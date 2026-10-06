"""LLM 适配层：把「文本形态的工具调用」归一化为 langchain 标准的 tool_calls。

## 背景（为什么需要这层）

langchain 1.x 的 agent 图**只认** AIMessage.tool_calls 这个结构化字段来决定
"要不要进工具节点"。但本机的 qwen2.5-coder:latest 在 ollama 上不吐结构化
tool_calls，而是把工具调用**写成纯文本 JSON** 塞在 content 里：

    content: '{"name": "search_knowledge_base", "arguments": {"query": "..."}}'
    tool_calls: []

（探针 tests/probe_tool_calling.py 的实测输出。根因是该模型的 ollama 聊天模板
没有原生的 tools 段，ollama 把工具定义降级为文本注入，模型只能模仿着输出 JSON。）

Agent 图看到空 tool_calls 就以为模型直接给了最终答案，循环结束 ——
29 题全部「工具×0」就是这么来的。

## 这层做什么

继承 BaseChatModel 造一个壳：
    _generate() 内部照常调 ChatOllama（工具定义照传），
    拿到响应后做一次归一化：content 是工具调用 JSON → 重组成
    AIMessage(content="", tool_calls=[{name, args, id, type}])，
    不是 → 原样透传。

对 create_agent 完全透明：它 bind_tools / invoke 的接口都没变，
只是"看见"的 tool_calls 从空变成了有。

## 边界处理

- content 可能是 str 也可能是 [{"type":"text","text":...}] 列表
- JSON 可能带 ```json 代码围栏或前后缀文字 → 先整串解析，失败再掐头去尾
- arguments / parameters / input 三种键名都认（不同模型习惯不同）
- **name 必须在已绑定的工具名单里才认** —— 防止把正文里恰好长得像 JSON
  的内容误判成工具调用，这是这层最重要的安全阀
"""
import json
import re
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from langchain_ollama import ChatOllama


def _content_to_text(content) -> str:
    """AIMessage.content 兼容：str 直返，list-of-dict 拼接 text 段，其他转 str。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content
        )
    return str(content or "")


def _extract_json_objects(text: str) -> list[dict]:
    """从文本里抠出候选 JSON 对象。

    顺序：整串直接解析（最常见：content 就是纯 JSON）→
    第一个 { 到最后一个 } 的贪心截取（前后有废话时）。
    两种解析可能命中同一个对象，返回前去重（按键排序序列化当指纹）。
    """
    text = text.strip()
    # 去掉 markdown 代码围栏（模型偶尔加 ```json ... ```）
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    candidates = []
    try:
        candidates.append(json.loads(text))
    except Exception:
        pass
    if "{" in text and "}" in text:
        try:
            candidates.append(json.loads(text[text.index("{"): text.rindex("}") + 1]))
        except Exception:
            pass
    objs, seen = [], set()
    for c in candidates:
        if not isinstance(c, dict):
            continue
        fp = json.dumps(c, sort_keys=True, ensure_ascii=False)
        if fp not in seen:
            seen.add(fp)
            objs.append(c)
    return objs


# 模型不守格式时的变体（实测 qwen2.5-coder 全都出现过）：
#   ① 裸 JSON：      {"name": "...", "arguments": {...}}
#   ② 方括号简写：    [search_knowledge_base {"query": "..."}]
#   ③ qwen 官方格式： <tool_call>{"name": ..., "arguments": {...}}</tool_call>
_BRACKET_TOOL_RE = re.compile(
    r"\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*?\})\s*\]", re.S
)
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S
)


def _normalize_call(name, args, known_names) -> dict | None:
    """校验并归一化一次工具调用；name 不在已知工具名单里则拒收。"""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = None
    if name in known_names and isinstance(args, dict):
        return {
            "name": name,
            "args": args,
            "id": "call_" + uuid.uuid4().hex[:8],
            "type": "tool_call",
        }
    return None


class JsonToolCallChatModel(BaseChatModel):
    """把文本形态的工具调用归一化为标准 tool_calls 的聊天模型壳。"""

    model_name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0

    # pydantic 私有属性：不进模型校验，只在运行期用
    _inner: ChatOllama = PrivateAttr(default=None)
    _bound_tools: list = PrivateAttr(default_factory=list)

    def model_post_init(self, __context) -> None:
        self._inner = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    @property
    def _llm_type(self) -> str:
        return "json-tool-call-chat-model"

    def bind_tools(self, tools, **kwargs):
        """记录绑定的工具；其余 kwargs（tool_choice 等）ollama 后端不支持，忽略。"""
        clone = self.model_copy(deep=False)
        clone._bound_tools = list(tools)
        return clone

    @property
    def _tool_names(self) -> set:
        names = set()
        for t in self._bound_tools:
            if isinstance(t, BaseTool):
                names.add(t.name)
            elif isinstance(t, dict):
                names.add(t.get("name", ""))
            elif isinstance(t, type):
                names.add(getattr(t, "name", ""))
        return names

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        # 工具定义照常传给 ollama —— 模型得先"知道"有这个工具，才谈得上调用
        inner = self._inner
        if self._bound_tools:
            inner = inner.bind_tools(self._bound_tools)
        resp = inner.invoke(messages, stop=stop) if stop else inner.invoke(messages)

        # 已经是结构化 tool_calls（换了支持原生工具的模型时）→ 原样透传
        if resp.tool_calls:
            return ChatResult(generations=[ChatGeneration(message=resp)])

        text = _content_to_text(resp.content)
        known = self._tool_names
        calls = []

        # ① + ③：标准 JSON（可能在 <tool_call> 标签里）→ 走 JSON 对象解析
        for m in _TOOL_CALL_TAG_RE.finditer(text):
            try:
                obj = json.loads(m.group(1))
            except Exception:
                continue
            c = _normalize_call(
                obj.get("name", ""), obj.get("arguments", obj.get("parameters")), known
            )
            if c:
                calls.append(c)

        # ②：方括号简写变体
        for m in _BRACKET_TOOL_RE.finditer(text):
            try:
                args = json.loads(m.group(2))
            except Exception:
                continue
            c = _normalize_call(m.group(1), args, known)
            if c:
                calls.append(c)

        # 裸 JSON（最常见）—— 上面的标签/方括号没命中才试，且要求"名字在工具名单里"
        if not calls:
            for obj in _extract_json_objects(text):
                c = _normalize_call(
                    obj.get("name", ""),
                    obj.get("arguments", obj.get("parameters", obj.get("input"))),
                    known,
                )
                if c:
                    calls.append(c)

        if calls:
            # 重组：content 置空，工具调用进结构化字段 —— agent 图靠它路由
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="", tool_calls=calls))]
            )
        # 普通回答（含模型拒答/闲聊）→ 原样透传
        return ChatResult(generations=[ChatGeneration(message=resp)])
