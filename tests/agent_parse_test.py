"""验证 RagAgent.answer() 的**消息解析逻辑** —— 不需要 ollama。

为什么要单独测这一段：
    真实 Agent 跑一次要几十秒且依赖模型服务，但"从消息列表里抠出答案 / 工具轨迹 /
    引用来源"这段纯解析逻辑是**最容易写错、也最容易悄悄错**的地方
    （content 可能是 str 也可能是 list、tool_calls 可能是 dict 也可能是对象）。
    用一个假的 messages 列表喂进去，秒级验证，且覆盖了多步调用 / 无工具调用 / 拒答三种形态。

跑法：python tests/agent_parse_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import RagAgent  # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  —— {detail}" if detail else ""))


class FakeMsg:
    def __init__(self, type_, content="", tool_calls=None):
        self.type = type_
        self.content = content
        self.tool_calls = tool_calls or []


TOOL_OUT_1 = "[1] 来源：rrf.md\nRRF 用 1/(k+rank) 融合。\n\n[2] 来源：bm25.md\nBM25 靠词频打分。"
TOOL_OUT_2 = "[1] 来源：rrf.md\nRRF 的 k 推荐 60。\n\n[2] 来源：eval.md\n严格口径是 Hit@K 全词。"


def parse(messages, question="测试问题"):
    """直接调 answer() 的解析部分：绕开 graph.invoke，用一个 stub 顶替。"""
    a = RagAgent.__new__(RagAgent)  # 不走 __init__，避免加载 FAISS / 连 ollama

    class StubGraph:
        def invoke(self, _inputs):
            # 顺带验证入参格式是 1.x 的 {"messages": [...]}
            assert "messages" in _inputs, "入参必须是 {'messages': [...]}"
            assert _inputs["messages"][0]["role"] == "user"
            return {"messages": messages}

    a.graph = StubGraph()
    return a.answer(question)


print("=" * 72)
print("一、多步调用：两次检索 + 带引用编号的最终答案")
print("=" * 72)

msgs = [
    FakeMsg("human", "RRF 的 k 取多少？"),
    FakeMsg("ai", "", [{"name": "search_knowledge_base", "args": {"query": "RRF k"}}]),
    FakeMsg("tool", TOOL_OUT_1),
    FakeMsg("ai", "", [{"name": "search_knowledge_base", "args": {"query": "严格评估口径"}}]),
    FakeMsg("tool", TOOL_OUT_2),
    FakeMsg("ai", "RRF 的 k 推荐取 60 [1]，严格口径用 Hit@K 全词 [2]。"),
]
out = parse(msgs)
check("答案被正确提取", out["answer"].startswith("RRF 的 k 推荐取 60"), out["answer"][:30])
check("工具调用次数 = 2", out["tool_calls"] == 2, str(out["tool_calls"]))
check("轨迹记录了工具名", out["steps"] == ["search_knowledge_base"] * 2, str(out["steps"]))
srcs = [s["source"] for s in out["sources"]]
# 注意断言的是**按块保留**：rrf.md 在两轮检索里各出现一次，两条都要留下。
# 早先这里写的是"去重"断言（rrf.md 只留一次）—— 那是错的：同一文件的不同块是
# 不同证据，去重会丢掉第 2 块，而关键词常在第 2 块（实测把覆盖率从 0.95 拉到 0.79）。
check("两次检索的来源都提取到（按块保留，同文件多块不丢）",
      srcs == ["rrf.md", "bm25.md", "rrf.md", "eval.md"], str(srcs))
check("来源 id 连续", [s["id"] for s in out["sources"]] == [1, 2, 3, 4])
# excerpt 必须带正文：只留来源名会让评估脚本关键词匹配全落空
check("excerpt 带回了块正文（非空）",
      all(s["excerpt"] for s in out["sources"]) and "1/(k+rank)" in out["sources"][0]["excerpt"],
      repr(out["sources"][0]["excerpt"][:30]))

print()
print("=" * 72)
print("二、没调工具就直接答（模型偷懒的情形，必须能识别出来）")
print("=" * 72)

lazy = [
    FakeMsg("human", "今天天气怎么样？"),
    FakeMsg("ai", "我只是个知识库助手，不知道天气。"),
]
out2 = parse(lazy)
check("没调工具时 tool_calls=0（这是要监控的坏行为）", out2["tool_calls"] == 0)
check("答案仍然被提取", "不知道天气" in out2["answer"])
check("无来源时 sources 为空而不是报错", out2["sources"] == [])

print()
print("=" * 72)
print("三、拒答形态：检索了但没查到")
print("=" * 72)

refuse = [
    FakeMsg("human", "火星分公司地址？"),
    FakeMsg("ai", "", [{"name": "search_knowledge_base", "args": {"query": "火星分公司"}}]),
    FakeMsg("tool", "【检索结果】知识库中没有找到与该查询相关的片段。"),
    FakeMsg("ai", "知识库中没有找到相关内容。"),
]
out3 = parse(refuse)
check("拒答答案被提取", "没有找到相关内容" in out3["answer"], out3["answer"])
check("检索过但无来源（正确反映真实情况）", out3["sources"] == [])
check("工具调用次数记到了 1", out3["tool_calls"] == 1)

print()
print("=" * 72)
print("四、content 形态兼容（str / list-of-dict，本地模型两种都会出现）")
print("=" * 72)

list_content = [
    FakeMsg("human", "问"),
    FakeMsg("ai", [{"type": "text", "text": "这是列表形态的回答 [1]。"}]),
]
out4 = parse(list_content)
check("list 形态 content 能拼成文本", out4["answer"] == "这是列表形态的回答 [1]。", out4["answer"])

print()
print("=" * 72)
print("五、适配层的文本解析（llm_adapter，纯函数不需要模型）")
print("=" * 72)

from llm_adapter import JsonToolCallChatModel, _content_to_text, _extract_json_objects  # noqa: E402
from tools import build_search_tool as _bst  # noqa: E402

# 只借它解析纯函数，不真调模型：用 __new__ 绕开模型初始化
probe = JsonToolCallChatModel.__new__(JsonToolCallChatModel)
from retriever import HybridRetriever as _HR  # noqa: E402

probe._bound_tools = []  # 解析测试不用真工具；name 校验用假名单
probe._tool_names  # 确认属性可访问
KNOWN = {"search_knowledge_base"}


def parse_text(text):
    """复刻 _generate 里的解析路径（三种格式 + name 白名单校验）。"""
    import json as _json
    from llm_adapter import _BRACKET_TOOL_RE, _TOOL_CALL_TAG_RE, _normalize_call

    calls = []
    for m in _TOOL_CALL_TAG_RE.finditer(text):
        try:
            obj = _json.loads(m.group(1))
        except Exception:
            continue
        c = _normalize_call(obj.get("name", ""), obj.get("arguments", obj.get("parameters")), KNOWN)
        if c:
            calls.append(c)
    for m in _BRACKET_TOOL_RE.finditer(text):
        try:
            args = _json.loads(m.group(2))
        except Exception:
            continue
        c = _normalize_call(m.group(1), args, KNOWN)
        if c:
            calls.append(c)
    if not calls:
        for obj in _extract_json_objects(text):
            c = _normalize_call(
                obj.get("name", ""),
                obj.get("arguments", obj.get("parameters", obj.get("input"))),
                KNOWN,
            )
            if c:
                calls.append(c)
    return calls


# ① 裸 JSON（qwen2.5-coder 最常见输出）
c1 = parse_text('{"name": "search_knowledge_base", "arguments": {"query": "RRF k"}}')
check("① 裸 JSON 能解析", len(c1) == 1 and c1[0]["name"] == "search_knowledge_base" and c1[0]["args"] == {"query": "RRF k"})

# ①b 带代码围栏 + 前后缀文字
c1b = parse_text('好的，我来查一下。\n```json\n{"name": "search_knowledge_base", "arguments": {"query": "k"}}\n```')
check("①b 围栏+废话包裹能解析", len(c1b) == 1, f"{len(c1b)} 条")

# ② 方括号简写（实机出现过的变体）
c2 = parse_text('请稍等，我正在检索... [search_knowledge_base {"query":"RRF融合的k常数"}]')
check("② [name {json}] 变体能解析", len(c2) == 1 and c2[0]["args"] == {"query": "RRF融合的k常数"})

# ③ qwen 官方 <tool_call> 标签（模板要求但 coder 模型经常忘写的格式）
c3 = parse_text('<tool_call>\n{"name": "search_knowledge_base", "arguments": {"query": "k"}}\n</tool_call>')
check("③ <tool_call> 标签能解析", len(c3) == 1)

# 安全阀：name 不在白名单里的 JSON 不许当成工具调用
c_bad = parse_text('{"name": "delete_database", "arguments": {"confirm": true}}')
check("④ 非白名单工具被拒收（安全阀）", len(c_bad) == 0)

# 普通回答不误判
c_plain = parse_text("RRF 的 k 推荐取 60，这是论文默认值。")
check("⑤ 普通回答不误判为工具调用", len(c_plain) == 0)

# 去重：整串解析和贪心截取命中同一个 JSON 时不能翻倍
c_dup = parse_text('{"name": "search_knowledge_base", "arguments": {"query": "RRF k"}}')
check("⑥ 同一调用不重复解析", len(c_dup) == 1, f"{len(c_dup)} 条")

# _content_to_text 的两种形态
check("⑦ _content_to_text: str 直返", _content_to_text("abc") == "abc")
check("⑧ _content_to_text: list 拼接", _content_to_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab")

print()
print("=" * 72)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
if FAIL:
    print("失败：" + ", ".join(FAIL))
print("=" * 72)
sys.exit(1 if FAIL else 0)
