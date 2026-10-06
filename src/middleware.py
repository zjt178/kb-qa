"""强制检索 middleware：把「要不要检索」从模型的自由裁量变成结构保障。

## 为什么需要它

换用 qwen2.5:7b 后工具调用率从 4/8 升到 6/8，但残余失败仍是同一个模式：
    **模型不检索就直接作答**，且答案里带编造的引用标记（实测出现过
    "ronics[1] 提到，BM25 算法中的参数 k1 和 b ..."，[1] 是凭空捏的）。

提示词层面已经写了「硬性规则：回答前必须先检索」，但**提示词只能"请求"模型，
管不住模型**。要真正兜住，得在 agent 图的层面把这一步做成结构保障。

## 它做什么

`before_model` 钩子在**每次模型调用前**触发，此时能读到完整的 state：
    - 如果 messages 里**还没有任何 ToolMessage**（本次会话一次都没检索）
    - 且用户问题**不是纯寒暄**
    → 直接用检索器跑一次，把结果作为 ToolMessage 追加进 state。

于是模型看到的上下文里**已经躺着检索结果**了——它想偷懒也无从偷懒，
因为资料就在眼前，且系统提示要求它基于资料作答。

## 边界

- 寒暄（你好 / 谢谢 / 你是谁）**不注入** —— 强制检索会让"你好"也去翻知识库，很蠢
- 只在**首次**模型调用前注入；之后的轮次由模型自己决定要不要补检索
- 注入用的是与 `search_knowledge_base` 工具**完全相同**的检索器与格式化逻辑，
  保证"注入的"和"模型自己调的"结果一致（同一函数，不是两套实现）
"""
import re

from langchain.agents.middleware import before_model
from langchain_core.messages import AIMessage, ToolMessage

from tools import build_search_tool

# 纯寒暄 / 元问题——不需要检索知识库
_SMALLTALK = re.compile(
    r"^\s*(你好|您好|hi|hello|hey|哈喽|嗨|在吗|谢谢|感谢|多谢|"
    r"你是谁|你叫什么|你会什么|你能做什么|介绍一下你自己|再见|拜拜)",
    re.I,
)


def _is_smalltalk(text: str) -> bool:
    return bool(_SMALLTALK.match(text or ""))


def build_forced_retrieval_middleware(retriever=None, top_k: int | None = None):
    """造一个「首次模型调用前强制检索」的 middleware。

    用 build_search_tool 同一套逻辑取数，避免两处实现漂移。
    tool = build_search_tool(retriever=retriever, top_k=top_k)
    但工具本身要 invoke({"query": ...}) 才出结果，且返回已带 [n] 来源： 标记。
    """
    tool = build_search_tool(retriever=retriever, top_k=top_k)

    @before_model
    def forced_retrieval(state, runtime) -> dict | None:
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])

        # 已经检索过 → 不再插手（后续检索交给模型自己判断）
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return None

        # 取最后一条 HumanMessage 当查询
        question = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "human":
                question = str(getattr(m, "content", "") or "")
                break
        if not question or _is_smalltalk(question):
            return None

        # 替模型把检索做完（用的是同一个工具，结果格式一致）
        result = tool.invoke({"query": question})
        synthetic_call_id = "forced_retrieval_0"

        # 注入两条消息：
        #   ① AIMessage 带着 tool_calls —— 让 agent 图"看到"这次调用（轨迹可统计）
        #   ② ToolMessage 带检索结果 —— 模型据此作答
        # 只注入 ToolMessage 的话，图里的 tool_calls 计数会漏掉这次强制检索。
        ai_with_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": question},
                    "id": synthetic_call_id,
                    "type": "tool_call",
                }
            ],
        )
        tool_msg = ToolMessage(content=result, tool_call_id=synthetic_call_id)
        return {"messages": [ai_with_call, tool_msg]}

    return forced_retrieval
