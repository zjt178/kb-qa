"""阶段3：工具调用型 Agent —— LLM 自己决定"要不要检索、检索什么、检索几次"。

和阶段1/2 的关键区别：
    阶段1/2 是**固定链路**（prompt | llm）：检索必然发生、只发生一次，
    检索用的 query 就是用户原话 —— 用户问得含糊，检索就跟着烂。
    Agent 把「改写问题、决定检索几次」交给模型。这既是多步推理能带来增益的
    唯一来源，也是它可能变差的来源（模型可能不检索就硬答，或反复检索同一问题）。

**版本陷阱（这个坑值得单独记）**：
    langchain 1.x **删掉了** `create_react_agent` 和 `AgentExecutor`（0.x 时代的
    教科书 API，网上 90% 的示例还是它）。1.x 只剩 `create_agent`，且调用方式完全变了：

        # 0.x（现在 import 会直接报错）
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, ...)
        result = agent_executor.invoke({"input": q})

        # 1.x
        graph = create_agent(model=llm, tools=tools, system_prompt="...")
        result = graph.invoke({"messages": [{"role": "user", "content": q}]})

    不要把 prompt 里的 ReAct 格式硬塞进 1.x —— 它的循环靠**模型原生 tool calling**
    驱动（AIMessage.tool_calls），不再靠正则解析 "Action:" 文本，
    比 0.x 的文本协议稳得多。
"""
import re

from langchain.agents import create_agent

from config import AGENT_LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from llm_adapter import JsonToolCallChatModel
from middleware import build_forced_retrieval_middleware
from tools import build_search_tool, extract_sources

SYSTEM_PROMPT = """你是一个企业知识库问答助手，可以调用工具从知识库检索资料。

工作方式：
1.【硬性规则】除纯寒暄外，回答任何问题前必须先调用 search_knowledge_base 检索，
   禁止在未检索时向用户追问澄清，也禁止凭自己的知识直接回答。
   —— 知识库问答里，一次多余检索的代价是几秒钟；不检索就答的代价是编造。
2. 检索前把问题改写成包含关键术语的查询词（专有名词保留原文，不要改写术语）。
3. 首次检索覆盖不了问题时，换关键词最多再检索 2 次；仍不足则如实说明。
4. 资料足够后立即作答，不要对同一个问题反复检索。
5. 判断"资料是否足够"看**检索结果里有没有能支撑答案的内容**，不要因为结果里
   没有原话照搬就判定为"没有"——允许基于片段归纳作答。

回答要求：
- 用简体中文，简洁直接，不超过 300 字
- 关键论断后标注来源编号，如 [1]，编号对应检索结果里的 [1][2]
- **只有当检索结果确实与问题无关时**，才回答"知识库中没有找到相关内容"；
  只要检索结果里有可用信息，就必须直接作答，**禁止在答案开头套用这句拒答模板**。
  （实测踩坑：模型会把拒答句当成固定开头，先写"没有找到相关内容"再补正确答案——
   这是错误的。要么拒答、要么作答，不要两者都写。）"""


class RagAgent:
    """把 HybridRetriever 包成工具，交给 langchain 1.x 的 agent 图调度。

    verbose 默认关：29 条题 × 多步推理的日志会淹掉评估结果，要调试再开。
    """

    def __init__(self, retriever=None, top_k: int | None = None, verbose: bool = False,
                 force_retrieval: bool = True):
        self.top_k = top_k or TOP_K
        self.tool = build_search_tool(retriever=retriever, top_k=self.top_k)
        # 模型用 AGENT_LLM_MODEL（默认 qwen2.5:7b）：Agent 依赖原生 tool calling，
        # qwen2.5-coder 的"文本模仿"式调用格式不稳定（详见 config.py 注释）。
        # JsonToolCallChatModel 仍保留：即使换了模型，它对三种文本变体依然兜底，
        # 对结构化 tool_calls 则原样透传 —— 两层防御，互不冲突。
        llm = JsonToolCallChatModel(
            model_name=AGENT_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,  # Agent 要稳：同一问题别每次走不同的工具路径
        )
        # 强制检索 middleware（见 middleware.py）：提示词只能"请求"模型先检索，
        # 管不住模型偷懒不检索就硬答（实测出现过"不检索 + 编造 [1] 引用"）。
        # 这层在首次模型调用前替它把检索做完，把"要不要检索"变成结构保障。
        middleware = []
        if force_retrieval:
            middleware.append(
                build_forced_retrieval_middleware(retriever=retriever, top_k=self.top_k)
            )
        self.graph = create_agent(
            model=llm,
            tools=[self.tool],
            system_prompt=SYSTEM_PROMPT,
            middleware=tuple(middleware),
        )

    def answer(self, question: str) -> dict:
        """跑一次 Agent，返回答案 + 引用来源 + 轨迹元数据。

        为什么要返回 tool_calls / steps：单跳指标（Hit@K）衡量不了多步推理，
        「检索了几次、走没走弯路」是 Agent 独有的观测维度，评估脚本要用。
        """
        result = self.graph.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result.get("messages", [])

        observations = []   # 每次工具返回的原文，用于提取来源
        tool_names = []     # 工具调用轨迹
        answer = ""
        for m in messages:
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    # 兼容两种形状：dict（1.x 常见）和带 .name 的对象
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    tool_names.append(str(name))
            # ToolMessage 的 content 就是工具返回值；最后一条 AIMessage 是最终答案
            if getattr(m, "type", "") == "tool":
                observations.append(str(getattr(m, "content", "")))
            elif getattr(m, "type", "") == "ai" and not tool_calls:
                answer = _as_text(getattr(m, "content", ""))

        return {
            "answer": answer.strip(),
            "sources": extract_sources(observations),
            "tool_calls": len(tool_names),
            "steps": tool_names,
        }


def _as_text(content) -> str:
    """AIMessage.content 可能是 str，也可能是 [{'type':'text','text':...}] 列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(str(c.get("text", "")))
            else:
                parts.append(str(c))
        return "".join(parts)
    return str(content or "")
