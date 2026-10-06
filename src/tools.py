"""阶段3：把检索层封装成 LangChain Tool。

一个 Tool 能不能被模型正确选中，取决于三件事：
    1. **名字**：动词 + 名词，一眼看出它能干什么（`search_knowledge_base` 而不是 `retriever`）
    2. **description**：这是**最影响选择准确率**的一项 —— 模型只看这段文字决定调不调用它。
       要写清 ① 它能做什么 ② 什么时候必须用 ③ 什么时候不要用（边界）。
       只写"检索知识库"是典型错误：模型不知道"什么算需要检索"。
    3. **入参 schema**：参数名和类型要让模型能填对（这里 query 是必填字符串）。

顺手做了两层工程约束：
    - **结果截断**：Top-5 原文可能 2500 字，全塞回去会挤爆上下文，故每块只回 400 字。
      工具返回值也是 prompt 的一部分。
    - **来源提取**：返回文本里带 `[n] 来源：xxx`，Agent 靠它组装引用溯源，
      不用回头再查一次检索器（那会多一次 IO）。
"""
import re

from langchain_core.tools import tool

from retriever import HybridRetriever

# 工具返回给模型的单块最大字符数（不是检索层参数，纯粹为了控上下文长度）
EXCERPT_CHARS = 400

DESCRIPTION = """在企业内部知识库中做语义 + 关键词混合检索，返回最相关的文档片段。

什么时候必须调用：
- 用户问题涉及公司/产品的具体事实：业务流程、内部规范、技术方案、配置参数、术语定义
- 问题里出现知识库可能记载的专有名词、代号、指标名

什么时候不要调用：
- 纯闲聊、打招呼、算术、写代码等与知识库内容无关的请求
- 上一轮已经检索到足够证据、能直接回答时（避免重复检索同一问题）"""


def build_search_tool(retriever: HybridRetriever | None = None, top_k: int = 5):
    """返回一个绑定到指定检索器实例的 Tool。

    retriever 用**注入**而不是在函数里 new：检索器初始化要加载 FAISS 索引 +
    分词 + 构建 BM25 语料，很贵；每次提问都重建纯属浪费，交给调用方持有单例。
    """
    retriever = retriever or HybridRetriever()

    @tool("search_knowledge_base", description=DESCRIPTION)
    def search_knowledge_base(query: str) -> str:
        """在企业知识库中检索与 query 最相关的文档片段。"""
        docs = retriever.search(query, k=top_k)
        if not docs:
            # 明确告知"没找到"，而不是返回空串 —— 空串会让模型以为工具坏了，
            # 转而用参数化知识编答案，这正是 RAG 要防的事。
            return "【检索结果】知识库中没有找到与该查询相关的片段。"
        lines = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            lines.append(f"[{i}] 来源：{src}\n{d.page_content.strip()[:EXCERPT_CHARS]}")
        return "\n\n".join(lines)

    return search_knowledge_base


# 工具返回文本的块结构：`[n] 来源：<path>\n<正文>`
_BLOCK_RE = re.compile(r"\[(\d+)\]\s*来源：([^\n]+)\n(.*?)(?=\n\[\d+\]\s*来源：|\Z)", re.S)


def extract_sources(observations: list[str]) -> list[dict]:
    """从工具返回的文本里抠出引用来源（按**块**保留，同文件的多块都留）。

    单独抽成函数有个好处：Agent 和 MCP Server 两条链路都复用它，
    引用格式只有一处定义 —— 改了 `[n] 来源：` 只需改这里。

    **两个踩过的坑（都很隐蔽，务必别再犯）**：

    ① **只返回来源名、不带正文** → 评估脚本拿 `page_content=<路径>` 算关键词，
       正文是空的 → 全部"未命中"，看着像检索彻底失效。→ 必须带回 `excerpt`。

    ② **按文件名去重** → 同一文件的第 2/3 块被丢弃。而关键词恰恰常出现在第 2 块
       （实测 Q13/Q18/Q20 三道题都因这个原因从"命中"变"未命中"，把 Agent 的
       关键词覆盖率从 0.95 拉低到 0.79）。**同一文件的不同块是不同证据，不能去重。**
       引用展示时如需合并同文件，交给上层做（保留 source 字段即可）。
    """
    sources = []
    for obs in observations:
        for m in _BLOCK_RE.finditer(str(obs)):
            sources.append(
                {
                    "id": len(sources) + 1,
                    "source": m.group(2).strip(),
                    "excerpt": m.group(3).strip(),
                }
            )
    return sources
