"""阶段3：把 kb-qa 的检索能力做成一个标准 MCP Server。

为什么要做成 MCP 而不是内部函数：
    内部函数只能被**当前这个应用**调用 —— 想从 Claude Desktop、Cursor、
    或者另一个 Agent 里用同一套知识库，就得复制一遍代码。
    MCP（Model Context Protocol）把这层能力标准化成"服务"：
    谁实现协议谁就能接，检索逻辑只有一份。
    这就是"检索能力服务化"和"检索能力内嵌"的本质区别。

传输方式用 stdio：MCP 客户端会把这个脚本当子进程拉起来，
通过标准输入输出上的 JSON-RPC 通信 —— 不需要端口、不需要网络配置。
（要跨机器访问再换 streamable-http。）

**版本注意**：官方 SDK 走到 2.x 后，`mcp.server.fastmcp.FastMCP` 改名为
`mcp.server.mcpserver.MCPServer`（1.x 的教程/博客全是旧名字，照抄会 ImportError）。
这里写了兼容分支，两个版本都能跑。

跑法：
    python scripts/mcp_server.py          # 由 MCP 客户端拉起，一般不手动跑
调试：
    npx @modelcontextprotocol/inspector python scripts/mcp_server.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x 的旧名字
    from mcp.server.fastmcp import FastMCP as _Server

from config import RECALL_K, RETRIEVER_MODE, RRF_K, TOP_K  # noqa: E402
from retriever import HybridRetriever  # noqa: E402

mcp = _Server("kb-qa")

# 检索器单例：MCP Server 是长驻进程，索引只加载一次，后续每次调用直接复用。
# 如果写成一个裸函数每次 new 一个，第一次请求就要等好几秒加载 FAISS。
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = TOP_K) -> str:
    """在企业知识库中做语义 + 关键词混合检索，返回最相关的文档片段。

    什么时候用：
    - 用户问公司/产品的具体事实：业务流程、内部规范、技术方案、配置参数、术语定义
    - 问题里出现知识库可能记载的专有名词、代号、指标名

    什么时候不要用：
    - 纯闲聊、打招呼、算术、写代码等与知识库内容无关的请求
    - 已经拿到足够证据、能直接回答时（避免重复检索同一问题）

    Args:
        query: 检索用的自然语言查询。可以是用户原话，也可以是模型改写后的关键词，
               建议把专有名词保留原文（BM25 那一路靠字面匹配）。
        top_k: 返回几条，默认 5。问题很宽泛时调到 8，问题很具体时 3 就够。
    """
    docs = get_retriever().search(query, k=top_k)
    if not docs:
        return "【检索结果】知识库中没有找到与该查询相关的片段。"
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        lines.append(f"[{i}] 来源：{src}\n{d.page_content.strip()}")
    return "\n\n".join(lines)


@mcp.tool()
def list_documents() -> str:
    """列出知识库里当前有哪些文档，用于判断"该查什么"或确认某个主题是否被收录。"""
    from chunk_store import load_chunks

    chunks = load_chunks()
    if not chunks:
        return "知识库为空。"
    # 按来源聚合：一个文档被切成多块，这里要还原成"有几个文档"
    per_source: dict[str, int] = {}
    for c in chunks:
        src = c.get("metadata", {}).get("source", "unknown")
        per_source[src] = per_source.get(src, 0) + 1
    lines = [f"知识库共 {len(per_source)} 篇文档 / {len(chunks)} 个切片："]
    lines += [f"- {src}（{n} 块）" for src, n in sorted(per_source.items())]
    return "\n".join(lines)


@mcp.tool()
def get_retrieval_config() -> str:
    """返回当前检索配置（模式 / Top-K / 召回数）。

    有了它，MCP 客户端才知道"这个知识库现在用的是哪种检索策略"——
    否则你在一边改了 .env，另一边的 Agent 完全不知道结果为什么变了。
    """
    return (
        f"检索模式：{RETRIEVER_MODE}\n"
        f"TOP_K：{TOP_K}\n"
        f"每路召回：RECALL_K={RECALL_K}，RRF 平滑常数 k={RRF_K}"
    )


if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输
