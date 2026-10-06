"""阶段3 离线验收 A：不依赖 ollama 的部分 —— 工具定义 + BM25 通路 + 来源提取。

分两段的原因：
    Agent 端到端指标必须等 ollama 在线；但**向量无关**的那部分（工具三要素、
    BM25 字面检索、RRF 融合、引用格式）离线就能全验。
    能拆就拆，不因为"模型服务没开"整体卡住。

跑法：python tests/agent_offline_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools import build_search_tool, extract_sources  # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  —— {detail}" if detail else ""))


print("=" * 72)
print("一、工具层：Tool 三要素（名字 / description / 入参 schema）")
print("=" * 72)

tool = build_search_tool()
check("工具名是 动词_名词 形式", tool.name == "search_knowledge_base", tool.name)
desc = tool.description
check("description 有边界说明（含'不要调用'）", "不要调用" in desc)
check("description 足够长（模型靠它决策）", len(desc) > 100, f"{len(desc)} 字")
check("入参 schema 含 query(string)", "query" in tool.args, str(tool.args))

print()
print("=" * 72)
print("二、检索通路（绕开 ollama）：BM25 字面检索 + RRF 融合能独立工作")
print("=" * 72)

from retriever import HybridRetriever, tokenize  # noqa: E402

r = HybridRetriever.__new__(HybridRetriever)  # 不走 __init__，绕开 FAISS+ollama
from chunk_store import load_chunks  # noqa: E402

r.chunks = load_chunks()
r.mode = "hybrid"
r._bm25_index = None
r._reranker = None
print(f"  语料块数：{len(r.chunks)}")

check("jieba 分词可用（BM25 的前提）", len(tokenize("混合检索的RRF融合")) > 1,
      str(tokenize("混合检索的RRF融合")[:5]) + "...")

bm = r._bm25_search("RRF 的 k 取多少", 5)
check("BM25 字面检索有结果", len(bm) > 0, f"{len(bm)} 条")
check("BM25 命中了含 RRF 的块", any("RRF" in d.page_content or "rrf" in d.page_content.lower() for d in bm))

print()
print("  字面术语查询对比（这是 hybrid 相对 vector 唯一有增益的场景）：")
for q in ["RRF 的 k 取多少", "k1 和 b 参数", "多证据题为什么会失效"]:
    hits = r._bm25_search(q, 3)
    top = hits[0].page_content[:36].replace("\n", " ") if hits else "（无命中）"
    print(f"    「{q}」→ {len(hits)} 条  首条：{top}...")
check("BM25 在字面术语查询上有召回", len(r._bm25_search("RRF 的 k 取多少", 5)) > 0)

# RRF 融合是纯计算，不用 ollama
from langchain_core.documents import Document  # noqa: E402

fusion = HybridRetriever._rrf(
    [[Document(page_content="A"), Document(page_content="B")],
     [Document(page_content="B"), Document(page_content="C")]],
    60,
)
order = [d.page_content for d in fusion]
check("RRF 把两路都认可的块排到最前（B 优先）", order[0] == "B", "→".join(order))
check("RRF 结果未丢块", set(order) == {"A", "B", "C"})

print()
print("=" * 72)
print("三、来源提取（Agent 的引用溯源依赖它）")
print("=" * 72)

sample = "[1] 来源：a.md\n内容A\n\n[2] 来源：b.md\n内容B\n\n[3] 来源：a.md\n内容A2"
# 只传一次 observation —— 早先这里传的是 [sample, sample]（把同一段喂两遍），
# 测出来的是"跨 observation 去重"，掩盖了真正要紧的场景：**同一文件有多块**。
srcs = extract_sources([sample])
check("能提取出来源", len(srcs) == 3, str([s["source"] for s in srcs]))
check("同一文件的多个块都保留（不去重）", [s["source"] for s in srcs] == ["a.md", "b.md", "a.md"], str([s["source"] for s in srcs]))
check("excerpt 带回块正文（不是空串）", srcs[0]["excerpt"] == "内容A", repr(srcs[0]["excerpt"]))
check("id 从 1 连续", [s["id"] for s in srcs] == list(range(1, len(srcs) + 1)))
check("无来源时不报错", extract_sources(["没有来源标记的文本"]) == [])

print()
print("=" * 72)
print("四、MCP 协议层：Server 能被标准协议列出并调用（3 个工具全部实调）")
print("=" * 72)

import asyncio  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("mcp_server", ROOT / "scripts" / "mcp_server.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
server = mod.mcp


async def probe():
    tools = await server.list_tools()
    names = sorted(t.name for t in tools)
    print(f"  MCP 暴露工具：{names}")
    check("暴露 3 个工具", len(tools) == 3, str(names))
    for expect in ("search_knowledge_base", "list_documents", "get_retrieval_config"):
        check(f"包含 {expect}", expect in names)

    async def call(name, args):
        res = await server.call_tool(name, args)
        if isinstance(res, tuple):          # 1.x 返回 (content, structured)
            res = res[0]
        if hasattr(res, "content"):         # 2.x 返回 CallToolResult
            res = res.content
        item = res[0]
        return getattr(item, "text", str(item))

    t = await call("get_retrieval_config", {})
    check("get_retrieval_config 实调成功", "检索模式" in t, t.splitlines()[0])

    t = await call("list_documents", {})
    check("list_documents 实调成功（列出文档）", "篇文档" in t, t.splitlines()[0])
    print(f"    {t.splitlines()[0]}")

    # search 走向量，需要 ollama；这里只断言"调用不抛异常且返回文本"
    try:
        t = await call("search_knowledge_base", {"query": "RRF", "top_k": 2})
        check("search_knowledge_base 经 MCP 实调成功", "] 来源：" in t, t[:40].replace("\n", " "))
    except Exception as e:
        check("search_knowledge_base 经 MCP 实调成功", False,
              f"{type(e).__name__}（需 ollama 在线）: {str(e)[:60]}")


asyncio.run(probe())

print()
print("=" * 72)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
if FAIL:
    print("失败：")
    for f in FAIL:
        print("  - " + f)
print("=" * 72)
sys.exit(1 if FAIL else 0)
