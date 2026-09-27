"""RAG 主流程：检索 -> 组装上下文 -> 生成 -> 引用溯源。

这是在线链路的骨架。阶段2的混合检索+重排，替换的是 retrieve() 里的实现。
"""
from vectorstore import load_index
from generator import build_chain
from config import TOP_K


def format_context(docs: list) -> tuple[str, list]:
    """把检索到的文档块编号排版，并抽出引用元数据。

    返回 (context_str, sources)，sources 里的 id 与 context 里的 [n] 对应。
    """
    parts, sources = [], []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] {d.page_content.strip()}")
        sources.append(
            {
                "id": i,
                "source": d.metadata.get("source", "unknown"),
                "excerpt": d.page_content.strip()[:120],
            }
        )
    return "\n\n".join(parts), sources


class RAGPipeline:
    """懒加载单例用法：服务启动后第一次请求时才建索引/链，方便调试。"""

    def __init__(self):
        self.vs = None
        self.chain = None

    def _ensure_ready(self):
        if self.vs is None:
            self.vs = load_index()
        if self.chain is None:
            self.chain = build_chain()

    def retrieve(self, question: str, k: int):
        """检索层 —— 阶段2在这里升级为 混合检索 + 重排。"""
        return self.vs.similarity_search(question, k=k)

    def answer(self, question: str, k: int | None = None) -> dict:
        self._ensure_ready()
        docs = self.retrieve(question, k=k or TOP_K)
        if not docs:
            return {"answer": "知识库为空或未命中任何内容。", "sources": []}
        context, sources = format_context(docs)
        text = self.chain.invoke({"context": context, "question": question})
        return {"answer": text, "sources": sources}
