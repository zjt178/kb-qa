"""RAG 主流程：检索 -> 组装上下文 -> 生成 -> 引用溯源。

这是在线链路的骨架。阶段2的混合检索+重排，替换的是 retrieve() 里的实现。
"""
from retriever import HybridRetriever   #从retriever.py导入HybridRetriever类，负责检索（向量/BM25/混合/重排）
from generator import build_chain   #从generator.py导入build_chain函数，目的是构建生成器链
from config import TOP_K            #从config.py导入TOP_K变量，TOP_K表示检索到的文档块数量


def format_context(docs: list) -> tuple[str, list]:
    #定义函数format_context，用于输入文档列表，希望返回上下文字符串和引用元数据列表
    """把检索到的文档块编号排版，并抽出引用元数据。

    返回 (context_str, sources)，sources 里的 id 与 context 里的 [n] 对应。
    """
    parts, sources = [], []
    #定义两个空列表parts和sources，用于存储上下文字符串和引用元数据列表
    for i, d in enumerate(docs, 1):
        #遍历文档列表，enumerate函数用于获取文档编号和文档内容
        parts.append(f"[{i}] {d.page_content.strip()}")
        #给parts内容加上编号【i】，并去除文档内容的前后空格
        sources.append(
            {
                "id": i,
                "source": d.metadata.get("source", "unknown"),
                "excerpt": d.page_content.strip()[:120],
            }
        )
        #给sources内容加上编号【i】，并提取文档内容的前120个字符作为摘要
    return "\n\n".join(parts), sources
    #把parts列表中的内容用\n\n连接起来，形成上下文字符串，并返回上下文字符串和引用元数据列表


class RAGPipeline:
    """懒加载单例用法：服务启动后第一次请求时才建索引/链，方便调试。"""

    def __init__(self):
        self.retriever = None
        self.chain = None

    def _ensure_ready(self):
        if self.retriever is None:
            self.retriever = HybridRetriever()
        if self.chain is None:
            self.chain = build_chain()

    def retrieve(self, question: str, k: int):
        """检索层 —— 交给 HybridRetriever，模式由 .env 的 RETRIEVER_MODE 决定。"""
        return self.retriever.search(question, k)

    def answer(self, question: str, k: int | None = None) -> dict:
        self._ensure_ready()
        docs = self.retrieve(question, k=k or TOP_K)
        if not docs:
            return {"answer": "知识库为空或未命中任何内容。", "sources": []}
        context, sources = format_context(docs)
        text = self.chain.invoke({"context": context, "question": question})
        return {"answer": text, "sources": sources}
