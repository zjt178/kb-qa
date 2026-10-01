"""检索层 —— 阶段2 的主角：向量检索 / BM25 / 混合检索（RRF）/ 重排。

三种模式，由 .env 的 RETRIEVER_MODE 控制，也可以代码里传参覆盖：
    vector          只用向量检索          —— baseline，用来对比
    hybrid          向量 + BM25，RRF 融合 —— 日常推荐
    hybrid+rerank   融合后再 cross-encoder 精排 —— 质量最高，慢一点

读代码建议从 search() 开始，它就是"一次检索"的完整流程。
"""
import re

from langchain_core.documents import Document

from chunk_store import load_chunks
from config import RECALL_K, RERANK_MODEL, RETRIEVER_MODE, RRF_K, TOP_K
from vectorstore import load_index


# ---------------- 分词：中文检索绕不过去的一步 ----------------
def tokenize(text: str) -> list[str]:
    """把一句话切成词。

    为什么 BM25 必须分词？BM25 靠"词出现多少次、这个词有多稀有"打分，
    而中文没有空格，不切词就只能整句比对，召回率会惨不忍睹。
    优先用 jieba（准确）；没装 jieba 就退化成字符二元组（能用，稍差）。
    """
    try:
        import jieba
    except ImportError:
        return _char_ngrams(text)
    return [w for w in jieba.lcut(text) if w.strip()]


def _char_ngrams(text: str, n: int = 2) -> list[str]:
    s = re.sub(r"\s+", "", text)
    if len(s) < n:
        return list(s)
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def chunk_to_doc(chunk: dict) -> Document:
    """把 chunks.json 里的一条记录还原成 LangChain 的 Document。"""
    return Document(page_content=chunk["content"], metadata=chunk.get("metadata", {}))


class HybridRetriever:
    def __init__(self, mode: str | None = None):
        self.mode = (mode or RETRIEVER_MODE).strip().lower()
        self.vs = load_index()
        self.chunks = load_chunks()
        self._bm25_index = None
        self._reranker = None
        # 老索引没有 chunks.json，BM25 无从下手 —— 优雅降级而不是直接崩
        if self.mode != "vector" and not self.chunks:
            print("[warn] 找不到 storage/index/chunks.json（这是升级前建的索引），")
            print("       已自动降级为 vector 模式。跑 python scripts/ingest.py --force 即可启用混合检索。")
            self.mode = "vector"

    # ---------------- 第一路：向量检索 ----------------
    def _vector_search(self, question: str, k: int) -> list[Document]:
        return self.vs.similarity_search(question, k=k)

    # ---------------- 第二路：BM25 关键词检索 ----------------
    def _bm25(self):
        if self._bm25_index is None:
            from rank_bm25 import BM25Okapi

            corpus = [tokenize(c["content"]) for c in self.chunks]
            self._bm25_index = BM25Okapi(corpus)
        return self._bm25_index

    def _bm25_search(self, question: str, k: int) -> list[Document]:
        bm25 = self._bm25()
        scores = bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        docs = []
        for i in order:
            if scores[i] <= 0:  # 一个词都没命中，不要污染结果
                break
            docs.append(chunk_to_doc(self.chunks[i]))
        return docs

    # ---------------- 融合：RRF ----------------
    @staticmethod
    def _rrf(doc_lists: list[list[Document]], k_const: int) -> list[Document]:
        """Reciprocal Rank Fusion：不看分数，只看排名。

        为什么不用加权求和？向量分数是余弦距离、BM25 分数是词频权重，
        两者量纲完全不同，权重怎么调都是玄学。RRF 用 1/(k+排名) 累加，
        天然免疫量纲问题，还不用调参 —— 这是工业界的默认做法。
        """
        scores: dict[str, float] = {}
        store: dict[str, Document] = {}
        for docs in doc_lists:
            for rank, d in enumerate(docs, 1):
                key = d.page_content  # 用正文当唯一键：同一个块在两路里内容一致
                scores[key] = scores.get(key, 0.0) + 1.0 / (k_const + rank)
                store.setdefault(key, d)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [store[key] for key, _ in ranked]

    # ---------------- 精排：cross-encoder ----------------
    def _rerank(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    # ---------------- 对外入口 ----------------
    def search(self, question: str, k: int | None = None) -> list[Document]:
        k = k or TOP_K
        if self.mode == "vector":
            return self._vector_search(question, k)

        fused = self._rrf(
            [
                self._vector_search(question, RECALL_K),
                self._bm25_search(question, RECALL_K),
            ],
            RRF_K,
        )
        if self.mode == "hybrid" or not fused:
            return fused[:k]

        # 双塔（向量）负责"recall 够全"，cross-encoder 负责"排序够准"
        # —— 重排模型把问题和每块拼接后一起过一遍，判断相关性更准，但慢，所以只精排前 N 条
        pairs = [(question, d.page_content) for d in fused]
        scores = self._rerank().predict(pairs)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [fused[i] for i in order]
