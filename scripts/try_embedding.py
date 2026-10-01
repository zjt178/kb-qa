import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(model="qllama/bge-small-zh-v1.5:latest", base_url="http://localhost:11434")

sentences = [
    "向量数据库用于存储和检索高维向量",
    "FAISS 是常用的向量检索库",
    "今天天气不错，适合出去散步",
]

vecs = [emb.embed_query(s) for s in sentences]
print("每个句子的向量维度：", len(vecs[0]))
print("前 5 个数字长这样：", [round(x, 3) for x in vecs[0][:5]])
print()

def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)

print("句1 vs 句2（都讲向量）：", round(cos(vecs[0], vecs[1]), 3))
print("句1 vs 句3（毫不相关）：", round(cos(vecs[0], vecs[2]), 3))
