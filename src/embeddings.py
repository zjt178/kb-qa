"""嵌入模型封装。阶段2如果换 Milvus + 服务化嵌入，只改这个文件。"""
from langchain_ollama import OllamaEmbeddings

from config import EMBED_MODEL, OLLAMA_BASE_URL


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
