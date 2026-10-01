"""语料块的中转站：建库时把切好的块落盘，检索时读回来。

为什么要单独存一份？因为向量库里存的是"块 + 向量"，
而 BM25 和重排模型需要的是"块的原始文本"，单独存一份最省事、也最透明
（chunks.json 是纯文本，你可以直接打开看每块长什么样）。
"""
import json
from pathlib import Path

from config import INDEX_DIR


def chunks_path() -> Path:
    return Path(INDEX_DIR) / "chunks.json"


def save_chunks(chunks: list) -> None:
    """把 LangChain Document 列表存成 [[id, 文本, 元数据], ...]，按原顺序。"""
    path = chunks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"content": d.page_content, "metadata": d.metadata or {}} for d in chunks
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_chunks() -> list[dict]:
    """读回块列表。没有文件返回空列表（说明是改造前建的老索引）。"""
    path = chunks_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
