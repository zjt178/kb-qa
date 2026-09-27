"""向量库封装。MVP 用 FAISS（零部署成本）；阶段2换成 Milvus 时只动这里。"""
from pathlib import Path
import shutil

from langchain_community.vectorstores import FAISS

from embeddings import get_embeddings
from config import INDEX_DIR


def load_index():
    """加载已有索引。不存在则报错并提示先建库。"""
    path = Path(INDEX_DIR)
    if not path.exists():
        raise FileNotFoundError(
            f"索引目录 {INDEX_DIR} 不存在。请先运行: python scripts/ingest.py"
        )
    return FAISS.load_local(
        str(path), get_embeddings(), allow_dangerous_deserialization=True
    )


def build_index(docs: list, force: bool = False):
    """用文档块构建 FAISS 索引并落盘。force=True 时删除旧索引重建。"""
    path = Path(INDEX_DIR)
    if path.exists() and force:
        shutil.rmtree(path)
    if path.exists():
        print(f"索引已存在于 {INDEX_DIR}，跳过构建（force=True 可强制重建）")
        return load_index()
    print(f"开始嵌入 {len(docs)} 个文档块...")
    vs = FAISS.from_documents(docs, get_embeddings())
    path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))
    return vs
