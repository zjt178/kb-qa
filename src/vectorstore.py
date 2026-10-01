"""向量库封装。MVP 用 FAISS（零部署成本）；阶段2换成 Milvus 时只动这里。"""
from pathlib import Path    #跨平台写文件路径
import shutil               #删除整个索引文件夹

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
    """用文档块构建 FAISS 索引并落盘。

    force=True 时重建：先在新目录里构建成功，再替换旧索引。
    绝不能先删后建——万一嵌入失败（比如 ollama 没启动），旧索引就白丢了。
    """
    path = Path(INDEX_DIR)
    if path.exists() and not force:
        print(f"索引已存在于 {INDEX_DIR}，跳过构建（force=True 可强制重建）")
        return load_index()

    print(f"开始嵌入 {len(docs)} 个文档块...")
    vs = FAISS.from_documents(docs, get_embeddings())  # 可能失败的一步放在最前面

    tmp = path.with_name(path.name + "_tmp")  # 先落到临时目录
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(tmp))

    if path.exists():  # 新索引建好了，这时才敢删旧的
        shutil.rmtree(path)
    tmp.rename(path)
    return vs
