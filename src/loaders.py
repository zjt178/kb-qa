"""离线链路第一步：文档加载 + 清洗 + 切分。"""
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# 中文切分要显式给中文标点做分隔符，否则容易把句子拦腰切断
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def load_documents(docs_dir: str = DOCS_DIR) -> list:
    """扫描 docs_dir，按扩展名选择 loader。支持 .md/.txt/.pdf"""
    docs = []
    root = Path(docs_dir)
    if not root.exists():
        return docs
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() in {".md", ".txt"}:
            docs.extend(TextLoader(str(f), encoding="utf-8").load())
        elif f.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(f)).load())
    return docs


def split_documents(docs: list) -> list:
    """递归字符切分。chunk_size/overlap 在 .env 里调，是阶段2的实验变量。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    return splitter.split_documents(docs)
