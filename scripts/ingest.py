"""离线建库：data/docs/* -> 切分 -> 嵌入 -> FAISS 索引。

用法:
    python scripts/ingest.py            # 首次建库
    python scripts/ingest.py --force    # 删掉旧索引重建（改了切分参数后用）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loaders import load_documents, split_documents  # noqa: E402
from vectorstore import build_index  # noqa: E402
from chunk_store import save_chunks  # noqa: E402


def main():
    force = "--force" in sys.argv
    docs = load_documents()
    if not docs:
        print("data/docs/ 下没有 .md/.txt/.pdf 文件，先放语料进来。")
        return
    print(f"加载了 {len(docs)} 个文档片段，开始切分...")
    chunks = split_documents(docs)
    print(f"切分完成：{len(chunks)} 个块（chunk 参数见 .env）")
    build_index(chunks, force=force)
    save_chunks(chunks)  # 阶段2：给 BM25 / 重排留一份原始文本
    print(f"索引已保存到 storage/index/（含 chunks.json），可以启动服务了：uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
