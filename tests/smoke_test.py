"""冒烟测试：跑一次完整问答，验证 MVP 闭环。

用法: python tests/smoke_test.py
前提: 已运行 scripts/ingest.py，ollama 服务在线。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import RAGPipeline  # noqa: E402


def main():
    pipe = RAGPipeline()
    questions = [
        "什么是RAG？它解决什么问题？",       # 应命中知识库内容
        "本公司2030年在火星的分公司地址是？",  # 应回答"知识库中没有找到相关内容"
    ]
    for q in questions:
        print(f"\nQ: {q}")
        out = pipe.answer(q)
        print(f"A: {out['answer']}")
        print(f"引用: {[s['source'] for s in out['sources']]}")
        assert out["answer"], "答案为空，链路有问题"
    print("\n冒烟测试通过 ✓")


if __name__ == "__main__":
    main()
