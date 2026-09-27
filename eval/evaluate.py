"""阶段2评估脚本（当前版本：跑测试集 + 存结果，RAGAS 打分留接口）。

用法: python eval/evaluate.py
测试集: eval/questions.jsonl，每行 {"question": "...", "ground_truth": "..."}
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import RAGPipeline  # noqa: E402


def load_questions() -> list[dict]:
    path = ROOT / "eval" / "questions.jsonl"
    if not path.exists():
        print("缺少 eval/questions.jsonl，先按模板补 20-30 条测试问题。")
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    questions = load_questions()
    if not questions:
        return
    pipe = RAGPipeline()
    results = []
    for q in questions:
        t0 = time.time()
        out = pipe.answer(q["question"])
        results.append(
            {
                "question": q["question"],
                "ground_truth": q.get("ground_truth", ""),
                "answer": out["answer"],
                "retrieved_sources": [s["source"] for s in out["sources"]],
                "latency_ms": round((time.time() - t0) * 1000),
            }
        )
        print(f"OK  {q['question'][:30]}  ({results[-1]['latency_ms']}ms)")

    out_path = ROOT / "eval" / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n共 {len(results)} 条，结果已存 {out_path}")
    print("TODO(阶段2): 接入 RAGAS 计算 faithfulness / answer_relevancy / context_precision，")
    print("             与优化前 baseline 对比，产出指标表 —— 这是简历的核心数字。")


if __name__ == "__main__":
    main()
