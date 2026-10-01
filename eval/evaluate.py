"""阶段2评估脚本：跑测试集 → 算指标 → 出报告，用于"优化前 vs 优化后"对比。

用法:
    python eval/evaluate.py                    # 用 .env 里的 RETRIEVER_MODE
    python eval/evaluate.py --mode vector      # 跑 baseline
    python eval/evaluate.py --mode hybrid      # 跑混合检索

指标（自建轻量版，不依赖 RAGAS 的重依赖）：
    检索层   Hit@K      前 K 条里有没有命中关键词（关键词命中 ≈ 找到了对的资料）
             MRR        第一个命中的位置倒数，衡量"排得够不够前"
    生成层   忠实度      LLM 打分 1-5：答案是否完全基于检索到的内容（防幻觉）
             拒答准确率  该拒答的问题，是否真的拒答了
    工程层   平均延迟

测试集格式 eval/questions.jsonl，每行一个 JSON：
    {"question": "...", "ground_truth": "...", "keywords": ["词1", "词2"]}
    {"question": "知识库里没有的问题", "must_refuse": true}
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 注意：必须在 import config 之前改环境变量，因为 config 在导入时就读 .env
for i, arg in enumerate(sys.argv):
    if arg == "--mode" and i + 1 < len(sys.argv):
        os.environ["RETRIEVER_MODE"] = sys.argv[i + 1]

from config import RETRIEVER_MODE, TOP_K  # noqa: E402
from pipeline import RAGPipeline  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
# 用完整 mode 名做文件名：split('+')[0] 会让 hybrid+rerank 覆盖掉 hybrid 的报告，
# 而三模式对比恰恰最需要这两份同时在。
REPORT = ROOT / "eval" / f"report-{RETRIEVER_MODE.replace('+', '_')}.md"
HISTORY = ROOT / "eval" / "history.jsonl"

JUDGE_PROMPT = """你是一个严格的评估员。请判断【答案】是否完全基于【参考资料】，有没有编造参考资料之外的内容。

评分标准：
5 = 完全基于资料，且有引用
4 = 基本基于资料，个别措辞超出但不影响事实
3 = 部分内容找不到资料依据
2 = 有明显编造
1 = 基本是编造

参考资料：
{context}

答案：
{answer}

请只输出一个 1-5 的整数，不要输出任何文字、标点或解释。你的评分："""

# judge 不守格式时的兜底词表。顺序要紧：先判否定（"不完全基于"里也含"完全基于"）。
_SCORE_HINTS = (
    (1, ("基本是编造", "完全是编造")),
    (3, ("部分内容找不到", "不完全基于", "部分基于", "有些内容没有")),
    (5, ("完全基于", "没有编造", "均有依据")),
)


def parse_judge_score(raw) -> int:
    """从 judge 的输出里抠出 1-5 的分数。

    本地小模型经常不守格式 —— 直接回一句「【答案】完全基于【参考资料】。」，
    重试三次也未必吐数字。与其把这条样本丢掉、让忠实度的分母在几次运行之间
    悄悄漂移（22 / 22 / 23 就是这么来的），不如从措辞兜底推断：
    宁可分数粗糙，也不要样本集不一致。
    """
    text = str(raw)
    m = re.search(r"[1-5]", text)
    if m:
        return int(m.group())
    for score, hints in _SCORE_HINTS:
        if any(h in text for h in hints):
            return score
    return 0

REFUSE_WORDS = ("没有找到", "未找到", "没有相关", "无法回答", "无法提供", "没有提到", "不包含", "知识库中")


def load_questions() -> list[dict]:
    if not QUESTIONS.exists():
        return []
    return [
        json.loads(line)
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def hit_at_k(docs: list, keywords: list) -> tuple[bool, int, float]:
    """返回 (是否命中任一关键词, 命中位置1-based，0 表示未命中, 关键词覆盖率 0-1)。

    coverage=1.0 表示"证据齐全"：检索结果里所有关键词都能找到。
    coverage 低说明只沾到边、缺证据 —— 这是比"沾边即命中"严格得多的口径。
    """
    if not keywords or not docs:
        return (False, 0, 0.0)
    merged = "\n".join(d.page_content for d in docs)
    covered = [kw for kw in keywords if kw in merged]
    if not covered:
        return (False, 0, 0.0)
    rank = 0
    for i, d in enumerate(docs, 1):
        if any(kw in d.page_content for kw in keywords):
            rank = i
            break
    return (True, rank, len(covered) / len(keywords))


def judge_faithfulness(llm, context: str, answer: str, retries: int = 2) -> int:
    """让本地模型给忠实度打分（LLM-as-judge），失败自动重试。

    为什么要重试：返回 0 的条目会被 main() 排除出 faithful 列表，
    于是「忠实度平均分」的分母在两次运行之间悄悄变了 —— 看起来在比同一个指标，
    实际比的是不同的样本集。这是评估里最隐蔽的一类脏数据，必须堵住。
    """
    msg = JUDGE_PROMPT.format(context=context[:4000], answer=answer)
    for attempt in range(retries + 1):
        try:
            raw = llm.invoke(msg).content
        except Exception as e:  # 评估不能因为一次调用失败就整体中断
            print(f"   [judge 调用失败 {attempt + 1}/{retries + 1}] {e}")
            continue
        score = parse_judge_score(raw)
        if score:
            return score
        print(f"   [judge 输出无法解析 {attempt + 1}/{retries + 1}] {str(raw)[:40]!r}")
    return 0


def main():
    questions = load_questions()
    if not questions:
        print("缺少 eval/questions.jsonl，先按模板补 20-30 条测试问题。")
        return

    llm = None
    needs_judge = any(q.get("ground_truth") for q in questions)
    if needs_judge:
        from langchain_ollama import ChatOllama

        from config import LLM_MODEL, OLLAMA_BASE_URL

        llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    pipe = RAGPipeline()
    print(f"检索模式: {RETRIEVER_MODE}  测试集: {len(questions)} 条  TOP_K={TOP_K}\n")

    rows, hits, rr, faithful, latencies = [], 0, 0.0, [], []
    right_refuse, should_refuse = 0, 0
    full_cover, cover_sum = 0, 0.0

    limit = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])
    if limit:
        questions = questions[:limit]
        print(f"[--limit {limit}] 只跑前 {limit} 条（自检用，别当正式结果）")

    for i, q in enumerate(questions, 1):
        question = q["question"]
        keywords = q.get("keywords", [])
        t0 = time.time()
        out = pipe.answer(question)
        latency = round((time.time() - t0) * 1000)
        latencies.append(latency)

        # 检索层：拿内部命中的文档块做判定（重新检索一次，不额外调用大模型）
        docs = pipe.retrieve(question, TOP_K)
        hit, rank, coverage = hit_at_k(docs, keywords)
        if keywords:
            hits += int(hit)
            rr += (1.0 / rank) if hit else 0.0
            full_cover += int(coverage >= 1.0)
            cover_sum += coverage

        # 生成层
        if q.get("must_refuse"):
            should_refuse += 1
            ok_refuse = any(w in out["answer"] for w in REFUSE_WORDS)
            right_refuse += int(ok_refuse)
            verdict = "拒绝成功" if ok_refuse else "⚠ 该拒没拒"
        else:
            context = "\n\n".join(d.page_content for d in docs)
            score = judge_faithfulness(llm, context, out["answer"]) if llm else 0
            if score:
                faithful.append(score)
            verdict = f"忠实度 {score}" if score else "-"

        rows.append(
            {
                "no": i,
                "question": question,
                "hit": hit,
                "rank": rank,
                "coverage": coverage,
                "n_kw": len(keywords),
                "verdict": verdict,
                "latency_ms": latency,
                "answer_preview": out["answer"][:60].replace("\n", " "),
            }
        )
        cov_txt = f"覆盖 {round(coverage * len(keywords))}/{len(keywords)}" if keywords else ""
        print(f"{i:>3}. {'命中' if hit else ('拒答' if q.get('must_refuse') else '未命中')}  {cov_txt:<9} {verdict:<12} {latency:>6}ms  {question[:26]}")

    n_kw = sum(1 for q in questions if q.get("keywords"))
    summary = {
        "mode": RETRIEVER_MODE,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "questions": len(questions),
        "hit_at_k": round(hits / n_kw, 3) if n_kw else None,
        "hit_all_kw": round(full_cover / n_kw, 3) if n_kw else None,
        "avg_kw_coverage": round(cover_sum / n_kw, 3) if n_kw else None,
        "mrr": round(rr / n_kw, 3) if n_kw else None,
        "faithfulness": round(sum(faithful) / len(faithful), 2) if faithful else None,
        # 把分母也记下来：两次运行的忠实度只有在样本数一致时才真正可比
        "faithfulness_n": len(faithful),
        "refuse_accuracy": round(right_refuse / should_refuse, 3) if should_refuse else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies)),
    }

    lines = [
        f"# 评估报告 · {summary['mode']}",
        "",
        f"- 时间：{summary['time']}",
        f"- 测试集：{summary['questions']} 条（含关键词 {n_kw} 条 / 应拒答 {should_refuse} 条）",
        f"- TOP_K={TOP_K}",
        "",
        "| 指标 | 数值 | 含义 |",
        "|------|------|------|",
        f"| Hit@{TOP_K} | {summary['hit_at_k']} | 前 {TOP_K} 条里至少沾到一个关键词的比例（宽松口径） |",
        f"| Hit@{TOP_K}(全词) | {summary['hit_all_kw']} | 前 {TOP_K} 条覆盖了**全部**关键词的比例（严格口径 = 证据齐全） |",
        f"| 关键词覆盖率 | {summary['avg_kw_coverage']} | 平均每个问题命中了几成关键词 |",
        f"| MRR | {summary['mrr']} | 正确资料排得越靠前越高（满分 1） |",
        f"| 忠实度 | {summary['faithfulness']} | 1-5 分，答案是否完全基于检索内容（有效样本 {summary['faithfulness_n']} 条） |",
        f"| 拒答准确率 | {summary['refuse_accuracy']} | 该拒答的问题真的拒答了 |",
        f"| 平均延迟 | {summary['avg_latency_ms']} ms | 单次问答耗时 |",
        "",
        "## 逐条明细",
        "",
        "| # | 命中 | 排名 | 覆盖 | 判定 | 延迟 | 问题 |",
        "|---|------|------|------|------|------|------|",
    ]
    for r in rows:
        cov = f"{round(r['coverage'] * r['n_kw'])}/{r['n_kw']}" if r["n_kw"] else "—"
        lines.append(
            f"| {r['no']} | {'✓' if r['hit'] else '—'} | {r['rank'] or '—'} | {cov} | {r['verdict']} | {r['latency_ms']}ms | {r['question'][:30]} |"
        )

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print("\n—— 汇总 ——")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n报告已写入 {REPORT}")
    print(f"历史指标追加到 {HISTORY}（换模式再跑一次，就有 before/after 对比了）")


if __name__ == "__main__":
    main()
