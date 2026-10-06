"""批量实测：指定模型 + 适配层的**工具调用成功率**。

这是「29 题全部工具×0」问题的验收脚本：
  对每道题跑一次完整 Agent，记录：是否调了工具、调了几次、答案拿到了没。

跑法：
  python tests/probe_agent_batch.py                # 用 .env 里 AGENT_LLM_MODEL
  python tests/probe_agent_batch.py qwen2.5-coder:latest
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL = sys.argv[1] if len(sys.argv) > 1 else None
if MODEL:
    import os
    os.environ["AGENT_LLM_MODEL"] = MODEL

from agent import RagAgent  # noqa: E402  (必须在设好环境变量后导入)
from config import AGENT_LLM_MODEL  # noqa: E402

# 8 道代表性题：3 道术语型（必须靠检索）、2 道拒答题（应检索后拒答）、
# 2 道多证据题（需 2 次检索）、1 道寒暄（不应检索）
CASES = [
    ("术语型", "RRF 融合的 k 常数取多少？"),
    ("术语型", "BM25 里的 k1 和 b 参数是干什么的？"),
    ("术语型", "文档分块的时候 chunk size 一般怎么定？"),
    ("拒答题", "公司食堂今天中午的菜单是什么？"),
    ("拒答题", "帮我订一张明天去上海的高铁票"),
    ("多证据", "我们的检索链路用了哪几种召回，它们各自负责什么？"),
    ("多证据", "为什么加了重排之后指标反而下降了？"),
    ("寒暄", "你好"),
]

print(f"模型: {AGENT_LLM_MODEL}")
print("=" * 78)
agent = RagAgent()

ok = 0
for i, (kind, q) in enumerate(CASES, 1):
    t0 = time.time()
    try:
        run = agent.answer(q)
        dt = time.time() - t0
        n = run["tool_calls"]
        ans = run["answer"][:60].replace("\n", " ")
        mark = "OK " if n > 0 or kind == "寒暄" else "BAD"
        if mark == "OK ":
            ok += 1
        print(f"[{mark}] {i}. ({kind}) {q}")
        print(f"        工具×{n}  来源×{len(run['sources'])}  {dt:.1f}s")
        print(f"        答: {ans}")
    except Exception as e:
        print(f"[ERR] {i}. ({kind}) {q}")
        print(f"        {type(e).__name__}: {e}")

print("=" * 78)
print(f"工具调用正常: {ok}/{len(CASES)}")
