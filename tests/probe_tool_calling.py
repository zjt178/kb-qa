"""探针：某个 ollama 模型到底能不能吐 tool_calls（Agent 依赖原生 tool calling）。

三种方式层层排查，每层只加一个变量：
  A. 裸 bind_tools + 一个"明显必须检索"的问题 —— 最小复现（langchain 路径）
  B. 换成系统提示里明确要求调工具 —— 排除提示词因素
  C. 原始 HTTP /api/chat —— 绕开 langchain，看 ollama 原生协议返回什么

跑法：
  python tests/probe_tool_calling.py                # 用 .env 里 LLM_MODEL
  python tests/probe_tool_calling.py qwen2.5:7b     # 指定模型
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_ollama import ChatOllama

from config import LLM_MODEL, OLLAMA_BASE_URL
from tools import build_search_tool

MODEL = sys.argv[1] if len(sys.argv) > 1 else LLM_MODEL

llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
tool = build_search_tool()
bound = llm.bind_tools([tool])

Q = "RRF 融合的 k 常数取多少？"


def _calls(msg):
    """把 tool_calls 归一化成 list（不同 langchain 版本形态不一）。"""
    tc = getattr(msg, "tool_calls", None) or []
    return list(tc)


print(f"模型: {MODEL}   问题: {Q}")
print("=" * 72)

# ---- A. 裸 bind_tools，不额外提示 ----
print("\n[A] bind_tools + 裸问题（不加任何提示）")
a = bound.invoke(Q)
print("  content:", repr(str(a.content)[:100]))
print("  tool_calls:", _calls(a) if _calls(a) else "（空）")
print("  additional_kwargs keys:", list(getattr(a, "additional_kwargs", {}).keys()))

# ---- B. 明确指令要求调用工具 ----
print("\n[B] bind_tools + 系统指令『必须调用 search_knowledge_base 工具』")
b = bound.invoke([
    ("system", "回答任何问题之前，你必须先调用 search_knowledge_base 工具检索资料。"),
    ("human", Q),
])
print("  content:", repr(str(b.content)[:100]))
print("  tool_calls:", _calls(b) if _calls(b) else "（空）")

# ---- C. 原始 HTTP：绕开 langchain，看 ollama API 本身返回什么 ----
print("\n[C] 原始 /api/chat（tools 走 ollama 原生协议）")
import urllib.request

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": Q},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "在企业知识库中检索与查询最相关的文档片段。",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ],
    "stream": False,
    "options": {"temperature": 0},
}
req = urllib.request.Request(
    OLLAMA_BASE_URL.rstrip("/") + "/api/chat",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read())
msg = data.get("message", {})
print("  message keys:", list(msg.keys()))
print("  tool_calls:", json.dumps(msg.get("tool_calls"), ensure_ascii=False) if msg.get("tool_calls") else "（空）")
print("  content:", repr(msg.get("content", "")[:200]))

print("\n结论:")
if _calls(a) or _calls(b) or msg.get("tool_calls"):
    print("  模型能吐 tool_calls —— 这个模型可用于原生 tool calling Agent")
else:
    print("  三种方式都拿不到 tool_calls —— %s 不支持（或没启用）工具调用" % MODEL)
    print("  需换支持 tools 的模型，或依赖 llm_adapter 的文本协议兜底")
