"""环境配置：全部从 .env 读取，改参数不用动代码。"""
import os                             # os模块用于获取环境变量
from pathlib import Path
from dotenv import load_dotenv        # dotenv模块用于读取 .env 文件

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# Agent 链路模型（默认与 LLM_MODEL 相同）。
# 为什么单独留一个开关：Agent 依赖**原生 tool calling**，对模型工具协议遵循要求高。
# 实测（2026-10-06）：qwen2.5-coder 的工具调用是"文本模仿"式 —— 模板里明明写了
# <tool_call> 格式，它却输出裸 JSON / [name {json}] 等各种变体，格式不稳定，
# 29 题一度全军覆没（0 次工具调用）。qwen2.5:7b 这类指令模型才可靠。
# 固定链路（pipeline）继续用 LLM_MODEL，不受影响。
AGENT_LLM_MODEL = os.getenv("AGENT_LLM_MODEL", LLM_MODEL)

# 关键：把相对路径锚定到项目根目录（config.py 的上一级），
# 这样无论从 PowerShell、PyCharm 还是别的什么地方启动，路径都不会断。
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / os.getenv("DOCS_DIR", "data/docs")
INDEX_DIR = ROOT / os.getenv("INDEX_DIR", "storage/index")

# 切分参数 —— 阶段2调优的主要对象
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# 检索返回条数
TOP_K = int(os.getenv("TOP_K", "5"))

# ---- 阶段2：检索升级配置 ----
# vector = 只用向量检索（baseline）| hybrid = 向量+BM25 融合 | hybrid+rerank = 再加重排
RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "vector")
# 混合检索时每路各召回多少条，融合后再截到 TOP_K
RECALL_K = int(os.getenv("RECALL_K", "20"))
# RRF 融合的平滑常数（论文推荐 60，一般不用改）
RRF_K = int(os.getenv("RRF_K", "60"))
# 重排模型（需 pip install sentence-transformers，首次运行会下载约 1GB）
# 这里既可以填 HuggingFace 模型名，也可以填**模型目录** —— 本地路径按项目根解析。
# 为什么要支持本地目录：Windows 没开管理员 / 开发者模式时创建不了符号链接，
# HF 缓存的 snapshots/ 下会只剩一堆 0 字节空壳（blobs/ 里的权重其实完好），
# 加载时报「config.json is not a valid JSON file」。用
#     snapshot_download(..., local_dir="models/bge-reranker-base")
# 绕开符号链接机制即可。
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
_local_rerank = ROOT / RERANK_MODEL
if _local_rerank.is_dir():
    RERANK_MODEL = str(_local_rerank)
