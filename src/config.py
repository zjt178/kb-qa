"""环境配置：全部从 .env 读取，改参数不用动代码。"""
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")
INDEX_DIR = os.getenv("INDEX_DIR", "storage/index")

# 切分参数 —— 阶段2调优的主要对象
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# 检索返回条数
TOP_K = int(os.getenv("TOP_K", "5"))
