"""FastAPI 服务。启动: uvicorn api.main:app --reload --port 8000"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import RAGPipeline

app = FastAPI(title="kb-qa 企业知识库问答", version="0.1.0")


@app.get("/", include_in_schema=False)
def home():
    """中文问答页面。"""
    return FileResponse(Path(__file__).resolve().parents[1] / "static" / "index.html")

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    id: int
    source: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """问答接口：返回带引用标注的答案 + 引用溯源列表。"""
    result = get_pipeline().answer(req.question, k=req.top_k)
    return AskResponse(**result)


app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parents[1] / "static"), html=True), name="static")
