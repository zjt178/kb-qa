# kb-qa — 企业知识库问答系统

RAG 在线链路实战项目：多格式文档入库 → 检索 → 带引用溯源的回答 → FastAPI 服务 → 评估闭环。

## 架构

```
离线链路（建库）                     在线链路（问答）
data/docs/*.md|pdf                  用户提问
  │ 加载 + 清洗                        │
  ▼                                   ▼
RecursiveCharacterTextSplitter ──► 混合检索（阶段2：BM25+向量+重排）
  │ 切分                              │
  ▼                                   ▼
bge-m3 嵌入 ──► FAISS 索引 ──────► LLM 生成（带 [1][2] 引用标注）
                                      │
                                      ▼
                                 答案 + 引用溯源（/ask 接口）
                                      │
                                      ▼
                                 RAGAS 评估 + LangSmith 追踪（阶段2）
```

## 快速开始（阶段 1 验收线：curl 调 /ask，答案引用指向正确原文）

```bash
# 1. 安装依赖（建议先建 venv）
pip install -r requirements.txt

# 2. 确认 ollama 在跑，且拉了两个模型
ollama pull qwen2.5:7b
ollama pull bge-m3

# 3. 复制环境配置
cp .env.example .env

# 4. 放语料：把你的 .md / .txt / .pdf 丢进 data/docs/
#    （仓库里放了一份 sample_doc.md，可以直接先用它测通）

# 5. 建库（离线链路）
python scripts/ingest.py

# 6. 起服务（在线链路）
uvicorn api.main:app --reload --port 8000

# 7. 验收
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？它解决什么问题？"}'
```

返回应包含 `answer`（带 [1][2] 标注）和 `sources`（引用了哪些原文块）。

## 路线图

- [x] 阶段 1：RAG MVP（本项目初始代码）
- [ ] 阶段 2：混合检索（BM25 + 向量）+ bge-reranker 重排 + RAGAS 评估（`eval/`）
- [ ] 阶段 3：Agent 化 —— 检索/SQL 查询封装为 tools + 自建 MCP Server（`agent/`）
- [ ] 阶段 4：README 打磨 + 指标数据 + 简历定稿

## 目录说明

```
kb-qa/
├── data/docs/          语料目录（.md/.txt/.pdf）
├── src/                核心代码
│   ├── config.py       环境配置
│   ├── loaders.py      文档加载 + 切分
│   ├── embeddings.py   嵌入模型
│   ├── vectorstore.py  FAISS 索引（阶段2可换 Milvus）
│   ├── generator.py    提示词 + LLM + OutputParser
│   └── pipeline.py     RAG 主流程
├── api/main.py         FastAPI 服务
├── scripts/ingest.py   离线建库脚本
├── eval/               测试集 + 评估（阶段2）
├── tests/smoke_test.py 冒烟测试
└── TODO.md             看不懂代码清单 + 踩坑记录（每周更新）
```

## 开发约定

1. 每个阶段结束打一次 git tag（如 `v0.1-mvp`），repo 的演进史本身就是简历素材。
2. 看不懂的代码不要当场深挖，记进 `TODO.md`，周末统一处理。
3. 所有优化必须有前后指标对比，凭感觉的优化不算优化。
