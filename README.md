# kb-qa — RAG 检索策略的量化评估与收益边界定位

一个可复现的中文 RAG 项目：多格式文档建库 → 检索 → 带引用溯源的回答 → FastAPI 服务 → **可量化的评估闭环**。

> **这个项目最值钱的不是"跑通了"，而是把一个反直觉结论量了出来：加 cross-encoder 精排，指标反而掉了。** 下面有完整的三模式对比、失效案例和归因。

## 实测结论（40 篇语料 / 122 块 / 29 题，TOP_K=5）

| 指标 | vector（baseline） | hybrid（RRF 融合） | hybrid+rerank |
|------|------|------|------|
| Hit@5（宽松，沾一个关键词） | 0.952 | **1.000** | 0.952 |
| Hit@5（严格，前 5 条覆盖全部关键词） | 0.857 | **0.905** | 0.810 |
| 平均关键词覆盖率 | 0.905 | **0.952** | 0.881 |
| MRR | **0.873** | 0.869 | 0.769 |
| 忠实度（LLM-as-judge，1-5） | 4.29 | 4.29 | 4.17 |
| 拒答准确率 | 1.0 | 1.0 | 1.0 |
| 平均延迟 | 2640 ms | 2910 ms | 8429 ms |

**结论**

1. **hybrid 是性价比最优解**：严格证据覆盖率 +4.8pt、关键词覆盖率 +4.7pt，忠实度完全持平，延迟无明显代价。
2. **cross-encoder 精排在 122 块这个规模上是负优化**：三项检索指标全线垫底，严格口径（0.810）甚至低于不做任何优化的 baseline；延迟涨到 3.2 倍。原因是 pointwise 判据会拆散多证据问题的证据块（见下）。
3. **这套语料上向量检索已接近天花板**：21 道关键词题中 **18 道三模式结果完全一致**，差异只出现在 3 道多证据题上。增益集中在字面术语型查询（如"k1 和 b 参数""RRF 的 k 取多少"）。

### 多证据题失效案例（#29：同样的问题为什么两次答案不同）

| 模式 | 关键词覆盖 | 说明 |
|------|------|------|
| vector | 1/2 | 只有一块证据进前五 |
| hybrid | **2/2** | 两块证据齐全且排第 1 |
| hybrid+rerank | **0/2** | 两块证据**全被踢出**前五 |

召回阶段两路各带偏好，RRF 只问"两边是否都认可"，多证据题的两块证据都能进前五。而 cross-encoder 把「问题 + **单个**块」拼起来打分，本质在问"这一个块自己能不能把问题答完"——只含半份答案的块被判为相关性不足。**hybrid 好不容易凑齐的证据，被精排亲手拆散了。** 两次独立运行均在同一题、同一位置复现，不是噪声。

> 全部细节见 `eval/report-vector.md`、`eval/report-hybrid.md`、`eval/report-hybrid_rerank.md`；被推翻的早期版本存档在 `eval/archive/`；每次运行的原始指标追加在 `eval/history.jsonl`。

## 架构

```
离线链路（建库）                        在线链路（问答）
data/docs/**/*.md|txt|pdf               用户提问
  │ 加载 + 清洗                            │
  ▼                                        ▼
RecursiveCharacterTextSplitter          混合检索（三模式可切）
  │ 切分                                   │  ├─ 向量检索（FAISS，语义）
  ▼                                        │  └─ BM25（jieba 分词，字面）
bge-small-zh 嵌入 ──► FAISS 索引 ────────► │        └→ RRF 融合 (k=60)
  │                                        │           └→ [可选] cross-encoder 精排
  └──► chunks.json（BM25/重排所需的原文）  ▼
                                        LLM 生成（带 [1][2] 引用标注 + 拒答围栏）
                                           │
                                           ▼
                                     答案 + 引用溯源（POST /ask）
                                           │
                                           ▼
                                 评估闭环：Hit@K / MRR / 覆盖率 / 忠实度 / 拒答 / 延迟
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
#    要用 hybrid+rerank 模式再加（会拉 torch，约 2GB）：
#    pip install -r requirements-rerank.txt

# 2. 确认 ollama 在跑，并拉模型
ollama pull qwen2.5:7b              # 生成模型（本机实测用的是 qwen2.5-coder:latest）
ollama pull qllama/bge-small-zh-v1.5 # 嵌入模型（512 维，25MB，本地友好）

# 3. 配置
cp .env.example .env                 # 按需改 LLM_MODEL / EMBED_MODEL / RETRIEVER_MODE

# 4. 语料：把 .md / .txt / .pdf 放进 data/docs/（仓库已带 40 篇演示语料）

# 5. 建库
python scripts/ingest.py --force

# 6. 验收：冒烟测试
python tests/smoke_test.py           # 两个问题都答对；第二个答"没找到"才算对

# 7. 起服务
uvicorn api.main:app --reload --port 8000
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？它解决什么问题？"}'
```

返回含 `answer`（带 [1][2] 标注）与 `sources`（引用了哪些原文块）。

## 检索模式切换

改 `.env` 里的 `RETRIEVER_MODE`，或用参数覆盖：

```bash
python eval/evaluate.py --mode vector         # baseline
python eval/evaluate.py --mode hybrid         # 向量 ∪ BM25 → RRF 融合
python eval/evaluate.py --mode hybrid+rerank  # 再叠 cross-encoder 精排
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `RETRIEVER_MODE` | `vector` | `vector` / `hybrid` / `hybrid+rerank` |
| `RECALL_K` | 20 | 每路召回条数，融合后再截到 `TOP_K` |
| `RRF_K` | 60 | RRF 平滑常数（论文推荐值） |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 可填 HF 模型名，也可填**本地目录**（相对项目根） |

> **为什么 RRF 而不是加权求和**：BM25 分数无上界，余弦相似度在 [-1,1]，量纲不同；且两者分布随语料漂移，权重没法稳定标定。RRF 只用**排名**做倒数求和，天然对量纲免疫。

## 评估口径

`eval/questions.jsonl` 每行一题，字段：

| 字段 | 作用 |
|------|------|
| `question` | 问题原文 |
| `keywords` | 该问题的必备关键词，用于算覆盖率 |
| `must_refuse` | 该题在语料里无依据，正确行为是拒答 |

指标：

- **Hit@K**：前 K 条里沾到任一关键词（宽松口径，容易虚高）
- **Hit@K(全词)**：前 K 条覆盖**全部**关键词（严格口径 ≈ 证据齐全，这是真正有区分度的指标）
- **关键词覆盖率**：平均命中几成关键词
- **MRR**：第一个正确结果的排名倒数（衡量排序质量，不只衡量"有没有"）
- **忠实度**：LLM-as-judge，答案是否完全基于检索内容；报告里同时输出有效样本数 `faithfulness_n`，**分母不齐时不要比较均值**
- **拒答准确率**：无依据问题是否真的拒答
- **延迟**：端到端单次问答耗时

> 自检技巧：`python eval/evaluate.py --mode vector --limit 3` 只跑 3 题，几十秒验证链路，再跑全量。

## 路线图

- [x] **阶段 1：RAG MVP** — 加载/切分/嵌入/入库 + 检索 + 生成 + 引用溯源 + FastAPI + 中文 Web UI（tag `v0.1-mvp`）
- [x] **阶段 2：检索优化与评估闭环** — 混合检索（BM25+向量，RRF）+ cross-encoder 精排 + 29 题评测集 + 三模式指标对比（tag `v0.2-retrieval`）
- [ ] 阶段 3：Agent 化 — 检索/SQL 封装为 tools + 自建 MCP Server（`agent/`）
- [ ] 阶段 4：README 打磨 + 指标数据 + 简历定稿

## 目录说明

```
kb-qa/
├── data/docs/              语料（18 篇领域文档 + distractors/ 跨域干扰 + samedomain/ 同域干扰）
├── src/                    核心代码
│   ├── config.py           环境配置 + 路径锚定项目根
│   ├── loaders.py          文档加载 + 切分
│   ├── embeddings.py       嵌入模型
│   ├── vectorstore.py      FAISS 索引（先建临时目录、成功才替换旧索引）
│   ├── chunk_store.py      chunks.json：BM25 / 重排所需的原文
│   ├── retriever.py        三模式检索器（vector / hybrid / hybrid+rerank）
│   ├── generator.py        系统提示词 + LLM + OutputParser（LCEL 链）
│   └── pipeline.py         RAG 主流程
├── api/main.py             FastAPI 服务
├── scripts/ingest.py       离线建库脚本
├── eval/                   评测集 + 评估脚本 + 三份报告 + history.jsonl + archive/
├── tests/smoke_test.py     冒烟测试
├── models/                 本地重排模型（不进 git）
├── storage/index/          FAISS 索引 + chunks.json（不进 git）
└── TODO.md                 看不懂代码清单 + 踩坑记录（每周更新）
```

## 开发约定

1. 每个阶段结束打一次 git tag（`v0.1-mvp` / `v0.2-retrieval`），repo 的演进史本身就是简历素材。
2. 看不懂的代码不要当场深挖，记进 `TODO.md`，周末统一处理。
3. **所有优化必须有前后指标对比，凭感觉的优化不算优化。**
4. 指标对不上时，先怀疑度量实现，再怀疑被测系统。
