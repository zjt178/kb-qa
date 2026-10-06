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
- [x] **阶段 3：Agent 化与能力服务化** — 检索封装为 LangChain Tool + 多步 Agent + 自建 MCP Server；量化出「此规模下 Agent 无检索增益、延迟 3.5×」的**收益边界**（tag `v0.3-agent`）
- [ ] 阶段 4：README 打磨 + 指标数据 + 简历定稿

## 阶段 3：把检索能力服务化

阶段 1/2 的检索是**固定链路**：检索一定发生、只发生一次，query 就是用户原话 —— 用户问得含糊，检索跟着烂。阶段 3 把「要不要检索、检索什么、检索几次」交给模型决定。

### 实测结论：Agent 化在这个规模上没有检索增益，代价是 3 倍延迟

同一天、同一份语料、同一套 29 题，两个链路正面对照（`hybrid` 检索，TOP_K=5）：

| 指标 | hybrid（固定链路） | hybrid + Agent | 差异 |
|------|------|------|------|
| Hit@5（宽松） | 1.000 | 1.000 | 持平 |
| **Hit@5（严格全词）** | **0.905** | **0.905** | **完全一致** |
| 平均关键词覆盖率 | 0.952 | 0.952 | **完全一致** |
| MRR | 0.869 | 0.845 | −2.4pt |
| 忠实度（LLM-as-judge，n=24） | 4.25 | 4.33 | +0.08 |
| 拒答准确率 | 1.0 | 1.0 | 持平 |
| **平均延迟** | **3436 ms** | **10142 ms** | **2.95×** |
| 平均工具调用 | — | 1.0 | — |

**怎么读这张表**

1. **检索层三项指标完全一致**，忠实度还略高一点 —— 说明 Agent 链路**没有把质量做坏**，工程上是可靠的。
2. **但也没有变好**，而延迟涨到近 3 倍。根因在 `平均工具调用 = 1.0`：Agent 每次都只检索一次、query 就是用户原话 —— **它实际退化成了固定链路，只是多花了一轮模型调用**。
3. **为什么模型不改写 query**：本项目的 middleware 为了保证「一定会检索」，在首次模型调用前就替模型把检索做完了。这是**有意用"必然检索"换"稳定"，代价是放弃了 query 改写**。若要吃到 Agent 的改写增益，应改为**让模型先自主决定、只在它偷懒时兜底**（见 `middleware.py` 的 `force_retrieval=False` 开关）。
4. **天花板不在 Agent，在语料规模**：阶段 2 已量化出「29 题中 26 题三模式结果一致」—— 122 块的语料上向量检索已接近上限。Agent 要产生增益，前提是**语料大到单次检索必然漏**（需要多轮补检）。

> **这条结论比"我做了个 Agent"值钱**：它回答了「**什么时候不该上 Agent**」。多步推理的收益上界，由「单次检索的召回缺口」决定 —— 缺口为零时，Agent 只有成本没有收益。

三个新增件：

| 文件 | 作用 |
|------|------|
| `src/tools.py` | 检索能力封装为 LangChain Tool（`search_knowledge_base`），含工具描述设计与结果截断 + 逐块引用溯源 |
| `src/agent.py` | 工具调用型 Agent（langchain 1.x `create_agent`），多步推理 + 引用溯源 + 轨迹记录 |
| `src/middleware.py` | 强制检索 middleware：首次模型调用前替模型完成检索，把"要不要检索"变成结构保障 |
| `src/llm_adapter.py` | LLM 适配层：把文本形态的工具调用归一化为标准 `tool_calls`（安全阀 + 去重） |
| `scripts/mcp_server.py` | 自建 MCP Server（stdio），把同一套检索能力经标准协议暴露出去 |

> **langchain 1.x 的 API 变更**：1.x 移除了 0.x 的 `create_react_agent` / `AgentExecutor`（网上多数 ReAct 教程仍是旧 API，照抄会 ImportError）。1.x 用 `create_agent(model, tools, system_prompt=)` 返回一张图，`invoke({"messages": [...]})`；循环靠模型**原生 tool calling** 驱动，不再依赖正则解析 `Action:` 文本协议，稳定性明显更好。

**为什么要做成 MCP 而不是内部函数**：内部函数只能被当前应用调用；想从 Claude Desktop / Cursor / 另一个 Agent 用同一套知识库就得复制代码。MCP 把能力标准化成「服务」，检索逻辑只有一份。

**工具描述是选择准确率的第一变量**。`description` 不能只写「检索知识库」——模型不知道什么算需要检索。必须写清三件事：能做什么、什么时候**必须**用、什么时候**不要**用。

### Agent 的隐形天花板：模型的工具协议遵循能力

**踩坑实录**：阶段3 首次跑通后，29 题**全部「工具×0」**，Hit@K 全 0 —— 代码无报错，但 Agent 一次都没调工具。三层剥离定位：

| 层 | 验证方法 | 结论 |
|---|---|---|
| ① 代码是否透传 tools | 绕开 langchain，用原始 HTTP 打 ollama `/api/chat` 带 `tools` 字段 | 能收到响应 → **透传没问题** |
| ② 模型吐的是结构化还是文本 | `llm.bind_tools([t]).invoke(q)` 看 `tool_calls` 与 `content` | `tool_calls: []`，工具调用被写成**纯文本 JSON** |
| ③ 模型模板是否声明协议 | `ollama show <model> --template` | 模板里**有** `<tool_call>` 段却仍不遵守 → **模型能力问题** |

**根因**：`qwen2.5-coder` 是代码补全向的变体，工具协议遵循弱，输出格式五花八门（裸 JSON / `[name {json}]` / `<tool_call>` 标签）。而 langchain 1.x 的 agent 图**只认结构化 `AIMessage.tool_calls`**，看不见就直接结束循环。

**消融实验（8 题探针：术语型 3 / 拒答型 2 / 多证据 2 / 寒暄 1）**

| 配置 | 工具调用正常 | 失败模式 |
|---|---|---|
| `qwen2.5-coder:latest` + 适配层 | **4/8** | **不检索反而反问用户**（"请提供具体的指标名称"）——意愿问题，适配层救不了 |
| `qwen2.5:7b` | **6/8** | **不检索就作答 + 编造引用标记**（实测出现 `ronics[1] 提到...`，`[1]` 是凭空捏的） |
| `qwen2.5:7b` + **强制检索 middleware** | **8/8** | — |

**三层修复与渐进验证**：

1. `src/llm_adapter.py`：把三种文本变体归一化为标准 `tool_calls`（带"name 必须在工具名单里"的安全阀 + 去重）。**但它只救格式，救不了意愿**——coder 实测仅 4/8。
2. **换模型** `AGENT_LLM_MODEL=qwen2.5:7b`：指令模型原生支持 tools（探针 3/3 吐结构化）。升到 6/8。
3. `src/middleware.py`：**强制检索 middleware**（`before_model` 钩子）——首次模型调用前若尚无 ToolMessage 且问题非寒暄，**替模型把检索做完并注入结果**。升到 8/8。

> **这条比指标更值钱**：① **Agent 的天花板首先取决于模型的工具协议遵循能力**，其次才是提示词与工程；② **能用结构保证的，不要指望提示词**——提示词只能"请求"模型，middleware 才能"替它做完"。
>
> 还有一个**提示词副作用**的坑值得记：为防幻觉写的拒答条款「资料不足时回答『知识库中没有找到相关内容』」，被模型当成了**固定开头模板**，出现"先写没找到、再补正确答案"的矛盾答案。修法是条件化措辞 + 显式禁止「两者都写」。**看到"先否后肯"的矛盾答案，先怀疑提示词模板，不要以为是检索失败。**

跑法：

```bash
# Agent 端到端评估（需要 ollama 在线）
python eval/evaluate.py --mode hybrid --agent

# MCP Server（由客户端拉起，一般不手动跑）
python scripts/mcp_server.py
# 调试用 Inspector：
#   npx @modelcontextprotocol/inspector .venv/Scripts/python.exe scripts/mcp_server.py

# 离线验收（不需要 ollama，共 32 项断言）
python tests/agent_offline_test.py   # 工具定义 / BM25 通路 / RRF / 来源提取 / MCP 协议层实调
python tests/agent_parse_test.py     # Agent 消息解析 + 适配层三格式解析（21 项）

# 模型工具协议探针（三层剥离定位"Agent 不调工具"）
python tests/probe_tool_calling.py qwen2.5:7b

# 8 题端到端批量实测（术语/拒答/多证据/寒暄四类，统计工具调用成功率）
python tests/probe_agent_batch.py
```

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
│   ├── tools.py            检索能力 → LangChain Tool（工具描述 + 结果截断 + 来源提取）
│   ├── agent.py            工具调用型 Agent（1.x create_agent，多步调用 + 引用溯源）
│   ├── middleware.py       强制检索 middleware（首次调用前替模型完成检索，结构保障）
│   ├── llm_adapter.py      LLM 适配层（文本形态工具调用 → 标准 tool_calls）
│   ├── generator.py        系统提示词 + LLM + OutputParser（LCEL 链）
│   └── pipeline.py         RAG 主流程
├── api/main.py             FastAPI 服务
├── scripts/ingest.py       离线建库脚本
├── scripts/mcp_server.py   自建 MCP Server（stdio，3 个工具）
├── eval/                   评测集 + 评估脚本 + 三份报告 + history.jsonl + archive/
├── tests/smoke_test.py     冒烟测试
├── tests/agent_offline_test.py  阶段3 离线验收：工具/MCP 协议层（不需要 ollama）
├── tests/agent_parse_test.py    阶段3 离线验收：Agent 消息解析逻辑
├── models/                 本地重排模型（不进 git）
├── storage/index/          FAISS 索引 + chunks.json（不进 git）
└── TODO.md                 看不懂代码清单 + 踩坑记录（每周更新）
```

## 开发约定

1. 每个阶段结束打一次 git tag（`v0.1-mvp` / `v0.2-retrieval`），repo 的演进史本身就是简历素材。
2. 看不懂的代码不要当场深挖，记进 `TODO.md`，周末统一处理。
3. **所有优化必须有前后指标对比，凭感觉的优化不算优化。**
4. 指标对不上时，先怀疑度量实现，再怀疑被测系统。
