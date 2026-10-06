# TODO — 看不懂代码清单 & 踩坑记录

> 用法：项目里遇到看不懂的代码/报错，**先记到这里继续往前推**，周末统一处理。
> 处理完的条目写一行"结论"，这份文档本身就是学习记录 + 面试素材。

## 看不懂清单

| # | 位置（文件:行） | 问题描述 | 状态 | 结论 |
|---|----------------|---------|------|------|
| 1 | src/generator.py:34 | `\|` 管道符把 prompt/llm/parser 串起来，底层是怎么调度的？（Runnable 协议） | ☑ | LCEL 语法：`a \| b` 实际调用 `a.__or__(b)`，LangChain 组件都实现了 `__or__`，返回 `RunnableSequence`。数据流：dict → prompt 填模板 → llm 产出 AIMessage → parser 抽出 str。免费获得 stream/batch/异步能力 |
| 2 | src/vectorstore.py:24 | `allow_dangerous_deserialization=True` 为什么危险，FAISS 存的是什么 | ☑ | FAISS 落盘两个文件：index.faiss（纯向量+索引结构）、index.pkl（pickle 序列化的 id→原文+metadata 账本）。加载 pkl 等于执行反序列化，恶意构造的 pkl 可执行任意代码，所以只对自己建的索引开这个开关 |
| 3 | src/retriever.py | RRF 为什么用"排名倒数求和"而不是把两路分数加权平均？ | ☑ | BM25 分数无上界、余弦相似度在 [-1,1]，量纲不同不能直接相加；两者分布还随语料漂移，权重标定不出来。RRF 只用**名次**算 `1/(k+rank)`，对量纲和异常值免疫，代价是丢掉了"分数差多少"的信息 |
| 4 | src/retriever.py:rerank | cross-encoder 理论上更强，为什么实测反而更差？ | ☑ | 判据不同：RRF 问"两路是否都认可这块"，cross-encoder 把「问题+**单个**块」拼起来打分，问的是"这一块自己能不能答完问题"。多证据题各块只含半份答案，于是被精排判为低相关踢出。**精排的粒度和答案的粒度不匹配时，越准的模型错得越狠** |
| 5 | src/chunk_store.py | BM25 和重排都需要原文，为什么不能直接从 FAISS 里取？ | ☑ | FAISS 的 `index.pkl` 里确实存了原文，但那是 pickle 序列化的"账本"，读出要反序列化、还要处理 id 映射；单独落一份 chunks.json 让检索层不依赖向量库内部结构，换 Milvus / 换索引方案时检索层不用动 |

## 踩坑记录

| 日期 | 现象 | 原因 | 解决 |
|------|------|------|------|
| 2026-09-27 | `pip install -r requirements.txt` 报 UnicodeDecodeError: 'gbk' codec | requirements.txt 里有中文注释，Windows 中文系统的 pip 默认按 GBK 编码读文件，UTF-8 中文解码失败 | 已修复：requirements.txt 改为纯 ASCII（英文注释）。教训：Windows 下给 pip 读的文件不要写中文 |
| 2026-09-30 | `ingest.py --force` 报 `Failed to connect to Ollama` | Ollama 服务没启动（os error 10061 = 连接被拒） | 打开 Ollama 后重跑。教训：连接类报错先查服务，代码类报错才看 Traceback |
| 2026-10-01 | **`ingest.py --force` 建库失败后，旧索引被删光了** | `build_index()` 原来是"先 rmtree 删旧索引、再嵌入建新索引"。嵌入这一步依赖 ollama，它一失败就变成"旧的没了、新的也没建成"，索引全丢（连 chunks.json 一起） | 已修复 `vectorstore.py`：改为**先在临时目录建好新索引，成功后才删旧的**（build → 存 tmp → rmtree 旧 → rename）。已实测验证：ollama 挂掉时旧索引完整保留。教训：**破坏性操作要放在可能失败的操作之后**，永远不要"先删后建" |
| 2026-10-01 | `--mode hybrid+rerank` 报缺模块，补完一个又缺下一个：先 `scikit-learn`，再 `mpmath` | `sentence-transformers` 的依赖声明不全（torch→sympy→mpmath 这条链在实际导入时才触发） | 已装齐，并把重排的可选依赖单独列成 `requirements-rerank.txt`。教训：**重型可选依赖单独一个 requirements 文件**，主链路别被它拖累 |
| 2026-10-01 | **重排模型下载"成功"了，一加载就报 `config.json is not a valid JSON file`** | 缓存里 `snapshots/` 下全是 **0 字节空壳**，但 `blobs/` 里权重是完好的（1.06GB）。根因：Windows 无管理员权限 / 未开开发者模式时**创建不了符号链接**，HF 的 `snapshots/` 靠软链指向 `blobs/`，软链建不起来就只剩空文件 | 两条路：① `snapshot_download(..., local_dir=models/xxx)` 绕开软链机制；② 本地复制也被安全策略拦时，按文件头把 `blobs/` 里的哈希文件**认出来手工改名复制**。已在 `config.py` 支持 `RERANK_MODEL` 填本地目录。教训：**"下载成功"不等于"能用"**，装完模型必须真的加载一次跑个 predict |
| 2026-10-01 | **差点得出错误结论：修复前 vector 忠实度 4.86 vs hybrid 4.55，看着像"hybrid 拖低了忠实度"** | judge 被要求"只输出一个 1-5 的数字"，qwen2.5-coder 却回「【答案】完全基于【参考资料】。」，正则匹配不到数字 → 返回 0 并被静默丢弃。两次运行丢的不是同一批题，分母从 22 变到 24，均值就飘了 | 已修复：judge 加重试 + `parse_judge_score()` 措辞兜底（注意"先否定后肯定"，否则「不完全基于」会被误判成 5）+ 报告暴露 `faithfulness_n`。修复后 vector 4.29 / hybrid 4.29，完全持平。教训：**指标对不上时，先怀疑度量实现，再怀疑被测系统；均值必须连分母一起看** |
| 2026-10-01 | `--mode hybrid+rerank` 跑完，`report-hybrid.md` 被覆盖了 | 报告路径写成 `f"report-{RETRIEVER_MODE.split('+')[0]}.md"`，`hybrid+rerank` 切出 `hybrid`，把上一种模式的报告盖掉 | 已修复为 `RETRIEVER_MODE.replace('+', '_')`。教训：**模式名当文件名要整体转义**，别只取第一段 |
| 2026-10-06 | 照文档 `from mcp.server.fastmcp import FastMCP` 直接 `ImportError: No module named 'mcp.server.fastmcp'` | 装的是 **mcp 2.3.0**，官方 SDK 到 2.x 把 `FastMCP` 改名为 `mcp.server.mcpserver.MCPServer`（网上教程和 1.x 文档全是旧名字） | `scripts/mcp_server.py` 写成**兼容分支**：先试 `MCPServer`，ImportError 再退回 `FastMCP`。教训：**照抄教程前先看装的是哪个大版本**，2.x 的 `call_tool` 返回 `CallToolResult` 而不是元组 |
| 2026-10-06 | **`from langchain.agents import AgentExecutor, create_react_agent` 直接 ImportError** —— 本机装的是 langchain **1.4.2** | langchain 1.x **移除了** 0.x 时代的 `create_react_agent` / `AgentExecutor`（网上 90% 的 ReAct 教程还是这套 API）。1.x 只剩 `create_agent`，且调用方式完全变了 | 改用 1.x API：`create_agent(model=llm, tools=[...], system_prompt=...)` → `graph.invoke({"messages":[{"role":"user","content":q}]})`。**重要认知**：1.x 的循环靠模型**原生 tool calling**（`AIMessage.tool_calls`）驱动，不再靠正则解析 "Action:" 文本 —— 所以不要再往 prompt 里塞 ReAct 格式，那是 0.x 的文本协议 |
| 2026-10-06 | 离线测试里 `res[0]` 报 `TypeError: 'CallToolResult' object is not subscriptable` | 同上：mcp 1.x 的 `call_tool` 返回 `(content, structured)` 元组，2.x 返回带 `.content` 属性的 `CallToolResult` | 测试代码里两种形态都判一下（`isinstance(res, tuple)` / `hasattr(res, 'content')`）。教训：**跨大版本的可选依赖，取值处都要写兼容** |
| 2026-10-06 | **Agent 端到端 29 题全部「工具×0」，Hit@K 全 0、拒答率 0.4** —— 代码无报错，但一次工具都不调 | **qwen2.5-coder 不遵守工具协议**。探针证实：`bind_tools` 后模型的 `tool_calls` 恒为空，工具调用被写成**纯文本 JSON** 塞在 `content` 里。根因是该 coder 变体把工具定义当普通文本模仿，输出格式五花八门（裸 JSON / `[name {json}]` / `<tool_call>` 标签），而 langchain 1.x 的 agent 图只认结构化 `AIMessage.tool_calls` | 两层修复：① 新增 `src/llm_adapter.py`（`JsonToolCallChatModel`）把三种文本变体归一化为标准 `tool_calls`，带"name 必须在工具名单里"的安全阀 + 去重；② **换模型**：`AGENT_LLM_MODEL=qwen2.5:7b`（指令模型原生支持 tools）。实测结论：**适配层单独不够**——coder+适配层 8 题实测仅 4/8，失败的题是模型**压根不调工具而反问用户**（"请提供具体的指标名称"），这类"意愿问题"适配层救不了，必须换模型 |
| 2026-10-06 | 探针输出"三种方式都拿不到 tool_calls"，一度怀疑是 langchain 没把 tools 透传给 ollama | 用原始 urllib 直接打 ollama `/api/chat` 带 `tools` 字段，同样拿不到 → **排除透传问题，确认是模型本身**。另 `ollama show <model> --template` 显示模板里**有** `<tool_call>` 段却仍不遵守 → 模型能力问题，非配置问题 | 教训：**排查"Agent 不调工具"要按层剥离**——① 代码能不能透传 tools ② 模型吐的是结构化还是文本 ③ 模型模板是否声明了协议。三层各自独立，别混在一起猜 |
| 2026-10-06 | 换 7b 后仍 2/8 失败：**模型不检索就作答，答案里带编造的引用标记**（实测出现 `ronics[1] 提到，BM25 算法中的参数 k1 和 b ...`，`[1]` 是凭空捏的） | 提示词里"回答前必须先检索"是**硬性规则**，但**提示词只能"请求"模型，管不住模型**。模型判断"这题我会"就直接答了 | 新增 `src/middleware.py`：用 langchain 的 `before_model` 钩子，在**首次模型调用前**检查——若尚无 ToolMessage 且问题非寒暄，**替模型把检索做完并以 ToolMessage 注入**。效果：8 题探针从 6/8 → **8/8**，幻觉题被彻底兜住。教训：**能用结构保证的，不要指望提示词** |
| 2026-10-06 | middleware 修好后，多证据题答案开头出现固定套话「知识库中没有找到相关内容。」**紧跟着才是正确答案** | 排查发现检索结果完全正确（[1] 就是 RAG 在线链路原文），模型**也用上了**（准确复述了四步）——问题是 system_prompt 里那句拒答条款被模型当成了**格式模板**，先套模板再补答案。典型"提示词副作用"：为防幻觉写的规则，变成了新缺陷 | 改提示词为**条件式**：明确"只有当检索结果确实与问题无关时才拒答；只要结果里可用信息就必须直接答，禁止在开头套用拒答模板"，并显式点出"要么拒答、要么作答，不要两者都写"。修复后 #1/#2/#6/#7 答案均直接作答。教训：**看到"矛盾答案"（先否后肯）先查提示词模板，不要以为是检索失败** |
| 2026-10-06 | **Agent 端到端跑完，hit_all_kw 0.714，比固定链路 hybrid 的 0.905 低 19pt**，看着像"Agent 让检索变差了" | 两层级联的**度量 bug**：① `extract_sources` 只返回来源名、`excerpt` 恒空 → 评估脚本 `Document(page_content=<路径>)` 里没正文 → **全部"未命中"（0.0）**；② 补上 excerpt 后仍偏低，查出 `extract_sources` **按文件名去重**，同文件第 2/3 块被丢弃——而关键词**常在第 2 块**（Q13 的 `chunk_overlap`、Q18 的 `工具描述`、Q20 的 `N+M` 全因这个从命中变未命中） | ① excerpt 提正文（正则按 `[n] 来源：<path>\n<正文>` 块解析）；② **按块保留不去重**（同一文件的不同块是不同证据）。修复后三道题 `False→True`，sources 从 3/2/2 恢复到完整 5 条。另排查确认**截断 400 字对覆盖率零影响**（0.9524 vs 0.9524，21 题无一受影响），排除了这个嫌疑。教训：**两次都是度量 bug 伪装成"检索失效"**——先怀疑度量实现，再怀疑被测系统，这句要刻在脑子里 |

## 阶段 3 验收（Agent + MCP · 进行中）

- [x] `src/tools.py`：`HybridRetriever` 封装为 LangChain Tool（`search_knowledge_base`）
- [x] Tool 三要素设计：名字（动词_名词）/ description（含"什么时候不要调用"的边界）/ 入参 schema
- [x] 工具返回值两层工程约束：每块截断 400 字控上下文 + `[n] 来源：` 标记供引用溯源
- [x] `src/agent.py`：工具调用型 Agent（langchain 1.x `create_agent`，靠模型原生 tool calling 驱动）
- [x] `scripts/mcp_server.py`：自建 MCP Server（stdio），暴露 3 个工具（检索 / 列文档 / 读配置）
- [x] `src/llm_adapter.py`：LLM 适配层，把文本形态工具调用归一化为标准 `tool_calls`（三种变体 + 安全阀 + 去重）
- [x] `src/middleware.py`：**强制检索 middleware**（`before_model` 钩子，首次调用前替模型完成检索并注入）—— 把"要不要检索"从模型裁量变成结构保障
- [x] `eval/evaluate.py` 加 `--agent`：同一套指标口径跑 Agent 链路，额外统计**平均工具调用次数**
- [x] `tests/agent_offline_test.py` 离线验收：**20/21 通过**（工具定义 + BM25 通路 + RRF + 来源提取 + MCP 协议层实调；唯一失败项是需 ollama 的检索调用）
- [x] `tests/agent_parse_test.py` 离线验收：**21/21 通过**（Agent 消息解析 + 适配层三格式解析 + 安全阀 + 去重）
- [x] `tests/probe_tool_calling.py` 模型工具协议探针（三层剥离定位根因，可传模型名对比）
- [x] `tests/probe_agent_batch.py` 8 题端到端批量实测（术语/拒答/多证据/寒暄四类，统计工具调用成功率）
- [x] **定位并修复「工具×0」根因**：qwen2.5-coder 不遵守工具协议 → 适配层兜底 + 换 `qwen2.5:7b`
- [x] **消融实验（8 题探针，四类题）**：coder+适配层 **4/8** → 7b **6/8** → 7b+强制检索 middleware **8/8**
- [ ] **进行中**：`python eval/evaluate.py --mode hybrid --agent` 29 题端到端指标
- [ ] **待办**：Agent 与固定链路的**对照实验**（同一评估集，看多步检索有没有真增益、增益在哪几类题）
- [ ] **待办**：Agent 链路下忠实度判定缺完整上下文（工具返回的是截断片段）→ 需要给 Agent 单独设计端到端评估口径
- [ ] **待办**：MCP 客户端接入实测（Claude Desktop / Inspector），验证跨进程调用

## 阶段 0/1 检查表（第 1 周验收 · 已全部通过，tag `v0.1-mvp`）

- [x] ollama 在跑，模型就位（本机实际跑的是 `qwen2.5-coder:latest` + `qllama/bge-small-zh-v1.5:latest`；README 里的 `qwen2.5:7b` / `bge-m3` 是推荐配置，可随时切）
- [x] `pip install -r requirements.txt` 成功
- [x] `python scripts/ingest.py --force` 建库成功，storage/index/ 有文件
- [x] `python tests/smoke_test.py` 两个问题都答对（第二个答"没找到"才算对）
- [x] `uvicorn api.main:app` 起服务，curl /ask 返回带 [1][2] 的答案
- [x] git init + 首次提交，打 tag `v0.1-mvp`
- [x] 看不懂清单至少填 3 条

## 阶段 2 验收（检索优化 · 已完成，tag `v0.2-retrieval`）

- [x] eval/questions.jsonl 扩到 29 条（21 条含关键词、5 条应拒答、4 条多证据题）
- [x] rank-bm25 + jieba 加入 requirements；sentence-transformers 等重排依赖单列 `requirements-rerank.txt`
- [x] retrieve() 升级为 `HybridRetriever`：向量 ∪ BM25 → RRF 融合 → 可选 cross-encoder 精排
- [x] 自建评估脚本 `eval/evaluate.py`（**没上 RAGAS**：需要额外装 ragas + 配置 LLM judge，自己写的指标完全够用且更可控）
- [x] 产出 baseline(vector) vs hybrid vs hybrid+rerank 三模式指标对比表
- [x] 语料分级：18 篇领域文档 + 10 篇跨域干扰 + 12 篇同域干扰 = 40 篇 / 122 块
- [x] 评估口径严格化：新增 `Hit@K(全词)` 与 `关键词覆盖率`（原来的宽松 Hit@K 已双双封顶 1.0，没有区分度）
- [ ] 可选：FAISS 换 Milvus（`vectorstore.py` 单文件替换）—— 判断为**当前规模下不必要**，留到语料上千块再说
- [ ] 可选：分块策略消融（chunk_size 300 / 500 / 800 对比）—— 阶段 2 唯一没做的正交变量
