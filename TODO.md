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

## 阶段 3 预埋（Agent + MCP）

- [ ] `src/retriever.py` 已是干净接口，可直接包成 LangChain Tool
- [ ] 待办：检索 tool 的 `description` 怎么写才能让模型选对工具（Tool 三要素里最影响选择准确率的那一项）
- [ ] 待办：自建 MCP Server，把检索能力暴露出去
- [ ] 待办：Agent 多步推理场景下，怎么给一次问答做端到端评估（单跳指标不够用）
