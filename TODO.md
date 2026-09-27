# TODO — 看不懂代码清单 & 踩坑记录

> 用法：项目里遇到看不懂的代码/报错，**先记到这里继续往前推**，周末统一处理。
> 处理完的条目写一行"结论"，这份文档本身就是学习记录 + 面试素材。

## 看不懂清单

| # | 位置（文件:行） | 问题描述 | 状态 | 结论 |
|---|----------------|---------|------|------|
| 1 | src/generator.py:34 | `\|` 管道符把 prompt/llm/parser 串起来，底层是怎么调度的？（Runnable 协议） | ☐ | |
| 2 | src/vectorstore.py:24 | `allow_dangerous_deserialization=True` 为什么危险，FAISS 存的是什么 | ☐ | |
| 3 | | | ☐ | |

## 踩坑记录

| 日期 | 现象 | 原因 | 解决 |
|------|------|------|------|
| 2026-09-27 | `pip install -r requirements.txt` 报 UnicodeDecodeError: 'gbk' codec | requirements.txt 里有中文注释，Windows 中文系统的 pip 默认按 GBK 编码读文件，UTF-8 中文解码失败 | 已修复：requirements.txt 改为纯 ASCII（英文注释）。教训：Windows 下给 pip 读的文件不要写中文 |

## 阶段 0 检查表（第 1 周验收）

- [ ] ollama 在跑，`ollama pull qwen2.5:7b` 和 `bge-m3` 完成
- [ ] `pip install -r requirements.txt` 成功
- [ ] `python scripts/ingest.py --force` 建库成功，storage/index/ 有文件
- [ ] `python tests/smoke_test.py` 两个问题都答对（第二个答"没找到"才算对）
- [ ] `uvicorn api.main:app` 起服务，curl /ask 返回带 [1][2] 的答案
- [ ] git init + 首次提交，打 tag `v0.1-mvp`
- [ ] 看不懂清单至少填 3 条

## 阶段 2 预埋（先不做，做完 MVP 再看）

- [ ] eval/questions.jsonl 扩到 20-30 条（含应拒答的问题）
- [ ] rank_bm25 + sentence-transformers 加入 requirements
- [ ] retrieve() 升级：向量检索 ∪ BM25 检索 → bge-reranker 重排 → 取 top_k
- [ ] RAGAS 打分，产出 baseline vs 优化后指标表
- [ ] 可选：FAISS 换 Milvus（vectorstore.py 单文件替换）
