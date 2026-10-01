# 16 LCEL 与 Runnable

## 概述

LCEL（LangChain Expression Language，LangChain 表达式语言）是 LangChain 的组合语法，用管道符 `|` 把提示词模板、模型、输出解析器等组件串成一条链。`prompt | llm | parser` 这行代码看起来像 Shell 管道，但它在 Python 里能工作，靠的是一个非常基础的机制——**运算符重载**。本文讲清它的原理、数据流，以及它为什么取代了老式的 Chain。

## 管道符的底层机制

Python 里 `a | b` 这个表达式，如果 `a` 是普通整数，它是按位或运算。但如果 `a` 所属的类定义了 `__or__` 方法，Python 就会调用 `a.__or__(b)`，返回值完全由这个方法的实现决定。

LangChain 的所有核心组件（PromptTemplate、ChatModel、OutputParser）都继承自同一个基类 `Runnable`，而 `Runnable` 实现了 `__or__`：

```python
class Runnable:
    def __or__(self, other):
        return RunnableSequence(self, other)
```

所以 `prompt | llm` 并不是"对提示词和模型做或运算"，而是**创建了一个 RunnableSequence 对象**，把这个组件的执行顺序记下来。继续 `| parser`，就变成嵌套的序列。

验证方法很简单：

```python
chain = prompt | llm | parser
print(type(chain))   # <class 'langchain_core.runnables.base.RunnableSequence'>
```

课上学到的名词，落到了一个真实的类上。

## 数据流：每一环只认上一环的输出

调用 `chain.invoke({"context": "...", "question": "..."})` 时，数据按顺序流过每一环：

```C
dict 输入
  ↓ prompt  ——— 把 {context} {question} 填进模板，产出 ChatPromptValue（格式化好的完整提示词）
  ↓ llm     ——— 拿提示词调用大模型，产出 AIMessage（模型的原始回复对象，含 content 等字段）
  ↓ parser  ——— StrOutputParser 抽出 .content，产出纯字符串
字符串输出
```

关键理解：**每一环只认识上一环的输出格式**。

- PromptTemplate 的输入是 dict（变量名 → 值），输出是格式化后的提示词
- ChatModel 的输入是提示词，输出是 AIMessage 对象（不是字符串！）
- StrOutputParser 的输入是 AIMessage，输出是纯文本

这也解释了为什么 `parser` 这一环在这条链里是必要的：如果直接用 LLM 的输出，拿到的是 AIMessage 对象，除了正文还带着 token 用量、结束原因等元数据。要得到干净的字符串，就得挂一个 StrOutputParser——**它做的事就是** **`message.content`** **一下**。

## Runnable 带来的免费能力

因为所有组件统一实现 Runnable 接口，整条链**自动获得**这些方法，不需要额外写代码：

- `invoke(input)`：单次同步调用，最常用
- `batch([input1, input2, ...])`：批量调用，内部会自动并发
- `stream(input)`：流式输出，逐 token 产出，实现"打字机效果"
- `ainvoke()` / `abatch()` / `astream()`：异步版本

这就是 LCEL 的核心卖点：**统一接口带来的可组合性**。你不用为"支持流式"单独写一套逻辑，链本身就能 stream。

## 为什么取代了老的 LLMChain

LangChain 早期用 `LLMChain` 这类类来组织流程：

```python
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(question="...")
```

问题在于：

1. **不统一**——每种 Chain 有各自的输入输出约定，组合起来要做适配
2. **不可组合**——想把两个 Chain 串起来、或者给 Chain 中间插一个处理步骤，没有统一的方式
3. **能力受限**——流式、批量、异步要靠各 Chain 自己实现，支持得参差不齐

LCEL 把"一切皆 Runnable"作为统一抽象，于是组合、嵌套、复用、流式全部自然地成立。新版本的 LangChain 中 LCEL 已经是默认写法。

## 常见误区

**误区一：以为** **`|`** **是语法糖。**
它不是语法糖，是运算符重载。理解这一点，你就能自己实现一个支持 `|` 的类。

**误区二：以为** **`chain.invoke()`** **的输入是字符串。**
取决于链的结构。链首是 PromptTemplate 时，输入必须是 dict，key 要和自己定义的变量名一致。传错 key 会报"缺少变量"的错误。

**误区三：以为链只能线性。**
Runnable 支持分支（RunnableBranch）、并行（RunnableParallel）、透传（RunnablePassthrough）。RAG 里常见的"把检索结果和原始问题一起传给模型"就是用 `RunnableParallel` + `RunnablePassthrough` 实现的。

## 要点清单

- LCEL 是 LangChain 的组合语法，不是独立语言
- `|` 靠 `__or__` 运算符重载实现，返回 RunnableSequence
- 数据流：dict → 提示词 → AIMessage → 字符串，每环只认上环输出
- StrOutputParser 的本质就是取 `message.content`
- 统一 Runnable 接口免费带来 invoke / batch / stream / 异步
- 取代 LLMChain 的原因是统一抽象带来的可组合性
- 链可以分支、并行、透传，不限于线性

