# 18 MCP 协议与在 LangChain 中的使用

## 概述

MCP（Model Context Protocol，模型上下文协议）是一套标准化协议，目标是让大模型应用能以**统一的方式**连接外部工具和数据源。它由 Anthropic 在 2024 年底提出，2025 年开始被主流工具采纳。对做 Agent 的人来说，MCP 的价值在于：它把"给 Agent 接工具"这件事从"每个项目都要重写一遍适配代码"变成了"装一个标准的 Server 就能用"。

## 它要解决的问题：N×M 适配地狱

在没有 MCP 的时代，假设你有 N 个大模型应用、M 个外部工具和数据源（数据库、文件系统、GitHub、Slack……），要全部打通就得写 N×M 份适配代码。每换一个应用，所有工具的对接方式都要重写。

MCP 把接口标准化后，变成：

- 每个工具/数据源实现**一个 MCP Server**（M 份）
- 每个应用实现**一个 MCP Client**（N 份）
- 总工作量从 N×M 降到 N+M

这和当年 USB 取代一堆专用接口是同一个思路：**标准化的价值在于把两两对接变成统一插拔**。

## 三层架构

MCP 的架构分三层：

**Host（宿主）**
运行大模型的应用本身。例如 Claude Desktop、Cursor、你自己写的 Agent 程序。Host 负责发起对话、管理整个会话。

**Client（客户端）**
Host 内部负责与 Server 通信的连接器。**一个 Server 对应一个 Client**，Client 维护与那个 Server 的一对一连接，负责协议握手、消息收发、能力协商。

**Server（服务端）**
真正暴露能力的程序。它把一组工具、资源、提示词模板包装好，按 MCP 协议对外提供。

三者的关系：**一个 Host 可以同时连接多个 Server**（通过各自的 Client）。比如你的 Agent 同时连了"文件系统 Server"和"PostgreSQL Server"，那它就有了读文件和查数据库两套能力。

## Server 暴露的三类能力

这是 MCP 设计的核心：

1. **Tools（工具）**——模型可以**主动调用**的函数。比如 `read_file(path)`、`query_database(sql)`。调用会改变状态或返回数据。这类对应 Agent 里的 Tool。
2. **Resources（资源）**——模型可以**读取**的数据，像文件内容、数据库表结构、日志片段。和 Tools 的区别是 Resources 是"被读取的上下文"，不产生副作用。
3. **Prompts（提示词模板）**——Server 预置的提示词模板，供 Host 直接使用。让工具提供者能把自己的最佳实践（怎么用我这些工具）随着 Server 一起分发。

举例：一个"公司知识库 Server"可以暴露

- Tool：`search_docs(query)` —— 检索知识库
- Resource：`docs://handbook` —— 员工手册全文
- Prompt：`summarize_meeting` —— 会议纪要总结模板

## 传输方式

MCP 支持两种传输：

- **stdio（标准输入输出）**：本地进程间通信。Host 启动 Server 子进程，通过 stdin/stdout 交换 JSON-RPC 消息。优点是零网络配置、性能好、适合本地工具（文件系统、本地数据库）
- **HTTP / SSE**：远程通信。Server 部署在远端，通过 HTTP 传输，SSE（Server-Sent Events）用于服务端推送。适合团队共享的服务、云端 API 封装

选择原则很简单：**本地能力用 stdio，共享能力用 HTTP**。

## 在 LangChain 中使用

LangChain 通过 `langchain-mcp-adapters` 这个包做桥接，核心思路是：**把 MCP Server 暴露的 Tools 转换成 LangChain 的 Tool 对象，然后直接塞进 Agent**。

流程大致是：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]},
})
tools = await client.get_tools()     # 拿到一批 LangChain Tool 对象
agent = create_react_agent(llm, tools)   # 直接给 Agent 用
```

关键理解：**转换之后，MCP 工具和本地写的** **`@tool`** **函数在 Agent 看来没有任何区别**。Agent 的决策机制、工具描述的作用、function calling 的流程全部一样——MCP 改变的只是"工具的来源和分发方式"，没有改变 Agent 的工作方式。

这就是为什么第 17 篇（Agent 与 Tool）是学 MCP 的前置：不懂 Tool 的三要素和 function calling 机制，MCP 就只是一堆名词。

## 为什么值得学

- **时间窗口**：MCP 是 2024 年底才提出的协议，2025 年才普及，真正在生产项目里用过的人还不多
- **生态在快速扩张**：GitHub、Slack、Postgres、文件系统等都有官方或社区 Server，接入成本极低
- **架构思维**：MCP 背后的"标准化消除 N×M 适配"是通用的架构思考方式，不限于 AI 领域

## 常见误区

**误区一：MCP 是模型的能力。**
不是。MCP 是应用层协议，模型完全不知道 MCP 的存在。协议发生在 Host 和 Server 之间，模型看到的仍然只是"有哪些工具可用"。

**误区二：MCP 取代了 Agent。**
不取代。MCP 管的是"工具从哪来、怎么接"，Agent 管的是"什么时候用哪个工具"。两者是正交的。

**误区三：Server 必须是远端服务。**
最常用的形态恰恰是本地 stdio 的 Server，一个进程而已，`npx` 就能起。

## 要点清单

- MCP = Model Context Protocol，工具连接的标准化协议
- 解决的痛点是 N×M 适配，标准化后降到 N+M
- 三层架构：Host（应用）、Client（连接器，一对一）、Server（能力提供方）
- 一个 Host 可连多个 Server，每个 Server 对应一个 Client
- Server 暴露三类能力：Tools（可调用）、Resources（可读）、Prompts（模板）
- 本地用 stdio 传输，远程用 HTTP/SSE
- langchain-mcp-adapters 把 MCP 工具转成 LangChain Tool，转换后与本地工具无差别
- MCP 管"工具来源"，Agent 管"工具选择"，两者正交

