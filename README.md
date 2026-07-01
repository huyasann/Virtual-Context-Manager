# Virtual Context Manager (vctx)

**基于虚拟内存管理的 LLM 扩宽上下文系统**

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 核心思想

大语言模型的上下文窗口（如 200k tokens）相当于 CPU 的物理内存——容量有限，满了就得换出。vctx 借鉴操作系统的虚拟内存机制，为 LLM 构建了一套分层上下文管理系统：

```
┌─────────────────────────────────────────────────────────┐
│  Primary Context（主上下文，模型直接可见）                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  System Prompt + VC Index（目录）+ 最近几轮对话     │  │
│  │  ≈ 20k tokens，始终保持轻量                         │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │ 模型通过 tool call 按需读取      │
│                         ▼                                │
│  Virtual Context（虚拟上下文，完整历史）                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Block 001: ROS2 架构讨论（12k tokens）            │  │
│  │  Block 002: CAN 总线调试（8k tokens）              │  │
│  │  Block 003: Docker 优化方案（15k tokens）          │  │
│  │  ... 原始数据完整保留，无限扩展                      │  │
│  └───────────────────────────────────────────────────┘  │
│                         │                                │
│                         ▼                                │
│  时间衰减换血（importance decay）                         │
│  长期未访问的记忆自动降权 → 保持知识库新鲜度               │
└─────────────────────────────────────────────────────────┘
```

## 与现有方案的对比

| 维度 | 传统长上下文 | RAG 检索增强 | **vctx 虚拟上下文** |
|------|-------------|-------------|-------------------|
| 信息保真度 | 100%（但注意力退化） | 有损（依赖检索质量） | **100%（原始数据完整保存）** |
| 理论容量 | 硬上限 200k | 无上限 | **无上限** |
| 检索方式 | 全文塞入（注意力稀释） | embedding 语义搜索 | **目录索引 + 按需读取** |
| 注意力质量 | 随长度衰减 | 只对检索结果有效 | **始终满血（只聚焦当前需要的块）** |
| 模型适配 | 需要长上下文模型 | 需要 embedding 模型 | **任意支持 tool call 的模型** |

## 作为 MCP Server 运行（推荐）

vctx 以 [MCP (Model Context Protocol)](https://modelcontextprotocol.io) Server 形式运行，与 Claude Code 等支持 MCP 的客户端无缝集成。

### 安装

```bash
git clone https://github.com/huyasann/Virtual-Context-Manager.git
cd Virtual-Context-Manager
pip install -r requirements.txt
```

### 注册到 Claude Code

在 `~/.claude/settings.json` 中添加：

```json
{
  "mcpServers": {
    "vctx": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

重启 Claude Code 即可生效。

### 提供的工具

| 工具 | 用途 | 触发时机 |
|------|------|---------|
| `vctx_archive` | 归档对话到虚拟上下文 | 上下文快满时，或用户要求保存 |
| `vctx_read` | 按 ID 读取历史块 | 用户提到过去的话题 |
| `vctx_search` | 关键词搜索历史 | 不确定信息在哪个块时 |
| `vctx_list` | 查看完整目录 | 用户问"你记住了什么" |
| `vctx_decay` | 时间衰减换血 | 定期清理过期记忆 |
| `vctx_delete` | 删除指定块 | 用户要求遗忘某段记忆 |

## 使用示例

```
你: 我们来讨论一下项目的数据库选型
助手: [讨论 PostgreSQL vs MySQL...]

你: 把刚才的讨论存一下
助手: [调用 vctx_archive → 归档成功，block_id: 260701-a3f2c1]

你: （继续其他话题... 100 轮后）

你: 之前我们数据库选的什么？理由是什么？
助手: [调用 vctx_list → 看到 "数据库选型" 主题块]
      [调用 vctx_read("260701-a3f2c1") → 读取完整讨论]
      你们选了 PostgreSQL，主要理由是需要 JSONB 支持...
```

## 数据存储

所有数据存储在本地 SQLite 文件：

```
~/.vctx/memory.db    # 虚拟上下文块（原文 + 元数据）
```

- 零外部依赖，单文件可备份
- 支持 WAL 模式并发读写
- 可用任何 SQLite 工具直接查看

## 未来路线图

- [ ] Embedding 语义搜索（替换当前关键词匹配）
- [ ] 异步归档（后台线程，不阻塞对话）
- [ ] HTTP 代理模式（兼容 OpenAI API 格式）
- [ ] 多会话隔离
- [ ] 自动归档触发（水位线监控 + 自动搬家）
- [ ] 知识图谱增量更新（参考 LightRAG）

## 理论基础

vctx 的设计受以下项目启发：

| 项目 | 借鉴机制 |
|------|---------|
| [Letta (MemGPT)](https://github.com/letta-ai/letta) | 分层内存模型、水位线触发、Heartbeat 机制 |
| [Mem0](https://github.com/mem0ai/mem0) | 增量记忆提取、实体级去重、冲突解决 |
| [LightRAG](https://github.com/HKUDS/LightRAG) | hash-first 去重、低成本实体增量更新、纯 SQLite 存储 |

vctx 的核心创新在于：**不压缩原始数据，而是用目录索引替换上下文内容**。模型通过目录浏览虚拟上下文，按需读取完整块，实现了"记住目录就等于记住整本书"的效果。

## License

MIT
