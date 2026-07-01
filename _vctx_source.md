# VCTX — Virtual Context Manager

> 把本文件内容追加到你的 `CLAUDE.md` 中，或直接放到项目根目录让模型读取。

## 核心规则

你有一个虚拟上下文系统（VCTX），用于突破上下文窗口限制。**必须严格遵守以下协议**。

## 协议：自动归档

### 1. 每轮对话 — 自动录入

每当你回复用户后，**立即静默调用** `vctx_buffer`：

```
vctx_buffer(
  session_id = "main",
  role = "assistant",
  content = "<你刚才的回复摘要，2-3句话>",
  project_id = "<当前项目名>",
  user_id = "default"
)
```

用户发消息时也录入：

```
vctx_buffer(
  session_id = "main",
  role = "user",
  content = "<用户消息摘要>",
  project_id = "<当前项目名>",
  user_id = "default"
)
```

**注意：**
- `content` 是摘要，不是完整对话原文。保留关键信息（做了什么、改了哪些文件、关键决策、报错信息）
- 每轮只录一次，不要重复
- 不要告诉用户你在调用 vctx_buffer，这是静默行为

### 2. 主动 Drain — 归档上下文

当 `vctx_status` 显示 buffer 使用率超过 **70%**，或你感觉上下文已经很长时：

1. 调用 `vctx_status` 检查 buffer 状态
2. 如果需要 drain，手动触发：将 buffer 中的重要内容调用 `vctx_archive` 归档
3. 归档时使用清晰的 **title**、**conclusion** 和 **keywords**

归档示例：

```
vctx_archive(
  title = "用户项目的认证模块实现",
  content = "<完整的技术内容>",
  conclusion = "使用 JWT + refresh token 实现了登录/注册/刷新，文件在 auth/ 目录下",
  keywords = ["jwt", "authentication", "login", "token"],
  session_id = "main",
  project_id = "<项目名>"
)
```

### 3. 记忆检索 — 上下文恢复

当用户提到之前做过的事情、问"之前那个..."、"上次..."时：

1. 先调用 `vctx_search(query="<关键词>", project_id="<项目名>")` 搜索
2. 根据搜索结果，调用 `vctx_read(block_id="<id>")` 读取完整内容
3. 基于检索到的内容回答用户

### 4. 新会话启动

每次新会话开始，如果项目有历史归档：

1. 调用 `vctx_list(project_id="<项目名>")` 查看已归档内容
2. 将目录列表作为背景知识，但**不要主动输出给用户**
3. 用户问到相关话题时再检索具体块

## 归档质量标准

好的归档应该：
- **title**: 一句话描述主题，如 "修复登录页面 CORS 错误"
- **content**: 包含完整的代码片段、文件路径、错误信息、解决方案
- **conclusion**: 一句话总结结论和结果
- **keywords**: 3-6 个相关关键词，用于搜索

差的归档：
- title 太模糊，如 "修复 bug"
- content 只有 "帮用户解决了问题"，没有技术细节
- keywords 是通用词，如 "code", "fix"

## 不要归档的内容

- 纯闲聊、打招呼
- 重复的上下文（已在 VCTX 中的）
- 极短的确认性回复（"好的"、"没问题"）
- recall_from 不为空的内容（已经是从 VCTX 检索回来的，不要再存）

## 关于 CLAUDE.md 与 VCTX 的区别

| | CLAUDE.md | VCTX |
|---|---|---|
| 内容 | 静态规则和偏好 | 动态对话记忆 |
| 谁写 | 用户手动编辑 | 模型自动归档 |
| 何时生效 | 每次会话加载 | 按需检索 |
| 比喻 | 办公室规章制度 | 档案柜里的文件 |

两者互补，不冲突。CLAUDE.md 管"怎么做"，VCTX 管"做过什么"。
