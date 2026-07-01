# VCTX — Virtual Context Manager

> 把本文件内容追加到你的 `CLAUDE.md` 中，或直接放到项目根目录让模型读取。

## 你有 VCTX

你的项目接入了 VCTX（虚拟上下文管理器），提供 `vctx_archive`、`vctx_search`、`vctx_read`、`vctx_list` 工具，用于长期记忆存储。

## 规则（共 3 条）

### 规则 1：Compact 后必归档

当你在上下文中看到类似以下的系统消息时：

> "The conversation so far has been compacted" / "context has been compacted" / "summary above"

**立即执行：**

1. 读取 compact summary 中提到的所有关键文件（用 Read/Grep 工具）
2. 按主题拆分，每个主题调用一次 `vctx_archive`
3. 调用 `vctx_list` 确认归档完成
4. 告知用户："上下文已压缩，关键内容已归档到 VCTX，可随时检索。"

### 规则 2：重大任务完成后 checkpoint

完成一个重要任务（功能实现、bug 修复、架构决策）后，调用一次 `vctx_archive` 保存结果。

判断标准：如果这个任务花了超过 3 轮对话，就值得归档。

### 规则 3：用户提到历史内容时检索

当用户说"之前那个..."、"上次做的..."、"之前改过..."时：

1. `vctx_search(query="<关键词>")` 搜索
2. `vctx_read(block_id="<id>")` 读取完整内容
3. 基于检索内容回答

新会话开始时，`vctx_list` 查看已归档目录（不要主动输出给用户）。

## 归档格式

```
vctx_archive(
  title = "做什么 — 一句话结果",
  content = """
## 背景
为什么做这件事

## 做了什么
具体改动：文件路径、函数名、代码片段

## 关键决策
为什么选择这个方案而不是其他

## 结果
最终状态、测试结果、遗留问题

## 相关文件
- path/to/file1.py
- path/to/file2.md
""",
  conclusion = "一句话总结：做了X，结果是Y",
  keywords = ["关键词1", "关键词2", "关键词3"],
  project_id = "<项目名>"
)
```

## 归档示例（好 vs 差）

**好的归档：**
- title: "给 VCTX 所有工具添加 project_id 过滤"
- content: 包含改动的文件、函数签名变化、INSERT 语句变化、测试结果
- conclusion: "9 个工具全部支持 project_id/user_id 可选参数，smoke test 通过"
- keywords: ["vctx", "project_id", "multi-session", "filter"]

**差的归档：**
- title: "代码修改"
- content: "帮用户改了一些代码"
- conclusion: "改完了"
- keywords: ["code", "change"]

## 不要归档

- 纯闲聊、打招呼、确认（"好的"、"没问题"）
- 已在 VCTX 中存在的内容（先搜索确认）
- 没有技术细节的纯讨论
- compact summary 本身（必须先读取源文件丰富内容后再归档）

## 关于 CLAUDE.md 与 VCTX

| | CLAUDE.md | VCTX |
|---|---|---|
| 内容 | 静态规则和偏好 | 动态对话记忆 |
| 谁写 | 用户手动编辑 | 模型自动归档 |
| 何时生效 | 每次会话加载 | 按需检索 |
