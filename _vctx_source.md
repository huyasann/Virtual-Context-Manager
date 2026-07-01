# VCTX - Compact Recovery Rules

> Put this content in `CLAUDE.md` when VCTX should act as a low-cost recovery memory, not a per-turn logger.

## Core Strategy

Use VCTX only at recovery/checkpoint moments.

Default:

- Do not call `vctx_buffer` every turn.
- Do not archive routine short exchanges.
- Do not archive raw compact summaries without enriching them first.

Main flow:

```text
normal work
-> compact happens or a substantial task completes
-> extract anchors
-> read source files when available
-> archive enriched checkpoint
-> recover later with vctx_list / vctx_search / vctx_read
```

## Rule 1: After Compact, Archive Once

If the conversation has just been compacted, resumed from a summary, or earlier raw context has been replaced by a compact summary, immediately create one VCTX checkpoint.

Do not store the compact summary as-is.

Before archiving:

1. Extract anchors from the compact summary:
   - project name
   - file paths
   - commands
   - error messages
   - APIs/tools
   - decisions
   - unfinished next steps
2. Read source files mentioned by the summary when they are available.
3. Use the file contents plus the compact summary to create an enriched checkpoint.

If the summary mentions decisions that are not recoverable from files, store them explicitly and label them as summary-derived.

## Rule 2: Checkpoint Completed Substantial Tasks

When a task took more than a few turns or produced durable state, archive one checkpoint at task completion.

Durable state includes:

- code changes
- config changes
- environment setup
- debugging root cause
- architecture decisions
- important paths, commands, or test results

Do not checkpoint pure chat, simple confirmations, or tasks with no useful future recovery value.

## Rule 3: Recall Only When Needed

Use VCTX retrieval only when:

- the user asks to continue previous work
- the user refers to earlier context
- compact happened and you need missing details
- the current task depends on archived decisions or prior fixes

Recall flow:

```text
vctx_list
vctx_search(query="<specific topic>")
vctx_read(block_id="<relevant id>")
```

Read only the most relevant 1-3 blocks. Do not dump the full archive unless the user asks.

## Archive Template

Use `vctx_archive`:

```text
vctx_archive(
  title = "specific checkpoint title",
  content = "enriched recovery details: compact-summary anchors, source-file facts, exact paths, commands, errors, code/config changes, decisions, verification, and next steps",
  conclusion = "one-sentence current state or outcome",
  keywords = ["project", "feature", "file", "tool", "error"],
  session_id = "main"
)
```

Do not pass unsupported arguments such as `project_id`.

## Archive Quality Bar

A useful checkpoint must answer:

- What was being done?
- What changed?
- Which files/commands/errors matter?
- What decisions were made?
- What is verified?
- What remains next?

Good checkpoint:

- `title`: "Fix Claude Desktop VCTX MCP config"
- `content`: includes actual config path, exact JSON key, server path, verification command, and result
- `conclusion`: "Claude Desktop 3p reads MCP config from Local Claude-3p, not Roaming Claude."
- `keywords`: `["Claude Desktop", "VCTX", "MCP", "claude_desktop_config", "Claude-3p"]`

Bad checkpoint:

- stores only "we fixed the MCP issue"
- stores only raw compact summary
- omits file paths or commands
- uses generic keywords like `code`, `fix`, `task`

## Do Not Archive

Do not archive:

- pure small talk
- simple acknowledgements
- duplicate content already in VCTX
- raw compact summaries without enrichment
- speculative discussion with no decision
- tasks with no code/config/decision/debugging output
- secrets, unless the user explicitly asks to store them

## Relationship Between CLAUDE.md And VCTX

| Item | CLAUDE.md | VCTX |
|---|---|---|
| Purpose | Static rules and preferences | Dynamic recovery memory |
| Written by | User or project maintainer | Model checkpoints important state |
| Loaded when | Every session | Retrieved on demand |
| Analogy | Operating manual | Project notebook |

CLAUDE.md says how to work. VCTX stores what happened.
