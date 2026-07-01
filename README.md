# VCTX: A Virtual Memory Approach to Expanding Context Windows for Large Language Models

**VCTX：基于虚拟内存管理的 LLM 扩宽上下文系统**

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Abstract

Large Language Models (LLMs) are constrained by fixed context windows (e.g., 200k tokens), which limits their effectiveness in long-running conversations and iterative development workflows. Existing approaches—long-context scaling and Retrieval-Augmented Generation (RAG)—each suffer from fundamental trade-offs: the former degrades attention quality (the "Lost in the Middle" problem), while the latter introduces information loss through semantic compression. We propose **VCTX (Virtual Context eXtension)**, a system that borrows the hierarchical memory management paradigm from operating systems to achieve theoretically unbounded context for LLMs without modifying the underlying model. VCTX introduces a **directory-index mechanism**: when the context window approaches capacity, conversation history is moved verbatim into a persistent "Virtual Context" store, and a lightweight semantic index (the "VC Index") replaces it in the active context window. The model navigates this virtual context through tool calls, reading full blocks on demand—akin to a page fault handler in virtual memory systems. We present the system architecture, describe the core mechanisms of context draining, directory generation, and temporal decay-based memory rotation, and discuss the theoretical implications for LLM context management.

**Keywords:** Large Language Models, Context Management, Virtual Memory, Retrieval-Augmented Generation, Long-term Memory

---

## 1. Introduction

### 1.1 The Context Window Bottleneck

Modern LLMs support context windows ranging from 128k to 1M+ tokens. While these windows are large, they remain fundamentally finite—and in practice, effective utilization degrades significantly as context length increases. Research on the "Lost in the Middle" phenomenon [1] demonstrates that models exhibit a U-shaped attention curve: information placed at the beginning and end of long contexts receives disproportionately higher attention than information in the middle.

This creates a paradox: **increasing the context window does not proportionally increase the model's effective memory capacity.**

### 1.2 Limitations of Existing Approaches

**Long-context models** (e.g., 200k–1M windows) address the symptom but not the cause. Beyond attention degradation, each inference call processes the entire context, creating quadratic cost growth and latency issues.

**Retrieval-Augmented Generation (RAG)** externalizes memory to vector databases, but introduces a fundamental trade-off: the embedding-based retrieval step is lossy. Relevant context may not be retrieved, and irrelevant context may be retrieved instead. The model never "knows what it doesn't know."

**Memory-augmented systems** like MemGPT/Letta [2] introduce layered memory (core memory, archival memory) with state-machine-driven management, but rely on LLM-generated summaries for archival—introducing information loss at the point of storage.

### 1.3 Our Approach: Virtual Memory for LLMs

We observe that the problem of LLM context management is structurally analogous to the problem of physical memory management in operating systems. An OS with limited physical RAM does not attempt to fit all data in RAM; instead, it maintains a **page table** that maps virtual addresses to physical (or disk-backed) storage, and loads pages on demand via **page faults**.

VCTX applies this paradigm directly:

| OS Virtual Memory | VCTX |
|---|---|
| Physical RAM | Primary Context (model's active window) |
| Page Table | VC Index (semantic directory) |
| Swap / Disk | Virtual Context (persistent store) |
| Page Fault | Tool Call (`vc_read`) |
| Page Replacement | Context Drain (watermark-triggered) |
| Working Set | Recent conversation window |

The key insight: **the model does not need to hold all history in its context—it needs to know that the history exists and where to find it.** A compact directory (3–8k tokens) can represent an arbitrarily large conversation history, and the model can selectively retrieve relevant blocks through tool calls.

---

## 2. Related Work

### 2.1 Letta (formerly MemGPT)

Letta [2] introduced the concept of tiered memory for LLMs: Core Memory (always in context), In-context Messages (sliding window), and Archival Memory (external store). Its state machine uses a "heartbeat" mechanism to allow multi-step autonomous operations within a single user turn.

**What we borrow:** The tiered memory concept and watermark-triggered eviction.

**Where we diverge:** Letta's archival process compresses messages into summaries, introducing information loss. VCTX preserves original content verbatim and replaces it with an index.

### 2.2 Mem0

Mem0 [3] focuses on incremental memory extraction with entity-level conflict resolution. Memories are stored as (entity, attribute) pairs with temporal metadata, enabling smart upsert semantics.

**What we borrow:** Entity-level deduplication and temporal decay for memory freshness.

**Where we diverge:** Mem0 operates on extracted facts; VCTX operates on raw conversation blocks.

### 2.3 LightRAG / Nano-GraphRAG

LightRAG [4] demonstrates that lightweight knowledge graph construction is feasible with local SQLite storage and hash-first deduplication, avoiding heavyweight graph database infrastructure.

**What we borrow:** Hash-first deduplication, pure SQLite storage architecture, incremental entity updates.

**Where we diverge:** VCTX does not construct a knowledge graph; it maintains a flat block store with a semantic index.

### 2.4 The Lost in the Middle Problem

Liu et al. [1] empirically demonstrated that LLM performance degrades for information positioned in the middle of long inputs. This finding directly motivates VCTX's design: rather than filling the context window with all historical data (where middle content will be poorly attended), VCTX keeps the context window small and focused, with a directory that positions all blocks at equal retrieval accessibility.

---

## 3. System Design

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Primary Context (L1)                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  System Prompt (1k)                                   │  │
│  │  + VC Index / Directory (3-8k)                        │  │
│  │  + Recent N turns (10-15k)                            │  │
│  │  = ~20k tokens total, never exceeds capacity          │  │
│  └────────────────────────┬──────────────────────────────┘  │
│                           │ Model issues tool call           │
│                           ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Virtual Context (L2) — Persistent Store               │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐              │  │
│  │  │ Block 001│ │ Block 002│ │ Block 003│  ...          │  │
│  │  │ Full text│ │ Full text│ │ Full text│               │  │
│  │  └──────────┘ └──────────┘ └──────────┘              │  │
│  │  Storage: SQLite (WAL mode)                           │  │
│  │  Dedup: SHA-256 fingerprint + semantic similarity     │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Memory Rotation (Temporal Decay)                     │  │
│  │  importance *= decay_factor ^ (days_since_access)     │  │
│  │  importance < threshold → candidate for eviction      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Context Lifecycle State Machine

The system operates through four states:

```
IDLE ──[user input]──→ ACCUMULATING ──[tokens ≥ 170k]──→ DRAINING
  ↑                                                       │
  │                                                       ▼
  └──[archive complete]── ARCHIVING ◀──[clone submitted to worker]
```

**IDLE**: Waiting for user input. No context accumulation.

**ACCUMULATING**: Messages are appended to a buffer. Token count is tracked incrementally (O(1) per message). Three thresholds are monitored:
- `< 140k`: Normal operation
- `140k–170k`: System injects a "pressure hint" into the prompt, advising the model to be concise
- `≥ 170k`: Triggers the DRAINING state

**DRAINING**: The system performs:
1. Deep-copy the full context snapshot (O(n), memory-only, no I/O)
2. Sliding window retention: keep the most recent ~20k tokens of complete conversation turns
3. Submit the evicted snapshot to an async archival queue
4. Immediately return to ACCUMULATING with the trimmed context

**ARCHIVING**: A background worker processes the evicted snapshot:
1. Filter recalled content (anti-recursion)
2. Generate VC Index via LLM (the only lossy step—only produces a directory, not a summary)
3. Store original text verbatim in SQLite
4. Update the VC Index in the primary context

### 3.3 The VC Index: Directory, Not Summary

The core innovation of VCTX is the **VC Index**—a structured directory that replaces the original conversation content in the model's context window. Unlike summarization (which compresses information), the VC Index is a **pointer structure** that preserves full retrieval capability.

**Example VC Index:**

```markdown
## Virtual Context Directory
> 15 topic blocks | 170,032 tokens | 2026-06-15 ~ 2026-06-30

- [001] **ROS2 Communication Architecture** (12,400 tokens)
  Key decision: Use DDS over custom TCP | QoS: RELIABLE
  Keywords: ROS2, DDS, QoS, pub/sub, topic

- [002] **CAN Bus Driver Debugging** (8,200 tokens)
  Key finding: Baud rate 500k, filter ID 0x100-0x1FF
  Keywords: CAN, SocketCAN, bitrate, filter

- [003] **Docker Multi-stage Build Optimization** (15,800 tokens)
  Result: 3-stage build, 2.1GB → 340MB
  Keywords: Docker, multi-stage, alpine, layer cache
```

**Why this works better than summarization:**
- Each block is a self-contained retrieval unit with enough context for the model to decide relevance
- The model retains agency: it chooses which blocks to read, rather than being fed a pre-determined summary
- The index scales linearly (3–8k tokens) regardless of total history size

### 3.4 Anti-Recursion: Preventing Memory Loops

When the model recalls content from the Virtual Context (via `vc_read`), that content re-enters the primary context. Without safeguards, the next drain cycle would re-archive this recalled content, creating infinite loops of redundant storage.

VCTX employs a three-layer defense:

**Layer 1 — Tagging:** Recalled content is injected with metadata tags (`recall_from: block_id`, `recalled_at: timestamp`).

**Layer 2 — Filtering:** During drain, messages with recall metadata are excluded from archival. Only their access count is updated in the database.

**Layer 3 — Deduplication:** Non-recalled content is fingerprinted (SHA-256). Matching fingerprints skip archival. Semantic similarity (cosine > 0.95) triggers merge instead of insert.

### 3.5 Temporal Decay and Memory Rotation

To prevent unbounded growth of the Virtual Context, VCTX implements a time-decay mechanism inspired by cache replacement algorithms:

```
importance(t) = importance(t₀) × 0.97^(days_since_last_access)
```

- Blocks accessed recently retain high importance
- Blocks untouched for 30+ days decay toward the eviction threshold (0.1)
- Decay does not delete data—it lowers priority for VC Index inclusion
- Decayed blocks remain searchable via `vctx_search` but are excluded from the directory

This creates a natural "memory rotation" analogous to LRU page replacement.

---

## 4. Implementation

### 4.1 Deployment Model: MCP Server

VCTX is implemented as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io) Server, enabling seamless integration with any MCP-compatible client (e.g., Claude Code, VS Code extensions, custom agents).

The MCP model is particularly well-suited because it aligns exactly with the virtual memory metaphor:
- The MCP client (LLM host) is the CPU
- The MCP server (VCTX) is the memory controller
- Tool calls are memory bus transactions

### 4.2 Tool Interface

| Tool | OS Analogy | Function |
|---|---|---|
| `vctx_archive` | Page-out | Move conversation to Virtual Context |
| `vctx_read` | Page-in / Page Fault | Retrieve specific block by ID |
| `vctx_search` | TLB Lookup | Keyword search across Virtual Context |
| `vctx_list` | Page Table Dump | View full directory of archived content |
| `vctx_decay` | Clock Algorithm Sweep | Apply temporal decay, identify eviction candidates |
| `vctx_delete` | Page Free | Remove specific blocks |

### 4.3 Storage Layer

All persistent data is stored in a single SQLite file (`~/.vctx/memory.db`):

```sql
CREATE TABLE blocks (
    block_id    TEXT PRIMARY KEY,
    title       TEXT,
    content     TEXT NOT NULL,       -- Full original text, uncompressed
    token_count INTEGER,
    keywords    TEXT,                -- JSON array for search
    conclusion  TEXT,                -- One-line summary for index
    created_at  TEXT,
    last_access TEXT,
    importance  REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0
);
```

Design choices:
- **WAL mode** for concurrent read/write safety
- **No external dependencies** (no vector database, no graph database)
- **Single-file portability** (backup = copy one file)

---

## 5. Theoretical Analysis

### 5.1 Information Retention Comparison

| Property | Long Context | RAG | VCTX |
|---|---|---|---|
| Storage fidelity | 100% (in context) | Lossy (embedding) | **100% (verbatim)** |
| Retrieval fidelity | 100% (if attended) | Depends on recall@k | **100% (exact block)** |
| Attention quality | Degrades with length | Good (small context) | **Good (small context)** |
| Theoretical capacity | Hard limit (200k) | Unlimited | **Unlimited** |
| Model requirement | Long-context model | Embedding model | **Any tool-calling model** |

### 5.2 The "Lost in the Middle" Mitigation

In a 200k-token context, information at position 100k receives significantly less attention than information at position 0 or 200k [1]. VCTX sidesteps this entirely:

- The primary context is always ~20k tokens → no middle degradation
- All historical blocks are equally accessible via the VC Index → no positional bias
- Retrieved blocks are small (~4–8k) → full attention on relevant content

### 5.3 Cost Analysis (Projected)

| Scenario | Tokens per turn (Long Context) | Tokens per turn (VCTX) |
|---|---|---|
| Turn 1 (fresh) | ~1,000 | ~1,000 |
| Turn 100 | ~50,000 | ~20,000 (context capped) |
| Turn 500 | ~200,000 (at limit) | ~20,000 + occasional tool calls |
| Turn 1000 | Impossible | ~20,000 + tool calls |

VCTX maintains near-constant per-turn cost regardless of conversation length.

### 5.4 Latency Profile

The drain operation is the only potentially blocking step. VCTX mitigates this through:
- **Async archival**: Drain submits to a background queue; the main conversation continues immediately
- **O(1) handoff**: The drain itself is a memory copy + queue put, completing in microseconds
- **Background LLM call**: Directory generation runs asynchronously and does not block the user

---

## 6. Hypotheses and Planned Experiments

> **Note:** The following experiments are planned but not yet executed. Results will be added in future versions.

### 6.1 Experiment 1: Memory Recall Accuracy

**Setup:** Conduct 500-turn conversations on technical topics. At random intervals, ask the model to recall specific facts from earlier turns.

**Baselines:**
- Pure long-context (200k window, no memory management)
- Standard RAG (embedding-based retrieval from conversation history)
- MemGPT-style (summarize-and-archive)

**Metrics:** Recall@1, Recall@5, factual accuracy

**Hypothesis:** VCTX achieves comparable or superior recall accuracy to long-context models while maintaining constant per-turn cost, because the directory index provides targeted retrieval without attention degradation.

### 6.2 Experiment 2: Attention Quality Under Load

**Setup:** Place a specific fact at position P in a 200-token context vs. in a VCTX block, measure the model's ability to answer questions about it.

**Hypothesis:** VCTX's retrieved-block attention (on a small, focused context) exceeds the diluted attention of a full 200k window, particularly for facts that would otherwise fall in the "Lost in the Middle" zone.

### 6.3 Experiment 3: Memory Freshness via Temporal Decay

**Setup:** Seed 100 blocks with varying access patterns. Apply decay over simulated 90-day period. Measure whether the VC Index correctly retains frequently-accessed blocks and degrades rarely-accessed ones.

**Hypothesis:** The temporal decay mechanism maintains a VC Index that reflects the user's actual information needs, with high-recall for active topics and graceful degradation for dormant ones.

### 6.4 Experiment 4: Anti-Recursion Effectiveness

**Setup:** Conduct conversations that require repeated recall of the same blocks. Measure storage growth over 10 drain cycles.

**Hypothesis:** The three-layer anti-recursion mechanism (tagging → filtering → dedup) prevents storage bloat, with recalled content contributing zero additional storage.

---

## 7. Discussion

### 7.1 Advantages Over Existing Approaches

1. **Lossless by design**: Unlike summarization-based approaches, VCTX never discards information. The VC Index is a pointer structure, not a compression.

2. **Attention-optimal**: The model always operates on a small, focused context, avoiding the "Lost in the Middle" degradation inherent to long-context approaches.

3. **Model-agnostic**: VCTX works with any LLM that supports tool calling—no long-context fine-tuning, no specialized embedding models.

4. **Zero infrastructure**: A single SQLite file. No vector database, no graph database, no external services.

### 7.2 Limitations

1. **Tool calling dependency**: The model must be capable of deciding when and how to use `vc_read` and `vc_search`. Models with weak tool-calling abilities may not leverage the Virtual Context effectively.

2. **Directory generation quality**: The VC Index is generated by an LLM call. Poor-quality indexing (missing topics, vague conclusions) degrades retrieval effectiveness.

3. **Cold start problem**: With no archived history, the system behaves identically to a standard LLM. Benefits only emerge after sufficient conversation history accumulates.

4. **Keyword search limitations**: The current implementation uses keyword-based search for `vctx_search`. Semantic search (via embeddings) would improve recall but adds dependency complexity.

### 7.3 Future Work

- **Embedding-based semantic search** for the Virtual Context
- **HTTP proxy mode** for universal API compatibility
- **Multi-session isolation** with per-user Virtual Contexts
- **Automatic drain triggering** based on watermark detection
- **Knowledge graph integration** for entity-level reasoning across blocks
- **Evaluation experiments** as outlined in Section 6

---

## 8. Conclusion

We presented VCTX, a system that applies operating system virtual memory principles to LLM context management. By replacing compressed context with a lightweight directory index and preserving original conversation data in a persistent store, VCTX achieves theoretically unbounded context while maintaining constant per-turn cost and high attention quality. The system requires no model modifications, no external infrastructure, and works with any tool-calling LLM. We believe this paradigm—**"remember the index, not the book"**—offers a practical and theoretically grounded path toward truly persistent LLM memory.

---

## References

[1] Liu, N. F., et al. "Lost in the Middle: How Language Models Use Long Transforms." *NeurIPS*, 2023.

[2] Packer, C., et al. "MemGPT: Towards LLMs as Operating Systems." *arXiv:2310.08560*, 2023.

[3] Mem0. "The Memory Layer for AI Agents." https://github.com/mem0ai/mem0

[4] Guo, Z., et al. "LightRAG: Simple and Fast Retrieval-Augmented Generation." *arXiv:2410.05779*, 2024.

[5] Anthropic. "Model Context Protocol Specification." https://modelcontextprotocol.io

---

## License

MIT
