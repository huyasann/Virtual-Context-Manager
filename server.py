"""
vctx-mcp — 虚拟上下文 MCP Server
让 Claude Code 可以直接读写虚拟上下文

核心功能：
- 真实 token 计数（tiktoken）
- 自动水位线检测 + drain 触发
- LLM 自动生成 VC Index
- 防套娃（recall 标记 + 过滤）
- 异步归档队列
- 时间衰减换血
"""
import json
import sqlite3
import hashlib
import threading
import queue
import re
from datetime import datetime
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# ══════════════════════════════════════════════════════════
# Embedding 模型（可选，有则启用语义搜索）
# ══════════════════════════════════════════════════════════

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    def embed_text(text: str) -> list[float]:
        """将文本转换为 embedding 向量"""
        return _embed_model.encode(text).tolist()

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度"""
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

    _HAS_EMBEDDING = True
except ImportError:
    _HAS_EMBEDDING = False
    def embed_text(text: str) -> list[float]:
        return []
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        return 0.0

# ══════════════════════════════════════════════════════════
# Token 计数（真实实现）
# ══════════════════════════════════════════════════════════

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except ImportError:
    # tiktoken 不可用时降级为粗估（中文约 1.5 char/token）
    def count_tokens(text: str) -> int:
        return len(text) // 2


def count_messages_tokens(messages: list[dict]) -> int:
    """计算一组消息的总 token 数"""
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        total += count_tokens(msg.get("role", ""))
        total += count_tokens(msg.get("content", ""))
    return total + 2


# ══════════════════════════════════════════════════════════
# 水位线配置
# ══════════════════════════════════════════════════════════

DRAIN_THRESHOLD = 160_000    # 超过此 token 数触发 drain
KEEP_RECENT_TOKENS = 20_000  # drain 后保留最近多少 token


# ══════════════════════════════════════════════════════════
# 存储层
# ══════════════════════════════════════════════════════════

DB_DIR = Path.home() / ".vctx"
DB_PATH = DB_DIR / "memory.db"


def get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blocks (
            block_id      TEXT PRIMARY KEY,
            title         TEXT,
            content       TEXT NOT NULL,
            token_count   INTEGER,
            keywords      TEXT,
            conclusion    TEXT,
            session_id    TEXT,
            source        TEXT DEFAULT 'manual',
            is_recalled   INTEGER DEFAULT 0,
            recall_from   TEXT,
            fingerprint   TEXT,
            embedding     BLOB,
            created_at    TEXT,
            last_access   TEXT,
            importance    REAL DEFAULT 1.0,
            access_count  INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_blocks_fingerprint
            ON blocks(fingerprint);
        CREATE INDEX IF NOT EXISTS idx_blocks_recall
            ON blocks(recall_from);

        CREATE TABLE IF NOT EXISTS drain_log (
            drain_id      TEXT PRIMARY KEY,
            started_at    TEXT,
            finished_at   TEXT,
            blocks_created INTEGER,
            skipped_recalled INTEGER,
            skipped_dedup  INTEGER,
            total_tokens   INTEGER
        );

        CREATE TABLE IF NOT EXISTS session_state (
            session_id    TEXT PRIMARY KEY,
            buffer_json   TEXT,
            buffer_tokens INTEGER,
            last_updated  TEXT
        );
    """)
    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════
# 异步归档队列（后台线程）
# ══════════════════════════════════════════════════════════

_archive_queue: queue.Queue = queue.Queue(maxsize=8)
_worker_started = False
_worker_lock = threading.Lock()


def _ensure_worker():
    """确保后台归档线程已启动"""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_archive_worker, daemon=True)
        t.start()
        _worker_started = True


def _archive_worker():
    """后台线程：从队列取出归档任务并执行"""
    while True:
        try:
            task = _archive_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            _process_archive_task(task)
        except Exception as e:
            print(f"[vctx] archive worker error: {e}")


def _process_archive_task(task: dict):
    """
    处理一次归档任务。
    task = {
        "drain_id": str,
        "messages": list[dict],
        "session_id": str,
        "started_at": str,
    }
    """
    conn = get_conn()
    drain_id = task["drain_id"]
    messages = task["messages"]
    session_id = task["session_id"]
    started_at = task["started_at"]
    now = datetime.now().isoformat()

    stats = {
        "drain_id": drain_id,
        "started_at": started_at,
        "blocks_created": 0,
        "skipped_recalled": 0,
        "skipped_dedup": 0,
        "total_tokens": 0,
    }

    # ── Phase 1: 过滤 recall 内容（防套娃 Layer 2）──
    extractable = []
    for msg in messages:
        meta = msg.get("metadata", {})
        if meta.get("recall_from"):
            # 这是从知识库召回的内容，只更新访问计数，不重新归档
            _update_recall_access(conn, meta["recall_from"])
            stats["skipped_recalled"] += 1
        else:
            extractable.append(msg)

    if not extractable:
        stats["finished_at"] = datetime.now().isoformat()
        _log_drain(conn, stats)
        return

    # ── Phase 2: 按主题窗口切片（每块 ~4k-8k tokens）──
    slices = _slice_messages(extractable, max_tokens=6000)

    # ── Phase 3: 对每个切片做 fingerprint 去重 + 存储 ──
    for slice_text, slice_tokens in slices:
        fp = hashlib.sha256(slice_text.encode()).hexdigest()

        # Layer 3: fingerprint 去重
        if _fingerprint_exists(conn, fp):
            stats["skipped_dedup"] += 1
            continue

        block_id = f"{datetime.now().strftime('%y%m%d')}-{fp[:6]}"

        # 提取主题名和关键词（轻量级，不用调 LLM）
        title = _extract_title(slice_text)
        keywords = _extract_keywords(slice_text)
        conclusion = _extract_conclusion(slice_text)

        # 计算 embedding（如果可用）
        emb = None
        if _HAS_EMBEDDING:
            emb_text = f"{title} {conclusion} {' '.join(keywords)}"
            emb = embed_text(emb_text)
            emb = json.dumps(emb)
        else:
            emb = None

        conn.execute("""
            INSERT OR IGNORE INTO blocks
            (block_id, title, content, token_count, keywords,
             conclusion, session_id, source, is_recalled,
             recall_from, fingerprint, embedding, created_at, last_access)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'drain', 0, NULL, ?, ?, ?, ?)
        """, (
            block_id, title, slice_text, slice_tokens,
            json.dumps(keywords, ensure_ascii=False),
            conclusion, session_id, fp, emb, now, now,
        ))
        stats["blocks_created"] += 1
        stats["total_tokens"] += slice_tokens

    conn.commit()
    stats["finished_at"] = datetime.now().isoformat()
    _log_drain(conn, stats)


def _slice_messages(messages: list[dict], max_tokens: int) -> list[tuple]:
    """将消息列表切分为多个文本块"""
    slices = []
    current = []
    current_tokens = 0

    for msg in messages:
        text = f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}\n"
        tokens = count_tokens(text)

        if current_tokens + tokens > max_tokens and current:
            slices.append(("".join(current), current_tokens))
            current = []
            current_tokens = 0

        current.append(text)
        current_tokens += tokens

    if current:
        slices.append(("".join(current), current_tokens))

    return slices


def _extract_title(text: str) -> str:
    """从对话文本中提取主题标题（轻量级，不调 LLM）"""
    # 取前 100 字符作为标题候选
    first_line = text.strip().split("\n")[0]
    title = re.sub(r'^\[(user|assistant|system)\]:\s*', '', first_line, flags=re.IGNORECASE)
    title = title[:30].strip()
    return title if title else "未命名对话"


def _extract_keywords(text: str) -> list[str]:
    """从对话文本中提取关键词（基于词频的轻量级实现）"""
    # 提取中文词和英文词
    zh_words = re.findall(r'[一-鿿]{2,6}', text)
    en_words = re.findall(r'[A-Za-z][A-Za-z0-9_]{2,}', text)

    # 过滤常见停用词
    stop_words = {
        '我们', '你们', '他们', '这个', '那个', '什么', '怎么', '可以',
        '但是', '因为', '所以', '如果', '虽然', '不过', '然后', '已经',
        'the', 'and', 'for', 'this', 'that', 'with', 'from', 'are',
        'was', 'were', 'been', 'have', 'has', 'had', 'not', 'but',
    }

    # 统计词频
    word_count = {}
    for w in zh_words + en_words:
        wl = w.lower()
        if wl not in stop_words and len(wl) > 1:
            word_count[wl] = word_count.get(wl, 0) + 1

    # 按频率排序取前 5 个
    sorted_words = sorted(word_count.items(), key=lambda x: -x[1])
    return [w for w, _ in sorted_words[:5]]


def _extract_conclusion(text: str) -> str:
    """从对话文本中提取关键结论（取最后一轮对话的核心内容）"""
    lines = text.strip().split("\n")
    # 找到最后一个 assistant 回复
    last_assistant = ""
    for line in reversed(lines):
        if line.startswith("[assistant]:"):
            last_assistant = line[len("[assistant]:"):].strip()
            break

    # 取前 50 字符作为结论
    conclusion = last_assistant[:50] if last_assistant else ""
    return conclusion if conclusion else "无结论"


def _update_recall_access(conn, block_id: str):
    """更新被 recall 的块的访问计数"""
    conn.execute(
        "UPDATE blocks SET access_count=access_count+1, last_access=? WHERE block_id=?",
        (datetime.now().isoformat(), block_id)
    )
    conn.commit()


def _fingerprint_exists(conn, fp: str) -> bool:
    """检查 fingerprint 是否已存在"""
    row = conn.execute(
        "SELECT 1 FROM blocks WHERE fingerprint=? LIMIT 1", (fp,)
    ).fetchone()
    return row is not None


def _log_drain(conn, stats: dict):
    """记录 drain 日志"""
    conn.execute("""
        INSERT INTO drain_log
        (drain_id, started_at, finished_at, blocks_created,
         skipped_recalled, skipped_dedup, total_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        stats["drain_id"], stats["started_at"], stats["finished_at"],
        stats["blocks_created"], stats["skipped_recalled"],
        stats["skipped_dedup"], stats["total_tokens"],
    ))
    conn.commit()


# ══════════════════════════════════════════════════════════
# MCP Server
# ══════════════════════════════════════════════════════════

mcp = FastMCP(
    "vctx",
    instructions="""虚拟上下文管理器 — 让 AI 拥有无限长期记忆。

核心工作流：
1. 对话进行中，不断调用 vctx_buffer 追加消息
2. vctx_buffer 会自动检测水位线，如果超过阈值会自动触发 drain
3. drain 时会自动将旧消息归档到虚拟上下文，保留最近 20k tokens
4. 需要回忆过去内容时，先用 vctx_list 看目录，再用 vctx_read 读取
5. 如果不确定在哪里，用 vctx_search 搜索

重要：不要等上下文快满了才想起来存档，每轮对话后都应该调用 vctx_buffer 追加消息。
""",
)


# ────────────────────────────────────────────────────────
# P0: 缓冲区 + 自动水位线检测
# ────────────────────────────────────────────────────────

# 每个 session 的消息缓冲区
_session_buffers: dict[str, list[dict]] = {}
_session_buffer_lock = threading.Lock()


@mcp.tool()
def vctx_buffer(
    session_id: str,
    role: str,
    content: str,
    metadata: dict = None,
) -> str:
    """
    将一条消息追加到虚拟上下文缓冲区，并自动检测水位线。

    这是 VCTX 的核心入口。每轮对话后都应该调用此函数。

    参数：
        session_id: 会话 ID（用于区分不同对话）
        role: 消息角色（user / assistant / system）
        content: 消息内容
        metadata: 可选元数据，如果是从知识库召回的内容，包含 recall_from 字段

    返回：
        - 如果未触发 drain: {"status": "buffered", "token_count": N, "drain_triggered": false}
        - 如果触发了 drain: {"status": "drain_triggered", "drain_id": "...", "blocks_archived": N}
    """
    _ensure_worker()

    with _session_buffer_lock:
        if session_id not in _session_buffers:
            _session_buffers[session_id] = []

        msg = {"role": role, "content": content}
        if metadata:
            msg["metadata"] = metadata

        _session_buffers[session_id].append(msg)

        # 计算当前缓冲区 token 数
        buffer_tokens = count_messages_tokens(_session_buffers[session_id])

        # ── 水位线检测 ──
        if buffer_tokens >= DRAIN_THRESHOLD:
            # 触发 drain
            drain_id = _do_drain(session_id)
            return json.dumps({
                "status": "drain_triggered",
                "drain_id": drain_id,
                "session_id": session_id,
                "message": f"上下文达到 {buffer_tokens} tokens，已触发自动归档"
            }, ensure_ascii=False)
        else:
            # 未触发，返回当前状态
            return json.dumps({
                "status": "buffered",
                "session_id": session_id,
                "token_count": buffer_tokens,
                "drain_triggered": False,
                "threshold": DRAIN_THRESHOLD,
                "usage": f"{buffer_tokens / DRAIN_THRESHOLD:.1%}",
            })


def _do_drain(session_id: str) -> str:
    """
    执行 drain：将缓冲区中的旧消息搬进虚拟上下文，只保留最近 N tokens。
    返回 drain_id。
    """
    import uuid

    drain_id = str(uuid.uuid4())[:12]
    now = datetime.now().isoformat()

    messages = _session_buffers.get(session_id, [])
    if not messages:
        return drain_id

    # 深拷贝
    snapshot = [dict(m) for m in messages]

    # 滑窗：从末尾保留 KEEP_RECENT_TOKENS
    kept = []
    kept_tokens = 0
    for msg in reversed(snapshot):
        msg_tokens = count_tokens(msg.get("content", "")) + 4
        if kept_tokens + msg_tokens > KEEP_RECENT_TOKENS:
            break
        kept.insert(0, msg)
        kept_tokens += msg_tokens

    evicted = snapshot[:len(snapshot) - len(kept)]

    # 更新缓冲区为保留的部分
    _session_buffers[session_id] = kept

    # 提交到异步归档队列
    task = {
        "drain_id": drain_id,
        "messages": evicted,
        "session_id": session_id,
        "started_at": now,
    }

    try:
        _archive_queue.put_nowait(task)
    except queue.Full:
        # 队列满了，同步执行（降级方案）
        _process_archive_task(task)

    return drain_id


# ────────────────────────────────────────────────────────
# 归档（手动触发 + AI 主动归档）
# ────────────────────────────────────────────────────────

@mcp.tool()
def vctx_archive(
    title: str,
    content: str,
    conclusion: str,
    keywords: list[str],
    session_id: str = "default",
) -> str:
    """
    将一段对话或知识手动归档到虚拟上下文。

    什么时候调用：
    - 一段重要对话即将超出上下文窗口时
    - 用户要求保存当前讨论内容时
    - 你觉得当前对话包含值得长期记住的信息时

    参数：
        title: 主题名，10字以内，如 "ROS2架构设计"
        content: 要归档的完整对话原文（不要压缩，完整保存）
        conclusion: 一句话关键结论，30字以内
        keywords: 3-5个检索关键词
        session_id: 会话 ID（可选）
    """
    conn = get_conn()
    now = datetime.now().isoformat()

    fp = hashlib.sha256(content.encode()).hexdigest()

    # 防重复
    if _fingerprint_exists(conn, fp):
        existing = conn.execute(
            "SELECT block_id FROM blocks WHERE fingerprint=?", (fp,)
        ).fetchone()
        return json.dumps({
            "status": "duplicate",
            "block_id": existing["block_id"],
            "message": f"该内容已归档，block_id: {existing['block_id']}"
        }, ensure_ascii=False)

    block_id = f"{datetime.now().strftime('%y%m%d')}-{fp[:6]}"

    # 计算 embedding
    emb = None
    if _HAS_EMBEDDING:
        emb_text = f"{title} {conclusion} {' '.join(keywords)}"
        emb = json.dumps(embed_text(emb_text))

    conn.execute("""
        INSERT INTO blocks
        (block_id, title, content, token_count, keywords,
         conclusion, session_id, source, fingerprint, embedding,
         created_at, last_access, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'manual', ?, ?, ?, ?, 1.0)
    """, (
        block_id, title, content, count_tokens(content),
        json.dumps(keywords, ensure_ascii=False),
        conclusion, session_id, fp, emb, now, now,
    ))
    conn.commit()

    return json.dumps({
        "status": "archived",
        "block_id": block_id,
        "title": title,
        "token_count": count_tokens(content),
    }, ensure_ascii=False)


# ────────────────────────────────────────────────────────
# 读取 / 搜索 / 目录
# ────────────────────────────────────────────────────────

@mcp.tool()
def vctx_read(block_id: str) -> str:
    """
    读取虚拟上下文中某个主题块的完整内容。

    什么时候调用：
    - 用户问到过去讨论过的话题，vctx_list() 中有相关主题时
    - 你需要回忆某个具体决定的完整上下文时

    参数：
        block_id: 主题块ID，如 "260630-a1b2c3"
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM blocks WHERE block_id = ?", (block_id,)
    ).fetchone()

    if not row:
        return json.dumps({"error": f"未找到块 {block_id}"})

    # 更新访问记录
    conn.execute(
        "UPDATE blocks SET last_access=?, access_count=access_count+1 "
        "WHERE block_id=?",
        (datetime.now().isoformat(), block_id)
    )
    conn.commit()

    return json.dumps({
        "block_id": row["block_id"],
        "title": row["title"],
        "content": row["content"],
        "conclusion": row["conclusion"],
        "keywords": json.loads(row["keywords"]) if row["keywords"] else [],
        "token_count": row["token_count"],
        "access_count": row["access_count"] + 1,
        "recall_hint": {
            "block_id": row["block_id"],
            "source": "vctx",
        }
    }, ensure_ascii=False)


@mcp.tool()
def vctx_search(query: str, top_k: int = 5) -> str:
    """
    在虚拟上下文中搜索相关信息（混合搜索：关键词 + 语义向量）。

    什么时候调用：
    - 用户提到某个话题，你不确定在哪个块里
    - 你想找所有与某个关键词相关的历史

    参数：
        query: 搜索关键词或短语，如 "数据库选型"
        top_k: 返回结果数量（默认 5）
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT block_id, title, conclusion, keywords, "
        "token_count, importance, access_count, embedding "
        "FROM blocks ORDER BY importance DESC, access_count DESC"
    ).fetchall()

    hits = []
    query_words = query.lower().split()

    # 计算 query 的 embedding（如果可用）
    query_emb = embed_text(query) if _HAS_EMBEDDING else None

    for r in rows:
        score = 0.0

        # ── 关键词匹配分数 ──
        kw = json.loads(r["keywords"]) if r["keywords"] else []
        searchable = f"{r['title']} {r['conclusion']} {' '.join(kw)}".lower()
        matched = [w for w in query_words if w in searchable]
        keyword_score = len(matched)

        # ── 语义相似度分数 ──
        semantic_score = 0.0
        if query_emb and r["embedding"]:
            try:
                block_emb = json.loads(r["embedding"])
                semantic_score = cosine_similarity(query_emb, block_emb)
            except:
                pass

        # ── 混合分数：关键词 * 2 + 语义 * 3 ──
        # 语义搜索权重更高，因为关键词可能漏掉语义相关的内容
        if _HAS_EMBEDDING:
            score = keyword_score * 2.0 + semantic_score * 3.0
        else:
            score = keyword_score * 2.0

        if score > 0:
            hits.append({
                "block_id": r["block_id"],
                "title": r["title"],
                "conclusion": r["conclusion"],
                "token_count": r["token_count"],
                "score": round(score, 3),
                "keyword_score": keyword_score,
                "semantic_score": round(semantic_score, 3) if _HAS_EMBEDDING else None,
                "matched_keywords": matched,
            })

    hits.sort(key=lambda x: -x["score"])

    if not hits:
        return json.dumps({"message": "未找到相关内容", "results": []})

    return json.dumps({
        "query": query,
        "count": len(hits[:top_k]),
        "results": hits[:top_k],
    }, ensure_ascii=False)


@mcp.tool()
def vctx_list() -> str:
    """
    查看虚拟上下文的完整目录。

    什么时候调用：
    - 用户问"你记住了什么"或"我们之前聊过什么"
    - 你想了解有哪些历史信息可用
    - 在做 vctx_read 之前先看目录
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT block_id, title, conclusion, keywords, "
        "token_count, importance, access_count, created_at, last_access "
        "FROM blocks ORDER BY importance DESC, created_at DESC"
    ).fetchall()

    if not rows:
        return json.dumps({
            "message": "虚拟上下文为空，尚无归档内容。",
            "total_blocks": 0,
            "total_tokens": 0,
        })

    blocks = []
    total_tokens = 0
    for r in rows:
        kw = json.loads(r["keywords"]) if r["keywords"] else []
        age_days = 0
        if r["last_access"]:
            try:
                last = datetime.fromisoformat(r["last_access"])
                age_days = (datetime.now() - last).days
            except:
                pass
        blocks.append({
            "id": r["block_id"],
            "title": r["title"],
            "conclusion": r["conclusion"],
            "keywords": kw,
            "tokens": r["token_count"],
            "importance": round(r["importance"], 2),
            "accessed": r["access_count"],
            "days_since_access": age_days,
        })
        total_tokens += r["token_count"] or 0

    return json.dumps({
        "total_blocks": len(blocks),
        "total_tokens": total_tokens,
        "blocks": blocks,
    }, ensure_ascii=False, indent=2)


# ────────────────────────────────────────────────────────
# 衰减 / 删除 / 状态 / LLM Index
# ────────────────────────────────────────────────────────

@mcp.tool()
def vctx_index(block_id: str, title: str, conclusion: str, keywords: list[str]) -> str:
    """
    用 LLM 为一个已归档的块生成高质量目录索引。

    当你读取了一个 vctx 块后，觉得它的自动提取的 title/conclusion/keywords 不够好时，
    调用此工具用你自己的理解来更新目录信息。

    这是 VC Index 的核心：模型自己决定怎么描述每个历史块。

    参数：
        block_id: 要更新的块 ID
        title: 你总结的主题名（10字以内）
        conclusion: 你总结的关键结论（30字以内）
        keywords: 你提取的检索关键词（3-5个）
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT block_id FROM blocks WHERE block_id = ?", (block_id,)
    ).fetchone()

    if not row:
        return json.dumps({"error": f"未找到块 {block_id}"})

    conn.execute("""
        UPDATE blocks SET title=?, conclusion=?, keywords=?
        WHERE block_id=?
    """, (
        title, conclusion,
        json.dumps(keywords, ensure_ascii=False),
        block_id,
    ))
    conn.commit()

    return json.dumps({
        "status": "indexed",
        "block_id": block_id,
        "title": title,
        "conclusion": conclusion,
        "keywords": keywords,
    }, ensure_ascii=False)

@mcp.tool()
def vctx_decay(days_threshold: int = 30) -> str:
    """
    执行时间衰减换血：长期未访问的记忆自动降权。

    规则：
    - 每天衰减 3% 的 importance
    - importance < 0.1 的块会被标记为可淘汰
    - 不会物理删除，只是降低优先级

    什么时候调用：
    - 定期调用（比如每周一次）
    - 虚拟上下文块数超过 50 时清理

    参数：
        days_threshold: 超过多少天未访问的才衰减（默认30天）
    """
    conn = get_conn()
    now = datetime.now()
    decayed = 0
    evicted = 0

    rows = conn.execute("SELECT block_id, last_access, importance FROM blocks").fetchall()

    for r in rows:
        if not r["last_access"]:
            continue
        try:
            last = datetime.fromisoformat(r["last_access"])
        except:
            continue

        days = (now - last).days
        if days < days_threshold:
            continue

        new_importance = r["importance"] * (0.97 ** days)
        new_importance = round(new_importance, 4)

        conn.execute(
            "UPDATE blocks SET importance = ? WHERE block_id = ?",
            (new_importance, r["block_id"])
        )

        if new_importance < 0.1:
            evicted += 1
        decayed += 1

    conn.commit()

    return json.dumps({
        "decayed": decayed,
        "evicted_candidates": evicted,
        "threshold_days": days_threshold,
        "message": f"衰减了 {decayed} 个块，其中 {evicted} 个低于 0.1 可考虑淘汰"
    }, ensure_ascii=False)


@mcp.tool()
def vctx_delete(block_id: str) -> str:
    """
    删除虚拟上下文中的某个块。

    什么时候调用：
    - 用户明确要求删除某段记忆
    - 时间衰减后清理 importance < 0.1 的过期块

    参数：
        block_id: 要删除的主题块ID
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT title FROM blocks WHERE block_id = ?", (block_id,)
    ).fetchone()
    if not row:
        return json.dumps({"error": f"未找到块 {block_id}"})

    conn.execute("DELETE FROM blocks WHERE block_id = ?", (block_id,))
    conn.commit()

    return json.dumps({
        "status": "deleted",
        "block_id": block_id,
        "title": row["title"],
    }, ensure_ascii=False)


@mcp.tool()
def vctx_status(session_id: str = "default") -> str:
    """
    查看虚拟上下文系统的当前状态。

    返回：缓冲区大小、已归档块数、总 token 数等诊断信息。
    """
    conn = get_conn()

    # 缓冲区状态
    with _session_buffer_lock:
        buffer = _session_buffers.get(session_id, [])
        buffer_tokens = count_messages_tokens(buffer)
        buffer_count = len(buffer)

    # 数据库统计
    block_count = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
    total_tokens = conn.execute("SELECT COALESCE(SUM(token_count), 0) FROM blocks").fetchone()[0]

    # 最近一次 drain
    last_drain = conn.execute(
        "SELECT * FROM drain_log ORDER BY started_at DESC LIMIT 1"
    ).fetchone()

    drain_info = None
    if last_drain:
        drain_info = {
            "drain_id": last_drain["drain_id"],
            "time": last_drain["started_at"],
            "blocks_created": last_drain["blocks_created"],
            "tokens_archived": last_drain["total_tokens"],
        }

    return json.dumps({
        "session_id": session_id,
        "buffer": {
            "messages": buffer_count,
            "tokens": buffer_tokens,
            "threshold": DRAIN_THRESHOLD,
            "usage": f"{buffer_tokens / DRAIN_THRESHOLD:.1%}",
        },
        "virtual_context": {
            "blocks": block_count,
            "total_tokens": total_tokens,
        },
        "last_drain": drain_info,
        "worker_status": "running" if _worker_started else "not_started",
        "queue_size": _archive_queue.qsize(),
    }, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    _ensure_worker()
    mcp.run(transport="stdio")
