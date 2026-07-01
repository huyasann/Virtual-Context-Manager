"""
vctx-mcp: Virtual Context Manager MCP server.

The server stores conversation blocks verbatim in SQLite, exposes a compact
directory through MCP tools, and lets the model read full blocks on demand.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


DB_DIR = Path.home() / ".vctx"
DB_PATH = DB_DIR / "memory.db"

DRAIN_THRESHOLD = 160_000
KEEP_RECENT_TOKENS = 20_000
ARCHIVE_SLICE_TOKENS = 6_000


try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))

except ImportError:

    def count_tokens(text: str) -> int:
        text = text or ""
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin = len(text) - cjk
        return max(1, int(cjk * 1.5 + latin / 4))


try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _EMBED_MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    _HAS_EMBEDDING = True

    def embed_text(text: str) -> list[float]:
        return _EMBED_MODEL.encode(text or "").tolist()

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        if denom == 0:
            return 0.0
        return float(np.dot(a_np, b_np) / denom)

except ImportError:
    _HAS_EMBEDDING = False

    def embed_text(text: str) -> list[float]:
        return []

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        return 0.0


@contextmanager
def db_conn():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(
            """
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
                embedding     TEXT,
                created_at    TEXT,
                last_access   TEXT,
                importance    REAL DEFAULT 1.0,
                access_count  INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS drain_log (
                drain_id         TEXT PRIMARY KEY,
                started_at       TEXT,
                finished_at      TEXT,
                blocks_created   INTEGER,
                skipped_recalled INTEGER,
                skipped_dedup    INTEGER,
                total_tokens     INTEGER
            );
            """
        )
        _migrate_schema(conn)
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_blocks_fingerprint
                ON blocks(fingerprint);
            CREATE INDEX IF NOT EXISTS idx_blocks_session
                ON blocks(session_id);
            CREATE INDEX IF NOT EXISTS idx_blocks_recall
                ON blocks(recall_from);
            """
        )
        conn.commit()
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after early prototype databases were created."""
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(blocks)").fetchall()
    }
    columns = {
        "session_id": "TEXT",
        "source": "TEXT DEFAULT 'manual'",
        "is_recalled": "INTEGER DEFAULT 0",
        "recall_from": "TEXT",
        "fingerprint": "TEXT",
        "embedding": "TEXT",
        "importance": "REAL DEFAULT 1.0",
        "access_count": "INTEGER DEFAULT 0",
    }
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE blocks ADD COLUMN {name} {ddl}")


def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 2
    for msg in messages:
        total += 4
        total += count_tokens(str(msg.get("role", "")))
        total += count_tokens(str(msg.get("content", "")))
    return total


def normalize_keywords(keywords: list[str] | None) -> list[str]:
    if not keywords:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for keyword in keywords:
        item = str(keyword).strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result[:12]


def json_dumps(data: Any, *, indent: int | None = None) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent)


def make_block_id(content: str) -> tuple[str, str]:
    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{datetime.now().strftime('%y%m%d')}-{fingerprint[:6]}", fingerprint


def tokenize_query(text: str) -> list[str]:
    text = (text or "").lower()
    terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", text)
    singles = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = [a + b for a, b in zip(singles, singles[1:])]
    seen: set[str] = set()
    result: list[str] = []
    for term in terms + bigrams:
        if term not in seen:
            seen.add(term)
            result.append(term)
    return result


def row_keywords(row: sqlite3.Row) -> list[str]:
    if not row["keywords"]:
        return []
    try:
        loaded = json.loads(row["keywords"])
        return loaded if isinstance(loaded, list) else []
    except json.JSONDecodeError:
        return []


def build_embedding(title: str, conclusion: str, keywords: list[str]) -> str | None:
    if not _HAS_EMBEDDING:
        return None
    text = f"{title} {conclusion} {' '.join(keywords)}"
    return json.dumps(embed_text(text))


def _fingerprint_exists(conn: sqlite3.Connection, fingerprint: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT block_id FROM blocks WHERE fingerprint=? LIMIT 1",
        (fingerprint,),
    ).fetchone()


def _update_recall_access(conn: sqlite3.Connection, block_id: str) -> None:
    conn.execute(
        "UPDATE blocks SET access_count=access_count+1, last_access=? WHERE block_id=?",
        (datetime.now().isoformat(), block_id),
    )


def _slice_messages(messages: list[dict[str, Any]], max_tokens: int) -> list[tuple[str, int]]:
    slices: list[tuple[str, int]] = []
    current: list[str] = []
    current_tokens = 0

    for msg in messages:
        text = f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}\n"
        tokens = count_tokens(text)
        if current and current_tokens + tokens > max_tokens:
            slices.append(("".join(current), current_tokens))
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += tokens

    if current:
        slices.append(("".join(current), current_tokens))
    return slices


def _extract_title(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    title = re.sub(r"^\[(user|assistant|system)\]:\s*", "", first_line, flags=re.I)
    title = title.strip()
    return title[:60] if title else "Untitled conversation"


def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,6}", text)
    stop_words = {
        "the", "and", "for", "this", "that", "with", "from", "have", "has",
        "我们", "你们", "他们", "这个", "那个", "什么", "怎么", "可以",
        "但是", "因为", "所以", "如果", "然后", "已经", "一个",
    }
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for word in words:
        label = word.strip()
        key = label.lower()
        if len(key) < 2 or key in stop_words:
            continue
        counts[key] = counts.get(key, 0) + 1
        labels.setdefault(key, label)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [labels[key] for key, _ in ranked[:8]]


def _extract_conclusion(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.startswith("[assistant]:"):
            conclusion = line[len("[assistant]:") :].strip()
            return conclusion[:120] if conclusion else "No assistant conclusion"
    return text.strip()[:120] or "No conclusion"


def _log_drain(conn: sqlite3.Connection, stats: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO drain_log
        (drain_id, started_at, finished_at, blocks_created,
         skipped_recalled, skipped_dedup, total_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            stats["drain_id"],
            stats["started_at"],
            stats["finished_at"],
            stats["blocks_created"],
            stats["skipped_recalled"],
            stats["skipped_dedup"],
            stats["total_tokens"],
        ),
    )


_archive_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)
_worker_started = False
_worker_lock = threading.Lock()
_session_buffers: dict[str, list[dict[str, Any]]] = {}
_session_buffer_lock = threading.Lock()


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        thread = threading.Thread(target=_archive_worker, daemon=True)
        thread.start()
        _worker_started = True


def _archive_worker() -> None:
    while True:
        try:
            task = _archive_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            _process_archive_task(task)
        except Exception as exc:
            print(f"[vctx] archive worker error: {exc}")
        finally:
            _archive_queue.task_done()


def _process_archive_task(task: dict[str, Any]) -> None:
    drain_id = task["drain_id"]
    messages = task["messages"]
    session_id = task["session_id"]
    started_at = task["started_at"]

    stats = {
        "drain_id": drain_id,
        "started_at": started_at,
        "blocks_created": 0,
        "skipped_recalled": 0,
        "skipped_dedup": 0,
        "total_tokens": 0,
        "finished_at": datetime.now().isoformat(),
    }

    with db_conn() as conn:
        extractable: list[dict[str, Any]] = []
        for msg in messages:
            metadata = msg.get("metadata") or {}
            recall_from = metadata.get("recall_from")
            if recall_from:
                _update_recall_access(conn, str(recall_from))
                stats["skipped_recalled"] += 1
            else:
                extractable.append(msg)

        for slice_text, slice_tokens in _slice_messages(extractable, ARCHIVE_SLICE_TOKENS):
            block_id, fingerprint = make_block_id(slice_text)
            if _fingerprint_exists(conn, fingerprint):
                stats["skipped_dedup"] += 1
                continue

            title = _extract_title(slice_text)
            keywords = _extract_keywords(slice_text)
            conclusion = _extract_conclusion(slice_text)
            embedding = build_embedding(title, conclusion, keywords)
            now = datetime.now().isoformat()

            conn.execute(
                """
                INSERT INTO blocks
                (block_id, title, content, token_count, keywords,
                 conclusion, session_id, source, is_recalled,
                 recall_from, fingerprint, embedding, created_at, last_access)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'drain', 0, NULL, ?, ?, ?, ?)
                """,
                (
                    block_id,
                    title,
                    slice_text,
                    slice_tokens,
                    json.dumps(keywords, ensure_ascii=False),
                    conclusion,
                    session_id,
                    fingerprint,
                    embedding,
                    now,
                    now,
                ),
            )
            stats["blocks_created"] += 1
            stats["total_tokens"] += slice_tokens

        stats["finished_at"] = datetime.now().isoformat()
        _log_drain(conn, stats)


def _do_drain(session_id: str) -> str:
    drain_id = str(uuid.uuid4())[:12]
    messages = _session_buffers.get(session_id, [])
    if not messages:
        return drain_id

    snapshot = [dict(msg) for msg in messages]
    kept: list[dict[str, Any]] = []
    kept_tokens = 0

    for msg in reversed(snapshot):
        msg_tokens = count_tokens(str(msg.get("content", ""))) + 4
        if kept and kept_tokens + msg_tokens > KEEP_RECENT_TOKENS:
            break
        kept.insert(0, msg)
        kept_tokens += msg_tokens

    evicted = snapshot[: len(snapshot) - len(kept)]
    _session_buffers[session_id] = kept

    if not evicted:
        return drain_id

    task = {
        "drain_id": drain_id,
        "messages": evicted,
        "session_id": session_id,
        "started_at": datetime.now().isoformat(),
    }
    try:
        _archive_queue.put_nowait(task)
    except queue.Full:
        _process_archive_task(task)
    return drain_id


mcp = FastMCP(
    "vctx",
    instructions=(
        "Virtual Context Manager. Use vctx_buffer to record conversation turns, "
        "vctx_list or vctx_search to inspect archived memory, and vctx_read to "
        "load full archived blocks when the user asks about prior work."
    ),
)


@mcp.tool()
def vctx_buffer(
    session_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Append a message to a session buffer and trigger drain at the high watermark."""
    _ensure_worker()
    if role not in {"user", "assistant", "system", "tool"}:
        return json_dumps({"error": "role must be one of user, assistant, system, tool"})

    with _session_buffer_lock:
        buffer = _session_buffers.setdefault(session_id, [])
        msg: dict[str, Any] = {"role": role, "content": content}
        if metadata:
            msg["metadata"] = metadata
        buffer.append(msg)
        buffer_tokens = count_messages_tokens(buffer)

        if buffer_tokens >= DRAIN_THRESHOLD:
            drain_id = _do_drain(session_id)
            return json_dumps(
                {
                    "status": "drain_triggered",
                    "session_id": session_id,
                    "drain_id": drain_id,
                    "token_count": buffer_tokens,
                    "threshold": DRAIN_THRESHOLD,
                    "message": "Buffer reached the high watermark; archival drain has been queued.",
                }
            )

        return json_dumps(
            {
                "status": "buffered",
                "session_id": session_id,
                "messages": len(buffer),
                "token_count": buffer_tokens,
                "drain_triggered": False,
                "threshold": DRAIN_THRESHOLD,
                "usage": f"{buffer_tokens / DRAIN_THRESHOLD:.1%}",
            }
        )


@mcp.tool()
def vctx_archive(
    title: str,
    content: str,
    conclusion: str,
    keywords: list[str],
    session_id: str = "default",
) -> str:
    """Manually archive an important conversation or knowledge block."""
    block_id, fingerprint = make_block_id(content)
    normalized_keywords = normalize_keywords(keywords)
    embedding = build_embedding(title, conclusion, normalized_keywords)
    now = datetime.now().isoformat()

    with db_conn() as conn:
        existing = _fingerprint_exists(conn, fingerprint)
        if existing:
            return json_dumps(
                {
                    "status": "duplicate",
                    "block_id": existing["block_id"],
                    "message": "This content is already archived.",
                }
            )

        conn.execute(
            """
            INSERT INTO blocks
            (block_id, title, content, token_count, keywords,
             conclusion, session_id, source, fingerprint, embedding,
             created_at, last_access, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'manual', ?, ?, ?, ?, 1.0)
            """,
            (
                block_id,
                title,
                content,
                count_tokens(content),
                json.dumps(normalized_keywords, ensure_ascii=False),
                conclusion,
                session_id,
                fingerprint,
                embedding,
                now,
                now,
            ),
        )

    return json_dumps(
        {
            "status": "archived",
            "block_id": block_id,
            "title": title,
            "token_count": count_tokens(content),
        }
    )


@mcp.tool()
def vctx_read(block_id: str) -> str:
    """Read a full archived block by block_id."""
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM blocks WHERE block_id = ?", (block_id,)).fetchone()
        if not row:
            return json_dumps({"error": f"Block not found: {block_id}"})

        conn.execute(
            "UPDATE blocks SET last_access=?, access_count=access_count+1 WHERE block_id=?",
            (datetime.now().isoformat(), block_id),
        )

        return json_dumps(
            {
                "block_id": row["block_id"],
                "title": row["title"],
                "content": row["content"],
                "conclusion": row["conclusion"],
                "keywords": row_keywords(row),
                "token_count": row["token_count"],
                "access_count": row["access_count"] + 1,
                "recall_hint": {"block_id": row["block_id"], "source": "vctx"},
            }
        )


@mcp.tool()
def vctx_search(query: str, top_k: int = 5) -> str:
    """Search archived blocks with keyword scoring and optional embedding similarity."""
    top_k = max(1, min(int(top_k), 20))
    query_terms = tokenize_query(query)
    query_embedding = embed_text(query) if _HAS_EMBEDDING else None

    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT block_id, title, conclusion, keywords, token_count,
                   importance, access_count, embedding
            FROM blocks
            ORDER BY importance DESC, access_count DESC, created_at DESC
            """
        ).fetchall()

    hits: list[dict[str, Any]] = []
    for row in rows:
        keywords = row_keywords(row)
        searchable = f"{row['title']} {row['conclusion']} {' '.join(keywords)}".lower()
        matched = [term for term in query_terms if term in searchable]

        keyword_score = float(len(matched))
        title_boost = 1.0 if query.lower() and query.lower() in str(row["title"]).lower() else 0.0
        semantic_score = 0.0
        if query_embedding and row["embedding"]:
            try:
                semantic_score = cosine_similarity(query_embedding, json.loads(row["embedding"]))
            except (json.JSONDecodeError, TypeError, ValueError):
                semantic_score = 0.0

        score = keyword_score * 2.0 + title_boost + semantic_score * 3.0
        if score <= 0:
            continue

        hits.append(
            {
                "block_id": row["block_id"],
                "title": row["title"],
                "conclusion": row["conclusion"],
                "token_count": row["token_count"],
                "score": round(score, 3),
                "keyword_score": keyword_score,
                "semantic_score": round(semantic_score, 3) if _HAS_EMBEDDING else None,
                "matched_terms": matched,
            }
        )

    hits.sort(key=lambda item: (-item["score"], item["title"]))
    return json_dumps(
        {
            "query": query,
            "count": min(len(hits), top_k),
            "results": hits[:top_k],
            "search_mode": "hybrid" if _HAS_EMBEDDING else "keyword",
            "message": None if hits else "No matching archived blocks found.",
        }
    )


@mcp.tool()
def vctx_list() -> str:
    """List archived virtual-context blocks."""
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT block_id, title, conclusion, keywords, token_count,
                   importance, access_count, created_at, last_access
            FROM blocks
            ORDER BY importance DESC, created_at DESC
            """
        ).fetchall()

    blocks: list[dict[str, Any]] = []
    total_tokens = 0
    for row in rows:
        age_days = 0
        if row["last_access"]:
            try:
                age_days = (datetime.now() - datetime.fromisoformat(row["last_access"])).days
            except ValueError:
                age_days = 0
        total_tokens += row["token_count"] or 0
        blocks.append(
            {
                "id": row["block_id"],
                "title": row["title"],
                "conclusion": row["conclusion"],
                "keywords": row_keywords(row),
                "tokens": row["token_count"],
                "importance": round(row["importance"], 3),
                "accessed": row["access_count"],
                "days_since_access": age_days,
            }
        )

    return json_dumps(
        {
            "message": None if blocks else "Virtual context is empty.",
            "total_blocks": len(blocks),
            "total_tokens": total_tokens,
            "blocks": blocks,
        },
        indent=2,
    )


@mcp.tool()
def vctx_index(block_id: str, title: str, conclusion: str, keywords: list[str]) -> str:
    """Update the directory metadata for an archived block."""
    normalized_keywords = normalize_keywords(keywords)
    embedding = build_embedding(title, conclusion, normalized_keywords)

    with db_conn() as conn:
        row = conn.execute("SELECT block_id FROM blocks WHERE block_id=?", (block_id,)).fetchone()
        if not row:
            return json_dumps({"error": f"Block not found: {block_id}"})

        conn.execute(
            """
            UPDATE blocks
            SET title=?, conclusion=?, keywords=?, embedding=?
            WHERE block_id=?
            """,
            (
                title,
                conclusion,
                json.dumps(normalized_keywords, ensure_ascii=False),
                embedding,
                block_id,
            ),
        )

    return json_dumps(
        {
            "status": "indexed",
            "block_id": block_id,
            "title": title,
            "conclusion": conclusion,
            "keywords": normalized_keywords,
        }
    )


@mcp.tool()
def vctx_decay(days_threshold: int = 30) -> str:
    """Apply temporal decay to blocks not accessed for at least days_threshold days."""
    now = datetime.now()
    decayed = 0
    evicted_candidates = 0

    with db_conn() as conn:
        rows = conn.execute("SELECT block_id, last_access, importance FROM blocks").fetchall()
        for row in rows:
            if not row["last_access"]:
                continue
            try:
                last_access = datetime.fromisoformat(row["last_access"])
            except ValueError:
                continue
            days = (now - last_access).days
            if days < days_threshold:
                continue

            new_importance = round(float(row["importance"]) * (0.97**days), 4)
            conn.execute(
                "UPDATE blocks SET importance=? WHERE block_id=?",
                (new_importance, row["block_id"]),
            )
            decayed += 1
            if new_importance < 0.1:
                evicted_candidates += 1

    return json_dumps(
        {
            "decayed": decayed,
            "evicted_candidates": evicted_candidates,
            "threshold_days": days_threshold,
            "message": (
                f"Decayed {decayed} block(s); {evicted_candidates} block(s) "
                "are below importance 0.1 and may be deleted."
            ),
        }
    )


@mcp.tool()
def vctx_delete(block_id: str) -> str:
    """Delete an archived block."""
    with db_conn() as conn:
        row = conn.execute("SELECT title FROM blocks WHERE block_id=?", (block_id,)).fetchone()
        if not row:
            return json_dumps({"error": f"Block not found: {block_id}"})
        conn.execute("DELETE FROM blocks WHERE block_id=?", (block_id,))

    return json_dumps({"status": "deleted", "block_id": block_id, "title": row["title"]})


@mcp.tool()
def vctx_status(session_id: str = "default") -> str:
    """Return buffer, archive, drain, worker, and storage diagnostics."""
    with _session_buffer_lock:
        buffer = _session_buffers.get(session_id, [])
        buffer_messages = len(buffer)
        buffer_tokens = count_messages_tokens(buffer)

    with db_conn() as conn:
        block_count = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        total_tokens = conn.execute("SELECT COALESCE(SUM(token_count), 0) FROM blocks").fetchone()[0]
        last_drain = conn.execute(
            "SELECT * FROM drain_log ORDER BY started_at DESC LIMIT 1"
        ).fetchone()

    drain_info = None
    if last_drain:
        drain_info = {
            "drain_id": last_drain["drain_id"],
            "started_at": last_drain["started_at"],
            "finished_at": last_drain["finished_at"],
            "blocks_created": last_drain["blocks_created"],
            "tokens_archived": last_drain["total_tokens"],
            "skipped_recalled": last_drain["skipped_recalled"],
            "skipped_dedup": last_drain["skipped_dedup"],
        }

    return json_dumps(
        {
            "session_id": session_id,
            "buffer": {
                "messages": buffer_messages,
                "tokens": buffer_tokens,
                "threshold": DRAIN_THRESHOLD,
                "usage": f"{buffer_tokens / DRAIN_THRESHOLD:.1%}",
            },
            "virtual_context": {
                "blocks": block_count,
                "total_tokens": total_tokens,
                "database": str(DB_PATH),
            },
            "last_drain": drain_info,
            "worker_status": "running" if _worker_started else "not_started",
            "queue_size": _archive_queue.qsize(),
            "embedding_search": _HAS_EMBEDDING,
        },
        indent=2,
    )


if __name__ == "__main__":
    _ensure_worker()
    mcp.run(transport="stdio")
