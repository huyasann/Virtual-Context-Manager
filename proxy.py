"""
VCTX proxy: OpenAI-compatible and Anthropic-compatible memory middleware.

The proxy sits between a client and an upstream LLM gateway. It recalls relevant
VCTX blocks before forwarding non-streaming requests, injects a compact memory
section, and archives substantial completed turns as checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


DB_DIR = Path.home() / ".vctx"
DB_PATH = DB_DIR / "memory.db"

UPSTREAM_BASE_URL = os.getenv("VCTX_UPSTREAM_BASE_URL", "").rstrip("/")
UPSTREAM_API_KEY = os.getenv("VCTX_UPSTREAM_API_KEY", "")
RECALL_TOP_K = int(os.getenv("VCTX_RECALL_TOP_K", "3"))
MAX_MEMORY_CHARS = int(os.getenv("VCTX_MAX_MEMORY_CHARS", "12000"))
RECALL_MIN_SCORE = float(os.getenv("VCTX_RECALL_MIN_SCORE", "2.0"))
CHECKPOINT_MIN_CHARS = int(os.getenv("VCTX_CHECKPOINT_MIN_CHARS", "2500"))
CHECKPOINT_EVERY_N_TURNS = int(os.getenv("VCTX_CHECKPOINT_EVERY_N_TURNS", "8"))
CHECKPOINT_STREAMING = os.getenv("VCTX_CHECKPOINT_STREAMING", "1") not in {"0", "false", "False"}
PROJECT_HEADER = os.getenv("VCTX_PROJECT_HEADER", "x-vctx-project")
USER_HEADER = os.getenv("VCTX_USER_HEADER", "x-vctx-user")
SESSION_HEADER = os.getenv("VCTX_SESSION_HEADER", "x-vctx-session")
DEFAULT_PROJECT_ID = os.getenv("VCTX_PROJECT_ID", "")
DEFAULT_USER_ID = os.getenv("VCTX_USER_ID", "")

app = FastAPI(title="vctx-proxy", version="0.1.0")
_turn_counters: dict[str, int] = {}
_turn_buffers: dict[str, list[dict[str, str]]] = {}


@contextmanager
def db_conn():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
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
                project_id    TEXT,
                user_id       TEXT,
                source        TEXT DEFAULT 'proxy',
                fingerprint   TEXT,
                created_at    TEXT,
                last_access   TEXT,
                importance    REAL DEFAULT 1.0,
                access_count  INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_blocks_fingerprint ON blocks(fingerprint);
            CREATE TABLE IF NOT EXISTS proxy_trace (
                trace_id           TEXT PRIMARY KEY,
                started_at         TEXT,
                finished_at        TEXT,
                duration_ms        INTEGER,
                protocol           TEXT,
                path               TEXT,
                stream             INTEGER DEFAULT 0,
                project_id         TEXT,
                user_id            TEXT,
                session_id         TEXT,
                query_preview      TEXT,
                recalled_block_ids TEXT,
                recalled_scores    TEXT,
                injected           INTEGER DEFAULT 0,
                checkpoint_block_id TEXT,
                checkpoint_status  TEXT,
                upstream_status    INTEGER,
                error              TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_proxy_trace_started
                ON proxy_trace(started_at);
            CREATE INDEX IF NOT EXISTS idx_proxy_trace_project
                ON proxy_trace(project_id);
            """
        )
        migrate_schema(conn)
        conn.commit()
        yield conn
        conn.commit()
    finally:
        conn.close()


def migrate_schema(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(blocks)").fetchall()}
    columns = {
        "session_id": "TEXT",
        "project_id": "TEXT",
        "user_id": "TEXT",
        "source": "TEXT DEFAULT 'proxy'",
        "fingerprint": "TEXT",
        "importance": "REAL DEFAULT 1.0",
        "access_count": "INTEGER DEFAULT 0",
    }
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE blocks ADD COLUMN {name} {ddl}")

    existing_trace = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(proxy_trace)").fetchall()
    }
    trace_columns = {
        "checkpoint_status": "TEXT",
    }
    for name, ddl in trace_columns.items():
        if name not in existing_trace:
            conn.execute(f"ALTER TABLE proxy_trace ADD COLUMN {name} {ddl}")


def rough_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


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


def parse_keywords(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def recall_memory(
    query: str,
    top_k: int = RECALL_TOP_K,
    max_chars: int = MAX_MEMORY_CHARS,
    project_id: str = "",
    user_id: str = "",
    min_score: float = RECALL_MIN_SCORE,
) -> list[dict[str, Any]]:
    terms = tokenize_query(query)
    if not terms:
        return []

    with db_conn() as conn:
        filters: list[str] = []
        params: list[Any] = []
        if project_id:
            filters.append("COALESCE(project_id, '') = ?")
            params.append(project_id)
        if user_id:
            filters.append("COALESCE(user_id, '') = ?")
            params.append(user_id)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        rows = conn.execute(
            f"""
            SELECT block_id, title, content, conclusion, keywords, token_count,
                   importance, access_count, project_id, user_id
            FROM blocks
            {where}
            ORDER BY importance DESC, access_count DESC, created_at DESC
            """,
            params,
        ).fetchall()

        hits: list[dict[str, Any]] = []
        for row in rows:
            keywords = parse_keywords(row["keywords"])
            title = str(row["title"] or "").lower()
            conclusion = str(row["conclusion"] or "").lower()
            keyword_text = " ".join(str(item) for item in keywords).lower()
            content = str(row["content"] or "").lower()
            title_hits = [term for term in terms if term in title]
            conclusion_hits = [term for term in terms if term in conclusion]
            keyword_hits = [term for term in terms if term in keyword_text]
            content_hits = [term for term in terms if term in content]
            matched = sorted(set(title_hits + conclusion_hits + keyword_hits + content_hits))
            score = (
                len(title_hits) * 4
                + len(keyword_hits) * 3
                + len(conclusion_hits) * 2
                + len(content_hits) * 1
                + float(row["importance"] or 1.0) * 0.2
                + min(int(row["access_count"] or 0), 20) * 0.05
            )
            if not matched or score < min_score:
                continue
            hits.append(
                {
                    "block_id": row["block_id"],
                    "title": row["title"],
                    "content": row["content"],
                    "conclusion": row["conclusion"],
                    "keywords": keywords,
                    "score": score,
                    "token_count": row["token_count"],
                    "matched_terms": matched,
                    "project_id": row["project_id"] or "",
                    "user_id": row["user_id"] or "",
                }
            )

        hits.sort(key=lambda item: (-item["score"], item["title"] or ""))
        selected: list[dict[str, Any]] = []
        used = 0
        for hit in hits[: max(top_k * 3, top_k)]:
            content = hit["content"] or ""
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(content) > remaining:
                content = content[:remaining] + "\n[truncated]"
            hit = {**hit, "content": content}
            selected.append(hit)
            used += len(content)
            if len(selected) >= top_k:
                break

        for hit in selected:
            conn.execute(
                "UPDATE blocks SET access_count=access_count+1, last_access=? WHERE block_id=?",
                (datetime.now().isoformat(), hit["block_id"]),
            )

    return selected


def format_memory(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return ""
    parts = [
        "Relevant VCTX memory. Use as background context; do not quote it unless useful."
    ]
    for idx, memory in enumerate(memories, 1):
        parts.append(
            "\n".join(
                [
                    f"[{idx}] {memory['title']} ({memory['block_id']}, score={memory.get('score', 0):.2f})",
                    f"Conclusion: {memory.get('conclusion') or ''}",
                    f"Keywords: {', '.join(memory.get('keywords') or [])}",
                    f"Matched: {', '.join(memory.get('matched_terms') or [])}",
                    "Content:",
                    memory.get("content") or "",
                ]
            )
        )
    return "\n\n".join(parts)


def request_scope(request: Request) -> dict[str, str]:
    project_id = request.headers.get(PROJECT_HEADER, DEFAULT_PROJECT_ID).strip()
    user_id = request.headers.get(USER_HEADER, DEFAULT_USER_ID).strip()
    session_id = request.headers.get(SESSION_HEADER, "").strip()
    if not session_id:
        session_id = project_id or user_id or "vctx-proxy"
    return {"project_id": project_id, "user_id": user_id, "session_id": session_id}


def preview_text(text: str, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def start_trace(
    *,
    protocol: str,
    path: str,
    stream: bool,
    scope: dict[str, str],
    user_text: str,
    memories: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "trace_id": uuid.uuid4().hex[:16],
        "started_at": datetime.now().isoformat(),
        "started_monotonic": time.perf_counter(),
        "protocol": protocol,
        "path": path,
        "stream": stream,
        "project_id": scope["project_id"],
        "user_id": scope["user_id"],
        "session_id": scope["session_id"],
        "query_preview": preview_text(user_text),
        "recalled_block_ids": [memory["block_id"] for memory in memories],
        "recalled_scores": [round(float(memory.get("score") or 0.0), 3) for memory in memories],
        "injected": bool(memories),
        "checkpoint_status": "not_attempted",
    }


def finish_trace(
    trace: dict[str, Any],
    *,
    upstream_status: int | None = None,
    checkpoint_block_id: str | None = None,
    checkpoint_status: str | None = None,
    error: str = "",
) -> None:
    finished_at = datetime.now().isoformat()
    duration_ms = int((time.perf_counter() - float(trace["started_monotonic"])) * 1000)
    try:
        with db_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO proxy_trace
                (trace_id, started_at, finished_at, duration_ms, protocol, path,
                 stream, project_id, user_id, session_id, query_preview,
                 recalled_block_ids, recalled_scores, injected, checkpoint_block_id,
                 checkpoint_status, upstream_status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace["trace_id"],
                    trace["started_at"],
                    finished_at,
                    duration_ms,
                    trace["protocol"],
                    trace["path"],
                    1 if trace["stream"] else 0,
                    trace["project_id"],
                    trace["user_id"],
                    trace["session_id"],
                    trace["query_preview"],
                    json.dumps(trace["recalled_block_ids"], ensure_ascii=False),
                    json.dumps(trace["recalled_scores"], ensure_ascii=False),
                    1 if trace["injected"] else 0,
                    checkpoint_block_id or "",
                    checkpoint_status or trace.get("checkpoint_status") or "not_attempted",
                    upstream_status,
                    preview_text(error, limit=600),
                ),
            )
    except Exception as exc:
        print(f"[vctx-proxy] trace write skipped: {exc}")


def extract_openai_user_text(payload: dict[str, Any]) -> str:
    for msg in reversed(payload.get("messages", [])):
        if msg.get("role") == "user":
            return content_to_text(msg.get("content"))
    return ""


def extract_anthropic_user_text(payload: dict[str, Any]) -> str:
    for msg in reversed(payload.get("messages", [])):
        if msg.get("role") == "user":
            return content_to_text(msg.get("content"))
    return ""


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def inject_openai_memory(payload: dict[str, Any], memory_text: str) -> dict[str, Any]:
    if not memory_text:
        return payload
    cloned = dict(payload)
    messages = list(cloned.get("messages", []))
    memory_msg = {"role": "system", "content": f"<VCTX_MEMORY>\n{memory_text}\n</VCTX_MEMORY>"}
    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    messages.insert(insert_at, memory_msg)
    cloned["messages"] = messages
    return cloned


def inject_anthropic_memory(payload: dict[str, Any], memory_text: str) -> dict[str, Any]:
    if not memory_text:
        return payload
    cloned = dict(payload)
    existing = cloned.get("system", "")
    memory = f"<VCTX_MEMORY>\n{memory_text}\n</VCTX_MEMORY>"
    if isinstance(existing, str) and existing:
        cloned["system"] = f"{existing}\n\n{memory}"
    elif isinstance(existing, list):
        cloned["system"] = existing + [{"type": "text", "text": memory}]
    else:
        cloned["system"] = memory
    return cloned


def extract_openai_assistant_text(response: dict[str, Any]) -> str:
    try:
        message = response["choices"][0]["message"]
        return content_to_text(message.get("content"))
    except (KeyError, IndexError, TypeError):
        return ""


def extract_anthropic_assistant_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in response.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def maybe_checkpoint(
    user_text: str,
    assistant_text: str,
    session_id: str = "proxy",
    project_id: str = "",
    user_id: str = "",
    protocol: str = "proxy",
) -> str | None:
    if not user_text and not assistant_text:
        return None

    key = f"{protocol}:{project_id}:{user_id}:{session_id}"
    _turn_counters[key] = _turn_counters.get(key, 0) + 1
    turn_buffer = _turn_buffers.setdefault(key, [])
    turn_buffer.append({"user": user_text, "assistant": assistant_text})

    combined = "\n\n".join(
        f"[user]\n{turn['user']}\n\n[assistant]\n{turn['assistant']}"
        for turn in turn_buffer
    )
    if len(combined) < CHECKPOINT_MIN_CHARS and _turn_counters[key] % CHECKPOINT_EVERY_N_TURNS != 0:
        return None
    if len(combined) < CHECKPOINT_MIN_CHARS:
        return None

    block_id = archive_block(
        title=derive_title(user_text),
        content=combined,
        conclusion=derive_conclusion(assistant_text),
        keywords=derive_keywords(combined),
        session_id=session_id,
        project_id=project_id,
        user_id=user_id,
        source="vctx-proxy",
    )
    turn_buffer.clear()
    return block_id


def derive_title(text: str) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else "Proxy checkpoint"
    return line[:80]


def derive_conclusion(text: str) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else "Checkpoint saved by proxy"
    return line[:160]


def derive_keywords(text: str) -> list[str]:
    terms = tokenize_query(text)
    stop = {"the", "and", "for", "this", "that", "with", "from", "user", "assistant"}
    result: list[str] = []
    for term in terms:
        if term not in stop and term not in result:
            result.append(term)
        if len(result) >= 8:
            break
    return result


def archive_block(
    *,
    title: str,
    content: str,
    conclusion: str,
    keywords: list[str],
    session_id: str,
    project_id: str = "",
    user_id: str = "",
    source: str = "vctx-proxy",
) -> str:
    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
    block_id = f"{datetime.now().strftime('%y%m%d')}-{fingerprint[:6]}"
    now = datetime.now().isoformat()
    with db_conn() as conn:
        existing = conn.execute(
            """
            SELECT block_id FROM blocks
            WHERE fingerprint=?
              AND COALESCE(project_id, '')=?
              AND COALESCE(user_id, '')=?
            LIMIT 1
            """,
            (fingerprint, project_id, user_id),
        ).fetchone()
        if existing:
            return existing["block_id"]
        conn.execute(
            """
            INSERT INTO blocks
            (block_id, title, content, token_count, keywords, conclusion,
             session_id, project_id, user_id, source, fingerprint, created_at, last_access, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0)
            """,
            (
                block_id,
                title,
                content,
                rough_tokens(content),
                json.dumps(keywords, ensure_ascii=False),
                conclusion,
                session_id,
                project_id,
                user_id,
                source,
                fingerprint,
                now,
                now,
            ),
        )
    return block_id


def upstream_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    internal_headers = {PROJECT_HEADER.lower(), USER_HEADER.lower(), SESSION_HEADER.lower()}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in {"host", "content-length", "connection"} or lower in internal_headers:
            continue
        headers[key] = value
    if UPSTREAM_API_KEY:
        headers["authorization"] = f"Bearer {UPSTREAM_API_KEY}"
        headers["x-api-key"] = UPSTREAM_API_KEY
    return headers


def upstream_url(path: str) -> str:
    if not UPSTREAM_BASE_URL:
        raise RuntimeError("VCTX_UPSTREAM_BASE_URL is not configured")
    return f"{UPSTREAM_BASE_URL}{path}"


async def forward_json(path: str, payload: dict[str, Any], request: Request) -> Response:
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(upstream_url(path), headers=upstream_headers(request), json=payload)
    return JSONResponse(status_code=response.status_code, content=response.json())


def extract_sse_text(raw: bytes, protocol: str) -> str:
    text = raw.decode("utf-8", errors="ignore")
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if protocol == "openai":
            for choice in event.get("choices", []) or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    parts.append(content)
        else:
            event_type = event.get("type")
            if event_type == "content_block_delta":
                delta = event.get("delta") or {}
                if isinstance(delta.get("text"), str):
                    parts.append(delta["text"])
            elif event_type == "content_block_start":
                block = event.get("content_block") or {}
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif "content" in event:
                parts.append(content_to_text(event.get("content")))
    return "".join(parts)


async def forward_stream(
    path: str,
    payload: dict[str, Any],
    request: Request,
    *,
    user_text: str,
    scope: dict[str, str],
    protocol: str,
    trace: dict[str, Any],
) -> StreamingResponse:
    client = httpx.AsyncClient(timeout=None)
    try:
        upstream = client.stream("POST", upstream_url(path), headers=upstream_headers(request), json=payload)
        response = await upstream.__aenter__()
    except Exception as exc:
        await client.aclose()
        finish_trace(trace, error=str(exc))
        raise
    collected = bytearray()

    async def iterator():
        checkpoint_block_id = None
        checkpoint_status = "not_attempted"
        error = ""
        try:
            async for chunk in response.aiter_bytes():
                if CHECKPOINT_STREAMING:
                    collected.extend(chunk)
                yield chunk
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            if not CHECKPOINT_STREAMING:
                checkpoint_status = "disabled"
            elif not (200 <= response.status_code < 300):
                checkpoint_status = "skipped_upstream_status"
            else:
                assistant_text = extract_sse_text(bytes(collected), protocol)
                checkpoint_block_id = maybe_checkpoint(
                    user_text,
                    assistant_text,
                    session_id=scope["session_id"],
                    project_id=scope["project_id"],
                    user_id=scope["user_id"],
                    protocol=f"{protocol}-stream",
                )
                checkpoint_status = "saved" if checkpoint_block_id else "skipped_threshold"
            finish_trace(
                trace,
                upstream_status=response.status_code,
                checkpoint_block_id=checkpoint_block_id,
                checkpoint_status=checkpoint_status,
                error=error,
            )
            await upstream.__aexit__(None, None, None)
            await client.aclose()

    return StreamingResponse(
        iterator(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "text/event-stream"),
    )


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "upstream_configured": bool(UPSTREAM_BASE_URL),
        "db": str(DB_PATH),
        "recall_top_k": RECALL_TOP_K,
        "max_memory_chars": MAX_MEMORY_CHARS,
        "recall_min_score": RECALL_MIN_SCORE,
        "checkpoint_min_chars": CHECKPOINT_MIN_CHARS,
        "checkpoint_streaming": CHECKPOINT_STREAMING,
        "project_header": PROJECT_HEADER,
        "user_header": USER_HEADER,
        "session_header": SESSION_HEADER,
    }


@app.get("/vctx/status")
async def vctx_status() -> dict[str, Any]:
    with db_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        proxy_blocks = conn.execute(
            "SELECT COUNT(*) FROM blocks WHERE source='vctx-proxy'"
        ).fetchone()[0]
        trace_count = conn.execute("SELECT COUNT(*) FROM proxy_trace").fetchone()[0]
        projects = [
            row[0] or ""
            for row in conn.execute(
                "SELECT DISTINCT COALESCE(project_id, '') FROM blocks ORDER BY 1 LIMIT 50"
            ).fetchall()
        ]
    return {
        "ok": True,
        "db": str(DB_PATH),
        "blocks": total,
        "proxy_checkpoints": proxy_blocks,
        "proxy_traces": trace_count,
        "projects": projects,
        "turn_buffers": {key: len(value) for key, value in _turn_buffers.items()},
    }


def trace_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "trace_id": row["trace_id"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "duration_ms": row["duration_ms"],
        "protocol": row["protocol"],
        "path": row["path"],
        "stream": bool(row["stream"]),
        "project_id": row["project_id"] or "",
        "user_id": row["user_id"] or "",
        "session_id": row["session_id"] or "",
        "query_preview": row["query_preview"] or "",
        "recalled_block_ids": parse_keywords(row["recalled_block_ids"]),
        "recalled_scores": parse_keywords(row["recalled_scores"]),
        "injected": bool(row["injected"]),
        "checkpoint_block_id": row["checkpoint_block_id"] or "",
        "checkpoint_status": row["checkpoint_status"] or "",
        "upstream_status": row["upstream_status"],
        "error": row["error"] or "",
    }


@app.get("/vctx/traces")
async def vctx_traces(limit: int = 20, project: str = "") -> dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    filters: list[str] = []
    params: list[Any] = []
    if project:
        filters.append("COALESCE(project_id, '') = ?")
        params.append(project)
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    with db_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM proxy_trace
            {where}
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
    return {"count": len(rows), "traces": [trace_row_to_dict(row) for row in rows]}


@app.post("/vctx/recall")
async def vctx_recall(request: Request) -> dict[str, Any]:
    payload = await request.json()
    scope = request_scope(request)
    query = str(payload.get("query") or "")
    top_k = int(payload.get("top_k") or RECALL_TOP_K)
    memories = recall_memory(
        query,
        top_k=top_k,
        project_id=scope["project_id"],
        user_id=scope["user_id"],
        min_score=float(payload.get("min_score") or RECALL_MIN_SCORE),
    )
    return {"query": query, "scope": scope, "count": len(memories), "memories": memories}


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request) -> Response:
    payload = await request.json()
    scope = request_scope(request)
    user_text = extract_openai_user_text(payload)
    memories = recall_memory(user_text, project_id=scope["project_id"], user_id=scope["user_id"])
    injected = inject_openai_memory(payload, format_memory(memories))
    trace = start_trace(
        protocol="openai",
        path="/v1/chat/completions",
        stream=bool(payload.get("stream")),
        scope=scope,
        user_text=user_text,
        memories=memories,
    )

    if payload.get("stream"):
        return await forward_stream(
            "/v1/chat/completions",
            injected,
            request,
            user_text=user_text,
            scope=scope,
            protocol="openai",
            trace=trace,
        )

    checkpoint_block_id = None
    checkpoint_status = "not_attempted"
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            upstream_response = await client.post(
                upstream_url("/v1/chat/completions"),
                headers=upstream_headers(request),
                json=injected,
            )

        try:
            data = upstream_response.json()
        except json.JSONDecodeError:
            finish_trace(
                trace,
                upstream_status=upstream_response.status_code,
                checkpoint_status="skipped_non_json",
            )
            return Response(
                status_code=upstream_response.status_code,
                content=upstream_response.content,
                media_type=upstream_response.headers.get("content-type"),
            )

        if upstream_response.is_success:
            checkpoint_block_id = maybe_checkpoint(
                user_text,
                extract_openai_assistant_text(data),
                session_id=scope["session_id"],
                project_id=scope["project_id"],
                user_id=scope["user_id"],
                protocol="openai",
            )
            checkpoint_status = "saved" if checkpoint_block_id else "skipped_threshold"
        else:
            checkpoint_status = "skipped_upstream_status"
        finish_trace(
            trace,
            upstream_status=upstream_response.status_code,
            checkpoint_block_id=checkpoint_block_id,
            checkpoint_status=checkpoint_status,
        )
        return JSONResponse(status_code=upstream_response.status_code, content=data)
    except Exception as exc:
        finish_trace(trace, error=str(exc))
        raise


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Response:
    payload = await request.json()
    scope = request_scope(request)
    user_text = extract_anthropic_user_text(payload)
    memories = recall_memory(user_text, project_id=scope["project_id"], user_id=scope["user_id"])
    injected = inject_anthropic_memory(payload, format_memory(memories))
    trace = start_trace(
        protocol="anthropic",
        path="/v1/messages",
        stream=bool(payload.get("stream")),
        scope=scope,
        user_text=user_text,
        memories=memories,
    )

    if payload.get("stream"):
        return await forward_stream(
            "/v1/messages",
            injected,
            request,
            user_text=user_text,
            scope=scope,
            protocol="anthropic",
            trace=trace,
        )

    checkpoint_block_id = None
    checkpoint_status = "not_attempted"
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            upstream_response = await client.post(
                upstream_url("/v1/messages"),
                headers=upstream_headers(request),
                json=injected,
            )

        try:
            data = upstream_response.json()
        except json.JSONDecodeError:
            finish_trace(
                trace,
                upstream_status=upstream_response.status_code,
                checkpoint_status="skipped_non_json",
            )
            return Response(
                status_code=upstream_response.status_code,
                content=upstream_response.content,
                media_type=upstream_response.headers.get("content-type"),
            )

        if upstream_response.is_success:
            checkpoint_block_id = maybe_checkpoint(
                user_text,
                extract_anthropic_assistant_text(data),
                session_id=scope["session_id"],
                project_id=scope["project_id"],
                user_id=scope["user_id"],
                protocol="anthropic",
            )
            checkpoint_status = "saved" if checkpoint_block_id else "skipped_threshold"
        else:
            checkpoint_status = "skipped_upstream_status"
        finish_trace(
            trace,
            upstream_status=upstream_response.status_code,
            checkpoint_block_id=checkpoint_block_id,
            checkpoint_status=checkpoint_status,
        )
        return JSONResponse(status_code=upstream_response.status_code, content=data)
    except Exception as exc:
        finish_trace(trace, error=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VCTX proxy.")
    parser.add_argument("--host", default=os.getenv("VCTX_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VCTX_PROXY_PORT", "8787")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("proxy:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
