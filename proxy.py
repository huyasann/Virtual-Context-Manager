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
CHECKPOINT_MIN_CHARS = int(os.getenv("VCTX_CHECKPOINT_MIN_CHARS", "2500"))
CHECKPOINT_EVERY_N_TURNS = int(os.getenv("VCTX_CHECKPOINT_EVERY_N_TURNS", "8"))

app = FastAPI(title="vctx-proxy", version="0.1.0")
_turn_counter = 0
_turn_buffer: list[dict[str, str]] = []


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
                source        TEXT DEFAULT 'proxy',
                fingerprint   TEXT,
                created_at    TEXT,
                last_access   TEXT,
                importance    REAL DEFAULT 1.0,
                access_count  INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_blocks_fingerprint ON blocks(fingerprint);
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
        "source": "TEXT DEFAULT 'proxy'",
        "fingerprint": "TEXT",
        "importance": "REAL DEFAULT 1.0",
        "access_count": "INTEGER DEFAULT 0",
    }
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE blocks ADD COLUMN {name} {ddl}")


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


def recall_memory(query: str, top_k: int = RECALL_TOP_K, max_chars: int = MAX_MEMORY_CHARS) -> list[dict[str, Any]]:
    terms = tokenize_query(query)
    if not terms:
        return []

    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT block_id, title, content, conclusion, keywords, token_count,
                   importance, access_count
            FROM blocks
            ORDER BY importance DESC, access_count DESC, created_at DESC
            """
        ).fetchall()

        hits: list[dict[str, Any]] = []
        for row in rows:
            keywords = parse_keywords(row["keywords"])
            searchable = f"{row['title']} {row['conclusion']} {' '.join(keywords)}".lower()
            matched = [term for term in terms if term in searchable]
            if not matched:
                continue
            score = len(matched) * 2 + float(row["importance"] or 1.0)
            hits.append(
                {
                    "block_id": row["block_id"],
                    "title": row["title"],
                    "content": row["content"],
                    "conclusion": row["conclusion"],
                    "keywords": keywords,
                    "score": score,
                    "token_count": row["token_count"],
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
                    f"[{idx}] {memory['title']} ({memory['block_id']})",
                    f"Conclusion: {memory.get('conclusion') or ''}",
                    f"Keywords: {', '.join(memory.get('keywords') or [])}",
                    "Content:",
                    memory.get("content") or "",
                ]
            )
        )
    return "\n\n".join(parts)


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


def maybe_checkpoint(user_text: str, assistant_text: str, session_id: str = "proxy") -> None:
    global _turn_counter
    if not user_text and not assistant_text:
        return
    _turn_counter += 1
    _turn_buffer.append({"user": user_text, "assistant": assistant_text})

    combined = "\n\n".join(
        f"[user]\n{turn['user']}\n\n[assistant]\n{turn['assistant']}"
        for turn in _turn_buffer
    )
    if len(combined) < CHECKPOINT_MIN_CHARS and _turn_counter % CHECKPOINT_EVERY_N_TURNS != 0:
        return
    if len(combined) < CHECKPOINT_MIN_CHARS:
        return

    archive_block(
        title=derive_title(user_text),
        content=combined,
        conclusion=derive_conclusion(assistant_text),
        keywords=derive_keywords(combined),
        session_id=session_id,
        source="proxy-checkpoint",
    )
    _turn_buffer.clear()


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
    source: str,
) -> str:
    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
    block_id = f"{datetime.now().strftime('%y%m%d')}-{fingerprint[:6]}"
    now = datetime.now().isoformat()
    with db_conn() as conn:
        existing = conn.execute(
            "SELECT block_id FROM blocks WHERE fingerprint=? LIMIT 1", (fingerprint,)
        ).fetchone()
        if existing:
            return existing["block_id"]
        conn.execute(
            """
            INSERT INTO blocks
            (block_id, title, content, token_count, keywords, conclusion,
             session_id, source, fingerprint, created_at, last_access, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0)
            """,
            (
                block_id,
                title,
                content,
                rough_tokens(content),
                json.dumps(keywords, ensure_ascii=False),
                conclusion,
                session_id,
                source,
                fingerprint,
                now,
                now,
            ),
        )
    return block_id


def upstream_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in {"host", "content-length", "connection"}:
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


async def forward_stream(path: str, payload: dict[str, Any], request: Request) -> StreamingResponse:
    client = httpx.AsyncClient(timeout=None)
    upstream = client.stream("POST", upstream_url(path), headers=upstream_headers(request), json=payload)
    response = await upstream.__aenter__()

    async def iterator():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
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
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request) -> Response:
    payload = await request.json()
    user_text = extract_openai_user_text(payload)
    memories = recall_memory(user_text)
    injected = inject_openai_memory(payload, format_memory(memories))

    if payload.get("stream"):
        return await forward_stream("/v1/chat/completions", injected, request)

    async with httpx.AsyncClient(timeout=None) as client:
        upstream_response = await client.post(
            upstream_url("/v1/chat/completions"),
            headers=upstream_headers(request),
            json=injected,
        )

    try:
        data = upstream_response.json()
    except json.JSONDecodeError:
        return Response(
            status_code=upstream_response.status_code,
            content=upstream_response.content,
            media_type=upstream_response.headers.get("content-type"),
        )

    if upstream_response.is_success:
        maybe_checkpoint(user_text, extract_openai_assistant_text(data), session_id="openai-proxy")
    return JSONResponse(status_code=upstream_response.status_code, content=data)


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Response:
    payload = await request.json()
    user_text = extract_anthropic_user_text(payload)
    memories = recall_memory(user_text)
    injected = inject_anthropic_memory(payload, format_memory(memories))

    if payload.get("stream"):
        return await forward_stream("/v1/messages", injected, request)

    async with httpx.AsyncClient(timeout=None) as client:
        upstream_response = await client.post(
            upstream_url("/v1/messages"),
            headers=upstream_headers(request),
            json=injected,
        )

    try:
        data = upstream_response.json()
    except json.JSONDecodeError:
        return Response(
            status_code=upstream_response.status_code,
            content=upstream_response.content,
            media_type=upstream_response.headers.get("content-type"),
        )

    if upstream_response.is_success:
        maybe_checkpoint(user_text, extract_anthropic_assistant_text(data), session_id="anthropic-proxy")
    return JSONResponse(status_code=upstream_response.status_code, content=data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VCTX proxy.")
    parser.add_argument("--host", default=os.getenv("VCTX_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VCTX_PROXY_PORT", "8787")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("proxy:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
