"""
vctx-mcp — 虚拟上下文 MCP Server
让 Claude Code 可以直接读写虚拟上下文
"""
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# ────────────────────────────────────────────────────────
# 存储
# ────────────────────────────────────────────────────────

DB_DIR = Path.home() / ".vctx"
DB_PATH = DB_DIR / "memory.db"


def get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blocks (
            block_id    TEXT PRIMARY KEY,
            title       TEXT,
            content     TEXT NOT NULL,
            token_count INTEGER,
            keywords    TEXT,
            conclusion  TEXT,
            created_at  TEXT,
            last_access TEXT,
            importance  REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS raw_messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            role        TEXT,
            content     TEXT,
            archived_at TEXT,
            block_id    TEXT
        );
    """)
    conn.commit()
    return conn


# ────────────────────────────────────────────────────────
# MCP Server
# ────────────────────────────────────────────────────────

mcp = FastMCP(
    "vctx",
    instructions="虚拟上下文管理器 — 让 AI 拥有无限长期记忆。当对话即将超出上下文窗口时，使用 vctx_archive 归档。需要回忆过去的内容时，先用 vctx_list 查看目录，再用 vctx_read 读取。",
)


@mcp.tool()
def vctx_archive(
    title: str,
    content: str,
    conclusion: str,
    keywords: list[str],
) -> str:
    """
    将一段对话或知识归档到虚拟上下文。

    什么时候调用：
    - 一段重要对话即将超出上下文窗口时
    - 用户要求保存当前讨论内容时
    - 你觉得当前对话包含值得长期记住的信息时

    参数：
        title: 主题名，10字以内，如 "ROS2架构设计"
        content: 要归档的完整对话原文（不要压缩，完整保存）
        conclusion: 一句话关键结论，30字以内
        keywords: 3-5个检索关键词
    """
    conn = get_conn()

    # 生成 block_id（基于内容哈希，防重复）
    fp = hashlib.sha256(content.encode()).hexdigest()[:12]
    block_id = f"{datetime.now().strftime('%y%m%d')}-{fp[:6]}"

    # 检查是否已存在
    existing = conn.execute(
        "SELECT block_id FROM blocks WHERE block_id = ?", (block_id,)
    ).fetchone()
    if existing:
        return f"该内容已归档，block_id: {block_id}"

    now = datetime.now().isoformat()
    conn.execute("""
        INSERT INTO blocks
        (block_id, title, content, token_count, keywords,
         conclusion, created_at, last_access, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0)
    """, (
        block_id, title, content, len(content) // 2,  # 粗估 token
        json.dumps(keywords, ensure_ascii=False),
        conclusion, now, now,
    ))
    conn.commit()

    return json.dumps({
        "status": "archived",
        "block_id": block_id,
        "title": title,
        "message": f"已归档: [{block_id}] {title}"
    }, ensure_ascii=False)


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
        "access_count": row["access_count"] + 1,
    }, ensure_ascii=False)


@mcp.tool()
def vctx_search(query: str) -> str:
    """
    在虚拟上下文中搜索相关信息。

    什么时候调用：
    - 用户提到某个话题，你不确定在哪个块里
    - 你想找所有与某个关键词相关的历史

    参数：
        query: 搜索关键词或短语，如 "数据库选型"
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT block_id, title, conclusion, keywords, access_count "
        "FROM blocks ORDER BY importance DESC, access_count DESC"
    ).fetchall()

    hits = []
    query_words = query.lower().split()

    for r in rows:
        kw = json.loads(r["keywords"]) if r["keywords"] else []
        searchable = f"{r['title']} {r['conclusion']} {' '.join(kw)}".lower()
        matched = [w for w in query_words if w in searchable]
        if matched:
            hits.append({
                "block_id": r["block_id"],
                "title": r["title"],
                "conclusion": r["conclusion"],
                "match_score": len(matched),
            })

    hits.sort(key=lambda x: -x["match_score"])

    if not hits:
        return json.dumps({"message": "未找到相关内容", "results": []})

    return json.dumps({
        "query": query,
        "count": len(hits[:5]),
        "results": hits[:5],
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
        "FROM blocks ORDER BY created_at DESC"
    ).fetchall()

    if not rows:
        return "虚拟上下文为空，尚无归档内容。"

    blocks = []
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

    return json.dumps({
        "total_blocks": len(blocks),
        "blocks": blocks,
    }, ensure_ascii=False, indent=2)


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

        # 衰减公式: importance *= 0.97^days_since_access
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


# ────────────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
