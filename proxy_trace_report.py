"""Read-only JSON report for VCTX proxy trace records."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


DEFAULT_DB = Path.home() / ".vctx" / "memory.db"

TRACE_COLUMNS = [
    "trace_id",
    "started_at",
    "finished_at",
    "duration_ms",
    "protocol",
    "path",
    "stream",
    "project_id",
    "user_id",
    "session_id",
    "query_preview",
    "request_chars",
    "message_count",
    "compact_candidate",
    "compact_reason",
    "prompt_completion_used",
    "prompt_completion_chars",
    "prompt_completion_risk",
    "prompt_completion_reason",
    "recalled_block_ids",
    "recalled_scores",
    "injected",
    "checkpoint_block_id",
    "checkpoint_status",
    "upstream_status",
    "error",
]

JSON_COLUMNS = {"compact_reason", "recalled_block_ids", "recalled_scores"}
BOOL_COLUMNS = {"stream", "compact_candidate", "prompt_completion_used", "injected"}


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report VCTX proxy traces as JSON.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB path")
    parser.add_argument("--limit", type=int, default=20, help="Maximum traces to return")
    parser.add_argument("--project", help="Filter by project_id")
    parser.add_argument("--compact-only", action="store_true", help="Only show compact candidates")
    return parser.parse_args()


def readonly_connect(db_path: Path) -> sqlite3.Connection:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def decode_value(column: str, value: Any) -> Any:
    if value is None:
        return None
    if column in JSON_COLUMNS and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if column in BOOL_COLUMNS and isinstance(value, int):
        return bool(value)
    return value


def fetch_traces(
    conn: sqlite3.Connection,
    limit: int,
    project: str | None,
    compact_only: bool,
) -> list[dict[str, Any]]:
    columns = table_columns(conn, "proxy_trace")
    if not columns:
        return []

    selected = [column for column in TRACE_COLUMNS if column in columns]
    if not selected:
        return []

    select_sql = ", ".join(f'"{column}"' for column in selected)
    sql = f'SELECT {select_sql} FROM "proxy_trace"'
    params: list[Any] = []

    if project is not None:
        if "project_id" not in columns:
            return []
        sql += ' WHERE "project_id" = ?'
        params.append(project)
    if compact_only:
        if "compact_candidate" not in columns:
            return []
        sql += " AND " if " WHERE " in sql else " WHERE "
        sql += '"compact_candidate" = 1'

    if "started_at" in columns:
        sql += ' ORDER BY "started_at" DESC'
    elif "finished_at" in columns:
        sql += ' ORDER BY "finished_at" DESC'

    sql += " LIMIT ?"
    params.append(max(0, limit))

    rows = conn.execute(sql, params).fetchall()
    return [
        {column: decode_value(column, row[column]) for column in TRACE_COLUMNS if column in selected}
        for row in rows
    ]


def main() -> int:
    args = parse_args()
    db_path = Path(args.db).expanduser()

    if not db_path.exists():
        print(json_dumps({"traces": [], "message": "proxy_trace table not found"}))
        return 0

    try:
        with readonly_connect(db_path) as conn:
            if not table_columns(conn, "proxy_trace"):
                print(json_dumps({"traces": [], "message": "proxy_trace table not found"}))
                return 0
            print(json_dumps({"traces": fetch_traces(conn, args.limit, args.project, args.compact_only)}))
            return 0
    except sqlite3.Error as exc:
        print(json_dumps({"traces": [], "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
