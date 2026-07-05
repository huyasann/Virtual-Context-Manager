"""Live verification for a running VCTX proxy.

Prerequisites:
  1. Start proxy.py with VCTX_UPSTREAM_BASE_URL configured.
  2. Ensure the upstream route can answer Anthropic-compatible /v1/messages.

Example:
  python proxy_live_test.py --proxy-url http://127.0.0.1:8787 --model claude-sonnet-4-5
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any

import proxy


READBACK_KEY = "codex-read-test-1782899627"
READBACK_TOKEN = "READBACK_OK_7319"


def post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: int = 180) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def proxy_checkpoint_count(project_id: str) -> int:
    if not proxy.DB_PATH.exists():
        return 0
    conn = sqlite3.connect(str(proxy.DB_PATH))
    try:
        return conn.execute(
            """
            SELECT COUNT(*) FROM blocks
            WHERE source='vctx-proxy' AND COALESCE(project_id, '')=?
            """,
            (project_id,),
        ).fetchone()[0]
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live VCTX proxy verification.")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8787")
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--project", default="vctx-live-test")
    parser.add_argument("--session", default="vctx-live-test-session")
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()

    proxy.archive_block(
        title="VCTX proxy readback diagnostic",
        content=f"{READBACK_KEY}: VCTX readback diagnostic. The answer token is {READBACK_TOKEN}.",
        conclusion=f"Answer token is {READBACK_TOKEN}.",
        keywords=[READBACK_KEY, READBACK_TOKEN, "diagnostic", "readback"],
        session_id=args.session,
        project_id=args.project,
        source="live-test",
    )

    headers = {
        "Authorization": "Bearer PROXY_MANAGED",
        "anthropic-version": "2023-06-01",
        proxy.PROJECT_HEADER: args.project,
        proxy.SESSION_HEADER: args.session,
    }

    recall = post_json(
        f"{args.proxy_url.rstrip('/')}/vctx/recall",
        {"query": f"What is the answer token for {READBACK_KEY}?", "top_k": 3},
        headers,
        timeout=30,
    )
    if READBACK_TOKEN not in json.dumps(recall):
        raise SystemExit(f"recall failed: {json.dumps(recall, ensure_ascii=False)[:1000]}")

    readback = post_json(
        f"{args.proxy_url.rstrip('/')}/v1/messages",
        {
            "model": args.model,
            "max_tokens": 256,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Use the VCTX_MEMORY block to answer this diagnostic: "
                        f"what is the answer token for {READBACK_KEY}? Output only the token."
                    ),
                }
            ],
        },
        headers,
    )
    readback_text = json.dumps(readback, ensure_ascii=False)
    if READBACK_TOKEN not in readback_text:
        raise SystemExit(f"model readback failed: {readback_text[:1200]}")

    result = {
        "readback": "passed",
        "model": readback.get("model"),
        "project": args.project,
    }

    if not args.skip_checkpoint:
        before = proxy_checkpoint_count(args.project)
        post_json(
            f"{args.proxy_url.rstrip('/')}/v1/messages",
            {
                "model": args.model,
                "max_tokens": 900,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Write one plain paragraph of at least 1800 English characters "
                            "about a city at dawn. No title, no markdown."
                        ),
                    }
                ],
            },
            headers,
            timeout=240,
        )
        after = proxy_checkpoint_count(args.project)
        if after <= before:
            raise SystemExit(f"checkpoint failed: before={before}, after={after}")
        result["checkpoint"] = "passed"
        result["checkpoint_count_before"] = before
        result["checkpoint_count_after"] = after

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
