"""Deep live test for VCTX proxy routed to Mimo through CC Switch.

This script expects:
  - VCTX proxy running, usually on http://127.0.0.1:8787
  - VCTX proxy upstream configured to CC Switch, usually http://127.0.0.1:15721
  - CC Switch route returning Mimo-compatible Anthropic responses

It verifies recall, project isolation, non-streaming checkpoint, and streaming
checkpoint on real model traffic.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import proxy


ALPHA_KEY = "mimo-deep-alpha-readback-260705"
ALPHA_TOKEN = "ALPHA_MIMO_VCTX_OK_260705"
BETA_KEY = "mimo-deep-beta-readback-260705"
BETA_TOKEN = "BETA_MIMO_VCTX_OK_260705"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


def post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: int = 240) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    return f"HTTP {exc.code}: {body[:1000]}"


def get_json(url: str, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_project_traces(base_url: str, project: str, limit: int = 50) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"project": project, "limit": limit})
    data = get_json(f"{base_url.rstrip('/')}/vctx/traces?{query}")
    return data.get("traces", [])


def post_stream(url: str, body: dict[str, Any], headers: dict[str, str], timeout: int = 240) -> tuple[str, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    raw = bytearray()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            raw.extend(chunk)
    raw_text = raw.decode("utf-8", errors="ignore")
    return raw_text, proxy.extract_sse_text(bytes(raw), "anthropic")


def checkpoint_count(project_id: str) -> int:
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


def seed_memory(project: str, session: str) -> None:
    proxy.archive_block(
        title="Mimo deep test alpha memory",
        content=f"{ALPHA_KEY}: The alpha answer token is {ALPHA_TOKEN}.",
        conclusion=f"Alpha answer token is {ALPHA_TOKEN}.",
        keywords=[ALPHA_KEY, ALPHA_TOKEN, "alpha", "mimo", "deep-test"],
        session_id=session,
        project_id=project,
        source="mimo-deep-test",
    )
    proxy.archive_block(
        title="Mimo deep test beta memory",
        content=f"{BETA_KEY}: The beta answer token is {BETA_TOKEN}.",
        conclusion=f"Beta answer token is {BETA_TOKEN}.",
        keywords=[BETA_KEY, BETA_TOKEN, "beta", "mimo", "deep-test"],
        session_id=session,
        project_id=f"{project}-other",
        source="mimo-deep-test",
    )


def anthropic_body(model: str, content: str, max_tokens: int = 256, stream: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "stream": stream,
        "messages": [{"role": "user", "content": content}],
    }


def run(args: argparse.Namespace) -> list[Check]:
    base = args.proxy_url.rstrip("/")
    project = args.project
    session = args.session
    headers = {
        "Authorization": "Bearer PROXY_MANAGED",
        "anthropic-version": "2023-06-01",
        proxy.PROJECT_HEADER: project,
        proxy.SESSION_HEADER: session,
    }
    beta_headers = {**headers, proxy.PROJECT_HEADER: f"{project}-other"}
    checks: list[Check] = []

    health = get_json(f"{base}/healthz")
    if not health.get("ok") or not health.get("upstream_configured"):
        raise RuntimeError(f"proxy health failed: {health}")
    checks.append(Check("proxy_health", "passed", json.dumps(health, ensure_ascii=False)))

    seed_memory(project, session)
    checks.append(Check("seed_memory", "passed", f"project={project}"))

    alpha_recall = post_json(
        f"{base}/vctx/recall",
        {"query": f"What is the alpha answer token for {ALPHA_KEY}?", "top_k": 3},
        headers,
    )
    alpha_recall_text = json.dumps(alpha_recall, ensure_ascii=False)
    if ALPHA_TOKEN not in alpha_recall_text or BETA_TOKEN in alpha_recall_text:
        raise RuntimeError(f"alpha recall isolation failed: {alpha_recall_text[:1200]}")
    checks.append(Check("alpha_recall_isolated", "passed", f"count={alpha_recall.get('count')}"))

    beta_recall = post_json(
        f"{base}/vctx/recall",
        {"query": f"What is the beta answer token for {BETA_KEY}?", "top_k": 3},
        beta_headers,
    )
    beta_recall_text = json.dumps(beta_recall, ensure_ascii=False)
    if BETA_TOKEN not in beta_recall_text or ALPHA_TOKEN in beta_recall_text:
        raise RuntimeError(f"beta recall isolation failed: {beta_recall_text[:1200]}")
    checks.append(Check("beta_recall_isolated", "passed", f"count={beta_recall.get('count')}"))

    unrelated = post_json(
        f"{base}/vctx/recall",
        {"query": "banana smoothie airport weather", "top_k": 3},
        headers,
    )
    if unrelated.get("count") != 0:
        raise RuntimeError(f"unrelated recall should be empty: {unrelated}")
    checks.append(Check("unrelated_recall_suppressed", "passed"))

    readback = post_json(
        f"{base}/v1/messages",
        anthropic_body(
            args.model,
            (
                f"Use VCTX_MEMORY to answer this diagnostic. What is the alpha answer token "
                f"for {ALPHA_KEY}? Output only the token."
            ),
            max_tokens=256,
            stream=False,
        ),
        headers,
    )
    readback_text = json.dumps(readback, ensure_ascii=False)
    if ALPHA_TOKEN not in readback_text:
        raise RuntimeError(f"non-stream readback failed: {readback_text[:1600]}")
    checks.append(Check("nonstream_readback", "passed", f"model={readback.get('model')}"))

    stream_raw, stream_text = post_stream(
        f"{base}/v1/messages",
        anthropic_body(
            args.model,
            (
                f"Use VCTX_MEMORY to answer this streaming diagnostic. What is the alpha "
                f"answer token for {ALPHA_KEY}? Output only the token."
            ),
            max_tokens=256,
            stream=True,
        ),
        headers,
    )
    if ALPHA_TOKEN not in stream_raw and ALPHA_TOKEN not in stream_text:
        raise RuntimeError(f"stream readback failed: text={stream_text!r}, raw={stream_raw[:1200]!r}")
    checks.append(Check("stream_readback", "passed", f"text_len={len(stream_text)}"))

    before_nonstream = checkpoint_count(project)
    post_json(
        f"{base}/v1/messages",
        anthropic_body(
            args.model,
            "Write one plain paragraph of at least 1900 English characters about a city at dawn. No markdown.",
            max_tokens=1000,
            stream=False,
        ),
        headers,
        timeout=300,
    )
    after_nonstream = checkpoint_count(project)
    if after_nonstream <= before_nonstream:
        raise RuntimeError(f"non-stream checkpoint failed: before={before_nonstream}, after={after_nonstream}")
    checks.append(Check("nonstream_checkpoint", "passed", f"{before_nonstream}->{after_nonstream}"))

    before_stream = checkpoint_count(project)
    _, long_stream_text = post_stream(
        f"{base}/v1/messages",
        anthropic_body(
            args.model,
            "Write one plain paragraph of at least 1900 English characters about a railway station at night. No markdown.",
            max_tokens=1000,
            stream=True,
        ),
        headers,
        timeout=300,
    )
    time.sleep(0.5)
    after_stream = checkpoint_count(project)
    if after_stream <= before_stream:
        raise RuntimeError(
            f"stream checkpoint failed: before={before_stream}, after={after_stream}, stream_len={len(long_stream_text)}"
        )
    checks.append(Check("stream_checkpoint", "passed", f"{before_stream}->{after_stream}, text_len={len(long_stream_text)}"))

    traces = get_project_traces(base, project)
    if not traces:
        raise RuntimeError("expected proxy traces for project")
    if not any(
        trace.get("protocol") == "anthropic"
        and trace.get("path") == "/v1/messages"
        and not trace.get("stream")
        and trace.get("injected")
        and trace.get("upstream_status") == 200
        for trace in traces
    ):
        raise RuntimeError(f"missing non-stream injected trace: {json.dumps(traces[:5], ensure_ascii=False)}")
    if not any(
        trace.get("protocol") == "anthropic"
        and trace.get("stream")
        and trace.get("checkpoint_status") == "saved"
        and trace.get("checkpoint_block_id")
        for trace in traces
    ):
        raise RuntimeError(f"missing stream checkpoint trace: {json.dumps(traces[:5], ensure_ascii=False)}")
    checks.append(Check("proxy_trace_records", "passed", f"count={len(traces)}"))

    if args.try_openai:
        try:
            openai = post_json(
                f"{base}/v1/chat/completions",
                {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Use VCTX_MEMORY to answer: alpha token for {ALPHA_KEY}. Output only token.",
                        }
                    ],
                    "max_tokens": 256,
                    "stream": False,
                },
                headers,
            )
            openai_text = json.dumps(openai, ensure_ascii=False)
            status = "passed" if ALPHA_TOKEN in openai_text else "unsupported_or_failed"
            checks.append(Check("openai_compat_readback", status, openai_text[:600]))
        except urllib.error.HTTPError as exc:
            checks.append(Check("openai_compat_readback", "unsupported_or_failed", http_error_detail(exc)))
        except (urllib.error.URLError, TimeoutError) as exc:
            checks.append(Check("openai_compat_readback", "unsupported_or_failed", str(exc)))

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deep live Mimo test through VCTX proxy.")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8787")
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--project", default="mimo-deep-test")
    parser.add_argument("--session", default="mimo-deep-test-session")
    parser.add_argument("--try-openai", action="store_true")
    args = parser.parse_args()

    checks = run(args)
    print(
        json.dumps(
            {
                "status": "passed",
                "model": args.model,
                "project": args.project,
                "checks": [check.__dict__ for check in checks],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
