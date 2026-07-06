"""Live test for same-model prompt completion through VCTX proxy."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from typing import Any

import proxy


def post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: int = 300) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live prompt-completion verification.")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8787")
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--project", default="prompt-completion-live-test")
    parser.add_argument("--session", default="prompt-completion-live-test-session")
    args = parser.parse_args()

    base = args.proxy_url.rstrip("/")
    health = get_json(f"{base}/healthz")
    if not health.get("prompt_completion_enabled"):
        raise SystemExit("VCTX_PROMPT_COMPLETION is not enabled on the running proxy")

    headers = {
        "Authorization": "Bearer PROXY_MANAGED",
        "anthropic-version": "2023-06-01",
        proxy.PROJECT_HEADER: args.project,
        proxy.SESSION_HEADER: args.session,
    }
    post_json(
        f"{base}/v1/messages",
        {
            "model": args.model,
            "max_tokens": 160,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "Continue the coding task in this repository.",
                }
            ],
        },
        headers,
    )

    query = urllib.parse.urlencode({"project": args.project, "limit": 5})
    traces = get_json(f"{base}/vctx/traces?{query}").get("traces", [])
    if not traces:
        raise SystemExit("No traces found for prompt completion test")
    latest = traces[0]
    if not latest.get("prompt_completion_used"):
        raise SystemExit(f"Prompt completion was not used: {json.dumps(latest, ensure_ascii=False)}")
    if latest.get("prompt_completion_risk") not in {"low", "medium"}:
        raise SystemExit(f"Prompt completion risk was not accepted: {json.dumps(latest, ensure_ascii=False)}")
    if latest.get("prompt_completion_chars", 0) <= 0:
        raise SystemExit(f"Prompt completion text was empty: {json.dumps(latest, ensure_ascii=False)}")

    print(
        json.dumps(
            {
                "status": "passed",
                "model": args.model,
                "project": args.project,
                "trace_id": latest.get("trace_id"),
                "prompt_completion_chars": latest.get("prompt_completion_chars"),
                "prompt_completion_risk": latest.get("prompt_completion_risk"),
                "prompt_completion_reason": latest.get("prompt_completion_reason"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
