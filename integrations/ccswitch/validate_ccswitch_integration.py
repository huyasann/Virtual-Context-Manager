from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODULE = ROOT / "vctx_memory.rs"
PATCH = ROOT / "ccswitch-vctx.patch"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    require(MODULE.exists(), "missing vctx_memory.rs")
    require(PATCH.exists(), "missing ccswitch-vctx.patch")

    module = MODULE.read_text(encoding="utf-8")
    patch = PATCH.read_text(encoding="utf-8")

    for symbol in [
        "pub fn apply_request_memory",
        "pub fn maybe_checkpoint_response",
        "fn recall_memory",
        "fn inject_top_level_system",
        "fn archive_checkpoint",
    ]:
        require(symbol in module, f"missing symbol: {symbol}")

    for table_field in [
        "CREATE TABLE IF NOT EXISTS blocks",
        "block_id TEXT PRIMARY KEY",
        "content TEXT NOT NULL",
        "fingerprint TEXT",
        "source TEXT DEFAULT 'manual'",
    ]:
        require(table_field in module, f"missing VCTX schema field: {table_field}")

    for patch_hook in [
        "pub(crate) mod vctx_memory;",
        "VctxMemoryConfig",
        "get_vctx_memory_config",
        "apply_request_memory(",
        "maybe_checkpoint_response(",
    ]:
        require(patch_hook in patch, f"patch missing hook: {patch_hook}")

    require(
        'matches!(app_type, "claude" | "claude-desktop")' in module,
        "Anthropic/Claude top-level system branch is missing",
    )
    require(
        '"role": "system"' in module,
        "OpenAI/Codex messages system branch is missing",
    )
    require(
        "<VCTX_MEMORY" in module,
        "memory injection marker is missing",
    )
    require(
        "source, is_recalled, recall_from" in module,
        "checkpoint insert does not match VCTX block columns",
    )

    # Catch accidental stdout redirection failures where git diff text is empty
    # or truncated to only one file.
    diff_headers = re.findall(r"^diff --git ", patch, flags=re.MULTILINE)
    require(len(diff_headers) >= 6, "patch looks truncated")

    print("ccswitch VCTX integration package: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
