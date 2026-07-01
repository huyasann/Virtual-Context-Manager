"""Local smoke tests for VCTX proxy request adaptation and memory injection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import proxy


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        proxy.DB_DIR = Path(tmp)
        proxy.DB_PATH = Path(tmp) / "memory.db"
        proxy.archive_block(
            title="Claude Desktop VCTX MCP config",
            content=(
                "Claude Desktop 3p reads MCP config from "
                "C:/Users/22240/AppData/Local/Claude-3p/claude_desktop_config.json. "
                "The VCTX server path is C:/Users/22240/projects/vctx-mcp/server.py."
            ),
            conclusion="Use Local Claude-3p config, not Roaming Claude.",
            keywords=["Claude Desktop", "VCTX", "MCP", "Claude-3p"],
            session_id="test",
            source="test",
        )

        memories = proxy.recall_memory("继续修 Claude Desktop 的 VCTX MCP 问题")
        assert memories, "expected at least one recalled block"
        memory_text = proxy.format_memory(memories)
        assert "Claude-3p" in memory_text

        openai_payload = {
            "model": "test",
            "messages": [{"role": "user", "content": "继续修 Claude Desktop 的 VCTX MCP 问题"}],
        }
        injected_openai = proxy.inject_openai_memory(openai_payload, memory_text)
        assert injected_openai["messages"][0]["role"] == "system"
        assert "VCTX_MEMORY" in injected_openai["messages"][0]["content"]

        anthropic_payload = {
            "model": "test",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "继续修 Claude Desktop 的 VCTX MCP 问题"}],
        }
        injected_anthropic = proxy.inject_anthropic_memory(anthropic_payload, memory_text)
        assert "VCTX_MEMORY" in injected_anthropic["system"]

    print("vctx-proxy smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
