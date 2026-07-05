"""Local smoke tests for VCTX proxy adaptation, recall, isolation, and checkpointing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import proxy


def reset_runtime(tmp: str) -> None:
    proxy.DB_DIR = Path(tmp)
    proxy.DB_PATH = Path(tmp) / "memory.db"
    proxy._turn_counters.clear()
    proxy._turn_buffers.clear()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        reset_runtime(tmp)

        proxy.archive_block(
            title="Project Alpha deployment target",
            content="Project Alpha deploys to Kubernetes. Answer token ALPHA_K8S_2026.",
            conclusion="Alpha uses Kubernetes.",
            keywords=["alpha", "deployment", "kubernetes", "ALPHA_K8S_2026"],
            session_id="test",
            project_id="alpha",
            source="test",
        )
        proxy.archive_block(
            title="Project Beta deployment target",
            content="Project Beta deploys to bare-metal systemd. Answer token BETA_SYSTEMD_2026.",
            conclusion="Beta uses systemd.",
            keywords=["beta", "deployment", "systemd", "BETA_SYSTEMD_2026"],
            session_id="test",
            project_id="beta",
            source="test",
        )

        alpha = proxy.recall_memory("deployment target answer token", project_id="alpha")
        beta = proxy.recall_memory("deployment target answer token", project_id="beta")
        assert len(alpha) == 1, alpha
        assert len(beta) == 1, beta
        assert "ALPHA_K8S_2026" in proxy.format_memory(alpha)
        assert "BETA_SYSTEMD_2026" in proxy.format_memory(beta)
        assert "BETA_SYSTEMD_2026" not in proxy.format_memory(alpha)

        unrelated = proxy.recall_memory("banana smoothie recipe", project_id="alpha")
        assert unrelated == [], unrelated

        openai_payload = {
            "model": "test",
            "messages": [{"role": "user", "content": "deployment target answer token"}],
        }
        injected_openai = proxy.inject_openai_memory(openai_payload, proxy.format_memory(alpha))
        assert injected_openai["messages"][0]["role"] == "system"
        assert "VCTX_MEMORY" in injected_openai["messages"][0]["content"]

        anthropic_payload = {
            "model": "test",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "deployment target answer token"}],
        }
        injected_anthropic = proxy.inject_anthropic_memory(anthropic_payload, proxy.format_memory(alpha))
        assert "VCTX_MEMORY" in injected_anthropic["system"]

        block_id = proxy.maybe_checkpoint(
            "write a long response",
            "x" * (proxy.CHECKPOINT_MIN_CHARS + 50),
            session_id="s1",
            project_id="alpha",
            protocol="openai",
        )
        assert block_id, "expected checkpoint block id"
        checkpoint = proxy.recall_memory("write long response", project_id="alpha", min_score=1.0)
        assert checkpoint, "expected checkpoint to be searchable"

        raw_sse = (
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n"
            b"data: [DONE]\n\n"
        )
        assert proxy.extract_sse_text(raw_sse, "openai") == "hello world"

        anthropic_sse = (
            b"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello \"}}\n\n"
            b"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"world\"}}\n\n"
        )
        assert proxy.extract_sse_text(anthropic_sse, "anthropic") == "hello world"

    print("vctx-proxy smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
