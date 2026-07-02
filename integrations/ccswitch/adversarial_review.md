# Adversarial Review: CC Switch VCTX Middleware

## Verdict

Prototype viability: plausible, but not proven production-ready until the patch
passes `cargo check` inside a Rust-enabled CC Switch checkout and one real
Claude Code Desktop turn is observed through CC Switch logs.

The design logic is sound: VCTX belongs in the proxy hot path, not in MCP/Skill
instructions, because only the proxy reliably sees every request and response.

## Attack Surface And Failure Modes

| Risk | Severity | Evidence | Mitigation |
|---|---:|---|---|
| Patch fails to compile against future CC Switch versions | High | It edits internal structs: `RequestContext`, `RequestForwarder`, `types` | Treat as integration patch, pin tested CC Switch commit/version |
| Anthropic protocol broken by wrong system injection | High | Anthropic expects top-level `system`, not `messages[role=system]` | Middleware branches on `app_type` and uses top-level `system` for `claude` / `claude-desktop` |
| Context pollution from irrelevant recall | Medium | Current Rust recall is keyword-only | Keep `recallTopK=3`, `maxMemoryChars=2400`; later add embedding or score threshold tuning |
| Sensitive memory leakage across users/projects | High | Default DB path is global `~/.vctx/memory.db` | Add `project_id/user_id/session_id` filters before multi-user use |
| Streaming responses not checkpointed | Medium | Current hook archives only non-streaming bodies | Add SSE collector after request injection path is verified |
| Duplicate checkpoints | Low | Fingerprint dedup is implemented | Existing `blocks.fingerprint` index is reused |
| Disk lock / SQLite contention | Medium | CC Switch and MCP may write same DB | SQLite WAL helps; still needs timeout/backoff in Rust module if load increases |
| Model sees memory and over-trusts it | Medium | Injected block is system-level context | Mark as retrieved memory and keep excerpts bounded; do not inject secrets |
| No UI toggle yet | Medium | Config exists only as settings JSON | Add CC Switch settings UI after backend compile is verified |

## Can It Work With CC Switch Routing To Mimo?

Likely yes, if CC Switch keeps the request in Anthropic-compatible or
OpenAI-compatible JSON before forwarding to Mimo. The middleware runs before
provider routing, so the selected downstream model is mostly irrelevant.

Cases expected to work:

- Claude Code Desktop -> CC Switch `/v1/messages` -> Mimo Anthropic-compatible
- Codex/OpenAI client -> CC Switch chat completions path -> Mimo/OpenAI-like

Cases needing extra testing:

- Provider adapters that transform request shape after VCTX injection.
- Gemini-native routes.
- Streaming-only clients where checkpointing is desired.

## Minimum Acceptance Criteria

- `git apply integrations/ccswitch/ccswitch-vctx.patch` succeeds in a clean CC
  Switch checkout.
- `cargo check` succeeds.
- A request matching an archived VCTX keyword emits `[VCTX] injected memory`.
- Claude Code Desktop still receives a valid model response through CC Switch.
- A long non-streaming response creates one new `source='cc-switch-proxy'`
  block in `~/.vctx/memory.db`.

## Current Confidence

- Architecture: medium-high.
- Patch correctness without Rust compile: medium-low.
- End-to-end usability through Claude Code Desktop + CC Switch + Mimo: medium,
  pending live log verification.

## Local Validation Run

Date: 2026-07-02

Observed checks:

- `validate_ccswitch_integration.py`: passed.
- `git apply --check integrations/ccswitch/ccswitch-vctx.patch` against a clean
  local CC Switch clone: passed.
- Existing `proxy_smoke_test.py`: passed in the `pytorch_env` conda environment.
- CC Switch local proxy health:
  - `GET http://127.0.0.1:15721/health`: healthy.
  - `GET http://127.0.0.1:15721/status`: running.
- Direct Anthropic-compatible request through CC Switch:
  - Endpoint: `POST http://127.0.0.1:15721/v1/messages`
  - Result: response returned with `model="mimo-v2.5-pro"`.

Checks not completed:

- `cargo check`: local shell did not have `cargo`.
- Existing `smoke_test.py`: timed out locally after 124 seconds in this run.
- Live Claude Code Desktop UI observation: not automated from this repository.

Interpretation:

The current machine can route CC Switch traffic to Mimo, and the extracted VCTX
patch applies cleanly to CC Switch source. The remaining blocker for "can be
used normally" is Rust compilation plus installing/running the patched CC Switch
binary.
