# CC Switch VCTX Middleware Prototype

This folder contains the portable, VCTX-specific part of the CC Switch memory
middleware experiment. It does not vendor CC Switch.

## Files

- `vctx_memory.rs` - standalone Rust middleware module.
- `ccswitch-vctx.patch` - integration patch against CC Switch proxy internals.
- `adversarial_review.md` - risk review and current usability judgement.
- `validate_ccswitch_integration.py` - structural validation for this extracted
  integration package.

## What The Patch Adds

The patch hooks VCTX into CC Switch's local proxy path:

```text
client -> CC Switch proxy -> VCTX request recall/injection -> provider router
provider response -> VCTX checkpoint -> client
```

Behavior:

- Reads memory blocks from `~/.vctx/memory.db`.
- Recalls relevant blocks from the latest user message.
- Injects bounded `<VCTX_MEMORY>` into:
  - Anthropic/Claude style: top-level `system`
  - OpenAI/Codex style: `messages[role=system]`
- Archives substantial non-streaming responses back into `blocks`.
- No-ops when the VCTX database does not exist.

## Apply To A CC Switch Checkout

From a CC Switch repo root:

```bash
git apply /path/to/Virtual-Context-Manager/integrations/ccswitch/ccswitch-vctx.patch
cargo check
```

If `cargo` is unavailable on the machine, the patch can still be inspected with:

```bash
python /path/to/Virtual-Context-Manager/integrations/ccswitch/validate_ccswitch_integration.py
```

## Runtime Verification With Claude Code Desktop + CC Switch

1. Enable CC Switch local proxy takeover for Claude Code Desktop.
2. Route CC Switch to the target provider/model, for example Mimo.
3. Ensure VCTX has at least one archived block:

   ```bash
   python smoke_test.py
   ```

4. Start Claude Code Desktop and ask for something that matches the archived
   block keywords.
5. Check CC Switch logs for:

   ```text
   [VCTX] injected memory
   ```

6. For non-streaming responses longer than `checkpointMinChars`, check:

   ```text
   [VCTX] archived response checkpoint
   ```

## Current Prototype Limitations

- Streaming response checkpointing is not implemented yet.
- Recall is keyword-only in the Rust middleware.
- No UI switch is wired yet; config is read from CC Switch `settings` key
  `vctx_memory_config`, defaulting to enabled/no-op.
- This is a patch prototype, not a stable CC Switch plugin API.
