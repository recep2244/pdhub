---
phase: 02-mutagenesis-workflow-integrity
plan: "03"
status: complete
commit: 2b44839
requirements_closed:
  - MUT-05
---

# Plan 02-03 Summary: Backend Overrides Wired + Timing Log Model Name

**Completed:** 2026-02-21

## What Was Done

Three surgical changes to close MUT-05 — expert panel backend selections were
being collected in session state but silently ignored during pipeline execution.

### Task 1: meeting.py — Model name in timing log

Changed one line in `_call_llm()` (line 121):

```python
# Before
tok_info = f", {out_tok} tok, {out_tok / elapsed:.0f} tok/s"
# After
tok_info = f", {out_tok} tok, {out_tok / elapsed:.0f} tok/s ({model})"
```

The `model` variable (`agent.resolved_model`) is already defined at line 90.
Result: timing lines now read `[LLMBaselineReviewAgent] 3.2s, 120 tok, 37 tok/s (qwen2.5:14b)`.

### Task 2: 10_mutation_scanner.py — Override wiring

**`_run_phase1()`** — added at function entry:
```python
from protein_design_hub.web.agent_helpers import _temporary_llm_override
provider, model = _expert_review_overrides()
```
And wrapped `orchestrator.run(...)` with `with _temporary_llm_override(provider, model):`.

**`_run_phase2()`** — same pattern: `_expert_review_overrides()` at entry,
`_temporary_llm_override` wrapping `orchestrator.run_with_context(ctx)`.

**UI caption** — added before the `col1, col2 = st.columns(2)` block (Approve & Continue):
```python
try:
    from protein_design_hub.core.config import get_settings
    cfg = get_settings().llm
    _ov_provider, _ov_model = _expert_review_overrides()
    effective_provider = _ov_provider if _ov_provider and _ov_provider != "current" else cfg.provider
    effective_model = _ov_model if _ov_model else cfg.model
    st.caption(f"Using: `{effective_model}` @ `{effective_provider}`")
except Exception:
    pass  # Caption is informational; never block the UI
```

## Verification

- `grep -n "_temporary_llm_override" 10_mutation_scanner.py` → lines 1672, 1902 (one in each phase runner)
- `grep -n "tok/s.*{model}" meeting.py` → line 121 confirmed
- `grep -n "Using:.*effective_model" 10_mutation_scanner.py` → line 1821 confirmed
- Syntax checks: both files parse cleanly
- `pytest tests/ -x -q` → **46 passed**

## Anti-patterns Avoided

- Did NOT permanently mutate `get_settings().llm.*` — used `_temporary_llm_override` which handles save/restore/reset_llm_client
- Did NOT call `reset_llm_client()` directly
- Did NOT change agent constructors — global settings mutation via `_temporary_llm_override` propagates automatically

## Commit

`2b44839` — feat(02-03): wire expert backend overrides into Phase 1/2 and add model to timing log
