# Phase 2: Mutagenesis Workflow Integrity - Research

**Researched:** 2026-02-21
**Domain:** Streamlit session state, Python file I/O, multi-agent orchestrator kwargs
**Confidence:** HIGH (all findings from direct codebase inspection)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Approval gate UX:**
- "Approve & Continue" button lives at the bottom of the suggestions table (user scrolls through all suggestions first)
- If clicked with no mutations selected: show inline error message ("Select at least one mutation to approve"), block — do not proceed
- If user tries to start Phase 2 without going through the approval step: show a confirmation dialog ("No mutations approved — are you sure?"), not a hard block; user can override if they want to use fallback path

**Disk persistence format:**
- Format: JSON (human-readable, easy to inspect/debug, no extra dependencies)
- Save timing: auto-save when Phase 1 pipeline completes (not on approve click — the raw suggestions + low-confidence positions are saved immediately after Phase 1 agents finish)
- Load behavior: auto-load silently when user returns to a job with Phase 1 results on disk; show a subtle indicator "Loaded from previous session" but no prompt

**Backend override verification:**
- Observable evidence: BOTH the per-call timing log AND a UI preview before Phase 2 starts
  - Timing log should include the model name: `[LLMBaselineReviewAgent] 3.2s, 120 tok, 37 tok/s (qwen2.5:14b)`
  - UI shows "Using: [model] @ [provider]" confirmation before Phase 2 button is active
- Override scope: ALL Phase 2 mutagenesis agents use the selected backend (not just review agents)

### Claude's Discretion
- Exact JSON schema for persisted Phase 1 state (what fields, how nested)
- File naming convention for the persisted state file within the job directory
- Exact wording of the confirmation dialog for the no-approval case
- How the timing log model name is appended (prefix, suffix, or bracketed field)

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MUT-01 | Phase 2 is blocked if no explicit user approval from Phase 1 (`approved_mutations` empty = no execution) | `MutationExecutionAgent.run()` at line 83-103 reads `context.extra["approved_mutations"]`; falls back silently to saturation — must be blocked at the UI level before `_run_phase2()` is called |
| MUT-02 | "Approve & Continue" button in UI enforces approval before Phase 2 runs | Current button at line 1680 is already named "Approve & Execute Mutations" and sets `ctx.extra["approved_mutations"]` before calling `_run_phase2(ctx)` — needs name change + inline error on zero selection |
| MUT-03 | Phase 1 results (approved mutations, suggestions, low-confidence positions) persisted to job directory on disk when Phase 1 completes | No persistence exists today; `mutagenesis_context` lives only in `st.session_state`; save point is immediately after `_run_phase1()` succeeds |
| MUT-04 | UI can load previous Phase 1 results when returning to a job (survives browser close) | No load mechanism exists; needs file detection on page load + silent restore into `st.session_state.mutagenesis_context` |
| MUT-05 | Per-expert backend overrides flow verified end-to-end (session state → orchestrator kwargs → agent constructor) | Session state keys are correctly set by UI but NOT passed into `AgentOrchestrator(mode="mutagenesis_post")` — the `_run_phase1()` and `_run_phase2()` calls ignore them entirely |
</phase_requirements>

---

## Summary

Phase 2 fixes three correctness gaps: the approval gate, disk persistence, and backend override propagation. All three problems have been directly verified in the codebase. The research provides exact file locations, line numbers, variable names, and data flow for each fix.

The approval gate is the most critical bug. `MutationExecutionAgent.run()` silently falls back to saturation mutagenesis at low-confidence positions if `context.extra["approved_mutations"]` is empty (lines 83-103 of `mutagenesis_agents.py`). The current UI button ("Approve & Execute Mutations", line 1680) already sets `approved_mutations` before calling `_run_phase2()` — but there is no guard against the user clicking "Run Agent Analysis" again and starting Phase 1 fresh, then jumping to Phase 2 via a different path. The real gap is that `_run_phase2(ctx)` never validates that `ctx.extra["approved_mutations"]` is non-empty before launching the orchestrator.

Disk persistence is completely absent. `st.session_state.mutagenesis_context` holds the entire `WorkflowContext` object in memory only. There is no code path that saves the Phase 1 results to disk after `_run_phase1()` completes. The job directory structure is already established (`_ensure_mutagenesis_job_dir()` at line 616), so the save path is clear: `{job_dir}/phase1_state.json`.

Backend overrides are collected by the UI into three session state keys (`mut_review_provider`, `mut_review_model`, `mut_review_custom_provider`) and resolved by `_expert_review_overrides()` at line 418. This function is called correctly for the manual expert panels in the page, but `_run_phase1()` (line 1552) and `_run_phase2()` (line 1734) both construct `AgentOrchestrator` without passing any provider/model kwargs. The LLM agents (`LLMBaselineReviewAgent`, `LLMMutationSuggestionAgent`, `LLMMutationResultsAgent`) accept `**kwargs` in their constructors but do not currently accept a `provider_override` or `model_override` parameter — they rely entirely on the global `get_settings().llm` configuration.

**Primary recommendation:** Implement three focused, surgical changes: (1) add empty-check + confirmation dialog in `_run_phase2()`, (2) add `_save_phase1_state()` and `_load_phase1_state()` functions using the existing `_ensure_mutagenesis_job_dir()` path, (3) add `provider_override`/`model_override` parameters to all mutagenesis LLM agents and thread them through `_temporary_llm_override` in `_run_phase1()` and `_run_phase2()`.

---

## Standard Stack

This phase uses only libraries already present in the codebase. No new dependencies.

### Core

| Component | Location | Purpose | Why Standard |
|-----------|----------|---------|--------------|
| `json` stdlib | Python stdlib | Serialize/deserialize Phase 1 state | Already used in `mutagenesis_agents.py` (line 10); no new dependency |
| `pathlib.Path` | Python stdlib | File path operations | Already the codebase standard for all path operations |
| `streamlit` session state | `st.session_state` | Phase completion tracking, context storage | Existing pattern throughout `10_mutation_scanner.py` |
| `_temporary_llm_override` | `web/agent_helpers.py` line 584 | Thread-safe provider/model override for a single operation | Already implements the correct pattern: save → override → restore |
| `AgentOrchestrator(**kwargs)` | `agents/orchestrator.py` | Pass provider/model to LLM agents | `**kwargs` already forwarded to all agent constructors |

### Supporting

| Component | Location | Purpose | When to Use |
|-----------|----------|---------|-------------|
| `st.dialog` decorator | Streamlit | Confirmation dialog for Phase 2 without prior approval | For the confirmation dialog (no-approval warning); replaces ad-hoc modal patterns |
| `_ensure_mutagenesis_job_dir()` | `10_mutation_scanner.py` line 616 | Establish consistent job directory | Call this before writing phase1_state.json |
| `reset_llm_client()` | `agents/meeting.py` line 72 | Invalidate cached LLM client after provider change | Called inside `_temporary_llm_override`; no direct call needed |

---

## Architecture Patterns

### Current Data Flow (Phase 1 → Phase 2)

```
_run_phase1(sequence)
  └── AgentOrchestrator(mode="mutagenesis_pre")  # NO provider/model kwargs
      └── runs 7 agents, fills context.extra:
          - "mutation_suggestions": {...}
          - "baseline_low_confidence_positions": [int, ...]
          - "baseline_review": str (summary)
          - "mutation_suggestion_raw": str
  └── st.session_state.mutagenesis_context = result.context  # in-memory only

_render_approval_step(phase1_ctx)
  └── st.data_editor(...)  # editable table
  └── st.button("Approve & Execute Mutations")
      └── approved = _parse_approved_mutations(included, sequence)
      └── ctx.extra["approved_mutations"] = approved
      └── st.session_state.mutagenesis_context = ctx
      └── _run_phase2(ctx)

_run_phase2(ctx)
  └── AgentOrchestrator(mode="mutagenesis_post")  # NO provider/model kwargs
      └── runs 4 agents (MutationExecutionAgent reads approved_mutations)
  └── st.session_state.mutagenesis_phase2_context = result.context
```

### Pattern 1: Approval Gate (MUT-01, MUT-02)

**What:** Guard `_run_phase2()` with an explicit check that `ctx.extra["approved_mutations"]` is non-empty. The current button already sets this field — the missing piece is a check inside `_run_phase2()` itself (which is the enforcement point the backend agent also relies on).

**The actual bug:** The button at line 1679 already has `approve_disabled = len(included) == 0` which disables the button when nothing is selected. However, the button label is wrong ("Approve & Execute Mutations" instead of "Approve & Continue"), and there is no confirmation dialog for the no-approval-bypass scenario described in CONTEXT.md.

**The no-approval bypass scenario:** If a user runs Phase 1, then somehow calls `_run_phase2()` without going through the approval step (e.g., state mutation bug, page refresh with stale context), `approved_mutations` would be empty and the agent would silently use the saturation fallback. The fix is:

```python
# In _run_phase2(ctx):
def _run_phase2(ctx):
    """Execute Phase 2 mutagenesis pipeline."""
    approved = ctx.extra.get("approved_mutations", [])
    if not approved:
        # MUT-01: enforce the gate — no silent fallback
        st.error(
            "No approved mutations found. Please use the approval table above "
            "to review and approve mutations before running Phase 2."
        )
        return
    # ... rest of Phase 2 launch
```

**The confirmation dialog for no-prior-approval (CONTEXT.md decision):** The dialog is for a different case — when a user tries to trigger Phase 2 and the `mutagenesis_context.extra["approved_mutations"]` is empty AND they haven't gone through the approval flow at all (not the button-disabled case). Use `@st.dialog` decorator pattern:

```python
@st.dialog("Run Phase 2 without approved mutations?")
def _confirm_phase2_no_approval():
    st.warning("No mutations have been explicitly approved. Phase 2 will use the automatic fallback (saturation at low-confidence positions).")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Proceed anyway", type="primary"):
            st.session_state._phase2_confirmed = True
            st.rerun()
    with col2:
        if st.button("Cancel"):
            st.rerun()
```

### Pattern 2: Disk Persistence (MUT-03, MUT-04)

**What:** Serialize the Phase 1 `WorkflowContext.extra` fields to JSON immediately after `_run_phase1()` succeeds.

**Save point:** End of `_run_phase1()`, after `st.session_state.mutagenesis_context = result.context`.

**Load point:** At the top of the agent tab render function, before checking `phase1_done`.

**File path:** `{job_dir}/phase1_state.json` where `job_dir = _ensure_mutagenesis_job_dir()`.

**WorkflowContext is NOT directly JSON-serializable.** It contains `Path` objects, dataclass instances (`Sequence`, `PredictionResult`, etc.), and complex nested structures. Only `context.extra` and specific primitive fields need to be persisted.

**Fields to persist (research finding):**

From `context.extra` after Phase 1:
- `"mutation_suggestions"` — dict with `positions`, `strategy`, `rationale`, `source`
- `"baseline_low_confidence_positions"` — list of ints
- `"baseline_review"` — string (LLM summary)
- `"mutation_suggestion_raw"` — string (raw LLM output)
- `"mutation_suggestion_source"` — string ("llm" or "fallback")

From context root:
- `context.job_id` — string
- `context.sequences[0].sequence` — string
- `context.sequences[0].id` — string
- `context.output_dir` — string (for reconstruction)

**Schema (Claude's discretion — recommended):**
```json
{
  "schema_version": 1,
  "saved_at": "2026-02-21T10:30:00Z",
  "job_id": "mutagenesis_session_20260221_103000",
  "sequence_id": "my_protein",
  "sequence": "MKTAYIAKQRQISFVK...",
  "output_dir": "./outputs",
  "mutation_suggestions": {...},
  "baseline_low_confidence_positions": [12, 34, 56],
  "baseline_review": "The wild-type structure...",
  "mutation_suggestion_raw": "...",
  "mutation_suggestion_source": "llm"
}
```

**Reconstructing `WorkflowContext` from JSON:** Phase 2 needs a real `WorkflowContext` object (the orchestrator passes it to agents). The load path must reconstruct a minimal context from the saved fields. `context.prediction_results` is NOT needed for Phase 2 — only `approved_mutations`, `sequences`, `job_id`, `output_dir`, `job_dir`, and `mutation_suggestions`/`baseline_low_confidence_positions` (for potential fallback).

```python
def _save_phase1_state(ctx: WorkflowContext, job_dir: Path) -> None:
    """Persist Phase 1 results to disk for session resume."""
    state = {
        "schema_version": 1,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "job_id": ctx.job_id,
        "output_dir": str(ctx.output_dir),
        "sequence_id": ctx.sequences[0].id if ctx.sequences else "",
        "sequence": ctx.sequences[0].sequence if ctx.sequences else "",
        "mutation_suggestions": ctx.extra.get("mutation_suggestions"),
        "baseline_low_confidence_positions": ctx.extra.get("baseline_low_confidence_positions", []),
        "baseline_review": ctx.extra.get("baseline_review", ""),
        "mutation_suggestion_raw": ctx.extra.get("mutation_suggestion_raw", ""),
        "mutation_suggestion_source": ctx.extra.get("mutation_suggestion_source", "unknown"),
    }
    path = job_dir / "phase1_state.json"
    path.write_text(json.dumps(state, indent=2, default=str))


def _load_phase1_state(job_dir: Path) -> Optional[WorkflowContext]:
    """Load Phase 1 results from disk. Returns None if not found or invalid."""
    path = job_dir / "phase1_state.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1:
            return None
        from protein_design_hub.agents.context import WorkflowContext
        from protein_design_hub.core.types import Sequence
        ctx = WorkflowContext(
            job_id=data["job_id"],
            output_dir=Path(data["output_dir"]),
        )
        ctx.job_dir = job_dir
        if data.get("sequence"):
            ctx.sequences = [Sequence(id=data["sequence_id"], sequence=data["sequence"])]
        ctx.extra["mutation_suggestions"] = data.get("mutation_suggestions")
        ctx.extra["baseline_low_confidence_positions"] = data.get("baseline_low_confidence_positions", [])
        ctx.extra["baseline_review"] = data.get("baseline_review", "")
        ctx.extra["mutation_suggestion_raw"] = data.get("mutation_suggestion_raw", "")
        ctx.extra["mutation_suggestion_source"] = data.get("mutation_suggestion_source", "unknown")
        return ctx
    except Exception:
        return None
```

### Pattern 3: Backend Override Propagation (MUT-05)

**What:** Thread the `mut_review_provider` / `mut_review_model` session state into the Phase 1 and Phase 2 orchestrator calls AND into the mutagenesis LLM agents.

**Current state:** `_expert_review_overrides()` at line 418 correctly reads the session state and returns `(provider, model)`. This is called for manual expert panels. But `_run_phase1()` (line 1552) and `_run_phase2()` (line 1734) both construct `AgentOrchestrator` without calling `_expert_review_overrides()` or passing any LLM parameters.

**The `_temporary_llm_override` pattern** (in `agent_helpers.py` line 584) is the correct approach. It saves current LLM settings, applies overrides, yields for the operation, then restores. This is already used in `render_all_experts_panel()` at line 845.

**Fix for `_run_phase1()` and `_run_phase2()`:**
```python
def _run_phase1(sequence: str):
    from web.agent_helpers import _temporary_llm_override
    provider, model = _expert_review_overrides()
    # ...
    with _temporary_llm_override(provider, model):
        result = orchestrator.run(input_path=fasta_path, predictors=["esmfold_api"])
```

**Fix for timing log (model name):** The timing log in `meeting.py` line 122 is:
```python
print(f"  [{agent.title}] {elapsed:.1f}s{tok_info}")
```
The model name used is `agent.resolved_model` (line 90). Adding it to the log requires either:
- Appending it to `tok_info`: `f", {out_tok} tok, {out_tok / elapsed:.0f} tok/s ({model})"` — recommended
- Or appending it after `elapsed`: change the print to include model

The resolved model is `agent.resolved_model` already available in `_call_llm()` at line 90.

**UI "Using: [model] @ [provider]" preview:** Add this as a static caption in the agent tab, before the Phase 2 button is shown:
```python
provider, model = _expert_review_overrides()
effective_provider = provider or cfg.provider
effective_model = model or cfg.model
st.caption(f"Using: `{effective_model}` @ `{effective_provider}`")
```

### Recommended Implementation Order

1. **Approval gate** (MUT-01, MUT-02) — lowest risk, purest change to `_run_phase2()` + button rename
2. **Backend override propagation** (MUT-05) — timing log + `_temporary_llm_override` wrapper + UI caption
3. **Disk persistence** (MUT-03, MUT-04) — `_save_phase1_state()` + `_load_phase1_state()` + auto-load on page render

### Anti-Patterns to Avoid

- **Serializing the entire `WorkflowContext`:** It contains unpicklable/non-JSON-serializable objects (`Path`, dataclass instances, prediction results). Serialize only the `extra` fields needed for Phase 2.
- **Re-running Phase 1 silently on load:** Loading from disk should restore the context as-is, not trigger any computation.
- **Global settings mutation for override:** Do not permanently mutate `get_settings().llm.*` for Phase 2 overrides. Use `_temporary_llm_override` which restores settings after the call.
- **Calling `reset_llm_client()` outside of `_temporary_llm_override`:** The existing pattern in `agent_helpers.py` handles this correctly; don't duplicate it.
- **Blocking Phase 2 with a hard error when no approval exists:** CONTEXT.md specifies a confirmation dialog, not a hard block. Users must be able to override.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Temporary LLM provider switch | Custom save/restore logic | `_temporary_llm_override()` in `agent_helpers.py` line 584 | Already handles reset, restore, cache invalidation correctly |
| Confirmation modal/dialog | HTML injection or session state polling | `@st.dialog` Streamlit decorator | Native Streamlit pattern; handles rerun lifecycle correctly |
| JSON serialization of complex types | Custom recursive serializer | `json.dumps(..., default=str)` | Already used in `mutagenesis_agents.py` line 427; handles Path, datetime |
| LLM client reconnection after override | Manual client teardown | `reset_llm_client()` inside `_temporary_llm_override` | Already called in the helper |

---

## Common Pitfalls

### Pitfall 1: `st.dialog` rerun lifecycle

**What goes wrong:** `@st.dialog` decorated functions execute in a separate Streamlit script run. If you set session state inside the dialog and call `st.rerun()`, the dialog closes but the state change may not persist if the calling code also called `st.rerun()`.
**Why it happens:** Streamlit's rerun cycle; dialogs are rendered as overlays that disappear on rerun.
**How to avoid:** Use a sentinel session state key (e.g., `st.session_state._phase2_confirmed`) that persists after dialog close. Check it at the start of the render function.
**Warning signs:** Button click in dialog appears to do nothing; Phase 2 never starts.

### Pitfall 2: `WorkflowContext` reconstruction misses `job_dir`

**What goes wrong:** Loaded context has `output_dir` set but `job_dir = None`. When `MutationExecutionAgent.run()` calls `context.with_job_dir()`, it creates a new job dir using a fresh timestamp, losing the connection to the session's job directory.
**Why it happens:** `job_dir` is derived lazily from `output_dir / job_id` in `context.with_job_dir()` — if `job_dir` is None, it auto-creates. But the session's job dir path is `mutagenesis_session_YYYYMMDD_HHMMSS/` which differs from what `with_job_dir()` would compute.
**How to avoid:** Explicitly set `ctx.job_dir = job_dir` when reconstructing context from disk.
**Warning signs:** Phase 2 creates a new timestamped directory different from the Phase 1 job dir.

### Pitfall 3: `_temporary_llm_override` is not re-entrant

**What goes wrong:** If `_run_phase1()` uses `_temporary_llm_override` and the same call is somehow re-entered (Streamlit double-click), the global `settings.llm.*` may be left in an override state.
**Why it happens:** The `finally` block in `_temporary_llm_override` restores settings, but only if the try block ran. A crash before the `yield` would leave settings corrupted.
**How to avoid:** The existing implementation already handles this correctly — `previous` is captured before overriding, so the `finally` block always restores. No additional protection needed.
**Warning signs:** LLM calls after Phase 1 unexpectedly use the wrong model/provider.

### Pitfall 4: Phase 1 state JSON saved before `job_dir` is established

**What goes wrong:** `_run_phase1()` calls `orchestrator.run(input_path=fasta_path)` which creates the context via `WorkflowContext(...)`. The `job_id` is derived from `input_path` (the tempfile name like `tmpXXXXXX`), not from the mutagenesis session dir.
**Why it happens:** `_run_phase1()` uses a temporary FASTA file; the context's `output_dir` is from settings and `job_id` is from the tempfile name. The mutagenesis session dir (`_ensure_mutagenesis_job_dir()`) is separate.
**How to avoid:** Save phase1_state.json to `_ensure_mutagenesis_job_dir()`, not to `context.with_job_dir()`. These are different directories.
**Warning signs:** `phase1_state.json` ends up in `outputs/tmpXXXXXX_DATE/` instead of `outputs/mutagenesis_session_DATE/`.

### Pitfall 5: `_parse_approved_mutations` column name dependency

**What goes wrong:** `_parse_approved_mutations()` at line 1701 uses `row["Position"]`, `row["WT AA"]`, `row["Target AAs"]`. If the DataFrame is reconstructed from JSON (e.g., for a "re-approve" flow), column names might differ.
**Why it happens:** The editable dataframe uses display-friendly column names; JSON uses different keys.
**How to avoid:** The save schema for `mutation_suggestions` preserves the raw `positions` list (not the DataFrame). The approval table is always rebuilt fresh from `ctx.extra["mutation_suggestions"]["positions"]`. No round-trip through the DataFrame format is needed.

---

## Code Examples

### Exact timing log format (current vs required)

**Current** (meeting.py line 121-122):
```python
tok_info = f", {out_tok} tok, {out_tok / elapsed:.0f} tok/s"
print(f"  [{agent.title}] {elapsed:.1f}s{tok_info}")
# Output: [LLMBaselineReviewAgent] 3.2s, 120 tok, 37 tok/s
```

**Required** (CONTEXT.md decision):
```python
tok_info = f", {out_tok} tok, {out_tok / elapsed:.0f} tok/s ({model})"
print(f"  [{agent.title}] {elapsed:.1f}s{tok_info}")
# Output: [LLMBaselineReviewAgent] 3.2s, 120 tok, 37 tok/s (qwen2.5:14b)
```

The `model` variable is already available at line 90 as `model = agent.resolved_model`.

### Phase 1 auto-load detection

Where to call `_load_phase1_state()`: at the start of the agent tab render, between line 1493 and the `phase1_done` check. The job dir must come from `_ensure_mutagenesis_job_dir()`.

```python
# At the top of the agent tab render (around line 1493):
phase1_ctx = st.session_state.get("mutagenesis_context")

# Auto-load from disk if not in session state
if phase1_ctx is None:
    job_dir = _ensure_mutagenesis_job_dir()
    loaded_ctx = _load_phase1_state(job_dir)
    if loaded_ctx is not None:
        st.session_state.mutagenesis_context = loaded_ctx
        phase1_ctx = loaded_ctx
        st.caption("Loaded from previous session")

phase1_done = phase1_ctx is not None and phase1_ctx.extra.get("mutation_suggestions") is not None
```

### Backend override end-to-end wire

**In `_run_phase1()` (line 1552):**
```python
from protein_design_hub.web.agent_helpers import _temporary_llm_override
provider, model = _expert_review_overrides()

with st.status("Running Phase 1 — Agent Analysis...", expanded=True) as status:
    # ...
    with _temporary_llm_override(provider, model):
        result = orchestrator.run(
            input_path=fasta_path,
            predictors=["esmfold_api"],
        )
```

**In `_run_phase2()` (line 1726):**
```python
from protein_design_hub.web.agent_helpers import _temporary_llm_override
provider, model = _expert_review_overrides()

with st.status("Running Phase 2 — Executing Mutations...", expanded=True) as status:
    # ...
    with _temporary_llm_override(provider, model):
        result = orchestrator.run_with_context(ctx)
```

**UI preview caption (before Phase 2 button):**
```python
try:
    cfg = get_settings().llm.resolve()
    provider, model = _expert_review_overrides()
    effective_provider = provider or cfg.provider
    effective_model = model or cfg.model
    st.caption(f"Using: `{effective_model}` @ `{effective_provider}`")
except Exception:
    pass
```

---

## Key Facts Found in Research

### Q1: Where exactly is `context.extra["approved_mutations"]` read? What happens if empty?

**Answer (HIGH confidence):** `MutationExecutionAgent.run()` in `mutagenesis_agents.py` lines 83-103.

```python
approved = context.extra.get("approved_mutations", [])
if not approved:
    low_conf = context.extra.get("baseline_low_confidence_positions", [])
    if low_conf:
        # SILENT FALLBACK: builds saturation list from top 5 low-conf positions
        approved = [{"residue": pos, "wt_aa": seq[pos-1], "targets": ["*"]} for pos in low_conf[:5]]
        logger.info("No approved mutations; falling back to saturation at top-%d low-confidence positions", len(approved))
    else:
        return AgentResult.fail("No approved mutations and no low-confidence positions available.")
```

If `approved_mutations` is empty AND `baseline_low_confidence_positions` is non-empty, the agent silently executes saturation mutagenesis at up to 5 positions. The UI never shows this to the user.

### Q2: What is the job directory structure? Where would a phase1_state.json be written?

**Answer (HIGH confidence):** Two separate directory concepts exist:

1. **Agent pipeline job dir:** `{settings.output.base_dir}/{job_id}/` where `job_id` is derived from the temp FASTA filename (e.g., `outputs/tmpXXXXXX_20260221_103000/`). This is `context.with_job_dir()`.

2. **Mutagenesis session dir:** `{settings.output.base_dir}/mutagenesis_session_{YYYYMMDD_HHMMSS}/` managed by `_ensure_mutagenesis_job_dir()` at line 616. This is where manual expert panels save meetings.

**The phase1_state.json should be written to the mutagenesis session dir** (not the pipeline job dir), because:
- The session dir survives across Phase 1 and Phase 2 runs
- It is already tracked in `st.session_state.mutagenesis_job_dir`
- The pipeline job dir is a tempfile-derived path that changes each time Phase 1 is re-run

**Write path:** `_ensure_mutagenesis_job_dir() / "phase1_state.json"`
**Real example of existing session dirs:** `outputs/mutagenesis_session_20260210_023110/`

### Q3: How does the current timing log work? Where to add model name?

**Answer (HIGH confidence):** `meeting.py` lines 106-122.

The `model` variable is defined at line 90 as `model = agent.resolved_model`. The timing print is at line 122. Adding model name requires one-line change to `tok_info` construction.

Current format: `  [LLMBaselineReviewAgent] 3.2s, 120 tok, 37 tok/s`
Required format: `  [LLMBaselineReviewAgent] 3.2s, 120 tok, 37 tok/s (qwen2.5:14b)`

Change is in the `if out_tok and elapsed > 0:` block at line 120-121.

### Q4: How do session state keys flow (or fail to flow) into agent construction?

**Answer (HIGH confidence):** The session state keys are NOT currently passed to agents at all.

**What exists:**
- `st.session_state.mut_review_provider` — set by UI at line 893
- `st.session_state.mut_review_model` — set by UI at line 964
- `_expert_review_overrides()` — reads these and returns `(provider, model)` tuple; called at lines 1187, 2166, 2595, 2834

**What is missing:**
- `_run_phase1()` at line 1552: constructs `AgentOrchestrator(mode="mutagenesis_pre", ...)` with no provider/model kwargs
- `_run_phase2()` at line 1734: constructs `AgentOrchestrator(mode="mutagenesis_post", ...)` with no provider/model kwargs

**The gap:** The LLM agents inside the orchestrator call `_call_llm()` in `meeting.py`, which calls `_get_llm_client()` which reads `get_settings().llm`. The `_temporary_llm_override` context manager in `agent_helpers.py` temporarily mutates `settings.llm.*` so all subsequent `_get_llm_client()` calls pick up the override. This is the correct fix path — no changes to agent constructors needed.

### Q5: What Streamlit session state keys exist for tracking phase completion?

**Answer (HIGH confidence):**

| Key | Default | Set when | Read where |
|-----|---------|----------|------------|
| `mutagenesis_context` | `None` | Phase 1 completes (line 1564) | `phase1_done` check (line 1493-1494) |
| `mutagenesis_phase2_context` | `None` | Phase 2 completes (line 1743) | Phase 2 results render (line 1516) |
| `mut_review_provider` | `"current"` | UI interaction (line 893) | `_expert_review_overrides()` (line 428) |
| `mut_review_model` | `""` | UI interaction (line 964) | `_expert_review_overrides()` (line 429) |
| `mut_review_custom_provider` | `""` | UI interaction (line 908) | `_expert_review_overrides()` (line 430) |
| `mutagenesis_job_dir` | `""` | `_ensure_mutagenesis_job_dir()` (line 630) | `_meeting_save_dir()` |

---

## Open Questions

1. **`st.dialog` availability in current Streamlit version**
   - What we know: `@st.dialog` was introduced in Streamlit 1.36. The project uses Streamlit (version not pinned in CLAUDE.md).
   - What's unclear: Exact installed version.
   - Recommendation: Check `pip show streamlit` at plan execution time. If `@st.dialog` unavailable, use `st.session_state` + conditional render as fallback confirmation pattern.

2. **Whether Phase 1 override should also affect non-LLM agents**
   - What we know: `MutationExecutionAgent` and `MutationComparisonAgent` are pure Python (no LLM calls); the override only matters for `LLMBaselineReviewAgent`, `LLMMutationSuggestionAgent`, `LLMMutationResultsAgent`.
   - What's unclear: None — CONTEXT.md says "ALL Phase 2 mutagenesis agents use the selected backend". The computational agents don't call LLM so they are unaffected regardless. The three LLM agents in Phase 2 are all covered by `_temporary_llm_override`.
   - Recommendation: `_temporary_llm_override` wrapping the entire `orchestrator.run_with_context()` call covers all LLM agents in the pipeline. No per-agent override needed.

3. **What happens if `_ensure_mutagenesis_job_dir()` returns a new dir on page reload**
   - What we know: If `st.session_state.mutagenesis_job_dir` is empty (browser close + reload), `_ensure_mutagenesis_job_dir()` creates a NEW timestamped directory.
   - What's unclear: How to find the previous session's job dir on reload.
   - Recommendation: The load flow must search the `outputs/` directory for the most recent `mutagenesis_session_*/phase1_state.json` file when session state has no job dir. Add a helper `_find_latest_phase1_state()` that lists `settings.output.base_dir` for `mutagenesis_session_*` dirs, sorts by mtime, and returns the newest `phase1_state.json`.

---

## Sources

### Primary (HIGH confidence)
All findings from direct codebase inspection. File paths and line numbers verified.

- `src/protein_design_hub/web/pages/10_mutation_scanner.py` — approval step (1609-1699), `_run_phase1` (1526-1574), `_run_phase2` (1722-1747), `_expert_review_overrides` (418-440), session state defaults (390-413), `_ensure_mutagenesis_job_dir` (616-632)
- `src/protein_design_hub/agents/mutagenesis_agents.py` — `MutationExecutionAgent.run()` (82-274), approved_mutations reading (83-103), report writing (397-451)
- `src/protein_design_hub/agents/orchestrator.py` — `_build_mutagenesis_post_approval_pipeline` (140-161), `AgentOrchestrator.__init__` (170-214), `run_with_context` (263-306)
- `src/protein_design_hub/agents/meeting.py` — `_call_llm` (79-123), timing log format (106-122), `_get_llm_client` (48-69), `reset_llm_client` (72-76)
- `src/protein_design_hub/web/agent_helpers.py` — `_temporary_llm_override` (584-629), `switch_llm_provider` (102-118)
- `src/protein_design_hub/agents/context.py` — `WorkflowContext` definition (16-59), `with_job_dir()` (54-59)
- `src/protein_design_hub/agents/llm_guided.py` — `LLMBaselineReviewAgent.__init__` (1053-1063), `LLMMutationSuggestionAgent.__init__` (1198-1208), `LLMMutationResultsAgent.__init__` (1388-1398)

### Secondary (MEDIUM confidence)
- Streamlit `@st.dialog` API — from Streamlit release notes knowledge; recommend confirming installed version at plan time

---

## Metadata

**Confidence breakdown:**
- MUT-01 (approval gate): HIGH — exact code path traced, root cause confirmed
- MUT-02 (button gate): HIGH — current button behavior verified, fix is a rename + guard
- MUT-03 (disk save): HIGH — no persistence code exists; job dir structure confirmed from real output directories
- MUT-04 (disk load): HIGH — load point identified; `WorkflowContext` reconstruction pattern designed
- MUT-05 (backend overrides): HIGH — exact gap identified (`_run_phase1`/`_run_phase2` don't call `_expert_review_overrides()`), correct fix pattern (`_temporary_llm_override`) identified from existing usage in `render_all_experts_panel`

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (stable Python/Streamlit codebase; no external APIs involved)
