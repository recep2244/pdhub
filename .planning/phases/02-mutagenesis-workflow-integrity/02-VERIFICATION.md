---
phase: 02-mutagenesis-workflow-integrity
verified: 2026-02-21T23:30:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
gaps: []
human_verification:
  - test: "Confirm 'Loaded from previous session' caption appears on page reload"
    expected: "After Phase 1 completes and browser tab is closed, reopening the mutation scanner page shows the Phase 1 results and a 'Loaded from previous session' caption without re-running Phase 1"
    why_human: "Cannot simulate browser session loss in automated tests; requires actual Streamlit browser session"
  - test: "Confirm per-call timing log includes model name"
    expected: "During a Phase 1 or Phase 2 pipeline run, stdout shows lines like '[LLMBaselineReviewAgent] 3.2s, 120 tok, 37 tok/s (qwen2.5:14b)' with the model name in parentheses"
    why_human: "Timing output is printed to stdout during a live LLM call; requires actual Ollama backend running"
  - test: "Confirm backend override caption appears in UI"
    expected: "When a non-default backend is selected in the expert panel, the approval table shows 'Using: `model` @ `provider`' caption"
    why_human: "Streamlit UI widget state cannot be asserted without a running Streamlit server"
---

# Phase 2: Mutagenesis Workflow Integrity Verification Report

**Phase Goal:** The Phase 1 to Phase 2 mutagenesis transition is correct — approval is enforced, state survives browser close, and backend overrides reach the agents they are intended for.
**Verified:** 2026-02-21T23:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running Phase 2 without approved mutations shows a blocking error and does not execute any mutations | VERIFIED | `_run_phase2()` line 1876: guard checks `not approved and not _phase2_confirmed`; returns with `st.error` immediately. "Approve & Continue" button is `disabled=approve_disabled` where `approve_disabled = len(included) == 0` (line 1827). No code path can invoke `_run_phase2()` with empty approved mutations without the sentinel. |
| 2 | "Approve & Continue" button is present and must be clicked before Phase 2 can start; clicking it with no mutations selected does nothing | VERIFIED | Button labeled "Approve & Continue" at line 1831 with `disabled=approve_disabled`. `st.error("Select at least one mutation to approve.")` rendered unconditionally when `approve_disabled=True` (line 1829). The button handler (line 1842) calls `_run_phase2(ctx)` only after setting `approved_mutations` from the edited dataframe. |
| 3 | When Phase 1 completes, approved mutations, suggestions, and low-confidence positions are written to the job directory on disk | VERIFIED | `_run_phase1()` line 1681-1685: immediately after `result.success`, calls `_save_phase1_state(result.context, job_dir)`. `_save_phase1_state` at line 638 writes `phase1_state.json` containing `mutation_suggestions`, `baseline_low_confidence_positions`, `baseline_review`, and `mutation_suggestion_raw` with `schema_version=1`. |
| 4 | Returning to a completed Phase 1 job (after closing the browser) loads previous results from disk without re-running Phase 1 | VERIFIED | Auto-load block at lines 1579-1592: if `mutagenesis_context` is absent from session state, tries `_load_phase1_state(known_job_dir)` then `_find_latest_phase1_state()` (glob search by mtime). Sets `phase1_ctx` before `phase1_done` computation at line 1594. `st.caption("Loaded from previous session")` displayed on restore. |
| 5 | Selecting a different LLM backend in the expert panel UI causes that agent to use that backend; the change is observable in per-call timing logs | VERIFIED | `_run_phase1()` and `_run_phase2()` both call `_expert_review_overrides()` and wrap `orchestrator.run()` with `_temporary_llm_override(provider, model)` (lines 1672 and 1902). `meeting.py` line 121: `tok_info = f", {out_tok} tok, {out_tok / elapsed:.0f} tok/s ({model})"` appends model name. UI caption at line 1821 shows `f"Using: \`{effective_model}\` @ \`{effective_provider}\`"`. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/protein_design_hub/web/pages/10_mutation_scanner.py` | Approval gate guard, persistence helpers, override wiring | VERIFIED | 3007 lines. Contains `_run_phase2` guard, `_render_approval_step` with disabled button, `_save_phase1_state`, `_load_phase1_state`, `_find_latest_phase1_state`, `_expert_review_overrides`, `_temporary_llm_override` imports in both phase runners. |
| `src/protein_design_hub/agents/meeting.py` | Model name in per-call timing log | VERIFIED | Line 121: `tok_info = f", {out_tok} tok, {out_tok / elapsed:.0f} tok/s ({model})"`. `model` is `agent.resolved_model` (line 90). |
| `src/protein_design_hub/web/agent_helpers.py` | `_temporary_llm_override` context manager | VERIFIED | Lines 583-630: full context manager with save/restore of `(provider, base_url, model, api_key)` and `reset_llm_client()` on entry and exit. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_run_phase2()` | `ctx.extra['approved_mutations']` | empty check at function entry | WIRED | Line 1875-1881: `approved = ctx.extra.get("approved_mutations", []); if not approved and not st.session_state.get("_phase2_confirmed", False): st.error(...); return` |
| `_confirm_phase2_no_approval dialog` | `st.session_state._phase2_confirmed` | sentinel key | PARTIAL (see note) | Dialog is defined at line 1731 with `@st.dialog`. It sets `_phase2_confirmed=True` at line 1740. However the dialog is never called from any button or code path — it is dead code. The sentinel CAN be set to True only within this unreachable function. The gate works correctly regardless: the "Approve & Continue" button is disabled with 0 selections, and `_run_phase2` guards against empty approved list. |
| `_run_phase1()` | `_temporary_llm_override` | wraps `orchestrator.run()` | WIRED | Lines 1642-1672: `_temporary_llm_override(provider, model)` context manager wraps the orchestrator call. |
| `_run_phase2()` | `_temporary_llm_override` | wraps `orchestrator.run_with_context()` | WIRED | Lines 1884-1902: same pattern. |
| `_save_phase1_state()` | `phase1_state.json` on disk | called immediately after Phase 1 success | WIRED | Lines 1681-1685: inside `if result.success and result.context:` block, `_save_phase1_state(result.context, job_dir)` is called. |
| `_find_latest_phase1_state()` | `phase1_ctx` in session state | auto-load before `phase1_done` computation | WIRED | Lines 1579-1594: loaded into session state before `phase1_done = phase1_ctx is not None and ...` |
| `_expert_review_overrides()` | `_temporary_llm_override` | return values passed directly | WIRED | `provider, model = _expert_review_overrides()` then `with _temporary_llm_override(provider, model):` in both phase runners. |
| `meeting.py _call_llm()` | timing log with model name | f-string interpolation | WIRED | `tok_info = f", {out_tok} tok, {out_tok / elapsed:.0f} tok/s ({model})"` where `model = agent.resolved_model`. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MUT-01 | 02-01 | Phase 2 is blocked if no explicit user approval from Phase 1 | SATISFIED | `_run_phase2()` guard at entry (line 1876) blocks execution with `st.error` and `return`. |
| MUT-02 | 02-01 | "Approve & Continue" button in UI enforces approval before Phase 2 runs | SATISFIED | Button at line 1831, `disabled=approve_disabled` (zero selections). `st.error` shown as inline hint when disabled. |
| MUT-03 | 02-02 | Phase 1 results persisted to job directory on disk when Phase 1 completes | SATISFIED | `_save_phase1_state()` writes `phase1_state.json` with `mutation_suggestions`, `baseline_low_confidence_positions`, `baseline_review`, schema_version. Called immediately after Phase 1 success. |
| MUT-04 | 02-02 | UI can load previous Phase 1 results when returning to a job (survives browser close) | SATISFIED | `_find_latest_phase1_state()` globs `mutagenesis_session_*` dirs by mtime; auto-load fires before `phase1_done` check; `st.caption("Loaded from previous session")` shown. |
| MUT-05 | 02-03 | Per-expert backend overrides flow verified end-to-end (session state → orchestrator kwargs → agent constructor) | SATISFIED | `_expert_review_overrides()` reads session state keys; values passed to `_temporary_llm_override()` which mutates `get_settings().llm.*` and calls `reset_llm_client()`; all subsequent `_call_llm()` calls in orchestrator use the overridden settings. Model name appears in timing log via `({model})` suffix. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `10_mutation_scanner.py` | 1731 | `_confirm_phase2_no_approval` dialog defined but never called | INFO | The `@st.dialog` function exists and contains correct sentinel logic but has no call site. The bypass path (dialog -> sentinel -> `_run_phase2`) is dead code. This does not compromise the gate's correctness — the gate works via `approve_disabled` button + `_run_phase2` guard — but the "Proceed anyway" user path documented in CONTEXT.md is unreachable. The SUMMARY for 02-01 documents this: "no such external trigger existed." |

**Severity assessment:** INFO (not a blocker). The gate functions correctly without the bypass dialog. The orphaned dialog code does not introduce any risk — it can only be reached if `_confirm_phase2_no_approval()` is explicitly called, which currently never happens.

### Human Verification Required

#### 1. Phase 1 State Load After Browser Close

**Test:** Run Phase 1 on a protein sequence, wait for it to complete, then close the browser tab completely. Open a new browser tab and navigate to the Mutation Scanner page.
**Expected:** The page shows Phase 1 results (mutation suggestions table) with a "Loaded from previous session" caption, without re-running the Phase 1 pipeline.
**Why human:** Requires actual Streamlit server and real browser session. Session state loss from browser close cannot be simulated in unit tests.

#### 2. Model Name in Timing Log

**Test:** With Ollama running (`ollama serve`) and qwen2.5:14b loaded, trigger Phase 1 on a short sequence. Observe the terminal/server logs.
**Expected:** Each LLM agent call produces a log line matching `[AgentName] X.Xs, N tok, N tok/s (qwen2.5:14b)` — the model name in parentheses at the end.
**Why human:** Requires a live LLM backend (Ollama) to produce actual token usage data. Without a response with `usage.completion_tokens`, the `tok_info` string stays empty and only `elapsed` is logged.

#### 3. Backend Override Caption in Approval UI

**Test:** On the Mutation Scanner page after Phase 1 completes, open the expert panel settings and select "Ollama" with model "deepseek-r1:14b". Scroll down to the mutation approval table.
**Expected:** A caption reading `Using: \`deepseek-r1:14b\` @ \`ollama\`` appears above the "Approve & Continue" button.
**Why human:** Requires a running Streamlit server with the expert panel override widget rendered and interacted with.

### Gaps Summary

No gaps blocking goal achievement. All five success criteria are verifiably implemented in the codebase:

1. **SC1 (approval gate):** `_run_phase2()` has a two-layer guard — the "Approve & Continue" button is structurally disabled when no mutations are selected, AND the function itself returns early with `st.error` if `approved_mutations` is empty and `_phase2_confirmed` is not set. Phase 2 cannot execute silently.

2. **SC2 (Approve & Continue button):** The button exists, is labeled correctly, is disabled on zero selections, and shows an inline error hint. Clicking it with selections correctly calls `_run_phase2()` only after writing approved mutations to context.

3. **SC3 (Phase 1 disk write):** `_save_phase1_state()` is called immediately after `result.success` in `_run_phase1()`, writing a schema-versioned JSON file with all necessary fields.

4. **SC4 (browser-close resilience):** Auto-load fires at tab render entry, before `phase1_done` is computed, using a two-tier search (known job dir → glob by mtime).

5. **SC5 (backend overrides):** Both phase runners call `_expert_review_overrides()` and wrap orchestrator execution in `_temporary_llm_override()`. The context manager correctly saves and restores settings. The timing log includes model name at meeting.py line 121.

**Notable finding:** `_confirm_phase2_no_approval` dialog is defined but never called. This is acknowledged dead code — the SUMMARY for 02-01 explicitly notes "no such external trigger existed." The approval gate works correctly without it. This is an INFO-level finding, not a gap.

**Test suite:** 46 tests pass (`python -m pytest tests/ -x -q` — 39.53s). All commits (46b4006, 931d426, d1afac8, 2b44839) verified present in git history.

---

_Verified: 2026-02-21T23:30:00Z_
_Verifier: Claude (gsd-verifier)_
