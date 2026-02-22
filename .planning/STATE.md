# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** A reliable, end-to-end protein design workflow where a researcher goes from sequence to structure to expert analysis to mutagenesis to report without data loss, silent failures, or manual workarounds.
**Current focus:** Phase 3 - Performance & Reliability (complete)

## Current Position

Phase: 4 of 8 (Test Coverage)
Plan: 2 of 2 in current phase
Status: Phase 4 complete — 04-01 complete, 04-02 complete
Last activity: 2026-02-22 — 04-02 complete: TEST-03 failure modes + TEST-01 Phase 1/Phase 2 integration; 56 tests pass

Progress: [########░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 5 min
- Total execution time: 0.5 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-git-and-code-health | 2 | 6 min | 3 min |
| 02-mutagenesis-workflow-integrity | 3 | 20 min | 7 min |
| 03-performance-and-reliability | 2 | 4 min | 2 min |

**Recent Trend:**
- Last 5 plans: 02-01 (12 min), 02-02 (4 min), 02-03 (4 min), 03-01 (2 min), 03-02 (2 min)
- Trend: fast (simple surgical edits)

*Updated after each plan completion*
| Phase 04-test-coverage P01 | 5 | 2 tasks | 1 files |
| Phase 04-test-coverage P02 | 12 | 2 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- qwen2.5:14b as default LLM (best quality/fit for 12 GB VRAM)
- Phase 1/Phase 2 mutagenesis split (allows user review before expensive execution) — approval gate enforced by 02-01
- OST comprehensive scoring per mutant flagged for revisit — Phase 3 addresses cap/optional flag
- [01-01] *.bak confirmed safe to delete: current 10_mutation_scanner.py is superset of .bak (contains _render_manual_tab_settings() refactor)
- [01-01] *.bak added to Temporary files section of .gitignore alongside *.tmp
- [01-02] Canonical class name is MutagenesisPipelineReportAgent (fixes missing 's' typo in MutagenesiReportAgent)
- [01-02] Backward-compatible alias MutagenesiReportAgent = MutagenesisPipelineReportAgent kept at module level
- [02-01] _run_phase2() guard checks both empty approved_mutations AND the _phase2_confirmed sentinel so dialog bypass is allowed without a hard block
- [02-01] Inline st.error shown unconditionally when zero mutations selected (above disabled button) as persistent UX hint
- [02-01] Sentinel _phase2_confirmed reset AFTER calling _run_phase2() in bypass path (not before) to ensure guard sees True value
- [02-02] Serialize only context.extra primitive fields to phase1_state.json — full WorkflowContext is not JSON-serializable
- [02-02] Save to _ensure_mutagenesis_job_dir() path (mutagenesis session dir), NOT context.with_job_dir() (temp FASTA job dir — different directory)
- [02-02] No st.rerun() after auto-load — Streamlit reads newly-set session state in same script execution cycle
- [02-02] ctx.job_dir set explicitly after WorkflowContext construction to avoid mis-derived path from with_job_dir()
- [02-03] _temporary_llm_override wraps orchestrator.run() in both _run_phase1() and _run_phase2() — NOT the AgentOrchestrator constructor
- [02-03] Model name appended in parentheses to tok_info: tok/s ({model}) — uses agent.resolved_model already computed at line 90 of meeting.py
- [02-03] UI caption uses try/except to never block UI; falls back to get_settings().llm.{provider,model} when no override selected
- [03-01] _check_scanner_api() defined as function (not bare module code) to avoid circular import issues at import time
- [03-01] OST cap threshold is 3 distinct positions; >3 auto-disables OST, ost_force_override bypasses cap
- [03-01] ost_auto_disabled_reason stored in context.extra as human-readable string for downstream UI display
- [03-02] Warnings appear at TOP of _run_phase2() before st.status spinner so user sees them before long execution begins
- [03-02] Force OST checkbox only shown when >3 positions selected (matches OST cap threshold from 03-01)
- [03-02] Module-level logger added to llm_guided.py; both inline import logging occurrences replaced with logger.warning()
- [03-02] context.extra as UI signal bus: agent writes warning string, UI reads and renders st.warning before expensive execution
- [Phase 04-test-coverage]: [04-01] _load_scanner_page uses patch() for UI helpers + real streamlit with AttrDict session_state; plan's pure MagicMock failed because ui.py holds real st refs at import time
- [Phase 04-test-coverage]: [04-01] AttrDict(dict) subclass supports both session_state['key'] and session_state.key attribute-style access used by 10_mutation_scanner.py at module level
- [Phase 04-test-coverage]: [04-02] MutationScanner imported from protein_design_hub.analysis.mutation_scanner (not mutagenesis_agents which does not re-export it)
- [Phase 04-test-coverage]: [04-02] TEST-01 uses patch.object(real_scanner, "predict_single") + _build_scanner patch to return real instance — real __init__ exercised, no HTTP calls
- [Phase 04-test-coverage]: [04-02] TEST-01 excludes MutationComparisonAgent from Phase 2 pipeline — only MutationExecutionAgent needed to verify approved_mutations → mutation_results flow

### Pending Todos

None yet.

### Blockers/Concerns

- ~~`mutagenesis_agents.py` is untracked in git~~ — RESOLVED by 01-01
- ~~Class name typo MutagenesiReportAgent (missing 's')~~ — RESOLVED by 01-02
- ~~Missing approval gate between mutagenesis Phase 1 and Phase 2 is a live correctness bug~~ — RESOLVED by 02-01
- ~~No Phase 1 state persistence (data loss on browser close)~~ — RESOLVED by 02-02
- ~~Expert backend overrides silently ignored during pipeline execution~~ — RESOLVED by 02-03
- ~~OST runs on too many positions causing multi-hour runtimes~~ — RESOLVED by 03-01 (cap) + 03-02 (Force OST override UI)
- ~~LLM fallback decision invisible to user~~ — RESOLVED by 03-02 (context.extra warning + st.warning)
- ~~No Phase 1 to Phase 2 integration test~~ — RESOLVED by 04-02 (TEST-01 integration test)

## Session Continuity

Last session: 2026-02-22
Stopped at: Completed 04-02-PLAN.md (TEST-03 failure modes + TEST-01 Phase 1/Phase 2 integration; 2 tasks; 56 tests pass)
Resume file: None
