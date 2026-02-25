---
phase: 06-new-agent-workflows
plan: "02"
subsystem: agents-ui
tags: [streamlit, pipeline-modes, ui, nanobody, binding-affinity]

# Dependency graph
requires:
  - phase: 06-01
    provides: "nanobody_llm and binding_affinity pipeline modes in orchestrator.py"
provides:
  - "_PIPELINE_MODES dict wired into Tab 1 of 11_agents.py with four selectable modes"
  - "Dynamic mode_str derivation from session_state + widget, powering expander and steps preview"
  - "use_llm = mode_str != 'step' replacing brittle string-contains check"
affects:
  - 11_agents.py Tab 1 (Run Pipeline) - pipeline mode selectbox, LLM Settings visibility, steps preview

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_PIPELINE_MODES dict at top of with tabs[1]: block; mode_str derived from session_state before widget, re-bound after"
    - "use_llm = mode_str != 'step' replaces 'LLM' in pm string matching"

key-files:
  created: []
  modified:
    - src/protein_design_hub/web/pages/11_agents.py

key-decisions:
  - "_PIPELINE_MODES defined inside with tabs[1]: block (not module-level) — keeps it scoped to Tab 1 as specified"
  - "mode_str derived from session_state.get('p_mode', ...) at top of tab to make it available to expander (which renders before the selectbox widget in source order)"
  - "mode_str re-bound after st.selectbox() call so the actual widget value is used for everything below (steps preview, execution block)"
  - "use_llm = mode_str != 'step' replaces brittle 'LLM' in pm check — works correctly for all four modes"

patterns-established:
  - "Dict-based mode mapping: _PIPELINE_MODES[pm] -> mode_str; single source of truth for label-to-string mapping"

requirements-completed:
  - AGT-01
  - AGT-02

# Metrics
duration: 2min
completed: 2026-02-25
---

# Phase 6 Plan 02: Four-Mode Pipeline Selector with Dynamic Steps Preview Summary

**_PIPELINE_MODES dict replacing binary LLM/step selectbox — four modes (llm, step, nanobody_llm, binding_affinity) with dynamic expander, steps preview, and execution block all driven by mode_str**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-25T23:22:09Z
- **Completed:** 2026-02-25T23:22:35Z (awaiting Task 2 checkpoint approval)
- **Tasks completed:** 1 of 2 (Task 2 is human-verify checkpoint)
- **Files modified:** 1

## Accomplishments

- Added `_PIPELINE_MODES` dict with four entries: `"LLM-guided (recommended)"` -> `"llm"`, `"Step-only (fast, no LLM)"` -> `"step"`, `"Antibody / Nanobody Design"` -> `"nanobody_llm"`, `"Binding Affinity Analysis"` -> `"binding_affinity"`
- Derived `mode_str` at the top of `with tabs[1]:` from `session_state.get("p_mode", ...)` so the expander (rendered before the selectbox) uses the correct mode on re-runs
- Re-bound `mode_str = _PIPELINE_MODES[pm]` after the `st.selectbox()` call so the actual widget value drives all downstream logic
- Updated `use_llm = mode_str != "step"` — LLM Settings appear for all three LLM-using modes
- Updated `_pipeline_table_markdown(mode_str)` in expander — no longer hardcoded to `"llm"`
- Updated `AgentOrchestrator(mode=mode_str)` in pipeline steps preview
- Updated `mode = mode_str` in the execution block

## Task Commits

1. **Task 1: Replace binary pipeline mode selector with _PIPELINE_MODES mapping** - `7e71e82` (feat)

## Files Created/Modified

- `src/protein_design_hub/web/pages/11_agents.py` — five targeted edits to Tab 1: _PIPELINE_MODES dict at top, expander updated, selectbox expanded to 4 options, steps preview updated, execution block updated

## Decisions Made

- `_PIPELINE_MODES` defined inside `with tabs[1]:` block as specified (not module-level) — keeps scope clean and consistent with plan instructions
- `mode_str` derived from `session_state.get("p_mode", "LLM-guided (recommended)")` at top of tab (before widget renders) — necessary so the expander can use the correct mode on re-runs; defaults to `"llm"` on first page load
- `use_llm = mode_str != "step"` is correct for all four modes: `"llm"`, `"nanobody_llm"`, `"binding_affinity"` all show LLM Settings; `"step"` hides them

## Deviations from Plan

None - plan executed exactly as written. Task 1 commit pre-existed from prior session (7e71e82, dated 2026-02-23); all five edits verified present and correct.

## Issues Encountered

None.

## Checkpoint Status

Task 2 (human-verify) gate is ACTIVE. Human must confirm the four pipeline modes appear and behave correctly in the Streamlit UI before this plan can be marked complete.

## User Setup Required

Start Streamlit: `streamlit run src/protein_design_hub/web/Home.py` and navigate to Agent Pipeline page, Tab 1 (Run Pipeline).

## Self-Check

## Self-Check: PASSED

- `src/protein_design_hub/web/pages/11_agents.py` - FOUND
- Commit `7e71e82` - FOUND (feat(06-02): replace binary pipeline mode selector)
- All syntax and assertion checks passed

## Next Phase Readiness

- Task 2 (human-verify) pending user approval
- Once approved: plan is complete; Phase 6 Plan 3 (if any) can proceed

---
*Phase: 06-new-agent-workflows*
*Completed: 2026-02-25 (partial - awaiting checkpoint)*
