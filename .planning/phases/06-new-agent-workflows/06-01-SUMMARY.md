---
phase: 06-new-agent-workflows
plan: "01"
subsystem: agents
tags: [orchestrator, pipeline, nanobody, binding-affinity, llm-agents]

# Dependency graph
requires:
  - phase: 05-reporting
    provides: ReportAgent used as the final agent in both new pipelines
provides:
  - "_build_nanobody_llm_pipeline: 12-agent pipeline with NANOBODY_TEAM_MEMBERS forced on all 7 LLM agents"
  - "_build_binding_affinity_pipeline: 6-agent pipeline with Biophysicist-led evaluation review"
  - "AgentOrchestrator modes 'nanobody_llm' and 'binding_affinity' registered and dispatchable by name"
affects:
  - 06-02-PLAN (UI plan that exposes these modes as selectable options)
  - cli/commands (any CLI caller using AgentOrchestrator by mode string)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Specialised pipeline mode pattern: builder function + elif branch in AgentOrchestrator.__init__"
    - "Forced vs setdefault team injection: nanobody uses direct assignment (team is mandatory), binding_affinity uses setdefault (caller may override)"

key-files:
  created: []
  modified:
    - src/protein_design_hub/agents/orchestrator.py

key-decisions:
  - "nanobody_llm uses direct assignment nb_kwargs['team_members'] = NANOBODY_TEAM_MEMBERS — caller overrides are intentionally ignored because the nanobody team IS the defining characteristic of this mode"
  - "binding_affinity uses setdefault for both team_lead and team_members — callers can override if needed (more flexible mode)"
  - "_build_binding_affinity_pipeline inserted just before AgentOrchestrator class (after mutagenesis_post builder) per plan's placement instruction"
  - "_build_nanobody_llm_pipeline inserted between _build_llm_guided_pipeline and _build_mutagenesis_pre_approval_pipeline per plan's placement instruction"

patterns-established:
  - "New pipeline mode pattern: add builder function at module level + elif branch in __init__ + update docstring"
  - "Imports inside builder functions (not at module top-level) to avoid circular imports — consistent with all existing builders"

requirements-completed:
  - AGT-01
  - AGT-02

# Metrics
duration: 2min
completed: 2026-02-23
---

# Phase 6 Plan 01: New Agent Workflows - Orchestrator Pipeline Modes Summary

**Two new AgentOrchestrator modes — `nanobody_llm` (12-agent, Immunologist-led) and `binding_affinity` (6-agent, Biophysicist-led) — registered in orchestrator.py with forced/setdefault team injection respectively**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-23T16:15:40Z
- **Completed:** 2026-02-23T16:18:25Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `_build_nanobody_llm_pipeline()` returning a 12-agent pipeline with `NANOBODY_TEAM_MEMBERS` (Immunologist, Structural Biologist, ML Specialist, Scientific Critic) forced onto all 7 LLM meeting agents
- Added `_build_binding_affinity_pipeline()` returning a focused 6-agent pipeline (Input, Predict, Eval, Compare, LLMEvalReview with Biophysicist + Critic, Report)
- Registered both modes via `elif` branches in `AgentOrchestrator.__init__` with updated docstring
- All 56 existing tests pass with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Add _build_nanobody_llm_pipeline builder** - `fd9c5ed` (feat)
2. **Task 2: Add _build_binding_affinity_pipeline and register both modes** - `0a99a01` (feat)

## Files Created/Modified

- `src/protein_design_hub/agents/orchestrator.py` - Added two builder functions (_build_nanobody_llm_pipeline at line ~107, _build_binding_affinity_pipeline before AgentOrchestrator class), two elif dispatch branches in __init__, updated docstring

## Decisions Made

- `nanobody_llm` uses **direct assignment** for `team_members` (not setdefault) — the Immunologist must always be present in this mode; any caller attempt to pass a different team is silently overridden by design
- `binding_affinity` uses **setdefault** for both `team_lead` and `team_members` — this is a more general pipeline where downstream callers (e.g. UI with custom expert selection) may legitimately override the default Biophysicist+Critic team
- Both builder functions use local imports (inside function body) to avoid circular imports, consistent with the existing builder pattern in this file

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Both new pipeline modes are fully wired and dispatchable by name
- Plan 06-02 (UI) can now offer `nanobody_llm` and `binding_affinity` as selectable options in the Streamlit design page
- No blockers for Plan 06-02

---
*Phase: 06-new-agent-workflows*
*Completed: 2026-02-23*
