---
phase: 03-performance-and-reliability
plan: "01"
subsystem: agents
tags: [mutagenesis, openstructure, inspect, version-gate, position-cap]

# Dependency graph
requires:
  - phase: 02-mutagenesis-workflow-integrity
    provides: mutagenesis_agents.py with MutationExecutionAgent and _build_scanner

provides:
  - Import-time version gate via _check_scanner_api() using inspect.signature
  - OST position-count cap guard auto-disabling OST when >3 distinct positions approved
  - run_ost parameter on _build_scanner() replacing silent try/except TypeError fallback

affects:
  - 03-performance-and-reliability
  - 04-integration-testing

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Import-time API contract verification using inspect.signature (fail fast at import, not at runtime)"
    - "Context.extra-based feature flag pattern: ost_force_override / ost_auto_disabled_reason"

key-files:
  created: []
  modified:
    - src/protein_design_hub/agents/mutagenesis_agents.py

key-decisions:
  - "_check_scanner_api() defined as a function (not bare module-level code) to avoid circular import issues at module scope"
  - "_check_scanner_api() called at module level so bad MutationScanner installs raise ImportError on first import, not on first pipeline run"
  - "OST cap threshold set at 3 distinct positions (not mutations) — a saturation scan at 4 positions would be ~76 OST calls; capping prevents multi-hour hangs"
  - "Force override stored in context.extra['ost_force_override'] (not a constructor param) so UI checkbox can set it without rebuilding context"
  - "ost_auto_disabled_reason stored as human-readable string in context.extra for UI display in later plans"

patterns-established:
  - "Fail-fast pattern: verify external API contract at import time, not at first call"
  - "Auto-disable pattern: expensive optional features guarded by position count with context.extra override escape hatch"

requirements-completed: [PERF-01, PERF-03]

# Metrics
duration: 2min
completed: 2026-02-21
---

# Phase 3 Plan 01: Mutagenesis Backend Fixes Summary

**Import-time MutationScanner API version gate via inspect.signature and OST auto-disable guard when >3 distinct positions approved**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-21T22:26:15Z
- **Completed:** 2026-02-21T22:27:30Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Replaced silent `try/except TypeError` version fallback in `_build_scanner()` with an import-time `inspect.signature` check in `_check_scanner_api()` — wrong MutationScanner installs now fail immediately with a clear upgrade message
- Added `run_ost: bool = True` parameter to `_build_scanner()` enabling callers to disable OST without reconstructing the scanner
- Added PERF-01 guard in `MutationExecutionAgent.run()` that counts distinct residue positions and auto-disables OST when >3 positions, storing a human-readable reason in `context.extra['ost_auto_disabled_reason']`

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace try/except TypeError with inspect.signature version gate** - `e19a197` (feat)
2. **Task 2: OST position-count cap guard** - `e19a197` (feat, same commit — both tasks modify the same file and were staged together)

## Files Created/Modified
- `src/protein_design_hub/agents/mutagenesis_agents.py` - Added `import inspect`, `_check_scanner_api()`, module-level call, rewrote `_build_scanner()` with `run_ost` param, added OST cap guard in `MutationExecutionAgent.run()`

## Decisions Made
- `_check_scanner_api()` is a function (not bare module code) to avoid potential circular import issues when the check runs at import time
- OST cap threshold is 3 positions: >3 triggers auto-disable, <=3 enables OST as before, `ost_force_override=True` bypasses the cap entirely
- The reason string is stored in `context.extra['ost_auto_disabled_reason']` (not just logged) so downstream UI pages can surface it to the user without re-deriving the logic

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The second `git commit` attempt showed "nothing to commit" because both task changes were already captured in the first commit (both tasks modify `mutagenesis_agents.py`; all edits were staged before the first commit ran).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `_check_scanner_api()` and `_build_scanner(run_ost=...)` are ready for use by 03-02 (UI checkbox for Force OST)
- `context.extra['ost_auto_disabled_reason']` is populated and available for Phase 3 UI plans to surface to the user
- All 18 pipeline integrity tests pass

---
*Phase: 03-performance-and-reliability*
*Completed: 2026-02-21*
