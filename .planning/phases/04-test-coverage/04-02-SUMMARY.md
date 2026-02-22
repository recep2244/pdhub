---
phase: 04-test-coverage
plan: "02"
subsystem: testing
tags: [pytest, mutagenesis, mock, integration-test, MutationExecutionAgent, MutationScanner]

# Dependency graph
requires:
  - phase: 04-01
    provides: "tests/test_mutagenesis_workflow.py with TEST-02/TEST-04/TEST-05 (6 tests); _load_scanner_page helper; _AttrDict class"
provides:
  - "_make_mock_scanner() helper (MagicMock configured for MutationExecutionAgent expectations)"
  - "TEST-03: 3 MutationExecutionAgent failure mode tests (WT fail, all-fail, partial-fail)"
  - "TEST-01: Phase 1 to Phase 2 integration test using real MutationScanner instance with patched predict_single"
  - "10 total test functions in test_mutagenesis_workflow.py, all 5 requirement groups covered"
affects: [future mutagenesis test extensions, Phase 5+]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "patch _build_scanner at full dotted module path (protein_design_hub.agents.mutagenesis_agents._build_scanner)"
    - "Use patch.object(real_instance, method) to patch instance methods while keeping real __init__"
    - "call_count list closure for differentiating WT vs mutation calls in side_effect callables"
    - "Mock agent classes (BaseAgent subclass) for Phase 1 mock pipeline without LLM calls"

key-files:
  created: []
  modified:
    - tests/test_mutagenesis_workflow.py

key-decisions:
  - "MutationScanner imported from protein_design_hub.analysis.mutation_scanner (not mutagenesis_agents which does not re-export it)"
  - "calculate_biophysical_metrics patched directly on real_scanner instance in integration test (not via _build_scanner mock)"
  - "TEST-01 excludes MutationComparisonAgent from Phase 2 pipeline (requires additional context keys not produced by mock Phase 1)"
  - "plan template import 'from protein_design_hub.web.pages import mutation_scanner' removed - unnecessary since MutationScanner imports directly from analysis module"

patterns-established:
  - "Separation of WT call vs mutation calls via call_count closure: robust across any number of approved mutations"
  - "Real MutationScanner.__init__ exercised (no HTTP calls) + predict_single patched on instance = hermetic integration test"

requirements-completed: [TEST-01, TEST-03]

# Metrics
duration: 12min
completed: 2026-02-22
---

# Phase 4 Plan 02: MutationExecutionAgent Failure Mode Tests + Phase 1/2 Integration Summary

**TEST-03 failure modes (WT fail, all-fail, partial-fail) + TEST-01 Phase 1-to-Phase 2 integration using real MutationScanner instance with HTTP calls patched via patch.object**

## Performance

- **Duration:** 12 min
- **Started:** 2026-02-22T02:59:00Z
- **Completed:** 2026-02-22T03:11:43Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `_make_mock_scanner()` shared helper: MagicMock configured with predict_single, scan_position, calculate_biophysical_metrics return values for hermetic MutationExecutionAgent testing
- Added 3 TEST-03 tests: WT prediction failure detection, all-mutations-failure path, partial-success path
- Added TEST-01 integration test: 3 mock Phase 1 agents write approved_mutations → real MutationExecutionAgent in Phase 2 reads and produces mutation_results; MutationScanner.__init__ exercised without network calls
- 56 total tests passing across full test suite (0 regressions in test_agent_pipeline_integrity.py 18 tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add _make_mock_scanner helper and TEST-03 failure mode tests** - `ebf1788` (test)
2. **Task 2: Add TEST-01 Phase 1 to Phase 2 integration test** - `b202c6e` (test)

**Plan metadata:** (to be committed with docs)

_Note: Both tasks extend tests/test_mutagenesis_workflow.py — no new files created_

## Files Created/Modified

- `tests/test_mutagenesis_workflow.py` - Extended from 235 lines to 546 lines; 4 new test functions + _make_mock_scanner helper + 3 new mock agent classes

## Decisions Made

- **MutationScanner import path**: Plan template suggested `from protein_design_hub.agents.mutagenesis_agents import MutationScanner` but MutationScanner is not re-exported from mutagenesis_agents. Corrected to `from protein_design_hub.analysis.mutation_scanner import MutationScanner`.
- **calculate_biophysical_metrics in integration test**: Patched directly on the real_scanner instance (not via _build_scanner) since MutationExecutionAgent calls `scanner.calculate_biophysical_metrics()` after each successful targeted mutation predict_single call.
- **plan's web page import removed**: `from protein_design_hub.web.pages import mutation_scanner as _ms_mod` was in the plan template as an optional fallback but unnecessary — MutationScanner is available directly from the analysis module.
- **MutationComparisonAgent excluded from Phase 2 test pipeline**: Requires `mutation_results` already in context to function, and the mock Phase 1 doesn't produce the format it expects. The key invariant (approved_mutations → mutation_results) is fully validated by MutationExecutionAgent alone.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected MutationScanner import path in TEST-01**
- **Found during:** Task 2 (TEST-01 integration test)
- **Issue:** Plan template import `from protein_design_hub.agents.mutagenesis_agents import MutationScanner` raises ImportError — MutationScanner is not re-exported from mutagenesis_agents
- **Fix:** Changed to `from protein_design_hub.analysis.mutation_scanner import MutationScanner` (the actual module where it lives)
- **Files modified:** tests/test_mutagenesis_workflow.py
- **Verification:** `python -c "from protein_design_hub.analysis.mutation_scanner import MutationScanner; print(MutationScanner)"` returns the class; test passes
- **Committed in:** b202c6e (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in plan template)
**Impact on plan:** Single corrected import path. All required behavior is tested correctly. No scope creep.

## Issues Encountered

None — both tasks executed cleanly. The import correction was discovered via a quick test and resolved immediately.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 5 TEST requirement groups now covered across 10 test functions in test_mutagenesis_workflow.py
- Phase 4 (04-test-coverage) is now complete: both plans (04-01 and 04-02) delivered
- 56 total tests pass; test suite is stable with 0 failures
- Ready for Phase 5 when planned

---
*Phase: 04-test-coverage*
*Completed: 2026-02-22*
