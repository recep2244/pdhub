---
phase: 04-test-coverage
plan: "01"
subsystem: testing
tags: [pytest, importlib, streamlit-mock, mutagenesis, unit-tests]

# Dependency graph
requires:
  - phase: 03-performance-and-reliability
    provides: LLMMutationSuggestionAgent fallback path (context.extra warning) + MutationExecutionAgent empty-input guard
  - phase: 02-mutagenesis-workflow-integrity
    provides: _parse_approved_mutations() in 10_mutation_scanner.py + MutationExecutionAgent behaviour
provides:
  - TEST-02: _parse_approved_mutations() parser unit tests (4 cases)
  - TEST-04: LLMMutationSuggestionAgent JSON parse fallback test
  - TEST-05: Empty agent output surfaces error rather than silently continuing
  - Importlib loading pattern for digit-prefixed Streamlit pages in tests
affects:
  - 04-02 (Plan 02 can reuse _load_scanner_page helper and add TEST-01/TEST-03)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Streamlit page importlib pattern: patch UI helpers + use real streamlit with AttrDict session_state"
    - "patch.object on agent instance methods to bypass LLM calls in unit tests"
    - "_EmptyOutputAgent stub pattern for testing downstream failure detection"

key-files:
  created:
    - tests/test_mutagenesis_workflow.py
  modified: []

key-decisions:
  - "_load_scanner_page uses patch() for ui helpers (inject_base_css, sidebar_nav, etc.) + real streamlit with AttrDict session_state; plan's pure MagicMock approach for sys.modules['streamlit'] failed because ui.py/agent_helpers.py hold their own real st references at import time"
  - "AttrDict(dict) subclass supports both st.session_state['key'] and st.session_state.key = val access patterns that 10_mutation_scanner.py uses at module level"
  - "All 6 tests written in single Task 1 commit since Task 1 and Task 2 are logically sequential file writes with no intermediate verification needed"

patterns-established:
  - "Importlib + patch pattern: load Streamlit pages in tests without a running Streamlit server"
  - "AgentOrchestrator(agents=[stub, real_agent], stop_on_failure=True) for testing downstream failure propagation"

requirements-completed: [TEST-02, TEST-04, TEST-05]

# Metrics
duration: 5min
completed: 2026-02-22
---

# Phase 4 Plan 01: Mutagenesis Workflow Unit Tests Summary

**6 behavioural unit tests covering _parse_approved_mutations() parser correctness, LLMMutationSuggestionAgent JSON fallback warning, and MutationExecutionAgent empty-input failure — all passing, 52 total tests green**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-22T02:31:57Z
- **Completed:** 2026-02-22T02:37:00Z
- **Tasks:** 2 (both written in first commit; second commit was empty — no changes staged)
- **Files modified:** 1

## Accomplishments
- Established importlib loading pattern for `10_mutation_scanner.py` that works in headless test environment without a running Streamlit server
- TEST-02 (4 cases): `_parse_approved_mutations()` contracts documented — correct input, renamed columns (KeyError/empty), empty DataFrame, malformed Target AAs
- TEST-04 (1 case): `LLMMutationSuggestionAgent` sets `mutation_suggestion_warning` in `context.extra` when LLM output lacks `MUTATION_PLAN_JSON` footer
- TEST-05 (1 case): `MutationExecutionAgent` returns failure with descriptive message when `approved_mutations=[]` and `baseline_low_confidence_positions=[]`
- No regressions: 18 existing pipeline integrity tests + all other tests still pass (52 total)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test file with importlib helper and TEST-02 unit tests** - `c8661d1` (test)
2. **Task 2: TEST-04 and TEST-05** - included in c8661d1 (all 6 tests written together)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `tests/test_mutagenesis_workflow.py` - 234 lines; contains `_load_scanner_page` helper, `_AttrDict` session_state class, TEST-02 x4, TEST-04 x1, TEST-05 x1

## Decisions Made
- `_load_scanner_page` uses `patch()` for specific UI helper functions (inject_base_css, sidebar_nav, sidebar_system_status, page_header, section_header, workflow_breadcrumb, agent_sidebar_status) plus real streamlit with `_AttrDict` session_state. Plan's original `MagicMock()` approach for `sys.modules["streamlit"]` fails because `ui.py` and `agent_helpers.py` import streamlit at their own module level and hold real references that are not replaced by patching sys.modules after the fact.
- `_AttrDict` subclass of `dict` provides both `dict.get()` semantics and attribute-style `st.session_state.foo = bar` access that the scanner page uses at module level (line 503+).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Replaced pure MagicMock streamlit approach with patch-based UI helper mocking**
- **Found during:** Task 1 (creating `_load_scanner_page` helper)
- **Issue:** Plan specified `MagicMock()` for `sys.modules["streamlit"]`, but `ui.py` and `agent_helpers.py` import streamlit at their own module level and retain the real `st` reference. When they call `st.sidebar.columns([0.08, 0.92])`, the real streamlit's `columns()` returns something unpacked as 2 values, but with MagicMock the return value cannot be unpacked as `col1, col2 = ...`. Attempted fix with `MagicMock.columns = lambda *a, **k: [MagicMock(), MagicMock()]` hit a second wall: `get_gpu_status_html()` in ui.py uses f-string formatting on MagicMock values raising `TypeError: unsupported format string`.
- **Fix:** Use real streamlit, patch the 7 specific module-level UI helper calls that cause side effects, and replace `st.session_state` with `_AttrDict` to support attribute-style writes.
- **Files modified:** `tests/test_mutagenesis_workflow.py` (written from scratch with correct approach)
- **Verification:** `python -m pytest tests/test_mutagenesis_workflow.py -v` — 6 passed
- **Committed in:** c8661d1 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (blocking — import infrastructure)
**Impact on plan:** Fix is necessary for correct test infrastructure. No functional scope creep; all required test behaviours are validated as specified.

## Issues Encountered
- None beyond the deviation above (cleanly resolved).

## User Setup Required
None - no external service configuration required. All tests run headlessly with no network calls.

## Next Phase Readiness
- `_load_scanner_page()` helper and `_AttrDict` class in `test_mutagenesis_workflow.py` are ready for Plan 04-02 to add TEST-01 and TEST-03 tests
- Plan 02 context should note: import via `from tests.test_mutagenesis_workflow import _SCANNER_PAGE` or re-call `_load_scanner_page()` — both work
- All 52 existing tests green; new test file established as canonical mutagenesis test location

---
*Phase: 04-test-coverage*
*Completed: 2026-02-22*

## Self-Check: PASSED
