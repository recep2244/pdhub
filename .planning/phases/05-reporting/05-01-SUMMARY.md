---
phase: 05-reporting
plan: "01"
subsystem: ui
tags: [plotly, streamlit, charts, plddt, ost, mutation-ranking, fpdf2, kaleido]

# Dependency graph
requires:
  - phase: 02-mutagenesis-workflow-integrity
    provides: mutation_comparison dict with ranked_mutations, ost_lddt/ost_rmsd_ca/ost_qs_score per mutant
  - phase: 03-performance-and-reliability
    provides: OST cap/override logic writing ost fields into ranked_mutations
provides:
  - _build_ranking_figure: reusable Plotly bar chart of mutation improvement scores (REP-01)
  - _build_plddt_figure: reusable Plotly line chart WT vs top-3 mutant pLDDT traces (REP-02)
  - _render_ranking_chart: Streamlit wrapper for REP-01
  - _render_plddt_chart: Streamlit wrapper for REP-02
  - _render_ost_table: conditional OST lDDT/RMSD/QS-score table (REP-03)
  - fpdf2 and kaleido declared in pyproject.toml for Plan 02 PDF/HTML export
affects:
  - 05-reporting plan 02 (PDF/HTML export uses _build_ranking_figure and _build_plddt_figure directly)

# Tech tracking
tech-stack:
  added:
    - fpdf2>=2.8.0 (PDF export, declared in pyproject.toml)
    - kaleido>=1.2.0 (chart image rendering for export, declared in pyproject.toml)
  patterns:
    - Shared figure builder pattern: _build_* returns go.Figure reusable by both display and export
    - Guard pattern: OST table silently skips when no ost_lddt present (OST was disabled)
    - Guard pattern: pLDDT chart skipped with info message when mutation_wt_plddt_per_residue absent
    - Category thresholds: beneficial >0 (#22c55e), detrimental <-0.5 (#ef4444), neutral (#9ca3af)

key-files:
  created: []
  modified:
    - src/protein_design_hub/web/pages/10_mutation_scanner.py
    - pyproject.toml

key-decisions:
  - "_build_ranking_figure uses single go.Bar trace with per-bar marker_color list (NOT separate traces per category) to preserve sorted x-axis order"
  - "OST table reads ost_lddt directly from ranked mutation dicts, not from ctx.extra['ost_metrics'] (that key does not exist)"
  - "pLDDT chart guard uses mutation_wt_plddt_per_residue from ctx.extra; info message shown when absent rather than silent skip"
  - "_build_* functions are module-level shared builders, making them available to Plan 02 export without duplication"

patterns-established:
  - "Shared figure builder pattern: _build_X returns go.Figure; _render_X wraps it for Streamlit display; export functions call _build_X directly"
  - "OST conditional rendering: guard with r.get('ost_lddt') is not None check, silent skip when OST was disabled"

requirements-completed: [REP-01, REP-02, REP-03]

# Metrics
duration: 5min
completed: 2026-02-23
---

# Phase 05 Plan 01: Mutation Scanner Visual Charts Summary

**Plotly bar chart (improvement scores), pLDDT line chart (WT vs top-3 mutants), and OST structural metrics table added to Phase 2 results panel with shared figure builders reusable by Plan 02 export**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-23T05:13:25Z
- **Completed:** 2026-02-23T05:18:27Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `_build_ranking_figure` and `_build_plddt_figure` as shared Plotly figure builders — both display and export (Plan 02) use these same functions without duplication
- Added `_render_ranking_chart`, `_render_plddt_chart`, `_render_ost_table` wired into `_render_phase2_results` immediately after the ranked mutations table
- Declared `fpdf2>=2.8.0` and `kaleido>=1.2.0` in `pyproject.toml` so fresh installs include them

## Task Commits

Each task was committed atomically:

1. **Task 1: Add shared figure builders and chart render helpers** - `c968a77` (feat)
2. **Task 2: Declare fpdf2 and kaleido in pyproject.toml** - `0b7c9bd` (chore)

## Files Created/Modified
- `src/protein_design_hub/web/pages/10_mutation_scanner.py` - Five new functions added (124 insertions), three calls wired into `_render_phase2_results`
- `pyproject.toml` - Two new dependency entries: fpdf2>=2.8.0, kaleido>=1.2.0

## Decisions Made
- `_build_ranking_figure` uses a single `go.Bar` trace with a per-bar `marker_color` list instead of separate traces per category. Separate traces would disrupt the sorted x-axis order (Plotly concatenates traces rather than interleaving them).
- OST table accesses `ost_lddt`, `ost_rmsd_ca`, `ost_qs_score` directly from each ranked mutation dict. These keys are written by `MutationComparisonAgent`; there is no `ctx.extra["ost_metrics"]` key.
- pLDDT chart uses `ctx.extra.get("mutation_wt_plddt_per_residue", [])` as the guard. When absent, an `st.info` message is shown (not silent skip) so the researcher knows the data could be present on future runs.
- `_build_*` functions are module-level (not nested) so Plan 02 export can import and call them without going through Streamlit.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `tomllib` not available on Python 3.10 (stdlib only from 3.11+). Used `tomli` package instead for TOML validation. No impact on code or functionality.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `_build_ranking_figure` and `_build_plddt_figure` are ready for Plan 02 (PDF/HTML export) to import and use directly
- `fpdf2` and `kaleido` are declared in pyproject.toml; Plan 02 can import them without further setup
- OST table rendering is guarded correctly — Plan 02 export can use the same guard pattern

---
*Phase: 05-reporting*
*Completed: 2026-02-23*
