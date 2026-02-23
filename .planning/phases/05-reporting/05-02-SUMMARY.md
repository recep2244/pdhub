---
phase: 05-reporting
plan: "02"
subsystem: ui
tags: [fpdf2, kaleido, plotly, streamlit, pdf, html, export, base64, mutation-scanner]

# Dependency graph
requires:
  - phase: 05-reporting plan 01
    provides: _build_ranking_figure and _build_plddt_figure shared figure builders; fpdf2/kaleido declared in pyproject.toml
  - phase: 02-mutagenesis-workflow-integrity
    provides: mutation_comparison dict with ranked_mutations and ost fields; ctx.extra with mutation_wt_plddt_per_residue and mutation_interpretation
provides:
  - _strip_for_pdf: strips non-latin-1 chars and markdown markers so fpdf2 Helvetica does not crash on emoji/CJK (REP-04)
  - _embed_fig_in_pdf: exports plotly figure to PNG temp file with try/finally cleanup preventing /tmp accumulation (REP-04)
  - _build_report_pdf: builds complete bytes PDF with title, ranking chart, pLDDT chart, OST table (capped 20 rows), narrative (REP-04)
  - _build_report_html: builds self-contained HTML string with base64 PNG data URIs, html.escape on narrative (REP-05)
  - Export PDF and Export HTML download buttons wired into _render_phase2_results with session_state caching (REP-04, REP-05)
affects:
  - Future plans using _build_report_pdf or _build_report_html

# Tech tracking
tech-stack:
  added:
    - fpdf2 (already declared in pyproject.toml by plan 01; now actively imported and used)
    - kaleido (already declared in pyproject.toml by plan 01; now used via fig.to_image())
    - base64 stdlib (base64-embedding PNG into HTML data URIs)
    - re stdlib (non-latin-1 character stripping)
  patterns:
    - try/finally temp file cleanup: _embed_fig_in_pdf writes PNG to NamedTemporaryFile then calls os.unlink in finally block — prevents /tmp accumulation on exceptions
    - session_state caching with ctx_key invalidation: cache keyed to id(ctx), stale cache cleared when ctx changes
    - html.escape() for LLM output: prevents malformed HTML from angle brackets/ampersands in LLM narrative
    - bytes(pdf.output()) conversion: fpdf2 output() returns bytearray; st.download_button requires bytes

key-files:
  created: []
  modified:
    - src/protein_design_hub/web/pages/10_mutation_scanner.py

key-decisions:
  - "bytes(pdf.output()) conversion required — fpdf2 FPDF.output() returns bytearray, but st.download_button requires bytes type"
  - "_embed_fig_in_pdf uses try/finally around os.unlink to guarantee temp file removal even when pdf.image() raises an exception"
  - "session_state cache uses id(ctx) as invalidation key — different ctx object on page reload clears stale cached bytes"
  - "html.escape() applied to LLM narrative in both HTML function (via html_mod.escape) and stripped in PDF (via _strip_for_pdf)"
  - "_build_report_html imports html as html_mod inside function to avoid shadowing any outer html name"
  - "OST table capped at 20 rows in both PDF and HTML to prevent slowness with large mutation sets"

patterns-established:
  - "Export caching pattern: st.session_state[cached_X] checked before building; ctx id invalidation clears stale cache"
  - "Character safety pattern: PDF uses _strip_for_pdf (latin-1 only), HTML uses html.escape (XSS-safe)"

requirements-completed: [REP-04, REP-05]

# Metrics
duration: 3min
completed: 2026-02-23
---

# Phase 05 Plan 02: Mutation Scanner PDF and HTML Export Summary

**PDF (fpdf2) and self-contained HTML export added to mutation scanner Phase 2 results panel with base64-embedded charts, OST table, narrative, and session_state caching to prevent rebuild on Streamlit re-runs**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-23T15:40:13Z
- **Completed:** 2026-02-23T15:42:41Z
- **Tasks:** 2 auto tasks + 1 checkpoint (awaiting human verification)
- **Files modified:** 1

## Accomplishments
- Added `_strip_for_pdf` to sanitize fpdf2-incompatible characters (emoji, CJK, >U+00FF) and markdown markers before PDF embedding
- Added `_embed_fig_in_pdf` with try/finally temp file cleanup — prevents /tmp orphan files when kaleido or fpdf2 raises an exception mid-export
- Added `_build_report_pdf` returning `bytes` (not bytearray) with title, ranking chart PNG, pLDDT chart PNG (when WT data present), OST table (when OST enabled, capped 20 rows), and narrative text (stripped, 3000 char limit)
- Added `_build_report_html` returning a self-contained HTML string with base64 data URI PNGs, `html.escape()` on LLM narrative, inline CSS — no CDN or internet required
- Wired "Export PDF" and "Export HTML" `st.download_button` widgets into `_render_phase2_results` with `st.session_state` caching keyed to `id(ctx)` for invalidation

## Task Commits

Each task was committed atomically:

1. **Task 1: Add PDF and HTML export builder functions** - `92a894e` (feat)
2. **Task 2: Wire PDF and HTML export download buttons into _render_phase2_results** - `fa4a20c` (feat)

## Files Created/Modified
- `src/protein_design_hub/web/pages/10_mutation_scanner.py` - Four new export functions added (230 insertions); two download buttons wired into `_render_phase2_results` with session_state caching (39 insertions); three new imports at top of file (base64, re, from fpdf import FPDF)

## Decisions Made
- `bytes(pdf.output())` conversion is mandatory — fpdf2's `FPDF.output()` returns `bytearray`, but `st.download_button`'s `data` parameter requires `bytes`. Omitting the conversion causes a Streamlit type error at runtime.
- `_embed_fig_in_pdf` uses `try/finally` around `os.unlink` so temp PNG files are deleted even when `pdf.image()` raises (e.g., kaleido not installed, corrupt PNG). Without try/finally, each failed export leaks a file in `/tmp`.
- Session state caching uses `id(ctx)` as the invalidation key. When the user loads a new run, a new `ctx` object is created with a different identity, which triggers cache eviction.
- `_build_report_html` imports `html` as `html_mod` inside the function body to avoid any risk of shadowing the stdlib `html` name at module scope.
- OST table is capped at 20 rows in both PDF and HTML to prevent slowness. A full saturation scan can produce hundreds of mutants.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None - both tasks executed cleanly on first attempt.

## User Setup Required
None - fpdf2 and kaleido were already declared in pyproject.toml by Plan 01. No additional setup required.

## Next Phase Readiness
- PDF and HTML export complete — awaiting human verification (checkpoint) to confirm downloads work end-to-end in running Streamlit app
- After checkpoint approval, Phase 05 reporting is complete (REP-01 through REP-05 satisfied)
- Phase 06 can proceed once checkpoint is cleared

## Self-Check: PASSED
- FOUND: src/protein_design_hub/web/pages/10_mutation_scanner.py
- FOUND: .planning/phases/05-reporting/05-02-SUMMARY.md
- FOUND: commit 92a894e (feat(05-02): add PDF and HTML export builder functions)
- FOUND: commit fa4a20c (feat(05-02): wire PDF and HTML export download buttons)

---
*Phase: 05-reporting*
*Completed: 2026-02-23*
