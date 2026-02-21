# Phase 1: Git & Code Health - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix three concrete maintenance hazards: commit the untracked `mutagenesis_agents.py`, delete the orphaned `.bak` backup file, and rename the class name typo. No new features. No refactoring beyond the rename. This phase is done when git is clean, the backup is gone, and the canonical name is correct everywhere.

</domain>

<decisions>
## Implementation Decisions

### Pre-commit import audit
- Audit ALL import paths before committing mutagenesis_agents.py: orchestrator, registry, CLI, and web pages
- After committing, run the existing pipeline integrity tests (10-step and 12-step flows) to confirm nothing broke

### Backup file handling
- Diff `.bak` against current `10_mutation_scanner.py` before deleting — confirm nothing was accidentally lost in the refactor
- Add `*.bak` to `.gitignore` after deletion to prevent future occurrences

### Rename compatibility
- Rename `MutagenesiReportAgent` → `MutagenesisPipelineReportAgent` as the canonical class name
- Add import alias in same file: `MutagenesiReportAgent = MutagenesisPipelineReportAgent` (no deprecation warning — silent alias is sufficient)
- Update ALL call sites: orchestrator, registry, tests, web pages — use canonical name everywhere
- Add an import test confirming both old and new names are importable after the rename

### Claude's Discretion
- Commit message wording and granularity (one commit per fix vs combined)
- Exact diff tool/approach for comparing .bak vs current

</decisions>

<specifics>
## Specific Ideas

- No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-git-and-code-health*
*Context gathered: 2026-02-21*
