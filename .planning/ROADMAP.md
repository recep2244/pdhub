# Roadmap: Protein Design Hub

## Overview

Starting from a working multi-predictor pipeline with a recently added mutagenesis workflow, this roadmap stabilizes the codebase and extends it into four new capability areas. The first four phases fix real correctness gaps (untracked code, missing approval enforcement, silent failures, missing tests) before any feature work begins. Phases 5-8 deliver user-visible extensions in order of dependency: reporting first (pure UI, no new infrastructure), then new agent workflows (use existing agent system), then async jobs (new infrastructure other phases depend on), then new predictors (isolated integrations).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Git & Code Health** - Commit untracked code, delete backup artifact, fix class name typo (completed 2026-02-21)
- [x] **Phase 2: Mutagenesis Workflow Integrity** - Enforce approval gate, persist state to disk, verify backend overrides (completed 2026-02-21)
- [x] **Phase 3: Performance & Reliability** - Cap OST scoring, surface silent fallbacks, fix fragile scanner init (completed 2026-02-21)
- [ ] **Phase 4: Test Coverage** - Integration test Phase 1-2 flow, unit tests for parsing and failure modes
- [ ] **Phase 5: Reporting** - Mutation ranking charts, per-residue pLDDT visualization, OST metric table, PDF/HTML export
- [ ] **Phase 6: New Agent Workflows** - Antibody design pipeline, binding affinity analysis workflow
- [ ] **Phase 7: Async Job Queue** - Background job submission, sequential multi-protein execution, live status in UI
- [ ] **Phase 8: New Predictors** - AlphaFold3 API integration, OpenFold local GPU integration

## Phase Details

### Phase 1: Git & Code Health
**Goal**: The codebase is clean, traceable, and free of maintenance hazards — every file is committed, no orphaned backups exist, and class names match their usage
**Depends on**: Nothing (first phase)
**Requirements**: GIT-01, GIT-02, GIT-03
**Success Criteria** (what must be TRUE):
  1. `git status` shows `mutagenesis_agents.py` as tracked; orchestrator and registry imports resolve without error after a fresh clone
  2. `10_mutation_scanner.py.bak` no longer exists in the repository and is not restorable from git history
  3. `MutagenesisPipelineReportAgent` is the canonical class name; old name `MutagenesiReportAgent` resolves via an import alias without breaking existing code
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Commit mutagenesis_agents.py to git and delete orphaned .bak backup (GIT-01, GIT-02)
- [ ] 01-02-PLAN.md — Rename MutagenesiReportAgent to MutagenesisPipelineReportAgent with backward-compatible alias (GIT-03)

### Phase 2: Mutagenesis Workflow Integrity
**Goal**: The Phase 1 to Phase 2 mutagenesis transition is correct — approval is enforced, state survives browser close, and backend overrides reach the agents they are intended for
**Depends on**: Phase 1
**Requirements**: MUT-01, MUT-02, MUT-03, MUT-04, MUT-05
**Success Criteria** (what must be TRUE):
  1. Clicking "Run Phase 2" without approving mutations shows a blocking error and does not execute any mutations
  2. The "Approve and Continue" button is present and must be clicked before Phase 2 can start; clicking it with no mutations selected does nothing
  3. When Phase 1 completes, approved mutations, suggestions, and low-confidence positions are written to the job directory on disk
  4. Returning to a completed Phase 1 job (after closing the browser) loads previous results from disk without re-running Phase 1
  5. Selecting a different LLM backend in the expert panel UI causes that agent to use that backend; the change is observable in per-call timing logs
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Enforce Phase 1→2 approval gate: button rename, @st.dialog confirmation, _run_phase2() guard (MUT-01, MUT-02)
- [x] 02-02-PLAN.md — Persist Phase 1 results to disk and auto-load on page return (MUT-03, MUT-04)
- [x] 02-03-PLAN.md — Thread expert backend overrides into Phase 1 and Phase 2 orchestrator calls; add model name to timing log (MUT-05)

### Phase 3: Performance & Reliability
**Goal**: OST scoring does not silently run for hours, fallback paths announce themselves before executing, and version mismatches produce helpful errors
**Depends on**: Phase 2
**Requirements**: PERF-01, PERF-02, PERF-03
**Success Criteria** (what must be TRUE):
  1. Running mutagenesis with more than 3 mutation positions and OST enabled shows a warning and automatically disables OST comprehensive scoring; user can override with an explicit flag
  2. When LLM mutation plan parsing fails, the UI displays a clear warning ("LLM plan unparseable — falling back to saturation at N positions") before any mutations are executed
  3. Importing `mutagenesis_agents` against an incompatible MutationScanner version raises an `ImportError` with a message stating the minimum required version, not a silent `TypeError` at runtime
**Plans**: 2 plans

Plans:
- [ ] 03-01-PLAN.md — Import-time version gate (_check_scanner_api) and OST position-count cap in MutationExecutionAgent (PERF-03, PERF-01)
- [ ] 03-02-PLAN.md — Surface saturation fallback warning via context.extra and add Force OST checkbox to UI (PERF-02, PERF-01)

### Phase 4: Test Coverage
**Goal**: The mutagenesis workflow has automated tests covering the critical Phase 1 to Phase 2 transition, parsing edge cases, and agent failure modes so that refactoring cannot break the workflow silently
**Depends on**: Phase 3
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05
**Success Criteria** (what must be TRUE):
  1. `pytest` runs a Phase 1 to Phase 2 integration test using mock LLM responses and a real MutationScanner on a short sequence (under 50 residues) and passes
  2. `_parse_approved_mutations()` has unit tests that pass for: correct input, renamed columns, empty dataframe, and malformed input
  3. MutationExecutionAgent failure mode tests pass for: WT prediction fails, partial mutation failures, and all mutations fail
  4. LLMMutationSuggestionAgent has a test that passes when the `MUTATION_PLAN_JSON:` footer is missing from LLM output and confirms the saturation fallback activates
  5. A test exists that feeds a mock agent returning empty output into the LLM pipeline and confirms the pipeline surfaces an error rather than continuing silently
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md — Create test_mutagenesis_workflow.py with TEST-02 (parse_approved_mutations), TEST-04 (LLM fallback), TEST-05 (empty output reliability)
- [ ] 04-02-PLAN.md — Add TEST-03 (MutationExecutionAgent failure modes) and TEST-01 (Phase 1 to Phase 2 integration test)

### Phase 5: Reporting
**Goal**: Mutagenesis results are visually interpretable in the web UI and exportable to PDF and HTML for sharing
**Depends on**: Phase 4
**Requirements**: REP-01, REP-02, REP-03, REP-04, REP-05
**Success Criteria** (what must be TRUE):
  1. The mutation scanner UI shows a bar chart of improvement score per mutation, colored by mutation category, after Phase 2 completes
  2. The mutation scanner UI shows a per-residue pLDDT line chart comparing wildtype against the top 3 mutants
  3. The mutation scanner UI shows a table with OST metrics (lDDT, RMSD, QS-score) for each mutant when OST was enabled
  4. Clicking "Export PDF" in the mutation scanner UI downloads a PDF containing the ranking chart, pLDDT chart, OST metric table, and narrative summary
  5. Clicking "Export HTML" downloads a self-contained HTML file with the same content as the PDF export
**Plans**: TBD

### Phase 6: New Agent Workflows
**Goal**: Two new domain-specific workflows are available in the web UI using the existing agent infrastructure — antibody/nanobody design and binding affinity analysis
**Depends on**: Phase 5
**Requirements**: AGT-01, AGT-02
**Success Criteria** (what must be TRUE):
  1. The agents page in the web UI offers an "Antibody/Nanobody Design" pipeline mode that uses the nanobody team preset; running it on an antibody sequence produces a completed LLM-guided analysis report
  2. The agents page offers a "Binding Affinity Analysis" workflow that invokes the biophysicist persona against existing evaluation metrics; running it produces a binding affinity interpretation in the report
**Plans**: TBD

### Phase 7: Async Job Queue
**Goal**: Researchers can submit multiple pipeline runs without waiting for each to finish — jobs run in the background, and the jobs page shows live status without blocking
**Depends on**: Phase 6
**Requirements**: JOB-01, JOB-02, JOB-03
**Success Criteria** (what must be TRUE):
  1. Submitting a pipeline run from the design page returns a job ID immediately (within 2 seconds) and does not block the UI
  2. Submitting a second protein sequence while the first job is running adds it to a queue; the second job starts automatically when the first completes
  3. The jobs page shows each job's status (queued, running, complete, failed) and updates without requiring a page reload
**Plans**: TBD

### Phase 8: New Predictors
**Goal**: AlphaFold3 (via API) and OpenFold (local GPU) are available as predictor choices alongside the existing ColabFold, Chai-1, Boltz-2, ESMFold, and ESM3
**Depends on**: Phase 7
**Requirements**: PRED-01, PRED-02
**Success Criteria** (what must be TRUE):
  1. Selecting AlphaFold3 in the predictor settings and running a pipeline produces a structure prediction result using the AlphaFold3 API; the result appears in evaluation and comparison alongside other predictors
  2. Selecting OpenFold in the predictor settings on a machine with a compatible GPU runs a local OpenFold prediction; the result integrates into the pipeline the same way as other predictors
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Git & Code Health | 2/2 | Complete   | 2026-02-21 |
| 2. Mutagenesis Workflow Integrity | 3/3 | Complete   | 2026-02-21 |
| 3. Performance & Reliability | 2/2 | Complete    | 2026-02-21 |
| 4. Test Coverage | 1/2 | In Progress|  |
| 5. Reporting | 0/TBD | Not started | - |
| 6. New Agent Workflows | 0/TBD | Not started | - |
| 7. Async Job Queue | 0/TBD | Not started | - |
| 8. New Predictors | 0/TBD | Not started | - |
