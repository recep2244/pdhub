# Requirements: Protein Design Hub

**Defined:** 2026-02-21
**Core Value:** A reliable, end-to-end protein design workflow where a researcher goes from sequence → structure → expert analysis → mutagenesis → report without data loss, silent failures, or manual workarounds.

## v1 Requirements

### Git & Code Health

- [x] **GIT-01**: `mutagenesis_agents.py` committed to git with orchestrator/registry imports verified
- [x] **GIT-02**: `10_mutation_scanner.py.bak` deleted and confirmed safe to remove
- [x] **GIT-03**: `MutagenesiReportAgent` class renamed to `MutagenesisPipelineReportAgent` with backward-compatible import alias

### Mutagenesis Workflow

- [x] **MUT-01**: Phase 2 is blocked if no explicit user approval from Phase 1 (`approved_mutations` empty = no execution)
- [x] **MUT-02**: "Approve & Continue" button in UI enforces approval before Phase 2 runs
- [x] **MUT-03**: Phase 1 results (approved mutations, suggestions, low-confidence positions) persisted to job directory on disk when Phase 1 completes
- [x] **MUT-04**: UI can load previous Phase 1 results when returning to a job (survives browser close)
- [x] **MUT-05**: Per-expert backend overrides flow verified end-to-end (session state → orchestrator kwargs → agent constructor)

### Performance

- [x] **PERF-01**: OST comprehensive scoring made optional with a flag; disabled by default when >3 mutation positions
- [x] **PERF-02**: Silent saturation fallback (LLM plan parsing failure) surfaces a clear warning to user before executing
- [x] **PERF-03**: `_build_scanner()` TypeError fallback replaced with explicit version check and helpful ImportError message

### Testing

- [x] **TEST-01**: End-to-end integration test: Phase 1 → Phase 2 with mock LLM and real MutationScanner on short sequence
- [x] **TEST-02**: Unit test for `_parse_approved_mutations()` covering column rename, empty input, malformed input
- [x] **TEST-03**: Unit tests for MutationExecutionAgent failure modes (WT fails, partial failures, all mutations fail)
- [x] **TEST-04**: Test for LLMMutationSuggestionAgent JSON parse fallback path (missing MUTATION_PLAN_JSON footer)
- [x] **TEST-05**: LLM pipeline reliability test — agent returning bad/empty output is caught and surfaced, not silently swallowed

### Reporting

- [x] **REP-01**: Mutation ranking chart in web UI (improvement score vs mutation, colored by mutation type/category)
- [x] **REP-02**: Per-residue pLDDT visualization comparing WT vs top 3 mutants
- [x] **REP-03**: OST metric table (lDDT, RMSD, QS-score) displayed per mutant in mutation scanner UI
- [x] **REP-04**: PDF export of mutagenesis report with embedded charts and metric tables
- [x] **REP-05**: HTML export option for sharing results without PDF dependency

### New Agent Workflows

- [x] **AGT-01**: Antibody/nanobody design pipeline using existing nanobody team preset with LLM guidance
- [x] **AGT-02**: Binding affinity analysis workflow using existing biophysicist persona and evaluation metrics

### Async Jobs

- [ ] **JOB-01**: Background job queue — submit a pipeline run and receive a job ID
- [ ] **JOB-02**: Multiple proteins can be queued and run sequentially in background without blocking UI
- [ ] **JOB-03**: Job progress and status visible in web UI (jobs page) without blocking the page

### New Predictors

- [ ] **PRED-01**: AlphaFold3 integration (API-based)
- [ ] **PRED-02**: OpenFold integration (local, GPU)

## v2 Requirements

### Advanced Reporting

- **AREP-01**: OST metric weighting configurable in improvement score formula (currently hard-coded 0.6/0.4)
- **AREP-02**: Batch approval UI for reviewing many mutations at once with sortable table

### Performance Scaling

- **PSCL-01**: Parallel OST runs across mutation batch (currently serial)
- **PSCL-02**: Real-time job log streaming in web UI

### UI

- **UI-01**: Mobile-responsive Streamlit layout

## Out of Scope

| Feature | Reason |
|---------|--------|
| Cloud deployment / SaaS | Local GPU workstation use case; network latency would hurt predictor performance |
| Multi-user collaboration | Single researcher tool; no auth system planned |
| Real-time chat with agents | Batch pipeline model; interactive chat would require different architecture |
| Mobile app | Web-first; GPU workstation primary environment |
| OST parallel runs (v1) | Too complex for stabilization phase; deferred to v2 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| GIT-01 | Phase 1 | Complete |
| GIT-02 | Phase 1 | Complete |
| GIT-03 | Phase 1 | Complete |
| MUT-01 | Phase 2 | Complete |
| MUT-02 | Phase 2 | Complete |
| MUT-03 | Phase 2 | Complete |
| MUT-04 | Phase 2 | Complete |
| MUT-05 | Phase 2 | Complete |
| PERF-01 | Phase 3 | Complete |
| PERF-02 | Phase 3 | Complete |
| PERF-03 | Phase 3 | Complete |
| TEST-01 | Phase 4 | Complete |
| TEST-02 | Phase 4 | Complete |
| TEST-03 | Phase 4 | Complete |
| TEST-04 | Phase 4 | Complete |
| TEST-05 | Phase 4 | Complete |
| REP-01 | Phase 5 | Complete |
| REP-02 | Phase 5 | Complete |
| REP-03 | Phase 5 | Complete |
| REP-04 | Phase 5 | Complete |
| REP-05 | Phase 5 | Complete |
| AGT-01 | Phase 6 | Complete |
| AGT-02 | Phase 6 | Complete |
| JOB-01 | Phase 7 | Pending |
| JOB-02 | Phase 7 | Pending |
| JOB-03 | Phase 7 | Pending |
| PRED-01 | Phase 8 | Pending |
| PRED-02 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 28 total
- Mapped to phases: 28
- Unmapped: 0

---
*Requirements defined: 2026-02-21*
*Last updated: 2026-02-21 after roadmap creation*
