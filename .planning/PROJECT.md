# Protein Design Hub

## What This Is

An integrated computational platform for end-to-end protein design. Researchers input a sequence and get structure predictions from multiple predictors, 15+ evaluation metrics, LLM-guided expert analysis via 10 scientist personas, and a mutagenesis workflow — all through a 12-page Streamlit web UI and CLI. The platform runs locally on a GPU workstation using Ollama, with cloud LLM fallbacks.

## Core Value

A reliable, end-to-end protein design workflow where a researcher goes from sequence → structure → expert analysis → mutagenesis → report without data loss, silent failures, or manual workarounds.

## Requirements

### Validated

<!-- Inferred from existing codebase (.planning/codebase/) -->

- ✓ Multi-predictor structure prediction (ColabFold, Chai-1, Boltz-2, ESMFold, ESM3) — existing
- ✓ 15+ evaluation metrics (lDDT, TM-score, RMSD, QS-score, VoroMQA, CAD-score, OpenMM GBSA, SASA, clash) — existing
- ✓ 5-step computational pipeline (Input → Prediction → Evaluation → Comparison → Report) — existing
- ✓ 12-step LLM-guided pipeline with 10 scientist personas and 7 team presets — existing
- ✓ Mutagenesis workflow: Phase 1 (baseline review + mutation suggestions) — existing
- ✓ Mutagenesis workflow: Phase 2 (mutation execution + comparison + report) — existing
- ✓ Saturation mutagenesis scanning with OpenStructure per-mutant scoring — existing
- ✓ 12-page Streamlit web UI (design, predict, evaluate, compare, evolution, batch, settings, MSA, MPNN, jobs, mutation scanner, agents) — existing
- ✓ CLI interface with pipeline and agent commands — existing
- ✓ Ollama local GPU inference (qwen2.5:14b default) + 10+ cloud LLM backends — existing
- ✓ Protein design tools: ProteinMPNN, RFdiffusion, ESMif — existing
- ✓ LLM client caching and GPU TTL caching for performance — existing

### Active

<!-- Stabilization -->

- [ ] Phase 1→2 mutagenesis approval gate enforced (block Phase 2 if no explicit approval)
- [ ] Phase 1 approval state persisted to disk (survives browser close/restart)
- [ ] LLM pipeline reliability: agent failures surface clearly, no silent bad outputs
- [ ] OST scoring made optional with position cap (≤3 positions default, warn/disable above)
- [ ] `mutagenesis_agents.py` committed to git with registry/orchestrator integration verified
- [ ] `.bak` file deleted and mutation scanner refactor completed cleanly
- [ ] `MutagenesiReportAgent` class name typo fixed (`MutagenesisPipelineReportAgent`)
- [ ] Per-expert backend override flow documented and verified end-to-end
- [ ] Phase 1→2 integration test (mock LLM, real MutationScanner, small sequence)
- [ ] Unit tests for `_parse_approved_mutations()`, mutation execution failure modes
- [ ] LLMMutationSuggestionAgent JSON parsing fallback tested

<!-- Extensions (in order) -->

- [ ] Mutation ranking charts and OST metric visualization in web UI
- [ ] PDF/HTML report export with figures and metric tables
- [ ] New agent workflows: antibody design pipeline, binding affinity analysis
- [ ] Async job queue: run multiple proteins in background with progress tracking
- [ ] New structure predictors: AlphaFold3, OpenFold integration

### Out of Scope

- Mobile app — web-first, GPU workstation use case
- Real-time multi-user collaboration — single researcher tool
- Cloud deployment / SaaS — local GPU is the design constraint
- Real-time chat between researcher and agents — batch pipeline model

## Context

- Hardware: RTX 4080 Laptop (12 GB VRAM), i9-13900H (20 cores), 64 GB RAM
- Default LLM: qwen2.5:14b via Ollama (~9 GB, fits 12 GB VRAM)
- Cloud LLM fallbacks: Groq (300+ tok/s), Cerebras (450+ tok/s), SambaNova, DeepSeek, OpenAI, Gemini
- Current state: core pipeline solid, mutagenesis workflow recently added (Feb 2026) but has known gaps — approval gate, session persistence, test coverage
- Codebase map: `.planning/codebase/` (7 documents, mapped 2026-02-21)

## Constraints

- **Tech stack**: Python 3.10+, Streamlit, existing evaluation framework — no rewrites
- **GPU memory**: 12 GB VRAM — predictor selection and OST batch sizes must stay within budget
- **LLM API**: OpenAI-compatible throughout (works with Ollama and all cloud backends)
- **Backwards compatibility**: Existing job directories and context formats must remain readable

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| qwen2.5:14b as default LLM | Best quality/fit for 12 GB VRAM; far superior to llama3.2:3b | ✓ Good |
| OpenAI-compatible API for all LLM backends | Single client works with Ollama + 10 cloud providers | ✓ Good |
| Streamlit for web UI | Rapid iteration for research tools; 12 pages already built | ✓ Good |
| Phase 1/Phase 2 split for mutagenesis | Allows user review before executing expensive mutations | — Pending (approval gate not yet enforced) |
| OST comprehensive scoring per mutant | High-quality structural comparison but CPU-bound and slow | ⚠️ Revisit (needs cap/optional flag) |

---
*Last updated: 2026-02-21 after initialization*
