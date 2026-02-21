# Codebase Structure

**Analysis Date:** 2026-02-21

## Directory Layout

```
protein_design_hub/
├── src/protein_design_hub/
│   ├── agents/                 # Multi-agent orchestrator & LLM systems
│   │   ├── base.py            # BaseAgent interface, AgentResult
│   │   ├── context.py         # WorkflowContext (shared mutable state)
│   │   ├── orchestrator.py    # AgentOrchestrator, pipeline builders
│   │   ├── meeting.py         # Team/individual meetings, LLM calls
│   │   ├── llm_agent.py       # LLMAgent persona (scientist definition)
│   │   ├── scientists.py      # 10 scientist personas + teams
│   │   ├── registry.py        # Agent registry & defaults
│   │   │
│   │   ├── input_agent.py     # Step 1: Parse FASTA input
│   │   ├── prediction_agent.py # Step 2: Run structure predictors
│   │   ├── evaluation_agent.py # Step 3: Evaluate structures
│   │   ├── comparison_agent.py # Step 4: Rank & compare
│   │   ├── report_agent.py    # Step 5: Write reports
│   │   │
│   │   ├── llm_guided.py      # LLM review agents (7 classes)
│   │   ├── mutagenesis_agents.py # Mutation execution & comparison
│   │   │
│   │   ├── prompts.py         # Meeting prompt templates
│   │   ├── ollama_gpu.py      # GPU check with 60s cache
│   │
│   ├── core/
│   │   ├── config.py          # Pydantic settings (12 LLM provider presets)
│   │   ├── types.py           # Core types (Sequence, PredictionResult, etc.)
│   │
│   ├── web/                    # Streamlit frontend
│   │   ├── ui.py              # UI components & GPU detection
│   │   ├── pages/             # 12 Streamlit pages
│   │   │   ├── 0_design.py    # Main design/pipeline entry
│   │   │   ├── 1_predict.py   # Prediction interface
│   │   │   ├── 2_evaluate.py  # Structure evaluation
│   │   │   ├── 3_compare.py   # Result comparison
│   │   │   ├── 4_evolution.py # Evolutionary design
│   │   │   ├── 5_batch.py     # Batch processing
│   │   │   ├── 6_settings.py  # Configuration UI
│   │   │   ├── 7_msa.py       # MSA builder
│   │   │   ├── 8_mpnn.py      # ProteinMPNN design
│   │   │   ├── 9_jobs.py      # Job history
│   │   │   ├── 10_mutation_scanner.py # Mutagenesis workflow
│   │   │   ├── 11_agents.py   # Agent configuration UI
│   │
│   ├── predictors/             # Structure prediction integrations
│   │   ├── colabfold/         # ColabFold wrapper
│   │   ├── chai1/             # Chai-1 wrapper
│   │   ├── boltz2/            # Boltz-2 wrapper
│   │   ├── esmfold/           # ESMFold wrapper
│   │   ├── esm3/              # ESM3 wrapper
│   │
│   ├── evaluation/             # Structure evaluation metrics
│   │   ├── composite.py       # CompositeEvaluator (multi-metric)
│   │   ├── metrics/           # Individual metric implementations
│   │
│   ├── analysis/               # Structure analysis tools
│   ├── design/                 # Protein design generators
│   │   ├── generators/
│   │   ├── rfdiffusion/        # RFDiffusion wrapper
│   │   ├── proteinmpnn/        # ProteinMPNN wrapper
│   │   ├── esmif/              # ESMif wrapper
│   │
│   ├── evolution/              # Evolutionary search
│   ├── energy/                 # Energy functions (Rosetta, FoldX, etc.)
│   ├── biophysics/             # Biophysical analysis
│   ├── msa/                    # MSA generation & analysis
│   ├── io/                     # Input/output
│   │   ├── parsers/            # FASTA, PDB, etc.
│   │   ├── writers/            # Report writers
│   │
│   ├── pipeline/               # Sequential pipeline runner
│   │   ├── runner.py          # SequentialPipelineRunner
│   │
│   ├── cli/                    # Command-line interface
│   │   ├── commands/          # Subcommands (pipeline, agents, etc.)
│   │
│   ├── __init__.py
│
├── config/
│   ├── default.yaml           # Default settings
│
├── tests/
│   ├── test_*.py              # Unit & integration tests
│
├── .planning/codebase/         # This documentation
│   ├── ARCHITECTURE.md
│   ├── STRUCTURE.md
│   └── (other GSD docs)
```

## Directory Purposes

**`agents/`:**
- Purpose: Multi-agent orchestration, LLM-guided workflow, scientist personas
- Contains: BaseAgent interface, 5 step agents, 7 LLM review agents, mutagenesis agents, orchestrator
- Key files: `orchestrator.py` (pipeline builder/runner), `context.py` (shared state), `meeting.py` (LLM discussions)

**`core/`:**
- Purpose: Shared types and configuration
- Contains: Pydantic models for all subsystems, enums for molecule/metric types
- Key files: `config.py` (Settings with 12 LLM provider presets), `types.py` (Sequence, PredictionResult, etc.)

**`web/`:**
- Purpose: Streamlit-based interactive frontend
- Contains: 12+ pages for design, prediction, evaluation, mutagenesis
- Key files: `ui.py` (UI components, GPU detection), `pages/*.py` (Streamlit pages)

**`predictors/`:**
- Purpose: Wrappers around external structure prediction tools
- Contains: ColabFold, Chai-1, Boltz-2, ESMFold, ESM3 integrations
- Pattern: Each has config class in `core/config.py` and wrapper in `predictors/{name}/`

**`evaluation/`:**
- Purpose: Structure quality metrics and composite scoring
- Contains: CompositeEvaluator, individual metric classes
- Used by: EvaluationAgent, ComparisonAgent

**`design/`:**
- Purpose: Protein design generators (seq design + structure design)
- Contains: RFDiffusion, ProteinMPNN, ESMif wrappers and generators
- Used by: Design pages, mutagenesis workflow

**`pipeline/`:**
- Purpose: Sequential execution of computational steps
- Contains: SequentialPipelineRunner for running all predictors
- Used by: PredictionAgent

**`cli/`:**
- Purpose: Command-line interface entry points
- Contains: Subcommands for pipeline execution, agent configuration
- Key files: `commands/pipeline.py`, `commands/agents.py`

**`io/`:**
- Purpose: File I/O (parsing inputs, writing reports)
- Contains: FASTA parser, report writers
- Used by: InputAgent, ReportAgent

**`config/`:**
- Purpose: Default YAML configuration
- Files: `default.yaml` (loaded if no user config)
- Overridable by: Environment variables, Streamlit settings UI

**`tests/`:**
- Purpose: Unit and integration tests
- Pattern: Filenames match modules they test (e.g., `test_agent_pipeline_integrity.py`)

## Key File Locations

**Entry Points:**

- `src/protein_design_hub/cli/commands/pipeline.py`: CLI pipeline subcommand
- `src/protein_design_hub/web/pages/0_design.py`: Web UI main design page
- `src/protein_design_hub/agents/orchestrator.py`: Programmatic entry (AgentOrchestrator)

**Configuration:**

- `src/protein_design_hub/core/config.py`: Pydantic settings with 12 provider presets
- `config/default.yaml`: Default YAML configuration file
- `src/protein_design_hub/core/types.py`: Type definitions (enums, dataclasses)

**Core Logic:**

- `src/protein_design_hub/agents/context.py`: WorkflowContext (shared state)
- `src/protein_design_hub/agents/orchestrator.py`: Agent sequencing and pipeline building
- `src/protein_design_hub/agents/meeting.py`: LLM meeting execution, verdict extraction

**Step Agents:**

- `src/protein_design_hub/agents/input_agent.py`: FASTA parsing
- `src/protein_design_hub/agents/prediction_agent.py`: Run predictors
- `src/protein_design_hub/agents/evaluation_agent.py`: Score structures
- `src/protein_design_hub/agents/comparison_agent.py`: Rank and compare
- `src/protein_design_hub/agents/report_agent.py`: Generate reports

**LLM Agents:**

- `src/protein_design_hub/agents/llm_guided.py`: 7 LLM review agent classes
- `src/protein_design_hub/agents/scientists.py`: 10 scientist personas
- `src/protein_design_hub/agents/llm_agent.py`: LLMAgent dataclass definition
- `src/protein_design_hub/agents/prompts.py`: Meeting prompt templates

**Mutagenesis:**

- `src/protein_design_hub/agents/mutagenesis_agents.py`: Mutation execution & comparison
- `src/protein_design_hub/web/pages/10_mutation_scanner.py`: Mutagenesis UI

**Testing:**

- `tests/test_agent_pipeline_integrity.py`: Pipeline consistency tests
- `tests/test_web_smoke.py`: Web UI smoke tests

## Naming Conventions

**Files:**

- Snake case: `input_agent.py`, `orchestrator.py`, `default.yaml`
- Step agents: `{step}_agent.py` (e.g., `input_agent.py`, `prediction_agent.py`)
- LLM agents: `llm_guided.py` (all LLM review agents in one module)
- Web pages: `{number}_{name}.py` (e.g., `0_design.py`, `10_mutation_scanner.py`)

**Directories:**

- Plural nouns: `agents/`, `predictors/`, `evaluators/`, `metrics/`
- Domain areas: `design/`, `evolution/`, `analysis/`, `biophysics/`
- Infrastructure: `cli/`, `web/`, `io/`, `core/`, `pipeline/`

**Classes:**

- PascalCase agents: `InputAgent`, `PredictionAgent`, `LLMInputReviewAgent`
- Data classes: `WorkflowContext`, `AgentResult`, `ComparisonResult`
- Evaluators/runners: `CompositeEvaluator`, `SequentialPipelineRunner`

**Functions:**

- Snake case: `run_meeting()`, `_call_llm()`, `ensure_ollama_gpu()`
- Private helpers: Prefix with `_` (e.g., `_build_llm_guided_pipeline()`)
- Prompt builders: Prefix with verb (e.g., `team_meeting_start()`, `team_lead_initial()`)

## Where to Add New Code

**New Computational Step:**
- Implement: `src/protein_design_hub/agents/{step}_agent.py` inheriting from `BaseAgent`
- Register: Add to `AgentRegistry` in `src/protein_design_hub/agents/registry.py`
- Integrate: Add to pipeline builder in `src/protein_design_hub/agents/orchestrator.py`
- Test: Create `tests/test_{step}_agent.py`

**New LLM Review Agent:**
- Implement: Add class to `src/protein_design_hub/agents/llm_guided.py`
- Define team: Add agents to `src/protein_design_hub/agents/scientists.py` if needed
- Prompts: Add templates to `src/protein_design_hub/agents/prompts.py`
- Integrate: Add to `_build_llm_guided_pipeline()` in `orchestrator.py`

**New Web Page:**
- Create: `src/protein_design_hub/web/pages/{num}_{name}.py`
- Use: Import from `ui.py`, follow Streamlit page structure
- Session state: Use keys like `pdhub_*` prefix
- Progress: Use orchestrator's `progress_callback` parameter

**New Predictor Integration:**
- Create: `src/protein_design_hub/predictors/{name}/`
- Config: Add `{Name}Config` class to `src/protein_design_hub/core/config.py`
- Wrapper: Implement `{Name}Predictor` class
- Register: Add to `PredictorType` enum in `types.py`
- Pipeline: Register in `SequentialPipelineRunner` in `pipeline/runner.py`

**New Evaluation Metric:**
- Implement: New file or class in `src/protein_design_hub/evaluation/`
- Wrapper: Create evaluator class implementing metric interface
- Register: Add to `CompositeEvaluator` in `evaluation/composite.py`
- Config: Add `{MetricName}` to `MetricType` enum in `types.py`

**New Utility/Helper:**
- Shared helpers: `src/protein_design_hub/core/utils.py` (if it doesn't exist, create it)
- Domain-specific: Place in relevant module folder
- Convention: Use snake_case function names, add docstrings

## Special Directories

**`outputs/`:**
- Purpose: Job outputs (created at runtime)
- Structure: `outputs/{job_id}/` per job
  - `input/` - parsed sequences
  - `{predictor_name}/` - per-predictor results
  - `evaluation/` - evaluation metrics
  - `comparison_summary.json` - ranking results
  - `meetings/` - LLM discussion transcripts (JSON + Markdown)
  - `report/` - HTML report, verdicts.json, policy_log.json
- Generated: Yes (user-created at runtime)
- Committed: No (gitignored)

**`.planning/codebase/`:**
- Purpose: GSD documentation (this folder)
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md
- Generated: No (manually maintained or GSD-generated)
- Committed: Yes (documentation for future Claude instances)

**`config/`:**
- Purpose: Default configuration files
- Files: `default.yaml`
- Format: YAML, loaded by `get_settings()` at startup
- Committed: Yes (shared defaults)

**`.venv_immunebuilder/`:**
- Purpose: Python virtual environment
- Generated: Yes (pip install)
- Committed: No (gitignored)

---

*Structure analysis: 2026-02-21*
