# Architecture

**Analysis Date:** 2026-02-21

## Pattern Overview

**Overall:** Multi-agent orchestrator pattern with shared workflow context

**Key Characteristics:**
- Sequential agent pipeline (5 computational steps or up to 12 with LLM reviews)
- LLM-guided workflow: each computational step can be preceded/followed by LLM team meetings
- Virtual-Lab inspired: scientist personas + round-robin discussion + verdict tracking
- Shared mutable context (`WorkflowContext`) passed between agents
- Policy-based halt-on-failure with explicit override capability

## Layers

**Agent Layer:**
- Purpose: Execute specific computational or LLM tasks in the pipeline
- Location: `src/protein_design_hub/agents/`
- Contains: Base interface, 5 step agents, LLM review agents, mutagenesis agents
- Depends on: `WorkflowContext`, step-specific modules (predictors, evaluation, etc.)
- Used by: `AgentOrchestrator`

**Orchestration Layer:**
- Purpose: Coordinate agent execution and context threading
- Location: `src/protein_design_hub/agents/orchestrator.py`
- Contains: `AgentOrchestrator` (runs agent sequences), pipeline builders
- Depends on: Agent implementations, `WorkflowContext`
- Used by: CLI, Web UI, tests

**Meeting/Discussion Layer:**
- Purpose: Facilitate LLM team deliberation and individual review cycles
- Location: `src/protein_design_hub/agents/meeting.py`
- Contains: Meeting runner, LLM client caching, discussion persistence
- Depends on: `LLMAgent` definitions, prompt builders, OpenAI client
- Used by: LLM-guided agents

**Computational Layer:**
- Purpose: Encapsulate domain logic (prediction, evaluation, comparison)
- Location: `src/protein_design_hub/{predictors,evaluation,comparison,etc.}`
- Contains: Specific predictor wrappers, metrics evaluators, ranking logic
- Depends on: External tools (ColabFold, Chai-1, Boltz-2, etc.) via subprocess/APIs
- Used by: Step agents

**Configuration Layer:**
- Purpose: Centralize settings and enable runtime customization
- Location: `src/protein_design_hub/core/config.py`, `config/default.yaml`
- Contains: Pydantic models for all subsystems, provider presets
- Depends on: YAML files, environment variables
- Used by: All agents, web UI, CLI

**Web UI Layer:**
- Purpose: Provide Streamlit-based interactive frontend
- Location: `src/protein_design_hub/web/`
- Contains: 12+ pages for design, prediction, evaluation, settings, jobs
- Depends on: Orchestrator, agents, settings
- Used by: End users (web browser)

## Data Flow

**Main Pipeline (step-only, 5 agents):**

1. **Input Agent** → Parse FASTA → Populate `context.sequences`, `context.prediction_input`
2. **Prediction Agent** → Run predictors → Populate `context.prediction_results`
3. **Evaluation Agent** → Score structures → Populate `context.evaluation_results`
4. **Comparison Agent** → Rank by composite score → Populate `context.comparison_result`
5. **Report Agent** → Write reports → Persist to disk

**LLM-Guided Pipeline (12 agents, interleaved meetings):**

1. **InputAgent** (computational)
2. **LLMInputReviewAgent** → Team meeting: sequence validation → Store verdict in `context.step_verdicts["input_review"]`
3. **LLMPlanningAgent** → Team meeting: planning predictors/metrics → Store verdict
4. **PredictionAgent** (computational)
5. **LLMPredictionReviewAgent** → Team meeting: review prediction quality → Store verdict
6. **EvaluationAgent** (computational)
7. **ComparisonAgent** (computational)
8. **LLMEvaluationReviewAgent** → Team meeting: interpret results → Store verdict
9. **LLMRefinementReviewAgent** → Individual meeting: refinement strategy → Store verdict
10. **LLMMutagenesisPlanningAgent** → Team meeting: mutagenesis design → Store verdict
11. **LLMReportNarrativeAgent** → Team meeting: executive summary → Store verdict
12. **ReportAgent** (computational) → Persist verdicts and summaries

**Mutagenesis Pre-Approval (Phase 1, 7 agents):**

1. InputAgent → Parse sequences
2. LLMInputReviewAgent → Validate input
3. PredictionAgent → WT baseline structures
4. EvaluationAgent → Score WT
5. ComparisonAgent → Rank predictors
6. LLMBaselineReviewAgent → Review WT in detail
7. LLMMutationSuggestionAgent → Suggest mutations

**Mutagenesis Post-Approval (Phase 2, 4 agents):**

8. MutationExecutionAgent → Generate mutants + predict
9. MutationComparisonAgent → Compare vs WT
10. LLMMutationResultsAgent → Interpret results
11. MutagenesiReportAgent → Final report

**State Management:**

- `WorkflowContext` is the single mutable source of truth
- Each agent reads relevant fields and writes results to specific fields
- `step_verdicts` dict stores structured LLM decisions (PASS/WARN/FAIL + findings)
- `extra` dict holds dynamic data: meeting summaries, mutation plans, policy logs
- Progress callbacks enable UI updates during long operations

## Key Abstractions

**WorkflowContext (`agents/context.py`):**
- Purpose: Shared container for all pipeline state
- Contains: Input/output paths, sequences, prediction results, evaluation results, comparison results, LLM verdicts, extra data
- Used by: All agents as primary interface
- Pattern: Dataclass-based, mutable, passed by reference through agent chain

**BaseAgent + AgentResult (`agents/base.py`):**
- Purpose: Standardize agent interface and result reporting
- Pattern: Abstract base class, all agents implement `run(context) -> AgentResult`
- AgentResult carries: success flag, message, updated context, optional error

**LLMAgent (`agents/llm_agent.py`):**
- Purpose: Define scientist persona with expertise, goal, role
- Pattern: Frozen dataclass, system prompt auto-generated, model configurable per agent or global
- Usage: Team leads, team members, critics in meetings

**Meeting System (`agents/meeting.py`):**
- Purpose: Run structured LLM discussions (team or individual)
- Patterns:
  - **Team meetings**: Lead + members round-robin with initial/intermediate/final prompts
  - **Individual meetings**: Agent + critic iterate with feedback loop
  - All meetings persist JSON + Markdown transcripts
  - Verdict contract: final summary ends with `VERDICT_JSON:` line for structured extraction
  - Mutation plan contract: final summary can include `MUTATION_PLAN_JSON:` for mutations

**Verdict Tracking:**
- Purpose: Record LLM decision points throughout pipeline
- Format: `context.step_verdicts[step_key] = {"status": "PASS|WARN|FAIL", "key_findings": [...], "thresholds": {...}, "actions": [...]}`
- Enforcement: Orchestrator can halt on FAIL verdict (configurable)
- Persistence: ReportAgent saves verdicts to `report/verdicts.json`

**Policy Log:**
- Purpose: Track halt/override decisions for audit trail
- Format: Timestamped events in `context.extra["policy_log"]`
- Triggers: When LLM verdict is FAIL and `allow_failed_llm_steps` config is checked

## Entry Points

**CLI Entry (agents/orchestrator.py used by CLI):**
- Location: `src/protein_design_hub/cli/commands/pipeline.py`, `agents.py`
- Triggers: User runs `python -m protein_design_hub pipeline ...` or `...agents ...`
- Responsibilities: Parse CLI args, instantiate orchestrator, call `run()` or `run_with_context()`

**Web Entry (agents/orchestrator.py used by Streamlit):**
- Location: `src/protein_design_hub/web/pages/0_design.py`, `11_agents.py`
- Triggers: User interacts with design/agents page, clicks "Run Pipeline"
- Responsibilities: Collect web form inputs, build context, instantiate orchestrator, stream progress via callbacks

**Programmatic Entry:**
- Location: Any external Python code importing orchestrator
- Usage: `orchestrator = AgentOrchestrator(mode="llm"); result = orchestrator.run(input_path, ...)`
- Returns: `AgentResult` with final context

## Error Handling

**Strategy:** Fail-fast with context preservation

**Patterns:**

1. **Agent Failure:**
   - Agent returns `AgentResult.fail(message, error=exception)`
   - Orchestrator halts by default (checks `stop_on_failure=True`)
   - Context is preserved in the failed result for debugging

2. **LLM Verdict Failure:**
   - LLM meeting ends with verdict `status: "FAIL"`
   - Orchestrator logs event to `policy_log`
   - If `allow_failed_llm_steps=False` (default), pipeline halts with policy violation message
   - If `allow_failed_llm_steps=True`, pipeline continues with warning

3. **LLM Client Error:**
   - `_call_llm()` wrapped in try-except at meeting level
   - GPU check via `ensure_ollama_gpu()` (60s cached) prevents hung connections
   - Timeout set to 120s on OpenAI client
   - max_retries=2 on client initialization

4. **Subprocess Error:**
   - Step agents catch exceptions from external tools (predictors, evaluators)
   - Error message and traceback saved to `context` and returned
   - Reports written with error flags for transparency

## Cross-Cutting Concerns

**Logging:**
- Per-call timing printed to stdout: `[Agent] elapsed_time, tokens, tok/s`
- Meeting summaries include discussion transcripts (JSON + Markdown)
- Policy decisions logged to `context.extra["policy_log"]`

**Validation:**
- Input sequences validated in InputAgent (FASTA parser)
- LLM verdicts parsed from structured JSON suffix
- Mutation positions validated against sequence length
- Composite score calculation checks multiple metric sources

**Authentication:**
- LLM provider configured via `Settings.llm` (reads from config or env)
- OpenAI client created with base_url + api_key (supports Ollama, vLLM, cloud providers)
- Default Ollama on localhost requires no API key

**GPU Management:**
- `ollama_gpu.py` provides cached check (60s TTL) to avoid repeated shell calls
- Per-agent model override supported (default model: `qwen2.5:14b`)
- Perf flags passed to Ollama: `num_ctx=4096, num_batch=512, keep_alive=10m`

---

*Architecture analysis: 2026-02-21*
