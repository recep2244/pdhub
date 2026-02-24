# Protein Design Hub

Protein Design Hub is an integrated platform for protein structure prediction, evaluation, mutagenesis, and LLM-guided scientific interpretation.

It combines:
- computational predictors and quality metrics,
- a 5-step deterministic pipeline or a 12-step LLM-guided agent pipeline,
- a Streamlit UI with workflow-connected pages,
- and specialist scientist agents that review and interpret each stage.

See `AGENTS.md` for extended agent API examples.

## Core Capabilities

- Multi-predictor structure generation: ColabFold, Chai-1, Boltz-2, ESMFold variants, ESM3, and optional ImmuneBuilder flows.
- Deep structural evaluation: lDDT, TM-score, RMSD, QS-score, clash metrics, SASA/interface metrics, VoroMQA/CAD, OpenMM GBSA, and optional advanced metrics.
- Agent orchestration:
  - Step-only mode: fast compute pipeline.
  - LLM-guided mode: meetings + verdicts + policy gating.
- Mutagenesis workflow:
  - Baseline predictor benchmarking.
  - Baseline structure evaluation before mutation.
  - Expert recommendation of mutation targets.
  - Saturation and multi-mutation pipelines with optional OpenStructure comprehensive mutant-vs-baseline scoring.
- Full web workspace: Predict, Evaluate, Compare, Agents, Editor, Mutagenesis, Evolution, MPNN, Batch, MSA, Jobs, Settings.

## Pipeline Modes

### Step-only Pipeline (5 steps)

`Input -> Prediction -> Evaluation -> Comparison -> Report`

### LLM-guided Pipeline (12 steps)

`Input -> Input Review -> Planning -> Prediction -> Prediction Review -> Evaluation -> Comparison -> Evaluation Review -> Refinement Review -> Mutagenesis Planning -> Executive Summary -> Report`

LLM steps emit structured verdicts (`PASS`, `WARN`, `FAIL`) into `step_verdicts.json`.
By default, a `FAIL` verdict halts the pipeline.
Use `--allow-fail-verdicts` to continue and record an override event in `policy_log.json`.

## Scientist Agents

| Agent | Core expertise | Primary role in workflow |
|---|---|---|
| Principal Investigator | Project leadership, ML for structural biology, CASP standards | Leads meetings, aligns strategy, synthesizes decisions |
| Scientific Critic | Rigor, reproducibility, failure-mode analysis | Challenges assumptions, flags weak evidence and risks |
| Structural Biologist | 3D structure interpretation, domains, interfaces, validation | Interprets structural plausibility and functional geometry |
| Computational Biologist | Pipeline setup, MSA strategy, large-scale workflows | Tunes predictor setup and throughput/reproducibility tradeoffs |
| Machine Learning Specialist | AF/ESM/diffusion/inverse-folding model behavior | Selects and calibrates model usage and confidence interpretation |
| Immunologist | Antibody/nanobody engineering and interface biology | Guides immune-specific structure and mutation decisions |
| Protein Engineer | Stability/function engineering and mutational strategy | Proposes residue targets and mutation library strategy |
| Biophysicist | Energetics, solubility, assay planning | Interprets energy/quality metrics and experimental readiness |
| Digital Recep | Refinement methods (ReFOLD, AMBER, FastRelax, etc.) | Recommends targeted refinement plans and safeguards fold integrity |
| Liam | QA specialist (ModFOLD/ModFOLDdock/MultiFOLD/IntFOLD suite) | Performs independent quality assessment and confidence triage |

## Team Presets

| Team key | Composition focus |
|---|---|
| `default` | General prediction and evaluation |
| `design` | Rational design and engineering |
| `nanobody` | Antibody/nanobody development |
| `evaluation` | Quality and biophysical assessment |
| `refinement` | Structure refinement strategy |
| `mutagenesis` | Mutation scanning and design strategy |
| `mpnn_design` | Inverse folding / sequence design |
| `full_pipeline` | End-to-end core expert review |
| `all_experts` | Comprehensive review using all scientist personas |

## Installation

### Prerequisites

- Python `>=3.10`
- CUDA GPU recommended for predictors and local LLM speed
- Optional: Conda for OpenStructure installs

### Quick Setup

```bash
git clone https://github.com/recep2244/pdhub.git
cd pdhub

# Option A: environment file
conda env create -f environment.yaml
conda activate protein_design_hub

# Option B: existing environment
pip install -e .
```

Install predictors as needed:

```bash
pdhub install all
# or targeted
pdhub install predictor colabfold
pdhub install predictor chai1
pdhub install predictor boltz2
```

## LLM Backend (Default: Qwen on Ollama)

The default local provider is `ollama` with model `qwen2.5:14b`.

```bash
ollama pull qwen2.5:14b
ollama serve
pdhub agents status
```

GPU validation for Ollama:

```bash
ollama ps
journalctl -u ollama -n 120 --no-pager | rg -i "gpu|cuda|backend|vram"
```

## Quick Start

```bash
# System checks
pdhub status
pdhub pipeline status

# Step-only (fast)
pdhub pipeline run input.fasta

# Full LLM-guided
pdhub pipeline run input.fasta --llm

# Allow continuation even if an LLM verdict is FAIL
pdhub pipeline run input.fasta --llm --allow-fail-verdicts

# Provider/model override at runtime
pdhub pipeline run input.fasta --llm --provider deepseek
pdhub pipeline run input.fasta --llm --provider gemini --model gemini-2.5-flash

# Dry-run pipeline shape
pdhub pipeline plan input.fasta
pdhub pipeline plan input.fasta --llm

# LLM meeting tools
pdhub agents list
pdhub agents meet "Which predictor is best for this sequence?"

# Web UI
pdhub web --host localhost --port 8501
# then open http://localhost:8501 in your browser
```

## Web UI Modules

Main pages:
- `Home`
- `Predict`
- `Evaluate`
- `Compare`
- `Agents`
- `Editor`
- `Mutagenesis`
- `Evolution`
- `MPNN Lab`
- `Batch`
- `MSA`
- `Jobs`
- `Settings`

The UI is cross-linked so outputs from prediction/evaluation/mutagenesis can be reused across pages.

## Mutagenesis Workflow

The mutagenesis page (`src/protein_design_hub/web/pages/10_mutation_scanner.py`) supports:

- Baseline comparison across selected predictors.
- Baseline no-reference evaluation before mutation selection.
- Agent discussion for:
  - residue targeting after baseline prediction,
  - baseline metric interpretation,
  - post-scan interpretation and reporting.
- Single-position saturation mutagenesis.
- Multi-mutation combination search.
- Optional per-mutant extended metrics and OpenStructure comprehensive mutant-vs-baseline comparison.
- Saved scan jobs with provenance and meeting transcripts.

## CLI Reference

Primary commands:

- `pdhub pipeline run`: unified step-only or LLM-guided pipeline.
- `pdhub pipeline plan`: dry-run pipeline stages.
- `pdhub pipeline status`: predictor + LLM + GPU checks.
- `pdhub agents run`: shortcut for `pdhub pipeline run --llm`.
- `pdhub agents meet`: ad-hoc team or individual meetings.
- `pdhub compare run`: legacy monolithic or agent-based comparison modes.
- `pdhub predict run`, `pdhub evaluate run`, `pdhub design`, `pdhub energy`, `pdhub backbone`: focused workflows.

## Python API

Step-only:

```python
from pathlib import Path
from protein_design_hub.agents import AgentOrchestrator

orchestrator = AgentOrchestrator(mode="step")
result = orchestrator.run(input_path=Path("input.fasta"))
print(result.success, result.message)
```

LLM-guided with policy override control:

```python
from pathlib import Path
from protein_design_hub.agents import AgentOrchestrator

orchestrator = AgentOrchestrator(
    mode="llm",
    num_rounds=1,
    allow_failed_llm_steps=False,  # default safety policy
)
result = orchestrator.run(
    input_path=Path("input.fasta"),
    reference_path=Path("native.pdb"),
)
if result.success:
    ctx = result.context
    print(ctx.comparison_result.best_predictor)
    print(ctx.step_verdicts.keys())
```

## LLM Providers

Built-in provider presets in `src/protein_design_hub/core/config.py`:

| Provider | Type | Default model |
|---|---|---|
| `ollama` | local | `qwen2.5:14b` |
| `lmstudio` | local | `default` |
| `vllm` | local | `default` |
| `llamacpp` | local | `default` |
| `groq` | fast cloud | `llama-3.3-70b-versatile` |
| `cerebras` | fast cloud | `llama-3.3-70b` |
| `sambanova` | fast cloud | `Meta-Llama-3.3-70B-Instruct` |
| `deepseek` | cloud | `deepseek-chat` |
| `openai` | cloud | `gpt-4o` |
| `gemini` | cloud | `gemini-2.5-flash` |
| `kimi` | cloud | `kimi-k2` |
| `openrouter` | cloud | `meta-llama/llama-3.3-70b-instruct` |

## Configuration

Config load order:
1. `config/default.yaml`
2. `~/.protein_design_hub/config.yaml`
3. environment variables

Typical LLM block:

```yaml
llm:
  provider: "ollama"
  model: "qwen2.5:14b"
  temperature: 0.2
  max_tokens: 4096
  num_rounds: 1
```

## Output Structure

Pipeline job output:

```text
outputs/<job_id>/
  metadata.json
  prediction_summary.json
  input/
    sequences.fasta
  <predictor_name>/
    ...structures...
    scores.json
    status.json
  evaluation/
    comparison_summary.json
    <predictor_name>_metrics.json
  meetings/
    *.json
    *.md
  report/
    report.html
    agent_summaries.json
    step_verdicts.json
    policy_log.json
```

Mutagenesis job output (UI):

```text
outputs/<scan_job_id>/
  prediction_summary.json
  scan_results.json
  base_wt.pdb
  meetings/
    *.json
    *.md
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Project Structure

```text
src/protein_design_hub/
  agents/       # Orchestrator, scientist personas, meeting engine
  analysis/     # Mutation scanning and analysis workflows
  cli/          # Typer CLI commands
  core/         # Configuration, shared types
  evaluation/   # Metrics and composite evaluator
  pipeline/     # Predictor execution runners
  predictors/   # Predictor adapters/installers
  web/          # Streamlit app and pages
```

## License

MIT License

## Citation

If you use this project in research, cite the primary underlying tools (for example: AlphaFold2, ColabFold, Chai-1, Boltz, OpenStructure, ESM/ESMFold/ESM3, and other predictor/evaluator backends used in your run).
