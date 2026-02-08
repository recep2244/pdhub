# Multi-Agent Pipeline

The Protein Design Hub supports a modular agent-based pipeline that can run
in two modes — **step-only** (fast, no LLM) or **LLM-guided** (agents
plan, review, and interpret every stage via structured meetings).

---

## Quick Start

### Prerequisites

1. **Ollama** running locally (default LLM backend — free, no API key):

```bash
# Install Ollama (if not already)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Verify it's running
ollama list
```

2. **Protein Design Hub** installed:

```bash
pip install -e .
```

### Run the Pipeline

```bash
# FAST: Step-only (no LLM, computational pipeline only)
pdhub pipeline run input.fasta

# FULL: LLM-guided (agents plan + review + interpret every stage)
pdhub pipeline run input.fasta --llm

# With reference structure for evaluation
pdhub pipeline run input.fasta --llm -r reference.pdb

# Pick specific predictors
pdhub pipeline run input.fasta --llm -p colabfold,chai1

# Override LLM provider on the fly
pdhub pipeline run input.fasta --llm --provider deepseek
pdhub pipeline run input.fasta --llm --provider gemini --model gemini-2.5-flash
```

### Preview Pipeline Steps (Dry-Run)

```bash
pdhub pipeline plan input.fasta          # step-only plan
pdhub pipeline plan input.fasta --llm    # LLM-guided plan
```

### Check System Status

```bash
pdhub pipeline status     # Predictors + LLM + GPU
pdhub agents status       # LLM backend details
pdhub agents list         # All scientist agents
```

---

## Pipeline Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Step-only** | `pdhub pipeline run input.fasta` | 5 computational agents, no LLM |
| **LLM-guided** | `pdhub pipeline run input.fasta --llm` | 8 agents: 5 computation + 3 LLM meetings |
| **Custom** | Python API only | Any combination of agents |

### Step-only Pipeline (5 agents)

```
Input → Prediction → Evaluation → Comparison → Report
```

### LLM-guided Pipeline (8 agents)

```
Input → [Planning Meeting] → Prediction → [Prediction Review] →
Evaluation → Comparison → [Evaluation Review] → Report
```

---

## CLI Reference

### `pdhub pipeline run`

The main entry point. Runs everything from FASTA to HTML report.

```bash
pdhub pipeline run INPUT_FILE [OPTIONS]

Options:
  -o, --output PATH       Output directory (default: ./outputs)
  -r, --reference PATH    Reference PDB/CIF for evaluation
  -p, --predictors TEXT   Comma-separated predictors (default: all enabled)
  -j, --job-id TEXT       Custom job identifier
  --llm                   Enable LLM-guided pipeline
  --provider TEXT         Override LLM provider (ollama, deepseek, openai, ...)
  --model TEXT            Override LLM model name
  --rounds INTEGER        Discussion rounds per meeting
```

### `pdhub agents run`

Shortcut that always runs the LLM-guided pipeline (equivalent to `pdhub pipeline run --llm`).

```bash
pdhub agents run INPUT_FILE [OPTIONS]

Options:
  -o, --output PATH       Output directory
  -r, --reference PATH    Reference structure
  -p, --predictors TEXT   Comma-separated predictors
  -j, --job-id TEXT       Job identifier
  --provider TEXT         Override LLM provider (ollama, deepseek, openai, ...)
  --model TEXT            Override LLM model name
  --rounds INTEGER        Rounds per meeting
```

### `pdhub agents meet`

Run a standalone LLM meeting outside the pipeline (for ad-hoc discussions).

```bash
# Team meeting with default team
pdhub agents meet "Which predictor is best for a 300-residue monomer?"

# Use a specific team
pdhub agents meet "How to refine this structure?" --team refinement

# Individual meeting (agent + critic)
pdhub agents meet "Review this pLDDT distribution" --type individual --team evaluation

# More rounds for deeper discussion
pdhub agents meet "Design strategy for nanobody" --team nanobody --rounds 3

Available team presets: default, design, nanobody, evaluation, refinement
```

### `pdhub agents status`

Check LLM backend connectivity, model availability, and list all provider presets.

### `pdhub agents list`

Show all available scientist agents with their expertise and current model.

### `pdhub compare run` (legacy)

The older comparison command still works:

```bash
pdhub compare run input.fasta               # monolithic workflow
pdhub compare run input.fasta --agents       # step agents
pdhub compare run input.fasta --llm-agents   # LLM-guided
```

---

## Python API

### Step-only Pipeline

```python
from pathlib import Path
from protein_design_hub.agents import AgentOrchestrator

orchestrator = AgentOrchestrator(mode="step")
result = orchestrator.run(input_path=Path("input.fasta"))

if result.success:
    ctx = result.context
    for name, pr in ctx.prediction_results.items():
        print(f"{name}: {len(pr.structure_paths)} structures")
```

### LLM-guided Pipeline

```python
from pathlib import Path
from protein_design_hub.agents import AgentOrchestrator

orchestrator = AgentOrchestrator(mode="llm")
result = orchestrator.run(
    input_path=Path("input.fasta"),
    reference_path=Path("native.pdb"),
)

if result.success:
    ctx = result.context
    print("Best predictor:", ctx.comparison_result.best_predictor)
    print("Plan:", ctx.extra.get("plan", "")[:200])
    print("Prediction review:", ctx.extra.get("prediction_review", "")[:200])
    print("Evaluation review:", ctx.extra.get("evaluation_review", "")[:200])
```

### Custom Agent Chain

```python
from pathlib import Path
from protein_design_hub.agents import (
    InputAgent,
    PredictionAgent,
    AgentOrchestrator,
)
from protein_design_hub.agents.llm_guided import LLMPlanningAgent

# Only parse + plan + predict (skip eval/report)
orchestrator = AgentOrchestrator(agents=[
    InputAgent(),
    LLMPlanningAgent(num_rounds=2),
    PredictionAgent(),
])
result = orchestrator.run(input_path=Path("seq.fasta"))
```

### Standalone Meeting

```python
from pathlib import Path
from protein_design_hub.agents.meeting import run_meeting
from protein_design_hub.agents.scientists import (
    PRINCIPAL_INVESTIGATOR,
    STRUCTURAL_BIOLOGIST,
    DIGITAL_RECEP,
    LIAM,
    SCIENTIFIC_CRITIC,
)

summary = run_meeting(
    meeting_type="team",
    agenda="Review and refine the predicted structure of protein X.",
    save_dir=Path("./meetings"),
    team_lead=PRINCIPAL_INVESTIGATOR,
    team_members=(STRUCTURAL_BIOLOGIST, DIGITAL_RECEP, LIAM, SCIENTIFIC_CRITIC),
    num_rounds=2,
    return_summary=True,
)
print(summary)
```

### With Progress Callback

```python
from protein_design_hub.agents import AgentOrchestrator

def on_progress(stage, item, current, total):
    print(f"[{current}/{total}] {item}")

orchestrator = AgentOrchestrator(mode="llm", progress_callback=on_progress)
result = orchestrator.run(input_path=Path("input.fasta"))
```

---

## Step Agents

| Agent | Step | Responsibility |
|-------|------|----------------|
| **InputAgent** | 1 | Parse FASTA, create `PredictionInput`, save input and metadata |
| **PredictionAgent** | 2 | Run structure predictors (ColabFold, Chai-1, Boltz-2, ...) |
| **EvaluationAgent** | 3 | Evaluate structures (lDDT, TM-score, RMSD, ...) |
| **ComparisonAgent** | 4 | Rank predictors, build `ComparisonResult` |
| **ReportAgent** | 5 | Write comparison summary, HTML report, per-tool reports |

---

## LLM Scientist Agents

Inspired by the [Virtual Lab](https://github.com/zou-group/virtual-lab)
(Swanson et al., *Nature* 2025), the LLM layer adds **scientist personas**
that discuss research decisions via **team meetings** (round-robin +
synthesis) and **individual meetings** (agent + critic loop).

### McGuffin Lab Integration

This system is co-developed with the [McGuffin Lab](https://www.reading.ac.uk/bioinf/)
at the University of Reading. All agents have deep expertise in the lab's
bioinformatics server suite:

| Tool | URL | Description |
|------|-----|-------------|
| **IntFOLD7** | [reading.ac.uk/bioinf/IntFOLD](https://www.reading.ac.uk/bioinf/IntFOLD/) | Integrated prediction: 3D modelling + QA (self-estimates) + disorder (DISOclust) + domains (DomFOLD) + ligand binding (FunFOLD) |
| **MultiFOLD2** | [reading.ac.uk/bioinf/MultiFOLD](https://www.reading.ac.uk/bioinf/MultiFOLD/) | Tertiary + quaternary structure prediction with stoichiometry; top-ranked CASP16 server on hardest domain targets; outperforms AF3 on multimers in CAMEO |
| **ModFOLD9** | [reading.ac.uk/bioinf/ModFOLD](https://www.reading.ac.uk/bioinf/ModFOLD/) | Single-model & consensus QA: global score + p-value + per-residue error; CASP-benchmarked since v1 |
| **ModFOLDdock2** | [reading.ac.uk/bioinf/ModFOLDdock](https://www.reading.ac.uk/bioinf/ModFOLDdock/) | QA for protein complexes: ranked #1 at CASP16 for global (QSCORE) and local interface accuracy |
| **ReFOLD3** | [reading.ac.uk/bioinf/ReFOLD](https://www.reading.ac.uk/bioinf/ReFOLD/) | Quality-guided model refinement using gradual restraints based on predicted local quality and residue contacts |
| **FunFOLD5** | [reading.ac.uk/bioinf/FunFOLD](https://www.reading.ac.uk/bioinf/FunFOLD/) | Protein-ligand binding site prediction with QA-integrated prediction selection |
| **DISOclust** | [reading.ac.uk/bioinf/DISOclust](https://www.reading.ac.uk/bioinf/DISOclust/) | Intrinsically disordered region prediction from 3D model ensembles (ModFOLDclust + DISOPRED) |
| **DomFOLD** | [reading.ac.uk/bioinf/DomFOLD](https://www.reading.ac.uk/bioinf/DomFOLD/) | Domain boundary prediction using DomSSEA, DISOPRED, and HHsearch consensus |

Docker packages: [`mcguffin/multifold2`](https://hub.docker.com/r/mcguffin/multifold2) (MultiFOLD2 + MultiFOLD2_refine + ModFOLDdock2), [`mcguffin/multifold`](https://hub.docker.com/r/mcguffin/multifold) (MultiFOLD + MultiFOLD_refine + ModFOLDdock)

### Scientist Personas

| Agent | Expertise |
|-------|-----------|
| **Principal Investigator** | Project leadership, AI for structural biology, CASP evaluation standards |
| **Structural Biologist** | Structure prediction, structure-function relationships, IntFOLD7, MultiFOLD2, DomFOLD |
| **Computational Biologist** | Pipeline configuration, MSA, IntFOLD7 workflows, MultiFOLD2, ModFOLD9 QA |
| **Machine Learning Specialist** | Deep learning models (ESMFold, AlphaFold, ESM3, ProteinMPNN, Chai-1, Boltz-2) |
| **Immunologist** | Antibody/nanobody engineering, binding affinity, CDR loops |
| **Protein Engineer** | Rational design, directed evolution, stability, ProteinMPNN |
| **Biophysicist** | Thermodynamics, energetics, solubility |
| **Digital Recep** | Structure refinement expert — co-developer of **ReFOLD** (quality-guided refinement with gradual restraints from ModFOLD). Also: AMBER relaxation, GalaxyRefine, ModRefiner, Rosetta FastRelax, MultiFOLD_refine (AF2 recycling). Evaluates before/after using ModFOLD scores + MolProbity + Ramachandran |
| **Liam** | Model quality assessment expert — co-developer of **ModFOLD** (v1-9), **ModFOLDdock** (v1-2), **MultiFOLD** (v1-2), **IntFOLD** (v1-7), **ReFOLD**, **FunFOLD**, **DISOclust**, **DomFOLD**. Global + per-residue QA with p-values, interface QA (CASP16 #1), stoichiometry prediction, disorder detection, domain parsing, binding site prediction |
| **Scientific Critic** | Rigour, feasibility, error identification |

### Team Compositions

| Name | Lead | Members |
|------|------|---------|
| `default` | Principal Investigator | Structural Biologist, Computational Biologist, ML Specialist, Scientific Critic |
| `design` | PI | Structural Biologist, Protein Engineer, ML Specialist, Scientific Critic |
| `nanobody` | PI | Immunologist, Structural Biologist, ML Specialist, Scientific Critic |
| `evaluation` | PI | Structural Biologist, Biophysicist, Liam, Scientific Critic |
| `refinement` | PI | Digital Recep, Structural Biologist, Liam, Scientific Critic |

---

## LLM Configuration

The LLM backend is configured in `config/default.yaml` under the `llm` section.

### Provider Presets

| Provider | Type | Default Model | Base URL | API Key Source |
|----------|------|---------------|----------|---------------|
| `ollama` | local | llama3.2:latest | localhost:11434 | none needed |
| `lmstudio` | local | default | localhost:1234 | none needed |
| `vllm` | local | default | localhost:8000 | none needed |
| `llamacpp` | local | default | localhost:8080 | none needed |
| `deepseek` | cloud | deepseek-chat | api.deepseek.com | `DEEPSEEK_API_KEY` |
| `openai` | cloud | gpt-4o | api.openai.com | `OPENAI_API_KEY` |
| `gemini` | cloud | gemini-2.5-flash | generativelanguage.googleapis.com | `GEMINI_API_KEY` |
| `kimi` | cloud | kimi-k2 | api.moonshot.cn | `MOONSHOT_API_KEY` |

### Switching Providers

**Option 1**: Edit `config/default.yaml`:

```yaml
llm:
  provider: "deepseek"    # or ollama, openai, gemini, kimi
  temperature: 0.2
  max_tokens: 4096
  num_rounds: 1
```

**Option 2**: Override at runtime:

```bash
pdhub pipeline run input.fasta --llm --provider deepseek
pdhub pipeline run input.fasta --llm --provider gemini --model gemini-2.5-flash
```

**Option 3**: Set environment variables (for cloud providers):

```bash
export DEEPSEEK_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AI..."
export MOONSHOT_API_KEY="sk-..."
```

---

## Output Structure

After a pipeline run, outputs are organised as:

```
outputs/<job_id>/
  input/
    input.fasta
    metadata.json
  predictions/
    colabfold/
    chai1/
    ...
  evaluation/
    comparison_summary.json
  report/
    report.html
    report.json
  meetings/                  # only with --llm
    planning_meeting.json
    planning_meeting.md
    prediction_review.json
    prediction_review.md
    evaluation_review.json
    evaluation_review.md
```

---

## Registering Custom Agents

```python
from protein_design_hub.agents import BaseAgent, AgentResult, WorkflowContext, AgentRegistry

class MyAgent(BaseAgent):
    name = "my_step"
    description = "Custom step"

    def run(self, context: WorkflowContext) -> AgentResult:
        # ... use and update context ...
        return AgentResult.ok(context, "Done")

AgentRegistry.register("my_step", MyAgent)
```

---

## Extending to Other Workflows

The same pattern works for:

- **Design pipeline**: Input -> *Design Meeting* -> Design (ESMIF/MPNN/RFdiffusion) -> *Design Review* -> Prediction -> Evaluation -> Report
- **Mutation scanner**: Input -> *Mutation Strategy Meeting* -> Base prediction -> Saturation scan -> *Results Review* -> Report
- **Nanobody design**: Input -> *Nanobody Team Meeting* -> ESM -> AlphaFold-Multimer -> Rosetta -> *Scoring Review* -> Selection -> Report

Add new step agents and/or LLM meeting agents, compose them in a list, and
pass to `AgentOrchestrator(agents=[...])`.
