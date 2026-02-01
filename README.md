# Protein Design Hub

A unified platform for protein structure prediction and evaluation, integrating multiple state-of-the-art prediction tools with comprehensive quality metrics.

## Features

### Predictors
- **ColabFold** - AlphaFold2 with MMseqs2 for fast MSA generation
- **Chai-1** - Multi-molecule structure prediction (proteins, DNA, RNA, ligands)
- **Boltz-2** - Diffusion-based biomolecular structure prediction
- **ESMFold (local/API)** - Single-sequence folding (ESM-2); API for CPU-only
- **ESM3** - Multimodal generation (structure track, optional; local or Forge)

### Evaluation Metrics
- **lDDT / lDDT-PLI** - Local Distance Difference Tests (OpenStructure)
- **QS-score** - Quaternary Structure Score for complexes
- **TM-score** - Template Modeling Score (TMalign)
- **RMSD / GDT-TS / GDT-HA**
- **Clash score** - Steric clash proxy (heavy-atom contacts)
- **Contact energy** - Coarse residue-residue contact potential
- **SASA / Interface BSA / Salt bridges**
- **VoroMQA / CAD-score** - Voronota-based model quality metrics
- **OpenMM GBSA** - Implicit-solvent energy
- **Sequence recovery / disorder / shape complementarity**
- **Rosetta energy** - Optional ref2015 total score (requires PyRosetta)

### Mutation Scanner & Baselines
- Per-residue pLDDT line plots and low-confidence residue selection
- Saturation mutagenesis + multi-mutation design pipeline
- Baseline comparison: **ESMFold API (ESM-2)** vs **ESMFold v0 (ESM-1)** vs **ESM3**
- Optional **ImmuneBuilder** for antibody/nanobody/TCR structure prediction
- Optional AlphaFold DB lookup (≥90% identity/coverage) via EBI/UniProt BLAST

### Interfaces
- **CLI** - Full-featured command-line interface
- **Web UI** - Streamlit-based graphical interface

## Architecture Overview

```
FASTA/Sequence Input
   ├── Predictors (ColabFold / Chai-1 / Boltz-2 / ESMFold / ESM3)
   │     └── Structures + scores (pLDDT, confidence, metadata)
   ├── Evaluation (OpenStructure, TMalign, Voronota, OpenMM, optional Rosetta)
   │     └── Global + interface + per-residue metrics
   └── Mutation Scanner
         ├── Base structure + per-residue pLDDT plot
         ├── Saturation scan + multi-mutation pipeline
         └── Optional AFDB match + extended evaluation metrics
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- Conda (for OpenStructure)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/recep2244/pdhub.git
cd pdhub

# Run setup script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Activate environment
conda activate protein_design_hub

# Install prediction tools
pdhub install --all
```

### Manual Installation

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate protein_design_hub

# Install package
pip install -e .

# Install predictors
pdhub install predictor colabfold
pdhub install predictor chai1
pdhub install predictor boltz2
pdhub install predictor esmfold
```

### Optional Tooling (Recommended)

```bash
# ESM3 (gated HF model or Forge)
pdhub install predictor esm3
export ESM3_PYTHON=/path/to/esm3_env/bin/python
export ESM3_HF_TOKEN=...   # or ESM3_FORGE_TOKEN=...

# ImmuneBuilder (antibody/nanobody/TCR)
export IMMUNEBUILDER_PYTHON=/path/to/immunebuilder_env/bin/python

# Voronota + OpenMM metrics (for CAD-score / VoroMQA / GBSA)
# Ensure voronota/voronota-cadscore/voronota-voromqa are on PATH.
# OpenMM GBSA uses PDBFixer to repair missing atoms:
#   pip install git+https://github.com/openmm/pdbfixer
```

Notes:
- **ESM3** uses the EvolutionaryScale `esm` package (same import name as `fair-esm`).
  Use a separate environment for ESM3 to avoid conflicts with local ESMFold.

## Usage

### CLI Commands

```bash
# Check installation status
pdhub status

# Run prediction with all tools
pdhub predict run input.fasta --predictor all --output ./results

# Run single predictor
pdhub predict single input.fasta chai1

# Evaluate structure
pdhub evaluate run model.pdb --reference native.pdb --metrics lddt,tm_score,voromqa,openmm_gbsa

# Full comparison pipeline
pdhub compare run input.fasta --reference native.pdb --output ./comparison

# Launch web interface
pdhub web
```

### Python API

```python
from protein_design_hub import PredictionInput, Sequence
from protein_design_hub.pipeline import PredictionWorkflow
from protein_design_hub.evaluation import CompositeEvaluator

# Create input
sequences = [Sequence(id="my_protein", sequence="MKFLILLFNILCLFPVLAAD...")]
prediction_input = PredictionInput(
    job_id="my_job",
    sequences=sequences,
    num_models=5,
)

# Run predictions
workflow = PredictionWorkflow()
result = workflow.run(
    input_path="input.fasta",
    reference_path="reference.pdb",
    predictors=["colabfold", "chai1", "boltz2"],
)

# Evaluate structures
evaluator = CompositeEvaluator(metrics=["lddt", "tm_score", "rmsd"])
eval_result = evaluator.evaluate(model_path, reference_path)
print(f"lDDT: {eval_result.lddt:.3f}")
print(f"TM-score: {eval_result.tm_score:.3f}")
```

## Project Structure

```
protein_design_hub/
├── pyproject.toml
├── environment.yaml
├── config/
│   └── default.yaml
├── src/protein_design_hub/
│   ├── core/                 # Types, config, exceptions
│   ├── analysis/             # Mutation scanner + pipelines
│   ├── io/                   # Parsers and writers
│   ├── predictors/           # Predictor implementations
│   │   ├── colabfold/
│   │   ├── chai1/
│   │   ├── boltz2/
│   │   ├── esmfold/
│   │   └── esm3/
│   ├── evaluation/           # Quality metrics + composite evaluator
│   │   └── metrics/
│   ├── pipeline/             # Workflow orchestration
│   ├── cli/                  # Command-line interface
│   └── web/                  # Streamlit web UI
├── scripts/
│   └── setup_environment.sh
└── outputs/                  # Prediction results
```

## Configuration

Configuration is loaded from (in order of priority):
1. `config/default.yaml`
2. `~/.protein_design_hub/config.yaml`
3. Environment variables (prefixed with `PDHUB_`)

Example configuration:
```yaml
output:
  base_dir: "./outputs"
  save_all_models: true

predictors:
  colabfold:
    num_models: 5
    num_recycles: 3
    use_amber: false
  chai1:
    num_trunk_recycles: 3
    num_diffusion_timesteps: 200
  boltz2:
    recycling_steps: 3
    sampling_steps: 200

evaluation:
  metrics:
    - lddt
    - tm_score
    - qs_score
    - rmsd
```

## Output Structure

```
outputs/{job_id}/
├── metadata.json
├── input/
│   └── sequences.fasta
├── colabfold/
│   ├── structures/*.pdb
│   └── scores.json
├── chai1/
│   ├── structures/*.cif
│   └── scores.json
├── boltz2/
│   ├── structures/*.cif
│   └── scores.json
├── evaluation/
│   ├── comparison_summary.json
│   └── {predictor}_metrics.json
└── report/
    └── report.html
```

## Requirements

### Prediction Tools
- **ColabFold**: Installed via LocalColabFold
- **Chai-1**: `pip install chai_lab`
- **Boltz-2**: `pip install boltz`

### Evaluation Tools
- **OpenStructure**: `conda install -c bioconda openstructure`
- **TMalign**: Download from https://zhanggroup.org/TM-align/

## License

MIT License

## Citation

If you use Protein Design Hub in your research, please cite the underlying tools:

- **ColabFold**: Mirdita et al., Nature Methods 2022
- **Chai-1**: Chai Discovery, 2024
- **Boltz-2**: MIT, 2024
- **AlphaFold2**: Jumper et al., Nature 2021
- **OpenStructure**: Biasini et al., BMC Bioinformatics 2013
