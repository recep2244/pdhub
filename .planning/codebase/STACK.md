# Technology Stack

**Analysis Date:** 2026-02-21

## Languages

**Primary:**
- Python 3.10+ - All core code, predictors, agents, web UI
- YAML - Configuration (default.yaml, workflows)
- JSON - Data serialization, transcripts, verdicts
- Markdown - Documentation

**Secondary:**
- Shell (bash) - Installation scripts, tools management
- Dockerfile - Container configuration

## Runtime

**Environment:**
- Python 3.10, 3.11, 3.12 (supported versions in `pyproject.toml`)
- CUDA 11.x/12.x (recommended for GPU acceleration)
- Optional: Conda for environment management

**Package Manager:**
- pip (primary)
- conda (environment.yaml for bootstrap)
- Micromamba (separate OST environment for OpenStructure)

**Lockfile:**
- `pyproject.toml` with pinned minimum versions
- No lock file present (uses version ranges)

## Frameworks

**Core Application:**
- Streamlit 1.28.0+ - Web UI with multi-page app
- Typer 0.9.0+ - CLI framework
- FastAPI capabilities via Streamlit

**LLM & Agents:**
- OpenAI SDK 1.0.0+ - OpenAI-compatible API clients (Ollama, Groq, Cerebras, etc.)
- Pydantic 2.0.0+ - Data validation and settings management
- Pydantic-Settings 2.0.0+ - Environment variable management

**Scientific Computing:**
- BioPython 1.81+ - Protein sequence/structure parsing
- NumPy 1.24.0+ - Numerical arrays
- Pandas 2.0.0+ - Tabular data and analysis
- Plotly 5.18.0+ - Interactive visualizations

**Utilities:**
- Rich 13.0.0+ - Terminal formatting and progress bars
- PyYAML 6.0+ - Configuration file parsing
- Requests 2.28.0+ - HTTP client for external APIs

**Testing:**
- Pytest 7.0.0+ - Test runner
- Pytest-cov 4.0.0+ - Code coverage reporting

**Code Quality:**
- Black 23.0.0+ - Code formatter (line length 100)
- Ruff 0.1.0+ - Fast linter
- Mypy 1.0.0+ - Static type checking

## Key Dependencies

**Critical:**
- `pydantic>=2.0.0` - Runtime data validation for settings, configs, and type safety
- `streamlit>=1.28.0` - Web UI framework (architecture-critical)
- `openai>=1.0.0` - LLM backend (agent orchestration, meetings)
- `biopython>=1.81` - Protein structure parsing and validation

**Infrastructure:**
- `requests>=2.28.0` - HTTP calls to RCSB PDB, AlphaFold DB, EBI BLAST APIs
- `plotly>=5.18.0` - Interactive structural visualizations
- `numpy>=1.24.0` - Numerical operations (AlphaFold metrics, MSA)
- `pandas>=2.0.0` - Result aggregation and reporting

**Optional (conditionally imported):**
- `boltz` - Boltz-2 structure predictor (pip install)
- `chai` - Chai-1 multi-molecule predictor
- `colabfold` - LocalColabFold wrapper (git clone + pip)
- `openmm` - Molecular dynamics (OST/refinement metrics)
- `esm` - ESM/ESMFold embeddings and folding
- `rosetta` - Rosetta energy functions (binary/source install)

## Configuration

**Environment:**
- Primary: `config/default.yaml` (checked first)
- User: `~/.protein_design_hub/config.yaml`
- System: `/etc/protein_design_hub/config.yaml`
- Env vars: `PDHUB_*` prefix with nested delimiter `__` (e.g., `PDHUB_LLM__PROVIDER`)

**Key Configuration Files:**
- `config/default.yaml` - Predictors (ColabFold, Chai-1, Boltz-2), evaluation metrics, GPU settings, LLM provider presets
- `pyproject.toml` - Package metadata, dependencies, tool configs
- `environment.yaml` - Conda environment specification
- `.github/workflows/ci.yml` - GitHub Actions CI/CD

**Environment Variables (Secrets):**
- `OPENAI_API_KEY` - OpenAI API access
- `DEEPSEEK_API_KEY` - DeepSeek API access
- `GEMINI_API_KEY` - Google Gemini API access
- `GROQ_API_KEY` - Groq API access (fast cloud inference)
- `CEREBRAS_API_KEY` - Cerebras API access (fast cloud inference)
- `SAMBANOVA_API_KEY` - SambaNova API access (fast cloud inference)
- `MOONSHOT_API_KEY` - Kimi API access
- `OPENROUTER_API_KEY` - OpenRouter API access

## Platform Requirements

**Development:**
- Linux/macOS/Windows with CUDA toolkit 11.x or 12.x (GPU recommended)
- Git for cloning design tools (ProteinMPNN, RFdiffusion, ESMIF)
- Micromamba for separate OST environment
- Ollama for local LLM inference (optional, default provider)

**Production:**
- Docker: `Dockerfile` builds Python 3.10-slim + deps
- Docker Compose: `docker-compose.yml` for containerized Streamlit on port 8501
- Kubernetes capable (standard Python web container)
- GPU support via CUDA (nvidia-docker recommended)

**System Dependencies (from Dockerfile):**
- build-essential (compilation)
- git (tool installation)

## Build & Tooling

**Build System:**
- setuptools>=61.0 (via pyproject.toml)
- wheel packaging

**CI/CD:**
- GitHub Actions (`ci.yml`):
  - Runs on: ubuntu-latest
  - Python 3.10, 3.11 matrix
  - Linting (ruff), type checking (mypy), testing (pytest + cov)
  - Codecov integration

**Development Tools:**
- Pre-commit hooks: `.pre-commit-config.yaml` configured but optional enforcement

**Install System:**
- CLI installer: `pdhub install` command
- Tools installed to `~/.protein_design_hub/tools/` (configurable)
- Support for ColabFold (binary), ProteinMPNN (git), RFdiffusion (git), ESMIF (git), etc.

## External Model Registries

**Local Ollama:**
- Download URLs: `ollama pull qwen2.5:14b` (default)
- Alternative: `deepseek-r1:14b` (reasoning)
- Legacy migration: llama3.2:* → qwen2.5:14b

**PyPI (pip):**
- boltz, chai, colabfold, esm packages

**Git Repositories:**
- ProteinMPNN: github.com (git clone)
- RFdiffusion: github.com (git clone)
- ESMIF: (git clone)

## Python Version Support

**Supported:**
- 3.10 (primary, 10+ tested via CI)
- 3.11 (tested via CI)
- 3.12 (listed in classifiers)

**Minimum:** Python 3.10 (`requires-python = ">=3.10"`)

---

*Stack analysis: 2026-02-21*
