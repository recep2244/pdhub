# External Integrations

**Analysis Date:** 2026-02-21

## Protein Structure Predictors

**ColabFold (LocalColabFold):**
- What: Fast AlphaFold2 structure prediction with MMseqs2 MSA
- SDK/Client: `colabfold_batch` binary (installed via LocalColabFold)
- Config: `src/protein_design_hub/core/config.py` - `ColabFoldConfig`
  - MSA modes: mmseqs2_uniref_env (default), mmseqs2_uniref, single_sequence
  - Models: auto, alphafold2, alphafold2_ptm, alphafold2_multimer_v1/v2/v3
  - AMBER relaxation support
- Installer: `src/protein_design_hub/predictors/colabfold/installer.py`
- Searches: pixi-based (`~/.pixi/envs/`) or conda-based installation

**Chai-1:**
- What: Multi-molecule structure prediction (proteins, DNA, RNA, ligands)
- Package: `chai` (pip install via external source)
- Config: `src/protein_design_hub/core/config.py` - `Chai1Config`
  - ESM embeddings integration
  - Optional MSA server: https://api.colabfold.com
  - Template support
- Installer: `src/protein_design_hub/predictors/chai1/installer.py`
- Status: Optional (enabled in default.yaml)

**Boltz-2:**
- What: Latest biomolecular structure prediction (multi-chain, ligands, small molecules)
- Package: `boltz` (pip install)
- Config: `src/protein_design_hub/core/config.py` - `Boltz2Config`
  - MSA server integration (enabled by default)
  - Pairing strategies: greedy, complete
  - Affinity prediction support
- Installer: `src/protein_design_hub/predictors/boltz2/installer.py`
- Status: Optional (enabled in default.yaml)

**ESMFold:**
- What: Meta ESM-2 folding (fast, single-sequence or with MSA)
- Package: `esm` (pip install)
- Config: Predictors setup in settings
- Installer: `src/protein_design_hub/predictors/esmfold/installer.py`
- Uses: ESM-2 embeddings internally

**ESM3:**
- What: Meta's ESM-3 generative model for structure and sequence
- Package: `esm` (pip install)
- Config: `src/protein_design_hub/core/config.py` - optional
- Installer: `src/protein_design_hub/predictors/esm3/installer.py`
- Status: Optional

## Structure Design Tools

**ProteinMPNN:**
- What: Fixed-backbone sequence design via graph neural networks
- Installation: Git clone to `~/.protein_design_hub/tools/ProteinMPNN`
- Script: `ProteinMPNN/protein_mpnn_run.py`
- Installer: `src/protein_design_hub/design/proteinmpnn/installer.py`
- Used by: Web UI pages, design agents (MPNN design workflows)

**RFdiffusion:**
- What: Diffusion-based backbone generation and motif scaffolding
- Installation: Git clone to `~/.protein_design_hub/tools/RFdiffusion`
- Scripts: `RFdiffusion/scripts/run_inference.py`, `download_models.sh`
- Installer: `src/protein_design_hub/design/rfdiffusion/installer.py`
- Used by: Design page, backbone generation pipelines

**ESMIF:**
- What: ESM Inverse Folding for sequence design from structure
- Installation: Git clone to `~/.protein_design_hub/tools/ESMIF`
- Installer: `src/protein_design_hub/design/esmif/installer.py`
- Used by: Inverse folding design agents

## LLM Backends (OpenAI-compatible)

**Ollama (default local):**
- URL: http://localhost:11434/v1
- Model: `qwen2.5:14b` (default, ~9 GB VRAM)
- Alternative: `deepseek-r1:14b` (chain-of-thought reasoning)
- No API key required
- Config: `src/protein_design_hub/core/config.py` - `LLM_PROVIDER_PRESETS["ollama"]`
- Used: Agent meetings, scientist discussions, verdict generation
- File: `src/protein_design_hub/agents/meeting.py` - `_get_llm_client()`

**Groq (fast cloud):**
- URL: https://api.groq.com/openai/v1
- Model: `llama-3.3-70b-versatile`
- Speed: ~300+ tok/s
- Auth: `GROQ_API_KEY` env var
- Config: `src/protein_design_hub/core/config.py` - `LLM_PROVIDER_PRESETS["groq"]`
- Free tier available

**Cerebras (fast cloud):**
- URL: https://api.cerebras.ai/v1
- Model: `llama-3.3-70b`
- Speed: ~450+ tok/s
- Auth: `CEREBRAS_API_KEY` env var
- Free tier available

**SambaNova (fast cloud):**
- URL: https://api.sambanova.ai/v1
- Model: `Meta-Llama-3.3-70B-Instruct`
- Speed: ~200+ tok/s
- Auth: `SAMBANOVA_API_KEY` env var
- Free tier available

**DeepSeek (cost-effective cloud):**
- URL: https://api.deepseek.com/v1
- Model: `deepseek-chat`
- Cost: $0.28/1M input tokens
- Auth: `DEEPSEEK_API_KEY` env var

**OpenAI:**
- URL: https://api.openai.com/v1
- Model: `gpt-4o` (default)
- Auth: `OPENAI_API_KEY` env var
- Used for high-reliability meetings

**Google Gemini:**
- URL: https://generativelanguage.googleapis.com/v1beta/openai/
- Model: `gemini-2.5-flash`
- Auth: `GEMINI_API_KEY` env var
- Free tier available

**Kimi (Moonshot):**
- URL: https://api.moonshot.cn/v1
- Model: `kimi-k2`
- Auth: `MOONSHOT_API_KEY` env var

**OpenRouter:**
- URL: https://openrouter.ai/api/v1
- Model: `meta-llama/llama-3.3-70b-instruct` (default)
- Auth: `OPENROUTER_API_KEY` env var
- Access to many models

**LM Studio (local):**
- URL: http://localhost:1234/v1
- Default model: `default`
- No API key required
- Alternative to Ollama

**vLLM (fast local):**
- URL: http://localhost:8000/v1
- Default model: `default`
- No API key required
- Fastest local inference option

**LlamaCPP (local):**
- URL: http://localhost:8080/v1
- Default model: `default`
- No API key required
- CPU-capable local option

## Data & Structure Databases

**RCSB PDB:**
- API: https://data.rcsb.org/rest/v1/core/entry
- Download: https://files.rcsb.org/download
- Client: `src/protein_design_hub/io/fetch.py` - `PDBFetcher` class
- Usage: Structure lookups, template search, format conversion (pdb/cif/mmcif)
- Caching: Local cache in `~/.cache/pdhub_cache/pdb/`

**AlphaFold DB (AFDB):**
- API: https://alphafold.ebi.ac.uk/api/prediction
- Client: `src/protein_design_hub/io/afdb.py` - `AFDBFetcher` class
- Usage: AFDB structure lookup by UniProt ID
- MSA source: ColabFold API (https://api.colabfold.com) for MSA pre-computation

**EBI BLAST:**
- API: https://www.ebi.ac.uk/Tools/services/rest/ncbiblast
- Client: `src/protein_design_hub/io/afdb.py` - `run_blast_ebi_uniref()` function
- Usage: Sequence similarity search for template identification
- Returns: UniProt hits with identity and coverage

**ColabFold MSA Server:**
- URL: https://api.colabfold.com
- Config: Chai-1 and Boltz-2 can use for pre-computed MSAs
- Optional: Can be disabled for offline use

## Structure Evaluation & Metrics

**OpenStructure (OST):**
- Environment: Separate micromamba environment `ost`
- Client: `src/protein_design_hub/evaluation/ost_runner.py` - `OpenStructureRunner`
- Usage: Comprehensive structure metrics (CAD-score, VoroMQA, clash detection, etc.)
- Commands: Via subprocess in isolated conda environment
- Metrics: VoroMQA local score, CAD-score, clash detection

**Rosetta (optional):**
- Binaries: score_jd2, other command-line tools
- Client: `src/protein_design_hub/energy/rosetta.py` - functions
- Config: Rosetta home auto-detection or explicit path
- Usage: Energy functions, structure refinement scoring
- Installer: Must be built/installed separately (not pip)

**FoldX (optional):**
- Tool: Energy calculation and mutation effects
- Client: `src/protein_design_hub/energy/foldx.py`
- Usage: Mutagenesis energy deltas, stability predictions
- Status: Optional external tool

**TM-align:**
- Purpose: Structure alignment and TM-score calculation
- Binary: Auto-detected from PATH or config
- Client: `src/protein_design_hub/evaluation/metrics/tm_score.py`
- Usage: Structural similarity metric (0-1 scale)

**OpenMM:**
- Package: `openmm` (optional install)
- Client: `src/protein_design_hub/evaluation/metrics/openmm_gbsa.py`
- Usage: Generalized Born solvation energy (GBSA) for binding affinity
- GPU acceleration: CUDA backend when available

**VoronoTA:**
- Purpose: Structure quality assessment
- Client: `src/protein_design_hub/evaluation/metrics/voronota_*.py`
- Tools: VoroMQA, CAD-score evaluation
- Integration: Via OpenStructure runner or direct binary

## Mutagenesis & Analysis

**Mutagenesis Scanning:**
- Module: `src/protein_design_hub/analysis/mutation_scanner.py`
- Features: Saturation mutagenesis, multi-site mutations, baseline comparison
- Metrics: pAE, pLDDT, energy changes, interface metrics

**Agents:**
- Module: `src/protein_design_hub/agents/mutagenesis_agents.py`
- Specialists: Mutagenesis expert, baseline reviewer
- LLM-guided: Verdict generation, mutation scoring, candidate ranking

## Authentication & Secrets

**API Keys:**
- Environment variables (preferred): `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, etc.
- Config file (not recommended): `config/default.yaml` or user config
- Fallback: Preset defaults for local/free providers (Ollama, Groq, etc.)

**Security:**
- `.env*` files in .gitignore (never committed)
- Pydantic Settings with nested env var mapping (`__` delimiter)
- Client timeout: 120s (OpenAI client in `meeting.py`)
- Max retries: 2 (OpenAI client)

## Monitoring & Observability

**Logging:**
- Approach: Standard Python `print()` and logging module
- Per-call timing: Printed by `src/protein_design_hub/agents/meeting.py`
  - Format: `[Agent] 3.2s, 120 tok, 37 tok/s`
- Discussion transcripts: JSON + Markdown saved to disk
- Verdicts: `step_verdicts.json` per pipeline run
- Policy log: `policy_log.json` for override tracking

**Error Tracking:**
- No external service (local error handling)
- Exceptions: Custom types in `src/protein_design_hub/core/exceptions.py`
- Verdict-based gating: `FAIL` halts pipeline by default

**Coverage:**
- Tool: pytest-cov
- CI/CD: Codecov integration (GitHub Actions)
- Report: `coverage.xml` generated per run

## Webhooks & Callbacks

**Incoming:**
- None detected (Streamlit-based UI only)

**Outgoing:**
- None detected (batch processing, no external notifications)

**Data Exchange:**
- Uni-directional: Pull from PDB, AFDB, EBI
- File-based: Output to `./outputs/` directory
- No push to external services

## Container & Deployment

**Docker:**
- Base: `python:3.10-slim`
- Exposed: Port 8501 (Streamlit)
- Env: `PDHUB_OUTPUT_BASE_DIR=/app/outputs`
- Mount: config/, src/, outputs/

**Docker Compose:**
- Service: pdhub
- Volume persistence: outputs/, config/, src/
- Network: Host mapping 8501→8501
- Restart: unless-stopped

## Tool Paths & Installation

**Tools Directory:**
- Default: `~/.protein_design_hub/tools/`
- Config key: `installation.tools_dir`
- Contents:
  - `ColabFold/` - colabfold_batch binary
  - `ProteinMPNN/` - git clone
  - `RFdiffusion/` - git clone
  - `ESMIF/` - git clone

**Virtual Environments:**
- Primary: Current Python environment (pyproject.toml)
- OST: Separate micromamba `ost` environment
- Ollama: Separate process (no venv)

## GPU Management

**Ollama GPU:**
- Cache: 60-second TTL (avoid repeated `ollama ps` checks)
- File: `src/protein_design_hub/agents/ollama_gpu.py`
- Flags: `num_ctx=4096, num_batch=512, keep_alive=10m`

**CUDA Support:**
- GPU device: Configurable in `gpu.device` (default: cuda:0)
- Memory fraction: Configurable (default: 0.95)
- Clear cache between jobs: Configurable

---

*Integration audit: 2026-02-21*
