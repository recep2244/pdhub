"""Getting Started — Protein Design Hub complete user guide.

Step-by-step onboarding for every workflow, use-case track, and tool.
No installation or configuration required to read this page.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[3]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st
from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    section_header,
    info_box,
)
from protein_design_hub.web.agent_helpers import agent_sidebar_status

inject_base_css()
sidebar_nav(current="Guide")
sidebar_system_status()
agent_sidebar_status()

page_header(
    "Getting Started",
    "Complete guide to the Protein Design Hub — from first run to advanced workflows",
    "📖",
)

# ── Quick navigation inside the guide ───────────────────────────────────────
st.markdown("""
<style>
.guide-toc a { color: var(--pdhub-primary-light); text-decoration: none; font-weight:500; }
.guide-toc a:hover { text-decoration: underline; }
.guide-step { background: var(--pdhub-bg-card); border-left: 4px solid var(--pdhub-primary);
  padding: 12px 18px; border-radius: 0 10px 10px 0; margin: 8px 0; }
.guide-step-num { font-size: 0.75rem; color: var(--pdhub-primary-light);
  font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.guide-step-title { font-weight: 600; font-size: 1rem; color: var(--pdhub-text); margin: 2px 0; }
.guide-step-body { color: var(--pdhub-text-secondary); font-size: 0.88rem; line-height: 1.6; }
.guide-tip { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.25);
  border-radius: 10px; padding: 10px 16px; margin: 8px 0; font-size: 0.88rem; }
.guide-warn { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25);
  border-radius: 10px; padding: 10px 16px; margin: 8px 0; font-size: 0.88rem; }
.guide-code { font-family: 'IBM Plex Mono', monospace; background: var(--pdhub-bg-light);
  padding: 2px 6px; border-radius: 4px; font-size: 0.84rem; color: var(--pdhub-primary-light); }
.track-card { background: var(--pdhub-bg-card); border: 1px solid var(--pdhub-border);
  border-radius: 14px; padding: 18px 20px; height: 100%; }
.track-title { font-size: 1.05rem; font-weight: 700; color: var(--pdhub-text); margin-bottom: 6px; }
.track-body { color: var(--pdhub-text-secondary); font-size: 0.85rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Top-of-page overview strip ───────────────────────────────────────────────
col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
for col, icon, title, subtitle in [
    (col_ov1, "🔮", "Predict", "5 structure predictors"),
    (col_ov2, "🧬", "Design", "Editor, MPNN, Evolution"),
    (col_ov3, "🤖", "AI Agents", "10 specialist personas"),
    (col_ov4, "🧫", "Specialised", "Antibody · Plant · Batch"),
]:
    with col:
        st.markdown(
            f'<div style="background:var(--pdhub-bg-card);border:1px solid var(--pdhub-border);'
            f'border-radius:12px;padding:14px;text-align:center">'
            f'<div style="font-size:1.6rem">{icon}</div>'
            f'<div style="font-weight:600;color:var(--pdhub-text);margin:4px 0">{title}</div>'
            f'<div style="color:var(--pdhub-text-muted);font-size:.78rem">{subtitle}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Installation & First Run
# ════════════════════════════════════════════════════════════════════════════
section_header("1. Installation & First Run", icon="🚀")

with st.expander("1.1 — Prerequisites & Installation", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**System requirements**
- Python **3.10** or newer
- 8 GB RAM minimum (16 GB recommended)
- GPU with 8+ GB VRAM strongly recommended for local predictors
- CUDA 11.8+ (for GPU acceleration)

**Install from source**
```bash
git clone <repo-url>
cd protein-design-hub
pip install -e ".[dev]"
```

**Install with conda**
```bash
conda create -n pdhub python=3.11
conda activate pdhub
pip install -e ".[dev]"
```
""")
    with col_b:
        st.markdown("""
**Docker (recommended for reproducibility)**
```bash
docker compose up -d
# Web UI → http://localhost:8501
# PyMOL viewer → http://localhost:8592
```

**Run directly**
```bash
pdhub web
# or
streamlit run src/protein_design_hub/web/app.py
```

**Install predictors** (one-time, after pip install)
```bash
pdhub install colabfold    # AlphaFold2-based
pdhub install chai1        # Chai-1 (multi-chain)
pdhub install boltz2       # Boltz-2 (cofactor-aware)
# ESMFold uses public API — no install needed
```
""")

with st.expander("1.2 — Configure an LLM backend (for AI Agent features)"):
    st.markdown("""
The AI Agent pipeline requires an LLM. Three options — pick one:

| Option | Speed | Cost | Setup |
|--------|-------|------|-------|
| **Ollama (local)** | ★★★ | Free | Install Ollama + pull model |
| **Groq / Cerebras** | ★★★★★ | Free tier | API key in Settings |
| **OpenAI / DeepSeek** | ★★★★ | Pay-per-use | API key in Settings |

**Default: Ollama with qwen2.5:14b (fits in 12 GB VRAM)**
```bash
# Install Ollama: https://ollama.com/download
ollama pull qwen2.5:14b
```
Selecting the `ollama` provider in Settings automatically uses `qwen2.5:14b`. Optional upgrade: `ollama pull qwen3:14b` and pick it in the Model dropdown.

**Quick cloud option (no GPU needed):**
Open **Settings → LLM Provider**, select `groq`, and paste a free Groq API key.
""")

    st.markdown("""
<div class="guide-tip">
✅ <strong>Tip:</strong> You can use the entire pipeline (predict, evaluate, mutate, design) without an LLM.
The AI Agent features are additive — they interpret results, they don't produce them.
</div>
""", unsafe_allow_html=True)

with st.expander("1.3 — Verify your setup"):
    st.markdown("""
**Check everything is working on the Settings page:**
1. Navigate to **Settings** (⚙️ in sidebar)
2. In the **Predictor Status** panel — confirm at least one predictor shows ✅
3. In the **LLM Status** panel — send a test message (optional)
4. In the **System** panel — confirm GPU is detected (or CPU mode is acceptable)

**Quick smoke test**
```bash
pytest tests/test_web_smoke.py -q   # should show 14 passed
```
""")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Core Workflow (5 steps)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
section_header("2. Core Workflow — The 5-Step Pipeline", icon="⚙️")

st.markdown("""
Every study in this hub follows the same 5-step backbone.
You can start at any step and skip steps you don't need.
""")

steps = [
    ("Step 1", "🔮 Predict", "pages/1_predict.py",
     "Input an amino-acid sequence → choose a predictor → get a 3D structure (PDB) with per-residue confidence (pLDDT).",
     ["Paste a sequence or upload a FASTA file.",
      "Select a predictor: ESMFold (fast, API-based), ColabFold (most accurate), Chai-1 (multi-chain), Boltz-2 (cofactors).",
      "Click Predict. Results appear in the viewer and are saved to outputs/.",
      "The structure is automatically shared with other pages via session state."]),
    ("Step 2", "📊 Evaluate", "pages/2_evaluate.py",
     "Load a PDB → compute biophysical quality metrics → identify problem regions.",
     ["Upload a PDB or use the last predicted structure.",
      "Click Evaluate. Metrics computed: pLDDT mean, TM-score vs reference, RMSD, clash score, SASA, VoroMQA, Ramachandran.",
      "Optional: PTM Liability scanner, Tm prediction, OpenMM GBSA energy.",
      "Export a full PDF report."]),
    ("Step 3", "🧬 Mutagenesis", "pages/10_mutation_scanner.py",
     "Select positions → scan all 19 amino acid substitutions → rank by structural impact.",
     ["Load a sequence (or use the predicted one).",
      "Click 'Run Plant Biology Analysis' if working with plant proteins.",
      "Select 2–5 positions to scan (more = slower).",
      "Run Phase 1: LLM suggests positions. Run Phase 2: ESMFold predicts each variant.",
      "The ranked table shows delta pLDDT, RMSD, and NLR impact flags (🔴/🟡/🟢)."]),
    ("Step 4", "🎯 Design (MPNN)", "pages/8_mpnn.py",
     "Upload a backbone structure → ProteinMPNN generates new sequences that fold into it.",
     ["Upload a PDB backbone (use the predicted structure from Step 1).",
      "Set temperature (0.1 = conservative, 0.3 = balanced, 0.5+ = diverse).",
      "Optionally fix key positions, add sequence constraints, or enable wheat codon optimization.",
      "Run MPNN → get 8–32 designed sequences with pI/GRAVY/instability metrics.",
      "Send top sequence → Predict page to validate the design."]),
    ("Step 5", "⚖️ Compare / Refine", "pages/3_compare.py",
     "Run the same sequence through multiple predictors → compare confidence and structural agreement.",
     ["Paste sequence → select 2+ predictors.",
      "Compare pLDDT, TM-score (inter-predictor), clash scores side-by-side.",
      "High agreement = high structural confidence. Divergence = flexible / uncertain region.",
      "Send best structure to Evolution for further optimization."]),
]

for step_num, title, target, desc, substeps in steps:
    with st.container():
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(
                f'<div style="background:var(--pdhub-bg-card);border:1px solid var(--pdhub-border);'
                f'border-radius:12px;padding:18px 14px;text-align:center">'
                f'<div style="font-size:.75rem;color:var(--pdhub-primary-light);font-weight:700;letter-spacing:.08em">{step_num}</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:var(--pdhub-text);margin:6px 0">{title}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Open {title.split()[-1]}", key=f"guide_open_{step_num}", width='stretch'):
                st.switch_page(target)
        with c2:
            st.markdown(
                f'<div class="guide-step">'
                f'<div class="guide-step-body"><strong>{desc}</strong></div></div>',
                unsafe_allow_html=True,
            )
            for i, sub in enumerate(substeps, 1):
                st.markdown(
                    f'<div style="padding-left:18px;color:var(--pdhub-text-secondary);'
                    f'font-size:.87rem;line-height:1.7">{i}. {sub}</div>',
                    unsafe_allow_html=True,
                )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Use-Case Tracks
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
section_header("3. Use-Case Tracks", "Choose the path that matches your research goal", "🎯")

track_tab1, track_tab2, track_tab3, track_tab4, track_tab5 = st.tabs([
    "🏗️ De novo Design",
    "⚡ Stability Engineering",
    "🧫 Antibody Engineering",
    "🌾 Plant / Wheat Biology",
    "📦 High-Throughput Batch",
])

with track_tab1:
    st.markdown("""
### De novo Protein Design Track

**Goal:** Design a new protein sequence from scratch for a given fold.

#### Recommended path

| Step | Page | What to do |
|------|------|-----------|
| 1 | **Editor (0)** | Sketch the sequence: set length, fix anchor residues, add constraints |
| 2 | **Predict (1)** | Predict fold with ESMFold to check if the design is plausible |
| 3 | **MPNN Lab (8)** | Upload the predicted backbone → generate sequences that fold into it |
| 4 | **Predict (1)** | Validate each MPNN design: does it predict back to the same fold? |
| 5 | **Evaluate (2)** | Check biophysical quality: SASA, clash, pLDDT, instability index |
| 6 | **Evolution (4)** | Run directed evolution on the best design to optimize stability |

#### Tips
- Start with short sequences (50–100 aa) for fast iteration cycles
- pLDDT > 80 and sequence recovery > 35% are good signs
- Use temperature = 0.2 for MPNN on structured proteins, 0.4 for flexible loops
- Run Compare (3) on the final design across ColabFold + ESMFold for confidence
""")

with track_tab2:
    st.markdown("""
### Stability Engineering Track

**Goal:** Improve the thermostability or solubility of an existing protein.

#### Recommended path

| Step | Page | What to do |
|------|------|-----------|
| 1 | **Predict (1)** | Predict baseline structure for your wild-type sequence |
| 2 | **Evaluate (2)** | Identify unstable regions: low pLDDT, high clash score, exposed hydrophobics |
| 3 | **Mutagenesis (10)** | Select 3–5 unstable positions → run saturation scan |
| 4 | **Mutagenesis (10)** | Look for mutations with delta pLDDT > +2 and RMSD < 1.0 Å |
| 5 | **Evolution (4)** | Set fitness: stability 50% + pLDDT 50%; evolve for 20–50 generations |
| 6 | **Batch (5)** | Evaluate top variants from evolution; export for wet-lab validation |

#### Key metrics to optimise
- **Instability index** < 40 = predicted stable
- **GRAVY score** > 0 = tends to aggregate (avoid)
- **pLDDT** > 80 in core = good tertiary structure
- **Tm prediction** (Evaluate page, expander) — target ΔTm > 5°C

#### Multi-mutation design
In the Mutagenesis page → **Multi-Mutation Design** tab, combine top single
mutations into double/triple variants. Sort by improvement score.
""")

with track_tab3:
    st.markdown("""
### Antibody Engineering Track

**Goal:** Annotate, analyse, and engineer an antibody variable domain.

#### Recommended path

| Step | Page | What to do |
|------|------|-----------|
| 1 | **Antibody (12)** | Paste VH, VL-κ/λ, or VHH sequence → CDR annotation + chain type detection |
| 2 | **Antibody (12)** | Check **Immunogenicity** tab: MHC-II T-cell epitope heatmap, APR regions |
| 3 | **Antibody (12)** | Check **Developability** tab: humanness ≥ 85%, pI 6–8, instability < 40 |
| 4 | **Predict (1)** | Structure prediction — use ImmuneBuild for Fv, ESMFold for scFv/VHH |
| 5 | **Mutagenesis (10)** | Scan mutations in high-immunogenicity regions to reduce MHC-II risk |
| 6 | **Antibody (12)** | **Wet-Lab Plan** tab: recommended expression system, purification strategy |

#### Understanding CDR annotation schemes
- **Chothia**: best for structural loops (recommended for modelling)
- **Kabat**: canonical database numbering (best for literature comparison)
- **IMGT**: universal system (best for gene assignment)
- All three are shown simultaneously — use the scheme your target database uses

#### Key thresholds
- **Humanness ≥ 85%** — good germline identity; reduces immunogenicity risk
- **Immunogenicity score < 30** — low clinical risk (heuristic; confirm with NetMHCIIpan)
- **Developability index** — check charge patches, pI, VH/VL interface hydrophobics
- **CDR-H3** — pLDDT < 70 is *normal* (intrinsically flexible loop)

#### Paratope candidates
The Antibody page highlights positions likely involved in antigen contact:
CDR-H3 (all), CDR-H1/H2 (middle 70%), CDR-L1/L3 (all), CDR-L2 (C-terminal 3 aa).
Mutations at these positions change binding, not just stability.
""")

with track_tab4:
    st.markdown("""
### Plant / Wheat Biology Track

**Goal:** Analyse and engineer plant proteins — especially wheat NLR immune receptors.

#### Why wheat proteins need special handling
1. **Hexaploid genome** — T. aestivum has A/B/D homeologous copies
2. **Transit peptides** — chloroplast/mitochondrial TPs must be removed before structure prediction
3. **NLR domains** — TIR-NLR and CC-NLR have expected low pLDDT in linker regions
4. **Codon bias** — wheat codon preferences differ significantly from E. coli or mammalian

#### Recommended path

| Step | Page | What to do |
|------|------|-----------|
| 1 | **Mutagenesis (10)** | Paste NLR sequence → open **🌾 Plant Biology Analysis** expander → Run |
| 2 | | Check transit peptide detection: use mature sequence for all downstream steps |
| 3 | | Check NLR domain annotation: TIR / NBS-ARC / LRR boundaries + critical motifs |
| 4 | | Download wheat-optimized DNA (CAI > 0.8 target) |
| 5 | **Predict (1)** | Predict structure using **mature sequence** (TP removed) |
| 6 | **Evaluate (2)** | Check structure quality — note: TIR–NBS linker pLDDT < 70 is **expected** |
| 7 | **Mutagenesis (10)** | Scan LRR specificity positions (positions 11/12/14 of each repeat) |
| 8 | | NLR column in ranked table: 🔴 = likely LOF, ⚡ = autoactive GOF, 🎯 = altered specificity |
| 9 | **MPNN Lab (8)** | Design variants → enable **🌾 Wheat Codon Optimization** |
| 10 | **Agents (11)** | Select **plant_biology** team for NLR-informed interpretation |

#### NLR motif reference

| Motif | Domain | Critical? | Mutation impact |
|-------|--------|-----------|----------------|
| P-loop GXXXXGKT | NBS | **Yes** | K/T → anything = LOF (ATP binding lost) |
| MHD (last codon D) | ARC2 | **Yes** | D→A/V = GOF autoactive; D→K/R = LOF |
| W-box (Trp) | TIR | **Yes** | W→any = LOF (TIR-TIR signalling blocked) |
| LRR pos 11/12/14 | LRR | Moderate | Changes effector recognition specificity |
| TIR-NBS linker | — | No | pLDDT 50–70 is normal (disordered) |

#### Codon optimization guide

| Species | Tool output | CAI target |
|---------|------------|-----------|
| Wheat (*T. aestivum*) | Wheat-optimized FASTA | > 0.80 |
| Rice (*O. sativa*) | Rice-optimized FASTA | > 0.80 |
| Maize (*Z. mays*) | Maize-optimized FASTA | > 0.80 |

CAI = Codon Adaptation Index. Values > 0.80 predict good expression.
Cryptic splice sites (GT-AG) and poly-A signals (AATAAA) are auto-detected.
""")
    st.markdown("""
<div class="guide-tip">
🌱 <strong>Tip for N. benthamiana transient expression:</strong>
Agroinfiltration uses T-DNA delivery — use the pWBVEC / pGWB binary vector family,
35S or Ubi promoter, and GV3101 Agrobacterium strain. Co-express p19 silencing suppressor.
</div>
""", unsafe_allow_html=True)

with track_tab5:
    st.markdown("""
### High-Throughput Batch Track

**Goal:** Process dozens of sequences in parallel for screening or library evaluation.

#### Recommended path

| Step | Page | What to do |
|------|------|-----------|
| 1 | **Batch (5)** | Upload a multi-FASTA file (one entry per sequence) |
| 2 | | Choose predictor + metrics (pLDDT, instability, GRAVY, pI) |
| 3 | | Run batch — up to 20 sequences with progress tracking |
| 4 | | Review the ranking table and the **🧪 Wet-Lab Go/No-Go** panel |
| 5 | **Mutagenesis (10)** | Promote top sequences for individual saturation scanning |
| 6 | **Evolution (4)** | Promote the best sequence for directed evolution |

#### FASTA format for Batch
```
>protein_1|WT
MVTKEQIKSLQGLRSVK...
>protein_2|A42G
MVTKEQIKSLQGLRSVK...
>protein_3|K58R
MVTKEQIKSLQGLRSVK...
```

#### Batch Library Design
In the **Editor (0)** → **Library Design** tab:
- **Combinatorial**: define positions + allowed AAs → generate up to 2000 variants
- **Degenerate Codons**: NNK, NDT, etc. for DNA-level diversity
- Download as FASTA → upload directly to Batch page

#### Wet-Lab Go/No-Go in Batch
Each sequence gets a **GO / CONDITIONAL / NO-GO** verdict based on:
- Instability index (threshold 40)
- GRAVY score (hydrophobicity)
- Predicted pI (6–8 preferred for most systems)
- MW (< 100 kDa for standard expression)
""")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — All Pages Reference
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
section_header("4. All Pages — Quick Reference", icon="📋")

pages_ref = [
    ("🏠 Home", "app.py", "Dashboard with quick stats, workflow shortcuts, and recent activity."),
    ("🔮 Predict", "pages/1_predict.py",
     "Structure prediction. Supports ESMFold (API), ColabFold, Chai-1, Boltz-2, ESM3, ImmuneBuild. "
     "Input: sequence. Output: PDB + pLDDT per residue + 3D viewer."),
    ("📊 Evaluate", "pages/2_evaluate.py",
     "Structure quality analysis. Metrics: pLDDT, TM-score, RMSD, VoroMQA, QS-score, clash, SASA, "
     "Ramachandran. Optional: PTM liability, Tm prediction, GBSA energy. Export PDF report."),
    ("⚖️ Compare", "pages/3_compare.py",
     "Side-by-side predictor benchmarking. Same sequence → multiple predictors → metric comparison table + 3D overlays."),
    ("📈 Evolution", "pages/4_evolution.py",
     "Directed evolution. GA with 4 fitness functions (stability, solubility, pLDDT, recovery). "
     "Combinatorial library design. Biophysical pre-filter. Wet-lab readiness per variant."),
    ("📦 Batch", "pages/5_batch.py",
     "Multi-sequence processing. FASTA upload → parallel prediction + biophysics → ranked table + Wet-Lab Go/No-Go."),
    ("⚙️ Settings", "pages/6_settings.py",
     "Configure predictors, LLM backend (Ollama/Groq/OpenAI/DeepSeek/etc.), output directory, "
     "Ollama GPU settings, and API keys. Run system diagnostics."),
    ("🧬 MSA", "pages/7_msa.py",
     "Multiple sequence alignment analysis. Upload FASTA or paste aligned sequences. "
     "Conservation plot, entropy heatmap, consensus sequence."),
    ("🎯 MPNN Lab", "pages/8_mpnn.py",
     "ProteinMPNN inverse folding. Upload PDB backbone → design N sequences. "
     "Temperature control, fixed positions, sequence constraints. "
     "Wheat codon optimization output. Send to Predict for validation."),
    ("📁 Jobs", "pages/9_jobs.py",
     "Browse previous job directories. Reload and re-inspect past predictions and analyses. Filter by date/type."),
    ("🧬 Mutagenesis", "pages/10_mutation_scanner.py",
     "Saturation mutagenesis. Phase 1: LLM baseline analysis + position suggestions. "
     "Phase 2: ESMFold saturation scan. Ranked by delta pLDDT + RMSD. NLR-aware flags. "
     "Plant Biology panel: transit peptide + NLR domains + wheat codon optimization. PDF export."),
    ("🤖 Agents", "pages/11_agents.py",
     "LLM-guided full pipeline. 10 specialist personas (PI, Critic, Structural Biologist, Immunologist, "
     "Plant Biologist, Wet Lab Researcher, etc.). Team presets: default, design, nanobody, "
     "antibody, plant_biology, full_pipeline. Per-call timing + GPU usage display."),
    ("🧫 Antibody", "pages/12_antibody.py",
     "Antibody Engineering Workbench. CDR annotation (Chothia/IMGT/Kabat), chain-type detection, "
     "immunogenicity heatmap, developability metrics, Fc engineering guide, "
     "germline assignment, H3 kink detection, wet-lab plan."),
    ("📖 Guide", "pages/13_guide.py",
     "This page. Complete onboarding guide, use-case tracks, page reference, troubleshooting."),
]

for i, (title, target, desc) in enumerate(pages_ref):
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button(title, key=f"ref_btn_{i}", width='stretch'):
            st.switch_page(target)
    with c2:
        st.markdown(
            f'<div style="color:var(--pdhub-text-secondary);font-size:.88rem;'
            f'padding:8px 0;line-height:1.6">{desc}</div>',
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — AI Agent Guide
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
section_header("5. AI Agent Guide", "Getting the most out of the specialist agent system", "🤖")

with st.expander("5.1 — What agents do", expanded=True):
    st.markdown("""
Agents **do not replace computational tools** — they interpret results, catch issues, and suggest next steps.

Every page has two kinds of agent support:
1. **Inline insights** (`render_contextual_insight`) — short automated commentary on current results
2. **Expert panel** (`render_all_experts_panel`) — 3 expert opinions (immunologist, wet-lab, plant biologist) on demand
3. **Agent Pipeline** (page 11) — full multi-step LLM-guided run with 5–12 specialist agents sequentially
""")

with st.expander("5.2 — Team presets"):
    st.markdown("""
| Preset | Agents | Best for |
|--------|--------|---------|
| `default` | PI + Critic + Structural + ML Specialist | General protein engineering |
| `design` | PI + Protein Engineer + Structural + ML | De novo design |
| `nanobody` | PI + Structural + Wet Lab + Protein Eng | VHH / nanobody engineering |
| `evaluation` | Structural + Biophysicist + ML + Critic | Quality assessment focus |
| `mutagenesis` | Protein Eng + Structural + Biophysicist + Critic | Mutation analysis |
| `mpnn_design` | ML Specialist + Protein Eng + Structural | MPNN + ESM design |
| `antibody` | Immunologist + Structural + Wet Lab + Critic | Antibody optimisation |
| `plant_biology` | Plant Biologist + Structural + Wet Lab + Protein Eng | Wheat / NLR work |
| `full_pipeline` | PI + all 5 step agents | Comprehensive analysis |
| `all_experts` | All 10 personas | Maximum coverage |

Select a preset in the **Agents** page before running.
""")

with st.expander("5.3 — LLM performance tips"):
    st.markdown("""
**Speed optimizations already applied:**
- Ollama perf flags: `num_ctx=4096`, `num_batch=512`, `keep_alive=10m`
- Qwen3 models: `think: false` disables chain-of-thought tokens (saves 1–3 s/call)
- Client cached as singleton — no re-initialization between calls
- GPU TTL cache: 60 s for GPU availability check

**Per-call timing is displayed** after each agent completes, e.g.:
```
[Structural Biologist] 2.8s, 310 tok, 111 tok/s
```

**Recommended models by hardware:**
| GPU VRAM | Model | Tokens/s |
|----------|-------|---------|
| 12 GB | qwen2.5:14b (default) — qwen3:14b optional upgrade | ~25–30 tok/s |
| 8 GB | qwen2.5:7b | ~120 tok/s |
| 6 GB | qwen2.5:3b | ~160 tok/s |
| CPU only | llama3.2:3b | ~15 tok/s |

**Cloud alternatives** (no GPU required):
- **Groq** (llama-3.3-70b): ~800 tok/s, free tier available
- **Cerebras**: ~2000 tok/s, free tier available
- **DeepSeek**: cost-effective, high quality
""")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Structure Viewer Guide
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
section_header("6. Structure Viewer Controls", icon="🔬")

with st.expander("3Dmol.js viewer (all pages with structures)"):
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("""
**Mouse controls**
| Action | Gesture |
|--------|---------|
| Rotate | Click + drag |
| Zoom | Scroll wheel |
| Pan | Right-click + drag |
| Select residue | Click on atom |

**Toolbar buttons**
- **Cartoon / Surface / Sticks / Sphere / Ribbon** — change representation
- **pLDDT / Rainbow / Chain / SS** — change colouring
- **⟳ Spin** — toggle auto-rotation
- **⊙ Reset** — return to default view
- **📸** — save PNG screenshot
""")
    with col_v2:
        st.markdown("""
**Colour schemes explained**
- **pLDDT**: blue (>90, very high) → cyan → green → yellow → red (<50, low confidence)
- **Rainbow**: blue (N-terminus) → red (C-terminus)
- **Chain**: different colour per chain (useful for multimers)
- **SS**: secondary structure (helix = pink, sheet = yellow, loop = white)

**PyMOL viewer** (if PyMOL is installed)
- Renders at 720×450 px with anti-aliasing
- ~185 ms rotate, ~400 ms surface
- Accessible at sidebar → System Status → PyMOL port
""")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Metrics Reference
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
section_header("7. Metrics Reference", "What each number means and how to interpret it", "📐")

with st.expander("Structure quality metrics"):
    st.markdown("""
| Metric | Good | Acceptable | Poor | Notes |
|--------|------|-----------|------|-------|
| **pLDDT** | > 90 | 70–90 | < 70 | Per-residue confidence. < 70 in loops is normal. NLR linkers expected < 70 |
| **TM-score** | > 0.9 | 0.7–0.9 | < 0.7 | Structural similarity (0–1). > 0.5 = same fold |
| **RMSD** | < 1.0 Å | 1–2 Å | > 2 Å | Cα deviation. Compare same region only |
| **Clash score** | < 10 | 10–25 | > 25 | Steric clashes per 1000 atoms. MolProbity scale |
| **SASA** | context | context | context | Solvent accessible surface area. High = more exposed |
| **VoroMQA** | > 0.4 | 0.2–0.4 | < 0.2 | Global quality score from Voronoi tessellation |
| **Ramachandran** | > 98% | 95–98% | < 95% | Fraction of residues in allowed backbone regions |
""")

with st.expander("Sequence / biophysical metrics"):
    st.markdown("""
| Metric | Good | Watch | Poor | Notes |
|--------|------|-------|------|-------|
| **Instability index** | < 40 | 40–60 | > 60 | Predicted in vivo stability. > 40 = unstable |
| **GRAVY** | -0.5–0.0 | 0.0–0.5 | > 0.5 | Hydrophobicity. > 0 = aggregation prone |
| **pI** | 6–8 | 5–9 | outside | Isoelectric point. Near-neutral = easier purification |
| **MW** | < 60 kDa | 60–150 kDa | > 150 kDa | Molecular weight. Above 150 kDa = expression challenges |
| **Net charge** | -2 to +2 | ±5 | outside ±10 | At pH 7. Extreme charge causes aggregation |
| **Tm (predicted)** | > 60°C | 45–60°C | < 45°C | Melting temperature. > 60°C for most applications |
| **CAI (DNA)** | > 0.80 | 0.65–0.80 | < 0.65 | Codon Adaptation Index. > 0.8 = good expression |
""")

with st.expander("Immunology / antibody metrics"):
    st.markdown("""
| Metric | Good | Borderline | Flag |
|--------|------|-----------|------|
| **Humanness score** | ≥ 85% | 75–85% | < 75% |
| **Immunogenicity score** | < 30 | 30–60 | > 60 |
| **T-cell epitope density** | < 10% | 10–20% | > 20% |
| **MHC-II epitopes (High)** | 0 | 1 | ≥ 2 |
| **pI (antibody)** | 6.5–8.5 | 5.5–9.5 | outside |
| **CDR-H3 pLDDT** | > 60 | 50–60 | < 50 (or normal if loop is long) |

⚠️ Immunogenicity scores are heuristic (pan-DR PSSM). Confirm high-risk epitopes with NetMHCIIpan or EpiVax.
""")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Troubleshooting
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
section_header("8. Troubleshooting", "Common issues and how to fix them", "🔧")

faqs = [
    ("ESMFold returns 'Service unavailable'",
     "ESMFold uses the public ESM Metagenomic Atlas API which has rate limits. "
     "Wait 30 seconds and retry. For sequences > 400 aa, install local ESMFold: `pip install esm`. "
     "Alternatively switch to ColabFold for local prediction."),
    ("ColabFold is slow",
     "ColabFold runs full MSA search by default. Go to Settings → ColabFold → "
     "set `msa_mode = single_sequence` to skip MSA and run ~10× faster (lower accuracy). "
     "Also set `num_models = 1` and `num_recycles = 1` for quick prototyping."),
    ("LLM agent not responding / 'No LLM configured'",
     "1. Check Settings → LLM Provider. "
     "2. If using Ollama: run `ollama serve` in a terminal and `ollama pull qwen2.5:14b` (or `qwen3:14b` for the optional upgrade). "
     "3. If using a cloud API: paste your key in Settings and save. "
     "4. Confirm by clicking 'Test LLM' in Settings."),
    ("CUDA out of memory when running MPNN / prediction",
     "Reduce the number of sequences (MPNN: set num_sequences ≤ 8). "
     "For prediction, use ESMFold API instead of local ColabFold. "
     "In Settings, enable `use_gpu_relax = false` for ColabFold."),
    ("Plant Biology Analysis shows 'No NLR detected'",
     "1. Make sure you are using the **mature sequence** (TP removed). "
     "2. NLR detection requires ≥ 150 aa with NBS motifs present. "
     "3. The LRR detector needs at least 3 repeat units. "
     "4. If you have a partial domain (just TIR or just LRR), the full architecture will not be detected — this is expected."),
    ("Antibody CDR annotation gives unexpected boundaries",
     "The numbering scheme affects boundaries. "
     "Chothia is used for structure-based analysis; Kabat for database comparison. "
     "If results differ from your expectation, try IMGT (most universal). "
     "Verify chain type detection — VHH (nanobody) uses different boundaries than VH."),
    ("Mutation scanner Phase 2 takes very long",
     "Each position scans 19 mutations × 1 ESMFold call each. "
     "5 positions = 95 predictions ≈ 15–20 minutes on API, 3–5 minutes with local GPU. "
     "Start with 2–3 positions, then expand."),
    ("Docker container can't access GPU",
     "Add NVIDIA container toolkit: `apt-get install nvidia-container-toolkit`. "
     "In docker-compose.yml, add under the service: "
     "`deploy: resources: reservations: devices: - driver: nvidia count: 1 capabilities: [gpu]`"),
    ("'NaN' values in evaluation metrics",
     "This usually means the PDB file is missing atoms or has non-standard residues. "
     "Check the structure in the viewer — if it looks incomplete, re-predict. "
     "Amber relaxation in ColabFold can also fix missing atoms: set `use_amber = true` in Settings."),
    ("How do I export results?",
     "Each page has Download buttons: "
     "Evaluate → PDF report; "
     "Mutagenesis → CSV table + PDF; "
     "MPNN → FASTA of designed sequences; "
     "Plant Biology → mature sequence FASTA + wheat-optimized DNA FASTA. "
     "All predicted structures are saved to `outputs/` and accessible via the Jobs page."),
]

for q, a in faqs:
    with st.expander(f"❓ {q}"):
        st.markdown(f'<div style="color:var(--pdhub-text-secondary);font-size:.9rem;line-height:1.7">{a}</div>',
                    unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Example sequences to try right now
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
section_header("9. Try It Now — Example Sequences", icon="▶️")

examples = {
    "Human Ubiquitin (76 aa)": {
        "seq": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "note": "Classic stability benchmark. Compact β-grasp fold. pLDDT > 90 expected.",
    },
    "T1024 miniprotein (52 aa)": {
        "seq": "MAAHKGAEHVVKASLDAGVKTVAGGLVVKAKALGGKDATMHLVAATLKKGYM",
        "note": "De novo designed miniprotein. Ideal for testing design tools.",
    },
    "VHH nanobody domain (~120 aa)": {
        "seq": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVGAYVMDYWGQGTLVTVSS",
        "note": "VHH single-domain antibody. Use Antibody page for CDR annotation.",
    },
    "CC-NLR fragment (NBS region, ~100 aa)": {
        "seq": "MAEDAEAKRAAELEKLHRELRAKEVAKDQVSQLKAEVEELKSMLEAKELAEAKDAAVSLQGAVSGKTALSTIRGRLIEDRGKIGVIVDDDAEMFKAIRSAFDAAERVSQIRKLVPALDDFQKISELL",
        "note": "Plant NLR NBS domain fragment. Try the Plant Biology panel in Mutagenesis.",
    },
}

cols = st.columns(2)
for i, (name, info) in enumerate(examples.items()):
    with cols[i % 2]:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(info["note"])
            st.code(info["seq"], language=None)
            c_pred, c_mut = st.columns(2)
            with c_pred:
                if st.button("→ Predict", key=f"ex_pred_{i}", width='stretch'):
                    st.session_state["predict_sequence"] = info["seq"]
                    st.session_state["predict_name"] = name
                    st.switch_page("app_pages/1_predict.py")
            with c_mut:
                if st.button("→ Mutagenesis", key=f"ex_mut_{i}", width='stretch'):
                    st.session_state["sequence"] = info["seq"]
                    st.session_state["sequence_name"] = name
                    st.switch_page("app_pages/10_mutation_scanner.py")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:var(--pdhub-text-muted);font-size:.78rem;padding:12px">'
    'Protein Design Hub — Integrated computational biology platform<br>'
    'For issues and feedback: open a GitHub issue in the repository'
    '</div>',
    unsafe_allow_html=True,
)
