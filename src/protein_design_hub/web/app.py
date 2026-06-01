"""Protein Design Hub -- Home / Dashboard page."""

import sys, json, time
from pathlib import Path
from datetime import datetime

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st

from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    section_header,
    metric_card,
    info_box,
    detect_gpu,
    workflow_breadcrumb,
)
from protein_design_hub.web.agent_helpers import agent_sidebar_status

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Protein Design Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_base_css()
sidebar_nav(current="Home")
sidebar_system_status()
agent_sidebar_status()

# ── Hero header ──────────────────────────────────────────────────────
page_header(
    "Protein Design Hub",
    "Integrated platform for protein structure prediction, analysis, and design",
    "🧬",
)

# ── Quick Stats Row ──────────────────────────────────────────────────
gpu_info = detect_gpu()
gpu_available = gpu_info["available"]

try:
    from protein_design_hub.predictors.registry import PredictorRegistry
    num_predictors = len(PredictorRegistry.list_available())
except Exception:
    num_predictors = 0

job_dir = Path("outputs")
try:
    from protein_design_hub.core.config import get_settings
    settings = get_settings()
    job_dir = Path(settings.output.base_dir)
    job_dirs = [d for d in job_dir.iterdir() if d.is_dir()] if job_dir.exists() else []
    num_jobs = len(job_dirs)
except Exception:
    num_jobs = 0
    job_dirs = []

# Count structures
try:
    from protein_design_hub.web.ui import list_output_structures
    structures = list_output_structures(job_dir) if job_dir.exists() else []
    num_structures = len(structures)
except Exception:
    num_structures = 0
    structures = []

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card(num_predictors, "Predictors Available", "info", "🔮")
with c2:
    metric_card(num_jobs, "Jobs Completed", "success", "📁")
with c3:
    metric_card(num_structures, "Structures", "gradient", "🏗️")
with c4:
    metric_card("GPU" if gpu_available else "CPU", "Compute Mode", "gradient" if gpu_available else "warning", "⚡")
with c5:
    metric_card("v0.3", "Platform Version", "default", "📦")

# ── Workflow Hub ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_header("Workflows", "Launch a protein analysis pipeline", "🚀")

workflow_breadcrumb(
    ["Sequence Input", "Predict", "Evaluate", "Refine / Design", "Export"],
    current=0,
)

col_w1, col_w2, col_w3 = st.columns(3)

with col_w1:
    with st.container(border=True):
        st.markdown("#### 🔮 Structure Prediction")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem;margin-bottom:.75rem">'
            "Predict 3D structures from amino acid sequences using ESMFold, ColabFold, Chai-1, Boltz-2, and more.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Start Prediction", key="h_predict", type="primary", use_container_width=True):
            st.switch_page("pages/1_predict.py")

with col_w2:
    with st.container(border=True):
        st.markdown("#### 📊 Structure Evaluation")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem;margin-bottom:.75rem">'
            "Analyze structures with biophysical metrics: clash score, SASA, contact energy, Ramachandran analysis, and more.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Evaluate Structure", key="h_eval", type="primary", use_container_width=True):
            st.switch_page("pages/2_evaluate.py")

with col_w3:
    with st.container(border=True):
        st.markdown("#### ⚖️ Compare Predictors")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem;margin-bottom:.75rem">'
            "Benchmark multiple predictors on the same sequence. Side-by-side metric comparison with detailed analysis.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Compare", key="h_compare", type="primary", use_container_width=True):
            st.switch_page("pages/3_compare.py")

# ── Design & Engineering ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_header("Protein Engineering", "Design and optimize protein sequences", "🎯")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    with st.container(border=True):
        st.markdown("#### ✏️ Sequence Editor")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem;margin-bottom:.75rem">'
            "Interactive residue editor with amino acid property coloring, constraint management, and ligand support.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Open Editor", key="h_editor", use_container_width=True):
            st.switch_page("pages/0_design.py")

with col_d2:
    with st.container(border=True):
        st.markdown("#### 🧬 Mutation Scanner")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem;margin-bottom:.75rem">'
            "Saturation mutagenesis to identify beneficial mutations and evaluate stability impacts.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Scan Mutations", key="h_scan", use_container_width=True):
            st.switch_page("pages/10_mutation_scanner.py")

with col_d3:
    with st.container(border=True):
        st.markdown("#### 🎯 ProteinMPNN Lab")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem;margin-bottom:.75rem">'
            "Design novel sequences for fixed backbone structures using ProteinMPNN deep learning.</p>",
            unsafe_allow_html=True,
        )
        if st.button("MPNN Design", key="h_mpnn", use_container_width=True):
            st.switch_page("pages/8_mpnn.py")

# ── Agent Pipeline ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_header("AI Agent Pipeline", "LLM-guided analysis with specialist agents", "🤖")

col_a1, col_a2 = st.columns(2)

with col_a1:
    with st.container(border=True):
        st.markdown("#### 🤖 LLM-Guided Pipeline")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem">'
            "Run the full prediction pipeline with AI scientist agents that plan, "
            "review predictions, and interpret results. "
            "Powered by Ollama, DeepSeek, OpenAI, or any compatible backend.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Open Agent Pipeline", key="h_agents", type="primary", use_container_width=True):
            st.switch_page("pages/11_agents.py")

with col_a2:
    with st.container(border=True):
        st.markdown("#### 💬 Expert Meetings")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem">'
            "Run ad-hoc team or individual meetings with AI agents. "
            "Ask about design strategy, predictor selection, "
            "structure refinement, and quality assessment.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Start Meeting", key="h_meeting", use_container_width=True):
            st.switch_page("pages/11_agents.py")

# ── Additional Tools Grid ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_header("Tools & Utilities", "Specialized analysis capabilities", "🔧")

ct1, ct2, ct3, ct4 = st.columns(4)

with ct1:
    with st.container(border=True):
        st.markdown("**📈 Directed Evolution**")
        st.caption("Iterative optimization with fitness tracking")
        if st.button("Evolve", key="t_evolve", use_container_width=True):
            st.switch_page("pages/4_evolution.py")

with ct2:
    with st.container(border=True):
        st.markdown("**📦 Batch Processing**")
        st.caption("Process multiple sequences in parallel")
        if st.button("Batch", key="t_batch", use_container_width=True):
            st.switch_page("pages/5_batch.py")

with ct3:
    with st.container(border=True):
        st.markdown("**🧬 MSA & Phylogeny**")
        st.caption("Multiple sequence alignment analysis")
        if st.button("MSA", key="t_msa", use_container_width=True):
            st.switch_page("pages/7_msa.py")

with ct4:
    with st.container(border=True):
        st.markdown("**📁 Job Browser**")
        st.caption("Browse and reload past results")
        if st.button("Jobs", key="t_jobs", use_container_width=True):
            st.switch_page("pages/9_jobs.py")

# ── Getting Started ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_header("Getting Started", "New to the platform? Start here.", "📖")

col_gs1, col_gs2, col_gs3 = st.columns(3)
with col_gs1:
    with st.container(border=True):
        st.markdown("#### 📖 Full User Guide")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem;margin-bottom:.75rem">'
            "Step-by-step onboarding: installation, all workflows, use-case tracks "
            "(antibody, plant/wheat, de novo design), metrics reference, and troubleshooting.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Open Guide", key="h_guide", type="primary", use_container_width=True):
            st.switch_page("pages/13_guide.py")

with col_gs2:
    with st.container(border=True):
        st.markdown("#### 🧫 Antibody Engineering")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem;margin-bottom:.75rem">'
            "CDR annotation (Chothia/IMGT/Kabat), immunogenicity profiling, "
            "developability metrics, germline assignment, and wet-lab planning.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Antibody Workbench", key="h_antibody", use_container_width=True):
            st.switch_page("pages/12_antibody.py")

with col_gs3:
    with st.container(border=True):
        st.markdown("#### 🌾 Plant / Wheat Biology")
        st.markdown(
            '<p style="color:var(--pdhub-text-secondary);font-size:.88rem;margin-bottom:.75rem">'
            "Transit peptide detection, NLR domain annotation (TIR/NBS/LRR), "
            "wheat codon optimization (CAI), and NLR mutation impact flags.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Plant Biology Tools", key="h_plant", use_container_width=True):
            st.switch_page("pages/10_mutation_scanner.py")

# ── Recent Activity ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_header("Recent Activity", "Latest predictions and analyses", "📋")

if num_jobs == 0:
    info_box(
        "No jobs yet. Start a prediction to see activity here.",
        variant="info",
        icon="📭",
    )
else:
    recent_jobs = sorted(job_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[:6]
    cols = st.columns(3)
    for i, jdir in enumerate(recent_jobs):
        with cols[i % 3]:
            mtime = datetime.fromtimestamp(jdir.stat().st_mtime)
            age = datetime.now() - mtime
            if age.days > 0:
                age_str = f"{age.days}d ago"
            elif age.seconds > 3600:
                age_str = f"{age.seconds // 3600}h ago"
            else:
                age_str = f"{age.seconds // 60}m ago"

            has_pred = (jdir / "prediction_summary.json").exists()
            has_eval = (jdir / "evaluation").exists()
            tags = []
            if has_pred:
                tags.append("prediction")
            if has_eval:
                tags.append("evaluation")
            tag_str = " ".join(f'<span style="background:rgba(99,102,241,.12);color:var(--pdhub-primary-light);padding:2px 8px;border-radius:8px;font-size:.7rem;font-weight:600">{t}</span>' for t in tags)

            st.markdown(
                f'<div style="background:var(--pdhub-bg-card);border:1px solid var(--pdhub-border);'
                f'border-radius:12px;padding:14px 16px;margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-weight:600;font-size:.88rem;color:var(--pdhub-text)">{jdir.name[:30]}</span>'
                f'<span style="color:var(--pdhub-text-muted);font-size:.75rem">{age_str}</span></div>'
                f'<div style="margin-top:6px;display:flex;gap:6px">{tag_str}</div></div>',
                unsafe_allow_html=True,
            )

# ── System Status Footer ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

status_color = "#22c55e" if gpu_info["available"] else "#f59e0b"
status_text = f"GPU Active: {gpu_info['name']}" if gpu_info["available"] else "Running in CPU mode"

st.markdown(
    f'<div style="background:var(--pdhub-bg-card);border:1px solid var(--pdhub-border);'
    f'padding:.75rem 1.25rem;border-radius:12px;display:flex;align-items:center;justify-content:center;gap:12px">'
    f'<span style="width:10px;height:10px;background:{status_color};border-radius:50%"></span>'
    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.82rem;color:var(--pdhub-text-secondary)">{status_text}</span>'
    f'<span style="color:var(--pdhub-text-muted);font-size:.75rem;margin-left:auto">Protein Design Hub v0.3</span></div>',
    unsafe_allow_html=True,
)
