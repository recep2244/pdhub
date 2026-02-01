import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st

# Shared UI helpers
from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    section_header,
    metric_card,
    info_box,
)

# Page configuration
st.set_page_config(
    page_title="Protein Design Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_base_css()

sidebar_nav(current="Home")
sidebar_system_status()

# Main page content
page_header(
    "Protein Design Hub",
    "Integrated platform for protein structure prediction, analysis, and design",
    "🧬"
)

# Quick Stats - Use robust GPU detection
from protein_design_hub.web.ui import detect_gpu
gpu_info = detect_gpu()
gpu_available = gpu_info["available"]

try:
    from protein_design_hub.predictors.registry import PredictorRegistry
    num_predictors = len(PredictorRegistry.list_available())
except Exception:
    num_predictors = 0

try:
    from protein_design_hub.core.config import get_settings
    settings = get_settings()
    job_dir = Path(settings.output.base_dir)
    num_jobs = len([d for d in job_dir.iterdir() if d.is_dir()]) if job_dir.exists() else 0
except Exception:
    num_jobs = 0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    metric_card(num_predictors, "Predictors Available", "info", "🔮")
with col_m2:
    metric_card(num_jobs, "Jobs Completed", "success", "📁")
with col_m3:
    metric_card("GPU" if gpu_available else "CPU", "Compute Mode", "gradient" if gpu_available else "warning", "⚡")
with col_m4:
    metric_card("v0.3", "Version", "default", "📦")

# Primary Workflows
st.markdown("<br>", unsafe_allow_html=True)
section_header("Quick Start", "Primary workflows for protein analysis", "🚀")

col_w1, col_w2, col_w3 = st.columns(3)

with col_w1:
    with st.container(border=True):
        st.markdown("### 🔮 Structure Prediction")
        st.markdown(
            '<p style="color: var(--pdhub-text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">'
            "Predict 3D structures from amino acid sequences using state-of-the-art AI models including ESMFold, ColabFold, and more."
            "</p>",
            unsafe_allow_html=True,
        )
        if st.button("Start Prediction", key="h_predict", type="primary", use_container_width=True):
            st.switch_page("pages/1_predict.py")

with col_w2:
    with st.container(border=True):
        st.markdown("### 📊 Structure Evaluation")
        st.markdown(
            '<p style="color: var(--pdhub-text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">'
            "Analyze predicted structures with comprehensive biophysical metrics: energy scores, clash detection, Ramachandran analysis."
            "</p>",
            unsafe_allow_html=True,
        )
        if st.button("Evaluate Structure", key="h_eval", type="primary", use_container_width=True):
            st.switch_page("pages/2_evaluate.py")

with col_w3:
    with st.container(border=True):
        st.markdown("### 📁 Job Browser")
        st.markdown(
            '<p style="color: var(--pdhub-text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">'
            "Browse completed predictions, designs, and analyses. Load previous results for further exploration and comparison."
            "</p>",
            unsafe_allow_html=True,
        )
        if st.button("Browse Jobs", key="h_jobs", type="primary", use_container_width=True):
            st.switch_page("pages/9_jobs.py")

# Design Tools
st.markdown("<br>", unsafe_allow_html=True)
section_header("Design Tools", "Advanced protein engineering capabilities", "🎯")

col_d1, col_d2 = st.columns(2)

with col_d1:
    with st.container(border=True):
        st.markdown("### 🧬 Mutation Scanner")
        st.markdown(
            '<p style="color: var(--pdhub-text-secondary); font-size: 0.9rem;">'
            "Perform saturation mutagenesis to identify beneficial mutations. Scan positions systematically "
            "and evaluate stability impacts using structural predictions."
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Open Scanner", key="h_scan", use_container_width=True):
            st.switch_page("pages/10_mutation_scanner.py")

with col_d2:
    with st.container(border=True):
        st.markdown("### 🎯 ProteinMPNN Design")
        st.markdown(
            '<p style="color: var(--pdhub-text-secondary); font-size: 0.9rem;">'
            "Design novel sequences for fixed backbone structures using ProteinMPNN neural network. "
            "Configure sampling parameters and design constraints."
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Open MPNN Lab", key="h_mpnn", use_container_width=True):
            st.switch_page("pages/8_mpnn.py")

# Additional Tools Grid
st.markdown("<br>", unsafe_allow_html=True)
section_header("Additional Tools", "Specialized analysis and utilities", "🔧")

col_t1, col_t2, col_t3, col_t4 = st.columns(4)

with col_t1:
    with st.container(border=True):
        st.markdown("**⚖️ Compare**")
        st.caption("Benchmark predictors")
        if st.button("Compare", key="t_compare", use_container_width=True):
            st.switch_page("pages/3_compare.py")

with col_t2:
    with st.container(border=True):
        st.markdown("**📈 Evolution**")
        st.caption("Directed evolution")
        if st.button("Evolve", key="t_evolve", use_container_width=True):
            st.switch_page("pages/4_evolution.py")

with col_t3:
    with st.container(border=True):
        st.markdown("**📦 Batch**")
        st.caption("Batch processing")
        if st.button("Batch", key="t_batch", use_container_width=True):
            st.switch_page("pages/5_batch.py")

with col_t4:
    with st.container(border=True):
        st.markdown("**🧬 MSA**")
        st.caption("Sequence alignment")
        if st.button("MSA", key="t_msa", use_container_width=True):
            st.switch_page("pages/7_msa.py")

# System Status
st.markdown("<br>", unsafe_allow_html=True)

# Use robust GPU detection for footer status
if gpu_info["available"]:
    status_color = "#22c55e"
    status_text = f"GPU Active: {gpu_info['name']}"
else:
    status_color = "#f59e0b"
    status_text = "Running in CPU mode"

st.markdown(f"""
<div style="background: var(--pdhub-bg-card); border: 1px solid var(--pdhub-border); padding: 1rem 1.5rem; border-radius: var(--pdhub-border-radius-md); display: flex; align-items: center; justify-content: center; gap: 12px;">
    <span style="width: 10px; height: 10px; background: {status_color}; border-radius: 50%;"></span>
    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--pdhub-text-secondary);">{status_text}</span>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: var(--pdhub-text-muted); font-size: 0.75rem; padding: 2rem 0;">
        Protein Design Hub • Computational Biology Platform
    </div>
    """,
    unsafe_allow_html=True,
)
