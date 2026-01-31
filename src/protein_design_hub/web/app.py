import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st
import torch

# Shared UI helpers
from protein_design_hub.web.ui import (
    inject_base_css, 
    sidebar_nav, 
    sidebar_system_status,
    page_header,
    card_start,
    card_end,
    metric_card,
    status_badge,
    render_badge
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
    "PDHUB PRO",
    "THE SYNTHETIC BIOLOGY COMMAND CENTER // VERSION 0.2.2",
    ""
)

# 1. MISSION CONTROL (Primary Workflow)
st.markdown("<h2 style='text-align: center; font-weight: 800; letter-spacing: 0.1em; color: #64748b; margin-top: 2rem;'>MISSION CONTROL</h2>", unsafe_allow_html=True)

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

with col_ctrl1:
    with st.container(border=True):
        st.markdown("### 🔮 FOLD")
        st.caption("AI Structure Prediction")
        if st.button("LAUNCH PREDICTOR", key="h_fold", type="primary", use_container_width=True):
            st.switch_page("pages/1_predict.py")

with col_ctrl2:
    with st.container(border=True):
        st.markdown("### 📊 ANALYZE")
        st.caption("Biophysical Evaluation")
        if st.button("LAUNCH EVALUATOR", key="h_eval", type="primary", use_container_width=True):
            st.switch_page("pages/2_evaluate.py")

with col_ctrl3:
    with st.container(border=True):
        st.markdown("### 📁 ARCHIVE")
        st.caption("Project Jobs & History")
        if st.button("OPEN REPOSITORY", key="h_jobs", type="primary", use_container_width=True):
            st.switch_page("pages/9_jobs.py")

# 2. RESEARCH SUITE (Advanced Bento)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("### 🔬 Specialized Laboratoreis")

b_row1_col1, b_row1_col2 = st.columns([1, 1])

with b_row1_col1:
    with st.container(border=True):
        col_img, col_txt = st.columns([1, 2])
        with col_img:
            st.markdown("<div style='font-size: 5rem; text-align: center;'>🧬</div>", unsafe_allow_html=True)
        with col_txt:
            st.markdown("#### Deep Mutagenesis Scanner")
            st.markdown("In-silico scanning of beneficial mutations using enterprise-grade structural models.")
            if st.button("Enter Scanner", key="b_scan", use_container_width=True):
                st.switch_page("pages/10_mutation_scanner.py")

with b_row1_col2:
    with st.container(border=True):
        col_img, col_txt = st.columns([1, 2])
        with col_img:
            st.markdown("<div style='font-size: 5rem; text-align: center;'>🎯</div>", unsafe_allow_html=True)
        with col_txt:
            st.markdown("#### ProteinMPNN Studio")
            st.markdown("Professional de-novo sequence design for designated backbone architectures.")
            if st.button("Enter Studio", key="b_mpnn", use_container_width=True):
                st.switch_page("pages/8_mpnn.py")

# 3. TECHNICAL INFRASTRUCTURE
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🛰️ System Infrastructure")

inf_col1, inf_col2, inf_col3, inf_col4 = st.columns(4)

with inf_col1:
    with st.container(border=True):
        st.markdown("**⚖️ Benchmark**")
        if st.button("Compare", key="inf_compare", use_container_width=True):
            st.switch_page("pages/3_compare.py")

with inf_col2:
    with st.container(border=True):
        st.markdown("**📈 Evolution**")
        if st.button("Evolve", key="inf_evolve", use_container_width=True):
            st.switch_page("pages/4_evolution.py")

with inf_col3:
    with st.container(border=True):
        st.markdown("**📦 Batch**")
        if st.button("Batch", key="inf_batch", use_container_width=True):
            st.switch_page("pages/5_batch.py")

with inf_col4:
    with st.container(border=True):
        st.markdown("**⚙️ Config**")
        if st.button("Settings", key="inf_settings", use_container_width=True):
            st.switch_page("pages/6_settings.py")

# System Pulse
st.markdown("<br>", unsafe_allow_html=True)
try:
    import torch
    pulse_color = "#22d3ee" if torch.cuda.is_available() else "#f43f5e"
    pulse_text = "NVIDIA HPC CLUSTER ONLINE" if torch.cuda.is_available() else "CPU FALLBACK ACTIVE"
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 20px; display: flex; align-items: center; justify-content: center; gap: 15px;">
        <span style="width: 12px; height: 12px; background: {pulse_color}; border-radius: 50%; box-shadow: 0 0 15px {pulse_color}; animation: pulse 2s infinite;"></span>
        <span style="font-family: 'JetBrains Mono'; font-size: 0.8rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.1em;">{pulse_text}</span>
    </div>
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba({pulse_color}, 0.7); }}
        70% {{ transform: scale(1); box-shadow: 0 0 0 10px rgba({pulse_color}, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba({pulse_color}, 0); }}
    }}
    </style>
    """, unsafe_allow_html=True)
except Exception:
    pass

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #334155; font-size: 0.7rem; font-family: 'JetBrains Mono'; letter-spacing: 0.4em; padding: 4rem 0;">
        DESIGNED FOR EXCELLENCE // REPRODUCIBILITY GUARANTEED
    </div>
    """,
    unsafe_allow_html=True,
)
