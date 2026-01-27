"""Main Streamlit application for Protein Design Hub."""

import streamlit as st

# Shared UI helpers
from protein_design_hub.web.ui import inject_base_css, sidebar_nav, sidebar_system_status

# Page configuration
st.set_page_config(
    page_title="Protein Design Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .status-ok {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)
inject_base_css()

sidebar_nav(current="Home")
sidebar_system_status()

# Main page content
st.markdown('<p class="main-header">🧬 Protein Design Hub</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Unified protein structure prediction and evaluation</p>',
    unsafe_allow_html=True,
)

# Quick start guide
st.markdown("---")
st.markdown("### Quick Start")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### 1️⃣ Predict / Design")
    st.markdown(
        """
    Predict structures from sequences, or design sequences for a fixed backbone.
    """
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Predict", key="go_predict"):
            st.switch_page("pages/1_predict.py")
    with col_b:
        if st.button("ProteinMPNN", key="go_mpnn"):
            st.switch_page("pages/6_mpnn.py")

with col2:
    st.markdown("#### 2️⃣ Evaluate")
    st.markdown(
        """
    Analyze predicted structures with quality metrics
    like lDDT, TM-score, QS-score, and RMSD.
    """
    )
    if st.button("Go to Evaluate", key="go_evaluate"):
        st.switch_page("pages/2_evaluate.py")

with col3:
    st.markdown("#### 3️⃣ Compare")
    st.markdown(
        """
    Run all predictors and compare results to find
    the best prediction for your protein.
    """
    )
    if st.button("Go to Compare", key="go_compare"):
        st.switch_page("pages/3_compare.py")

with col4:
    st.markdown("#### 4️⃣ Jobs")
    st.markdown(
        """
    Browse recent outputs and jump back into Evaluate or MPNN.
    """
    )
    if st.button("Go to Jobs", key="go_jobs"):
        st.switch_page("pages/7_jobs.py")

# System status
st.markdown("---")
st.markdown("### System Status")

try:
    from protein_design_hub.predictors.registry import PredictorRegistry
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    # Predictor status
    col1, col2, col3 = st.columns(3)

    predictors = ["colabfold", "chai1", "boltz2", "esmfold", "esmfold_api"]
    columns = st.columns(3)

    for i, pred_name in enumerate(predictors):
        col = columns[i % 3]
        with col:
            try:
                predictor = PredictorRegistry.get(pred_name, settings)
                status = predictor.get_status()

                if status["installed"]:
                    st.success(f"**{pred_name.upper()}**: Installed")
                    st.caption(f"Version: {status.get('version', 'unknown')}")
                else:
                    st.error(f"**{pred_name.upper()}**: Not Installed")
            except Exception as e:
                st.warning(f"**{pred_name.upper()}**: Error checking status")

    # GPU status
    st.markdown("#### GPU Status")
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.success(f"**GPU Available**: {device_name} ({memory:.1f} GB)")
        else:
            st.warning("**GPU**: Not available - predictions will be slow")
    except ImportError:
        st.error("**PyTorch**: Not installed")

except Exception as e:
    st.error(f"Error loading system status: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        Protein Design Hub v0.1.0 |
        Integrating ColabFold, Chai-1, Boltz-2 & OpenStructure
    </div>
    """,
    unsafe_allow_html=True,
)
