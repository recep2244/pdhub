"""Main Streamlit application entry point for Protein Design Hub."""

import streamlit as st
from pathlib import Path

from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    metric_card,
    section_header,
    info_box,
)

st.set_page_config(
    page_title="Protein Design Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject base CSS theme
inject_base_css()

# Additional page-specific CSS
st.markdown(
    """
<style>
/* Main container */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* Hero section - using theme variables */
.home-hero {
    background: var(--pdhub-gradient);
    border-radius: var(--pdhub-border-radius-xl);
    padding: 50px 40px;
    color: white;
    text-align: center;
    margin-bottom: 40px;
    box-shadow: var(--pdhub-shadow-lg);
    position: relative;
    overflow: hidden;
}

.home-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -30%;
    width: 80%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
    pointer-events: none;
}

.home-hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    margin-bottom: 15px;
    position: relative;
    z-index: 1;
}

.home-hero-subtitle {
    font-size: 1.3rem;
    opacity: 0.95;
    max-width: 700px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

/* Workflow cards */
.workflow-card {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-lg);
    padding: 28px;
    box-shadow: var(--pdhub-shadow-sm);
    border: 1px solid var(--pdhub-border);
    transition: var(--pdhub-transition);
    height: 100%;
    cursor: pointer;
}

.workflow-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--pdhub-shadow-lg);
    border-color: var(--pdhub-primary);
}

.workflow-icon {
    font-size: 3rem;
    margin-bottom: 15px;
    display: block;
}

.workflow-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--pdhub-text);
    margin-bottom: 12px;
}

.workflow-desc {
    color: var(--pdhub-text-secondary);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Status cards */
.status-card {
    background: var(--pdhub-bg-gradient);
    border-radius: var(--pdhub-border-radius-md);
    padding: 24px;
    text-align: center;
    border: 1px solid var(--pdhub-border);
    transition: var(--pdhub-transition);
}

.status-card:hover {
    box-shadow: var(--pdhub-shadow-md);
}

.status-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: var(--pdhub-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.status-label {
    color: var(--pdhub-text-secondary);
    font-size: 0.9rem;
    margin-top: 4px;
}

/* Tool badges */
.tool-badge {
    display: inline-flex;
    align-items: center;
    padding: 8px 16px;
    border-radius: var(--pdhub-border-radius-full);
    font-size: 0.85rem;
    margin: 4px;
    font-weight: 500;
    transition: var(--pdhub-transition);
}

.tool-badge-ok {
    background: var(--pdhub-success-light);
    color: #155724;
    border: 1px solid #c3e6cb;
}

.tool-badge-missing {
    background: var(--pdhub-error-light);
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.tool-badge-warn {
    background: var(--pdhub-warning-light);
    color: #856404;
    border: 1px solid #ffeeba;
}

/* Quick start steps */
.step-card {
    display: flex;
    align-items: flex-start;
    padding: 18px;
    background: var(--pdhub-bg-light);
    border-radius: var(--pdhub-border-radius-md);
    margin: 12px 0;
    transition: var(--pdhub-transition);
    border: 1px solid transparent;
}

.step-card:hover {
    border-color: var(--pdhub-primary-light);
    box-shadow: var(--pdhub-shadow-sm);
}

.step-number {
    background: var(--pdhub-gradient);
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    margin-right: 16px;
    flex-shrink: 0;
    font-size: 1.1rem;
}

.step-content {
    flex: 1;
}

.step-title {
    font-weight: 600;
    color: var(--pdhub-text);
    margin-bottom: 6px;
    font-size: 1rem;
}

.step-desc {
    color: var(--pdhub-text-secondary);
    font-size: 0.9rem;
    line-height: 1.5;
}

/* Feature grid */
.feature-item {
    padding: 18px;
    border-left: 4px solid var(--pdhub-primary);
    background: var(--pdhub-bg-light);
    margin: 12px 0;
    border-radius: 0 var(--pdhub-border-radius-md) var(--pdhub-border-radius-md) 0;
    transition: var(--pdhub-transition);
}

.feature-item:hover {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left-color: var(--pdhub-primary-dark);
}

/* Pipeline diagram */
.pipeline-container {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 15px;
    padding: 30px;
    background: var(--pdhub-bg-gradient);
    border-radius: var(--pdhub-border-radius-lg);
    border: 1px solid var(--pdhub-border);
}

.pipeline-step {
    background: var(--pdhub-bg-card);
    padding: 16px 24px;
    border-radius: var(--pdhub-border-radius-md);
    box-shadow: var(--pdhub-shadow-sm);
    text-align: center;
    min-width: 110px;
    transition: var(--pdhub-transition);
    border: 1px solid var(--pdhub-border);
}

.pipeline-step:hover {
    transform: translateY(-3px);
    box-shadow: var(--pdhub-shadow-md);
    border-color: var(--pdhub-primary-light);
}

.pipeline-arrow {
    color: var(--pdhub-primary);
    font-size: 1.8rem;
    font-weight: bold;
}

/* Section title styling */
.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--pdhub-text);
    margin: 40px 0 20px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 2px;
    background: var(--pdhub-border);
    margin-left: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: var(--pdhub-text-secondary);
    padding: 30px;
    margin-top: 40px;
    border-top: 1px solid var(--pdhub-border);
}

.footer a {
    color: var(--pdhub-primary);
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}
</style>
""",
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
<div class="home-hero">
    <div class="home-hero-title">🧬 Protein Design Hub</div>
    <div class="home-hero-subtitle">
        Unified platform for protein structure prediction, sequence design, and comprehensive evaluation
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# System Status Overview
st.markdown('<div class="section-title">📊 System Status</div>', unsafe_allow_html=True)


def get_predictor_count():
    """Count installed predictors."""
    try:
        from protein_design_hub.core.config import get_settings
        from protein_design_hub.predictors.registry import PredictorRegistry

        settings = get_settings()
        installed = 0
        total = 0
        for name in PredictorRegistry.list_available():
            total += 1
            try:
                pred = PredictorRegistry.get(name, settings)
                if pred.installer.is_installed():
                    installed += 1
            except Exception:
                pass
        return installed, total
    except Exception:
        return 0, 5


def get_designer_count():
    """Count installed designers."""
    try:
        from protein_design_hub.core.config import get_settings
        from protein_design_hub.design.registry import DesignerRegistry

        settings = get_settings()
        installed = 0
        total = 0
        for name in DesignerRegistry.list_available():
            total += 1
            try:
                d = DesignerRegistry.get(name, settings)
                if d.installer.is_installed():
                    installed += 1
            except Exception:
                pass
        return installed, total
    except Exception:
        return 0, 2


def get_gpu_status():
    """Check GPU availability."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), True
        return "Not available", False
    except Exception:
        return "PyTorch not installed", False


def get_job_count():
    """Count recent jobs."""
    try:
        from protein_design_hub.core.config import get_settings

        settings = get_settings()
        output_dir = Path(settings.output.base_dir)
        if output_dir.exists():
            return len([d for d in output_dir.iterdir() if d.is_dir()])
        return 0
    except Exception:
        return 0


pred_installed, pred_total = get_predictor_count()
design_installed, design_total = get_designer_count()
gpu_name, gpu_available = get_gpu_status()
job_count = get_job_count()

col_status1, col_status2, col_status3, col_status4 = st.columns(4)

with col_status1:
    st.markdown(
        f"""
    <div class="status-card">
        <div class="status-value">{pred_installed}/{pred_total}</div>
        <div class="status-label">Predictors Ready</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_status2:
    st.markdown(
        f"""
    <div class="status-card">
        <div class="status-value">{design_installed}/{design_total}</div>
        <div class="status-label">Design Tools Ready</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_status3:
    gpu_icon = "✅" if gpu_available else "⚠️"
    st.markdown(
        f"""
    <div class="status-card">
        <div class="status-value">{gpu_icon}</div>
        <div class="status-label">GPU: {gpu_name[:20] if gpu_available else 'N/A'}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_status4:
    st.markdown(
        f"""
    <div class="status-card">
        <div class="status-value">{job_count}</div>
        <div class="status-label">Jobs Completed</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Main Workflows
st.markdown('<div class="section-title">🚀 Workflows</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
    <div class="workflow-card">
        <span class="workflow-icon">🎨</span>
        <div class="workflow-title">Sequence Design</div>
        <div class="workflow-desc">
            Design new protein sequences using ProteinMPNN, ESM-IF1, or AI-guided evolution.
            Start from a backbone structure or generate de novo.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Start Design →", key="btn_design", use_container_width=True, type="primary"):
        st.switch_page("pages/0_design.py")

with col2:
    st.markdown(
        """
    <div class="workflow-card">
        <span class="workflow-icon">🔮</span>
        <div class="workflow-title">Structure Prediction</div>
        <div class="workflow-desc">
            Predict 3D structures using ColabFold, Chai-1, Boltz-2, or ESMFold.
            Supports proteins, complexes, and ligand binding.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Start Prediction →", key="btn_predict", use_container_width=True, type="primary"):
        st.switch_page("pages/1_predict.py")

with col3:
    st.markdown(
        """
    <div class="workflow-card">
        <span class="workflow-icon">📊</span>
        <div class="workflow-title">Structure Evaluation</div>
        <div class="workflow-desc">
            Evaluate structures with 18+ metrics including lDDT, TM-score,
            DockQ, clash score, and energy calculations.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Start Evaluation →", key="btn_evaluate", use_container_width=True, type="primary"):
        st.switch_page("pages/2_evaluate.py")

with col4:
    st.markdown(
        """
    <div class="workflow-card">
        <span class="workflow-icon">🧪</span>
        <div class="workflow-title">Directed Evolution</div>
        <div class="workflow-desc">
            Run iterative design cycles with fitness landscape exploration,
            library generation, and automated optimization.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Start Evolution →", key="btn_evolution", use_container_width=True, type="primary"):
        st.switch_page("pages/4_evolution.py")

# Pipeline Overview
st.markdown('<div class="section-title">🔄 Design Pipeline</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="pipeline-container">
    <div class="pipeline-step">
        <div style="font-size: 1.8rem;">📥</div>
        <div style="font-weight: 600; margin-top: 8px;">Input</div>
        <div style="font-size: 0.8rem; color: var(--pdhub-text-secondary);">Sequence/Structure</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
        <div style="font-size: 1.8rem;">🎨</div>
        <div style="font-weight: 600; margin-top: 8px;">Design</div>
        <div style="font-size: 0.8rem; color: var(--pdhub-text-secondary);">MPNN/ESM-IF1</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
        <div style="font-size: 1.8rem;">🔮</div>
        <div style="font-weight: 600; margin-top: 8px;">Predict</div>
        <div style="font-size: 0.8rem; color: var(--pdhub-text-secondary);">AF2/Chai/Boltz</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
        <div style="font-size: 1.8rem;">📊</div>
        <div style="font-weight: 600; margin-top: 8px;">Evaluate</div>
        <div style="font-size: 0.8rem; color: var(--pdhub-text-secondary);">18+ Metrics</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
        <div style="font-size: 1.8rem;">🔄</div>
        <div style="font-weight: 600; margin-top: 8px;">Iterate</div>
        <div style="font-size: 0.8rem; color: var(--pdhub-text-secondary);">Optimize</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Two column layout for quick start and tools
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="section-title">📚 Quick Start Guide</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="step-card">
        <div class="step-number">1</div>
        <div class="step-content">
            <div class="step-title">Upload or Fetch Sequence</div>
            <div class="step-desc">Paste a sequence, upload FASTA, or fetch from UniProt/PDB/AlphaFold DB</div>
        </div>
    </div>
    <div class="step-card">
        <div class="step-number">2</div>
        <div class="step-content">
            <div class="step-title">Choose Your Workflow</div>
            <div class="step-desc">Design new sequences, predict structures, or evaluate existing models</div>
        </div>
    </div>
    <div class="step-card">
        <div class="step-number">3</div>
        <div class="step-content">
            <div class="step-title">Configure Parameters</div>
            <div class="step-desc">Set predictor options, design temperature, or evaluation metrics</div>
        </div>
    </div>
    <div class="step-card">
        <div class="step-number">4</div>
        <div class="step-content">
            <div class="step-title">Run & Analyze</div>
            <div class="step-desc">Execute the job and explore results with interactive visualizations</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown('<div class="section-title">🛠️ Available Tools</div>', unsafe_allow_html=True)

    # Predictors
    st.markdown("#### Structure Predictors")
    try:
        from protein_design_hub.core.config import get_settings
        from protein_design_hub.predictors.registry import PredictorRegistry

        settings = get_settings()
        predictor_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        for name in ["colabfold", "chai1", "boltz2", "esmfold", "esmfold_api"]:
            try:
                pred = PredictorRegistry.get(name, settings)
                installed = pred.installer.is_installed()
                badge_class = "tool-badge-ok" if installed else "tool-badge-missing"
                status = "✓" if installed else "✗"
                predictor_html += f'<span class="tool-badge {badge_class}">{name} {status}</span>'
            except Exception:
                predictor_html += f'<span class="tool-badge tool-badge-warn">{name} ⚠</span>'
        predictor_html += '</div>'

        st.markdown(predictor_html, unsafe_allow_html=True)
    except Exception:
        st.warning("Could not load predictor status")

    # Designers
    st.markdown("#### Design Tools")
    try:
        from protein_design_hub.design.registry import DesignerRegistry

        designer_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        for name in DesignerRegistry.list_available():
            try:
                d = DesignerRegistry.get(name, settings)
                installed = d.installer.is_installed()
                badge_class = "tool-badge-ok" if installed else "tool-badge-missing"
                status = "✓" if installed else "✗"
                designer_html += f'<span class="tool-badge {badge_class}">{name} {status}</span>'
            except Exception:
                designer_html += f'<span class="tool-badge tool-badge-warn">{name} ⚠</span>'
        designer_html += '</div>'

        st.markdown(designer_html, unsafe_allow_html=True)
    except Exception:
        st.warning("Could not load designer status")

    # Evaluation
    st.markdown("#### Evaluation Metrics")
    try:
        from protein_design_hub.evaluation.composite import CompositeEvaluator

        metrics = CompositeEvaluator.list_all_metrics()
        available = sum(1 for m in metrics if m["available"])
        st.markdown(
            f'<span class="tool-badge tool-badge-ok">{available}/{len(metrics)} metrics available ✓</span>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.warning("Could not load metrics status")

# Features Grid
st.markdown('<div class="section-title">✨ Key Features</div>', unsafe_allow_html=True)

col_feat1, col_feat2, col_feat3 = st.columns(3)

with col_feat1:
    st.markdown(
        """
    <div class="feature-item">
        <b>🔬 Multi-Predictor Support</b><br>
        <span style="color: var(--pdhub-text-secondary);">ColabFold, Chai-1, Boltz-2, ESMFold - run all and compare results</span>
    </div>
    <div class="feature-item">
        <b>🎯 Advanced Design</b><br>
        <span style="color: var(--pdhub-text-secondary);">ProteinMPNN, ESM-IF1 for fixed-backbone sequence design</span>
    </div>
    <div class="feature-item">
        <b>🧬 Ligand Support</b><br>
        <span style="color: var(--pdhub-text-secondary);">Design proteins with ligand awareness (Chai-1, Boltz-2)</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_feat2:
    st.markdown(
        """
    <div class="feature-item">
        <b>📊 18+ Evaluation Metrics</b><br>
        <span style="color: var(--pdhub-text-secondary);">lDDT, TM-score, DockQ, clash score, energy calculations</span>
    </div>
    <div class="feature-item">
        <b>🔄 Directed Evolution</b><br>
        <span style="color: var(--pdhub-text-secondary);">Iterative optimization with fitness landscape exploration</span>
    </div>
    <div class="feature-item">
        <b>⚗️ Biophysical Analysis</b><br>
        <span style="color: var(--pdhub-text-secondary);">pI, MW, solubility, aggregation propensity predictions</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_feat3:
    st.markdown(
        """
    <div class="feature-item">
        <b>📈 Interactive Visualizations</b><br>
        <span style="color: var(--pdhub-text-secondary);">3D structure viewer, PAE heatmaps, per-residue plots</span>
    </div>
    <div class="feature-item">
        <b>🔗 Database Integration</b><br>
        <span style="color: var(--pdhub-text-secondary);">Fetch from UniProt, PDB, AlphaFold DB directly</span>
    </div>
    <div class="feature-item">
        <b>📦 Batch Processing</b><br>
        <span style="color: var(--pdhub-text-secondary);">Process multiple sequences with parallel execution</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown(
    """
<div class="footer">
    <p><b>Protein Design Hub</b> v0.2.0</p>
    <p>
        <a href="https://github.com/recep2244/pdhub" target="_blank">GitHub</a> ·
        <a href="https://github.com/recep2244/pdhub/issues" target="_blank">Report Issue</a> ·
        Built with Streamlit
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar with new navigation
sidebar_nav(current="Home")
sidebar_system_status()

st.sidebar.markdown("---")
st.sidebar.markdown("### Resources")
st.sidebar.markdown(
    """
- [Documentation](https://github.com/recep2244/pdhub)
- [Report Issue](https://github.com/recep2244/pdhub/issues)
- CLI: `pdhub --help`
"""
)
