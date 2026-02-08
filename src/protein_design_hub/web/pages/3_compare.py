"""Compare page for Streamlit app - Professional UI Design."""

import streamlit as st
from pathlib import Path
import tempfile
import json

st.set_page_config(page_title="Compare - Protein Design Hub", page_icon="⚖️", layout="wide")

from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    section_header,
    info_box,
    metric_card,
    metric_card_with_context,
    card_start,
    card_end,
    empty_state,
    render_badge,
    status_badge,
    workflow_breadcrumb,
    cross_page_actions,
)
from protein_design_hub.web.agent_helpers import (
    render_agent_advice_panel,
    render_contextual_insight,
    agent_sidebar_status,
)

inject_base_css()
sidebar_nav(current="Compare")
sidebar_system_status()
agent_sidebar_status()

# =============================================================================
# Page-specific CSS for Compare page
# =============================================================================
COMPARE_CSS = """
<style>
/* Predictor Selection Pills */
.predictor-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 1rem 0;
}

.predictor-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    border-radius: 24px;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.25s ease;
    cursor: default;
    border: 2px solid transparent;
}

.predictor-pill-colabfold {
    background: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
    border-color: rgba(59, 130, 246, 0.3);
}

.predictor-pill-chai1 {
    background: rgba(168, 85, 247, 0.15);
    color: #a855f7;
    border-color: rgba(168, 85, 247, 0.3);
}

.predictor-pill-boltz2 {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border-color: rgba(245, 158, 11, 0.3);
}

.predictor-pill-esmfold {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border-color: rgba(34, 197, 94, 0.3);
}

.predictor-pill-esm3 {
    background: rgba(6, 182, 212, 0.15);
    color: #06b6d4;
    border-color: rgba(6, 182, 212, 0.3);
}

.predictor-pill .pill-icon {
    font-size: 1.1rem;
}

.predictor-pill .pill-check {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: currentColor;
    color: var(--pdhub-text-heading);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: bold;
}

/* Input Panel */
.input-panel {
    background: var(--pdhub-bg-card, rgba(20,20,30,0.8));
    border: 1px solid var(--pdhub-border, rgba(100,100,100,0.3));
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.input-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--pdhub-border, rgba(100,100,100,0.2));
}

.input-panel-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--pdhub-text, #f1f5f9);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Analysis Panel */
.analysis-panel {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
}

.analysis-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--pdhub-primary-light, #818cf8);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

.analysis-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(100,100,100,0.15);
}

.analysis-stat:last-child {
    border-bottom: none;
}

.analysis-label {
    font-size: 0.85rem;
    color: var(--pdhub-text-secondary, #a1a9b8);
}

.analysis-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--pdhub-text, #f1f5f9);
}

/* Run Section */
.run-section {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.08));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

.run-section.ready {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1));
    border-color: rgba(34, 197, 94, 0.3);
}

.run-section.disabled {
    background: var(--pdhub-bg-card, rgba(20,20,30,0.5));
    border-color: var(--pdhub-border, rgba(100,100,100,0.3));
}

.run-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
}

.run-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--pdhub-text, #f1f5f9);
}

.run-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.run-status-ready {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.run-status-pending {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* Comparison Metric Cards */
.compare-metric-card {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-md);
    padding: var(--pdhub-space-lg);
    text-align: center;
    border: 1px solid var(--pdhub-border);
    transition: var(--pdhub-transition);
}

.compare-metric-card:hover {
    box-shadow: var(--pdhub-shadow-md);
    border-color: var(--pdhub-primary-light);
}

.compare-metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: var(--pdhub-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.compare-metric-label {
    font-size: 0.8rem;
    color: var(--pdhub-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* Results Dashboard */
.results-header {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

.results-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--pdhub-text, #f1f5f9);
    margin-bottom: 0.5rem;
}

.results-subtitle {
    font-size: 0.9rem;
    color: var(--pdhub-text-secondary, #a1a9b8);
}

/* Ranking Display */
.ranking-item {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 1rem 1.25rem;
    background: var(--pdhub-bg-card, rgba(20,20,30,0.6));
    border: 1px solid var(--pdhub-border, rgba(100,100,100,0.3));
    border-radius: 12px;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.ranking-item:hover {
    border-color: var(--pdhub-primary, #6366f1);
    transform: translateX(4px);
}

.ranking-item.gold {
    background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 215, 0, 0.05));
    border-color: rgba(255, 215, 0, 0.4);
}

.ranking-item.silver {
    background: linear-gradient(135deg, rgba(192, 192, 192, 0.1), rgba(192, 192, 192, 0.05));
    border-color: rgba(192, 192, 192, 0.4);
}

.ranking-item.bronze {
    background: linear-gradient(135deg, rgba(205, 127, 50, 0.1), rgba(205, 127, 50, 0.05));
    border-color: rgba(205, 127, 50, 0.4);
}

.ranking-medal {
    font-size: 1.5rem;
    width: 40px;
    text-align: center;
}

.ranking-info {
    flex: 1;
}

.ranking-name {
    font-weight: 600;
    font-size: 1rem;
    color: var(--pdhub-text, #f1f5f9);
}

.ranking-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--pdhub-primary-light, #818cf8);
}

/* Viewer Panel */
.viewer-panel {
    background: var(--pdhub-bg-card, rgba(20,20,30,0.6));
    border: 1px solid var(--pdhub-border, rgba(100,100,100,0.3));
    border-radius: 16px;
    padding: 1rem;
    margin: 1rem 0;
}

.viewer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.75rem;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid var(--pdhub-border, rgba(100,100,100,0.2));
}

/* Reference Panel */
.reference-panel {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(99, 102, 241, 0.05));
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
}

/* Settings Summary */
.settings-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 0.75rem;
}

.setting-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    background: var(--pdhub-bg-light, rgba(255,255,255,0.05));
    border: 1px solid var(--pdhub-border, rgba(100,100,100,0.2));
    border-radius: 6px;
    font-size: 0.75rem;
    color: var(--pdhub-text-secondary, #a1a9b8);
}

.setting-tag-value {
    font-family: 'JetBrains Mono', monospace;
    color: var(--pdhub-text, #f1f5f9);
    font-weight: 600;
}
</style>
"""

st.markdown(COMPARE_CSS, unsafe_allow_html=True)

# Page header
page_header(
    "Compare Predictions",
    "Run multiple predictors and compare results with detailed analysis",
    "⚖️"
)

workflow_breadcrumb(
    ["Sequence Input", "Predict", "Evaluate", "Compare", "Refine / Design"],
    current=3,
)

with st.expander("📖 How comparison works", expanded=False):
    st.markdown("""
**This page compares multiple predictor outputs** for the same protein sequence.

**What gets compared:**
- **pLDDT / pTM** — per-predictor confidence scores
- **Structural quality** — clash score, Ramachandran, energy
- **Fold similarity** — RMSD and TM-score between predictions

**The composite ranking** weighs: lDDT (40%), TM-score (30%), clash quality (15%), pTM confidence (15%).

**When to use this page:**
- After running 2+ predictors on the same sequence
- To decide which structure to use for downstream analysis
- To identify regions where predictors disagree (may indicate disorder or flexibility)
    """)

# =============================================================================
# Predictor Configuration
# =============================================================================
PREDICTORS = {
    "ColabFold": {
        "id": "colabfold",
        "icon": "🔬",
        "color": "#3b82f6",
        "css_class": "predictor-pill-colabfold",
        "desc": "AlphaFold2 with MSA",
    },
    "Chai-1": {
        "id": "chai1",
        "icon": "🧪",
        "color": "#a855f7",
        "css_class": "predictor-pill-chai1",
        "desc": "Multi-modal diffusion",
    },
    "Boltz-2": {
        "id": "boltz2",
        "icon": "⚡",
        "color": "#f59e0b",
        "css_class": "predictor-pill-boltz2",
        "desc": "Fast diffusion model",
    },
}

# Example sequences
COMPARE_EXAMPLES = {
    "Select example...": "",
    "Ubiquitin (76 aa)": ">Ubiquitin\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "T1024 (52 aa)": ">T1024\nMAAHKGAEHVVKASLDAGVKTVAGGALVVKAKALGKDATMHLVAATLKKGYM",
    "Insulin A+B (51 aa)": ">Insulin\nGIVEQCCTSICSLYQLENYCN:FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
}

# =============================================================================
# Sidebar Settings
# =============================================================================
st.sidebar.header("Comparison Settings")

# Predictor selection in sidebar
predictors_to_run = st.sidebar.multiselect(
    "Predictors to compare",
    options=list(PREDICTORS.keys()),
    default=list(PREDICTORS.keys()),
    help="Select AI models to include in comparison"
)

predictor_map = {name: info["id"] for name, info in PREDICTORS.items()}

# Settings
with st.sidebar.expander("Prediction Settings", expanded=False):
    num_models = st.number_input("Models per predictor", value=5, min_value=1, max_value=10)
    num_recycles = st.number_input("Recycles", value=3, min_value=1, max_value=10)

with st.sidebar.expander("Evaluation Metrics", expanded=False):
    st.markdown("**Global Metrics**")
    eval_metrics = st.multiselect(
        "Global metrics",
        options=["lDDT", "BB-lDDT", "TM-score", "RMSD", "GDT-TS", "GDT-HA"],
        default=["lDDT", "TM-score"],
        label_visibility="collapsed"
    )

    st.markdown("**Interface Metrics**")
    interface_metrics = st.multiselect(
        "Interface metrics",
        options=["QS-score", "DockQ", "ICS", "IPS", "iLDDT", "Patch Scores"],
        default=["QS-score", "DockQ"],
        help="Only computed for multimeric structures",
        label_visibility="collapsed"
    )

# =============================================================================
# SECTION 1: Predictor Selection Display
# =============================================================================
section_header("Selected Predictors", "Models that will be compared", "🎯")

if predictors_to_run:
    pills_html = ['<div class="predictor-pills">']
    for name in predictors_to_run:
        info = PREDICTORS[name]
        pills_html.append(f'''
        <div class="predictor-pill {info["css_class"]}">
            <span class="pill-icon">{info["icon"]}</span>
            <span>{name}</span>
            <span class="pill-check"></span>
        </div>
        ''')
    pills_html.append('</div>')
    st.markdown(''.join(pills_html), unsafe_allow_html=True)

    # Settings summary
    st.markdown(f'''
    <div class="settings-summary">
        <div class="setting-tag">Models: <span class="setting-tag-value">{num_models}</span></div>
        <div class="setting-tag">Recycles: <span class="setting-tag-value">{num_recycles}</span></div>
        <div class="setting-tag">Metrics: <span class="setting-tag-value">{len(eval_metrics)}</span></div>
    </div>
    ''', unsafe_allow_html=True)
else:
    info_box("Select at least one predictor from the sidebar", variant="warning", icon="⚠️")

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# =============================================================================
# SECTION 2: Sequence Input with Analysis Panel
# =============================================================================
section_header("Input Sequence", "Enter protein sequence for comparison", "📝")

col_input, col_analysis = st.columns([2, 1])

with col_input:
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)

    # Quick actions row
    col_ex, col_up = st.columns([2, 1])
    with col_ex:
        selected_ex = st.selectbox(
            "Load Example",
            list(COMPARE_EXAMPLES.keys()),
            index=0,
            label_visibility="collapsed",
            help="Quick-load an example protein"
        )
    with col_up:
        uploaded = st.file_uploader(
            "Upload FASTA",
            type=["fasta", "fa"],
            key="compare_fasta",
            label_visibility="collapsed",
            help="Upload a FASTA file"
        )

    default_seq = COMPARE_EXAMPLES.get(selected_ex, "")
    if uploaded:
        default_seq = uploaded.read().decode()
        st.success(f"Loaded: {uploaded.name}")

    sequence_input = st.text_area(
        "Enter sequence (FASTA format)",
        value=default_seq,
        height=150,
        placeholder=">protein_name\nMKFLILLFNILCLFPVLAADNHGVGPQGAS...\n\nFor multi-chain complexes, use colon (:) separator",
        key="compare_seq",
        label_visibility="collapsed"
    )

    # Multi-chain indicator
    if sequence_input and ":" in sequence_input:
        st.info("🔗 **Multi-chain complex detected** — Chains separated by `:`")

    st.markdown('</div>', unsafe_allow_html=True)

with col_analysis:
    st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
    st.markdown('<div class="analysis-title">📊 Sequence Analysis</div>', unsafe_allow_html=True)

    if sequence_input:
        # Parse sequence
        lines = [l.strip() for l in sequence_input.strip().split('\n') if l.strip()]
        seq_only = ''.join(l for l in lines if not l.startswith('>'))
        chains = seq_only.split(':') if ':' in seq_only else [seq_only]
        total_len = sum(len(c) for c in chains)
        num_chains = len(chains)

        # Estimate MW (rough: 110 Da per aa)
        mw_kda = total_len * 0.11

        # Complexity
        if total_len < 150:
            complexity = "Low"
            complexity_icon = "⚡"
        elif total_len < 500:
            complexity = "Medium"
            complexity_icon = "⏱️"
        else:
            complexity = "High"
            complexity_icon = "🐢"

        st.markdown(f'''
        <div class="analysis-stat">
            <span class="analysis-label">Total Length</span>
            <span class="analysis-value">{total_len} aa</span>
        </div>
        <div class="analysis-stat">
            <span class="analysis-label">Chains</span>
            <span class="analysis-value">{num_chains}</span>
        </div>
        <div class="analysis-stat">
            <span class="analysis-label">Est. MW</span>
            <span class="analysis-value">{mw_kda:.1f} kDa</span>
        </div>
        <div class="analysis-stat">
            <span class="analysis-label">Complexity</span>
            <span class="analysis-value">{complexity_icon} {complexity}</span>
        </div>
        ''', unsafe_allow_html=True)

        # Chain breakdown for multi-chain
        if num_chains > 1:
            st.markdown("<div style='margin-top: 1rem; font-size: 0.8rem; color: var(--pdhub-text-secondary, #a1a9b8);'>Chain Lengths:</div>", unsafe_allow_html=True)
            for i, chain in enumerate(chains):
                st.markdown(f"<div style='font-size: 0.85rem; color: var(--pdhub-text, #f1f5f9); padding: 2px 0;'>Chain {chr(65+i)}: {len(chain)} aa</div>", unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="text-align: center; padding: 2rem 1rem; color: var(--pdhub-text-muted, #6b7280);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧬</div>
            <div>Enter a sequence to see analysis</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# SECTION 3: Reference Structure (Optional)
# =============================================================================
section_header("Reference Structure", "Optional: provide ground truth for evaluation", "📏")

col_ref, col_output = st.columns([1, 1])

with col_ref:
    st.markdown('<div class="reference-panel">', unsafe_allow_html=True)
    st.markdown("**Upload Reference PDB/CIF**")
    st.caption("Enables accuracy metrics (lDDT, TM-score, RMSD)")

    reference_file = st.file_uploader(
        "Upload reference structure",
        type=["pdb", "cif"],
        key="compare_ref",
        label_visibility="collapsed"
    )

    if reference_file:
        st.success(f"✅ Reference loaded: {reference_file.name}")
    else:
        st.markdown("<div style='color: #6b7280; font-size: 0.85rem; margin-top: 0.5rem;'>No reference - ranking will use pLDDT</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_output:
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown("**Output Settings**")

    try:
        from protein_design_hub.core.config import get_settings
        _settings = get_settings()
        default_out = str(_settings.output.base_dir)
    except Exception:
        default_out = "./outputs"

    output_dir = st.text_input("Output directory", value=default_out, key="compare_output", label_visibility="collapsed")
    job_name = st.text_input("Job name", placeholder="my_comparison_001", key="compare_job", label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# =============================================================================
# SECTION 4: Run Comparison
# =============================================================================
is_ready = bool(sequence_input) and bool(predictors_to_run)
section_class = "ready" if is_ready else "disabled"

st.markdown(f'<div class="run-section {section_class}">', unsafe_allow_html=True)

col_status, col_actions = st.columns([2, 1])

with col_status:
    st.markdown('<div class="run-header">', unsafe_allow_html=True)
    st.markdown('<span class="run-title">🚀 Run Comparison</span>', unsafe_allow_html=True)

    if is_ready:
        st.markdown(f'''
        <span class="run-status run-status-ready">
            Ready - {len(predictors_to_run)} predictors
        </span>
        ''', unsafe_allow_html=True)
    else:
        missing = []
        if not sequence_input:
            missing.append("sequence")
        if not predictors_to_run:
            missing.append("predictors")
        st.markdown(f'''
        <span class="run-status run-status-pending">
            Missing: {", ".join(missing)}
        </span>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if is_ready:
        st.markdown(f"Will run **{len(predictors_to_run)}** predictors with **{num_models}** models each")
        if reference_file:
            st.markdown(f"📏 Reference: `{reference_file.name}` — full evaluation metrics enabled")
        else:
            st.markdown("ℹ️ No reference — ranking by pLDDT confidence")

with col_actions:
    run_comparison = st.button(
        "🚀 Run Comparison",
        type="primary",
        use_container_width=True,
        disabled=not is_ready
    )

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Execution Logic
# =============================================================================
if run_comparison:
    if not sequence_input:
        st.error("Please provide input sequence")
    elif not predictors_to_run:
        st.error("Please select at least one predictor")
    else:
        try:
            from protein_design_hub.pipeline.workflow import PredictionWorkflow
            from protein_design_hub.core.config import get_settings
            from datetime import datetime

            settings = get_settings()
            settings.predictors.colabfold.num_models = num_models
            settings.predictors.colabfold.num_recycles = num_recycles

            # Save input to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
                tmp.write(sequence_input)
                input_path = Path(tmp.name)

            # Save reference if provided
            reference_path = None
            if reference_file:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(reference_file.name).suffix, delete=False
                ) as tmp:
                    tmp.write(reference_file.read())
                    reference_path = Path(tmp.name)

            # Setup job
            job_id = job_name or f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            predictor_list = [predictor_map[p] for p in predictors_to_run]

            # Run workflow with status
            with st.status("⏳ Running comparison pipeline...", expanded=True) as status:
                st.write("📝 Initializing workflow...")

                workflow = PredictionWorkflow(settings)

                st.write(f"🚀 Running {len(predictor_list)} predictors: {', '.join(predictors_to_run)}")

                result = workflow.run(
                    input_path=input_path,
                    output_dir=Path(output_dir),
                    reference_path=reference_path,
                    predictors=predictor_list,
                    job_id=job_id,
                )

                status.update(label="✅ Comparison Complete!", state="complete", expanded=False)

            # Store results in session state
            st.session_state["compare_result"] = result
            st.session_state["compare_job_id"] = job_id
            st.session_state["compare_output_dir"] = output_dir

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# =============================================================================
# SECTION 5: Results Display
# =============================================================================
if "compare_result" in st.session_state:
    result = st.session_state["compare_result"]
    job_id = st.session_state.get("compare_job_id", "")

    st.markdown("<br>", unsafe_allow_html=True)

    # Results header
    st.markdown(f'''
    <div class="results-header">
        <div class="results-title">✨ Comparison Results</div>
        <div class="results-subtitle">Job: {job_id}</div>
    </div>
    ''', unsafe_allow_html=True)

    # Best predictor highlight
    if result.best_predictor:
        st.markdown(f"### 🏆 Best Predictor: **{result.best_predictor.upper()}**")

    # Results tabs
    tab_ranking, tab_metrics, tab_viewer, tab_download = st.tabs([
        "🏆 Ranking", "📊 Metrics", "🔬 Structure Viewer", "💾 Downloads"
    ])

    with tab_ranking:
        section_header("Predictor Ranking", "Ordered by evaluation score", "🏆")

        if result.ranking:
            for i, (name, score) in enumerate(result.ranking, 1):
                rank_class = "gold" if i == 1 else "silver" if i == 2 else "bronze" if i == 3 else ""
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"

                st.markdown(f'''
                <div class="ranking-item {rank_class}">
                    <div class="ranking-medal">{medal}</div>
                    <div class="ranking-info">
                        <div class="ranking-name">{name.upper()}</div>
                    </div>
                    <div class="ranking-score">{score:.3f}</div>
                </div>
                ''', unsafe_allow_html=True)
            # Agent advice on comparison results
            ranking_ctx = "\n".join(
                f"- Rank {i}: {name} (score: {score:.3f})"
                for i, (name, score) in enumerate(result.ranking, 1)
            )
            render_agent_advice_panel(
                page_context=f"Predictor comparison ranking:\n{ranking_ctx}",
                default_question=(
                    "Which predictor produced the best structure and why? "
                    "Should I refine the top result or try different parameters?"
                ),
                expert="Computational Biologist",
                key_prefix="compare_agent",
            )
        else:
            empty_state("No ranking available", "Run comparison first", "📭")

    with tab_metrics:
        section_header("Detailed Metrics", "Per-predictor evaluation results", "📊")

        # Prediction summary table
        st.markdown("#### Prediction Results")
        pred_data = []
        for name, pred in result.prediction_results.items():
            pred_data.append({
                "Predictor": name.upper(),
                "Status": "✅ Success" if pred.success else "❌ Failed",
                "Structures": len(pred.structure_paths),
                "Runtime (s)": f"{pred.runtime_seconds:.1f}",
                "Best pLDDT": f"{max((s.plddt for s in pred.scores if s.plddt), default=0):.1f}"
                if pred.scores else "N/A",
            })

        import pandas as pd
        pred_df = pd.DataFrame(pred_data)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # Evaluation results (if reference provided)
        if result.evaluation_results:
            st.markdown("#### Global Metrics")
            eval_data = []
            for name, eval_result in result.evaluation_results.items():
                row = {
                    "Predictor": name.upper(),
                    "lDDT": f"{eval_result.lddt:.3f}" if eval_result.lddt else "N/A",
                    "TM-score": f"{eval_result.tm_score:.3f}" if eval_result.tm_score else "N/A",
                    "RMSD": f"{eval_result.rmsd:.2f}" if eval_result.rmsd else "N/A",
                }
                if eval_result.metadata:
                    if "bb_lddt" in eval_result.metadata:
                        row["BB-lDDT"] = f"{eval_result.metadata['bb_lddt']:.3f}"
                    if "gdt_ts" in eval_result.metadata:
                        row["GDT-TS"] = f"{eval_result.metadata['gdt_ts']:.3f}"
                    if "gdt_ha" in eval_result.metadata:
                        row["GDT-HA"] = f"{eval_result.metadata['gdt_ha']:.3f}"
                eval_data.append(row)

            eval_df = pd.DataFrame(eval_data)
            st.dataframe(eval_df, use_container_width=True, hide_index=True)

            # Interface metrics (for multimers)
            has_interface = any(
                er.qs_score is not None or (er.metadata and ("dockq" in er.metadata or "ics" in er.metadata))
                for er in result.evaluation_results.values()
            )

            if has_interface:
                st.markdown("#### Interface Metrics")
                interface_data = []
                for name, eval_result in result.evaluation_results.items():
                    row = {"Predictor": name.upper()}
                    row["QS-score"] = f"{eval_result.qs_score:.3f}" if eval_result.qs_score is not None else "N/A"

                    if eval_result.metadata:
                        meta = eval_result.metadata
                        row["DockQ"] = f"{meta['dockq']:.3f}" if isinstance(meta.get("dockq"), (int, float)) else "N/A"
                        row["ICS"] = f"{meta['ics']:.3f}" if isinstance(meta.get("ics"), (int, float)) else "N/A"
                        row["IPS"] = f"{meta['ips']:.3f}" if isinstance(meta.get("ips"), (int, float)) else "N/A"
                        row["iLDDT"] = f"{meta['ilddt']:.3f}" if isinstance(meta.get("ilddt"), (int, float)) else "N/A"

                    interface_data.append(row)

                interface_df = pd.DataFrame(interface_data)
                st.dataframe(interface_df, use_container_width=True, hide_index=True)

            # Comparison chart
            st.markdown("#### Visual Comparison")
            import plotly.graph_objects as go

            predictors = list(result.evaluation_results.keys())
            fig = go.Figure()

            for metric in ["lDDT", "TM-score"]:
                values = []
                for name in predictors:
                    er = result.evaluation_results[name]
                    if metric == "lDDT":
                        values.append(er.lddt if er.lddt else 0)
                    elif metric == "TM-score":
                        values.append(er.tm_score if er.tm_score else 0)

                if any(v > 0 for v in values):
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=[p.upper() for p in predictors],
                        y=values,
                    ))

            fig.update_layout(
                barmode="group",
                title="Predictor Metric Comparison",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_viewer:
        section_header("Structure Viewer", "Visualize predicted structures", "🔬")

        job_dir = Path(st.session_state.get("compare_output_dir", output_dir)) / job_id
        structure_files = list(job_dir.glob("**/*.pdb")) + list(job_dir.glob("**/*.cif"))

        if structure_files:
            col_select, col_view = st.columns([1, 2])

            with col_select:
                selected_structure = st.selectbox(
                    "Select structure",
                    structure_files,
                    format_func=lambda x: f"{x.parent.name}/{x.name}",
                    label_visibility="collapsed"
                )

                if selected_structure:
                    st.markdown(f"**File:** `{selected_structure.name}`")
                    st.markdown(f"**Predictor:** {selected_structure.parent.name}")

                    with open(selected_structure, "rb") as f:
                        st.download_button(
                            f"📥 Download",
                            data=f.read(),
                            file_name=selected_structure.name,
                            mime="chemical/x-pdb" if selected_structure.suffix == ".pdb" else "chemical/x-cif",
                            use_container_width=True
                        )

            with col_view:
                if selected_structure and selected_structure.exists():
                    from protein_design_hub.web.visualizations import create_structure_comparison_3d
                    import streamlit.components.v1 as components

                    ref_candidate = job_dir / "reference.pdb"
                    ref_path = ref_candidate if ref_candidate.exists() else None

                    html_view = create_structure_comparison_3d(
                        selected_structure,
                        ref_path,
                        highlight_differences=True
                    )
                    components.html(html_view, height=500)

                    if ref_path:
                        st.caption(f"Aligned with reference: {ref_path.name}")
        else:
            empty_state("No structures found", "Run a comparison to generate structures", "📭")

    with tab_download:
        section_header("Download Results", "Export structures and reports", "💾")

        job_dir = Path(st.session_state.get("compare_output_dir", output_dir)) / job_id

        st.info(f"📁 Results saved to: `{job_dir}`")

        col_files, col_report = st.columns(2)

        with col_files:
            st.markdown("**Structure Files**")
            structure_files = list(job_dir.glob("**/*.pdb")) + list(job_dir.glob("**/*.cif"))

            if structure_files:
                for sf in structure_files[:10]:  # Limit display
                    col_name, col_dl = st.columns([3, 1])
                    with col_name:
                        st.markdown(f"`{sf.parent.name}/{sf.name}`")
                    with col_dl:
                        with open(sf, "rb") as f:
                            st.download_button("📥 Download", data=f.read(), file_name=sf.name, key=f"dl_{sf.name}")

                if len(structure_files) > 10:
                    st.caption(f"...and {len(structure_files) - 10} more files")
            else:
                st.markdown("No structure files found")

        with col_report:
            st.markdown("**Reports**")
            report_path = job_dir / "report" / "report.html"
            if report_path.exists():
                st.download_button(
                    "📥 Download HTML Report",
                    data=report_path.read_text(),
                    file_name="comparison_report.html",
                    mime="text/html",
                    use_container_width=True
                )
            else:
                st.markdown("No report generated")

            summary_path = job_dir / "prediction_summary.json"
            if summary_path.exists():
                st.download_button(
                    "📥 Download JSON Summary",
                    data=summary_path.read_text(),
                    file_name="prediction_summary.json",
                    mime="application/json",
                    use_container_width=True
                )

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

# =============================================================================
# SECTION 6: Load Existing Results
# =============================================================================
section_header("Load Existing Results", "View previous comparison jobs", "📂")

existing_results = st.text_input(
    "Path to job directory",
    placeholder="/path/to/outputs/compare_20240101_120000",
    key="load_results",
    label_visibility="collapsed"
)

if existing_results and Path(existing_results).exists():
    job_path = Path(existing_results)

    # Find structure files
    structure_files = list(job_path.glob("**/*.pdb")) + list(job_path.glob("**/*.cif"))

    if structure_files:
        st.success(f"✅ Found {len(structure_files)} structure(s) in `{job_path.name}`")

        col_list, col_view = st.columns([1, 2])

        with col_list:
            selected_structure = st.selectbox(
                "Select structure to view",
                structure_files,
                format_func=lambda x: f"{x.parent.name}/{x.name}",
                key="load_struct_select"
            )

            if selected_structure:
                with open(selected_structure, "rb") as f:
                    st.download_button(
                        f"📥 Download {selected_structure.name}",
                        data=f.read(),
                        file_name=selected_structure.name,
                        mime="chemical/x-pdb" if selected_structure.suffix == ".pdb" else "chemical/x-cif",
                        use_container_width=True
                    )

        with col_view:
            if selected_structure:
                from protein_design_hub.web.visualizations import create_structure_comparison_3d
                import streamlit.components.v1 as components

                ref_candidate = job_path / "reference.pdb"
                ref_path = ref_candidate if ref_candidate.exists() else None

                html_view = create_structure_comparison_3d(
                    selected_structure,
                    ref_path,
                    highlight_differences=True
                )
                components.html(html_view, height=500)

                if ref_path:
                    st.caption(f"Showing alignment with reference: {ref_path.name}")
                else:
                    st.caption("No reference found in job folder for alignment")

    # Load summary if exists
    summary_file = job_path / "prediction_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

        st.markdown("### Results Summary")

        for pred_name, pred_info in summary.get("predictors", {}).items():
            with st.expander(f"🔮 **{pred_name.upper()}**", expanded=False):
                col_1, col_2, col_3 = st.columns(3)
                with col_1:
                    st_label = "✅ Success" if pred_info.get("success") else "❌ Failed"
                    st.markdown(f"**Status:** {st_label}")
                with col_2:
                    st.markdown(f"**Structures:** {pred_info.get('num_structures', 0)}")
                with col_3:
                    runtime = pred_info.get("runtime_seconds", 0)
                    st.markdown(f"**Runtime:** {runtime:.1f}s")

                if pred_info.get("structure_paths"):
                    st.markdown("**Structure files:**")
                    for path in pred_info["structure_paths"]:
                        st.text(f"  - {Path(path).name}")

elif existing_results:
    st.warning(f"Path not found: `{existing_results}`")
else:
    st.markdown('<div class="pdhub-muted">Enter a path to load existing comparison results</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

# =============================================================================
# Info Section
# =============================================================================
with st.expander("ℹ️ About Comparison", expanded=False):
    st.markdown("""
### How Comparison Works

1. **Prediction Phase**: Each selected predictor runs on your input sequence
2. **Evaluation Phase**: If a reference is provided, structures are evaluated
3. **Ranking Phase**: Predictors are ranked by evaluation scores (or pLDDT if no reference)

### Ranking Criteria

- **With reference**: Ranking by lDDT score
- **Without reference**: Ranking by pLDDT (predicted confidence)

### Evaluation Metrics

**Global Metrics:**
| Metric | Description | Range |
|--------|-------------|-------|
| **lDDT** | Local Distance Difference Test | 0-1 (higher is better) |
| **BB-lDDT** | Backbone-only lDDT | 0-1 (higher is better) |
| **TM-score** | Template Modeling score | 0-1 (higher is better) |
| **RMSD** | Root Mean Square Deviation | 0- (lower is better) |
| **GDT-TS/HA** | Global Distance Test | 0-1 (higher is better) |

**Interface Metrics (for multimers):**
| Metric | Description | Range |
|--------|-------------|-------|
| **QS-score** | Quaternary Structure score | 0-1 (higher is better) |
| **DockQ** | Docking Quality (fnat, iRMSD, lRMSD) | 0-1 (higher is better) |
| **ICS** | Interface Contact Similarity | 0-1 (higher is better) |
| **IPS** | Interface Patch Similarity | 0-1 (higher is better) |
| **iLDDT** | Inter-chain lDDT | 0-1 (higher is better) |
| **Patch Scores** | CASP15 local interface quality | 0-1 (higher is better) |

### Tips

- Provide a reference structure when available for more accurate comparison
- Longer sequences take more time to predict
- Each predictor runs sequentially to manage GPU memory
- Use Chai-1 or Boltz-2 for protein-ligand complexes
    """)
