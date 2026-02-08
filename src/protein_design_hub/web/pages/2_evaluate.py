"""Comprehensive Evaluation page with all OpenStructure metrics."""

import json
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

st.set_page_config(page_title="Evaluate - Protein Design Hub", page_icon="📊", layout="wide")

from protein_design_hub.web.ui import (
    get_selected_model,
    inject_base_css,
    list_output_structures,
    page_header,
    section_header,
    set_selected_model,
    sidebar_nav,
    sidebar_system_status,
    metric_card,
    metric_card_with_context,
    card_start,
    card_end,
    empty_state,
    render_badge,
    info_box,
    workflow_breadcrumb,
    cross_page_actions,
)
from protein_design_hub.web.agent_helpers import (
    render_agent_advice_panel,
    render_contextual_insight,
    agent_sidebar_status,
    render_all_experts_panel,
)

inject_base_css()
sidebar_nav(current="Evaluate")
sidebar_system_status()
agent_sidebar_status()

# =============================================================================
# Page-specific CSS for enhanced UI
# =============================================================================
EVALUATE_CSS = """
<style>
/* Evaluation Panel Styling */
.evaluation-panel {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.evaluation-panel-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--pdhub-primary-light, #818cf8);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

/* Input Card Styling */
.input-card {
    background: var(--pdhub-bg-card, rgba(18, 20, 28, 0.9));
    border: 1px solid var(--pdhub-border, rgba(255, 255, 255, 0.10));
    border-radius: 14px;
    padding: 1.25rem;
    transition: all 0.25s ease;
    margin-bottom: 1rem;
}

.input-card:hover {
    border-color: var(--pdhub-border-strong, rgba(255, 255, 255, 0.18));
}

.input-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--pdhub-border, rgba(255, 255, 255, 0.10));
}

.input-card-icon {
    font-size: 1.25rem;
}

.input-card-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--pdhub-text, #f1f5f9);
}

.input-card-subtitle {
    font-size: 0.8rem;
    color: var(--pdhub-text-muted, #6b7280);
}

/* Run Section Styling */
.run-section {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

.run-section.disabled {
    background: var(--pdhub-bg-card, rgba(20,20,30,0.5));
    border-color: var(--pdhub-border, rgba(100,100,100,0.3));
}

.run-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}

.run-section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--pdhub-text, #f1f5f9);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Status Indicator */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.status-ready {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-waiting {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.status-error {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Metric Grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 1rem 0;
}

.metric-tile {
    background: var(--pdhub-bg-card, rgba(18, 20, 28, 0.9));
    border: 1px solid var(--pdhub-border, rgba(255, 255, 255, 0.10));
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: all 0.2s ease;
}

.metric-tile:hover {
    border-color: var(--pdhub-primary, #6366f1);
    transform: translateY(-2px);
}

.metric-tile-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--pdhub-text-heading, #e5e7eb);
}

.metric-tile-label {
    font-size: 0.75rem;
    color: var(--pdhub-text-secondary, #a1a9b8);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.metric-tile-icon {
    font-size: 1.25rem;
    margin-bottom: 6px;
}

/* Metric Categories */
.metric-category {
    background: var(--pdhub-bg-card, rgba(18, 20, 28, 0.9));
    border: 1px solid var(--pdhub-border, rgba(255, 255, 255, 0.10));
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.metric-category-header {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--pdhub-text, #f1f5f9);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Settings Panel */
.settings-panel {
    background: var(--pdhub-bg-card, rgba(20,20,30,0.6));
    border: 1px solid var(--pdhub-border, rgba(100,100,100,0.3));
    border-radius: 12px;
    padding: 1rem;
    margin-top: 0.5rem;
}

.settings-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--pdhub-text-secondary, #a1a9b8);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--pdhub-border, rgba(100,100,100,0.2));
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

/* Workflow Steps */
.workflow-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 0.75rem 0;
}

.workflow-step-number {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--pdhub-primary, #6366f1);
    color: var(--pdhub-text-heading);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.85rem;
    flex-shrink: 0;
}

.workflow-step-content {
    flex: 1;
}

.workflow-step-title {
    font-weight: 600;
    color: var(--pdhub-text, #f1f5f9);
    margin-bottom: 2px;
}

.workflow-step-desc {
    font-size: 0.85rem;
    color: var(--pdhub-text-secondary, #a1a9b8);
}

/* Checklist Item */
.checklist-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--pdhub-bg-light, rgba(255, 255, 255, 0.05));
    margin: 4px 0;
}

.checklist-icon {
    width: 20px;
    text-align: center;
}

.checklist-done {
    color: #22c55e;
}

.checklist-pending {
    color: #6b7280;
}

/* File Status */
.file-status {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 10px;
    background: var(--pdhub-bg-light, rgba(255, 255, 255, 0.05));
    margin: 8px 0;
}

.file-status-icon {
    font-size: 1.1rem;
}

.file-status-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--pdhub-text, #f1f5f9);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.file-status-badge {
    font-size: 0.7rem;
    padding: 3px 8px;
    border-radius: 12px;
    font-weight: 600;
}

.file-status-badge-ok {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
}

.file-status-badge-optional {
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
}

/* Visual Placeholder */
.visual-placeholder {
    border: 2px dashed var(--pdhub-border, rgba(100,100,100,0.3));
    border-radius: 12px;
    height: 350px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--pdhub-bg-card, rgba(18, 20, 28, 0.5));
}

.visual-placeholder-content {
    text-align: center;
    color: var(--pdhub-text-muted, #6b7280);
}

.visual-placeholder-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    opacity: 0.6;
}
</style>
"""

st.markdown(EVALUATE_CSS, unsafe_allow_html=True)


def _summarize_metric_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    if isinstance(value, list):
        numeric = [v for v in value if isinstance(v, (int, float))]
        if numeric:
            mean_val = sum(numeric) / len(numeric)
            return f"mean {mean_val:.4f} (n={len(numeric)})"
        return f"{len(value)} items"
    if isinstance(value, dict):
        numeric_keys = {k: v for k, v in value.items() if isinstance(v, (int, float))}
        if numeric_keys:
            key, val = next(iter(numeric_keys.items()))
            return f"{key}: {val:.4f}"
        return f"{len(value)} fields"
    return str(value)


def _primary_metric_value(metric_name: str, result) -> Optional[str]:
    field_map = {
        "lddt": "lddt",
        "qs_score": "qs_score",
        "tm_score": "tm_score",
        "rmsd": "rmsd",
        "gdt_ts": "gdt_ts",
        "gdt_ha": "gdt_ha",
        "clash_score": "clash_score",
        "contact_energy": "contact_energy",
        "sasa": "sasa_total",
        "interface_bsa": "interface_bsa_total",
        "salt_bridges": "salt_bridge_count",
        "openmm_gbsa": "openmm_gbsa_energy_kj_mol",
        "rosetta_energy": "rosetta_total_score",
        "rosetta_score_jd2": "rosetta_score_jd2_total_score",
        "sequence_recovery": "sequence_recovery",
        "disorder": "disorder_fraction",
        "shape_complementarity": "shape_complementarity",
        "cad_score": "cad_score",
        "voromqa": "voromqa_score",
        "lddt_pli": None,
    }

    field = field_map.get(metric_name)
    if field and hasattr(result, field):
        return _summarize_metric_value(getattr(result, field))

    meta = getattr(result, "metadata", {}) or {}
    metric_meta = meta.get(metric_name)
    if isinstance(metric_meta, dict):
        for key in ("score", "lddt_pli", "cad_score", "voromqa_score", "value"):
            if key in metric_meta and isinstance(metric_meta[key], (int, float)):
                return f"{metric_meta[key]:.4f}"
        return _summarize_metric_value(metric_meta)
    return _summarize_metric_value(metric_meta)

# Page header
page_header(
    "Structure Evaluation",
    "Comprehensive evaluation with 18+ metrics including lDDT, TM-score, DockQ, and energy calculations",
    "📊",
)

workflow_breadcrumb(
    ["Sequence Input", "Predict", "Evaluate", "Refine / Design", "Export"],
    current=2,
)

with st.expander("📖 Understanding evaluation metrics", expanded=False):
    st.markdown("""
**Key metrics at a glance:**

| Metric | What it measures | Good value | Action if bad |
|--------|-----------------|------------|---------------|
| **pLDDT** | Per-residue confidence | > 90 | Re-predict with ColabFold |
| **Clash Score** | Steric clashes (MolProbity) | < 10 | Refine with AMBER |
| **TM-score** | Global fold vs reference | > 0.7 | Check alignment |
| **RMSD** | Deviation from reference (Å) | < 2.0 | Check domain alignment |
| **Contact Energy** | Residue-residue contacts | < -30 | Review packing |
| **SASA** | Solvent accessibility (Å²) | Depends on size | Check for exposed hydrophobics |

**How to evaluate:**
1. Upload a PDB/CIF file (from prediction) or select from recent outputs
2. Optionally upload a **reference** structure for RMSD/TM-score comparison
3. Click **Run Evaluation** to compute all metrics
4. Use the **AI Analysis** button to get expert interpretation
    """)

# Initialize metric selection
try:
    from protein_design_hub.evaluation.composite import CompositeEvaluator
    @st.cache_data(show_spinner=False)
    def _cached_list_metrics():
        return CompositeEvaluator.list_all_metrics()
    _all = _cached_list_metrics()
    _metric_names = [m["name"] for m in _all]
    _metric_info = {m["name"]: m for m in _all}
except Exception:
    _metric_names = ["clash_score", "contact_energy", "sasa"]
    _metric_info = {}

# Initialize session state for metrics
if "quick_metrics" not in st.session_state:
    st.session_state.quick_metrics = [
        "clash_score",
        "contact_energy",
        "sasa",
        "interface_bsa",
        "salt_bridges",
        "voromqa",
        "cad_score",
    ]

# Sidebar - Simplified Tool Status
with st.sidebar.expander("Tool Status", expanded=False):
    try:
        from protein_design_hub.evaluation.ost_runner import get_ost_runner
        runner = get_ost_runner()
        if runner.is_available():
            version = runner.get_version()
            st.success(f"OpenStructure: v{version}")
        else:
            st.error("OpenStructure: Not available")
            st.caption("Install: `micromamba create -n ost -c conda-forge -c bioconda openstructure`")
    except Exception as e:
        st.error(f"Error checking tools: {e}")

# =============================================================================
# SECTION 1: Evaluation Configuration (Main Content Area)
# =============================================================================
section_header("Evaluation Configuration", "Select metrics and parameters", "⚙️")

col_config, col_preview = st.columns([2, 1])

with col_config:
    # Evaluation Mode Selection
    st.markdown("##### Evaluation Mode")
    eval_mode = st.radio(
        "Mode",
        ["Quick (Design Ranking)", "Comprehensive (Full Analysis)"],
        index=0,
        horizontal=True,
        help="Quick mode for fast design screening, Comprehensive for detailed structural analysis",
        label_visibility="collapsed"
    )

    # Metric Selection
    st.markdown("##### Metrics Selection")

    # Group metrics by category
    ref_free_metrics = ["clash_score", "contact_energy", "sasa", "interface_bsa", "salt_bridges", "voromqa", "cad_score", "disorder"]
    ref_based_metrics = ["lddt", "tm_score", "rmsd", "gdt_ts", "gdt_ha", "qs_score", "sequence_recovery"]
    interface_metrics = ["dockq", "ics", "ips", "ilddt", "patch_scores", "lddt_pli", "shape_complementarity"]

    tab_refree, tab_refbased, tab_interface = st.tabs(["Reference-Free", "Reference-Based", "Interface"])

    with tab_refree:
        st.caption("Metrics that work without a reference structure - ideal for design ranking")
        available_ref_free = [m for m in ref_free_metrics if m in _metric_names]
        selected_ref_free = st.multiselect(
            "Reference-free metrics",
            options=available_ref_free,
            default=[m for m in st.session_state.quick_metrics if m in available_ref_free],
            key="ref_free_select",
            label_visibility="collapsed"
        )

    with tab_refbased:
        st.caption("Metrics requiring a reference structure for comparison")
        available_ref_based = [m for m in ref_based_metrics if m in _metric_names]
        selected_ref_based = st.multiselect(
            "Reference-based metrics",
            options=available_ref_based,
            default=["lddt", "tm_score", "rmsd"] if eval_mode.startswith("Comprehensive") else [],
            key="ref_based_select",
            label_visibility="collapsed"
        )

    with tab_interface:
        st.caption("Interface quality metrics for complexes")
        available_interface = [m for m in interface_metrics if m in _metric_names]
        selected_interface = st.multiselect(
            "Interface metrics",
            options=available_interface,
            default=[],
            key="interface_select",
            label_visibility="collapsed"
        )

    # Combine all selected metrics
    quick_metrics = list(set(selected_ref_free + selected_ref_based + selected_interface))
    st.session_state.quick_metrics = quick_metrics

    # Advanced Settings
    with st.expander("Advanced Parameters", expanded=False):
        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            st.markdown("**lDDT Settings**")
            lddt_radius = st.number_input("Inclusion radius (Å)", value=15.0, min_value=5.0, max_value=30.0)
            lddt_seq_sep = st.number_input("Sequence separation", value=0, min_value=0, max_value=10)

        with col_adv2:
            st.markdown("**RMSD Settings**")
            rmsd_atoms = st.selectbox("Atom selection", options=["CA", "backbone", "heavy", "all"], index=0)

            st.markdown("**Chain Mapping**")
            chem_seqid_thresh = st.slider("Chain grouping (%)", 50, 100, 95)
            map_seqid_thresh = st.slider("Mapping threshold (%)", 30, 100, 70)

with col_preview:
    # Metrics Summary Panel
    st.markdown("""
    <div class="evaluation-panel">
        <div class="evaluation-panel-title">Selected Metrics</div>
    """, unsafe_allow_html=True)

    if quick_metrics:
        ref_free_count = len([m for m in quick_metrics if m in ref_free_metrics])
        ref_based_count = len([m for m in quick_metrics if m in ref_based_metrics])
        interface_count = len([m for m in quick_metrics if m in interface_metrics])

        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <div style="font-size: 2.5rem; font-weight: 700; color: #60a5fa;">{len(quick_metrics)}</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">Total Metrics</div>
        </div>
        <div class="checklist-item">
            <span class="checklist-icon {'checklist-done' if ref_free_count > 0 else 'checklist-pending'}">{'✓' if ref_free_count > 0 else '○'}</span>
            <span>{ref_free_count} Reference-free</span>
        </div>
        <div class="checklist-item">
            <span class="checklist-icon {'checklist-done' if ref_based_count > 0 else 'checklist-pending'}">{'✓' if ref_based_count > 0 else '○'}</span>
            <span>{ref_based_count} Reference-based</span>
        </div>
        <div class="checklist-item">
            <span class="checklist-icon {'checklist-done' if interface_count > 0 else 'checklist-pending'}">{'✓' if interface_count > 0 else '○'}</span>
            <span>{interface_count} Interface</span>
        </div>
        """, unsafe_allow_html=True)

        # Show if reference is required
        needs_ref = any(m in ref_based_metrics or m in interface_metrics for m in quick_metrics)
        if needs_ref:
            st.markdown("""
            <div style="margin-top: 1rem; padding: 8px 12px; background: rgba(245, 158, 11, 0.15); border-radius: 8px; font-size: 0.8rem; color: #f59e0b;">
                ⚠️ Reference structure required
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; color: #6b7280;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
            <div>No metrics selected</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# SECTION 2: Input Structures
# =============================================================================
section_header("Input Structures", "Upload or select structures to evaluate", "📁")

# Initialize variables before column blocks to avoid NameError
model_file = None
chosen = None
reference_file = None

col_upload, col_visual = st.columns([1, 1])

with col_upload:
    # Model structure card
    st.markdown("""
    <div class="input-card">
        <div class="input-card-header">
            <span class="input-card-icon">🧬</span>
            <div>
                <div class="input-card-title">Model Structure</div>
                <div class="input-card-subtitle">Required - The structure to evaluate</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Check for pre-selected model from Jobs page
    try:
        from protein_design_hub.core.config import get_settings
        _settings = get_settings()
        recent = list_output_structures(Path(_settings.output.base_dir))
    except Exception:
        recent = []

    chosen = None
    sel = get_selected_model()
    if sel is not None and sel.exists():
        chosen = sel
        st.markdown(f"""
        <div class="file-status">
            <span class="file-status-icon">✅</span>
            <span class="file-status-name">{sel.name}</span>
            <span class="file-status-badge file-status-badge-ok">Selected</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Clear selection", key="clear_sel_model", type="secondary", use_container_width=True):
            set_selected_model(None)
            st.rerun()
    else:
        if recent:
            chosen = st.selectbox(
                "Choose from recent outputs",
                options=[None] + recent,
                format_func=lambda p: "— Select structure —" if p is None else p.name,
                index=0,
                key="recent_model"
            )

        model_file = st.file_uploader(
            "Or upload PDB/CIF file",
            type=["pdb", "cif", "mmcif"],
            key="model",
            help="Structure to evaluate",
        )

        if model_file:
            st.markdown(f"""
            <div class="file-status">
                <span class="file-status-icon">📄</span>
                <span class="file-status-name">{model_file.name}</span>
                <span class="file-status-badge file-status-badge-ok">Uploaded</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Reference structure card
    needs_reference = any(m in ref_based_metrics or m in interface_metrics for m in quick_metrics)
    ref_status = "Required" if needs_reference else "Optional"
    ref_color = "warning" if needs_reference else "info"

    st.markdown(f"""
    <div class="input-card">
        <div class="input-card-header">
            <span class="input-card-icon">📏</span>
            <div>
                <div class="input-card-title">Reference Structure</div>
                <div class="input-card-subtitle">{ref_status} - Ground truth for comparison</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    reference_file = st.file_uploader(
        "Upload reference PDB/CIF",
        type=["pdb", "cif", "mmcif"],
        key="reference",
        help="Ground truth structure for comparison",
    )

    if reference_file:
        st.markdown(f"""
        <div class="file-status">
            <span class="file-status-icon">✅</span>
            <span class="file-status-name">{reference_file.name}</span>
            <span class="file-status-badge file-status-badge-ok">Uploaded</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        if needs_reference:
            st.warning("Reference needed for selected metrics")
        else:
            st.markdown(f"""
            <div class="file-status">
                <span class="file-status-icon">ℹ️</span>
                <span class="file-status-name">Not provided</span>
                <span class="file-status-badge file-status-badge-optional">Optional</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Determine active model path for visualization
active_model_path = None
active_ref_path = None

if chosen:
    active_model_path = Path(chosen)
elif 'model_file' in dir() and model_file:
    # Save temp for visualization
    with tempfile.NamedTemporaryFile(suffix=Path(model_file.name).suffix, delete=False) as tmp:
        tmp.write(model_file.read())
        active_model_path = Path(tmp.name)
        model_file.seek(0) # Reset pointer

if reference_file:
    with tempfile.NamedTemporaryFile(suffix=Path(reference_file.name).suffix, delete=False) as tmp:
        tmp.write(reference_file.read())
        active_ref_path = Path(tmp.name)
        reference_file.seek(0)

with col_visual:
    st.markdown("""
    <div class="input-card" style="height: 100%;">
        <div class="input-card-header">
            <span class="input-card-icon">🔬</span>
            <div>
                <div class="input-card-title">Visual Inspection</div>
                <div class="input-card-subtitle">3D structure preview</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if active_model_path:
        from protein_design_hub.web.visualizations import (
            create_structure_comparison_3d,
            create_protein_info_table,
            create_plddt_sequence_viewer,
            create_expandable_section,
            create_model_quality_summary,
        )
        import streamlit.components.v1 as components

        st.caption(f"Viewing: `{active_model_path.name}`")
        html_view = create_structure_comparison_3d(
            active_model_path,
            active_ref_path,
            highlight_differences=True
        )
        components.html(html_view, height=350)

        # Extract sequence and pLDDT from structure for enhanced display
        try:
            from Bio.PDB import PDBParser, MMCIFParser
            from Bio.SeqUtils import seq1

            if active_model_path.suffix.lower() in ['.cif', '.mmcif']:
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)

            structure = parser.get_structure('model', str(active_model_path))

            # Get sequence and pLDDT from B-factors
            sequence = ""
            plddt_values = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':
                            sequence += seq1(residue.resname)
                            if 'CA' in residue:
                                plddt_values.append(residue['CA'].get_bfactor())
                    break
                break

            if sequence and plddt_values:
                mean_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else 0

                # Quick stats
                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-tile">
                        <div class="metric-tile-icon">📐</div>
                        <div class="metric-tile-value">{len(sequence)}</div>
                        <div class="metric-tile-label">Residues</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-tile-icon">⭐</div>
                        <div class="metric-tile-value">{mean_plddt:.1f}</div>
                        <div class="metric-tile-label">Avg pLDDT</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Sequence with Confidence", expanded=False):
                    seq_html = create_plddt_sequence_viewer(
                        sequence[:100] + ("..." if len(sequence) > 100 else ""),
                        plddt_values[:100] if len(plddt_values) > 100 else plddt_values,
                        label=active_model_path.stem[:15],
                        show_ruler=True
                    )
                    components.html(seq_html, height=150, scrolling=True)
                    if len(sequence) > 100:
                        st.caption(f"Showing first 100 of {len(sequence)} residues")
        except Exception as e:
            pass  # Silently skip enhanced view if parsing fails

    else:
        st.markdown("""
        <div class="visual-placeholder">
            <div class="visual-placeholder-content">
                <div class="visual-placeholder-icon">🧬</div>
                <div>Upload or select a structure</div>
                <div style="font-size: 0.8rem; margin-top: 4px;">3D viewer will appear here</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# SECTION 3: Run Evaluation
# =============================================================================
section_header("Run Evaluation", "Execute selected metrics on your structure", "🚀")

# Determine readiness
has_model = chosen is not None or model_file is not None
has_reference = reference_file is not None
needs_reference_for_metrics = any(m in ref_based_metrics or m in interface_metrics for m in quick_metrics)
is_ready = has_model and (not needs_reference_for_metrics or has_reference)

# Status indicators
run_section_class = "" if is_ready else "disabled"

st.markdown(f'<div class="run-section {run_section_class}">', unsafe_allow_html=True)

col_status, col_actions = st.columns([2, 1])

with col_status:
    st.markdown('<div class="run-section-title">🎯 Evaluation Status</div>', unsafe_allow_html=True)

    # Checklist
    st.markdown(f"""
    <div class="checklist-item">
        <span class="checklist-icon {'checklist-done' if has_model else 'checklist-pending'}">{'✓' if has_model else '○'}</span>
        <span>Model structure {'loaded' if has_model else 'required'}</span>
    </div>
    <div class="checklist-item">
        <span class="checklist-icon {'checklist-done' if has_reference else ('checklist-pending' if needs_reference_for_metrics else 'checklist-done')}">{'✓' if has_reference or not needs_reference_for_metrics else '○'}</span>
        <span>Reference structure {'loaded' if has_reference else ('required' if needs_reference_for_metrics else 'not needed')}</span>
    </div>
    <div class="checklist-item">
        <span class="checklist-icon {'checklist-done' if quick_metrics else 'checklist-pending'}">{'✓' if quick_metrics else '○'}</span>
        <span>{len(quick_metrics)} metrics selected</span>
    </div>
    """, unsafe_allow_html=True)

with col_actions:
    # Status badge
    if is_ready:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <span class="status-indicator status-ready">✓ Ready to Run</span>
        </div>
        """, unsafe_allow_html=True)
    elif not has_model:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <span class="status-indicator status-waiting">Upload Structure</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <span class="status-indicator status-waiting">Upload Reference</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Action buttons
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    run_quick = st.button(
        "⚡ Run Quick Evaluation",
        type="primary",
        use_container_width=True,
        disabled=not has_model
    )

with col_btn2:
    run_comprehensive = st.button(
        "🔬 Run Comprehensive",
        type="secondary",
        use_container_width=True,
        disabled=not (has_model and has_reference)
    )

# =============================================================================
# Evaluation Execution
# =============================================================================
if run_quick:
    if not model_file and chosen is None:
        st.error("Please upload a model structure (or pick a recent one).")
    else:
        try:
            # Resolve model path
            if chosen is not None and model_file is None:
                model_path = Path(chosen)
            else:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(model_file.name).suffix, delete=False
                ) as tmp:
                    tmp.write(model_file.read())
                    model_path = Path(tmp.name)

            reference_path = None
            if reference_file is not None:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(reference_file.name).suffix, delete=False
                ) as tmp:
                    tmp.write(reference_file.read())
                    reference_path = Path(tmp.name)

            # Filter metrics if reference missing
            from protein_design_hub.evaluation.composite import CompositeEvaluator
            from protein_design_hub.core.config import get_settings

            settings = get_settings()
            evaluator = CompositeEvaluator(metrics=quick_metrics, settings=settings)

            if reference_path is None:
                available = CompositeEvaluator.list_all_metrics()
                ref_required = {m["name"]: m["requires_reference"] for m in available}
                filtered = [m for m in quick_metrics if not ref_required.get(m, False)]
                if filtered != quick_metrics:
                    st.warning(
                        "Reference not provided; skipping reference-required metrics: "
                        + ", ".join(sorted(set(quick_metrics) - set(filtered)))
                    )
                evaluator = CompositeEvaluator(metrics=filtered, settings=settings)

            with st.spinner("Computing metrics..."):
                result = evaluator.evaluate(model_path, reference_path)

            st.session_state.quick_eval = result.to_dict()

            # Results header
            st.markdown("""
            <div class="results-header">
                <div class="results-title">✅ Evaluation Complete</div>
                <div class="results-subtitle">Quick evaluation finished successfully</div>
            </div>
            """, unsafe_allow_html=True)

            # Metric results grid
            st.markdown("### Results Summary")

            # --- Metric cards with scientific context ---
            # Structural
            struct_items = []
            if result.clash_score is not None:
                struct_items.append(("Clash Score", result.clash_score, "clash_score", "💥"))
            if result.sasa_total is not None:
                struct_items.append(("SASA (A\u00b2)", result.sasa_total, "sasa", "\U0001f310"))
            if result.interface_bsa_total is not None:
                struct_items.append(("Interface BSA", result.interface_bsa_total, "interface_bsa", "\U0001f517"))

            if struct_items:
                st.markdown("#### Structural Metrics")
                cols = st.columns(len(struct_items))
                for i, (label, val, mname, icon) in enumerate(struct_items):
                    with cols[i]:
                        metric_card_with_context(val, label, metric_name=mname, icon=icon)

            # Energy
            energy_items = []
            if result.contact_energy is not None:
                energy_items.append(("Contact Energy", result.contact_energy, "contact_energy", "\U0001f50b"))
            if result.salt_bridge_count is not None:
                energy_items.append(("Salt Bridges", result.salt_bridge_count, "salt_bridges", "\U0001f9c2"))

            if energy_items:
                st.markdown("#### Energy Metrics")
                cols = st.columns(len(energy_items))
                for i, (label, val, mname, icon) in enumerate(energy_items):
                    with cols[i]:
                        metric_card_with_context(val, label, metric_name=mname, icon=icon)

            # Quality
            quality_items = []
            if result.tm_score is not None:
                quality_items.append(("TM-Score", result.tm_score, "tm_score", "\U0001f4cf"))
            if result.rmsd is not None:
                quality_items.append(("RMSD (A)", result.rmsd, "rmsd", "\U0001f4d0"))
            if result.lddt is not None:
                quality_items.append(("lDDT", result.lddt, "lddt", "\U0001f3af"))
            if result.cad_score is not None:
                quality_items.append(("CAD-score", result.cad_score, "cad_score", "\U0001f9ed"))
            if result.voromqa_score is not None:
                quality_items.append(("VoroMQA", result.voromqa_score, "voromqa_score", "\U0001f9ea"))

            if quality_items:
                st.markdown("#### Quality Metrics")
                cols = st.columns(min(len(quality_items), 4))
                for i, (label, val, mname, icon) in enumerate(quality_items):
                    with cols[i % len(cols)]:
                        metric_card_with_context(val, label, metric_name=mname, icon=icon)

            # Detailed metrics table
            st.markdown("### 📋 All Computed Metrics")
            metric_info = {m["name"]: m for m in CompositeEvaluator.list_all_metrics()}
            metric_rows = []
            for metric_name in quick_metrics:
                info = metric_info.get(metric_name, {})
                value = _primary_metric_value(metric_name, result)
                if value is not None:
                    status = "OK"
                elif info.get("requires_reference") and reference_path is None:
                    status = "Skipped (needs reference)"
                elif info.get("available") is False:
                    status = "Unavailable"
                else:
                    status = "No result"

                metric_rows.append(
                    {
                        "Metric": metric_name,
                        "Status": status,
                        "Value": value or "",
                        "Requires Reference": bool(info.get("requires_reference")),
                    }
                )

            if metric_rows:
                st.dataframe(metric_rows, use_container_width=True, hide_index=True)

            with st.expander("Raw metric outputs"):
                st.json(result.metadata or {})

            # Save results if part of a job
            if chosen:
                try:
                    chosen_path = Path(chosen)
                    # Infer job dir: outputs/job_id/predictor/file.pdb -> outputs/job_id
                    job_dir = chosen_path.parent.parent
                    if (job_dir / "prediction_summary.json").exists():
                        eval_dir = job_dir / "evaluation"
                        eval_dir.mkdir(exist_ok=True)
                        with open(eval_dir / "quick_metrics.json", "w") as f:
                            json.dump(result.to_dict(), f, indent=2)
                        st.info(f"💾 Evaluation saved to job: {job_dir.name}")
                except Exception as e:
                    st.warning(f"Could not save evaluation to job directory: {e}")

            st.download_button(
                "📥 Download JSON",
                data=json.dumps(result.to_dict(), indent=2),
                file_name="evaluation.json",
                mime="application/json",
                use_container_width=True,
            )

            # Agent advice on evaluation results
            eval_ctx_parts = []
            if result.clash_score is not None:
                eval_ctx_parts.append(f"Clash Score: {result.clash_score:.2f}")
            if result.contact_energy is not None:
                eval_ctx_parts.append(f"Contact Energy: {result.contact_energy:.2f}")
            if result.sasa_total is not None:
                eval_ctx_parts.append(f"SASA: {result.sasa_total:.1f} Å²")
            if result.lddt is not None:
                eval_ctx_parts.append(f"lDDT: {result.lddt:.3f}")
            if result.tm_score is not None:
                eval_ctx_parts.append(f"TM-score: {result.tm_score:.3f}")
            if result.rmsd is not None:
                eval_ctx_parts.append(f"RMSD: {result.rmsd:.2f} Å")
            if result.cad_score is not None:
                eval_ctx_parts.append(f"CAD-score: {result.cad_score:.3f}")
            if result.voromqa_score is not None:
                eval_ctx_parts.append(f"VoroMQA: {result.voromqa_score:.3f}")

            if eval_ctx_parts:
                # AI contextual insight
                render_contextual_insight(
                    "Evaluation",
                    {p.split(":")[0].strip(): p.split(":")[1].strip() for p in eval_ctx_parts},
                    key_prefix="eval_ctx",
                )

                render_agent_advice_panel(
                    page_context="Evaluation metrics:\n" + "\n".join(f"- {p}" for p in eval_ctx_parts),
                    default_question=(
                        "Interpret these evaluation metrics. Is this structure suitable for "
                        "downstream use (docking, design)? Should it be refined?"
                    ),
                    expert="Liam",
                    key_prefix="eval_agent",
                )

                # All-experts investigation
                eval_ctx = "Evaluation metrics:\n" + "\n".join(f"- {p}" for p in eval_ctx_parts)
                render_all_experts_panel(
                    "🧠 All-Expert Investigation (evaluation results)",
                    agenda=(
                        "Interpret the evaluation metrics and advise on structural quality, "
                        "refinement needs, and readiness for downstream tasks."
                    ),
                    context=eval_ctx,
                    questions=(
                        "Is the model quality sufficient for downstream use (docking/design)?",
                        "Which metrics indicate potential issues or refinement targets?",
                        "What specific refinement or validation steps should be run next?",
                    ),
                    key_prefix="eval_all",
                )

            # Cross-page navigation
            st.markdown("---")
            section_header("Next Steps", "Continue your workflow", "➡️")
            cross_page_actions([
                {"label": "Compare Predictors", "page": "pages/3_compare.py", "icon": "⚖️"},
                {"label": "Scan Mutations", "page": "pages/10_mutation_scanner.py", "icon": "🧬"},
                {"label": "MPNN Design", "page": "pages/8_mpnn.py", "icon": "🎯"},
            ])

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            import traceback

            with st.expander("Error details"):
                st.code(traceback.format_exc())

if run_comprehensive:
    if not model_file and chosen is None:
        st.error("Please upload a model structure or select a recent output.")
    elif not reference_file:
        st.error("Please upload a reference structure for comparison")
    else:
        try:
            # Resolve model path (uploaded or recent output)
            if chosen is not None and model_file is None:
                model_path = Path(chosen)
            else:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(model_file.name).suffix, delete=False
                ) as tmp:
                    tmp.write(model_file.read())
                    model_path = Path(tmp.name)

            with tempfile.NamedTemporaryFile(
                suffix=Path(reference_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(reference_file.read())
                reference_path = Path(tmp.name)

            with st.spinner("🔬 Computing all metrics... This may take a moment."):
                # Use comprehensive evaluation
                from protein_design_hub.evaluation.composite import CompositeEvaluator
                from protein_design_hub.core.config import get_settings

                settings = get_settings()
                settings.evaluation.lddt.inclusion_radius = lddt_radius
                settings.evaluation.lddt.sequence_separation = lddt_seq_sep

                evaluator = CompositeEvaluator(settings=settings)

                if eval_mode == "Comprehensive (All Levels)":
                    # Use comprehensive evaluation
                    results = evaluator.evaluate_comprehensive(model_path, reference_path)
                else:
                    # Quick evaluation
                    result = evaluator.evaluate(model_path, reference_path)
                    results = {
                        "global": {
                            "lddt": result.lddt,
                            "rmsd_ca": result.rmsd,
                            "tm_score": result.tm_score,
                            "qs_score": result.qs_score,
                        },
                        "per_residue": {"lddt": result.lddt_per_residue or []},
                        "per_chain": {},
                        "interface": {},
                    }

            st.success("✅ Evaluation Complete!")

            # Store results in session state
            st.session_state.eval_results = results
            
            # Save results if part of a job
            if chosen:
                try:
                    chosen_path = Path(chosen)
                    job_dir = chosen_path.parent.parent
                    if (job_dir / "prediction_summary.json").exists():
                        eval_dir = job_dir / "evaluation"
                        eval_dir.mkdir(exist_ok=True)
                        with open(eval_dir / "comprehensive_metrics.json", "w") as f:
                            json.dump(results, f, indent=2)
                        st.info(f"💾 Comprehensive evaluation saved to job: {job_dir.name}")
                except Exception as e:
                    st.warning(f"Could not save evaluation to job directory: {e}")

            # ========== GLOBAL METRICS ==========
            st.markdown(
                '<div class="section-header"><h3>🌍 Global Metrics</h3></div>',
                unsafe_allow_html=True,
            )

            global_metrics = results.get("global", {})

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                lddt = global_metrics.get("lddt")
                if lddt is not None:
                    quality = "success" if lddt > 0.7 else "warning" if lddt > 0.5 else "error"
                    st.markdown(
                        f"""
                    <div class="metric-card metric-card-{quality}">
                        <div class="metric-value">{lddt:.4f}</div>
                        <div class="metric-label">lDDT</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric("lDDT", "N/A")

            with col2:
                rmsd = global_metrics.get("rmsd_ca") or global_metrics.get("rmsd")
                if rmsd is not None:
                    quality = "success" if rmsd < 2 else "warning" if rmsd < 4 else "error"
                    st.markdown(
                        f"""
                    <div class="metric-card metric-card-{quality}">
                        <div class="metric-value">{rmsd:.2f} Å</div>
                        <div class="metric-label">RMSD (Cα)</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric("RMSD", "N/A")

            with col3:
                tm = global_metrics.get("tm_score")
                if tm is not None:
                    quality = "success" if tm > 0.7 else "warning" if tm > 0.5 else "error"
                    st.markdown(
                        f"""
                    <div class="metric-card metric-card-{quality}">
                        <div class="metric-value">{tm:.4f}</div>
                        <div class="metric-label">TM-score</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric("TM-score", "N/A")

            with col4:
                qs = global_metrics.get("qs_score")
                if qs is not None:
                    quality = "success" if qs > 0.7 else "warning" if qs > 0.5 else "error"
                    st.markdown(
                        f"""
                    <div class="metric-card metric-card-{quality}">
                        <div class="metric-value">{qs:.4f}</div>
                        <div class="metric-label">QS-score</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric("QS-score", "N/A")

            # Additional global metrics - Row 2
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                gdt_ts = global_metrics.get("gdt_ts")
                if gdt_ts is not None:
                    st.metric("GDT-TS", f"{gdt_ts:.4f}")

            with col6:
                gdt_ha = global_metrics.get("gdt_ha")
                if gdt_ha is not None:
                    st.metric("GDT-HA", f"{gdt_ha:.4f}")

            with col7:
                rmsd_bb = global_metrics.get("rmsd_backbone")
                if rmsd_bb is not None:
                    st.metric("RMSD (Backbone)", f"{rmsd_bb:.2f} Å")

            with col8:
                lddt_mean = global_metrics.get("lddt_mean")
                if lddt_mean is not None:
                    st.metric("lDDT Mean", f"{lddt_mean:.4f}")

            # Additional metrics - Row 3 (BB-lDDT, iLDDT, CAD)
            col9, col10, col11, col12 = st.columns(4)

            with col9:
                bb_lddt = global_metrics.get("bb_lddt")
                if bb_lddt is not None:
                    st.metric("BB-lDDT (Backbone)", f"{bb_lddt:.4f}")

            with col10:
                ilddt = global_metrics.get("ilddt")
                if ilddt is not None:
                    st.metric("iLDDT (Inter-chain)", f"{ilddt:.4f}")

            with col11:
                cad = global_metrics.get("cad_score")
                if cad is not None:
                    st.metric("CAD-score", f"{cad:.4f}")

            with col12:
                patch = global_metrics.get("patch_score")
                if patch is not None:
                    st.metric("Patch Score (CASP15)", f"{patch:.4f}")

            # ========== DOCKQ AND INTERFACE QUALITY ==========
            interface = results.get("interface", {})
            dockq = global_metrics.get("dockq")

            if dockq is not None or interface.get("dockq_details"):
                st.markdown(
                    '<div class="section-header"><h3>🎯 DockQ Interface Quality</h3></div>',
                    unsafe_allow_html=True,
                )

                # Global DockQ
                col_dq1, col_dq2, col_dq3, col_dq4 = st.columns(4)

                with col_dq1:
                    if dockq is not None:
                        # DockQ classification
                        if dockq >= 0.8:
                            dockq_class = "High Quality"
                            quality = "success"
                        elif dockq >= 0.49:
                            dockq_class = "Medium Quality"
                            quality = "info"
                        elif dockq >= 0.23:
                            dockq_class = "Acceptable"
                            quality = "warning"
                        else:
                            dockq_class = "Incorrect"
                            quality = "error"

                        st.markdown(
                            f"""
                        <div class="metric-card metric-card-{quality}">
                            <div class="metric-value">{dockq:.4f}</div>
                            <div class="metric-label">DockQ ({dockq_class})</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                # Per-interface DockQ details
                dockq_details = interface.get("dockq_details", [])
                if isinstance(dockq_details, list) and len(dockq_details) > 0:
                    with st.expander("📊 Per-Interface DockQ Details"):
                        import pandas as pd

                        df_dockq = pd.DataFrame(
                            [
                                {
                                    "Interface": f"Interface {i+1}",
                                    "DockQ": d.get("dockq", 0),
                                    "fnat": d.get("fnat", 0),
                                    "fnonnat": d.get("fnonnat", 0),
                                    "iRMSD (Å)": d.get("irmsd", 0),
                                    "lRMSD (Å)": d.get("lrmsd", 0),
                                }
                                for i, d in enumerate(dockq_details)
                            ]
                        )
                        st.dataframe(df_dockq, use_container_width=True, hide_index=True)

                        # DockQ component explanation
                        st.markdown(
                            """
                        **DockQ Components:**
                        - **fnat**: Fraction of native contacts preserved
                        - **fnonnat**: Fraction of non-native contacts (false positives)
                        - **iRMSD**: Interface RMSD (Å) - backbone atoms within 10Å of interface
                        - **lRMSD**: Ligand RMSD (Å) - smaller chain after superposition on larger
                        """
                        )

            # ========== ICS AND IPS METRICS ==========
            ics = global_metrics.get("ics") or interface.get("ics")
            ips = global_metrics.get("ips") or interface.get("ips")

            if ics is not None or ips is not None:
                st.markdown(
                    '<div class="section-header"><h3>🔗 Interface Contact & Patch Similarity</h3></div>',
                    unsafe_allow_html=True,
                )

                col_ics1, col_ics2, col_ics3, col_ics4 = st.columns(4)

                with col_ics1:
                    if ics is not None:
                        st.metric("ICS (F1)", f"{ics:.4f}")

                with col_ics2:
                    ics_prec = interface.get("ics_precision")
                    if ics_prec is not None:
                        st.metric("ICS Precision", f"{ics_prec:.4f}")

                with col_ics3:
                    ics_rec = interface.get("ics_recall")
                    if ics_rec is not None:
                        st.metric("ICS Recall", f"{ics_rec:.4f}")

                with col_ics4:
                    if ips is not None:
                        st.metric("IPS (Jaccard)", f"{ips:.4f}")

                st.caption(
                    "ICS = Interface Contact Similarity (how well contacts are preserved), IPS = Interface Patch Similarity (spatial overlap)"
                )

            # ========== PATCH SCORES (CASP15) ==========
            patch_scores = interface.get("patch_scores", [])
            if patch_scores and isinstance(patch_scores, list) and len(patch_scores) > 0:
                st.markdown(
                    '<div class="section-header"><h3>🏆 Patch Scores (CASP15 Local Interface)</h3></div>',
                    unsafe_allow_html=True,
                )

                import pandas as pd

                df_patch = pd.DataFrame(patch_scores)
                if not df_patch.empty:
                    st.dataframe(df_patch, use_container_width=True, hide_index=True)

                    # Patch score visualization
                    if "score" in df_patch.columns:
                        import plotly.express as px

                        fig = px.bar(df_patch, y="score", title="Patch Scores by Interface Region")
                        fig.update_layout(yaxis_range=[0, 1], height=300)
                        st.plotly_chart(fig, use_container_width=True)

            # lDDT Quality Distribution
            quality_cats = global_metrics.get("lddt_quality_categories", {})
            if quality_cats:
                st.markdown("#### 📊 lDDT Quality Distribution")

                total = sum(quality_cats.values())
                if total > 0:
                    very_high = quality_cats.get("very_high_gt_90", 0)
                    confident = quality_cats.get("confident_70_90", 0)
                    low = quality_cats.get("low_50_70", 0)
                    very_low = quality_cats.get("very_low_lt_50", 0)

                    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                    with col_q1:
                        st.metric("Very High (>90%)", f"{very_high} ({very_high/total*100:.1f}%)")
                    with col_q2:
                        st.metric("Confident (70-90%)", f"{confident} ({confident/total*100:.1f}%)")
                    with col_q3:
                        st.metric("Low (50-70%)", f"{low} ({low/total*100:.1f}%)")
                    with col_q4:
                        st.metric("Very Low (<50%)", f"{very_low} ({very_low/total*100:.1f}%)")

                    # Visual bar
                    import plotly.graph_objects as go

                    fig = go.Figure(
                        go.Bar(
                            x=[very_high, confident, low, very_low],
                            y=[
                                "Very High (>90%)",
                                "Confident (70-90%)",
                                "Low (50-70%)",
                                "Very Low (<50%)",
                            ],
                            orientation="h",
                            marker_color=["#0053d6", "#65cbf3", "#ffdb13", "#ff7d45"],
                            text=[
                                f"{x} ({x/total*100:.1f}%)"
                                for x in [very_high, confident, low, very_low]
                            ],
                            textposition="inside",
                        )
                    )
                    fig.update_layout(
                        title="Residue Quality Distribution",
                        height=200,
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ========== PER-RESIDUE METRICS ==========
            per_residue = results.get("per_residue", {})
            lddt_per_res = per_residue.get("lddt", [])

            if lddt_per_res:
                st.markdown(
                    '<div class="section-header"><h3>📍 Per-Residue Metrics</h3></div>',
                    unsafe_allow_html=True,
                )

                import plotly.express as px
                import plotly.graph_objects as go
                import pandas as pd

                # lDDT per residue plot
                df = pd.DataFrame(
                    {
                        "Residue": range(1, len(lddt_per_res) + 1),
                        "lDDT": lddt_per_res,
                    }
                )

                fig = px.line(df, x="Residue", y="lDDT", title="lDDT per Residue")

                # Add colored background regions for quality
                fig.add_hrect(y0=0.9, y1=1.0, fillcolor="#0053d6", opacity=0.1, line_width=0)
                fig.add_hrect(y0=0.7, y1=0.9, fillcolor="#65cbf3", opacity=0.1, line_width=0)
                fig.add_hrect(y0=0.5, y1=0.7, fillcolor="#ffdb13", opacity=0.1, line_width=0)
                fig.add_hrect(y0=0.0, y1=0.5, fillcolor="#ff7d45", opacity=0.1, line_width=0)

                fig.update_layout(yaxis_range=[0, 1], height=400)
                fig.update_traces(line_color="#667eea")
                st.plotly_chart(fig, use_container_width=True)

                # Detailed residue table
                lddt_details = per_residue.get("lddt_details", [])
                if lddt_details:
                    with st.expander("📋 Detailed Per-Residue Data"):
                        df_details = pd.DataFrame(lddt_details)
                        if not df_details.empty:
                            # Color code by quality
                            def color_lddt(val):
                                if val >= 0.9:
                                    return "background-color: #0053d6; color: #e5e7eb"
                                elif val >= 0.7:
                                    return "background-color: #65cbf3"
                                elif val >= 0.5:
                                    return "background-color: #ffdb13"
                                else:
                                    return "background-color: #ff7d45; color: #e5e7eb"

                            if "lddt" in df_details.columns:
                                styled = df_details.style.applymap(color_lddt, subset=["lddt"])
                                st.dataframe(styled, use_container_width=True, height=400)
                            else:
                                st.dataframe(df_details, use_container_width=True, height=400)

            # ========== PER-CHAIN METRICS ==========
            per_chain = results.get("per_chain", {})
            if per_chain:
                st.markdown(
                    '<div class="section-header"><h3>🔗 Per-Chain Metrics</h3></div>',
                    unsafe_allow_html=True,
                )

                chain_data = []
                for chain_name, chain_metrics in per_chain.items():
                    if isinstance(chain_metrics, dict):
                        row = {"Chain": chain_name}
                        if "lddt" in chain_metrics and isinstance(chain_metrics["lddt"], dict):
                            row.update(chain_metrics["lddt"])
                        else:
                            row.update(chain_metrics)
                        chain_data.append(row)

                if chain_data:
                    import pandas as pd

                    df_chains = pd.DataFrame(chain_data)
                    st.dataframe(df_chains, use_container_width=True, hide_index=True)

                    # Chain comparison chart
                    if "mean_lddt" in df_chains.columns:
                        import plotly.express as px

                        fig = px.bar(
                            df_chains,
                            x="Chain",
                            y="mean_lddt",
                            title="lDDT by Chain",
                            color="mean_lddt",
                            color_continuous_scale="RdYlGn",
                        )
                        fig.update_layout(yaxis_range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)

            # ========== INTERFACE METRICS ==========
            interface = results.get("interface", {})
            if interface and not interface.get("qs_note"):
                st.markdown(
                    '<div class="section-header"><h3>🤝 Interface Metrics</h3></div>',
                    unsafe_allow_html=True,
                )

                if interface.get("qs_error"):
                    st.warning(f"QS-score error: {interface['qs_error']}")
                else:
                    col_if1, col_if2 = st.columns(2)

                    with col_if1:
                        if interface.get("chain_mapping"):
                            st.markdown("**Chain Mapping:**")
                            st.json(interface["chain_mapping"])

                    with col_if2:
                        if interface.get("mapped_target_chains"):
                            st.markdown(
                                f"**Mapped Target Chains:** {', '.join(interface['mapped_target_chains'])}"
                            )
                        if interface.get("mapped_model_chains"):
                            st.markdown(
                                f"**Mapped Model Chains:** {', '.join(interface['mapped_model_chains'])}"
                            )

            elif interface.get("qs_note"):
                st.info(f"ℹ️ {interface['qs_note']}")

            # ========== CONTACT INFORMATION ==========
            if global_metrics.get("lddt_total_contacts"):
                st.markdown(
                    '<div class="section-header"><h3>📊 Contact Statistics</h3></div>',
                    unsafe_allow_html=True,
                )

                col_c1, col_c2, col_c3 = st.columns(3)

                with col_c1:
                    st.metric("Total Contacts", global_metrics.get("lddt_total_contacts", "N/A"))

                with col_c2:
                    st.metric(
                        "Conserved Contacts", global_metrics.get("lddt_conserved_contacts", "N/A")
                    )

                with col_c3:
                    total = global_metrics.get("lddt_total_contacts", 0)
                    conserved = global_metrics.get("lddt_conserved_contacts", 0)
                    if total > 0:
                        pct = conserved / total * 100
                        st.metric("Conservation Rate", f"{pct:.1f}%")

            # ========== PAE VISUALIZATION ==========
            # Check for PAE data in results or uploaded files
            pae_data = results.get("pae") or results.get("predicted_aligned_error")

            # Try to load PAE from output directory if not in results
            if pae_data is None:
                # Look for PAE JSON files
                pae_files = list(model_path.parent.glob("*pae*.json")) + list(
                    model_path.parent.glob("*scores*.json")
                )
                for pae_file in pae_files:
                    try:
                        from protein_design_hub.web.visualizations import load_pae_from_json

                        pae_data = load_pae_from_json(pae_file)
                        if pae_data:
                            break
                    except Exception:
                        pass

            if pae_data:
                st.markdown(
                    '<div class="section-header"><h3>🎨 Predicted Aligned Error (PAE)</h3></div>',
                    unsafe_allow_html=True,
                )

                try:
                    from protein_design_hub.web.visualizations import create_pae_heatmap

                    fig_pae = create_pae_heatmap(pae_data)
                    st.plotly_chart(fig_pae, use_container_width=True)

                    st.caption(
                        "PAE shows the expected position error (Å) when residue X is used to align the structure. Low values (green) indicate high confidence in relative positions."
                    )
                except Exception as e:
                    st.warning(f"Could not render PAE: {e}")

            # ========== CONTACT MAP VISUALIZATION ==========
            st.markdown(
                '<div class="section-header"><h3>🔗 Contact Map Analysis</h3></div>',
                unsafe_allow_html=True,
            )

            col_cm1, col_cm2 = st.columns(2)

            with col_cm1:
                contact_threshold = st.slider("Contact threshold (Å)", 4.0, 12.0, 8.0, 0.5)

            with col_cm2:
                show_comparison = st.checkbox("Show model vs reference comparison", value=True)

            try:
                from protein_design_hub.web.visualizations import (
                    compute_contact_map_from_structure,
                    create_contact_map,
                )

                model_contacts = compute_contact_map_from_structure(model_path)

                if show_comparison and reference_path:
                    ref_contacts = compute_contact_map_from_structure(reference_path)
                    fig_cm = create_contact_map(
                        model_contacts, ref_contacts, threshold=contact_threshold
                    )
                else:
                    fig_cm = create_contact_map(model_contacts, threshold=contact_threshold)

                st.plotly_chart(fig_cm, use_container_width=True)

                # Contact statistics
                model_contact_count = (model_contacts < contact_threshold).sum() - len(
                    model_contacts
                )
                st.caption(
                    f"Model has {model_contact_count // 2} contacts at {contact_threshold}Å threshold"
                )

            except Exception as e:
                st.info(f"Contact map visualization requires Biopython and scipy: {e}")

            # ========== DOWNLOAD RESULTS ==========
            st.markdown("---")

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                st.download_button(
                    "📥 Download Full Results (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name="evaluation_results_comprehensive.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with col_dl2:
                # Generate text report
                report = []
                report.append("=" * 60)
                report.append("COMPREHENSIVE STRUCTURE EVALUATION REPORT")
                report.append("=" * 60)
                report.append("")
                report.append("GLOBAL METRICS:")
                report.append("-" * 40)
                for key, val in global_metrics.items():
                    if isinstance(val, (int, float)):
                        report.append(
                            f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}"
                        )
                report.append("")
                report.append(f"Per-residue data: {len(lddt_per_res)} residues")
                report.append(f"Chains analyzed: {list(per_chain.keys())}")

                st.download_button(
                    "📥 Download Report (TXT)",
                    data="\n".join(report),
                    file_name="evaluation_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            import traceback

            with st.expander("Error details"):
                st.code(traceback.format_exc())

# ========== METRIC DESCRIPTIONS ==========
with st.expander("ℹ️ Metric Descriptions"):
    st.markdown(
        """
    ### 🎯 Global Metrics

    #### lDDT (Local Distance Difference Test)
    Measures local structural accuracy by comparing interatomic distances between model and reference.
    - **Range**: 0-1 (higher is better)
    - **Interpretation**:
        - >0.9: Very high confidence (excellent)
        - 0.7-0.9: Confident (good)
        - 0.5-0.7: Low confidence
        - <0.5: Very low confidence (poor)

    #### BB-lDDT (Backbone lDDT)
    Backbone-only lDDT using CA atoms for proteins or C3' for nucleotides.
    - More lenient than full lDDT as it ignores side chains.

    #### iLDDT (Inter-chain lDDT)
    lDDT computed only on inter-chain contacts.
    - Specifically measures interface accuracy independent of individual chain quality.

    #### RMSD (Root Mean Square Deviation)
    Measures average distance between aligned atoms after optimal superposition.
    - **Units**: Angstroms (Å)
    - **Interpretation**: Lower is better
        - <2Å: Excellent structural similarity
        - 2-4Å: Good similarity
        - >4Å: Significant deviations

    #### TM-score (Template Modeling Score)
    Measures global structural similarity, length-normalized.
    - **Range**: 0-1 (higher is better)
    - **Interpretation**:
        - >0.5: Same fold
        - >0.7: High structural similarity
        - >0.9: Nearly identical structures

    #### CAD-score (Contact Area Difference)
    Measures contact surface area similarity using Voronoi tessellation.
    - **Range**: 0-1 (higher is better)
    - **Requires**: voronota-cadscore external tool

    #### VoroMQA
    Voronoi-based model quality assessment (single-structure).
    - **Range**: higher is better (unitless)
    - **Requires**: voronota-voromqa external tool

    #### GDT-TS/GDT-HA (Global Distance Test)
    - **GDT-TS**: Percentage of residues within 1, 2, 4, 8 Å of reference
    - **GDT-HA**: High-accuracy version (0.5, 1, 2, 4 Å thresholds)

    ---

    ### 🎯 Interface Metrics

    #### DockQ (Docking Quality Score)
    Comprehensive interface quality metric combining:
    - **fnat**: Fraction of native contacts preserved (0-1)
    - **fnonnat**: Fraction of false positive contacts (lower is better)
    - **iRMSD**: Interface RMSD - backbone atoms within 10Å of interface
    - **lRMSD**: Ligand RMSD - smaller chain after superposition on larger

    **DockQ Classification:**
    | DockQ | Classification |
    |-------|----------------|
    | ≥0.80 | High Quality |
    | 0.49-0.80 | Medium Quality |
    | 0.23-0.49 | Acceptable |
    | <0.23 | Incorrect |

    #### QS-score (Quaternary Structure Score)
    Evaluates overall interface quality in multimeric structures.
    - **Range**: 0-1 (higher is better)
    - **Note**: Only computed for complexes with multiple chains

    #### ICS (Interface Contact Similarity)
    Measures how well interface contacts are preserved.
    - **Precision**: Fraction of predicted contacts that are native
    - **Recall**: Fraction of native contacts that are predicted
    - **F1**: Harmonic mean of precision and recall

    #### IPS (Interface Patch Similarity)
    Measures spatial overlap of interface patches using Jaccard coefficient.
    - **Range**: 0-1 (higher is better)
    - Focuses on the physical location of interface residues

    #### Patch Scores (CASP15)
    Local interface quality scores used in CASP15 assessment.
    - Evaluates specific interface regions independently
    - Useful for identifying well-modeled vs. poorly-modeled interface regions

    ---

    ### 📍 Per-Residue Metrics
    Per-residue lDDT values identify local regions of high/low accuracy.

    ### 🔗 Per-Chain Metrics
    Chain-level aggregated metrics useful for analyzing multimeric structures.

    ### 🤝 Interface Metrics
    Chain mapping and interface quality for protein-protein interfaces.
    """
    )

# OpenStructure status is shown in the sidebar "Tool Status" expander above
