"""Jobs browser page."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from protein_design_hub.web.ui import (
    get_selected_backbone,
    get_selected_model,
    inject_base_css,
    list_jobs,
    page_header,
    section_header,
    info_box,
    empty_state,
    status_badge,
    set_selected_backbone,
    set_selected_model,
    sidebar_nav,
    sidebar_system_status,
)

st.set_page_config(page_title="Jobs - Protein Design Hub", page_icon="📁", layout="wide")
inject_base_css()

# Page-specific styling
st.markdown("""
<style>
.job-card {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-lg);
    padding: var(--pdhub-space-lg);
    margin-bottom: var(--pdhub-space-md);
    border: 1px solid var(--pdhub-border);
    transition: var(--pdhub-transition);
}

.job-card:hover {
    box-shadow: var(--pdhub-shadow-md);
    border-color: var(--pdhub-primary-light);
}

.job-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: var(--pdhub-space-md);
}

.job-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--pdhub-text);
    margin-bottom: 4px;
}

.job-meta {
    font-size: 0.85rem;
    color: var(--pdhub-text-secondary);
}

.job-path {
    font-family: monospace;
    font-size: 0.75rem;
    color: var(--pdhub-text-muted);
    background: var(--pdhub-bg-light);
    padding: 4px 8px;
    border-radius: var(--pdhub-border-radius-sm);
    margin-top: 8px;
    word-break: break-all;
}

.artifact-badges {
    display: flex;
    gap: var(--pdhub-space-sm);
    flex-wrap: wrap;
    margin: var(--pdhub-space-md) 0;
}

.artifact-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: var(--pdhub-border-radius-full);
    font-size: 0.8rem;
    font-weight: 500;
}

.artifact-badge-ok {
    background: var(--pdhub-success-light);
    color: #155724;
}

.artifact-badge-missing {
    background: var(--pdhub-bg-light);
    color: var(--pdhub-text-muted);
}

.selection-banner {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-radius: var(--pdhub-border-radius-md);
    padding: var(--pdhub-space-md) var(--pdhub-space-lg);
    margin-bottom: var(--pdhub-space-lg);
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid #90caf9;
}

.selection-info {
    font-family: monospace;
    font-size: 0.85rem;
    color: var(--pdhub-text);
}
</style>
""", unsafe_allow_html=True)

# Header
page_header(
    "Job Browser",
    "Browse recent output folders and quickly load structures for evaluation or design",
    "📁"
)

sidebar_nav(current="Jobs")
sidebar_system_status()

try:
    from protein_design_hub.core.config import get_settings

    settings = get_settings()
    base_dir = Path(settings.output.base_dir)
except Exception:
    base_dir = Path("./outputs")

# Sidebar controls
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
limit = st.sidebar.slider("Max jobs to show", min_value=10, max_value=200, value=50, step=10)
show_only_with_prediction = st.sidebar.checkbox("Only with predictions", value=False)

jobs = list_jobs(base_dir, limit=limit)

if show_only_with_prediction:
    jobs = [j for j in jobs if j["has_prediction"]]

if not jobs:
    empty_state(
        title="No jobs found",
        message=f"No job folders found under {base_dir}. Run some predictions to see them here.",
        icon="📭"
    )
    st.stop()


def _pick_structure_from_prediction_summary(summary_path: Path) -> Optional[Path]:
    try:
        data = json.loads(summary_path.read_text())
        predictors = data.get("predictors", {})
        for _, p in predictors.items():
            for sp in p.get("structure_paths", []) or []:
                cand = Path(sp)
                if cand.exists():
                    return cand
        return None
    except Exception:
        return None


selected_model = get_selected_model()
selected_backbone = get_selected_backbone()

# Show current selection
if selected_model or selected_backbone:
    st.markdown("""
    <div class="selection-banner">
        <div>
            <strong>🎯 Current Selection</strong><br>
            <span class="selection-info">
    """, unsafe_allow_html=True)

    if selected_model:
        st.markdown(f'Model: <code>{selected_model}</code>', unsafe_allow_html=True)
    if selected_backbone:
        st.markdown(f'Backbone: <code>{selected_backbone}</code>', unsafe_allow_html=True)

    st.markdown('</span></div>', unsafe_allow_html=True)

    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("Clear Selection", use_container_width=True, type="secondary"):
            set_selected_model(None)
            set_selected_backbone(None)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Job count info
section_header("Recent Jobs", f"{len(jobs)} jobs found", "📋")

for job in jobs:
    job_id = job["job_id"]
    job_path = Path(job["path"])
    dt = datetime.fromtimestamp(job["mtime"]).strftime("%Y-%m-%d %H:%M")

    inferred_model = None
    if job["has_prediction"]:
        inferred_model = _pick_structure_from_prediction_summary(job["prediction_summary"])

    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 2, 1.5])

        with col1:
            st.markdown(f"""
            <div class="job-title">📁 {job_id}</div>
            <div class="job-meta">🕐 {dt}</div>
            """, unsafe_allow_html=True)

            # Artifact badges
            badges_html = '<div class="artifact-badges">'
            if job['has_prediction']:
                badges_html += '<span class="artifact-badge artifact-badge-ok">✅ Prediction</span>'
            if job['has_design']:
                badges_html += '<span class="artifact-badge artifact-badge-ok">✅ Design</span>'
            if job['has_compare']:
                badges_html += '<span class="artifact-badge artifact-badge-ok">✅ Evaluation</span>'
            if job.get('has_evolution'):
                 badges_html += '<span class="artifact-badge artifact-badge-ok" style="background:#e8f5e9; border-color:#81c784; color:#2e7d32;">✅ Evolution</span>'
            if job.get('has_scan'):
                 badges_html += '<span class="artifact-badge artifact-badge-ok" style="background:#f3e5f5; border-color:#ba68c8; color:#7b1fa2;">✅ Scan</span>'
            badges_html += '</div>'
            st.markdown(badges_html, unsafe_allow_html=True)

        with col2:
            if inferred_model is not None:
                st.markdown("**Structure:**")
                st.code(str(inferred_model.name), language=None)
            elif job.get('has_evolution'):
                st.markdown("**Type:**")
                st.caption("Iterative Optimization")
            elif job.get('has_scan'):
                st.markdown("**Type:**")
                st.caption("Mutation Scan")
            else:
                st.caption("No details available")

        with col3:
            st.markdown("**Actions**")
            # Prediction/Evaluation paths
            if inferred_model is not None:
                if st.button("📊 Evaluate", key=f"eval_{job_id}", use_container_width=True):
                    set_selected_model(inferred_model)
                    st.switch_page("pages/2_evaluate.py")

                if st.button("🎯 MPNN Design", key=f"mpnn_{job_id}", use_container_width=True):
                    set_selected_backbone(inferred_model)
                    st.switch_page("pages/8_mpnn.py")

            # Evolution path
            if job.get('has_evolution'):
                if st.button("🧬 View Evo", key=f"view_evo_{job_id}", use_container_width=True):
                    st.session_state["evolution_job_to_load"] = str(job["path"])
                    st.switch_page("pages/4_evolution.py")

            # Scan path
            if job.get('has_scan'):
                if st.button("🔬 View Scan", key=f"view_scan_{job_id}", use_container_width=True):
                    st.session_state["scan_job_to_load"] = str(job["path"])
                    st.switch_page("pages/10_mutation_scanner.py")
            
            if not (inferred_model or job.get('has_evolution') or job.get('has_scan')):
                st.caption("No actions available")
