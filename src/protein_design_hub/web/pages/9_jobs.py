"""Jobs browser page."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from protein_design_hub.web.agent_helpers import (
    agent_sidebar_status,
    render_contextual_insight,
    render_agent_advice_panel,
    render_all_experts_panel,
)

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
    metric_card,  # Added for new UI
    render_badge, # Added for new UI
    card_start,   # Added for new UI
    card_end,     # Added for new UI
    workflow_breadcrumb,
    cross_page_actions,
)

st.set_page_config(page_title="Jobs - Protein Design Hub", page_icon="📁", layout="wide")
inject_base_css()
sidebar_nav(current="Jobs")
sidebar_system_status()
agent_sidebar_status()

# Header
page_header(
    "Job Browser",
    "Explore, analyze, and load your protein design experiments.",
    "📁"
)

workflow_breadcrumb(
    ["Run Jobs", "Browse Results", "Analyze", "Export"],
    current=1,
)

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
col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
with col_stats1: metric_card(len(jobs), "Total Jobs", "info", "📁")
with col_stats2: metric_card(sum(1 for j in jobs if j["has_prediction"]), "Predictions", "gradient", "🔮")
with col_stats3: metric_card(len([j for j in jobs if j.get('has_scan') or j.get('has_evolution')]), "Analyses", "warning", "🧬")
with col_stats4: metric_card(sum(j.get("meeting_count", 0) for j in jobs), "Meetings", "default", "🧠")

st.markdown("<br>", unsafe_allow_html=True)
section_header("Registry Index", f"Showing {len(jobs)} most recent runs", "📋")

# Grid layout for jobs
cols_per_row = 2
for i in range(0, len(jobs), cols_per_row):
    row_jobs = jobs[i:i+cols_per_row]
    cols = st.columns(cols_per_row)
    
    for idx, job in enumerate(row_jobs):
        with cols[idx]:
            job_id = job["job_id"]
            dt = datetime.fromtimestamp(job["mtime"]).strftime("%m/%d %H:%M")
            
            inferred_model = None
            if job["has_prediction"]:
                inferred_model = _pick_structure_from_prediction_summary(job["prediction_summary"])
            
            with st.container(border=True):
                st.markdown(f"**📁 {job_id}**")
                st.markdown(f'<div class="pdhub-muted">⏱️ {dt}</div>', unsafe_allow_html=True)

                # Artifacts
                badge_html = []
                if job['has_prediction']:
                    badge_html.append('<span style="background:#22c55e;color:#e5e7eb;padding:2px 8px;border-radius:4px;font-size:0.75rem;margin-right:4px;">Predict</span>')
                if job['has_design']:
                    badge_html.append('<span style="background:#667eea;color:#e5e7eb;padding:2px 8px;border-radius:4px;font-size:0.75rem;margin-right:4px;">Design</span>')
                if job['has_compare']:
                    badge_html.append('<span style="background:#60a5fa;color:#e5e7eb;padding:2px 8px;border-radius:4px;font-size:0.75rem;margin-right:4px;">Eval</span>')
                if job.get('has_evolution'):
                    badge_html.append('<span style="background:#f59e0b;color:#e5e7eb;padding:2px 8px;border-radius:4px;font-size:0.75rem;margin-right:4px;">Evo</span>')
                if job.get('has_scan'):
                    badge_html.append('<span style="background:#8b5cf6;color:#e5e7eb;padding:2px 8px;border-radius:4px;font-size:0.75rem;margin-right:4px;">Scan</span>')
                if job.get('has_meetings'):
                    badge_html.append(
                        f'<span style="background:#ec4899;color:#e5e7eb;padding:2px 8px;border-radius:4px;'
                        f'font-size:0.75rem;margin-right:4px;">Meetings:{job.get("meeting_count", 0)}</span>'
                    )

                if badge_html:
                    st.markdown(f'<div style="margin: 0.5rem 0;">{"".join(badge_html)}</div>', unsafe_allow_html=True)

                # At-a-glance stats from prediction_summary
                if job["has_prediction"] and job.get("prediction_summary"):
                    try:
                        _ps = json.loads(Path(job["prediction_summary"]).read_text())
                        _best_plddt = None
                        _total_runtime = 0.0
                        for _pred in _ps.get("predictors", {}).values():
                            _total_runtime += _pred.get("runtime_seconds", 0) or 0
                            for _score in _pred.get("scores", []) or []:
                                _p = _score.get("plddt")
                                if _p and (_best_plddt is None or _p > _best_plddt):
                                    _best_plddt = _p
                        _stats = []
                        if _best_plddt:
                            _plddt_color = "#22c55e" if _best_plddt >= 70 else "#f59e0b" if _best_plddt >= 50 else "#ef4444"
                            _stats.append(f'<span style="color:{_plddt_color};font-weight:600;">pLDDT {_best_plddt:.0f}</span>')
                        if _total_runtime > 0:
                            _stats.append(f'<span style="color:#a1a9b8;">{_total_runtime/60:.1f}min</span>')
                        if _stats:
                            st.markdown(f'<div style="font-size:0.8rem;margin:2px 0;">{"  ·  ".join(_stats)}</div>', unsafe_allow_html=True)
                    except Exception:
                        pass
                if inferred_model:
                    st.caption(f"Best: {inferred_model.name}")

                # Actions
                act_cols = st.columns(2)

                with act_cols[0]:
                    if job["has_prediction"]:
                        if st.button("📊 Evaluate", key=f"eval_{job_id}", use_container_width=True, disabled=not inferred_model):
                            if inferred_model:
                                set_selected_model(inferred_model)
                                st.switch_page("pages/2_evaluate.py")
                        if not inferred_model:
                            st.caption("No structure file found")

                    if job.get("has_evolution"):
                        if st.button("🧬 View Evo", key=f"vevo_{job_id}", use_container_width=True):
                            st.session_state["evolution_job_to_load"] = str(job["path"])
                            st.switch_page("pages/4_evolution.py")

                with act_cols[1]:
                    if job.get("has_scan"):
                        if st.button("🔬 View Scan", key=f"vscan_{job_id}", use_container_width=True):
                            st.session_state["scan_job_to_load"] = str(job["path"])
                            st.switch_page("pages/10_mutation_scanner.py")
                    elif inferred_model:
                        if st.button("🎯 Design", key=f"des_{job_id}", use_container_width=True):
                            set_selected_backbone(inferred_model)
                            st.switch_page("pages/8_mpnn.py")

# Job details section
st.markdown("---")
section_header("Job Details", "Select a job to view details", "📋")

if jobs:
    selected_job_id = st.selectbox(
        "Select job for details",
        options=[j["job_id"] for j in jobs],
        format_func=lambda x: x,
        key="job_details_select"
    )

    selected_job = next((j for j in jobs if j["job_id"] == selected_job_id), None)

    if selected_job:
        job_path = selected_job["path"]
        st.session_state["active_job_dir"] = str(job_path)

        col_detail, col_files = st.columns([1, 1])

        with col_detail:
            st.markdown("##### Job Information")
            st.markdown(f"**Path:** `{job_path}`")
            st.markdown(f"**Created:** {datetime.fromtimestamp(selected_job['mtime']).strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Meetings:** {selected_job.get('meeting_count', 0)}")

            # Load summary if exists
            summary_path = job_path / "prediction_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    st.markdown("##### Prediction Summary")
                    st.json(summary, expanded=False)
                except Exception as e:
                    st.warning(f"Could not load summary: {e}")

        with col_files:
            st.markdown("##### Files")
            try:
                files = list(job_path.glob("**/*"))
                pdb_files = [f for f in files if f.suffix == ".pdb"]
                json_files = [f for f in files if f.suffix == ".json"]
                meeting_json = sorted((job_path / "meetings").glob("*.json")) if (job_path / "meetings").exists() else []

                st.markdown(f"📦 **{len(pdb_files)}** PDB files")
                st.markdown(f"📄 **{len(json_files)}** JSON files")
                st.markdown(f"📁 **{len(files)}** total files")
                st.markdown(f"🧠 **{len(meeting_json)}** meeting transcripts")

                if pdb_files:
                    with st.expander("View PDB files"):
                        for pdb in pdb_files[:10]:
                            st.markdown(f"- `{pdb.name}`")
                        if len(pdb_files) > 10:
                            st.caption(f"...and {len(pdb_files) - 10} more")
                if meeting_json:
                    with st.expander("View meeting transcripts"):
                        for mt in meeting_json[:20]:
                            st.markdown(f"- `{mt.name}`")
                        if len(meeting_json) > 20:
                            st.caption(f"...and {len(meeting_json) - 20} more")
            except Exception as e:
                st.warning(f"Could not list files: {e}")

        job_ctx = "\n".join([
            f"Job ID: {selected_job['job_id']}",
            f"Has prediction: {selected_job.get('has_prediction', False)}",
            f"Has compare/eval: {selected_job.get('has_compare', False)}",
            f"Has design: {selected_job.get('has_design', False)}",
            f"Has mutation scan: {selected_job.get('has_scan', False)}",
            f"Has evolution: {selected_job.get('has_evolution', False)}",
            f"Meeting transcripts: {selected_job.get('meeting_count', 0)}",
            f"Job path: {job_path}",
        ])

        job_data = {
            "Job ID": selected_job['job_id'],
            "Has prediction": selected_job.get('has_prediction', False),
            "Has design": selected_job.get('has_design', False),
            "Has mutation scan": selected_job.get('has_scan', False),
            "Has evolution": selected_job.get('has_evolution', False),
            "Meeting transcripts": selected_job.get('meeting_count', 0),
        }
        render_contextual_insight(
            "Jobs",
            job_data,
            key_prefix=f"job_ctx_{selected_job['job_id']}",
        )

        render_agent_advice_panel(
            page_context=job_ctx,
            default_question=(
                "Is this job output sufficient for decision-making? "
                "What workflow step should be executed next?"
            ),
            expert="Computational Biologist",
            key_prefix=f"job_agent_{selected_job['job_id']}",
        )

        render_all_experts_panel(
            "All-Expert Review (selected job)",
            agenda=(
                "Review this job's artifacts and determine quality, risks, and the "
                "best next action in the workflow."
            ),
            context=job_ctx,
            questions=(
                "Is this job output sufficient for decision-making or does it need re-run/refinement?",
                "Which downstream step should be executed next for this job?",
                "Any reliability risks or missing artifacts to fix first?",
            ),
            key_prefix=f"job_all_{selected_job['job_id']}",
            save_dir=job_path / "meetings",
        )
