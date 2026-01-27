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
    set_selected_backbone,
    set_selected_model,
    sidebar_nav,
    sidebar_system_status,
)

st.set_page_config(page_title="Jobs - Protein Design Hub", page_icon="🗂️", layout="wide")
inject_base_css()
sidebar_nav(current="Jobs")
sidebar_system_status()

st.title("🗂️ Jobs")
st.caption("Browse recent output folders and jump into Evaluate/MPNN with one click.")

try:
    from protein_design_hub.core.config import get_settings

    settings = get_settings()
    base_dir = Path(settings.output.base_dir)
except Exception:
    base_dir = Path("./outputs")

st.sidebar.markdown("---")
st.sidebar.markdown("## Browser")
limit = st.sidebar.slider("Max jobs", min_value=10, max_value=200, value=50, step=10)

jobs = list_jobs(base_dir, limit=limit)
if not jobs:
    st.info(f"No job folders found under `{base_dir}`.")
    st.stop()


def _pick_structure_from_prediction_summary(summary_path: Path) -> Optional[Path]:
    try:
        data = json.loads(summary_path.read_text())
        predictors = data.get("predictors", {})
        # Prefer any predictor that has a structure path
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

if selected_model or selected_backbone:
    st.markdown("### Current selection")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        if selected_model:
            st.code(f"Model: {selected_model}")
        if selected_backbone:
            st.code(f"Backbone: {selected_backbone}")
    with col_b:
        if st.button("Clear", use_container_width=True):
            set_selected_model(None)
            set_selected_backbone(None)
            st.rerun()

st.markdown("---")

for job in jobs:
    job_id = job["job_id"]
    job_path = Path(job["path"])

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
        with col1:
            st.subheader(job_id)
            dt = datetime.fromtimestamp(job["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
            st.caption(dt)
            st.caption(str(job_path))

        with col2:
            st.markdown("**Artifacts**")
            st.write(
                f"Predict: {'✅' if job['has_prediction'] else '—'}  "
                f"Design: {'✅' if job['has_design'] else '—'}  "
                f"Compare: {'✅' if job['has_compare'] else '—'}"
            )

        inferred_model = None
        if job["has_prediction"]:
            inferred_model = _pick_structure_from_prediction_summary(job["prediction_summary"])

        with col3:
            st.markdown("**Quick actions**")
            if inferred_model is not None and st.button(
                "Use as model → Evaluate", key=f"eval_{job_id}", use_container_width=True
            ):
                set_selected_model(inferred_model)
                st.switch_page("pages/2_evaluate.py")

            if inferred_model is not None and st.button(
                "Use as backbone → MPNN", key=f"mpnn_{job_id}", use_container_width=True
            ):
                set_selected_backbone(inferred_model)
                st.switch_page("pages/6_mpnn.py")

        with col4:
            st.markdown("**Preview**")
            if inferred_model is not None:
                st.code(str(inferred_model))
            else:
                st.caption("No structure detected in prediction summary.")
