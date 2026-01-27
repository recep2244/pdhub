"""Shared Streamlit UI helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

SESSION_SELECTED_MODEL = "pdhub_selected_model_path"
SESSION_SELECTED_BACKBONE = "pdhub_selected_backbone_path"


def inject_base_css() -> None:
    st.markdown(
        """
        <style>
          .pdhub-muted { color: #6c757d; }
          .pdhub-badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; }
          .pdhub-badge-ok { background: #e6ffed; color: #0f5132; border: 1px solid #b7ebc6; }
          .pdhub-badge-warn { background: #fff4e5; color: #7a4f01; border: 1px solid #ffd8a8; }
          .pdhub-badge-err { background: #ffe3e3; color: #842029; border: 1px solid #ffb3b3; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_nav(current: str | None = None) -> None:
    """Render a simple sidebar navigation."""
    st.sidebar.markdown("## Navigation")

    def go(label: str, target: str) -> None:
        disabled = current == label
        if st.sidebar.button(label, use_container_width=True, disabled=disabled):
            try:
                st.switch_page(target)
            except Exception:
                st.sidebar.info("Navigation not available in this Streamlit version.")

    go("Home", "app.py")
    go("Design", "pages/0_design.py")
    go("Predict", "pages/1_predict.py")
    go("Evaluate", "pages/2_evaluate.py")
    go("Compare", "pages/3_compare.py")
    go("Settings", "pages/4_settings.py")
    go("Batch", "pages/5_batch.py")
    go("MPNN Design", "pages/6_mpnn.py")
    go("Jobs", "pages/7_jobs.py")


def sidebar_system_status() -> None:
    """Render quick system status (predictors + GPU)."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## System Status")

    try:
        from protein_design_hub.core.config import get_settings
        from protein_design_hub.predictors.registry import PredictorRegistry

        settings = get_settings()
        predictor_names: List[str] = PredictorRegistry.list_available()
        predictor_names = [
            p
            for p in predictor_names
            if p in {"colabfold", "chai1", "boltz2", "esmfold", "esmfold_api"}
        ]
        predictor_names += [
            p for p in PredictorRegistry.list_available() if p not in predictor_names
        ]

        for name in predictor_names:
            try:
                pred = PredictorRegistry.get(name, settings)
                status = pred.get_status()
                installed = bool(status.get("installed"))
                version = status.get("version") or "-"
                badge_cls = "pdhub-badge-ok" if installed else "pdhub-badge-err"
                st.sidebar.markdown(
                    f"**{name}** "
                    f"<span class='pdhub-badge {badge_cls}'>{'installed' if installed else 'missing'}</span>"
                    f"<div class='pdhub-muted'>v{version}</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                st.sidebar.markdown(
                    f"**{name}** <span class='pdhub-badge pdhub-badge-warn'>error</span>",
                    unsafe_allow_html=True,
                )

    except Exception:
        st.sidebar.caption("Predictor status unavailable.")

    st.sidebar.markdown("### GPU")
    try:
        import torch

        if torch.cuda.is_available():
            st.sidebar.success(torch.cuda.get_device_name(0))
        else:
            st.sidebar.warning("Not available")
    except Exception:
        st.sidebar.caption("PyTorch not installed.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Design Tools")
    try:
        from protein_design_hub.core.config import get_settings
        from protein_design_hub.design.registry import DesignerRegistry
        from protein_design_hub.design.generators.registry import GeneratorRegistry

        settings = get_settings()
        for name in DesignerRegistry.list_available():
            try:
                d = DesignerRegistry.get(name, settings)
                ok = d.installer.is_installed()
                st.sidebar.markdown(
                    f"**{name}** "
                    f"<span class='pdhub-badge {'pdhub-badge-ok' if ok else 'pdhub-badge-err'}'>"
                    f"{'installed' if ok else 'missing'}</span>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
        for name in GeneratorRegistry.list_available():
            try:
                g = GeneratorRegistry.get(name, settings)
                ok = g.installer.is_installed()
                st.sidebar.markdown(
                    f"**{name}** "
                    f"<span class='pdhub-badge {'pdhub-badge-ok' if ok else 'pdhub-badge-err'}'>"
                    f"{'installed' if ok else 'missing'}</span>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
    except Exception:
        st.sidebar.caption("Design tool status unavailable.")

    st.sidebar.markdown("## Evaluation")
    try:
        from protein_design_hub.evaluation.ost_runner import get_ost_runner

        runner = get_ost_runner()
        if runner.is_available():
            st.sidebar.markdown(
                "<span class='pdhub-badge pdhub-badge-ok'>OpenStructure OK</span>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                "<span class='pdhub-badge pdhub-badge-warn'>OpenStructure missing</span>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass


def list_output_structures(base_dir: Path, limit: int = 200) -> List[Path]:
    """List recent structure files under outputs."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    exts = {".pdb", ".cif", ".mmcif"}
    paths = [p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[:limit]


def list_jobs(base_dir: Path, limit: int = 50) -> List[Dict[str, Any]]:
    """List job directories under base_dir, newest first."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    jobs: List[Dict[str, Any]] = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            mtime = p.stat().st_mtime
        except Exception:
            continue

        job = {
            "job_id": p.name,
            "path": p,
            "mtime": mtime,
            "prediction_summary": p / "prediction_summary.json",
            "design_summary": p / "design_summary.json",
            "comparison_summary": p / "evaluation" / "comparison_summary.json",
        }
        job["has_prediction"] = job["prediction_summary"].exists()
        job["has_design"] = job["design_summary"].exists()
        job["has_compare"] = job["comparison_summary"].exists()
        jobs.append(job)

    jobs.sort(key=lambda x: x["mtime"], reverse=True)
    return jobs[:limit]


def set_selected_model(path: Optional[Path]) -> None:
    if path is None:
        st.session_state.pop(SESSION_SELECTED_MODEL, None)
        return
    st.session_state[SESSION_SELECTED_MODEL] = str(Path(path))


def set_selected_backbone(path: Optional[Path]) -> None:
    if path is None:
        st.session_state.pop(SESSION_SELECTED_BACKBONE, None)
        return
    st.session_state[SESSION_SELECTED_BACKBONE] = str(Path(path))


def get_selected_model() -> Optional[Path]:
    v = st.session_state.get(SESSION_SELECTED_MODEL)
    return Path(v) if v else None


def get_selected_backbone() -> Optional[Path]:
    v = st.session_state.get(SESSION_SELECTED_BACKBONE)
    return Path(v) if v else None
