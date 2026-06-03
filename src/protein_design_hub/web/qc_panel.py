"""Research-backed QC panel for the Evaluate page.

Renders the unified protein-qc assessment (composite score, metric gates,
sequence liabilities, biophysical properties) and the ipSAE interface verdict
when a PAE matrix is available. Self-contained: pulls the sequence from session
state so call sites only pass the metrics dict (+ optional job dir).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from protein_design_hub.analysis import protein_qc
from protein_design_hub.evaluation import ipsae
from protein_design_hub.web.ui import metric_card, status_badge, info_box, section_header


# Map this app's metric keys → protein-qc metric keys.
_KEY_MAP = {
    "plddt": ["plddt", "mean_plddt", "global_plddt"],
    "iptm": ["iptm", "iptm_score"],
    "ptm": ["ptm", "ptm_score"],
    "pae_interaction": ["pae_interaction", "interface_pae", "pae_int"],
    "shape_complementarity": ["shape_complementarity", "sc"],
    "sc_rmsd": ["sc_rmsd", "scrmsd"],
    "esm2_pll_normalized": ["esm2_pll_normalized", "esm2_pll"],
}


def _session_sequence() -> str:
    for k in ("_extracted_sequence", "_ptm_cached_seq", "_ptm_seq_input",
              "_wl_cached_seq", "sequence"):
        v = st.session_state.get(k)
        if isinstance(v, str) and len(v) >= 10 and v.isalpha():
            return v.upper()
    return ""


def _collect_metrics(global_metrics: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for qc_key, candidates in _KEY_MAP.items():
        for c in candidates:
            if c in global_metrics and isinstance(global_metrics[c], (int, float)):
                out[qc_key] = float(global_metrics[c])
                break
    return out


def render_research_qc(global_metrics: Optional[Dict] = None,
                       job_dir: Optional[Path] = None) -> None:
    """Render the QC + ipSAE panel inside an expander."""
    global_metrics = global_metrics or {}
    sequence = _session_sequence()
    metrics = _collect_metrics(global_metrics)
    if not metrics and not sequence:
        return  # nothing to assess

    with st.expander("🔬 Research-backed Design QC  ·  composite · liabilities · ipSAE", expanded=False):
        st.caption(
            "Thresholds from binder-design competitions (protein-qc skill). "
            "Individual metrics are weak pre-screens — the **composite** is the ranking signal."
        )
        assessment = protein_qc.assess_design(metrics, sequence or None, level="standard")

        # Verdict + composite
        c1, c2 = st.columns([0.62, 0.38])
        with c1:
            variant = "success" if assessment.passed else ("warning" if assessment.composite else "info")
            info_box(assessment.verdict, variant=variant, title="Quality verdict")
        with c2:
            if assessment.composite is not None:
                metric_card(f"{assessment.composite:.2f}", "Composite QC (0–1)",
                            "success" if assessment.passed else "warning", "🎯")

        # Metric gates
        if assessment.checks:
            st.markdown("**Metric gates**")
            rows = []
            for ch in assessment.checks:
                if ch.get("pass") is None:
                    continue
                arrow = "≥" if ch["dir"] == "high" else "≤"
                rows.append({
                    "Metric": ch["metric"], "Level": ch["level"],
                    "Value": round(ch["value"], 3),
                    "Threshold": f'{arrow} {ch["threshold"]}',
                    "Pass": "✅" if ch["pass"] else "❌",
                })
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)

        # Sequence liabilities + biophysical
        if sequence:
            lc, bc = st.columns(2)
            with lc:
                st.markdown("**Sequence liabilities**")
                liab = assessment.liabilities
                badge = status_badge(f'risk {liab.get("risk", 0)}',
                                     "ok" if liab.get("risk", 0) < 20 else
                                     "warning" if liab.get("risk", 0) < 50 else "error")
                st.markdown(badge, unsafe_allow_html=True)
                if liab.get("flags"):
                    for f in liab["flags"]:
                        st.markdown(f"- {f}")
                else:
                    st.caption("No major liabilities detected.")
            with bc:
                st.markdown("**Biophysical**")
                bp = assessment.biophysical
                b1, b2, b3 = st.columns(3)
                with b1:
                    metric_card(bp.get("gravy", "—"), "GRAVY", "default")
                with b2:
                    metric_card(bp.get("pi", "—"), "pI", "default")
                with b3:
                    inst = bp.get("instability")
                    metric_card(inst if inst is not None else "—", "Instability",
                                "success" if (inst or 99) < 40 else "warning")

        # ipSAE (needs PAE)
        st.markdown("**Interface confidence (ipSAE)**")
        pae = ipsae.load_pae_from_dir(job_dir) if job_dir else None
        if pae is None:
            st.caption("⚪ ipSAE needs a ≥2-chain prediction with a PAE matrix "
                       "(Chai-1 / Boltz). None found for this result.")
        else:
            chains = st.session_state.get("_chain_lengths")
            if not chains:
                st.caption(f"PAE found ({pae.shape[0]}×{pae.shape[0]}) but chain "
                           "boundaries unknown — re-run the complex predictor to record them.")
            else:
                res = ipsae.compute_ipsae(pae, chains)
                v = "success" if res.get("passed") else "warning"
                info_box(ipsae.verdict(res), variant=v)
