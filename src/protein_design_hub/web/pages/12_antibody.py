"""Antibody Engineering Workbench — CDR annotation, immunogenicity profiling,
germline analysis & developability metrics for therapeutic mAb development.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[3]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st

st.set_page_config(
    page_title="Antibody Engineering - Protein Design Hub",
    page_icon="🧫",
    layout="wide",
)

import html as _html
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    section_header,
    metric_card,
    metric_card_with_context,
    card_start,
    card_end,
    info_box,
    workflow_breadcrumb,
    cross_page_actions,
    render_badge,
)
from protein_design_hub.web.agent_helpers import (
    render_agent_advice_panel,
    render_contextual_insight,
    render_ml_stats_panel,
    agent_sidebar_status,
    render_all_experts_panel,
)
from protein_design_hub.web.shared_context import set_page_results, render_workflow_status_bar

# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

inject_base_css()
sidebar_nav(current="Antibody")
sidebar_system_status()
agent_sidebar_status()

with st.sidebar:
    with st.expander("About this page", expanded=False):
        st.markdown(
            """
**Antibody Engineering Workbench** provides:

- **CDR Annotation** — Chothia, IMGT, and Kabat numbering schemes with
  color-coded sequence maps and canonical class assignment.
- **MHC-II T-cell Epitope Scanning** — Pan-DR PSSM-based immunogenicity
  scoring with per-residue heatmaps and high-risk region identification.
- **Germline V-gene Assignment** — Closest human IGHV/IGKV/IGLV gene,
  identity %, and framework mutation catalogue.
- **Developability Metrics** — pI, instability index, GRAVY, PTM liabilities,
  and a therapeutic criteria checklist.
"""
        )

# ---------------------------------------------------------------------------
# Render header + workflow bar
# ---------------------------------------------------------------------------

page_header(
    "🧫 Antibody Engineering Workbench",
    "CDR annotation, immunogenicity profiling, germline analysis & developability",
    "12",
)
render_workflow_status_bar()

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

if "ab_sequence" not in st.session_state:
    st.session_state["ab_sequence"] = ""
if "ab_analysis_result" not in st.session_state:
    st.session_state["ab_analysis_result"] = None
if "ab_numbering_scheme" not in st.session_state:
    st.session_state["ab_numbering_scheme"] = "chothia"

# ---------------------------------------------------------------------------
# Helper: cached analysis runners
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _run_antibody_numbering(sequence: str):
    """Cache AntibodyAnnotation for a given sequence."""
    from protein_design_hub.analysis.antibody_numbering import annotate_antibody
    return annotate_antibody(sequence)


@st.cache_data(show_spinner=False)
def _run_immunogenicity(sequence: str):
    """Cache ImmunogenicityResult for a given sequence."""
    from protein_design_hub.analysis.immunogenicity import ImmunogenicityAnalyzer
    return ImmunogenicityAnalyzer().analyze(sequence)


@st.cache_data(show_spinner=False)
def _run_ptm_scan(sequence: str):
    """Cache PTMScanResult for a given sequence."""
    from protein_design_hub.analysis.ptm_scanner import PTMLiabilityScanner
    return PTMLiabilityScanner().scan(sequence)


@st.cache_data(show_spinner=False)
def _run_sequence_metrics(sequence: str):
    """Cache SequenceMetrics for a given sequence."""
    from protein_design_hub.analysis.sequence_metrics import compute_sequence_metrics
    return compute_sequence_metrics(sequence)


# ---------------------------------------------------------------------------
# Helper: auto-detect chain badge
# ---------------------------------------------------------------------------

_CHAIN_COLORS = {
    "VH": "#4f46e5",
    "VL_kappa": "#0891b2",
    "VL_lambda": "#0d9488",
    "VHH": "#7c3aed",
    "Unknown": "#6b7280",
}

_CHAIN_LABELS = {
    "VH": "VH (Heavy)",
    "VL_kappa": "VL-κ (Light)",
    "VL_lambda": "VL-λ (Light)",
    "VHH": "VHH / Nanobody",
    "Unknown": "Unknown",
}


def _chain_badge_html(chain_type: str, confidence: float) -> str:
    color = _CHAIN_COLORS.get(chain_type, "#6b7280")
    label = _CHAIN_LABELS.get(chain_type, chain_type)
    pct = f"{confidence:.0%}"
    return (
        f'<span style="background:{color};color:#fff;padding:3px 10px;'
        f'border-radius:12px;font-size:0.82rem;font-weight:600;">'
        f"{label} &nbsp; {pct}</span>"
    )


# ---------------------------------------------------------------------------
# Helper: color-coded CDR sequence display
# ---------------------------------------------------------------------------

_CDR_COLORS = {
    "CDR-H1": "#FF6B6B",
    "CDR-L1": "#FF6B6B",
    "CDR1":   "#FF6B6B",
    "CDR-H2": "#FFD93D",
    "CDR-L2": "#FFD93D",
    "CDR2":   "#FFD93D",
    "CDR-H3": "#6BCB77",
    "CDR-L3": "#6BCB77",
    "CDR3":   "#6BCB77",
}


def _build_cdr_sequence_html(sequence: str, cdrs, paratope_candidates: List[int]) -> str:
    """Build a color-coded HTML representation of the sequence."""
    # Build per-position region map (0-indexed)
    region_map = ["FR"] * len(sequence)
    for cdr in cdrs:
        for i in range(cdr.start - 1, min(cdr.end, len(sequence))):
            region_map[i] = cdr.name

    paratope_set = set(p - 1 for p in paratope_candidates)  # 0-indexed

    html_parts = []
    # Tick line every 10 residues
    tick_html = '<div style="font-family:monospace;font-size:0.75em;color:#9ca3af;letter-spacing:0;margin-bottom:2px;">'
    for i in range(len(sequence)):
        if i % 10 == 0:
            tick_html += f'<span style="display:inline-block;width:10ch;text-align:left">{i + 1}</span>'
    tick_html += "</div>"

    seq_html = '<div style="line-height:1.8;word-break:break-all;">'
    line_buffer = ""
    for i, aa in enumerate(sequence):
        if i > 0 and i % 60 == 0:
            seq_html += line_buffer + "<br/>"
            line_buffer = ""
        region = region_map[i]
        color = _CDR_COLORS.get(region, "transparent")
        bg = color if region != "FR" else "rgba(255,255,255,0.07)"
        text_color = "#1a1a1a" if region != "FR" else "inherit"
        border = "2px solid rgba(255,255,255,0.6)" if i in paratope_set else "none"
        weight = "bold" if i in paratope_set else "normal"
        span = (
            f'<span style="background-color:{bg};color:{text_color};'
            f"padding:2px 3px;font-family:monospace;font-size:0.85em;"
            f"border-radius:2px;border:{border};font-weight:{weight};"
            f'margin:1px;" title="pos {i+1} | {region}">{_html.escape(aa)}</span>'
        )
        line_buffer += span
    if line_buffer:
        seq_html += line_buffer
    seq_html += "</div>"

    # CDR label legend
    legend_html = '<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;font-size:0.8em;">'
    seen = []
    for cdr in cdrs:
        name = cdr.name
        if name in seen:
            continue
        seen.append(name)
        color = _CDR_COLORS.get(name, "#9ca3af")
        legend_html += (
            f'<span style="background:{color};color:#1a1a1a;padding:2px 8px;'
            f'border-radius:10px;font-weight:600;">'
            f"{_html.escape(name)} ({cdr.length} aa)</span>"
        )
    legend_html += (
        '<span style="background:rgba(255,255,255,0.1);color:inherit;padding:2px 8px;'
        'border-radius:10px;border:1px solid rgba(255,255,255,0.15);">Framework</span>'
    )
    legend_html += "</div>"

    return tick_html + seq_html + legend_html


# ---------------------------------------------------------------------------
# ── INPUT SECTION ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

section_header("Sequence Input", icon="🔬")

col_seq, col_load = st.columns([4, 1])

with col_seq:
    seq_input = st.text_area(
        "Antibody Variable Domain Sequence",
        value=st.session_state["ab_sequence"],
        height=120,
        placeholder=(
            "Paste VH, VL, or VHH sequence here...\n"
            "Example: EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS..."
        ),
        key="_ab_seq_input",
    )

with col_load:
    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("Load from\nPredictor", use_container_width=True, key="ab_load_predict"):
        pred_seq = st.session_state.get("predict_sequence", "")
        if pred_seq:
            st.session_state["ab_sequence"] = pred_seq
            st.rerun()
        else:
            st.warning("No sequence found in the Predictor page.")

# Sync input to session state
if seq_input != st.session_state["ab_sequence"]:
    st.session_state["ab_sequence"] = seq_input
    st.session_state["ab_analysis_result"] = None  # invalidate cache

sequence = st.session_state["ab_sequence"].strip().upper()

# Character count + auto-detect chain badge
char_col, badge_col = st.columns([2, 3])
with char_col:
    if sequence:
        st.caption(f"{len(sequence)} residues")
    else:
        st.caption("0 residues")

with badge_col:
    if len(sequence) >= 50:
        try:
            _preview_ann = _run_antibody_numbering(sequence)
            st.markdown(
                _chain_badge_html(_preview_ann.chain_type, _preview_ann.chain_confidence),
                unsafe_allow_html=True,
            )
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Analyse button
# ---------------------------------------------------------------------------

analyze_clicked = st.button(
    "🔬 Analyze",
    type="primary",
    use_container_width=True,
    key="ab_analyze_btn",
    disabled=len(sequence) < 10,
)

if analyze_clicked and len(sequence) >= 10:
    with st.spinner("Analyzing antibody sequence..."):
        ann_result = None
        imm_result = None
        ptm_result = None
        metrics_result = None
        ann_error = None
        imm_error = None
        ptm_error = None
        metrics_error = None

        try:
            ann_result = _run_antibody_numbering(sequence)
        except Exception as exc:
            ann_error = str(exc)

        try:
            imm_result = _run_immunogenicity(sequence)
        except Exception as exc:
            imm_error = str(exc)

        try:
            ptm_result = _run_ptm_scan(sequence)
        except Exception as exc:
            ptm_error = str(exc)

        try:
            metrics_result = _run_sequence_metrics(sequence)
        except Exception as exc:
            metrics_error = str(exc)

        st.session_state["ab_analysis_result"] = {
            "sequence": sequence,
            "ann": ann_result,
            "ann_error": ann_error,
            "imm": imm_result,
            "imm_error": imm_error,
            "ptm": ptm_result,
            "ptm_error": ptm_error,
            "metrics": metrics_result,
            "metrics_error": metrics_error,
        }

# ---------------------------------------------------------------------------
# ── RESULTS ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

stored = st.session_state.get("ab_analysis_result")

if stored is None or stored.get("sequence") != sequence:
    if not sequence:
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#6b7280;">'
            "<h3>Enter a sequence above to begin analysis</h3>"
            "<p>Supports VH, VL-κ, VL-λ, and VHH / nanobody sequences.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()
    else:
        st.info("Click **Analyze** to run the full antibody workbench.")
        st.stop()

# Unpack stored results (guard against bare-mode execution where st.stop() is a no-op)
if not stored:
    import sys as _sys
    _sys.exit(0)

ann: Optional[object] = stored.get("ann")
imm: Optional[object] = stored.get("imm")
ptm: Optional[object] = stored.get("ptm")
metrics: Optional[object] = stored.get("metrics")
ann_error: Optional[str] = stored.get("ann_error")
imm_error: Optional[str] = stored.get("imm_error")
ptm_error: Optional[str] = stored.get("ptm_error")
metrics_error: Optional[str] = stored.get("metrics_error")

# ---------------------------------------------------------------------------
# Check if this is an antibody sequence at all
# ---------------------------------------------------------------------------

if ann and ann.chain_type == "Unknown" and ann.chain_confidence < 0.4:
    st.warning(
        "This sequence does not appear to be an antibody variable domain. "
        "CDR annotation may be unreliable. Immunogenicity and developability "
        "results are still shown below."
    )

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tab_cdr, tab_imm, tab_dev, tab_fc, tab_wl = st.tabs(
    ["CDR Annotation", "Immunogenicity", "Developability", "Fc Engineering Guide", "🧪 Wet-Lab Plan"]
)

# ============================================================================
# TAB 1 — CDR ANNOTATION
# ============================================================================

with tab_cdr:
    if ann_error:
        st.error(f"CDR annotation failed: {ann_error}")
    elif ann is None:
        st.info("CDR annotation not available.")
    else:
        # ── Chain detection banner ───────────────────────────────────────────
        germline_text = (
            f"Germline: **{ann.germline.gene}** (family {ann.germline.family}, "
            f"{ann.germline.identity:.1f}% identity)"
            if ann.germline
            else "Germline: not assigned"
        )
        info_box(
            f"Chain: **{_CHAIN_LABELS.get(ann.chain_type, ann.chain_type)}** "
            f"| Confidence: **{ann.chain_confidence:.1%}** "
            f"| {germline_text} "
            f"| Humanness: **{ann.humanness_score:.1f}%**",
            kind="info",
        )

        # ── Numbering scheme selector ────────────────────────────────────────
        scheme = st.radio(
            "Numbering scheme",
            ["Chothia", "IMGT", "Kabat"],
            horizontal=True,
            key="ab_numbering_scheme_radio",
            index=["chothia", "imgt", "kabat"].index(
                st.session_state.get("ab_numbering_scheme", "chothia")
            ),
        )
        st.session_state["ab_numbering_scheme"] = scheme.lower()
        active_cdrs = (
            ann.cdrs
            if scheme == "Chothia"
            else (ann.cdrs_imgt if scheme == "IMGT" else ann.cdrs_kabat)
        )

        # ── Color-coded CDR map ──────────────────────────────────────────────
        section_header("CDR Map", icon="🗺")
        cdr_html = _build_cdr_sequence_html(
            ann.sequence, active_cdrs, ann.paratope_candidates
        )
        st.markdown(
            f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;'
            f'padding:16px;overflow-x:auto;">{cdr_html}</div>',
            unsafe_allow_html=True,
        )
        if ann.paratope_candidates:
            st.caption(
                f"Bold + bordered residues are paratope candidates "
                f"({len(ann.paratope_candidates)} positions): "
                + ", ".join(str(p) for p in ann.paratope_candidates[:15])
            )

        # ── CDR table ───────────────────────────────────────────────────────
        section_header(f"CDR Regions ({scheme})", icon="📋")
        if active_cdrs:
            cdr_df = pd.DataFrame(
                [
                    {
                        "CDR Name": c.name,
                        "Start": c.start,
                        "End": c.end,
                        "Length": c.length,
                        "Sequence": c.sequence,
                        "Canonical Class": c.canonical_class or "—",
                    }
                    for c in active_cdrs
                ]
            )
            st.dataframe(cdr_df, use_container_width=True, hide_index=True)
        else:
            st.info("No CDRs detected under this numbering scheme.")

        # ── Germline assignment ──────────────────────────────────────────────
        section_header("Germline Assignment", icon="🧬")
        if ann.germline:
            g = ann.germline
            gcol1, gcol2, gcol3, gcol4 = st.columns(4)
            with gcol1:
                metric_card(g.gene, "Germline Gene", variant="gradient", icon="🧬")
            with gcol2:
                metric_card(g.family, "Gene Family", variant="info", icon="🏷")
            with gcol3:
                metric_card(f"{g.identity:.1f}%", "Identity", variant="success", icon="✅")
            with gcol4:
                metric_card(str(g.n_mutations), "FR Mutations", variant="warning", icon="🔄")
        else:
            st.warning("No germline V-gene could be assigned for this sequence.")

        # Humanness progress bar
        st.markdown("**Humanness Score**")
        hm_val = int(min(100, max(0, ann.humanness_score)))
        hm_color = "#22c55e" if hm_val >= 85 else ("#f59e0b" if hm_val >= 70 else "#ef4444")
        st.progress(hm_val / 100)
        st.markdown(
            f'<span style="color:{hm_color};font-weight:700;">{hm_val}%</span> humanness',
            unsafe_allow_html=True,
        )

        # Framework mutation badges
        if ann.framework_mutations:
            st.markdown("**Framework Mutations vs. Germline:**")
            badge_html = " ".join(
                f'<span style="background:#e0e7ff;color:#3730a3;padding:2px 8px;'
                f'border-radius:10px;font-size:0.82em;font-family:monospace;">'
                f"{_html.escape(m)}</span>"
                for m in ann.framework_mutations[:20]
            )
            if len(ann.framework_mutations) > 20:
                badge_html += f' <span style="color:#6b7280;font-size:0.8em">+{len(ann.framework_mutations)-20} more</span>'
            st.markdown(badge_html, unsafe_allow_html=True)
        else:
            st.success("No framework mutations detected relative to closest germline.")

        # ── H3 kink indicator ───────────────────────────────────────────────
        section_header("CDR-H3 Conformation", icon="🔩")
        if ann.chain_type in ("VH", "VHH"):
            if ann.h3_kink_motif:
                st.success(
                    "**H3 Kinked** — CDR-H3 adopts a kinked base conformation "
                    "(W/Y at FR3 + WGXXD/N kink motif detected). Canonical in "
                    "~80% of human VH domains; confirm by structure."
                )
            else:
                st.info(
                    "**H3 Extended** — CDR-H3 does not show the canonical KINK motif. "
                    "This may indicate an unusual loop conformation; confirm by structure."
                )
        else:
            st.caption("H3 kink detection applies to VH / VHH chains only.")


# ============================================================================
# TAB 2 — IMMUNOGENICITY
# ============================================================================

with tab_imm:
    if imm_error:
        st.error(f"Immunogenicity analysis failed: {imm_error}")
    elif imm is None:
        st.info("Immunogenicity analysis not available.")
    else:
        # ── Risk header metrics ──────────────────────────────────────────────
        risk_colors = {
            "Low": "#22c55e",
            "Moderate": "#f59e0b",
            "High": "#ef4444",
            "Very High": "#7f1d1d",
        }
        risk_variants = {
            "Low": "success",
            "Moderate": "warning",
            "High": "error",
            "Very High": "error",
        }
        n_high_tcell = sum(1 for e in imm.t_cell_epitopes if e.risk == "High")
        n_med_tcell = sum(1 for e in imm.t_cell_epitopes if e.risk == "Medium")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            metric_card(
                f"{imm.immunogenicity_score:.0f}/100",
                "Immunogenicity Score",
                variant=risk_variants.get(imm.risk_label, "default"),
                icon="🎯",
            )
        with mc2:
            metric_card(
                imm.risk_label,
                "Risk Level",
                variant=risk_variants.get(imm.risk_label, "default"),
                icon="⚠️",
            )
        with mc3:
            metric_card(
                f"{imm.t_cell_epitope_density * 100:.1f}%",
                "T-cell Epitope Density",
                variant="warning" if imm.t_cell_epitope_density > 0.15 else "success",
                icon="🦠",
            )
        with mc4:
            metric_card(
                f"{n_high_tcell}H / {n_med_tcell}M",
                "MHC-II Epitopes (High/Med)",
                variant="error" if n_high_tcell > 2 else "warning",
                icon="🔬",
            )

        st.caption(
            "⚠️ Heuristic scores only — MHC-II scoring uses a pan-DR PSSM "
            "(T-cell 50%, APR 30%, B-cell 20%). Not a validated clinical predictor; "
            "confirm high-risk epitopes with NetMHCIIpan or EpiVax."
        )

        # ── Per-residue T-cell heatmap ───────────────────────────────────────
        section_header("Per-Residue MHC-II T-cell Epitope Risk", icon="🌡")
        if imm.per_residue_tcell:
            fig_tcell = go.Figure(
                go.Heatmap(
                    z=[imm.per_residue_tcell],
                    colorscale="RdYlGn_r",
                    zmin=0,
                    zmax=100,
                    showscale=True,
                    colorbar=dict(title="MHC-II score"),
                )
            )
            fig_tcell.update_layout(
                title="Per-residue MHC-II T-cell Epitope Risk",
                xaxis_title="Residue Position",
                height=220,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            # Annotate CDR regions as vertical shapes
            if ann and not ann_error:
                shape_colors = {
                    "CDR-H1": "rgba(255,107,107,0.35)",
                    "CDR-H2": "rgba(255,217,61,0.35)",
                    "CDR-H3": "rgba(107,203,119,0.35)",
                    "CDR-L1": "rgba(255,107,107,0.35)",
                    "CDR-L2": "rgba(255,217,61,0.35)",
                    "CDR-L3": "rgba(107,203,119,0.35)",
                    "CDR1":   "rgba(255,107,107,0.35)",
                    "CDR2":   "rgba(255,217,61,0.35)",
                    "CDR3":   "rgba(107,203,119,0.35)",
                }
                shapes = []
                annotations_fig = []
                for cdr in ann.cdrs:
                    shapes.append(
                        dict(
                            type="rect",
                            x0=cdr.start - 1.5,
                            x1=cdr.end - 0.5,
                            y0=-0.5,
                            y1=0.5,
                            line=dict(
                                color=shape_colors.get(cdr.name, "rgba(99,102,241,0.4)"),
                                width=2,
                                dash="dash",
                            ),
                            fillcolor="rgba(0,0,0,0)",
                        )
                    )
                    annotations_fig.append(
                        dict(
                            x=(cdr.start + cdr.end) / 2 - 1,
                            y=0.48,
                            text=cdr.name,
                            showarrow=False,
                            font=dict(size=9, color="#374151"),
                            xanchor="center",
                            yanchor="top",
                        )
                    )
                fig_tcell.update_layout(shapes=shapes, annotations=annotations_fig)
            st.plotly_chart(fig_tcell, use_container_width=True)

        # ── Top T-cell epitopes table ────────────────────────────────────────
        section_header("Top T-cell Epitopes", icon="📊")
        if imm.t_cell_epitopes:
            epi_df = pd.DataFrame(
                [
                    {
                        "Position": e.position,
                        "9-mer Peptide": e.peptide,
                        "Score": f"{e.score:.1f}",
                        "Risk": e.risk,
                        "15-mer Context": e.context_15mer,
                    }
                    for e in imm.t_cell_epitopes
                ]
            )

            def _color_risk_rows(row):
                if row["Risk"] == "High":
                    return ["background-color: rgba(239,68,68,0.15)"] * len(row)
                if row["Risk"] == "Medium":
                    return ["background-color: rgba(245,158,11,0.10)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                epi_df.style.apply(_color_risk_rows, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No significant MHC-II T-cell epitopes detected.")

        # ── Per-residue B-cell epitope chart ─────────────────────────────────
        section_header("Per-Residue B-cell Epitope Profile", icon="📈")
        if imm.per_residue_bcell:
            x_vals = list(range(1, len(imm.per_residue_bcell) + 1))
            fig_bcell = go.Figure()
            fig_bcell.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=imm.per_residue_bcell,
                    mode="lines",
                    line=dict(color="#3b82f6", width=2),
                    name="B-cell score",
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.12)",
                )
            )
            fig_bcell.add_hline(
                y=55,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text="Threshold (55)",
                annotation_position="top right",
            )
            # Shade above-threshold regions
            above = [v for v in imm.per_residue_bcell if v >= 55]
            if above:
                regions_above = []
                in_region = False
                start_r = 0
                for i, v in enumerate(imm.per_residue_bcell):
                    if v >= 55 and not in_region:
                        in_region = True
                        start_r = i
                    elif v < 55 and in_region:
                        in_region = False
                        regions_above.append((start_r + 1, i))
                if in_region:
                    regions_above.append((start_r + 1, len(imm.per_residue_bcell)))
                for r_start, r_end in regions_above:
                    fig_bcell.add_vrect(
                        x0=r_start - 0.5,
                        x1=r_end + 0.5,
                        fillcolor="rgba(239,68,68,0.12)",
                        layer="below",
                        line_width=0,
                    )
            fig_bcell.update_layout(
                xaxis_title="Residue Position",
                yaxis_title="B-cell Epitope Score",
                height=220,
                margin=dict(l=20, r=20, t=30, b=30),
            )
            st.plotly_chart(fig_bcell, use_container_width=True)

        # ── APR regions ──────────────────────────────────────────────────────
        section_header("Aggregation-Prone Regions (APR)", icon="⚡")
        if imm.apr_regions:
            for apr in imm.apr_regions:
                color = "#ef4444" if apr.risk == "High" else "#f59e0b"
                st.markdown(
                    f'<div style="background:rgba({("239,68,68" if apr.risk=="High" else "245,158,11")},0.12);'
                    f"border-left:4px solid {color};border-radius:6px;"
                    f'padding:10px 14px;margin-bottom:8px;">'
                    f"<strong>APR [{apr.risk}]</strong> &nbsp; "
                    f"Residues {apr.start}–{apr.end} &nbsp; "
                    f'<code style="background:#1f2937;color:#f3f4f6;padding:2px 6px;border-radius:3px;">'
                    f"{_html.escape(apr.sequence)}</code> &nbsp; "
                    f"Score: {apr.score:.3f}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No aggregation-prone regions detected.")

        # ── Summary text ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            f"**{n_high_tcell} high-risk MHC-II epitope{'s' if n_high_tcell != 1 else ''} detected. "
            f"Immunogenicity risk: {imm.risk_label}.** "
            "For therapeutic use, regions with score >60 should be redesigned by introducing "
            "charged or Pro residues at P1/P4/P9 anchor positions."
        )


# ============================================================================
# TAB 3 — DEVELOPABILITY
# ============================================================================

with tab_dev:
    # ── Biophysical properties ────────────────────────────────────────────────
    section_header("Biophysical Properties", icon="⚗️")

    if metrics_error:
        st.error(f"Sequence metrics failed: {metrics_error}")
        metrics = None

    if metrics:
        mw_kda = metrics.molecular_weight / 1000.0
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            pi_ok = 5.0 <= metrics.isoelectric_point <= 8.0
            metric_card(
                f"{metrics.isoelectric_point:.1f}",
                "Isoelectric Point (pI)",
                variant="success" if pi_ok else "warning",
                icon="⚡",
            )
        with d2:
            ii_ok = metrics.instability_index < 40
            metric_card(
                f"{metrics.instability_index:.1f}" + ("" if ii_ok else " ⚠️"),
                "Instability Index",
                variant="success" if ii_ok else "warning",
                icon="🔥",
            )
        with d3:
            gravy_ok = metrics.gravy < 0
            metric_card(
                f"{metrics.gravy:.3f}" + ("" if gravy_ok else " ⚠️ Hydrophobic"),
                "GRAVY",
                variant="success" if gravy_ok else "warning",
                icon="💧",
            )
        with d4:
            metric_card(
                f"{mw_kda:.1f} kDa",
                "Molecular Weight",
                variant="info",
                icon="⚖️",
            )
    else:
        info_box("Sequence metrics could not be computed (BioPython required).", kind="warning")

    # ── Therapeutic antibody criteria checklist ───────────────────────────────
    section_header("Therapeutic Antibody Criteria", icon="✅")
    with st.expander("Therapeutic Antibody Criteria Checklist", expanded=True):
        def _criterion_row(passed: bool, label: str, detail: str = "") -> None:
            icon = "✅" if passed else "❌"
            color = "#22c55e" if passed else "#ef4444"
            detail_html = (
                f' <span style="color:#6b7280;font-size:0.85em">{_html.escape(detail)}</span>'
                if detail
                else ""
            )
            st.markdown(
                f'<div style="padding:4px 0;">{icon} '
                f'<span style="color:{color};font-weight:600;">{_html.escape(label)}</span>'
                f"{detail_html}</div>",
                unsafe_allow_html=True,
            )

        pi_criterion = metrics and 5.0 <= metrics.isoelectric_point <= 8.0
        ii_criterion = metrics and metrics.instability_index < 40
        gravy_criterion = metrics and metrics.gravy < 0
        humanness_criterion = ann and not ann_error and ann.humanness_score >= 85.0

        # Check for CDR glycosylation
        cdr_glyco_criterion = True
        if ptm and not ptm_error:
            for lia in ptm.liabilities:
                if lia.in_cdr and "glycosylat" in lia.liability_type.lower():
                    cdr_glyco_criterion = False
                    break

        _criterion_row(
            bool(pi_criterion),
            "pI 5–8 (preferred for low viscosity)",
            f"pI = {metrics.isoelectric_point:.1f}" if metrics else "N/A",
        )
        _criterion_row(
            bool(ii_criterion),
            "Instability Index <40 (stable)",
            f"II = {metrics.instability_index:.1f}" if metrics else "N/A",
        )
        _criterion_row(
            bool(gravy_criterion),
            "GRAVY < 0 (hydrophilic)",
            f"GRAVY = {metrics.gravy:.3f}" if metrics else "N/A",
        )
        _criterion_row(
            bool(humanness_criterion),
            "Humanness ≥85% (low immunogenicity)",
            f"{ann.humanness_score:.1f}%" if (ann and not ann_error) else "N/A",
        )
        _criterion_row(
            cdr_glyco_criterion,
            "No CDR glycosylation sites",
        )

    # ── PTM liabilities ───────────────────────────────────────────────────────
    section_header("PTM Liabilities", icon="🔬")

    if ptm_error:
        st.error(f"PTM scan failed: {ptm_error}")
    elif ptm is None:
        st.info("PTM scan not available.")
    else:
        ptm_col1, ptm_col2 = st.columns([1, 3])
        with ptm_col1:
            ptm_variant = (
                "error"
                if ptm.risk_label in ("High", "Critical")
                else ("warning" if ptm.risk_label == "Medium" else "success")
            )
            metric_card(
                f"{ptm.risk_score:.0f}/100",
                f"PTM Risk ({ptm.risk_label})",
                variant=ptm_variant,
                icon="⚗️",
            )

        if ptm.liabilities:
            ptm_df = pd.DataFrame(
                [
                    {
                        "Type": lia.liability_type,
                        "Position": lia.position,
                        "Motif": lia.motif,
                        "Risk": lia.risk,
                        "In CDR": "Yes" if lia.in_cdr else "No",
                        "Description": lia.description,
                    }
                    for lia in ptm.liabilities
                ]
            )

            def _color_ptm_rows(row):
                if row["In CDR"] == "Yes":
                    return ["background-color: rgba(245,158,11,0.18)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                ptm_df.style.apply(_color_ptm_rows, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            cdr_ptm_count = sum(1 for lia in ptm.liabilities if lia.in_cdr)
            if cdr_ptm_count > 0:
                st.warning(
                    f"**{cdr_ptm_count} PTM {'liability' if cdr_ptm_count == 1 else 'liabilities'} "
                    f"detected in CDR regions** — these directly affect binding and must be "
                    f"addressed before advancing to clinical development."
                )
            else:
                st.success("No PTM liabilities detected in CDR regions.")

            if ptm.manufacturability_flags:
                st.markdown("**Manufacturability Flags:**")
                for flag in ptm.manufacturability_flags:
                    st.markdown(f"- {flag}")
        else:
            st.success("No PTM liabilities detected.")


# ============================================================================
# TAB 4 — Fc ENGINEERING GUIDE
# ============================================================================

with tab_fc:
    section_header("Fc Engineering Guide", icon="🔧")

    seq_len = len(sequence)
    is_full_ab = seq_len > 200

    if not is_full_ab:
        info_box(
            "Variable domain only — Fc engineering not applicable. "
            f"(Sequence is {seq_len} residues; full IgG typically >600 residues.)",
            kind="info",
        )
    else:
        st.warning(
            f"This sequence is {seq_len} residues and may contain an Fc region. "
            "Fc engineering options are shown below."
        )
        # Try to detect IgG subclass
        seq_upper = sequence.upper()
        detected_subclass = "Unknown"
        if "SCPAPEL" in seq_upper:
            detected_subclass = "IgG4"
        elif "CPAPELLGG" in seq_upper:
            detected_subclass = "IgG1 or IgG2"
        elif "APELLGG" in seq_upper:
            detected_subclass = "IgG1 (likely)"

        if detected_subclass != "Unknown":
            st.success(f"Detected subclass: **{detected_subclass}** (motif-based)")
        else:
            st.info("IgG subclass could not be determined from sequence motifs alone.")

    st.markdown("---")

    # ── Known Engineering Mutations table ────────────────────────────────────
    section_header("Known Fc Engineering Mutations", icon="📋")
    fc_data = pd.DataFrame(
        [
            {
                "Mutation": "LALA (L234A/L235A)",
                "Effect": "Ablates FcγR binding, removes ADCC",
                "Subclass Compatibility": "IgG1, IgG2",
                "Clinical Use": "Silencing mAbs",
            },
            {
                "Mutation": "LALAPG (+P329G)",
                "Effect": "Complete FcγR silence + Protein A binding retained",
                "Subclass Compatibility": "IgG1",
                "Clinical Use": "Full silencing",
            },
            {
                "Mutation": "YTE (M252Y/S254T/T256E)",
                "Effect": "3–5× half-life extension via enhanced FcRn binding",
                "Subclass Compatibility": "IgG1",
                "Clinical Use": "Extended half-life",
            },
            {
                "Mutation": "LS (M428L/N434S)",
                "Effect": "2–5× half-life extension via FcRn pH-dependent binding",
                "Subclass Compatibility": "IgG1",
                "Clinical Use": "Extended half-life",
            },
            {
                "Mutation": "GASDALIE",
                "Effect": "Enhanced ADCC/ADCP via increased FcγRIIIa affinity",
                "Subclass Compatibility": "IgG1",
                "Clinical Use": "Tumor immunotherapy",
            },
            {
                "Mutation": "N297A / N297Q",
                "Effect": "Removes Fc glycosylation, reduces ADCC, reduces immunogenicity",
                "Subclass Compatibility": "IgG1, IgG4",
                "Clinical Use": "Reduces ADCC, reduces immunogenicity",
            },
            {
                "Mutation": "S228P",
                "Effect": "Prevents half-antibody formation by stabilising hinge disulfide",
                "Subclass Compatibility": "IgG4",
                "Clinical Use": "Stabilises IgG4",
            },
        ]
    )
    st.dataframe(fc_data, use_container_width=True, hide_index=True)

    # ── Effector function guide ───────────────────────────────────────────────
    with st.expander("Fc Effector Function Selection Guide", expanded=False):
        st.markdown(
            """
| Format | Effector Profile | Best Use Case |
|--------|-----------------|---------------|
| **IgG1 (native)** | ADCC + ADCP + CDC | Tumor killing, viral clearance, CD20/HER2 |
| **IgG4 (native)** | Minimal effector | Autoimmune, receptor agonism (no killing needed) |
| **LALA / LALAPG** | Silent Fc | Receptor occupancy without immune killing |
| **YTE / LS** | Extended half-life | Monthly or bi-monthly dosing schedules |
| **GASDALIE** | Boosted ADCC/ADCP | High-efficacy tumor immunotherapy |
| **N297A/Q** | Aglycosyl (no ADCC) | Reduce immunogenicity, simplify manufacturing |
| **S228P (IgG4)** | Stable monovalent | Bispecifics, stable IgG4 therapeutics |

**Key decision criteria:**
- Need to kill target cells? → Native IgG1 or GASDALIE
- Need receptor block without killing? → LALA / LALAPG on IgG1, or IgG4
- Chronic disease / patient compliance? → YTE or LS for half-life extension
- Manufacturing simplification? → N297A/Q (aglycosyl, no Protein A glycan dependency)
- IgG4 instability concern? → Always include S228P in IgG4 therapeutics
"""
        )

# ============================================================================
# TAB 5 — WET-LAB PLAN
# ============================================================================

with tab_wl:
    section_header("Wet-Lab Readiness & Expression Plan", icon="🧪")
    st.markdown(
        "Complete **expression system recommendation**, step-by-step **purification strategy**, "
        "**codon risk** assessment, and an evidence-based **Go/No-Go decision** for advancing "
        "this antibody to wet lab."
    )

    # Run or retrieve wet-lab report
    _ab_wl_cache_key = f"_ab_wl_report_{hash(sequence)}"
    _ab_wl_report = st.session_state.get(_ab_wl_cache_key)

    _run_wl_plan = st.button(
        "Generate Wet-Lab Plan",
        type="primary",
        key="ab_wl_run_btn",
        help="Analyse expression suitability, purification strategy, and codon risks for this antibody sequence.",
    )

    if _run_wl_plan:
        try:
            from protein_design_hub.analysis.wet_lab_advisor import build_wet_lab_report as _build_wl
            _chain_type_for_wl = ann.chain_type if ann else None
            with st.spinner("Generating wet-lab readiness report…"):
                _ab_wl_report = _build_wl(
                    sequence,
                    is_antibody=True,
                    chain_type=_chain_type_for_wl,
                    codon_hosts=["E. coli", "HEK293/CHO"],
                )
            st.session_state[_ab_wl_cache_key] = _ab_wl_report
        except Exception as _wle:
            st.error(f"Wet-lab plan failed: {_wle}")
            import traceback
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            _ab_wl_report = None

    if _ab_wl_report is None:
        info_box("Click **Generate Wet-Lab Plan** to run the analysis.", kind="info")
    else:
        _verd_col = {"GO": "#22c55e", "CONDITIONAL": "#f59e0b", "NO-GO": "#ef4444"}
        _verd_icon = {"GO": "✅", "CONDITIONAL": "⚠️", "NO-GO": "❌"}
        _vc = _verd_col.get(_ab_wl_report.go_nogo, "#6b7280")
        _vi = _verd_icon.get(_ab_wl_report.go_nogo, "❓")

        # ── Verdict banner ────────────────────────────────────────────────────
        st.markdown(
            f'<div style="background:rgba(0,0,0,0.35);border-left:6px solid {_vc};'
            f'padding:18px 22px;border-radius:10px;margin:14px 0;">'
            f'<span style="font-size:1.5rem;font-weight:700;color:{_vc};">{_vi} {_ab_wl_report.go_nogo}</span>'
            f'<p style="color:#e2e8f0;margin:8px 0 0 0;font-size:1rem;">{_ab_wl_report.go_nogo_reason}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Synopsis ──────────────────────────────────────────────────────────
        st.info(_ab_wl_report.synopsis)

        # ── Quick biophysics ──────────────────────────────────────────────────
        _wlc1, _wlc2, _wlc3, _wlc4, _wlc5 = st.columns(5)
        _wlc1.metric("MW", f"{_ab_wl_report.mw_kda:.1f} kDa")
        _wlc2.metric("pI", f"{_ab_wl_report.pi:.1f}")
        _wlc3.metric("GRAVY", f"{_ab_wl_report.gravy:.3f}")
        _wlc4.metric("Instability II", f"{_ab_wl_report.instability_index:.1f}")
        _wlc5.metric("SS Pairs", f"{_ab_wl_report.disulfide_pairs_predicted}")

        # ── Criteria checklist ────────────────────────────────────────────────
        st.markdown("---")
        _wlcc1, _wlcc2 = st.columns(2)
        with _wlcc1:
            st.markdown("**✅ Criteria Met**")
            for _c in _ab_wl_report.criteria_met:
                st.markdown(f'<div style="color:#22c55e;padding:3px 0;">✅ {_c}</div>', unsafe_allow_html=True)
            if not _ab_wl_report.criteria_met:
                st.caption("None")
        with _wlcc2:
            st.markdown("**⚠️ Watch / ❌ Failed**")
            for _c in _ab_wl_report.criteria_watch:
                st.markdown(f'<div style="color:#f59e0b;padding:3px 0;">⚠️ {_c}</div>', unsafe_allow_html=True)
            for _c in _ab_wl_report.criteria_failed:
                st.markdown(f'<div style="color:#ef4444;padding:3px 0;">❌ {_c}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Expression system recommendations ──────────────────────────────────
        section_header("Expression System Recommendations", icon="🔬")
        _es_df_rows = []
        for _es in _ab_wl_report.expression_systems:
            _es_df_rows.append({
                "System": _es.system,
                "Score /100": f"{_es.score:.0f}",
                "Est. Yield (mg/L)": f"{_es.estimated_yield_mgl[0]:.0f}–{_es.estimated_yield_mgl[1]:.0f}",
                "Timeline": f"{_es.timeline_days[0]}–{_es.timeline_days[1]} days",
                "Cost/mg": f"${_es.cost_per_mg_usd[0]:.0f}–${_es.cost_per_mg_usd[1]:.0f}",
            })
        st.dataframe(pd.DataFrame(_es_df_rows), use_container_width=True, hide_index=True)

        _top = _ab_wl_report.expression_systems[0]
        with st.expander(f"📋 Detailed Protocol: {_top.system}", expanded=True):
            if _top.rationale:
                st.markdown("**Why recommended:**")
                for _r in _top.rationale:
                    st.markdown(f"- {_r}")
            if _top.caveats:
                st.markdown("**Caveats:**")
                for _cv in _top.caveats:
                    st.warning(_cv)
            st.markdown("**Step-by-step protocol:**")
            for _step_i, _pn in enumerate(_top.protocol_notes, 1):
                st.markdown(f"{_step_i}. {_pn}")

        st.markdown("---")

        # ── Purification strategy ──────────────────────────────────────────────
        section_header("Purification Strategy", icon="🧫")
        st.caption(
            f"Estimated total recovery: **{_ab_wl_report.purification.overall_recovery_pct:.0f}%** | "
            f"Target purity: **{_ab_wl_report.purification.final_purity_target}**"
        )
        for _ps in _ab_wl_report.purification.steps:
            with st.expander(f"Step {_ps.step_number}: {_ps.method}", expanded=(_ps.step_number <= 2)):
                _psc1, _psc2 = st.columns([3, 1])
                with _psc1:
                    st.markdown(f"**Column/Resin:** {_ps.resin_or_column}")
                    st.markdown(f"**Buffers:** {_ps.buffer_system}")
                with _psc2:
                    metric_card(f"{_ps.expected_yield_pct:.0f}%", "Step Recovery", variant="info", icon="📊")
                    metric_card(f"{_ps.expected_purity_pct:.0f}%", "Purity After Step", variant="success", icon="✨")
                if _ps.notes:
                    st.info(_ps.notes)

        with st.expander("QC Checklist (post-purification)", expanded=True):
            for _qcl in _ab_wl_report.purification.qc_checklist:
                st.markdown(f"- {_qcl}")

        st.info(_ab_wl_report.purification.concentration_notes)

        st.markdown("---")

        # ── Codon risk ─────────────────────────────────────────────────────────
        section_header("Codon Risk Assessment", icon="🧬")
        _risk_icons = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        for _cr in _ab_wl_report.codon_risks:
            _cr_icon = _risk_icons.get(_cr.risk_level, "⚪")
            with st.expander(
                f"{_cr_icon} {_cr.host} — {_cr.risk_level} risk",
                expanded=(_cr.risk_level != "Low"),
            ):
                for _flag in _cr.rare_codons:
                    st.markdown(f"- {_flag}")
                st.info(_cr.recommendation)

        # ── Signal peptide / secretion hint ────────────────────────────────────
        if _ab_wl_report.has_signal_peptide_hint:
            st.warning(
                "**Signal peptide detected (heuristic):** The N-terminal region has a hydrophobic "
                "stretch consistent with a signal peptide. If this is a secreted antibody, verify with "
                "SignalP-6 and remove the SP before structural prediction of the mature domain."
            )

# ============================================================================
# AGENT PANEL
# ============================================================================

if ann and not ann_error and imm and not imm_error and ptm and not ptm_error and metrics:
    st.markdown("---")
    section_header("Expert AI Panel", icon="🤖")

    n_high_tcell_agent = sum(1 for e in imm.t_cell_epitopes if e.risk == "High")
    cdr_ptm_count_agent = sum(1 for p in ptm.liabilities if p.in_cdr)

    _ab_agent_ctx = "\n".join(
        [
            f"Chain type: {ann.chain_type} (confidence: {ann.chain_confidence:.1%})",
            (
                f"Germline: {ann.germline.gene} ({ann.germline.identity:.1f}% identity)"
                if ann.germline
                else "Germline: N/A"
            ),
            f"Humanness score: {ann.humanness_score:.1f}%",
            f"CDRs (Chothia): {', '.join(f'{c.name}={c.sequence}' for c in ann.cdrs)}",
            f"Framework mutations: {', '.join(ann.framework_mutations[:10]) if ann.framework_mutations else 'None'}",
            f"Immunogenicity score: {imm.immunogenicity_score:.1f} ({imm.risk_label})",
            f"High-risk T-cell epitopes: {n_high_tcell_agent}",
            f"PTM risk: {ptm.risk_score:.1f} ({ptm.risk_label})",
            f"CDR PTM count: {cdr_ptm_count_agent}",
            f"pI: {metrics.isoelectric_point:.1f}, GRAVY: {metrics.gravy:.3f}, "
            f"Instability: {metrics.instability_index:.1f}",
        ]
    )

    render_agent_advice_panel(
        page_context=_ab_agent_ctx,
        default_question=(
            "Based on CDR annotation, germline assignment, and immunogenicity profile, "
            "what are the most important design changes to improve this antibody for therapeutic use?"
        ),
        expert="Immunologist",
        key_prefix="ab_agent",
    )

    render_all_experts_panel(
        "🧫 Immunologist Expert Team Review",
        agenda=(
            "Comprehensive therapeutic antibody review: CDR quality, germline humanness, "
            "immunogenicity risk, developability, and wet lab advancement strategy."
        ),
        context=_ab_agent_ctx,
        questions=(
            f"The germline identity is {ann.humanness_score:.0f}% and "
            f"{n_high_tcell_agent} high-risk MHC-II epitopes were detected — "
            "is this antibody ready for clinical candidate nomination, or what specific "
            "humanization/deimmunization changes are needed?",
            f"With {cdr_ptm_count_agent} PTM liabilities in CDR regions, "
            "which CDR-specific chemical liabilities (deamidation/isomerization/glycosylation) "
            "must be fixed before IND-enabling studies, and what are the exact mutations to make?",
            "From a wet lab prioritization perspective: what is the ordered experimental plan — "
            "gene synthesis, transient HEK293 expression, Protein A purification, SEC-HPLC, "
            "DSF Tm, SPR kinetics — and at what Tm/yield/purity thresholds does this candidate "
            "advance or fail?",
        ),
        key_prefix="ab_all",
    )

elif stored is not None:
    # Partial results — show whatever agent panels we can
    st.markdown("---")
    partial_ctx = f"Antibody sequence length: {len(sequence)} residues"
    if ann and not ann_error:
        partial_ctx += f"\nChain: {ann.chain_type} ({ann.chain_confidence:.1%} confidence)"
    if imm and not imm_error:
        partial_ctx += f"\nImmunogenicity: {imm.immunogenicity_score:.1f} ({imm.risk_label})"
    render_agent_advice_panel(
        page_context=partial_ctx,
        default_question="What are the key concerns for this antibody sequence?",
        expert="Immunologist",
        key_prefix="ab_agent_partial",
    )

# ---------------------------------------------------------------------------
# Cross-page actions
# ---------------------------------------------------------------------------

cross_page_actions(
    [
        {"label": "Scan Mutations", "page": "pages/10_mutation_scanner.py", "icon": "🧬"},
        {"label": "Predict Structure", "page": "pages/1_predict.py", "icon": "🔮"},
        {"label": "Evaluate Structure", "page": "pages/2_evaluate.py", "icon": "📊"},
        {"label": "Scan PTMs", "page": "pages/2_evaluate.py", "icon": "⚗️"},
    ]
)
