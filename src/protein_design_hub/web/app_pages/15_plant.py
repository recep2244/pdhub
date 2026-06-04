"""Plant / Wheat Biology Track — guided flow (Sequence → Localisation → NLR →
Codon & Export) over a shared Workbench. First-class track (blueprint decision #1)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st

from protein_design_hub.web.ui import (
    inject_base_css, page_header, section_header, info_box, metric_card,
    status_badge, scientific_insight, insight_bar, sidebar_nav, sidebar_system_status,
    wizard, wizard_nav, wizard_reset, Workbench,
)
from protein_design_hub.analysis import (
    transit_peptide as tp, nlr_domains as nlr, codon_optimization as co,
)

inject_base_css()
sidebar_nav(current="Plant / Wheat")
sidebar_system_status()
try:
    from protein_design_hub.web.agent_helpers import agent_sidebar_status
    agent_sidebar_status()
except Exception:
    pass

page_header(
    "Plant / Wheat Biology",
    "Guided analysis for N. benthamiana / wheat engineering: targeting, NLR "
    "architecture, and host codon optimisation — carried across steps in your Workbench.",
    "🌾",
)

wb = Workbench("plant")
STEPS = ["Sequence", "Localisation", "NLR Architecture", "Codon & Export"]

_DEMO = ("MASSMLSSAAVATRSNVAQANMVAPFTGLKSSAAFPVTRKQNLDITSIASNGGRVQC"
         "MKTAYIAKQRQISFVKSHFSRQLEERLGGIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK")

# ── Workbench status bar ─────────────────────────────────────────────────────
_seq = wb.get("seq")
_parts = []
if _seq:
    _parts.append(status_badge(f"{len(_seq)} aa", "primary"))
    if wb.get("tp_type"):
        _parts.append(status_badge(f"signal · {wb.get('tp_type')}", "info"))
    if wb.get("is_nlr"):
        _parts.append(status_badge(f"NLR · {wb.get('nlr_class')}", "ok"))
if _parts:
    st.markdown('<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px">'
                + " ".join(_parts) + "</div>", unsafe_allow_html=True)

cur = wizard(STEPS, key="plant_wiz")
st.markdown("")


# ============================================================================
# STEP 0 — Sequence
# ============================================================================
if cur == 0:
    section_header("Input Sequence", "Paste a plant protein (or load the demo)", "🧬")
    txt = st.text_area("Protein sequence (single-letter amino acids)",
                       value=wb.get("seq", ""), height=120, key="plant_seq_input")
    c1, _ = st.columns([0.3, 0.7])
    with c1:
        if st.button("Load demo (chloroplast + NLR)", width="stretch"):
            wb.set("seq", _DEMO)
            st.rerun()
    cleaned = "".join(ch for ch in (txt or "").upper() if ch.isalpha())
    if cleaned and cleaned != wb.get("seq"):
        wb.set("seq", cleaned)
        # invalidate downstream cached calls
        for k in ("tp_type", "is_nlr", "nlr_class"):
            wb.data.pop(k, None)
    if wb.get("seq"):
        st.success(f"Sequence ready — {len(wb.get('seq'))} residues.")
    else:
        info_box("Paste a sequence or load the demo to begin the guided analysis.",
                 variant="info", title="Awaiting input")
    wizard_nav(STEPS, key="plant_wiz", can_advance=bool(wb.get("seq")),
               advance_hint="Enter a sequence to continue.")


# ============================================================================
# STEP 1 — Localisation
# ============================================================================
elif cur == 1:
    section_header("Subcellular Targeting", "N-terminal transit/signal peptide prediction", "🧭")
    st.caption("Heuristic N-terminal signal prediction — confidence bands are qualitative. "
               "Confirm with TargetP-2 / SignalP-6 before committing a construct.")
    seq = wb.get("seq", "")
    pred = tp.predict_transit_peptide(seq)
    wb.set("tp_type", pred.tp_type)
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card(pred.tp_type or "none", "Predicted signal",
                    "success" if pred.has_tp else "default", "🎯")
    with m2:
        metric_card(f"{pred.confidence:.0%}", "Confidence",
                    "success" if pred.confidence >= 0.66 else "warning" if pred.confidence >= 0.33 else "error")
    with m3:
        metric_card(pred.cleavage_site if pred.cleavage_site else "—", "Cleavage site", "info", "✂")
    insight_bar(pred.confidence, 0, 1, "Signal confidence")
    if pred.has_tp:
        scientific_insight(
            f"Predicted <b>{pred.tp_type}</b> signal (cleavage ~{pred.cleavage_site}). {pred.rationale} "
            "The mature protein (post-cleavage) is what folds in the target compartment — predict "
            "structure on the <b>mature</b> sequence, not the precursor.",
            title="Targeting · interpretation")
        st.markdown("**Mature sequence (post-cleavage)**")
        st.code(pred.mature_sequence(seq), language=None)
    else:
        scientific_insight(
            "No strong N-terminal targeting signal — likely cytosolic/nucleocytoplasmic, or an "
            f"internal/C-terminal signal. {pred.rationale}", title="Targeting · interpretation", icon="🧪")
    wizard_nav(STEPS, key="plant_wiz")


# ============================================================================
# STEP 2 — NLR architecture
# ============================================================================
elif cur == 2:
    section_header("NLR Immune Receptor", "TIR/CC–NBS–LRR domain architecture", "🛡")
    ann = nlr.annotate_nlr(wb.get("seq", ""))
    wb.set("is_nlr", ann.is_nlr)
    wb.set("nlr_class", ann.nlr_class or "NLR")
    if ann.is_nlr:
        b1, b2 = st.columns([0.4, 0.6])
        with b1:
            metric_card(ann.nlr_class or "NLR", "Class", "success", "🛡")
        with b2:
            st.markdown(status_badge(f"NLR · {ann.confidence:.0%} confidence", "ok"), unsafe_allow_html=True)
            st.caption(ann.summary)
        if ann.domains:
            st.markdown("**Domain architecture**")
            st.dataframe([{"Domain": getattr(d, "domain_type", getattr(d, "name", "?")),
                           "Start": getattr(d, "start", "—"), "End": getattr(d, "end", "—")}
                          for d in ann.domains], width="stretch", hide_index=True)
        scientific_insight(
            "NLRs trigger immunity through the <b>NBS</b> nucleotide-binding switch and <b>LRR</b> "
            "recognition surface. P-loop/NBS mutations abolish signalling; LRR-surface changes retune "
            "recognition. Treat <b>LRR loops</b> as the engineering target and the NBS as conserved.",
            title="NLR · engineering guidance")
    else:
        info_box(f"No NLR architecture detected (no NBS/LRR signature). {ann.summary}",
                 variant="info", title="Not an NLR")
    wizard_nav(STEPS, key="plant_wiz")


# ============================================================================
# STEP 3 — Codon & Export
# ============================================================================
elif cur == 3:
    section_header("Host Codon Optimisation", "Back-translate & optimise for expression host", "🧬")
    host = st.selectbox("Expression host", ["wheat", "rice", "maize"], index=0,
                        help="Codon usage tables available for wheat, rice and maize.")
    try:
        res = co.optimize_for_wheat(wb.get("seq", ""), species=host)
    except Exception as exc:
        info_box(f"Codon optimisation failed for host '{host}': {exc}", variant="error")
        res = None
    if res is not None:
        k1, k2, k3 = st.columns(3)
        with k1:
            metric_card(f"{res.cai:.2f}", f"CAI ({host})",
                        "success" if res.cai >= 0.8 else "warning" if res.cai >= 0.6 else "error", "📈")
        with k2:
            metric_card(f"{res.gc_content:.0%}", "GC content",
                        "success" if 0.40 <= res.gc_content <= 0.60 else "warning")
        with k3:
            nflags = len(res.cryptic_splice_sites) + len(res.poly_a_signals)
            metric_card(nflags, "Sequence liabilities", "success" if nflags == 0 else "warning", "⚠")
        insight_bar(res.cai, 0, 1, f"Codon Adaptation Index ({host})")
        st.caption("CAI is scaled 0–1; higher = better-matched to host codon usage (≥ 0.8 is the target).")
        try:
            st.markdown("**CAI across hosts** (0–1, higher = better)")
            st.bar_chart(co.compare_species_cai(wb.get("seq", "")))
        except Exception:
            pass
        for w in (res.warnings or []):
            st.markdown(f"- ⚠ {w}")
        scientific_insight(
            f"Optimised for <b>{host}</b>: CAI {res.cai:.2f}, GC {res.gc_content:.0%}. "
            + ("No cryptic splice / poly-A liabilities — expression-ready. " if nflags == 0 else
               f"{nflags} liability flag(s) — review before synthesis. ")
            + "CAI ≥ 0.8 and GC 40–60% are the practical targets for stable cereal expression.",
            title="Expression · interpretation")
        st.download_button("⬇ Download codon-optimised FASTA", data=res.fasta,
                           file_name=f"optimised_{host}.fasta", mime="text/plain", width="stretch")

    if wizard_nav(STEPS, key="plant_wiz", finish_label="✓ New sequence") == "finish":
        wb.clear()
        wizard_reset("plant_wiz")
        st.rerun()
