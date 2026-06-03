"""Plant / Wheat Biology Track — transit-peptide & localisation, NLR domain
annotation, and host codon optimisation. First-class track (blueprint decision #1)."""

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
    "Engineer plant proteins for expression in N. benthamiana / wheat: subcellular "
    "targeting, NLR immune-receptor architecture, and host codon optimisation.",
    "🌾",
)

_DEMO = ("MASSMLSSAAVATRSNVAQANMVAPFTGLKSSAAFPVTRKQNLDITSIASNGGRVQC"
         "MKTAYIAKQRQISFVKSHFSRQLEERLGGIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK")

seq = st.text_area("Protein sequence (single-letter amino acids)", value="",
                   placeholder=_DEMO, height=110, key="plant_seq").strip().upper()
c1, c2 = st.columns([0.25, 0.75])
with c1:
    use_demo = st.button("Load demo (chloroplast + NLR)", width="stretch")
if use_demo:
    st.session_state["plant_seq"] = _DEMO
    st.rerun()
seq = "".join(ch for ch in seq if ch.isalpha())

if not seq:
    info_box("Paste a plant protein sequence (or load the demo) to run localisation, "
             "NLR annotation, and wheat codon optimisation.", variant="info", title="Awaiting input")
    st.stop()

tab_loc, tab_nlr, tab_codon = st.tabs(
    ["🧭 Localisation & Transit Peptide", "🛡 NLR Domain Architecture", "🧬 Codon Optimisation"])

# ── Localisation ─────────────────────────────────────────────────────────────
with tab_loc:
    section_header("Subcellular Targeting", "N-terminal transit/signal peptide prediction", "🧭")
    st.caption("Heuristic N-terminal signal prediction — confidence bands are qualitative. "
               "Confirm calls and cleavage sites with TargetP-2 / SignalP-6 before committing a construct.")
    pred = tp.predict_transit_peptide(seq)
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
            f"Predicted <b>{pred.tp_type}</b> signal (cleavage ~{pred.cleavage_site}). "
            f"{pred.rationale} The mature protein (post-cleavage) is what folds in the target compartment — "
            "predict structure on the <b>mature</b> sequence, not the precursor.",
            title="Targeting · interpretation")
        mature = pred.mature_sequence(seq)
        st.markdown("**Mature sequence (post-cleavage)**")
        st.code(mature, language=None)
    else:
        scientific_insight(
            "No strong N-terminal targeting signal detected — the protein is likely "
            "cytosolic/nucleocytoplasmic, or uses an internal/C-terminal signal. "
            f"{pred.rationale}", title="Targeting · interpretation", icon="🧪")

# ── NLR architecture ─────────────────────────────────────────────────────────
with tab_nlr:
    section_header("NLR Immune Receptor", "TIR/CC–NBS–LRR domain architecture", "🛡")
    ann = nlr.annotate_nlr(seq)
    if ann.is_nlr:
        b1, b2 = st.columns([0.4, 0.6])
        with b1:
            metric_card(ann.nlr_class or "NLR", "Class", "success", "🛡")
        with b2:
            st.markdown(status_badge(f"NLR · {ann.confidence:.0%} confidence", "ok"), unsafe_allow_html=True)
            st.caption(ann.summary)
        if ann.domains:
            st.markdown("**Domain architecture**")
            rows = []
            for d in ann.domains:
                rows.append({
                    "Domain": getattr(d, "domain_type", getattr(d, "name", "?")),
                    "Start": getattr(d, "start", "—"),
                    "End": getattr(d, "end", "—"),
                })
            st.dataframe(rows, width="stretch", hide_index=True)
        scientific_insight(
            "NLRs trigger immunity through the <b>NBS</b> nucleotide-binding switch and "
            "<b>LRR</b> recognition surface. Mutations in the P-loop/NBS abolish signalling; "
            "LRR-surface changes retune pathogen recognition. Treat <b>CDR-like LRR loops</b> "
            "as the engineering target and the NBS as conserved/off-limits.",
            title="NLR · engineering guidance")
    else:
        info_box("No NLR architecture detected (no NBS/LRR signature). "
                 f"{ann.summary}", variant="info", title="Not an NLR")

# ── Codon optimisation ───────────────────────────────────────────────────────
with tab_codon:
    section_header("Host Codon Optimisation", "Back-translate & optimise for expression host", "🧬")
    host = st.selectbox("Expression host", ["wheat", "rice", "maize"], index=0,
                        help="Codon usage tables available for wheat, rice and maize "
                             "(N. benthamiana table not yet bundled).")
    try:
        res = co.optimize_for_wheat(seq, species=host)
    except Exception as exc:
        info_box(f"Codon optimisation failed for host '{host}': {exc}", variant="error")
        st.stop()
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
    st.caption("CAI is scaled 0–1; higher = better-matched to host codon usage (≥ 0.8 is the practical target).")

    # cross-host comparison
    try:
        comp = co.compare_species_cai(seq)
        st.markdown("**CAI across hosts** (0–1, higher = better)")
        st.bar_chart(comp)
    except Exception:
        pass

    if res.warnings:
        for w in res.warnings:
            st.markdown(f"- ⚠ {w}")
    scientific_insight(
        f"Optimised for <b>{host}</b>: CAI {res.cai:.2f}, GC {res.gc_content:.0%}. "
        + ("No cryptic splice / poly-A liabilities — construct is expression-ready. "
           if nflags == 0 else
           f"{nflags} liability flag(s) — review before synthesis (cryptic splice sites / poly-A signals trigger mis-processing in planta). ")
        + "CAI ≥ 0.8 and GC 40–60% are the practical targets for stable cereal expression.",
        title="Expression · interpretation")

    st.download_button("⬇ Download codon-optimised FASTA", data=res.fasta,
                       file_name=f"optimised_{host}.fasta", mime="text/plain", width="stretch")
