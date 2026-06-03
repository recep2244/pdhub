"""De novo binder design — campaign planning, tool availability, RFdiffusion
backbone generation, and campaign-health assessment."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st

from protein_design_hub.web.ui import (
    inject_base_css, page_header, section_header, info_box, metric_card,
    status_badge, sidebar_nav, sidebar_system_status,
)
from protein_design_hub.design.campaign import (
    plan_campaign, assess_health, recommend_stack,
)

st.set_page_config(page_title="Binder Design - Protein Design Hub", page_icon="🔗", layout="wide")
inject_base_css()
sidebar_nav(current="Binder Design")
sidebar_system_status()
try:
    from protein_design_hub.web.agent_helpers import agent_sidebar_status
    agent_sidebar_status()
except Exception:
    pass

page_header(
    "De Novo Binder Design",
    "Plan a binder campaign end-to-end: size the funnel, estimate cost & yield, "
    "generate backbones, and track campaign health.",
    "🔗",
)


# ── Tool availability ────────────────────────────────────────────────────────
def _tool_availability() -> dict:
    avail = {}
    try:
        from protein_design_hub.design.generators.registry import GeneratorRegistry
        gens = [g.lower() for g in GeneratorRegistry.list_available()]
        avail["RFdiffusion"] = "rfdiffusion" in gens
    except Exception:
        avail["RFdiffusion"] = False
    try:
        from protein_design_hub.design.registry import DesignerRegistry  # type: ignore
        ds = [d.lower() for d in DesignerRegistry.list_available()]
        avail["ProteinMPNN"] = any("mpnn" in d for d in ds)
    except Exception:
        avail["ProteinMPNN"] = Path(PROJECT_SRC, "protein_design_hub/design/proteinmpnn").exists()
    import shutil
    avail["BindCraft"] = shutil.which("bindcraft") is not None
    avail["BoltzGen"] = shutil.which("boltzgen") is not None
    return avail

section_header("Design Toolchain", "Available backbone & sequence generators", "🧰")
_avail = _tool_availability()
cols = st.columns(len(_avail))
for col, (tool, ok) in zip(cols, _avail.items()):
    with col:
        st.markdown(
            f'<div style="text-align:center;padding:6px 0">{status_badge(tool, "ok" if ok else "warning")}'
            f'<div style="font-size:.72rem;color:var(--pdhub-text-muted);margin-top:6px">'
            f'{"ready" if ok else "not installed"}</div></div>',
            unsafe_allow_html=True,
        )
if not _avail.get("RFdiffusion"):
    info_box(
        "RFdiffusion is not installed — you can still plan the campaign and estimate "
        "yield/cost below. Install it from the Settings page to enable backbone generation.",
        variant="info", title="Planning available without GPU tools",
    )


# ── Campaign planner ─────────────────────────────────────────────────────────
section_header("Campaign Planner", "Size the funnel, estimate cost & expected yield", "📐")

with st.form("campaign_plan"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target = st.number_input("Target binders", 1, 200, 10,
                                 help="Diverse, QC-passing binders you want at the end")
    with c2:
        difficulty = st.selectbox("Target difficulty", ["easy", "medium", "difficult"], index=1,
                                  help="Easy=concave pocket/known binders; difficult=flat/convex, novel")
    with c3:
        priority = st.selectbox("Priority", ["standard", "higher_success", "all_atom", "fast"],
                                help="Drives the recommended tool stack")
    with c4:
        predictor = st.selectbox("Validation predictor", ["chai", "boltz2", "colabfold", "esmfold"],
                                 help="Chai/Boltz give PAE → enables ipSAE interface ranking")
    submitted = st.form_submit_button("📐 Plan campaign", type="primary", width="stretch")

if submitted or st.session_state.get("_binder_plan"):
    if submitted:
        st.session_state["_binder_plan"] = plan_campaign(
            int(target), difficulty, priority, predictor=predictor)
    plan = st.session_state["_binder_plan"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card(f"{plan.backbones:,}", "Backbones to generate", "info", "🧬")
    with m2:
        metric_card(f"{plan.sequences:,}", "Sequences (MPNN)", "info", "✍")
    with m3:
        metric_card(f"{plan.expected_passing_low}–{plan.expected_passing_high}",
                    "Expected QC-passing", "success", "✅")
    with m4:
        metric_card(f"${plan.est_cost_usd:,.0f}", f"Est. cost · ~{plan.est_hours:.0f}h", "gradient", "💵")

    st.markdown("**Design funnel**")
    total = max(plan.funnel[0]["count"], 1)
    funnel_rows = [{"Stage": s["stage"], "Count": f'{s["count"]:,}',
                    "% of start": f'{100*s["count"]/total:.1f}%'} for s in plan.funnel]
    st.dataframe(funnel_rows, width="stretch", hide_index=True)

    info_box(
        f"<b>Recommended stack:</b> {plan.recommended_stack}<br>"
        f"<span style='color:var(--pdhub-text-secondary)'>{plan.rationale}</span>",
        variant="tip", title=f"{plan.difficulty.title()} target · ~{int(plan.pass_rate*100)}% pass rate",
    )
    for n in plan.notes:
        st.caption("• " + n)


# ── Backbone generation (RFdiffusion) ────────────────────────────────────────
section_header("Backbone Generation", "RFdiffusion de novo backbones", "🧱")
if not _avail.get("RFdiffusion"):
    st.caption("⚪ RFdiffusion unavailable — install to enable. Planning above works regardless.")
else:
    with st.form("rfd_gen"):
        g1, g2 = st.columns(2)
        with g1:
            n_designs = st.number_input("Number of backbones", 1, 500, 8)
            contigs = st.text_input("Contig map", value="100-100",
                                    help="RFdiffusion contigmap.contigs, e.g. '100-100' or 'A1-100/0 50-50'")
        with g2:
            target_pdb = st.text_input("Target PDB (optional, for binders)", value="",
                                       help="Path to target structure for binder design")
        go = st.form_submit_button("🧱 Generate backbones", type="primary", width="stretch")
    if go:
        try:
            from protein_design_hub.design.generators.registry import get_generator
            from protein_design_hub.design.generators.types import BackboneInput
            out_dir = Path("outputs") / f"binder_bb_{datetime.now():%Y%m%d_%H%M%S}"
            out_dir.mkdir(parents=True, exist_ok=True)
            bi = BackboneInput(
                job_id=out_dir.name, output_dir=out_dir,
                num_designs=int(n_designs), contigs=contigs or None,
                input_pdb=Path(target_pdb) if target_pdb else None,
            )
            with st.spinner("Running RFdiffusion…"):
                gen = get_generator("rfdiffusion")
                res = gen.generate(bi)
            if getattr(res, "success", False):
                info_box(f"Generated {len(res.backbone_paths)} backbone(s) → {out_dir}",
                         variant="success", title="Backbones ready")
                for p in res.backbone_paths[:10]:
                    st.caption(f"• {p}")
            else:
                info_box(getattr(res, "error_message", "generation failed"), variant="error")
        except Exception as e:
            info_box(f"Backbone generation failed: {e}", variant="error")


# ── Campaign health ──────────────────────────────────────────────────────────
section_header("Campaign Health", "Diagnose a running campaign from observed pass rates", "🩺")
h1, h2, h3 = st.columns(3)
with h1:
    plddt_pass = st.slider("pLDDT pass fraction", 0.0, 1.0, 0.4, 0.01)
with h2:
    iptm_pass = st.slider("ipTM pass fraction", 0.0, 1.0, 0.3, 0.01)
with h3:
    scrmsd_pass = st.slider("scRMSD pass fraction", 0.0, 1.0, 0.5, 0.01)

health = assess_health(plddt_pass, iptm_pass, scrmsd_pass)
hv = "success" if "EXCELLENT" in health["health"] or "GOOD" in health["health"] else (
    "warning" if "MARGINAL" in health["health"] else "error")
info_box(f"<b>{health['health']}</b> — overall pass {health['overall_pass']*100:.1f}%<br>"
         f"<span style='color:var(--pdhub-text-secondary)'>{health['action']}</span>",
         variant=hv, title="Campaign health")
for d in health["diagnostics"]:
    st.markdown(f"- ⚠ {d}")
