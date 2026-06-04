"""De novo binder design — a guided campaign wizard (Define → Generate →
Validate → Brief) over a shared Workbench run-context."""

from __future__ import annotations

import sys
import json
import re as _re
from datetime import datetime
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st

from protein_design_hub.web.ui import (
    inject_base_css, page_header, section_header, info_box, metric_card,
    status_badge, sidebar_nav, sidebar_system_status, wizard, wizard_nav,
    wizard_reset, Workbench,
)
from protein_design_hub.design.campaign import plan_campaign, assess_health

# Project root (…/protein_design_hub), not the process CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[4]

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
    "A guided campaign: define the goal, generate backbones, validate, and export "
    "a campaign brief — state carries across steps in your Workbench.",
    "🔗",
)

wb = Workbench("binder")
STEPS = ["Define & Plan", "Generate Backbones", "Validate", "Campaign Brief"]

# ── Workbench status bar (carried state) ─────────────────────────────────────
_plan = wb.get("plan")
_bb = wb.get("backbones") or []
_parts = []
if _plan is not None:
    _parts.append(status_badge(f"plan · {_plan.target_binders} binders · {_plan.difficulty}", "primary"))
    _parts.append(status_badge(f"predict · {_plan.recommended_stack.split()[0]}", "info"))
if _bb:
    _parts.append(status_badge(f"{len(_bb)} backbones", "ok"))
if wb.get("health"):
    _parts.append(status_badge(wb.get('health')['health'].split()[-1].lower(), "info"))
if _parts:
    st.markdown(
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px">'
        + " ".join(_parts) + "</div>", unsafe_allow_html=True)

cur = wizard(STEPS, key="binder_wiz")
st.markdown("")


def _tool_availability() -> dict:
    avail = {}
    try:
        from protein_design_hub.design.generators.registry import GeneratorRegistry
        avail["RFdiffusion"] = "rfdiffusion" in [g.lower() for g in GeneratorRegistry.list_available()]
    except Exception:
        avail["RFdiffusion"] = False
    try:
        from protein_design_hub.design.registry import DesignerRegistry  # type: ignore
        avail["ProteinMPNN"] = any("mpnn" in d.lower() for d in DesignerRegistry.list_available())
    except Exception:
        avail["ProteinMPNN"] = Path(PROJECT_SRC, "protein_design_hub/design/proteinmpnn").exists()
    import shutil
    avail["BindCraft"] = shutil.which("bindcraft") is not None
    avail["BoltzGen"] = shutil.which("boltzgen") is not None
    return avail


_CONTIG_TOKEN = _re.compile(r"^(?:[A-Za-z]?\d+-\d+|\d+|0)$")


def _valid_contig(spec: str) -> bool:
    if not spec or not spec.strip():
        return True
    tokens = [t for t in _re.split(r"[\s/]+", spec.strip()) if t]
    return bool(tokens) and all(_CONTIG_TOKEN.match(tok) for tok in tokens)


# ============================================================================
# STEP 0 — Define & Plan
# ============================================================================
if cur == 0:
    section_header("Design Toolchain", "Available backbone & sequence generators", "🧰")
    _avail = _tool_availability()
    wb.set("toolchain", _avail)
    cols = st.columns(len(_avail))
    for col, (tool, ok) in zip(cols, _avail.items()):
        with col:
            st.markdown(
                f'<div style="text-align:center;padding:6px 0">{status_badge(tool, "ok" if ok else "warning")}'
                f'<div style="font-size:.72rem;color:var(--pdhub-text-muted);margin-top:6px">'
                f'{"ready" if ok else "not installed"}</div></div>', unsafe_allow_html=True)

    section_header("Campaign Planner", "Size the funnel, estimate cost & expected yield", "📐")
    with st.form("campaign_plan"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            target = st.number_input("Target binders", 1, 200, 10)
        with c2:
            difficulty = st.selectbox("Target difficulty", ["easy", "medium", "difficult"], index=1)
        with c3:
            priority = st.selectbox("Priority", ["standard", "higher_success", "all_atom", "fast"])
        with c4:
            predictor = st.selectbox("Validation predictor", ["chai", "boltz2", "colabfold", "esmfold"],
                                     help="Chai/Boltz2 give inter-chain PAE → ipSAE interface ranking.")
        submitted = st.form_submit_button("📐 Plan campaign", type="primary", width="stretch")

    if predictor in {"esmfold"}:
        info_box(f"<b>{predictor}</b> produces no inter-chain PAE → ipSAE / interface ranking "
                 "(the core binder readout) is unavailable. Use chai / boltz2 to rank complexes.",
                 variant="warning", title="No interface ranking with this predictor")
    elif predictor == "colabfold":
        st.caption("ColabFold gives inter-chain PAE only in multimer mode with a paired MSA.")

    if submitted:
        wb.set("plan", plan_campaign(int(target), difficulty, priority, predictor=predictor))

    plan = wb.get("plan")
    if plan is not None:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card(f"{plan.backbones:,}", "Backbones", "info", "🧬")
        with m2:
            metric_card(f"{plan.sequences:,}", "Sequences (MPNN)", "info", "✍")
        with m3:
            metric_card(f"{plan.expected_passing_low}–{plan.expected_passing_high}", "QC-passing", "success", "✅")
        with m4:
            metric_card(f"${plan.est_cost_usd:,.0f}", f"Cost · ~{plan.est_hours:.0f}h", "gradient", "💵")
        st.markdown("**Design funnel**")
        total = max(plan.funnel[0]["count"], 1)
        st.dataframe([{"Stage": s["stage"], "Count": f'{s["count"]:,}',
                       "% of start": f'{100*s["count"]/total:.1f}%'} for s in plan.funnel],
                     width="stretch", hide_index=True)
        info_box(f"<b>Recommended stack:</b> {plan.recommended_stack}<br>"
                 f"<span style='color:var(--pdhub-text-secondary)'>{plan.rationale}</span>",
                 variant="tip", title=f"{plan.difficulty.title()} target · ~{int(plan.pass_rate*100)}% pass rate")

    wizard_nav(STEPS, key="binder_wiz", can_advance=wb.has("plan"),
               advance_hint="Plan a campaign to continue.")


# ============================================================================
# STEP 1 — Generate Backbones
# ============================================================================
elif cur == 1:
    section_header("Backbone Generation", "RFdiffusion de novo backbones", "🧱")
    plan = wb.get("plan")
    if plan is not None:
        st.caption(f"Plan calls for ~{plan.backbones:,} backbones for {plan.target_binders} target binders.")
    avail = (wb.get("toolchain") or {}).get("RFdiffusion", False)
    if not avail:
        info_box("RFdiffusion is not installed — backbone generation is disabled. You can still "
                 "validate and export the campaign brief; install RFdiffusion from Settings to enable.",
                 variant="info", title="Generation unavailable")
    else:
        with st.form("rfd_gen"):
            g1, g2 = st.columns(2)
            with g1:
                n_designs = st.number_input("Number of backbones", 1, 500,
                                            min(int(plan.backbones) if plan else 8, 500))
                contigs = st.text_input("Contig map", value="100-100",
                                        help="e.g. '100-100' or 'A1-100/0 50-50'")
            with g2:
                target_pdb = st.text_input("Target PDB (optional)", value="")
            go = st.form_submit_button("🧱 Generate backbones", type="primary", width="stretch")
        if go and not _valid_contig(contigs):
            info_box(f"Contig map <code>{contigs}</code> is not valid RFdiffusion grammar.",
                     variant="error", title="Invalid contig map")
        elif go:
            try:
                from protein_design_hub.design.generators.registry import get_generator
                from protein_design_hub.design.generators.types import BackboneInput
                out_dir = PROJECT_ROOT / "outputs" / f"binder_bb_{datetime.now():%Y%m%d_%H%M%S}"
                out_dir.mkdir(parents=True, exist_ok=True)
                bi = BackboneInput(job_id=out_dir.name, output_dir=out_dir,
                                   num_designs=int(n_designs), contigs=contigs or None,
                                   input_pdb=Path(target_pdb) if target_pdb else None)
                with st.spinner("Running RFdiffusion…"):
                    res = get_generator("rfdiffusion").generate(bi)
                if getattr(res, "success", False):
                    wb.set("backbones", [str(p) for p in res.backbone_paths])
                    wb.set("out_dir", str(out_dir))
                    info_box(f"Generated {len(res.backbone_paths)} backbone(s) → {out_dir}",
                             variant="success", title="Backbones ready")
                else:
                    info_box(getattr(res, "error_message", "generation failed"), variant="error")
            except Exception as e:
                info_box(f"Backbone generation failed: {e}", variant="error")

    if wb.get("backbones"):
        st.markdown(f"**{len(wb.get('backbones'))} backbone(s) in this run**")
        for p in wb.get("backbones")[:10]:
            st.caption(f"• {p}")
        st.info("Next: design sequences for these backbones in **MPNN Lab**, then fold & rank "
                "with Chai/Boltz to compute ipSAE.")

    wizard_nav(STEPS, key="binder_wiz")


# ============================================================================
# STEP 2 — Validate (campaign health)
# ============================================================================
elif cur == 2:
    section_header("Campaign Health", "Diagnose from observed (or expected) pass rates", "🩺")
    st.caption("What-if calculator: the verdict is computed live from the slider values — "
               "enter the pass fractions you observe to get a diagnosis.")
    h1, h2, h3 = st.columns(3)
    with h1:
        plddt_pass = st.slider("pLDDT pass fraction", 0.0, 1.0, 0.4, 0.01)
    with h2:
        iptm_pass = st.slider("ipTM/ipSAE pass fraction", 0.0, 1.0, 0.3, 0.01)
    with h3:
        scrmsd_pass = st.slider("scRMSD pass fraction", 0.0, 1.0, 0.5, 0.01)
    health = assess_health(plddt_pass, iptm_pass, scrmsd_pass)
    wb.set("health", health)
    hv = "success" if ("EXCELLENT" in health["health"] or "GOOD" in health["health"]) else (
        "warning" if "MARGINAL" in health["health"] else "error")
    info_box(f"<b>{health['health']}</b> — overall pass {health['overall_pass']*100:.1f}%<br>"
             f"<span style='color:var(--pdhub-text-secondary)'>{health['action']}</span>",
             variant=hv, title="Campaign health")
    for d in health["diagnostics"]:
        st.markdown(f"- ⚠ {d}")
    wizard_nav(STEPS, key="binder_wiz")


# ============================================================================
# STEP 3 — Campaign Brief (export)
# ============================================================================
elif cur == 3:
    section_header("Campaign Brief", "Bundle the run into a downloadable brief", "📦")
    plan = wb.get("plan")
    if plan is None:
        info_box("No campaign plan in this run — go back to step 1 to define one.", variant="warning")
    else:
        brief = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "target_binders": plan.target_binders,
            "difficulty": plan.difficulty,
            "expected_pass_rate": plan.pass_rate,
            "backbones_planned": plan.backbones,
            "sequences_planned": plan.sequences,
            "expected_qc_passing": [plan.expected_passing_low, plan.expected_passing_high],
            "est_cost_usd": plan.est_cost_usd,
            "est_hours": plan.est_hours,
            "recommended_stack": plan.recommended_stack,
            "funnel": plan.funnel,
            "backbones_generated": wb.get("backbones") or [],
            "health": wb.get("health"),
        }
        md = [f"# Binder Campaign Brief", "",
              f"- **Target binders:** {plan.target_binders}  ·  **Difficulty:** {plan.difficulty}  ·  "
              f"**Pass rate:** ~{int(plan.pass_rate*100)}%",
              f"- **Stack:** {plan.recommended_stack}",
              f"- **Plan:** {plan.backbones:,} backbones → {plan.sequences:,} sequences → "
              f"{plan.expected_passing_low}–{plan.expected_passing_high} QC-passing",
              f"- **Estimate:** ${plan.est_cost_usd:,.0f} · ~{plan.est_hours:.0f}h",
              f"- **Backbones generated this run:** {len(wb.get('backbones') or [])}", ""]
        if wb.get("health"):
            md.append(f"- **Health:** {wb.get('health')['health']} — {wb.get('health')['action']}")
        md_text = "\n".join(md)

        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card(f"{plan.target_binders}", "Target binders", "info", "🎯")
        with m2:
            metric_card(f"{len(wb.get('backbones') or [])}", "Backbones generated", "success", "🧱")
        with m3:
            metric_card(f"${plan.est_cost_usd:,.0f}", "Est. cost", "gradient", "💵")
        st.markdown(md_text)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button("⬇ Brief (Markdown)", md_text, file_name="binder_campaign_brief.md",
                               mime="text/markdown", width="stretch")
        with d2:
            st.download_button("⬇ Brief (JSON)", json.dumps(brief, indent=2, default=str),
                               file_name="binder_campaign_brief.json", mime="application/json", width="stretch")
        st.caption("To produce a full ordered candidate list, run ProteinMPNN on the backbones "
                   "(MPNN Lab) → fold with Chai/Boltz → rank by ipSAE on the Jobs page.")

    if wizard_nav(STEPS, key="binder_wiz", finish_label="✓ Start new campaign") == "finish":
        wb.clear()
        wizard_reset("binder_wiz")
        st.rerun()
