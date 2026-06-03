"""Binder-design campaign planning and health assessment.

Turns a high-level goal ("I want N binders against this target") into a concrete,
costed plan: how many backbones/sequences to generate, the expected QC funnel,
time/cost estimates, and a recommended tool stack. Also assesses the health of a
running campaign from observed pass rates. Pure logic — no heavy deps — so it is
unit-testable and safe to import anywhere.

Numbers come from the ``campaign-manager`` skill (binder-design competition
benchmarks). They are planning estimates, not guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Expected end-to-end QC pass rate by target difficulty (fraction of designs that
# clear pLDDT/ipTM/scRMSD). Skill: 15-20% easy, 5-10% difficult.
PASS_RATE = {"easy": 0.18, "medium": 0.12, "difficult": 0.07}

# 50x rule: generate ~50x the final count (pass rate × diversity clustering).
OVERSAMPLE = 50
SEQ_PER_BACKBONE = 8
CLUSTER_RETENTION = 0.04        # passing → diverse finals (~4%, i.e. ~50x → N)

# Per-tool cost/throughput (Modal pricing, skill table).
TOOL_COSTS = {
    "rfdiffusion": {"gpu": "A10G", "rate": 1.20, "per_h": 250},   # backbones/hour
    "proteinmpnn": {"gpu": "T4",   "rate": 0.60, "per_h": 2667},  # sequences/hour
    "esm2":        {"gpu": "A10G", "rate": 1.20, "per_h": 8000},  # PLL scores/hour
    "colabfold":   {"gpu": "A100", "rate": 4.50, "per_h": 1000},  # predictions/hour
    "chai":        {"gpu": "A100", "rate": 4.50, "per_h": 500},
    "esmfold":     {"gpu": "A10G", "rate": 1.20, "per_h": 2000},
}

# Tool-stack recommendation by scenario (skill tool-selection guide).
TOOL_GUIDE = [
    ("higher_success", "BindCraft", "Integrated design+validation loop → highest hit rate"),
    ("all_atom",       "BoltzGen",  "Side-chain-aware all-atom diffusion for precise geometry"),
    ("difficult",      "RFdiffusion + ProteinMPNN + ColabFold", "Diverse backbones, AF2 validation for hard targets"),
    ("fast",           "ESMFold + ESM2", "Fast screening, no PAE — cheap first pass"),
    ("standard",       "RFdiffusion + ProteinMPNN + Chai/Boltz", "Proven miniprotein stack with PAE for ipSAE"),
]


@dataclass
class CampaignPlan:
    target_binders: int
    difficulty: str
    pass_rate: float
    backbones: int
    sequences: int
    expected_passing_low: int
    expected_passing_high: int
    expected_finals: int
    funnel: List[Dict] = field(default_factory=list)
    est_cost_usd: float = 0.0
    est_hours: float = 0.0
    recommended_stack: str = ""
    rationale: str = ""
    notes: List[str] = field(default_factory=list)


def recommend_stack(difficulty: str = "medium", priority: str = "standard") -> Dict:
    """Pick a tool stack from priority ('higher_success'|'all_atom'|'fast'|
    'standard') and difficulty."""
    key = priority
    if difficulty == "difficult" and priority == "standard":
        key = "difficult"
    for k, stack, why in TOOL_GUIDE:
        if k == key:
            return {"stack": stack, "rationale": why}
    return {"stack": "RFdiffusion + ProteinMPNN + Chai/Boltz", "rationale": TOOL_GUIDE[-1][2]}


def plan_campaign(
    target_binders: int = 10,
    difficulty: str = "medium",
    priority: str = "standard",
    seq_per_backbone: int = SEQ_PER_BACKBONE,
    predictor: str = "chai",
) -> CampaignPlan:
    """Plan a binder campaign: sizing, funnel, cost/time, recommended stack."""
    difficulty = difficulty if difficulty in PASS_RATE else "medium"
    pr = PASS_RATE[difficulty]

    # Size to land target_binders finals after pass-rate + clustering.
    backbones = max(int(round(target_binders / max(CLUSTER_RETENTION, 1e-6) / seq_per_backbone)), 50)
    sequences = backbones * seq_per_backbone
    passing_mid = sequences * pr
    passing_low = int(round(passing_mid * 0.83))
    passing_high = int(round(passing_mid * 1.25))
    finals = max(int(round(sequences * CLUSTER_RETENTION)), 1)

    # Funnel stages
    funnel = [
        {"stage": "Backbones (RFdiffusion)", "count": backbones},
        {"stage": "Sequences (ProteinMPNN)", "count": sequences},
        {"stage": "ESM2 PLL filter (>0.0)", "count": int(sequences * 0.6)},
        {"stage": f"Structure pred ({predictor})", "count": int(sequences * 0.6)},
        {"stage": "Pass QC (pLDDT/ipTM/scRMSD)", "count": int(round(passing_mid))},
        {"stage": "Diverse finals (clustered)", "count": finals},
    ]

    # Cost/time across the stack
    cost, hours = _estimate_cost_time(backbones, sequences, predictor)

    rec = recommend_stack(difficulty, priority)
    notes = [
        f"~{OVERSAMPLE}x oversampling: generating {sequences:,} sequences to land ~{finals} diverse binders.",
        f"{difficulty.title()} target → ~{int(pr*100)}% end-to-end QC pass rate.",
        "Order experimentally by composite QC + ipSAE_min; cluster to maximise diversity.",
    ]
    return CampaignPlan(
        target_binders=target_binders, difficulty=difficulty, pass_rate=pr,
        backbones=backbones, sequences=sequences,
        expected_passing_low=passing_low, expected_passing_high=passing_high,
        expected_finals=finals, funnel=funnel,
        est_cost_usd=round(cost, 2), est_hours=round(hours, 1),
        recommended_stack=rec["stack"], rationale=rec["rationale"], notes=notes,
    )


def _estimate_cost_time(backbones: int, sequences: int, predictor: str) -> tuple:
    cost = hours = 0.0
    plan = [
        ("rfdiffusion", backbones),
        ("proteinmpnn", sequences),
        ("esm2", sequences),
        (predictor if predictor in TOOL_COSTS else "chai", int(sequences * 0.6)),
    ]
    for tool, n in plan:
        spec = TOOL_COSTS[tool]
        h = n / spec["per_h"]
        hours += h
        cost += h * spec["rate"]
    return cost, hours


def assess_health(plddt_pass: float, iptm_pass: float, scrmsd_pass: float,
                  overall_pass: Optional[float] = None) -> Dict:
    """Assess a running campaign from observed per-stage pass fractions (0–1)."""
    if overall_pass is None:
        overall_pass = plddt_pass * iptm_pass * scrmsd_pass
    if overall_pass > 0.15:
        health, action = "🟢 EXCELLENT", "Proceed — high yield, you can shrink the run."
    elif overall_pass > 0.10:
        health, action = "🟢 GOOD", "Proceed — normal yield."
    elif overall_pass > 0.05:
        health, action = "🟡 MARGINAL", "Proceed but oversample more, or revisit hotspots."
    else:
        health, action = "🔴 POOR", "Stop — re-scope target/hotspots or switch to BindCraft."

    diagnostics: List[str] = []
    if plddt_pass < 0.20:
        diagnostics.append("Low pLDDT pass — backbones may be undesignable; increase diversity / iterations.")
    if iptm_pass < 0.20:
        diagnostics.append("Low ipTM pass — interface not forming; check hotspots / target surface.")
    if scrmsd_pass < 0.50:
        diagnostics.append("Low scRMSD pass — sequences don't fold to the backbone; raise ProteinMPNN sampling or filter PLL harder.")
    return {"overall_pass": round(overall_pass, 4), "health": health,
            "action": action, "diagnostics": diagnostics}
