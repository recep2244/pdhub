"""Physics-aware scoring for mutagenesis hits.

Background
----------
The original mutagenesis ranking used ``0.6·Δmean_pLDDT + 0.4·Δlocal_pLDDT``.
pLDDT is the structure predictor's *confidence*, not a stability or fitness
signal — a mutation that merely makes ESMFold more confident was being labelled
"beneficial" even when it destabilised the protein. Meanwhile the scanner already
computes real physics per mutant (OpenMM-GBSA energy, OST lDDT/RMSD vs WT) that
the ranking ignored.

This module composes those signals into a single, transparent ``improvement``
score, signed so that **higher = more beneficial**:

    score = w_stab·(−ΔΔG)            stability (heuristic/FoldX ΔΔG, kcal/mol)
          + w_esm2·ΔLL               ESM-2 zero-shot fitness (single-sequence)
          − w_conserv·conservation   MSA family conservation penalty
          − w_am·(AM − threshold)    AlphaMissense pathogenicity (flag/tiebreak)
          + w_fold·fold_agreement    fold preserved (OST lDDT / RMSD vs WT)
          − clash_penalty            new steric clashes vs WT
          − ptm_penalty              newly-introduced PTM liabilities

**pLDDT is NOT a ranking term.** Per the gate-vs-rank contract it is used
*only* as a viability gate: if the mutant fold collapses (mean pLDDT below
``MIN_VIABLE_PLDDT``) the candidate is flagged non-viable and heavily demoted.
The structure predictor's *confidence* (Δmean_pLDDT) never moves a viable
candidate up or down the ranking — two mutants that differ only in pLDDT rank
identically. ``delta_mean_plddt`` is still reported for transparency, with a
weight of exactly zero.

The heuristic ΔΔG is sequence-based and robust; GBSA single-point energies on
two *different* predicted structures are noisy, so GBSA is reported but given a
small, sign-only weight. Every call returns a component breakdown so the report
and LLM agents can show *why* a mutation ranked where it did.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ── Composite-score weights (documented, tunable) ──────────────────────────
W_STABILITY = 1.0      # per kcal/mol of −ΔΔG  (stabilising → positive)
W_FOLD = 5.0           # per unit of (lDDT − LDDT_REF)
W_CLASH = 0.05         # per clash-score unit introduced vs WT
W_RMSD = 0.5           # per Å of CA-RMSD beyond RMSD_TOL (fallback when no lDDT)
PTM_PENALTY = 0.75     # per newly-introduced high-risk PTM liability
GBSA_PENALTY = 0.2     # sign-only nudge from ΔGBSA (noisy → small)
W_ESM2 = 0.15          # per unit of ESM-2 Δlog-likelihood (fitness/conservation)
W_AM = 2.0             # per unit of (AlphaMissense score − threshold), human only
W_DEV = 0.4            # developability penalty weight (aggregation/charge)
W_CONSERV = 1.5        # MSA family-conservation penalty (mutating conserved sites)
CONSERV_FLOOR = 0.6    # conservation above this starts to penalise

# Aromatic residues — the clearest single-residue aggregation liability for a
# sequence-only screen. (Aliphatic swaps like A→V are common stabilising core
# mutations and are NOT penalised without surface-exposure information.)
_AGG_PRONE = set("FWY")

LDDT_REF = 0.90        # lDDT at/above this = fold preserved
RMSD_TOL = 1.0         # Å of CA-RMSD tolerated before penalising
MIN_VIABLE_PLDDT = 50.0  # mutant mean pLDDT below this → fold likely collapsed

# Classification thresholds on the composite score (≈ kcal/mol scale).
BENEFICIAL_CUTOFF = 0.25
DETRIMENTAL_CUTOFF = -0.25

# Structurally critical residues whose *removal* is high-risk.
_CRITICAL_TO_REMOVE = {
    "C": "possible disulfide partner",
    "G": "backbone flexibility (turns/hinges)",
    "P": "backbone rigidity (turn/kink)",
}


def _extract_gbsa(extra_metrics: Dict[str, Any]) -> Optional[float]:
    """Pull a GBSA / potential energy (kJ/mol) from an extra_metrics dict, if present."""
    if not isinstance(extra_metrics, dict):
        return None
    for key in ("openmm_gbsa_energy_kj_mol", "openmm_potential_energy_kj_mol"):
        v = extra_metrics.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def substitution_risk(wt_aa: str, mut_aa: str, exposure: str = "unknown") -> Dict[str, Any]:
    """Per-substitution risk summary used to gate/annotate suggestions.

    Returns ``ddg_kcal`` (heuristic ΔΔG, + = destabilising), an ``effect`` string,
    a 0–1 ``risk`` score (higher = riskier to make), and human-readable ``flags``.
    """
    from protein_design_hub.biophysics.stability import estimate_ddg_mutation

    wt_aa = (wt_aa or "").upper()
    mut_aa = (mut_aa or "").upper()
    ddg, effect = estimate_ddg_mutation(wt_aa, mut_aa, exposure)

    flags: List[str] = []
    risk = 0.0
    # Destabilising substitutions are risky.
    if ddg > 0:
        risk += min(ddg / 3.0, 1.0) * 0.6
    # Removing a structurally critical WT residue.
    if wt_aa in _CRITICAL_TO_REMOVE and mut_aa != wt_aa:
        flags.append(f"removes {wt_aa} ({_CRITICAL_TO_REMOVE[wt_aa]})")
        risk += 0.3
    # Introducing a free cysteine (mispairing risk).
    if mut_aa == "C" and wt_aa != "C":
        flags.append("introduces free Cys (mispairing/oxidation risk)")
        risk += 0.2
    # Proline introduction inside (likely) structured regions.
    if mut_aa == "P" and wt_aa != "P":
        flags.append("introduces Pro (backbone constraint)")
        risk += 0.15
    return {
        "ddg_kcal": ddg,
        "effect": effect,
        "risk": round(min(risk, 1.0), 3),
        "flags": flags,
    }


def developability_delta(wt_aa: str, mt_aa: str) -> Dict[str, object]:
    """Sequence-based developability risk for a substitution (aggregation/charge).

    Returns ``{risk: 0..1, flags: [...]}``. Captures the two cheap, robust
    liabilities a protein engineer screens for when picking expressible
    candidates: introducing an aggregation-prone hydrophobic residue, and large
    surface-charge swings.
    """
    from protein_design_hub.analysis.protein_utils import AA_PROPERTIES

    wt_aa, mt_aa = (wt_aa or "").upper(), (mt_aa or "").upper()
    flags: list = []
    risk = 0.0
    if wt_aa not in AA_PROPERTIES or mt_aa not in AA_PROPERTIES or wt_aa == mt_aa:
        return {"risk": 0.0, "flags": flags}

    # Aggregation: introducing an aromatic where WT wasn't is a developability risk.
    if mt_aa in _AGG_PRONE and wt_aa not in _AGG_PRONE:
        risk += 0.3
        flags.append("↑aggregation propensity (aromatic introduced)")
    # Charge swing (affects solubility / pI).
    dq = AA_PROPERTIES[mt_aa]["charge"] - AA_PROPERTIES[wt_aa]["charge"]
    if abs(dq) >= 1:
        risk += 0.2 * min(abs(dq), 2)
        flags.append(f"charge change {dq:+.0f}")
    return {"risk": round(min(risk, 1.0), 3), "flags": flags}


def conservation_profile(alignment, method: str = "composite"):
    """Per-position MSA conservation (0–1, higher = more conserved).

    Thin wrapper over ``msa.conservation.calculate_conservation`` so the
    mutagenesis scorer can consume an alignment (list of aligned sequences, or a
    FASTA/A3M path) without importing the MSA package directly. Returns ``None``
    on any failure (graceful — the composite simply omits the conservation term).
    """
    try:
        from protein_design_hub.msa.conservation import calculate_conservation
        scores = calculate_conservation(alignment, method=method)
        return [float(s) for s in scores] if scores else None
    except Exception:
        return None


def introduces_ptm_liability(
    wt_sequence: str,
    mutant_sequence: str,
    _scanner: Any = None,
) -> int:
    """Return (mutant high-risk PTM count − WT high-risk PTM count).

    Positive means the mutation *introduces* developability liabilities
    (new N-glycosylation sequon, deamidation/isomerisation motif, etc.).
    Sequence-based and cheap; reuses the validated PTMLiabilityScanner.
    """
    try:
        from protein_design_hub.analysis.ptm_scanner import PTMLiabilityScanner
        scanner = _scanner or PTMLiabilityScanner()
        wt_hr = len(scanner.scan(wt_sequence).high_risk())
        mut_hr = len(scanner.scan(mutant_sequence).high_risk())
        return int(mut_hr - wt_hr)
    except Exception:
        return 0


def composite_mutation_score(
    mutant: Dict[str, Any],
    wt_metrics: Optional[Dict[str, Any]] = None,
    sequence: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute the physics-aware improvement score for one mutant result dict.

    Args:
        mutant: a mutation result dict (keys: original_aa, mutant_aa, position,
            delta_mean_plddt, mean_plddt, rmsd_to_base, ost_lddt, ost_rmsd_ca,
            clash_score, extra_metrics, …).
        wt_metrics: ``context.extra['mutation_wt_metrics']`` for WT baselines
            (clash_score, extra_metrics with GBSA).
        sequence: WT sequence, enabling the PTM-introduction check.

    Returns:
        (score, components) where ``components`` is a transparent breakdown.
    """
    wt_metrics = wt_metrics or {}
    comp: Dict[str, Any] = {}
    flags: List[str] = []

    wt_aa = str(mutant.get("original_aa", "") or "")
    mut_aa = str(mutant.get("mutant_aa", "") or "")

    # ── stability (FoldX ΔΔG preferred; heuristic fallback) ────────────────
    sub = substitution_risk(wt_aa, mut_aa)
    foldx_ddg = mutant.get("foldx_ddg_kcal_mol")
    if isinstance(foldx_ddg, (int, float)):
        ddg = float(foldx_ddg)
        comp["ddg_source"] = "foldx"
        comp["ddg_effect"] = ("stabilising" if ddg < -1 else "destabilising" if ddg > 1 else "neutral")
    else:
        ddg = sub["ddg_kcal"]
        comp["ddg_source"] = "heuristic"
        comp["ddg_effect"] = sub["effect"]
    comp["ddg_kcal"] = ddg
    stability_term = W_STABILITY * (-ddg)
    flags.extend(sub["flags"])

    # ── developability (aggregation / charge) ──────────────────────────────
    dev = developability_delta(wt_aa, mut_aa)
    dev_penalty = -W_DEV * float(dev["risk"])
    if dev["flags"]:
        flags.extend(dev["flags"])

    # ── fold agreement (OST lDDT preferred, else RMSD) ─────────────────────
    ost_lddt = mutant.get("ost_lddt")
    rmsd = mutant.get("ost_rmsd_ca", mutant.get("rmsd_to_base"))
    if isinstance(ost_lddt, (int, float)):
        fold_term = W_FOLD * (float(ost_lddt) - LDDT_REF)
        comp["ost_lddt"] = float(ost_lddt)
    elif isinstance(rmsd, (int, float)):
        fold_term = -W_RMSD * max(0.0, float(rmsd) - RMSD_TOL)
        comp["rmsd_ca"] = float(rmsd)
    else:
        fold_term = 0.0

    # ── confidence: REPORTED ONLY, never ranked ────────────────────────────
    # pLDDT is a gate, not a ranking signal (see module docstring). We surface
    # Δmean_pLDDT for transparency but it contributes exactly zero to the score.
    d_plddt = mutant.get("delta_mean_plddt", 0.0) or 0.0
    conf_term = 0.0  # hard zero — pLDDT must not influence ranking

    # ── clash penalty (new clashes vs WT) ──────────────────────────────────
    clash_penalty = 0.0
    mut_clash = mutant.get("clash_score")
    wt_clash = wt_metrics.get("clash_score")
    if isinstance(mut_clash, (int, float)) and isinstance(wt_clash, (int, float)):
        extra_clash = max(0.0, float(mut_clash) - float(wt_clash))
        clash_penalty = -W_CLASH * extra_clash
        if extra_clash > 0:
            flags.append(f"+{extra_clash:.0f} clash vs WT")

    # ── PTM liability introduction ─────────────────────────────────────────
    ptm_penalty = 0.0
    pos = mutant.get("position")
    if sequence and isinstance(pos, int) and 1 <= pos <= len(sequence) and mut_aa in "ACDEFGHIKLMNPQRSTVWY":
        mutant_seq = sequence[: pos - 1] + mut_aa + sequence[pos:]
        new_liab = introduces_ptm_liability(sequence, mutant_seq)
        if new_liab > 0:
            ptm_penalty = -PTM_PENALTY * new_liab
            flags.append(f"introduces {new_liab} PTM liability(ies)")
        comp["ptm_liability_delta"] = new_liab

    # ── ESM-2 zero-shot Δlog-likelihood (fitness/conservation; distinct from
    # the ΔΔG stability term above) ─────────────────────────────────────────
    esm2_term = 0.0
    esm2_delta = mutant.get("esm2_delta_ll")
    if isinstance(esm2_delta, (int, float)):
        esm2_term = W_ESM2 * float(esm2_delta)   # negative Δ (deleterious) lowers score
        comp["esm2_delta_ll"] = round(float(esm2_delta), 3)
        verdict = mutant.get("esm2_verdict")
        if verdict in ("strongly_deleterious", "likely_deleterious"):
            flags.append(f"ESM-2 {verdict.replace('_', ' ')}")

    # ── AlphaMissense pathogenicity (human canonical only) ─────────────────
    am_term = 0.0
    am_score = mutant.get("am_score")
    if isinstance(am_score, (int, float)):
        am_thr = mutant.get("am_threshold", 0.564)
        am_term = -W_AM * (float(am_score) - float(am_thr))  # pathogenic → lowers score
        comp["am_score"] = round(float(am_score), 3)
        if float(am_score) >= float(am_thr):
            flags.append(f"AlphaMissense likely pathogenic ({am_score:.2f})")

    # ── MSA family conservation: mutating a highly-conserved column is risky ──
    # Complementary to ESM-2 (single-sequence): this is alignment-derived, so it
    # flags positions conserved across homologues (often functional/structural).
    conserv_penalty = 0.0
    conserv = mutant.get("conservation")
    if isinstance(conserv, (int, float)):
        comp["conservation"] = round(float(conserv), 3)
        if conserv > CONSERV_FLOOR:
            conserv_penalty = -W_CONSERV * (float(conserv) - CONSERV_FLOOR)
            flags.append(f"MSA-conserved position ({conserv:.2f})")

    # ── GBSA (noisy → small, sign-only nudge) ──────────────────────────────
    gbsa_term = 0.0
    mut_gbsa = _extract_gbsa(mutant.get("extra_metrics", {}))
    wt_gbsa = _extract_gbsa(wt_metrics.get("extra_metrics", {}))
    if mut_gbsa is not None and wt_gbsa is not None:
        d_gbsa = mut_gbsa - wt_gbsa  # lower energy = more stable
        comp["delta_gbsa_kj_mol"] = round(d_gbsa, 1)
        if d_gbsa < 0:
            gbsa_term = GBSA_PENALTY
        elif d_gbsa > 0:
            gbsa_term = -GBSA_PENALTY

    # ── confidence gate ────────────────────────────────────────────────────
    mean_plddt = mutant.get("mean_plddt")
    gate_ok = True
    if isinstance(mean_plddt, (int, float)) and float(mean_plddt) < MIN_VIABLE_PLDDT:
        gate_ok = False
        flags.append(f"mutant fold low-confidence (pLDDT {float(mean_plddt):.0f}<{MIN_VIABLE_PLDDT:.0f})")

    score = (stability_term + fold_term + conf_term + clash_penalty
             + ptm_penalty + gbsa_term + esm2_term + am_term + dev_penalty + conserv_penalty)
    if not gate_ok:
        score -= 1.0  # heavy demotion — prediction unreliable

    comp.update({
        "stability_term": round(stability_term, 3),
        "fold_term": round(fold_term, 3),
        "confidence_term": round(conf_term, 3),
        "clash_penalty": round(clash_penalty, 3),
        "ptm_penalty": round(ptm_penalty, 3),
        "gbsa_term": round(gbsa_term, 3),
        "esm2_term": round(esm2_term, 3),
        "am_term": round(am_term, 3),
        "dev_penalty": round(dev_penalty, 3),
        "conservation_penalty": round(conserv_penalty, 3),
        "gate_ok": gate_ok,
        "flags": flags,
        "score": round(score, 3),
    })
    return round(score, 4), comp


def classify_score(score: float) -> str:
    """Map a composite score to beneficial / neutral / detrimental."""
    if score > BENEFICIAL_CUTOFF:
        return "beneficial"
    if score < DETRIMENTAL_CUTOFF:
        return "detrimental"
    return "neutral"


def position_risk_summary(sequence: str, positions: List[int]) -> Dict[int, Dict[str, Any]]:
    """Per-position mutability risk for the suggestion agent (data-backed gate).

    For each 1-indexed position, summarise the WT residue's criticality so the
    LLM's "avoid conserved/critical residues" guidance is backed by an actual
    signal rather than a single-sequence guess.
    """
    out: Dict[int, Dict[str, Any]] = {}
    for p in positions:
        if not (isinstance(p, int) and 1 <= p <= len(sequence)):
            continue
        wt = sequence[p - 1].upper()
        notes: List[str] = []
        risk = 0.0
        if wt in _CRITICAL_TO_REMOVE:
            notes.append(_CRITICAL_TO_REMOVE[wt])
            risk += 0.4
        # crude local context: Cys may be disulfide-bonded; Gly/Pro in runs = turns
        out[p] = {"wt_aa": wt, "risk": round(min(risk, 1.0), 3), "notes": notes}
    return out
