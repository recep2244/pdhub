"""Protein-design quality control — research-backed thresholds, sequence-liability
detection, composite ranking, and a sequential filtering pipeline.

Thresholds and the composite formula come from the ``protein-qc`` skill (binder
design competitions + published benchmarks). Key principle: individual metrics
are weak pre-screening filters (ROC AUC ~0.65); **composite scoring is required**
for meaningful ranking — these gates remove poor designs, they do not predict
affinity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Thresholds (standard / stringent). Direction = "high" (≥ is better) or "low".
# --------------------------------------------------------------------------- #
THRESHOLDS: Dict[str, Dict] = {
    # Structural confidence
    "plddt":            {"std": 85.0, "strict": 90.0, "dir": "high", "level": "structural", "scale": 100},
    "ptm":              {"std": 0.70, "strict": 0.80, "dir": "high", "level": "structural"},
    "sc_rmsd":          {"std": 2.0,  "strict": 1.5,  "dir": "low",  "level": "structural"},
    # Binding / interface
    "iptm":             {"std": 0.50, "strict": 0.60, "dir": "high", "level": "binding"},
    "ipsae":            {"std": 0.61, "strict": 0.70, "dir": "high", "level": "binding"},
    "pae_interaction":  {"std": 12.0, "strict": 10.0, "dir": "low",  "level": "binding"},
    "shape_complementarity": {"std": 0.50, "strict": 0.60, "dir": "high", "level": "binding"},
    "interface_dg":     {"std": -10.0, "strict": -15.0, "dir": "low", "level": "binding"},
    # Expression / developability
    "instability":      {"std": 40.0, "strict": 30.0, "dir": "low",  "level": "expression"},
    "gravy":            {"std": 0.4,  "strict": 0.2,  "dir": "low",  "level": "expression"},
    "esm2_pll_normalized": {"std": 0.0, "strict": 0.2, "dir": "high", "level": "expression"},
}

_KD = {  # Kyte-Doolittle hydropathy
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
_HYDROPHOBIC = set("AILMFVWY")


# --------------------------------------------------------------------------- #
# Sequence-level liability detection (design-level checks)
# --------------------------------------------------------------------------- #
def sequence_liabilities(seq: str) -> Dict:
    """Detect manufacturability/expression liabilities in a sequence.

    Returns counts, a list of human flags, and an additive risk score (0–100).
    """
    s = (seq or "").upper().strip()
    flags: List[str] = []
    risk = 0.0

    cys = s.count("C")
    if cys % 2 == 1:
        flags.append(f"Odd cysteine count ({cys}) — unpaired disulfide risk")
        risk += 25

    import re
    deam = re.findall(r"N[GST]", s)          # NG/NS/NT deamidation/isomerisation
    if deam:
        flags.append(f"{len(deam)} deamidation motif(s) (NG/NS/NT)")
        risk += min(15, 3 * len(deam))

    poly = re.findall(r"[KR]{3,}", s)        # polybasic → proteolysis
    if poly:
        flags.append(f"{len(poly)} polybasic run(s) (≥3 K/R) — proteolysis risk")
        risk += min(15, 5 * len(poly))

    hyd = re.findall(rf"[{''.join(_HYDROPHOBIC)}]{{6,}}", s)  # aggregation
    if hyd:
        flags.append(f"{len(hyd)} hydrophobic run(s) (≥6) — aggregation risk")
        risk += min(20, 7 * len(hyd))

    if s and s.count("M") and s[0] != "M":
        pass  # informational only

    return {
        "cysteine_count": cys,
        "cysteine_even": cys % 2 == 0,
        "deamidation_sites": len(deam),
        "polybasic_runs": len(poly),
        "hydrophobic_runs": len(hyd),
        "flags": flags,
        "risk": round(min(risk, 100.0), 1),
    }


def biophysical_properties(seq: str) -> Dict:
    """GRAVY, theoretical pI, MW, aromaticity, and (if BioPython is present)
    the Guruprasad instability index.

    Delegates to :func:`protein_design_hub.biophysics.properties.compute_properties`
    — the single canonical implementation — while preserving this function's
    historical return shape: keys ``length``, ``gravy``, ``pi``, ``mw`` and
    ``instability`` are always present, and ``aromaticity`` is added only when
    BioPython is available (matching prior behaviour for downstream callers).
    """
    s = "".join(c for c in (seq or "").upper() if c in _KD)
    if not s:
        return {"gravy": None, "pi": None, "mw": None, "instability": None, "length": 0}

    from ..biophysics import properties as _props

    full = _props.compute_properties(s)
    out = {
        "length": full["length"],
        "gravy": full["gravy"],
        "pi": full["pi"],
        "mw": full["mw"],
        "instability": full["instability"],
    }
    if _props._has_biopython():  # historical: aromaticity only with BioPython
        out["aromaticity"] = full["aromaticity"]
    return out


_PKA = {"Cterm": 3.55, "Nterm": 7.5, "D": 4.05, "E": 4.45, "C": 9.0,
        "Y": 10.0, "H": 5.98, "K": 10.0, "R": 12.0}


def _charge_at_ph(seq: str, ph: float) -> float:
    pos = 1.0 / (1 + 10 ** (ph - _PKA["Nterm"]))
    pos += seq.count("K") / (1 + 10 ** (ph - _PKA["K"]))
    pos += seq.count("R") / (1 + 10 ** (ph - _PKA["R"]))
    pos += seq.count("H") / (1 + 10 ** (ph - _PKA["H"]))
    neg = 1.0 / (1 + 10 ** (_PKA["Cterm"] - ph))
    neg += seq.count("D") / (1 + 10 ** (_PKA["D"] - ph))
    neg += seq.count("E") / (1 + 10 ** (_PKA["E"] - ph))
    neg += seq.count("C") / (1 + 10 ** (_PKA["C"] - ph))
    neg += seq.count("Y") / (1 + 10 ** (_PKA["Y"] - ph))
    return pos - neg


def _isoelectric_point(seq: str) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _charge_at_ph(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


_AA_MW = {"A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
          "Q": 128.13, "E": 129.12, "G": 57.05, "H": 137.14, "I": 113.16,
          "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18, "P": 97.12,
          "S": 87.08, "T": 101.10, "W": 186.21, "Y": 163.18, "V": 99.13}


def _molecular_weight(seq: str) -> float:
    return sum(_AA_MW.get(c, 110.0) for c in seq) + 18.02


# --------------------------------------------------------------------------- #
# Metric gating + composite ranking
# --------------------------------------------------------------------------- #
def check_metric(name: str, value: Optional[float], level: str = "stringent") -> Dict:
    """Return {pass, threshold, dir, level} for one metric against a threshold."""
    cfg = THRESHOLDS.get(name)
    if cfg is None or value is None:
        return {"metric": name, "value": value, "pass": None, "reason": "no threshold/value"}
    thr = cfg["strict"] if level == "stringent" else cfg["std"]
    ok = value >= thr if cfg["dir"] == "high" else value <= thr
    return {"metric": name, "value": value, "threshold": thr, "dir": cfg["dir"],
            "level": cfg["level"], "pass": bool(ok)}


def composite_score(metrics: Dict[str, float]) -> Optional[float]:
    """Weighted composite (protein-qc skill). Re-normalises over available terms
    so a missing metric doesn't zero the score. Returns 0–1 or None."""
    plddt = metrics.get("plddt")
    plddt = (plddt / 100.0) if plddt is not None and plddt > 1.5 else plddt
    pae = metrics.get("pae_interaction")
    terms = [
        (0.30, plddt),
        (0.20, metrics.get("iptm", metrics.get("ipsae"))),
        (0.20, (1 - pae / 20.0) if pae is not None else None),
        (0.15, metrics.get("shape_complementarity")),
        (0.15, metrics.get("esm2_pll_normalized")),
    ]
    avail = [(w, v) for w, v in terms if v is not None]
    if not avail:
        return None
    wsum = sum(w for w, _ in avail)
    return round(sum(w * max(0.0, min(1.0, float(v))) for w, v in avail) / wsum, 4)


@dataclass
class DesignAssessment:
    composite: Optional[float]
    checks: List[Dict]
    liabilities: Dict = field(default_factory=dict)
    biophysical: Dict = field(default_factory=dict)
    passed: bool = False
    verdict: str = ""


def assess_design(metrics: Dict[str, float], sequence: Optional[str] = None,
                  level: str = "stringent") -> DesignAssessment:
    """Full QC of one design: gate every provided metric, score the composite,
    and (if a sequence is given) add liabilities + biophysical properties."""
    checks = [check_metric(k, metrics.get(k), level) for k in THRESHOLDS if k in metrics]
    comp = composite_score(metrics)
    liab = sequence_liabilities(sequence) if sequence else {}
    biop = biophysical_properties(sequence) if sequence else {}

    hard_fail = any(c.get("pass") is False for c in checks)
    cys_fail = bool(liab) and not liab.get("cysteine_even", True)
    passed = (not hard_fail) and (not cys_fail) and (comp is None or comp >= 0.5)

    if comp is None:
        v = "⚪ Insufficient metrics for composite ranking"
    elif passed:
        v = f"🟢 Passes QC ({level}) — composite {comp:.2f}"
    else:
        failed = [c["metric"] for c in checks if c.get("pass") is False]
        if cys_fail:
            failed.append("odd-cysteines")
        v = f"🔴 Fails QC — composite {comp:.2f}; " + (", ".join(failed) or "below composite floor")
    return DesignAssessment(comp, checks, liab, biop, passed, v)


def filter_designs(rows: List[Dict], level: str = "standard") -> Dict:
    """Sequential multi-stage filter (structural → self-consistency → binding →
    plausibility → expression). Returns survivors, per-stage counts, and drop
    reasons — mirrors the protein-qc skill pipeline."""
    stages = [
        ("structural confidence", lambda r: check_metric("plddt", r.get("plddt"), level).get("pass") is not False),
        ("self-consistency",      lambda r: check_metric("sc_rmsd", r.get("sc_rmsd"), level).get("pass") is not False),
        ("binding quality",       lambda r: check_metric("iptm", r.get("iptm"), level).get("pass") is not False
                                           and check_metric("pae_interaction", r.get("pae_interaction"), level).get("pass") is not False),
        ("sequence plausibility", lambda r: check_metric("esm2_pll_normalized", r.get("esm2_pll_normalized"), level).get("pass") is not False),
        ("expression checks",     lambda r: sequence_liabilities(r.get("sequence", "")).get("cysteine_even", True)
                                           and check_metric("instability", r.get("instability"), level).get("pass") is not False),
    ]
    survivors = list(rows)
    counts = {"input": len(rows)}
    drops: List[Dict] = []
    for name, keep in stages:
        kept = []
        for r in survivors:
            if keep(r):
                kept.append(r)
            else:
                drops.append({"id": r.get("id", r.get("sequence", "?")[:12]), "stage": name})
        survivors = kept
        counts[name] = len(survivors)
    for r in survivors:
        r.setdefault("composite", composite_score(r))
    survivors.sort(key=lambda r: (r.get("composite") or 0.0), reverse=True)
    return {"survivors": survivors, "counts": counts, "dropped": drops}
