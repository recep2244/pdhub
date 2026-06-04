"""Batch job handlers — pure functions (no Streamlit) so they can run in a
fragment tick, an out-of-process worker, or a test.

Each handler takes params and returns {status, result, error}. Implemented kinds
mirror what the batch UI offers: ESMFold-API structure prediction and
sequence-level biophysics.
"""

from __future__ import annotations

from typing import Dict

ESMFOLD_MAX_AA = 400


def run_prediction(sequence: str) -> Dict:
    """Fold a single sequence via the ESMFold Atlas API (≤400 aa, single chain)."""
    seq = (sequence or "").strip().upper()
    if len(seq) > ESMFOLD_MAX_AA:
        return {"status": "failed", "result": None,
                "error": f"Sequence too long for ESMFold API ({len(seq)} aa > {ESMFOLD_MAX_AA})"}
    try:
        import requests
        resp = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            data=seq, headers={"Content-Type": "text/plain"}, timeout=120,
        )
    except Exception as exc:  # network/timeout
        return {"status": "failed", "result": None, "error": f"request failed: {exc}"}
    if resp.status_code != 200:
        return {"status": "failed", "result": None, "error": f"API error {resp.status_code}"}
    plddt = []
    for line in resp.text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                plddt.append(float(line[60:66]))
            except ValueError:
                pass
    return {"status": "complete", "error": None,
            "result": {"pdb": resp.text,
                       "plddt": (sum(plddt) / len(plddt)) if plddt else 0.0}}


def run_biophysics(sequence: str) -> Dict:
    """Sequence-level biophysical properties + solubility for one sequence."""
    try:
        from protein_design_hub.biophysics import (
            calculate_mw, calculate_pi, calculate_gravy, calculate_instability_index,
        )
        from protein_design_hub.biophysics.solubility import SolubilityPredictor
        seq = (sequence or "").strip().upper()
        sol = SolubilityPredictor(sequence=seq).predict()
        return {"status": "complete", "error": None, "result": {
            "mw": calculate_mw(seq),
            "pi": calculate_pi(seq),
            "gravy": calculate_gravy(seq),
            "instability": calculate_instability_index(seq),
            "solubility_score": sol["solubility_score"],
            "aggregation": sol.get("aggregation_propensity", 0),
            "overall": sol.get("overall_assessment", ""),
        }}
    except Exception as exc:
        return {"status": "failed", "result": None, "error": str(exc)}


# kind → handler. ``kind`` is the batch config 'type'.
HANDLERS = {
    "prediction": lambda p: run_prediction(p.get("sequence", "")),
    "biophysics": lambda p: run_biophysics(p.get("sequence", "")),
}


def run_one(kind: str, params: Dict) -> Dict:
    """Dispatch one batch job by kind. Never raises — returns a status dict."""
    handler = HANDLERS.get(kind)
    if handler is None:
        return {"status": "failed", "result": None,
                "error": f"batch kind '{kind}' not implemented (use prediction or biophysics)"}
    try:
        return handler(params)
    except Exception as exc:
        return {"status": "failed", "result": None, "error": str(exc)}
