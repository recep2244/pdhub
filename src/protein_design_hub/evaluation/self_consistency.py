"""Design self-consistency QC (scTM / scRMSD / sc-lDDT).

The field-standard de-novo design check: take a *designed sequence*, predict its
structure, and compare that prediction back to the *target backbone* it was
designed for. A sequence is "designable" when its predicted fold matches the
target (high scTM / low scRMSD) — i.e. the design and structure agree.

Uses the ESMFold API for the round-trip prediction (fast, single-chain) and the
fixed OpenStructure ``compare-structures`` runner for scTM/scRMSD/sc-lDDT.
Thresholds follow common practice (e.g. RFdiffusion/ProteinMPNN pipelines):
scTM ≥ 0.5 and scRMSD ≤ 2 Å = self-consistent.

Graceful: returns ``{"status": "unavailable", ...}`` if ESMFold/OST are missing,
so callers can treat it as an optional QC gate.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Optional

SC_TM_PASS = 0.5      # scTM ≥ → designable
SC_RMSD_PASS = 2.0    # scRMSD ≤ Å → designable


def self_consistency(
    designed_sequence: str,
    target_pdb: Path,
    predicted_pdb: Optional[Path] = None,
) -> Dict:
    """Score design↔structure self-consistency for one designed sequence.

    Args:
        designed_sequence: the designed amino-acid sequence.
        target_pdb: the backbone the sequence was designed onto (reference).
        predicted_pdb: optional precomputed prediction of ``designed_sequence``;
            if omitted, it is folded via the ESMFold API.

    Returns ``{status, sc_tm, sc_rmsd, sc_lddt, passed, n_residues}``.
    """
    target_pdb = Path(target_pdb)
    if not target_pdb.exists():
        return {"status": "error", "error": f"target not found: {target_pdb}"}

    tmp_dir = tempfile.mkdtemp(prefix="self_consistency_")
    try:
        # 1) structure of the designed sequence (fold if not supplied)
        if predicted_pdb is None:
            try:
                from protein_design_hub.analysis.esm_dimer import fold_sequence_esm_api
                pdb_str = fold_sequence_esm_api(designed_sequence)
            except Exception as e:  # noqa: BLE001
                return {"status": "unavailable", "error": f"prediction failed: {e}"}
            predicted_pdb = Path(tmp_dir) / "predicted.pdb"
            predicted_pdb.write_text(pdb_str)
        predicted_pdb = Path(predicted_pdb)

        # 2) compare predicted vs target via the fixed OST runner
        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner
            runner = get_ost_runner()
            if not runner.is_available():
                return {"status": "unavailable", "error": "OpenStructure not available"}
            raw = runner.run_compare_structures(
                predicted_pdb, target_pdb, metrics=["lddt", "rigid-scores", "tm-score"],
            )
        except Exception as e:  # noqa: BLE001
            return {"status": "unavailable", "error": f"OST compare failed: {e}"}

        if not isinstance(raw, dict) or raw.get("status") not in (None, "SUCCESS"):
            return {"status": "error", "error": (raw or {}).get("exception", "compare failed")}

        sc_tm = raw.get("tm_score")
        sc_rmsd = raw.get("rmsd")
        sc_lddt = raw.get("lddt")
        passed = (
            isinstance(sc_tm, (int, float)) and sc_tm >= SC_TM_PASS
            and (not isinstance(sc_rmsd, (int, float)) or sc_rmsd <= SC_RMSD_PASS)
        )
        return {
            "status": "SUCCESS",
            "sc_tm": sc_tm,
            "sc_rmsd": sc_rmsd,
            "sc_lddt": sc_lddt,
            "passed": bool(passed),
            "n_residues": len(designed_sequence),
            "thresholds": {"sc_tm>=": SC_TM_PASS, "sc_rmsd<=": SC_RMSD_PASS},
        }
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def verdict(result: Dict) -> str:
    """One-line human verdict from a self_consistency result."""
    if result.get("status") != "SUCCESS":
        return f"self-consistency unavailable ({result.get('error', result.get('status'))})"
    tm, rmsd = result.get("sc_tm"), result.get("sc_rmsd")
    tag = "🟢 self-consistent (designable)" if result.get("passed") else "🔴 NOT self-consistent"
    return f"{tag} — scTM={tm:.3f}" + (f", scRMSD={rmsd:.2f} Å" if isinstance(rmsd, (int, float)) else "")
