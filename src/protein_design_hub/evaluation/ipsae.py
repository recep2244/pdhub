"""ipSAE — interprotein Score from Aligned Errors.

Implements the ipSAE interface-confidence score (Dunbrack lab,
https://www.biorxiv.org/content/10.1101/2025.02.10.637595), which ranks
predicted protein–protein interfaces ~1.4x more precisely than ipTM/iPAE for
designed binders. Operates on a predicted-aligned-error (PAE) matrix plus chain
boundaries, so it works for any predictor that emits PAE (Chai-1, Boltz, AF2/3);
ESMFold has no PAE and is reported as unavailable.

Supporting metrics: pDockQ (Bryant 2022) and LIS (Local Interaction Score).

Recommended thresholds (from the ipsae skill):
    ipSAE_min > 0.61 (standard) / > 0.70 (stringent)   — primary filter
    LIS       > 0.35 / > 0.45                            — interface quality
    pDockQ    > 0.50 / > 0.60                            — supporting
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Thresholds (skill: protein-qc / ipsae)
IPSAE_PASS = 0.61
IPSAE_STRINGENT = 0.70
LIS_PASS = 0.35
PDOCKQ_PASS = 0.50

# pDockQ logistic constants (Bryant et al. 2022, Nat. Commun.)
_PDOCKQ_L, _PDOCKQ_X0, _PDOCKQ_K, _PDOCKQ_B = 0.724, 152.611, 0.052, 0.018


def _ptm_d0(n: int) -> float:
    """AlphaFold pTM distance scale d0 for *n* aligned residues (clamped > 0)."""
    n_eff = max(int(n), 19)
    return max(1.0, 1.24 * (n_eff - 15) ** (1.0 / 3.0) - 1.8)


def _directional_ipsae(pae: np.ndarray, idx_a: Sequence[int], idx_b: Sequence[int],
                       pae_cutoff: float) -> float:
    """ipSAE for chain A attending to chain B (asymmetric).

    For each residue i in A, average a pTM-style transform of the inter-chain
    PAE over partners j in B with PAE(i, j) < cutoff (d0 set by the per-residue
    contact count), then take the best-scoring residue.
    """
    sub = pae[np.ix_(list(idx_a), list(idx_b))]
    best = 0.0
    for row in sub:
        valid = row < pae_cutoff
        n0 = int(valid.sum())
        if n0 == 0:
            continue
        d0 = _ptm_d0(n0)
        ptm = 1.0 / (1.0 + (row[valid] / d0) ** 2)
        best = max(best, float(ptm.mean()))
    return best


def _lis(pae: np.ndarray, idx_a: Sequence[int], idx_b: Sequence[int],
         cutoff: float = 12.0) -> float:
    """Local Interaction Score: mean of (cutoff-PAE)/cutoff over inter-chain
    pairs with PAE < cutoff, both directions."""
    ab = pae[np.ix_(list(idx_a), list(idx_b))].ravel()
    ba = pae[np.ix_(list(idx_b), list(idx_a))].ravel()
    both = np.concatenate([ab, ba])
    mask = both < cutoff
    if not mask.any():
        return 0.0
    return float(((cutoff - both[mask]) / cutoff).mean())


def _pdockq(plddt: np.ndarray, coords: np.ndarray, idx_a: Sequence[int],
            idx_b: Sequence[int], contact_dist: float = 8.0) -> Optional[float]:
    """pDockQ from interface CA contacts and mean interface pLDDT.

    *plddt* in 0–100 or 0–1; *coords* are CA xyz (N, 3). Returns None if there
    are no inter-chain CA contacts within *contact_dist*.
    """
    ca_a = coords[list(idx_a)]
    ca_b = coords[list(idx_b)]
    # pairwise CA-CA distances
    d = np.linalg.norm(ca_a[:, None, :] - ca_b[None, :, :], axis=-1)
    contacts = d < contact_dist
    n_contacts = int(contacts.sum())
    if n_contacts == 0:
        return None
    a_if = np.unique(np.where(contacts.any(axis=1))[0])
    b_if = np.unique(np.where(contacts.any(axis=0))[0])
    pl = np.asarray(plddt, dtype=float)
    if pl.max() > 1.5:                       # normalise 0–100 → 0–1
        pl = pl / 100.0
    if_plddt = np.concatenate([pl[list(idx_a)][a_if], pl[list(idx_b)][b_if]]).mean()
    x = if_plddt * np.log(max(n_contacts, 1))
    return float(_PDOCKQ_L / (1 + np.exp(-_PDOCKQ_K * (x - _PDOCKQ_X0))) + _PDOCKQ_B)


def compute_ipsae(
    pae: np.ndarray,
    chain_lengths: Sequence[int],
    pae_cutoff: float = 10.0,
    plddt: Optional[Sequence[float]] = None,
    ca_coords: Optional[np.ndarray] = None,
    predictor_name: Optional[str] = None,
) -> Dict:
    """Compute ipSAE (+ pDockQ, LIS) for every chain pair in a predicted complex.

    Args:
        pae: (N, N) predicted aligned error matrix in Å.
        chain_lengths: residues per chain, summing to N (e.g. [120, 95]).
        pae_cutoff: PAE threshold for inter-chain contacts (10–15 Å typical).
        plddt: optional per-residue pLDDT (length N) for pDockQ.
        ca_coords: optional CA coordinates (N, 3) for pDockQ.
        predictor_name: optional origin engine. When given, ESMFold (and any
            other non-PAE predictor) is rejected with status ``"unavailable"``
            *before* scoring — ipSAE is meaningless without genuine PAE. When
            ``None`` (the default), no engine check is performed and the SUCCESS
            path is identical to before — callers that already hold a real PAE
            matrix are unaffected.

    Returns dict with status, ipsae_min (worst pair = the binding-relevant one),
    best pair, per-pair table, and a pass flag.
    """
    # ── interface-capability gate (opt-in via predictor_name) ──────────────
    if predictor_name is not None:
        from protein_design_hub.evaluation.interface_guard import (
            InterfaceCapabilityError,
            assert_interface_capable,
        )
        try:
            assert_interface_capable(predictor_name)
        except InterfaceCapabilityError as exc:
            return {"status": "unavailable", "error": str(exc),
                    "predictor": predictor_name}

    pae = np.asarray(pae, dtype=float)
    n = pae.shape[0]
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        return {"status": "error", "error": "PAE must be a square matrix"}
    if sum(chain_lengths) != n:
        return {"status": "error",
                "error": f"chain_lengths sum {sum(chain_lengths)} != PAE dim {n}"}
    if len(chain_lengths) < 2:
        return {"status": "unavailable", "error": "single chain — ipSAE needs ≥2 chains"}

    # chain residue-index ranges
    bounds, start = [], 0
    for L in chain_lengths:
        bounds.append(range(start, start + L))
        start += L
    coords = np.asarray(ca_coords, dtype=float) if ca_coords is not None else None

    pairs: List[Dict] = []
    for a in range(len(bounds)):
        for b in range(a + 1, len(bounds)):
            ia, ib = bounds[a], bounds[b]
            d_ab = _directional_ipsae(pae, ia, ib, pae_cutoff)
            d_ba = _directional_ipsae(pae, ib, ia, pae_cutoff)
            entry = {
                "chain_a": chr(65 + a), "chain_b": chr(65 + b),
                "ipsae": round(min(d_ab, d_ba), 4),
                "ipsae_asym": [round(d_ab, 4), round(d_ba, 4)],
                "lis": round(_lis(pae, ia, ib), 4),
            }
            if coords is not None and plddt is not None:
                pq = _pdockq(np.asarray(plddt, float), coords, ia, ib)
                entry["pdockq"] = round(pq, 4) if pq is not None else None
            pairs.append(entry)

    ipsae_min = min(p["ipsae"] for p in pairs)
    best = max(pairs, key=lambda p: p["ipsae"])
    return {
        "status": "SUCCESS",
        "ipsae_min": ipsae_min,
        "best_pair": f'{best["chain_a"]}-{best["chain_b"]}',
        "best_ipsae": best["ipsae"],
        "lis": best.get("lis"),
        "pdockq": best.get("pdockq"),
        "pairs": pairs,
        "pae_cutoff": pae_cutoff,
        "passed": bool(ipsae_min >= IPSAE_PASS),
    }


def load_pae_from_dir(pred_dir: Path) -> Optional[np.ndarray]:
    """Best-effort load of a PAE matrix from a prediction output directory.

    Recognises Boltz/Chai ``*.npz`` (key ``pae``/``predicted_aligned_error``),
    raw ``*.npy``, and AF-style JSON (``pae``/``predicted_aligned_error``).
    Returns None if no PAE artifact is found.
    """
    pred_dir = Path(pred_dir)
    if not pred_dir.exists():
        return None
    for npz in sorted(pred_dir.rglob("*pae*.npz")) + sorted(pred_dir.rglob("*.npz")):
        try:
            data = np.load(npz)
            for key in ("pae", "predicted_aligned_error", "pae_chain", "arr_0"):
                if key in data:
                    return np.asarray(data[key], dtype=float)
        except Exception:
            continue
    for npy in sorted(pred_dir.rglob("*pae*.npy")):
        try:
            return np.asarray(np.load(npy), dtype=float)
        except Exception:
            continue
    import json
    for js in sorted(pred_dir.rglob("*.json")):
        try:
            obj = json.loads(js.read_text())
            obj = obj[0] if isinstance(obj, list) and obj else obj
            for key in ("pae", "predicted_aligned_error"):
                if isinstance(obj, dict) and key in obj:
                    return np.asarray(obj[key], dtype=float)
        except Exception:
            continue
    return None


def verdict(result: Dict) -> str:
    """One-line human verdict from a compute_ipsae result."""
    if result.get("status") == "unavailable":
        return "⚪ ipSAE unavailable (needs a ≥2-chain prediction with PAE — e.g. Chai/Boltz)"
    if result.get("status") != "SUCCESS":
        return f"⚪ ipSAE error: {result.get('error', 'unknown')}"
    v = result["ipsae_min"]
    if v >= IPSAE_STRINGENT:
        tier = "🟢 high-confidence interface"
    elif v >= IPSAE_PASS:
        tier = "🟢 likely interface"
    elif v >= 0.4:
        tier = "🟡 weak / uncertain interface"
    else:
        tier = "🔴 no credible interface"
    extra = f", LIS {result['lis']:.2f}" if result.get("lis") is not None else ""
    pq = f", pDockQ {result['pdockq']:.2f}" if result.get("pdockq") is not None else ""
    return f"{tier} — ipSAE_min {v:.2f}{extra}{pq} (pass ≥ {IPSAE_PASS})"
