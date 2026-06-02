"""ESM-2 zero-shot variant-effect prediction (masked-marginal, Meier 2021).

Implements the protocol used in the BIOS6380 teaching notebook: mask the variant
position, take the log-softmax over the amino-acid vocabulary at that position,
and report the substitution score as the change in log-likelihood::

    Δ = log P(mutant | context) − log P(wild-type | context)

Negative Δ = the mutation is less likely than wild-type under the protein
language model (evolutionarily deleterious); Δ near 0 / positive = tolerated.
This is a *fitness / conservation* signal, NOT a stability (ΔΔG) measurement —
it is kept distinct from the FoldX/heuristic stability term in the composite
mutation score.

A single masked forward pass yields the full 20-AA distribution at a position,
so a saturation scan costs N forward passes (one per position), not 19·N.

Model and device come from ``core.config``; default ``esm2_t33_650M_UR50D``
(650M) on auto device (GPU when there is free VRAM, else CPU). The model is
loaded lazily and cached as a process singleton; per-position log-prob vectors
are cached so repeated/saturation lookups are free.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

_AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Δ log-likelihood verdict thresholds (from the BIOS6380 notebook).
_VERDICTS = (
    (-7.5, "strongly_deleterious"),
    (-5.0, "likely_deleterious"),
    (-2.0, "ambiguous"),
)

# ESM-2 positional context limit (incl. BOS/EOS); window long sequences.
_MAX_TOKENS = 1022
_WINDOW = 1022  # residues kept around the variant for long sequences


def verdict_for_delta(delta: float) -> str:
    """Map a Δ log-likelihood to a categorical verdict."""
    for cutoff, label in _VERDICTS:
        if delta < cutoff:
            return label
    return "tolerated"


class ESM2VariantScorer:
    """Masked-marginal ESM-2 scorer with a cached model and per-position cache."""

    def __init__(self, model_name: Optional[str] = None, device: str = "auto"):
        self.model_name = model_name or "esm2_t33_650M_UR50D"
        self.device_pref = device
        self._model = None
        self._alphabet = None
        self._bc = None
        self._device = None
        self._logp_cache: Dict[str, Dict[str, float]] = {}

    # ── availability ───────────────────────────────────────────────────────
    @staticmethod
    def is_available() -> bool:
        try:
            import esm  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def get_requirements(self) -> str:
        return "pip install fair-esm torch  (ESM-2 zero-shot variant effect)"

    # ── lazy load ──────────────────────────────────────────────────────────
    def _resolve_device(self) -> str:
        import torch
        pref = (self.device_pref or "auto").lower()
        if pref == "cpu" or not torch.cuda.is_available():
            return "cpu"
        if pref == "cuda":
            return "cuda"
        # auto: use GPU only if there is comfortable free VRAM (≈ model + acts).
        try:
            free, _ = torch.cuda.mem_get_info()
            need = 3.5e9 if "650M" in self.model_name else (1.0e9 if "150M" in self.model_name else 0.3e9)
            return "cuda" if free > need else "cpu"
        except Exception:
            return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import esm
        import torch
        loader = getattr(esm.pretrained, self.model_name)
        model, alphabet = loader()
        model.eval()
        self._device = self._resolve_device()
        model = model.to(self._device)
        self._model = model
        self._alphabet = alphabet
        self._bc = alphabet.get_batch_converter()

    # ── core: masked log-probs at one position ───────────────────────────────
    def _window(self, sequence: str, pos0: int):
        """Return (windowed_seq, pos0_in_window) so long seqs fit the context."""
        if len(sequence) <= _MAX_TOKENS:
            return sequence, pos0
        half = _WINDOW // 2
        start = max(0, min(pos0 - half, len(sequence) - _WINDOW))
        return sequence[start:start + _WINDOW], pos0 - start

    def _logprobs_at(self, sequence: str, pos0: int) -> Dict[str, float]:
        """Masked-marginal log-prob for all 20 AAs at 0-indexed ``pos0``."""
        key = f"{self.model_name}:{hashlib.md5(sequence.encode()).hexdigest()}:{pos0}"
        if key in self._logp_cache:
            return self._logp_cache[key]

        import torch
        self._ensure_loaded()
        win_seq, win_pos = self._window(sequence, pos0)
        _, _, tokens = self._bc([("seq", win_seq)])
        tokens = tokens.to(self._device)
        mask_pos = win_pos + 1  # +1 for the prepended BOS token
        tokens[0, mask_pos] = self._alphabet.mask_idx
        with torch.no_grad():
            logits = self._model(tokens)["logits"]
            logp = torch.log_softmax(logits, dim=-1)[0, mask_pos]
        out = {aa: float(logp[self._alphabet.get_idx(aa)]) for aa in _AA20}
        self._logp_cache[key] = out
        return out

    # ── public API ───────────────────────────────────────────────────────────
    def score_variant(self, sequence: str, position: int, wt: str, mt: str) -> Dict:
        """Score a single substitution. ``position`` is 1-indexed.

        Returns ``{esm2_ll_wt, esm2_ll_mt, esm2_delta_ll, verdict}``.
        """
        pos0 = position - 1
        wt = wt.upper()
        mt = mt.upper()
        logp = self._logprobs_at(sequence, pos0)
        ll_wt = logp.get(wt)
        ll_mt = logp.get(mt)
        if ll_wt is None or ll_mt is None:
            return {"esm2_ll_wt": ll_wt, "esm2_ll_mt": ll_mt, "esm2_delta_ll": None, "verdict": "n/a"}
        delta = ll_mt - ll_wt
        return {
            "esm2_ll_wt": round(ll_wt, 4),
            "esm2_ll_mt": round(ll_mt, 4),
            "esm2_delta_ll": round(delta, 4),
            "verdict": verdict_for_delta(delta),
        }

    def score_position(self, sequence: str, position: int) -> Dict[str, float]:
        """All-19-substitution Δ log-likelihood at 1-indexed ``position`` (one fwd pass).

        Returns ``{mutant_aa: delta_ll}`` (wild-type maps to 0.0).
        """
        pos0 = position - 1
        wt = sequence[pos0].upper()
        logp = self._logprobs_at(sequence, pos0)
        ll_wt = logp[wt]
        return {aa: round(logp[aa] - ll_wt, 4) for aa in _AA20}

    def score_saturation(self, sequence: str, positions: List[int]) -> Dict[int, Dict[str, float]]:
        """Δ log-likelihood matrix for each requested 1-indexed position."""
        return {p: self.score_position(sequence, p) for p in positions
                if 1 <= p <= len(sequence)}

    def conservation(self, sequence: str, position: int) -> float:
        """WT log-prob at a position — higher = WT more strongly preferred (conserved)."""
        pos0 = position - 1
        return self._logprobs_at(sequence, pos0)[sequence[pos0].upper()]


# ── process-wide singleton (model load is expensive) ───────────────────────
_scorer: Optional[ESM2VariantScorer] = None
_scorer_key: str = ""


def get_esm2_scorer(model_name: Optional[str] = None, device: Optional[str] = None) -> ESM2VariantScorer:
    """Return a cached ESM2VariantScorer; rebuilt only when model/device change.

    Defaults: ``esm2_t33_650M_UR50D`` on auto device. Override per-process with
    env vars ``PDHUB_ESM2_MODEL`` / ``PDHUB_ESM2_DEVICE``.
    """
    global _scorer, _scorer_key
    import os
    name = model_name or os.environ.get("PDHUB_ESM2_MODEL") or "esm2_t33_650M_UR50D"
    device = device or os.environ.get("PDHUB_ESM2_DEVICE") or "auto"
    key = f"{name}|{device}"
    if _scorer is None or _scorer_key != key:
        _scorer = ESM2VariantScorer(name, device)
        _scorer_key = key
    return _scorer
