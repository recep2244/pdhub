"""Interface-capability guard for predictor engines.

The blueprint draws a hard line between predictors that can support *interface*
decisions (binding/complex confidence) and those that cannot. Interface scores
such as ipSAE, pDockQ and LIS are all derived from a predicted-aligned-error
(PAE) matrix. ESMFold is a single-sequence folder that emits pLDDT only — it has
no PAE and therefore *cannot* be used for any interface decision. Chai-1, Boltz
and AlphaFold (2/3, incl. Multimer) all emit PAE and are interface-capable.

This module centralises that policy so every interface code path (ipSAE, future
docking metrics, …) rejects ESMFold-origin inputs the same way, with a clear,
testable status — rather than silently producing a meaningless score.
"""

from __future__ import annotations

from typing import Set

# Engines that emit a PAE matrix and may drive interface decisions.
INTERFACE_CAPABLE_ENGINES: Set[str] = {
    "chai1",
    "chai",
    "boltz1",
    "boltz2",
    "boltz",
    "alphafold",
    "alphafold2",
    "alphafold3",
    "af2",
    "af3",
    "alphafold-multimer",
    "af2-multimer",
}

# Engines that are explicitly *not* interface-capable (no PAE). Kept distinct
# from "unknown" so callers can give a precise reason for the rejection.
NON_INTERFACE_ENGINES: Set[str] = {
    "esmfold",
    "esm",
    "esm2",
    "omegafold",  # single-sequence; no PAE
}


class InterfaceCapabilityError(ValueError):
    """Raised when an interface decision is attempted on a non-PAE predictor."""


def _normalize(predictor_name: str) -> str:
    """Lower-case, strip, and collapse separators so ``ESMFold``/``esm-fold``
    /``ESM_Fold`` all map to the same canonical key."""
    return (predictor_name or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")


# Pre-normalise the lookup sets once.
_CAPABLE = {_normalize(e) for e in INTERFACE_CAPABLE_ENGINES}
_NON = {_normalize(e) for e in NON_INTERFACE_ENGINES}


def is_interface_capable(predictor_name: str) -> bool:
    """Return ``True`` iff *predictor_name* emits PAE and may drive interface
    decisions (Chai/Boltz/AlphaFold), ``False`` otherwise (ESMFold, unknown).

    Unknown engines return ``False`` — interface decisions are opt-in to a known
    PAE-emitting predictor, never assumed.
    """
    return _normalize(predictor_name) in _CAPABLE


def assert_interface_capable(predictor_name: str) -> None:
    """Raise :class:`InterfaceCapabilityError` if *predictor_name* cannot support
    an interface decision (no PAE). No-op for capable engines.
    """
    key = _normalize(predictor_name)
    if key in _CAPABLE:
        return
    if key in _NON:
        raise InterfaceCapabilityError(
            f"{predictor_name!r} emits no PAE (single-sequence predictor) — it "
            f"cannot be used for interface decisions. Use Chai-1, Boltz, or "
            f"AlphaFold for complex/binding confidence."
        )
    raise InterfaceCapabilityError(
        f"{predictor_name!r} is not a known interface-capable predictor. "
        f"Interface decisions require a PAE-emitting engine "
        f"({', '.join(sorted(INTERFACE_CAPABLE_ENGINES))})."
    )
