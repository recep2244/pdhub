"""Rule-based developability gate — a deterministic GO / NO-GO decision for
ordering a sequence (gene synthesis / expression).

This combines the design-level liability detector and biophysical-property
calculator from :mod:`protein_design_hub.analysis.protein_qc` into a single,
deterministic gate that produces a hard, orderable boolean:

* **Hard blockers** (NO-GO): free thiol (odd cysteine count), forbidden /
  invalid residues, an empty / non-AA sequence, or an *egregious* biophysical
  value (far outside the tiered bounds).
* **Warnings** (still orderable): instability, GRAVY, or pI outside the
  preferred band but not catastrophic, plus the sequence-liability flags.

The gate is intentionally rule-based and dependency-free (BioPython is used by
``biophysical_properties`` only when present; absence merely degrades the
instability index to ``None``). It performs a back-translation sanity check —
not an actual codon back-translation, but the equivalent guarantee that every
residue is a valid, codon-encodable amino acid — so a flagged sequence can be
handed to a gene-synthesis vendor without surprises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .protein_qc import biophysical_properties, sequence_liabilities

# Canonical, codon-encodable amino acids (no ambiguity / non-standard codes).
_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Tiered biophysical bounds. ``warn`` brackets the preferred band (flag only);
# ``hard`` brackets the egregious band (auto NO-GO when crossed).
_INSTABILITY_WARN = 40.0   # Guruprasad: >40 predicted unstable
_INSTABILITY_HARD = 60.0   # grossly unstable → NO-GO
_GRAVY_WARN = 0.4          # |GRAVY| above this → solubility/aggregation flag
_GRAVY_HARD = 1.5          # extreme hydropathy (either sign) → NO-GO
_PI_WARN_LOW = 5.0         # pI inside [WARN_LOW, WARN_HIGH] is comfortable
_PI_WARN_HIGH = 9.0
_PI_HARD_LOW = 3.5         # pI outside [HARD_LOW, HARD_HIGH] → NO-GO
_PI_HARD_HIGH = 11.0


@dataclass
class DevReport:
    """Outcome of the developability gate.

    Attributes:
        orderable: Hard, deterministic GO / NO-GO boolean. ``True`` only when
            there are zero blockers.
        blockers: Reasons the sequence must not be ordered (NO-GO causes).
        warnings: Non-blocking concerns worth reviewing before ordering.
        properties: Merged biophysical properties + liability summary.
    """

    orderable: bool
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)


def _back_translation_ok(seq: str) -> List[str]:
    """Back-translation sanity check.

    Returns a list of blocker strings (empty when the sequence is a clean,
    fully codon-encodable amino-acid string). This guarantees every character
    maps to at least one codon, i.e. the sequence can be synthesised.
    """
    blockers: List[str] = []
    if not seq:
        blockers.append("Empty sequence — nothing to order")
        return blockers
    invalid = sorted({c for c in seq if c not in _VALID_AA})
    if invalid:
        blockers.append(
            "Forbidden/invalid residue(s) "
            f"({', '.join(invalid)}) — not codon-encodable"
        )
    return blockers


def developability_report(seq: str) -> DevReport:
    """Run the rule-based developability gate on a single amino-acid sequence.

    The decision is deterministic and order-independent:

    1. Back-translation sanity (valid, non-empty, codon-encodable AAs).
    2. Free thiol — an odd cysteine count is a hard NO-GO.
    3. Biophysical bounds — instability / GRAVY / pI inside the egregious band
       block; merely outside the preferred band only warn.
    4. Remaining sequence-liability flags become warnings.

    Args:
        seq: Single-letter amino-acid sequence (case-insensitive, whitespace
            tolerated).

    Returns:
        A :class:`DevReport`. ``orderable`` is ``True`` iff no blockers fired.
    """
    s = "".join((seq or "").upper().split())

    blockers: List[str] = []
    warnings: List[str] = []

    # 1. Back-translation / validity sanity check (also catches empty input).
    blockers.extend(_back_translation_ok(s))

    # Liabilities + biophysics are computed on whatever was supplied; the
    # underlying helpers already ignore non-AA characters defensively.
    liab = sequence_liabilities(s)
    biop = biophysical_properties(s)

    # 2. Free thiol — odd cysteine count is a hard blocker.
    if not liab.get("cysteine_even", True):
        blockers.append(
            f"Free thiol — odd cysteine count ({liab.get('cysteine_count')}); "
            "unpaired disulfide"
        )

    # 3. Tiered biophysical gating (only when validity passed and props exist).
    if not blockers:
        instab = biop.get("instability")
        if instab is not None:
            if instab > _INSTABILITY_HARD:
                blockers.append(
                    f"Instability index {instab} > {_INSTABILITY_HARD} — "
                    "grossly unstable"
                )
            elif instab > _INSTABILITY_WARN:
                warnings.append(
                    f"Instability index {instab} > {_INSTABILITY_WARN} — "
                    "predicted unstable"
                )

        gravy = biop.get("gravy")
        if gravy is not None:
            if abs(gravy) > _GRAVY_HARD:
                blockers.append(
                    f"GRAVY {gravy} (|GRAVY| > {_GRAVY_HARD}) — extreme "
                    "hydrophobicity"
                )
            elif abs(gravy) > _GRAVY_WARN:
                warnings.append(
                    f"GRAVY {gravy} (|GRAVY| > {_GRAVY_WARN}) — "
                    "solubility/aggregation concern"
                )

        pi = biop.get("pi")
        if pi is not None:
            if pi < _PI_HARD_LOW or pi > _PI_HARD_HIGH:
                blockers.append(
                    f"Theoretical pI {pi} outside "
                    f"[{_PI_HARD_LOW}, {_PI_HARD_HIGH}] — extreme charge"
                )
            elif pi < _PI_WARN_LOW or pi > _PI_WARN_HIGH:
                warnings.append(
                    f"Theoretical pI {pi} outside preferred "
                    f"[{_PI_WARN_LOW}, {_PI_WARN_HIGH}] band"
                )

    # 4. Remaining liability flags become warnings (free-thiol already handled
    #    above as a blocker, so drop its flag to avoid duplication).
    for flag in liab.get("flags", []):
        if "cysteine" in flag.lower():
            continue
        warnings.append(flag)

    properties: Dict = {**biop, "liabilities": liab}

    return DevReport(
        orderable=not blockers,
        blockers=blockers,
        warnings=warnings,
        properties=properties,
    )
