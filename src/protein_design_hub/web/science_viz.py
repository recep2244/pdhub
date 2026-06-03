"""Canonical scientific-visualisation helpers for Protein Design Hub.

This is the single source of truth for how the web app encodes scientific
meaning in colour and small HTML widgets:

* The **AlphaFold pLDDT confidence ramp** (blue -> cyan -> yellow -> orange),
  used *fill-only* (never as text colour on a dark background, where the
  yellow/orange bands fail contrast).
* A **colour-vision-deficiency (CVD) safe property-class palette** for the 20
  amino acids, grouped by physicochemical class.
* A **diverging effect scale** for mutation effects (purple stabilising ->
  neutral grey -> orange destabilising) that is *deliberately distinct* from
  the confidence ramp so the two encodings are never confused.

It also provides engine-aware helpers (``pae_domain``) and pure-string HTML
builders (``takeaway_banner``, ``campaign_funnel``).  Nothing here imports or
calls Streamlit, so every function is unit-testable in a headless CI without
heavy dependencies.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

__all__ = [
    "PLDDT_RAMP",
    "PLDDT_RAMP_NOTE",
    "PROPERTY_PALETTE",
    "AA_PROPERTY_CLASS",
    "EFFECT_SCALE",
    "CONFIDENCE_BANDS",
    "confidence_color",
    "confidence_band",
    "property_color",
    "effect_color",
    "pae_domain",
    "takeaway_banner",
    "campaign_funnel",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: AlphaFold / ColabFold pLDDT confidence ramp, ordered low -> high confidence.
#: Each entry is ``(lower_bound_inclusive, hex)``.  These four anchor colours
#: are the canonical AlphaFold ramp.
PLDDT_RAMP: Tuple[Tuple[float, str], ...] = (
    (0.0, "#ff7d45"),   # very low   (pLDDT < 50)   orange
    (50.0, "#ffdb13"),  # low        (50 <= pLDDT < 70) yellow
    (70.0, "#65cbf3"),  # confident  (70 <= pLDDT < 90) cyan
    (90.0, "#0053d6"),  # very high  (pLDDT >= 90)   blue
)

#: Reminder that the ramp colours are tuned for structure *fill* only.  The
#: yellow (#ffdb13) and orange (#ff7d45) bands do not meet WCAG text contrast
#: against the dark UI, so they must never be used as text colour.
PLDDT_RAMP_NOTE: str = (
    "pLDDT ramp colours are for structure fill only. Yellow (#ffdb13) and "
    "orange (#ff7d45) fail text contrast on the dark UI; use a band label "
    "instead of colouring text with these values."
)

#: Confidence band thresholds (lower bound inclusive), high -> low.
CONFIDENCE_BANDS: Tuple[Tuple[float, str], ...] = (
    (90.0, "very_high"),
    (70.0, "high"),
    (50.0, "low"),
    (0.0, "very_low"),
)

#: Physicochemical class for each amino acid (one-letter code).
AA_PROPERTY_CLASS: Dict[str, str] = {
    # hydrophobic / aliphatic
    "A": "hydrophobic", "V": "hydrophobic", "L": "hydrophobic",
    "I": "hydrophobic", "M": "hydrophobic",
    # aromatic
    "F": "aromatic", "W": "aromatic", "Y": "aromatic",
    # polar uncharged
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar", "C": "polar",
    # positive (basic)
    "K": "positive", "R": "positive", "H": "positive",
    # negative (acidic)
    "D": "negative", "E": "negative",
    # special / conformational
    "G": "special", "P": "special",
}

#: CVD-safe palette keyed by property class.  Hues chosen to remain
#: distinguishable under deuteranopia/protanopia (Okabe-Ito derived, plus a
#: neutral grey for the special class).
PROPERTY_PALETTE: Dict[str, str] = {
    "hydrophobic": "#e69f00",  # orange
    "aromatic": "#cc79a7",     # reddish purple
    "polar": "#56b4e9",        # sky blue
    "positive": "#0072b2",     # deep blue
    "negative": "#d55e00",     # vermillion
    "special": "#999999",      # neutral grey
    "unknown": "#444444",      # fallback for X/gap/unknown
}

#: Diverging effect scale for mutation effects.  Purple = stabilising
#: (favourable, negative ddG), neutral grey at zero, orange = destabilising
#: (unfavourable, positive ddG).  Ordered most-stabilising -> most-destabilising.
#: Each entry is ``(upper_bound_exclusive, hex)`` keyed on a ddG-like value in
#: kcal/mol; the final ``inf`` bucket catches everything above.
EFFECT_SCALE: Tuple[Tuple[float, str], ...] = (
    (-2.0, "#5e3c99"),  # strongly stabilising  (deep purple)
    (-0.5, "#b2abd8"),  # mildly stabilising    (light purple)
    (0.5, "#bdbdbd"),   # neutral               (grey)
    (2.0, "#fdb863"),   # mildly destabilising  (light orange)
    (float("inf"), "#e66101"),  # strongly destabilising (deep orange)
)


# ---------------------------------------------------------------------------
# Confidence encoding
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def confidence_color(value: float, kind: str = "plddt") -> str:
    """Return the canonical fill colour (hex) for a confidence value.

    Args:
        value: The confidence metric value.
        kind: One of ``"plddt"``, ``"iptm"`` or ``"pae"``.

            * ``"plddt"`` expects 0-100 (higher is better).
            * ``"iptm"`` expects 0-1 (higher is better); scaled to 0-100.
            * ``"pae"`` expects Angstroms (lower is better); inverted onto the
              0-100 confidence scale assuming a 30 A worst case.

    Returns:
        A hex colour string drawn from :data:`PLDDT_RAMP`.
    """
    k = kind.lower()
    if k == "plddt":
        scaled = _clamp(float(value), 0.0, 100.0)
    elif k == "iptm":
        scaled = _clamp(float(value), 0.0, 1.0) * 100.0
    elif k == "pae":
        # Lower PAE is better; map 0 A -> 100 (best), 30 A -> 0 (worst).
        scaled = _clamp(100.0 - (float(value) / 30.0) * 100.0, 0.0, 100.0)
    else:
        raise ValueError(f"Unknown confidence kind: {kind!r}")

    color = PLDDT_RAMP[0][1]
    for lower, hex_color in PLDDT_RAMP:
        if scaled >= lower:
            color = hex_color
    return color


def confidence_band(value: float) -> str:
    """Map a 0-100 pLDDT-like value to a band label.

    Returns one of ``"very_high"`` (>=90), ``"high"`` (>=70), ``"low"``
    (>=50) or ``"very_low"`` (<50).
    """
    v = float(value)
    for lower, label in CONFIDENCE_BANDS:
        if v >= lower:
            return label
    return CONFIDENCE_BANDS[-1][1]


# ---------------------------------------------------------------------------
# Amino-acid property colour
# ---------------------------------------------------------------------------

def property_color(aa: str) -> str:
    """Return the CVD-safe property-class colour for an amino acid.

    Accepts one- or three-letter codes (case-insensitive). Unknown residues,
    gaps and ``X`` fall back to the ``"unknown"`` colour.
    """
    if not aa:
        return PROPERTY_PALETTE["unknown"]
    code = aa.strip().upper()
    if len(code) == 3:
        code = _THREE_TO_ONE.get(code, "X")
    one = code[:1]
    cls = AA_PROPERTY_CLASS.get(one, "unknown")
    return PROPERTY_PALETTE.get(cls, PROPERTY_PALETTE["unknown"])


_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ---------------------------------------------------------------------------
# Mutation effect colour
# ---------------------------------------------------------------------------

def effect_color(ddg: float) -> str:
    """Return the diverging effect colour (hex) for a ddG-like value.

    Negative ddG (stabilising) maps to purple, near-zero to grey, positive
    (destabilising) to orange. See :data:`EFFECT_SCALE`.
    """
    v = float(ddg)
    for upper, hex_color in EFFECT_SCALE:
        if v < upper:
            return hex_color
    return EFFECT_SCALE[-1][1]


# ---------------------------------------------------------------------------
# Engine-aware PAE domain
# ---------------------------------------------------------------------------

def pae_domain(engine: Union[str, float, int, None] = None,
               max_error: Union[float, None] = None) -> Tuple[float, float]:
    """Return the ``(min, max)`` PAE colour-scale domain for an engine.

    Different predictors emit PAE on different maxima (AlphaFold caps at
    ~31.75 A, Boltz/Chai commonly report up to 32 A, ESMFold ~30 A).  Passing
    an explicit ``max_error`` overrides the engine lookup; passing a numeric
    ``engine`` is treated as ``max_error`` for convenience.

    Args:
        engine: Predictor name (case-insensitive) or a numeric max error.
        max_error: Explicit upper bound in Angstroms.

    Returns:
        ``(0.0, upper)`` domain tuple.
    """
    if max_error is not None:
        return (0.0, float(max_error))
    if isinstance(engine, (int, float)):
        return (0.0, float(engine))

    name = (engine or "").strip().lower()
    table = {
        "alphafold": 31.75,
        "alphafold2": 31.75,
        "af2": 31.75,
        "af3": 31.75,
        "colabfold": 31.75,
        "boltz": 32.0,
        "boltz1": 32.0,
        "boltz2": 32.0,
        "chai": 32.0,
        "chai1": 32.0,
        "esmfold": 30.0,
        "esm": 30.0,
    }
    return (0.0, table.get(name, 30.0))


# ---------------------------------------------------------------------------
# Pure-HTML builders (no Streamlit)
# ---------------------------------------------------------------------------

#: Accent/background colours per banner level (CVD-safe, distinct from ramp).
_BANNER_LEVELS: Dict[str, Tuple[str, str]] = {
    "success": ("#3fe0c5", "rgba(63, 224, 197, 0.10)"),
    "info": ("#56b4e9", "rgba(86, 180, 233, 0.10)"),
    "warning": ("#e69f00", "rgba(230, 159, 0, 0.10)"),
    "danger": ("#e66101", "rgba(230, 97, 1, 0.10)"),
}


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def takeaway_banner(text: str, level: str = "info") -> str:
    """Build a single-line "key takeaway" banner as an HTML string.

    Args:
        text: The takeaway message (escaped before insertion).
        level: One of ``"success"``, ``"info"``, ``"warning"``, ``"danger"``.

    Returns:
        A non-empty HTML ``<div>`` string. Never calls Streamlit.
    """
    accent, bg = _BANNER_LEVELS.get(level.lower(), _BANNER_LEVELS["info"])
    safe = _escape(text)
    return (
        f'<div class="pdhub-takeaway pdhub-takeaway--{_escape(level.lower())}" '
        f'style="border-left:4px solid {accent};background:{bg};'
        f'padding:0.6rem 0.9rem;border-radius:6px;margin:0.5rem 0;'
        f'font-family:\'Hanken Grotesk\',sans-serif;color:#e6edf3;">'
        f'<span style="color:{accent};font-weight:600;'
        f'text-transform:uppercase;font-size:0.72rem;letter-spacing:0.06em;'
        f'margin-right:0.5rem;">{_escape(level.lower())}</span>'
        f'<span>{safe}</span>'
        f'</div>'
    )


def campaign_funnel(stages: Sequence[Dict[str, object]]) -> str:
    """Build an HTML funnel summarising a design campaign's stages.

    Args:
        stages: Ordered sequence of stage dicts. Recognised keys:

            * ``"label"`` / ``"name"`` — stage name (required-ish).
            * ``"count"`` / ``"n"`` — number of designs at this stage.
            * ``"color"`` — optional bar accent colour (hex).

    Returns:
        A non-empty HTML string. Empty input still yields a valid wrapper.
    """
    stages = list(stages or [])
    counts: List[float] = []
    for s in stages:
        raw = s.get("count", s.get("n", 0))
        try:
            counts.append(float(raw))
        except (TypeError, ValueError):
            counts.append(0.0)
    top = max(counts) if counts else 0.0

    rows: List[str] = []
    default_accent = "#3fe0c5"
    for stage, count in zip(stages, counts):
        label = _escape(stage.get("label", stage.get("name", "stage")))
        accent = _escape(str(stage.get("color", default_accent)))
        pct = (count / top * 100.0) if top > 0 else 0.0
        retained = ""
        if top > 0:
            retained = f"{count / top * 100.0:.0f}% of input"
        rows.append(
            f'<div class="pdhub-funnel__row" style="margin:0.35rem 0;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;'
            f'color:#c9d1d9;margin-bottom:0.15rem;">'
            f'<span>{label}</span>'
            f'<span>{count:g} <span style="color:#8b949e;">{retained}</span></span>'
            f'</div>'
            f'<div style="background:rgba(255,255,255,0.06);border-radius:4px;'
            f'height:14px;overflow:hidden;">'
            f'<div style="width:{max(pct, 0.0):.1f}%;height:100%;'
            f'background:{accent};border-radius:4px;"></div>'
            f'</div>'
            f'</div>'
        )
    body = "".join(rows) if rows else (
        '<div style="color:#8b949e;font-size:0.8rem;">No campaign stages.</div>'
    )
    return (
        '<div class="pdhub-funnel" '
        'style="font-family:\'Hanken Grotesk\',sans-serif;'
        'padding:0.5rem 0;">' + body + '</div>'
    )
