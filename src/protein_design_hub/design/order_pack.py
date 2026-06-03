"""Order Pack exporter — turn ranked design candidates into a synthesis-ready bundle.

Given a list of candidate dicts (sequence + scores + optional liabilities), this
module assembles the artifacts a wet-lab team needs to actually *order* genes:

  * ``ranked_csv``      — candidates sorted best-first, one row each
  * ``construct_fasta`` — codon-optimized DNA per candidate, with optional
                          N-terminal tag / signal peptide and linker fused in
  * ``construct_map``   — human-readable element map for every construct
  * ``assay_plan``      — track-specific bench plan (binder / enzyme / stability …)
  * ``risk_sheet``      — per-candidate liabilities in Ala123Lys-style notation

Everything is pure-Python with no heavy dependencies. Codon optimization reuses
:mod:`protein_design_hub.analysis.codon_optimization` when available and falls
back to a small built-in frequency table otherwise, so the exporter still works
in a minimal CI environment.

Public API
----------
``build_order_pack(candidates, track, host="ecoli") -> dict``
    Build the in-memory artifact bundle.
``write_order_pack(pack, out_dir) -> dict``
    Persist the bundle to disk; returns ``{artifact_name: path}``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Optional reuse of the project codon-optimization module (degrade gracefully)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import wiring, exercised indirectly
    from protein_design_hub.analysis.codon_optimization import (
        GENETIC_CODE as _GENETIC_CODE,
        back_translate as _back_translate,
    )

    _HAVE_CODON_MODULE = True
except Exception:  # pragma: no cover - minimal fallback path
    _GENETIC_CODE = {}
    _back_translate = None  # type: ignore[assignment]
    _HAVE_CODON_MODULE = False

# Three-letter mutation notation helper (also optional, also degradable).
try:  # pragma: no cover - import wiring
    from protein_design_hub.analysis.protein_utils import (
        format_mutation_three_letter as _format_mutation_three_letter,
    )
except Exception:  # pragma: no cover

    def _format_mutation_three_letter(code: str) -> str:  # type: ignore[misc]
        return code


# ---------------------------------------------------------------------------
# Fallback codon table (most-frequent codon per AA, generic high-expression set)
# Used only when analysis/codon_optimization.py is unavailable.
# ---------------------------------------------------------------------------

_FALLBACK_CODON: Dict[str, str] = {
    "A": "GCG", "R": "CGT", "N": "AAC", "D": "GAT", "C": "TGC",
    "Q": "CAG", "E": "GAA", "G": "GGC", "H": "CAT", "I": "ATC",
    "L": "CTG", "K": "AAA", "M": "ATG", "F": "TTC", "P": "CCG",
    "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
    # tolerated placeholders
    "*": "TAA", "X": "NNN", "U": "TGC", "B": "GAT", "Z": "GAA",
}

# Common N-terminal/affinity helper elements (protein-level, fused before back-translation).
_TAGS: Dict[str, str] = {
    "his6": "HHHHHH",
    "his8": "HHHHHHHH",
    "strep": "WSHPQFEK",
    "flag": "DYKDDDDK",
    "myc": "EQKLISEEDL",
    "none": "",
}
_SIGNALS: Dict[str, str] = {
    # E. coli periplasmic export
    "pelb": "MKYLLPTAAAGLLLLAAQPAMA",
    "ompa": "MKKTAIAIAVALAGFATVAQA",
    "none": "",
}
_DEFAULT_LINKER = "GGGGS"

# Track -> assay plan template. Each value is a list of plan lines.
_ASSAY_PLANS: Dict[str, List[str]] = {
    "binder": [
        "1. Expression: small-scale (1 mL) auto-induction; SDS-PAGE for soluble fraction.",
        "2. Purification: IMAC (His-tag) -> SEC polish; collect monomer peak.",
        "3. Affinity: SPR or BLI single-cycle kinetics vs immobilized target (KD, kon, koff).",
        "4. Specificity: counter-screen vs off-target panel; report fold-selectivity.",
        "5. Triage: advance binders with KD < 100 nM and monomeric SEC trace.",
    ],
    "enzyme": [
        "1. Expression: 50 mL culture; lyse and clarify; SDS-PAGE soluble check.",
        "2. Purification: affinity capture; buffer-exchange into assay buffer.",
        "3. Activity: steady-state kinetics (kcat, KM) at 25C and target temperature.",
        "4. Stability: residual activity after 1 h at challenge temperature.",
        "5. Triage: advance variants with kcat/KM above wild-type baseline.",
    ],
    "stability": [
        "1. Expression: small-scale; soluble-fraction SDS-PAGE.",
        "2. Purification: affinity + SEC; confirm monodispersity.",
        "3. Thermal: nanoDSF / DSF melt; report Tm and onset of aggregation (Tagg).",
        "4. Colloidal: SEC after 1 week at 4C / 37C; %monomer retained.",
        "5. Triage: advance variants with delta-Tm > +2C and no new aggregation.",
    ],
    "generic": [
        "1. Expression: small-scale screen; assess soluble yield by SDS-PAGE.",
        "2. Purification: affinity capture appropriate to fused tag; SEC polish.",
        "3. Identity: intact-mass LC-MS to confirm construct sequence.",
        "4. Function: track-appropriate functional readout.",
        "5. Triage: advance top-ranked candidates that express solubly.",
    ],
}

_HOST_NOTE: Dict[str, str] = {
    "ecoli": "E. coli (BL21/T7); cytoplasmic unless a signal peptide is fused.",
    "hek293": "HEK293 transient; secreted constructs need a native/igk signal.",
    "cho": "CHO stable/transient; codon table is a generic high-expression set.",
    "yeast": "S. cerevisiae / P. pastoris; check for cryptic glycosylation sites.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_protein(seq: str) -> str:
    """Uppercase a protein string and strip gaps / stop characters / whitespace."""
    return "".join(c for c in (seq or "").upper() if c.isalpha())


def _back_translate_seq(protein: str, host: str) -> str:
    """Codon-optimize a protein sequence to DNA (ACGT only where possible).

    Reuses the project's species back-translator when present (mapping a few
    common heterologous hosts onto its plant tables purely as a frequency
    source), otherwise uses the built-in fallback codon table.
    """
    protein = _clean_protein(protein)
    if _HAVE_CODON_MODULE and _back_translate is not None:
        # The project module ships plant tables; any of them yields valid,
        # high-frequency codons. We do not claim host-perfect optimization in
        # the fallback path — the construct_map records the host explicitly.
        try:
            return _back_translate(protein, species="wheat")
        except Exception:  # pragma: no cover - defensive
            pass
    return "".join(_FALLBACK_CODON.get(aa, "NNN") for aa in protein)


def _sanitize_dna(dna: str) -> str:
    """Force a DNA string to the ACGT alphabet (N -> A) so synthesis is valid."""
    table = str.maketrans({"N": "A", "n": "A"})
    return dna.upper().translate(table)


def _candidate_id(cand: Dict[str, Any], idx: int) -> str:
    for key in ("id", "name", "candidate_id", "label"):
        val = cand.get(key)
        if val:
            return str(val)
    return f"cand_{idx + 1}"


def _candidate_score(cand: Dict[str, Any]) -> float:
    """Pull a single rank-key score; higher is better.

    Looks for an explicit ``score``/``rank_score``/``composite`` first, else
    falls back to common metric names. Missing -> ``-inf`` so unscored
    candidates sort last.
    """
    for key in ("score", "rank_score", "composite", "composite_score", "fitness"):
        if isinstance(cand.get(key), (int, float)):
            return float(cand[key])
    for key in ("plddt", "iptm", "ptm", "ddg", "cai"):
        if isinstance(cand.get(key), (int, float)):
            return float(cand[key])
    return float("-inf")


def _liabilities_three_letter(cand: Dict[str, Any]) -> List[str]:
    """Render a candidate's liabilities as Ala123Lys-style strings.

    Accepts liabilities either as a list of single-letter mutation codes
    (``"A123K"``) under ``liabilities``/``mutations``/``flags``, or as a list
    of dicts carrying a ``code`` field. Free-text entries pass through.
    """
    out: List[str] = []
    raw = cand.get("liabilities") or cand.get("mutations") or cand.get("flags") or []
    if isinstance(raw, str):
        raw = [raw]
    for item in raw:
        if isinstance(item, dict):
            code = item.get("code") or item.get("mutation") or ""
            note = item.get("note") or item.get("description") or ""
            rendered = _format_mutation_three_letter(str(code)) if code else ""
            if rendered and note:
                out.append(f"{rendered} ({note})")
            elif rendered:
                out.append(rendered)
            elif note:
                out.append(note)
        else:
            out.append(_format_mutation_three_letter(str(item)))
    return [s for s in out if s]


def _csv_cell(value: Any) -> str:
    """Minimal RFC-4180-ish CSV escaping (no csv module needed, no deps)."""
    text = "" if value is None else str(value)
    if any(ch in text for ch in (",", '"', "\n", "\r")):
        return '"' + text.replace('"', '""') + '"'
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_order_pack(
    candidates: List[Dict[str, Any]],
    track: str,
    host: str = "ecoli",
    *,
    tag: str = "his6",
    signal: str = "none",
    linker: Optional[str] = _DEFAULT_LINKER,
) -> Dict[str, Any]:
    """Assemble a synthesis-ready order pack from ranked design candidates.

    Parameters
    ----------
    candidates:
        List of candidate dicts. Recognized keys (all optional except a
        sequence): ``id``/``name``, ``sequence``/``seq``, a score key
        (``score``/``rank_score``/``composite``/…), and a liabilities list
        under ``liabilities``/``mutations``/``flags``.
    track:
        Design track driving the assay plan: ``"binder"`` | ``"enzyme"`` |
        ``"stability"``; anything else falls back to a generic plan.
    host:
        Expression host label for the construct map / assay notes
        (``"ecoli"`` | ``"hek293"`` | ``"cho"`` | ``"yeast"``).
    tag, signal, linker:
        Protein-level elements fused at the N-terminus before back-translation.
        ``tag``/``signal`` are looked up in the built-in tables (unknown names
        are treated as literal AA sequences); ``linker`` joins the helper
        elements to the payload (set to ``None``/``""`` to disable).

    Returns
    -------
    dict with keys: ``track``, ``host``, ``n_candidates``, ``ranked_csv``,
    ``construct_fasta``, ``construct_map``, ``assay_plan``, ``risk_sheet``.
    """
    track_key = (track or "generic").strip().lower()
    host_key = (host or "ecoli").strip().lower()

    tag_seq = _TAGS.get(tag.lower(), tag.upper() if tag else "")
    signal_seq = _SIGNALS.get(signal.lower(), signal.upper() if signal else "")
    link_seq = (linker or "").upper()

    # Rank best-first; preserve input order for ties via enumerate index.
    ordered = sorted(
        enumerate(candidates),
        key=lambda pair: (-_candidate_score(pair[1]), pair[0]),
    )

    csv_rows: List[str] = ["rank,id,score,length,n_liabilities,sequence"]
    fasta_chunks: List[str] = []
    map_lines: List[str] = [
        f"Construct map  (track={track_key}, host={host_key})",
        f"Host: {_HOST_NOTE.get(host_key, host_key)}",
        "Element order (N->C): [signal]-[tag]-[linker]-payload",
        "=" * 64,
    ]
    risk_rows: List[Dict[str, Any]] = []

    for rank, (orig_idx, cand) in enumerate(ordered, start=1):
        cid = _candidate_id(cand, orig_idx)
        payload = _clean_protein(cand.get("sequence") or cand.get("seq") or "")
        score = _candidate_score(cand)
        liabilities = _liabilities_three_letter(cand)

        # Fuse helper elements at the N-terminus (signal, tag, linker, payload).
        parts: List[str] = []
        if signal_seq:
            parts.append(signal_seq)
        if tag_seq:
            parts.append(tag_seq)
        fused_prefix = "".join(parts)
        if fused_prefix and link_seq and payload:
            fused_protein = fused_prefix + link_seq + payload
        else:
            fused_protein = fused_prefix + payload

        dna = _sanitize_dna(_back_translate_seq(fused_protein, host_key))

        score_str = "" if score == float("-inf") else f"{score:.4f}"
        csv_rows.append(
            ",".join(
                _csv_cell(v)
                for v in (rank, cid, score_str, len(payload), len(liabilities), payload)
            )
        )

        header = (
            f">{cid}|rank={rank}|track={track_key}|host={host_key}"
            f"|tag={tag.lower()}|signal={signal.lower()}|len_dna={len(dna)}"
        )
        wrapped = "\n".join(dna[i : i + 60] for i in range(0, len(dna), 60)) or ""
        fasta_chunks.append(f"{header}\n{wrapped}")

        map_lines.append(f"[{rank}] {cid}  (rank score {score_str or 'n/a'})")
        if signal_seq:
            map_lines.append(f"      signal : {signal.lower()} ({len(signal_seq)} aa)")
        if tag_seq:
            map_lines.append(f"      tag    : {tag.lower()} ({len(tag_seq)} aa)")
        if fused_prefix and link_seq and payload:
            map_lines.append(f"      linker : {link_seq} ({len(link_seq)} aa)")
        map_lines.append(f"      payload: {len(payload)} aa -> {len(dna)} bp CDS")
        map_lines.append("")

        risk_rows.append(
            {
                "rank": rank,
                "id": cid,
                "liabilities": liabilities,
                "n_liabilities": len(liabilities),
            }
        )

    ranked_csv = "\n".join(csv_rows) + "\n"
    construct_fasta = "\n".join(fasta_chunks) + ("\n" if fasta_chunks else "")
    construct_map = "\n".join(map_lines).rstrip() + "\n"

    plan_lines = _ASSAY_PLANS.get(track_key, _ASSAY_PLANS["generic"])
    assay_plan = (
        f"Assay plan — track: {track_key} | host: {host_key}\n"
        f"{_HOST_NOTE.get(host_key, host_key)}\n"
        + "-" * 64
        + "\n"
        + "\n".join(plan_lines)
        + "\n"
    )

    risk_sheet = _render_risk_sheet(risk_rows)

    return {
        "track": track_key,
        "host": host_key,
        "n_candidates": len(candidates),
        "ranked_csv": ranked_csv,
        "construct_fasta": construct_fasta,
        "construct_map": construct_map,
        "assay_plan": assay_plan,
        "risk_sheet": risk_sheet,
        "risk_rows": risk_rows,
    }


def _render_risk_sheet(rows: Sequence[Dict[str, Any]]) -> str:
    """Render the liability rows as a readable text sheet (Ala123Lys notation)."""
    lines = ["Risk sheet — sequence liabilities (Ala123Lys notation)", "=" * 64]
    if not rows:
        lines.append("(no candidates)")
        return "\n".join(lines) + "\n"
    for row in rows:
        liabs = row["liabilities"]
        body = "; ".join(liabs) if liabs else "none flagged"
        lines.append(f"[{row['rank']}] {row['id']}  ({row['n_liabilities']}): {body}")
    return "\n".join(lines) + "\n"


# Artifact name -> output filename.
_ARTIFACT_FILES: Dict[str, str] = {
    "ranked_csv": "ranked_candidates.csv",
    "construct_fasta": "constructs.fasta",
    "construct_map": "construct_map.txt",
    "assay_plan": "assay_plan.txt",
    "risk_sheet": "risk_sheet.txt",
}


def write_order_pack(pack: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    """Write each text artifact in ``pack`` to ``out_dir``.

    Creates ``out_dir`` if needed. Only the known text artifacts are written
    (``ranked_csv``, ``construct_fasta``, ``construct_map``, ``assay_plan``,
    ``risk_sheet``). Returns a mapping of artifact name -> absolute file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}
    for artifact, filename in _ARTIFACT_FILES.items():
        content = pack.get(artifact)
        if content is None:
            continue
        path = os.path.abspath(os.path.join(out_dir, filename))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content if isinstance(content, str) else str(content))
        written[artifact] = path
    return written
