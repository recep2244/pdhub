"""Wheat (Triticum aestivum) codon optimization and expression analysis.

Pure-Python, no external dependencies.

Provides:
  - Triticum aestivum codon usage table (from NCBI GenBank / Kazusa CUT DB)
  - Codon Adaptation Index (CAI) calculation
  - Back-translation (protein → optimised DNA)
  - Cryptic splice site detection (GT-AG, GC-AG, AT-AC; plant-specific contexts)
  - Poly-A signal detection (AATAAA, ATTAAA, TATAAA variants)
  - Comparison to Oryza sativa and Zea mays codon tables

References
----------
* Sharp & Li 1987 (CAI definition)
* Nakamura et al. 2000 (Kazusa CUT DB, codon usage tables)
* Goodstein et al. 2012 (Phytozome, wheat expressed sequences)
* Reddy 2007 (plant pre-mRNA splicing signals)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Genetic code
# ---------------------------------------------------------------------------

GENETIC_CODE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CAT": "H", "CAC": "H",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Reverse map: AA → list of codons
_AA_TO_CODONS: Dict[str, List[str]] = {}
for codon, aa in GENETIC_CODE.items():
    if aa != "*":
        _AA_TO_CODONS.setdefault(aa, []).append(codon)


# ---------------------------------------------------------------------------
# Codon usage tables (relative synonymous codon usage, RSCU)
# Values are per-codon fractional frequency within synonymous family.
# Source: Kazusa CUT DB + NCBI Triticum aestivum entries.
# ---------------------------------------------------------------------------

WHEAT_CODON_FREQ: Dict[str, float] = {
    # Ala
    "GCT": 0.38, "GCC": 0.28, "GCA": 0.22, "GCG": 0.12,
    # Arg
    "CGT": 0.28, "CGC": 0.18, "CGA": 0.10, "CGG": 0.14, "AGA": 0.18, "AGG": 0.12,
    # Asn
    "AAT": 0.40, "AAC": 0.60,
    # Asp
    "GAT": 0.46, "GAC": 0.54,
    # Cys
    "TGT": 0.44, "TGC": 0.56,
    # Gln
    "CAA": 0.36, "CAG": 0.64,
    # Glu
    "GAA": 0.38, "GAG": 0.62,
    # Gly
    "GGT": 0.32, "GGC": 0.28, "GGA": 0.22, "GGG": 0.18,
    # His
    "CAT": 0.38, "CAC": 0.62,
    # Ile
    "ATT": 0.38, "ATC": 0.46, "ATA": 0.16,
    # Leu
    "TTA": 0.06, "TTG": 0.16, "CTT": 0.22, "CTC": 0.22, "CTA": 0.08, "CTG": 0.26,
    # Lys
    "AAA": 0.34, "AAG": 0.66,
    # Met
    "ATG": 1.00,
    # Phe
    "TTT": 0.42, "TTC": 0.58,
    # Pro
    "CCT": 0.32, "CCC": 0.22, "CCA": 0.28, "CCG": 0.18,
    # Ser
    "TCT": 0.22, "TCC": 0.22, "TCA": 0.14, "TCG": 0.08, "AGT": 0.18, "AGC": 0.16,
    # Thr
    "ACT": 0.32, "ACC": 0.34, "ACA": 0.22, "ACG": 0.12,
    # Trp
    "TGG": 1.00,
    # Tyr
    "TAT": 0.40, "TAC": 0.60,
    # Val
    "GTT": 0.32, "GTC": 0.24, "GTA": 0.14, "GTG": 0.30,
}

RICE_CODON_FREQ: Dict[str, float] = {
    "GCT": 0.20, "GCC": 0.32, "GCA": 0.18, "GCG": 0.30,
    "CGT": 0.20, "CGC": 0.26, "CGA": 0.08, "CGG": 0.22, "AGA": 0.14, "AGG": 0.10,
    "AAT": 0.28, "AAC": 0.72, "GAT": 0.32, "GAC": 0.68,
    "TGT": 0.32, "TGC": 0.68, "CAA": 0.26, "CAG": 0.74,
    "GAA": 0.28, "GAG": 0.72, "GGT": 0.18, "GGC": 0.38, "GGA": 0.18, "GGG": 0.26,
    "CAT": 0.28, "CAC": 0.72, "ATT": 0.28, "ATC": 0.58, "ATA": 0.14,
    "TTA": 0.04, "TTG": 0.10, "CTT": 0.14, "CTC": 0.24, "CTA": 0.04, "CTG": 0.44,
    "AAA": 0.26, "AAG": 0.74, "ATG": 1.00,
    "TTT": 0.32, "TTC": 0.68, "CCT": 0.18, "CCC": 0.28, "CCA": 0.22, "CCG": 0.32,
    "TCT": 0.12, "TCC": 0.26, "TCA": 0.10, "TCG": 0.12, "AGT": 0.14, "AGC": 0.26,
    "ACT": 0.18, "ACC": 0.42, "ACA": 0.16, "ACG": 0.24,
    "TGG": 1.00, "TAT": 0.28, "TAC": 0.72,
    "GTT": 0.20, "GTC": 0.26, "GTA": 0.10, "GTG": 0.44,
}

MAIZE_CODON_FREQ: Dict[str, float] = {
    "GCT": 0.18, "GCC": 0.30, "GCA": 0.20, "GCG": 0.32,
    "CGT": 0.18, "CGC": 0.22, "CGA": 0.10, "CGG": 0.20, "AGA": 0.18, "AGG": 0.12,
    "AAT": 0.30, "AAC": 0.70, "GAT": 0.34, "GAC": 0.66,
    "TGT": 0.36, "TGC": 0.64, "CAA": 0.30, "CAG": 0.70,
    "GAA": 0.32, "GAG": 0.68, "GGT": 0.20, "GGC": 0.30, "GGA": 0.20, "GGG": 0.30,
    "CAT": 0.32, "CAC": 0.68, "ATT": 0.30, "ATC": 0.54, "ATA": 0.16,
    "TTA": 0.04, "TTG": 0.12, "CTT": 0.16, "CTC": 0.22, "CTA": 0.06, "CTG": 0.40,
    "AAA": 0.28, "AAG": 0.72, "ATG": 1.00,
    "TTT": 0.36, "TTC": 0.64, "CCT": 0.20, "CCC": 0.26, "CCA": 0.22, "CCG": 0.32,
    "TCT": 0.14, "TCC": 0.24, "TCA": 0.12, "TCG": 0.12, "AGT": 0.16, "AGC": 0.22,
    "ACT": 0.20, "ACC": 0.40, "ACA": 0.18, "ACG": 0.22,
    "TGG": 1.00, "TAT": 0.32, "TAC": 0.68,
    "GTT": 0.18, "GTC": 0.24, "GTA": 0.12, "GTG": 0.46,
}

SPECIES_TABLES = {
    "wheat": WHEAT_CODON_FREQ,
    "rice":  RICE_CODON_FREQ,
    "maize": MAIZE_CODON_FREQ,
}

# Maximum-frequency codon per AA per species (for back-translation)
def _max_freq_codon_table(freq_table: Dict[str, float]) -> Dict[str, str]:
    best: Dict[str, str] = {}
    for aa, codons in _AA_TO_CODONS.items():
        valid = [c for c in codons if c in freq_table]
        if valid:
            best[aa] = max(valid, key=lambda c: freq_table[c])
    return best


_MAX_CODON: Dict[str, Dict[str, str]] = {
    species: _max_freq_codon_table(table)
    for species, table in SPECIES_TABLES.items()
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CodonOptimizationResult:
    """Result of codon optimization for a protein sequence."""

    protein_sequence: str
    dna_sequence: str
    species: str
    cai: float
    gc_content: float
    cryptic_splice_sites: List[Dict]
    poly_a_signals: List[Dict]
    rare_codon_positions: List[Dict]
    warnings: List[str] = field(default_factory=list)
    fasta: str = ""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def back_translate(protein_seq: str, species: str = "wheat") -> str:
    """Translate protein to DNA using the most-frequent codon per species.

    Parameters
    ----------
    protein_seq:
        Single-letter amino acid sequence (uppercase, no gaps).
    species:
        "wheat" | "rice" | "maize"

    Returns
    -------
    DNA coding sequence (uppercase, no stop codon appended).
    """
    species = species.lower()
    if species not in _MAX_CODON:
        raise ValueError(f"Unknown species '{species}'. Use: {list(_MAX_CODON)}")
    table = _MAX_CODON[species]
    dna = []
    for aa in protein_seq.upper().replace("*", ""):
        codon = table.get(aa)
        if codon is None:
            codon = "NNN"  # unknown / ambiguous AA
        dna.append(codon)
    return "".join(dna)


def calculate_cai(dna_seq: str, species: str = "wheat") -> float:
    """Codon Adaptation Index (Sharp & Li 1987).

    CAI = geometric mean of RSCU(codon) / max_RSCU(synonymous_family)
    Range 0–1; > 0.8 = high expression expected.

    Parameters
    ----------
    dna_seq:
        CDS in-frame, length divisible by 3 (without stop codon).
    species:
        "wheat" | "rice" | "maize"
    """
    species = species.lower()
    freq_table = SPECIES_TABLES.get(species, WHEAT_CODON_FREQ)
    codons = [dna_seq[i:i+3].upper() for i in range(0, len(dna_seq) - 2, 3)]
    log_sum = 0.0
    count = 0
    for codon in codons:
        aa = GENETIC_CODE.get(codon, "?")
        if aa in ("*", "?") or aa == "M" or aa == "W":  # single-codon AAs excluded
            continue
        synonyms = [c for c in _AA_TO_CODONS.get(aa, []) if c in freq_table]
        if not synonyms:
            continue
        max_freq = max(freq_table[c] for c in synonyms)
        codon_freq = freq_table.get(codon, 1e-6)
        if max_freq > 0:
            log_sum += math.log(codon_freq / max_freq)
            count += 1
    if count == 0:
        return 0.0
    return round(math.exp(log_sum / count), 4)


def gc_content(dna_seq: str) -> float:
    """GC content of a DNA sequence (0–1)."""
    dna = dna_seq.upper()
    gc = sum(1 for nt in dna if nt in "GC")
    return round(gc / max(1, len(dna)), 4)


def find_rare_codons(
    dna_seq: str, species: str = "wheat", threshold: float = 0.10
) -> List[Dict]:
    """Find codons with frequency below `threshold` in the species codon table.

    Parameters
    ----------
    threshold:
        Fractional frequency below which a codon is considered "rare" (default 0.10).

    Returns
    -------
    List of dicts: {position, codon, aa, frequency, species}
    """
    species = species.lower()
    freq_table = SPECIES_TABLES.get(species, WHEAT_CODON_FREQ)
    rare = []
    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i+3].upper()
        aa = GENETIC_CODE.get(codon, "?")
        if aa in ("*", "?"):
            continue
        freq = freq_table.get(codon, 0.0)
        if freq < threshold:
            rare.append({
                "aa_position": i // 3 + 1,
                "codon": codon,
                "aa": aa,
                "frequency": freq,
                "species": species,
            })
    return rare


# ---------------------------------------------------------------------------
# Cryptic splice site detection
# ---------------------------------------------------------------------------

# Plant splice site consensus sequences (GT-AG rule dominant, GC-AG minor)
# Donor: exon(3 nt) | GT(AG)AAGT  → [AG]GTAAGT is the 5' splice site
# Acceptor: YAG | exon → pyrimidine-rich stretch ending in AG
_DONOR_PATTERNS = [
    (r"[AG]GTAAGT",  "GT-AG 5' splice donor (canonical)"),
    (r"[AG]GTATGT",  "GT-AG 5' splice donor (variant)"),
    (r"[AG]GCAAGT",  "GC-AG 5' splice donor (minor)"),
    (r"ATATCC",       "AT-AC 5' splice donor (U12, rare)"),
]
_ACCEPTOR_PATTERNS = [
    (r"[CT]{4,}AG",   "poly-pyrimidine 3' splice acceptor (canonical)"),
    (r"TTTTTTCAG",    "strong poly-T acceptor (plant-preferred)"),
]


def detect_cryptic_splice_sites(dna_seq: str) -> List[Dict]:
    """Detect potential cryptic splice sites in a coding DNA sequence.

    Returns list of dicts: {position, sequence, type, description, risk}
    """
    dna = dna_seq.upper()
    hits = []
    for pattern, desc in _DONOR_PATTERNS + _ACCEPTOR_PATTERNS:
        for m in re.finditer(pattern, dna):
            risk = "high" if "canonical" in desc else "medium"
            hits.append({
                "position": m.start() + 1,
                "sequence": m.group(),
                "type": "donor" if "donor" in desc else "acceptor",
                "description": desc,
                "risk": risk,
            })
    return hits


# ---------------------------------------------------------------------------
# Poly-A signal detection
# ---------------------------------------------------------------------------

_POLYA_SIGNALS = [
    ("AATAAA", "Canonical poly-A signal (strongest)"),
    ("ATTAAA", "Variant poly-A signal (common)"),
    ("TATAAA", "Variant poly-A signal (plant)"),
    ("AGTAAA", "Variant poly-A signal (plant)"),
    ("AAGAAA", "Variant poly-A signal (plant)"),
    ("AATATA", "Variant poly-A signal (plant)"),
    ("AATACA", "Variant poly-A signal (plant)"),
]


def detect_poly_a_signals(dna_seq: str) -> List[Dict]:
    """Detect poly-A signals that could prematurely terminate transcription.

    Returns list of dicts: {position, sequence, description, risk}
    """
    dna = dna_seq.upper()
    hits = []
    for motif, desc in _POLYA_SIGNALS:
        for m in re.finditer(motif, dna):
            risk = "high" if motif == "AATAAA" else "medium"
            hits.append({
                "position": m.start() + 1,
                "sequence": m.group(),
                "description": desc,
                "risk": risk,
            })
    return hits


# ---------------------------------------------------------------------------
# High-level optimisation
# ---------------------------------------------------------------------------


def optimize_for_wheat(
    protein_seq: str,
    species: str = "wheat",
    gene_name: str = "gene",
) -> CodonOptimizationResult:
    """Full codon optimisation pipeline for a plant protein.

    1. Back-translate using most-frequent codons for the target species.
    2. Calculate CAI.
    3. Detect cryptic splice sites and poly-A signals.
    4. Identify rare codons.
    5. Produce FASTA output.

    Parameters
    ----------
    protein_seq:
        Single-letter amino acid sequence.
    species:
        "wheat" | "rice" | "maize"
    gene_name:
        Name for FASTA header.

    Returns
    -------
    CodonOptimizationResult dataclass.
    """
    prot = protein_seq.upper().replace("*", "").replace("-", "")
    dna  = back_translate(prot, species=species)
    cai  = calculate_cai(dna, species=species)
    gc   = gc_content(dna)
    splice_sites = detect_cryptic_splice_sites(dna)
    polya        = detect_poly_a_signals(dna)
    rare         = find_rare_codons(dna, species=species)

    warnings: List[str] = []
    if cai < 0.60:
        warnings.append(f"Low CAI ({cai:.2f}): expression in {species} may be suboptimal.")
    if gc < 0.40 or gc > 0.70:
        warnings.append(f"GC content {gc * 100:.1f}% is outside 40–70% optimal range for plant expression.")
    high_splice = [s for s in splice_sites if s["risk"] == "high"]
    if high_splice:
        warnings.append(
            f"{len(high_splice)} high-risk cryptic splice donor/acceptor sites detected. "
            "Recommend silent mutation to disrupt GT/AG dinucleotides."
        )
    high_polya = [p for p in polya if p["risk"] == "high"]
    if high_polya:
        warnings.append(
            f"{len(high_polya)} canonical poly-A signal(s) (AATAAA) detected in CDS. "
            "These can cause premature transcription termination."
        )
    if len(rare) > len(prot) * 0.10:
        warnings.append(
            f"{len(rare)} rare codons (>{len(rare)/len(prot)*100:.0f}% of CDS): "
            "back-translation uses optimal codons but check specific rare positions."
        )

    fasta = f">{gene_name}|{species}_optimized|CAI={cai:.3f}|GC={gc*100:.1f}%\n"
    # Wrap at 60 chars
    for i in range(0, len(dna), 60):
        fasta += dna[i:i+60] + "\n"

    return CodonOptimizationResult(
        protein_sequence=prot,
        dna_sequence=dna,
        species=species,
        cai=cai,
        gc_content=gc,
        cryptic_splice_sites=splice_sites,
        poly_a_signals=polya,
        rare_codon_positions=rare,
        warnings=warnings,
        fasta=fasta,
    )


def compare_species_cai(protein_seq: str) -> Dict[str, float]:
    """Calculate CAI for each supported species for comparison."""
    prot = protein_seq.upper().replace("*", "").replace("-", "")
    results: Dict[str, float] = {}
    for sp in SPECIES_TABLES:
        dna = back_translate(prot, species=sp)
        results[sp] = calculate_cai(dna, species=sp)
    return results
