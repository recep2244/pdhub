"""Protein analysis utilities for Protein Design Hub."""

from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import re

# Amino acid properties
AA_PROPERTIES = {
    'A': {'name': 'Alanine', 'code': 'Ala', 'hydrophobic': True, 'charge': 0, 'mw': 89.1, 'pI': 6.0},
    'C': {'name': 'Cysteine', 'code': 'Cys', 'hydrophobic': False, 'charge': 0, 'mw': 121.2, 'pI': 5.1},
    'D': {'name': 'Aspartate', 'code': 'Asp', 'hydrophobic': False, 'charge': -1, 'mw': 133.1, 'pI': 2.8},
    'E': {'name': 'Glutamate', 'code': 'Glu', 'hydrophobic': False, 'charge': -1, 'mw': 147.1, 'pI': 3.2},
    'F': {'name': 'Phenylalanine', 'code': 'Phe', 'hydrophobic': True, 'charge': 0, 'mw': 165.2, 'pI': 5.5},
    'G': {'name': 'Glycine', 'code': 'Gly', 'hydrophobic': False, 'charge': 0, 'mw': 75.1, 'pI': 6.0},
    'H': {'name': 'Histidine', 'code': 'His', 'hydrophobic': False, 'charge': 0.1, 'mw': 155.2, 'pI': 7.6},
    'I': {'name': 'Isoleucine', 'code': 'Ile', 'hydrophobic': True, 'charge': 0, 'mw': 131.2, 'pI': 6.0},
    'K': {'name': 'Lysine', 'code': 'Lys', 'hydrophobic': False, 'charge': 1, 'mw': 146.2, 'pI': 9.7},
    'L': {'name': 'Leucine', 'code': 'Leu', 'hydrophobic': True, 'charge': 0, 'mw': 131.2, 'pI': 6.0},
    'M': {'name': 'Methionine', 'code': 'Met', 'hydrophobic': True, 'charge': 0, 'mw': 149.2, 'pI': 5.7},
    'N': {'name': 'Asparagine', 'code': 'Asn', 'hydrophobic': False, 'charge': 0, 'mw': 132.1, 'pI': 5.4},
    'P': {'name': 'Proline', 'code': 'Pro', 'hydrophobic': False, 'charge': 0, 'mw': 115.1, 'pI': 6.3},
    'Q': {'name': 'Glutamine', 'code': 'Gln', 'hydrophobic': False, 'charge': 0, 'mw': 146.2, 'pI': 5.7},
    'R': {'name': 'Arginine', 'code': 'Arg', 'hydrophobic': False, 'charge': 1, 'mw': 174.2, 'pI': 10.8},
    'S': {'name': 'Serine', 'code': 'Ser', 'hydrophobic': False, 'charge': 0, 'mw': 105.1, 'pI': 5.7},
    'T': {'name': 'Threonine', 'code': 'Thr', 'hydrophobic': False, 'charge': 0, 'mw': 119.1, 'pI': 5.6},
    'V': {'name': 'Valine', 'code': 'Val', 'hydrophobic': True, 'charge': 0, 'mw': 117.1, 'pI': 6.0},
    'W': {'name': 'Tryptophan', 'code': 'Trp', 'hydrophobic': True, 'charge': 0, 'mw': 204.2, 'pI': 5.9},
    'Y': {'name': 'Tyrosine', 'code': 'Tyr', 'hydrophobic': True, 'charge': 0, 'mw': 181.2, 'pI': 5.7},
}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}


def parse_multichain_sequence(sequence_input: str) -> List[Dict[str, Any]]:
    """
    Parse multi-chain sequence input.

    Supports formats:
    - Single sequence: "MKFLILL..."
    - Colon-separated: "MKFLILL:ADEFGHIK"
    - FASTA with multiple entries
    - Chain notation: "A:MKFLILL:B:ADEFGHIK"

    Returns list of dicts with chain_id, sequence, description
    """
    chains = []
    sequence_input = sequence_input.strip()

    # Check if FASTA format
    if sequence_input.startswith(">"):
        lines = sequence_input.split("\n")
        current_header = ""
        current_seq = []
        chain_idx = 0

        for line in lines:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    seq = "".join(current_seq)
                    chain_id = chr(ord("A") + chain_idx)
                    chains.append({
                        "chain_id": chain_id,
                        "sequence": seq,
                        "description": current_header
                    })
                    chain_idx += 1
                    current_seq = []
                current_header = line[1:].strip()
            elif line:
                current_seq.append(line.upper())

        # Last sequence
        if current_seq:
            seq = "".join(current_seq)
            # Check for colon separator in sequence
            if ":" in seq:
                for part in seq.split(":"):
                    if part and not part.isalpha():
                        continue
                    if part:
                        chain_id = chr(ord("A") + chain_idx)
                        chains.append({
                            "chain_id": chain_id,
                            "sequence": part,
                            "description": current_header
                        })
                        chain_idx += 1
            else:
                chain_id = chr(ord("A") + chain_idx)
                chains.append({
                    "chain_id": chain_id,
                    "sequence": seq,
                    "description": current_header
                })
    else:
        # Check for chain notation: A:SEQ:B:SEQ
        chain_pattern = r'^([A-Z]):([A-Z]+)(?::([A-Z]):([A-Z]+))*$'
        if re.match(chain_pattern, sequence_input):
            parts = sequence_input.split(":")
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    chains.append({
                        "chain_id": parts[i],
                        "sequence": parts[i + 1],
                        "description": f"Chain {parts[i]}"
                    })
        # Simple colon separator
        elif ":" in sequence_input:
            parts = sequence_input.split(":")
            for i, part in enumerate(parts):
                if part:
                    chain_id = chr(ord("A") + i)
                    chains.append({
                        "chain_id": chain_id,
                        "sequence": part.upper(),
                        "description": f"Chain {chain_id}"
                    })
        # Single sequence
        else:
            chains.append({
                "chain_id": "A",
                "sequence": sequence_input.upper(),
                "description": "Chain A"
            })

    return chains


def calculate_sequence_properties(sequence: str) -> Dict[str, Any]:
    """Calculate biophysical properties of a protein sequence."""
    sequence = sequence.upper()
    valid_aa = set(AA_PROPERTIES.keys())
    sequence = "".join(c for c in sequence if c in valid_aa)

    if not sequence:
        return {}

    # Molecular weight
    mw = sum(AA_PROPERTIES.get(aa, {}).get('mw', 0) for aa in sequence) - (len(sequence) - 1) * 18.015

    # Net charge at pH 7
    charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in sequence)

    # GRAVY (Grand Average of Hydropathicity)
    gravy = sum(HYDROPHOBICITY.get(aa, 0) for aa in sequence) / len(sequence) if sequence else 0

    # Isoelectric point estimation (simplified)
    n_pos = sequence.count('K') + sequence.count('R') + sequence.count('H')
    n_neg = sequence.count('D') + sequence.count('E')
    if n_pos + n_neg > 0:
        pI = 7.0 + (n_pos - n_neg) * 0.5  # Simplified estimation
    else:
        pI = 7.0
    pI = max(3.0, min(12.0, pI))

    # Instability index (simplified)
    instability_weights = {
        'W': {'W': 1}, 'D': {'G': 1, 'P': 1}, 'P': {'P': 1},
        # Add more as needed
    }
    instability = 0
    for i in range(len(sequence) - 1):
        aa1, aa2 = sequence[i], sequence[i + 1]
        if aa1 in instability_weights and aa2 in instability_weights.get(aa1, {}):
            instability += instability_weights[aa1][aa2]
    instability_index = (10.0 / len(sequence)) * instability if sequence else 0

    # Composition
    composition = {}
    for aa in sequence:
        composition[aa] = composition.get(aa, 0) + 1

    # Hydrophobic fraction
    hydrophobic_count = sum(1 for aa in sequence if AA_PROPERTIES.get(aa, {}).get('hydrophobic', False))
    hydrophobic_fraction = hydrophobic_count / len(sequence) if sequence else 0

    return {
        'length': len(sequence),
        'molecular_weight': round(mw, 2),
        'molecular_weight_kda': round(mw / 1000, 2),
        'net_charge': charge,
        'gravy': round(gravy, 3),
        'isoelectric_point': round(pI, 2),
        'instability_index': round(instability_index, 2),
        'hydrophobic_fraction': round(hydrophobic_fraction, 3),
        'composition': composition,
    }


def predict_secondary_structure_propensity(sequence: str) -> Dict[str, float]:
    """
    Predict secondary structure propensity using Chou-Fasman parameters.
    Returns propensities for helix, sheet, and coil.
    """
    # Chou-Fasman propensities (simplified)
    helix_prop = {
        'A': 1.42, 'L': 1.21, 'E': 1.51, 'M': 1.45, 'Q': 1.11,
        'K': 1.16, 'R': 0.98, 'H': 1.00, 'V': 1.06, 'I': 1.08,
        'Y': 0.69, 'C': 0.70, 'W': 1.08, 'F': 1.13, 'T': 0.83,
        'G': 0.57, 'N': 0.67, 'P': 0.57, 'S': 0.77, 'D': 1.01,
    }
    sheet_prop = {
        'A': 0.83, 'L': 1.30, 'E': 0.37, 'M': 1.05, 'Q': 1.10,
        'K': 0.74, 'R': 0.93, 'H': 0.87, 'V': 1.70, 'I': 1.60,
        'Y': 1.47, 'C': 1.19, 'W': 1.37, 'F': 1.38, 'T': 1.19,
        'G': 0.75, 'N': 0.89, 'P': 0.55, 'S': 0.75, 'D': 0.54,
    }

    sequence = sequence.upper()
    if not sequence:
        return {'helix': 0, 'sheet': 0, 'coil': 0}

    helix_score = sum(helix_prop.get(aa, 1.0) for aa in sequence) / len(sequence)
    sheet_score = sum(sheet_prop.get(aa, 1.0) for aa in sequence) / len(sequence)
    coil_score = 2.0 - helix_score - sheet_score  # Simplified

    total = helix_score + sheet_score + max(0, coil_score)

    return {
        'helix': round(helix_score / total, 3) if total > 0 else 0.33,
        'sheet': round(sheet_score / total, 3) if total > 0 else 0.33,
        'coil': round(max(0, coil_score) / total, 3) if total > 0 else 0.34,
    }


def predict_aggregation_propensity(sequence: str) -> Dict[str, Any]:
    """
    Predict aggregation propensity using simplified TANGO-like approach.
    """
    # Simplified aggregation propensity scores
    agg_scores = {
        'V': 0.8, 'I': 0.9, 'L': 0.7, 'F': 1.0, 'Y': 0.6,
        'W': 0.5, 'M': 0.4, 'A': 0.3, 'G': 0.1, 'P': -0.5,
        'C': 0.4, 'S': 0.0, 'T': 0.1, 'N': -0.2, 'Q': -0.1,
        'D': -0.8, 'E': -0.7, 'K': -0.9, 'R': -0.8, 'H': -0.3,
    }

    sequence = sequence.upper()
    if len(sequence) < 5:
        return {'aggregation_prone': False, 'score': 0, 'hotspots': []}

    # Calculate sliding window scores
    window_size = 7
    hotspots = []
    scores = []

    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        score = sum(agg_scores.get(aa, 0) for aa in window) / window_size
        scores.append(score)
        if score > 0.5:
            hotspots.append({
                'start': i + 1,
                'end': i + window_size,
                'sequence': window,
                'score': round(score, 3)
            })

    avg_score = sum(scores) / len(scores) if scores else 0

    return {
        'aggregation_prone': avg_score > 0.3 or len(hotspots) > 2,
        'average_score': round(avg_score, 3),
        'hotspots': hotspots,
        'num_hotspots': len(hotspots),
    }


def predict_solubility(sequence: str) -> Dict[str, Any]:
    """
    Predict protein solubility using sequence features.
    """
    props = calculate_sequence_properties(sequence)

    if not props:
        return {'soluble': False, 'score': 0}

    # Factors affecting solubility
    gravy = props.get('gravy', 0)
    charge = props.get('net_charge', 0)
    length = props.get('length', 0)

    # Simple solubility score
    # More hydrophobic = less soluble
    # More charged = more soluble
    solubility_score = 0.5
    solubility_score -= gravy * 0.1  # Hydrophobicity reduces solubility
    solubility_score += abs(charge) * 0.02  # Charge increases solubility

    # Length penalty (very long proteins less soluble)
    if length > 500:
        solubility_score -= (length - 500) * 0.0001

    solubility_score = max(0, min(1, solubility_score))

    return {
        'soluble': solubility_score > 0.4,
        'score': round(solubility_score, 3),
        'factors': {
            'hydrophobicity': 'high' if gravy > 0 else 'low',
            'net_charge': charge,
            'length_penalty': length > 500,
        }
    }


def calculate_conservation_score(
    query_sequence: str,
    aligned_sequences: List[str],
    method: str = "shannon"
) -> List[float]:
    """
    Calculate per-position conservation scores from MSA.

    Args:
        query_sequence: The query sequence
        aligned_sequences: List of aligned sequences
        method: "shannon" for Shannon entropy or "identity" for percent identity

    Returns:
        List of conservation scores per position (0-1, higher = more conserved)
    """
    import math

    if not aligned_sequences:
        return [0.5] * len(query_sequence)

    all_seqs = [query_sequence] + aligned_sequences
    seq_length = len(query_sequence)

    scores = []
    for pos in range(seq_length):
        # Get amino acids at this position
        aas = []
        for seq in all_seqs:
            if pos < len(seq) and seq[pos] != '-':
                aas.append(seq[pos].upper())

        if not aas:
            scores.append(0.5)
            continue

        if method == "identity":
            # Percent identity with query
            query_aa = query_sequence[pos].upper()
            identity = sum(1 for aa in aas if aa == query_aa) / len(aas)
            scores.append(identity)
        else:
            # Shannon entropy
            aa_counts = {}
            for aa in aas:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1

            total = len(aas)
            entropy = 0
            for count in aa_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)

            # Normalize entropy (max entropy for 20 AA is log2(20) ≈ 4.32)
            max_entropy = math.log2(min(20, total))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Convert to conservation (1 - normalized_entropy)
            conservation = 1 - normalized_entropy
            scores.append(conservation)

    return scores


def detect_domains(sequence: str) -> List[Dict[str, Any]]:
    """
    Simple domain detection based on sequence patterns.
    Returns list of potential domains/motifs.
    """
    domains = []
    sequence = sequence.upper()

    # Common motifs
    motifs = {
        'N-glycosylation': r'N[^P][ST][^P]',
        'Protein kinase C phosphorylation': r'[ST]..[RK]',
        'Casein kinase II phosphorylation': r'[ST]..[DE]',
        'N-myristoylation': r'G[^EDRKHPFYW]..[STAGCN][^P]',
        'ATP/GTP binding (P-loop)': r'[AG]....GK[ST]',
        'EF-hand calcium-binding': r'D.{2}[DN].{3}[DENQ]',
        'Zinc finger (C2H2)': r'C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H',
        'Leucine zipper': r'L.{6}L.{6}L.{6}L',
    }

    for motif_name, pattern in motifs.items():
        import re
        for match in re.finditer(pattern, sequence):
            domains.append({
                'name': motif_name,
                'start': match.start() + 1,
                'end': match.end(),
                'sequence': match.group(),
                'type': 'motif'
            })

    return domains


def validate_sequence(sequence: str) -> Dict[str, Any]:
    """
    Validate a protein sequence.
    """
    errors = []
    warnings = []

    sequence = sequence.strip()

    # Check for empty
    if not sequence:
        errors.append("Sequence is empty")
        return {'valid': False, 'errors': errors, 'warnings': warnings}

    # Remove FASTA header if present
    if sequence.startswith(">"):
        lines = sequence.split("\n")
        sequence = "".join(l for l in lines[1:] if not l.startswith(">"))

    # Remove whitespace
    sequence = "".join(sequence.split())

    # Check for invalid characters
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
    invalid_chars = set(c.upper() for c in sequence) - valid_aa - set(":-")
    if invalid_chars:
        errors.append(f"Invalid characters: {', '.join(sorted(invalid_chars))}")

    # Check length
    clean_seq = "".join(c for c in sequence.upper() if c in valid_aa)
    if len(clean_seq) < 10:
        warnings.append("Sequence is very short (< 10 residues)")
    if len(clean_seq) > 2000:
        warnings.append("Sequence is very long (> 2000 residues), prediction may be slow")

    # Check for unusual composition
    if clean_seq:
        x_count = clean_seq.count('X')
        if x_count > len(clean_seq) * 0.1:
            warnings.append(f"High proportion of unknown residues (X): {x_count}")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'cleaned_sequence': clean_seq,
        'length': len(clean_seq),
    }
