"""Scientific metric interpretation and threshold-based context.

Provides automatic interpretation for all protein structure metrics used
across the web UI. Every metric gets a color, quality label, and a plain-English
description so that non-expert users can understand results immediately.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ── Metric threshold definitions ─────────────────────────────────────
# Each entry: metric_name -> list of (threshold, label, color, description)
# Thresholds are checked top-down; first match wins.
# For "lower is better" metrics, thresholds are checked with <=
# For "higher is better" metrics, thresholds are checked with >=

_HIGHER_IS_BETTER = {
    "plddt": [
        (90, "Excellent", "#22c55e", "Atomic-level accuracy; suitable for detailed analysis"),
        (70, "Good", "#3b82f6", "Reliable backbone; some side-chain uncertainty"),
        (50, "Moderate", "#f59e0b", "Use as rough model only; low confidence regions"),
        (0, "Poor", "#ef4444", "Structure likely incorrect; consider re-prediction"),
    ],
    "ptm": [
        (0.8, "Excellent", "#22c55e", "Very high confidence in overall fold"),
        (0.6, "Good", "#3b82f6", "Reliable global topology"),
        (0.4, "Moderate", "#f59e0b", "Some uncertainty in global fold"),
        (0, "Poor", "#ef4444", "Low confidence in predicted topology"),
    ],
    "iptm": [
        (0.8, "Excellent", "#22c55e", "High-confidence interface prediction"),
        (0.6, "Good", "#3b82f6", "Reliable interface contacts"),
        (0.4, "Moderate", "#f59e0b", "Interface may have errors"),
        (0, "Poor", "#ef4444", "Interface prediction unreliable"),
    ],
    "lddt": [
        (0.9, "Excellent", "#22c55e", "Near-native local accuracy across all residues"),
        (0.7, "Good", "#3b82f6", "Correct overall fold with minor local deviations"),
        (0.5, "Moderate", "#f59e0b", "Significant local errors; use with caution"),
        (0, "Poor", "#ef4444", "Poor local accuracy; structure may be wrong"),
    ],
    "tm_score": [
        (0.7, "High similarity", "#22c55e", "Nearly identical fold to reference"),
        (0.5, "Same fold", "#3b82f6", "Same structural family as reference"),
        (0.3, "Partial", "#f59e0b", "Some structural similarity but different fold"),
        (0, "Different", "#ef4444", "Structurally dissimilar to reference"),
    ],
    "gdt_ts": [
        (0.7, "Excellent", "#22c55e", "Most residues within 4 Å of reference"),
        (0.5, "Good", "#3b82f6", "Reasonable global structural match"),
        (0.3, "Moderate", "#f59e0b", "Partial match to reference"),
        (0, "Poor", "#ef4444", "Very different from reference"),
    ],
    "gdt_ha": [
        (0.5, "Excellent", "#22c55e", "High-accuracy structural match (2 Å threshold)"),
        (0.3, "Good", "#3b82f6", "Good high-accuracy match"),
        (0.15, "Moderate", "#f59e0b", "Partial high-accuracy match"),
        (0, "Poor", "#ef4444", "Poor high-accuracy match"),
    ],
    "qs_score": [
        (0.7, "Excellent", "#22c55e", "Interface very well preserved"),
        (0.5, "Good", "#3b82f6", "Interface reasonably preserved"),
        (0.3, "Moderate", "#f59e0b", "Interface partially correct"),
        (0, "Poor", "#ef4444", "Interface prediction failed"),
    ],
    "dockq": [
        (0.80, "High quality", "#22c55e", "Excellent docking; near-native interface"),
        (0.49, "Medium quality", "#3b82f6", "Acceptable interface prediction"),
        (0.23, "Acceptable", "#f59e0b", "Marginal interface; use with caution"),
        (0, "Incorrect", "#ef4444", "Interface prediction is incorrect"),
    ],
    "cad_score": [
        (0.7, "Excellent", "#22c55e", "Very similar contact surfaces"),
        (0.5, "Good", "#3b82f6", "Reasonable contact similarity"),
        (0.3, "Moderate", "#f59e0b", "Partial contact agreement"),
        (0, "Poor", "#ef4444", "Very different contact patterns"),
    ],
    "voromqa_score": [
        (0.4, "Good", "#22c55e", "High-quality model by Voronoi MQA"),
        (0.3, "Acceptable", "#3b82f6", "Reasonable model quality"),
        (0.2, "Marginal", "#f59e0b", "Below-average model quality"),
        (0, "Poor", "#ef4444", "Low-quality model"),
    ],
    "sequence_recovery": [
        (0.5, "High", "#22c55e", "Design closely matches natural sequence"),
        (0.3, "Moderate", "#3b82f6", "Some sequence similarity to natural"),
        (0.15, "Low", "#f59e0b", "Significant sequence divergence"),
        (0, "Very low", "#ef4444", "Minimal sequence recovery"),
    ],
    "ramachandran_favored": [
        (98, "Excellent", "#22c55e", "Nearly all residues in favored regions (>98%)"),
        (95, "Good", "#3b82f6", "Most residues in favored regions"),
        (90, "Concerning", "#f59e0b", "Too many residues outside favored regions"),
        (0, "Poor", "#ef4444", "Significant backbone geometry issues"),
    ],
    "ramachandran_allowed": [
        (99.8, "Excellent", "#22c55e", "Virtually no Ramachandran outliers"),
        (99, "Good", "#3b82f6", "Very few Ramachandran outliers"),
        (97, "Concerning", "#f59e0b", "Notable Ramachandran outliers present"),
        (0, "Poor", "#ef4444", "Many Ramachandran outliers; check backbone"),
    ],
    "sc_tm": [
        (0.9, "Excellent", "#22c55e", "Self-consistency: design folds as intended"),
        (0.7, "Good", "#3b82f6", "Design likely folds correctly"),
        (0.5, "Marginal", "#f59e0b", "Design may not fold as intended"),
        (0, "Fail", "#ef4444", "Design does not recapitulate target fold"),
    ],
    "pae_interaction": [
        (10, "High confidence", "#22c55e", "Strong interface confidence (PAE < 10)"),
        (20, "Moderate", "#3b82f6", "Reasonable interface prediction"),
        (25, "Low", "#f59e0b", "Interface may be incorrectly predicted"),
        (0, "N/A", "#94a3b8", "Insufficient data"),
    ],
    "mpnn_score": [
        (0.8, "Excellent", "#22c55e", "High ProteinMPNN log-likelihood; favorable design"),
        (0.5, "Good", "#3b82f6", "Acceptable design score"),
        (0.3, "Moderate", "#f59e0b", "Design may have suboptimal packing"),
        (0, "Poor", "#ef4444", "Low confidence in sequence-structure compatibility"),
    ],
    "packing_density": [
        (0.7, "Well-packed", "#22c55e", "Dense hydrophobic core"),
        (0.5, "Acceptable", "#3b82f6", "Reasonable interior packing"),
        (0.3, "Loose", "#f59e0b", "Under-packed core; may reduce stability"),
        (0, "Hollow", "#ef4444", "Core cavities present; redesign recommended"),
    ],
    # ── McGuffin Lab / ModFOLD-family metrics ────────────────────────
    "modfold_global": [
        (0.5, "High confidence", "#22c55e", "ModFOLD global score > 0.5; model likely correct"),
        (0.3, "Moderate", "#3b82f6", "ModFOLD global score moderate; some uncertainty"),
        (0.15, "Low", "#f59e0b", "ModFOLD score below reliable threshold"),
        (0, "Very low", "#ef4444", "ModFOLD indicates unreliable model"),
    ],
    "modfold9_score": [
        (0.5, "High confidence", "#22c55e", "ModFOLD9: high-quality model (CASP standard)"),
        (0.3, "Moderate", "#3b82f6", "ModFOLD9: reasonable model quality"),
        (0.15, "Low", "#f59e0b", "ModFOLD9: below-average quality"),
        (0, "Very low", "#ef4444", "ModFOLD9: unreliable model"),
    ],
    "modfold_pvalue": [
        (0.99, "Significant", "#22c55e", "ModFOLD p-value indicates significant model (1-p shown)"),
        (0.95, "Good", "#3b82f6", "ModFOLD p-value < 0.05; likely correct"),
        (0.9, "Marginal", "#f59e0b", "ModFOLD p-value 0.05-0.1; borderline significance"),
        (0, "Not significant", "#ef4444", "ModFOLD p-value > 0.1; model unreliable"),
    ],
    "modfold_confidence": [
        (0.8, "High", "#22c55e", "ModFOLD confidence high; proceed with downstream analysis"),
        (0.5, "Moderate", "#3b82f6", "ModFOLD confidence moderate; interpret with caution"),
        (0.3, "Low", "#f59e0b", "ModFOLD confidence low; consider refinement via ReFOLD"),
        (0, "Very low", "#ef4444", "ModFOLD confidence very low; discard or re-predict"),
    ],
    "modfold_local_score": [
        (0.7, "Accurate", "#22c55e", "Per-residue ModFOLD score: near-native local accuracy"),
        (0.5, "Reasonable", "#3b82f6", "Per-residue: acceptable local accuracy"),
        (0.3, "Uncertain", "#f59e0b", "Per-residue: significant local uncertainty"),
        (0, "Unreliable", "#ef4444", "Per-residue: local structure likely incorrect"),
    ],
    "modfoldock_qscore": [
        (0.7, "Excellent", "#22c55e", "ModFOLDdock2 QSCORE: excellent interface quality"),
        (0.5, "Good", "#3b82f6", "ModFOLDdock2 QSCORE: reasonable interface"),
        (0.3, "Marginal", "#f59e0b", "ModFOLDdock2 QSCORE: interface may have errors"),
        (0, "Poor", "#ef4444", "ModFOLDdock2 QSCORE: interface prediction unreliable"),
    ],
    "modfoldock_interface": [
        (0.7, "High quality", "#22c55e", "ModFOLDdock2: interface residues accurately predicted"),
        (0.5, "Acceptable", "#3b82f6", "ModFOLDdock2: reasonable interface prediction"),
        (0.3, "Low", "#f59e0b", "ModFOLDdock2: interface accuracy below reliable threshold"),
        (0, "Poor", "#ef4444", "ModFOLDdock2: interface prediction failed"),
    ],
    "oligo_lddt": [
        (0.7, "Excellent", "#22c55e", "Oligomeric lDDT high; multimer contacts well predicted"),
        (0.5, "Good", "#3b82f6", "Oligomeric lDDT reasonable; interface mostly correct"),
        (0.3, "Moderate", "#f59e0b", "Oligomeric lDDT moderate; interface may have errors"),
        (0, "Poor", "#ef4444", "Oligomeric lDDT low; multimer assembly likely incorrect"),
    ],
    "multifold_score": [
        (0.5, "High confidence", "#22c55e", "MultiFOLD2 prediction: high-quality structure"),
        (0.3, "Moderate", "#3b82f6", "MultiFOLD2: reasonable prediction quality"),
        (0.15, "Low", "#f59e0b", "MultiFOLD2: below-average prediction"),
        (0, "Very low", "#ef4444", "MultiFOLD2: unreliable prediction"),
    ],
    "intfold_score": [
        (0.5, "High confidence", "#22c55e", "IntFOLD7: high-quality integrated prediction"),
        (0.3, "Moderate", "#3b82f6", "IntFOLD7: reasonable quality"),
        (0.15, "Low", "#f59e0b", "IntFOLD7: below-average prediction"),
        (0, "Very low", "#ef4444", "IntFOLD7: unreliable prediction"),
    ],
}

_LOWER_IS_BETTER = {
    "rmsd": [
        (2.0, "Excellent", "#22c55e", "Near-identical to reference (< 2 Å)"),
        (4.0, "Good", "#3b82f6", "Reasonable structural agreement"),
        (8.0, "Moderate", "#f59e0b", "Significant deviations from reference"),
        (999, "Poor", "#ef4444", "Very different from reference"),
    ],
    "clash_score": [
        (10, "Excellent", "#22c55e", "Minimal steric clashes; well-packed (MolProbity standard)"),
        (30, "Acceptable", "#3b82f6", "Some clashes; may need refinement"),
        (60, "Concerning", "#f59e0b", "Many clashes; refinement recommended"),
        (9999, "Poor", "#ef4444", "Severe steric problems"),
    ],
    "rotamer_outliers": [
        (1, "Excellent", "#22c55e", "Very few side-chain outliers (< 1%)"),
        (5, "Acceptable", "#3b82f6", "Some side-chain issues"),
        (15, "Concerning", "#f59e0b", "Many rotamer outliers; check side chains"),
        (9999, "Poor", "#ef4444", "Severe side-chain geometry problems"),
    ],
    "c_beta_deviations": [
        (0, "Perfect", "#22c55e", "No C-beta deviations detected"),
        (2, "Good", "#3b82f6", "Minimal C-beta deviations"),
        (10, "Concerning", "#f59e0b", "Several C-beta deviations; check model"),
        (9999, "Poor", "#ef4444", "Many C-beta deviations; backbone errors likely"),
    ],
    "contact_energy": [
        (-100, "Favorable", "#22c55e", "Strong favorable contacts"),
        (-30, "Acceptable", "#3b82f6", "Reasonable contact network"),
        (0, "Neutral", "#f59e0b", "Weak contact network"),
        (9999, "Unfavorable", "#ef4444", "Unfavorable contact energetics"),
    ],
    "pae": [
        (5, "Excellent", "#22c55e", "Very high confidence in relative positions (PAE < 5 Å)"),
        (10, "Good", "#3b82f6", "Reliable inter-residue distance predictions"),
        (20, "Moderate", "#f59e0b", "Moderate uncertainty; some domain positions may be wrong"),
        (9999, "Poor", "#ef4444", "High positional uncertainty; relative arrangement unreliable"),
    ],
    "max_pae": [
        (10, "Excellent", "#22c55e", "Even worst regions have reasonable confidence"),
        (20, "Acceptable", "#3b82f6", "Some regions with lower confidence"),
        (25, "Concerning", "#f59e0b", "Regions with very low confidence present"),
        (9999, "Poor", "#ef4444", "Extremely low confidence in some regions"),
    ],
}

# Metrics where interpretation depends on context (not simply higher/lower)
_NEUTRAL_METRICS = {
    "sasa": "Solvent-accessible surface area in Å². Depends on protein size; ~100 Å² per residue is typical.",
    "interface_bsa": "Buried surface area at interface. > 1000 Å² = significant interface; > 2000 Å² = extensive.",
    "salt_bridges": "Number of salt bridges. Each contributes ~1-5 kcal/mol to stability depending on context.",
    "disorder_fraction": "Fraction of predicted disordered residues. > 0.3 may indicate intrinsically disordered protein.",
    "openmm_gbsa": "Solvation free energy (kJ/mol). More negative = more favorable. Best for relative comparisons.",
    "rosetta_energy": "Rosetta energy score (REU). Normalise per residue: < -2 REU/res = well-folded.",
    "molecular_weight_kda": "Molecular weight in kDa. Important for expression system selection.",
    "isoelectric_point": "Predicted isoelectric point (pI). Buffer pH should differ by > 1 unit for solubility.",
    "gravy": "Grand Average of Hydropathy. Positive = hydrophobic (membrane?), negative = hydrophilic (soluble).",
    "instability_index": "Values > 40 suggest the protein may be unstable in vitro (Guruprasad et al.).",
    "charge": "Net charge at pH 7. Highly charged proteins (|charge| > 20) may aggregate or bind non-specifically.",
    "num_residues": "Total number of amino acid residues in the protein chain.",
    "n_chains": "Number of polypeptide chains in the predicted structure.",
    "mpnn_temperature": "ProteinMPNN sampling temperature. 0.1 = conservative, 0.3 = moderate, 0.5+ = diverse.",
    "num_mutations": "Number of mutations introduced relative to wild-type sequence.",
    "ddg": "Predicted change in folding free energy (kcal/mol). Negative = stabilising. FoldX uncertainty ~1 kcal/mol.",
    "foldx_ddg": "FoldX-computed ddG (kcal/mol). < -1 = likely stabilising; > 1 = likely destabilising.",
    "conservation_score": "Evolutionary conservation at this position. High = functionally important, risky to mutate.",
    # ── McGuffin Lab neutral metrics ─────────────────────────────────
    "disorder_score": "DISOclust disorder probability per residue. > 0.5 = disordered. Combines ModFOLDclust + DISOPRED.",
    "funfold_confidence": "FunFOLD binding-site prediction confidence. Context-dependent; compare relative rankings.",
    "domain_boundary": "DomFOLD predicted domain boundary position. Check for multi-domain architecture.",
    "refold_improvement": "ReFOLD GDT-TS improvement after refinement. Positive = refinement helped.",
    "stoichiometry_prediction": "MultiFOLD2 predicted stoichiometry (e.g. homo-dimer, trimer). Compare with experimental data.",
}


def interpret_metric(name: str, value: float) -> Dict[str, Any]:
    """Interpret a metric value and return context.

    Returns a dict with keys:
        - label: quality label (e.g. "Excellent", "Poor")
        - color: hex color for UI display
        - description: plain-English interpretation
        - direction: "higher_better", "lower_better", or "neutral"
        - value: the original value
        - name: the metric name
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    # Check higher-is-better metrics
    if name_lower in _HIGHER_IS_BETTER:
        for threshold, label, color, desc in _HIGHER_IS_BETTER[name_lower]:
            if value >= threshold:
                return {
                    "label": label,
                    "color": color,
                    "description": desc,
                    "direction": "higher_better",
                    "value": value,
                    "name": name,
                }
        # Fallback
        entry = _HIGHER_IS_BETTER[name_lower][-1]
        return {"label": entry[1], "color": entry[2], "description": entry[3],
                "direction": "higher_better", "value": value, "name": name}

    # Check lower-is-better metrics
    if name_lower in _LOWER_IS_BETTER:
        for threshold, label, color, desc in _LOWER_IS_BETTER[name_lower]:
            if value <= threshold:
                return {
                    "label": label,
                    "color": color,
                    "description": desc,
                    "direction": "lower_better",
                    "value": value,
                    "name": name,
                }
        entry = _LOWER_IS_BETTER[name_lower][-1]
        return {"label": entry[1], "color": entry[2], "description": entry[3],
                "direction": "lower_better", "value": value, "name": name}

    # Neutral / context-dependent metrics
    desc = _NEUTRAL_METRICS.get(name_lower, f"Metric: {name}")
    return {
        "label": "N/A",
        "color": "#94a3b8",
        "description": desc,
        "direction": "neutral",
        "value": value,
        "name": name,
    }


def get_quality_color(name: str, value: float) -> str:
    """Return just the color for a metric value."""
    return interpret_metric(name, value)["color"]


def get_quality_label(name: str, value: float) -> str:
    """Return just the quality label for a metric value."""
    return interpret_metric(name, value)["label"]


def format_metric(name: str, value: float) -> str:
    """Format a metric value with appropriate precision."""
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    if name_lower in ("plddt",):
        return f"{value:.1f}"
    if name_lower in ("rmsd",):
        return f"{value:.2f}"
    if name_lower in ("clash_score", "contact_energy", "sasa", "interface_bsa"):
        return f"{value:.1f}"
    if name_lower in ("salt_bridges",):
        return f"{int(value)}"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def metric_context_html(name: str, value: float) -> str:
    """Return a small HTML snippet showing the metric with context coloring."""
    ctx = interpret_metric(name, value)
    formatted = format_metric(name, value)
    return (
        f'<span style="color:{ctx["color"]};font-weight:600;font-family:\'JetBrains Mono\',monospace">'
        f'{formatted}</span> '
        f'<span style="color:{ctx["color"]};font-size:.78rem;font-weight:500">'
        f'{ctx["label"]}</span>'
    )


def metric_badge_html(name: str, value: float) -> str:
    """Return a colored badge HTML for a metric."""
    ctx = interpret_metric(name, value)
    formatted = format_metric(name, value)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'background:rgba({_hex_to_rgb(ctx["color"])},0.12);'
        f'border:1px solid rgba({_hex_to_rgb(ctx["color"])},0.3);'
        f'color:{ctx["color"]};padding:3px 10px;border-radius:12px;'
        f'font-size:.8rem;font-weight:600">'
        f'{formatted} {ctx["label"]}</span>'
    )


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B' string."""
    h = hex_color.lstrip("#")
    return f"{int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


# ── Predictor metadata ──────────────────────────────────────────────

PREDICTOR_INFO = {
    "colabfold": {
        "name": "ColabFold",
        "icon": "🔬",
        "color": "#3b82f6",
        "desc": "AlphaFold2 with MSA",
        "speed": "Medium",
        "accuracy": "Excellent",
        "gpu": True,
    },
    "esmfold": {
        "name": "ESMFold",
        "icon": "🧬",
        "color": "#22c55e",
        "desc": "Single-sequence, fast",
        "speed": "Fast",
        "accuracy": "Good",
        "gpu": False,
    },
    "chai1": {
        "name": "Chai-1",
        "icon": "🧪",
        "color": "#a855f7",
        "desc": "Multi-modal diffusion",
        "speed": "Slow",
        "accuracy": "Excellent",
        "gpu": True,
    },
    "boltz2": {
        "name": "Boltz-2",
        "icon": "⚡",
        "color": "#f59e0b",
        "desc": "Fast diffusion model",
        "speed": "Medium",
        "accuracy": "Very Good",
        "gpu": True,
    },
    "esm3": {
        "name": "ESM3",
        "icon": "🌊",
        "color": "#06b6d4",
        "desc": "Next-gen language model",
        "speed": "Fast",
        "accuracy": "Good",
        "gpu": False,
    },
    # ── McGuffin Lab integrated servers ──────────────────────────────
    "intfold7": {
        "name": "IntFOLD7",
        "icon": "🏗️",
        "color": "#8b5cf6",
        "desc": "Integrated prediction + QA + function (Reading)",
        "speed": "Medium",
        "accuracy": "Excellent",
        "gpu": False,
        "url": "https://www.reading.ac.uk/bioinf/IntFOLD/",
    },
    "multifold2": {
        "name": "MultiFOLD2",
        "icon": "🔗",
        "color": "#ec4899",
        "desc": "Tertiary + quaternary + stoichiometry (CASP16 top)",
        "speed": "Medium",
        "accuracy": "Excellent",
        "gpu": False,
        "url": "https://www.reading.ac.uk/bioinf/MultiFOLD/",
    },
    "modfold9": {
        "name": "ModFOLD9",
        "icon": "✅",
        "color": "#14b8a6",
        "desc": "Model quality assessment (global + local + p-value)",
        "speed": "Fast",
        "accuracy": "Excellent",
        "gpu": False,
        "url": "https://www.reading.ac.uk/bioinf/ModFOLD/",
    },
    "modfoldock2": {
        "name": "ModFOLDdock2",
        "icon": "🤝",
        "color": "#f97316",
        "desc": "Multimer interface QA (CASP16 #1 for QSCORE)",
        "speed": "Fast",
        "accuracy": "Excellent",
        "gpu": False,
        "url": "https://www.reading.ac.uk/bioinf/ModFOLDdock/",
    },
    "refold3": {
        "name": "ReFOLD3",
        "icon": "🔧",
        "color": "#06b6d4",
        "desc": "Quality-guided model refinement",
        "speed": "Medium",
        "accuracy": "Good",
        "gpu": False,
        "url": "https://www.reading.ac.uk/bioinf/ReFOLD/",
    },
    "funfold5": {
        "name": "FunFOLD5",
        "icon": "🎯",
        "color": "#84cc16",
        "desc": "Protein-ligand binding site prediction",
        "speed": "Medium",
        "accuracy": "Good",
        "gpu": False,
        "url": "https://www.reading.ac.uk/bioinf/FunFOLD/",
    },
}


def get_predictor_info(predictor_id: str) -> Dict[str, Any]:
    """Get metadata for a predictor by ID."""
    return PREDICTOR_INFO.get(predictor_id.lower(), {
        "name": predictor_id,
        "icon": "🔮",
        "color": "#6366f1",
        "desc": predictor_id,
        "speed": "Unknown",
        "accuracy": "Unknown",
        "gpu": False,
    })
