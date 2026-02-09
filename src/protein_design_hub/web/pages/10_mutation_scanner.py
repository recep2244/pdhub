"""Interactive Mutation Scanner with ESMFold-based saturation mutagenesis.

This page provides:
1. Sequence input with auto ESMFold prediction
2. Interactive residue selection for mutation scanning
3. Automatic saturation mutagenesis (all 19 AA mutations)
4. Comprehensive metric calculation (pLDDT, RMSD, Clash Score, SASA)
5. Mutation ranking and recommendations
6. Side-by-side structure comparison
"""

import os
import sys
from pathlib import Path
import json
import tempfile
import time
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import plotly.graph_objects as go

PROJECT_SRC = Path(__file__).resolve().parents[3]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st

from protein_design_hub.analysis.mutation_scanner import (
    MutationScanner,
    SaturationMutagenesisResult,
    MutationResult,
    MultiMutationResult,
    MultiMutationVariant,
)
from protein_design_hub.io.afdb import AFDBClient, AFDBMatch, normalize_sequence
from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    section_header,
    info_box,
    metric_card_with_context,
    workflow_breadcrumb,
    cross_page_actions,
)
from protein_design_hub.web.agent_helpers import (
    render_agent_advice_panel,
    render_contextual_insight,
    agent_sidebar_status,
    render_all_experts_panel,
)
from datetime import datetime
from types import SimpleNamespace

st.set_page_config(
    page_title="Mutation Scanner - Protein Design Hub",
    page_icon="🔬",
    layout="wide"
)

# Base theme + navigation
inject_base_css()
sidebar_nav(current="Mutagenesis")
sidebar_system_status()
agent_sidebar_status()

# Workflow breadcrumb
page_header(
    "Mutation Scanner",
    "Saturation mutagenesis and multi-mutation design with stability analysis",
    "🧬",
)
workflow_breadcrumb(
    ["Predict Structure", "Evaluate", "Scan Mutations", "Design"],
    current=2,
)

with st.expander("📖 How mutation scanning works", expanded=False):
    st.markdown("""
**Saturation mutagenesis** tests every possible amino acid substitution at selected positions:
1. Enter your protein sequence (or use a predicted structure)
2. Select residue positions to scan (core residues for stability, surface for binding)
3. The scanner predicts structures for all 19 mutations at each position
4. Metrics (pLDDT, RMSD vs WT, clash score) rank stabilising vs destabilising mutations

**Multi-mutation design** combines beneficial single mutations into variants.

**Tips:**
- Start with 2-3 positions to test the workflow (scanning 5+ positions takes longer)
- **Buried positions** (low SASA) are riskier but can dramatically affect stability
- **Surface positions** are safer to mutate and good for engineering binding
- Mutations preserving pLDDT > 80 and RMSD < 1.0 Å are the safest candidates
    """)

# Enhanced CSS for mutation scanner interface
st.markdown("""
<style>
/* Main container */
.main .block-container {
    padding: 1rem 2rem;
    max-width: 100%;
}

/* Residue grid */
.residue-button {
    font-weight: bold;
    transition: all 0.2s ease;
}

/* Metric cards */
.metric-card {
    background: var(--pdhub-bg-card);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: var(--pdhub-shadow-sm);
    border: 1px solid var(--pdhub-border);
    margin-bottom: 10px;
    color: var(--pdhub-text);
}

.metric-card-success { border-left: 5px solid var(--pdhub-success); }
.metric-card-danger { border-left: 5px solid var(--pdhub-error); }

.metric-value { font-size: 1.5rem; font-weight: bold; color: var(--pdhub-text); }
.metric-delta { font-size: 1rem; font-weight: bold; }
.delta-positive { color: var(--pdhub-success); }
.delta-negative { color: var(--pdhub-error); }
</style>
""", unsafe_allow_html=True)


# ── LLM Expert Review Helpers ───────────────────────────────────────

def _coerce_float_list(values: List[Any]) -> List[float]:
    cleaned = []
    for v in values:
        try:
            cleaned.append(float(v))
        except Exception:
            cleaned.append(float("nan"))
    if cleaned and max(cleaned) <= 1.0:
        cleaned = [v * 100.0 for v in cleaned]
    return cleaned


def _summarize_confidence(
    sequence: str,
    values: List[Any],
    is_immunebuilder: bool,
    top_k: int = 12,
) -> Tuple[str, List[int]]:
    if not sequence or not values:
        return "", []
    cleaned = _coerce_float_list(values)
    pairs = []
    for i, v in enumerate(cleaned[: len(sequence)]):
        if v == v:  # not NaN
            pairs.append((i + 1, v))
    if not pairs:
        return "", []

    if is_immunebuilder:
        pairs.sort(key=lambda x: x[1], reverse=True)
        label = "Highest error positions (Å)"
        threshold = 5.0
        flagged = [pos for pos, v in pairs if v >= threshold]
    else:
        pairs.sort(key=lambda x: x[1])
        label = "Lowest-confidence positions (pLDDT)"
        threshold = 70.0
        flagged = [pos for pos, v in pairs if v <= threshold]

    top = pairs[:top_k]
    lines = [f"{sequence[p-1]}{p} ({v:.1f})" for p, v in top]
    return f"{label}: " + ", ".join(lines), flagged


def _format_baseline_summary(baseline_results: Dict[str, Dict[str, Any]], label_by_value: Dict[str, str]) -> str:
    lines = []
    for pred_id, data in baseline_results.items():
        name = label_by_value.get(pred_id, pred_id)
        mean = data.get("mean_plddt")
        runtime = data.get("runtime_seconds")
        status = "OK" if data.get("success") else "FAILED"
        mean_str = f"{mean:.2f}" if mean is not None else "N/A"
        runtime_str = f"{runtime:.1f}s" if runtime else "N/A"
        lines.append(f"- {name}: mean={mean_str}, runtime={runtime_str}, status={status}")
    return "\n".join(lines)


def _format_top_mutations(mutations: List[MutationResult], is_immunebuilder: bool, top_k: int = 8) -> str:
    if not mutations:
        return ""
    rows = []
    delta_label = "ΔpLDDT"
    if is_immunebuilder:
        delta_label = "ΔError"
    for m in mutations[:top_k]:
        parts = [f"{m.mutation_code}", f"{delta_label}={m.delta_mean_plddt:+.2f}"]
        if m.rmsd_to_base is not None:
            parts.append(f"RMSD={m.rmsd_to_base:.2f}Å")
        if m.tm_score_to_base is not None:
            parts.append(f"TM-score={m.tm_score_to_base:.2f}")
        if m.clash_score is not None:
            parts.append(f"Clash={m.clash_score:.1f}")
        if m.sasa_total is not None:
            parts.append(f"SASA={m.sasa_total:.0f}")
        ost_lddt = get_ost_global_metric(m, "lddt")
        ost_rmsd = get_ost_global_metric(m, "rmsd_ca")
        ost_qs = get_ost_global_metric(m, "qs_score")
        if ost_lddt is not None:
            parts.append(f"OST-lDDT={ost_lddt:.3f}")
        if ost_rmsd is not None:
            parts.append(f"OST-RMSD(CA)={ost_rmsd:.2f}Å")
        if ost_qs is not None:
            parts.append(f"OST-QS={ost_qs:.3f}")
        rows.append("- " + ", ".join(parts))
    return "\n".join(rows)


def _format_top_variants(variants: List[MultiMutationVariant], is_immunebuilder: bool, top_k: int = 6) -> str:
    if not variants:
        return ""
    delta_label = "ΔpLDDT"
    local_label = "Δlocal pLDDT"
    if is_immunebuilder:
        delta_label = "ΔError"
        local_label = "Δlocal error"
    rows = []
    for v in variants[:top_k]:
        parts = [
            f"{v.mutation_code}",
            f"{delta_label}={v.delta_mean_plddt:+.2f}",
            f"{local_label}={v.delta_local_plddt:+.2f}",
        ]
        if getattr(v, "rmsd_to_base", None) is not None:
            parts.append(f"RMSD={v.rmsd_to_base:.2f}Å")
        if getattr(v, "tm_score_to_base", None) is not None:
            parts.append(f"TM-score={v.tm_score_to_base:.2f}")
        ost_lddt = get_ost_global_metric(v, "lddt")
        ost_rmsd = get_ost_global_metric(v, "rmsd_ca")
        ost_qs = get_ost_global_metric(v, "qs_score")
        if ost_lddt is not None:
            parts.append(f"OST-lDDT={ost_lddt:.3f}")
        if ost_rmsd is not None:
            parts.append(f"OST-RMSD(CA)={ost_rmsd:.2f}Å")
        if ost_qs is not None:
            parts.append(f"OST-QS={ost_qs:.3f}")
        rows.append("- " + ", ".join(parts))
    return "\n".join(rows)


def _format_base_eval_summary(base_eval: Dict[str, Any]) -> str:
    """Compact one-line summary of baseline structure evaluation."""
    result = base_eval.get("result", {})
    parts = []
    for key, label, fmt in [
        ("clash_score", "clash", "{:.1f}"),
        ("sasa_total", "SASA", "{:.0f}"),
        ("voromqa_score", "VoroMQA", "{:.3f}"),
        ("cad_score", "CAD", "{:.3f}"),
        ("openmm_potential_energy_kj_mol", "OpenMM", "{:.1f}"),
        ("contact_energy", "contact_E", "{:.1f}"),
        ("disorder_fraction", "disorder", "{:.2%}"),
    ]:
        val = result.get(key)
        if isinstance(val, (int, float)):
            parts.append(f"{label}={fmt.format(float(val))}")
    return ", ".join(parts) if parts else "No baseline metrics available."


def run_baseline_structure_evaluation(
    structure_path: Path,
    metrics: List[str],
) -> Dict[str, Any]:
    """Evaluate a baseline structure with selected no-reference metrics."""
    from protein_design_hub.evaluation.composite import CompositeEvaluator

    evaluator = CompositeEvaluator(metrics=metrics)
    result = evaluator.evaluate(structure_path, reference_path=None)
    return {
        "structure_path": str(structure_path),
        "selected_metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "result": result.to_dict(),
    }


# Amino acid data
AMINO_ACIDS = {
    'A': {'name': 'Alanine', 'code': 'Ala'}, 'C': {'name': 'Cysteine', 'code': 'Cys'},
    'D': {'name': 'Aspartate', 'code': 'Asp'}, 'E': {'name': 'Glutamate', 'code': 'Glu'},
    'F': {'name': 'Phenylalanine', 'code': 'Phe'}, 'G': {'name': 'Glycine', 'code': 'Gly'},
    'H': {'name': 'Histidine', 'code': 'His'}, 'I': {'name': 'Isoleucine', 'code': 'Ile'},
    'K': {'name': 'Lysine', 'code': 'Lys'}, 'L': {'name': 'Leucine', 'code': 'Leu'},
    'M': {'name': 'Methionine', 'code': 'Met'}, 'N': {'name': 'Asparagine', 'code': 'Asn'},
    'P': {'name': 'Proline', 'code': 'Pro'}, 'Q': {'name': 'Glutamine', 'code': 'Gln'},
    'R': {'name': 'Arginine', 'code': 'Arg'}, 'S': {'name': 'Serine', 'code': 'Ser'},
    'T': {'name': 'Threonine', 'code': 'Thr'}, 'V': {'name': 'Valine', 'code': 'Val'},
    'W': {'name': 'Tryptophan', 'code': 'Trp'}, 'Y': {'name': 'Tyrosine', 'code': 'Tyr'},
}

# Handle external job loading
if st.session_state.get("scan_job_to_load"):
    try:
        job_path = Path(st.session_state["scan_job_to_load"])
        summary_path = job_path / "scan_results.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)

            # Reconstruct dummy object for UI compatibility
            if "positions" in data and "variants" in data:
                res = SimpleNamespace(**data)
                res.variants = [SimpleNamespace(**v) for v in data.get("variants", [])]
                res.ranked_variants = sorted(
                    [v for v in res.variants if v.success],
                    key=lambda v: v.improvement_score if hasattr(v, "improvement_score") else 0,
                    reverse=True,
                )
                st.session_state.multi_scan_results = res
                st.session_state.selected_positions = set(res.positions or [])
                st.session_state.sequence = data.get("sequence", "")
                st.success(f"Successfully loaded multi-scan: {job_path.name}")
            else:
                res = SimpleNamespace(**data)
                res.mutations = [SimpleNamespace(**m) for m in data.get("mutations", [])]
                res.ranked_mutations = sorted(
                    [m for m in res.mutations if m.success],
                    key=lambda x: x.improvement_score if hasattr(x, 'improvement_score') else 0,
                    reverse=True
                )
                st.session_state.scan_results = res
                st.session_state.sequence = data.get("sequence", "")
                st.session_state.selected_position = data.get("position")
                st.success(f"Successfully loaded scan: {job_path.name}")
        st.session_state.pop("scan_job_to_load")
    except Exception as e:
        st.error(f"Error loading job: {e}")

def init_session_state():
    default_eval_metrics = ["openmm_gbsa", "cad_score", "voromqa"]
    defaults = {
        'sequence': '', 'sequence_name': 'my_protein',
        'sequence_input_raw': '',
        'base_structure': None, 'base_plddt': None, 'base_plddt_per_residue': None,
        'base_structure_path': None,
        'selected_position': None, 'scan_results': None,
        'selected_positions': set(), 'multi_scan_results': None,
        'comparison_mutation': None,
        'multi_comparison_variant': None,
        'show_advanced_predictors': False,
        'immunebuilder_mode': 'antibody',
        'immune_chain_a': None,
        'immune_chain_b': None,
        'immune_active_chain': 'A',
        'immune_parse_error': None,
        'mutation_predictor': 'esmfold_api',
        'mutation_eval_enabled': True,
        'mutation_eval_metrics': default_eval_metrics,
        'mutation_ost_comprehensive': False,
        'baseline_predictors': ['esmfold_api'],
        'baseline_results': None,
        'baseline_sequence': None,
        'baseline_evaluation': None,
        'baseline_evaluation_predictor': None,
        'mut_review_provider': "current",
        'mut_review_model': "",
        'mut_review_custom_provider': "",
        'scanner': MutationScanner(
            predictor='esmfold_api',
            evaluation_metrics=default_eval_metrics,
            run_openstructure_comprehensive=False,
        ),
        'afdb_enabled': False,
        'afdb_email': os.getenv("EBI_EMAIL", ""),
        'afdb_cache': {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


def _expert_review_overrides() -> Tuple[str, str]:
    """Return provider/model overrides for expert panels on this page."""
    mode = st.session_state.get("mut_review_provider", "current")
    model = (st.session_state.get("mut_review_model") or "").strip()
    custom_provider = (st.session_state.get("mut_review_custom_provider") or "").strip()

    if mode == "current":
        return "", model
    if mode == "custom":
        return custom_provider, model
    if mode == "ollama":
        return "ollama", model or "qwen2.5:14b"
    if mode == "deepseek":
        return "deepseek", model or "deepseek-chat"
    return mode, model

def current_eval_metrics() -> List[str]:
    if not st.session_state.get("mutation_eval_enabled"):
        return []
    return st.session_state.get("mutation_eval_metrics", [])

def build_scanner(predictor_id: str) -> MutationScanner:
    eval_metrics = current_eval_metrics()
    run_ost = bool(st.session_state.get("mutation_ost_comprehensive", False))
    if predictor_id == "immunebuilder":
        return MutationScanner(
            predictor=predictor_id,
            immunebuilder_mode=st.session_state.immunebuilder_mode,
            immune_chain_a=st.session_state.immune_chain_a,
            immune_chain_b=st.session_state.immune_chain_b,
            immune_active_chain=st.session_state.immune_active_chain,
            evaluation_metrics=eval_metrics,
            run_openstructure_comprehensive=run_ost,
        )
    return MutationScanner(
        predictor=predictor_id,
        evaluation_metrics=eval_metrics,
        run_openstructure_comprehensive=run_ost,
    )

# Optional AFDB lookup in sidebar
with st.sidebar.expander("🔍 AFDB Match", expanded=False):
    afdb_enabled = st.checkbox(
        "Fetch related AFDB structure (>=90% identity & coverage)",
        value=st.session_state.get("afdb_enabled", False),
        help="Uses EBI BLAST against UniProt and fetches AFDB if a close match exists.",
    )
    afdb_email = st.text_input(
        "EBI email (required by BLAST)",
        value=st.session_state.get("afdb_email", os.getenv("EBI_EMAIL", "")),
        help="EBI asks for an email address when submitting BLAST jobs.",
    )
    st.caption("AFDB lookup can take ~1-2 minutes depending on sequence length.")
    st.session_state.afdb_enabled = afdb_enabled
    st.session_state.afdb_email = afdb_email.strip()

def run_saturation_mutagenesis(sequence, position):
    """Run saturation mutagenesis using the backend scanner."""
    with st.status("Running Mutation Scanner...", expanded=True) as status:
        st.write("Initializing...")
        
        def progress_callback(current, total, message):
            st.write(f"Testing {message}...")
        
        scanner = st.session_state.scanner
        results = scanner.scan_position(
            sequence, 
            position, 
            progress_callback=progress_callback
        )
        
        status.update(label="Scan Complete!", state="complete", expanded=False)
        return results

def parse_positions(text, max_len):
    positions = set()
    if not text:
        return positions
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if start.isdigit() and end.isdigit():
                for p in range(int(start), int(end) + 1):
                    if 1 <= p <= max_len:
                        positions.add(p)
        elif part.isdigit():
            p = int(part)
            if 1 <= p <= max_len:
                positions.add(p)
    return positions

def parse_ab_fasta(text: str) -> Dict[str, str]:
    chains: Dict[str, str] = {}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].strip().upper()
            if header:
                current = header[0]
                if current not in {"A", "B"}:
                    current = None
            continue
        if current in {"A", "B"}:
            seq = "".join(c for c in line.upper() if c in AMINO_ACIDS)
            chains[current] = chains.get(current, "") + seq
    if "A" not in chains or "B" not in chains:
        raise ValueError("FASTA must include >A and >B chains for ImmuneBuilder.")
    return chains

def get_extra_metric(obj: Any, metric_name: str, field: str) -> Optional[float]:
    extra = getattr(obj, "extra_metrics", None) or {}
    metric = extra.get(metric_name)
    if isinstance(metric, dict):
        val = metric.get(field)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def get_ost_global_metric(obj: Any, field: str) -> Optional[float]:
    """Read one global OpenStructure metric from obj.extra_metrics."""
    extra = getattr(obj, "extra_metrics", None) or {}
    ost = extra.get("ost_comprehensive")
    if not isinstance(ost, dict):
        return None
    global_metrics = ost.get("global")
    if not isinstance(global_metrics, dict):
        return None
    val = global_metrics.get(field)
    if isinstance(val, (int, float)):
        return float(val)
    return None

def get_afdb_match_cached(sequence: str, email: str) -> Tuple[Optional[AFDBMatch], Optional[str]]:
    if not sequence:
        return None, None
    cache = st.session_state.setdefault("afdb_cache", {})
    client = AFDBClient()
    cache_key = client.cache_key(sequence)
    cached = cache.get(cache_key)
    if cached:
        if isinstance(cached, dict) and cached.get("error"):
            return None, cached.get("error")
        try:
            return AFDBMatch.from_dict(cached), None
        except Exception as exc:
            return None, str(exc)

    match, error = client.find_match(
        sequence,
        min_identity=90.0,
        min_coverage=90.0,
        email=email,
    )
    if match:
        cache[cache_key] = match.to_dict()
    else:
        cache[cache_key] = {"error": error}
    return match, error

def run_multi_mutation_pipeline(sequence, positions, top_k, max_variants, only_beneficial=True, max_positions=6):
    """Run multi-position mutation pipeline with ESMFold evaluation."""
    with st.status("Running Multi-Mutation Pipeline...", expanded=True) as status:
        def progress_callback(stage, current, total, message):
            st.write(f"[{stage}] {message} ({current}/{total})")

        scanner = st.session_state.scanner
        results = scanner.scan_positions(
            sequence,
            positions,
            top_k=top_k,
            max_variants=max_variants,
            max_positions=max_positions,
            only_beneficial=only_beneficial,
            progress_callback=progress_callback,
        )

        status.update(label="Multi-scan Complete!", state="complete", expanded=False)
        return results

def render_heatmap(results):
    mutations = results.mutations
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    original_aa = results.original_aa
    is_immunebuilder = getattr(results, "predictor", "") == "immunebuilder"
    
    values = []
    hover_texts = []
    colors = []
    
    for aa in aa_order:
        if aa == original_aa:
            values.append(0)
            colors.append('#9ca3af')
            hover_texts.append(f"{aa} (WT)")
        else:
            mut = next((m for m in mutations if m.mutant_aa == aa), None)
            if mut and mut.success:
                delta = mut.delta_mean_plddt
                values.append(delta)
                colors.append('#22c55e' if delta > 0 else '#ef4444')
                delta_label = "ΔpLDDT"
                if is_immunebuilder:
                    delta_label = "ΔError"
                hover_texts.append(
                    f"<b>{mut.mutation_code}</b><br>"
                    f"{delta_label}: {delta:+.2f}"
                    + (f"<br>RMSD: {mut.rmsd_to_base:.2f} Å" if mut.rmsd_to_base else "")
                )
            else:
                values.append(None)
                colors.append('#6b7280')
                hover_texts.append("Failed")
                
    fig = go.Figure(data=go.Bar(
        x=aa_order, y=values, marker_color=colors,
        hovertext=hover_texts, hoverinfo='text'
    ))
    y_title = "ΔpLDDT"
    if is_immunebuilder:
        y_title = "ΔError (Å)"
    fig.update_layout(
        title=f"Mutation Stability ({y_title}) at {results.original_aa}{results.position}",
        yaxis_title=y_title, height=350
    )
    st.plotly_chart(fig, use_container_width=True)

def run_baseline_comparison(
    sequence: str,
    predictors: List[str],
    immunebuilder_mode: str,
    immune_chain_a: Optional[str],
    immune_chain_b: Optional[str],
    immune_active_chain: str,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    with st.status("Running baseline comparison...", expanded=True) as status:
        for pred in predictors:
            st.write(f"🔬 {pred}...")
            if pred == "immunebuilder":
                scanner = MutationScanner(
                    predictor=pred,
                    immunebuilder_mode=immunebuilder_mode,
                    immune_chain_a=immune_chain_a,
                    immune_chain_b=immune_chain_b,
                    immune_active_chain=immune_active_chain,
                )
            else:
                scanner = MutationScanner(predictor=pred)
            try:
                start = time.time()
                pdb, plddt, path = scanner.predict_single(sequence, f"baseline_{pred}")
                runtime = time.time() - start
                mean_plddt = sum(plddt) / len(plddt) if plddt else 0.0
                results[pred] = {
                    "mean_plddt": mean_plddt,
                    "runtime_seconds": runtime,
                    "throughput": (1.0 / runtime) if runtime > 0 else None,
                    "structure_path": str(path) if path else None,
                    "success": True,
                }
            except Exception as e:
                results[pred] = {
                    "mean_plddt": None,
                    "runtime_seconds": None,
                    "throughput": None,
                    "structure_path": None,
                    "success": False,
                    "error": str(e),
                }
        status.update(label="Baseline comparison complete", state="complete", expanded=False)
    return results

# Predictor selection
show_advanced = st.checkbox(
    "Show advanced predictors",
    value=st.session_state.get("show_advanced_predictors", False),
    help="Show local ESMFold variants and ESM3 (requires separate environments).",
)
st.session_state.show_advanced_predictors = show_advanced

predictor_options = {
    "ESM1 (legacy ESMFold v0)": "esmfold_v0",
    "ESMFold2 (ESM-2, local)": "esmfold_v1",
    "ESMFold API (ESM-2, <=400 aa)": "esmfold_api",
    "ESM3 (local or Forge)": "esm3",
    "ImmuneBuilder (antibody/nanobody/TCR)": "immunebuilder",
}
label_by_value = {v: k for k, v in predictor_options.items()}

baseline_option_labels = ["ESMFold API (ESM-2, <=400 aa)"]
if show_advanced:
    baseline_option_labels += [
        "ESM1 (legacy ESMFold v0)",
        "ESMFold2 (ESM-2, local)",
        "ESM3 (local or Forge)",
        "ImmuneBuilder (antibody/nanobody/TCR)",
    ]

# Baseline comparison (tick-box style)
baseline_defaults = [
    label_by_value[p] for p in st.session_state.get("baseline_predictors", []) if p in label_by_value
]
baseline_defaults = [label for label in baseline_defaults if label in baseline_option_labels]
baseline_labels = st.multiselect(
    "Baseline predictors (tick to compare)",
    options=baseline_option_labels,
    default=baseline_defaults or ["ESMFold API (ESM-2, <=400 aa)"],
    help="Run base-structure predictions for comparison (pLDDT + runtime).",
)
st.caption("Note: the ESM1 baseline uses legacy ESMFold v0 for structure generation.")
baseline_predictors = [predictor_options[label] for label in baseline_labels]
if baseline_predictors != st.session_state.get("baseline_predictors"):
    st.session_state.baseline_predictors = baseline_predictors

if st.button("🧪 Run Baseline Comparison", use_container_width=True, disabled=not baseline_predictors):
    if not st.session_state.sequence:
        st.warning("Provide a sequence first.")
    elif "immunebuilder" in baseline_predictors and not (
        st.session_state.immune_chain_a and st.session_state.immune_chain_b
    ):
        st.error("ImmuneBuilder requires FASTA with >A and >B chains before running baseline.")
    else:
        st.session_state.baseline_results = run_baseline_comparison(
            st.session_state.sequence,
            baseline_predictors,
            st.session_state.immunebuilder_mode,
            st.session_state.immune_chain_a,
            st.session_state.immune_chain_b,
            st.session_state.immune_active_chain,
        )
        st.session_state.baseline_sequence = st.session_state.sequence
        st.session_state.baseline_evaluation = None
        st.session_state.baseline_evaluation_predictor = None

# Expert-review backend override for all mutagenesis meetings on this page
with st.expander("🧠 Expert Review Backend (optional second opinion)", expanded=False):
    try:
        from protein_design_hub.core.config import LLM_PROVIDER_PRESETS

        provider_options = ["current", "ollama", "deepseek"] + [
            p for p in LLM_PROVIDER_PRESETS.keys()
            if p not in {"ollama", "deepseek"}
        ] + ["custom"]
        provider_default_model = {
            provider: preset[1] for provider, preset in LLM_PROVIDER_PRESETS.items()
        }
    except Exception:
        provider_options = ["current", "ollama", "deepseek", "custom"]
        provider_default_model = {
            "ollama": "qwen2.5:14b",
            "deepseek": "deepseek-chat",
        }

    selected_provider = st.session_state.get("mut_review_provider", "current")
    if selected_provider not in provider_options:
        selected_provider = "current"

    st.session_state.mut_review_provider = st.selectbox(
        "Expert panel provider",
        options=provider_options,
        index=provider_options.index(selected_provider),
        format_func=lambda x: {
            "current": "Current configured provider",
            "ollama": "Ollama (local, recommended: qwen2.5:14b)",
            "deepseek": "DeepSeek (secondary cloud check)",
            "custom": "Custom provider/model",
        }.get(x, x),
        key="mut_review_provider_select",
    )

    current_override = st.session_state.mut_review_provider
    if current_override == "custom":
        st.session_state.mut_review_custom_provider = st.text_input(
            "Custom provider ID",
            value=st.session_state.get("mut_review_custom_provider", ""),
            key="mut_review_custom_provider_input",
            help="Example: openrouter, custom, or another provider preset name.",
        ).strip()
        st.session_state.mut_review_model = st.text_input(
            "Custom model ID",
            value=st.session_state.get("mut_review_model", ""),
            key="mut_review_model_input",
            help="Leave empty to use provider default if available.",
        )
        st.caption(
            "Set provider/model here only for mutagenesis expert panels. "
            "This does not change global app configuration."
        )
    elif current_override == "current":
        st.session_state.mut_review_custom_provider = ""
        st.session_state.mut_review_model = st.text_input(
            "Model override (optional)",
            value=st.session_state.get("mut_review_model", ""),
            key="mut_review_model_input",
            help="Leave empty to keep the globally configured model.",
        ).strip()
    else:
        st.session_state.mut_review_custom_provider = ""
        suggested_model = provider_default_model.get(current_override, "")
        st.session_state.mut_review_model = st.text_input(
            "Model (optional)",
            value=st.session_state.get("mut_review_model", suggested_model) or suggested_model,
            key="mut_review_model_input",
            help="Leave as default unless you need a specific model.",
        ).strip()
        if current_override == "deepseek":
            st.caption("Requires `DEEPSEEK_API_KEY` in the environment.")

if st.session_state.get("baseline_results"):
    if st.session_state.get("baseline_sequence") != st.session_state.sequence:
        st.warning("Baseline results were computed for a different sequence. Re-run for current sequence.")
    baseline_rows = []
    for pred_id, data in st.session_state.baseline_results.items():
        metric_label = "Mean pLDDT"
        if pred_id == "immunebuilder":
            metric_label = "Mean error (Å)"
        baseline_rows.append(
            {
                "Predictor": label_by_value.get(pred_id, pred_id),
                "Metric": metric_label,
                "Mean": data.get("mean_plddt"),
                "Runtime (s)": data.get("runtime_seconds"),
                "Throughput (structures/s)": data.get("throughput"),
                "Status": "OK" if data.get("success") else "FAILED",
                "Error": data.get("error", ""),
            }
        )
    st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True)

    successful_structures: Dict[str, Path] = {}
    for pred_id, data in st.session_state.baseline_results.items():
        sp = data.get("structure_path")
        if data.get("success") and sp:
            p = Path(sp)
            if p.exists():
                successful_structures[pred_id] = p

    # Action: evaluate selected baseline structure with no-reference metrics.
    if successful_structures:
        st.markdown("#### Baseline Structure Evaluation")
        st.caption("Evaluate a baseline structure before mutagenesis to guide residue selection.")

        eval_col1, eval_col2 = st.columns([2, 3])
        with eval_col1:
            pred_options = list(successful_structures.keys())
            default_pred = st.session_state.get("baseline_evaluation_predictor")
            if default_pred not in pred_options:
                default_pred = pred_options[0]
            eval_pred = st.selectbox(
                "Baseline structure to evaluate",
                options=pred_options,
                index=pred_options.index(default_pred),
                format_func=lambda p: label_by_value.get(p, p),
                key="baseline_eval_predictor",
            )
        with eval_col2:
            try:
                from protein_design_hub.evaluation.composite import CompositeEvaluator

                metric_info = CompositeEvaluator.list_all_metrics()
                available_no_ref = [
                    m["name"] for m in metric_info
                    if m.get("available") and not m.get("requires_reference")
                ]
            except Exception:
                available_no_ref = []

            default_eval_metrics = [
                m for m in ["clash_score", "sasa", "voromqa", "cad_score", "openmm_gbsa", "contact_energy", "disorder"]
                if m in available_no_ref
            ]
            selected_baseline_metrics = st.multiselect(
                "Metrics",
                options=available_no_ref,
                default=default_eval_metrics,
                key="baseline_eval_metrics",
                help="Reference-free metrics only. Use this to triage structural quality before mutagenesis.",
            )

        run_eval_btn = st.button(
            "📊 Evaluate Baseline Structure",
            use_container_width=True,
            type="primary",
            disabled=not successful_structures or not selected_baseline_metrics,
        )
        if run_eval_btn:
            try:
                with st.spinner("Evaluating baseline structure..."):
                    payload = run_baseline_structure_evaluation(
                        successful_structures[eval_pred],
                        selected_baseline_metrics,
                    )
                st.session_state.baseline_evaluation = payload
                st.session_state.baseline_evaluation_predictor = eval_pred
                st.success("Baseline evaluation completed.")
            except Exception as exc:
                st.error(f"Baseline evaluation failed: {exc}")

    base_eval = st.session_state.get("baseline_evaluation")
    if base_eval:
        eval_pred = st.session_state.get("baseline_evaluation_predictor")
        eval_pred_label = label_by_value.get(eval_pred, eval_pred) if eval_pred else "baseline"
        st.markdown(f"#### Baseline Evaluation Results ({eval_pred_label})")
        eval_result = base_eval.get("result", {})

        met1, met2, met3, met4 = st.columns(4)
        with met1:
            if eval_result.get("clash_score") is not None:
                metric_card_with_context(
                    f"{float(eval_result['clash_score']):.1f}",
                    "Clash Score",
                    "Lower is better (<10 ideal)",
                    status="success" if float(eval_result["clash_score"]) < 10 else "warning",
                    icon="💥",
                )
        with met2:
            if eval_result.get("voromqa_score") is not None:
                metric_card_with_context(
                    f"{float(eval_result['voromqa_score']):.3f}",
                    "VoroMQA",
                    "Higher is better (>0.40 good)",
                    status="success" if float(eval_result["voromqa_score"]) > 0.40 else "warning",
                    icon="📏",
                )
        with met3:
            if eval_result.get("cad_score") is not None:
                metric_card_with_context(
                    f"{float(eval_result['cad_score']):.3f}",
                    "CAD-score",
                    "Higher indicates better local agreement",
                    status="info",
                    icon="🧩",
                )
        with met4:
            if eval_result.get("sasa_total") is not None:
                metric_card_with_context(
                    f"{float(eval_result['sasa_total']):.0f}",
                    "SASA (A^2)",
                    "Surface exposure",
                    status="default",
                    icon="🌊",
                )

        rows = []
        for field, label in [
            ("clash_score", "Clash score"),
            ("clash_count", "Clash count"),
            ("sasa_total", "SASA (A^2)"),
            ("voromqa_score", "VoroMQA"),
            ("cad_score", "CAD-score"),
            ("contact_energy", "Contact energy"),
            ("openmm_potential_energy_kj_mol", "OpenMM potential (kJ/mol)"),
            ("openmm_gbsa_energy_kj_mol", "OpenMM GBSA (kJ/mol)"),
            ("disorder_fraction", "Disorder fraction"),
        ]:
            val = eval_result.get(field)
            if val is not None:
                rows.append({"Metric": label, "Value": val})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        errors = (eval_result.get("metadata") or {}).get("errors", [])
        if errors:
            with st.expander("Metric warnings/errors"):
                for e in errors:
                    st.markdown(f"- {e}")

    # All-expert residue targeting suggestions
    base_seq = st.session_state.get("sequence", "")
    base_plddt = st.session_state.get("base_plddt_per_residue") or []
    is_immune = (
        ("immunebuilder" in baseline_predictors)
        or (st.session_state.get("mutation_predictor") == "immunebuilder")
    )
    conf_summary, flagged_positions = _summarize_confidence(base_seq, base_plddt, is_immune)
    selected_positions_sorted = sorted(st.session_state.get("selected_positions", set()))
    selected_str = (
        ", ".join(f"{base_seq[p-1]}{p}" for p in selected_positions_sorted[:15])
        + (" ..." if len(selected_positions_sorted) > 15 else "")
    ) if selected_positions_sorted and base_seq else "None"

    baseline_summary = _format_baseline_summary(st.session_state.baseline_results, label_by_value)
    base_eval_summary = _format_base_eval_summary(base_eval) if base_eval else ""
    context_lines = [
        f"Sequence length: {len(base_seq)}" if base_seq else "",
        f"Selected positions: {selected_str}",
        conf_summary,
        "Baseline predictors:\n" + baseline_summary if baseline_summary else "",
        "Baseline evaluation:\n" + base_eval_summary if base_eval_summary else "",
    ]
    context = "\n".join([line for line in context_lines if line])
    review_provider, review_model = _expert_review_overrides()

    agenda = (
        "Review the baseline structure predictions and per-residue confidence. "
        "Use confidence, baseline quality metrics, and structural context to suggest "
        "which residues should be targeted for mutagenesis (stability, function, or "
        "binding improvements). Prioritize a short list of positions."
    )
    questions = (
        "Which residues are the best candidates to mutate based on confidence, "
        "structural context (surface vs core), and predicted stability?",
        "Which residues should be avoided due to functional importance or "
        "structural risk?",
        "Provide 5-10 residue positions to target, ranked high to low, with short rationale for each.",
    )
    render_all_experts_panel(
        "🧠 All-Expert Residue Targeting (after baseline)",
        agenda=agenda,
        context=context,
        questions=questions,
        key_prefix="mut_baseline",
        provider_override=review_provider,
        model_override=review_model,
    )

    if base_eval:
        eval_agenda = (
            "Interpret the baseline structure evaluation and decide if the wild-type "
            "structure quality is sufficient for mutagenesis-driven decisions."
        )
        eval_questions = (
            "Are baseline quality metrics acceptable for mutation planning?",
            "Which structural regions look fragile and should be treated cautiously?",
            "Should we run additional validation before launching large mutagenesis scans?",
        )
        render_all_experts_panel(
            "🧠 All-Expert Baseline Evaluation Review",
            agenda=eval_agenda,
            context="Baseline evaluation summary:\n" + base_eval_summary,
            questions=eval_questions,
            key_prefix="mut_baseline_eval",
            provider_override=review_provider,
            model_override=review_model,
        )

# Predictor for mutation scans
mutation_option_labels = ["ESMFold API (ESM-2, <=400 aa)"]
if show_advanced:
    mutation_option_labels += [
        "ESM1 (legacy ESMFold v0)",
        "ESMFold2 (ESM-2, local)",
        "ESM3 (local or Forge)",
        "ImmuneBuilder (antibody/nanobody/TCR)",
    ]

current_mut_label = label_by_value.get(st.session_state.get("mutation_predictor"), "ESMFold API (ESM-2, <=400 aa)")
if current_mut_label not in mutation_option_labels:
    current_mut_label = "ESMFold API (ESM-2, <=400 aa)"

selected_label = st.selectbox(
    "Predictor for Mutation Scans",
    options=mutation_option_labels,
    index=mutation_option_labels.index(current_mut_label),
    help="Choose which model to fold each mutant for scoring.",
)
selected_predictor = predictor_options[selected_label]
if st.session_state.get("mutation_predictor") != selected_predictor:
    st.session_state.mutation_predictor = selected_predictor
    st.session_state.scanner = build_scanner(selected_predictor)
    st.session_state.baseline_evaluation = None
    st.session_state.baseline_evaluation_predictor = None

# Mutagenesis evaluation settings
st.markdown("#### Mutagenesis Evaluation")
with st.expander("🔬 Advanced Evaluation Metrics", expanded=False):
    prev_eval_enabled = st.session_state.get("mutation_eval_enabled", False)
    prev_eval_metrics = list(st.session_state.get("mutation_eval_metrics", []))
    prev_ost_enabled = bool(st.session_state.get("mutation_ost_comprehensive", False))

    eval_enabled = st.checkbox(
        "Run extended evaluation per mutant (OpenMM/Voronota/etc.)",
        value=prev_eval_enabled,
        help="Metrics requiring a reference will use the wild-type base structure.",
    )

    metric_options: List[str] = []
    if eval_enabled:
        try:
            from protein_design_hub.evaluation.composite import CompositeEvaluator

            metric_options = [m["name"] for m in CompositeEvaluator.list_all_metrics()]
        except Exception:
            metric_options = []

    selected_metrics = prev_eval_metrics
    if eval_enabled:
        selected_metrics = st.multiselect(
            "Metrics to compute",
            options=metric_options,
            default=[m for m in prev_eval_metrics if m in metric_options] or prev_eval_metrics,
            help="Use sparingly; some metrics (OpenMM, Rosetta) can be slow.",
        )

    ost_comprehensive = st.checkbox(
        "Run OpenStructure comprehensive comparison for each mutant (slow)",
        value=prev_ost_enabled,
        help=(
            "Compares each mutant against the baseline structure using OpenStructure "
            "global/per-chain/interface metrics. Use for shortlisted candidates."
        ),
    )
    st.caption(
        "Requires OpenStructure availability in your environment. "
        "If unavailable, results keep running and log a metric warning."
    )

    st.session_state.mutation_eval_enabled = eval_enabled
    st.session_state.mutation_eval_metrics = selected_metrics
    st.session_state.mutation_ost_comprehensive = ost_comprehensive

    if (
        (eval_enabled != prev_eval_enabled)
        or (selected_metrics != prev_eval_metrics)
        or (ost_comprehensive != prev_ost_enabled)
    ):
        st.session_state.scanner = build_scanner(st.session_state.mutation_predictor)
        st.session_state.scan_results = None
        st.session_state.multi_scan_results = None
        st.session_state.baseline_evaluation = None
        st.session_state.baseline_evaluation_predictor = None

if "esm3" in baseline_predictors or selected_predictor == "esm3":
    st.info("ESM3 uses the EvolutionaryScale `esm` package; ESMFold uses `fair-esm`. Use separate environments if needed.")

if "immunebuilder" in baseline_predictors or selected_predictor == "immunebuilder":
    st.info(
        "ImmuneBuilder requires FASTA with >A and >B chains. For antibodies/nanobodies, "
        "A is treated as heavy and B as light. Nanobody uses only chain A. "
        "Set IMMUNEBUILDER_PYTHON to a separate env with ImmuneBuilder installed."
    )

if "immunebuilder" in baseline_predictors or selected_predictor == "immunebuilder":
    st.markdown("#### ImmuneBuilder Settings")
    mode_label = st.selectbox(
        "Receptor type",
        options=["Antibody", "Nanobody", "TCR"],
        index=["antibody", "nanobody", "tcr"].index(st.session_state.get("immunebuilder_mode", "antibody")),
        help="Select the receptor type for ImmuneBuilder.",
    )
    st.session_state.immunebuilder_mode = mode_label.lower()
    if st.session_state.immunebuilder_mode == "nanobody":
        st.session_state.immune_active_chain = "A"

    chain_options = ["A", "B"]
    if st.session_state.immunebuilder_mode == "nanobody":
        chain_options = ["A"]
    chain_label = st.selectbox(
        "Chain to mutate",
        options=chain_options,
        index=0 if st.session_state.get("immune_active_chain", "A") == "A" else 0,
        help="Positions refer to the selected chain when using ImmuneBuilder.",
    )
    if st.session_state.get("immune_active_chain") != chain_label:
        st.session_state.immune_active_chain = chain_label
        if st.session_state.get("immune_chain_a") and st.session_state.get("immune_chain_b"):
            st.session_state.sequence = (
                st.session_state.immune_chain_a if chain_label == "A" else st.session_state.immune_chain_b
            )
        else:
            st.session_state.sequence = ""
        st.session_state.base_structure = None
        st.session_state.base_structure_path = None
        st.session_state.scan_results = None
        st.session_state.multi_scan_results = None
        st.session_state.baseline_evaluation = None
        st.session_state.baseline_evaluation_predictor = None

    if selected_predictor == "immunebuilder":
        st.session_state.scanner = build_scanner("immunebuilder")

# 1. Input
section_header("Input Sequence", "Paste your protein sequence for mutation analysis", "1️⃣")

seq_col, info_col = st.columns([3, 1])
with seq_col:
    immune_required = ("immunebuilder" in baseline_predictors) or (selected_predictor == "immunebuilder")
    if immune_required and st.session_state.sequence_input_raw and not st.session_state.immune_parse_error:
        if not (st.session_state.immune_chain_a and st.session_state.immune_chain_b):
            try:
                chains = parse_ab_fasta(st.session_state.sequence_input_raw)
                st.session_state.immune_chain_a = chains["A"]
                st.session_state.immune_chain_b = chains["B"]
                active_chain = st.session_state.get("immune_active_chain", "A")
                st.session_state.sequence = chains[active_chain]
                if selected_predictor == "immunebuilder":
                    st.session_state.scanner = build_scanner("immunebuilder")
            except Exception as e:
                st.session_state.immune_parse_error = str(e)
    placeholder = (
        ">A\nEVQLVESGGGLVQPGGSLRLSCAAS...\n>B\nDIQMTQSPSSLSASVGDRVT...\n"
        if immune_required else
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPP...\n\nPaste raw amino acid sequence (A-Y)"
    )
    new_seq = st.text_area(
        "Protein Sequence",
        value=st.session_state.sequence_input_raw,
        height=120 if immune_required else 100,
        placeholder=placeholder,
        label_visibility="collapsed"
    )
    if new_seq != st.session_state.sequence_input_raw:
        st.session_state.sequence_input_raw = new_seq
        st.session_state.immune_parse_error = None
        if immune_required:
            if new_seq.strip():
                try:
                    chains = parse_ab_fasta(new_seq)
                    st.session_state.immune_chain_a = chains["A"]
                    st.session_state.immune_chain_b = chains["B"]
                    active_chain = st.session_state.get("immune_active_chain", "A")
                    st.session_state.sequence = chains[active_chain]
                    if selected_predictor == "immunebuilder":
                        st.session_state.scanner = build_scanner("immunebuilder")
                except Exception as e:
                    st.session_state.immune_parse_error = str(e)
                    st.session_state.sequence = ""
            else:
                st.session_state.sequence = ""
        else:
            cleaned = "".join(c for c in new_seq.upper().strip() if c in AMINO_ACIDS)
            st.session_state.sequence = cleaned

        st.session_state.base_structure = None
        st.session_state.base_structure_path = None
        st.session_state.scan_results = None
        st.session_state.baseline_results = None
        st.session_state.baseline_sequence = None
        st.session_state.baseline_evaluation = None
        st.session_state.baseline_evaluation_predictor = None
        st.rerun()

    if st.session_state.immune_parse_error:
        st.error(st.session_state.immune_parse_error)

with info_col:
    st.markdown("##### Quick Load")
    if st.button("📋 Ubiquitin (76 aa)", use_container_width=True, type="secondary"):
        ubi = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        if ("immunebuilder" in baseline_predictors) or (selected_predictor == "immunebuilder"):
            st.session_state.sequence_input_raw = f">A\n{ubi}\n>B\n{ubi}\n"
        else:
            st.session_state.sequence = ubi
            st.session_state.sequence_input_raw = ubi
        st.session_state.sequence_name = "Ubiquitin"
        st.session_state.base_structure = None
        st.session_state.base_structure_path = None
        st.session_state.scan_results = None
        st.session_state.baseline_results = None
        st.session_state.baseline_sequence = None
        st.session_state.baseline_evaluation = None
        st.session_state.baseline_evaluation_predictor = None
        st.rerun()

    if st.button("📋 T1024 (52 aa)", use_container_width=True, type="secondary"):
        t1024 = "MAAHKGAEHVVKASLDAGVKTVAGGLVVKAKALGGKDATMHLVAATLKKGYM"
        if ("immunebuilder" in baseline_predictors) or (selected_predictor == "immunebuilder"):
            st.session_state.sequence_input_raw = f">A\n{t1024}\n>B\n{t1024}\n"
        else:
            st.session_state.sequence = t1024
            st.session_state.sequence_input_raw = t1024
        st.session_state.sequence_name = "T1024"
        st.session_state.base_structure = None
        st.session_state.base_structure_path = None
        st.session_state.scan_results = None
        st.session_state.baseline_results = None
        st.session_state.baseline_sequence = None
        st.session_state.baseline_evaluation = None
        st.session_state.baseline_evaluation_predictor = None
        st.rerun()

# 2. Base Prediction
if st.session_state.sequence:
    st.markdown("---")
    st.markdown("## 2️⃣ Base Structure")
    
    if not st.session_state.base_structure:
        if st.button("🚀 Predict Base Structure", type="primary"):
            with st.spinner("Predicting..."):
                pdb, plddt, path = st.session_state.scanner.predict_single(st.session_state.sequence, "base")
                st.session_state.base_structure = pdb
                st.session_state.base_structure_path = str(path) if path else None
                st.session_state.base_plddt = sum(plddt)/len(plddt)
                st.session_state.base_plddt_per_residue = plddt
                st.session_state.baseline_evaluation = None
                st.session_state.baseline_evaluation_predictor = None
                st.rerun()
    else:
        pred_label = label_by_value.get(st.session_state.mutation_predictor, st.session_state.mutation_predictor)
        metric_name = "Mean pLDDT"
        if st.session_state.mutation_predictor == "immunebuilder":
            metric_name = "Mean error (Å)"
        st.success(
            f"Base Structure Ready ({metric_name}: {st.session_state.base_plddt:.1f}) · {pred_label}"
        )

        if st.session_state.get("afdb_enabled"):
            with st.expander("🔍 Related AlphaFold DB Structure", expanded=False):
                if not st.session_state.get("afdb_email"):
                    st.info("Provide an EBI email in the sidebar to enable AFDB lookup.")
                else:
                    clean_seq = normalize_sequence(st.session_state.sequence)
                    if not clean_seq:
                        st.info("AFDB lookup skipped: no valid amino acids found.")
                    else:
                        with st.status("Searching AFDB via BLAST...", expanded=False) as status:
                            match, error = get_afdb_match_cached(
                                clean_seq,
                                st.session_state.get("afdb_email", ""),
                            )
                            if error:
                                status.update(label="AFDB lookup failed", state="error", expanded=False)
                            else:
                                status.update(label="AFDB lookup complete", state="complete", expanded=False)

                        if match and match.structure_path and Path(match.structure_path).exists():
                            col_afdb_view, col_afdb_meta = st.columns([3, 1])
                            with col_afdb_view:
                                from protein_design_hub.web.visualizations import create_structure_viewer
                                import streamlit.components.v1 as components
                                components.html(
                                    create_structure_viewer(Path(match.structure_path), height=350),
                                    height=370,
                                )
                            with col_afdb_meta:
                                st.markdown("#### AFDB Match")
                                st.write(f"UniProt: {match.uniprot_id}")
                                if match.entry_id:
                                    st.write(f"AFDB Entry: {match.entry_id}")
                                st.write(f"Identity: {match.identity:.1f}%")
                                st.write(f"Coverage: {match.coverage:.1f}%")
                                if match.evalue is not None:
                                    st.write(f"E-value: {match.evalue:.2e}")
                        elif error:
                            st.error(f"AFDB lookup error: {error}")
                        else:
                            st.info("No AFDB match found with ≥90% identity and coverage.")
        
        # 3. Selection
        st.markdown("### Residue Selection & pLDDT-Based Prioritization")
        seq = st.session_state.sequence
        plddt_per_res = st.session_state.base_plddt_per_residue or []

        col_plddt, col_select = st.columns([2, 1])
        with col_plddt:
            if plddt_per_res:
                if st.session_state.mutation_predictor == "immunebuilder":
                    threshold = st.slider("High-error threshold (Å)", 0.0, 20.0, 5.0)
                    if st.button("Auto-select high-error residues", use_container_width=True):
                        st.session_state.selected_positions = {
                            i + 1 for i, v in enumerate(plddt_per_res) if v > threshold
                        }
                        st.rerun()
                else:
                    threshold = st.slider("Low-confidence threshold (pLDDT)", 40, 90, 70)
                    if st.button("Auto-select low-confidence residues", use_container_width=True):
                        st.session_state.selected_positions = {
                            i + 1 for i, v in enumerate(plddt_per_res) if v < threshold
                        }
                        st.rerun()
                values = []
                for v in plddt_per_res:
                    try:
                        values.append(float(v))
                    except Exception:
                        values.append(float("nan"))
                if values and max(values) <= 1.0:
                    values = [v * 100.0 for v in values]
                try:
                    from protein_design_hub.web.visualizations import create_plddt_plot
                    title = "Per-Residue pLDDT (Base)"
                    if st.session_state.mutation_predictor == "immunebuilder":
                        title = "Per-Residue Error (Base)"
                    fig = create_plddt_plot(values, title=title)
                    for pos in st.session_state.selected_positions:
                        fig.add_vline(x=pos, line_dash="dash", line_color="#f59e0b", opacity=0.6)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Plotly pLDDT plot unavailable ({exc}). Showing fallback line chart.")
                    st.line_chart(values, height=300, use_container_width=True)
        with col_select:
            st.markdown("**Manual selection**")
            pos_text = st.text_input("Positions (e.g. 5,7-10)", value="")
            if st.button("Add positions", use_container_width=True):
                st.session_state.selected_positions |= parse_positions(pos_text, len(seq))
                st.rerun()
            if st.button("Clear selection", use_container_width=True):
                st.session_state.selected_positions = set()
                st.rerun()

        st.caption("Click residues to toggle. Selected positions drive multi-mutation design.")
        cols = st.columns(20)
        for i, aa in enumerate(seq):
            if i >= 120:
                break
            selected = (i + 1) in st.session_state.selected_positions
            color = "primary" if selected else "secondary"
            if cols[i % 20].button(aa, key=f"r_{i}", type=color, help=f"Pos {i+1}"):
                if selected:
                    st.session_state.selected_positions.remove(i + 1)
                else:
                    st.session_state.selected_positions.add(i + 1)
                st.session_state.selected_position = i + 1
                st.session_state.scan_results = None
                st.rerun()

        selected_positions_sorted = sorted(st.session_state.selected_positions)
        if selected_positions_sorted:
            st.info(
                "Selected positions: "
                + ", ".join(f"{seq[p-1]}{p}" for p in selected_positions_sorted[:20])
                + (" ..." if len(selected_positions_sorted) > 20 else "")
            )

        # Baseline expert guidance directly after base-structure prediction.
        # This runs even if the full baseline-comparison action was not used.
        if not st.session_state.get("baseline_results"):
            is_immune = st.session_state.get("mutation_predictor") == "immunebuilder"
            conf_summary, _ = _summarize_confidence(
                seq,
                st.session_state.get("base_plddt_per_residue") or [],
                is_immune,
            )
            selected_str = (
                ", ".join(f"{seq[p-1]}{p}" for p in selected_positions_sorted[:20])
                + (" ..." if len(selected_positions_sorted) > 20 else "")
            ) if selected_positions_sorted else "None"

            context_lines = [
                f"Sequence length: {len(seq)}",
                f"Predictor: {label_by_value.get(st.session_state.get('mutation_predictor'), st.session_state.get('mutation_predictor'))}",
                f"Baseline mean score: {st.session_state.get('base_plddt', 0.0):.2f}",
                f"Selected positions: {selected_str}",
                conf_summary,
            ]
            review_provider, review_model = _expert_review_overrides()
            render_all_experts_panel(
                "🧠 All-Expert Residue Targeting (after base prediction)",
                agenda=(
                    "Interpret baseline structure confidence and identify the most valuable "
                    "residue positions for mutagenesis."
                ),
                context="\n".join([line for line in context_lines if line]),
                questions=(
                    "Which residues should be mutated first and why?",
                    "Which residues should be avoided due to likely structural or functional risk?",
                    "Recommend a ranked shortlist of 5-10 positions for scanning.",
                ),
                key_prefix="mut_basepred",
                provider_override=review_provider,
                model_override=review_model,
            )

        # Single-position scan
        if st.session_state.selected_position:
            pos = st.session_state.selected_position
            st.markdown("#### Single-Position Saturation Scan")
            st.info(f"Selected: **{seq[pos-1]}{pos}**")

            if st.button(f"🔬 Run Saturation Mutagenesis at {pos}", type="primary"):
                results = run_saturation_mutagenesis(seq, pos)
                st.session_state.scan_results = results
                st.session_state.multi_scan_results = None

                # Create Job in outputs
                try:
                    from protein_design_hub.core.config import get_settings
                    settings = get_settings()
                    job_id = f"scan_{st.session_state.sequence_name}_{pos}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    job_dir = Path(settings.output.base_dir) / job_id
                    job_dir.mkdir(parents=True, exist_ok=True)

                    with open(job_dir / "scan_results.json", "w") as f:
                        json.dump(results.to_dict(), f, indent=2)

                    import shutil
                    if results.base_structure_path and results.base_structure_path.exists():
                        shutil.copy(results.base_structure_path, job_dir / "base_wt.pdb")

                    with open(job_dir / "prediction_summary.json", "w") as f:
                        json.dump({"job_id": job_id, "type": "scan", "status": "complete"}, f)

                    st.info(f"💾 Job saved as {job_id}")
                except Exception as e:
                    st.warning(f"Could not save job to outputs: {e}")

                st.rerun()

        # Multi-mutation pipeline
        if selected_positions_sorted and len(selected_positions_sorted) >= 2:
            st.markdown("#### Multi-Mutation Design Pipeline")
            col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
            with col_a:
                top_k = st.slider("Top mutations per position", 1, 5, 2)
            with col_b:
                max_variants = st.slider("Max combined variants", 1, 30, 12)
            with col_c:
                only_beneficial = st.checkbox("Only beneficial singles", value=True)
            with col_d:
                max_positions = st.slider("Max positions", 2, 10, 6)

            st.caption("Pipeline: run single-position scans → build multi-mutation combos → re-evaluate by residue pLDDT.")
            if len(selected_positions_sorted) > max_positions:
                st.warning(f"Selected {len(selected_positions_sorted)} positions. Reduce to ≤ {max_positions} to run.")
            if st.button("🧬 Run Multi-Mutation Pipeline", type="primary", disabled=len(selected_positions_sorted) > max_positions):
                results = run_multi_mutation_pipeline(
                    seq,
                    selected_positions_sorted,
                    top_k=top_k,
                    max_variants=max_variants,
                    only_beneficial=only_beneficial,
                    max_positions=max_positions,
                )
                st.session_state.multi_scan_results = results
                st.session_state.scan_results = None

                # Save job
                try:
                    from protein_design_hub.core.config import get_settings
                    settings = get_settings()
                    job_id = f"multi_scan_{st.session_state.sequence_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    job_dir = Path(settings.output.base_dir) / job_id
                    job_dir.mkdir(parents=True, exist_ok=True)

                    with open(job_dir / "scan_results.json", "w") as f:
                        json.dump(results.to_dict(), f, indent=2)

                    import shutil
                    if results.base_structure_path and results.base_structure_path.exists():
                        shutil.copy(results.base_structure_path, job_dir / "base_wt.pdb")

                    with open(job_dir / "prediction_summary.json", "w") as f:
                        json.dump({"job_id": job_id, "type": "multi_scan", "status": "complete"}, f)

                    st.info(f"💾 Job saved as {job_id}")
                except Exception as e:
                    st.warning(f"Could not save job to outputs: {e}")

                st.rerun()

# 4. Multi-Mutation Results
if st.session_state.multi_scan_results:
    res = st.session_state.multi_scan_results
    st.markdown("---")
    st.markdown("## 🧬 Multi-Mutation Results")
    pred_label = label_by_value.get(getattr(res, "predictor", None), getattr(res, "predictor", ""))
    if pred_label:
        st.caption(f"Predictor: {pred_label}")
    is_immunebuilder = getattr(res, "predictor", "") == "immunebuilder"
    delta_label = "ΔpLDDT"
    local_label = "Δlocal pLDDT"
    if is_immunebuilder:
        delta_label = "ΔError (Å)"
        local_label = "Δlocal error (Å)"

    def _variant_score(v):
        if hasattr(v, "improvement_score"):
            return v.improvement_score or 0
        return 0.6 * getattr(v, "delta_mean_plddt", 0) + 0.4 * getattr(v, "delta_local_plddt", 0)

    variants = sorted(
        [v for v in res.variants if getattr(v, "success", True)],
        key=_variant_score,
        reverse=True,
    )

    tab1, tab2, tab3 = st.tabs(["🏆 Top Variants", "📈 Metrics Table", "🔬 3D Comparison"])

    with tab1:
        if variants:
            best = variants[0]
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, rgba(16,185,129,0.15) 0%, rgba(34,197,94,0.15) 100%); padding: 15px; border-radius: 10px; border: 1px solid rgba(16,185,129,0.35); margin-bottom: 20px;">
                <h3 style="margin:0; color: #10b981;">🏆 Best Multi-Mutation: {best.mutation_code}</h3>
                <p style="margin:5px 0 0 0; color: #e2e8f0;">
                    {delta_label}: <b>{best.delta_mean_plddt:+.2f}</b> ·
                    {local_label}: <b>{best.delta_local_plddt:+.2f}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("#### 🚀 Validate in Predict")
                target_variant = st.selectbox(
                    "Select Variant",
                    options=variants[:10],
                    format_func=lambda v: f"{v.mutation_code} (Δ {v.delta_mean_plddt:+.2f})",
                    key="multi_predict_select",
                )
                if st.button("⚡ Send to Predict Page", type="primary", use_container_width=True):
                    mut_seq = list(res.sequence)
                    for pos, aa in zip(res.positions, target_variant.mutant_aas):
                        mut_seq[pos - 1] = aa
                    mut_seq = "".join(mut_seq)

                    st.session_state['incoming_prediction_job'] = {
                        'sequence': mut_seq,
                        'name': f"{st.session_state.sequence_name}_{target_variant.mutation_code}",
                        'source': 'mutation_scanner_multi',
                        'description': f"Multi-variant {target_variant.mutation_code}. Δmean pLDDT: {target_variant.delta_mean_plddt:+.2f}",
                    }
                    st.switch_page("pages/1_predict.py")

            with c2:
                st.markdown("#### 🔬 Quick Compare")
                if st.button("Load Best Variant", use_container_width=True):
                    st.session_state.multi_comparison_variant = best

        if variants:
            st.markdown("#### Other Top Candidates")
            for v in variants[1:5]:
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric(v.mutation_code, f"Δ {v.delta_mean_plddt:+.2f}")
                with col2:
                    st.caption(
                        f"Local pLDDT: {v.local_plddt_mean:.1f} (min {v.local_plddt_min:.1f}) | "
                        f"RMSD: {v.rmsd_to_base:.2f} Å" if getattr(v, "rmsd_to_base", None) else
                        f"Local pLDDT: {v.local_plddt_mean:.1f} (min {v.local_plddt_min:.1f})"
                    )
        else:
            st.info("No successful variants produced.")

    with tab2:
        include_cad = any(get_extra_metric(v, "cad_score", "cad_score") is not None for v in variants)
        include_voromqa = any(get_extra_metric(v, "voromqa", "voromqa_score") is not None for v in variants)
        include_openmm = any(
            get_extra_metric(v, "openmm_gbsa", "openmm_potential_energy_kj_mol") is not None
            for v in variants
        )
        ost_fields = [
            ("lddt", "OST lDDT"),
            ("rmsd_ca", "OST RMSD(CA, Å)"),
            ("qs_score", "OST QS-score"),
            ("tm_score", "OST TM-score"),
            ("gdt_ts", "OST GDT-TS"),
            ("gdt_ha", "OST GDT-HA"),
        ]
        include_ost = {
            field: any(get_ost_global_metric(v, field) is not None for v in variants)
            for field, _ in ost_fields
        }
        data = []
        for v in variants:
            mean_label = "Mean pLDDT"
            mean_delta_label = "Δ Mean"
            local_label_col = "Local pLDDT"
            local_delta_label = "Δ Local"
            if is_immunebuilder:
                mean_label = "Mean error (Å)"
                mean_delta_label = "Δ Mean"
                local_label_col = "Local error (Å)"
                local_delta_label = "Δ Local"
            data.append({
                "Variant": v.mutation_code,
                mean_label: f"{v.mean_plddt:.1f}",
                mean_delta_label: f"{v.delta_mean_plddt:+.2f}",
                local_label_col: f"{v.local_plddt_mean:.1f}",
                local_delta_label: f"{v.delta_local_plddt:+.2f}",
                "Min Local": f"{v.local_plddt_min:.1f}",
                "RMSD (Å)": f"{v.rmsd_to_base:.2f}" if getattr(v, "rmsd_to_base", None) else "N/A",
                "TM-score": f"{v.tm_score_to_base:.2f}" if getattr(v, "tm_score_to_base", None) else "N/A",
            })
            if include_cad:
                cad = get_extra_metric(v, "cad_score", "cad_score")
                data[-1]["CAD-score"] = f"{cad:.3f}" if cad is not None else "N/A"
            if include_voromqa:
                voro = get_extra_metric(v, "voromqa", "voromqa_score")
                data[-1]["VoroMQA"] = f"{voro:.3f}" if voro is not None else "N/A"
            if include_openmm:
                openmm = get_extra_metric(v, "openmm_gbsa", "openmm_potential_energy_kj_mol")
                data[-1]["OpenMM (kJ/mol)"] = f"{openmm:.1f}" if openmm is not None else "N/A"
            for field, label in ost_fields:
                if include_ost[field]:
                    value = get_ost_global_metric(v, field)
                    if value is None:
                        data[-1][label] = "N/A"
                    elif field in {"rmsd_ca"}:
                        data[-1][label] = f"{value:.2f}"
                    else:
                        data[-1][label] = f"{value:.3f}"
        st.dataframe(pd.DataFrame(data), use_container_width=True)

    with tab3:
        base_path = None
        if getattr(res, "base_structure_path", None):
            try:
                cand = Path(res.base_structure_path)
                if cand.exists():
                    base_path = cand
            except Exception:
                base_path = None

        variant = st.session_state.multi_comparison_variant or (variants[0] if variants else None)
        if variant:
            from protein_design_hub.web.visualizations import create_structure_viewer
            import streamlit.components.v1 as components

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Wild Type**")
                if base_path:
                    components.html(create_structure_viewer(base_path, height=300), height=320)
                elif st.session_state.base_structure:
                    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
                        f.write(st.session_state.base_structure)
                        p1 = f.name
                    components.html(create_structure_viewer(Path(p1), height=300), height=320)
                else:
                    st.info("Base structure not available.")

            with c2:
                st.markdown(f"**Variant {variant.mutation_code}**")
                try:
                    vpath = Path(variant.structure_path)
                    if vpath.exists():
                        components.html(create_structure_viewer(vpath, height=300), height=320)
                    else:
                        st.info("Variant structure not found.")
                except Exception:
                    st.info("Variant structure not found.")
        else:
            st.info("No variant selected.")

    # All-expert investigation of multi-mutation results
    if variants:
        top_variants_text = _format_top_variants(variants, is_immunebuilder, top_k=6)
        context_lines = [
            f"Sequence length: {len(res.sequence)}",
            f"Positions combined: {', '.join(str(p) for p in res.positions)}",
            "Top variants:\n" + top_variants_text if top_variants_text else "",
        ]
        context = "\n".join([line for line in context_lines if line])
        agenda = (
            "Investigate multi-mutation results. Evaluate which combined variants are "
            "most promising and whether any trade-offs or risks are apparent. "
            "Produce a detailed ranked report with explicit validation priorities."
        )
        questions = (
            "Which multi-mutation variants look most promising based on ΔpLDDT/Δerror and "
            "structural similarity?",
            "Are there red flags (high RMSD, interface disruption, packing issues)?",
            "What next validation steps would you recommend before experiments?",
            "Provide a ranked shortlist with rationale and confidence for each variant.",
        )
        review_provider, review_model = _expert_review_overrides()
        render_all_experts_panel(
            "🧠 All-Expert Investigation (multi-mutation results)",
            agenda=agenda,
            context=context,
            questions=questions,
            key_prefix="mut_multi",
            provider_override=review_provider,
            model_override=review_model,
        )

# 5. Single-Position Results
if st.session_state.scan_results:
    res = st.session_state.scan_results
    st.markdown("---")
    st.markdown("## 📊 Scan Results")
    pred_label = label_by_value.get(getattr(res, "predictor", None), getattr(res, "predictor", ""))
    total_time = getattr(res, "total_time", None)
    if pred_label and total_time:
        total_preds = 1 + len(getattr(res, "mutations", []))
        throughput = (total_preds / total_time) if total_time else None
        st.caption(
            f"Predictor: {pred_label} · Total time: {total_time:.1f}s · Throughput: "
            f"{throughput:.2f} structures/s"
        )
    is_immunebuilder = getattr(res, "predictor", "") == "immunebuilder"
    delta_label = "ΔpLDDT"
    if is_immunebuilder:
        delta_label = "ΔError (Å)"
    
    tab1, tab2, tab3 = st.tabs(["🏆 Recommendations", "📈 Detailed Metrics", "🔬 3D Comparison"])
    
    with tab1:
        render_heatmap(res)
        
        st.markdown("### Top Variants")
        
        best_mut = res.ranked_mutations[0] if res.ranked_mutations else None
        
        if best_mut:
            # Highlight Best Variant
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, rgba(16,185,129,0.15) 0%, rgba(34,197,94,0.15) 100%); padding: 15px; border-radius: 10px; border: 1px solid rgba(16,185,129,0.35); margin-bottom: 20px;">
                <h3 style="margin:0; color: #10b981;">🏆 Best Candidate: {best_mut.mutation_code}</h3>
                <p style="margin:5px 0 0 0; color: var(--pdhub-text, #e2e8f0);">
                    predicted to improve by <b>+{best_mut.delta_mean_plddt:.2f}</b> ({delta_label})
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Integration Controls
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("#### 🚀 Validation Pipeline")
                st.write("Run high-accuracy prediction (ColabFold/Chai-1/Boltz-2) on this variant.")
                
                # Predictor selection
                target_variant = st.selectbox(
                    "Select Variant to Predict",
                    options=res.ranked_mutations[:5],
                    format_func=lambda x: f"{x.mutation_code} (Δ {x.delta_mean_plddt:.2f})"
                )
                
                if st.button("⚡ Send to Predict Page", type="primary", use_container_width=True):
                    # Create the full mutant sequence
                    mut_seq = res.sequence[:res.position-1] + target_variant.mutant_aa + res.sequence[res.position:]
                    
                    # Store in session state for the Predict page
                    st.session_state['incoming_prediction_job'] = {
                        'sequence': mut_seq,
                        'name': f"{st.session_state.sequence_name}_{target_variant.mutation_code}",
                        'source': 'mutation_scanner',
                        'description': f"Variant {target_variant.mutation_code} from residue {res.position} scan. Expected ΔpLDDT: {target_variant.delta_mean_plddt:.2f}"
                    }
                    st.switch_page("pages/1_predict.py")

            with c2:
                st.markdown("#### 🔬 Quick Compare")
                st.write(f"Compare {target_variant.mutation_code} with Wild-Type")
                if st.button("Load into Structure Viewer", use_container_width=True):
                    st.session_state.comparison_mutation = target_variant

        # List other top variants
        st.markdown("#### Other Top Candidates")
        for mut in res.ranked_mutations[1:4]:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.metric(mut.mutation_code, f"Δ {mut.delta_mean_plddt:.2f}", 
                         delta_color="normal" if mut.is_beneficial else "inverse")
            with col2:
                st.caption(f"RMSD: {mut.rmsd_to_base:.2f} Å | Clash Score: {mut.clash_score} | SASA: {mut.sasa_total}")


    with tab2:
        include_cad = any(get_extra_metric(m, "cad_score", "cad_score") is not None for m in res.mutations)
        include_voromqa = any(get_extra_metric(m, "voromqa", "voromqa_score") is not None for m in res.mutations)
        include_openmm = any(
            get_extra_metric(m, "openmm_gbsa", "openmm_potential_energy_kj_mol") is not None
            for m in res.mutations
        )
        ost_fields = [
            ("lddt", "OST lDDT"),
            ("rmsd_ca", "OST RMSD(CA, Å)"),
            ("qs_score", "OST QS-score"),
            ("tm_score", "OST TM-score"),
            ("gdt_ts", "OST GDT-TS"),
            ("gdt_ha", "OST GDT-HA"),
        ]
        include_ost = {
            field: any(get_ost_global_metric(m, field) is not None for m in res.mutations)
            for field, _ in ost_fields
        }
        data = []
        for m in res.mutations:
            if m.success:
                mean_label = "Mean pLDDT"
                delta_label = "Δ pLDDT"
                if is_immunebuilder:
                    mean_label = "Mean error (Å)"
                    delta_label = "Δ Error (Å)"
                data.append({
                    "Mutation": m.mutation_code,
                    mean_label: f"{m.mean_plddt:.1f}",
                    delta_label: f"{m.delta_mean_plddt:+.2f}",
                    "RMSD (Å)": f"{m.rmsd_to_base:.2f}" if m.rmsd_to_base else "N/A",
                    "Clash Score": f"{m.clash_score:.2f}" if m.clash_score else "N/A",
                    "SASA (Å²)": f"{m.sasa_total:.0f}" if m.sasa_total else "N/A",
                    "TM-score": f"{m.tm_score_to_base:.2f}" if m.tm_score_to_base else "N/A"
                })
                if include_cad:
                    cad = get_extra_metric(m, "cad_score", "cad_score")
                    data[-1]["CAD-score"] = f"{cad:.3f}" if cad is not None else "N/A"
                if include_voromqa:
                    voro = get_extra_metric(m, "voromqa", "voromqa_score")
                    data[-1]["VoroMQA"] = f"{voro:.3f}" if voro is not None else "N/A"
                if include_openmm:
                    openmm = get_extra_metric(m, "openmm_gbsa", "openmm_potential_energy_kj_mol")
                    data[-1]["OpenMM (kJ/mol)"] = f"{openmm:.1f}" if openmm is not None else "N/A"
                for field, label in ost_fields:
                    if include_ost[field]:
                        value = get_ost_global_metric(m, field)
                        if value is None:
                            data[-1][label] = "N/A"
                        elif field in {"rmsd_ca"}:
                            data[-1][label] = f"{value:.2f}"
                        else:
                            data[-1][label] = f"{value:.3f}"
        st.dataframe(pd.DataFrame(data), use_container_width=True)

    with tab3:
        if st.session_state.comparison_mutation:
            mut = st.session_state.comparison_mutation
            c1, c2 = st.columns(2)
            
            from protein_design_hub.web.visualizations import create_structure_viewer
            import streamlit.components.v1 as components

            with c1:
                st.markdown("**Wild Type**")
                with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
                    f.write(st.session_state.base_structure)
                    p1 = f.name
                components.html(create_structure_viewer(Path(p1), height=300), height=320)
                
            with c2:
                st.markdown(f"**Mutant {mut.mutation_code}**")
                with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
                    f.write(mut.structure_path.read_text())
                    p2 = f.name
                components.html(create_structure_viewer(Path(p2), height=300), height=320)
        else:
            st.info("Select a mutation from Recommendations to compare.")

    # Agent advice on mutation results
    if res.ranked_mutations:
        top5 = res.ranked_mutations[:5]
        mut_ctx_parts = [
            f"- {m.mutation_code}: ΔpLDDT={m.delta_mean_plddt:+.2f}"
            for m in top5
        ]
        render_agent_advice_panel(
            page_context=(
                f"Saturation mutagenesis results at position {res.position} of "
                f"{len(res.sequence)}-residue protein.\n"
                f"Wild-type residue: {res.sequence[res.position-1]}\n"
                f"Top mutations:\n" + "\n".join(mut_ctx_parts)
            ),
            default_question=(
                "Which of these mutations looks most promising for stability? "
                "Are there any concerns about these substitutions?"
            ),
            expert="Protein Engineer",
            key_prefix="mut_agent",
        )

        # All-expert investigation of single-position mutagenesis
        is_immune = getattr(res, "predictor", "") == "immunebuilder"
        top_mut_text = _format_top_mutations(res.ranked_mutations, is_immune, top_k=8)
        context_lines = [
            f"Sequence length: {len(res.sequence)}",
            f"Position scanned: {res.original_aa}{res.position}",
            "Top mutations:\n" + top_mut_text if top_mut_text else "",
        ]
        context = "\n".join([line for line in context_lines if line])
        agenda = (
            "Investigate saturation mutagenesis results for the selected position. "
            "Evaluate the top mutations and identify any risks or trade-offs. "
            "Generate a detailed recommendation report."
        )
        questions = (
            "Which mutations are most promising for stability or function?",
            "Are there any substitutions that should be avoided and why?",
            "Recommend the next 3-5 mutants to validate with high-accuracy predictors.",
            "Provide a ranked list with confidence, risks, and follow-up checks per mutant.",
        )
        review_provider, review_model = _expert_review_overrides()
        render_all_experts_panel(
            "🧠 All-Expert Investigation (single-position results)",
            agenda=agenda,
            context=context,
            questions=questions,
            key_prefix="mut_single",
            provider_override=review_provider,
            model_override=review_model,
        )
