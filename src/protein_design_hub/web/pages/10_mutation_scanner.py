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
        'baseline_predictors': ['esmfold_api'],
        'baseline_results': None,
        'baseline_sequence': None,
        'scanner': MutationScanner(
            predictor='esmfold_api',
            evaluation_metrics=default_eval_metrics,
        ),
        'afdb_enabled': False,
        'afdb_email': os.getenv("EBI_EMAIL", ""),
        'afdb_cache': {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def current_eval_metrics() -> List[str]:
    if not st.session_state.get("mutation_eval_enabled"):
        return []
    return st.session_state.get("mutation_eval_metrics", [])

def build_scanner(predictor_id: str) -> MutationScanner:
    eval_metrics = current_eval_metrics()
    if predictor_id == "immunebuilder":
        return MutationScanner(
            predictor=predictor_id,
            immunebuilder_mode=st.session_state.immunebuilder_mode,
            immune_chain_a=st.session_state.immune_chain_a,
            immune_chain_b=st.session_state.immune_chain_b,
            immune_active_chain=st.session_state.immune_active_chain,
            evaluation_metrics=eval_metrics,
        )
    return MutationScanner(predictor=predictor_id, evaluation_metrics=eval_metrics)

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

# Mutagenesis evaluation settings
st.markdown("#### Mutagenesis Evaluation")
with st.expander("🔬 Advanced Evaluation Metrics", expanded=False):
    prev_eval_enabled = st.session_state.get("mutation_eval_enabled", False)
    prev_eval_metrics = list(st.session_state.get("mutation_eval_metrics", []))

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

    st.session_state.mutation_eval_enabled = eval_enabled
    st.session_state.mutation_eval_metrics = selected_metrics

    if (eval_enabled != prev_eval_enabled) or (selected_metrics != prev_eval_metrics):
        st.session_state.scanner = build_scanner(st.session_state.mutation_predictor)
        st.session_state.scan_results = None
        st.session_state.multi_scan_results = None

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
        st.session_state.scan_results = None
        st.session_state.multi_scan_results = None

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
        st.session_state.scan_results = None
        st.session_state.baseline_results = None
        st.session_state.baseline_sequence = None
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
        st.session_state.scan_results = None
        st.session_state.baseline_results = None
        st.session_state.baseline_sequence = None
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
        st.session_state.scan_results = None
        st.session_state.baseline_results = None
        st.session_state.baseline_sequence = None
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
                st.session_state.base_plddt = sum(plddt)/len(plddt)
                st.session_state.base_plddt_per_residue = plddt
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
