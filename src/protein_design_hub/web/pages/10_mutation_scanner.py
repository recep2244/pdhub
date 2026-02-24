"""Interactive Mutation Scanner with ESMFold-based saturation mutagenesis.

This page provides:
1. Sequence input with auto ESMFold prediction
2. Interactive residue selection for mutation scanning
3. Automatic saturation mutagenesis (all 19 AA mutations)
4. Comprehensive metric calculation (pLDDT, RMSD, Clash Score, SASA)
5. Mutation ranking and recommendations
6. Side-by-side structure comparison
"""

import base64
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF

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
    render_ml_stats_panel,
    agent_sidebar_status,
    render_all_experts_panel,
)
from protein_design_hub.web.shared_context import set_page_results, render_workflow_status_bar
from datetime import datetime, timezone
from types import SimpleNamespace

logger = logging.getLogger(__name__)

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
render_workflow_status_bar()
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


def _score_semantics(predictor_id: str) -> Dict[str, Any]:
    """Return normalized score semantics for predictor-specific confidence metrics."""
    if predictor_id == "immunebuilder":
        return {
            "metric_name": "mean_error",
            "metric_label": "Mean error (Å)",
            "delta_label": "ΔError (Å)",
            "higher_is_better": False,
        }
    return {
        "metric_name": "mean_plddt",
        "metric_label": "Mean pLDDT",
        "delta_label": "ΔpLDDT",
        "higher_is_better": True,
    }


def _format_baseline_summary(baseline_results: Dict[str, Dict[str, Any]], label_by_value: Dict[str, str]) -> str:
    lines = []
    for pred_id, data in baseline_results.items():
        name = label_by_value.get(pred_id, pred_id)
        metric_label = data.get("metric_label") or _score_semantics(pred_id)["metric_label"]
        metric_value = data.get("metric_value")
        if metric_value is None:
            metric_value = data.get("mean_plddt")
        direction = "high=better" if data.get("higher_is_better", pred_id != "immunebuilder") else "low=better"
        runtime = data.get("runtime_seconds")
        status = "OK" if data.get("success") else "FAILED"
        mean_str = f"{metric_value:.2f}" if metric_value is not None else "N/A"
        runtime_str = f"{runtime:.1f}s" if runtime else "N/A"
        lines.append(
            f"- {name}: {metric_label}={mean_str} ({direction}), "
            f"runtime={runtime_str}, status={status}"
        )
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
                st.session_state.mutagenesis_job_dir = str(job_path)
                st.session_state.active_job_dir = str(job_path)
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
                st.session_state.mutagenesis_job_dir = str(job_path)
                st.session_state.active_job_dir = str(job_path)
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
        'mutagenesis_job_dir': "",
        'scan_error': None,
        'multi_scan_error': None,
        'last_scan_request': None,
        'last_multi_scan_request': None,
        'scanner': _create_scanner_compat(
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

def _expert_review_overrides() -> Tuple[str, str]:
    """Return provider/model overrides for expert panels on this page."""
    def _norm_ollama(model_name: str) -> str:
        try:
            from protein_design_hub.core.config import normalize_ollama_model_name
            return normalize_ollama_model_name(model_name)
        except Exception:
            raw = str(model_name or "")
            return "qwen2.5:14b" if raw.strip().lower() in {"llama3.2", "llama3.2:latest"} else raw

    mode = st.session_state.get("mut_review_provider", "current")
    model = (st.session_state.get("mut_review_model") or "").strip()
    custom_provider = (st.session_state.get("mut_review_custom_provider") or "").strip()

    if mode == "current":
        return "", model
    if mode == "custom":
        return custom_provider, model
    if mode == "ollama":
        return "ollama", _norm_ollama(model or "qwen2.5:14b")
    if mode == "deepseek":
        return "deepseek", model or "deepseek-chat"
    return mode, model

def current_eval_metrics() -> List[str]:
    if not st.session_state.get("mutation_eval_enabled"):
        return []
    return st.session_state.get("mutation_eval_metrics", [])


def _create_scanner_compat(**kwargs) -> MutationScanner:
    """Create MutationScanner across old/new signatures.

    Some environments may still resolve an older installed package version
    where `run_openstructure_comprehensive` is not accepted yet.
    """
    try:
        return MutationScanner(**kwargs)
    except TypeError as exc:
        if "run_openstructure_comprehensive" not in str(exc):
            raise
        fallback_kwargs = dict(kwargs)
        run_ost = bool(fallback_kwargs.pop("run_openstructure_comprehensive", False))
        scanner = MutationScanner(**fallback_kwargs)
        # Best-effort runtime compatibility for downstream checks.
        setattr(scanner, "run_openstructure_comprehensive", run_ost)
        return scanner

def build_scanner(predictor_id: str) -> MutationScanner:
    eval_metrics = current_eval_metrics()
    run_ost = bool(st.session_state.get("mutation_ost_comprehensive", False))
    if predictor_id == "immunebuilder":
        return _create_scanner_compat(
            predictor=predictor_id,
            immunebuilder_mode=st.session_state.immunebuilder_mode,
            immune_chain_a=st.session_state.immune_chain_a,
            immune_chain_b=st.session_state.immune_chain_b,
            immune_active_chain=st.session_state.immune_active_chain,
            evaluation_metrics=eval_metrics,
            run_openstructure_comprehensive=run_ost,
        )
    return _create_scanner_compat(
        predictor=predictor_id,
        evaluation_metrics=eval_metrics,
        run_openstructure_comprehensive=run_ost,
    )

init_session_state()

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
        try:
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
            return results, None
        except Exception as exc:
            status.update(label="Scan failed", state="error", expanded=True)
            return None, str(exc)

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


def _ensure_mutagenesis_job_dir() -> Path:
    """Return active mutagenesis job dir, creating one when needed."""
    current = (st.session_state.get("mutagenesis_job_dir") or "").strip()
    if current:
        p = Path(current)
        if p.exists():
            return p

    from protein_design_hub.core.config import get_settings

    base = Path(get_settings().output.base_dir)
    job_id = f"mutagenesis_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir = base / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    st.session_state.mutagenesis_job_dir = str(job_dir)
    st.session_state.active_job_dir = str(job_dir)
    return job_dir


def _save_phase1_state(ctx: "WorkflowContext", job_dir: Path) -> None:
    """Persist Phase 1 results to disk for session resume (MUT-03).

    Serializes only the context.extra fields needed for Phase 2 — not the full
    WorkflowContext which contains non-JSON-serializable objects.
    """
    state = {
        "schema_version": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "job_id": ctx.job_id,
        "output_dir": str(ctx.output_dir),
        "sequence_id": ctx.sequences[0].id if ctx.sequences else "",
        "sequence": ctx.sequences[0].sequence if ctx.sequences else "",
        "mutation_suggestions": ctx.extra.get("mutation_suggestions"),
        "baseline_low_confidence_positions": ctx.extra.get("baseline_low_confidence_positions", []),
        "baseline_review": ctx.extra.get("baseline_review", ""),
        "mutation_suggestion_raw": ctx.extra.get("mutation_suggestion_raw", ""),
        "mutation_suggestion_source": ctx.extra.get("mutation_suggestion_source", "unknown"),
    }
    path = job_dir / "phase1_state.json"
    path.write_text(json.dumps(state, indent=2, default=str))


def _load_phase1_state(job_dir: Path) -> Optional["WorkflowContext"]:
    """Load Phase 1 results from a known job directory (MUT-04).

    Returns None if phase1_state.json does not exist or cannot be parsed.
    """
    path = job_dir / "phase1_state.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1:
            return None
        from protein_design_hub.agents.context import WorkflowContext
        from protein_design_hub.core.types import Sequence as ProteinSequence
        ctx = WorkflowContext(
            job_id=data["job_id"],
            output_dir=Path(data["output_dir"]),
        )
        ctx.job_dir = job_dir  # Critical: set explicitly to avoid mis-derived path
        if data.get("sequence"):
            ctx.sequences = [
                ProteinSequence(id=data.get("sequence_id", ""), sequence=data["sequence"])
            ]
        ctx.extra["mutation_suggestions"] = data.get("mutation_suggestions")
        ctx.extra["baseline_low_confidence_positions"] = data.get(
            "baseline_low_confidence_positions", []
        )
        ctx.extra["baseline_review"] = data.get("baseline_review", "")
        ctx.extra["mutation_suggestion_raw"] = data.get("mutation_suggestion_raw", "")
        ctx.extra["mutation_suggestion_source"] = data.get("mutation_suggestion_source", "unknown")
        return ctx
    except Exception:
        return None


def _find_latest_phase1_state() -> Optional["WorkflowContext"]:
    """Search output directory for the most recent phase1_state.json (MUT-04).

    Used when st.session_state.mutagenesis_job_dir is empty (browser close + reload).
    Looks in settings.output.base_dir for mutagenesis_session_* directories sorted by mtime.
    Returns reconstructed WorkflowContext or None if no valid state found.
    """
    try:
        from protein_design_hub.core.config import get_settings
        base_dir = Path(get_settings().output.base_dir)
        session_dirs = sorted(
            base_dir.glob("mutagenesis_session_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for session_dir in session_dirs:
            ctx = _load_phase1_state(session_dir)
            if ctx is not None:
                return ctx
        return None
    except Exception:
        return None


def _meeting_save_dir() -> Path:
    """Meeting directory scoped to the active mutagenesis job/session."""
    out = _ensure_mutagenesis_job_dir() / "meetings"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _activate_job_dir(job_dir: Path) -> None:
    """Track the latest mutagenesis result job for provenance."""
    st.session_state.mutagenesis_job_dir = str(job_dir)
    st.session_state.active_job_dir = str(job_dir)


def run_multi_mutation_pipeline(sequence, positions, top_k, max_variants, only_beneficial=True, max_positions=6):
    """Run multi-position mutation pipeline with ESMFold evaluation."""
    with st.status("Running Multi-Mutation Pipeline...", expanded=True) as status:
        try:
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
            return results, None
        except Exception as exc:
            status.update(label="Multi-scan failed", state="error", expanded=True)
            return None, str(exc)

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
    eval_metrics = st.session_state.get("mutation_eval_metrics", [])
    run_ost = bool(st.session_state.get("mutation_ost_comprehensive", False))
    with st.status("Running baseline comparison...", expanded=True) as status:
        for pred in predictors:
            st.write(f"🔬 {pred}...")
            sem = _score_semantics(pred)
            if pred == "immunebuilder":
                scanner = _create_scanner_compat(
                    predictor=pred,
                    immunebuilder_mode=immunebuilder_mode,
                    immune_chain_a=immune_chain_a,
                    immune_chain_b=immune_chain_b,
                    immune_active_chain=immune_active_chain,
                    evaluation_metrics=eval_metrics,
                    run_openstructure_comprehensive=run_ost,
                )
            else:
                scanner = _create_scanner_compat(
                    predictor=pred,
                    evaluation_metrics=eval_metrics,
                    run_openstructure_comprehensive=run_ost,
                )
            try:
                start = time.time()
                pdb, plddt, path = scanner.predict_single(sequence, f"baseline_{pred}")
                runtime = time.time() - start
                mean_plddt = sum(plddt) / len(plddt) if plddt else 0.0
                results[pred] = {
                    "mean_plddt": mean_plddt,
                    "metric_name": sem["metric_name"],
                    "metric_label": sem["metric_label"],
                    "metric_value": mean_plddt,
                    "higher_is_better": sem["higher_is_better"],
                    "runtime_seconds": runtime,
                    "throughput": (1.0 / runtime) if runtime > 0 else None,
                    "structure_path": str(path) if path else None,
                    "success": True,
                }
            except Exception as e:
                results[pred] = {
                    "mean_plddt": None,
                    "metric_name": sem["metric_name"],
                    "metric_label": sem["metric_label"],
                    "metric_value": None,
                    "higher_is_better": sem["higher_is_better"],
                    "runtime_seconds": None,
                    "throughput": None,
                    "structure_path": None,
                    "success": False,
                    "error": str(e),
                }
        status.update(label="Baseline comparison complete", state="complete", expanded=False)
    return results

# Static predictor label → ID mapping (used by both tabs)
predictor_options = {
    "ESM1 (legacy ESMFold v0)": "esmfold_v0",
    "ESMFold2 (ESM-2, local)": "esmfold_v1",
    "ESMFold API (ESM-2, <=400 aa)": "esmfold_api",
    "ESM3 (local or Forge)": "esm3",
    "ImmuneBuilder (antibody/nanobody/TCR)": "immunebuilder",
}
label_by_value = {v: k for k, v in predictor_options.items()}


def _render_manual_tab_settings():
    """Render all manual-tab-specific predictor/baseline/eval settings."""
    # Predictor selection
    show_advanced = st.checkbox(
        "Show advanced predictors",
        value=st.session_state.get("show_advanced_predictors", False),
        help="Show local ESMFold variants and ESM3 (requires separate environments).",
    )
    st.session_state.show_advanced_predictors = show_advanced

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

        # Quick-switch buttons for common local models
        st.markdown("**Quick model switch (local Ollama):**")
        qs_cols = st.columns(3)
        with qs_cols[0]:
            if st.button("qwen2.5:14b (fast)", key="qs_qwen", use_container_width=True):
                st.session_state.mut_review_provider = "ollama"
                st.session_state.mut_review_model = "qwen2.5:14b"
                st.session_state.mut_review_custom_provider = ""
                st.rerun()
        with qs_cols[1]:
            if st.button("deepseek-r1:14b (reasoning)", key="qs_dsr1", use_container_width=True):
                st.session_state.mut_review_provider = "ollama"
                st.session_state.mut_review_model = "deepseek-r1:14b"
                st.session_state.mut_review_custom_provider = ""
                st.rerun()
        with qs_cols[2]:
            if st.button("DeepSeek Cloud", key="qs_dscloud", use_container_width=True):
                st.session_state.mut_review_provider = "deepseek"
                st.session_state.mut_review_model = "deepseek-chat"
                st.session_state.mut_review_custom_provider = ""
                st.rerun()

        st.session_state.mut_review_provider = st.selectbox(
            "Expert panel provider",
            options=provider_options,
            index=provider_options.index(selected_provider),
            format_func=lambda x: {
                "current": "Current configured provider",
                "ollama": "Ollama (local, qwen2.5:14b / deepseek-r1:14b)",
                "deepseek": "DeepSeek Cloud (secondary check, requires API key)",
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
            current_override_model = st.session_state.get("mut_review_model", "")
            if str(current_override_model).strip().lower() in {"llama3.2", "llama3.2:latest"}:
                current_override_model = "qwen2.5:14b"
            st.session_state.mut_review_model = st.text_input(
                "Model override (optional)",
                value=current_override_model,
                key="mut_review_model_input",
                help="Leave empty to keep the globally configured model.",
            ).strip()
        else:
            st.session_state.mut_review_custom_provider = ""
            suggested_model = provider_default_model.get(current_override, "")
            if current_override == "ollama":
                try:
                    from protein_design_hub.core.config import normalize_ollama_model_name
                    suggested_model = normalize_ollama_model_name(suggested_model or "qwen2.5:14b")
                    current_model = normalize_ollama_model_name(
                        st.session_state.get("mut_review_model", suggested_model) or suggested_model
                    )
                except Exception:
                    current_model = st.session_state.get("mut_review_model", suggested_model) or suggested_model
                    if str(current_model).strip().lower() in {"llama3.2", "llama3.2:latest"}:
                        current_model = "qwen2.5:14b"
                # Offer recommended local models as a selectbox
                try:
                    from protein_design_hub.core.config import OLLAMA_RECOMMENDED_MODELS
                    rec_ids = [m[0] for m in OLLAMA_RECOMMENDED_MODELS]
                    rec_labels = {m[0]: f"{m[0]} — {m[1]}" for m in OLLAMA_RECOMMENDED_MODELS}
                except Exception:
                    rec_ids = ["qwen2.5:14b", "deepseek-r1:14b"]
                    rec_labels = {
                        "qwen2.5:14b": "qwen2.5:14b — Fast general-purpose (default)",
                        "deepseek-r1:14b": "deepseek-r1:14b — Deep reasoning",
                    }
                if current_model not in rec_ids:
                    rec_ids = [current_model] + rec_ids
                    rec_labels[current_model] = current_model
                sel_idx = rec_ids.index(current_model) if current_model in rec_ids else 0
                st.session_state.mut_review_model = st.selectbox(
                    "Local model",
                    options=rec_ids,
                    index=sel_idx,
                    format_func=lambda m: rec_labels.get(m, m),
                    key="mut_review_model_input",
                    help="qwen2.5:14b is fast; deepseek-r1:14b provides deeper reasoning.",
                )
            else:
                current_model = st.session_state.get("mut_review_model", suggested_model) or suggested_model
                st.session_state.mut_review_model = st.text_input(
                    "Model (optional)",
                    value=current_model,
                    key="mut_review_model_input",
                    help="Leave as default unless you need a specific model.",
                ).strip()
            if current_override == "deepseek":
                st.caption("Requires `DEEPSEEK_API_KEY` in the environment.")

    active_mut_job = (st.session_state.get("mutagenesis_job_dir") or "").strip()
    if active_mut_job:
        st.caption(f"Active mutagenesis session: `{active_mut_job}`")

    if st.session_state.get("baseline_results"):
        if st.session_state.get("baseline_sequence") != st.session_state.sequence:
            st.warning("Baseline results were computed for a different sequence. Re-run for current sequence.")
        baseline_rows = []
        for pred_id, data in st.session_state.baseline_results.items():
            metric_label = data.get("metric_label") or _score_semantics(pred_id)["metric_label"]
            metric_value = data.get("metric_value")
            if metric_value is None:
                metric_value = data.get("mean_plddt")
            direction = "high=better" if data.get("higher_is_better", pred_id != "immunebuilder") else "low=better"
            baseline_rows.append(
                {
                    "Predictor": label_by_value.get(pred_id, pred_id),
                    "Metric": metric_label,
                    "Value": metric_value,
                    "Direction": direction,
                    "Runtime (s)": data.get("runtime_seconds"),
                    "Throughput (structures/s)": data.get("throughput"),
                    "Status": "OK" if data.get("success") else "FAILED",
                    "Error": data.get("error", ""),
                }
            )
        st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True)

        # Auto-insight on baseline comparison
        baseline_insight_data = {}
        for pred_id, data in st.session_state.baseline_results.items():
            pred_name = label_by_value.get(pred_id, pred_id)
            if data.get("success"):
                baseline_insight_data[pred_name] = f"{data.get('metric_value', 'N/A'):.2f}" if data.get("metric_value") is not None else "N/A"
            else:
                baseline_insight_data[pred_name] = "FAILED"
        baseline_insight_data["Sequence length"] = len(st.session_state.get("sequence", ""))
        render_contextual_insight(
            "Mutation",
            baseline_insight_data,
            key_prefix="mut_baseline_ctx",
        )

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

        render_agent_advice_panel(
            page_context=context,
            default_question=(
                "Based on the baseline comparison, which residues should I target "
                "for mutagenesis? Consider confidence, surface exposure, and "
                "structural importance."
            ),
            expert="Protein Engineer",
            key_prefix="mut_baseline_advice",
        )

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
            save_dir=_meeting_save_dir(),
        )

        if base_eval:
            render_agent_advice_panel(
                page_context="Baseline evaluation summary:\n" + base_eval_summary,
                default_question=(
                    "Is this baseline structure quality sufficient for mutation planning? "
                    "What should I be cautious about?"
                ),
                expert="Protein Engineer",
                key_prefix="mut_baseval_advice",
            )

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
                save_dir=_meeting_save_dir(),
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

    return baseline_predictors, selected_predictor


# ── Two-tab layout: Agent Pipeline + Advanced / Manual ─────────
tab_agent, tab_manual = st.tabs(["Agent Pipeline", "Advanced / Manual"])


# ══════════════════════════════════════════════════════════════════
#  TAB 1 — Agent-Governed Mutagenesis Pipeline
# ══════════════════════════════════════════════════════════════════
_EXAMPLE_SEQUENCES = {
    "Ubiquitin (76 aa)": ("Ubiquitin", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"),
    "T1024 (52 aa)": ("T1024", "MAAHKGAEHVVKASLDAGVKTVAGGLVVKAKALGGKDATMHLVAATLKKGYM"),
    "Hemoglobin alpha (51 aa)": ("HbA_fragment", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"),
    "Insulin B-chain (30 aa)": ("Insulin_B", "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
}


def _render_agent_pipeline_tab():
    """Render the agent-governed mutagenesis pipeline UI."""

    section_header(
        "Agent-Governed Mutagenesis",
        "LLM scientists drive the workflow: analyse baseline, suggest mutations, execute, interpret",
        "🤖",
    )

    # ── A. Sequence Input ────────────────────────────────────────
    seq = st.session_state.get("sequence", "")
    if not seq:
        st.markdown("#### Load a sequence")

        # Example sequence buttons
        st.markdown("**Quick Load — Example Sequences:**")
        ex_cols = st.columns(len(_EXAMPLE_SEQUENCES))
        for i, (label, (name, sequence)) in enumerate(_EXAMPLE_SEQUENCES.items()):
            with ex_cols[i]:
                if st.button(label, key=f"agent_ex_{i}", use_container_width=True):
                    st.session_state.sequence = sequence
                    st.session_state.sequence_name = name
                    st.session_state.sequence_input_raw = sequence
                    st.session_state.mutagenesis_context = None
                    st.session_state.mutagenesis_phase2_context = None
                    st.rerun()

        st.markdown("**Or paste your own sequence:**")
        agent_seq = st.text_area(
            "Protein sequence",
            height=100,
            placeholder="Paste amino acid sequence (single letter codes)...",
            key="agent_seq_input",
        )
        if agent_seq:
            cleaned = "".join(c.upper() for c in agent_seq if c.upper() in _VALID_AAS)
            if cleaned:
                st.session_state.sequence = cleaned
                st.session_state.sequence_name = "agent_input"
                st.session_state.sequence_input_raw = cleaned
                seq = cleaned
                st.rerun()
            else:
                st.error("No valid amino acid characters found.")
                return
        else:
            return

    # Sequence is loaded — show info and allow changing
    col_seq_info, col_seq_actions = st.columns([3, 1])
    with col_seq_info:
        _mw = len(seq) * 110 / 1000
        _hydrophobic = sum(1 for aa in seq if aa in "AVILMFYWP")
        _charged = sum(1 for aa in seq if aa in "DEKRH")
        _hpct = 100 * _hydrophobic / max(len(seq), 1)
        _cpct = 100 * _charged / max(len(seq), 1)
        st.success(
            f"Sequence loaded: **{st.session_state.get('sequence_name', 'protein')}** "
            f"· {len(seq)} aa · ~{_mw:.1f} kDa · "
            f"{_hpct:.0f}% hydrophobic · {_cpct:.0f}% charged"
        )
    with col_seq_actions:
        if st.button("Change Sequence", key="agent_change_seq", use_container_width=True):
            st.session_state.sequence = ""
            st.session_state.mutagenesis_context = None
            st.session_state.mutagenesis_phase2_context = None
            st.rerun()

    if len(seq) > 400:
        st.warning(
            f"Sequence is {len(seq)} residues — ESMFold API limit is 400. "
            "The pipeline will attempt prediction but may fail. "
            "Consider using local ESMFold for longer sequences."
        )

    # ── B. Phase 1: Understanding + Baseline + Suggestion ────────
    st.markdown("---")
    st.markdown("### Phase 1: Analyse Baseline & Suggest Mutations")

    # Check LLM availability
    llm_ok = False
    try:
        from protein_design_hub.core.config import get_settings
        cfg = get_settings().llm.resolve()
        llm_ok = bool(cfg.base_url and cfg.model)
    except Exception:
        pass

    if not llm_ok:
        st.warning(
            "LLM is not configured. The agent pipeline requires an LLM provider. "
            "Configure one in the Settings page or use the **Advanced / Manual** tab."
        )

    # MUT-04: auto-load Phase 1 results from disk if not already in session state
    phase1_ctx = st.session_state.get("mutagenesis_context")
    if phase1_ctx is None:
        # Try known job dir first (same session, page refresh)
        _stored_job_dir = st.session_state.get("mutagenesis_job_dir", "")
        _loaded_ctx = None
        if _stored_job_dir:
            _loaded_ctx = _load_phase1_state(Path(_stored_job_dir))
        # Fall back to searching all session dirs (browser close + reload)
        if _loaded_ctx is None:
            _loaded_ctx = _find_latest_phase1_state()
        if _loaded_ctx is not None:
            st.session_state.mutagenesis_context = _loaded_ctx
            phase1_ctx = _loaded_ctx
            st.caption("Loaded from previous session")

    phase1_done = phase1_ctx is not None and phase1_ctx.extra.get("mutation_suggestions") is not None

    if not phase1_done:
        if st.button(
            "Run Agent Analysis",
            disabled=not llm_ok,
            use_container_width=True,
            type="primary",
            key="run_phase1",
        ):
            _run_phase1(seq)
            st.rerun()
    else:
        with st.expander("Phase 1 completed — show agent summaries", expanded=False):
            _show_phase1_summaries(phase1_ctx)

        # ── C. Human Approval ────────────────────────────────────
        st.markdown("---")
        st.markdown("### Review & Approve Mutations")

        approved = phase1_ctx.extra.get("approved_mutations", [])
        if not approved and not st.session_state.get("_phase2_confirmed", False):
            # No approval yet — show the approval table
            _render_approval_step(phase1_ctx)
        elif not approved and st.session_state.get("_phase2_confirmed", False):
            # User confirmed bypass via dialog — proceed with Phase 2 (automatic fallback)
            _run_phase2(phase1_ctx)
            st.session_state._phase2_confirmed = False  # reset sentinel after use
            st.rerun()
        else:
            _render_approval_step(phase1_ctx)

    # ── D. Phase 2 Results ───────────────────────────────────────
    phase2_ctx = st.session_state.get("mutagenesis_phase2_context")
    if phase2_ctx is not None:
        st.markdown("---")
        st.markdown("### Phase 2: Results")
        _render_phase2_results(phase2_ctx)


_VALID_AAS_SET = set("ACDEFGHIKLMNPQRSTVWY")


def _run_phase1(sequence: str):
    """Execute Phase 1 mutagenesis pipeline."""
    import tempfile

    from protein_design_hub.agents.orchestrator import AgentOrchestrator
    from protein_design_hub.web.agent_helpers import _temporary_llm_override

    provider, model = _expert_review_overrides()

    # Write sequence to temp FASTA
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False,
    ) as f:
        f.write(f">{st.session_state.get('sequence_name', 'protein')}\n")
        f.write(sequence + "\n")
        fasta_path = Path(f.name)

    with st.status("Running Phase 1 — Agent Analysis...", expanded=True) as status:
        step_container = st.empty()
        steps_seen: list[str] = []

        def _progress(stage, item, current, total):
            from protein_design_hub.agents.orchestrator import _AGENT_LABELS
            label = _AGENT_LABELS.get(item, item)
            steps_seen.append(label)
            step_container.markdown(
                f"**[{current}/{total}]** {label}"
            )

        orchestrator = AgentOrchestrator(
            mode="mutagenesis_pre",
            progress_callback=_progress,
            allow_failed_llm_steps=True,
        )

        with _temporary_llm_override(provider, model):
            result = orchestrator.run(
                input_path=fasta_path,
                predictors=["esmfold_api"],
            )

        if result.success and result.context:
            st.session_state.mutagenesis_context = result.context
            # MUT-03: persist Phase 1 results to disk for session resume
            try:
                job_dir = _ensure_mutagenesis_job_dir()
                _save_phase1_state(result.context, job_dir)
            except Exception as _save_err:
                logger.warning("Phase 1 state save failed: %s", _save_err)
            status.update(label="Phase 1 complete!", state="complete")
        else:
            status.update(label="Phase 1 failed", state="error")
            st.error(f"Pipeline failed: {result.message}")

    # Clean up temp file
    try:
        fasta_path.unlink(missing_ok=True)
    except Exception:
        pass


def _show_phase1_summaries(ctx):
    """Display Phase 1 meeting summaries."""
    summary_keys = [
        ("input_review", "Input Review"),
        ("prediction_review", "Prediction Review"),
        ("evaluation_review", "Evaluation Review"),
        ("baseline_review", "Baseline Review"),
        ("mutation_suggestion_raw", "Mutation Suggestions"),
    ]
    for key, label in summary_keys:
        text = ctx.extra.get(key, "")
        if text:
            with st.expander(f"**{label}**"):
                st.markdown(text[:2000])

    # Show verdicts
    if ctx.step_verdicts:
        cols = st.columns(min(len(ctx.step_verdicts), 4))
        for i, (step, verdict) in enumerate(ctx.step_verdicts.items()):
            status = verdict.get("status", "?")
            color = {"PASS": "green", "WARN": "orange", "FAIL": "red"}.get(status, "gray")
            with cols[i % len(cols)]:
                st.markdown(
                    f"<div style='text-align:center; padding:8px; "
                    f"border-radius:8px; border:2px solid {color};'>"
                    f"<b>{step.replace('_', ' ').title()}</b><br>"
                    f"<span style='color:{color}; font-size:1.2em;'>{status}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


@st.dialog("Run Phase 2 without approved mutations?")
def _confirm_phase2_no_approval() -> None:
    """Confirmation dialog when Phase 2 is triggered without prior approval."""
    st.warning(
        "No mutations have been explicitly approved. "
        "Phase 2 will use the automatic fallback (saturation at low-confidence positions)."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Proceed anyway", type="primary"):
            st.session_state._phase2_confirmed = True
            st.rerun()
    with col2:
        if st.button("Cancel"):
            st.rerun()


def _render_approval_step(ctx):
    """Render the mutation approval UI with editable dataframe."""
    suggestions = ctx.extra.get("mutation_suggestions")
    source = ctx.extra.get("mutation_suggestion_source", "unknown")

    if not suggestions or not suggestions.get("positions"):
        st.warning("No mutation suggestions were generated. Try re-running Phase 1.")
        if st.button("Reset Phase 1", key="reset_phase1"):
            st.session_state.mutagenesis_context = None
            st.rerun()
        return

    if source == "fallback":
        st.info(
            "LLM mutation plan could not be parsed. Using fallback: "
            "saturation mutagenesis at low-confidence positions."
        )

    st.markdown(f"**Strategy:** {suggestions.get('strategy', 'N/A')}")
    if suggestions.get("rationale"):
        st.caption(suggestions["rationale"])

    # Build editable dataframe
    positions = suggestions["positions"]
    df_data = []
    for p in positions:
        targets = p.get("targets", ["*"])
        targets_str = ", ".join(targets) if targets != ["*"] else "All 19 AAs"
        df_data.append({
            "Include": True,
            "Position": p["residue"],
            "WT AA": p["wt_aa"],
            "Target AAs": targets_str,
            "Rationale": p.get("rationale", ""),
        })

    df = pd.DataFrame(df_data)

    st.markdown("Edit the table below — uncheck **Include** to skip positions, modify targets as needed.")
    edited_df = st.data_editor(
        df,
        column_config={
            "Include": st.column_config.CheckboxColumn("Include", default=True),
            "Position": st.column_config.NumberColumn("Position", disabled=True),
            "WT AA": st.column_config.TextColumn("WT AA", disabled=True),
            "Target AAs": st.column_config.TextColumn(
                "Target AAs",
                help="Comma-separated target amino acids, or 'All 19 AAs' for saturation",
            ),
            "Rationale": st.column_config.TextColumn("Rationale", disabled=True),
        },
        use_container_width=True,
        num_rows="fixed",
        key="mutation_editor",
    )

    # Count variants
    included = edited_df[edited_df["Include"] == True]
    total_variants = 0
    for _, row in included.iterrows():
        targets_str = row["Target AAs"].strip()
        if targets_str in ("All 19 AAs", "*"):
            total_variants += 19
        else:
            total_variants += len([t.strip() for t in targets_str.split(",") if t.strip()])

    st.caption(f"**{len(included)} positions selected, ~{total_variants} variants to test**")

    try:
        from protein_design_hub.core.config import get_settings
        cfg = get_settings().llm
        _ov_provider, _ov_model = _expert_review_overrides()
        effective_provider = _ov_provider if _ov_provider and _ov_provider != "current" else cfg.provider
        effective_model = _ov_model if _ov_model else cfg.model
        st.caption(f"Using: `{effective_model}` @ `{effective_provider}`")
    except Exception:
        pass  # Caption is informational; never block the UI

    # Position count and OST override
    _n_selected = int(edited_df["Include"].sum()) if "Include" in edited_df.columns else len(edited_df)
    _distinct_positions = _n_selected  # each row = one position
    st.caption(f"{_distinct_positions} position(s) selected.")
    if _distinct_positions > 3:
        _force_ost = st.checkbox(
            "Force OST scoring even with >3 positions (slow — may take hours)",
            value=False,
            key="force_ost_override",
            help="OST comprehensive scoring is auto-disabled when >3 positions to prevent multi-hour runs. "
                 "Enable this to override.",
        )
        ctx.extra["ost_force_override"] = _force_ost
    else:
        ctx.extra["ost_force_override"] = False

    col1, col2 = st.columns(2)
    with col1:
        approve_disabled = len(included) == 0
        if approve_disabled:
            st.error("Select at least one mutation to approve.")
        if st.button(
            "Approve & Continue",
            disabled=approve_disabled,
            use_container_width=True,
            type="primary",
            key="approve_mutations",
        ):
            # Parse approved mutations
            approved = _parse_approved_mutations(included, ctx.sequences[0].sequence)
            ctx.extra["approved_mutations"] = approved
            st.session_state.mutagenesis_context = ctx
            st.session_state._phase2_confirmed = False  # reset sentinel
            _run_phase2(ctx)
            st.rerun()

    with col2:
        if st.button("Reset Pipeline", use_container_width=True, key="reset_all"):
            st.session_state.mutagenesis_context = None
            st.session_state.mutagenesis_phase2_context = None
            st.rerun()


def _parse_approved_mutations(df, sequence: str) -> list:
    """Parse the edited approval dataframe into the format expected by MutationExecutionAgent."""
    approved = []
    for _, row in df.iterrows():
        pos = int(row["Position"])
        wt = row["WT AA"]
        targets_str = str(row["Target AAs"]).strip()
        if targets_str in ("All 19 AAs", "*"):
            targets = ["*"]
        else:
            targets = [t.strip().upper() for t in targets_str.split(",") if t.strip()]
            targets = [t for t in targets if t in _VALID_AAS_SET and t != wt]
        if targets:
            approved.append({
                "residue": pos,
                "wt_aa": wt,
                "targets": targets,
            })
    return approved


def _run_phase2(ctx):
    """Execute Phase 2 mutagenesis pipeline."""
    approved = ctx.extra.get("approved_mutations", [])
    if not approved and not st.session_state.get("_phase2_confirmed", False):
        st.error(
            "No approved mutations found. Please use the approval table above "
            "to review and approve mutations before running Phase 2."
        )
        return

    # PERF-02: Surface LLM saturation fallback warning before mutations execute
    _fallback_warn = ctx.extra.get("mutation_suggestion_warning")
    if _fallback_warn:
        st.warning(_fallback_warn)

    # PERF-01: Surface OST auto-disable warning before mutations execute
    if ctx.extra.get("ost_auto_disabled"):
        st.warning(ctx.extra.get("ost_auto_disabled_reason", "OST scoring was automatically disabled."))

    from protein_design_hub.agents.orchestrator import AgentOrchestrator
    from protein_design_hub.web.agent_helpers import _temporary_llm_override

    provider, model = _expert_review_overrides()

    with st.status("Running Phase 2 — Executing Mutations...", expanded=True) as status:
        step_container = st.empty()

        def _progress(stage, item, current, total):
            from protein_design_hub.agents.orchestrator import _AGENT_LABELS
            label = _AGENT_LABELS.get(item, item)
            step_container.markdown(f"**[{current}/{total}]** {label}")

        orchestrator = AgentOrchestrator(
            mode="mutagenesis_post",
            progress_callback=_progress,
            allow_failed_llm_steps=True,
        )

        with _temporary_llm_override(provider, model):
            result = orchestrator.run_with_context(ctx)

        if result.success and result.context:
            st.session_state.mutagenesis_phase2_context = result.context
            status.update(label="Phase 2 complete!", state="complete")
        else:
            status.update(label="Phase 2 failed", state="error")
            st.error(f"Phase 2 failed: {result.message}")


def _render_phase2_results(ctx):
    """Render Phase 2 mutation results."""
    comparison = ctx.extra.get("mutation_comparison", {})
    if not comparison:
        st.warning("No comparison results available.")
        return

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tested", comparison.get("total_mutations", 0))
    c2.metric("Successful", comparison.get("successful_count", 0))
    c3.metric("Beneficial", comparison.get("beneficial_count", 0))
    c4.metric("Detrimental", comparison.get("detrimental_count", 0))

    # Ranked mutations table
    ranked = comparison.get("ranked_mutations", [])
    if ranked:
        st.markdown("#### Ranked Mutations")
        table_data = []
        for i, r in enumerate(ranked, 1):
            rmsd = r.get("rmsd_to_base")
            table_data.append({
                "Rank": i,
                "Mutation": r.get("mutation_code", "?"),
                "Score": round(r.get("improvement_score", 0), 3),
                "Delta pLDDT": round(r.get("delta_mean_plddt", 0), 2),
                "RMSD": round(rmsd, 2) if rmsd is not None else None,
                "Clash": round(r.get("clash_score", 0), 1) if r.get("clash_score") is not None else None,
            })
        result_df = pd.DataFrame(table_data)

        # Color-code the Score column
        def _color_score(val):
            if val is None:
                return ""
            if val > 0:
                return "color: green"
            elif val < -0.5:
                return "color: red"
            return ""

        styled = result_df.style.map(_color_score, subset=["Score", "Delta pLDDT"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Mutation ranking chart (REP-01)
    if ranked:
        st.markdown("#### Mutation Ranking Chart")
        _render_ranking_chart(ranked)

    # Per-residue pLDDT chart (REP-02)
    wt_per_res = ctx.extra.get("mutation_wt_plddt_per_residue", [])
    if wt_per_res:
        st.markdown("#### Per-Residue pLDDT")
        _render_plddt_chart(ranked, wt_per_res)
    elif ranked:
        st.info("Per-residue pLDDT chart not available (data missing from Phase 2 run).")

    # OST structural metrics table (REP-03)
    if ranked:
        _render_ost_table(ranked)

    # Best mutation highlight
    best = comparison.get("best_overall")
    if best and best.get("improvement_score", 0) > 0:
        st.success(
            f"Best mutation: **{best.get('mutation_code', '?')}** — "
            f"improvement score {best.get('improvement_score', 0):.3f}, "
            f"delta pLDDT {best.get('delta_mean_plddt', 0):+.2f}"
        )

    # LLM interpretation
    interpretation = ctx.extra.get("mutation_interpretation", "")
    if interpretation:
        with st.expander("LLM Interpretation", expanded=True):
            st.markdown(interpretation[:3000])

    # Heatmap for saturation positions
    by_position = comparison.get("by_position", {})
    for pos_str, pos_data in by_position.items():
        all_muts = pos_data.get("all", [])
        if len(all_muts) >= 10:  # Likely saturation
            with st.expander(f"Saturation Heatmap — Position {pos_str}"):
                _render_saturation_heatmap(all_muts, int(pos_str))

    # Download buttons
    st.markdown("---")
    dl_col1, dl_col2 = st.columns(2)

    job_dir = ctx.job_dir
    if job_dir:
        json_path = job_dir / "mutagenesis" / "mutagenesis_report.json"
        md_path = job_dir / "mutagenesis" / "mutagenesis_summary.md"

        if json_path.exists():
            with dl_col1:
                st.download_button(
                    "Download JSON Report",
                    data=json_path.read_text(),
                    file_name="mutagenesis_report.json",
                    mime="application/json",
                    key="dl_json",
                )
        if md_path.exists():
            with dl_col2:
                st.download_button(
                    "Download Markdown Summary",
                    data=md_path.read_text(),
                    file_name="mutagenesis_summary.md",
                    mime="text/markdown",
                    key="dl_md",
                )

    # Export buttons (PDF + HTML) — cached in session_state to avoid rebuild per Streamlit re-run
    if ranked:
        st.markdown("#### Export Report")
        exp_col1, exp_col2 = st.columns(2)

        # Cache key tied to current context id — invalidated if page reloads with new context
        ctx_key = id(ctx)
        if st.session_state.get("_report_ctx_key") != ctx_key:
            # Context changed — clear stale cache
            st.session_state.pop("cached_report_pdf", None)
            st.session_state.pop("cached_report_html", None)
            st.session_state["_report_ctx_key"] = ctx_key

        with exp_col1:
            if "cached_report_pdf" not in st.session_state:
                with st.spinner("Building PDF..."):
                    st.session_state["cached_report_pdf"] = _build_report_pdf(ctx, comparison)
            st.download_button(
                "Export PDF",
                data=st.session_state["cached_report_pdf"],
                file_name="mutagenesis_report.pdf",
                mime="application/pdf",
                key="dl_pdf",
                use_container_width=True,
            )

        with exp_col2:
            if "cached_report_html" not in st.session_state:
                with st.spinner("Building HTML..."):
                    st.session_state["cached_report_html"] = _build_report_html(ctx, comparison)
            st.download_button(
                "Export HTML",
                data=st.session_state["cached_report_html"],
                file_name="mutagenesis_report.html",
                mime="text/html",
                key="dl_html",
                use_container_width=True,
            )


def _build_ranking_figure(ranked: list) -> "go.Figure":
    """Bar chart of improvement score per mutation, colored by category.
    Used by both _render_ranking_chart (Streamlit display) and _build_report_pdf/_build_report_html (export).
    Single go.Bar trace with per-bar marker_color — preserves sorted x-axis order.
    """
    CATEGORY_COLORS = {"beneficial": "#22c55e", "neutral": "#9ca3af", "detrimental": "#ef4444"}

    def _cat(score: float) -> str:
        if score > 0:
            return "beneficial"
        if score < -0.5:
            return "detrimental"
        return "neutral"

    mutations = [r.get("mutation_code", "?") for r in ranked]
    scores = [r.get("improvement_score", 0) for r in ranked]
    cats = [_cat(s) for s in scores]
    colors = [CATEGORY_COLORS[c] for c in cats]

    fig = go.Figure(data=go.Bar(
        x=mutations,
        y=scores,
        marker_color=colors,
        text=cats,
        hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Mutation Ranking by Improvement Score",
        xaxis_title="Mutation",
        yaxis_title="Improvement Score",
        height=400,
        showlegend=False,
    )
    return fig


def _build_plddt_figure(ranked: list, wt_per_res: list) -> "go.Figure":
    """Per-residue pLDDT line chart: Wildtype vs top 3 mutants.
    Used by both Streamlit display and export functions.
    Guards against missing plddt_per_residue (saturation failure paths).
    """
    MUTANT_COLORS = ["#22c55e", "#f59e0b", "#ef4444"]
    fig = go.Figure()
    if wt_per_res:
        residues = list(range(1, len(wt_per_res) + 1))
        fig.add_trace(go.Scatter(
            x=residues,
            y=wt_per_res,
            name="Wildtype",
            line={"color": "#6366f1", "width": 2},
        ))
    for mut, color in zip(ranked[:3], MUTANT_COLORS):
        per_res = mut.get("plddt_per_residue") or []
        if not per_res:
            continue
        residues = list(range(1, len(per_res) + 1))
        fig.add_trace(go.Scatter(
            x=residues,
            y=per_res,
            name=mut.get("mutation_code", "?"),
            line={"color": color, "width": 1.5, "dash": "dot"},
        ))
    fig.update_layout(
        title="Per-Residue pLDDT: Wildtype vs Top Mutants",
        xaxis_title="Residue",
        yaxis_title="pLDDT",
        yaxis={"range": [0, 100]},
        height=400,
    )
    return fig


def _render_ranking_chart(ranked: list) -> None:
    """Render mutation ranking bar chart in Streamlit (REP-01)."""
    fig = _build_ranking_figure(ranked)
    st.plotly_chart(fig, use_container_width=True)


def _render_plddt_chart(ranked: list, wt_per_res: list) -> None:
    """Render per-residue pLDDT comparison chart in Streamlit (REP-02)."""
    fig = _build_plddt_figure(ranked, wt_per_res)
    st.plotly_chart(fig, use_container_width=True)


def _render_ost_table(ranked: list) -> None:
    """OST structural metrics table, shown only when OST was enabled (REP-03).
    Keys are on each ranked mutation dict: ost_lddt, ost_rmsd_ca, ost_qs_score.
    Access via ranked[i].get("ost_lddt") NOT via ctx.extra["ost_metrics"] (that key does not exist).
    """
    ost_rows = [r for r in ranked if r.get("ost_lddt") is not None]
    if not ost_rows:
        return  # OST was disabled — silently skip

    rows = []
    for r in ost_rows:
        rows.append({
            "Mutation": r.get("mutation_code", "?"),
            "Score": round(r.get("improvement_score", 0), 3),
            "lDDT": round(r["ost_lddt"], 3),
            "RMSD (CA)": round(r["ost_rmsd_ca"], 2) if r.get("ost_rmsd_ca") is not None else None,
            "QS-score": round(r["ost_qs_score"], 3) if r.get("ost_qs_score") is not None else None,
        })

    st.markdown("#### OST Structural Metrics")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _strip_for_pdf(text: str) -> str:
    """Remove characters fpdf2 Helvetica cannot render (emoji, CJK, high-Unicode).
    Also strip common markdown markers for cleaner PDF text.
    """
    # Remove non-latin-1 characters
    text = re.sub(r"[^\x00-\xFF]+", "", text)
    # Strip markdown bold/italic/header markers
    text = re.sub(r"[*#`]+", "", text)
    return text.strip()


def _embed_fig_in_pdf(pdf: "FPDF", fig: "go.Figure", width_mm: int = 180) -> None:
    """Export plotly figure to PNG temp file, embed in PDF, delete temp file.
    try/finally ensures cleanup even on exception (prevents /tmp accumulation).
    fpdf2 image() requires file path — bytes not accepted.
    """
    png_bytes = fig.to_image(format="png", width=700, height=350)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_bytes)
        tmp_path = f.name
    try:
        pdf.image(tmp_path, w=width_mm)
    finally:
        os.unlink(tmp_path)


def _build_report_pdf(ctx, comparison: dict) -> bytes:
    """Build PDF mutagenesis report as bytes.
    Returns bytes (not bytearray) — st.download_button requires bytes.
    Uses _embed_fig_in_pdf for try/finally temp file cleanup.
    OST table capped at 20 rows to prevent large-count slowness.
    Narrative stripped of non-latin-1 and markdown via _strip_for_pdf.
    """
    ranked = comparison.get("ranked_mutations", [])
    wt_per_res = ctx.extra.get("mutation_wt_plddt_per_residue", [])
    interpretation = ctx.extra.get("mutation_interpretation", "")

    ranking_fig = _build_ranking_figure(ranked)
    plddt_fig = _build_plddt_figure(ranked, wt_per_res) if wt_per_res else None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Mutagenesis Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)

    # Summary
    pdf.set_font("Helvetica", size=11)
    best = comparison.get("best_overall")
    if best:
        line = (
            f"Best mutation: {best.get('mutation_code', '?')} "
            f"(score {best.get('improvement_score', 0):.3f})"
        )
        pdf.cell(0, 8, _strip_for_pdf(line), new_x="LMARGIN", new_y="NEXT")
    total = comparison.get("total_mutations", 0)
    beneficial = comparison.get("beneficial_count", 0)
    detrimental = comparison.get("detrimental_count", 0)
    pdf.cell(
        0, 8,
        f"Total: {total}  |  Beneficial: {beneficial}  |  Detrimental: {detrimental}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(3)

    # Ranking chart
    if ranked:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Mutation Ranking", new_x="LMARGIN", new_y="NEXT")
        _embed_fig_in_pdf(pdf, ranking_fig)
        pdf.ln(4)

    # pLDDT chart
    if plddt_fig is not None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Per-Residue pLDDT", new_x="LMARGIN", new_y="NEXT")
        _embed_fig_in_pdf(pdf, plddt_fig)
        pdf.ln(4)

    # OST metric table
    ost_rows = [r for r in ranked if r.get("ost_lddt") is not None]
    if ost_rows:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "OST Structural Metrics", new_x="LMARGIN", new_y="NEXT")
        col_widths = [40, 30, 30, 35, 35]
        headers = ["Mutation", "Score", "lDDT", "RMSD (CA)", "QS-score"]
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(241, 245, 249)
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 8, h, border=1, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", size=10)
        for i, r in enumerate(ost_rows[:20]):
            fill = i % 2 == 1
            if fill:
                pdf.set_fill_color(248, 250, 252)
            vals = [
                r.get("mutation_code", "?"),
                f"{r.get('improvement_score', 0):.3f}",
                f"{r.get('ost_lddt', 0):.3f}",
                f"{r['ost_rmsd_ca']:.2f}" if r.get("ost_rmsd_ca") is not None else "-",
                f"{r['ost_qs_score']:.3f}" if r.get("ost_qs_score") is not None else "-",
            ]
            for w, v in zip(col_widths, vals):
                pdf.cell(w, 7, v, border=1, fill=fill)
            pdf.ln()
        pdf.ln(4)

    # Narrative summary
    if interpretation:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Narrative Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 6, _strip_for_pdf(interpretation[:3000]))

    return bytes(pdf.output())


def _build_report_html(ctx, comparison: dict) -> str:
    """Build self-contained HTML mutagenesis report.
    Uses base64-embedded PNG images (no CDN, no internet required).
    html.escape() applied to LLM narrative to prevent broken HTML from <, >, & in output.
    Output ~60 KB for 3-chart report.
    """
    import html as html_mod

    ranked = comparison.get("ranked_mutations", [])
    wt_per_res = ctx.extra.get("mutation_wt_plddt_per_residue", [])
    interpretation = ctx.extra.get("mutation_interpretation", "")

    def _to_b64_png(fig: "go.Figure") -> str:
        return base64.b64encode(
            fig.to_image(format="png", width=700, height=350)
        ).decode("utf-8")

    ranking_b64 = ""
    if ranked:
        ranking_b64 = _to_b64_png(_build_ranking_figure(ranked))

    plddt_b64 = ""
    if wt_per_res:
        plddt_b64 = _to_b64_png(_build_plddt_figure(ranked, wt_per_res))

    # Summary block
    best = comparison.get("best_overall")
    summary_html = ""
    if best:
        summary_html = (
            f"<div class='summary'>"
            f"<p><strong>Best mutation:</strong> {html_mod.escape(str(best.get('mutation_code', '?')))} "
            f"(score: {best.get('improvement_score', 0):.3f})</p>"
            f"<p>Total: {comparison.get('total_mutations', 0)} | "
            f"Beneficial: {comparison.get('beneficial_count', 0)} | "
            f"Detrimental: {comparison.get('detrimental_count', 0)}</p></div>"
        )

    # Charts
    ranking_section = (
        f"<h2>Mutation Ranking</h2>"
        f"<img src='data:image/png;base64,{ranking_b64}' alt='Mutation Ranking Chart'>"
        if ranking_b64 else ""
    )
    plddt_section = (
        f"<h2>Per-Residue pLDDT</h2>"
        f"<img src='data:image/png;base64,{plddt_b64}' alt='pLDDT Chart'>"
        if plddt_b64 else ""
    )

    # OST table
    ost_rows = [r for r in ranked if r.get("ost_lddt") is not None]
    ost_table_html = ""
    if ost_rows:
        rows_html = ""
        for r in ost_rows[:20]:
            rmsd = f"{r['ost_rmsd_ca']:.2f}" if r.get("ost_rmsd_ca") is not None else "-"
            qs = f"{r['ost_qs_score']:.3f}" if r.get("ost_qs_score") is not None else "-"
            rows_html += (
                f"<tr>"
                f"<td>{html_mod.escape(str(r.get('mutation_code', '?')))}</td>"
                f"<td>{r.get('improvement_score', 0):.3f}</td>"
                f"<td>{r.get('ost_lddt', 0):.3f}</td>"
                f"<td>{rmsd}</td>"
                f"<td>{qs}</td>"
                f"</tr>"
            )
        ost_table_html = (
            f"<h2>OST Structural Metrics</h2>"
            f"<table><tr><th>Mutation</th><th>Score</th>"
            f"<th>lDDT</th><th>RMSD (CA)</th><th>QS-score</th></tr>"
            f"{rows_html}</table>"
        )

    # Narrative
    narrative_html = (
        f"<h2>Narrative Summary</h2><p>{html_mod.escape(interpretation[:3000])}</p>"
        if interpretation else ""
    )

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Mutagenesis Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:0 auto;padding:20px;color:#1e293b}}
h1{{color:#1e293b}}h2{{color:#334155;border-bottom:2px solid #e2e8f0;padding-bottom:8px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}
th{{background:#f1f5f9;padding:8px 12px;text-align:left;border:1px solid #e2e8f0}}
td{{padding:7px 12px;border:1px solid #e2e8f0}}
tr:nth-child(even){{background:#f8fafc}}
img{{max-width:100%;height:auto;margin:12px 0;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}}
.summary{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:16px;margin:16px 0}}
</style></head>
<body>
<h1>Mutagenesis Report</h1>
{summary_html}
{ranking_section}
{plddt_section}
{ost_table_html}
{narrative_html}
</body></html>"""


def _render_saturation_heatmap(mutations: list, position: int):
    """Render a bar chart heatmap for saturation mutagenesis at one position."""
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    values = []
    colors = []
    hover_texts = []

    wt_aa = mutations[0].get("original_aa", "?") if mutations else "?"

    for aa in aa_order:
        if aa == wt_aa:
            values.append(0)
            colors.append("#9ca3af")
            hover_texts.append(f"{aa} (WT)")
        else:
            mut = next(
                (m for m in mutations if m.get("mutant_aa") == aa and m.get("success")),
                None,
            )
            if mut:
                delta = mut.get("delta_mean_plddt", 0)
                values.append(delta)
                colors.append("#22c55e" if delta > 0 else "#ef4444")
                hover_texts.append(
                    f"<b>{mut.get('mutation_code', '?')}</b><br>"
                    f"Delta pLDDT: {delta:+.2f}"
                )
            else:
                values.append(None)
                colors.append("#6b7280")
                hover_texts.append("Failed")

    fig = go.Figure(data=go.Bar(
        x=aa_order, y=values, marker_color=colors,
        hovertext=hover_texts, hoverinfo="text",
    ))
    fig.update_layout(
        title=f"Saturation at {wt_aa}{position}",
        yaxis_title="Delta pLDDT",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


with tab_agent:
    _render_agent_pipeline_tab()


# ══════════════════════════════════════════════════════════════════
#  TAB 2 — Advanced / Manual (existing scanner)
# ══════════════════════════════════════════════════════════════════
with tab_manual:

    # Render all predictor / baseline / evaluation settings first
    baseline_predictors, selected_predictor = _render_manual_tab_settings()

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
                                        create_structure_viewer(
                                            Path(match.structure_path),
                                            height=380,
                                            show_toolbar=True,
                                            title=getattr(match, "uniprot_id", ""),
                                        ),
                                        height=400,
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
                    save_dir=_meeting_save_dir(),
                )

            # Single-position scan
            if st.session_state.selected_position:
                pos = st.session_state.selected_position
                st.markdown("#### Single-Position Saturation Scan")
                st.info(f"Selected: **{seq[pos-1]}{pos}**")

                if st.session_state.get("scan_error"):
                    st.error(f"Last saturation scan failed: {st.session_state.scan_error}")
                    retry_req = st.session_state.get("last_scan_request") or {}
                    can_retry = (
                        retry_req.get("sequence") == seq
                        and retry_req.get("position") == pos
                    )
                    if st.button(
                        f"↻ Retry Saturation Scan at {pos}",
                        type="secondary",
                        disabled=not can_retry,
                        key=f"retry_scan_{pos}",
                    ):
                        results, err = run_saturation_mutagenesis(seq, pos)
                        if err:
                            st.session_state.scan_error = err
                        else:
                            st.session_state.scan_error = None
                            st.session_state.scan_results = results
                            st.session_state.multi_scan_results = None
                        st.rerun()

                if st.button(f"🔬 Run Saturation Mutagenesis at {pos}", type="primary"):
                    st.session_state.last_scan_request = {
                        "sequence": seq,
                        "position": pos,
                        "timestamp": datetime.now().isoformat(),
                    }
                    results, err = run_saturation_mutagenesis(seq, pos)
                    if err:
                        st.session_state.scan_error = err
                        st.session_state.scan_results = None
                        st.session_state.multi_scan_results = None
                        st.rerun()

                    st.session_state.scan_error = None
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
                            payload = results.to_dict()
                            payload["_metric_semantics"] = _score_semantics(getattr(results, "predictor", ""))
                            json.dump(payload, f, indent=2)

                        import shutil
                        if results.base_structure_path and results.base_structure_path.exists():
                            shutil.copy(results.base_structure_path, job_dir / "base_wt.pdb")

                        with open(job_dir / "prediction_summary.json", "w") as f:
                            json.dump({"job_id": job_id, "type": "scan", "status": "complete"}, f)

                        _activate_job_dir(job_dir)
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

                st.caption(
                    "Pipeline: run single-position scans → build multi-mutation combos "
                    "→ re-score each combined variant against the baseline."
                )
                if len(selected_positions_sorted) > max_positions:
                    st.warning(f"Selected {len(selected_positions_sorted)} positions. Reduce to ≤ {max_positions} to run.")
                if st.session_state.get("multi_scan_error"):
                    st.error(f"Last multi-mutation run failed: {st.session_state.multi_scan_error}")
                    can_retry_multi = bool(st.session_state.get("last_multi_scan_request"))
                    if st.button(
                        "↻ Retry Multi-Mutation Pipeline",
                        type="secondary",
                        disabled=not can_retry_multi,
                        key="retry_multi_scan",
                    ):
                        req = st.session_state.get("last_multi_scan_request") or {}
                        results, err = run_multi_mutation_pipeline(
                            req.get("sequence", seq),
                            req.get("positions", selected_positions_sorted),
                            top_k=req.get("top_k", top_k),
                            max_variants=req.get("max_variants", max_variants),
                            only_beneficial=req.get("only_beneficial", only_beneficial),
                            max_positions=req.get("max_positions", max_positions),
                        )
                        if err:
                            st.session_state.multi_scan_error = err
                        else:
                            st.session_state.multi_scan_error = None
                            st.session_state.multi_scan_results = results
                            st.session_state.scan_results = None
                        st.rerun()
                if st.button("🧬 Run Multi-Mutation Pipeline", type="primary", disabled=len(selected_positions_sorted) > max_positions):
                    st.session_state.last_multi_scan_request = {
                        "sequence": seq,
                        "positions": selected_positions_sorted,
                        "top_k": top_k,
                        "max_variants": max_variants,
                        "only_beneficial": only_beneficial,
                        "max_positions": max_positions,
                        "timestamp": datetime.now().isoformat(),
                    }
                    results, err = run_multi_mutation_pipeline(
                        seq,
                        selected_positions_sorted,
                        top_k=top_k,
                        max_variants=max_variants,
                        only_beneficial=only_beneficial,
                        max_positions=max_positions,
                    )
                    if err:
                        st.session_state.multi_scan_error = err
                        st.session_state.multi_scan_results = None
                        st.session_state.scan_results = None
                        st.rerun()

                    st.session_state.multi_scan_error = None
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
                            payload = results.to_dict()
                            payload["_metric_semantics"] = _score_semantics(getattr(results, "predictor", ""))
                            json.dump(payload, f, indent=2)

                        import shutil
                        if results.base_structure_path and results.base_structure_path.exists():
                            shutil.copy(results.base_structure_path, job_dir / "base_wt.pdb")

                        with open(job_dir / "prediction_summary.json", "w") as f:
                            json.dump({"job_id": job_id, "type": "multi_scan", "status": "complete"}, f)

                        _activate_job_dir(job_dir)
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

        # Auto-insight on multi-mutation results
        multi_insight_data: Dict[str, Any] = {
            "Positions combined": ", ".join(str(p) for p in res.positions),
            "Sequence length": len(res.sequence),
            "Predictor": pred_label or "unknown",
            "Total variants": len(getattr(res, "variants", [])),
        }
        successful = [v for v in getattr(res, "variants", []) if getattr(v, "success", True)]
        if successful:
            best_v = max(successful, key=lambda v: getattr(v, "delta_mean_plddt", 0))
            multi_insight_data["Best variant"] = best_v.mutation_code
            multi_insight_data["Best delta"] = f"{best_v.delta_mean_plddt:+.2f}"
        # Save to shared context
        _all_variants = getattr(res, "variants", [])
        _successful_variants = [v for v in _all_variants if getattr(v, "success", True)]
        set_page_results("MutationScanner", {
            "num_variants": len(_successful_variants),
            "positions": list(res.positions) if hasattr(res, "positions") else [],
            "best_delta_plddt": max(
                (getattr(v, "delta_mean_plddt", 0) for v in _successful_variants), default=0
            ),
            "scan_type": "multi_position",
        })

        # ML stats panel for mutation results
        if _successful_variants:
            _mut_records = [{
                "delta_pLDDT": getattr(v, "delta_mean_plddt", 0),
                "delta_local_pLDDT": getattr(v, "delta_local_plddt", 0),
                "improvement_score": getattr(v, "improvement_score", 0) or 0,
            } for v in _successful_variants[:100]]
            render_ml_stats_panel(
                _mut_records,
                numeric_keys=["delta_pLDDT", "delta_local_pLDDT", "improvement_score"],
                target_key="delta_pLDDT",
                page_name="Mutation Scanner",
                key_prefix="mut_ml_stats",
            )

        render_contextual_insight(
            "Mutation",
            multi_insight_data,
            key_prefix="mut_multi_ctx",
        )

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
                    if st.button(
                        "⚡ Send to Predict Page",
                        type="primary",
                        use_container_width=True,
                        key="mut_multi_send_predict",
                    ):
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
                    with st.container(border=True):
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            _delta = v.delta_mean_plddt
                            _color = "#22c55e" if _delta > 0 else "#ef4444"
                            st.markdown(
                                f'<div style="font-size:15px;font-weight:700;color:{_color};">'
                                f'{v.mutation_code}</div>'
                                f'<div style="font-size:13px;color:{_color};">Δ {_delta:+.2f}</div>',
                                unsafe_allow_html=True,
                            )
                        with col2:
                            _rmsd = getattr(v, "rmsd_to_base", None)
                            _local = getattr(v, "local_plddt_mean", None)
                            details = []
                            if _local is not None:
                                details.append(f"Local pLDDT {_local:.1f}")
                            if _rmsd is not None:
                                details.append(f"RMSD {_rmsd:.2f} Å")
                            st.caption("  ·  ".join(details) if details else "")
                            # Inline structure viewer if structure exists
                            _vpath = getattr(v, "structure_path", None)
                            if _vpath and Path(_vpath).exists():
                                with st.expander(f"🔬 View {v.mutation_code} structure"):
                                    import streamlit.components.v1 as _cmp
                                    from protein_design_hub.web.visualizations import create_structure_viewer as _csv
                                    _cmp.html(
                                        _csv(Path(_vpath), height=260, show_toolbar=True, title=v.mutation_code),
                                        height=280,
                                    )
                        with col3:
                            if st.button("Send to Predict", key=f"send_{v.mutation_code}", use_container_width=True, type="secondary"):
                                _ms = list(res.sequence)
                                for _p, _aa in zip(res.positions, v.mutant_aas):
                                    _ms[_p - 1] = _aa
                                st.session_state['incoming_prediction_job'] = {
                                    'sequence': "".join(_ms),
                                    'name': f"variant_{v.mutation_code}",
                                    'source': 'mutation_scanner',
                                }
                                st.switch_page("pages/1_predict.py")
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
                        components.html(create_structure_viewer(base_path, height=320, show_toolbar=True, title="Wild Type"), height=340)
                    elif st.session_state.base_structure:
                        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
                            f.write(st.session_state.base_structure)
                            p1 = f.name
                        components.html(create_structure_viewer(Path(p1), height=320, show_toolbar=True, title="Wild Type"), height=340)
                    else:
                        st.info("Base structure not available.")

                with c2:
                    st.markdown(f"**Variant {variant.mutation_code}**")
                    try:
                        vpath = Path(variant.structure_path)
                        if vpath.exists():
                            components.html(create_structure_viewer(vpath, height=320, show_toolbar=True, title=variant.mutation_code), height=340)
                        else:
                            st.info("Variant structure not found.")
                    except Exception:
                        st.info("Variant structure not found.")
            else:
                st.info("No variant selected.")

        # Agent advice + all-expert investigation of multi-mutation results
        if variants:
            top_variants_text = _format_top_variants(variants, is_immunebuilder, top_k=6)
            context_lines = [
                f"Sequence length: {len(res.sequence)}",
                f"Positions combined: {', '.join(str(p) for p in res.positions)}",
                "Top variants:\n" + top_variants_text if top_variants_text else "",
            ]
            context = "\n".join([line for line in context_lines if line])

            render_agent_advice_panel(
                page_context=context,
                default_question=(
                    "Which of these multi-mutation variants is most promising? "
                    "Are there trade-offs between stability and function?"
                ),
                expert="Protein Engineer",
                key_prefix="mut_multi_advice",
            )
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
                save_dir=_meeting_save_dir(),
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

        # Auto-insight on scan results
        scan_insight_data: Dict[str, Any] = {
            "Position": f"{res.original_aa}{res.position}",
            "Sequence length": len(res.sequence),
            "Predictor": pred_label or "unknown",
        }
        if res.ranked_mutations:
            best = res.ranked_mutations[0]
            scan_insight_data["Best mutation"] = best.mutation_code
            scan_insight_data["Best delta"] = f"{best.delta_mean_plddt:+.2f}"
            beneficial_count = sum(1 for m in res.mutations if getattr(m, "is_beneficial", False))
            scan_insight_data["Beneficial mutations"] = f"{beneficial_count}/{len(res.mutations)}"
        # Save single-position scan to shared context
        set_page_results("MutationScanner", {
            "position": res.position,
            "original_aa": res.original_aa,
            "num_mutations": len(getattr(res, "mutations", [])),
            "best_mutation": res.ranked_mutations[0].mutation_code if res.ranked_mutations else "",
            "best_delta_plddt": res.ranked_mutations[0].delta_mean_plddt if res.ranked_mutations else 0,
            "scan_type": "single_position",
        })

        # ML stats panel for single-position scan
        if getattr(res, "mutations", []):
            _scan_records = [{
                "delta_pLDDT": getattr(m, "delta_mean_plddt", 0),
                "delta_local_pLDDT": getattr(m, "delta_local_plddt", 0),
                "mutant_pLDDT": getattr(m, "mutant_mean_plddt", 0) or 0,
                "score": getattr(m, "improvement_score", 0) or 0,
            } for m in res.mutations[:20]]
            render_ml_stats_panel(
                _scan_records,
                numeric_keys=["delta_pLDDT", "delta_local_pLDDT", "mutant_pLDDT", "score"],
                target_key="delta_pLDDT",
                page_name="Single-Position Scan",
                key_prefix="scan_ml_stats",
            )

        render_contextual_insight(
            "Mutation",
            scan_insight_data,
            key_prefix="mut_scan_ctx",
        )

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
                
                    if st.button(
                        "⚡ Send to Predict Page",
                        type="primary",
                        use_container_width=True,
                        key="mut_single_send_predict",
                    ):
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

            # List other top variants with inline structure viewers
            st.markdown("#### Other Top Candidates")
            for mut in res.ranked_mutations[1:4]:
                with st.container(border=True):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        _mcolor = "#22c55e" if mut.is_beneficial else "#ef4444"
                        st.markdown(
                            f'<div style="font-weight:700;color:{_mcolor};">{mut.mutation_code}</div>'
                            f'<div style="font-size:13px;color:{_mcolor};">Δ {mut.delta_mean_plddt:+.2f}</div>',
                            unsafe_allow_html=True,
                        )
                    with col2:
                        _rmsd = getattr(mut, "rmsd_to_base", None)
                        _clash = getattr(mut, "clash_score", None)
                        _sasa = getattr(mut, "sasa_total", None)
                        _parts = []
                        if _rmsd: _parts.append(f"RMSD {_rmsd:.2f} Å")
                        if _clash is not None: _parts.append(f"Clash {_clash}")
                        if _sasa: _parts.append(f"SASA {_sasa:.0f} Ų")
                        st.caption("  ·  ".join(_parts))
                        _mpath = getattr(mut, "structure_path", None)
                        if _mpath:
                            try:
                                _mpath = Path(_mpath) if not isinstance(_mpath, Path) else _mpath
                                if _mpath.exists():
                                    with st.expander(f"🔬 View {mut.mutation_code}"):
                                        import streamlit.components.v1 as _cmp2
                                        from protein_design_hub.web.visualizations import create_structure_viewer as _csv2
                                        _cmp2.html(
                                            _csv2(_mpath, height=240, show_toolbar=True, title=mut.mutation_code),
                                            height=260,
                                        )
                            except Exception:
                                pass


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
                    components.html(
                        create_structure_viewer(Path(p1), height=320, show_toolbar=True, title="Wild Type"),
                        height=340,
                    )

                with c2:
                    st.markdown(f"**Mutant {mut.mutation_code}**")
                    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
                        f.write(mut.structure_path.read_text())
                        p2 = f.name
                    components.html(
                        create_structure_viewer(
                            Path(p2), height=320, show_toolbar=True, title=mut.mutation_code
                        ),
                        height=340,
                    )
                    # Inline agent interpretation of the comparison
                    _delta = getattr(mut, "delta_mean_plddt", None)
                    _rmsd = getattr(mut, "rmsd_to_base", None)
                    if _delta is not None:
                        _color = "#22c55e" if _delta > 0 else "#ef4444"
                        st.markdown(
                            f'<div style="font-size:13px;margin-top:6px;">'
                            f'<span style="color:{_color};font-weight:700;">ΔpLDDT {_delta:+.2f}</span>'
                            + (f' · RMSD {_rmsd:.2f} Å' if _rmsd else '')
                            + '</div>',
                            unsafe_allow_html=True,
                        )
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
                save_dir=_meeting_save_dir(),
            )
