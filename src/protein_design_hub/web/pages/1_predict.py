"""Prediction page for Streamlit app - Professional UI Design."""

import streamlit as st
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import tempfile
import json

from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    section_header,
    metric_card,
    metric_card_with_context,
    info_box,
    empty_state,
    workflow_breadcrumb,
    cross_page_actions,
)
from protein_design_hub.web.agent_helpers import (
    render_agent_advice_panel,
    render_contextual_insight,
    agent_sidebar_status,
)
from protein_design_hub.web.visualizations import (
    create_structure_viewer,
    create_plddt_plot,
    create_pae_heatmap
)
from protein_design_hub.io.afdb import AFDBClient, AFDBMatch, normalize_sequence
from protein_design_hub.analysis.protein_utils import (
    parse_multichain_sequence,
    calculate_sequence_properties,
    predict_secondary_structure_propensity,
    predict_aggregation_propensity,
    predict_solubility,
    detect_domains,
    validate_sequence
)

# Page config
st.set_page_config(
    page_title="Predict - Protein Design Hub",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Page-specific CSS for enhanced UI
# =============================================================================
PREDICT_CSS = """
<style>
/* Predictor Selection Cards */
.predictor-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 1rem 0;
}

.predictor-card {
    background: var(--pdhub-bg-card, rgba(20,20,30,0.8));
    border: 2px solid var(--pdhub-border, rgba(100,100,100,0.3));
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    position: relative;
}

.predictor-card:hover {
    border-color: var(--pdhub-primary, #6366f1);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
}

.predictor-card.selected {
    border-color: var(--pdhub-primary, #6366f1);
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
    box-shadow: 0 0 0 1px var(--pdhub-primary, #6366f1), 0 8px 24px rgba(99, 102, 241, 0.2);
}

.predictor-card.selected::after {
    content: "✓";
    position: absolute;
    top: 8px;
    right: 10px;
    background: var(--pdhub-primary, #6366f1);
    color: var(--pdhub-text-heading);
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: bold;
}

.predictor-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.predictor-name {
    font-weight: 600;
    font-size: 1rem;
    color: var(--pdhub-text, #f1f5f9);
    margin-bottom: 0.25rem;
}

.predictor-desc {
    font-size: 0.75rem;
    color: var(--pdhub-text-muted, #6b7280);
    line-height: 1.3;
}

/* Sequence Input Area */
.sequence-input-container {
    background: var(--pdhub-bg-card, rgba(20,20,30,0.8));
    border: 1px solid var(--pdhub-border, rgba(100,100,100,0.3));
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.sequence-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--pdhub-border, rgba(100,100,100,0.2));
}

.sequence-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--pdhub-text, #f1f5f9);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Analysis Panel */
.analysis-panel {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
}

.analysis-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--pdhub-primary-light, #818cf8);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

.analysis-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(100,100,100,0.15);
}

.analysis-stat:last-child {
    border-bottom: none;
}

.analysis-label {
    font-size: 0.85rem;
    color: var(--pdhub-text-secondary, #a1a9b8);
}

.analysis-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--pdhub-text, #f1f5f9);
}

/* Run Button Section */
.run-section {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

.run-section.disabled {
    background: var(--pdhub-bg-card, rgba(20,20,30,0.5));
    border-color: var(--pdhub-border, rgba(100,100,100,0.3));
}

/* Settings Panel */
.settings-panel {
    background: var(--pdhub-bg-card, rgba(20,20,30,0.6));
    border: 1px solid var(--pdhub-border, rgba(100,100,100,0.3));
    border-radius: 12px;
    padding: 1rem;
    margin-top: 0.5rem;
}

.settings-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--pdhub-text-secondary, #a1a9b8);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--pdhub-border, rgba(100,100,100,0.2));
}

/* Results Dashboard */
.results-header {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

.results-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--pdhub-text, #f1f5f9);
    margin-bottom: 0.5rem;
}

.results-subtitle {
    font-size: 0.9rem;
    color: var(--pdhub-text-secondary, #a1a9b8);
}

/* Chain Table */
.chain-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.85rem;
    margin-top: 0.75rem;
}

.chain-table th {
    text-align: left;
    padding: 0.5rem 0.75rem;
    color: var(--pdhub-text-muted, #6b7280);
    font-weight: 500;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    border-bottom: 1px solid var(--pdhub-border, rgba(100,100,100,0.3));
}

.chain-table td {
    padding: 0.5rem 0.75rem;
    color: var(--pdhub-text, #f1f5f9);
    border-bottom: 1px solid var(--pdhub-border, rgba(100,100,100,0.15));
}

.chain-table tr:last-child td {
    border-bottom: none;
}

/* Complexity Badge */
.complexity-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.complexity-low {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.complexity-medium {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.complexity-high {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
</style>
"""

# =============================================================================
# Example Proteins
# =============================================================================
EXAMPLES = {
    "": ("Select an example...", ""),
    "ubiquitin": ("Ubiquitin (76 aa)", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"),
    "t1024": ("T1024 - CASP14 (52 aa)", "MAAHKGAEHVVKASLDAGVKTVAGGGALVVKAKALGKDATMHLVAATLKKGYM"),
    "insulin": ("Insulin - Multi-chain (51 aa)", "GIVEQCCTSICSLYQLENYCN:FVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
    "gfp": ("GFP (238 aa)", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"),
}

# Predictor definitions with metadata
PREDICTORS = {
    "ColabFold": {
        "id": "colabfold",
        "icon": "🔬",
        "color": "#3b82f6",
        "desc": "AlphaFold2 with MSA",
        "speed": "Medium",
        "accuracy": "Excellent",
    },
    "ESMFold": {
        "id": "esmfold",
        "icon": "🧬",
        "color": "#22c55e",
        "desc": "Single-sequence, fast",
        "speed": "Fast",
        "accuracy": "Good",
    },
    "Chai-1": {
        "id": "chai1",
        "icon": "🧪",
        "color": "#a855f7",
        "desc": "Multi-modal diffusion",
        "speed": "Slow",
        "accuracy": "Excellent",
    },
    "Boltz-2": {
        "id": "boltz2",
        "icon": "⚡",
        "color": "#f59e0b",
        "desc": "Fast diffusion model",
        "speed": "Medium",
        "accuracy": "Very Good",
    },
    "ESM3": {
        "id": "esm3",
        "icon": "🌊",
        "color": "#06b6d4",
        "desc": "Next-gen language model",
        "speed": "Fast",
        "accuracy": "Good",
    },
}

# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_data(show_spinner=False)
def analyze_sequence(sequence: str) -> Dict[str, Any]:
    """Analyze input sequence with comprehensive properties. Cached by sequence."""
    if not sequence or sequence.strip() == "":
        return {}

    analysis = {}
    validation = validate_sequence(sequence)
    analysis["valid"] = validation["valid"]
    analysis["errors"] = validation.get("errors", [])
    analysis["warnings"] = validation.get("warnings", [])

    chains = parse_multichain_sequence(sequence)
    analysis["chains"] = chains
    analysis["num_chains"] = len(chains)
    analysis["total_length"] = sum(len(c["sequence"]) for c in chains)
    analysis["lengths"] = [len(c["sequence"]) for c in chains]

    chain_properties = []
    for chain in chains:
        props = calculate_sequence_properties(chain["sequence"])
        agg = predict_aggregation_propensity(chain["sequence"])
        sol = predict_solubility(chain["sequence"])

        chain_properties.append({
            "chain_id": chain["chain_id"],
            "length": len(chain["sequence"]),
            "mw_kda": props.get("molecular_weight_kda", 0),
            "net_charge": props.get("net_charge", 0),
            "pI": props.get("isoelectric_point", 7.0),
            "gravy": props.get("gravy", 0),
            "aggregation": agg,
            "solubility": sol,
        })
    analysis["chain_properties"] = chain_properties

    combined_seq = "".join(c["sequence"] for c in chains)
    analysis["overall_properties"] = calculate_sequence_properties(combined_seq)

    # Complexity estimation
    if analysis["total_length"] < 150:
        analysis["complexity"] = "Low"
    elif analysis["total_length"] < 500:
        analysis["complexity"] = "Medium"
    else:
        analysis["complexity"] = "High"

    return analysis


def render_predictor_cards(selected: List[str]) -> str:
    """Render predictor selection cards as HTML."""
    cards = []
    for name, info in PREDICTORS.items():
        is_selected = name in selected
        selected_class = "selected" if is_selected else ""
        cards.append(f"""
        <div class="predictor-card {selected_class}" style="--card-color: {info['color']};">
            <div class="predictor-icon">{info['icon']}</div>
            <div class="predictor-name">{name}</div>
            <div class="predictor-desc">{info['desc']}</div>
        </div>
        """)
    return f'<div class="predictor-grid">{"".join(cards)}</div>'


def render_chain_table(chains: List[Dict[str, Any]]) -> str:
    """Render chain breakdown table."""
    if not chains or len(chains) <= 1:
        return ""

    rows = []
    for c in chains:
        charge = c.get("net_charge", 0)
        charge_str = f"+{charge:.0f}" if charge > 0 else f"{charge:.0f}"
        rows.append(f"""
        <tr>
            <td><strong style="color: #60a5fa;">Chain {c['chain_id']}</strong></td>
            <td>{c['length']} aa</td>
            <td>{c['mw_kda']:.1f} kDa</td>
            <td>{charge_str}</td>
        </tr>
        """)

    return f"""
    <table class="chain-table">
        <thead><tr><th>Chain</th><th>Length</th><th>MW</th><th>Charge</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    """


def get_afdb_match_cached(sequence: str, email: str) -> Tuple[Optional[AFDBMatch], Optional[str]]:
    """Get AFDB match with caching."""
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

    match, error = client.find_match(sequence, min_identity=90.0, min_coverage=90.0, email=email)
    cache[cache_key] = match.to_dict() if match else {"error": error}
    return match, error


# =============================================================================
# Main Application
# =============================================================================

def main():
    inject_base_css()
    st.markdown(PREDICT_CSS, unsafe_allow_html=True)

    # Sidebar
    sidebar_nav(current="Predict")
    sidebar_system_status()
    agent_sidebar_status()

    # Page Header
    page_header(
        "Structure Prediction",
        "Transform protein sequences into accurate 3D structures using AI",
        "🔮"
    )

    # Workflow breadcrumb
    workflow_breadcrumb(
        ["Sequence Input", "Predict", "Evaluate", "Refine / Design", "Export"],
        current=1,
    )

    # Quick start guide
    with st.expander("📖 Quick start: How to predict a structure", expanded=False):
        st.markdown("""
**Step-by-step:**
1. **Select a predictor** below (ESMFold is fastest, ColabFold is most accurate)
2. **Enter your sequence** in the input area (paste raw AA sequence or upload FASTA)
3. **Click "Run Prediction"** and wait for results
4. **Review** pLDDT scores and 3D structure viewer

**Example sequence** — Human ubiquitin (76 residues):
```
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
```

**Which predictor to choose?**
- **ESMFold** — fastest (seconds), no MSA needed, good for well-known folds
- **ColabFold** — most accurate for novel proteins, uses MSA, ~2-10 min
- **Chai-1** / **Boltz-2** — best for complexes and ligand-bound proteins, GPU required
        """)

    # Initialize session state
    if "selected_predictors" not in st.session_state:
        st.session_state.selected_predictors = ["ColabFold"]
    if "job_running" not in st.session_state:
        st.session_state.job_running = False
    if "job_complete" not in st.session_state:
        st.session_state.job_complete = False

    # Handle incoming jobs from other pages
    incoming = st.session_state.get('incoming_prediction_job')

    # =========================================================================
    # SECTION 1: Predictor Selection
    # =========================================================================
    section_header("Select Prediction Engine", "Choose one or more AI models", "🎯")

    # Predictor multiselect with visual feedback
    col_select, col_info = st.columns([3, 1])

    with col_select:
        selected_predictors = st.multiselect(
            "Prediction Tools",
            options=list(PREDICTORS.keys()),
            default=st.session_state.selected_predictors,
            help="Select AI models for structure prediction. Multiple selections enable comparison.",
            label_visibility="collapsed"
        )
        st.session_state.selected_predictors = selected_predictors

        # Visual cards showing selection
        if selected_predictors:
            cards_html = []
            for name in selected_predictors:
                info = PREDICTORS[name]
                cards_html.append(f"""
                <div style="display: inline-flex; align-items: center; gap: 8px;
                            background: rgba({int(info['color'][1:3], 16)}, {int(info['color'][3:5], 16)}, {int(info['color'][5:7], 16)}, 0.15);
                            border: 1px solid {info['color']}40; border-radius: 8px; padding: 8px 14px; margin: 4px;">
                    <span style="font-size: 1.1rem;">{info['icon']}</span>
                    <span style="font-weight: 600; color: {info['color']};">{name}</span>
                </div>
                """)
            st.markdown(f"<div style='margin-top: 0.75rem;'>{''.join(cards_html)}</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div class="analysis-panel" style="text-align: center;">
            <div style="font-size: 2rem; font-weight: 700; color: #60a5fa;">{len(selected_predictors)}</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">Models Selected</div>
        </div>
        """, unsafe_allow_html=True)

    # Advanced settings expander
    if selected_predictors:
        with st.expander("⚙️ Model Settings", expanded=False):
            settings_cols = st.columns(min(len(selected_predictors), 3))
            settings_ui = {}

            for idx, pred_name in enumerate(selected_predictors):
                info = PREDICTORS[pred_name]
                with settings_cols[idx % len(settings_cols)]:
                    st.markdown(f"""
                    <div style="background: {info['color']}15; border: 1px solid {info['color']}40;
                                border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
                        <span style="font-weight: 600; color: {info['color']};">{info['icon']} {pred_name}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if pred_name == "ColabFold":
                        settings_ui["colabfold"] = {
                            "num_models": st.number_input("Models", 1, 5, 5, key="cf_m"),
                            "num_recycles": st.number_input("Recycles", 1, 24, 3, key="cf_r"),
                            "use_amber": st.checkbox("AMBER Relax", False, key="cf_a"),
                        }
                    elif pred_name == "ESMFold":
                        settings_ui["esmfold"] = {
                            "num_recycles": st.number_input("Recycles", 1, 8, 4, key="esm_r"),
                        }
                    elif pred_name == "Chai-1":
                        settings_ui["chai1"] = {
                            "num_trunk_recycles": st.number_input("Trunk Recycles", 1, 10, 3, key="chai_r"),
                            "num_diffusion_timesteps": st.number_input("Diffusion Steps", 50, 500, 200, key="chai_d"),
                        }
                    elif pred_name == "Boltz-2":
                        settings_ui["boltz2"] = {
                            "sampling_steps": st.number_input("Sampling Steps", 50, 500, 200, key="boltz_s"),
                        }
                    elif pred_name == "ESM3":
                        settings_ui["esm3"] = {
                            "temperature": st.slider("Temperature", 0.1, 1.5, 0.7, key="esm3_t"),
                        }
    else:
        settings_ui = {}

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2: Sequence Input
    # =========================================================================
    section_header("Input Sequence", "Enter or upload your protein sequence", "📝")

    col_input, col_analysis = st.columns([2, 1])

    with col_input:
        # Quick actions row
        col_example, col_upload = st.columns([2, 1])

        with col_example:
            example_options = {v[0]: k for k, v in EXAMPLES.items()}
            selected_example_label = st.selectbox(
                "Load Example",
                options=list(example_options.keys()),
                index=0,
                label_visibility="collapsed",
                help="Quick-load an example protein"
            )
            selected_example_key = example_options.get(selected_example_label, "")

        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload FASTA",
                type=["fasta", "fa", "txt"],
                label_visibility="collapsed",
                help="Upload a FASTA file"
            )

        # Determine default sequence
        default_seq = ""
        if selected_example_key and selected_example_key in EXAMPLES:
            name, seq = EXAMPLES[selected_example_key]
            default_seq = f">{name.split('(')[0].strip()}\n{seq}"

        if uploaded_file:
            default_seq = uploaded_file.read().decode("utf-8")
            st.success(f"Loaded: {uploaded_file.name}")

        if incoming:
            default_seq = f">{incoming['name']}\n{incoming['sequence']}"
            st.success(f"Loaded from: {incoming.get('description', 'external source')}")

        # Main sequence input
        sequence_input = st.text_area(
            "Protein Sequence",
            value=default_seq,
            height=180,
            placeholder=">protein_name\nMKFLILLFNILCLFPVLAADNHGVGPQGAS...\n\nFor complexes, separate chains with colon (:)\n>complex\nCHAIN_A:CHAIN_B",
            label_visibility="collapsed"
        )

        # Multi-chain indicator
        if sequence_input and ":" in sequence_input:
            st.info("🔗 **Multi-chain complex detected** — Chains separated by `:`")

    with col_analysis:
        st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
        st.markdown('<div class="analysis-title">📊 Sequence Analysis</div>', unsafe_allow_html=True)

        if sequence_input:
            analysis = analyze_sequence(sequence_input)

            if analysis.get("errors"):
                for err in analysis["errors"]:
                    st.error(f"❌ {err}")
            else:
                # Core metrics
                props = analysis.get("overall_properties", {})
                total_len = analysis.get("total_length", 0)
                num_chains = analysis.get("num_chains", 1)
                complexity = analysis.get("complexity", "Unknown")

                st.markdown(f"""
                <div class="analysis-stat">
                    <span class="analysis-label">Length</span>
                    <span class="analysis-value">{total_len} aa</span>
                </div>
                <div class="analysis-stat">
                    <span class="analysis-label">Chains</span>
                    <span class="analysis-value">{num_chains}</span>
                </div>
                <div class="analysis-stat">
                    <span class="analysis-label">MW</span>
                    <span class="analysis-value">{props.get('molecular_weight_kda', 0):.1f} kDa</span>
                </div>
                <div class="analysis-stat">
                    <span class="analysis-label">pI</span>
                    <span class="analysis-value">{props.get('isoelectric_point', 7.0):.1f}</span>
                </div>
                """, unsafe_allow_html=True)

                # Complexity badge
                complexity_class = f"complexity-{complexity.lower()}"
                st.markdown(f"""
                <div style="margin-top: 1rem; text-align: center;">
                    <span class="complexity-badge {complexity_class}">
                        {'⚡' if complexity == 'Low' else '⏱️' if complexity == 'Medium' else '🐢'} {complexity} Complexity
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Chain breakdown for multi-chain
                if num_chains > 1:
                    chain_table = render_chain_table(analysis.get("chain_properties", []))
                    if chain_table:
                        st.markdown(chain_table, unsafe_allow_html=True)

                # Warnings
                for warn in analysis.get("warnings", []):
                    st.warning(f"⚠️ {warn}")

                # Aggregation check
                for cp in analysis.get("chain_properties", []):
                    if cp.get("aggregation", {}).get("aggregation_prone"):
                        st.warning(f"⚠️ Chain {cp['chain_id']}: Aggregation hotspots")
                        break
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem 1rem; color: #6b7280;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧬</div>
                <div>Enter a sequence to see analysis</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Optional: Template & MSA (collapsed by default)
    with st.expander("📎 Advanced: Template & MSA Input", expanded=False):
        col_template, col_msa = st.columns(2)

        with col_template:
            st.markdown("**🏗️ Template Structure**")
            st.caption("Guide prediction with a known structure")
            template_file = st.file_uploader("Upload PDB/CIF", type=["pdb", "cif"], key="template")
            if template_file:
                st.session_state["template_file"] = template_file
                st.success(f"Template: {template_file.name}")

        with col_msa:
            st.markdown("**📊 Pre-computed MSA**")
            st.caption("Provide evolutionary information")
            msa_file = st.file_uploader("Upload A3M/FASTA", type=["a3m", "fasta", "sto"], key="msa")
            if msa_file:
                st.session_state["msa_file"] = msa_file
                st.success(f"MSA: {msa_file.name}")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3: Run Prediction
    # =========================================================================
    is_ready = bool(sequence_input) and bool(selected_predictors)

    section_class = "" if is_ready else "disabled"
    st.markdown(f'<div class="run-section {section_class}">', unsafe_allow_html=True)

    col_job, col_actions = st.columns([2, 1])

    with col_job:
        st.markdown("##### 🚀 Run Prediction")
        job_name = st.text_input(
            "Job Name",
            value=incoming['name'] if incoming else "",
            placeholder="my_experiment_001",
            label_visibility="collapsed"
        )

    with col_actions:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if is_ready:
            st.success(f"Ready: {len(selected_predictors)} model(s) selected")
        elif not sequence_input:
            st.warning("Enter a sequence above")
        else:
            st.warning("Select at least one model")

        col_run, col_demo = st.columns(2)
        with col_run:
            run_btn = st.button(
                "🚀 Predict",
                type="primary",
                use_container_width=True,
                disabled=not is_ready
            )
        with col_demo:
            demo_btn = st.button(
                "📂 Demo",
                type="secondary",
                use_container_width=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # Execution Logic
    # =========================================================================
    if run_btn:
        st.session_state["job_running"] = True
        st.session_state["job_complete"] = False
        st.session_state["results"] = None

    if demo_btn:
        demo_pdb = Path("outputs/scan_Ubiquitin_5_20260129_025431/base_wt.pdb")
        if demo_pdb.exists():
            from protein_design_hub.core.types import PredictionResult, StructureScore, PredictorType
            mock_res = PredictionResult(
                job_id="demo_job",
                predictor=PredictorType.COLABFOLD,
                success=True,
                structure_paths=[demo_pdb],
                scores=[StructureScore(plddt=94.5, ptm=0.88)],
                runtime_seconds=1.0
            )
            st.session_state["results"] = {"ColabFold": mock_res}
            st.session_state["job_complete"] = True
            st.rerun()
        else:
            st.error("Demo structure not found. Run a prediction first.")

    if st.session_state.get("job_running"):
        from protein_design_hub.pipeline.workflow import PredictionWorkflow
        from protein_design_hub.core.config import get_settings

        global_settings = get_settings()
        predictor_ids = [PREDICTORS[p]["id"] for p in selected_predictors]

        # Apply settings
        if "colabfold" in settings_ui:
            global_settings.predictors.colabfold.num_models = settings_ui["colabfold"]["num_models"]
            global_settings.predictors.colabfold.num_recycles = settings_ui["colabfold"]["num_recycles"]

        workflow = PredictionWorkflow(global_settings)

        with st.status("🔮 Running Structure Prediction...", expanded=True) as status:
            st.write("📝 Preparing input...")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
                tmp.write(sequence_input)
                input_path = Path(tmp.name)

            job_id = job_name or f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.write(f"🚀 Job: **{job_id}**")

            try:
                results = workflow.run_prediction_only(
                    input_path=input_path,
                    predictors=predictor_ids,
                    job_id=job_id
                )
                st.session_state["results"] = results
                st.session_state["job_complete"] = True
                status.update(label="✅ Prediction Complete!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                status.update(label="❌ Failed", state="error")

            st.session_state["job_running"] = False
            st.rerun()

    # =========================================================================
    # SECTION 4: Results Dashboard
    # =========================================================================
    if st.session_state.get("job_complete") and st.session_state.get("results"):
        res_dict = st.session_state["results"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="results-header">
            <div class="results-title">✨ Prediction Results</div>
            <div class="results-subtitle">Structure prediction completed successfully</div>
        </div>
        """, unsafe_allow_html=True)

        tab_struct, tab_metrics, tab_download = st.tabs(["🧬 3D Structure", "📊 Quality Metrics", "💾 Downloads"])

        with tab_struct:
            # Find best structure
            best_pdb, best_plddt, best_name = None, -1, ""
            for name, res in res_dict.items():
                if res.success and res.structure_paths:
                    for i, path in enumerate(res.structure_paths):
                        score = res.scores[i].plddt if i < len(res.scores) else 0
                        if score > best_plddt:
                            best_plddt, best_pdb, best_name = score, path, name

            col_viewer, col_info = st.columns([3, 1])

            with col_viewer:
                if best_pdb and best_pdb.exists():
                    st.components.v1.html(create_structure_viewer(best_pdb), height=500)
                else:
                    empty_state("No Structure", "Structure file not found", "🔬")

            with col_info:
                if best_name:
                    st.markdown(f"#### 🏆 Best Result")
                    st.markdown(f"**Model:** {best_name}")
                    metric_card(f"{best_plddt:.1f}", "pLDDT Score", "success", "⭐")

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("📊 Evaluate Structure", use_container_width=True, type="primary"):
                        from protein_design_hub.web.ui import set_selected_model
                        set_selected_model(best_pdb)
                        st.switch_page("pages/2_evaluate.py")

                    if st.button("🧬 Run Mutations", use_container_width=True):
                        st.switch_page("pages/10_mutation_scanner.py")

        with tab_metrics:
            all_scores = []
            for name, res in res_dict.items():
                if res.success and res.scores:
                    for i, score in enumerate(res.scores):
                        all_scores.append({
                            "Predictor": name,
                            "Model": i + 1,
                            "pLDDT": score.plddt or 0,
                            "pTM": score.ptm or 0,
                            "ipTM": score.iptm or 0,
                        })

            if all_scores:
                # Summary cards with scientific context
                best = max(all_scores, key=lambda x: x["pLDDT"])
                avg = sum(s["pLDDT"] for s in all_scores) / len(all_scores)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    metric_card_with_context(best['pLDDT'], "Best pLDDT", metric_name="plddt", icon="🏆")
                with c2:
                    metric_card_with_context(avg, "Average pLDDT", metric_name="plddt", icon="📈")
                with c3:
                    ptm_val = best.get('pTM', 0)
                    if ptm_val:
                        metric_card_with_context(ptm_val, "pTM Score", metric_name="ptm", icon="🎯")
                    else:
                        metric_card("N/A", "pTM Score", "default", "🎯")
                with c4:
                    metric_card(str(len(all_scores)), "Models Generated", "info", "🔢")

                st.markdown("---")

                # Score table
                import pandas as pd
                df = pd.DataFrame(all_scores)
                st.dataframe(
                    df.style.format({"pLDDT": "{:.1f}", "pTM": "{:.3f}", "ipTM": "{:.3f}"})
                    .background_gradient(subset=["pLDDT"], cmap="RdYlGn"),
                    use_container_width=True,
                )

                # AI Scientific Analysis
                st.markdown("---")
                render_contextual_insight(
                    "Prediction",
                    {"Best pLDDT": f"{best['pLDDT']:.1f}",
                     "Average pLDDT": f"{avg:.1f}",
                     "pTM": f"{best.get('pTM', 0):.3f}",
                     "Models": len(all_scores)},
                    key_prefix="predict_ctx",
                )

                # Agent advice on prediction results
                scores_ctx = "\n".join(
                    f"- {s['Predictor']} model {s['Model']}: pLDDT={s['pLDDT']:.1f}, pTM={s.get('pTM', 0):.2f}"
                    for s in all_scores
                )
                render_agent_advice_panel(
                    page_context=f"Prediction results:\n{scores_ctx}",
                    default_question="Based on these prediction scores, what is the quality of this structure and what should I do next?",
                    expert="Structural Biologist",
                    key_prefix="predict_agent",
                )

                # Cross-page actions
                st.markdown("---")
                section_header("Next Steps", "Continue your analysis workflow", "➡️")
                cross_page_actions([
                    {"label": "Evaluate Structure", "page": "pages/2_evaluate.py", "icon": "📊"},
                    {"label": "Compare Predictors", "page": "pages/3_compare.py", "icon": "⚖️"},
                    {"label": "Scan Mutations", "page": "pages/10_mutation_scanner.py", "icon": "🧬"},
                    {"label": "MPNN Design", "page": "pages/8_mpnn.py", "icon": "🎯"},
                ])

        with tab_download:
            output_files = []
            for name, res in res_dict.items():
                if res.success and res.structure_paths:
                    for path in res.structure_paths:
                        if path.exists():
                            output_files.append({"name": path.name, "path": path, "predictor": name, "size": path.stat().st_size})

            if output_files:
                st.markdown(f"**{len(output_files)} structure file(s) ready for download**")

                for f in output_files:
                    col_name, col_btn = st.columns([3, 1])
                    with col_name:
                        st.markdown(f"📄 `{f['name']}` — {f['predictor']} ({f['size']/1024:.1f} KB)")
                    with col_btn:
                        with open(f["path"], "rb") as file:
                            st.download_button("Download", file.read(), f["name"], "chemical/x-pdb", key=f"dl_{f['name']}", use_container_width=True)

                st.markdown("---")
                import zipfile, io
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as zf:
                    for f in output_files:
                        zf.write(f["path"], f["name"])
                st.download_button("📦 Download All as ZIP", buf.getvalue(), "prediction_results.zip", "application/zip", key="dl_all", use_container_width=True)
            else:
                empty_state("No Files", "No structure files found", "📭")


if __name__ == "__main__":
    main()
