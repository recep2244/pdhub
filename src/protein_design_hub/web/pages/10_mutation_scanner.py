"""Interactive Mutation Scanner with ESMFold-based saturation mutagenesis.

This page provides:
1. Sequence input with auto ESMFold prediction
2. Interactive residue selection for mutation scanning
3. Automatic saturation mutagenesis (all 19 AA mutations)
4. Comprehensive metric calculation (pLDDT, RMSD, Clash Score, SASA)
5. Mutation ranking and recommendations
6. Side-by-side structure comparison
"""

import streamlit as st
from pathlib import Path
import json
import tempfile
import time
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.graph_objects as go

from protein_design_hub.analysis.mutation_scanner import MutationScanner, SaturationMutagenesisResult, MutationResult
from datetime import datetime
from types import SimpleNamespace

st.set_page_config(
    page_title="Mutation Scanner - Protein Design Hub",
    page_icon="🔬",
    layout="wide"
)

# Enhanced CSS for mutation scanner interface
st.markdown("""
<style>
/* Main container */
.main .block-container {
    padding: 1rem 2rem;
    max-width: 100%;
}

/* Gradient headers */
.gradient-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 0.5rem;
}

/* Residue grid */
.residue-button {
    font-weight: bold;
    transition: all 0.2s ease;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border: 1px solid #dee2e6;
    margin-bottom: 10px;
}

.metric-card-success { border-left: 5px solid #28a745; }
.metric-card-danger { border-left: 5px solid #dc3545; }

.metric-value { font-size: 1.5rem; font-weight: bold; color: #333; }
.metric-delta { font-size: 1rem; font-weight: bold; }
.delta-positive { color: #28a745; }
.delta-negative { color: #dc3545; }
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
            # We don't need all fields, just what the UI uses
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
    defaults = {
        'sequence': '', 'sequence_name': 'my_protein',
        'base_structure': None, 'base_plddt': None, 'base_plddt_per_residue': None,
        'selected_position': None, 'scan_results': None,
        'comparison_mutation': None,
        'scanner': MutationScanner(use_api=True)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

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

def render_heatmap(results):
    mutations = results.mutations
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    original_aa = results.original_aa
    
    values = []
    hover_texts = []
    colors = []
    
    for aa in aa_order:
        if aa == original_aa:
            values.append(0)
            colors.append('#808080')
            hover_texts.append(f"{aa} (WT)")
        else:
            mut = next((m for m in mutations if m.mutant_aa == aa), None)
            if mut and mut.success:
                delta = mut.delta_mean_plddt
                values.append(delta)
                colors.append('#28a745' if delta > 0 else '#dc3545')
                hover_texts.append(
                    f"<b>{mut.mutation_code}</b><br>"
                    f"ΔpLDDT: {delta:+.2f}<br>"
                    f"RMSD: {mut.rmsd_to_base:.2f} Å" if mut.rmsd_to_base else ""
                )
            else:
                values.append(None)
                colors.append('#cccccc')
                hover_texts.append("Failed")
                
    fig = go.Figure(data=go.Bar(
        x=aa_order, y=values, marker_color=colors,
        hovertext=hover_texts, hoverinfo='text'
    ))
    fig.update_layout(
        title=f"Mutation Stability (ΔpLDDT) at {results.original_aa}{results.position}",
        yaxis_title="ΔpLDDT", height=350
    )
    st.plotly_chart(fig, use_container_width=True)

# Main UI
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 class="gradient-header">🔬 Mutation Scanner</h1>
    <p style="color: #666;">Comprehensive saturation mutagenesis with full biophysical metrics</p>
</div>
""", unsafe_allow_html=True)

# 1. Input
st.markdown("## 1️⃣ Input Sequence")
seq_col, info_col = st.columns([3, 1])
with seq_col:
    new_seq = st.text_area("Sequence", value=st.session_state.sequence, height=100)
    if new_seq != st.session_state.sequence:
        cleaned = "".join(c for c in new_seq.upper().strip() if c in AMINO_ACIDS)
        st.session_state.sequence = cleaned
        st.session_state.base_structure = None
        st.session_state.scan_results = None
        st.rerun()

with info_col:
    if st.button("Load Ubiquitin"):
        st.session_state.sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        st.session_state.sequence_name = "Ubiquitin"
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
        st.success(f"Base Structure Ready (Mean pLDDT: {st.session_state.base_plddt:.1f})")
        
        # 3. Selection
        st.markdown("### Select Residue to Scan")
        seq = st.session_state.sequence
        pldit = st.session_state.base_plddt_per_residue
        
        cols = st.columns(20)
        for i, aa in enumerate(seq):
            if i >= 100: break # simple view limit for demo
            color = "primary" if (i+1) == st.session_state.selected_position else "secondary"
            if cols[i % 20].button(aa, key=f"r_{i}", type=color, help=f"Pos {i+1}"):
                st.session_state.selected_position = i + 1
                st.session_state.scan_results = None
                st.rerun()
        
        if st.session_state.selected_position:
            pos = st.session_state.selected_position
            st.info(f"Selected: **{seq[pos-1]}{pos}**")
            
            if st.button(f"🔬 Run Saturation Mutagenesis at {pos}", type="primary"):
                results = run_saturation_mutagenesis(seq, pos)
                st.session_state.scan_results = results
                
                # Create Job in outputs
                try:
                    from protein_design_hub.core.config import get_settings
                    settings = get_settings()
                    job_id = f"scan_{st.session_state.sequence_name}_{pos}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    job_dir = Path(settings.output.base_dir) / job_id
                    job_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Store scan results
                    with open(job_dir / "scan_results.json", "w") as f:
                        json.dump(results.to_dict(), f, indent=2)
                    
                    # Copy structures to job dir
                    # Results contains base_structure_path and mut.structure_path
                    import shutil
                    if results.base_structure_path and results.base_structure_path.exists():
                        shutil.copy(results.base_structure_path, job_dir / "base_wt.pdb")
                    
                    # Save a dummy prediction_summary for Job browser detection
                    with open(job_dir / "prediction_summary.json", "w") as f:
                        json.dump({"job_id": job_id, "type": "scan", "status": "complete"}, f)
                        
                    st.info(f"💾 Job saved as {job_id}")
                except Exception as e:
                    st.warning(f"Could not save job to outputs: {e}")
                
                st.rerun()

# 4. Results
if st.session_state.scan_results:
    res = st.session_state.scan_results
    st.markdown("---")
    st.markdown("## 📊 Scan Results")
    
    tab1, tab2, tab3 = st.tabs(["🏆 Recommendations", "📈 Detailed Metrics", "🔬 3D Comparison"])
    
    with tab1:
        render_heatmap(res)
        
        st.markdown("### Top Variants")
        
        best_mut = res.ranked_mutations[0] if res.ranked_mutations else None
        
        if best_mut:
            # Highlight Best Variant
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%); padding: 15px; border-radius: 10px; border: 1px solid #28a745; margin-bottom: 20px;">
                <h3 style="margin:0; color: #155724;">🏆 Best Candidate: {best_mut.mutation_code}</h3>
                <p style="margin:5px 0 0 0; color: #155724;">
                    predicted to improve pLDDT by <b>+{best_mut.delta_mean_plddt:.2f}</b>
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
        data = []
        for m in res.mutations:
            if m.success:
                data.append({
                    "Mutation": m.mutation_code,
                    "Mean pLDDT": f"{m.mean_plddt:.1f}",
                    "Δ pLDDT": f"{m.delta_mean_plddt:+.2f}",
                    "RMSD (Å)": f"{m.rmsd_to_base:.2f}" if m.rmsd_to_base else "N/A",
                    "Clash Score": f"{m.clash_score:.2f}" if m.clash_score else "N/A",
                    "SASA (Å²)": f"{m.sasa_total:.0f}" if m.sasa_total else "N/A",
                    "TM-score": f"{m.tm_score_to_base:.2f}" if m.tm_score_to_base else "N/A"
                })
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
