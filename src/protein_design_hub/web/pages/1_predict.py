"""Prediction page for Streamlit app."""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import time
import tempfile

from protein_design_hub.web.ui import (
    inject_base_css, 
    sidebar_nav, 
    sidebar_system_status, 
    page_header,
    metric_card,
    render_badge
)
from protein_design_hub.web.visualizations import (
    create_structure_viewer,
    create_plddt_plot,
    create_pae_heatmap
)

# Set page config
st.set_page_config(
    page_title="Predict - Protein Design Hub",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Example proteins
EXAMPLES = {
    "T1024 (CASP14 Target)": ">T1024\nMAAHKGAEHVVKASLDAGVKTVAGGALVVKAKALGKDATMHLVAATLKKGYM",
    "Ubiquitin": ">Ubiquitin\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "Insulin": ">Insulin\nGIVEQCCTSICSLYQLENYCN:FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
}

def analyze_sequence(sequence: str) -> Dict[str, Any]:
    """Analyze input sequence for basic properties."""
    if not sequence or sequence.strip() == "":
        return {}
        
    analysis = {}
    
    # Simple FASTA parsing
    lines = sequence.strip().split("\n")
    seq_parts = []
    
    for line in lines:
        if line.startswith(">"):
            continue
        seq_parts.append(line.strip())
        
    full_seq = "".join(seq_parts)
    
    # Check for chains (multimer)
    chains = full_seq.split(":")
    
    analysis["num_chains"] = len(chains)
    analysis["total_length"] = sum(len(c) for c in chains)
    analysis["lengths"] = [len(c) for c in chains]
    
    # Check characters
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_chars = set(full_seq.upper()) - valid_chars - set(":")
    
    analysis["valid"] = len(invalid_chars) == 0
    analysis["invalid_chars"] = list(invalid_chars)
    
    # Estimate complexity
    analysis["complexity"] = "Low" if analysis["total_length"] < 200 else "Medium" if analysis["total_length"] < 800 else "High"
    
    return analysis

def render_predictor_settings(selected_predictors: List[str]) -> Dict[str, Any]:
    """Render sidebar settings for selected predictors."""
    settings = {}
    
    # Initialize defaults
    settings["colabfold"] = {"num_models": 5, "num_recycles": 3, "use_amber": False}
    
    if "ColabFold" in selected_predictors:
        st.sidebar.markdown("---")
        with st.sidebar.expander("🔬 ColabFold Settings", expanded=True):
            settings["colabfold"]["num_models"] = st.number_input( "Models", 1, 5, 5)
            settings["colabfold"]["num_recycles"] = st.number_input("Recycles", 1, 24, 3)
            settings["colabfold"]["use_amber"] = st.checkbox("AMBER Relax", False)
            
    if "Chai-1" in selected_predictors:
        st.sidebar.markdown("---")
        with st.sidebar.expander("🧪 Chai-1 Settings", expanded=True):
            settings["chai1"] = {}
            settings["chai1"]["num_trunk_recycles"] = st.number_input("Trunk Recycles", 1, 10, 3)
            settings["chai1"]["num_diffusion_timesteps"] = st.number_input("Diffusion Steps", 50, 500, 200)

    if "Boltz-2" in selected_predictors:
        st.sidebar.markdown("---")
        with st.sidebar.expander("⚡ Boltz-2 Settings", expanded=True):
            settings["boltz2"] = {}
            settings["boltz2"]["sampling_steps"] = st.number_input("Sampling Steps", 50, 500, 200)
            
    return settings

def main():
    inject_base_css()
    
    # --- Header ---
    page_header(
        "Structure Prediction",
        "Generate 3D structures from protein sequences using state-of-the-art AI models.",
        "🔮"
    )
    
    sidebar_nav(current="Predict")
    sidebar_system_status()
    
    # --- Sidebar ---
    st.sidebar.header("Configuration")
    
    # Predictor Selection
    predictor_map = {
        "ColabFold": "colabfold",
        "Chai-1": "chai1",
        "Boltz-2": "boltz2",
        "ESMFold": "esmfold",
    }
    
    selected_labels = st.sidebar.multiselect(
        "Select Predictors",
        options=list(predictor_map.keys()),
        default=["ColabFold"],
        help="Select one or more tools to run in parallel comparison"
    )
    
    selected_predictors = [predictor_map[l] for l in selected_labels]
    settings_ui = render_predictor_settings(selected_labels)
    
    # --- Main Content ---
    
    # 1. Input Section
    st.markdown("### 1. Input Sequence")
    
    col_input, col_info = st.columns([2, 1])
    
    with col_input:
        # Quick load examples
        selected_example = st.selectbox(
            "Load Example (Optional)", 
            ["Select...", *EXAMPLES.keys()],
            index=0
        )
        
        default_seq = ""
        if selected_example != "Select...":
            default_seq = EXAMPLES[selected_example]
            
        # Handle incoming jobs from Mutation Scanner (overrides example)
        incoming = st.session_state.get('incoming_prediction_job')
        if incoming:
            default_seq = f">{incoming['name']}\n{incoming['sequence']}"
            if 'description' in incoming:
                st.success(f"✅ Loaded: {incoming['description']}")
            # We don't clear it immediately so re-runs work, 
            # but ideally we might clear upon successful edit or submission.
        
        sequence_input = st.text_area(
            "Protein Sequence (FASTA format)",
            value=default_seq,
            height=180,
            placeholder=">Target_1\nMKFLILLFNILCLFPVLAADNHGVGPQGAS..."
        )
        
    with col_info:
        # Real-time Sequence Analysis
        if sequence_input:
            analysis = analyze_sequence(sequence_input)
            if analysis:
                st.markdown("##### Sequence Analysis")
                
                # Check metrics
                if analysis.get("valid", True):
                    metric_card(analysis["total_length"], "Residues", "info", "📏")
                    
                    if analysis["num_chains"] > 1:
                        metric_card(analysis["num_chains"], "Chains (Multimer)", "warning", "🔗")
                    else:
                        render_badge("Monomer", "primary")
                        
                    st.progress(min(analysis["total_length"] / 1000, 1.0), text="Length Complexity")
                else:
                    st.error(f"Invalid Characters Found: {', '.join(analysis['invalid_chars'])}")
        else:
            st.info("👈 Enter a sequence to analyze details")

    # 2. Controls
    st.markdown("### 2. Job Control")
    
    col_run, col_status = st.columns([1, 2])
    
    with col_run:
        default_name = incoming['name'] if incoming else ""
        job_name = st.text_input("Job Name (Optional)", value=default_name, placeholder="my_experiment_1")
        run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True, disabled=not sequence_input)
        
    if run_btn:
        st.session_state["job_running"] = True
        st.session_state["job_complete"] = False
        st.session_state["results"] = None
        
    # --- Execution Logic ---
    if st.session_state.get("job_running"):
        from protein_design_hub.pipeline.workflow import PredictionWorkflow
        from protein_design_hub.core.config import get_settings
        
        global_settings = get_settings()
        # Update settings from UI
        if "colabfold" in selected_predictors:
            global_settings.predictors.colabfold.num_models = settings_ui["colabfold"]["num_models"]
            global_settings.predictors.colabfold.num_recycles = settings_ui["colabfold"]["num_recycles"]
            
        workflow = PredictionWorkflow(global_settings)
        
        with st.status("Running Predictions...", expanded=True) as status:
            # 1. Prepare Input
            st.write("📝 Preparing sequences...")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
                tmp.write(sequence_input)
                input_path = Path(tmp.name)
            
            # 2. Run Job
            job_id = job_name or f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.write(f"🚀 Running job: {job_id}")
            
            try:
                # We use run_prediction_only for this page unless user wants evaluation here too
                results = workflow.run_prediction_only(
                    input_path=input_path,
                    predictors=selected_predictors,
                    job_id=job_id
                )
                
                st.session_state["results"] = results
                st.session_state["job_complete"] = True
                status.update(label="Prediction Complete!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                status.update(label="Failed", state="error")
            
            st.session_state["job_running"] = False
            st.rerun()

    # 3. Results Dashboard
    if st.session_state.get("job_complete") and st.session_state.get("results"):
        res_dict = st.session_state["results"]
        st.markdown("---")
        st.header("✨ Results Dashboard")
        
        tab_struct, tab_metrics, tab_files = st.tabs(["🧬 3D Structure", "📊 Quality Metrics", "💾 Downloads"])
        
        with tab_struct:
            # Find the best structure across all successful predictors
            best_pdb = None
            best_plddt = -1
            best_pred_name = ""
            
            for name, res in res_dict.items():
                if res.success and res.structure_paths:
                    for i, path in enumerate(res.structure_paths):
                        score = res.scores[i].plddt if i < len(res.scores) else 0
                        if score > best_plddt:
                            best_plddt = score
                            best_pdb = path
                            best_pred_name = name
            
            col_viewer, col_details = st.columns([3, 1])
            with col_viewer:
                if best_pdb and best_pdb.exists():
                    st.components.v1.html(create_structure_viewer(best_pdb), height=500)
                else:
                    st.info("No structure files found to display.")
                
            with col_details:
                if best_pred_name:
                    st.markdown(f"#### Best: {best_pred_name.upper()}")
                    metric_card(f"{best_plddt:.1f}", "pLDDT (Global)", "success", "🌟")
                    
                    if st.button("📊 Go to Evaluation", use_container_width=True):
                        from protein_design_hub.web.ui import set_selected_model
                        set_selected_model(best_pdb)
                        st.switch_page("pages/2_evaluate.py")
                else:
                    st.warning("No successful predictions.")

        
        with tab_metrics:
            st.warning("Metrics plots will be generated from PredictionResult data.")
            
        with tab_files:
            st.success("Job data saved to `outputs/folder`")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.button("📥 Download all PDBs (ZIP)", use_container_width=True)
            with col_d2:
                st.button("📥 Download Report (PDF)", use_container_width=True)

if __name__ == "__main__":
    main()
