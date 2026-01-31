"""Prediction page for Streamlit app."""

import streamlit as st
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import time
import tempfile
import json

from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    metric_card,
    render_badge,
    card_start,
    card_end,
    empty_state
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
    """Analyze input sequence with comprehensive properties."""
    if not sequence or sequence.strip() == "":
        return {}

    analysis = {}

    # Validate sequence first
    validation = validate_sequence(sequence)
    analysis["valid"] = validation["valid"]
    analysis["errors"] = validation.get("errors", [])
    analysis["warnings"] = validation.get("warnings", [])

    # Parse multi-chain
    chains = parse_multichain_sequence(sequence)
    analysis["chains"] = chains
    analysis["num_chains"] = len(chains)
    analysis["total_length"] = sum(len(c["sequence"]) for c in chains)
    analysis["lengths"] = [len(c["sequence"]) for c in chains]

    # Per-chain analysis
    chain_properties = []
    for chain in chains:
        props = calculate_sequence_properties(chain["sequence"])
        ss_prop = predict_secondary_structure_propensity(chain["sequence"])
        agg = predict_aggregation_propensity(chain["sequence"])
        sol = predict_solubility(chain["sequence"])
        domains = detect_domains(chain["sequence"])

        chain_properties.append({
            "chain_id": chain["chain_id"],
            "length": len(chain["sequence"]),
            "mw_kda": props.get("molecular_weight_kda", 0),
            "net_charge": props.get("net_charge", 0),
            "gravy": props.get("gravy", 0),
            "pI": props.get("isoelectric_point", 7.0),
            "ss_propensity": ss_prop,
            "aggregation": agg,
            "solubility": sol,
            "domains": domains,
        })
    analysis["chain_properties"] = chain_properties

    # Overall properties (combined)
    combined_seq = "".join(c["sequence"] for c in chains)
    analysis["overall_properties"] = calculate_sequence_properties(combined_seq)

    # Estimate complexity
    if analysis["total_length"] < 200:
        analysis["complexity"] = "Low"
        analysis["complexity_color"] = "success"
    elif analysis["total_length"] < 800:
        analysis["complexity"] = "Medium"
        analysis["complexity_color"] = "warning"
    else:
        analysis["complexity"] = "High"
        analysis["complexity_color"] = "error"

    return analysis


def render_chain_table(chains: List[Dict[str, Any]]) -> str:
    """Render a visual chain table."""
    if not chains or len(chains) <= 1:
        return ""

    rows = []
    for chain in chains:
        chain_id = chain.get("chain_id", "?")
        length = chain.get("length", 0)
        mw = chain.get("mw_kda", 0)
        charge = chain.get("net_charge", 0)
        charge_icon = "+" if charge > 0 else ("−" if charge < 0 else "")
        rows.append(f"""
            <tr>
                <td style="font-weight: 600; color: #60a5fa;">Chain {chain_id}</td>
                <td>{length} aa</td>
                <td>{mw:.1f} kDa</td>
                <td>{charge_icon}{abs(charge)}</td>
            </tr>
        """)

    return f"""
    <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem;">
        <thead>
            <tr style="border-bottom: 1px solid rgba(100,100,100,0.3); color: #94a3b8;">
                <th style="text-align: left; padding: 0.5rem;">Chain</th>
                <th style="text-align: left; padding: 0.5rem;">Length</th>
                <th style="text-align: left; padding: 0.5rem;">MW</th>
                <th style="text-align: left; padding: 0.5rem;">Charge</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """


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
        "ESM3": "esm3",
    }
    
    selected_labels = st.sidebar.multiselect(
        "Select Predictors",
        options=list(predictor_map.keys()),
        default=["ColabFold"],
        help="Select one or more tools to run in parallel comparison"
    )
    
    selected_predictors = [predictor_map[l] for l in selected_labels]
    settings_ui = render_predictor_settings(selected_labels)

    st.sidebar.markdown("---")
    with st.sidebar.expander("🔍 AFDB Match", expanded=False):
        afdb_enabled = st.checkbox(
            "Fetch related AFDB structure (>=90% identity & coverage)",
            value=st.session_state.get("afdb_enabled", False),
            help="Uses EBI BLAST against UniProt and fetches AFDB if a close match exists.",
        )
        afdb_email = st.text_input(
            "EBI email (recommended)",
            value=st.session_state.get("afdb_email", os.getenv("EBI_EMAIL", "")),
            help="EBI BLAST requests ask for an email address.",
        )
        st.caption("AFDB lookup can take ~1-2 minutes depending on sequence length.")

        st.session_state.afdb_enabled = afdb_enabled
        st.session_state.afdb_email = afdb_email.strip()
    
    # --- Main Content ---
    st.markdown("---")

    # Tabs for input modes
    input_tab, template_tab, msa_tab = st.tabs(["📝 Sequence Input", "🏗️ Template (Optional)", "📊 MSA (Optional)"])

    with input_tab:
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

            sequence_input = st.text_area(
                "Protein Sequence (FASTA format)",
                value=default_seq,
                height=180,
                placeholder=">Target_1\nMKFLILLFNILCLFPVLAADNHGVGPQGAS...\n\nFor multi-chain: use : separator\n>Complex\nCHAIN_A_SEQ:CHAIN_B_SEQ"
            )

            # Multi-chain helper
            if sequence_input and ":" in sequence_input:
                st.info("🔗 Multi-chain sequence detected. Chains are separated by ':'")

        with col_info:
            # Real-time Sequence Analysis
            if sequence_input:
                analysis = analyze_sequence(sequence_input)
                if analysis:
                    st.markdown("##### Real-time Analysis")

                    # Show errors if any
                    if analysis.get("errors"):
                        for err in analysis["errors"]:
                            st.error(f"❌ {err}")
                    elif analysis.get("warnings"):
                        for warn in analysis["warnings"]:
                            st.warning(f"⚠️ {warn}")

                    # Metrics
                    if analysis.get("valid", True) or not analysis.get("errors"):
                        m1, m2 = st.columns(2)
                        with m1:
                            metric_card(analysis["total_length"], "Residues", "info", "📏")
                        with m2:
                            if analysis["num_chains"] > 1:
                                metric_card(analysis["num_chains"], "Chains", "warning", "🔗")
                            else:
                                metric_card("Single", "Mode", "gradient", "👤")

                        # Complexity indicator
                        complexity_color = analysis.get("complexity_color", "info")
                        st.markdown(f"""
                        <div style="margin-top: 1rem; padding: 0.5rem; border-radius: 8px;
                                    background: rgba(100,100,100,0.1); text-align: center;">
                            <span style="font-size: 0.75rem; color: #94a3b8;">Complexity:</span>
                            <span style="font-weight: 600; color: {'#22c55e' if complexity_color == 'success' else '#f59e0b' if complexity_color == 'warning' else '#ef4444'};">
                                {analysis.get('complexity', 'Unknown')}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Chain breakdown for multi-chain
                        if analysis["num_chains"] > 1:
                            chain_table = render_chain_table(analysis.get("chain_properties", []))
                            if chain_table:
                                st.markdown(chain_table, unsafe_allow_html=True)

                        # Biophysical summary
                        props = analysis.get("overall_properties", {})
                        if props:
                            with st.expander("🔬 Biophysical Properties", expanded=False):
                                st.markdown(f"""
                                - **MW**: {props.get('molecular_weight_kda', 0):.1f} kDa
                                - **pI**: {props.get('isoelectric_point', 7.0):.1f}
                                - **Charge**: {props.get('net_charge', 0):+d}
                                - **GRAVY**: {props.get('gravy', 0):.2f}
                                - **Hydrophobic**: {props.get('hydrophobic_fraction', 0)*100:.1f}%
                                """)

                        # Aggregation warning
                        chain_props = analysis.get("chain_properties", [])
                        for cp in chain_props:
                            agg = cp.get("aggregation", {})
                            if agg.get("aggregation_prone"):
                                st.warning(f"⚠️ Chain {cp['chain_id']}: Potential aggregation hotspots detected")
                                break
            else:
                empty_state("Ready for Sequence", "Analysis will appear here", "🔬")

    with template_tab:
        st.markdown("### 🏗️ Template Structure")
        st.caption("Upload a reference structure to guide prediction (PDB format)")

        template_file = st.file_uploader(
            "Upload Template PDB",
            type=["pdb", "cif"],
            key="template_upload",
            help="Upload a known structure to use as a template for prediction"
        )

        if template_file:
            st.success(f"✅ Template loaded: {template_file.name}")
            # Save to session
            st.session_state["template_file"] = template_file
            st.session_state["template_name"] = template_file.name

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                template_chain = st.text_input("Template Chain ID", value="A", max_chars=1)
                st.session_state["template_chain"] = template_chain
            with col_t2:
                template_weight = st.slider("Template Weight", 0.0, 1.0, 0.5, 0.1)
                st.session_state["template_weight"] = template_weight
        else:
            st.info("No template uploaded. Prediction will use de novo folding.")
            st.session_state["template_file"] = None

    with msa_tab:
        st.markdown("### 📊 Multiple Sequence Alignment")
        st.caption("Upload pre-computed MSA to improve prediction accuracy (A3M format)")

        msa_file = st.file_uploader(
            "Upload MSA (A3M format)",
            type=["a3m", "fasta", "sto"],
            key="msa_upload",
            help="Upload a pre-computed MSA for better evolutionary information"
        )

        if msa_file:
            st.success(f"✅ MSA loaded: {msa_file.name}")
            st.session_state["msa_file"] = msa_file
            st.session_state["msa_name"] = msa_file.name

            # Parse and show stats
            try:
                msa_content = msa_file.read().decode("utf-8")
                msa_file.seek(0)  # Reset for later use
                num_seqs = msa_content.count(">")
                st.markdown(f"""
                <div style="padding: 0.75rem; background: rgba(34,197,94,0.1); border-radius: 8px; margin-top: 0.5rem;">
                    <span style="font-weight: 600;">📈 MSA Statistics</span><br>
                    <span style="font-size: 0.85rem; color: #94a3b8;">Sequences: {num_seqs}</span>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass

            use_custom_msa = st.checkbox("Use uploaded MSA instead of generating new", value=True)
            st.session_state["use_custom_msa"] = use_custom_msa
        else:
            st.info("No MSA uploaded. Prediction will generate MSA automatically (ColabFold) or skip (ESMFold).")
            st.session_state["msa_file"] = None

    # 2. Controls
    st.markdown("### 2. Job Control")
    
    col_run, col_status = st.columns([1, 2])
    
    with col_run:
        default_name = incoming['name'] if incoming else ""
        job_name = st.text_input("Job Name (Optional)", value=default_name, placeholder="my_experiment_1")
        
        c1, c2 = st.columns(2)
        with c1:
            run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True, disabled=not sequence_input)
        with c2:
            demo_btn = st.button("🧬 Load Demo Structure", type="secondary", use_container_width=True)
        
    if run_btn:
        st.session_state["job_running"] = True
        st.session_state["job_complete"] = False
        st.session_state["results"] = None
        
    if demo_btn:
        # Load the existing PDB file as a demo
        demo_pdb = Path("outputs/scan_Ubiquitin_5_20260129_025431/base_wt.pdb")
        if demo_pdb.exists():
            from protein_design_hub.core.types import PredictionResult, StructureScore, PredictorType
            # Mock results for visualization
            mock_res = PredictionResult(
                job_id="demo_job",
                predictor=PredictorType.COLABFOLD,
                success=True,
                structure_paths=[demo_pdb],
                scores=[StructureScore(plddt=94.5, ptm=0.88)],
                runtime_seconds=1.0
            )
            st.session_state["results"] = {"demo": mock_res}
            st.session_state["job_complete"] = True
            st.success("✅ Demo structure loaded successfully.")
        else:
            st.error("Demo structure file not found. Please run a job first to generate outputs.")
        
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

            if st.session_state.get("afdb_enabled") and sequence_input:
                st.markdown("---")
                st.markdown("### 🔗 Related AlphaFold DB Structure")
                clean_seq = normalize_sequence(sequence_input)
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
                            st.components.v1.html(
                                create_structure_viewer(Path(match.structure_path), height=450),
                                height=470,
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

        
        with tab_metrics:
            st.markdown("### 📊 Quality Metrics")

            # Collect all scores
            all_scores = []
            for name, res in res_dict.items():
                if res.success and res.scores:
                    for i, score in enumerate(res.scores):
                        all_scores.append({
                            "predictor": name,
                            "model": i + 1,
                            "pLDDT": score.plddt or 0,
                            "pTM": score.ptm or 0,
                            "ipTM": score.iptm or 0,
                            "confidence": score.confidence or score.plddt or 0,
                        })

            if all_scores:
                # Summary metrics grid
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                best_score = max(all_scores, key=lambda x: x["pLDDT"])
                avg_plddt = sum(s["pLDDT"] for s in all_scores) / len(all_scores)

                with col_m1:
                    metric_card(f"{best_score['pLDDT']:.1f}", "Best pLDDT", "success", "🏆")
                with col_m2:
                    metric_card(f"{avg_plddt:.1f}", "Avg pLDDT", "info", "📈")
                with col_m3:
                    if best_score.get("pTM"):
                        metric_card(f"{best_score['pTM']:.2f}", "pTM Score", "gradient", "🎯")
                    else:
                        metric_card("N/A", "pTM Score", "muted", "🎯")
                with col_m4:
                    metric_card(str(len(all_scores)), "Models", "info", "🔢")

                st.markdown("---")

                # Per-model breakdown
                st.markdown("##### Per-Model Scores")
                import pandas as pd
                df = pd.DataFrame(all_scores)
                st.dataframe(
                    df.style.format({
                        "pLDDT": "{:.1f}",
                        "pTM": "{:.3f}",
                        "ipTM": "{:.3f}",
                        "confidence": "{:.1f}"
                    }).background_gradient(subset=["pLDDT", "confidence"], cmap="RdYlGn"),
                    use_container_width=True
                )

                # pLDDT distribution plot
                st.markdown("##### pLDDT Distribution")
                if best_pdb and best_pdb.exists():
                    # Try to show per-residue pLDDT
                    try:
                        fig_plddt = create_plddt_plot(best_pdb)
                        if fig_plddt:
                            st.plotly_chart(fig_plddt, use_container_width=True)
                    except Exception as e:
                        st.info(f"Per-residue pLDDT plot not available: {e}")

                # Quality interpretation
                st.markdown("##### Quality Interpretation")
                if best_score["pLDDT"] >= 90:
                    st.success("🌟 **Excellent** - Very high confidence prediction. Structure is likely accurate at atomic level.")
                elif best_score["pLDDT"] >= 70:
                    st.info("✅ **Good** - High confidence. Backbone likely correct, some side-chain uncertainty.")
                elif best_score["pLDDT"] >= 50:
                    st.warning("⚠️ **Moderate** - Low confidence. Consider this as a rough model only.")
                else:
                    st.error("❌ **Poor** - Very low confidence. Structure may be largely incorrect.")

            else:
                st.info("No quality metrics available.")

        with tab_files:
            st.markdown("### 💾 Downloads")

            # List all output files
            output_files = []
            for name, res in res_dict.items():
                if res.success and res.structure_paths:
                    for path in res.structure_paths:
                        if path.exists():
                            output_files.append({
                                "name": path.name,
                                "path": path,
                                "predictor": name,
                                "size": path.stat().st_size
                            })

            if output_files:
                st.markdown(f"**{len(output_files)} structure files available**")

                for f in output_files:
                    col_fn, col_dl = st.columns([3, 1])
                    with col_fn:
                        size_kb = f["size"] / 1024
                        st.markdown(f"📄 `{f['name']}` ({f['predictor']}) - {size_kb:.1f} KB")
                    with col_dl:
                        with open(f["path"], "rb") as file:
                            st.download_button(
                                "Download",
                                data=file.read(),
                                file_name=f["name"],
                                mime="chemical/x-pdb",
                                key=f"dl_{f['name']}",
                                use_container_width=True
                            )

                st.markdown("---")

                # Bulk download
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if st.button("📦 Download all as ZIP", use_container_width=True):
                        import zipfile
                        import io
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                            for f in output_files:
                                zf.write(f["path"], f["name"])
                        zip_buffer.seek(0)
                        st.download_button(
                            "💾 Save ZIP",
                            data=zip_buffer.getvalue(),
                            file_name="prediction_results.zip",
                            mime="application/zip",
                            key="dl_zip"
                        )
                with col_d2:
                    st.button("📥 Download Report (PDF)", use_container_width=True, disabled=True)
                    st.caption("Report generation coming soon")
            else:
                st.info("No output files found.")

if __name__ == "__main__":
    main()
