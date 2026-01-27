"""Compare page for Streamlit app."""

import streamlit as st
from pathlib import Path
import tempfile
import json

st.set_page_config(page_title="Compare - Protein Design Hub", page_icon="⚖️", layout="wide")

from protein_design_hub.web.ui import inject_base_css, sidebar_nav, sidebar_system_status

inject_base_css()
sidebar_nav(current="Compare")
sidebar_system_status()

st.title("⚖️ Compare Predictions")
st.markdown("Run all predictors and compare results with visual analysis")

# Add custom CSS for better layout
st.markdown(
    """
<style>
.metric-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    margin: 5px;
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #1f77b4;
}
.metric-label {
    font-size: 12px;
    color: #666;
}
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("Settings")

# Predictor selection
predictors_to_run = st.sidebar.multiselect(
    "Predictors to run",
    options=["ColabFold", "Chai-1", "Boltz-2"],
    default=["ColabFold", "Chai-1", "Boltz-2"],
)

predictor_map = {
    "ColabFold": "colabfold",
    "Chai-1": "chai1",
    "Boltz-2": "boltz2",
}

# Settings
with st.sidebar.expander("Prediction Settings"):
    num_models = st.number_input("Models per predictor", value=5, min_value=1, max_value=10)
    num_recycles = st.number_input("Recycles", value=3, min_value=1, max_value=10)

with st.sidebar.expander("Evaluation Settings"):
    st.markdown("**Global Metrics**")
    eval_metrics = st.multiselect(
        "Global metrics",
        options=["lDDT", "BB-lDDT", "TM-score", "RMSD", "GDT-TS", "GDT-HA"],
        default=["lDDT", "TM-score"],
    )

    st.markdown("**Interface Metrics**")
    interface_metrics = st.multiselect(
        "Interface metrics",
        options=["QS-score", "DockQ", "ICS", "IPS", "iLDDT", "Patch Scores"],
        default=["QS-score", "DockQ"],
        help="Only computed for multimeric structures",
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Sequence")

    input_method = st.radio(
        "Input method",
        options=["Paste sequence", "Upload FASTA"],
        horizontal=True,
        key="compare_input",
    )

    if input_method == "Paste sequence":
        sequence_input = st.text_area(
            "Enter sequence (FASTA format)",
            height=150,
            placeholder=">protein\nMKFLILLFNILCLFPVLAADNHGVGPQGAS...",
            key="compare_seq",
        )
    else:
        uploaded = st.file_uploader("Upload FASTA", type=["fasta", "fa"], key="compare_fasta")
        sequence_input = uploaded.read().decode() if uploaded else None

with col2:
    st.subheader("Reference (Optional)")

    reference_file = st.file_uploader(
        "Upload reference structure",
        type=["pdb", "cif"],
        key="compare_ref",
    )

    if reference_file:
        st.success(f"Reference: {reference_file.name}")

    st.subheader("Output")
    try:
        from protein_design_hub.core.config import get_settings

        _settings = get_settings()
        default_out = str(_settings.output.base_dir)
    except Exception:
        default_out = "./outputs"

    output_dir = st.text_input("Output directory", value=default_out, key="compare_output")
    job_name = st.text_input("Job name (optional)", key="compare_job")

# Run comparison
st.markdown("---")

if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
    if not sequence_input:
        st.error("Please provide input sequence")
    elif not predictors_to_run:
        st.error("Please select at least one predictor")
    else:
        try:
            from protein_design_hub.pipeline.workflow import PredictionWorkflow
            from protein_design_hub.core.config import get_settings
            from datetime import datetime

            settings = get_settings()
            settings.predictors.colabfold.num_models = num_models
            settings.predictors.colabfold.num_recycles = num_recycles

            # Save input to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
                tmp.write(sequence_input)
                input_path = Path(tmp.name)

            # Save reference if provided
            reference_path = None
            if reference_file:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(reference_file.name).suffix, delete=False
                ) as tmp:
                    tmp.write(reference_file.read())
                    reference_path = Path(tmp.name)

            # Setup job
            job_id = job_name or f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            predictor_list = [predictor_map[p] for p in predictors_to_run]

            # Progress tracking
            progress_bar = st.progress(0)
            status_container = st.container()

            # Run workflow
            workflow = PredictionWorkflow(settings)

            with status_container:
                st.info("Running comparison pipeline...")

                result = workflow.run(
                    input_path=input_path,
                    output_dir=Path(output_dir),
                    reference_path=reference_path,
                    predictors=predictor_list,
                    job_id=job_id,
                )

            progress_bar.progress(1.0)

            # Display results
            st.success("**Comparison Complete!**")

            # Best predictor
            if result.best_predictor:
                st.markdown(f"### 🏆 Best Predictor: **{result.best_predictor.upper()}**")

            # Prediction results table
            st.markdown("### Prediction Results")

            pred_data = []
            for name, pred in result.prediction_results.items():
                pred_data.append(
                    {
                        "Predictor": name.upper(),
                        "Status": "✓ Success" if pred.success else "✗ Failed",
                        "Structures": len(pred.structure_paths),
                        "Runtime (s)": f"{pred.runtime_seconds:.1f}",
                        "Best pLDDT": f"{max((s.plddt for s in pred.scores if s.plddt), default=0):.1f}"
                        if pred.scores
                        else "N/A",
                    }
                )

            import pandas as pd

            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            # Evaluation results (if reference provided)
            if result.evaluation_results:
                st.markdown("### Evaluation Results")

                # Global metrics table
                st.markdown("#### Global Metrics")
                eval_data = []
                for name, eval_result in result.evaluation_results.items():
                    row = {
                        "Predictor": name.upper(),
                        "lDDT": f"{eval_result.lddt:.3f}" if eval_result.lddt else "N/A",
                        "TM-score": f"{eval_result.tm_score:.3f}"
                        if eval_result.tm_score
                        else "N/A",
                        "RMSD (Å)": f"{eval_result.rmsd:.2f}" if eval_result.rmsd else "N/A",
                    }
                    # Add additional metrics if available in metadata
                    if eval_result.metadata:
                        if "bb_lddt" in eval_result.metadata:
                            row["BB-lDDT"] = f"{eval_result.metadata['bb_lddt']:.3f}"
                        if "gdt_ts" in eval_result.metadata:
                            row["GDT-TS"] = f"{eval_result.metadata['gdt_ts']:.3f}"
                        if "gdt_ha" in eval_result.metadata:
                            row["GDT-HA"] = f"{eval_result.metadata['gdt_ha']:.3f}"
                    eval_data.append(row)

                eval_df = pd.DataFrame(eval_data)
                st.dataframe(eval_df, use_container_width=True, hide_index=True)

                # Interface metrics table (for multimers)
                has_interface_metrics = any(
                    er.qs_score is not None
                    or (er.metadata and ("dockq" in er.metadata or "ics" in er.metadata))
                    for er in result.evaluation_results.values()
                )

                if has_interface_metrics:
                    st.markdown("#### Interface Metrics")
                    interface_data = []
                    for name, eval_result in result.evaluation_results.items():
                        row = {"Predictor": name.upper()}

                        if eval_result.qs_score is not None:
                            row["QS-score"] = f"{eval_result.qs_score:.3f}"
                        else:
                            row["QS-score"] = "N/A"

                        if eval_result.metadata:
                            meta = eval_result.metadata
                            row["DockQ"] = (
                                f"{meta.get('dockq', 'N/A'):.3f}"
                                if isinstance(meta.get("dockq"), (int, float))
                                else "N/A"
                            )
                            row["ICS"] = (
                                f"{meta.get('ics', 'N/A'):.3f}"
                                if isinstance(meta.get("ics"), (int, float))
                                else "N/A"
                            )
                            row["IPS"] = (
                                f"{meta.get('ips', 'N/A'):.3f}"
                                if isinstance(meta.get("ips"), (int, float))
                                else "N/A"
                            )
                            row["iLDDT"] = (
                                f"{meta.get('ilddt', 'N/A'):.3f}"
                                if isinstance(meta.get("ilddt"), (int, float))
                                else "N/A"
                            )

                        interface_data.append(row)

                    interface_df = pd.DataFrame(interface_data)
                    st.dataframe(interface_df, use_container_width=True, hide_index=True)

                # Comparison chart
                st.markdown("### Metric Comparison")

                import plotly.graph_objects as go

                metrics_to_plot = []
                for eval_result in result.evaluation_results.values():
                    if eval_result.lddt:
                        metrics_to_plot.append("lDDT")
                        break

                if metrics_to_plot:
                    fig = go.Figure()

                    predictors = list(result.evaluation_results.keys())

                    for metric in ["lDDT", "TM-score"]:
                        values = []
                        for name in predictors:
                            er = result.evaluation_results[name]
                            if metric == "lDDT":
                                values.append(er.lddt if er.lddt else 0)
                            elif metric == "TM-score":
                                values.append(er.tm_score if er.tm_score else 0)

                        if any(v > 0 for v in values):
                            fig.add_trace(
                                go.Bar(
                                    name=metric,
                                    x=[p.upper() for p in predictors],
                                    y=values,
                                )
                            )

                    fig.update_layout(
                        barmode="group",
                        title="Predictor Comparison",
                        yaxis_title="Score",
                        yaxis_range=[0, 1],
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Ranking
            if result.ranking:
                st.markdown("### Ranking")
                for i, (name, score) in enumerate(result.ranking, 1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    st.markdown(f"{medal} **{name.upper()}** - Score: {score:.3f}")

            # Output info
            job_dir = Path(output_dir) / job_id
            st.info(f"Results saved to: `{job_dir}`")

            # Download report
            report_path = job_dir / "report" / "report.html"
            if report_path.exists():
                st.download_button(
                    "📥 Download HTML Report",
                    data=open(report_path).read(),
                    file_name="comparison_report.html",
                    mime="text/html",
                )

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback

            st.code(traceback.format_exc())

# Structure Viewer Section
st.markdown("---")
st.subheader("🔬 Structure Viewer")

# Check for existing results
existing_results = st.text_input(
    "Load existing results (path to job directory)", key="load_results"
)

if existing_results and Path(existing_results).exists():
    job_path = Path(existing_results)

    # Find structure files
    structure_files = list(job_path.glob("**/*.pdb")) + list(job_path.glob("**/*.cif"))

    if structure_files:
        selected_structure = st.selectbox(
            "Select structure to view",
            structure_files,
            format_func=lambda x: f"{x.parent.name}/{x.name}",
        )

        if selected_structure:
            # Try to use py3Dmol for 3D visualization
            try:
                import py3Dmol
                import stmol

                with open(selected_structure) as f:
                    structure_content = f.read()

                file_ext = selected_structure.suffix.lower()
                mol_format = "cif" if file_ext == ".cif" else "pdb"

                viewer = py3Dmol.view(width=700, height=500)
                viewer.addModel(structure_content, mol_format)
                viewer.setStyle({"cartoon": {"color": "spectrum"}})
                viewer.setBackgroundColor("white")
                viewer.zoomTo()

                stmol.showmol(viewer, height=500, width=700)

            except ImportError:
                st.warning(
                    "Install py3Dmol and stmol for 3D visualization: `pip install py3Dmol stmol`"
                )

                # Fallback: show structure info
                st.markdown("**Structure Info:**")
                st.code(f"File: {selected_structure.name}")

                # Try to parse basic info
                try:
                    from Bio.PDB import PDBParser, MMCIFParser

                    if selected_structure.suffix.lower() == ".pdb":
                        parser = PDBParser(QUIET=True)
                        structure = parser.get_structure("structure", str(selected_structure))
                    else:
                        parser = MMCIFParser(QUIET=True)
                        structure = parser.get_structure("structure", str(selected_structure))

                    num_chains = len(list(structure.get_chains()))
                    num_residues = len(list(structure.get_residues()))
                    num_atoms = len(list(structure.get_atoms()))

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Chains", num_chains)
                    with col_b:
                        st.metric("Residues", num_residues)
                    with col_c:
                        st.metric("Atoms", num_atoms)

                except Exception as e:
                    st.text(f"Could not parse structure: {e}")

            # Download button
            with open(selected_structure, "rb") as f:
                st.download_button(
                    f"📥 Download {selected_structure.name}",
                    data=f.read(),
                    file_name=selected_structure.name,
                    mime="chemical/x-pdb"
                    if selected_structure.suffix == ".pdb"
                    else "chemical/x-cif",
                )

    # Load summary if exists
    summary_file = job_path / "prediction_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

        st.markdown("### Results Summary")

        for pred_name, pred_info in summary.get("predictors", {}).items():
            with st.expander(f"**{pred_name.upper()}**"):
                col_1, col_2, col_3 = st.columns(3)
                with col_1:
                    status = "✓ Success" if pred_info.get("success") else "✗ Failed"
                    st.markdown(f"**Status:** {status}")
                with col_2:
                    st.markdown(f"**Structures:** {pred_info.get('num_structures', 0)}")
                with col_3:
                    runtime = pred_info.get("runtime_seconds", 0)
                    st.markdown(f"**Runtime:** {runtime:.1f}s")

                if pred_info.get("structure_paths"):
                    st.markdown("**Structure files:**")
                    for path in pred_info["structure_paths"]:
                        st.text(f"  - {Path(path).name}")

else:
    st.info("Enter a path to load existing results, or run a new comparison above.")

# Info section
with st.expander("ℹ️ About Comparison"):
    st.markdown(
        """
    ### How Comparison Works

    1. **Prediction Phase**: Each selected predictor runs on your input sequence
    2. **Evaluation Phase**: If a reference is provided, structures are evaluated
    3. **Ranking Phase**: Predictors are ranked by evaluation scores (or pLDDT if no reference)

    ### Ranking Criteria

    - **With reference**: Ranking by lDDT score
    - **Without reference**: Ranking by pLDDT (predicted confidence)

    ### Evaluation Metrics

    **Global Metrics:**
    | Metric | Description | Range |
    |--------|-------------|-------|
    | **lDDT** | Local Distance Difference Test | 0-1 (higher is better) |
    | **BB-lDDT** | Backbone-only lDDT | 0-1 (higher is better) |
    | **TM-score** | Template Modeling score | 0-1 (higher is better) |
    | **RMSD** | Root Mean Square Deviation | 0-∞ Å (lower is better) |
    | **GDT-TS/HA** | Global Distance Test | 0-1 (higher is better) |

    **Interface Metrics (for multimers):**
    | Metric | Description | Range |
    |--------|-------------|-------|
    | **QS-score** | Quaternary Structure score | 0-1 (higher is better) |
    | **DockQ** | Docking Quality (fnat, iRMSD, lRMSD) | 0-1 (higher is better) |
    | **ICS** | Interface Contact Similarity | 0-1 (higher is better) |
    | **IPS** | Interface Patch Similarity | 0-1 (higher is better) |
    | **iLDDT** | Inter-chain lDDT | 0-1 (higher is better) |
    | **Patch Scores** | CASP15 local interface quality | 0-1 (higher is better) |

    ### Tips

    - Provide a reference structure when available for more accurate comparison
    - Longer sequences take more time to predict
    - Each predictor runs sequentially to manage GPU memory
    - Use Chai-1 or Boltz-2 for protein-ligand complexes
    """
    )
