"""Compare page for Streamlit app."""

import streamlit as st
from pathlib import Path
import tempfile
import json

st.set_page_config(page_title="Compare - Protein Design Hub", page_icon="⚖️", layout="wide")

st.title("⚖️ Compare Predictions")
st.markdown("Run all predictors and compare results to find the best prediction")

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
    eval_metrics = st.multiselect(
        "Evaluation metrics",
        options=["lDDT", "TM-score", "QS-score", "RMSD"],
        default=["lDDT", "TM-score"],
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
    output_dir = st.text_input("Output directory", value="./outputs", key="compare_output")
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
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
                tmp.write(sequence_input)
                input_path = Path(tmp.name)

            # Save reference if provided
            reference_path = None
            if reference_file:
                with tempfile.NamedTemporaryFile(suffix=Path(reference_file.name).suffix, delete=False) as tmp:
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
                pred_data.append({
                    "Predictor": name.upper(),
                    "Status": "✓ Success" if pred.success else "✗ Failed",
                    "Structures": len(pred.structure_paths),
                    "Runtime (s)": f"{pred.runtime_seconds:.1f}",
                    "Best pLDDT": f"{max((s.plddt for s in pred.scores if s.plddt), default=0):.1f}" if pred.scores else "N/A",
                })

            import pandas as pd
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            # Evaluation results (if reference provided)
            if result.evaluation_results:
                st.markdown("### Evaluation Results")

                eval_data = []
                for name, eval_result in result.evaluation_results.items():
                    eval_data.append({
                        "Predictor": name.upper(),
                        "lDDT": f"{eval_result.lddt:.3f}" if eval_result.lddt else "N/A",
                        "TM-score": f"{eval_result.tm_score:.3f}" if eval_result.tm_score else "N/A",
                        "RMSD (Å)": f"{eval_result.rmsd:.2f}" if eval_result.rmsd else "N/A",
                        "QS-score": f"{eval_result.qs_score:.3f}" if eval_result.qs_score else "N/A",
                    })

                eval_df = pd.DataFrame(eval_data)
                st.dataframe(eval_df, use_container_width=True, hide_index=True)

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
                            fig.add_trace(go.Bar(
                                name=metric,
                                x=[p.upper() for p in predictors],
                                y=values,
                            ))

                    fig.update_layout(
                        barmode='group',
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

# Info section
with st.expander("ℹ️ About Comparison"):
    st.markdown("""
    ### How Comparison Works

    1. **Prediction Phase**: Each selected predictor runs on your input sequence
    2. **Evaluation Phase**: If a reference is provided, structures are evaluated
    3. **Ranking Phase**: Predictors are ranked by evaluation scores (or pLDDT if no reference)

    ### Ranking Criteria

    - **With reference**: Ranking by lDDT score
    - **Without reference**: Ranking by pLDDT (predicted confidence)

    ### Tips

    - Provide a reference structure when available for more accurate comparison
    - Longer sequences take more time to predict
    - Each predictor runs sequentially to manage GPU memory
    """)
