"""Evaluation page for Streamlit app."""

import streamlit as st
from pathlib import Path
import tempfile
import json

st.set_page_config(page_title="Evaluate - Protein Design Hub", page_icon="📊", layout="wide")

st.title("📊 Structure Evaluation")
st.markdown("Evaluate predicted structures using quality metrics")

# Sidebar
st.sidebar.header("Metrics")

available_metrics = {
    "lDDT": ("lddt", "Local Distance Difference Test"),
    "TM-score": ("tm_score", "Template Modeling Score"),
    "QS-score": ("qs_score", "Quaternary Structure Score"),
    "RMSD": ("rmsd", "Root Mean Square Deviation"),
}

selected_metrics = st.sidebar.multiselect(
    "Select metrics",
    options=list(available_metrics.keys()),
    default=["lDDT", "TM-score", "RMSD"],
)

# Metric settings
with st.sidebar.expander("Metric Settings"):
    st.markdown("**lDDT Settings**")
    lddt_radius = st.number_input("Inclusion radius (Å)", value=15.0, min_value=5.0, max_value=30.0)
    lddt_seq_sep = st.number_input("Sequence separation", value=0, min_value=0, max_value=10)

    st.markdown("**RMSD Settings**")
    rmsd_atoms = st.selectbox(
        "Atom selection",
        options=["CA", "backbone", "heavy", "all"],
        index=0,
    )

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Structure")
    model_file = st.file_uploader(
        "Upload model structure",
        type=["pdb", "cif", "mmcif"],
        key="model",
    )

    if model_file:
        st.success(f"Loaded: {model_file.name}")

with col2:
    st.subheader("Reference Structure (Optional)")
    reference_file = st.file_uploader(
        "Upload reference structure",
        type=["pdb", "cif", "mmcif"],
        key="reference",
    )

    if reference_file:
        st.success(f"Loaded: {reference_file.name}")
    else:
        st.info("Reference required for lDDT, TM-score, QS-score")

# Run evaluation
st.markdown("---")

if st.button("📊 Run Evaluation", type="primary", use_container_width=True):
    if not model_file:
        st.error("Please upload a model structure")
    elif not selected_metrics:
        st.error("Please select at least one metric")
    else:
        try:
            from protein_design_hub.evaluation.composite import CompositeEvaluator
            from protein_design_hub.core.config import get_settings

            settings = get_settings()
            settings.evaluation.lddt.inclusion_radius = lddt_radius
            settings.evaluation.lddt.sequence_separation = lddt_seq_sep

            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(suffix=Path(model_file.name).suffix, delete=False) as tmp:
                tmp.write(model_file.read())
                model_path = Path(tmp.name)

            reference_path = None
            if reference_file:
                with tempfile.NamedTemporaryFile(suffix=Path(reference_file.name).suffix, delete=False) as tmp:
                    tmp.write(reference_file.read())
                    reference_path = Path(tmp.name)

            # Get metric names
            metric_names = [available_metrics[m][0] for m in selected_metrics]

            # Run evaluation
            with st.spinner("Evaluating..."):
                evaluator = CompositeEvaluator(metrics=metric_names, settings=settings)

                # Check availability
                available = evaluator.get_available_metrics()
                unavailable = [m for m, v in available.items() if not v]
                if unavailable:
                    requirements = evaluator.get_metric_requirements()
                    for m in unavailable:
                        st.warning(f"**{m}** not available: {requirements.get(m, 'Unknown dependency')}")

                result = evaluator.evaluate(model_path, reference_path)

            # Display results
            st.success("**Evaluation Complete!**")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if result.lddt is not None:
                    st.metric("lDDT", f"{result.lddt:.4f}")
                else:
                    st.metric("lDDT", "N/A")

            with col2:
                if result.tm_score is not None:
                    st.metric("TM-score", f"{result.tm_score:.4f}")
                else:
                    st.metric("TM-score", "N/A")

            with col3:
                if result.qs_score is not None:
                    st.metric("QS-score", f"{result.qs_score:.4f}")
                else:
                    st.metric("QS-score", "N/A")

            with col4:
                if result.rmsd is not None:
                    st.metric("RMSD", f"{result.rmsd:.4f} Å")
                else:
                    st.metric("RMSD", "N/A")

            # Additional metrics
            if result.gdt_ts is not None or result.gdt_ha is not None:
                st.markdown("#### Additional Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    if result.gdt_ts:
                        st.metric("GDT-TS", f"{result.gdt_ts:.4f}")
                with col2:
                    if result.gdt_ha:
                        st.metric("GDT-HA", f"{result.gdt_ha:.4f}")

            # Per-residue scores
            if result.lddt_per_residue:
                st.markdown("#### Per-Residue lDDT")

                import plotly.express as px
                import pandas as pd

                df = pd.DataFrame({
                    "Residue": range(1, len(result.lddt_per_residue) + 1),
                    "lDDT": result.lddt_per_residue,
                })

                fig = px.line(df, x="Residue", y="lDDT", title="lDDT per Residue")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            # Download results
            results_json = json.dumps(result.to_dict(), indent=2)
            st.download_button(
                "📥 Download Results",
                data=results_json,
                file_name="evaluation_results.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# Metric descriptions
with st.expander("ℹ️ Metric Descriptions"):
    st.markdown("""
    ### lDDT (Local Distance Difference Test)
    Measures local structural accuracy by comparing interatomic distances.
    - **Range**: 0-1 (higher is better)
    - **Interpretation**: >0.7 good, >0.9 excellent

    ### TM-score (Template Modeling Score)
    Measures global structural similarity, normalized by target length.
    - **Range**: 0-1 (higher is better)
    - **Interpretation**: >0.5 same fold, >0.8 highly similar

    ### QS-score (Quaternary Structure Score)
    Evaluates interface quality in multimeric structures.
    - **Range**: 0-1 (higher is better)
    - **Best for**: Evaluating protein complexes

    ### RMSD (Root Mean Square Deviation)
    Measures average distance between aligned atoms.
    - **Units**: Angstroms (Å)
    - **Interpretation**: Lower is better; <2Å excellent, <4Å good
    """)
