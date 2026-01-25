"""Prediction page for Streamlit app."""

import streamlit as st
from pathlib import Path
import tempfile

st.set_page_config(page_title="Predict - Protein Design Hub", page_icon="🔮", layout="wide")

st.title("🔮 Structure Prediction")
st.markdown("Run protein structure predictions using ColabFold, Chai-1, or Boltz-2")

# Sidebar - Predictor selection
st.sidebar.header("Settings")

predictor_options = {
    "ColabFold": "colabfold",
    "Chai-1": "chai1",
    "Boltz-2": "boltz2",
    "All Predictors": "all",
}

selected_predictors = st.sidebar.multiselect(
    "Select Predictors",
    options=list(predictor_options.keys()),
    default=["ColabFold"],
)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    num_models = st.number_input("Number of models", min_value=1, max_value=10, value=5)
    num_recycles = st.number_input("Number of recycles", min_value=1, max_value=20, value=3)

    st.markdown("---")
    st.markdown("**ColabFold Settings**")
    use_amber = st.checkbox("Use AMBER relaxation", value=False)
    use_templates = st.checkbox("Use templates", value=False)
    msa_mode = st.selectbox(
        "MSA mode",
        options=["mmseqs2_uniref_env", "mmseqs2_uniref", "single_sequence"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Chai-1 Settings**")
    chai_trunk_recycles = st.number_input("Trunk recycles", min_value=1, max_value=10, value=3, key="chai_recycles")
    chai_diffusion_steps = st.number_input("Diffusion timesteps", min_value=50, max_value=500, value=200)

    st.markdown("---")
    st.markdown("**Boltz-2 Settings**")
    boltz_recycling = st.number_input("Recycling steps", min_value=1, max_value=10, value=3, key="boltz_recycles")
    boltz_sampling = st.number_input("Sampling steps", min_value=50, max_value=500, value=200)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Sequence")

    input_method = st.radio(
        "Input method",
        options=["Paste sequence", "Upload FASTA file"],
        horizontal=True,
    )

    if input_method == "Paste sequence":
        sequence_input = st.text_area(
            "Enter protein sequence (FASTA format)",
            height=200,
            placeholder=">protein_name\nMKFLILLFNILCLFPVLAADNHGVGPQGAS...",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload FASTA file",
            type=["fasta", "fa", "faa"],
        )
        if uploaded_file is not None:
            sequence_input = uploaded_file.read().decode("utf-8")
            st.text_area("File content", sequence_input, height=200, disabled=True)
        else:
            sequence_input = None

with col2:
    st.subheader("Job Settings")

    job_name = st.text_input("Job name (optional)")

    output_dir = st.text_input(
        "Output directory",
        value="./outputs",
    )

    # Template upload (optional)
    st.markdown("---")
    st.markdown("**Optional: Template Structure**")
    template_file = st.file_uploader(
        "Upload template PDB/CIF",
        type=["pdb", "cif"],
    )

# Run prediction button
st.markdown("---")

if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
    if not sequence_input:
        st.error("Please provide input sequence")
    elif not selected_predictors:
        st.error("Please select at least one predictor")
    else:
        try:
            from protein_design_hub.pipeline.runner import SequentialPipelineRunner
            from protein_design_hub.core.types import PredictionInput, Sequence
            from protein_design_hub.core.config import get_settings
            from protein_design_hub.io.parsers.fasta import FastaParser
            from datetime import datetime

            settings = get_settings()

            # Update settings
            settings.predictors.colabfold.num_models = num_models
            settings.predictors.colabfold.num_recycles = num_recycles
            settings.predictors.colabfold.use_amber = use_amber
            settings.predictors.colabfold.use_templates = use_templates
            settings.predictors.colabfold.msa_mode = msa_mode
            settings.predictors.chai1.num_trunk_recycles = chai_trunk_recycles
            settings.predictors.chai1.num_diffusion_timesteps = chai_diffusion_steps
            settings.predictors.boltz2.recycling_steps = boltz_recycling
            settings.predictors.boltz2.sampling_steps = boltz_sampling

            # Parse sequence
            parser = FastaParser()
            sequences = parser.parse(sequence_input)

            # Create job
            job_id = job_name or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            job_dir = Path(output_dir) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            prediction_input = PredictionInput(
                job_id=job_id,
                sequences=sequences,
                output_dir=job_dir,
                num_models=num_models,
                num_recycles=num_recycles,
            )

            # Get predictor list
            predictor_list = []
            for p in selected_predictors:
                if p == "All Predictors":
                    predictor_list = ["colabfold", "chai1", "boltz2"]
                    break
                else:
                    predictor_list.append(predictor_options[p])

            # Run predictions
            runner = SequentialPipelineRunner(settings)

            progress_bar = st.progress(0)
            status_text = st.empty()

            results = {}
            for i, pred_name in enumerate(predictor_list):
                status_text.text(f"Running {pred_name}...")
                progress_bar.progress((i) / len(predictor_list))

                try:
                    result = runner.run_single_predictor(pred_name, prediction_input)
                    results[pred_name] = result

                    if result.success:
                        st.success(f"✓ {pred_name}: {len(result.structure_paths)} structures ({result.runtime_seconds:.1f}s)")
                    else:
                        st.error(f"✗ {pred_name}: {result.error_message}")
                except Exception as e:
                    st.warning(f"⊘ {pred_name}: {e}")

                progress_bar.progress((i + 1) / len(predictor_list))

            status_text.text("Complete!")
            progress_bar.progress(1.0)

            # Save summary
            summary_path = runner.save_results(results, job_dir)

            st.success(f"**Predictions complete!**")
            st.info(f"Results saved to: `{job_dir}`")

            # Download results
            st.download_button(
                "📥 Download Summary",
                data=open(summary_path).read(),
                file_name="prediction_summary.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# Help section
with st.expander("ℹ️ Help"):
    st.markdown("""
    ### Input Format

    Enter your sequence in FASTA format:
    ```
    >protein_name optional description
    MKFLILLFNILCLFPVLAADNHGVGPQGASLGLLDNALLFLSSHFTSDL
    ```

    For **multimers**, separate chains with a colon `:`:
    ```
    >complex
    MKFLILLFNILCLFPVLAAD:MNFLLSFVFVFLLPFVLVAD
    ```

    ### Predictor Comparison

    | Feature | ColabFold | Chai-1 | Boltz-2 |
    |---------|-----------|--------|---------|
    | Speed | Fast | Medium | Medium |
    | Multimer | ✓ | ✓ | ✓ |
    | DNA/RNA | ✗ | ✓ | ✓ |
    | Ligands | ✗ | ✓ | ✓ |
    | Templates | ✓ | ✗ | ✗ |
    """)
