"""Prediction page for Streamlit app."""

import streamlit as st
from pathlib import Path
import tempfile

st.set_page_config(page_title="Predict - Protein Design Hub", page_icon="🔮", layout="wide")

from protein_design_hub.web.ui import inject_base_css, sidebar_nav, sidebar_system_status

inject_base_css()
sidebar_nav(current="Predict")
sidebar_system_status()

st.title("🔮 Structure Prediction")
st.markdown("Run structure prediction, then download results for evaluation and comparison.")

# Sidebar - Predictor selection
st.sidebar.header("Predictor Selection")

predictor_options = {
    "ColabFold": "colabfold",
    "Chai-1": "chai1",
    "Boltz-2": "boltz2",
    "ESMFold (local)": "esmfold",
    "ESMFold (API)": "esmfold_api",
}

selected_predictors = st.sidebar.multiselect(
    "Select Predictors",
    options=list(predictor_options.keys()),
    default=["ColabFold"],
)

run_all_installed = st.sidebar.checkbox("Run all installed predictors", value=False)
if run_all_installed:
    try:
        from protein_design_hub.core.config import get_settings
        from protein_design_hub.predictors.registry import PredictorRegistry

        settings = get_settings()
        selected_predictors = []
        for label, name in predictor_options.items():
            try:
                pred = PredictorRegistry.get(name, settings)
                if pred.installer.is_installed():
                    selected_predictors.append(label)
            except Exception:
                continue
        if not selected_predictors:
            st.sidebar.warning("No predictors detected as installed.")
    except Exception:
        selected_predictors = list(predictor_options.keys())

st.sidebar.caption("Tip: `esmfold_api` works without GPU, but needs internet access.")

# Initialize settings dictionary
settings_dict = {}

# ColabFold Settings
st.sidebar.markdown("---")
with st.sidebar.expander("🔬 ColabFold Settings", expanded="ColabFold" in selected_predictors):
    st.markdown("**Basic Settings**")
    cf_num_models = st.number_input(
        "Number of models (1-5)", min_value=1, max_value=5, value=5, key="cf_num_models"
    )
    cf_num_recycles = st.number_input(
        "Number of recycles", min_value=1, max_value=48, value=3, key="cf_num_recycles"
    )
    cf_num_ensemble = st.number_input(
        "Number of ensembles", min_value=1, max_value=8, value=1, key="cf_num_ensemble"
    )
    cf_num_seeds = st.number_input(
        "Number of seeds", min_value=1, max_value=10, value=1, key="cf_num_seeds"
    )
    cf_random_seed = st.number_input("Random seed (0=random)", min_value=0, value=0, key="cf_seed")

    st.markdown("**MSA Settings**")
    cf_msa_mode = st.selectbox(
        "MSA mode",
        options=["mmseqs2_uniref_env", "mmseqs2_uniref", "single_sequence"],
        index=0,
        key="cf_msa_mode",
        help="mmseqs2_uniref_env: Full MSA, mmseqs2_uniref: Uniref only, single_sequence: No MSA",
    )
    cf_max_seq = st.number_input(
        "Max MSA sequences", min_value=64, max_value=2048, value=512, key="cf_max_seq"
    )
    cf_max_extra_seq = st.number_input(
        "Max extra sequences", min_value=64, max_value=4096, value=1024, key="cf_max_extra"
    )
    cf_pair_mode = st.selectbox(
        "Pairing mode",
        options=["unpaired", "paired", "unpaired_paired"],
        index=2,
        key="cf_pair_mode",
        help="For multimers: how to pair MSAs",
    )

    st.markdown("**Model Settings**")
    cf_model_type = st.selectbox(
        "Model type",
        options=[
            "auto",
            "alphafold2",
            "alphafold2_ptm",
            "alphafold2_multimer_v1",
            "alphafold2_multimer_v2",
            "alphafold2_multimer_v3",
        ],
        index=0,
        key="cf_model_type",
    )
    cf_rank_by = st.selectbox(
        "Rank models by",
        options=["auto", "plddt", "ptm", "iptm", "multimer"],
        index=0,
        key="cf_rank_by",
    )

    st.markdown("**Relaxation Settings**")
    cf_use_amber = st.checkbox("Use AMBER relaxation", value=False, key="cf_amber")
    if cf_use_amber:
        cf_num_relax = st.number_input(
            "Structures to relax", min_value=0, max_value=10, value=0, key="cf_num_relax"
        )
        cf_use_gpu_relax = st.checkbox("Use GPU for relaxation", value=True, key="cf_gpu_relax")
    else:
        cf_num_relax = 0
        cf_use_gpu_relax = True

    st.markdown("**Template Settings**")
    cf_use_templates = st.checkbox("Use templates", value=False, key="cf_templates")

    st.markdown("**Early Stopping**")
    cf_stop_score = st.number_input(
        "Stop at pLDDT score", min_value=0.0, max_value=100.0, value=100.0, key="cf_stop"
    )
    cf_recycle_tol = st.number_input(
        "Recycle early stop tolerance", min_value=0.0, max_value=5.0, value=0.5, key="cf_tol"
    )

    st.markdown("**Output Options**")
    cf_save_single = st.checkbox("Save single representations", value=False, key="cf_single")
    cf_save_pair = st.checkbox("Save pair representations", value=False, key="cf_pair")
    cf_save_all = st.checkbox("Save all outputs", value=False, key="cf_all")
    cf_use_dropout = st.checkbox("Use dropout (uncertainty)", value=False, key="cf_dropout")

    settings_dict["colabfold"] = {
        "num_models": cf_num_models,
        "num_recycles": cf_num_recycles,
        "num_ensemble": cf_num_ensemble,
        "num_seeds": cf_num_seeds,
        "random_seed": cf_random_seed if cf_random_seed > 0 else None,
        "msa_mode": cf_msa_mode,
        "max_seq": cf_max_seq,
        "max_extra_seq": cf_max_extra_seq,
        "pair_mode": cf_pair_mode,
        "model_type": cf_model_type,
        "rank_by": cf_rank_by,
        "use_amber": cf_use_amber,
        "num_relax": cf_num_relax,
        "use_gpu_relax": cf_use_gpu_relax,
        "use_templates": cf_use_templates,
        "stop_at_score": cf_stop_score,
        "recycle_early_stop_tolerance": cf_recycle_tol,
        "save_single_representations": cf_save_single,
        "save_pair_representations": cf_save_pair,
        "save_all": cf_save_all,
        "use_dropout": cf_use_dropout,
    }

# Chai-1 Settings
with st.sidebar.expander("🧪 Chai-1 Settings", expanded="Chai-1" in selected_predictors):
    st.markdown("**Core Settings**")
    ch_trunk_recycles = st.number_input(
        "Trunk recycles", min_value=1, max_value=20, value=3, key="ch_trunk"
    )
    ch_diffusion_steps = st.number_input(
        "Diffusion timesteps", min_value=50, max_value=1000, value=200, key="ch_diffusion"
    )
    ch_diffusion_samples = st.number_input(
        "Diffusion samples", min_value=1, max_value=20, value=5, key="ch_samples"
    )
    ch_trunk_samples = st.number_input(
        "Trunk samples", min_value=1, max_value=10, value=1, key="ch_trunk_samples"
    )
    ch_seed = st.number_input("Random seed (0=random)", min_value=0, value=0, key="ch_seed")

    st.markdown("**ESM Embeddings**")
    ch_use_esm = st.checkbox("Use ESM embeddings", value=True, key="ch_esm")

    st.markdown("**MSA Settings**")
    ch_use_msa = st.checkbox("Use MSA server", value=False, key="ch_msa")
    ch_msa_url = st.text_input(
        "MSA server URL", value="https://api.colabfold.com", key="ch_msa_url"
    )

    st.markdown("**Template Settings**")
    ch_use_templates = st.checkbox("Use templates server", value=False, key="ch_templates")

    st.markdown("**Memory Settings**")
    ch_low_memory = st.checkbox("Low memory mode", value=True, key="ch_low_mem")

    st.markdown("**Device Settings**")
    ch_device = st.text_input(
        "Device (e.g., cuda:0)", value="", key="ch_device", help="Leave empty for auto-detection"
    )

    settings_dict["chai1"] = {
        "num_trunk_recycles": ch_trunk_recycles,
        "num_diffn_timesteps": ch_diffusion_steps,
        "num_diffn_samples": ch_diffusion_samples,
        "num_trunk_samples": ch_trunk_samples,
        "seed": ch_seed if ch_seed > 0 else None,
        "use_esm_embeddings": ch_use_esm,
        "use_msa_server": ch_use_msa,
        "msa_server_url": ch_msa_url,
        "use_templates_server": ch_use_templates,
        "low_memory": ch_low_memory,
        "device": ch_device if ch_device else None,
    }

# Boltz-2 Settings
with st.sidebar.expander("⚡ Boltz-2 Settings", expanded="Boltz-2" in selected_predictors):
    st.markdown("**Model Selection**")
    bz_model = st.selectbox("Model version", options=["boltz2", "boltz1"], index=0, key="bz_model")

    st.markdown("**Core Settings**")
    bz_recycling = st.number_input(
        "Recycling steps", min_value=1, max_value=20, value=3, key="bz_recycling"
    )
    bz_sampling = st.number_input(
        "Sampling steps", min_value=50, max_value=1000, value=200, key="bz_sampling"
    )
    bz_diffusion_samples = st.number_input(
        "Diffusion samples", min_value=1, max_value=20, value=1, key="bz_samples"
    )
    bz_seed = st.number_input("Random seed (0=random)", min_value=0, value=0, key="bz_seed")

    st.markdown("**Sampling Parameters**")
    bz_step_scale = st.number_input(
        "Step scale (0=default)", min_value=0.0, max_value=5.0, value=0.0, key="bz_step"
    )

    st.markdown("**MSA Settings**")
    bz_use_msa = st.checkbox(
        "Use MSA server", value=True, key="bz_msa", help="Required for best results"
    )
    bz_msa_pairing = st.selectbox(
        "MSA pairing strategy", options=["greedy", "complete"], index=0, key="bz_pairing"
    )
    bz_max_msa = st.number_input(
        "Max MSA sequences", min_value=256, max_value=16384, value=8192, key="bz_max_msa"
    )
    bz_subsample = st.checkbox("Subsample MSA", value=True, key="bz_subsample")
    if bz_subsample:
        bz_num_subsample = st.number_input(
            "Subsampled MSA size", min_value=128, max_value=4096, value=1024, key="bz_num_sub"
        )
    else:
        bz_num_subsample = 1024

    st.markdown("**Potentials/Steering**")
    bz_potentials = st.checkbox("Use potentials", value=False, key="bz_potentials")

    st.markdown("**Affinity Prediction**")
    bz_affinity_mw = st.checkbox("Affinity MW correction", value=False, key="bz_affinity")

    st.markdown("**Output Options**")
    bz_output_format = st.selectbox(
        "Output format", options=["mmcif", "pdb"], index=0, key="bz_format"
    )
    bz_write_pae = st.checkbox("Write full PAE matrix", value=True, key="bz_pae")
    bz_write_pde = st.checkbox("Write full PDE matrix", value=False, key="bz_pde")
    bz_write_embed = st.checkbox("Write embeddings", value=False, key="bz_embed")

    st.markdown("**Performance**")
    bz_devices = st.number_input(
        "Number of devices", min_value=1, max_value=8, value=1, key="bz_devices"
    )
    bz_accelerator = st.selectbox(
        "Accelerator", options=["gpu", "cpu", "tpu"], index=0, key="bz_accel"
    )
    bz_workers = st.number_input(
        "Dataloader workers", min_value=0, max_value=16, value=2, key="bz_workers"
    )
    bz_no_kernels = st.checkbox(
        "Disable optimized kernels",
        value=True,
        key="bz_kernels",
        help="Set False if cuequivariance_torch is installed",
    )

    settings_dict["boltz2"] = {
        "model": bz_model,
        "recycling_steps": bz_recycling,
        "sampling_steps": bz_sampling,
        "diffusion_samples": bz_diffusion_samples,
        "seed": bz_seed if bz_seed > 0 else None,
        "step_scale": bz_step_scale if bz_step_scale > 0 else None,
        "use_msa_server": bz_use_msa,
        "msa_pairing_strategy": bz_msa_pairing,
        "max_msa_seqs": bz_max_msa,
        "subsample_msa": bz_subsample,
        "num_subsampled_msa": bz_num_subsample,
        "use_potentials": bz_potentials,
        "affinity_mw_correction": bz_affinity_mw,
        "output_format": bz_output_format,
        "write_full_pae": bz_write_pae,
        "write_full_pde": bz_write_pde,
        "write_embeddings": bz_write_embed,
        "devices": bz_devices,
        "accelerator": bz_accelerator,
        "num_workers": bz_workers,
        "no_kernels": bz_no_kernels,
    }

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Sequence")

    input_method = st.radio(
        "Input method",
        options=["Paste sequence", "Upload FASTA file", "Fetch from database"],
        horizontal=True,
    )

    if input_method == "Paste sequence":
        sequence_input = st.text_area(
            "Enter protein sequence (FASTA format)",
            height=200,
            placeholder=">protein_name\nMKFLILLFNILCLFPVLAADNHGVGPQGAS...",
        )
    elif input_method == "Upload FASTA file":
        uploaded_file = st.file_uploader(
            "Upload FASTA file",
            type=["fasta", "fa", "faa"],
        )
        if uploaded_file is not None:
            sequence_input = uploaded_file.read().decode("utf-8")
            st.text_area("File content", sequence_input, height=200, disabled=True)
        else:
            sequence_input = None
    else:
        # Fetch from database
        sequence_input = None

        fetch_col1, fetch_col2 = st.columns([1, 2])
        with fetch_col1:
            fetch_source = st.selectbox("Database", ["UniProt", "PDB", "AlphaFold DB"])

        with fetch_col2:
            if fetch_source == "UniProt":
                fetch_id = st.text_input("UniProt ID", placeholder="P12345 or EGFR_HUMAN")
            elif fetch_source == "PDB":
                fetch_id = st.text_input("PDB ID", placeholder="1ABC")
            else:
                fetch_id = st.text_input(
                    "UniProt ID", placeholder="P12345", help="For AlphaFold DB lookup"
                )

        if "fetched_sequence" not in st.session_state:
            st.session_state.fetched_sequence = None
            st.session_state.fetched_structure = None

        if st.button("📥 Fetch Sequence", disabled=not fetch_id):
            with st.spinner(f"Fetching from {fetch_source}..."):
                try:
                    if fetch_source == "UniProt":
                        from protein_design_hub.io.fetch import UniProtFetcher, parse_fasta

                        fetcher = UniProtFetcher()
                        result = fetcher.fetch_sequence(fetch_id.strip())

                        if result.success:
                            sequences = parse_fasta(result.data)
                            if sequences:
                                st.session_state.fetched_sequence = result.data
                                st.success(f"Fetched {len(sequences[0][1])} residues")
                        else:
                            st.error(result.error)

                    elif fetch_source == "PDB":
                        from protein_design_hub.io.fetch import PDBFetcher
                        from Bio.PDB import PDBParser
                        from Bio.SeqUtils import seq1

                        fetcher = PDBFetcher()
                        result = fetcher.fetch_structure(fetch_id.strip())

                        if result.success:
                            parser = PDBParser(QUIET=True)
                            structure = parser.get_structure("pdb", str(result.file_path))

                            seqs = []
                            for model in structure:
                                for chain in model:
                                    residues = [r for r in chain if r.id[0] == " "]
                                    if residues:
                                        sequence = "".join(seq1(r.resname) for r in residues)
                                        seqs.append(f">{fetch_id}_{chain.id}\n{sequence}")
                                break

                            if seqs:
                                st.session_state.fetched_sequence = "\n\n".join(seqs)
                                st.session_state.fetched_structure = result.data
                                st.success(f"Fetched {len(seqs)} chain(s)")
                        else:
                            st.error(result.error)

                    else:  # AlphaFold DB
                        from protein_design_hub.io.fetch import AlphaFoldDBFetcher
                        from Bio.PDB import PDBParser
                        from Bio.SeqUtils import seq1

                        fetcher = AlphaFoldDBFetcher()
                        result = fetcher.fetch_structure(fetch_id.strip())

                        if result.success:
                            parser = PDBParser(QUIET=True)
                            structure = parser.get_structure("af", str(result.file_path))

                            for model in structure:
                                for chain in model:
                                    residues = [r for r in chain if r.id[0] == " "]
                                    if residues:
                                        sequence = "".join(seq1(r.resname) for r in residues)
                                        st.session_state.fetched_sequence = (
                                            f">AF_{fetch_id}\n{sequence}"
                                        )
                                        st.session_state.fetched_structure = result.data
                                        st.success(f"Fetched {len(sequence)} residues")
                                    break
                                break
                        else:
                            st.error(result.error)

                except ImportError as e:
                    st.error(f"Required package not available: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.fetched_sequence:
            sequence_input = st.text_area(
                "Fetched sequence",
                value=st.session_state.fetched_sequence,
                height=200,
            )

with col2:
    st.subheader("Job Settings")

    job_name = st.text_input("Job name (optional)")

    try:
        from protein_design_hub.core.config import get_settings

        _settings = get_settings()
        default_out = str(_settings.output.base_dir)
    except Exception:
        default_out = "./outputs"

    output_dir = st.text_input(
        "Output directory",
        value=default_out,
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
            # Apply ColabFold settings
            for key, value in settings_dict.get("colabfold", {}).items():
                if value is not None and hasattr(settings.predictors.colabfold, key):
                    setattr(settings.predictors.colabfold, key, value)

            # Apply Chai-1 settings
            for key, value in settings_dict.get("chai1", {}).items():
                if value is not None and hasattr(settings.predictors.chai1, key):
                    setattr(settings.predictors.chai1, key, value)

            # Apply Boltz-2 settings
            for key, value in settings_dict.get("boltz2", {}).items():
                if value is not None and hasattr(settings.predictors.boltz2, key):
                    setattr(settings.predictors.boltz2, key, value)

            # Parse sequence
            parser = FastaParser()
            sequences = parser.parse(sequence_input)

            if len(sequences) > 1:
                st.info(f"Detected multimer input: {len(sequences)} chains.")

            # Create job
            job_id = job_name or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            job_dir = Path(output_dir) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            prediction_input = PredictionInput(
                job_id=job_id,
                sequences=sequences,
                output_dir=job_dir,
                num_models=settings_dict.get("colabfold", {}).get("num_models", 5),
                num_recycles=settings_dict.get("colabfold", {}).get("num_recycles", 3),
            )

            # Get predictor list
            predictor_list = [predictor_options[p] for p in selected_predictors]

            # Warn about missing installations early
            try:
                from protein_design_hub.predictors.registry import PredictorRegistry

                missing = []
                for pred_name in predictor_list:
                    pred = PredictorRegistry.get(pred_name, settings)
                    if not pred.installer.is_installed():
                        missing.append(pred_name)
                if missing:
                    st.warning(
                        "Some selected predictors are not installed: "
                        + ", ".join(missing)
                        + ". Use the Settings page or run `pdhub install predictor <name>`."
                    )
            except Exception:
                pass

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
                        st.success(
                            f"✓ {pred_name}: {len(result.structure_paths)} structures ({result.runtime_seconds:.1f}s)"
                        )
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
    st.markdown(
        """
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
    """
    )
