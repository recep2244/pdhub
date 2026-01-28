"""Batch Processing page for running multiple predictions/designs."""

import streamlit as st
from pathlib import Path
import json
import tempfile
import time

st.set_page_config(page_title="Batch - Protein Design Hub", page_icon="📦", layout="wide")

# Custom CSS
st.markdown("""
<style>
.batch-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    border-left: 4px solid #667eea;
}
.job-pending { border-left-color: #ffc107; }
.job-running { border-left-color: #17a2b8; }
.job-complete { border-left-color: #28a745; }
.job-failed { border-left-color: #dc3545; }
.progress-container {
    background: #e9ecef;
    border-radius: 10px;
    height: 20px;
    overflow: hidden;
}
.progress-bar {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    height: 100%;
    transition: width 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'batch_jobs' not in st.session_state:
    st.session_state.batch_jobs = []
if 'batch_running' not in st.session_state:
    st.session_state.batch_running = False

# Title
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 2.5rem;">
        📦 Batch Processing
    </h1>
    <p style="color: #666;">Run multiple predictions, designs, or evaluations in parallel</p>
</div>
""", unsafe_allow_html=True)

# Main tabs
main_tabs = st.tabs(["📥 Input", "⚙️ Configure", "🚀 Run", "📊 Results"])

# === INPUT TAB ===
with main_tabs[0]:
    st.markdown("### 📥 Input Sequences")

    input_method = st.radio(
        "Input method",
        ["Paste sequences", "Upload FASTA", "Upload CSV"],
        horizontal=True
    )

    sequences = []

    if input_method == "Paste sequences":
        st.markdown("**Paste sequences (one per line or FASTA format):**")

        text_input = st.text_area(
            "Sequences",
            height=200,
            placeholder=">protein_1\nMKFLILLFNILCLFPVLAADNHGVGPQGAS...\n>protein_2\nMGSSHHHHHHSSGLVPRGSHM...",
            key="batch_seq_input"
        )

        if text_input:
            lines = text_input.strip().split('\n')
            current_name = None
            current_seq = []

            for line in lines:
                line = line.strip()
                if line.startswith('>'):
                    if current_name and current_seq:
                        sequences.append({
                            'name': current_name,
                            'sequence': ''.join(current_seq)
                        })
                    current_name = line[1:].split()[0]
                    current_seq = []
                elif line:
                    if current_name is None:
                        current_name = f"sequence_{len(sequences) + 1}"
                    current_seq.append(''.join(c for c in line.upper() if c in "ACDEFGHIKLMNPQRSTVWY"))

            if current_name and current_seq:
                sequences.append({
                    'name': current_name,
                    'sequence': ''.join(current_seq)
                })

    elif input_method == "Upload FASTA":
        uploaded = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "fna"])

        if uploaded:
            content = uploaded.read().decode()
            lines = content.strip().split('\n')
            current_name = None
            current_seq = []

            for line in lines:
                line = line.strip()
                if line.startswith('>'):
                    if current_name and current_seq:
                        sequences.append({
                            'name': current_name,
                            'sequence': ''.join(current_seq)
                        })
                    current_name = line[1:].split()[0]
                    current_seq = []
                elif line:
                    current_seq.append(''.join(c for c in line.upper() if c in "ACDEFGHIKLMNPQRSTVWY"))

            if current_name and current_seq:
                sequences.append({
                    'name': current_name,
                    'sequence': ''.join(current_seq)
                })

    else:  # CSV
        uploaded = st.file_uploader("Upload CSV file", type=["csv", "tsv"])

        if uploaded:
            import pandas as pd

            try:
                df = pd.read_csv(uploaded)
                st.dataframe(df.head())

                col_name = st.selectbox("Name column", df.columns.tolist())
                col_seq = st.selectbox("Sequence column", df.columns.tolist())

                if st.button("Parse CSV"):
                    for _, row in df.iterrows():
                        seq = ''.join(c for c in str(row[col_seq]).upper() if c in "ACDEFGHIKLMNPQRSTVWY")
                        if seq:
                            sequences.append({
                                'name': str(row[col_name]),
                                'sequence': seq
                            })
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

    # Show parsed sequences
    if sequences:
        st.markdown(f"### ✅ Parsed {len(sequences)} sequences")

        # Store in session state
        st.session_state.batch_sequences = sequences

        # Preview table
        import pandas as pd
        preview_df = pd.DataFrame([
            {'Name': s['name'], 'Length': len(s['sequence']), 'Sequence': s['sequence'][:30] + '...'}
            for s in sequences
        ])
        st.dataframe(preview_df, use_container_width=True)

        # Validation
        invalid = [s for s in sequences if len(s['sequence']) < 10]
        if invalid:
            st.warning(f"{len(invalid)} sequences are very short (<10 residues)")

        long_seqs = [s for s in sequences if len(s['sequence']) > 1000]
        if long_seqs:
            st.info(f"{len(long_seqs)} sequences are >1000 residues (may take longer)")


# === CONFIGURE TAB ===
with main_tabs[1]:
    st.markdown("### ⚙️ Job Configuration")

    if 'batch_sequences' not in st.session_state or not st.session_state.batch_sequences:
        st.warning("Please input sequences first")
    else:
        # Job type selection
        job_type = st.selectbox(
            "Task type",
            ["Structure Prediction", "Sequence Design (Inverse Folding)", "Structure Evaluation", "Biophysical Analysis"]
        )

        st.markdown("---")

        if job_type == "Structure Prediction":
            st.markdown("#### Prediction Settings")

            predictor = st.selectbox(
                "Predictor",
                ["ESMFold (API)", "ESMFold (Local)", "ColabFold", "Chai-1", "Boltz-2"]
            )

            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                num_models = st.slider("Models per sequence", 1, 5, 1)
                if predictor == "ColabFold":
                    use_templates = st.checkbox("Use templates")
                    msa_mode = st.selectbox("MSA mode", ["mmseqs2_uniref_env", "mmseqs2_uniref"])

            with col_opt2:
                if predictor in ["Chai-1", "Boltz-2"]:
                    num_recycles = st.slider("Recycles", 1, 10, 3)

            st.session_state.batch_config = {
                'type': 'prediction',
                'predictor': predictor,
                'num_models': num_models,
            }

        elif job_type == "Sequence Design (Inverse Folding)":
            st.markdown("#### Design Settings")

            designer = st.selectbox("Designer", ["ProteinMPNN", "ESM-IF1"])

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                num_designs = st.slider("Designs per structure", 1, 10, 4)
                temperature = st.slider("Sampling temperature", 0.1, 2.0, 0.1)

            with col_d2:
                if designer == "ProteinMPNN":
                    backbone_noise = st.slider("Backbone noise", 0.0, 1.0, 0.0)

            st.info("Note: Requires PDB structures as input instead of sequences")

            st.session_state.batch_config = {
                'type': 'design',
                'designer': designer,
                'num_designs': num_designs,
                'temperature': temperature,
            }

        elif job_type == "Structure Evaluation":
            st.markdown("#### Evaluation Settings")

            metrics = st.multiselect(
                "Metrics to calculate",
                ["pLDDT", "RMSD", "TM-score", "Clash score", "SASA", "Contact energy",
                 "Disorder", "Shape complementarity", "Rosetta energy"],
                default=["pLDDT", "Clash score", "SASA"]
            )

            reference_option = st.radio(
                "Reference structure",
                ["None (single structure metrics)", "Upload reference", "AlphaFold DB"]
            )

            st.session_state.batch_config = {
                'type': 'evaluation',
                'metrics': metrics,
            }

        else:  # Biophysical Analysis
            st.markdown("#### Analysis Settings")

            analyses = st.multiselect(
                "Analyses to run",
                ["Basic properties (MW, pI, GRAVY)", "Solubility prediction",
                 "Stability estimation", "Disorder prediction", "Aggregation propensity"],
                default=["Basic properties (MW, pI, GRAVY)", "Solubility prediction"]
            )

            st.session_state.batch_config = {
                'type': 'biophysics',
                'analyses': analyses,
            }

        # Execution settings
        st.markdown("---")
        st.markdown("#### Execution Settings")

        col_exec1, col_exec2 = st.columns(2)

        with col_exec1:
            parallel_jobs = st.slider("Parallel jobs", 1, 8, 2)
            retry_failed = st.checkbox("Retry failed jobs", value=True)

        with col_exec2:
            save_intermediate = st.checkbox("Save intermediate results", value=True)
            output_dir = st.text_input("Output directory", value="./batch_output")


# === RUN TAB ===
with main_tabs[2]:
    st.markdown("### 🚀 Run Batch Jobs")

    if 'batch_sequences' not in st.session_state or not st.session_state.batch_sequences:
        st.warning("Please input sequences first")
    elif 'batch_config' not in st.session_state:
        st.warning("Please configure job settings first")
    else:
        sequences = st.session_state.batch_sequences
        config = st.session_state.batch_config

        st.markdown(f"""
        <div class="batch-card">
            <h4>Job Summary</h4>
            <p><b>Task:</b> {config['type'].title()}</p>
            <p><b>Sequences:</b> {len(sequences)}</p>
            <p><b>Total jobs:</b> {len(sequences)}</p>
        </div>
        """, unsafe_allow_html=True)

        col_run, col_status = st.columns([1, 2])

        with col_run:
            if st.button("▶️ Start Batch", type="primary",
                         use_container_width=True,
                         disabled=st.session_state.batch_running):

                st.session_state.batch_running = True
                st.session_state.batch_jobs = []

                # Create jobs
                for seq in sequences:
                    st.session_state.batch_jobs.append({
                        'name': seq['name'],
                        'sequence': seq['sequence'],
                        'status': 'pending',
                        'result': None,
                        'error': None,
                    })

                # Process jobs
                progress_bar = st.progress(0)
                status_container = st.container()

                for i, job in enumerate(st.session_state.batch_jobs):
                    progress_bar.progress((i + 1) / len(st.session_state.batch_jobs))
                    job['status'] = 'running'

                    with status_container:
                        st.text(f"Processing: {job['name']} ({i + 1}/{len(st.session_state.batch_jobs)})")

                    try:
                        # Run based on job type
                        if config['type'] == 'prediction':
                            # Quick ESMFold API prediction
                            if config['predictor'] == "ESMFold (API)" and len(job['sequence']) <= 400:
                                import requests

                                response = requests.post(
                                    "https://api.esmatlas.com/foldSequence/v1/pdb/",
                                    data=job['sequence'],
                                    headers={"Content-Type": "text/plain"},
                                    timeout=120,
                                )

                                if response.status_code == 200:
                                    # Extract pLDDT
                                    plddt_values = []
                                    for line in response.text.split('\n'):
                                        if line.startswith("ATOM") and line[12:16].strip() == "CA":
                                            try:
                                                plddt_values.append(float(line[60:66]))
                                            except:
                                                pass

                                    job['result'] = {
                                        'pdb': response.text,
                                        'plddt': sum(plddt_values) / len(plddt_values) if plddt_values else 0,
                                    }
                                    job['status'] = 'complete'
                                else:
                                    job['status'] = 'failed'
                                    job['error'] = f"API error: {response.status_code}"
                            else:
                                job['status'] = 'failed'
                                job['error'] = "Predictor not available for batch mode"

                        elif config['type'] == 'biophysics':
                            try:
                                from protein_design_hub.biophysics import calculate_all_properties
                                from protein_design_hub.biophysics.solubility import SolubilityPredictor

                                props = calculate_all_properties(job['sequence'])
                                sol_pred = SolubilityPredictor()
                                sol = sol_pred.predict(job['sequence'])

                                job['result'] = {
                                    'mw': props.molecular_weight,
                                    'pi': props.isoelectric_point,
                                    'gravy': props.gravy,
                                    'instability': props.instability_index,
                                    'solubility_score': sol['solubility_score'],
                                }
                                job['status'] = 'complete'

                            except ImportError:
                                job['status'] = 'failed'
                                job['error'] = "Biophysics module not available"

                        else:
                            job['status'] = 'failed'
                            job['error'] = f"Job type {config['type']} not implemented"

                    except Exception as e:
                        job['status'] = 'failed'
                        job['error'] = str(e)

                st.session_state.batch_running = False
                st.success("Batch processing complete!")
                st.rerun()

            if st.button("⏹️ Stop", disabled=not st.session_state.batch_running):
                st.session_state.batch_running = False
                st.warning("Batch stopped")

        with col_status:
            if st.session_state.batch_jobs:
                complete = sum(1 for j in st.session_state.batch_jobs if j['status'] == 'complete')
                failed = sum(1 for j in st.session_state.batch_jobs if j['status'] == 'failed')
                pending = sum(1 for j in st.session_state.batch_jobs if j['status'] == 'pending')
                running = sum(1 for j in st.session_state.batch_jobs if j['status'] == 'running')

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Complete", complete)
                col_s2.metric("Failed", failed)
                col_s3.metric("Pending", pending)
                col_s4.metric("Running", running)


# === RESULTS TAB ===
with main_tabs[3]:
    st.markdown("### 📊 Batch Results")

    if not st.session_state.batch_jobs:
        st.info("Run a batch job to see results")
    else:
        jobs = st.session_state.batch_jobs

        # Summary
        complete = [j for j in jobs if j['status'] == 'complete']
        failed = [j for j in jobs if j['status'] == 'failed']

        st.markdown(f"**Completed:** {len(complete)}/{len(jobs)} | **Failed:** {len(failed)}")

        # Results table
        if complete:
            st.markdown("#### Completed Jobs")

            config = st.session_state.get('batch_config', {})

            if config.get('type') == 'prediction':
                import pandas as pd

                data = []
                for job in complete:
                    result = job.get('result', {})
                    data.append({
                        'Name': job['name'],
                        'Length': len(job['sequence']),
                        'pLDDT': f"{result.get('plddt', 0):.1f}",
                    })

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

                # Download all structures
                if st.button("📥 Download All Structures (ZIP)"):
                    import io
                    import zipfile

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for job in complete:
                            if job.get('result', {}).get('pdb'):
                                zf.writestr(f"{job['name']}.pdb", job['result']['pdb'])

                    st.download_button(
                        "📥 Download ZIP",
                        zip_buffer.getvalue(),
                        "batch_structures.zip",
                        mime="application/zip"
                    )

            elif config.get('type') == 'biophysics':
                import pandas as pd

                data = []
                for job in complete:
                    result = job.get('result', {})
                    data.append({
                        'Name': job['name'],
                        'Length': len(job['sequence']),
                        'MW (Da)': f"{result.get('mw', 0):.0f}",
                        'pI': f"{result.get('pi', 0):.2f}",
                        'GRAVY': f"{result.get('gravy', 0):.2f}",
                        'Instability': f"{result.get('instability', 0):.1f}",
                        'Solubility': f"{result.get('solubility_score', 0):.2f}",
                    })

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "batch_biophysics.csv",
                    mime="text/csv"
                )

        # Failed jobs
        if failed:
            st.markdown("#### Failed Jobs")

            for job in failed:
                st.error(f"**{job['name']}**: {job.get('error', 'Unknown error')}")

        # Full results JSON
        st.markdown("---")
        if st.button("📥 Download Full Results (JSON)"):
            results_json = json.dumps(st.session_state.batch_jobs, indent=2, default=str)
            st.download_button(
                "Download JSON",
                results_json,
                "batch_results.json",
                mime="application/json"
            )
