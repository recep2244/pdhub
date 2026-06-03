"""Batch Processing page for running multiple predictions/designs."""

import streamlit as st
from pathlib import Path
import json

from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    section_header,
    info_box,
    metric_card,
    workflow_breadcrumb,
)
from protein_design_hub.web.agent_helpers import (
    render_contextual_insight,
    render_agent_advice_panel,
    agent_sidebar_status,
    render_all_experts_panel,
    observed_scoring_section,
)


# Base theme + navigation
inject_base_css()
sidebar_nav(current="Batch")
sidebar_system_status()
agent_sidebar_status()

# Custom CSS
st.markdown("""
<style>
.batch-card {
    background: var(--pdhub-bg-card);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid var(--pdhub-border);
    border-left: 4px solid var(--pdhub-primary);
    color: var(--pdhub-text);
}
.job-pending { border-left-color: var(--pdhub-warning); }
.job-running { border-left-color: var(--pdhub-info); }
.job-complete { border-left-color: var(--pdhub-success); }
.job-failed { border-left-color: var(--pdhub-error); }
.progress-container {
    background: var(--pdhub-bg-light);
    border-radius: 10px;
    height: 20px;
    overflow: hidden;
    border: 1px solid var(--pdhub-border);
}
.progress-bar {
    background: var(--pdhub-gradient);
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

# Page Header
page_header(
    "Batch Processing",
    "Run multiple predictions, designs, or evaluations in parallel",
    "📦"
)

workflow_breadcrumb(
    ["Configure Batch", "Run Predictions", "Evaluate All", "Export"],
    current=0,
)

# Main tabs
main_tabs = st.tabs(["📥 Input", "⚙️ Configure", "🚀 Run", "📊 Results"])

# Example batch sequences
BATCH_EXAMPLE = """>Ubiquitin
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
>T1024
MAAHKGAEHVVKASLDAGVKTVAGGLVVKAKALGGKDATMHLVAATLKKGYM
>Villin_HP35
LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"""

# === INPUT TAB ===
with main_tabs[0]:
    section_header("Input Sequences", "Add multiple sequences for batch processing", "📥")

    col_method, col_load = st.columns([3, 1])
    with col_method:
        input_method = st.radio(
            "Input method",
            ["Paste sequences", "Upload FASTA", "Upload CSV"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with col_load:
        if st.button("📋 Load Example", width='stretch', type="secondary"):
            st.session_state.batch_seq_input = BATCH_EXAMPLE
            st.rerun()

    sequences = []

    if input_method == "Paste sequences":
        info_box(
            "Paste sequences in FASTA format (>name followed by sequence). Each sequence will be processed separately.",
            variant="info"
        )

        text_input = st.text_area(
            "Sequences",
            height=200,
            placeholder=">protein_1\nMKFLILLFNILCLFPVLAADNHGVGPQGAS...\n>protein_2\nMGSSHHHHHHSSGLVPRGSHM...",
            key="batch_seq_input",
            label_visibility="collapsed"
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
        st.dataframe(preview_df, width='stretch')

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
        st.caption(
            "Batch mode currently runs **Structure Prediction (ESMFold API)** and "
            "**Biophysical Analysis** end-to-end. Other task types are not yet wired into "
            "the batch runner and are disabled below."
        )

        # Job type selection — only implemented task types are selectable
        _IMPLEMENTED_TASKS = ["Structure Prediction", "Biophysical Analysis"]
        job_type = st.selectbox(
            "Task type",
            _IMPLEMENTED_TASKS,
            help="Sequence Design and Structure Evaluation are not yet supported in batch mode.",
        )

        st.markdown("---")

        if job_type == "Structure Prediction":
            st.markdown("#### Prediction Settings")

            # Only ESMFold (API) is implemented in the batch runner.
            predictor = st.selectbox(
                "Predictor",
                ["ESMFold (API)"],
                help="Only the ESMFold API path is implemented in batch mode "
                     "(per-sequence ≤400 aa). ColabFold/Chai-1/Boltz-2 batch support is planned.",
            )
            st.caption(
                "ESMFold (API) folds single chains up to 400 aa with no MSA. "
                "It returns pLDDT only — no global lDDT/TM or interface (ipTM/PAE) accuracy."
            )

            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                num_models = st.slider(
                    "Models per sequence", 1, 1, 1, disabled=True,
                    help="ESMFold API returns a single model per sequence.",
                )

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
        st.caption(
            "Batch jobs run sequentially in-process. Parallelism, retry, and output-dir "
            "persistence are not yet implemented, so the controls below are disabled."
        )

        col_exec1, col_exec2 = st.columns(2)

        with col_exec1:
            parallel_jobs = st.slider("Parallel jobs", 1, 8, 1, disabled=True)
            retry_failed = st.checkbox("Retry failed jobs", value=False, disabled=True)

        with col_exec2:
            save_intermediate = st.checkbox("Save intermediate results", value=False, disabled=True)
            output_dir = st.text_input("Output directory", value="./batch_output", disabled=True)


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
                         width='stretch',
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
                n_jobs = len(st.session_state.batch_jobs)
                progress_bar = st.progress(0)
                status_text = st.empty()
                live_table = st.empty()

                _completed = 0
                _failed = 0

                for i, job in enumerate(st.session_state.batch_jobs):
                    job['status'] = 'running'
                    status_text.markdown(
                        f"⚙️ **Processing** `{job['name']}` — "
                        f"{i+1}/{n_jobs} &nbsp;|&nbsp; "
                        f"✅ {_completed} &nbsp;❌ {_failed}"
                    )

                    try:
                        if config['type'] == 'prediction':
                            if config['predictor'] == "ESMFold (API)" and len(job['sequence']) <= 400:
                                import requests
                                response = requests.post(
                                    "https://api.esmatlas.com/foldSequence/v1/pdb/",
                                    data=job['sequence'],
                                    headers={"Content-Type": "text/plain"},
                                    timeout=120,
                                )
                                if response.status_code == 200:
                                    plddt_values = []
                                    for line in response.text.split('\n'):
                                        if line.startswith("ATOM") and line[12:16].strip() == "CA":
                                            try:
                                                plddt_values.append(float(line[60:66]))
                                            except Exception:
                                                pass
                                    job['result'] = {
                                        'pdb': response.text,
                                        'plddt': sum(plddt_values) / len(plddt_values) if plddt_values else 0,
                                    }
                                    job['status'] = 'complete'
                                    _completed += 1
                                else:
                                    job['status'] = 'failed'
                                    job['error'] = f"API error {response.status_code}"
                                    _failed += 1
                            elif len(job['sequence']) > 400 and config['predictor'] == "ESMFold (API)":
                                job['status'] = 'failed'
                                job['error'] = f"Sequence too long for API ({len(job['sequence'])} aa > 400 limit)"
                                _failed += 1
                            else:
                                job['status'] = 'failed'
                                job['error'] = f"Predictor '{config['predictor']}' not available in batch mode — use ESMFold (API)"
                                _failed += 1

                        elif config['type'] == 'biophysics':
                            from protein_design_hub.biophysics import (
                                calculate_mw, calculate_pi, calculate_gravy,
                                calculate_instability_index,
                            )
                            from protein_design_hub.biophysics.solubility import SolubilityPredictor
                            seq = job['sequence']
                            sol_pred = SolubilityPredictor(sequence=seq)
                            sol = sol_pred.predict()
                            job['result'] = {
                                'mw': calculate_mw(seq),
                                'pi': calculate_pi(seq),
                                'gravy': calculate_gravy(seq),
                                'instability': calculate_instability_index(seq),
                                'solubility_score': sol['solubility_score'],
                                'aggregation': sol.get('aggregation_propensity', 0),
                                'overall': sol.get('overall_assessment', ''),
                            }
                            job['status'] = 'complete'
                            _completed += 1

                        else:
                            job['status'] = 'failed'
                            job['error'] = f"Job type '{config['type']}' not implemented in batch mode"
                            _failed += 1

                    except Exception as e:
                        job['status'] = 'failed'
                        job['error'] = str(e)
                        _failed += 1

                    progress_bar.progress((i + 1) / n_jobs)

                status_text.empty()
                st.session_state.batch_running = False
                if _failed == 0:
                    st.success(f"✅ All {_completed} jobs completed successfully!")
                elif _completed == 0:
                    st.error(f"❌ All {_failed} jobs failed — check the Results tab for details.")
                else:
                    st.warning(f"⚠️ Completed {_completed}/{n_jobs} jobs. {_failed} failed — see Results tab.")
                st.rerun()

            st.caption(
                "Batch runs synchronously and cannot be interrupted mid-run; "
                "the page is responsive again once all jobs finish."
            )

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

                # pLDDT QC filter — confidence axis only
                _plddt_thresh = st.slider(
                    "Min pLDDT for downstream use",
                    min_value=30.0, max_value=95.0, value=70.0, step=5.0,
                    help=">70 = confident; >90 = very high confidence (per-residue self-confidence)",
                    key="qc_plddt_thresh",
                )
                _pass_plddt = [j for j in complete if j.get("result", {}).get("plddt", 0) >= _plddt_thresh]
                if _pass_plddt:
                    st.success(f"✅ **{len(_pass_plddt)}/{len(complete)} structures pass pLDDT ≥ {_plddt_thresh:.0f}**")
                else:
                    st.warning(f"No structures pass pLDDT ≥ {_plddt_thresh:.0f}. Lower the threshold or use a different predictor.")
                st.caption(
                    "pLDDT is ESMFold's per-residue **self-confidence**, not measured accuracy. "
                    "A confident-but-wrong fold can still score high, and ESMFold gives no "
                    "global lDDT/TM-score and no interface accuracy (ipTM/PAE). For binders or "
                    "complexes, validate interface quality with ipSAE/PAE from Chai/Boltz."
                )

                st.dataframe(df, width='stretch')

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

                # Observed scoring for batch-predicted structures.
                # Write PDB strings to a stable per-session dir (one file per job name)
                # so reruns overwrite instead of leaking a new tempfile every render.
                import tempfile as _tmpmod
                import re as _re
                _batch_dir = Path(_tmpmod.gettempdir()) / "pdhub_batch_pdb"
                _batch_dir.mkdir(parents=True, exist_ok=True)
                _batch_pdb_paths = []
                for _job in complete:
                    _pdb_str = _job.get("result", {}).get("pdb")
                    if _pdb_str:
                        _safe = _re.sub(r"[^A-Za-z0-9_.-]", "_", str(_job["name"]))[:60]
                        _ppath = _batch_dir / f"{_safe}.pdb"
                        _ppath.write_text(_pdb_str)
                        _batch_pdb_paths.append(_ppath)
                if _batch_pdb_paths:
                    observed_scoring_section(
                        model_paths=_batch_pdb_paths,
                        section_key="batch_obs",
                    )

            elif config.get('type') == 'biophysics':
                import pandas as pd

                data = []
                for job in complete:
                    result = job.get('result', {})
                    instability = result.get('instability', 0)
                    sol = result.get('solubility_score', 0)
                    agg = result.get('aggregation', 0)
                    data.append({
                        'Name': job['name'],
                        'Length': len(job['sequence']),
                        'MW (Da)': f"{result.get('mw', 0):.0f}",
                        'pI': f"{result.get('pi', 0):.2f}",
                        'GRAVY': f"{result.get('gravy', 0):.2f}",
                        'Instability': f"{instability:.1f}",
                        'Stable?': '✅' if instability < 40 else '⚠️',
                        'Solubility': f"{sol:.2f}",
                        'Aggregation': f"{agg:.2f}",
                        'Assessment': result.get('overall', ''),
                    })

                df = pd.DataFrame(data)

                # QC filter
                with st.expander("🔍 QC Filter", expanded=False):
                    st.markdown("Filter sequences by biophysical criteria for downstream work:")
                    _qc_col1, _qc_col2, _qc_col3 = st.columns(3)
                    with _qc_col1:
                        _pi_min = st.number_input("Min pI", value=5.0, min_value=1.0, max_value=14.0, step=0.1, key="qc_pi_min")
                        _pi_max = st.number_input("Max pI", value=9.0, min_value=1.0, max_value=14.0, step=0.1, key="qc_pi_max")
                    with _qc_col2:
                        _instab_max = st.number_input("Max instability index", value=40.0, min_value=0.0, max_value=200.0, step=1.0,
                                                       help="< 40 = stable protein (Parker scale)", key="qc_instab")
                        _gravy_max = st.number_input("Max GRAVY", value=0.0, min_value=-5.0, max_value=5.0, step=0.1,
                                                      help="Negative = hydrophilic (better solubility)", key="qc_gravy")
                    with _qc_col3:
                        _sol_min = st.number_input("Min solubility score", value=0.5, min_value=0.0, max_value=1.0, step=0.05, key="qc_sol")

                    _pass_qc = []
                    for job in complete:
                        r = job.get("result", {})
                        _pi = r.get("pi", 7.0)
                        _inst = r.get("instability", 999.0)
                        _grav = r.get("gravy", 0.0)
                        _sol = r.get("solubility_score", 0.0)
                        if _pi_min <= _pi <= _pi_max and _inst <= _instab_max and _grav <= _gravy_max and _sol >= _sol_min:
                            _pass_qc.append(job["name"])

                    if _pass_qc:
                        st.success(f"✅ **{len(_pass_qc)}/{len(complete)} sequences pass QC:** " + ", ".join(_pass_qc[:10])
                                   + ("..." if len(_pass_qc) > 10 else ""))
                    else:
                        st.warning("No sequences pass current QC criteria — relax the filters.")

                st.dataframe(df, width='stretch')

                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "batch_biophysics.csv",
                    mime="text/csv"
                )

                # ── Wet-Lab Go/No-Go per sequence ──────────────────────────
                with st.expander("🧪 Wet-Lab Go/No-Go Assessment (all sequences)", expanded=False):
                    st.markdown(
                        "Quick wet-lab advancement decision for each completed sequence "
                        "based on biophysical properties."
                    )
                    try:
                        from protein_design_hub.analysis.wet_lab_advisor import build_wet_lab_report as _build_wlr
                        _wl_rows = []
                        for _bj in complete[:20]:  # cap at 20 to avoid long waits
                            _bjseq = _bj.get("sequence", "")
                            if len(_bjseq) < 10:
                                continue
                            try:
                                _bjr = _build_wlr(_bjseq, is_antibody=False, codon_hosts=["E. coli", "HEK293/CHO"])
                                _top_sys = _bjr.expression_systems[0] if _bjr.expression_systems else None
                                _wl_rows.append({
                                    "Name": _bj["name"],
                                    "Verdict": _bjr.go_nogo,
                                    "Top System": _top_sys.system if _top_sys else "N/A",
                                    "Est. Yield": (
                                        f"{_top_sys.estimated_yield_mgl[0]:.0f}–{_top_sys.estimated_yield_mgl[1]:.0f} mg/L"
                                        if _top_sys else "N/A"
                                    ),
                                    "Timeline": (
                                        f"{_top_sys.timeline_days[0]}–{_top_sys.timeline_days[1]} days"
                                        if _top_sys else "N/A"
                                    ),
                                    "Key Issue": (
                                        _bjr.criteria_failed[0][:60] + "…"
                                        if _bjr.criteria_failed
                                        else (_bjr.criteria_watch[0][:60] + "…" if _bjr.criteria_watch else "—")
                                    ),
                                })
                            except Exception:
                                pass
                        if _wl_rows:
                            _wl_df = pd.DataFrame(_wl_rows)
                            st.dataframe(_wl_df, width='stretch', hide_index=True)
                            _go_count = sum(1 for r in _wl_rows if r["Verdict"] == "GO")
                            _cond_count = sum(1 for r in _wl_rows if r["Verdict"] == "CONDITIONAL")
                            _nogo_count = sum(1 for r in _wl_rows if r["Verdict"] == "NO-GO")
                            st.markdown(
                                f"**Summary:** {_go_count} GO ✅ | {_cond_count} CONDITIONAL ⚠️ | {_nogo_count} NO-GO ❌"
                            )
                        else:
                            st.info("No sequences with sufficient length for wet-lab assessment.")
                    except Exception as _wlbatch_e:
                        st.warning(f"Wet-lab assessment unavailable: {_wlbatch_e}")

            # All-experts review for batch outcomes
            config = st.session_state.get('batch_config', {})
            success_rate = (len(complete) / len(jobs)) if jobs else 0.0
            sample_names = ", ".join(j["name"] for j in complete[:10])
            if len(complete) > 10:
                sample_names += ", ..."
            context_lines = [
                f"Batch type: {config.get('type', 'unknown')}",
                f"Total jobs: {len(jobs)}",
                f"Completed: {len(complete)}",
                f"Failed: {len(failed)}",
                f"Success rate: {success_rate:.1%}",
                f"Sample completed job names: {sample_names or 'none'}",
            ]

            batch_data = {
                "Batch type": config.get('type', 'unknown'),
                "Total jobs": len(jobs),
                "Completed": len(complete),
                "Failed": len(failed),
                "Success rate": f"{success_rate:.1%}",
            }
            render_contextual_insight(
                "Batch",
                batch_data,
                key_prefix="batch_ctx",
            )

            render_agent_advice_panel(
                page_context="\n".join(context_lines),
                default_question=(
                    "What do the failure patterns suggest? Which completed "
                    "jobs should be prioritized for downstream analysis?"
                ),
                expert="Computational Biologist",
                key_prefix="batch_agent",
            )

            _batch_domain_ctx = "\n".join([
                *context_lines,
                "",
                "Wet lab synthesis budget: ~$100-200/gene construct; typical screening batch = 4-16 variants. "
                "Go criteria: pLDDT>80, instability<40, GRAVY<0, no high-risk CDR PTMs.",
                "Immunology: therapeutic candidates need pLDDT>80 AND humanness>85% AND no high MHC-II epitopes.",
                "Plant biology: select constructs with correct codon usage for target plant host "
                "(Arabidopsis/tobacco/rice codon tables differ); include p19 silencing suppressor "
                "in Agrobacterium mix for highest transient expression yield.",
            ])
            render_all_experts_panel(
                "All-Expert Review (batch job)",
                agenda=(
                    "Assess batch run quality and identify which designs are ready for wet lab advancement "
                    "from immunology, wet lab, and plant biology experimental perspectives."
                ),
                context=_batch_domain_ctx,
                questions=(
                    "Do the batch failure/success patterns suggest setup issues (sequence length limits, "
                    "predictor limitations with certain folds like LRR arrays or coiled-coil NLRs) "
                    "or expected noise — and for therapeutic antibody batches, what fraction of designs "
                    "have acceptable pLDDT AND low immunogenicity risk?",
                    "Based on batch results, which top 3-5 sequences should be ordered for gene synthesis "
                    "and wet lab expression screening this week? Selection criteria: pLDDT threshold, "
                    "instability index, GRAVY, absence of PTM liabilities in CDR/active site regions, "
                    "and absence of rare codons for the target expression host.",
                    "From a plant biology perspective: for plant protein batches (enzymes, NLR immune "
                    "receptors, designer binders for crop protection), which completed predictions show "
                    "stable, well-folded candidates for Agrobacterium-mediated transient expression in "
                    "N. benthamiana — and what construct architecture (35S promoter, signal peptide, "
                    "His/Strep tag, p19 co-expression) maximizes protein yield?",
                ),
                key_prefix="batch_all",
            )

        # Failed jobs
        if failed:
            st.markdown("#### Failed Jobs")

            for job in failed:
                st.error(f"**{job['name']}**: {job.get('error', 'Unknown error')}")

        # Full results JSON
        st.markdown("---")
        results_json = json.dumps(st.session_state.batch_jobs, indent=2, default=str)
        st.download_button(
            "📥 Download Full Results (JSON)",
            results_json,
            "batch_results.json",
            mime="application/json",
        )
