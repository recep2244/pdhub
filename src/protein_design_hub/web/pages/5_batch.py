"""Batch Processing page for multiple sequence predictions."""

import streamlit as st
from pathlib import Path
import tempfile
import time
from datetime import datetime

st.set_page_config(page_title="Batch - Protein Design Hub", page_icon="📦", layout="wide")

from protein_design_hub.web.ui import inject_base_css, sidebar_nav, sidebar_system_status

inject_base_css()
sidebar_nav(current="Batch")
sidebar_system_status()

# Custom CSS
st.markdown(
    """
<style>
.job-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #667eea;
}
.job-card-running {
    border-left-color: #f39c12;
    animation: pulse 2s infinite;
}
.job-card-completed {
    border-left-color: #27ae60;
}
.job-card-failed {
    border-left-color: #e74c3c;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
}
.status-pending { background: #f39c12; color: white; }
.status-running { background: #3498db; color: white; }
.status-completed { background: #27ae60; color: white; }
.status-failed { background: #e74c3c; color: white; }
.status-cancelled { background: #95a5a6; color: white; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📦 Batch Processing")
st.markdown("Submit multiple sequences for batch prediction and track job progress")

# Initialize session state
if "batch_sequences" not in st.session_state:
    st.session_state.batch_sequences = []

# Sidebar - Job Queue
st.sidebar.header("🔄 Job Queue")

try:
    from protein_design_hub.core.job_manager import get_job_manager, JobStatus

    job_manager = get_job_manager()

    # Show active jobs
    running_jobs = job_manager.list_jobs(status=JobStatus.RUNNING, limit=5)
    pending_jobs = job_manager.list_jobs(status=JobStatus.PENDING, limit=10)

    if running_jobs:
        st.sidebar.markdown("**Running:**")
        for job in running_jobs:
            st.sidebar.markdown(
                f"""
            <div class="job-card job-card-running">
                <b>{job.name}</b><br>
                <small>{job.progress_message or 'Processing...'}</small><br>
                <progress value="{job.progress}" max="1" style="width:100%"></progress>
            </div>
            """,
                unsafe_allow_html=True,
            )

    if pending_jobs:
        st.sidebar.markdown("**Pending:**")
        for job in pending_jobs:
            st.sidebar.text(f"• {job.name}")

    # Queue stats
    st.sidebar.markdown("---")
    all_jobs = job_manager.list_jobs(limit=100)
    completed = sum(1 for j in all_jobs if j.status == JobStatus.COMPLETED)
    failed = sum(1 for j in all_jobs if j.status == JobStatus.FAILED)
    st.sidebar.metric("Completed Jobs", completed)
    st.sidebar.metric("Failed Jobs", failed)

except ImportError as e:
    st.sidebar.warning(f"Job manager not available: {e}")
    job_manager = None

# Main content
tab_submit, tab_queue, tab_history = st.tabs(["📝 Submit Jobs", "📋 Job Queue", "📜 History"])

# ========== SUBMIT TAB ==========
with tab_submit:
    st.subheader("Submit Batch Prediction")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Input Sequences")

        input_method = st.radio(
            "Input method",
            ["Paste multiple sequences", "Upload multi-FASTA file", "Add sequences one by one"],
            horizontal=True,
        )

        if input_method == "Paste multiple sequences":
            sequences_text = st.text_area(
                "Paste sequences (FASTA format, multiple sequences)",
                height=300,
                placeholder=""">protein_1
MKFLILLFNILCLFPVLAADNHGVGPQGAS...

>protein_2
MAEGEITTFTALTEKFNLPPGNYKKPKLLY...

>protein_3
MVLSPADKTNVKAAWGKVGAHAGEYGAEAL...""",
            )

            if sequences_text:
                # Parse sequences
                sequences = []
                current_name = None
                current_seq = []

                for line in sequences_text.strip().split("\n"):
                    if line.startswith(">"):
                        if current_name and current_seq:
                            sequences.append(
                                {
                                    "name": current_name,
                                    "sequence": "".join(current_seq),
                                }
                            )
                        current_name = line[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line.strip())

                if current_name and current_seq:
                    sequences.append(
                        {
                            "name": current_name,
                            "sequence": "".join(current_seq),
                        }
                    )

                st.session_state.batch_sequences = sequences
                st.success(f"Parsed {len(sequences)} sequences")

        elif input_method == "Upload multi-FASTA file":
            uploaded = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "faa"])
            if uploaded:
                content = uploaded.read().decode()
                sequences = []
                current_name = None
                current_seq = []

                for line in content.strip().split("\n"):
                    if line.startswith(">"):
                        if current_name and current_seq:
                            sequences.append(
                                {
                                    "name": current_name,
                                    "sequence": "".join(current_seq),
                                }
                            )
                        current_name = line[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line.strip())

                if current_name and current_seq:
                    sequences.append(
                        {
                            "name": current_name,
                            "sequence": "".join(current_seq),
                        }
                    )

                st.session_state.batch_sequences = sequences
                st.success(f"Loaded {len(sequences)} sequences from file")

        else:
            # Add one by one
            with st.form("add_sequence"):
                seq_name = st.text_input("Sequence name")
                seq_content = st.text_area("Sequence", height=100)
                if st.form_submit_button("Add Sequence"):
                    if seq_name and seq_content:
                        st.session_state.batch_sequences.append(
                            {
                                "name": seq_name,
                                "sequence": seq_content.replace("\n", "").replace(" ", ""),
                            }
                        )
                        st.success(f"Added {seq_name}")
                        st.rerun()

        # Show current sequences
        if st.session_state.batch_sequences:
            st.markdown("### Sequences to Process")

            for i, seq in enumerate(st.session_state.batch_sequences):
                col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
                with col_s1:
                    st.text(f"{seq['name']} ({len(seq['sequence'])} aa)")
                with col_s2:
                    st.text(f"{seq['sequence'][:20]}...")
                with col_s3:
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.batch_sequences.pop(i)
                        st.rerun()

            if st.button("Clear All", type="secondary"):
                st.session_state.batch_sequences = []
                st.rerun()

    with col2:
        st.markdown("### Settings")

        predictor = st.selectbox(
            "Predictor",
            ["ColabFold", "Chai-1", "Boltz-2", "ESMFold (Fast)"],
        )

        predictor_map = {
            "ColabFold": "colabfold",
            "Chai-1": "chai1",
            "Boltz-2": "boltz2",
            "ESMFold (Fast)": "esmfold",
        }

        num_models = st.number_input("Models per sequence", 1, 5, 1)
        num_recycles = st.number_input("Recycles", 1, 10, 3)

        try:
            from protein_design_hub.core.config import get_settings

            _settings = get_settings()
            default_out = str(_settings.output.base_dir / "batch")
        except Exception:
            default_out = "./batch_outputs"

        output_dir = st.text_input("Output directory", default_out)

        # Priority
        priority = st.selectbox("Priority", ["Normal", "High", "Low"])

        # Notification
        notify_email = st.text_input("Notification email (optional)")

    # Submit button
    st.markdown("---")

    if st.button("🚀 Submit Batch Job", type="primary", use_container_width=True):
        if not st.session_state.batch_sequences:
            st.error("No sequences to process")
        elif job_manager is None:
            st.error("Job manager not available")
        else:
            # Create jobs for each sequence
            submitted_jobs = []

            for seq in st.session_state.batch_sequences:
                job = job_manager.create_job(
                    job_type="prediction",
                    name=f"{predictor} - {seq['name']}",
                    input_data={
                        "sequences": [{"id": seq["name"], "sequence": seq["sequence"]}],
                        "predictor": predictor_map[predictor],
                        "num_models": num_models,
                        "num_recycles": num_recycles,
                        "output_dir": output_dir,
                    },
                )
                job_manager.submit_job(job)
                submitted_jobs.append(job)

            st.success(f"Submitted {len(submitted_jobs)} jobs to the queue!")

            # Show job IDs
            with st.expander("Job IDs"):
                for job in submitted_jobs:
                    st.text(f"{job.id}: {job.name}")

            # Clear sequences
            st.session_state.batch_sequences = []

# ========== QUEUE TAB ==========
with tab_queue:
    st.subheader("Current Job Queue")

    if job_manager:
        # Refresh button
        if st.button("🔄 Refresh"):
            st.rerun()

        # Running jobs
        running = job_manager.list_jobs(status=JobStatus.RUNNING)
        if running:
            st.markdown("### 🔄 Running")
            for job in running:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{job.name}**")
                    st.progress(job.progress)
                    st.caption(job.progress_message or "Processing...")
                with col2:
                    started = datetime.fromisoformat(job.started_at) if job.started_at else None
                    if started:
                        elapsed = (datetime.now() - started).total_seconds()
                        st.text(f"⏱️ {elapsed:.0f}s")
                with col3:
                    if st.button("Cancel", key=f"cancel_{job.id}"):
                        job_manager.cancel_job(job.id)
                        st.rerun()

        # Pending jobs
        pending = job_manager.list_jobs(status=JobStatus.PENDING)
        if pending:
            st.markdown("### ⏳ Pending")
            for job in pending:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"• {job.name}")
                with col2:
                    if st.button("Cancel", key=f"cancel_{job.id}"):
                        job_manager.cancel_job(job.id)
                        st.rerun()

        if not running and not pending:
            st.info("No jobs in queue")
    else:
        st.warning("Job manager not available")

# ========== HISTORY TAB ==========
with tab_history:
    st.subheader("Job History")

    if job_manager:
        col_filter1, col_filter2, col_filter3 = st.columns(3)

        with col_filter1:
            status_filter = st.selectbox(
                "Status",
                ["All", "Completed", "Failed", "Cancelled"],
            )

        with col_filter2:
            type_filter = st.selectbox(
                "Type",
                ["All", "prediction", "evaluation", "comparison"],
            )

        with col_filter3:
            limit = st.number_input("Show", 10, 100, 25)

        # Get jobs
        status_map = {
            "Completed": JobStatus.COMPLETED,
            "Failed": JobStatus.FAILED,
            "Cancelled": JobStatus.CANCELLED,
        }

        status = status_map.get(status_filter)
        job_type = type_filter if type_filter != "All" else None

        jobs = job_manager.list_jobs(status=status, job_type=job_type, limit=limit)

        if jobs:
            for job in jobs:
                status_class = job.status.value
                card_class = f"job-card job-card-{status_class}"

                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.markdown(f"**{job.name}**")
                    st.caption(f"ID: {job.id} | Created: {job.created_at[:19]}")

                with col2:
                    badge_class = f"status-{status_class}"
                    st.markdown(
                        f'<span class="status-badge {badge_class}">{job.status.value.upper()}</span>',
                        unsafe_allow_html=True,
                    )

                with col3:
                    if job.completed_at and job.started_at:
                        start = datetime.fromisoformat(job.started_at)
                        end = datetime.fromisoformat(job.completed_at)
                        duration = (end - start).total_seconds()
                        st.text(f"⏱️ {duration:.1f}s")

                with col4:
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if job.status == JobStatus.COMPLETED:
                            if st.button("📁", key=f"view_{job.id}", help="View results"):
                                st.session_state.view_job = job.id
                    with col_btn2:
                        if st.button("🗑️", key=f"del_{job.id}", help="Delete"):
                            job_manager.delete_job(job.id)
                            st.rerun()

                # Show error if failed
                if job.status == JobStatus.FAILED and job.error_message:
                    st.error(f"Error: {job.error_message}")

                # Show results if viewing
                if st.session_state.get("view_job") == job.id:
                    with st.expander("Results", expanded=True):
                        st.json(job.output_data)

                st.markdown("---")

            # Clear history button
            if st.button("🗑️ Clear Completed Jobs"):
                job_manager.clear_completed()
                st.rerun()
        else:
            st.info("No jobs in history")
    else:
        st.warning("Job manager not available")

# Help section
with st.expander("ℹ️ Help"):
    st.markdown(
        """
    ### Batch Processing

    Submit multiple protein sequences for structure prediction in batch mode.

    **Features:**
    - Queue multiple sequences for prediction
    - Track job progress in real-time
    - View job history and results
    - Cancel pending or running jobs

    **Workflow:**
    1. Add sequences (paste, upload, or one-by-one)
    2. Select predictor and settings
    3. Submit batch job
    4. Monitor progress in Job Queue tab
    5. View results in History tab

    **Tips:**
    - Use ESMFold for fast initial predictions
    - Use ColabFold for highest accuracy
    - Longer sequences take more time
    - GPU memory limits batch size
    """
    )
