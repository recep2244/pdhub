"""Settings page for Streamlit app with comprehensive configuration options."""

import streamlit as st
from pathlib import Path
import json
from typing import Iterable, List, Tuple

st.set_page_config(page_title="Settings - Protein Design Hub", page_icon="⚙️", layout="wide")

from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    section_header,
    status_badge,
)

inject_base_css()
sidebar_nav(current="Settings")
sidebar_system_status()

# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Custom CSS based on theme
theme_css = """
<style>
.settings-card {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-md);
    padding: 20px;
    margin: 12px 0;
    border: 1px solid var(--pdhub-border);
    color: var(--pdhub-text);
}
.settings-card-dark {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-md);
    padding: 20px;
    margin: 12px 0;
    color: var(--pdhub-text);
    border: 1px solid var(--pdhub-border);
}
.status-ok {
    color: var(--pdhub-success);
    font-weight: bold;
}
.status-error {
    color: var(--pdhub-error);
    font-weight: bold;
}
.status-warning {
    color: var(--pdhub-warning);
    font-weight: bold;
}
.metric-card {
    background: var(--pdhub-gradient);
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    color: white;
    margin: 5px;
}
.section-header {
    border-bottom: 1px solid var(--pdhub-border);
    padding-bottom: 8px;
    margin-bottom: 16px;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}
.muted-text {
    color: var(--pdhub-text-secondary);
    font-size: 0.9rem;
}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

st.title("⚙️ Settings & Configuration")
st.markdown("Configure Protein Design Hub, manage installations, and customize your experience")

# Utility helpers
def _render_badges(items: Iterable[Tuple[str, str]]) -> None:
    badges = " ".join(status_badge(label, status) for label, status in items if label)
    if badges:
        st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)

# Tabs
# Tabs
tabs = st.tabs(
    [
        "📊 System Status",
        "📦 Installations",
        "🎨 Preferences",
        "🔧 Configuration",
        "📁 Cache & Data",
        "📋 Logs",
    ]
)

# ==================== TAB 1: System Status ====================
with tabs[0]:
    section_header("System Status Overview", "Live hardware and tool health signals.", "📊")

    # Refresh button
    col_refresh, col_spacer = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh Status", key="refresh_status"):
            st.rerun()

    # GPU Status Section
    st.markdown("### 🖥️ GPU Status")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_cols = st.columns(4)

            with gpu_cols[0]:
                st.markdown(
                    """
                <div class="metric-card">
                    <h3>GPU</h3>
                    <p style="font-size: 14px;">"""
                    + torch.cuda.get_device_name(0)
                    + """</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with gpu_cols[1]:
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h3>{total_mem:.1f} GB</h3>
                    <p style="font-size: 14px;">Total VRAM</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with gpu_cols[2]:
                free_mem = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated(0)
                ) / 1e9
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h3>{free_mem:.1f} GB</h3>
                    <p style="font-size: 14px;">Available</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with gpu_cols[3]:
                cuda_version = torch.version.cuda or "N/A"
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h3>CUDA {cuda_version}</h3>
                    <p style="font-size: 14px;">Version</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Memory usage bar
            allocated = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            usage_pct = (allocated / total) * 100

            st.progress(
                usage_pct / 100,
                text=f"VRAM Usage: {allocated:.2f} / {total:.1f} GB ({usage_pct:.1f}%)",
            )

        else:
            st.warning("⚠️ No CUDA GPU detected - predictions will use CPU (much slower)")

    except ImportError:
        st.error("PyTorch not installed")

    st.markdown("---")

    # Predictor Status
    st.markdown("### 🧬 Predictors Status")

    try:
        from protein_design_hub.predictors.registry import PredictorRegistry
        from protein_design_hub.core.config import get_settings

        settings = get_settings()

        pred_cols = st.columns(3)

        predictor_info = {
            "colabfold": {"icon": "🔬", "name": "ColabFold", "desc": "AlphaFold2 + MMseqs2"},
            "chai1": {"icon": "🧪", "name": "Chai-1", "desc": "Diffusion-based predictor"},
            "boltz2": {"icon": "⚡", "name": "Boltz-2", "desc": "Fast diffusion model"},
            "esmfold": {"icon": "🧠", "name": "ESMFold", "desc": "Fast single-sequence folding"},
            "esmfold_api": {"icon": "🌐", "name": "ESMFold API", "desc": "Remote folding (no GPU)"},
            "esm3": {"icon": "🧬", "name": "ESM3", "desc": "Multimodal generation (structure track)"},
        }

        for i, (pred_id, info) in enumerate(predictor_info.items()):
            with pred_cols[i % 3]:
                try:
                    predictor = PredictorRegistry.get(pred_id, settings)
                    status = predictor.get_status()

                    with st.container(border=True):
                        st.markdown(f"#### {info['icon']} {info['name']}")
                        st.caption(info["desc"])

                        if status["installed"]:
                            version = status.get("version", "unknown")
                            _render_badges([("Installed", "ok"), (f"v{version}", "primary")])
                        else:
                            _render_badges([("Not installed", "error")])

                        # Features
                        features = []
                        if status.get("supports_multimer"):
                            features.append("🔗 Multimer")
                        if status.get("supports_templates"):
                            features.append("📄 Templates")
                        if status.get("supports_msa"):
                            features.append("📊 MSA")
                        if status.get("output_format"):
                            features.append(f"🗂️ {status.get('output_format')}")

                        if features:
                            _render_badges([(feat, "info") for feat in features])

                        if status.get("needs_update"):
                            latest = status.get("latest_version")
                            _render_badges([("Update available", "warning")])
                            if latest:
                                st.caption(f"Latest: v{latest}")

                except Exception as e:
                    st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Error loading predictor status: {e}")

    st.markdown("---")

    # Evaluation Tools Status
    st.markdown("### 📊 Evaluation Tools")

    try:
        from protein_design_hub.evaluation.composite import CompositeEvaluator

        metrics_info = CompositeEvaluator.list_all_metrics()

        eval_cols = st.columns(3)

        for i, metric in enumerate(metrics_info):
            with eval_cols[i % 3]:
                if metric["available"]:
                    st.success(f"✓ {metric['name'].upper()}")
                    st.caption(metric["description"])
                else:
                    st.error(f"✗ {metric['name'].upper()}")
                    st.caption(metric.get("requirements", "Not available"))

        # OpenStructure check
        st.markdown("**OpenStructure Integration:**")
        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner

            runner = get_ost_runner()
            if runner.is_available():
                st.success("✓ OpenStructure available (comprehensive metrics enabled)")
            else:
                st.warning("OpenStructure not found - install for advanced metrics")
        except Exception:
            st.warning("OpenStructure integration not available")

    except Exception as e:
        st.error(f"Error loading evaluation tools: {e}")

# ==================== TAB 2: Installations ====================
with tabs[1]:
    section_header("Install & Update Tools", "Manage predictors and evaluation dependencies.", "📦")

    try:
        from protein_design_hub.predictors.registry import PredictorRegistry
        from protein_design_hub.core.config import get_settings

        settings = get_settings()

        # Individual predictors
        st.markdown("### Predictors")

        for pred_id, info in predictor_info.items():
            with st.expander(f"{info['icon']} {info['name']}", expanded=False):
                predictor = PredictorRegistry.get(pred_id, settings)
                is_installed = predictor.installer.is_installed()

                col1, col2 = st.columns([2, 1])

                with col1:
                    if is_installed:
                        version = predictor.installer.get_installed_version()
                        _render_badges([("Installed", "ok"), (f"v{version or 'unknown'}", "primary")])
                        st.markdown(f"**Location:** `{predictor.installer.get_install_path()}`")
                    else:
                        _render_badges([("Not installed", "error")])

                    # Installation requirements
                    st.markdown("**Requirements:**")
                    st.caption(predictor.installer.get_requirements())

                with col2:
                    if is_installed:
                        if st.button("🔄 Update", key=f"update_{pred_id}", use_container_width=True):
                            with st.spinner(f"Updating {info['name']}..."):
                                success = predictor.installer.update()
                                if success:
                                    st.success("Updated!")
                                    st.rerun()
                                else:
                                    st.error("Update failed")

                        if st.button(
                            "🗑️ Uninstall", key=f"uninstall_{pred_id}", use_container_width=True
                        ):
                            st.warning("This will remove the installation. Proceed?")
                            if st.button("Confirm Uninstall", key=f"confirm_{pred_id}"):
                                with st.spinner("Uninstalling..."):
                                    success = predictor.installer.uninstall()
                                    if success:
                                        st.success("Uninstalled!")
                                        st.rerun()
                    else:
                        if st.button(
                            "📥 Install",
                            key=f"install_{pred_id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            with st.spinner(f"Installing {info['name']}..."):
                                success = predictor.installer.install()
                                if success:
                                    st.success("Installed!")
                                    st.rerun()
                                else:
                                    st.error("Installation failed - check logs")

        # Batch operations
        st.markdown("---")
        st.markdown("### Batch Operations")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 Install All Missing", use_container_width=True, type="primary"):
                progress = st.progress(0)
                status = st.empty()

                predictors = PredictorRegistry.list_available()
                for i, name in enumerate(predictors):
                    status.text(f"Checking {name}...")
                    predictor = PredictorRegistry.get(name, settings)
                    if not predictor.installer.is_installed():
                        status.text(f"Installing {name}...")
                        predictor.installer.install()
                    progress.progress((i + 1) / len(predictors))

                status.text("Complete!")
                st.success("All predictors processed!")

        with col2:
            if st.button("🔄 Update All", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()

                predictors = PredictorRegistry.list_available()
                for i, name in enumerate(predictors):
                    status.text(f"Updating {name}...")
                    predictor = PredictorRegistry.get(name, settings)
                    if predictor.installer.is_installed():
                        predictor.installer.update()
                    progress.progress((i + 1) / len(predictors))

                status.text("Complete!")
                st.success("All predictors updated!")

        with col3:
            if st.button("🧪 Verify All", use_container_width=True):
                for name in PredictorRegistry.list_available():
                    predictor = PredictorRegistry.get(name, settings)

                    if not predictor.installer.is_installed():
                        st.warning(f"{name}: Not installed")
                        continue

                    with st.spinner(f"Verifying {name}..."):
                        success, message = predictor.verify_installation()

                        if success:
                            st.success(f"{name}: {message}")
                        else:
                            st.error(f"{name}: {message}")

        # Evaluation tools installation
        st.markdown("---")
        st.markdown("### Evaluation Tools")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**TMalign**")
            st.caption("For TM-score computation")
            if st.button("Install TMalign", key="install_tmalign"):
                st.code("conda install -c bioconda tmalign")

        with col2:
            st.markdown("**OpenStructure**")
            st.caption("For comprehensive metrics (lDDT, DockQ, etc.)")
            if st.button("Install OpenStructure", key="install_ost"):
                st.code("micromamba create -n ost -c conda-forge -c bioconda openstructure")

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback

        st.code(traceback.format_exc())

# ==================== TAB 3: Preferences ====================
with tabs[2]:
    section_header("User Preferences", "Customize how results are displayed and saved.", "🎨")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Display Settings")

        # Theme selection (placeholder - Streamlit theming is limited)
        theme_option = st.selectbox(
            "Theme",
            options=["Light", "Dark", "System"],
            index=0,
            help="Note: Full dark mode requires Streamlit config changes",
        )

        if theme_option == "Dark":
            st.info("To enable dark mode, add to `.streamlit/config.toml`:")
            st.code(
                """[theme]
base="dark"
primaryColor="#667eea"
backgroundColor="#1e1e1e"
secondaryBackgroundColor="#2d2d2d"
textColor="#ffffff"
"""
            )

        # Results display
        st.markdown("### Results Display")
        show_confidence = st.checkbox("Show confidence regions in plots", value=True)
        default_colorscheme = st.selectbox(
            "Default structure coloring",
            options=["pLDDT", "Chain", "Residue type", "B-factor"],
            index=0,
        )
        default_style = st.selectbox(
            "Default structure style", options=["Cartoon", "Stick", "Sphere", "Surface"], index=0
        )

    with col2:
        st.markdown("### Prediction Defaults")

        default_predictor = st.selectbox(
            "Default predictor", options=["ColabFold", "Chai-1", "Boltz-2"], index=0
        )

        default_num_models = st.number_input(
            "Default number of models", min_value=1, max_value=5, value=5
        )

        default_num_recycles = st.number_input(
            "Default number of recycles", min_value=1, max_value=48, value=3
        )

        st.markdown("### Notification Settings")
        notify_on_complete = st.checkbox("Browser notification on completion", value=False)
        play_sound = st.checkbox("Play sound on completion", value=False)

    # Save preferences
    st.markdown("---")

    if st.button("💾 Save Preferences", type="primary"):
        preferences = {
            "theme": theme_option,
            "show_confidence": show_confidence,
            "default_colorscheme": default_colorscheme,
            "default_style": default_style,
            "default_predictor": default_predictor,
            "default_num_models": default_num_models,
            "default_num_recycles": default_num_recycles,
            "notify_on_complete": notify_on_complete,
            "play_sound": play_sound,
        }

        # Save to file
        prefs_path = Path.home() / ".pdhub" / "preferences.json"
        prefs_path.parent.mkdir(parents=True, exist_ok=True)
        prefs_path.write_text(json.dumps(preferences, indent=2))

        st.success(f"Preferences saved to {prefs_path}")

# ==================== TAB 4: Configuration ====================
with tabs[3]:
    section_header("Advanced Configuration", "Power-user settings for output, GPU, and network.", "🔧")

    try:
        from protein_design_hub.core.config import get_settings
        import yaml

        settings = get_settings()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Output Settings")

            output_dir = st.text_input(
                "Base output directory",
                value=str(settings.output.base_dir),
                help="Default directory for prediction outputs",
            )

            save_all = st.checkbox(
                "Save all models",
                value=settings.output.save_all_models,
                help="Save all generated models, not just the best",
            )

            gen_report = st.checkbox(
                "Generate HTML report",
                value=settings.output.generate_report,
                help="Auto-generate HTML summary report",
            )

            st.markdown("### Evaluation Settings")

            eval_metrics = st.multiselect(
                "Default metrics",
                options=["lddt", "tm_score", "rmsd", "qs_score", "lddt_pli"],
                default=settings.evaluation.metrics,
            )

            lddt_radius = st.number_input(
                "lDDT inclusion radius (Å)",
                min_value=5.0,
                max_value=20.0,
                value=float(settings.evaluation.lddt.inclusion_radius),
                help="Distance cutoff for lDDT calculation",
            )

        with col2:
            st.markdown("### GPU Settings")

            gpu_device = st.text_input(
                "GPU device", value=settings.gpu.device, help="e.g., 'cuda:0' or 'cuda:1'"
            )

            mem_fraction = st.slider(
                "Memory fraction",
                min_value=0.5,
                max_value=1.0,
                value=settings.gpu.memory_fraction,
                help="Fraction of GPU memory to use",
            )

            clear_cache = st.checkbox(
                "Clear cache between jobs",
                value=settings.gpu.clear_cache_between_jobs,
                help="Free GPU memory between predictions",
            )

            st.markdown("### Network Settings")

            msa_server = st.text_input(
                "MSA Server URL",
                value="https://api.colabfold.com",
                help="ColabFold MSA server endpoint",
            )

            request_timeout = st.number_input(
                "Request timeout (seconds)", min_value=30, max_value=600, value=120
            )

        # Save configuration
        st.markdown("---")

        col_save, col_reset = st.columns(2)

        with col_save:
            if st.button("💾 Save Configuration", type="primary", use_container_width=True):
                try:
                    # Update settings
                    settings.output.base_dir = Path(output_dir)
                    settings.output.save_all_models = save_all
                    settings.output.generate_report = gen_report
                    settings.gpu.device = gpu_device
                    settings.gpu.memory_fraction = mem_fraction
                    settings.gpu.clear_cache_between_jobs = clear_cache
                    settings.evaluation.metrics = eval_metrics
                    settings.evaluation.lddt.inclusion_radius = lddt_radius

                    # Save to file
                    config_path = Path.home() / ".pdhub" / "config.yaml"
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    settings.to_yaml(config_path)

                    st.success(f"Configuration saved to {config_path}")
                except Exception as e:
                    st.error(f"Error saving: {e}")

        with col_reset:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                config_path = Path.home() / ".pdhub" / "config.yaml"
                if config_path.exists():
                    config_path.unlink()
                    st.success("Configuration reset - restart to apply")
                    st.rerun()

        # View full config
        with st.expander("View Full Configuration"):
            config_dict = settings.model_dump()
            st.code(yaml.dump(config_dict, default_flow_style=False), language="yaml")

        # Export/Import
        st.markdown("---")
        st.markdown("### Export / Import")

        col1, col2 = st.columns(2)

        with col1:
            config_yaml = yaml.dump(settings.model_dump(), default_flow_style=False)
            st.download_button(
                "📤 Export Configuration",
                data=config_yaml,
                file_name="pdhub_config.yaml",
                mime="text/yaml",
                use_container_width=True,
            )

        with col2:
            uploaded_config = st.file_uploader("📥 Import Configuration", type=["yaml", "yml"])
            if uploaded_config:
                try:
                    new_config = yaml.safe_load(uploaded_config.read())
                    st.json(new_config)
                    if st.button("Apply Imported Config"):
                        config_path = Path.home() / ".pdhub" / "config.yaml"
                        config_path.parent.mkdir(parents=True, exist_ok=True)
                        config_path.write_text(yaml.dump(new_config))
                        st.success("Configuration imported - restart to apply")
                except Exception as e:
                    st.error(f"Error loading config: {e}")

    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        import traceback

        st.code(traceback.format_exc())

# ==================== TAB 5: Cache & Data ====================
with tabs[4]:
    section_header("Cache & Data Management", "Inspect caches, clean artifacts, and manage outputs.", "📁")

    # Cache locations
    st.markdown("### Cache Locations")

    cache_dirs = {
        "PDB Cache": Path.home() / ".pdhub" / "cache" / "pdb",
        "UniProt Cache": Path.home() / ".pdhub" / "cache" / "uniprot",
        "AlphaFold Cache": Path.home() / ".pdhub" / "cache" / "alphafold",
        "MSA Cache": Path.home() / ".pdhub" / "cache" / "msa",
        "Model Weights": Path.home() / ".cache" / "torch" / "hub",
        "Job Database": Path.home() / ".pdhub" / "jobs.db",
    }

    for name, path in cache_dirs.items():
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"**{name}**")
                st.caption(str(path))

            with col2:
                if path.exists():
                    if path.is_file():
                        size = path.stat().st_size / 1e6
                        st.caption(f"{size:.1f} MB")
                    else:
                        try:
                            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
                            st.caption(f"{size:.1f} MB")
                        except Exception:
                            st.caption("N/A")
                else:
                    st.caption("Not created")

            with col3:
                if path.exists():
                    if st.button("🗑️ Clear", key=f"clear_{name}", use_container_width=True):
                        try:
                            if path.is_file():
                                path.unlink()
                            else:
                                import shutil

                                shutil.rmtree(path)
                            st.success(f"Cleared {name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    st.markdown("---")

    # Batch cache operations
    st.markdown("### Cache Operations")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧹 Clear All Caches", use_container_width=True):
            import shutil

            cache_root = Path.home() / ".pdhub" / "cache"
            if cache_root.exists():
                shutil.rmtree(cache_root)
                st.success("All caches cleared!")

    with col2:
        if st.button("🗑️ Clear Job History", use_container_width=True):
            db_path = Path.home() / ".pdhub" / "jobs.db"
            if db_path.exists():
                db_path.unlink()
                st.success("Job history cleared!")

    with col3:
        total_size = 0
        for name, path in cache_dirs.items():
            if path.exists():
                if path.is_file():
                    total_size += path.stat().st_size
                else:
                    try:
                        total_size += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    except:
                        pass

        st.metric("Total Cache Size", f"{total_size / 1e9:.2f} GB")

    # Output data management
    st.markdown("---")
    st.markdown("### Output Data")

    try:
        from protein_design_hub.core.config import get_settings

        settings = get_settings()

        output_path = settings.output.base_dir
        if output_path.exists():
            jobs = list(output_path.iterdir())
            st.info(f"Output directory: `{output_path}` ({len(jobs)} jobs)")

            if jobs:
                with st.expander("View Jobs"):
                    for job_dir in sorted(jobs, reverse=True)[:20]:
                        if job_dir.is_dir():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.text(job_dir.name)
                            with col2:
                                if st.button("🗑️", key=f"del_job_{job_dir.name}"):
                                    import shutil

                                    shutil.rmtree(job_dir)
                                    st.rerun()
        else:
            st.info(f"Output directory not yet created: `{output_path}`")

    except Exception as e:
        st.warning(f"Could not load output settings: {e}")

# ==================== TAB 6: Logs ====================
with tabs[5]:
    section_header("Application Logs", "Inspect runtime logs and export diagnostics.", "📋")

    log_files = {
        "Application Log": Path.home() / ".pdhub" / "pdhub.log",
        "Prediction Log": Path.home() / ".pdhub" / "predictions.log",
        "Error Log": Path.home() / ".pdhub" / "errors.log",
    }

    for name, log_path in log_files.items():
        with st.expander(name, expanded=False):
            if log_path.exists():
                content = log_path.read_text()
                lines = content.split("\n")

                # Show last N lines
                num_lines = st.slider(f"Show last N lines", 10, 500, 100, key=f"lines_{name}")
                st.code("\n".join(lines[-num_lines:]), language="log")

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        f"📥 Download {name}",
                        data=content,
                        file_name=log_path.name,
                        mime="text/plain",
                    )
                with col2:
                    if st.button(f"🗑️ Clear {name}", key=f"clear_log_{name}"):
                        log_path.write_text("")
                        st.success("Log cleared")
                        st.rerun()
            else:
                st.info(f"Log file not created yet: `{log_path}`")

    # System info
    st.markdown("---")
    st.markdown("### System Information")

    import platform
    import sys

    sys_info = {
        "Python Version": sys.version.split()[0],
        "Platform": platform.platform(),
        "Processor": platform.processor() or "N/A",
    }

    try:
        import torch

        sys_info["PyTorch Version"] = torch.__version__
        sys_info["CUDA Available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            sys_info["CUDA Version"] = torch.version.cuda or "N/A"
    except ImportError:
        sys_info["PyTorch"] = "Not installed"

    for key, value in sys_info.items():
        st.text(f"{key}: {value}")

    # Copy system info
    sys_info_text = "\n".join(f"{k}: {v}" for k, v in sys_info.items())
    st.download_button(
        "📋 Copy System Info", data=sys_info_text, file_name="system_info.txt", mime="text/plain"
    )
