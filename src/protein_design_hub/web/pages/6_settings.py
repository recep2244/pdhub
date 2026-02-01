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
    page_header,
    section_header,
    status_badge,
    metric_card,
    info_box,
)

inject_base_css()
sidebar_nav(current="Settings")
sidebar_system_status()

# Page header
page_header(
    "Settings & Configuration",
    "Manage predictors, configure preferences, and monitor system health",
    "⚙️"
)

# Utility helpers
def _render_badges(items: Iterable[Tuple[str, str]]) -> None:
    badges = " ".join(status_badge(label, status) for label, status in items if label)
    if badges:
        st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;">{badges}</div>', unsafe_allow_html=True)

# Predictor info dictionary (shared across tabs)
predictor_info = {
    "colabfold": {"icon": "🔬", "name": "ColabFold", "desc": "AlphaFold2 + MMseqs2"},
    "chai1": {"icon": "🧪", "name": "Chai-1", "desc": "Diffusion-based predictor"},
    "boltz2": {"icon": "⚡", "name": "Boltz-2", "desc": "Fast diffusion model"},
    "esmfold": {"icon": "🧠", "name": "ESMFold", "desc": "Fast single-sequence folding"},
    "esmfold_api": {"icon": "🌐", "name": "ESMFold API", "desc": "Remote folding (no GPU)"},
    "esm3": {"icon": "🧬", "name": "ESM3", "desc": "Multimodal generation"},
}

# Tabs
tabs = st.tabs([
    "📊 System Status",
    "📦 Installations",
    "🎨 Preferences",
    "🔧 Configuration",
    "📁 Cache & Data",
    "📋 Logs",
])

# ==================== TAB 1: System Status ====================
with tabs[0]:
    section_header("System Overview", "Hardware and tool health monitoring", "📊")

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh", key="refresh_status", use_container_width=True):
            st.rerun()

    # GPU Status - Using robust detection
    st.markdown("### GPU Status")

    from protein_design_hub.web.ui import detect_gpu
    gpu_info = detect_gpu()

    if gpu_info["available"]:
        gpu_name = gpu_info["name"]
        total_mem = gpu_info["memory_total_gb"]
        free_mem = gpu_info["memory_free_gb"]

        # Try to get CUDA version from torch
        cuda_version = "N/A"
        try:
            import torch
            if torch.version.cuda:
                cuda_version = torch.version.cuda
        except Exception:
            pass

        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        with col_g1:
            short_name = gpu_name.split()[-1] if gpu_name else "GPU"
            metric_card(short_name, "GPU Device", "gradient", "🎮")
        with col_g2:
            metric_card(f"{total_mem:.1f} GB", "Total VRAM", "info", "💾")
        with col_g3:
            metric_card(f"{free_mem:.1f} GB", "Available", "success", "✅")
        with col_g4:
            if gpu_info["driver_version"]:
                metric_card(gpu_info["driver_version"][:10], "Driver", "info", "🔧")
            else:
                metric_card(cuda_version, "CUDA Version", "info", "🔧")

        # Memory usage bar
        used_mem = total_mem - free_mem
        usage_pct = (used_mem / total_mem) * 100 if total_mem > 0 else 0
        st.progress(usage_pct / 100, text=f"VRAM Usage: {used_mem:.2f} / {total_mem:.1f} GB ({usage_pct:.1f}%)")

        # Show detection source
        if gpu_info["source"] == "nvidia-smi":
            st.caption("ℹ️ GPU detected via nvidia-smi (PyTorch CUDA init may need restart)")
    else:
        info_box("No CUDA GPU detected. Predictions will use CPU (much slower).", "warning", "CPU Mode")

    st.markdown("---")

    # Predictor Status
    st.markdown("### Predictors")

    try:
        from protein_design_hub.predictors.registry import PredictorRegistry
        from protein_design_hub.core.config import get_settings

        settings = get_settings()
        pred_cols = st.columns(3)

        for i, (pred_id, info) in enumerate(predictor_info.items()):
            with pred_cols[i % 3]:
                try:
                    predictor = PredictorRegistry.get(pred_id, settings)
                    status = predictor.get_status()

                    with st.container(border=True):
                        st.markdown(f"**{info['icon']} {info['name']}**")
                        st.caption(info["desc"])

                        if status["installed"]:
                            version = status.get("version", "unknown")
                            _render_badges([("Installed", "ok"), (f"v{version}", "primary")])
                        else:
                            _render_badges([("Not installed", "error")])

                        # Features
                        features = []
                        if status.get("supports_multimer"):
                            features.append(("Multimer", "info"))
                        if status.get("supports_templates"):
                            features.append(("Templates", "info"))
                        if status.get("supports_msa"):
                            features.append(("MSA", "info"))

                        if features:
                            _render_badges(features)

                except Exception as e:
                    with st.container(border=True):
                        st.markdown(f"**{info['icon']} {info['name']}**")
                        st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Error loading predictor status: {e}")

    st.markdown("---")

    # Evaluation Tools
    st.markdown("### Evaluation Tools")

    try:
        from protein_design_hub.evaluation.composite import CompositeEvaluator

        metrics_info = CompositeEvaluator.list_all_metrics()
        eval_cols = st.columns(3)

        for i, metric in enumerate(metrics_info):
            with eval_cols[i % 3]:
                if metric["available"]:
                    st.success(f"✓ {metric['name'].upper()}")
                    st.caption(metric["description"][:50] + "..." if len(metric["description"]) > 50 else metric["description"])
                else:
                    st.error(f"✗ {metric['name'].upper()}")

        # OpenStructure check
        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner
            runner = get_ost_runner()
            if runner.is_available():
                info_box("OpenStructure available - comprehensive metrics enabled", "success", "OpenStructure")
            else:
                info_box("Install OpenStructure for advanced metrics", "warning", "OpenStructure")
        except Exception:
            pass

    except Exception as e:
        st.warning(f"Could not load evaluation tools: {e}")

# ==================== TAB 2: Installations ====================
with tabs[1]:
    section_header("Install & Update", "Manage predictor installations", "📦")

    try:
        from protein_design_hub.predictors.registry import PredictorRegistry
        from protein_design_hub.core.config import get_settings

        settings = get_settings()

        # Individual predictors
        st.markdown("### Predictors")

        for pred_id, info in predictor_info.items():
            with st.expander(f"{info['icon']} {info['name']}", expanded=False):
                try:
                    predictor = PredictorRegistry.get(pred_id, settings)
                    is_installed = predictor.installer.is_installed()

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        if is_installed:
                            version = predictor.installer.get_installed_version()
                            _render_badges([("Installed", "ok"), (f"v{version or 'unknown'}", "primary")])
                            st.markdown(f"**Location:** `{predictor.installer.get_install_path()}`")
                        else:
                            _render_badges([("Not installed", "error")])

                        st.caption(f"**Requirements:** {predictor.installer.get_requirements()}")

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

                            if st.button("🗑️ Uninstall", key=f"uninstall_{pred_id}", use_container_width=True, type="secondary"):
                                st.warning("Confirm uninstall?")
                                if st.button("Confirm", key=f"confirm_{pred_id}"):
                                    predictor.installer.uninstall()
                                    st.rerun()
                        else:
                            if st.button("📥 Install", key=f"install_{pred_id}", type="primary", use_container_width=True):
                                with st.spinner(f"Installing {info['name']}..."):
                                    success = predictor.installer.install()
                                    if success:
                                        st.success("Installed!")
                                        st.rerun()
                                    else:
                                        st.error("Installation failed")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")

        # Batch operations
        st.markdown("### Batch Operations")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 Install All Missing", use_container_width=True, type="primary"):
                progress = st.progress(0)
                status_text = st.empty()

                predictors = PredictorRegistry.list_available()
                for i, name in enumerate(predictors):
                    status_text.text(f"Checking {name}...")
                    predictor = PredictorRegistry.get(name, settings)
                    if not predictor.installer.is_installed():
                        status_text.text(f"Installing {name}...")
                        predictor.installer.install()
                    progress.progress((i + 1) / len(predictors))

                status_text.text("Complete!")
                st.success("All predictors processed!")

        with col2:
            if st.button("🔄 Update All", use_container_width=True):
                progress = st.progress(0)
                status_text = st.empty()

                predictors = PredictorRegistry.list_available()
                for i, name in enumerate(predictors):
                    status_text.text(f"Updating {name}...")
                    predictor = PredictorRegistry.get(name, settings)
                    if predictor.installer.is_installed():
                        predictor.installer.update()
                    progress.progress((i + 1) / len(predictors))

                status_text.text("Complete!")
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

        st.markdown("---")

        # Evaluation tools
        st.markdown("### Evaluation Tools")

        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container(border=True):
                st.markdown("**TMalign**")
                st.caption("For TM-score computation")
                if st.button("Show Install Command", key="show_tmalign"):
                    st.code("conda install -c bioconda tmalign")

        with col2:
            with st.container(border=True):
                st.markdown("**OpenStructure**")
                st.caption("Comprehensive metrics (lDDT, DockQ)")
                if st.button("Show Install Command", key="show_ost"):
                    st.code("micromamba create -n ost -c conda-forge -c bioconda openstructure")

        with col3:
            with st.container(border=True):
                st.markdown("**Voronota**")
                st.caption("CAD-score / VoroMQA")
                if st.button("Show Install Command", key="show_voronota"):
                    st.code("conda install -c bioconda voronota")

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        with st.expander("Details"):
            st.code(traceback.format_exc())

# ==================== TAB 3: Preferences ====================
with tabs[2]:
    section_header("User Preferences", "Customize display and defaults", "🎨")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### Display Settings")

            default_colorscheme = st.selectbox(
                "Structure coloring",
                options=["pLDDT", "Chain", "Residue type", "B-factor"],
                index=0,
            )

            default_style = st.selectbox(
                "Structure style",
                options=["Cartoon", "Stick", "Sphere", "Surface"],
                index=0
            )

            show_confidence = st.checkbox("Show confidence regions in plots", value=True)

    with col2:
        with st.container(border=True):
            st.markdown("### Prediction Defaults")

            default_predictor = st.selectbox(
                "Default predictor",
                options=["ColabFold", "Chai-1", "Boltz-2", "ESMFold"],
                index=0
            )

            default_num_models = st.number_input(
                "Number of models",
                min_value=1, max_value=5, value=5
            )

            default_num_recycles = st.number_input(
                "Number of recycles",
                min_value=1, max_value=48, value=3
            )

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        with st.container(border=True):
            st.markdown("### Notifications")
            notify_on_complete = st.checkbox("Browser notification on completion", value=False)
            play_sound = st.checkbox("Play sound on completion", value=False)

    with col4:
        pass  # Reserved for future settings

    st.markdown("---")

    if st.button("💾 Save Preferences", type="primary", use_container_width=True):
        preferences = {
            "show_confidence": show_confidence,
            "default_colorscheme": default_colorscheme,
            "default_style": default_style,
            "default_predictor": default_predictor,
            "default_num_models": default_num_models,
            "default_num_recycles": default_num_recycles,
            "notify_on_complete": notify_on_complete,
            "play_sound": play_sound,
        }

        prefs_path = Path.home() / ".pdhub" / "preferences.json"
        prefs_path.parent.mkdir(parents=True, exist_ok=True)
        prefs_path.write_text(json.dumps(preferences, indent=2))

        st.success(f"Preferences saved to {prefs_path}")

# ==================== TAB 4: Configuration ====================
with tabs[3]:
    section_header("Advanced Configuration", "Output, GPU, and network settings", "🔧")

    try:
        from protein_design_hub.core.config import get_settings
        import yaml

        settings = get_settings()

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown("### Output Settings")

                output_dir = st.text_input(
                    "Base output directory",
                    value=str(settings.output.base_dir),
                )

                save_all = st.checkbox(
                    "Save all models",
                    value=settings.output.save_all_models,
                )

                gen_report = st.checkbox(
                    "Generate HTML report",
                    value=settings.output.generate_report,
                )

            with st.container(border=True):
                st.markdown("### Evaluation Metrics")

                from protein_design_hub.evaluation.composite import CompositeEvaluator

                available_metrics = [m["name"] for m in CompositeEvaluator.list_all_metrics()]
                eval_metrics = st.multiselect(
                    "Default metrics",
                    options=available_metrics,
                    default=[m for m in settings.evaluation.metrics if m in available_metrics],
                )

                lddt_radius = st.number_input(
                    "lDDT inclusion radius (Å)",
                    min_value=5.0, max_value=20.0,
                    value=float(settings.evaluation.lddt.inclusion_radius),
                )

        with col2:
            with st.container(border=True):
                st.markdown("### GPU Settings")

                gpu_device = st.text_input(
                    "GPU device",
                    value=settings.gpu.device,
                    help="e.g., 'cuda:0' or 'cuda:1'"
                )

                mem_fraction = st.slider(
                    "Memory fraction",
                    min_value=0.5, max_value=1.0,
                    value=settings.gpu.memory_fraction,
                )

                clear_cache = st.checkbox(
                    "Clear cache between jobs",
                    value=settings.gpu.clear_cache_between_jobs,
                )

            with st.container(border=True):
                st.markdown("### Network Settings")

                msa_server = st.text_input(
                    "MSA Server URL",
                    value="https://api.colabfold.com",
                )

                request_timeout = st.number_input(
                    "Request timeout (seconds)",
                    min_value=30, max_value=600, value=120
                )

        st.markdown("---")

        col_save, col_reset = st.columns(2)

        with col_save:
            if st.button("💾 Save Configuration", type="primary", use_container_width=True):
                try:
                    settings.output.base_dir = Path(output_dir)
                    settings.output.save_all_models = save_all
                    settings.output.generate_report = gen_report
                    settings.gpu.device = gpu_device
                    settings.gpu.memory_fraction = mem_fraction
                    settings.gpu.clear_cache_between_jobs = clear_cache
                    settings.evaluation.metrics = eval_metrics
                    settings.evaluation.lddt.inclusion_radius = lddt_radius

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

        # View/Export config
        with st.expander("📋 View Full Configuration"):
            config_dict = settings.model_dump()
            st.code(yaml.dump(config_dict, default_flow_style=False), language="yaml")

        st.markdown("---")

        col_exp, col_imp = st.columns(2)

        with col_exp:
            config_yaml = yaml.dump(settings.model_dump(), default_flow_style=False)
            st.download_button(
                "📤 Export Configuration",
                data=config_yaml,
                file_name="pdhub_config.yaml",
                mime="text/yaml",
                use_container_width=True,
            )

        with col_imp:
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
        with st.expander("Details"):
            st.code(traceback.format_exc())

# ==================== TAB 5: Cache & Data ====================
with tabs[4]:
    section_header("Cache & Data Management", "Manage storage and clean up", "📁")

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
                st.caption(str(path)[:60] + "..." if len(str(path)) > 60 else str(path))

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

    # Batch operations
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
                            col1, col2 = st.columns([4, 1])
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
    section_header("Application Logs", "View and export logs", "📋")

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

                num_lines = st.slider(f"Show last N lines", 10, 500, 100, key=f"lines_{name}")
                st.code("\n".join(lines[-num_lines:]), language="log")

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        f"📥 Download",
                        data=content,
                        file_name=log_path.name,
                        mime="text/plain",
                        use_container_width=True,
                    )
                with col2:
                    if st.button(f"🗑️ Clear", key=f"clear_log_{name}", use_container_width=True):
                        log_path.write_text("")
                        st.success("Log cleared")
                        st.rerun()
            else:
                st.info(f"Log file not created yet")

    st.markdown("---")

    # System info
    st.markdown("### System Information")

    import platform
    import sys

    # Use robust GPU detection
    gpu_sys_info = detect_gpu()

    sys_info = {
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
        "Processor": platform.processor() or "N/A",
    }

    try:
        import torch
        sys_info["PyTorch"] = torch.__version__
    except ImportError:
        sys_info["PyTorch"] = "Not installed"

    # Add GPU info using robust detection
    sys_info["GPU Available"] = str(gpu_sys_info["available"])
    if gpu_sys_info["available"]:
        sys_info["GPU Name"] = gpu_sys_info["name"]
        sys_info["GPU Memory"] = f"{gpu_sys_info['memory_total_gb']:.1f} GB"
        if gpu_sys_info["driver_version"]:
            sys_info["Driver Version"] = gpu_sys_info["driver_version"]
        sys_info["Detection"] = gpu_sys_info["source"]

    col1, col2 = st.columns(2)

    with col1:
        for key, value in list(sys_info.items())[:len(sys_info)//2 + 1]:
            st.text(f"{key}: {value}")

    with col2:
        for key, value in list(sys_info.items())[len(sys_info)//2 + 1:]:
            st.text(f"{key}: {value}")

    sys_info_text = "\n".join(f"{k}: {v}" for k, v in sys_info.items())
    st.download_button(
        "📋 Copy System Info",
        data=sys_info_text,
        file_name="system_info.txt",
        mime="text/plain",
        use_container_width=True,
    )
