"""Settings page for Streamlit app."""

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Settings - Protein Design Hub", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings")
st.markdown("Configure Protein Design Hub and manage installations")

# Tabs
tab1, tab2, tab3 = st.tabs(["Installation Status", "Install/Update", "Configuration"])

# Tab 1: Installation Status
with tab1:
    st.subheader("🔧 Installation Status")

    try:
        from protein_design_hub.predictors.registry import PredictorRegistry
        from protein_design_hub.core.config import get_settings

        settings = get_settings()

        # Refresh button
        if st.button("🔄 Refresh Status"):
            st.rerun()

        # Predictor status
        col1, col2, col3 = st.columns(3)

        for i, name in enumerate(["colabfold", "chai1", "boltz2"]):
            col = [col1, col2, col3][i]

            with col:
                try:
                    predictor = PredictorRegistry.get(name, settings)
                    status = predictor.get_status()

                    st.markdown(f"### {name.upper()}")

                    if status["installed"]:
                        st.success("✓ Installed")
                        st.markdown(f"**Version:** {status.get('version', 'unknown')}")

                        if status.get("latest_version"):
                            if status.get("needs_update"):
                                st.warning(f"Update available: {status['latest_version']}")
                            else:
                                st.info(f"Latest: {status['latest_version']}")
                    else:
                        st.error("✗ Not Installed")

                    # Features
                    st.markdown("**Features:**")
                    features = []
                    if status.get("supports_multimer"):
                        features.append("Multimer")
                    if status.get("supports_templates"):
                        features.append("Templates")
                    if status.get("supports_msa"):
                        features.append("MSA")
                    st.caption(", ".join(features) if features else "Basic prediction")

                except Exception as e:
                    st.error(f"Error: {e}")

        # GPU Status
        st.markdown("---")
        st.subheader("🖥️ GPU Status")

        try:
            import torch
            if torch.cuda.is_available():
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Device", torch.cuda.get_device_name(0))

                with col2:
                    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    st.metric("Total Memory", f"{total_mem:.1f} GB")

                with col3:
                    allocated = torch.cuda.memory_allocated(0) / 1e9
                    st.metric("Allocated", f"{allocated:.2f} GB")

                st.success("GPU is available for predictions")
            else:
                st.warning("No GPU available - predictions will be slow")

        except ImportError:
            st.error("PyTorch not installed")

        # Evaluation tools
        st.markdown("---")
        st.subheader("📊 Evaluation Tools")

        from protein_design_hub.evaluation.composite import CompositeEvaluator

        metrics_info = CompositeEvaluator.list_all_metrics()

        for metric in metrics_info:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{metric['name'].upper()}** - {metric['description']}")
            with col2:
                if metric["available"]:
                    st.success("Available")
                else:
                    st.error("Missing")
                    st.caption(metric.get("requirements", ""))

    except Exception as e:
        st.error(f"Error loading status: {e}")

# Tab 2: Install/Update
with tab2:
    st.subheader("📦 Install & Update Tools")

    try:
        from protein_design_hub.predictors.registry import PredictorRegistry
        from protein_design_hub.core.config import get_settings

        settings = get_settings()

        # Install section
        st.markdown("### Install Predictors")

        col1, col2, col3 = st.columns(3)

        for i, name in enumerate(["colabfold", "chai1", "boltz2"]):
            col = [col1, col2, col3][i]

            with col:
                predictor = PredictorRegistry.get(name, settings)
                is_installed = predictor.installer.is_installed()

                st.markdown(f"**{name.upper()}**")

                if is_installed:
                    st.info(f"Installed v{predictor.installer.get_installed_version() or 'unknown'}")
                    if st.button(f"Update {name}", key=f"update_{name}"):
                        with st.spinner(f"Updating {name}..."):
                            success = predictor.installer.update()
                            if success:
                                st.success("Updated!")
                            else:
                                st.error("Update failed")
                else:
                    st.warning("Not installed")
                    if st.button(f"Install {name}", key=f"install_{name}"):
                        with st.spinner(f"Installing {name}..."):
                            success = predictor.installer.install()
                            if success:
                                st.success("Installed!")
                                st.rerun()
                            else:
                                st.error("Installation failed")

        # Batch operations
        st.markdown("---")
        st.markdown("### Batch Operations")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Install All", use_container_width=True):
                progress = st.progress(0)
                predictors = PredictorRegistry.list_available()

                for i, name in enumerate(predictors):
                    st.write(f"Installing {name}...")
                    predictor = PredictorRegistry.get(name, settings)
                    if not predictor.installer.is_installed():
                        predictor.installer.install()
                    progress.progress((i + 1) / len(predictors))

                st.success("All predictors installed!")

        with col2:
            if st.button("🔄 Update All", use_container_width=True):
                progress = st.progress(0)
                predictors = PredictorRegistry.list_available()

                for i, name in enumerate(predictors):
                    st.write(f"Updating {name}...")
                    predictor = PredictorRegistry.get(name, settings)
                    if predictor.installer.is_installed():
                        predictor.installer.update()
                    progress.progress((i + 1) / len(predictors))

                st.success("All predictors updated!")

        # Verification
        st.markdown("---")
        st.markdown("### Verify Installation")

        if st.button("🧪 Run Verification Tests"):
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

    except Exception as e:
        st.error(f"Error: {e}")

# Tab 3: Configuration
with tab3:
    st.subheader("⚙️ Configuration")

    try:
        from protein_design_hub.core.config import get_settings
        import yaml

        settings = get_settings()

        # Display current config
        st.markdown("### Current Configuration")

        config_dict = settings.model_dump()
        config_yaml = yaml.dump(config_dict, default_flow_style=False)

        st.code(config_yaml, language="yaml")

        # Edit configuration
        st.markdown("---")
        st.markdown("### Edit Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Output Settings**")
            output_dir = st.text_input("Output directory", value=str(settings.output.base_dir))
            save_all = st.checkbox("Save all models", value=settings.output.save_all_models)
            gen_report = st.checkbox("Generate HTML report", value=settings.output.generate_report)

        with col2:
            st.markdown("**GPU Settings**")
            gpu_device = st.text_input("GPU device", value=settings.gpu.device)
            clear_cache = st.checkbox("Clear cache between jobs", value=settings.gpu.clear_cache_between_jobs)
            mem_fraction = st.slider("Memory fraction", 0.5, 1.0, settings.gpu.memory_fraction)

        if st.button("💾 Save Configuration"):
            try:
                settings.output.base_dir = Path(output_dir)
                settings.output.save_all_models = save_all
                settings.output.generate_report = gen_report
                settings.gpu.device = gpu_device
                settings.gpu.clear_cache_between_jobs = clear_cache
                settings.gpu.memory_fraction = mem_fraction

                # Save to file
                config_path = Path.home() / ".protein_design_hub" / "config.yaml"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                settings.to_yaml(config_path)

                st.success(f"Configuration saved to {config_path}")
            except Exception as e:
                st.error(f"Error saving configuration: {e}")

        # Export/Import
        st.markdown("---")
        st.markdown("### Export/Import")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "📤 Export Configuration",
                data=config_yaml,
                file_name="pdhub_config.yaml",
                mime="text/yaml",
            )

        with col2:
            uploaded_config = st.file_uploader("📥 Import Configuration", type=["yaml", "yml"])
            if uploaded_config:
                try:
                    new_config = yaml.safe_load(uploaded_config.read())
                    st.json(new_config)
                    if st.button("Apply Imported Config"):
                        st.info("Configuration imported (restart required)")
                except Exception as e:
                    st.error(f"Error loading config: {e}")

    except Exception as e:
        st.error(f"Error loading configuration: {e}")
