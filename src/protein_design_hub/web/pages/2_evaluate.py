"""Comprehensive Evaluation page with all OpenStructure metrics."""

import streamlit as st
from pathlib import Path
import tempfile
import json

st.set_page_config(page_title="Evaluate - Protein Design Hub", page_icon="📊", layout="wide")

from protein_design_hub.web.ui import (
    get_selected_model,
    inject_base_css,
    list_output_structures,
    page_header,
    set_selected_model,
    sidebar_nav,
    sidebar_system_status,
)

inject_base_css()

# Page header
page_header(
    "Structure Evaluation",
    "Comprehensive evaluation with 18+ metrics including lDDT, TM-score, DockQ, and energy calculations",
    "📊"
)

sidebar_nav(current="Evaluate")
sidebar_system_status()

# Custom CSS
st.markdown(
    """
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    margin: 5px 0;
}
.metric-card-success {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}
.metric-card-warning {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}
.metric-value {
    font-size: 32px;
    font-weight: bold;
}
.metric-label {
    font-size: 14px;
    opacity: 0.9;
}
.section-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    margin: 20px 0 10px 0;
}
.quality-bar {
    display: flex;
    height: 30px;
    border-radius: 8px;
    overflow: hidden;
    margin: 10px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📊 Comprehensive Structure Evaluation")
st.markdown(
    "Evaluate predicted structures using OpenStructure metrics at global, per-residue, per-chain, and interface levels"
)

# Sidebar
st.sidebar.header("⚙️ Evaluation Settings")

# Quick evaluation (design-focused, reference optional)
st.sidebar.subheader("Quick Evaluation (Recommended)")
try:
    from protein_design_hub.evaluation.composite import CompositeEvaluator

    _all = CompositeEvaluator.list_all_metrics()
    _metric_names = [m["name"] for m in _all]
    quick_metrics = st.sidebar.multiselect(
        "Metrics",
        options=_metric_names,
        default=["clash_score", "contact_energy", "sasa", "interface_bsa", "salt_bridges"],
        help="Reference-free metrics are great for design ranking. Reference-based metrics need a reference structure.",
    )
except Exception:
    quick_metrics = ["clash_score", "contact_energy", "sasa"]

# Metric selection - Global
st.sidebar.subheader("Global Metrics")
compute_lddt = st.sidebar.checkbox("lDDT (Local Distance)", value=True)
compute_bb_lddt = st.sidebar.checkbox("BB-lDDT (Backbone only)", value=False)
compute_rmsd = st.sidebar.checkbox("RMSD & GDT (Rigid Scores)", value=True)
compute_tm = st.sidebar.checkbox("TM-score", value=True)
compute_cad = st.sidebar.checkbox("CAD-score", value=False, help="Requires voronota")

# Interface metrics
st.sidebar.subheader("Interface Metrics")
compute_qs = st.sidebar.checkbox("QS-score (Quaternary)", value=True)
compute_dockq = st.sidebar.checkbox(
    "DockQ (Interface Quality)", value=True, help="fnat, irmsd, lrmsd"
)
compute_ilddt = st.sidebar.checkbox("iLDDT (Inter-chain lDDT)", value=False)
compute_ics = st.sidebar.checkbox(
    "ICS (Contact Similarity)", value=True, help="Precision/Recall/F1"
)
compute_ips = st.sidebar.checkbox("IPS (Patch Similarity)", value=True, help="Jaccard coefficient")
compute_patch = st.sidebar.checkbox(
    "Patch Scores (CASP15)", value=True, help="Local interface quality"
)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    st.markdown("**lDDT Settings**")
    lddt_radius = st.number_input("Inclusion radius (Å)", value=15.0, min_value=5.0, max_value=30.0)
    lddt_seq_sep = st.number_input("Sequence separation", value=0, min_value=0, max_value=10)

    st.markdown("**RMSD Settings**")
    rmsd_atoms = st.selectbox("Atom selection", options=["CA", "backbone", "heavy", "all"], index=0)

    st.markdown("**DockQ Settings**")
    dockq_capri_peptide = st.checkbox(
        "CAPRI Peptide Mode", value=False, help="For small peptide docking"
    )

    st.markdown("**Chain Mapping**")
    chem_seqid_thresh = st.slider("Chain grouping threshold (%)", 50, 100, 95)
    map_seqid_thresh = st.slider("Mapping threshold (%)", 30, 100, 70)

# Evaluation mode
eval_mode = st.sidebar.radio(
    "Evaluation Mode", ["Comprehensive (All Levels)", "Quick (Global Only)"], index=0
)

# Main content
col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.markdown("### 📁 Model Structure")
    st.caption("Upload a structure, or select one from recent outputs.")
    try:
        from protein_design_hub.core.config import get_settings

        _settings = get_settings()
        recent = list_output_structures(Path(_settings.output.base_dir))
    except Exception:
        recent = []

    chosen = None
    sel = get_selected_model()
    if sel is not None and sel.exists():
        chosen = sel
        st.success("Using selected model from Jobs")
        st.code(str(chosen))
        if st.button("Clear selected model", key="clear_sel_model"):
            set_selected_model(None)
            st.rerun()

    if recent:
        default_index = 0
        if chosen is not None and chosen in recent:
            default_index = 1 + recent.index(chosen)
        chosen = st.selectbox(
            "Use recent output (optional)",
            options=[None] + recent,
            format_func=lambda p: "—" if p is None else str(p),
            index=default_index,
        )
    model_file = st.file_uploader(
        "Upload model structure",
        type=["pdb", "cif", "mmcif"],
        key="model",
        help="Structure to evaluate",
    )
    if model_file:
        st.success(f"✅ {model_file.name}")
    elif chosen is not None:
        st.info(f"Using: `{chosen}`")

with col_upload2:
    st.markdown("### 📁 Reference Structure")
    reference_file = st.file_uploader(
        "Upload reference structure",
        type=["pdb", "cif", "mmcif"],
        key="reference",
        help="Ground truth structure for comparison",
    )
    if reference_file:
        st.success(f"✅ {reference_file.name}")
    else:
        st.warning("⚠️ Reference required for lDDT/TM/QS/RMSD; design metrics can run without it.")

# Run evaluation
st.markdown("---")

if st.button("⚡ Run Quick Evaluation", type="primary", use_container_width=True):
    if not model_file and chosen is None:
        st.error("Please upload a model structure (or pick a recent one).")
    else:
        try:
            # Resolve model path
            if chosen is not None and model_file is None:
                model_path = Path(chosen)
            else:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(model_file.name).suffix, delete=False
                ) as tmp:
                    tmp.write(model_file.read())
                    model_path = Path(tmp.name)

            reference_path = None
            if reference_file is not None:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(reference_file.name).suffix, delete=False
                ) as tmp:
                    tmp.write(reference_file.read())
                    reference_path = Path(tmp.name)

            # Filter metrics if reference missing
            from protein_design_hub.evaluation.composite import CompositeEvaluator
            from protein_design_hub.core.config import get_settings

            settings = get_settings()
            evaluator = CompositeEvaluator(metrics=quick_metrics, settings=settings)

            if reference_path is None:
                available = CompositeEvaluator.list_all_metrics()
                ref_required = {m["name"]: m["requires_reference"] for m in available}
                filtered = [m for m in quick_metrics if not ref_required.get(m, False)]
                if filtered != quick_metrics:
                    st.warning(
                        "Reference not provided; skipping reference-required metrics: "
                        + ", ".join(sorted(set(quick_metrics) - set(filtered)))
                    )
                evaluator = CompositeEvaluator(metrics=filtered, settings=settings)

            with st.spinner("Computing metrics..."):
                result = evaluator.evaluate(model_path, reference_path)

            st.success("✅ Quick evaluation complete")
            st.session_state.quick_eval = result.to_dict()

            st.markdown("### Results")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if result.clash_score is not None:
                    st.metric("Clash score", f"{result.clash_score:.2f}")
                if result.sasa_total is not None:
                    st.metric("SASA", f"{result.sasa_total:.1f} Å²")
            with col_b:
                if result.contact_energy is not None:
                    st.metric("Contact energy", f"{result.contact_energy:.3f}")
                if result.contact_energy_per_residue is not None:
                    st.metric("Contact energy / res", f"{result.contact_energy_per_residue:.3f}")
            with col_c:
                if result.tm_score is not None:
                    st.metric("TM-score", f"{result.tm_score:.4f}")
                if result.rmsd is not None:
                    st.metric("RMSD", f"{result.rmsd:.2f} Å")

            st.download_button(
                "📥 Download JSON",
                data=json.dumps(result.to_dict(), indent=2),
                file_name="evaluation.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            import traceback

            with st.expander("Error details"):
                st.code(traceback.format_exc())

if st.button("🚀 Run Comprehensive Evaluation", type="primary", use_container_width=True):
    if not model_file:
        st.error("Please upload a model structure")
    elif not reference_file:
        st.error("Please upload a reference structure for comparison")
    else:
        try:
            # Save files temporarily
            with tempfile.NamedTemporaryFile(
                suffix=Path(model_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(model_file.read())
                model_path = Path(tmp.name)

            with tempfile.NamedTemporaryFile(
                suffix=Path(reference_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(reference_file.read())
                reference_path = Path(tmp.name)

            with st.spinner("🔬 Computing all metrics... This may take a moment."):
                # Use comprehensive evaluation
                from protein_design_hub.evaluation.composite import CompositeEvaluator
                from protein_design_hub.core.config import get_settings

                settings = get_settings()
                settings.evaluation.lddt.inclusion_radius = lddt_radius
                settings.evaluation.lddt.sequence_separation = lddt_seq_sep

                evaluator = CompositeEvaluator(settings=settings)

                if eval_mode == "Comprehensive (All Levels)":
                    # Use comprehensive evaluation
                    results = evaluator.evaluate_comprehensive(model_path, reference_path)
                else:
                    # Quick evaluation
                    result = evaluator.evaluate(model_path, reference_path)
                    results = {
                        "global": {
                            "lddt": result.lddt,
                            "rmsd_ca": result.rmsd,
                            "tm_score": result.tm_score,
                            "qs_score": result.qs_score,
                        },
                        "per_residue": {"lddt": result.lddt_per_residue or []},
                        "per_chain": {},
                        "interface": {},
                    }

            st.success("✅ Evaluation Complete!")

            # Store results in session state
            st.session_state.eval_results = results

            # ========== GLOBAL METRICS ==========
            st.markdown(
                '<div class="section-header"><h3>🌍 Global Metrics</h3></div>',
                unsafe_allow_html=True,
            )

            global_metrics = results.get("global", {})

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                lddt = global_metrics.get("lddt")
                if lddt is not None:
                    quality = "success" if lddt > 0.7 else "warning" if lddt > 0.5 else ""
                    st.markdown(
                        f"""
                    <div class="metric-card metric-card-{quality}">
                        <div class="metric-value">{lddt:.4f}</div>
                        <div class="metric-label">lDDT</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric("lDDT", "N/A")

            with col2:
                rmsd = global_metrics.get("rmsd_ca") or global_metrics.get("rmsd")
                if rmsd is not None:
                    quality = "success" if rmsd < 2 else "warning" if rmsd < 4 else ""
                    st.markdown(
                        f"""
                    <div class="metric-card metric-card-{quality}">
                        <div class="metric-value">{rmsd:.2f} Å</div>
                        <div class="metric-label">RMSD (Cα)</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric("RMSD", "N/A")

            with col3:
                tm = global_metrics.get("tm_score")
                if tm is not None:
                    quality = "success" if tm > 0.7 else "warning" if tm > 0.5 else ""
                    st.markdown(
                        f"""
                    <div class="metric-card metric-card-{quality}">
                        <div class="metric-value">{tm:.4f}</div>
                        <div class="metric-label">TM-score</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric("TM-score", "N/A")

            with col4:
                qs = global_metrics.get("qs_score")
                if qs is not None:
                    quality = "success" if qs > 0.7 else "warning" if qs > 0.5 else ""
                    st.markdown(
                        f"""
                    <div class="metric-card metric-card-{quality}">
                        <div class="metric-value">{qs:.4f}</div>
                        <div class="metric-label">QS-score</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric("QS-score", "N/A")

            # Additional global metrics - Row 2
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                gdt_ts = global_metrics.get("gdt_ts")
                if gdt_ts is not None:
                    st.metric("GDT-TS", f"{gdt_ts:.4f}")

            with col6:
                gdt_ha = global_metrics.get("gdt_ha")
                if gdt_ha is not None:
                    st.metric("GDT-HA", f"{gdt_ha:.4f}")

            with col7:
                rmsd_bb = global_metrics.get("rmsd_backbone")
                if rmsd_bb is not None:
                    st.metric("RMSD (Backbone)", f"{rmsd_bb:.2f} Å")

            with col8:
                lddt_mean = global_metrics.get("lddt_mean")
                if lddt_mean is not None:
                    st.metric("lDDT Mean", f"{lddt_mean:.4f}")

            # Additional metrics - Row 3 (BB-lDDT, iLDDT, CAD)
            col9, col10, col11, col12 = st.columns(4)

            with col9:
                bb_lddt = global_metrics.get("bb_lddt")
                if bb_lddt is not None:
                    st.metric("BB-lDDT (Backbone)", f"{bb_lddt:.4f}")

            with col10:
                ilddt = global_metrics.get("ilddt")
                if ilddt is not None:
                    st.metric("iLDDT (Inter-chain)", f"{ilddt:.4f}")

            with col11:
                cad = global_metrics.get("cad_score")
                if cad is not None:
                    st.metric("CAD-score", f"{cad:.4f}")

            with col12:
                patch = global_metrics.get("patch_score")
                if patch is not None:
                    st.metric("Patch Score (CASP15)", f"{patch:.4f}")

            # ========== DOCKQ AND INTERFACE QUALITY ==========
            interface = results.get("interface", {})
            dockq = global_metrics.get("dockq")

            if dockq is not None or interface.get("dockq_details"):
                st.markdown(
                    '<div class="section-header"><h3>🎯 DockQ Interface Quality</h3></div>',
                    unsafe_allow_html=True,
                )

                # Global DockQ
                col_dq1, col_dq2, col_dq3, col_dq4 = st.columns(4)

                with col_dq1:
                    if dockq is not None:
                        # DockQ classification
                        if dockq >= 0.8:
                            dockq_class = "High Quality"
                            quality = "success"
                        elif dockq >= 0.49:
                            dockq_class = "Medium Quality"
                            quality = ""
                        elif dockq >= 0.23:
                            dockq_class = "Acceptable"
                            quality = "warning"
                        else:
                            dockq_class = "Incorrect"
                            quality = "warning"

                        st.markdown(
                            f"""
                        <div class="metric-card metric-card-{quality}">
                            <div class="metric-value">{dockq:.4f}</div>
                            <div class="metric-label">DockQ ({dockq_class})</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                # Per-interface DockQ details
                dockq_details = interface.get("dockq_details", [])
                if isinstance(dockq_details, list) and len(dockq_details) > 0:
                    with st.expander("📊 Per-Interface DockQ Details"):
                        import pandas as pd

                        df_dockq = pd.DataFrame(
                            [
                                {
                                    "Interface": f"Interface {i+1}",
                                    "DockQ": d.get("dockq", 0),
                                    "fnat": d.get("fnat", 0),
                                    "fnonnat": d.get("fnonnat", 0),
                                    "iRMSD (Å)": d.get("irmsd", 0),
                                    "lRMSD (Å)": d.get("lrmsd", 0),
                                }
                                for i, d in enumerate(dockq_details)
                            ]
                        )
                        st.dataframe(df_dockq, use_container_width=True, hide_index=True)

                        # DockQ component explanation
                        st.markdown(
                            """
                        **DockQ Components:**
                        - **fnat**: Fraction of native contacts preserved
                        - **fnonnat**: Fraction of non-native contacts (false positives)
                        - **iRMSD**: Interface RMSD (Å) - backbone atoms within 10Å of interface
                        - **lRMSD**: Ligand RMSD (Å) - smaller chain after superposition on larger
                        """
                        )

            # ========== ICS AND IPS METRICS ==========
            ics = global_metrics.get("ics") or interface.get("ics")
            ips = global_metrics.get("ips") or interface.get("ips")

            if ics is not None or ips is not None:
                st.markdown(
                    '<div class="section-header"><h3>🔗 Interface Contact & Patch Similarity</h3></div>',
                    unsafe_allow_html=True,
                )

                col_ics1, col_ics2, col_ics3, col_ics4 = st.columns(4)

                with col_ics1:
                    if ics is not None:
                        st.metric("ICS (F1)", f"{ics:.4f}")

                with col_ics2:
                    ics_prec = interface.get("ics_precision")
                    if ics_prec is not None:
                        st.metric("ICS Precision", f"{ics_prec:.4f}")

                with col_ics3:
                    ics_rec = interface.get("ics_recall")
                    if ics_rec is not None:
                        st.metric("ICS Recall", f"{ics_rec:.4f}")

                with col_ics4:
                    if ips is not None:
                        st.metric("IPS (Jaccard)", f"{ips:.4f}")

                st.caption(
                    "ICS = Interface Contact Similarity (how well contacts are preserved), IPS = Interface Patch Similarity (spatial overlap)"
                )

            # ========== PATCH SCORES (CASP15) ==========
            patch_scores = interface.get("patch_scores", [])
            if patch_scores and isinstance(patch_scores, list) and len(patch_scores) > 0:
                st.markdown(
                    '<div class="section-header"><h3>🏆 Patch Scores (CASP15 Local Interface)</h3></div>',
                    unsafe_allow_html=True,
                )

                import pandas as pd

                df_patch = pd.DataFrame(patch_scores)
                if not df_patch.empty:
                    st.dataframe(df_patch, use_container_width=True, hide_index=True)

                    # Patch score visualization
                    if "score" in df_patch.columns:
                        import plotly.express as px

                        fig = px.bar(df_patch, y="score", title="Patch Scores by Interface Region")
                        fig.update_layout(yaxis_range=[0, 1], height=300)
                        st.plotly_chart(fig, use_container_width=True)

            # lDDT Quality Distribution
            quality_cats = global_metrics.get("lddt_quality_categories", {})
            if quality_cats:
                st.markdown("#### 📊 lDDT Quality Distribution")

                total = sum(quality_cats.values())
                if total > 0:
                    very_high = quality_cats.get("very_high_gt_90", 0)
                    confident = quality_cats.get("confident_70_90", 0)
                    low = quality_cats.get("low_50_70", 0)
                    very_low = quality_cats.get("very_low_lt_50", 0)

                    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                    with col_q1:
                        st.metric("Very High (>90%)", f"{very_high} ({very_high/total*100:.1f}%)")
                    with col_q2:
                        st.metric("Confident (70-90%)", f"{confident} ({confident/total*100:.1f}%)")
                    with col_q3:
                        st.metric("Low (50-70%)", f"{low} ({low/total*100:.1f}%)")
                    with col_q4:
                        st.metric("Very Low (<50%)", f"{very_low} ({very_low/total*100:.1f}%)")

                    # Visual bar
                    import plotly.graph_objects as go

                    fig = go.Figure(
                        go.Bar(
                            x=[very_high, confident, low, very_low],
                            y=[
                                "Very High (>90%)",
                                "Confident (70-90%)",
                                "Low (50-70%)",
                                "Very Low (<50%)",
                            ],
                            orientation="h",
                            marker_color=["#0053d6", "#65cbf3", "#ffdb13", "#ff7d45"],
                            text=[
                                f"{x} ({x/total*100:.1f}%)"
                                for x in [very_high, confident, low, very_low]
                            ],
                            textposition="inside",
                        )
                    )
                    fig.update_layout(
                        title="Residue Quality Distribution",
                        height=200,
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ========== PER-RESIDUE METRICS ==========
            per_residue = results.get("per_residue", {})
            lddt_per_res = per_residue.get("lddt", [])

            if lddt_per_res:
                st.markdown(
                    '<div class="section-header"><h3>📍 Per-Residue Metrics</h3></div>',
                    unsafe_allow_html=True,
                )

                import plotly.express as px
                import plotly.graph_objects as go
                import pandas as pd

                # lDDT per residue plot
                df = pd.DataFrame(
                    {
                        "Residue": range(1, len(lddt_per_res) + 1),
                        "lDDT": lddt_per_res,
                    }
                )

                fig = px.line(df, x="Residue", y="lDDT", title="lDDT per Residue")

                # Add colored background regions for quality
                fig.add_hrect(y0=0.9, y1=1.0, fillcolor="#0053d6", opacity=0.1, line_width=0)
                fig.add_hrect(y0=0.7, y1=0.9, fillcolor="#65cbf3", opacity=0.1, line_width=0)
                fig.add_hrect(y0=0.5, y1=0.7, fillcolor="#ffdb13", opacity=0.1, line_width=0)
                fig.add_hrect(y0=0.0, y1=0.5, fillcolor="#ff7d45", opacity=0.1, line_width=0)

                fig.update_layout(yaxis_range=[0, 1], height=400)
                fig.update_traces(line_color="#667eea")
                st.plotly_chart(fig, use_container_width=True)

                # Detailed residue table
                lddt_details = per_residue.get("lddt_details", [])
                if lddt_details:
                    with st.expander("📋 Detailed Per-Residue Data"):
                        df_details = pd.DataFrame(lddt_details)
                        if not df_details.empty:
                            # Color code by quality
                            def color_lddt(val):
                                if val >= 0.9:
                                    return "background-color: #0053d6; color: white"
                                elif val >= 0.7:
                                    return "background-color: #65cbf3"
                                elif val >= 0.5:
                                    return "background-color: #ffdb13"
                                else:
                                    return "background-color: #ff7d45; color: white"

                            if "lddt" in df_details.columns:
                                styled = df_details.style.applymap(color_lddt, subset=["lddt"])
                                st.dataframe(styled, use_container_width=True, height=400)
                            else:
                                st.dataframe(df_details, use_container_width=True, height=400)

            # ========== PER-CHAIN METRICS ==========
            per_chain = results.get("per_chain", {})
            if per_chain:
                st.markdown(
                    '<div class="section-header"><h3>🔗 Per-Chain Metrics</h3></div>',
                    unsafe_allow_html=True,
                )

                chain_data = []
                for chain_name, chain_metrics in per_chain.items():
                    if isinstance(chain_metrics, dict):
                        row = {"Chain": chain_name}
                        if "lddt" in chain_metrics and isinstance(chain_metrics["lddt"], dict):
                            row.update(chain_metrics["lddt"])
                        else:
                            row.update(chain_metrics)
                        chain_data.append(row)

                if chain_data:
                    import pandas as pd

                    df_chains = pd.DataFrame(chain_data)
                    st.dataframe(df_chains, use_container_width=True, hide_index=True)

                    # Chain comparison chart
                    if "mean_lddt" in df_chains.columns:
                        import plotly.express as px

                        fig = px.bar(
                            df_chains,
                            x="Chain",
                            y="mean_lddt",
                            title="lDDT by Chain",
                            color="mean_lddt",
                            color_continuous_scale="RdYlGn",
                        )
                        fig.update_layout(yaxis_range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)

            # ========== INTERFACE METRICS ==========
            interface = results.get("interface", {})
            if interface and not interface.get("qs_note"):
                st.markdown(
                    '<div class="section-header"><h3>🤝 Interface Metrics</h3></div>',
                    unsafe_allow_html=True,
                )

                if interface.get("qs_error"):
                    st.warning(f"QS-score error: {interface['qs_error']}")
                else:
                    col_if1, col_if2 = st.columns(2)

                    with col_if1:
                        if interface.get("chain_mapping"):
                            st.markdown("**Chain Mapping:**")
                            st.json(interface["chain_mapping"])

                    with col_if2:
                        if interface.get("mapped_target_chains"):
                            st.markdown(
                                f"**Mapped Target Chains:** {', '.join(interface['mapped_target_chains'])}"
                            )
                        if interface.get("mapped_model_chains"):
                            st.markdown(
                                f"**Mapped Model Chains:** {', '.join(interface['mapped_model_chains'])}"
                            )

            elif interface.get("qs_note"):
                st.info(f"ℹ️ {interface['qs_note']}")

            # ========== CONTACT INFORMATION ==========
            if global_metrics.get("lddt_total_contacts"):
                st.markdown(
                    '<div class="section-header"><h3>📊 Contact Statistics</h3></div>',
                    unsafe_allow_html=True,
                )

                col_c1, col_c2, col_c3 = st.columns(3)

                with col_c1:
                    st.metric("Total Contacts", global_metrics.get("lddt_total_contacts", "N/A"))

                with col_c2:
                    st.metric(
                        "Conserved Contacts", global_metrics.get("lddt_conserved_contacts", "N/A")
                    )

                with col_c3:
                    total = global_metrics.get("lddt_total_contacts", 0)
                    conserved = global_metrics.get("lddt_conserved_contacts", 0)
                    if total > 0:
                        pct = conserved / total * 100
                        st.metric("Conservation Rate", f"{pct:.1f}%")

            # ========== PAE VISUALIZATION ==========
            # Check for PAE data in results or uploaded files
            pae_data = results.get("pae") or results.get("predicted_aligned_error")

            # Try to load PAE from output directory if not in results
            if pae_data is None:
                # Look for PAE JSON files
                pae_files = list(model_path.parent.glob("*pae*.json")) + list(
                    model_path.parent.glob("*scores*.json")
                )
                for pae_file in pae_files:
                    try:
                        from protein_design_hub.web.visualizations import load_pae_from_json

                        pae_data = load_pae_from_json(pae_file)
                        if pae_data:
                            break
                    except Exception:
                        pass

            if pae_data:
                st.markdown(
                    '<div class="section-header"><h3>🎨 Predicted Aligned Error (PAE)</h3></div>',
                    unsafe_allow_html=True,
                )

                try:
                    from protein_design_hub.web.visualizations import create_pae_heatmap

                    fig_pae = create_pae_heatmap(pae_data)
                    st.plotly_chart(fig_pae, use_container_width=True)

                    st.caption(
                        "PAE shows the expected position error (Å) when residue X is used to align the structure. Low values (green) indicate high confidence in relative positions."
                    )
                except Exception as e:
                    st.warning(f"Could not render PAE: {e}")

            # ========== CONTACT MAP VISUALIZATION ==========
            st.markdown(
                '<div class="section-header"><h3>🔗 Contact Map Analysis</h3></div>',
                unsafe_allow_html=True,
            )

            col_cm1, col_cm2 = st.columns(2)

            with col_cm1:
                contact_threshold = st.slider("Contact threshold (Å)", 4.0, 12.0, 8.0, 0.5)

            with col_cm2:
                show_comparison = st.checkbox("Show model vs reference comparison", value=True)

            try:
                from protein_design_hub.web.visualizations import (
                    compute_contact_map_from_structure,
                    create_contact_map,
                )

                model_contacts = compute_contact_map_from_structure(model_path)

                if show_comparison and reference_path:
                    ref_contacts = compute_contact_map_from_structure(reference_path)
                    fig_cm = create_contact_map(
                        model_contacts, ref_contacts, threshold=contact_threshold
                    )
                else:
                    fig_cm = create_contact_map(model_contacts, threshold=contact_threshold)

                st.plotly_chart(fig_cm, use_container_width=True)

                # Contact statistics
                model_contact_count = (model_contacts < contact_threshold).sum() - len(
                    model_contacts
                )
                st.caption(
                    f"Model has {model_contact_count // 2} contacts at {contact_threshold}Å threshold"
                )

            except Exception as e:
                st.info(f"Contact map visualization requires Biopython and scipy: {e}")

            # ========== DOWNLOAD RESULTS ==========
            st.markdown("---")

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                st.download_button(
                    "📥 Download Full Results (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name="evaluation_results_comprehensive.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with col_dl2:
                # Generate text report
                report = []
                report.append("=" * 60)
                report.append("COMPREHENSIVE STRUCTURE EVALUATION REPORT")
                report.append("=" * 60)
                report.append("")
                report.append("GLOBAL METRICS:")
                report.append("-" * 40)
                for key, val in global_metrics.items():
                    if isinstance(val, (int, float)):
                        report.append(
                            f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}"
                        )
                report.append("")
                report.append(f"Per-residue data: {len(lddt_per_res)} residues")
                report.append(f"Chains analyzed: {list(per_chain.keys())}")

                st.download_button(
                    "📥 Download Report (TXT)",
                    data="\n".join(report),
                    file_name="evaluation_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            import traceback

            with st.expander("Error details"):
                st.code(traceback.format_exc())

# ========== METRIC DESCRIPTIONS ==========
with st.expander("ℹ️ Metric Descriptions"):
    st.markdown(
        """
    ### 🎯 Global Metrics

    #### lDDT (Local Distance Difference Test)
    Measures local structural accuracy by comparing interatomic distances between model and reference.
    - **Range**: 0-1 (higher is better)
    - **Interpretation**:
        - >0.9: Very high confidence (excellent)
        - 0.7-0.9: Confident (good)
        - 0.5-0.7: Low confidence
        - <0.5: Very low confidence (poor)

    #### BB-lDDT (Backbone lDDT)
    Backbone-only lDDT using CA atoms for proteins or C3' for nucleotides.
    - More lenient than full lDDT as it ignores side chains.

    #### iLDDT (Inter-chain lDDT)
    lDDT computed only on inter-chain contacts.
    - Specifically measures interface accuracy independent of individual chain quality.

    #### RMSD (Root Mean Square Deviation)
    Measures average distance between aligned atoms after optimal superposition.
    - **Units**: Angstroms (Å)
    - **Interpretation**: Lower is better
        - <2Å: Excellent structural similarity
        - 2-4Å: Good similarity
        - >4Å: Significant deviations

    #### TM-score (Template Modeling Score)
    Measures global structural similarity, length-normalized.
    - **Range**: 0-1 (higher is better)
    - **Interpretation**:
        - >0.5: Same fold
        - >0.7: High structural similarity
        - >0.9: Nearly identical structures

    #### CAD-score (Contact Area Difference)
    Measures contact surface area similarity using Voronoi tessellation.
    - **Range**: 0-1 (higher is better)
    - **Requires**: voronota_cadscore external tool

    #### GDT-TS/GDT-HA (Global Distance Test)
    - **GDT-TS**: Percentage of residues within 1, 2, 4, 8 Å of reference
    - **GDT-HA**: High-accuracy version (0.5, 1, 2, 4 Å thresholds)

    ---

    ### 🎯 Interface Metrics

    #### DockQ (Docking Quality Score)
    Comprehensive interface quality metric combining:
    - **fnat**: Fraction of native contacts preserved (0-1)
    - **fnonnat**: Fraction of false positive contacts (lower is better)
    - **iRMSD**: Interface RMSD - backbone atoms within 10Å of interface
    - **lRMSD**: Ligand RMSD - smaller chain after superposition on larger

    **DockQ Classification:**
    | DockQ | Classification |
    |-------|----------------|
    | ≥0.80 | High Quality |
    | 0.49-0.80 | Medium Quality |
    | 0.23-0.49 | Acceptable |
    | <0.23 | Incorrect |

    #### QS-score (Quaternary Structure Score)
    Evaluates overall interface quality in multimeric structures.
    - **Range**: 0-1 (higher is better)
    - **Note**: Only computed for complexes with multiple chains

    #### ICS (Interface Contact Similarity)
    Measures how well interface contacts are preserved.
    - **Precision**: Fraction of predicted contacts that are native
    - **Recall**: Fraction of native contacts that are predicted
    - **F1**: Harmonic mean of precision and recall

    #### IPS (Interface Patch Similarity)
    Measures spatial overlap of interface patches using Jaccard coefficient.
    - **Range**: 0-1 (higher is better)
    - Focuses on the physical location of interface residues

    #### Patch Scores (CASP15)
    Local interface quality scores used in CASP15 assessment.
    - Evaluates specific interface regions independently
    - Useful for identifying well-modeled vs. poorly-modeled interface regions

    ---

    ### 📍 Per-Residue Metrics
    Per-residue lDDT values identify local regions of high/low accuracy.

    ### 🔗 Per-Chain Metrics
    Chain-level aggregated metrics useful for analyzing multimeric structures.

    ### 🤝 Interface Metrics
    Chain mapping and interface quality for protein-protein interfaces.
    """
    )

# OpenStructure status
with st.sidebar.expander("🔧 Tool Status"):
    try:
        from protein_design_hub.evaluation.ost_runner import get_ost_runner

        runner = get_ost_runner()

        if runner.is_available():
            version = runner.get_version()
            st.success(f"OpenStructure: v{version}")
        else:
            st.error("OpenStructure: Not available")
            st.info("Install: `micromamba create -n ost -c conda-forge -c bioconda openstructure`")
    except Exception as e:
        st.error(f"Error checking tools: {e}")
