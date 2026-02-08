"""ProteinMPNN sequence design page."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile

import streamlit as st

from protein_design_hub.analysis.sequence_metrics import compute_sequence_metrics
from protein_design_hub.web.ui import (
    get_selected_backbone,
    inject_base_css,
    page_header,
    section_header,
    info_box,
    metric_card,
    metric_card_with_context,
    progress_steps,
    set_selected_backbone,
    sidebar_nav,
    sidebar_system_status,
    list_output_structures,
    workflow_breadcrumb,
    cross_page_actions,
)
from protein_design_hub.web.agent_helpers import (
    render_agent_advice_panel,
    render_contextual_insight,
    agent_sidebar_status,
    render_all_experts_panel,
)

st.set_page_config(page_title="MPNN Design - Protein Design Hub", page_icon="🎯", layout="wide")
inject_base_css()
sidebar_nav(current="MPNN Lab")
sidebar_system_status()
agent_sidebar_status()

# Page-specific styling
st.markdown("""
<style>
.mpnn-input-card {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-lg);
    padding: var(--pdhub-space-lg);
    border: 1px solid var(--pdhub-border);
    margin-bottom: var(--pdhub-space-md);
}

.mpnn-settings-card {
    background: var(--pdhub-bg-elevated, rgba(28, 30, 42, 0.95));
    border-radius: var(--pdhub-border-radius-lg);
    padding: var(--pdhub-space-lg);
    border: 1px solid var(--pdhub-border);
}

.sequence-result {
    background: var(--pdhub-bg-card);
    border-radius: var(--pdhub-border-radius-md);
    padding: var(--pdhub-space-md);
    margin: var(--pdhub-space-sm) 0;
    border: 1px solid var(--pdhub-border);
    transition: var(--pdhub-transition);
}

.sequence-result:hover {
    border-color: var(--pdhub-primary-light);
    box-shadow: var(--pdhub-shadow-sm);
}

.sequence-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--pdhub-space-sm);
}

.sequence-id {
    font-weight: 600;
    color: var(--pdhub-primary);
}

.sequence-code {
    font-family: monospace;
    font-size: 0.85rem;
    background: var(--pdhub-bg-light);
    padding: var(--pdhub-space-sm);
    border-radius: var(--pdhub-border-radius-sm);
    word-break: break-all;
    color: var(--pdhub-text);
}

.metrics-row {
    display: flex;
    gap: var(--pdhub-space-md);
    margin-top: var(--pdhub-space-md);
    flex-wrap: wrap;
}

.mini-metric {
    background: var(--pdhub-bg-light);
    padding: 8px 14px;
    border-radius: var(--pdhub-border-radius-sm);
    text-align: center;
    min-width: 80px;
}

.mini-metric-value {
    font-weight: 600;
    color: var(--pdhub-text);
    font-size: 1rem;
}

.mini-metric-label {
    font-size: 0.7rem;
    color: var(--pdhub-text-secondary);
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# Header
page_header(
    "ProteinMPNN Design",
    "Fixed-backbone sequence design using ProteinMPNN neural network",
    "🎯"
)

workflow_breadcrumb(
    ["Predict Structure", "Evaluate", "Design Sequence (MPNN)", "Validate"],
    current=2,
)

with st.expander("📖 ProteinMPNN sequence design guide", expanded=False):
    st.markdown("""
**ProteinMPNN** designs new amino acid sequences that fold into a given backbone structure.

**How to use:**
1. Upload a PDB backbone (from prediction or experimental structure)
2. Set **temperature** (controls sequence diversity):
   - `0.1` = conservative (high sequence recovery, safe)
   - `0.3` = moderate (good diversity/quality balance, recommended)
   - `0.5+` = diverse (creative designs, may need validation)
3. Optionally **fix positions** you want to keep unchanged (e.g. active site residues)
4. Run and review designed sequences with biophysical metrics

**Self-consistency check:** Re-predict designed sequences with ESMFold/ColabFold. If TM-score > 0.9 to the template, the design likely folds correctly.

**Sequence recovery:** ProteinMPNN typically recovers 30-50% of native sequence. Higher recovery means the design is closer to natural proteins.
    """)

# Main content
col_a, col_b = st.columns([2, 1])

with col_a:
    section_header("Backbone Structure", "Upload or select a structure file", "🦴")

    try:
        from protein_design_hub.core.config import get_settings

        _settings = get_settings()
        recent = list_output_structures(Path(_settings.output.base_dir))
    except Exception:
        recent = []

    chosen = None
    sel = get_selected_backbone()

    if sel is not None and sel.exists():
        chosen = sel
        info_box(
            f"Using selected backbone from Jobs: <code>{sel.name}</code>",
            variant="success",
            title="Backbone Selected"
        )
        if st.button("Clear selected backbone", key="clear_sel_backbone", type="secondary"):
            set_selected_backbone(None)
            st.rerun()

    if recent:
        st.markdown("**Or choose from recent structures:**")
        default_index = 0
        if chosen is not None and chosen in recent:
            default_index = 1 + recent.index(chosen)
        chosen = st.selectbox(
            "Recent structures",
            options=[None] + recent,
            format_func=lambda p: "— Select a structure —" if p is None else str(p.name),
            index=default_index,
            label_visibility="collapsed"
        )

    st.markdown("**Or upload a new file:**")
    backbone_file = st.file_uploader(
        "Upload backbone (PDB/mmCIF)",
        type=["pdb", "cif", "mmcif"],
        label_visibility="collapsed"
    )

    # Preview Logic
    if backbone_file or chosen:
        st.markdown("### 🧬 Backbone Preview")
        preview_path = None
        if backbone_file:
             with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                 tmp.write(backbone_file.read())
                 backbone_file.seek(0) # Reset
                 preview_path = Path(tmp.name)
        elif chosen:
             preview_path = chosen
        
        if preview_path:
             from protein_design_hub.web.visualizations import create_structure_viewer
             import streamlit.components.v1 as components
             html = create_structure_viewer(preview_path, height=300, style="cartoon")
             components.html(html, height=320)

with col_b:
    section_header("Sampling Settings", "Configure design parameters", "⚙️")

    st.markdown('<div class="mpnn-settings-card">', unsafe_allow_html=True)

    num_seqs = st.number_input(
        "Number of sequences",
        min_value=1,
        max_value=256,
        value=16,
        help="How many sequence designs to generate"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Lower = more conservative, Higher = more diverse"
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=0,
        help="0 = random seed, or set a specific value for reproducibility"
    )

    auto_install = st.checkbox(
        "Auto-install ProteinMPNN",
        value=False,
        help="Automatically clone ProteinMPNN if not installed"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Design Constraints Section
    with st.expander("🎯 Design Constraints", expanded=False):
        st.markdown("##### Fixed Positions")
        fixed_positions = st.text_input(
            "Positions to keep fixed",
            placeholder="e.g., 1-10, 25, 30-35",
            help="These residues will not be redesigned"
        )

        st.markdown("##### Designable Chains")
        design_chains = st.text_input(
            "Chains to design",
            value="A",
            placeholder="e.g., A,B or A",
            help="Which chains to redesign (comma-separated)"
        )

        st.markdown("##### Amino Acid Bias")
        bias_mode = st.selectbox(
            "Bias type",
            ["None", "Favor hydrophilic", "Avoid cysteine", "Custom"],
            help="Apply bias to amino acid sampling"
        )

        if bias_mode == "Custom":
            st.text_input(
                "Favored residues",
                placeholder="e.g., A,L,V,I",
                help="Residues to favor during design"
            )
            st.text_input(
                "Avoided residues",
                placeholder="e.g., C,M,W",
                help="Residues to avoid during design"
            )

    # Advanced Options
    with st.expander("⚙️ Advanced Options", expanded=False):
        use_soluble_model = st.checkbox(
            "Use soluble model",
            value=False,
            help="Optimized for soluble proteins"
        )

        omit_aa = st.text_input(
            "Omit amino acids globally",
            placeholder="e.g., C,M",
            help="Never include these amino acids"
        )

        pssm_mode = st.checkbox(
            "Use PSSM bias",
            value=False,
            help="Apply position-specific scoring matrix bias"
        )

        if pssm_mode:
            pssm_file = st.file_uploader(
                "Upload PSSM file",
                type=["txt", "pssm"],
                help="BLAST-format PSSM file"
            )

st.markdown("---")

# Run button
if st.button("🚀 Run ProteinMPNN Design", type="primary", use_container_width=True):
    if backbone_file is None and chosen is None:
        info_box(
            "Please upload a backbone structure or select one from recent structures.",
            variant="error",
            title="No Structure Selected"
        )
    else:
        try:
            from protein_design_hub.core.config import get_settings
            from protein_design_hub.design.registry import get_designer
            from protein_design_hub.design.types import DesignInput

            settings = get_settings()

            if chosen is not None and backbone_file is None:
                backbone_path = Path(chosen)
            else:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(backbone_file.name).suffix, delete=False
                ) as tmp:
                    tmp.write(backbone_file.read())
                    backbone_path = Path(tmp.name)

            job_id = f"mpnn_{backbone_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            job_dir = settings.output.base_dir / job_id / "design" / "proteinmpnn"
            job_dir.mkdir(parents=True, exist_ok=True)

            designer = get_designer("proteinmpnn", settings)

            # Show progress
            progress_steps(["Initialize", "Design", "Analyze", "Complete"], current_step=1)

            with st.spinner("Running ProteinMPNN sequence design..."):
                result = designer.design(
                    DesignInput(
                        job_id=job_id,
                        backbone_path=backbone_path,
                        output_dir=job_dir,
                        num_sequences=int(num_seqs),
                        temperature=float(temperature),
                        seed=None if int(seed) == 0 else int(seed),
                    ),
                    auto_install=auto_install,
                )

            if not result.success:
                info_box(
                    result.error_message or "ProteinMPNN failed with unknown error",
                    variant="error",
                    title="Design Failed"
                )
            else:
                # Success!
                progress_steps(["Initialize", "Design", "Analyze", "Complete"], current_step=3)

                info_box(
                    f"Successfully designed {len(result.sequences)} sequences!",
                    variant="success",
                    title="Design Complete"
                )

                section_header("Results", f"{len(result.sequences)} sequences generated", "📊")

                # Show top sequences + metrics
                for i, s in enumerate(result.sequences[:20]):
                    with st.expander(f"**{s.id}**", expanded=(i < 3)):
                        st.markdown(f'<div class="sequence-code">{s.sequence}</div>', unsafe_allow_html=True)

                        try:
                            m = compute_sequence_metrics(s.sequence)
                            st.markdown("""
                            <div class="metrics-row">
                                <div class="mini-metric">
                                    <div class="mini-metric-value">{:.2f}</div>
                                    <div class="mini-metric-label">pI</div>
                                </div>
                                <div class="mini-metric">
                                    <div class="mini-metric-value">{:.2f}</div>
                                    <div class="mini-metric-label">Charge pH7</div>
                                </div>
                                <div class="mini-metric">
                                    <div class="mini-metric-value">{:.3f}</div>
                                    <div class="mini-metric-label">GRAVY</div>
                                </div>
                                <div class="mini-metric">
                                    <div class="mini-metric-value">{:.1f}</div>
                                    <div class="mini-metric-label">Instability</div>
                                </div>
                            </div>
                            """.format(
                                m.isoelectric_point,
                                m.net_charge_ph7,
                                m.gravy,
                                m.instability_index
                            ), unsafe_allow_html=True)
                        except Exception as e:
                            st.caption(f"Metrics unavailable: {e}")

                # Download FASTA
                st.markdown("---")
                fasta_lines = []
                for s in result.sequences:
                    fasta_lines.append(f">{s.id}")
                    fasta_lines.append(s.sequence)

                col_d1, col_d2 = st.columns([1, 1])
                with col_d1:
                    st.download_button(
                        "📥 Download All Sequences (FASTA)",
                        data="\n".join(fasta_lines) + "\n",
                        file_name=f"{job_id}_designed.fasta",
                        mime="text/plain",
                        use_container_width=True,
                        type="primary"
                    )

                with col_d2:
                    st.markdown("**Validation**")
                    if st.button("🔮 Fold Top Sequence (ESMFold)", use_container_width=True):
                         st.session_state.predict_sequence = result.sequences[0].sequence
                         st.session_state.predict_name = f"{result.sequences[0].id}_check"
                         st.switch_page("pages/1_predict.py")

                st.caption(f"Results saved to: `{job_dir}`")

                # Agent advice on MPNN design results
                seq_summaries = []
                for s in result.sequences[:5]:
                    try:
                        m = compute_sequence_metrics(s.sequence)
                        seq_summaries.append(
                            f"- {s.id}: len={len(s.sequence)}, pI={m.isoelectric_point:.1f}, "
                            f"charge={m.net_charge_ph7:.1f}, GRAVY={m.gravy:.3f}, "
                            f"instability={m.instability_index:.1f}"
                        )
                    except Exception:
                        seq_summaries.append(f"- {s.id}: len={len(s.sequence)}")

                render_agent_advice_panel(
                    page_context=(
                        f"ProteinMPNN designed {len(result.sequences)} sequences.\n"
                        f"Top 5 sequences:\n" + "\n".join(seq_summaries)
                    ),
                    default_question=(
                        "Evaluate these designed sequences. Which ones look most "
                        "promising for expression and stability? Any red flags?"
                    ),
                    expert="Machine Learning Specialist",
                    key_prefix="mpnn_agent",
                )

                render_all_experts_panel(
                    "All-Expert Review (MPNN design job)",
                    agenda=(
                        "Review ProteinMPNN design outputs and recommend which sequences "
                        "to prioritize for folding validation and experimental follow-up."
                    ),
                    context=(
                        f"Designed sequences: {len(result.sequences)}\n"
                        f"Backbone: {backbone_path.name}\n"
                        f"Top sequence summaries:\n" + "\n".join(seq_summaries)
                    ),
                    questions=(
                        "Which designed sequences should be prioritized first and why?",
                        "Any sequence-level red flags for expression/solubility/stability?",
                        "What validation stack should run next before experiments?",
                    ),
                    key_prefix="mpnn_all",
                )

        except Exception as e:
            info_box(
                f"An error occurred: {e}",
                variant="error",
                title="Error"
            )
            import traceback
            with st.expander("Show traceback"):
                st.code(traceback.format_exc())

# Tips section
st.markdown("---")
with st.expander("💡 Tips for using ProteinMPNN"):
    st.markdown("""**Temperature Settings:**
- **0.1** (default): Conservative designs, high sequence recovery
- **0.2-0.3**: Balanced diversity and recovery
- **0.5+**: More diverse, novel sequences

**Best Practices:**
- Start with a well-resolved backbone structure (< 2.5 Å resolution)
- Use 16-32 sequences initially, then filter by metrics
- Lower temperature for stability, higher for novelty
- Consider running multiple rounds with different temperatures""")
