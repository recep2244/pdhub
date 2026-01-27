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
    list_output_structures,
    set_selected_backbone,
    sidebar_nav,
    sidebar_system_status,
)

st.set_page_config(page_title="MPNN Design - Protein Design Hub", page_icon="🧬", layout="wide")
inject_base_css()
sidebar_nav(current="MPNN Design")
sidebar_system_status()

st.title("🧬 ProteinMPNN Sequence Design")
st.caption("Fixed-backbone sequence design (requires local ProteinMPNN + torch).")

col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("Backbone Structure")
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
        st.success("Using selected backbone from Jobs")
        st.code(str(chosen))
        if st.button("Clear selected backbone", key="clear_sel_backbone"):
            set_selected_backbone(None)
            st.rerun()

    if recent:
        default_index = 0
        if chosen is not None and chosen in recent:
            default_index = 1 + recent.index(chosen)
        chosen = st.selectbox(
            "Use recent structure (optional)",
            options=[None] + recent,
            format_func=lambda p: "—" if p is None else str(p),
            index=default_index,
        )

    backbone_file = st.file_uploader("Upload backbone (PDB/mmCIF)", type=["pdb", "cif", "mmcif"])

with col_b:
    st.subheader("Sampling")
    num_seqs = st.number_input("Sequences", min_value=1, max_value=256, value=16)
    temperature = st.slider("Temperature", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    seed = st.number_input("Seed (0=random)", min_value=0, value=0)
    auto_install = st.checkbox("Auto-install ProteinMPNN (git clone)", value=False)

st.markdown("---")

if st.button("🚀 Run ProteinMPNN", type="primary", use_container_width=True):
    if backbone_file is None and chosen is None:
        st.error("Please upload a backbone structure (or pick a recent one).")
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
            with st.spinner("Running ProteinMPNN..."):
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
                st.error(result.error_message or "ProteinMPNN failed")
            else:
                st.success(f"Designed {len(result.sequences)} sequences")

                # Show top sequences + metrics
                for s in result.sequences[:20]:
                    with st.expander(s.id):
                        st.code(s.sequence)
                        try:
                            m = compute_sequence_metrics(s.sequence)
                            cols = st.columns(4)
                            cols[0].metric("pI", f"{m.isoelectric_point:.2f}")
                            cols[1].metric("Charge pH7", f"{m.net_charge_ph7:.2f}")
                            cols[2].metric("GRAVY", f"{m.gravy:.3f}")
                            cols[3].metric("Instability", f"{m.instability_index:.2f}")
                        except Exception as e:
                            st.caption(f"Metrics unavailable: {e}")

                # Download FASTA
                fasta_lines = []
                for s in result.sequences:
                    fasta_lines.append(f">{s.id}")
                    fasta_lines.append(s.sequence)
                st.download_button(
                    "📥 Download FASTA",
                    data="\n".join(fasta_lines) + "\n",
                    file_name=f"{job_id}_designed.fasta",
                    mime="text/plain",
                    use_container_width=True,
                )

                st.caption(f"Saved under: `{job_dir}`")

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback

            with st.expander("Traceback"):
                st.code(traceback.format_exc())
