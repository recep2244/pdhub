"""Sequence and Ligand Design page for Streamlit app."""

import streamlit as st
from pathlib import Path
import json

st.set_page_config(page_title="Design - Protein Design Hub", page_icon="🧬", layout="wide")

# Amino acid data
AMINO_ACIDS = {
    'A': {'name': 'Alanine', 'code': 'Ala', 'properties': 'hydrophobic, small'},
    'C': {'name': 'Cysteine', 'code': 'Cys', 'properties': 'polar, disulfide bonds'},
    'D': {'name': 'Aspartic acid', 'code': 'Asp', 'properties': 'acidic, charged (-)'},
    'E': {'name': 'Glutamic acid', 'code': 'Glu', 'properties': 'acidic, charged (-)'},
    'F': {'name': 'Phenylalanine', 'code': 'Phe', 'properties': 'aromatic, hydrophobic'},
    'G': {'name': 'Glycine', 'code': 'Gly', 'properties': 'flexible, small'},
    'H': {'name': 'Histidine', 'code': 'His', 'properties': 'basic, aromatic'},
    'I': {'name': 'Isoleucine', 'code': 'Ile', 'properties': 'hydrophobic, branched'},
    'K': {'name': 'Lysine', 'code': 'Lys', 'properties': 'basic, charged (+)'},
    'L': {'name': 'Leucine', 'code': 'Leu', 'properties': 'hydrophobic, branched'},
    'M': {'name': 'Methionine', 'code': 'Met', 'properties': 'hydrophobic, sulfur'},
    'N': {'name': 'Asparagine', 'code': 'Asn', 'properties': 'polar, amide'},
    'P': {'name': 'Proline', 'code': 'Pro', 'properties': 'rigid, cyclic'},
    'Q': {'name': 'Glutamine', 'code': 'Gln', 'properties': 'polar, amide'},
    'R': {'name': 'Arginine', 'code': 'Arg', 'properties': 'basic, charged (+)'},
    'S': {'name': 'Serine', 'code': 'Ser', 'properties': 'polar, hydroxyl'},
    'T': {'name': 'Threonine', 'code': 'Thr', 'properties': 'polar, hydroxyl'},
    'V': {'name': 'Valine', 'code': 'Val', 'properties': 'hydrophobic, branched'},
    'W': {'name': 'Tryptophan', 'code': 'Trp', 'properties': 'aromatic, largest'},
    'Y': {'name': 'Tyrosine', 'code': 'Tyr', 'properties': 'aromatic, hydroxyl'},
}

AA_COLORS = {
    'A': '#8B0000', 'C': '#FFD700', 'D': '#FF0000', 'E': '#FF0000',
    'F': '#4169E1', 'G': '#808080', 'H': '#00CED1', 'I': '#006400',
    'K': '#0000FF', 'L': '#006400', 'M': '#FFD700', 'N': '#FFA500',
    'P': '#708090', 'Q': '#FFA500', 'R': '#0000FF', 'S': '#FFA500',
    'T': '#FFA500', 'V': '#006400', 'W': '#4169E1', 'Y': '#00CED1',
}

# Initialize session state
if 'sequences' not in st.session_state:
    st.session_state.sequences = []
if 'ligands' not in st.session_state:
    st.session_state.ligands = []
if 'design_history' not in st.session_state:
    st.session_state.design_history = []

st.title("🧬 Sequence & Ligand Design")
st.markdown("Design protein sequences and ligands for structure prediction")

# Create tabs
tab_sequence, tab_ligand, tab_preview = st.tabs(["📝 Sequence Design", "💊 Ligand Design", "👁️ Preview & Export"])

# === SEQUENCE DESIGN TAB ===
with tab_sequence:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Sequence")

        input_method = st.radio(
            "Input method",
            ["Upload FASTA", "Paste sequence", "Design from scratch"],
            horizontal=True,
            key="seq_input_method"
        )

        if input_method == "Upload FASTA":
            uploaded_file = st.file_uploader(
                "Upload FASTA file",
                type=["fasta", "fa", "faa"],
                key="fasta_upload"
            )
            if uploaded_file:
                content = uploaded_file.read().decode("utf-8")
                # Parse FASTA
                sequences = []
                current_header = ""
                current_seq = []
                for line in content.strip().split('\n'):
                    if line.startswith('>'):
                        if current_header:
                            sequences.append({
                                'name': current_header,
                                'sequence': ''.join(current_seq),
                                'type': 'protein'
                            })
                        current_header = line[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line.strip())
                if current_header:
                    sequences.append({
                        'name': current_header,
                        'sequence': ''.join(current_seq),
                        'type': 'protein'
                    })
                st.session_state.sequences = sequences
                st.success(f"Loaded {len(sequences)} sequence(s)")

        elif input_method == "Paste sequence":
            seq_name = st.text_input("Sequence name", value="protein_1")
            seq_text = st.text_area(
                "Enter amino acid sequence",
                placeholder="MKFLILLFNILCLFPVLAADNHGVGPQGAS...",
                height=150
            )
            mol_type = st.selectbox("Molecule type", ["protein", "dna", "rna"])

            if st.button("Add Sequence", key="add_seq"):
                if seq_text:
                    st.session_state.sequences.append({
                        'name': seq_name,
                        'sequence': seq_text.upper().replace(' ', '').replace('\n', ''),
                        'type': mol_type
                    })
                    st.success(f"Added sequence: {seq_name}")
                    st.rerun()

        else:  # Design from scratch
            st.markdown("**Build sequence residue by residue:**")
            new_seq_name = st.text_input("New sequence name", value="designed_protein")

            if 'building_sequence' not in st.session_state:
                st.session_state.building_sequence = ""

            # Quick add buttons for each AA
            st.markdown("**Add amino acid:**")
            aa_cols = st.columns(10)
            for i, aa in enumerate(AMINO_ACIDS.keys()):
                with aa_cols[i % 10]:
                    if st.button(aa, key=f"add_{aa}", help=f"{AMINO_ACIDS[aa]['name']}"):
                        st.session_state.building_sequence += aa
                        st.rerun()

            st.text_area(
                "Building sequence",
                value=st.session_state.building_sequence,
                height=100,
                disabled=True
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("Undo", key="undo_aa"):
                    if st.session_state.building_sequence:
                        st.session_state.building_sequence = st.session_state.building_sequence[:-1]
                        st.rerun()
            with col_b:
                if st.button("Clear", key="clear_aa"):
                    st.session_state.building_sequence = ""
                    st.rerun()
            with col_c:
                if st.button("Save Sequence", key="save_built"):
                    if st.session_state.building_sequence:
                        st.session_state.sequences.append({
                            'name': new_seq_name,
                            'sequence': st.session_state.building_sequence,
                            'type': 'protein'
                        })
                        st.session_state.building_sequence = ""
                        st.success(f"Saved: {new_seq_name}")
                        st.rerun()

    with col2:
        st.subheader("Sequence Editor")

        if st.session_state.sequences:
            # Select sequence to edit
            seq_names = [s['name'] for s in st.session_state.sequences]
            selected_seq_idx = st.selectbox(
                "Select sequence to edit",
                range(len(seq_names)),
                format_func=lambda x: seq_names[x],
                key="edit_seq_select"
            )

            current_seq = st.session_state.sequences[selected_seq_idx]

            # Display sequence with residue colors
            st.markdown("**Sequence visualization:**")
            seq_html = ""
            for i, aa in enumerate(current_seq['sequence']):
                color = AA_COLORS.get(aa, '#000000')
                seq_html += f'<span style="color:{color};font-family:monospace;font-size:14px;" title="Position {i+1}: {AMINO_ACIDS.get(aa, {}).get("name", aa)}">{aa}</span>'
                if (i + 1) % 50 == 0:
                    seq_html += "<br>"
                elif (i + 1) % 10 == 0:
                    seq_html += " "
            st.markdown(f'<div style="background-color:#f0f0f0;padding:10px;border-radius:5px;line-height:1.8;">{seq_html}</div>', unsafe_allow_html=True)

            st.markdown(f"**Length:** {len(current_seq['sequence'])} residues")

            # Residue-level editor
            st.markdown("---")
            st.markdown("### Edit Specific Residues")

            col_pos, col_new = st.columns([1, 2])

            with col_pos:
                edit_position = st.number_input(
                    "Position to edit",
                    min_value=1,
                    max_value=len(current_seq['sequence']),
                    value=1,
                    key="edit_pos"
                )
                current_aa = current_seq['sequence'][edit_position - 1]
                st.info(f"Current: **{current_aa}** ({AMINO_ACIDS.get(current_aa, {}).get('name', 'Unknown')})")

            with col_new:
                st.markdown("**Replace with:**")

                # Group amino acids by property
                st.markdown("*Hydrophobic:*")
                hydrophobic = ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P']
                h_cols = st.columns(8)
                for i, aa in enumerate(hydrophobic):
                    with h_cols[i]:
                        if st.button(aa, key=f"rep_h_{aa}", help=AMINO_ACIDS[aa]['name']):
                            seq_list = list(current_seq['sequence'])
                            old_aa = seq_list[edit_position - 1]
                            seq_list[edit_position - 1] = aa
                            st.session_state.sequences[selected_seq_idx]['sequence'] = ''.join(seq_list)
                            st.session_state.design_history.append({
                                'action': 'mutate',
                                'position': edit_position,
                                'from': old_aa,
                                'to': aa,
                                'sequence': current_seq['name']
                            })
                            st.rerun()

                st.markdown("*Polar:*")
                polar = ['S', 'T', 'N', 'Q', 'Y', 'C']
                p_cols = st.columns(6)
                for i, aa in enumerate(polar):
                    with p_cols[i]:
                        if st.button(aa, key=f"rep_p_{aa}", help=AMINO_ACIDS[aa]['name']):
                            seq_list = list(current_seq['sequence'])
                            old_aa = seq_list[edit_position - 1]
                            seq_list[edit_position - 1] = aa
                            st.session_state.sequences[selected_seq_idx]['sequence'] = ''.join(seq_list)
                            st.session_state.design_history.append({
                                'action': 'mutate',
                                'position': edit_position,
                                'from': old_aa,
                                'to': aa,
                                'sequence': current_seq['name']
                            })
                            st.rerun()

                st.markdown("*Charged:*")
                charged = ['D', 'E', 'K', 'R', 'H']
                c_cols = st.columns(5)
                for i, aa in enumerate(charged):
                    with c_cols[i]:
                        if st.button(aa, key=f"rep_c_{aa}", help=AMINO_ACIDS[aa]['name']):
                            seq_list = list(current_seq['sequence'])
                            old_aa = seq_list[edit_position - 1]
                            seq_list[edit_position - 1] = aa
                            st.session_state.sequences[selected_seq_idx]['sequence'] = ''.join(seq_list)
                            st.session_state.design_history.append({
                                'action': 'mutate',
                                'position': edit_position,
                                'from': old_aa,
                                'to': aa,
                                'sequence': current_seq['name']
                            })
                            st.rerun()

                st.markdown("*Special:*")
                special = ['G']
                if st.button('G', key="rep_g", help="Glycine - flexible"):
                    seq_list = list(current_seq['sequence'])
                    old_aa = seq_list[edit_position - 1]
                    seq_list[edit_position - 1] = 'G'
                    st.session_state.sequences[selected_seq_idx]['sequence'] = ''.join(seq_list)
                    st.session_state.design_history.append({
                        'action': 'mutate',
                        'position': edit_position,
                        'from': old_aa,
                        'to': 'G',
                        'sequence': current_seq['name']
                    })
                    st.rerun()

            # Design history
            if st.session_state.design_history:
                with st.expander("📜 Design History"):
                    for i, entry in enumerate(reversed(st.session_state.design_history[-10:])):
                        st.text(f"{entry['sequence']}: {entry['position']}{entry['from']}→{entry['to']}")
        else:
            st.info("No sequences loaded. Upload a FASTA file or create a new sequence.")

# === LIGAND DESIGN TAB ===
with tab_ligand:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Add Ligand")

        ligand_input = st.radio(
            "Input method",
            ["Enter SMILES", "Upload SMI file", "Common ligands"],
            horizontal=True,
            key="ligand_method"
        )

        if ligand_input == "Enter SMILES":
            ligand_name = st.text_input("Ligand name", value="ligand_1", key="lig_name")
            smiles_input = st.text_input(
                "SMILES string",
                placeholder="CCO for ethanol, CC(=O)O for acetic acid...",
                key="smiles_input"
            )

            if st.button("Add Ligand", key="add_ligand"):
                if smiles_input:
                    st.session_state.ligands.append({
                        'name': ligand_name,
                        'smiles': smiles_input,
                        'type': 'ligand'
                    })
                    st.success(f"Added ligand: {ligand_name}")
                    st.rerun()

        elif ligand_input == "Upload SMI file":
            smi_file = st.file_uploader(
                "Upload SMI/SMILES file",
                type=["smi", "smiles", "txt"],
                key="smi_upload"
            )
            if smi_file:
                content = smi_file.read().decode("utf-8")
                for line in content.strip().split('\n'):
                    parts = line.split()
                    if parts:
                        smiles = parts[0]
                        name = parts[1] if len(parts) > 1 else f"ligand_{len(st.session_state.ligands)+1}"
                        st.session_state.ligands.append({
                            'name': name,
                            'smiles': smiles,
                            'type': 'ligand'
                        })
                st.success(f"Loaded {len(st.session_state.ligands)} ligand(s)")
                st.rerun()

        else:  # Common ligands
            common_ligands = {
                "ATP": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
                "NAD+": "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)c1",
                "Heme": "CC1=C(CCC(=O)O)C2=Cc3c(C)c(C=C)c4C=C5C(C)=C(C=C)C6=CC7=C(CCC(=O)O)C(C)=C8C=c1n2[Fe]n34n56n78",
                "FAD": "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@H]([C@H](O)[C@@H]3O)n3cnc4c(N)ncnc43)c2cc1C",
                "Glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
                "Acetyl-CoA (simplified)": "CC(=O)SCCNC(=O)CCNC(=O)O",
                "Water": "O",
                "Ethanol": "CCO",
                "Benzene": "c1ccccc1",
            }

            selected_common = st.selectbox(
                "Select common ligand",
                list(common_ligands.keys()),
                key="common_lig"
            )

            st.code(common_ligands[selected_common], language=None)

            if st.button("Add Selected Ligand", key="add_common"):
                st.session_state.ligands.append({
                    'name': selected_common.replace(" ", "_").lower(),
                    'smiles': common_ligands[selected_common],
                    'type': 'ligand'
                })
                st.success(f"Added: {selected_common}")
                st.rerun()

    with col2:
        st.subheader("Ligand List")

        if st.session_state.ligands:
            for i, lig in enumerate(st.session_state.ligands):
                with st.container():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{lig['name']}**")
                        st.code(lig['smiles'][:50] + ("..." if len(lig['smiles']) > 50 else ""))
                    with col_b:
                        if st.button("🗑️", key=f"del_lig_{i}"):
                            st.session_state.ligands.pop(i)
                            st.rerun()
                    st.markdown("---")
        else:
            st.info("No ligands added yet.")

        # Try to visualize with rdkit if available
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            import io

            st.subheader("Ligand Visualization")
            if st.session_state.ligands:
                for lig in st.session_state.ligands[:3]:  # Show first 3
                    mol = Chem.MolFromSmiles(lig['smiles'])
                    if mol:
                        img = Draw.MolToImage(mol, size=(300, 200))
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        st.image(buf.getvalue(), caption=lig['name'])
        except ImportError:
            st.info("Install RDKit for ligand visualization: `pip install rdkit`")

# === PREVIEW & EXPORT TAB ===
with tab_preview:
    st.subheader("Design Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Sequences")
        if st.session_state.sequences:
            for seq in st.session_state.sequences:
                st.markdown(f"**{seq['name']}** ({seq['type']})")
                st.text(f"Length: {len(seq['sequence'])} residues")
                st.code(seq['sequence'][:80] + ("..." if len(seq['sequence']) > 80 else ""))
        else:
            st.info("No sequences designed")

    with col2:
        st.markdown("### Ligands")
        if st.session_state.ligands:
            for lig in st.session_state.ligands:
                st.markdown(f"**{lig['name']}**")
                st.code(lig['smiles'])
        else:
            st.info("No ligands added")

    st.markdown("---")
    st.subheader("Export & Run")

    col_export1, col_export2, col_export3 = st.columns(3)

    with col_export1:
        # Export as FASTA
        if st.session_state.sequences:
            fasta_content = ""
            for seq in st.session_state.sequences:
                fasta_content += f">{seq['name']}\n{seq['sequence']}\n"

            st.download_button(
                "📥 Download FASTA",
                data=fasta_content,
                file_name="designed_sequences.fasta",
                mime="text/plain"
            )

    with col_export2:
        # Export ligands as SMILES
        if st.session_state.ligands:
            smi_content = ""
            for lig in st.session_state.ligands:
                smi_content += f"{lig['smiles']} {lig['name']}\n"

            st.download_button(
                "📥 Download SMILES",
                data=smi_content,
                file_name="designed_ligands.smi",
                mime="text/plain"
            )

    with col_export3:
        # Export as JSON
        export_data = {
            'sequences': st.session_state.sequences,
            'ligands': st.session_state.ligands,
            'design_history': st.session_state.design_history
        }
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name="design_export.json",
            mime="application/json"
        )

    st.markdown("---")

    # Run prediction
    st.subheader("🚀 Run Prediction")

    if st.session_state.sequences or st.session_state.ligands:
        predictor_choice = st.selectbox(
            "Select predictor",
            ["Chai-1 (supports ligands)", "Boltz-2 (supports ligands)", "ColabFold (protein only)", "All predictors"],
            key="pred_choice"
        )

        output_dir = st.text_input("Output directory", value="./outputs/design_job")

        if st.button("🚀 Run Structure Prediction", type="primary", use_container_width=True):
            st.info("Preparing prediction job...")

            # Create combined input for predictors that support ligands
            try:
                from protein_design_hub.pipeline.runner import SequentialPipelineRunner
                from protein_design_hub.core.types import PredictionInput, Sequence, MoleculeType
                from protein_design_hub.core.config import get_settings
                from datetime import datetime

                settings = get_settings()

                # Build sequence list
                seq_list = []
                for seq in st.session_state.sequences:
                    mol_type = MoleculeType.PROTEIN
                    if seq['type'] == 'dna':
                        mol_type = MoleculeType.DNA
                    elif seq['type'] == 'rna':
                        mol_type = MoleculeType.RNA

                    seq_list.append(Sequence(
                        id=seq['name'],
                        sequence=seq['sequence'],
                        molecule_type=mol_type
                    ))

                # Add ligands
                for lig in st.session_state.ligands:
                    seq_list.append(Sequence(
                        id=lig['name'],
                        sequence=lig['smiles'],
                        molecule_type=MoleculeType.LIGAND
                    ))

                job_id = f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                job_dir = Path(output_dir)
                job_dir.mkdir(parents=True, exist_ok=True)

                prediction_input = PredictionInput(
                    job_id=job_id,
                    sequences=seq_list,
                    output_dir=job_dir,
                )

                # Determine predictors
                if "Chai-1" in predictor_choice:
                    predictors = ["chai1"]
                elif "Boltz-2" in predictor_choice:
                    predictors = ["boltz2"]
                elif "ColabFold" in predictor_choice:
                    predictors = ["colabfold"]
                else:
                    predictors = ["colabfold", "chai1", "boltz2"]

                runner = SequentialPipelineRunner(settings)

                progress = st.progress(0)
                status = st.empty()

                results = {}
                for i, pred_name in enumerate(predictors):
                    status.text(f"Running {pred_name}...")
                    progress.progress(i / len(predictors))

                    try:
                        result = runner.run_single_predictor(pred_name, prediction_input)
                        results[pred_name] = result

                        if result.success:
                            st.success(f"✓ {pred_name}: {len(result.structure_paths)} structures")
                        else:
                            st.error(f"✗ {pred_name}: {result.error_message}")
                    except Exception as e:
                        st.warning(f"⊘ {pred_name}: {e}")

                    progress.progress((i + 1) / len(predictors))

                status.text("Complete!")
                st.success(f"Results saved to: {job_dir}")

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Add sequences or ligands to run prediction")

# Amino acid reference
with st.expander("📖 Amino Acid Reference"):
    st.markdown("### Standard Amino Acids")

    aa_data = []
    for code, info in AMINO_ACIDS.items():
        aa_data.append({
            '1-letter': code,
            '3-letter': info['code'],
            'Name': info['name'],
            'Properties': info['properties']
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(aa_data), use_container_width=True)
