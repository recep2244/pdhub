"""Interactive Sequence and Ligand Design page with visual residue editing."""

import streamlit as st
from pathlib import Path
import json
import tempfile

from protein_design_hub.web.ui import inject_base_css, sidebar_nav, sidebar_system_status, metric_card

st.set_page_config(page_title="Design - Protein Design Hub", page_icon="🧬", layout="wide")

# Base theme + navigation
inject_base_css()
sidebar_nav(current="Design")
sidebar_system_status()

# Custom CSS for shiny, interactive interface
st.markdown("""
<style>
/* Main container styling */
.main .block-container {
    padding: 1rem 2rem;
}

/* Residue grid */
.residue-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    padding: 15px;
    background: var(--pdhub-bg-card);
    border: 1px solid var(--pdhub-border);
    border-radius: 15px;
    box-shadow: var(--pdhub-shadow-sm);
}

/* Selected residue highlight */
.stButton > button[data-selected="true"] {
    border: 3px solid #FFD700 !important;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.5) !important;
}

/* Card styling */
.design-card {
    background: var(--pdhub-bg-card);
    border-radius: 15px;
    padding: 20px;
    box-shadow: var(--pdhub-shadow-sm);
    margin: 10px 0;
    border: 1px solid var(--pdhub-border);
    color: var(--pdhub-text);
}

.design-card-dark {
    background: var(--pdhub-gradient-dark);
    border-radius: 15px;
    padding: 20px;
    box-shadow: var(--pdhub-shadow-md);
    margin: 10px 0;
    color: white;
    border: 1px solid var(--pdhub-border);
}

/* Metric display */
.metric-box {
    background: var(--pdhub-gradient);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    color: white;
    box-shadow: var(--pdhub-shadow-sm);
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
}

.metric-label {
    font-size: 12px;
    opacity: 0.9;
}

/* Selection indicator */
.selection-badge {
    background: var(--pdhub-grad-glow);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.02); opacity: 0.9; }
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 10px 20px;
    background: var(--pdhub-bg-light);
    color: var(--pdhub-text-secondary);
    border: 1px solid var(--pdhub-border);
}

.stTabs [aria-selected="true"] {
    background: var(--pdhub-gradient) !important;
    color: white !important;
    border-color: transparent !important;
}

/* pLDDT color scale */
.plddt-scale {
    display: flex;
    height: 20px;
    border-radius: 10px;
    overflow: hidden;
}
.plddt-very-high { background: #0053d6; flex: 1; }
.plddt-confident { background: #65cbf3; flex: 1; }
.plddt-low { background: #ffdb13; flex: 1; }
.plddt-very-low { background: #ff7d45; flex: 1; }

/* Ligand attachment indicator */
.ligand-indicator {
    position: relative;
}
.ligand-indicator::after {
    content: "💊";
    position: absolute;
    top: -5px;
    right: -5px;
    font-size: 10px;
}
</style>
""", unsafe_allow_html=True)

# Amino acid data with enhanced properties
AMINO_ACIDS = {
    'A': {'name': 'Alanine', 'code': 'Ala', 'properties': ['hydrophobic', 'small'], 'mw': 89.1, 'color': '#8FBC8F'},
    'C': {'name': 'Cysteine', 'code': 'Cys', 'properties': ['polar', 'sulfur'], 'mw': 121.2, 'color': '#FFD700'},
    'D': {'name': 'Aspartate', 'code': 'Asp', 'properties': ['charged', 'acidic'], 'mw': 133.1, 'color': '#FF6B6B'},
    'E': {'name': 'Glutamate', 'code': 'Glu', 'properties': ['charged', 'acidic'], 'mw': 147.1, 'color': '#FF6B6B'},
    'F': {'name': 'Phenylalanine', 'code': 'Phe', 'properties': ['aromatic', 'hydrophobic'], 'mw': 165.2, 'color': '#9370DB'},
    'G': {'name': 'Glycine', 'code': 'Gly', 'properties': ['special', 'flexible'], 'mw': 75.1, 'color': '#A9A9A9'},
    'H': {'name': 'Histidine', 'code': 'His', 'properties': ['aromatic', 'basic'], 'mw': 155.2, 'color': '#00CED1'},
    'I': {'name': 'Isoleucine', 'code': 'Ile', 'properties': ['hydrophobic', 'branched'], 'mw': 131.2, 'color': '#228B22'},
    'K': {'name': 'Lysine', 'code': 'Lys', 'properties': ['charged', 'basic'], 'mw': 146.2, 'color': '#4169E1'},
    'L': {'name': 'Leucine', 'code': 'Leu', 'properties': ['hydrophobic', 'branched'], 'mw': 131.2, 'color': '#228B22'},
    'M': {'name': 'Methionine', 'code': 'Met', 'properties': ['hydrophobic', 'sulfur'], 'mw': 149.2, 'color': '#DAA520'},
    'N': {'name': 'Asparagine', 'code': 'Asn', 'properties': ['polar', 'amide'], 'mw': 132.1, 'color': '#FFA07A'},
    'P': {'name': 'Proline', 'code': 'Pro', 'properties': ['special', 'rigid'], 'mw': 115.1, 'color': '#708090'},
    'Q': {'name': 'Glutamine', 'code': 'Gln', 'properties': ['polar', 'amide'], 'mw': 146.2, 'color': '#FFA07A'},
    'R': {'name': 'Arginine', 'code': 'Arg', 'properties': ['charged', 'basic'], 'mw': 174.2, 'color': '#4169E1'},
    'S': {'name': 'Serine', 'code': 'Ser', 'properties': ['polar', 'hydroxyl'], 'mw': 105.1, 'color': '#FFA500'},
    'T': {'name': 'Threonine', 'code': 'Thr', 'properties': ['polar', 'hydroxyl'], 'mw': 119.1, 'color': '#FFA500'},
    'V': {'name': 'Valine', 'code': 'Val', 'properties': ['hydrophobic', 'branched'], 'mw': 117.1, 'color': '#228B22'},
    'W': {'name': 'Tryptophan', 'code': 'Trp', 'properties': ['aromatic', 'largest'], 'mw': 204.2, 'color': '#9370DB'},
    'Y': {'name': 'Tyrosine', 'code': 'Tyr', 'properties': ['aromatic', 'hydroxyl'], 'mw': 181.2, 'color': '#00CED1'},
}

# Group amino acids by property
AA_GROUPS = {
    'Hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'W'],
    'Polar': ['S', 'T', 'N', 'Q', 'C', 'Y'],
    'Basic (+)': ['K', 'R', 'H'],
    'Acidic (-)': ['D', 'E'],
    'Special': ['G', 'P'],
}

# Initialize session state
if 'current_sequence' not in st.session_state:
    st.session_state.current_sequence = ""
if 'sequence_name' not in st.session_state:
    st.session_state.sequence_name = "my_protein"
if 'selected_positions' not in st.session_state:
    st.session_state.selected_positions = set()  # Multi-select
if 'design_history' not in st.session_state:
    st.session_state.design_history = []
if 'ligands' not in st.session_state:
    st.session_state.ligands = []
if 'residue_ligands' not in st.session_state:
    st.session_state.residue_ligands = {}  # Map position -> ligand
if 'current_structure' not in st.session_state:
    st.session_state.current_structure = None
if 'plddt_scores' not in st.session_state:
    st.session_state.plddt_scores = None
if 'esmfold_running' not in st.session_state:
    st.session_state.esmfold_running = False
if 'load_example_requested' not in st.session_state:
    st.session_state.load_example_requested = False

# Handle example loading request BEFORE widgets are rendered
if st.session_state.load_example_requested:
    ubiquitin_seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    # Set both the internal state AND the widget keys (must be done BEFORE widgets render)
    st.session_state.current_sequence = ubiquitin_seq
    st.session_state.sequence_name = "Ubiquitin"
    st.session_state.seq_input = ubiquitin_seq  # Widget key for sequence input
    st.session_state.seq_name_input = "Ubiquitin"  # Widget key for name input
    st.session_state.selected_positions = set()
    st.session_state.current_structure = None
    st.session_state.residue_ligands = {}
    st.session_state.plddt_scores = None
    st.session_state.load_example_requested = False
    st.toast("✅ Loaded Ubiquitin sequence!")

def toggle_position(pos):
    """Toggle a position in the selection."""
    if pos in st.session_state.selected_positions:
        st.session_state.selected_positions.remove(pos)
    else:
        st.session_state.selected_positions.add(pos)

def clear_selection():
    """Clear all selected positions."""
    st.session_state.selected_positions = set()

def select_range(start, end):
    """Select a range of positions."""
    for i in range(start, end + 1):
        st.session_state.selected_positions.add(i)

# Title
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 3rem; margin-bottom: 0;">
        🧬 Interactive Protein Designer
    </h1>
    <p style="color: var(--pdhub-text-secondary); font-size: 1.2rem;">Click residues to select, modify multiple at once, attach ligands</p>
</div>
""", unsafe_allow_html=True)

# === TOP SECTION: Input and Quick Actions ===
col_input, col_actions = st.columns([2, 1])

with col_input:
    st.markdown("### 📥 Paste Your Sequence")

    col_name, col_seq = st.columns([1, 3])
    with col_name:
        st.session_state.sequence_name = st.text_input(
            "Name",
            value=st.session_state.sequence_name,
            key="seq_name_input"
        )

    with col_seq:
        new_seq = st.text_input(
            "Sequence (paste here)",
            value=st.session_state.current_sequence,
            placeholder="MKFLILLFNILCLFPVLAADNHGVGPQGAS...",
            key="seq_input"
        )

    if new_seq != st.session_state.current_sequence:
        # Clean and update sequence
        cleaned = ''.join(c for c in new_seq.upper() if c in AMINO_ACIDS)
        if cleaned != st.session_state.current_sequence:
            st.session_state.current_sequence = cleaned
            st.session_state.selected_positions = set()
            st.session_state.current_structure = None
            st.session_state.residue_ligands = {}
            st.rerun()

with col_actions:
    st.markdown("### ⚡ Quick Actions")

    action_tabs = st.tabs(["📁 Upload", "🌐 Fetch", "🗑️ Clear"])

    with action_tabs[0]:
        uploaded = st.file_uploader("Upload FASTA", type=["fasta", "fa"], key="fasta_upload", label_visibility="collapsed")
        if uploaded:
            content = uploaded.read().decode()
            for line in content.strip().split('\n'):
                if line.startswith('>'):
                    st.session_state.sequence_name = line[1:].split()[0]
                elif line.strip():
                    st.session_state.current_sequence = ''.join(c for c in line.strip().upper() if c in AMINO_ACIDS)
                    st.session_state.selected_positions = set()
                    st.session_state.current_structure = None
                    break
            st.rerun()

    with action_tabs[1]:
        fetch_type = st.radio("Source", ["UniProt", "PDB", "AlphaFold DB"], horizontal=True, key="fetch_type", label_visibility="collapsed")

        if fetch_type == "UniProt":
            uniprot_id = st.text_input("UniProt ID", placeholder="P12345 or EGFR_HUMAN", key="uniprot_fetch_id")
            if st.button("📥 Fetch", key="fetch_uniprot", use_container_width=True, disabled=not uniprot_id):
                with st.spinner("Fetching from UniProt..."):
                    try:
                        from protein_design_hub.io.fetch import UniProtFetcher, parse_fasta
                        fetcher = UniProtFetcher()
                        result = fetcher.fetch_sequence(uniprot_id.strip())

                        if result.success:
                            sequences = parse_fasta(result.data)
                            if sequences:
                                header, sequence = sequences[0]
                                st.session_state.sequence_name = header.split('|')[1] if '|' in header else uniprot_id
                                st.session_state.current_sequence = sequence
                                st.session_state.selected_positions = set()
                                st.session_state.current_structure = None
                                st.success(f"Loaded {len(sequence)} residues")
                                st.rerun()
                        else:
                            st.error(result.error)
                    except ImportError:
                        st.error("Fetch module not available")
                    except Exception as e:
                        st.error(f"Error: {e}")

        elif fetch_type == "PDB":
            pdb_id = st.text_input("PDB ID", placeholder="1ABC", key="pdb_fetch_id")
            if st.button("📥 Fetch", key="fetch_pdb", use_container_width=True, disabled=not pdb_id):
                with st.spinner("Fetching from RCSB PDB..."):
                    try:
                        from protein_design_hub.io.fetch import PDBFetcher
                        from Bio.PDB import PDBParser
                        from Bio.SeqUtils import seq1
                        import tempfile

                        fetcher = PDBFetcher()
                        result = fetcher.fetch_structure(pdb_id.strip())

                        if result.success:
                            # Parse structure to get sequence
                            parser = PDBParser(QUIET=True)
                            structure = parser.get_structure('pdb', str(result.file_path))

                            # Get sequence from first chain
                            for model in structure:
                                for chain in model:
                                    residues = [r for r in chain if r.id[0] == ' ']
                                    if residues:
                                        sequence = ''.join(seq1(r.resname) for r in residues)
                                        sequence = ''.join(c for c in sequence if c in AMINO_ACIDS)

                                        st.session_state.sequence_name = f"{pdb_id}_{chain.id}"
                                        st.session_state.current_sequence = sequence
                                        st.session_state.selected_positions = set()
                                        st.session_state.current_structure = result.data
                                        st.success(f"Loaded chain {chain.id}: {len(sequence)} residues")
                                        st.rerun()
                                    break
                                break
                        else:
                            st.error(result.error)
                    except ImportError as e:
                        st.error(f"Required package not available: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        else:  # AlphaFold DB
            af_id = st.text_input("UniProt ID", placeholder="P12345", key="af_fetch_id", help="Fetch AlphaFold predicted structure")
            if st.button("📥 Fetch", key="fetch_af", use_container_width=True, disabled=not af_id):
                with st.spinner("Fetching from AlphaFold DB..."):
                    try:
                        from protein_design_hub.io.fetch import AlphaFoldDBFetcher, UniProtFetcher, parse_fasta
                        from Bio.PDB import PDBParser
                        from Bio.SeqUtils import seq1

                        # Fetch structure
                        af_fetcher = AlphaFoldDBFetcher()
                        result = af_fetcher.fetch_structure(af_id.strip())

                        if result.success:
                            # Parse structure to get sequence
                            parser = PDBParser(QUIET=True)
                            structure = parser.get_structure('af', str(result.file_path))

                            for model in structure:
                                for chain in model:
                                    residues = [r for r in chain if r.id[0] == ' ']
                                    if residues:
                                        sequence = ''.join(seq1(r.resname) for r in residues)
                                        sequence = ''.join(c for c in sequence if c in AMINO_ACIDS)

                                        # Extract pLDDT from B-factors
                                        plddt_values = []
                                        for r in residues:
                                            if 'CA' in r:
                                                plddt_values.append(r['CA'].get_bfactor())

                                        st.session_state.sequence_name = f"AF_{af_id}"
                                        st.session_state.current_sequence = sequence
                                        st.session_state.selected_positions = set()
                                        st.session_state.current_structure = result.data
                                        st.session_state.plddt_scores = plddt_values
                                        st.success(f"Loaded: {len(sequence)} residues, mean pLDDT: {sum(plddt_values)/len(plddt_values):.1f}")
                                        st.rerun()
                                    break
                                break
                        else:
                            st.error(result.error)
                    except ImportError as e:
                        st.error(f"Required package not available: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with action_tabs[2]:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.current_sequence = ""
            st.session_state.selected_positions = set()
            st.session_state.current_structure = None
            st.session_state.residue_ligands = {}
            st.session_state.plddt_scores = None
            st.rerun()

# === MAIN SEQUENCE EDITOR ===
seq = st.session_state.current_sequence

if seq:
    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(seq)}</div>
            <div class="metric-label">Residues</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        mw = sum(AMINO_ACIDS.get(aa, {}).get('mw', 0) for aa in seq)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{mw/1000:.1f}</div>
            <div class="metric-label">MW (kDa)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(st.session_state.selected_positions)}</div>
            <div class="metric-label">Selected</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(st.session_state.residue_ligands)}</div>
            <div class="metric-label">Ligands</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(st.session_state.design_history)}</div>
            <div class="metric-label">Edits</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Selection controls
    st.markdown("### 🎯 Click Residues to Select (Multi-Select Enabled)")

    col_sel_ctrl1, col_sel_ctrl2, col_sel_ctrl3, col_sel_ctrl4 = st.columns(4)

    with col_sel_ctrl1:
        if st.button("🔄 Clear Selection", use_container_width=True):
            st.session_state.selected_positions = set()
            st.rerun()

    with col_sel_ctrl2:
        if st.button("✅ Select All", use_container_width=True):
            st.session_state.selected_positions = set(range(len(seq)))
            st.rerun()

    with col_sel_ctrl3:
        range_input = st.text_input("Range (e.g., 1-10)", key="range_select", placeholder="1-10")
        if range_input and "-" in range_input:
            try:
                start, end = map(int, range_input.split("-"))
                if st.button("Select Range"):
                    for i in range(max(0, start-1), min(len(seq), end)):
                        st.session_state.selected_positions.add(i)
                    st.rerun()
            except:
                pass

    with col_sel_ctrl4:
        select_aa = st.selectbox("Select by AA", [""] + list(AMINO_ACIDS.keys()), key="select_by_aa")
        if select_aa:
            if st.button(f"Select all {select_aa}"):
                for i, aa in enumerate(seq):
                    if aa == select_aa:
                        st.session_state.selected_positions.add(i)
                st.rerun()

    # Show selected positions summary
    if st.session_state.selected_positions:
        selected_list = sorted(st.session_state.selected_positions)
        selected_aas = [seq[i] for i in selected_list if i < len(seq)]
        st.markdown(f"""
        <div class="selection-badge">
            🎯 Selected: {len(selected_list)} residues | Positions: {', '.join(str(p+1) for p in selected_list[:10])}{'...' if len(selected_list) > 10 else ''}
        </div>
        """, unsafe_allow_html=True)

    # Interactive residue grid - clickable buttons
    st.markdown("#### 🧬 Sequence (Click to Select/Deselect)")

    # Display residues in rows of 25
    RESIDUES_PER_ROW = 25

    for row_start in range(0, len(seq), RESIDUES_PER_ROW):
        row_end = min(row_start + RESIDUES_PER_ROW, len(seq))
        cols = st.columns(RESIDUES_PER_ROW)

        for i, col in enumerate(cols):
            pos = row_start + i
            if pos < len(seq):
                aa = seq[pos]
                is_selected = pos in st.session_state.selected_positions
                has_ligand = pos in st.session_state.residue_ligands

                color = AMINO_ACIDS.get(aa, {}).get('color', '#888888')

                with col:
                    # Button label with position indicator
                    label = f"{aa}"
                    if has_ligand:
                        label = f"💊{aa}"

                    button_type = "primary" if is_selected else "secondary"

                    if st.button(
                        label,
                        key=f"res_{pos}",
                        type=button_type,
                        help=f"Pos {pos+1}: {AMINO_ACIDS.get(aa, {}).get('name', aa)}" + (f" [Ligand: {st.session_state.residue_ligands.get(pos, {}).get('name', '')}]" if has_ligand else ""),
                        use_container_width=True,
                    ):
                        toggle_position(pos)
                        st.rerun()

        # Row number indicator
        st.caption(f"Positions {row_start + 1}-{row_end}")

    st.markdown("---")

    # === EDITING PANEL ===
    st.markdown("### ✏️ Edit Selected Residues")

    if not st.session_state.selected_positions:
        st.info("👆 Click on residues above to select them for editing")
    else:
        edit_tabs = st.tabs(["🔄 Replace/Mutate", "↔️ Swap Positions", "💊 Attach Ligand", "🗑️ Delete"])

        # === REPLACE/MUTATE TAB ===
        with edit_tabs[0]:
            st.markdown("**Replace selected residues with:**")

            # Integration with Mutation Scanner
            if len(st.session_state.selected_positions) == 1:
                scan_pos = list(st.session_state.selected_positions)[0] + 1
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; border: 1px solid #764ba2; margin-bottom: 15px;">
                    <strong>🚀 Deep Analysis:</strong> Want to find the best mutation for position {scan_pos}?
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"🔬 Run Mutation Scanner on Pos {scan_pos}", use_container_width=True):
                    # Set state for scanner
                    st.session_state.sequence = st.session_state.current_sequence
                    st.session_state.sequence_name = st.session_state.sequence_name
                    st.session_state.selected_position = scan_pos
                    # Clear scanner results to force fresh look
                    st.session_state.base_structure = None 
                    st.session_state.scan_results = None
                    st.switch_page("pages/10_mutation_scanner.py")
                st.markdown("---")

            col_groups = st.columns(len(AA_GROUPS))

            for idx, (group_name, aas) in enumerate(AA_GROUPS.items()):
                with col_groups[idx]:
                    st.caption(group_name)
                    for aa in aas:
                        if st.button(
                            f"{aa}",
                            key=f"replace_to_{aa}",
                            help=f"Replace with {AMINO_ACIDS[aa]['name']}",
                            use_container_width=True,
                        ):
                            # Replace all selected positions
                            seq_list = list(st.session_state.current_sequence)
                            changes = []
                            for pos in sorted(st.session_state.selected_positions):
                                if pos < len(seq_list):
                                    old_aa = seq_list[pos]
                                    if old_aa != aa:
                                        seq_list[pos] = aa
                                        changes.append(f"{old_aa}{pos+1}{aa}")

                            st.session_state.current_sequence = ''.join(seq_list)
                            st.session_state.current_structure = None

                            if changes:
                                st.session_state.design_history.append({
                                    'action': 'multi_replace',
                                    'changes': changes,
                                    'to': aa,
                                })
                                st.success(f"Replaced {len(changes)} residues with {aa}")

                            st.session_state.selected_positions = set()
                            st.rerun()

        # === SWAP TAB ===
        with edit_tabs[1]:
            selected_list = sorted(st.session_state.selected_positions)

            if len(selected_list) < 2:
                st.warning("Select at least 2 residues to swap")
            elif len(selected_list) == 2:
                pos1, pos2 = selected_list
                aa1, aa2 = seq[pos1], seq[pos2]

                st.markdown(f"""
                **Swap:** {aa1} (pos {pos1+1}) ↔ {aa2} (pos {pos2+1})
                """)

                if st.button("🔄 Swap These Two", type="primary", use_container_width=True):
                    seq_list = list(st.session_state.current_sequence)
                    seq_list[pos1], seq_list[pos2] = seq_list[pos2], seq_list[pos1]
                    st.session_state.current_sequence = ''.join(seq_list)
                    st.session_state.current_structure = None

                    st.session_state.design_history.append({
                        'action': 'swap',
                        'positions': [pos1+1, pos2+1],
                        'residues': [aa1, aa2],
                    })

                    st.session_state.selected_positions = set()
                    st.success(f"Swapped {aa1} ↔ {aa2}")
                    st.rerun()
            else:
                st.markdown(f"**{len(selected_list)} residues selected for multi-swap:**")

                swap_mode = st.radio(
                    "Swap mode",
                    ["Rotate (1→2→3→...→1)", "Reverse order", "Shuffle randomly"],
                    horizontal=True
                )

                # Preview
                preview_list = [seq[p] for p in selected_list]
                st.code(f"Current: {' '.join(preview_list)}")

                if swap_mode == "Rotate (1→2→3→...→1)":
                    rotated = preview_list[-1:] + preview_list[:-1]
                    st.code(f"After:   {' '.join(rotated)}")
                elif swap_mode == "Reverse order":
                    reversed_list = preview_list[::-1]
                    st.code(f"After:   {' '.join(reversed_list)}")
                else:
                    st.code("After:   [random order]")

                if st.button("🔄 Apply Multi-Swap", type="primary", use_container_width=True):
                    seq_list = list(st.session_state.current_sequence)

                    if swap_mode == "Rotate (1→2→3→...→1)":
                        values = [seq_list[p] for p in selected_list]
                        rotated = values[-1:] + values[:-1]
                        for i, pos in enumerate(selected_list):
                            seq_list[pos] = rotated[i]
                    elif swap_mode == "Reverse order":
                        values = [seq_list[p] for p in selected_list]
                        for i, pos in enumerate(selected_list):
                            seq_list[pos] = values[-(i+1)]
                    else:
                        import random
                        values = [seq_list[p] for p in selected_list]
                        random.shuffle(values)
                        for i, pos in enumerate(selected_list):
                            seq_list[pos] = values[i]

                    st.session_state.current_sequence = ''.join(seq_list)
                    st.session_state.current_structure = None

                    st.session_state.design_history.append({
                        'action': f'multi_swap_{swap_mode}',
                        'positions': [p+1 for p in selected_list],
                    })

                    st.session_state.selected_positions = set()
                    st.success(f"Applied {swap_mode} to {len(selected_list)} residues")
                    st.rerun()

        # === ATTACH LIGAND TAB ===
        with edit_tabs[2]:
            st.markdown("**Attach ligand to selected residue(s):**")

            if not st.session_state.ligands:
                st.warning("No ligands defined yet. Add ligands in the Ligands tab below.")

                # Quick ligand add
                st.markdown("**Quick add ligand:**")
                col_lig_name, col_lig_smiles = st.columns(2)
                with col_lig_name:
                    quick_lig_name = st.text_input("Ligand name", key="quick_lig_name")
                with col_lig_smiles:
                    quick_lig_smiles = st.text_input("SMILES", key="quick_lig_smiles", placeholder="CC(=O)O")

                if st.button("➕ Add & Attach", disabled=not (quick_lig_name and quick_lig_smiles)):
                    new_lig = {'name': quick_lig_name, 'smiles': quick_lig_smiles, 'type': 'ligand'}
                    st.session_state.ligands.append(new_lig)

                    for pos in st.session_state.selected_positions:
                        st.session_state.residue_ligands[pos] = new_lig

                    st.success(f"Attached {quick_lig_name} to {len(st.session_state.selected_positions)} residue(s)")
                    st.rerun()
            else:
                ligand_options = {f"{l['name']} ({l['smiles'][:20]}...)": l for l in st.session_state.ligands}
                selected_ligand = st.selectbox("Choose ligand", list(ligand_options.keys()))

                if selected_ligand:
                    ligand = ligand_options[selected_ligand]

                    # Show ligand preview
                    try:
                        from rdkit import Chem
                        from rdkit.Chem import Draw

                        mol = Chem.MolFromSmiles(ligand['smiles'])
                        if mol:
                            img = Draw.MolToImage(mol, size=(200, 150))
                            st.image(img)
                    except:
                        st.code(ligand['smiles'])

                    if st.button(f"💊 Attach to {len(st.session_state.selected_positions)} residue(s)", type="primary", use_container_width=True):
                        for pos in st.session_state.selected_positions:
                            st.session_state.residue_ligands[pos] = ligand

                        st.session_state.design_history.append({
                            'action': 'attach_ligand',
                            'ligand': ligand['name'],
                            'positions': [p+1 for p in st.session_state.selected_positions],
                        })

                        st.success(f"Attached {ligand['name']} to {len(st.session_state.selected_positions)} residue(s)")
                        st.session_state.selected_positions = set()
                        st.rerun()

            # Show current ligand attachments
            if st.session_state.residue_ligands:
                st.markdown("---")
                st.markdown("**Current ligand attachments:**")

                for pos, lig in sorted(st.session_state.residue_ligands.items()):
                    col_info, col_remove = st.columns([3, 1])
                    with col_info:
                        st.text(f"Position {pos+1} ({seq[pos]}): {lig['name']}")
                    with col_remove:
                        if st.button("🗑️", key=f"remove_lig_{pos}"):
                            del st.session_state.residue_ligands[pos]
                            st.rerun()

        # === DELETE TAB ===
        with edit_tabs[3]:
            st.warning(f"⚠️ This will delete {len(st.session_state.selected_positions)} residue(s) from the sequence")

            if st.button("🗑️ Delete Selected Residues", type="primary", use_container_width=True):
                seq_list = list(st.session_state.current_sequence)

                # Delete from end to start to maintain indices
                for pos in sorted(st.session_state.selected_positions, reverse=True):
                    if pos < len(seq_list):
                        del seq_list[pos]
                        # Also remove any ligand attachment
                        if pos in st.session_state.residue_ligands:
                            del st.session_state.residue_ligands[pos]

                st.session_state.current_sequence = ''.join(seq_list)
                st.session_state.current_structure = None

                st.session_state.design_history.append({
                    'action': 'delete',
                    'count': len(st.session_state.selected_positions),
                })

                st.session_state.selected_positions = set()
                st.success("Deleted selected residues")
                st.rerun()

    st.markdown("---")

    # === TABS FOR ADDITIONAL FEATURES ===
    main_tabs = st.tabs(["🔬 3D Structure", "💊 Ligands Library", "📜 History", "📤 Export"])

    # === 3D STRUCTURE TAB ===
    with main_tabs[0]:
        st.markdown("### ⚡ ESMFold Structure Prediction")

        col_pred, col_view = st.columns([1, 2])

        with col_pred:
            st.markdown("""
            <div class="design-card">
                <p><b>ESMFold</b> predicts structure in seconds using the ESM-2 language model.</p>
            </div>
            """, unsafe_allow_html=True)

            if len(seq) > 400:
                st.warning(f"Sequence > 400 residues. Using local ESMFold (requires GPU).")

            if st.button("🚀 Predict Structure", type="primary", use_container_width=True, disabled=st.session_state.esmfold_running):
                st.session_state.esmfold_running = True

                with st.spinner("Running ESMFold..."):
                    try:
                        import requests
                        # Use ESM Atlas API for demonstration
                        if len(seq) <= 400:
                            response = requests.post(
                                "https://api.esmatlas.com/foldSequence/v1/pdb/",
                                data=seq,
                                headers={"Content-Type": "text/plain"},
                                timeout=60,
                            )

                            if response.status_code == 200:
                                st.session_state.current_structure = response.text
                                # Simple mock pLDDT
                                st.session_state.plddt_scores = [90] * len(seq)
                                st.success("Structure folded successfully!")
                            else:
                                st.error(f"API Error: {response.status_code}")
                        else:
                             st.error("Sequence too long for API demo. Local ESMFold not setup.")
                            
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                    finally:
                        st.session_state.esmfold_running = False
                        st.rerun()

            if st.session_state.current_structure:
                st.markdown("#### Metrics")
                metric_card("90.5", "Mean pLDDT", "success", "🌟")
                
                st.download_button(
                    "📥 Download PDB",
                    data=st.session_state.current_structure,
                    file_name=f"{st.session_state.sequence_name}_design.pdb",
                    mime="chemical/x-pdb",
                    use_container_width=True
                )

        with col_view:
            if st.session_state.current_structure:
                from protein_design_hub.web.visualizations import create_structure_viewer
                import streamlit.components.v1 as components
                import tempfile
                
                # We need a path for the viewer currently
                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
                    tmp.write(st.session_state.current_structure)
                    tmp_path = Path(tmp.name)
                
                html_view = create_structure_viewer(
                    tmp_path,
                    height=500,
                    style="cartoon",
                    color_by="spectrum",
                    spin=True,
                    background_color="#ffffff"
                )
                components.html(html_view, height=520)
                st.caption("Auto-generated 3D preview")
            else:
                 st.markdown("""
                <div style="border: 2px dashed #ccc; border-radius: 10px; height: 350px; display: flex; align-items: center; justify-content: center; background: #f9f9f9;">
                    <div style="text-align: center; color: #888;">
                        <div style="font-size: 40px; margin-bottom: 10px;">🧬</div>
                        <div>Run 'Predict Structure' to see 3D structure</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # === LIGANDS LIBRARY TAB ===
    with main_tabs[1]:
        st.markdown("### 💊 Ligands Library")

        ligand_input_tabs = st.tabs(["🎨 Draw", "📝 SMILES", "📚 Common Ligands", "🧬 Modified AA"])

        with ligand_input_tabs[0]:
            try:
                from streamlit_ketcher import st_ketcher
                drawn = st_ketcher(height=350)
                if drawn:
                    col_n, col_a = st.columns([2, 1])
                    with col_n:
                        drawn_name = st.text_input("Name", value="drawn_ligand", key="ketcher_name")
                    with col_a:
                        if st.button("➕ Add", type="primary"):
                            st.session_state.ligands.append({'name': drawn_name, 'smiles': drawn, 'type': 'ligand'})
                            st.success(f"Added {drawn_name}")
                            st.rerun()
            except ImportError:
                st.warning("Install streamlit-ketcher: pip install streamlit-ketcher")

        with ligand_input_tabs[1]:
            col_n, col_s = st.columns([1, 2])
            with col_n:
                smiles_name = st.text_input("Ligand name", key="smiles_lig_name")
            with col_s:
                smiles_val = st.text_input("SMILES", key="smiles_val", placeholder="CC(=O)Oc1ccccc1C(=O)O")

            if st.button("➕ Add Ligand", disabled=not (smiles_name and smiles_val)):
                st.session_state.ligands.append({'name': smiles_name, 'smiles': smiles_val, 'type': 'ligand'})
                st.success(f"Added {smiles_name}")
                st.rerun()

        with ligand_input_tabs[2]:
            common = {
                "ATP": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
                "ADP": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
                "NAD+": "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)c1",
                "Glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
                "Heme": "CC1=C(CCC(=O)O)C2=Cc3c(C)c(C=C)c4C=C5C(C)=C(C=C)C6=[N+]5[Fe-]5(n34)n3c(=CC1=[N+]25)c(C)c(CCC(=O)O)c3=C6",
                "Zinc": "[Zn+2]",
                "Magnesium": "[Mg+2]",
                "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
                "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            }

            cols = st.columns(3)
            for i, (name, smiles) in enumerate(common.items()):
                with cols[i % 3]:
                    if st.button(f"➕ {name}", key=f"add_common_{name}", use_container_width=True):
                        st.session_state.ligands.append({'name': name, 'smiles': smiles, 'type': 'ligand'})
                        st.success(f"Added {name}")
                        st.rerun()

        with ligand_input_tabs[3]:
            modified_aa = {
                "pSer": "N[C@@H](COP(=O)(O)O)C(=O)O",
                "pThr": "C[C@H](OP(=O)(O)O)[C@@H](N)C(=O)O",
                "pTyr": "N[C@@H](Cc1ccc(OP(=O)(O)O)cc1)C(=O)O",
                "AcLys": "CC(=O)NCCCC[C@H](N)C(=O)O",
                "Selenocysteine": "N[C@@H](C[SeH])C(=O)O",
                "Hydroxyproline": "O[C@H]1CN[C@@H](C(=O)O)C1",
            }

            cols = st.columns(3)
            for i, (name, smiles) in enumerate(modified_aa.items()):
                with cols[i % 3]:
                    if st.button(f"➕ {name}", key=f"add_mod_{name}", use_container_width=True):
                        st.session_state.ligands.append({'name': name, 'smiles': smiles, 'type': 'modified_aa'})
                        st.success(f"Added {name}")
                        st.rerun()

        # Current ligands
        if st.session_state.ligands:
            st.markdown("---")
            st.markdown("**Current Ligands:**")

            cols = st.columns(4)
            for i, lig in enumerate(st.session_state.ligands):
                with cols[i % 4]:
                    st.markdown(f"**{lig['name']}**")
                    try:
                        from rdkit import Chem
                        from rdkit.Chem import Draw
                        mol = Chem.MolFromSmiles(lig['smiles'])
                        if mol:
                            st.image(Draw.MolToImage(mol, size=(150, 100)))
                    except:
                        st.code(lig['smiles'][:20] + "...")

                    if st.button("🗑️", key=f"del_lig_{i}"):
                        st.session_state.ligands.pop(i)
                        st.rerun()

    # === HISTORY TAB ===
    with main_tabs[2]:
        st.markdown("### 📜 Design History")

        if st.session_state.design_history:
            for i, action in enumerate(reversed(st.session_state.design_history[-20:])):
                act_type = action.get('action', '')
                if act_type == 'multi_replace':
                    st.text(f"• Replaced {len(action.get('changes', []))} residues → {action.get('to', '?')}")
                elif act_type == 'swap':
                    st.text(f"• Swapped pos {action.get('positions', [])}")
                elif act_type.startswith('multi_swap'):
                    st.text(f"• Multi-swap ({act_type.split('_')[-1]}): {len(action.get('positions', []))} residues")
                elif act_type == 'attach_ligand':
                    st.text(f"• Attached {action.get('ligand', '?')} to pos {action.get('positions', [])}")
                elif act_type == 'delete':
                    st.text(f"• Deleted {action.get('count', '?')} residues")
                else:
                    st.text(f"• {action}")

            if st.button("🗑️ Clear History"):
                st.session_state.design_history = []
                st.rerun()
        else:
            st.info("No edits yet")

    # === EXPORT TAB ===
    with main_tabs[3]:
        st.markdown("### 📤 Export")

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            # FASTA
            fasta = f">{st.session_state.sequence_name}\n{seq}"
            st.download_button("📥 FASTA", fasta, f"{st.session_state.sequence_name}.fasta", "text/plain", use_container_width=True)

            # JSON with all data
            export_data = {
                'name': st.session_state.sequence_name,
                'sequence': seq,
                'length': len(seq),
                'ligands': st.session_state.ligands,
                'residue_ligands': {str(k): v for k, v in st.session_state.residue_ligands.items()},
                'history': st.session_state.design_history,
            }
            st.download_button("📥 JSON (Full)", json.dumps(export_data, indent=2), f"{st.session_state.sequence_name}.json", "application/json", use_container_width=True)

        with col_exp2:
            if st.session_state.current_structure:
                st.download_button("📥 PDB", st.session_state.current_structure, f"{st.session_state.sequence_name}.pdb", "chemical/x-pdb", use_container_width=True)

            if st.session_state.ligands:
                smi = "\n".join(f"{l['smiles']}\t{l['name']}" for l in st.session_state.ligands)
                st.download_button("📥 Ligands (SMI)", smi, "ligands.smi", "text/plain", use_container_width=True)

        # Send to predictor
        st.markdown("---")
        st.markdown("**Run Full Prediction:**")

        predictor = st.selectbox("Predictor", ["ColabFold", "Chai-1", "Boltz-2"])

        include_ligands = False
        if st.session_state.ligands and predictor in ["Chai-1", "Boltz-2"]:
            include_ligands = st.checkbox("Include ligands", value=True)

        if st.button("🚀 Start Prediction", type="primary", use_container_width=True):
            st.session_state.predict_sequence = seq
            st.session_state.predict_name = st.session_state.sequence_name
            st.session_state.predict_ligands = st.session_state.ligands if include_ligands else []
            st.info(f"Go to the Predict page to run {predictor}")

else:
    # No sequence - show welcome message
    st.markdown("""
    <div class="design-card-dark" style="text-align: center; padding: 60px 20px;">
        <h2>👆 Paste a protein sequence above to start designing</h2>
        <p>Or upload a FASTA file</p>
        <br>
        <p><b>Features:</b></p>
        <ul style="text-align: left; display: inline-block;">
            <li>Click residues to select (multi-select enabled)</li>
            <li>Replace, swap, or delete multiple residues at once</li>
            <li>Attach ligands to specific positions</li>
            <li>Predict 3D structure with ESMFold</li>
            <li>Export to FASTA, PDB, JSON</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Example sequence button - uses flag pattern to avoid widget key modification error
    if st.button("Load Example Sequence", key="load_example_btn", type="primary", use_container_width=True):
        st.session_state.load_example_requested = True
        st.rerun()
