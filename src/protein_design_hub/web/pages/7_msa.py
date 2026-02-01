"""MSA and Evolutionary Analysis page."""

import streamlit as st
from pathlib import Path
import json

from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    section_header,
)

st.set_page_config(page_title="MSA Analysis - Protein Design Hub", page_icon="🧬", layout="wide")

# Base theme + navigation
inject_base_css()
sidebar_nav(current="MSA")
sidebar_system_status()

# Custom CSS
st.markdown("""
<style>
.conservation-high { background-color: #1a5276; color: white; }
.conservation-medium { background-color: #5dade2; color: white; }
.conservation-low { background-color: #aed6f1; color: black; }
.conservation-none { background-color: #f8f9fa; color: black; }
.msa-cell {
    display: inline-block;
    width: 20px;
    height: 24px;
    text-align: center;
    font-family: monospace;
    font-size: 12px;
    line-height: 24px;
}
.coevolution-card {
    background: var(--pdhub-gradient);
    border-radius: 12px;
    padding: 15px;
    color: white;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'msa_alignment' not in st.session_state:
    st.session_state.msa_alignment = None
if 'msa_names' not in st.session_state:
    st.session_state.msa_names = None

# Page Header
page_header(
    "MSA & Evolutionary Analysis",
    "Analyze conservation, coevolution, and ancestral sequences",
    "🧬"
)

# Main tabs
main_tabs = st.tabs(["📥 Input", "📊 Conservation", "🔗 Coevolution", "🌳 Ancestral", "📈 PSSM"])

# === INPUT TAB ===
with main_tabs[0]:
    section_header("Load MSA", "Upload or paste a multiple sequence alignment", "📥")

    col_method, col_info = st.columns([3, 1])
    with col_method:
        input_method = st.radio(
            "Input method",
            ["Upload file", "Paste alignment", "Fetch from UniProt"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with col_info:
        with st.popover("📖 Supported Formats"):
            st.markdown("""
            **Supported file formats:**
            - FASTA (.fasta, .fa)
            - A3M (.a3m)
            - Stockholm (.sto)
            - Clustal (.aln)
            """)

    if input_method == "Upload file":
        with st.container(border=True):
            uploaded = st.file_uploader(
                "Upload MSA file (FASTA, A3M, Stockholm, Clustal)",
                type=["fasta", "fa", "a3m", "sto", "aln"],
                help="Drag and drop or click to browse",
                label_visibility="collapsed"
            )

        if uploaded:
            content = uploaded.read().decode()

            # Parse FASTA-like format
            sequences = []
            names = []
            current_name = None
            current_seq = []

            for line in content.strip().split('\n'):
                line = line.strip()
                if line.startswith('>') or line.startswith('#=GS'):
                    if current_name and current_seq:
                        sequences.append(''.join(current_seq))
                        names.append(current_name)
                    current_name = line[1:].split()[0] if line.startswith('>') else line.split()[1]
                    current_seq = []
                elif line and not line.startswith('#'):
                    current_seq.append(line.replace(' ', ''))

            if current_name and current_seq:
                sequences.append(''.join(current_seq))
                names.append(current_name)

            if sequences:
                st.session_state.msa_alignment = sequences
                st.session_state.msa_names = names
                st.success(f"Loaded {len(sequences)} sequences, alignment length: {len(sequences[0])}")

    elif input_method == "Paste alignment":
        paste_input = st.text_area(
            "Paste alignment (FASTA format)",
            height=200,
            placeholder=">seq1\nMKFL--ILLFNI...\n>seq2\nMKFLILILLFNI..."
        )

        if paste_input:
            sequences = []
            names = []
            current_name = None
            current_seq = []

            for line in paste_input.strip().split('\n'):
                line = line.strip()
                if line.startswith('>'):
                    if current_name and current_seq:
                        sequences.append(''.join(current_seq))
                        names.append(current_name)
                    current_name = line[1:].split()[0]
                    current_seq = []
                elif line:
                    current_seq.append(line)

            if current_name and current_seq:
                sequences.append(''.join(current_seq))
                names.append(current_name)

            if sequences:
                st.session_state.msa_alignment = sequences
                st.session_state.msa_names = names
                st.success(f"Parsed {len(sequences)} sequences")

    else:  # Fetch from UniProt
        st.markdown("Generate MSA from UniProt using MMseqs2 or HHblits")

        uniprot_id = st.text_input("UniProt ID", placeholder="P12345")
        msa_method = st.selectbox("Method", ["MMseqs2 (fast)", "HHblits (sensitive)"])

        if st.button("🔍 Generate MSA", disabled=not uniprot_id):
            st.info("MSA generation would run MMseqs2/HHblits against UniRef databases")
            st.warning("This feature requires backend MSA generation tools to be installed")

    # Show alignment info
    if st.session_state.msa_alignment:
        alignment = st.session_state.msa_alignment
        names = st.session_state.msa_names

        st.markdown("---")
        st.markdown("### 📊 Alignment Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Sequences", len(alignment))
        with col2:
            st.metric("Length", len(alignment[0]))
        with col3:
            # Calculate average identity to first sequence
            identities = []
            for seq in alignment[1:]:
                matches = sum(1 for a, b in zip(alignment[0], seq) if a == b and a != '-')
                total = sum(1 for a, b in zip(alignment[0], seq) if a != '-' or b != '-')
                identities.append(matches / total if total > 0 else 0)
            avg_id = sum(identities) / len(identities) if identities else 1.0
            st.metric("Avg Identity", f"{avg_id * 100:.1f}%")
        with col4:
            # Gap percentage
            total_chars = sum(len(s) for s in alignment)
            gaps = sum(s.count('-') for s in alignment)
            gap_pct = gaps / total_chars * 100
            st.metric("Gap %", f"{gap_pct:.1f}%")

        # Preview alignment
        st.markdown("#### Alignment Viewer")
        
        try:
            from protein_design_hub.web.visualizations import create_msa_viewer
            import streamlit.components.v1 as components
            
            html = create_msa_viewer(alignment, names, height=300, max_sequences=50)
            components.html(html, height=320, scrolling=True)
        except ImportError:
            st.error("Visualization module (create_msa_viewer) not found")
        except Exception as e:
            st.error(f"Error visualizing alignment: {e}")


# === CONSERVATION TAB ===
with main_tabs[1]:
    st.markdown("### 📊 Conservation Analysis")

    if not st.session_state.msa_alignment:
        st.warning("Please load an MSA first")
    else:
        alignment = st.session_state.msa_alignment

        # Conservation method
        method = st.selectbox(
            "Conservation method",
            ["Shannon entropy", "Jensen-Shannon divergence", "Composite score"]
        )

        if st.button("🔬 Calculate Conservation"):
            try:
                from protein_design_hub.msa.conservation import (
                    ConservationCalculator,
                    calculate_conservation,
                )

                calculator = ConservationCalculator()
                results = calculator.analyze_alignment(alignment)

                # Store results
                st.session_state.conservation_results = results

                st.success("Conservation analysis complete!")

            except ImportError:
                st.error("MSA module not available")
            except Exception as e:
                st.error(f"Error: {e}")

        # Show results
        if 'conservation_results' in st.session_state:
            results = st.session_state.conservation_results

            st.markdown("---")

            # Conservation plot
            st.markdown("#### Conservation Profile")

            import pandas as pd

            df = pd.DataFrame([
                {
                    'Position': r.position + 1,
                    'Conservation': r.conservation_score,
                    'Entropy': r.shannon_entropy,
                    'Consensus': r.consensus_residue,
                    'Gap %': r.gap_frequency * 100,
                }
                for r in results
            ])

            st.line_chart(df.set_index('Position')['Conservation'])

            # Summary stats
            col1, col2, col3 = st.columns(3)

            with col1:
                highly_conserved = sum(1 for r in results if r.conservation_score > 0.8)
                st.metric("Highly conserved positions", highly_conserved)

            with col2:
                variable = sum(1 for r in results if r.conservation_score < 0.3)
                st.metric("Variable positions", variable)

            with col3:
                avg_cons = sum(r.conservation_score for r in results) / len(results)
                st.metric("Average conservation", f"{avg_cons:.3f}")

            # Download conservation data
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Conservation Data",
                csv,
                "conservation.csv",
                mime="text/csv"
            )

            # Conserved positions table
            st.markdown("#### Top Conserved Positions")

            top_conserved = sorted(results, key=lambda x: x.conservation_score, reverse=True)[:20]

            conserved_df = pd.DataFrame([
                {
                    'Position': r.position + 1,
                    'Consensus': r.consensus_residue,
                    'Conservation': f"{r.conservation_score:.3f}",
                    'Frequency': f"{r.consensus_frequency * 100:.1f}%",
                }
                for r in top_conserved
            ])

            st.dataframe(conserved_df, use_container_width=True)


# === COEVOLUTION TAB ===
with main_tabs[2]:
    st.markdown("### 🔗 Coevolution Analysis")

    if not st.session_state.msa_alignment:
        st.warning("Please load an MSA first")
    else:
        alignment = st.session_state.msa_alignment

        st.markdown("""
        Coevolving residues often indicate structural or functional contacts.
        This analysis uses mutual information with APC correction.
        """)

        col_params, col_run = st.columns([2, 1])

        with col_params:
            min_separation = st.slider("Minimum sequence separation", 3, 15, 5)
            top_pairs = st.slider("Top pairs to show", 10, 200, 50)

        with col_run:
            if st.button("🔬 Analyze Coevolution", type="primary"):
                with st.spinner("Calculating coevolution scores..."):
                    try:
                        from protein_design_hub.msa.coevolution import (
                            CoevolutionAnalyzer,
                            calculate_apc_corrected_mi,
                        )

                        analyzer = CoevolutionAnalyzer(min_sequence_separation=min_separation)
                        results = analyzer.analyze_alignment(alignment, top_pairs=top_pairs)

                        st.session_state.coevolution_results = results
                        st.success(f"Found {len(results)} coevolving pairs!")

                    except ImportError:
                        st.error("MSA module not available")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Show results
        if 'coevolution_results' in st.session_state:
            results = st.session_state.coevolution_results

            st.markdown("---")
            st.markdown("#### Top Coevolving Pairs")

            import pandas as pd

            df = pd.DataFrame([
                {
                    'Position i': r.position_i + 1,
                    'Position j': r.position_j + 1,
                    'MI (APC)': f"{r.apc_corrected_mi:.4f}",
                    'Normalized MI': f"{r.normalized_mi:.4f}",
                    'Contact Prob': f"{r.contact_probability:.3f}",
                }
                for r in results[:50]
            ])

            st.dataframe(df, use_container_width=True)

            # Contact map visualization
            st.markdown("#### Contact Map")

            try:
                import numpy as np
                import plotly.graph_objects as go

                # Create contact map matrix
                length = len(alignment[0])
                contact_map = np.zeros((length, length))

                for r in results:
                    contact_map[r.position_i, r.position_j] = r.apc_corrected_mi
                    contact_map[r.position_j, r.position_i] = r.apc_corrected_mi

                fig = go.Figure(data=go.Heatmap(
                    z=contact_map,
                    colorscale='Viridis',
                    showscale=True,
                ))

                fig.update_layout(
                    title="Coevolution Contact Map",
                    xaxis_title="Position",
                    yaxis_title="Position",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                st.info("Install plotly for contact map visualization: pip install plotly")

            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Coevolution Data",
                csv,
                "coevolution.csv",
                mime="text/csv"
            )


# === ANCESTRAL TAB ===
with main_tabs[3]:
    st.markdown("### 🌳 Ancestral Sequence Reconstruction")

    if not st.session_state.msa_alignment:
        st.warning("Please load an MSA first")
    else:
        alignment = st.session_state.msa_alignment
        names = st.session_state.msa_names

        st.markdown("""
        Reconstruct ancestral sequences to understand evolutionary history
        and identify stabilizing mutations.
        """)

        method = st.selectbox(
            "Reconstruction method",
            ["Maximum likelihood", "Maximum parsimony", "Consensus"]
        )

        if st.button("🌳 Reconstruct Ancestral Sequence", type="primary"):
            try:
                from protein_design_hub.msa.ancestral import (
                    AncestralReconstructor,
                    reconstruct_ancestral_sequence,
                )

                method_map = {
                    "Maximum likelihood": "likelihood",
                    "Maximum parsimony": "parsimony",
                    "Consensus": "consensus",
                }

                reconstructor = AncestralReconstructor(method=method_map[method])
                result = reconstructor.reconstruct(alignment, names=names)

                st.session_state.ancestral_result = result
                st.success("Ancestral reconstruction complete!")

            except ImportError:
                st.error("MSA module not available")
            except Exception as e:
                st.error(f"Error: {e}")

        # Show results
        if 'ancestral_result' in st.session_state:
            result = st.session_state.ancestral_result

            st.markdown("---")
            st.markdown("#### Reconstructed Ancestral Sequence")

            # Show sequence in chunks
            seq = result.sequence
            chunk_size = 60

            st.text_area(
                "Ancestral sequence",
                seq,
                height=100,
                label_visibility="collapsed"
            )

            # Confidence statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_conf = sum(result.confidence) / len(result.confidence)
                st.metric("Average confidence", f"{avg_conf:.3f}")

            with col2:
                high_conf = sum(1 for c in result.confidence if c > 0.8)
                st.metric("High confidence positions", high_conf)

            with col3:
                low_conf = sum(1 for c in result.confidence if c < 0.5)
                st.metric("Ambiguous positions", low_conf)

            # Confidence plot
            st.markdown("#### Per-Position Confidence")

            import pandas as pd

            conf_df = pd.DataFrame({
                'Position': range(1, len(result.confidence) + 1),
                'Confidence': result.confidence,
            })

            st.line_chart(conf_df.set_index('Position'))

            # Compare to extant sequences
            st.markdown("#### Comparison to Query Sequence")

            query = alignment[0].replace('-', '')
            ancestral = result.sequence.replace('-', '')

            differences = []
            for i, (q, a) in enumerate(zip(query, ancestral)):
                if q != a:
                    differences.append(f"{q}{i + 1}{a}")

            if differences:
                st.markdown(f"**{len(differences)} differences from query:**")
                st.code(' '.join(differences[:30]) + ('...' if len(differences) > 30 else ''))
            else:
                st.success("Ancestral sequence identical to query")

            # Generate variants
            st.markdown("---")
            st.markdown("#### Generate Ancestral Variants")

            num_variants = st.slider("Number of variants", 1, 20, 5)

            if st.button("🧬 Generate Variants"):
                try:
                    from protein_design_hub.msa.ancestral import AncestralReconstructor

                    reconstructor = AncestralReconstructor()
                    variants = reconstructor.generate_variants(result, num_variants=num_variants)

                    st.markdown("**Generated variants:**")
                    for i, var in enumerate(variants):
                        st.text(f"Variant {i + 1}: {var[:50]}...")

                except Exception as e:
                    st.error(f"Error: {e}")

            # Download
            fasta = f">ancestral_reconstructed\n{result.sequence}"
            st.download_button(
                "📥 Download Ancestral Sequence",
                fasta,
                "ancestral.fasta",
                mime="text/plain"
            )


# === PSSM TAB ===
with main_tabs[4]:
    st.markdown("### 📈 Position-Specific Scoring Matrix")

    if not st.session_state.msa_alignment:
        st.warning("Please load an MSA first")
    else:
        alignment = st.session_state.msa_alignment

        st.markdown("""
        Generate a PSSM for scoring sequences and identifying beneficial mutations.
        """)

        pseudocount = st.slider("Pseudocount", 0.0, 1.0, 0.0)

        if st.button("📊 Generate PSSM", type="primary"):
            try:
                from protein_design_hub.msa.pssm import PSSMCalculator, calculate_pssm

                calculator = PSSMCalculator(pseudocount=pseudocount)
                result = calculator.calculate_pssm(alignment)

                st.session_state.pssm_result = result
                st.success("PSSM generated!")

            except ImportError:
                st.error("MSA module not available")
            except Exception as e:
                st.error(f"Error: {e}")

        # Show results
        if 'pssm_result' in st.session_state:
            result = st.session_state.pssm_result

            st.markdown("---")

            # Consensus sequence
            st.markdown("#### Consensus Sequence")
            st.code(result.consensus_sequence[:100] + ('...' if len(result.consensus_sequence) > 100 else ''))

            # Information content
            st.markdown("#### Information Content Profile")

            import pandas as pd

            ic_df = pd.DataFrame({
                'Position': range(1, len(result.information_content) + 1),
                'Information Content': result.information_content,
            })

            st.line_chart(ic_df.set_index('Position'))

            st.metric("Total Information", f"{result.total_information:.2f} bits")

            # Score a sequence
            st.markdown("---")
            st.markdown("#### Score a Sequence")

            test_seq = st.text_input(
                "Enter sequence to score",
                placeholder="MKFLILLFNI..."
            )

            if test_seq and st.button("Score Sequence"):
                try:
                    from protein_design_hub.msa.pssm import PSSMCalculator

                    calculator = PSSMCalculator()
                    total_score, per_pos = calculator.score_sequence(test_seq, result)

                    st.metric("PSSM Score", f"{total_score:.2f}")

                    # Per-position scores
                    score_df = pd.DataFrame({
                        'Position': range(1, len(per_pos) + 1),
                        'Residue': list(test_seq[:len(per_pos)]),
                        'Score': per_pos,
                    })

                    st.line_chart(score_df.set_index('Position')['Score'])

                except Exception as e:
                    st.error(f"Error: {e}")

            # Suggest mutations
            st.markdown("---")
            st.markdown("#### Suggest Beneficial Mutations")

            query_seq = st.text_input(
                "Query sequence for mutation suggestions",
                value=alignment[0].replace('-', '')[:100],
                key="mutation_query"
            )

            threshold = st.slider("Minimum improvement threshold", 0.0, 2.0, 0.5)

            if query_seq and st.button("Find Mutations"):
                try:
                    from protein_design_hub.msa.pssm import PSSMCalculator

                    calculator = PSSMCalculator()
                    suggestions = calculator.suggest_mutations(query_seq, result, threshold=threshold)

                    if suggestions:
                        st.markdown(f"**Found {len(suggestions)} beneficial mutations:**")

                        import pandas as pd

                        mut_df = pd.DataFrame([
                            {
                                'Mutation': f"{s['original']}{s['position']}{s['mutant']}",
                                'Improvement': f"+{s['improvement']:.2f}",
                                'IC': f"{s['information_content']:.2f}",
                            }
                            for s in suggestions[:20]
                        ])

                        st.dataframe(mut_df, use_container_width=True)
                    else:
                        st.info("No mutations found above threshold")

                except Exception as e:
                    st.error(f"Error: {e}")
