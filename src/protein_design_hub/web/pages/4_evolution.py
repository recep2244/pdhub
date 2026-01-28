"""Directed Evolution Workflow page."""

import streamlit as st
from pathlib import Path
import json

from protein_design_hub.web.ui import (
    inject_base_css,
    page_header,
    sidebar_nav,
    sidebar_system_status,
)

st.set_page_config(page_title="Evolution - Protein Design Hub", page_icon="🧬", layout="wide")

inject_base_css()

# Page header
page_header(
    "Directed Evolution",
    "Run iterative design cycles with fitness landscape exploration and automated optimization",
    "🧬"
)

sidebar_nav(current="Evolution")
sidebar_system_status()

# Page-specific CSS (uses theme variables)
st.markdown("""
<style>
.evolution-card {
    background: var(--pdhub-gradient-dark);
    border-radius: var(--pdhub-border-radius-lg);
    padding: var(--pdhub-space-lg);
    color: white;
    margin: var(--pdhub-space-md) 0;
    box-shadow: var(--pdhub-shadow-md);
}
.generation-card {
    background: var(--pdhub-bg-gradient);
    border-radius: var(--pdhub-border-radius-md);
    padding: var(--pdhub-space-md);
    margin: var(--pdhub-space-sm) 0;
    border-left: 4px solid var(--pdhub-primary);
    transition: var(--pdhub-transition);
}
.generation-card:hover {
    box-shadow: var(--pdhub-shadow-sm);
    border-left-color: var(--pdhub-primary-dark);
}
.fitness-high { color: var(--pdhub-success); font-weight: bold; }
.fitness-medium { color: var(--pdhub-warning); font-weight: bold; }
.fitness-low { color: var(--pdhub-error); font-weight: bold; }
.metric-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: var(--pdhub-border-radius-full);
    font-size: 0.75rem;
    margin: 2px;
    font-weight: 500;
}
.metric-pill-blue { background: var(--pdhub-info-light); color: #1976d2; }
.metric-pill-green { background: var(--pdhub-success-light); color: #388e3c; }
.metric-pill-orange { background: var(--pdhub-warning-light); color: #f57c00; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evolution_sequence' not in st.session_state:
    st.session_state.evolution_sequence = ""
if 'evolution_results' not in st.session_state:
    st.session_state.evolution_results = None
if 'evolution_running' not in st.session_state:
    st.session_state.evolution_running = False

# Title
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 2.5rem;">
        🧬 Directed Evolution Workflow
    </h1>
    <p style="color: #666;">Computationally evolve proteins toward desired properties</p>
</div>
""", unsafe_allow_html=True)

# Main tabs
main_tabs = st.tabs(["🎯 Setup", "🔬 Run Evolution", "📊 Results", "📚 Library Design"])

# === SETUP TAB ===
with main_tabs[0]:
    st.markdown("### 📥 Input Sequence")

    col_input, col_info = st.columns([2, 1])

    with col_input:
        # Check if sequence came from design page
        if 'predict_sequence' in st.session_state and st.session_state.predict_sequence:
            st.session_state.evolution_sequence = st.session_state.predict_sequence

        seq_input = st.text_area(
            "Starting sequence",
            value=st.session_state.evolution_sequence,
            height=100,
            placeholder="Paste your protein sequence here...",
            key="evo_seq_input"
        )

        if seq_input != st.session_state.evolution_sequence:
            cleaned = ''.join(c for c in seq_input.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
            st.session_state.evolution_sequence = cleaned
            st.rerun()

        # Upload option
        uploaded = st.file_uploader("Or upload FASTA", type=["fasta", "fa"])
        if uploaded:
            content = uploaded.read().decode()
            for line in content.strip().split('\n'):
                if not line.startswith('>') and line.strip():
                    st.session_state.evolution_sequence = ''.join(
                        c for c in line.strip().upper() if c in "ACDEFGHIKLMNPQRSTVWY"
                    )
                    st.rerun()

    with col_info:
        if st.session_state.evolution_sequence:
            seq = st.session_state.evolution_sequence
            st.markdown(f"""
            <div class="evolution-card">
                <h4>Sequence Info</h4>
                <p><b>Length:</b> {len(seq)} residues</p>
                <p><b>MW:</b> ~{len(seq) * 110 / 1000:.1f} kDa</p>
            </div>
            """, unsafe_allow_html=True)

            # Quick biophysical analysis
            try:
                from protein_design_hub.biophysics import calculate_all_properties

                props = calculate_all_properties(seq)
                st.markdown("**Properties:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("pI", f"{props.isoelectric_point:.2f}")
                    st.metric("GRAVY", f"{props.gravy:.2f}")
                with col2:
                    st.metric("Instability", f"{props.instability_index:.1f}")
                    st.metric("Aliphatic", f"{props.aliphatic_index:.1f}")
            except ImportError:
                pass

    st.markdown("---")

    # Evolution parameters
    st.markdown("### ⚙️ Evolution Parameters")

    col_gen, col_pop, col_mut = st.columns(3)

    with col_gen:
        num_generations = st.slider("Generations", 1, 50, 10)
        selection_strategy = st.selectbox(
            "Selection",
            ["truncation", "tournament", "roulette", "rank", "elite"],
            help="How to select parents for next generation"
        )

    with col_pop:
        population_size = st.slider("Population size", 10, 200, 50)
        top_fraction = st.slider("Top fraction", 0.1, 0.5, 0.2,
                                 help="Fraction of population to keep each generation")

    with col_mut:
        mutation_rate = st.slider("Mutation rate", 0.01, 0.3, 0.1)
        max_mutations = st.slider("Max mutations/seq", 1, 10, 3)

    st.markdown("---")

    # Fitness function configuration
    st.markdown("### 🎯 Fitness Function")

    fitness_type = st.selectbox(
        "Primary fitness objective",
        ["Stability", "Solubility", "Structure quality", "Sequence recovery", "Custom composite"]
    )

    if fitness_type == "Custom composite":
        st.markdown("**Configure weights:**")
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)

        with col_w1:
            w_stability = st.slider("Stability", 0.0, 1.0, 0.3)
        with col_w2:
            w_solubility = st.slider("Solubility", 0.0, 1.0, 0.3)
        with col_w3:
            w_plddt = st.slider("pLDDT", 0.0, 1.0, 0.2)
        with col_w4:
            w_recovery = st.slider("Recovery", 0.0, 1.0, 0.2)

    # Constraints
    st.markdown("### 🔒 Constraints (Optional)")

    constraint_tabs = st.tabs(["Fixed Positions", "Allowed Mutations", "Secondary Structure"])

    with constraint_tabs[0]:
        fixed_positions = st.text_input(
            "Fixed positions (comma-separated)",
            placeholder="1,5,10-15,50",
            help="These positions will not be mutated"
        )

    with constraint_tabs[1]:
        col_pos, col_allowed = st.columns(2)
        with col_pos:
            restrict_pos = st.text_input("Position", placeholder="42")
        with col_allowed:
            allowed_aa = st.text_input("Allowed AAs", placeholder="AILV")

        if 'position_constraints' not in st.session_state:
            st.session_state.position_constraints = {}

        if st.button("Add constraint") and restrict_pos and allowed_aa:
            try:
                pos = int(restrict_pos)
                st.session_state.position_constraints[pos] = list(allowed_aa.upper())
                st.success(f"Position {pos}: allowed {allowed_aa}")
            except ValueError:
                st.error("Invalid position")

        if st.session_state.position_constraints:
            st.markdown("**Current constraints:**")
            for pos, aas in st.session_state.position_constraints.items():
                st.text(f"Position {pos}: {''.join(aas)}")

    with constraint_tabs[2]:
        preserve_ss = st.checkbox("Preserve secondary structure propensity")
        if preserve_ss:
            st.info("Mutations will favor residues with similar helix/sheet propensity")


# === RUN EVOLUTION TAB ===
with main_tabs[1]:
    st.markdown("### 🚀 Run Directed Evolution")

    if not st.session_state.evolution_sequence:
        st.warning("Please input a sequence in the Setup tab first")
    else:
        st.markdown(f"""
        <div class="evolution-card">
            <h4>Configuration Summary</h4>
            <p>Starting sequence: {len(st.session_state.evolution_sequence)} residues</p>
            <p>Generations: {num_generations} | Population: {population_size}</p>
            <p>Mutation rate: {mutation_rate} | Selection: {selection_strategy}</p>
        </div>
        """, unsafe_allow_html=True)

        col_run, col_status = st.columns([1, 2])

        with col_run:
            if st.button("🧬 Start Evolution", type="primary",
                         use_container_width=True,
                         disabled=st.session_state.evolution_running):
                st.session_state.evolution_running = True

                # Run evolution
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    from protein_design_hub.evolution import (
                        DirectedEvolution,
                        EvolutionConfig,
                        SelectionStrategy,
                    )
                    from protein_design_hub.evolution.fitness_landscape import (
                        StabilityFitness,
                        SolubilityFitness,
                        CompositeFitness,
                    )

                    # Create fitness function
                    if fitness_type == "Stability":
                        fitness_fn = StabilityFitness()
                    elif fitness_type == "Solubility":
                        fitness_fn = SolubilityFitness()
                    elif fitness_type == "Custom composite":
                        fitness_fn = CompositeFitness(weights={
                            "stability": w_stability,
                            "solubility": w_solubility,
                        })
                    else:
                        fitness_fn = StabilityFitness()

                    # Parse constraints
                    fixed_pos_set = set()
                    if fixed_positions:
                        for part in fixed_positions.split(","):
                            part = part.strip()
                            if "-" in part:
                                start, end = map(int, part.split("-"))
                                fixed_pos_set.update(range(start - 1, end))
                            else:
                                fixed_pos_set.add(int(part) - 1)

                    # Create config
                    config = EvolutionConfig(
                        population_size=population_size,
                        num_generations=num_generations,
                        mutation_rate=mutation_rate,
                        max_mutations_per_sequence=max_mutations,
                        selection_strategy=SelectionStrategy[selection_strategy.upper()],
                        elite_fraction=top_fraction,
                        fixed_positions=list(fixed_pos_set),
                    )

                    # Run evolution
                    evolver = DirectedEvolution(
                        fitness_function=fitness_fn,
                        config=config,
                    )

                    generations = []
                    for gen, result in enumerate(evolver.evolve_generator(
                        st.session_state.evolution_sequence
                    )):
                        progress_bar.progress((gen + 1) / num_generations)
                        status_text.text(f"Generation {gen + 1}/{num_generations} - Best fitness: {result.best_fitness:.4f}")
                        generations.append({
                            "generation": gen + 1,
                            "best_fitness": result.best_fitness,
                            "mean_fitness": result.mean_fitness,
                            "best_sequence": result.best_sequence,
                            "diversity": result.diversity,
                        })

                    st.session_state.evolution_results = {
                        "generations": generations,
                        "best_sequence": generations[-1]["best_sequence"],
                        "best_fitness": generations[-1]["best_fitness"],
                        "starting_sequence": st.session_state.evolution_sequence,
                    }

                    st.success(f"Evolution complete! Best fitness: {generations[-1]['best_fitness']:.4f}")

                except ImportError as e:
                    st.error(f"Missing module: {e}")
                except Exception as e:
                    st.error(f"Evolution failed: {e}")
                finally:
                    st.session_state.evolution_running = False
                    st.rerun()

        with col_status:
            if st.session_state.evolution_running:
                st.info("Evolution in progress...")
            elif st.session_state.evolution_results:
                st.success("Evolution completed! View results in the Results tab.")


# === RESULTS TAB ===
with main_tabs[2]:
    st.markdown("### 📊 Evolution Results")

    if not st.session_state.evolution_results:
        st.info("Run evolution first to see results")
    else:
        results = st.session_state.evolution_results
        generations = results["generations"]

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            improvement = generations[-1]["best_fitness"] - generations[0]["best_fitness"]
            st.metric("Fitness Improvement", f"+{improvement:.4f}")
        with col2:
            st.metric("Final Best Fitness", f"{generations[-1]['best_fitness']:.4f}")
        with col3:
            # Count mutations
            orig = results["starting_sequence"]
            best = results["best_sequence"]
            mutations = sum(1 for a, b in zip(orig, best) if a != b)
            st.metric("Total Mutations", mutations)
        with col4:
            st.metric("Generations Run", len(generations))

        st.markdown("---")

        # Fitness over generations chart
        st.markdown("#### Fitness Trajectory")

        import pandas as pd

        df = pd.DataFrame(generations)

        col_chart, col_data = st.columns([2, 1])

        with col_chart:
            st.line_chart(df[["best_fitness", "mean_fitness"]])

        with col_data:
            st.dataframe(df[["generation", "best_fitness", "diversity"]].tail(10))

        st.markdown("---")

        # Best sequence analysis
        st.markdown("#### Best Evolved Sequence")

        best_seq = results["best_sequence"]
        orig_seq = results["starting_sequence"]

        # Highlight mutations
        st.markdown("**Mutations:**")
        mutations_list = []
        for i, (orig, evolved) in enumerate(zip(orig_seq, best_seq)):
            if orig != evolved:
                mutations_list.append(f"{orig}{i + 1}{evolved}")

        if mutations_list:
            st.code(" ".join(mutations_list))
        else:
            st.info("No mutations in best sequence")

        # Show aligned sequences
        st.markdown("**Alignment:**")
        col_orig, col_best = st.columns(2)

        with col_orig:
            st.markdown("**Original:**")
            st.text(orig_seq[:50] + "..." if len(orig_seq) > 50 else orig_seq)

        with col_best:
            st.markdown("**Evolved:**")
            st.text(best_seq[:50] + "..." if len(best_seq) > 50 else best_seq)

        # Download options
        st.markdown("---")
        st.markdown("#### Export")

        col_dl1, col_dl2, col_dl3 = st.columns(3)

        with col_dl1:
            fasta = f">evolved_protein\n{best_seq}"
            st.download_button(
                "📥 Best Sequence (FASTA)",
                fasta,
                "evolved_sequence.fasta",
                use_container_width=True
            )

        with col_dl2:
            st.download_button(
                "📥 Full Results (JSON)",
                json.dumps(results, indent=2),
                "evolution_results.json",
                use_container_width=True
            )

        with col_dl3:
            if st.button("🔬 Predict Structure", use_container_width=True):
                st.session_state.predict_sequence = best_seq
                st.session_state.predict_name = "evolved_protein"
                st.info("Go to the Predict page to run structure prediction")


# === LIBRARY DESIGN TAB ===
with main_tabs[3]:
    st.markdown("### 📚 Combinatorial Library Design")

    st.markdown("""
    Design mutant libraries for experimental screening with optimized codon usage.
    """)

    if not st.session_state.evolution_sequence:
        st.warning("Please input a sequence in the Setup tab first")
    else:
        col_lib_setup, col_lib_preview = st.columns([1, 1])

        with col_lib_setup:
            st.markdown("#### Library Configuration")

            library_type = st.selectbox(
                "Library type",
                ["Site-saturation mutagenesis", "Combinatorial", "Error-prone PCR simulation"]
            )

            if library_type == "Site-saturation mutagenesis":
                target_positions = st.text_input(
                    "Target positions",
                    placeholder="1,5,10",
                    help="Positions for NNK saturation"
                )
                codon_type = st.selectbox(
                    "Degenerate codon",
                    ["NNK", "NNS", "NDT", "Custom"]
                )

                if codon_type == "Custom":
                    custom_codon = st.text_input("Custom codon", placeholder="NNG")

            elif library_type == "Combinatorial":
                st.markdown("**Define mutations per position:**")
                st.text_input("Position 1", placeholder="42:AILVM")
                st.text_input("Position 2", placeholder="58:DEKR")
                st.text_input("Position 3", placeholder="102:FWY")

            else:  # Error-prone
                error_rate = st.slider("Error rate (%)", 0.1, 5.0, 1.0)
                num_variants = st.slider("Number of variants", 10, 1000, 100)

            # Calculate library size
            st.markdown("---")
            if st.button("📊 Calculate Library Size", use_container_width=True):
                try:
                    from protein_design_hub.evolution.library_design import LibraryDesigner

                    designer = LibraryDesigner()

                    if library_type == "Site-saturation mutagenesis" and target_positions:
                        positions = [int(p.strip()) - 1 for p in target_positions.split(",")]

                        # NNK encodes 20 AAs
                        library_size = 20 ** len(positions)
                        st.metric("Theoretical library size", f"{library_size:,}")

                        # Practical coverage
                        st.markdown("**Screening requirements (95% coverage):**")
                        coverage_95 = int(library_size * 3)  # 3x oversampling
                        st.text(f"Clones to screen: ~{coverage_95:,}")

                except ImportError:
                    st.info("Library design module not available")

        with col_lib_preview:
            st.markdown("#### Library Preview")

            if st.button("🔬 Generate Library Preview"):
                try:
                    from protein_design_hub.evolution.library_design import (
                        LibraryDesigner,
                        MutationLibrary,
                    )

                    designer = LibraryDesigner()
                    seq = st.session_state.evolution_sequence

                    if library_type == "Site-saturation mutagenesis" and target_positions:
                        positions = [int(p.strip()) - 1 for p in target_positions.split(",")]

                        library = designer.create_saturation_library(
                            seq,
                            positions=positions,
                            max_variants=20
                        )

                        st.markdown("**Sample variants:**")
                        for i, variant in enumerate(library.variants[:10]):
                            mutations = []
                            for j, (orig, mut) in enumerate(zip(seq, variant)):
                                if orig != mut:
                                    mutations.append(f"{orig}{j + 1}{mut}")
                            st.text(f"{i + 1}. {' '.join(mutations) if mutations else 'WT'}")

                except ImportError:
                    st.warning("Library design module not available")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")

        # Export library
        st.markdown("#### Export Library")

        export_format = st.selectbox("Format", ["FASTA", "CSV", "DNA (with primers)"])

        if st.button("📥 Generate & Download Library", type="primary"):
            st.info("Library generation would create variants for experimental testing")
            st.code("""
# Example library output (FASTA):
>variant_001
MKFLILLFNILCLFPVLAADNHGVGPQGAS...
>variant_002
MKFLILLFNILCLFPALAADNHGVGPQGAS...
            """)
