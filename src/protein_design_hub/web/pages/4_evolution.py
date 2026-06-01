"""Directed Evolution Workflow page."""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import time

from protein_design_hub.web.ui import (
    inject_base_css,
    page_header,
    sidebar_nav,
    sidebar_system_status,
    metric_card,
    card_start,
    card_end,
    empty_state,
    render_badge,
    workflow_breadcrumb,
    cross_page_actions,
)
from protein_design_hub.web.agent_helpers import (
    render_contextual_insight,
    render_agent_advice_panel,
    render_ml_stats_panel,
    agent_sidebar_status,
    render_all_experts_panel,
    observed_scoring_section,
    render_pymolai_section,
    render_pymolai_chatbot,
    render_pymolai_viewer,
)
from protein_design_hub.web.shared_context import set_page_results, render_workflow_status_bar

st.set_page_config(page_title="Evolution - Protein Design Hub", page_icon="🧬", layout="wide")

inject_base_css()
sidebar_nav(current="Evolution")
sidebar_system_status()
agent_sidebar_status()

# Page header
page_header(
    "Directed Evolution",
    "Run iterative design cycles with fitness landscape exploration and automated optimization",
    "🧬"
)
render_workflow_status_bar()

workflow_breadcrumb(
    ["Design Sequence", "Predict", "Directed Evolution", "Select Best"],
    current=2,
)

with st.expander("📖 How directed evolution works", expanded=False):
    st.markdown("""
**Computational directed evolution** mimics natural evolution to optimize protein properties:

1. **Start** with a parent sequence (wild-type or designed)
2. **Generate** variants through random or targeted mutations
3. **Screen** using structure prediction (pLDDT, stability scores)
4. **Select** the best variants as parents for the next round
5. **Repeat** for multiple generations

**Parameters:**
- **Population size** — larger = more diverse exploration, slower per generation
- **Mutation rate** — 1-3 mutations/sequence is typical; more = higher risk
- **Fitness function** — pLDDT, stability, or custom metric to optimize
- **Generations** — 3-5 rounds usually converges for simple improvements
    """)

# Page-specific CSS (uses theme variables)
st.markdown("""
<style>
.evolution-card {
    background: var(--pdhub-gradient-dark);
    border-radius: var(--pdhub-border-radius-lg);
    padding: var(--pdhub-space-lg);
    color: var(--pdhub-text-heading);
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
.metric-pill-blue { background: var(--pdhub-info-light); color: var(--pdhub-info, #3b82f6); }
.metric-pill-green { background: var(--pdhub-success-light); color: var(--pdhub-success, #22c55e); }
.metric-pill-orange { background: var(--pdhub-warning-light); color: var(--pdhub-warning, #f59e0b); }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evolution_sequence' not in st.session_state:
    st.session_state.evolution_sequence = ""
if 'evolution_results' not in st.session_state:
    st.session_state.evolution_results = None
if 'evolution_running' not in st.session_state:
    st.session_state.evolution_running = False

# Handle external job loading
if st.session_state.get("evolution_job_to_load"):
    try:
        job_path = Path(st.session_state["evolution_job_to_load"])
        summary_path = job_path / "evolution_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                st.session_state.evolution_results = json.load(f)
            st.session_state.evolution_sequence = st.session_state.evolution_results.get("starting_sequence", "")
            st.success(f"Successfully loaded job: {job_path.name}")
        # Clear it so it doesn't reload on every rerun
        st.session_state.pop("evolution_job_to_load")
    except Exception as e:
        st.error(f"Error loading job: {e}")

# Main tabs
main_tabs = st.tabs(["🎯 Setup", "🔬 Run Evolution", "📊 Results", "📚 Library Design"])

# === SETUP TAB ===
with main_tabs[0]:
    st.markdown("### 📥 Input Sequence")

    # Quick-load example sequences
    _EVO_EXAMPLES = {
        "Ubiquitin (76 aa)": ("Ubiquitin", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"),
        "T1024 (52 aa)": ("T1024", "MAAHKGAEHVVKASLDAGVKTVAGGLVVKAKALGGKDATMHLVAATLKKGYM"),
        "Hemoglobin α (51 aa)": ("HbA_fragment", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"),
        "Insulin B (30 aa)": ("Insulin_B", "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
    }
    st.markdown("**⚡ Quick Load — Example Sequences:**")
    _evo_ex_cols = st.columns(len(_EVO_EXAMPLES))
    for _ei, (_elabel, (_ename, _eseq)) in enumerate(_EVO_EXAMPLES.items()):
        with _evo_ex_cols[_ei]:
            if st.button(_elabel, key=f"evo_ex_{_ei}", use_container_width=True, type="secondary"):
                st.session_state.evolution_sequence = _eseq
                st.rerun()

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
            seq_lines = []
            for line in content.strip().split('\n'):
                if line.startswith('>'):
                    continue
                if line.strip():
                    seq_lines.append(''.join(
                        c for c in line.strip().upper() if c in "ACDEFGHIKLMNPQRSTVWY"
                    ))
            if seq_lines:
                st.session_state.evolution_sequence = ''.join(seq_lines)
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
                from protein_design_hub.biophysics import (
                    calculate_pi, calculate_gravy,
                    calculate_instability_index, calculate_aliphatic_index,
                )
                st.markdown("**Properties:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("pI", f"{calculate_pi(seq):.2f}")
                    st.metric("GRAVY", f"{calculate_gravy(seq):.2f}")
                with col2:
                    st.metric("Instability", f"{calculate_instability_index(seq):.1f}")
                    st.metric("Aliphatic", f"{calculate_aliphatic_index(seq):.1f}")
            except Exception:
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
                            "plddt": w_plddt,
                            "recovery": w_recovery,
                        })
                    elif fitness_type == "Sequence recovery":
                        fitness_fn = CompositeFitness(weights={"recovery": 1.0})
                    elif fitness_type == "Structure quality":
                        fitness_fn = CompositeFitness(weights={"plddt": 1.0})
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

                    # Create config — all kwargs now exist on EvolutionConfig
                    config = EvolutionConfig(
                        population_size=population_size,
                        num_generations=num_generations,
                        mutation_rate=mutation_rate,
                        max_mutations_per_sequence=max_mutations,
                        selection_strategy=SelectionStrategy[selection_strategy.upper()],
                        elite_fraction=top_fraction,
                        fixed_positions=list(fixed_pos_set),
                        position_constraints=dict(st.session_state.get("position_constraints", {})),
                    )

                    # Run evolution — DirectedEvolution.run() returns EvolutionResult
                    evolver = DirectedEvolution(
                        parent_sequence=st.session_state.evolution_sequence,
                        fitness_function=fitness_fn,
                        config=config,
                    )

                    # Register a progress callback to update the UI during run
                    _progress_state = {"gen": 0}

                    def _on_gen(gen_result):
                        _progress_state["gen"] += 1
                        g = _progress_state["gen"]
                        progress_bar.progress(min(g / num_generations, 1.0))
                        status_text.text(
                            f"Generation {g}/{num_generations} — "
                            f"Best: {gen_result.best_fitness:.4f} | "
                            f"Mean: {gen_result.mean_fitness:.4f} | "
                            f"Diversity: {gen_result.diversity:.3f}"
                        )

                    evolver.add_callback(_on_gen)
                    evo_result = evolver.run()

                    generations = [
                        {
                            "generation": g.generation + 1,
                            "best_fitness": g.best_fitness,
                            "mean_fitness": g.mean_fitness,
                            "best_sequence": g.population[0].sequence if g.population else st.session_state.evolution_sequence,
                            "diversity": g.diversity,
                        }
                        for g in evo_result.generations
                    ]

                    st.session_state.evolution_results = {
                        "generations": generations,
                        "best_sequence": generations[-1]["best_sequence"],
                        "best_fitness": generations[-1]["best_fitness"],
                        "starting_sequence": st.session_state.evolution_sequence,
                    }

                    # Create Job in outputs
                    try:
                        from protein_design_hub.core.config import get_settings
                        settings = get_settings()
                        job_id = f"evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        job_dir = Path(settings.output.base_dir) / job_id
                        job_dir.mkdir(parents=True, exist_ok=True)
                        
                        with open(job_dir / "evolution_summary.json", "w") as f:
                            json.dump(st.session_state.evolution_results, f, indent=2)
                        
                        # Also save a dummy prediction_summary for Job browser detection
                        with open(job_dir / "prediction_summary.json", "w") as f:
                            json.dump({"job_id": job_id, "type": "evolution", "status": "complete"}, f)
                            
                        st.info(f"💾 Job saved as {job_id}")
                    except Exception as e:
                        st.warning(f"Could not save job to outputs: {e}")

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
            metric_card(f"+{improvement:.4f}", "Improvement", "success", "📈")
        with col2:
            metric_card(f"{generations[-1]['best_fitness']:.4f}", "Best Fitness", "gradient", "🏆")
        with col3:
            # Count mutations
            orig = results["starting_sequence"]
            best = results["best_sequence"]
            mutations = sum(1 for a, b in zip(orig, best) if a != b)
            metric_card(mutations, "Mutations", "warning", "🧬")
        with col4:
            metric_card(len(generations), "Generations", "info", "🔄")

        import pandas as pd

        # Save evolution results to shared context
        set_page_results("Evolution", {
            "best_fitness": generations[-1]["best_fitness"],
            "num_generations": len(generations),
            "improvement": generations[-1]["best_fitness"] - generations[0]["best_fitness"],
            "mutations_from_parent": sum(
                1 for a, b in zip(results["starting_sequence"], results["best_sequence"]) if a != b
            ),
            "best_sequence": results["best_sequence"][:40] + "...",
        })

        df = pd.DataFrame(generations)

        # ML stats panel on generation-by-generation data
        render_ml_stats_panel(
            generations,
            numeric_keys=["best_fitness", "mean_fitness", "diversity"],
            target_key="best_fitness",
            page_name="Evolution",
            key_prefix="evo_ml_stats",
        )

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

        # Sequence recovery
        if len(orig_seq) == len(best_seq):
            recovery = sum(a == b for a, b in zip(orig_seq, best_seq)) / len(orig_seq)
        else:
            recovery = 0.0

        # Highlight mutations
        mutations_list = []
        for i, (orig, evolved) in enumerate(zip(orig_seq, best_seq)):
            if orig != evolved:
                mutations_list.append((orig, i + 1, evolved))

        _evo_res_c1, _evo_res_c2, _evo_res_c3 = st.columns(3)
        with _evo_res_c1:
            st.metric("Sequence Recovery", f"{recovery:.1%}",
                      help="Fraction of positions identical to starting sequence")
        with _evo_res_c2:
            st.metric("Total Mutations", len(mutations_list))
        with _evo_res_c3:
            st.metric("Sequence Length", len(best_seq))

        if mutations_list:
            st.markdown("**Ranked mutation list:**")
            st.code("  ".join(f"{o}{p}{e}" for o, p, e in mutations_list))

            # Biophysics delta (parent vs evolved)
            try:
                from protein_design_hub.biophysics import (
                    calculate_pi, calculate_gravy, calculate_instability_index, calculate_mw,
                )
                _parent_pi = calculate_pi(orig_seq)
                _evolved_pi = calculate_pi(best_seq)
                _parent_ii = calculate_instability_index(orig_seq)
                _evolved_ii = calculate_instability_index(best_seq)
                _parent_gravy = calculate_gravy(orig_seq)
                _evolved_gravy = calculate_gravy(best_seq)
                _parent_mw = calculate_mw(orig_seq) / 1000
                _evolved_mw = calculate_mw(best_seq) / 1000

                st.markdown("**Biophysical delta (parent → evolved):**")
                _bp_cols = st.columns(4)
                _bp_cols[0].metric("pI", f"{_evolved_pi:.2f}", f"{_evolved_pi - _parent_pi:+.2f}")
                _bp_cols[1].metric("Instability Index",
                                   f"{_evolved_ii:.1f}",
                                   f"{_evolved_ii - _parent_ii:+.1f}",
                                   delta_color="inverse")
                _bp_cols[2].metric("GRAVY", f"{_evolved_gravy:.3f}", f"{_evolved_gravy - _parent_gravy:+.3f}")
                _bp_cols[3].metric("MW (kDa)", f"{_evolved_mw:.1f}", f"{_evolved_mw - _parent_mw:+.1f}")

                if _evolved_ii < 40 and _parent_ii >= 40:
                    st.success("Evolved variant crosses instability threshold (II < 40) — improved stability!")
                elif _evolved_ii >= 40 and _parent_ii < 40:
                    st.warning("Evolved variant crossed above instability threshold — check if mutations destabilised structure.")
            except Exception:
                pass
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
            if st.button("🧪 Wet-Lab Readiness", use_container_width=True, key="evo_wl_btn",
                         help="Check expression system, purification, and go/no-go for best variant"):
                try:
                    from protein_design_hub.analysis.wet_lab_advisor import build_wet_lab_report as _build_evo_wl
                    with st.spinner("Assessing wet-lab readiness…"):
                        _evo_wl_rep = _build_evo_wl(best_seq, is_antibody=False)
                    st.session_state["_evo_wl_report"] = _evo_wl_rep
                except Exception as _ewle:
                    st.error(f"Assessment failed: {_ewle}")

            _evo_wl = st.session_state.get("_evo_wl_report")
            if _evo_wl:
                _vvc = {"GO": "#22c55e", "CONDITIONAL": "#f59e0b", "NO-GO": "#ef4444"}
                _vvi = {"GO": "✅", "CONDITIONAL": "⚠️", "NO-GO": "❌"}
                st.markdown(
                    f'<div style="background:rgba(0,0,0,0.3);border-left:4px solid '
                    f'{_vvc.get(_evo_wl.go_nogo,"#6b7280")};padding:10px 14px;border-radius:6px;">'
                    f'<b style="color:{_vvc.get(_evo_wl.go_nogo,"#6b7280")};">'
                    f'{_vvi.get(_evo_wl.go_nogo,"?")} {_evo_wl.go_nogo}</b>'
                    f'<br><small style="color:#e2e8f0;">{_evo_wl.go_nogo_reason}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if _evo_wl.expression_systems:
                    _et = _evo_wl.expression_systems[0]
                    st.caption(f"Top system: {_et.system} ({_et.score:.0f}/100)")

        with col_dl3:
           st.markdown("**Structure Preview**")
           
           # State for evolution structure
           if 'evo_structure' not in st.session_state:
               st.session_state.evo_structure = None
           
           if st.button("🔮 Fold Best Variant (ESMFold)", use_container_width=True, disabled=st.session_state.get('evo_folding', False)):
                st.session_state.evo_folding = True
                with st.spinner("Folding best variant..."):
                    try:
                        import requests
                        if len(best_seq) <= 400:
                            response = requests.post(
                                "https://api.esmatlas.com/foldSequence/v1/pdb/",
                                data=best_seq,
                                headers={"Content-Type": "text/plain"},
                                timeout=60,
                            )
                            if response.status_code == 200:
                                st.session_state.evo_structure = response.text
                                st.success("Folded successfully!")
                            else:
                                st.error(f"API Error: {response.status_code}")
                        else:
                            st.error("Sequence too long for API demo")
                    except Exception as e:
                        st.error(f"Folding failed: {e}")
                    finally:
                        st.session_state.evo_folding = False
                        st.rerun()

        mutation_summary = ", ".join(mutations_list[:25]) if mutations_list else "none"
        if len(mutations_list) > 25:
            mutation_summary += ", ..."
        evo_context = "\n".join([
            f"Starting sequence length: {len(orig_seq)}",
            f"Generations: {len(generations)}",
            f"Best fitness: {generations[-1]['best_fitness']:.4f}",
            f"Mean fitness (last gen): {generations[-1]['mean_fitness']:.4f}",
            f"Best sequence mutations: {mutation_summary}",
        ])
        evo_data = {
            "Generations": len(generations),
            "Best fitness": f"{generations[-1]['best_fitness']:.4f}",
            "Mean fitness (last gen)": f"{generations[-1]['mean_fitness']:.4f}",
            "Mutations": len(mutations_list),
            "Starting sequence length": len(orig_seq),
        }
        render_contextual_insight(
            "Evolution",
            evo_data,
            key_prefix="evo_ctx",
        )

        render_agent_advice_panel(
            page_context=evo_context,
            default_question=(
                "Is the fitness improvement significant? Which mutations "
                "are most likely driving the improvement?"
            ),
            expert="Protein Engineer",
            key_prefix="evo_agent",
        )

        # Enrich context with trajectory shape and mutation burden
        _first_gen_fitness = generations[0]["best_fitness"]
        _last_gen_fitness = generations[-1]["best_fitness"]
        _fitness_gain = _last_gen_fitness - _first_gen_fitness
        _plateau_check = abs(generations[-1]["best_fitness"] - generations[-2]["best_fitness"]) < 1e-4 if len(generations) >= 2 else False
        _mean_last = generations[-1]["mean_fitness"]
        _fitness_gap = _last_gen_fitness - _mean_last  # best vs mean in last gen
        _mutation_burden = len(mutations_list)
        _enriched_evo_ctx = "\n".join([
            evo_context,
            "",
            f"Fitness gain across all generations: {_fitness_gain:+.4f} ({_first_gen_fitness:.4f} → {_last_gen_fitness:.4f})",
            f"Plateau detected (last gen delta < 1e-4): {'YES — may be converged' if _plateau_check else 'No — still improving'}",
            f"Best vs. mean fitness gap (last gen): {_fitness_gap:.4f} (large gap → high-fitness outlier, may be noisy)",
            f"Mutation burden: {_mutation_burden} mutations from WT",
        ])
        render_all_experts_panel(
            "All-Expert Review (evolution job)",
            agenda=(
                "Evaluate the directed evolution trajectory from structural biology, wet lab, "
                "immunology, and plant biology perspectives."
            ),
            context=_enriched_evo_ctx,
            questions=(
                f"The fitness improved by {_fitness_gain:+.4f} over {len(generations)} generations "
                f"{'and appears to have plateaued' if _plateau_check else 'with no clear plateau'} — "
                "is this gain convincing or likely noise/overfitting? Which subset of the "
                f"{_mutation_burden} mutations are most likely causal vs. hitchhiker, "
                "and is there epistasis risk that collapses fitness when mutations are separated?",
                f"From a wet lab perspective: what is the optimal validation plan for the evolved variant — "
                "synthesize top 3 single-mutant variants plus the multi-mutant best, express in "
                "E. coli/HEK293 (therapeutics) or Agrobacterium N. benthamiana (plant proteins), "
                "then screen by DSF for thermal stability and SPR/ELISA for function; "
                "at what throughput can your assay system handle these variants?",
                "From an immunology angle: do the evolved mutations introduce new T-cell epitope-prone "
                "hydrophobic stretches or remove canonical disulfide Cys residues? "
                "From a plant biology angle: do the mutations affect known phosphorylation sites "
                "(Ser/Thr-Pro motifs for plant kinases), ubiquitination lysines, or effector-interaction "
                "surfaces on NLR/LRR-RK proteins that mediate plant immune signaling?",
            ),
            key_prefix="evo_all",
        )

    # Structure Viewer Section
    if st.session_state.get('evo_structure'):
        st.markdown("---")
        st.markdown("#### 🧬 Best Variant Structure")
        
        col_v1, col_v2 = st.columns([3, 1])
        
        with col_v1:
            from protein_design_hub.web.visualizations import show_structure_with_pymol_fallback
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
                tmp.write(st.session_state.evo_structure)
                tmp_path = Path(tmp.name)

            show_structure_with_pymol_fallback(tmp_path, title="Best Evolved Variant", height=400)

            _evo_port = 0
            try:
                from protein_design_hub.web.pymol_server import get_pymol_server as _gps_evo
                _srv_evo = _gps_evo()
                if _srv_evo is not None:
                    _evo_port = _srv_evo.server_address[1]
            except Exception:
                pass
            render_pymolai_section(
                key_prefix="evolution_viewer",
                pymol_port=_evo_port,
                label="💬 Ask PyMolAI — Analyse evolved variant structure",
            )

            if tmp_path and Path(tmp_path).exists():
                observed_scoring_section(
                    model_paths=[Path(tmp_path)],
                    section_key="evo_obs",
                )
            
        with col_v2:
            st.info("Structure predicted by ESMFold")
            st.download_button(
                "📥 Download PDB",
                data=st.session_state.evo_structure,
                file_name=f"evolved_best.pdb",
                mime="chemical/x-pdb",
                use_container_width=True
            )



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
                st.markdown("**Define mutations per position (format: `position:AAs`)**")
                _comb_p1 = st.text_input("Position 1", placeholder="42:AILVM", key="comb_pos1")
                _comb_p2 = st.text_input("Position 2", placeholder="58:DEKR", key="comb_pos2")
                _comb_p3 = st.text_input("Position 3 (optional)", placeholder="102:FWY", key="comb_pos3")
                _comb_inputs = [x for x in [_comb_p1, _comb_p2, _comb_p3] if x.strip()]
                if _comb_inputs:
                    _comb_size = 1
                    for _ci in _comb_inputs:
                        if ":" in _ci:
                            _comb_size *= len(_ci.split(":", 1)[1])
                    st.caption(f"Combinatorial library size: **{_comb_size:,}** variants")

            else:  # Error-prone
                error_rate = st.slider("Error rate (%)", 0.1, 5.0, 1.0)
                num_variants = st.slider("Number of variants", 10, 1000, 100)

            # Calculate library size
            st.markdown("---")
            if st.button("📊 Calculate Library Size", use_container_width=True):
                try:
                    from protein_design_hub.evolution.library_design import LibraryDesigner

                    designer = LibraryDesigner(parent_sequence=st.session_state.evolution_sequence)

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

                    seq = st.session_state.evolution_sequence
                    designer = LibraryDesigner(parent_sequence=seq)

                    if library_type == "Site-saturation mutagenesis" and target_positions:
                        # target_positions from text_input are 1-indexed (human-readable)
                        positions_1idx = [int(p.strip()) - 1 for p in target_positions.split(",")]

                        library = designer.create_saturation_library(
                            seq,
                            positions=positions_1idx,
                            max_variants=20,
                        )

                        st.markdown("**Sample variants (first 10):**")
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

        _exp_col1, _exp_col2 = st.columns([2, 1])
        with _exp_col1:
            export_format = st.selectbox("Format", ["FASTA", "CSV", "DNA (with primers)"], key="lib_export_fmt")
        with _exp_col2:
            _biophys_filter = st.checkbox(
                "Pre-filter: instability < 40 & GRAVY < 0",
                value=True,
                key="lib_biophys_filter",
                help="Remove variants with high instability index or positive GRAVY before synthesis ordering",
            )

        if st.button("📥 Generate & Download Library", type="primary", key="lib_export_btn"):
            _lib_seq = st.session_state.evolution_sequence
            _lib_variants: list = []

            try:
                from protein_design_hub.evolution.library_design import LibraryDesigner

                _designer = LibraryDesigner(parent_sequence=_lib_seq)

                if library_type == "Site-saturation mutagenesis" and target_positions:
                    _positions_0 = [int(p.strip()) - 1 for p in target_positions.split(",")]
                    _lib = _designer.create_saturation_library(
                        _lib_seq, _positions_0, max_variants=1000
                    )
                    _lib_variants = _lib.variants

                elif library_type == "Combinatorial":
                    # Parse position:AAs entries from session state
                    _comb_defs = [
                        st.session_state.get("comb_pos1", ""),
                        st.session_state.get("comb_pos2", ""),
                        st.session_state.get("comb_pos3", ""),
                    ]
                    _comb_pos_aas: dict = {}
                    for _cdef in _comb_defs:
                        _cdef = (_cdef or "").strip()
                        if ":" in _cdef:
                            _cp, _caas = _cdef.split(":", 1)
                            try:
                                _comb_pos_aas[int(_cp.strip()) - 1] = list(_caas.strip().upper())
                            except ValueError:
                                pass
                    if _comb_pos_aas:
                        # Generate all combinations
                        import itertools
                        _positions_c = sorted(_comb_pos_aas.keys())
                        _aa_choices = [_comb_pos_aas[p] for p in _positions_c]
                        for _combo in itertools.product(*_aa_choices):
                            _s = list(_lib_seq)
                            for _pi, _aa in zip(_positions_c, _combo):
                                if 0 <= _pi < len(_s):
                                    _s[_pi] = _aa
                            _lib_variants.append("".join(_s))
                        # Cap at 2000 for performance
                        if len(_lib_variants) > 2000:
                            import random as _rnd
                            _lib_variants = _rnd.sample(_lib_variants, 2000)
                            st.warning("Library capped at 2000 variants (random sample from full combinatorial space).")
                    else:
                        st.warning("No valid combinatorial positions defined. Use format '42:AILVM'.")

                elif library_type == "Error-prone PCR simulation":
                    import random
                    _AAs = "ACDEFGHIKLMNPQRSTVWY"
                    for _ in range(int(num_variants)):
                        _s = list(_lib_seq)
                        _n_muts = max(1, int(len(_lib_seq) * error_rate / 100))
                        for _ in range(_n_muts):
                            _pos = random.randrange(len(_s))
                            _s[_pos] = random.choice(_AAs)
                        _lib_variants.append("".join(_s))

            except Exception as _lib_err:
                st.warning(f"Library generation error: {_lib_err}")

            # Biophysical pre-filter
            if _lib_variants and _biophys_filter:
                try:
                    from protein_design_hub.biophysics import (
                        calculate_instability_index, calculate_gravy,
                    )
                    _n_before = len(_lib_variants)
                    _lib_variants = [
                        v for v in _lib_variants
                        if calculate_instability_index(v) < 40 and calculate_gravy(v) < 0
                    ]
                    st.info(
                        f"Biophysical filter: {len(_lib_variants)}/{_n_before} variants pass "
                        f"(instability < 40 AND GRAVY < 0) — ready for gene synthesis ordering."
                    )
                except Exception:
                    st.caption("Biophysical pre-filter skipped (biophysics module unavailable).")

            if not _lib_variants:
                st.info("No variants generated — check positions are valid for this sequence.")
            else:
                st.success(f"Generated {len(_lib_variants)} variants ready for export.")

                if export_format == "FASTA":
                    _content = "\n".join(
                        f">variant_{i+1:04d}\n{v}"
                        for i, v in enumerate(_lib_variants)
                    )
                    _fname = "library.fasta"
                    _mime = "text/plain"

                elif export_format == "CSV":
                    import io as _io
                    _buf = _io.StringIO()
                    _buf.write("id,sequence,length\n")
                    for i, v in enumerate(_lib_variants):
                        _buf.write(f"variant_{i+1:04d},{v},{len(v)}\n")
                    _content = _buf.getvalue()
                    _fname = "library.csv"
                    _mime = "text/csv"

                else:  # DNA with primers
                    _CODON: dict = {
                        "A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT",
                        "G": "GGT", "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTG",
                        "M": "ATG", "N": "AAT", "P": "CCT", "Q": "CAA", "R": "CGT",
                        "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAT",
                    }
                    lines = []
                    for i, v in enumerate(_lib_variants[:200]):
                        dna = "".join(_CODON.get(aa, "NNN") for aa in v)
                        lines.append(f">variant_{i+1:04d}_dna\n{dna}")
                    _content = "\n".join(lines)
                    _fname = "library_dna.fasta"
                    _mime = "text/plain"

                st.download_button(
                    f"📥 Download {export_format} ({len(_lib_variants)} variants)",
                    data=_content,
                    file_name=_fname,
                    mime=_mime,
                    use_container_width=True,
                )

st.divider()
_evo_pymol_port = 0
try:
    from protein_design_hub.web.pymol_server import get_pymol_server as _gps_evo2
    _srv_evo2 = _gps_evo2()
    if _srv_evo2 is not None:
        _evo_pymol_port = _srv_evo2.server_address[1]
except Exception:
    pass
if _evo_pymol_port:
    import time as _evt
    _evts = int(_evt.time())
    _evh = "localhost"
    try:
        _evh = st.context.headers.get("host", "localhost").split(":")[0]
    except Exception:
        pass
    from protein_design_hub.web.ui import section_header as _evo_sh
    _evo_sh("PyMolAI Chat", "Molecular-visualization AI with live PyMOL access", "🔬")
    _ec1, _ev1 = st.columns([55, 45], gap="medium")
    with _ec1:
        render_pymolai_chatbot(key_prefix="evolution_pymolai", pymol_port=_evo_pymol_port)
    with _ev1:
        render_pymolai_viewer(_evo_pymol_port, "pdh-pymol-evolution", height=700)
