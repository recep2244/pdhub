# Phase 6: New Agent Workflows - Research

**Researched:** 2026-02-23
**Domain:** Python / Streamlit — extending existing agent orchestration infrastructure with two new pipeline modes surfaced in the web UI
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AGT-01 | Antibody/nanobody design pipeline using existing nanobody team preset with LLM guidance | `NANOBODY_TEAM_MEMBERS` and "nanobody" entry in `ALL_TEAMS` already exist in `scientists.py`; orchestrator accepts custom `agents` list; UI pipeline-mode selector is in Tab 1 of `11_agents.py` |
| AGT-02 | Binding affinity analysis workflow using existing biophysicist persona and evaluation metrics | `BIOPHYSICIST` agent exists in `scientists.py`; `context.evaluation_results` carries all computed metrics; a new LLM-only mode can run `LLMEvaluationReviewAgent` with BIOPHYSICIST as sole team member against existing evaluation data |
</phase_requirements>

---

## Summary

Phase 6 adds two purpose-built pipeline modes to the existing Agents page (`11_agents.py`). The entire infrastructure — agents, orchestrator, meeting runner, team presets, and the UI tab structure — already exists and is battle-tested. The work is almost entirely additive: (1) register two new orchestrator modes, (2) wire them into the Pipeline Settings selector in Tab 1, and (3) ensure the results panel renders the relevant LLM discussion summaries.

For AGT-01 (Antibody/Nanobody Design), the full 12-step LLM-guided pipeline is run unchanged but with `NANOBODY_TEAM_MEMBERS` (Immunologist, Structural Biologist, ML Specialist, Scientific Critic) substituted for the default team in every LLM agent. The `nanobody` team preset already exists in both `scientists.py` and the `_load_team_presets()` function in the UI. The only missing piece is a named orchestrator mode that passes this team composition through to every LLM agent constructor.

For AGT-02 (Binding Affinity Analysis), the goal is a lightweight LLM-only pipeline that interprets **already-computed** evaluation metrics from a previously-run job. It runs no new structure predictions. The pipeline is: load evaluation context from disk → run `LLMEvaluationReviewAgent` with `BIOPHYSICIST` as sole team member → produce a binding-affinity-focused narrative. This can be triggered from the same Tab 1 UI by pointing it at an existing job output directory.

**Primary recommendation:** Add two new orchestrator modes (`"nanobody_llm"` and `"binding_affinity"`). Register them in `_build_*` builder functions inside `orchestrator.py`. Add the two mode labels to the pipeline-mode `st.selectbox` in Tab 1. No new agent classes, no new files beyond these two touch points.

---

## Standard Stack

No new libraries are needed. This phase is pure orchestration configuration using what is already installed.

### Core (existing, already in use)
| Component | Location | Purpose | Why Standard |
|-----------|----------|---------|--------------|
| `scientists.py` | `agents/scientists.py` | Agent persona definitions and team presets | Single source of truth for all team compositions |
| `orchestrator.py` | `agents/orchestrator.py` | Pipeline builder and runner | Already handles `"llm"`, `"step"`, `"mutagenesis_pre"`, `"mutagenesis_post"` modes |
| `llm_guided.py` | `agents/llm_guided.py` | All LLM review agent classes | Fully parameterized — team_lead and team_members are constructor args |
| `11_agents.py` | `web/pages/11_agents.py` | Agents UI page with Tab 1 pipeline runner | Already has mode selector; just needs new options |
| `meeting.py` | `agents/meeting.py` | LLM meeting runner (cached client, GPU TTL) | Unchanged — all new modes reuse it |

### Key Existing Facts (HIGH confidence — verified by reading source)

- `LLMInputReviewAgent`, `LLMPlanningAgent`, `LLMPredictionReviewAgent`, `LLMEvaluationReviewAgent`, `LLMRefinementReviewAgent`, `LLMMutagenesisPlanningAgent`, `LLMReportNarrativeAgent` all accept `team_lead=` and `team_members=` as constructor kwargs.
- `NANOBODY_TEAM_MEMBERS = (IMMUNOLOGIST, STRUCTURAL_BIOLOGIST, MACHINE_LEARNING_SPECIALIST, SCIENTIFIC_CRITIC)` defined at module level in `scientists.py`.
- `BIOPHYSICIST` agent defined at module level in `scientists.py` with full thermodynamics/binding expertise.
- `ALL_TEAMS["nanobody"]` entry exists in `scientists.py` (used by Tab 2 meeting selector). Tab 1 pipeline runner does NOT use `ALL_TEAMS` — it uses hardcoded mode strings.
- `AgentOrchestrator.__init__` dispatches on mode string: `"llm"`, `"step"`, `"mutagenesis_pre"`, `"mutagenesis_post"`, or custom `agents=` list. A new `elif mode == "nanobody_llm":` block is the canonical extension point.
- `_AGENT_LABELS` dict in `orchestrator.py` maps step names to display labels; new modes reuse the same step names, no new entries needed.
- `_LLM_VERDICT_KEYS` dict maps step names to verdict storage keys; new modes reuse the same keys.
- `context.evaluation_results` is populated by `EvaluationAgent` and persists to `WorkflowContext` — available to any downstream LLM agent.
- `WorkflowContext` is serializable enough to pass between pipeline steps; `extra` dict carries all LLM meeting summaries.
- Tab 1 results panel renders `ctx.extra` keys: `input_review`, `plan`, `prediction_review`, `evaluation_review`, `refinement_review`, `mutagenesis_plan`, `executive_summary`. Any new keys from new workflows need to be added to `_lk` list at line 611 of `11_agents.py`.
- `_pipeline_table_markdown()` calls `AgentOrchestrator(mode=mode).describe_pipeline()` — new modes will auto-render in the pipeline preview table because they reuse existing `_AGENT_LABELS`.
- `_LLM_PIPELINE_STEPS` and `_STEP_PIPELINE_STEPS` are computed at module import time using modes `"llm"` and `"step"` — the header metric "Pipeline: N steps" is hardcoded to those two. New modes do not affect this.

---

## Architecture Patterns

### Pattern 1: New Orchestrator Mode via Builder Function

The established pattern (used for `mutagenesis_pre` and `mutagenesis_post`) is:

1. Define a `_build_<name>_pipeline(progress_callback, **kwargs) -> List[BaseAgent]` function in `orchestrator.py`.
2. Add an `elif mode == "<name>":` branch in `AgentOrchestrator.__init__` that calls the builder.
3. The builder assembles agents from existing classes, passing `team_lead` and `team_members` kwargs where needed.

```python
# Source: verified in orchestrator.py lines 59-104
def _build_nanobody_llm_pipeline(
    progress_callback=None, **kwargs
) -> List[BaseAgent]:
    """Full LLM-guided pipeline with nanobody team composition."""
    from protein_design_hub.agents.llm_guided import (
        LLMInputReviewAgent, LLMPlanningAgent, LLMPredictionReviewAgent,
        LLMEvaluationReviewAgent, LLMRefinementReviewAgent,
        LLMMutagenesisPlanningAgent, LLMReportNarrativeAgent,
    )
    from protein_design_hub.agents.scientists import (
        DEFAULT_TEAM_LEAD, NANOBODY_TEAM_MEMBERS,
    )
    from protein_design_hub.agents.input_agent import InputAgent
    from protein_design_hub.agents.prediction_agent import PredictionAgent
    from protein_design_hub.agents.evaluation_agent import EvaluationAgent
    from protein_design_hub.agents.comparison_agent import ComparisonAgent
    from protein_design_hub.agents.report_agent import ReportAgent

    nb_kwargs = dict(kwargs)
    nb_kwargs.setdefault("team_lead", DEFAULT_TEAM_LEAD)
    nb_kwargs.setdefault("team_members", NANOBODY_TEAM_MEMBERS)

    return [
        InputAgent(progress_callback=progress_callback),
        LLMInputReviewAgent(progress_callback=progress_callback, **nb_kwargs),
        LLMPlanningAgent(progress_callback=progress_callback, **nb_kwargs),
        PredictionAgent(progress_callback=progress_callback),
        LLMPredictionReviewAgent(progress_callback=progress_callback, **nb_kwargs),
        EvaluationAgent(progress_callback=progress_callback),
        ComparisonAgent(progress_callback=progress_callback),
        LLMEvaluationReviewAgent(progress_callback=progress_callback, **nb_kwargs),
        LLMRefinementReviewAgent(progress_callback=progress_callback, **nb_kwargs),
        LLMMutagenesisPlanningAgent(progress_callback=progress_callback, **nb_kwargs),
        LLMReportNarrativeAgent(progress_callback=progress_callback, **nb_kwargs),
        ReportAgent(progress_callback=progress_callback),
    ]
```

### Pattern 2: LLM-Only Pipeline (No Computation) for Binding Affinity

The binding affinity workflow interprets existing evaluation metrics. It does not re-run prediction or evaluation. This is a new pipeline composition pattern — but it fits within the same orchestrator `agents=` custom mode.

The pipeline for `"binding_affinity"` mode:
1. `InputAgent` — needed to populate `context.sequences` (minimal, just parses FASTA).
2. `LLMEvaluationReviewAgent` initialized with `BIOPHYSICIST` as sole team member — produces binding affinity interpretation.

The context must have `evaluation_results` pre-populated. There are two approaches:
- **Approach A (simpler):** Run the full step pipeline first (step mode to get evaluation_results), then run binding affinity LLM on top. The binding affinity mode would be a post-processing pipeline: `[LLMEvaluationReviewAgent(team_members=(BIOPHYSICIST,))]` applied to an existing context using `run_with_context()`.
- **Approach B:** Include `InputAgent + PredictionAgent + EvaluationAgent + ComparisonAgent + LLMEvaluationReviewAgent` (binding_affinity = step pipeline + single LLM review). This is self-contained and requires no pre-populated context.

**Use Approach B.** It is consistent with the existing pipeline model where every mode is self-contained. Label it with a Biophysicist-focused agenda in `LLMEvaluationReviewAgent`.

```python
# Source: verified in orchestrator.py + llm_guided.py pattern
def _build_binding_affinity_pipeline(
    progress_callback=None, **kwargs
) -> List[BaseAgent]:
    """Step pipeline + biophysicist binding affinity LLM review."""
    from protein_design_hub.agents.llm_guided import LLMEvaluationReviewAgent
    from protein_design_hub.agents.scientists import DEFAULT_TEAM_LEAD, BIOPHYSICIST, SCIENTIFIC_CRITIC
    from protein_design_hub.agents.input_agent import InputAgent
    from protein_design_hub.agents.prediction_agent import PredictionAgent
    from protein_design_hub.agents.evaluation_agent import EvaluationAgent
    from protein_design_hub.agents.comparison_agent import ComparisonAgent
    from protein_design_hub.agents.report_agent import ReportAgent

    ba_kwargs = dict(kwargs)
    ba_kwargs.setdefault("team_lead", DEFAULT_TEAM_LEAD)
    ba_kwargs.setdefault("team_members", (BIOPHYSICIST, SCIENTIFIC_CRITIC))

    return [
        InputAgent(progress_callback=progress_callback),
        PredictionAgent(progress_callback=progress_callback),
        EvaluationAgent(progress_callback=progress_callback),
        ComparisonAgent(progress_callback=progress_callback),
        LLMEvaluationReviewAgent(progress_callback=progress_callback, **ba_kwargs),
        ReportAgent(progress_callback=progress_callback),
    ]
```

### Pattern 3: Adding New Pipeline Modes to the UI Selector

The `st.selectbox("Pipeline mode", ...)` in Tab 1 of `11_agents.py` (lines 339-343) controls which mode string is passed to `AgentOrchestrator`. The current options are:

```python
# Source: 11_agents.py lines 339-343
pm = st.selectbox("Pipeline mode", [
    "LLM-guided (recommended)",
    "Step-only (fast, no LLM)",
], key="p_mode")
use_llm = "LLM" in pm
```

The `mode` variable is derived with `mode = "llm" if use_llm else "step"`. This must be extended to a multi-way mapping:

```python
# New pattern: map selectbox choice to mode string
_MODE_MAP = {
    "LLM-guided (recommended)":         "llm",
    "Step-only (fast, no LLM)":         "step",
    "Antibody / Nanobody Design":        "nanobody_llm",
    "Binding Affinity Analysis":         "binding_affinity",
}
pm = st.selectbox("Pipeline mode", list(_MODE_MAP.keys()), key="p_mode")
mode = _MODE_MAP[pm]
use_llm = mode not in ("step",)
```

The `use_llm` flag controls whether the LLM settings section is rendered. Both new modes use LLM, so they should show LLM settings.

### Pattern 4: Rendering New Mode Discussion Keys in the Results Panel

Tab 1 renders LLM discussion summaries by iterating `_lk` (list of `(context_key, label)` pairs) at line 611. The nanobody pipeline reuses the same `context.extra` keys as the standard LLM pipeline (`input_review`, `plan`, etc.) because it reuses the same agent classes. No new keys are needed for AGT-01.

For AGT-02, `LLMEvaluationReviewAgent` writes to `context.extra["evaluation_review"]` — already in `_lk`. No new keys needed for AGT-02 either.

However, the pipeline preview table (`_pipeline_table_markdown`) currently hardcodes `mode="llm"`. This must be updated to reflect the selected mode:

```python
# Change from hardcoded:
with st.expander("How does the pipeline work?"):
    st.markdown(_pipeline_table_markdown("llm"))

# Change to selected mode:
with st.expander("How does the pipeline work?"):
    st.markdown(_pipeline_table_markdown(mode))  # mode derived from selectbox
```

### Recommended File Change Map

```
orchestrator.py
├── Add: _build_nanobody_llm_pipeline()
├── Add: _build_binding_affinity_pipeline()
└── Add: elif mode == "nanobody_llm": / elif mode == "binding_affinity": branches

web/pages/11_agents.py (Tab 1 only)
├── Change: st.selectbox options to include two new labels
├── Change: mode derivation from binary to _MODE_MAP lookup
├── Change: use_llm derivation to handle new modes
└── Change: _pipeline_table_markdown() call to use selected mode
```

### Anti-Patterns to Avoid

- **Creating new agent classes for new workflows:** The new workflows are pure team-composition variants, not new logic. `LLMInputReviewAgent`, `LLMPlanningAgent`, etc. already accept `team_members=` — pass `NANOBODY_TEAM_MEMBERS` at construction time instead of subclassing.
- **Duplicating the `_build_llm_guided_pipeline` function:** Copy-paste with team swap. Use `nb_kwargs.setdefault("team_members", NANOBODY_TEAM_MEMBERS)` to inject the team without repeating the full function body.
- **Adding new context.extra keys for the nanobody workflow:** Nanobody reuses the same 12-step pipeline structure. All existing `_lk` keys will be populated correctly. No new keys means no UI changes to the results display.
- **Making binding affinity an LLM-only mode with no computation:** Without running `PredictionAgent + EvaluationAgent`, there are no metrics for the Biophysicist to interpret. The mode must be self-contained (steps + LLM review).
- **Modifying `_LLM_PIPELINE_STEPS` or `_STEP_PIPELINE_STEPS` module-level variables:** These are used in the header metric card and should remain pointing to the primary "llm" and "step" modes. The pipeline preview table update is the correct way to show new mode steps.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Nanobody-specific LLM prompts | New agent class with hardcoded nanobody context | `NANOBODY_TEAM_MEMBERS` passed to existing LLM agents | The Immunologist persona already has full CDR/VHH/humanisation expertise baked in |
| Binding affinity metric extraction | Custom metric parser | `context.evaluation_results` + `_evaluation_detail_text()` in `llm_guided.py` | This helper already extracts all metrics (Rosetta REU, FoldX ddG, SASA, BSA, clash, shape complementarity) |
| New meeting runner for domain workflows | Separate `run_meeting()` call path | `_LLMGuidedMixin._run_meeting_if_enabled()` | Already handles save_dir scoping, verdict extraction, summary persistence |
| Pipeline preview for new modes | New UI section | `_pipeline_table_markdown(mode)` with dynamic mode arg | Already works generically from `AgentOrchestrator(mode=...).describe_pipeline()` |

**Key insight:** The orchestrator's `agents=` list and the LLM agents' `team_members=` constructor arg together provide a complete composition point. New workflows are new compositions, not new components.

---

## Common Pitfalls

### Pitfall 1: `use_llm` Flag Not Updated for New Modes
**What goes wrong:** The current code sets `use_llm = "LLM" in pm` (string containment check). If the new mode labels contain "LLM" or "Analysis" but not "LLM", the LLM settings panel may be hidden even though the mode requires LLM.
**Why it happens:** Simple string check designed for exactly two options.
**How to avoid:** Switch to explicit mapping: `use_llm = mode not in ("step",)`. Both new modes require LLM.
**Warning signs:** LLM settings section (provider/model/rounds) not showing when nanobody or binding affinity mode selected.

### Pitfall 2: `_LLM_PIPELINE_COUNT` Header Metric Shows Wrong Count
**What goes wrong:** The header metric card shows `_LLM_PIPELINE_COUNT` (computed from `"llm"` mode) regardless of selected mode. For binding affinity (6 steps), it would still show 12.
**Why it happens:** `_LLM_PIPELINE_STEPS` is module-level, computed once at import with `"llm"`.
**How to avoid:** The header metric is acceptable as-is (it reflects the flagship LLM pipeline). Do NOT change the module-level variable. The pipeline preview table (per-mode) is the accurate reference.
**Warning signs:** None — this is a known acceptable approximation.

### Pitfall 3: `kwargs` Forwarding Overrides Explicit Team Args
**What goes wrong:** `AgentOrchestrator.__init__` calls `_build_nanobody_llm_pipeline(progress_callback=..., **kwargs)`. If the caller passes `team_members=DEFAULT_TEAM_MEMBERS` in kwargs (e.g. from the UI), it would override the nanobody team.
**Why it happens:** Python `**kwargs` merging — later values win if using `dict(kwargs)` then setting.
**How to avoid:** In `_build_nanobody_llm_pipeline`, use `nb_kwargs.setdefault("team_members", NANOBODY_TEAM_MEMBERS)` not direct assignment. `setdefault` preserves any explicit caller override. Actually for nanobody, the team IS the point — use direct assignment: `nb_kwargs["team_members"] = NANOBODY_TEAM_MEMBERS` to always enforce nanobody team.
**Warning signs:** Pipeline runs with default team instead of Immunologist-led team.

### Pitfall 4: `_pipeline_table_markdown()` Called with Wrong Mode
**What goes wrong:** The expander at line 266 hardcodes `_pipeline_table_markdown("llm")`. After adding new modes, the "How does the pipeline work?" expander always shows the 12-step LLM pipeline even when nanobody or binding affinity is selected.
**Why it happens:** Hardcoded string pre-dates multi-mode support.
**How to avoid:** Pass the selected `mode` variable to the call. Requires moving the `mode` derivation above the expander (currently the expander is before the mode selector in the code — check layout carefully).
**Warning signs:** Pipeline explanation table always shows 12 steps regardless of selected mode.

### Pitfall 5: `@st.cache_data` on `_get_pipeline_steps(mode)` Caches Correctly
**What goes wrong:** Concern that adding new modes breaks the `@st.cache_data(ttl=30)` cache on `_get_pipeline_steps`.
**Why it happens:** Potential cache miss issue.
**How to avoid:** The existing `_get_pipeline_steps(mode: str)` function already takes `mode` as a parameter, and `@st.cache_data` caches by argument value. New mode strings are new cache keys — no conflict. The function will cache nanobody/binding_affinity results separately.
**Warning signs:** None — this works correctly by design.

### Pitfall 6: Binding Affinity Verdict Key Not in `_LLM_VERDICT_KEYS`
**What goes wrong:** If the binding affinity pipeline is halted on FAIL verdict, the `run_with_context()` loop looks up the agent name in `_LLM_VERDICT_KEYS`. `LLMEvaluationReviewAgent` has name `"llm_evaluation_review"` — already in `_LLM_VERDICT_KEYS`. No issue.
**How to avoid:** N/A — already works. Document it here to avoid confusion.

---

## Code Examples

### Orchestrator Mode Dispatch (existing pattern to follow)

```python
# Source: verified in orchestrator.py lines 192-209
elif mode == "mutagenesis_pre":
    self.agents = _build_mutagenesis_pre_approval_pipeline(
        progress_callback=progress_callback, **kwargs,
    )
elif mode == "mutagenesis_post":
    self.agents = _build_mutagenesis_post_approval_pipeline(
        progress_callback=progress_callback, **kwargs,
    )
# New entries follow this exact pattern:
elif mode == "nanobody_llm":
    self.agents = _build_nanobody_llm_pipeline(
        progress_callback=progress_callback, **kwargs,
    )
elif mode == "binding_affinity":
    self.agents = _build_binding_affinity_pipeline(
        progress_callback=progress_callback, **kwargs,
    )
```

### UI Mode Selector Update (Tab 1 of 11_agents.py)

```python
# Source: current code at 11_agents.py lines 339-343
# BEFORE:
pm = st.selectbox("Pipeline mode", [
    "LLM-guided (recommended)",
    "Step-only (fast, no LLM)",
], key="p_mode")
use_llm = "LLM" in pm

# AFTER:
_PIPELINE_MODES = {
    "LLM-guided (recommended)":       "llm",
    "Step-only (fast, no LLM)":       "step",
    "Antibody / Nanobody Design":     "nanobody_llm",
    "Binding Affinity Analysis":      "binding_affinity",
}
pm = st.selectbox("Pipeline mode", list(_PIPELINE_MODES.keys()), key="p_mode")
mode_str = _PIPELINE_MODES[pm]
use_llm = mode_str != "step"
```

### LLM Agent Team Injection (pattern for nanobody builder)

```python
# Source: verified in llm_guided.py LLMInputReviewAgent.__init__ lines 364-373
# All LLM agents accept team_lead and team_members as constructor kwargs.
# Verified: LLMInputReviewAgent, LLMPlanningAgent, LLMPredictionReviewAgent,
# LLMEvaluationReviewAgent, LLMRefinementReviewAgent, LLMMutagenesisPlanningAgent,
# LLMReportNarrativeAgent.
LLMInputReviewAgent(
    progress_callback=progress_callback,
    team_lead=DEFAULT_TEAM_LEAD,
    team_members=NANOBODY_TEAM_MEMBERS,
    num_rounds=kwargs.get("num_rounds", 1),
)
```

### Evaluation Detail Available to Biophysicist

```python
# Source: verified in llm_guided.py lines 293-348
# _evaluation_detail_text() already extracts all binding-relevant metrics:
# Rosetta REU, FoldX ddG, GBSA energy, SASA, BSA (buried surface area),
# shape complementarity, salt bridges, VoroMQA, clash score.
# LLMEvaluationReviewAgent calls this automatically — no custom code needed.
eval_text = _evaluation_detail_text(context)
# Result passed directly into the meeting agenda.
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Single hardcoded team for all pipeline runs | Team injected via constructor kwargs | New workflows are pure composition — no code duplication |
| Binary `use_llm` flag (LLM vs step) | Multi-mode string mapping | Cleanly extensible for Phase 8 predictor modes too |

**No deprecated patterns apply.** This phase extends stable infrastructure.

---

## Open Questions

1. **Should binding affinity mode accept an existing job directory as input (skip prediction)?**
   - What we know: `AgentOrchestrator.run()` always starts fresh from a FASTA file. `run_with_context()` accepts a pre-populated context.
   - What's unclear: AGT-02 says "invokes the biophysicist persona against existing evaluation metrics." This could mean: (a) run full pipeline and then biophysicist review, or (b) pick up existing metrics from a previous job.
   - Recommendation: Implement Approach B (self-contained 6-step pipeline: input + prediction + evaluation + comparison + biophysicist LLM review + report). This satisfies the success criterion "running it produces a binding affinity interpretation in the report" without requiring cross-job context loading. Cross-job loading can be deferred.

2. **Does the Nanobody pipeline need a nanobody-specific system prompt in `LLMInputReviewAgent`?**
   - What we know: The Immunologist persona already has VHH/CDR/humanisation expertise. The agenda text ("identify protein type") will auto-route to antibody/nanobody discussion once the Immunologist is in the team.
   - What's unclear: Whether the Input Review agent should be pre-seeded with a nanobody-specific agenda.
   - Recommendation: No change needed. The Immunologist persona guides the discussion organically. Adding a nanobody preamble to the agenda would be a refinement, not a requirement for AGT-01.

3. **Should `_AGENT_LABELS` in `orchestrator.py` get new entries for new pipeline modes?**
   - What we know: `_AGENT_LABELS` maps step agent names (not mode names) to display strings. New modes reuse existing step agents. No new entries are needed.
   - Recommendation: No changes to `_AGENT_LABELS`.

---

## Sources

### Primary (HIGH confidence)
- Direct source reading of `src/protein_design_hub/agents/orchestrator.py` — verified all mode dispatch patterns, `_build_*` builder functions, `_AGENT_LABELS`, `_LLM_VERDICT_KEYS`
- Direct source reading of `src/protein_design_hub/agents/scientists.py` — verified `NANOBODY_TEAM_MEMBERS`, `BIOPHYSICIST`, `ALL_TEAMS`, all team compositions
- Direct source reading of `src/protein_design_hub/agents/llm_guided.py` — verified all LLM agent constructors accept `team_lead` and `team_members` kwargs; verified `_evaluation_detail_text()` content
- Direct source reading of `src/protein_design_hub/web/pages/11_agents.py` — verified Tab 1 pipeline runner code (mode selector, execution block, results rendering at `_lk`)
- Direct source reading of `src/protein_design_hub/agents/context.py` — verified `WorkflowContext` fields
- Direct source reading of `src/protein_design_hub/web/agent_helpers.py` — verified `render_all_experts_panel` and `_temporary_llm_override` patterns

### Secondary (MEDIUM confidence)
- `.planning/REQUIREMENTS.md` — AGT-01 and AGT-02 definitions confirmed
- `.planning/ROADMAP.md` — Phase 6 success criteria confirmed
- `.planning/STATE.md` — Phase 5 complete, Phase 6 is the active next phase

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all components verified by direct source reading
- Architecture patterns: HIGH — patterns verified against existing implementations (`mutagenesis_pre`/`mutagenesis_post` are exact precedents)
- Pitfalls: HIGH — all pitfalls identified from direct code analysis, not speculation
- Open questions: MEDIUM — implementation choice for AGT-02 input method; does not block planning

**Research date:** 2026-02-23
**Valid until:** 60 days — codebase is stable; no external dependencies involved
