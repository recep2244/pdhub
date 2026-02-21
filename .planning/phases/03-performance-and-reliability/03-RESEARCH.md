# Phase 3: Performance & Reliability - Research

**Researched:** 2026-02-21
**Domain:** Python guard clauses, Streamlit warning surfaces, version-checking patterns
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PERF-01 | OST comprehensive scoring made optional with a flag; disabled by default when >3 mutation positions | Guard logic in `_run_phase2()` counts approved positions and enforces auto-disable; `_build_scanner()` in `mutagenesis_agents.py` must respect or override that decision |
| PERF-02 | Silent saturation fallback (LLM plan parsing failure) surfaces a clear warning to user before executing | The fallback path in `LLMMutationSuggestionAgent.run()` currently logs only to `logger.warning`; surfacing to UI requires writing to `context.extra` and rendering in the web page before Phase 2 runs |
| PERF-03 | `_build_scanner()` TypeError fallback replaced with explicit version check and helpful ImportError message | Current `try/except TypeError` in `mutagenesis_agents._build_scanner()` hides version mismatches silently; replace with `inspect.signature` check at import time |
</phase_requirements>

---

## Summary

Phase 3 makes three targeted surgical changes to Python code — no new libraries, no new pages. All three requirements address the same class of problem: behaviour that should be loud (warnings, errors) is currently silent (catches exceptions, logs only to file). The fixes are confined to two files: `mutagenesis_agents.py` (PERF-01 and PERF-03) and the `LLMMutationSuggestionAgent` class in `llm_guided.py` (PERF-02), with a rendering addition to `10_mutation_scanner.py` for PERF-02 UI surface.

PERF-01 requires counting distinct `residue` positions in `approved_mutations` (not total variants) before calling `_build_scanner()` inside `MutationExecutionAgent.run()`. When the count exceeds 3 and `run_openstructure_comprehensive=True` was requested, OST must be forced off with a warning logged at WARNING level AND stored in `context.extra` for the UI to surface. An explicit boolean kwarg `force_ost=True` (or similar) passed into `MutationExecutionAgent` allows the user to override the auto-disable.

PERF-02 requires the saturation fallback branch inside `LLMMutationSuggestionAgent.run()` to write its warning message into `context.extra["mutation_suggestion_warning"]` (or equivalent) before `AgentResult.ok()` is returned, and the `_render_phase2_results()` or pre-phase-2 UI flow in `10_mutation_scanner.py` to check and display that key as `st.warning(...)` before showing mutation results.

PERF-03 replaces the runtime `try/except TypeError` in `mutagenesis_agents._build_scanner()` with a module-level import-time check using `inspect.signature(MutationScanner.__init__)`. If the parameter is absent, raise `ImportError` with a message stating the minimum required attribute (`run_openstructure_comprehensive`, added at the time the agent was written) and actionable upgrade instructions.

**Primary recommendation:** All three fixes are pure Python guard-clause patterns. No new dependencies. Each fix is a single focused code change of under 30 lines. Plan as three independent tasks, one per requirement.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `inspect` | built-in | Introspect function signatures at runtime | Only correct way to check parameter existence without calling the function |
| Python stdlib `warnings` | built-in | Issue `UserWarning` with stack trace | Standard Python warning mechanism; integrates with pytest `warns` |
| Streamlit `st.warning()` | already in project | Surface warnings in UI | Already used throughout the codebase for UI alerts |
| Python logging | already in project | `logger.warning(...)` for server-side logs | Already used in every agent |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `inspect.signature` | built-in | Check for parameter existence in `__init__` | PERF-03: version gate at import time |
| `inspect.Parameter` | built-in | Enumerate parameter names | Companion to `inspect.signature` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `inspect.signature` | `hasattr(MutationScanner, '_some_attr')` | Attribute check is fragile; signature check is the exact right tool — it checks the constructor contract |
| `inspect.signature` | `packaging.version` + a `__version__` string | `MutationScanner` has no `__version__`; adding one is more invasive than needed |
| `inspect.signature` | `try: MutationScanner(run_openstructure_comprehensive=False)` | Creates a side-effect object just to test the signature; wrong |

**Installation:** No new packages required.

---

## Architecture Patterns

### Recommended File Changes

```
src/protein_design_hub/agents/
├── mutagenesis_agents.py     # PERF-01 (MutationExecutionAgent.run guard) + PERF-03 (_build_scanner)
└── (no other files changed)

src/protein_design_hub/agents/
└── llm_guided.py             # PERF-02 (LLMMutationSuggestionAgent fallback warning)

src/protein_design_hub/web/pages/
└── 10_mutation_scanner.py    # PERF-02 (display mutation_suggestion_warning before Phase 2 results)
```

### Pattern 1: OST Position Cap (PERF-01)

**What:** Count distinct mutation positions BEFORE constructing the scanner. If >3 and OST was requested without explicit override, force OST off and log a warning.

**When to use:** At the start of `MutationExecutionAgent.run()`, before `_build_scanner()` is called.

**Current code path (in `mutagenesis_agents.py`):**

```python
# Current (lines 115):
scanner = _build_scanner(output_dir=mut_dir)
```

`_build_scanner()` currently always passes `run_openstructure_comprehensive=True` — there is no cap.

**New pattern:**

```python
# Source: project codebase pattern, aligned with success criterion
def run(self, context: WorkflowContext) -> AgentResult:
    approved = context.extra.get("approved_mutations", [])
    # ... existing early-exit logic ...

    # PERF-01: OST cap
    force_ost = context.extra.get("ost_force_override", False)
    n_positions = len({entry["residue"] for entry in approved})
    ost_requested = True  # _build_scanner always enables OST currently
    if ost_requested and n_positions > 3 and not force_ost:
        logger.warning(
            "OST comprehensive scoring auto-disabled: %d positions > 3 limit. "
            "Set context.extra['ost_force_override'] = True to override.",
            n_positions,
        )
        context.extra["ost_auto_disabled"] = True
        context.extra["ost_auto_disabled_reason"] = (
            f"Auto-disabled: {n_positions} positions > 3 position limit."
        )
        scanner = _build_scanner(output_dir=mut_dir, run_ost=False)
    else:
        scanner = _build_scanner(output_dir=mut_dir, run_ost=True)
```

`_build_scanner()` needs a `run_ost: bool = True` parameter to pass through:

```python
def _build_scanner(output_dir=None, run_ost: bool = True):
    kwargs = {
        "predictor": "esmfold_api",
        "evaluation_metrics": ["openmm_gbsa", "cad_score", "voromqa"],
        "run_openstructure_comprehensive": run_ost,
    }
    if output_dir:
        kwargs["output_dir"] = output_dir
    return MutationScanner(**kwargs)
    # Note: PERF-03 removes the try/except; PERF-03 is applied first
```

**UI surface:** In `10_mutation_scanner.py`, in `_run_phase2()` or in the pre-phase2 approval step, check `ctx.extra.get("ost_auto_disabled")` after Phase 2 runs (or before, if tracked in session state) and display `st.warning(...)`.

The success criterion says: "shows a warning and automatically disables OST ... user can override with an explicit flag." The explicit flag maps to `context.extra["ost_force_override"] = True`, settable via UI checkbox or CLI.

### Pattern 2: Fallback Warning Surface (PERF-02)

**What:** When `_parse_mutation_plan_from_summary()` returns `None` in `LLMMutationSuggestionAgent.run()`, write the warning text into `context.extra` before returning `AgentResult.ok()`. The web page reads this key and calls `st.warning()`.

**Current code (llm_guided.py ~line 1342):**

```python
# Current — only logs to file:
logging.getLogger(__name__).warning(
    "MUTATION_PLAN_JSON parsing failed; falling back to saturation."
)
```

**New pattern:**

```python
# Source: project pattern from _run_phase2() in 10_mutation_scanner.py
n_fallback = len(fallback_positions)
warning_msg = (
    f"LLM plan unparseable — falling back to saturation at {n_fallback} positions."
)
logger.warning(warning_msg)
context.extra["mutation_suggestion_warning"] = warning_msg
```

**Web page rendering (10_mutation_scanner.py):**

The warning must appear BEFORE mutations execute, i.e., before `_run_phase2()` is called or at the very top of `_run_phase2()`:

```python
def _run_phase2(ctx):
    # Display fallback warning if LLM plan parsing failed
    warning = ctx.extra.get("mutation_suggestion_warning")
    if warning:
        st.warning(warning)
    # ... rest of phase 2 ...
```

This satisfies the success criterion: "UI displays a clear warning ... before any mutations are executed." Phase 1 runs first and populates `ctx.extra`; Phase 2 reads it at the start of `_run_phase2()`.

### Pattern 3: Import-Time Version Gate (PERF-03)

**What:** At module import time in `mutagenesis_agents.py`, use `inspect.signature` to verify that `MutationScanner.__init__` accepts `run_openstructure_comprehensive`. If it does not, raise `ImportError` immediately with a helpful message.

**When to use:** Module-level, after the `from protein_design_hub.analysis.mutation_scanner import MutationScanner` import.

**Current code (mutagenesis_agents.py lines 37-44):**

```python
def _build_scanner(output_dir=None):
    from protein_design_hub.analysis.mutation_scanner import MutationScanner
    kwargs = {
        "predictor": "esmfold_api",
        "evaluation_metrics": [...],
        "run_openstructure_comprehensive": True,
    }
    try:
        return MutationScanner(**kwargs)
    except TypeError:
        # Older version — silently falls back
        kwargs.pop("run_openstructure_comprehensive", None)
        scanner = MutationScanner(**kwargs)
        setattr(scanner, "run_openstructure_comprehensive", True)
        return scanner
```

**New pattern:**

```python
# Source: Python stdlib inspect module
import inspect

# At module level (after imports), verify MutationScanner API
def _check_scanner_api() -> None:
    """Raise ImportError if MutationScanner is missing required parameters."""
    from protein_design_hub.analysis.mutation_scanner import MutationScanner
    sig = inspect.signature(MutationScanner.__init__)
    if "run_openstructure_comprehensive" not in sig.parameters:
        raise ImportError(
            "mutagenesis_agents requires MutationScanner with "
            "'run_openstructure_comprehensive' parameter (added 2025-Q4). "
            "Upgrade protein_design_hub or reinstall from source: "
            "'pip install -e .' from the project root."
        )

_check_scanner_api()
```

Then `_build_scanner()` drops the `try/except` entirely:

```python
def _build_scanner(output_dir=None, run_ost: bool = True):
    from protein_design_hub.analysis.mutation_scanner import MutationScanner
    kwargs = {
        "predictor": "esmfold_api",
        "evaluation_metrics": ["openmm_gbsa", "cad_score", "voromqa"],
        "run_openstructure_comprehensive": run_ost,
    }
    if output_dir:
        kwargs["output_dir"] = output_dir
    return MutationScanner(**kwargs)
```

**Key constraint:** `_check_scanner_api()` must run lazily (inside a function called at module level) rather than at the `import` statement level, to avoid circular import issues. The pattern above — a `_check_scanner_api()` function called once at module level — is the correct approach.

### Anti-Patterns to Avoid

- **Checking version strings:** `MutationScanner` has no `__version__`. Do not add one just for this check — parameter inspection is cleaner.
- **Catching `ImportError` from `_check_scanner_api()`:** The whole point is to let it propagate. Do not wrap it in another try/except.
- **Surfacing PERF-02 warning only in logs:** The success criterion explicitly says "UI displays." `st.warning()` in the web page is required.
- **OST cap by total variants instead of positions:** A 4-position run with targeted (non-saturation) mutations might only produce 8 variants — still cheap. The cap must count distinct positions, not total variants. The success criterion says ">3 mutation positions."
- **Calling `_check_scanner_api()` inside `_build_scanner()`:** This would repeat the check on every scanner creation. Call it once at module import time.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parameter existence check | Custom string parsing of `MutationScanner.__doc__` or source inspection | `inspect.signature(MutationScanner.__init__).parameters` | Stdlib, reliable, handles `*args/**kwargs` correctly |
| UI warning display | Custom HTML injection in Streamlit | `st.warning(message)` | Already used in codebase; correct semantic; renders with yellow warning box |
| Position counting | `len(approved)` | `len({e["residue"] for e in approved})` | Must count distinct positions, not total entries (a position can appear once per entry) |

**Key insight:** All three requirements are expressible with Python idioms already in use in this codebase. The test for whether this phase is done is behavioral, not architectural.

---

## Common Pitfalls

### Pitfall 1: Counting Variants Instead of Positions for OST Cap

**What goes wrong:** `len(approved_mutations)` counts entries, not positions. If the same position appears twice (different target AAs in two rows), it gets double-counted.

**Why it happens:** The `approved_mutations` list structure is `[{"residue": int, "targets": [...]}]`. Each dict is one position, but the code might confuse "approved entries" with "approved positions."

**How to avoid:** `n_positions = len({entry["residue"] for entry in approved})` — use a set comprehension on the `residue` field.

**Warning signs:** OST cap triggers for 2 positions when user selected only 2 distinct positions.

### Pitfall 2: PERF-02 Warning Displayed After Phase 2 Already Ran

**What goes wrong:** If the warning is only displayed in `_render_phase2_results()` (after Phase 2 completes), the success criterion "before any mutations are executed" is violated.

**Why it happens:** The natural place to display results is after completion. The warning needs to be in `_run_phase2()` at its very first line, before `orchestrator.run_with_context(ctx)` is called.

**How to avoid:** Place `st.warning(ctx.extra.get("mutation_suggestion_warning"))` at the TOP of `_run_phase2()`, before the `with st.status(...)` block.

**Warning signs:** Warning appears below mutation results table, not above the spinner.

### Pitfall 3: `_check_scanner_api()` Called at Class Definition Scope

**What goes wrong:** If the check runs at module parse time (not inside a function), circular imports in the `protein_design_hub` package can cause the check to fail even with a correct scanner version.

**Why it happens:** `mutagenesis_agents.py` imports from `protein_design_hub.agents.base` and `protein_design_hub.agents.context`, which may not have resolved `mutation_scanner` yet.

**How to avoid:** Wrap the check in a `_check_scanner_api()` function and call it at module level. The function body is only executed when the function is called (not when it is defined), so imports inside the function body are deferred.

**Warning signs:** `ImportError: cannot import name 'MutationScanner'` during module load, even with correct version.

### Pitfall 4: PERF-01 Override Flag Not Plumbed to UI

**What goes wrong:** `ost_force_override` is set in `context.extra` but there is no way for the user to set it from the web UI.

**Why it happens:** The phase description says "user can override with an explicit flag" — this flag must be surfaceable. If it lives only in code, it is not accessible.

**How to avoid:** Add a Streamlit checkbox in the Phase 2 section: "Force OST scoring even with >3 positions (slow)" that sets `ctx.extra["ost_force_override"] = True` before calling `_run_phase2(ctx)`. Alternative: a `force_ost` kwarg passed into `MutationExecutionAgent` constructor.

**Warning signs:** The override mechanism exists in code but is not reachable from the web UI.

---

## Code Examples

Verified patterns from project source:

### `inspect.signature` parameter check

```python
# Source: Python stdlib — https://docs.python.org/3/library/inspect.html#inspect.signature
import inspect

sig = inspect.signature(SomeClass.__init__)
if "some_param" not in sig.parameters:
    raise ImportError("SomeClass requires 'some_param' parameter ...")
```

This is the correct approach because:
- Works with positional, keyword, and default parameters
- Does not instantiate the class
- Raises at import time so the error is immediate

### Set comprehension for distinct positions

```python
# Source: project codebase pattern
approved = context.extra.get("approved_mutations", [])
n_positions = len({entry["residue"] for entry in approved})
```

### Streamlit warning display

```python
# Source: Streamlit docs + existing usage in 10_mutation_scanner.py
warning = ctx.extra.get("mutation_suggestion_warning")
if warning:
    st.warning(warning)
```

### Writing a warning into context.extra for downstream UI consumption

```python
# Source: project pattern — context.extra is the inter-agent communication channel
warning_msg = f"LLM plan unparseable — falling back to saturation at {n} positions."
logger.warning(warning_msg)
context.extra["mutation_suggestion_warning"] = warning_msg
```

---

## Key Code Locations

These are the exact edit sites for each requirement:

| Req | File | Location | Change |
|-----|------|----------|--------|
| PERF-01 | `agents/mutagenesis_agents.py` | `MutationExecutionAgent.run()` top, before `_build_scanner()` call | Add position count guard, conditionally disable OST, store in `context.extra` |
| PERF-01 | `agents/mutagenesis_agents.py` | `_build_scanner()` signature | Add `run_ost: bool = True` parameter; drop try/except (done by PERF-03) |
| PERF-01 | `web/pages/10_mutation_scanner.py` | `_run_phase2()` | Display `st.warning(ctx.extra["ost_auto_disabled_reason"])` if flag set |
| PERF-01 | `web/pages/10_mutation_scanner.py` | Phase 2 approval section | Add "Force OST" checkbox that sets `ctx.extra["ost_force_override"]` |
| PERF-02 | `agents/llm_guided.py` | `LLMMutationSuggestionAgent.run()` fallback branch (~line 1342) | Add `context.extra["mutation_suggestion_warning"] = warning_msg` |
| PERF-02 | `web/pages/10_mutation_scanner.py` | `_run_phase2()` top | Add `st.warning(ctx.extra.get("mutation_suggestion_warning"))` if present |
| PERF-03 | `agents/mutagenesis_agents.py` | Module level, after imports | Add `_check_scanner_api()` function + call |
| PERF-03 | `agents/mutagenesis_agents.py` | `_build_scanner()` | Remove `try/except TypeError` block entirely |

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `try/except TypeError` for scanner version compat | `inspect.signature` check at import time | Phase 3 | Errors surface immediately at import, not buried in mutation scan execution |
| Fallback path logs to `logger.warning` only | Fallback writes to `context.extra` + `st.warning` in UI | Phase 3 | User sees the fallback decision before mutations execute |
| OST always enabled in agent (uncapped) | OST auto-disabled for >3 positions unless overridden | Phase 3 | Prevents unintended multi-hour runs |

---

## Open Questions

1. **Where exactly should the PERF-01 override be exposed in the UI?**
   - What we know: The approval table is rendered by `_render_approval_step()` in `10_mutation_scanner.py`
   - What's unclear: Whether the override checkbox belongs in the approval table or in a sidebar setting
   - Recommendation: Place it inside `_render_approval_step()`, below the position count caption, as a `st.checkbox("Force OST scoring even with >3 positions (slow)")`. This keeps it contextual with the approval decision.

2. **Should PERF-01 also apply to the standalone mutation scanner page (the non-agent `10_mutation_scanner.py` saturation scan)?**
   - What we know: The success criterion says "Running mutagenesis with more than 3 mutation positions" — this refers to the agent pipeline (Phase 2), not the standalone scanner
   - What's unclear: Whether the standalone `run_saturation_mutagenesis()` also needs a cap
   - Recommendation: PERF-01 applies only to `MutationExecutionAgent` (agent pipeline). The standalone scanner is user-controlled and has its own OST checkbox. Do not add a cap there.

3. **Should `_check_scanner_api()` in PERF-03 be a pytest-testable unit, or only integration-testable?**
   - What we know: Phase 4 adds unit tests; Phase 3 must be correct behavior first
   - What's unclear: Whether PERF-03 test is in scope for Phase 3 or Phase 4
   - Recommendation: Phase 3 implements the behavior; Phase 4 (TEST-03/TEST-04) adds the test. Phase 3 verification: manually confirm that importing `mutagenesis_agents` against a patched scanner (missing the parameter) raises `ImportError` with the right message.

---

## Sources

### Primary (HIGH confidence)

- Project source: `/src/protein_design_hub/agents/mutagenesis_agents.py` — `_build_scanner()` try/except (lines 37-44), `MutationExecutionAgent.run()` structure
- Project source: `/src/protein_design_hub/agents/llm_guided.py` — `LLMMutationSuggestionAgent.run()` fallback branch (~lines 1339-1373)
- Project source: `/src/protein_design_hub/web/pages/10_mutation_scanner.py` — `_run_phase2()`, `_render_approval_step()`, `_create_scanner_compat()`
- Project source: `/src/protein_design_hub/analysis/mutation_scanner.py` — `MutationScanner.__init__` signature (line 311: `run_openstructure_comprehensive: bool = False`)
- Python stdlib: `inspect.signature` — https://docs.python.org/3/library/inspect.html#inspect.signature

### Secondary (MEDIUM confidence)

- Project REQUIREMENTS.md — exact success criteria for PERF-01, PERF-02, PERF-03
- Project ROADMAP.md — phase description and dependency (Phase 3 depends on Phase 2)

### Tertiary (LOW confidence)

None. All findings are from direct source code inspection.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries; all patterns are stdlib or already in codebase
- Architecture: HIGH — edit sites identified from source inspection with line numbers
- Pitfalls: HIGH — identified from reading actual current code (position count, warning timing, circular import)

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (30 days; codebase is stable, no fast-moving dependencies)
