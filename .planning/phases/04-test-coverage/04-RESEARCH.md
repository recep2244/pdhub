# Phase 4: Test Coverage - Research

**Researched:** 2026-02-22
**Domain:** Python testing with pytest — unit tests, mock-based integration tests, agent failure mode coverage
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TEST-01 | End-to-end integration test: Phase 1 → Phase 2 with mock LLM and real MutationScanner on short sequence | Architecture Pattern 3 (mock agent subclassing), MutationScanner interface documented in Code Examples |
| TEST-02 | Unit test for `_parse_approved_mutations()` covering column rename, empty input, malformed input | `_parse_approved_mutations` implementation fully read; column names (Position, WT AA, Target AAs, Include) confirmed; importable directly from page module |
| TEST-03 | Unit tests for MutationExecutionAgent failure modes (WT fails, partial failures, all mutations fail) | MutationExecutionAgent.run() branches fully mapped; `AgentResult.fail` path verified; mock via `_build_scanner` monkeypatch or agent subclass |
| TEST-04 | Test for LLMMutationSuggestionAgent JSON parse fallback (missing MUTATION_PLAN_JSON footer) | Fallback path confirmed in llm_guided.py lines 1341-1375; `context.extra["mutation_suggestion_warning"]` key confirmed |
| TEST-05 | LLM pipeline reliability test — agent returning empty/bad output is caught and surfaced | Orchestrator halt-on-fail logic fully documented; `_FailVerdictAgent` pattern already established in test_agent_pipeline_integrity.py |
</phase_requirements>

---

## Summary

Phase 4 adds a targeted suite of tests to `tests/test_mutagenesis_workflow.py` covering five specific behavioural properties of the mutagenesis workflow that Phases 1-3 implemented. The codebase already has 18 passing structural integrity tests in `tests/test_agent_pipeline_integrity.py` that establish the mock-agent subclass pattern. Phase 4 tests are additive — they do not modify any existing files besides adding the new test file.

The primary challenge is TEST-01 (Phase 1 → Phase 2 integration test): running the full 7+4 agent pipeline requires mocking both the LLM layer (`_run_meeting_if_enabled`) and the MutationScanner predictor network calls (`predict_single`, `scan_position`). The existing codebase provides clean seams for both: LLM agents call `_run_meeting_if_enabled` (a single method to patch), and `MutationExecutionAgent` calls `_build_scanner()` (a module-level factory to monkeypatch). The "real MutationScanner" requirement in TEST-01 means the scanner object is instantiated for real, only the HTTP predictor calls are patched.

The four unit tests (TEST-02 through TEST-05) are fully self-contained: they import the pure-Python functions and classes directly, construct minimal inputs, and assert on outputs. No filesystem, no network, no LLM required. All five requirements map cleanly to existing code paths with no ambiguity.

**Primary recommendation:** Write `tests/test_mutagenesis_workflow.py` with five test groups (one per requirement) using `pytest`, `unittest.mock.patch`, and pandas DataFrames for the parser tests. Keep TEST-01 focused on the state propagation (approved_mutations flows from Phase 1 output into Phase 2 context), not on structure quality.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | >=7.0.0 (from pyproject.toml dev deps) | Test runner, fixture system, assertion rewriting | Already in dev deps; all existing tests use it |
| unittest.mock | stdlib | Patch LLM calls, scanner HTTP requests, module-level factories | No new dependency; `patch`, `MagicMock`, `patch.object` cover all cases |
| pandas | >=2.0.0 (from pyproject.toml) | Construct test DataFrames for `_parse_approved_mutations` | Already a project dependency |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tempfile (stdlib) | stdlib | Create temp dirs for MutationScanner output_dir in integration test | TEST-01 needs a real tmp dir for scanner file output |
| pathlib (stdlib) | stdlib | Construct Path objects for temp dirs | Throughout test setup |
| pytest-cov | >=4.0.0 | Coverage reporting | Already in dev deps; run with `--cov` flag |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| unittest.mock.patch | pytest-mock (mocker fixture) | pytest-mock is cleaner syntax but requires an extra dependency; `unittest.mock` is zero-cost and already works |
| Direct DataFrame construction | factory fixtures | Fixtures add indirection; inline DataFrames are clearer for five distinct edge cases |

**Installation:** No new packages required. All dependencies are already in `pyproject.toml`.

---

## Architecture Patterns

### Recommended Project Structure

```
tests/
├── test_agent_pipeline_integrity.py   # existing 18 tests — DO NOT MODIFY
└── test_mutagenesis_workflow.py       # NEW — Phase 4 adds this file only
```

No `conftest.py` exists or is needed for Phase 4. Each test function is self-contained.

### Pattern 1: Mock Agent Subclass (established pattern)

**What:** Define a local `class _MockXxxAgent(BaseAgent)` inside the test file. The agent hardcodes its output into `context.extra` and returns `AgentResult.ok(context, "...")`. This pattern is already used in `test_agent_pipeline_integrity.py` for `_FailVerdictAgent` and `_TailAgent`.

**When to use:** Replacing any LLM-backed or network-backed agent in an orchestrator pipeline for integration tests.

**Example:**
```python
# Pattern from tests/test_agent_pipeline_integrity.py (existing, working)
class _FailVerdictAgent(BaseAgent):
    name = "llm_evaluation_review"
    def run(self, context: WorkflowContext) -> AgentResult:
        context.step_verdicts["evaluation_review"] = {"status": "FAIL", ...}
        return AgentResult.ok(context, "fail verdict emitted")
```

For Phase 4 integration test, the same pattern mocks LLMMutationSuggestionAgent:
```python
class _MockSuggestionAgent(BaseAgent):
    name = "llm_mutation_suggestion"
    def run(self, context: WorkflowContext) -> AgentResult:
        context.extra["mutation_suggestions"] = {
            "positions": [{"residue": 3, "wt_aa": "D", "targets": ["A"], "rationale": "test"}],
            "strategy": "targeted",
            "rationale": "mock",
        }
        context.extra["mutation_suggestion_source"] = "llm"
        context.extra["baseline_low_confidence_positions"] = [3]
        return AgentResult.ok(context, "mock suggestion")
```

### Pattern 2: Module-level Factory Monkeypatching

**What:** `MutationExecutionAgent.run()` calls `_build_scanner()` (module-level function in `mutagenesis_agents.py`) to get a `MutationScanner`. Patch that factory with `patch` to return a mock scanner. The mock scanner's `predict_single` and `calculate_biophysical_metrics` return canned values.

**When to use:** TEST-01 (integration test needs real `MutationExecutionAgent` but mocked HTTP calls), TEST-03 (failure mode tests need `predict_single` to raise).

**Example:**
```python
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

def _make_mock_scanner(tmp_dir, wt_plddt=None, fail_wt=False, fail_mutations=False):
    """Build a MagicMock that looks like MutationScanner to MutationExecutionAgent."""
    mock = MagicMock()
    pdb_stub = "ATOM  ..."
    wt_vals = wt_plddt or [75.0, 80.0, 60.0, 85.0, 70.0]
    wt_path = Path(tmp_dir) / "wildtype.pdb"
    wt_path.write_text(pdb_stub)

    if fail_wt:
        mock.predict_single.side_effect = RuntimeError("WT prediction failed")
    else:
        mock.predict_single.return_value = (pdb_stub, wt_vals, wt_path)
    mock.calculate_biophysical_metrics.return_value = {
        "clash_score": 5.0, "sasa_total": 1200.0, "rmsd": None, "tm_score": None,
        "extra_metrics": {},
    }
    return mock

def test_mutation_execution_wt_failure():
    with tempfile.TemporaryDirectory() as tmp:
        mock_scanner = _make_mock_scanner(tmp, fail_wt=True)
        with patch("protein_design_hub.agents.mutagenesis_agents._build_scanner", return_value=mock_scanner):
            agent = MutationExecutionAgent()
            ctx = WorkflowContext(job_id="test-wt-fail", output_dir=Path(tmp))
            ctx.sequences = [Sequence(id="test", sequence="ACDEFGHIKLM")]
            ctx.extra["approved_mutations"] = [{"residue": 3, "wt_aa": "D", "targets": ["A"]}]
            result = agent.run(ctx)
    assert result.success is False
    assert "wild-type" in result.message.lower()
```

### Pattern 3: Integration Test via Orchestrator.run_with_context

**What:** Build a custom agent list (mix of mock agents and real agents), set up a minimal `WorkflowContext` with a short sequence, run `orchestrator.run_with_context(context)`, assert on `context.extra` keys.

**When to use:** TEST-01 — Phase 1 to Phase 2 integration test.

**Critical detail:** The Phase 1 output that Phase 2 reads is `context.extra["approved_mutations"]`. In an integration test these two phases share the same `WorkflowContext` object. Phase 2's `MutationExecutionAgent` also calls `context.with_job_dir()`, which requires `context.output_dir` to be a writable directory (use `tempfile.TemporaryDirectory()`).

```python
def test_phase1_to_phase2_integration():
    with tempfile.TemporaryDirectory() as tmp:
        # Build Phase 1 pipeline with mock LLM agents
        from protein_design_hub.agents.orchestrator import AgentOrchestrator
        from protein_design_hub.agents.mutagenesis_agents import MutationExecutionAgent, MutationComparisonAgent

        phase1_agents = [
            _MockInputAgent(),          # sets context.sequences
            _MockBaselineReviewAgent(), # sets baseline_low_confidence_positions
            _MockSuggestionAgent(),     # sets approved_mutations + mutation_suggestions
        ]
        phase2_agents = [MutationExecutionAgent(), MutationComparisonAgent()]

        ctx = WorkflowContext(job_id="integ-test", output_dir=Path(tmp))

        # Run Phase 1
        orch1 = AgentOrchestrator(agents=phase1_agents)
        r1 = orch1.run_with_context(ctx)
        assert r1.success

        # Patch scanner for Phase 2
        mock_scanner = _make_mock_scanner(tmp)
        with patch("protein_design_hub.agents.mutagenesis_agents._build_scanner", return_value=mock_scanner):
            orch2 = AgentOrchestrator(agents=phase2_agents)
            r2 = orch2.run_with_context(r1.context)

        assert r2.success
        assert "mutation_results" in r2.context.extra
        assert any(m.get("success") for m in r2.context.extra["mutation_results"])
```

### Pattern 4: Direct Function Import for Unit Tests

**What:** Import `_parse_approved_mutations` directly from the page module. The function has no Streamlit dependencies (it only uses pandas). Build a pandas DataFrame that mimics the `st.data_editor` output.

**Example:**
```python
import pandas as pd
from protein_design_hub.web.pages import _10_mutation_scanner as scanner_page
# OR import via sys.path manipulation if module name starts with digit
```

**Critical detail:** The module filename `10_mutation_scanner.py` starts with a digit, which makes it un-importable via standard `import`. Use `importlib` or `sys.path`:

```python
import importlib.util, sys
from pathlib import Path

def _import_scanner_page():
    page_path = Path(__file__).parent.parent / "src/protein_design_hub/web/pages/10_mutation_scanner.py"
    spec = importlib.util.spec_from_file_location("mutation_scanner_page", page_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_PAGE = _import_scanner_page()
_parse_approved_mutations = _PAGE._parse_approved_mutations
```

**Alternative approach:** Extract `_parse_approved_mutations` into a non-Streamlit utility module (e.g., `web/agent_helpers.py`). This would make it directly importable. However, the task specifies testing the function as it stands, not refactoring it. Use `importlib` in the test.

**Warning:** `exec_module` will try to execute all module-level Streamlit calls. Patch `streamlit` before loading:

```python
import unittest.mock
with unittest.mock.patch.dict("sys.modules", {"streamlit": unittest.mock.MagicMock()}):
    spec.loader.exec_module(mod)
```

### Pattern 5: Empty Output / Bad Output Pipeline Test (TEST-05)

**What:** Extend the existing `_FailVerdictAgent` pattern to an agent that returns `AgentResult.ok()` but writes nothing useful to `context.extra`. Then verify the orchestrator either surfaces a FAIL verdict or the downstream agent fails explicitly.

**Key insight from codebase:** The orchestrator halts when `allow_failed_llm_steps=False` and an LLM verdict is FAIL. An agent returning empty output doesn't automatically trigger FAIL unless the *next* agent checks the missing key and returns `AgentResult.fail()`. Therefore TEST-05 should verify that `MutationExecutionAgent.run()` returns failure when `approved_mutations` is empty AND `baseline_low_confidence_positions` is also empty:

```python
def test_empty_agent_output_surfaces_error():
    class _EmptyOutputAgent(BaseAgent):
        name = "llm_mutation_suggestion"
        def run(self, context: WorkflowContext) -> AgentResult:
            # Writes nothing to context.extra
            return AgentResult.ok(context, "empty output")

    # Follow with MutationExecutionAgent which MUST fail on missing approved_mutations
    with tempfile.TemporaryDirectory() as tmp:
        ctx = WorkflowContext(job_id="empty-test", output_dir=Path(tmp))
        ctx.sequences = [Sequence(id="t", sequence="ACDEFGHIKL")]
        ctx.extra["approved_mutations"] = []
        ctx.extra["baseline_low_confidence_positions"] = []  # also empty

        orchestrator = AgentOrchestrator(
            agents=[_EmptyOutputAgent(), MutationExecutionAgent()],
            stop_on_failure=True,
        )
        result = orchestrator.run_with_context(ctx)
        assert result.success is False
        assert "no approved mutations" in result.message.lower()
```

### Anti-Patterns to Avoid

- **Importing the full Streamlit page without mocking `st`:** The page module has thousands of lines of Streamlit code that runs on import. Always mock `streamlit` in `sys.modules` before `exec_module`.
- **Using real ESMFold API in tests:** The ESMFold API has rate limits and requires network. Always patch `_build_scanner` or `predict_single` for any test involving `MutationExecutionAgent`.
- **Asserting on exact error message strings:** Messages may change during refactoring. Assert on substrings (`.lower()` contains) or on `result.success is False` + key presence in `context.extra`.
- **Sharing WorkflowContext between independent tests:** Each test must construct a fresh `WorkflowContext`. The dataclass is mutable and test isolation breaks if shared.
- **Calling `AgentOrchestrator(mode="mutagenesis_post")` in integration test without patching scanner:** The mode builds real agents including `MutationExecutionAgent`. The scanner HTTP call will fail. Always use `agents=` parameter with explicit lists or patch before calling `run_with_context`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP request mocking | Custom fake HTTP server | `unittest.mock.patch` on `predict_single` | `MutationScanner.predict_single` is a single method; patching at that level is cleaner and faster |
| LLM response generation | Actual Ollama call in tests | Mock agent subclass pattern | LLM responses are non-deterministic; tests must be deterministic |
| Temporary directory management | `os.mkdir` + manual cleanup | `tempfile.TemporaryDirectory()` as context manager | Automatic cleanup, thread-safe, cross-platform |
| DataFrame construction helpers | CSV file fixtures | Inline `pd.DataFrame({"Position": [3], "WT AA": ["D"], ...})` | Self-documenting, no file I/O, editable in one place |

**Key insight:** All the mocking infrastructure needed is in the Python stdlib. Adding pytest-mock, httpx, or respx would be unnecessary complexity for this scope.

---

## Common Pitfalls

### Pitfall 1: Digit-prefixed module import failure
**What goes wrong:** `from protein_design_hub.web.pages.10_mutation_scanner import _parse_approved_mutations` raises `SyntaxError` — Python identifiers cannot start with a digit.
**Why it happens:** Streamlit convention is to prefix pages with numbers for ordering. Python's import system cannot handle this directly.
**How to avoid:** Use `importlib.util.spec_from_file_location` with the full filesystem path. Mock `streamlit` in `sys.modules` before `exec_module` to prevent Streamlit's module-level code from running.
**Warning signs:** `SyntaxError` or `ModuleNotFoundError` when trying standard import.

### Pitfall 2: MutationExecutionAgent calls `context.with_job_dir()` which creates real directories
**What goes wrong:** `MutationExecutionAgent.run()` calls `context.with_job_dir()` at the start, which creates `{output_dir}/{job_id}/mutagenesis/structures/` on disk. If `output_dir` is the project root, test artefacts pollute the working tree.
**Why it happens:** `WorkflowContext.with_job_dir()` calls `mkdir(parents=True, exist_ok=True)`.
**How to avoid:** Always pass `output_dir=Path(tmp)` using a `tempfile.TemporaryDirectory()` context manager. The temp dir is cleaned up automatically.
**Warning signs:** Mysterious new directories appearing in the project root after running tests.

### Pitfall 3: Patching `_build_scanner` at wrong import path
**What goes wrong:** `patch("mutagenesis_agents._build_scanner", ...)` patches the definition site, but `MutationExecutionAgent.run()` has already imported the function into its own scope, so the patch has no effect.
**Why it happens:** Python's `patch` must target where the name is *used*, not where it is *defined*.
**How to avoid:** Patch at `"protein_design_hub.agents.mutagenesis_agents._build_scanner"` — the module where `MutationExecutionAgent.run()` calls it.
**Warning signs:** Test does not intercept the HTTP call; ESMFold API request goes out.

### Pitfall 4: MutationScanner mock missing `scan_position` return structure
**What goes wrong:** `scan_position` returns a `SaturationMutagenesisResult` dataclass. A plain `MagicMock()` returns a `MagicMock` for `.mutations`, and iterating over it does not produce `MutationResult` objects with `.to_dict()`.
**Why it happens:** `MutationExecutionAgent` calls `sat_result.mutations` and iterates, calling `m.to_dict()` on each.
**How to avoid:** Either mock `scan_position` to raise (testing the partial failure path) or build a real `SaturationMutagenesisResult` with a real `MutationResult` inside. For the failure mode test, raising is simpler and directly exercises the error handling branch.

### Pitfall 5: TEST-04 runs LLMMutationSuggestionAgent.run() which calls real LLM
**What goes wrong:** `LLMMutationSuggestionAgent.run()` calls `self._run_meeting_if_enabled(...)` which calls `run_meeting()` which tries to connect to Ollama.
**Why it happens:** The test exercises the agent's JSON parsing fallback path, but the agent calls LLM first.
**How to avoid:** Patch `LLMMutationSuggestionAgent._run_meeting_if_enabled` to return the canned string that is missing the `MUTATION_PLAN_JSON:` footer. Use `patch.object`:
```python
with patch.object(agent, "_run_meeting_if_enabled", return_value="The LLM said nothing useful."):
    result = agent.run(ctx)
```
**Warning signs:** Test hangs waiting for Ollama connection timeout.

### Pitfall 6: `Sequence` type import confusion
**What goes wrong:** `WorkflowContext.sequences` expects `List[Sequence]` where `Sequence` is `protein_design_hub.core.types.Sequence`, not `typing.Sequence`.
**Why it happens:** Both are named `Sequence`; the wrong import is common.
**How to avoid:** `from protein_design_hub.core.types import Sequence` explicitly, and verify the `id` and `sequence` attributes exist.

---

## Code Examples

Verified from direct source reading:

### `_parse_approved_mutations` — Expected DataFrame Column Names
```python
# Source: src/protein_design_hub/web/pages/10_mutation_scanner.py lines 1868-1886
# The function reads these exact column names from the DataFrame:
#   "Position" (int), "WT AA" (str), "Target AAs" (str)
# The caller filters on "Include" column before passing df to this function.
# The function is called with: _parse_approved_mutations(included, ctx.sequences[0].sequence)
# where `included = edited_df[edited_df["Include"] == True]`

import pandas as pd

# Correct input DataFrame
df_correct = pd.DataFrame({
    "Position": [3, 7],
    "WT AA": ["D", "G"],
    "Target AAs": ["A, G", "All 19 AAs"],
})
# Expected output:
# [{"residue": 3, "wt_aa": "D", "targets": ["A", "G"]},
#  {"residue": 7, "wt_aa": "G", "targets": ["*"]}]

# Renamed column test — simulate user editing column header
df_renamed = pd.DataFrame({
    "position": [3],  # lowercase — triggers KeyError
    "WT AA": ["D"],
    "Target AAs": ["A"],
})
# Expected: KeyError or empty list (function uses row["Position"] directly)

# Empty DataFrame
df_empty = pd.DataFrame(columns=["Position", "WT AA", "Target AAs"])
# Expected: []

# Malformed Target AAs — all invalid AAs
df_malformed = pd.DataFrame({
    "Position": [3],
    "WT AA": ["D"],
    "Target AAs": ["X1, 99, @@"],  # no valid AAs
})
# Expected: [] (no valid targets → entry is skipped)
```

### MutationExecutionAgent — Failure Mode Decision Tree
```python
# Source: src/protein_design_hub/agents/mutagenesis_agents.py lines 97-303

# 1. No approved_mutations AND no baseline_low_confidence_positions
#    → AgentResult.fail("No approved mutations and no low-confidence positions available.")

# 2. No sequences in context
#    → AgentResult.fail("No sequences in context for mutation execution.")

# 3. WT prediction fails (scanner.predict_single raises)
#    → AgentResult.fail("Wild-type structure prediction failed: ...")

# 4. Individual mutation fails (raises inside try/except)
#    → Records {"success": False, "error_message": str(e)} in all_results; continues

# 5. ALL mutations fail (no result has success=True)
#    → AgentResult.fail("All mutations failed during execution.")

# 6. Partial failures OK — returns AgentResult.ok with count
#    → AgentResult.ok(context, "Executed N/M mutations successfully.")
```

### LLMMutationSuggestionAgent — Fallback Path
```python
# Source: src/protein_design_hub/agents/llm_guided.py lines 1331-1375
# When _parse_mutation_plan_from_summary(summary, sequence) returns None:
#   - Writes context.extra["mutation_suggestion_warning"] = "LLM plan unparseable — ..."
#   - Falls back to saturation at context.extra["baseline_low_confidence_positions"][:5]
#   - If no low_conf positions: returns AgentResult.ok with warning about no suggestions

# The warning message key to assert on:
assert "mutation_suggestion_warning" in context.extra
assert "MUTATION_PLAN_JSON" in context.extra["mutation_suggestion_warning"]
# OR more robustly:
assert "unparseable" in context.extra["mutation_suggestion_warning"].lower()
```

### Orchestrator Empty-Output Reliability
```python
# Source: src/protein_design_hub/agents/orchestrator.py lines 263-306
# The orchestrator halts on:
#   1. agent.run() returns AgentResult with success=False (when stop_on_failure=True)
#   2. LLM verdict key is FAIL (when allow_failed_llm_steps=False)
# It does NOT automatically halt when an agent writes nothing to context.extra.
# Therefore TEST-05 must test via the downstream agent failing, not via orchestrator policy.
```

### Correct `Sequence` Type Construction
```python
# Source: src/protein_design_hub/agents/context.py + core/types.py
from protein_design_hub.core.types import Sequence
from protein_design_hub.agents.context import WorkflowContext
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    ctx = WorkflowContext(
        job_id="test-job",
        output_dir=Path(tmp),
    )
    ctx.sequences = [Sequence(id="test_protein", sequence="ACDEFGHIKL")]
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 46 tests in single file (STATE.md reference) | 18 tests (actual count after Phase 1-3 refactoring) | Phases 1-3 | New tests must target 18+N as baseline; don't assume 46 |
| No mutagenesis tests | `test_agent_pipeline_integrity.py` adds pipeline shape + parse tests | Phase 1-3 | Solid mock-agent pattern already established to follow |

**Deprecated/outdated:**
- STATE.md references "46 tests passing" — this was incorrect or from a different test file. The current `tests/` directory has only `test_agent_pipeline_integrity.py` with 18 tests and `test_web_smoke.py` (untracked). The planner should target 18 + new tests as the baseline.

---

## Open Questions

1. **Can `_parse_approved_mutations` be imported without executing Streamlit module-level code?**
   - What we know: The function itself has no Streamlit calls; it only uses `pandas`. But the module imports `streamlit` at the top level and calls `st.*` in the global scope during `exec_module`.
   - What's unclear: Whether patching `sys.modules["streamlit"]` with `MagicMock()` before `exec_module` is sufficient to prevent errors from all module-level Streamlit calls in a 3000-line file.
   - Recommendation: Write a small spike test first. If exec_module is too brittle, refactor `_parse_approved_mutations` into `web/agent_helpers.py` (which has no Streamlit calls) as a preparatory micro-task before the main test task.

2. **Does `test_web_smoke.py` (untracked) conflict with any Phase 4 test IDs?**
   - What we know: `test_web_smoke.py` is untracked in git (`?? tests/test_web_smoke.py` in git status). Its content is unknown.
   - What's unclear: Whether it imports from the same page module, causing fixture conflicts.
   - Recommendation: Planner should add a task to read `test_web_smoke.py` before writing Phase 4 tests and check for name conflicts.

---

## Sources

### Primary (HIGH confidence)
- Direct source reading: `src/protein_design_hub/agents/mutagenesis_agents.py` (full file) — MutationExecutionAgent branches, MutagenesisPipelineReportAgent structure
- Direct source reading: `src/protein_design_hub/agents/llm_guided.py` (lines 52-230, 1188-1375) — `_LLMGuidedMixin._run_meeting_if_enabled`, `_parse_mutation_plan_from_summary`, `LLMMutationSuggestionAgent.run` fallback path
- Direct source reading: `tests/test_agent_pipeline_integrity.py` (full file) — established mock-agent patterns, 18 test baseline confirmed
- Direct source reading: `src/protein_design_hub/agents/orchestrator.py` (full file) — pipeline halt logic, `run_with_context` flow
- Direct source reading: `src/protein_design_hub/agents/context.py` — `WorkflowContext` dataclass fields
- Direct source reading: `src/protein_design_hub/agents/base.py` — `AgentResult.fail`, `BaseAgent` interface
- Direct source reading: `src/protein_design_hub/analysis/mutation_scanner.py` (lines 1-400) — `MutationScanner.__init__` signature, `predict_single`, `scan_position`, `calculate_biophysical_metrics`
- Direct source reading: `pyproject.toml` — confirmed pytest>=7.0.0 and pytest-cov>=4.0.0 in dev deps
- Live command: `python -m pytest tests/test_agent_pipeline_integrity.py -q` — confirmed 18 tests pass

### Secondary (MEDIUM confidence)
- Grep on `10_mutation_scanner.py` for `_parse_approved_mutations`, `_VALID_AAS_SET`, column names — exact column names `"Position"`, `"WT AA"`, `"Target AAs"` confirmed from source
- Grep on `llm_guided.py` for `MUTATION_PLAN_JSON` and `mutation_suggestion_warning` — confirmed key names and exact warning message format

### Tertiary (LOW confidence)
- None. All claims are grounded in direct source reading.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already installed; no new deps
- Architecture patterns: HIGH — directly derived from existing test file patterns and source code
- Pitfalls: HIGH — all pitfalls confirmed by reading actual code that would trigger them
- Code examples: HIGH — all column names, key names, and branching logic confirmed from source

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (30 days; code base is stable, no fast-moving dependencies)
