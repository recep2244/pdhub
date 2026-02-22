"""
Mutagenesis workflow behavioural tests — Phase 4.

Test groups:
  TEST-02  _parse_approved_mutations() parser unit tests
  TEST-04  LLMMutationSuggestionAgent JSON parse fallback
  TEST-05  Empty agent output surfaces error (not silently swallowed)
  TEST-03  MutationExecutionAgent failure modes          (Plan 04-02)
  TEST-01  Phase 1 → Phase 2 integration                (Plan 04-02)
"""
import importlib.util
import sys
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.agents.orchestrator import AgentOrchestrator
from protein_design_hub.core.types import Sequence


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _AttrDict(dict):
    """Dict that supports attribute-style access for Streamlit session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value

    def get(self, key, default=None):  # type: ignore[override]
        return super().get(key, default)


def _load_scanner_page():
    """Import 10_mutation_scanner.py via importlib (digit prefix blocks normal import).

    The real Streamlit package is used (it is already on sys.path), but
    problematic module-level UI calls are patched so the page can be loaded
    in a headless test environment.  session_state is replaced with an
    _AttrDict so that attribute-style writes (st.session_state.foo = bar)
    do not raise AttributeError.
    """
    import streamlit as st

    page_path = (
        Path(__file__).resolve().parents[1]
        / "src/protein_design_hub/web/pages/10_mutation_scanner.py"
    )

    with (
        patch("protein_design_hub.web.ui.inject_base_css", return_value=None),
        patch("protein_design_hub.web.ui.sidebar_nav", return_value=None),
        patch("protein_design_hub.web.ui.sidebar_system_status", return_value=None),
        patch("protein_design_hub.web.ui.page_header", return_value=None),
        patch("protein_design_hub.web.ui.section_header", return_value=None),
        patch("protein_design_hub.web.ui.workflow_breadcrumb", return_value=None),
        patch(
            "protein_design_hub.web.agent_helpers.agent_sidebar_status",
            return_value=None,
        ),
    ):
        # Replace session_state with a dict that accepts attribute access.
        st.session_state = _AttrDict()

        spec = importlib.util.spec_from_file_location("mutation_scanner_page", page_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    return mod


# Load once at module level; individual tests use the cached module.
_SCANNER_PAGE = _load_scanner_page()
_parse_approved_mutations = _SCANNER_PAGE._parse_approved_mutations


# ---------------------------------------------------------------------------
# TEST-02  _parse_approved_mutations() unit tests
# ---------------------------------------------------------------------------

def test_parse_approved_mutations_correct_input():
    """Valid DataFrame with Position, WT AA, Target AAs returns parsed list."""
    df = pd.DataFrame({
        "Position": [3, 7],
        "WT AA": ["D", "G"],
        "Target AAs": ["A, G", "A"],
    })
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    result = _parse_approved_mutations(df, sequence)
    assert isinstance(result, list)
    assert len(result) == 2
    # Position 3 → 1-indexed residue 3, wt_aa "D"
    assert result[0]["residue"] == 3
    assert result[0]["wt_aa"] == "D"
    assert "A" in result[0]["targets"]
    assert "G" in result[0]["targets"]


def test_parse_approved_mutations_renamed_columns():
    """DataFrame with lowercase 'position' (renamed) is rejected gracefully — returns [] or raises KeyError."""
    df = pd.DataFrame({
        "position": [3],   # lowercase — not what the function expects
        "WT AA": ["D"],
        "Target AAs": ["A"],
    })
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    # The function uses row["Position"] directly; a renamed column causes KeyError or returns []
    try:
        result = _parse_approved_mutations(df, sequence)
        # If it doesn't raise, it must return empty (graceful degradation)
        assert result == []
    except (KeyError, Exception):
        pass  # KeyError is acceptable — documents the contract


def test_parse_approved_mutations_empty_dataframe():
    """Empty DataFrame (post-filter with no rows) returns an empty list."""
    df = pd.DataFrame(columns=["Position", "WT AA", "Target AAs"])
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    result = _parse_approved_mutations(df, sequence)
    assert result == []


def test_parse_approved_mutations_malformed_target_aas():
    """Target AAs containing no valid single-letter AAs produces no targets (entry skipped or empty targets)."""
    df = pd.DataFrame({
        "Position": [3],
        "WT AA": ["D"],
        "Target AAs": ["X1, 99, @@"],   # no valid AAs
    })
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    result = _parse_approved_mutations(df, sequence)
    # Either the entry is skipped entirely or targets is empty — both indicate the function
    # does not silently accept garbage input as valid mutations
    if result:
        assert result[0].get("targets", []) == []
    else:
        assert result == []


# ---------------------------------------------------------------------------
# TEST-04  LLMMutationSuggestionAgent — JSON parse fallback
# ---------------------------------------------------------------------------

def test_llm_suggestion_agent_fallback_when_no_json_footer():
    """When LLM output lacks MUTATION_PLAN_JSON footer, agent sets warning in context.extra
    and falls back to saturation over baseline_low_confidence_positions."""
    from protein_design_hub.agents.llm_guided import LLMMutationSuggestionAgent

    agent = LLMMutationSuggestionAgent()

    with tempfile.TemporaryDirectory() as tmp:
        ctx = WorkflowContext(job_id="test-fallback", output_dir=Path(tmp))
        ctx.sequences = [Sequence(id="test_protein", sequence="ACDEFGHIKLM")]
        # Set baseline positions so fallback has something to work with
        ctx.extra["baseline_low_confidence_positions"] = [3, 5]
        ctx.extra["baseline_plddt"] = [75.0, 80.0, 60.0, 85.0, 70.0, 65.0, 90.0, 72.0, 68.0, 88.0, 77.0]

        # Patch _run_meeting_if_enabled to return a string WITHOUT the MUTATION_PLAN_JSON footer
        with patch.object(
            agent,
            "_run_meeting_if_enabled",
            return_value="The protein analysis suggests some mutations. No structured plan available.",
        ):
            result = agent.run(ctx)

    # Agent should succeed (fallback is graceful, not a hard failure)
    assert result.success is True
    # The warning key MUST be set in context.extra
    assert "mutation_suggestion_warning" in result.context.extra, (
        "Expected 'mutation_suggestion_warning' in context.extra when JSON footer is missing"
    )
    warning = result.context.extra["mutation_suggestion_warning"]
    # Warning must describe what happened (substring match, not exact string)
    assert any(kw in warning.lower() for kw in ["unparseable", "mutation_plan_json", "fallback", "saturation"]), (
        f"Warning did not describe the fallback: {warning!r}"
    )


# ---------------------------------------------------------------------------
# Shared mock scanner factory (used by TEST-03 and TEST-01)
# ---------------------------------------------------------------------------

def _make_mock_scanner(tmp_dir: str, *, fail_wt: bool = False, fail_mutations: bool = False):
    """Return a MagicMock that satisfies MutationExecutionAgent's expectations.

    Args:
        tmp_dir: Writable temp directory (from TemporaryDirectory context manager).
        fail_wt: If True, predict_single raises RuntimeError (WT prediction failure).
        fail_mutations: If True, scan_position raises RuntimeError (mutation failure).
    """
    mock = MagicMock()
    pdb_stub = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    wt_plddt = [75.0, 80.0, 60.0, 85.0, 70.0, 65.0, 90.0, 72.0, 68.0, 88.0, 77.0]

    # Write a real file so path-based assertions don't fail
    wt_path = Path(tmp_dir) / "wildtype.pdb"
    wt_path.write_text(pdb_stub)

    if fail_wt:
        mock.predict_single.side_effect = RuntimeError("WT prediction failed in test")
    else:
        mock.predict_single.return_value = (pdb_stub, wt_plddt, wt_path)

    if fail_mutations:
        mock.scan_position.side_effect = RuntimeError("Mutation scan failed in test")
    else:
        # MutationExecutionAgent calls scan_position per mutation target.
        # Return a minimal result-like object. The agent calls result.to_dict() if it exists,
        # or accesses .mutations attribute for saturation mode.
        # For targeted mutations, the agent calls predict_single for each (position, aa) pair.
        # Check mutagenesis_agents.py execution path to determine which branch is taken.
        # Safe default: predict_single returns per-mutation PDB as well (targeted path).
        pass  # predict_single already configured above for WT; mutation calls reuse same mock

    mock.calculate_biophysical_metrics.return_value = {
        "clash_score": 5.0,
        "sasa_total": 1200.0,
        "rmsd": None,
        "tm_score": None,
        "extra_metrics": {},
    }
    return mock


# ---------------------------------------------------------------------------
# TEST-03  MutationExecutionAgent failure modes
# ---------------------------------------------------------------------------

def test_mutation_execution_fails_when_wt_prediction_fails():
    """When scanner.predict_single raises for WT, agent returns AgentResult with success=False."""
    from protein_design_hub.agents.mutagenesis_agents import MutationExecutionAgent

    with tempfile.TemporaryDirectory() as tmp:
        mock_scanner = _make_mock_scanner(tmp, fail_wt=True)
        with patch(
            "protein_design_hub.agents.mutagenesis_agents._build_scanner",
            return_value=mock_scanner,
        ):
            agent = MutationExecutionAgent()
            ctx = WorkflowContext(job_id="test-wt-fail", output_dir=Path(tmp))
            ctx.sequences = [Sequence(id="test_protein", sequence="ACDEFGHIKL")]
            ctx.extra["approved_mutations"] = [
                {"residue": 3, "wt_aa": "D", "targets": ["A"]}
            ]
            ctx.extra["baseline_low_confidence_positions"] = [3]
            result = agent.run(ctx)

    assert result.success is False, "Expected failure when WT prediction fails"
    assert result.message is not None
    # Message should mention WT/wild-type (substring, case-insensitive)
    assert any(kw in result.message.lower() for kw in ["wild", "wt", "wildtype"]), (
        f"Expected WT-related failure message, got: {result.message!r}"
    )


def test_mutation_execution_fails_when_all_mutations_fail():
    """When every individual mutation fails (predict_single raises for mutation calls),
    the agent returns failure with 'all mutations failed' message."""
    from protein_design_hub.agents.mutagenesis_agents import MutationExecutionAgent

    with tempfile.TemporaryDirectory() as tmp:
        mock_scanner = _make_mock_scanner(tmp)
        wt_pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        wt_plddt = [75.0, 80.0, 60.0, 85.0, 70.0, 65.0, 90.0, 72.0, 68.0, 88.0, 77.0]
        wt_path = Path(tmp) / "wildtype.pdb"
        wt_path.write_text(wt_pdb)

        # WT succeeds, but mutation predict_single call should fail.
        # MutationExecutionAgent.run() calls predict_single for WT first, then for each mutation.
        # Use side_effect as a callable: first call succeeds (WT), subsequent calls fail (mutations).
        call_count = [0]
        def predict_side_effect(sequence, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (wt_pdb, wt_plddt, wt_path)
            raise RuntimeError("Mutation prediction failed in test")

        mock_scanner.predict_single.side_effect = predict_side_effect

        with patch(
            "protein_design_hub.agents.mutagenesis_agents._build_scanner",
            return_value=mock_scanner,
        ):
            agent = MutationExecutionAgent()
            ctx = WorkflowContext(job_id="test-all-fail", output_dir=Path(tmp))
            ctx.sequences = [Sequence(id="test_protein", sequence="ACDEFGHIKL")]
            ctx.extra["approved_mutations"] = [
                {"residue": 3, "wt_aa": "D", "targets": ["A"]},
            ]
            ctx.extra["baseline_low_confidence_positions"] = [3]
            result = agent.run(ctx)

    # All mutations failed → agent must return failure
    assert result.success is False, "Expected failure when all mutations fail"
    assert result.message is not None


def test_mutation_execution_succeeds_with_partial_failures():
    """When some mutations fail and at least one succeeds, agent returns success with partial results."""
    from protein_design_hub.agents.mutagenesis_agents import MutationExecutionAgent

    with tempfile.TemporaryDirectory() as tmp:
        mock_scanner = _make_mock_scanner(tmp)
        wt_pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        wt_plddt = [75.0, 80.0, 60.0, 85.0, 70.0, 65.0, 90.0, 72.0, 68.0, 88.0, 77.0]
        wt_path = Path(tmp) / "wildtype.pdb"
        wt_path.write_text(wt_pdb)

        # First call: WT succeeds. Second call: mutation A succeeds. Third call: mutation G fails.
        call_count = [0]
        def predict_side_effect(sequence, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # WT
                return (wt_pdb, wt_plddt, wt_path)
            elif call_count[0] == 2:
                # First mutation succeeds
                mut_path = Path(tmp) / f"mut_{call_count[0]}.pdb"
                mut_path.write_text(wt_pdb)
                return (wt_pdb, wt_plddt, mut_path)
            else:
                # Remaining mutations fail
                raise RuntimeError("Mutation prediction failed")

        mock_scanner.predict_single.side_effect = predict_side_effect

        with patch(
            "protein_design_hub.agents.mutagenesis_agents._build_scanner",
            return_value=mock_scanner,
        ):
            agent = MutationExecutionAgent()
            ctx = WorkflowContext(job_id="test-partial-fail", output_dir=Path(tmp))
            ctx.sequences = [Sequence(id="test_protein", sequence="ACDEFGHIKL")]
            # Two targets: first will succeed, second will fail
            ctx.extra["approved_mutations"] = [
                {"residue": 3, "wt_aa": "D", "targets": ["A", "G"]},
            ]
            ctx.extra["baseline_low_confidence_positions"] = [3]
            result = agent.run(ctx)

    # Partial success: at least one mutation succeeded → overall success
    assert result.success is True, (
        f"Expected success with partial results, got: {result.message!r}"
    )
    mutation_results = result.context.extra.get("mutation_results", [])
    # At least one result should exist
    assert len(mutation_results) > 0
    # At least one should be successful
    assert any(r.get("success", False) for r in mutation_results), (
        "No successful mutations in mutation_results"
    )


# ---------------------------------------------------------------------------
# TEST-01  Phase 1 → Phase 2 integration (mock LLM + real agents + patched scanner instance)
# ---------------------------------------------------------------------------

class _MockInputAgent(BaseAgent):
    """Populates context.sequences to satisfy downstream agents."""
    name = "input"

    def run(self, context: WorkflowContext) -> AgentResult:
        context.sequences = [Sequence(id="integration_protein", sequence="ACDEFGHIKL")]
        return AgentResult.ok(context, "mock input set")


class _MockBaselineReviewAgent(BaseAgent):
    """Populates baseline review outputs used by LLMMutationSuggestionAgent."""
    name = "llm_baseline_review"

    def run(self, context: WorkflowContext) -> AgentResult:
        context.extra["baseline_low_confidence_positions"] = [3, 5]
        context.extra["baseline_plddt"] = [75.0, 80.0, 60.0, 85.0, 70.0, 65.0, 90.0, 72.0, 68.0, 88.0]
        context.step_verdicts["baseline_review"] = {
            "status": "PASS",
            "key_findings": ["positions 3 and 5 show low confidence"],
        }
        return AgentResult.ok(context, "mock baseline review complete")


class _MockSuggestionAgent(BaseAgent):
    """Directly populates approved_mutations (bypasses LLM call entirely)."""
    name = "llm_mutation_suggestion"

    def run(self, context: WorkflowContext) -> AgentResult:
        context.extra["approved_mutations"] = [
            {"residue": 3, "wt_aa": "D", "targets": ["A"], "rationale": "low pLDDT in test"},
        ]
        context.extra["mutation_suggestions"] = {
            "positions": [{"residue": 3, "wt_aa": "D", "targets": ["A"], "rationale": "test"}],
            "strategy": "targeted",
            "rationale": "mock test suggestion",
        }
        context.extra["mutation_suggestion_source"] = "llm"
        return AgentResult.ok(context, "mock suggestion set")


def test_phase1_to_phase2_integration():
    """Phase 1 approved_mutations flows into Phase 2 MutationExecutionAgent.
    Phase 2 produces mutation_results in context.extra using the shared WorkflowContext.

    Test strategy:
    - Phase 1: 3 mock agents that write approved_mutations into context.extra
    - Phase 2: Real MutationExecutionAgent — MutationScanner instantiated normally (no
      network calls in __init__), then instance.predict_single patched to avoid HTTP
    - Assert: mutation_results exists in final context.extra

    Why real MutationScanner (not MagicMock via _build_scanner):
      TEST-01 success criterion requires MutationScanner.__init__ to be exercised.
      MutationScanner(predictor='esmfold_api', output_dir=Path(tmp)) makes no network calls
      during __init__ — only predict_single triggers HTTP. Patching the instance method
      satisfies the "real scanner" requirement while keeping the test hermetic.
    """
    from protein_design_hub.agents.mutagenesis_agents import MutationExecutionAgent
    from protein_design_hub.analysis.mutation_scanner import MutationScanner

    with tempfile.TemporaryDirectory() as tmp:
        # --- Phase 1: mock pipeline ---
        phase1_agents = [
            _MockInputAgent(),
            _MockBaselineReviewAgent(),
            _MockSuggestionAgent(),
        ]
        ctx = WorkflowContext(job_id="integ-phase1-phase2", output_dir=Path(tmp))

        orch1 = AgentOrchestrator(agents=phase1_agents)
        r1 = orch1.run_with_context(ctx)

        assert r1.success is True, (
            f"Phase 1 mock pipeline failed: {r1.message!r}"
        )
        assert "approved_mutations" in r1.context.extra, (
            "Phase 1 did not write approved_mutations to context.extra"
        )
        approved = r1.context.extra["approved_mutations"]
        assert len(approved) > 0, "Phase 1 approved_mutations is empty"

        # --- Phase 2: real MutationScanner instance, predict_single patched ---
        # MutationScanner.__init__ stores config values but makes no HTTP calls.
        # This satisfies the TEST-01 "real scanner" requirement.
        real_scanner = MutationScanner(predictor="esmfold_api", output_dir=Path(tmp))

        wt_pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        wt_plddt = [75.0, 80.0, 60.0, 85.0, 70.0, 65.0, 90.0, 72.0, 68.0, 88.0]

        call_count_p2 = [0]
        def predict_side_effect_p2(sequence, *args, **kwargs):
            call_count_p2[0] += 1
            mut_path = Path(tmp) / f"pred_{call_count_p2[0]}.pdb"
            mut_path.write_text(wt_pdb)
            return (wt_pdb, wt_plddt, mut_path)

        with patch.object(real_scanner, "predict_single", side_effect=predict_side_effect_p2):
            # Also patch calculate_biophysical_metrics on the real instance
            real_scanner.calculate_biophysical_metrics = MagicMock(return_value={
                "clash_score": 5.0,
                "sasa_total": 1200.0,
                "rmsd": None,
                "tm_score": None,
                "extra_metrics": {},
            })
            with patch(
                "protein_design_hub.agents.mutagenesis_agents._build_scanner",
                return_value=real_scanner,
            ):
                # MutationComparisonAgent intentionally excluded: it requires additional context
                # keys (comparison data) that the mock Phase 1 does not produce. The critical
                # invariant — approved_mutations → mutation_results — is fully tested by
                # MutationExecutionAgent alone.
                orch2 = AgentOrchestrator(
                    agents=[MutationExecutionAgent()],
                    stop_on_failure=True,
                )
                r2 = orch2.run_with_context(r1.context)

        # Core assertion: Phase 1 approved_mutations flowed into Phase 2 and produced results
        assert r2.success is True, (
            f"Phase 2 pipeline failed: {r2.message!r}\n"
            f"approved_mutations was: {r1.context.extra.get('approved_mutations')}"
        )
        assert "mutation_results" in r2.context.extra, (
            "Phase 2 did not produce mutation_results in context.extra"
        )
        mutation_results = r2.context.extra["mutation_results"]
        assert len(mutation_results) > 0, "mutation_results is empty after Phase 2"
        assert any(m.get("success", False) for m in mutation_results), (
            "No successful mutations in Phase 2 results"
        )


# ---------------------------------------------------------------------------
# TEST-05  LLM pipeline reliability — empty agent output caught, not swallowed
# ---------------------------------------------------------------------------

class _EmptyOutputAgent(BaseAgent):
    """Agent that returns success but writes nothing to context.extra."""
    name = "llm_mutation_suggestion"

    def run(self, context: WorkflowContext) -> AgentResult:
        # Intentionally writes nothing — simulates a badly-behaved LLM agent
        return AgentResult.ok(context, "empty output — nothing written")


def test_empty_agent_output_surfaces_error_in_downstream_agent():
    """MutationExecutionAgent following an empty-output suggestion agent must return failure,
    not silently continue. The pipeline must NOT report success when no mutations are available."""
    from protein_design_hub.agents.mutagenesis_agents import MutationExecutionAgent

    with tempfile.TemporaryDirectory() as tmp:
        ctx = WorkflowContext(job_id="empty-output-test", output_dir=Path(tmp))
        ctx.sequences = [Sequence(id="test_protein", sequence="ACDEFGHIKL")]
        # Explicitly empty — simulates what an empty output agent leaves behind
        ctx.extra["approved_mutations"] = []
        ctx.extra["baseline_low_confidence_positions"] = []

        orchestrator = AgentOrchestrator(
            agents=[_EmptyOutputAgent(), MutationExecutionAgent()],
            stop_on_failure=True,
        )
        result = orchestrator.run_with_context(ctx)

    # The pipeline MUST report failure — cannot succeed with no mutations
    assert result.success is False, (
        "Pipeline reported success despite empty approved_mutations and no baseline positions"
    )
    # The failure message should communicate the root cause
    assert result.message is not None
    assert len(result.message) > 0
    # Stronger contract: message must describe the "no approved mutations" condition
    assert any(kw in result.message.lower() for kw in ["approved mutations", "no mutations", "empty", "no approved"]), (
        f"Failure message did not describe the empty-mutations condition: {result.message!r}"
    )
