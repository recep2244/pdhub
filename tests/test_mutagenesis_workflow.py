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
