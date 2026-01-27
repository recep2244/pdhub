from __future__ import annotations

from protein_design_hub.energy.paths import find_foldx_executable
from protein_design_hub.evaluation.metrics.openmm_gbsa import OpenMMGBSAMetric
from protein_design_hub.evaluation.metrics.rosetta_score_jd2 import RosettaScoreJd2Metric


def test_optional_integrations_are_optional():
    assert isinstance(OpenMMGBSAMetric().is_available(), bool)
    assert isinstance(RosettaScoreJd2Metric().is_available(), bool)
    # FoldX is proprietary; likely absent in CI/dev environments.
    _ = find_foldx_executable()
