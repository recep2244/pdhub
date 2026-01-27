from __future__ import annotations

from protein_design_hub.analysis.sequence_metrics import compute_sequence_metrics


def test_compute_sequence_metrics_basic():
    m = compute_sequence_metrics("ACDEFGHIKLMNPQRSTVWY")
    assert m.length == 20
    assert m.molecular_weight > 0
    assert 0 <= m.aromaticity <= 1
    assert -50 < m.net_charge_ph7 < 50
