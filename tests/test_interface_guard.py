"""Tests for the interface-capability guard and its ipSAE wiring."""

import numpy as np
import pytest

from protein_design_hub.evaluation.interface_guard import (
    InterfaceCapabilityError,
    assert_interface_capable,
    is_interface_capable,
)
from protein_design_hub.evaluation.ipsae import compute_ipsae


# ── is_interface_capable / assert_interface_capable ────────────────────────
@pytest.mark.parametrize("name", ["chai1", "chai", "boltz2", "boltz", "alphafold",
                                  "AlphaFold", "AF2", "alphafold-multimer"])
def test_capable_engines_allowed(name):
    assert is_interface_capable(name) is True
    assert_interface_capable(name)  # must not raise


@pytest.mark.parametrize("name", ["esmfold", "ESMFold", "esm-fold", "ESM_Fold", "esm2"])
def test_esmfold_blocked(name):
    assert is_interface_capable(name) is False
    with pytest.raises(InterfaceCapabilityError):
        assert_interface_capable(name)


def test_unknown_engine_blocked():
    assert is_interface_capable("mystery-folder") is False
    with pytest.raises(InterfaceCapabilityError):
        assert_interface_capable("mystery-folder")


# ── compute_ipsae wiring ────────────────────────────────────────────────────
def _toy_pae(n: int = 8) -> np.ndarray:
    """Low-PAE block matrix so a valid engine yields a real SUCCESS result."""
    return np.full((n, n), 3.0, dtype=float)


def test_ipsae_rejects_esmfold_origin():
    res = compute_ipsae(_toy_pae(), [4, 4], predictor_name="esmfold")
    assert res["status"] == "unavailable"
    assert "PAE" in res["error"]
    assert res["predictor"] == "esmfold"


def test_ipsae_allows_chai_and_boltz():
    for engine in ("chai1", "boltz2"):
        res = compute_ipsae(_toy_pae(), [4, 4], predictor_name=engine)
        assert res["status"] == "SUCCESS", engine
        assert "ipsae_min" in res


def test_ipsae_no_predictor_arg_keeps_success_path():
    """Existing callers pass no predictor_name → behaviour identical to before."""
    res_plain = compute_ipsae(_toy_pae(), [4, 4])
    res_named = compute_ipsae(_toy_pae(), [4, 4], predictor_name="boltz2")
    assert res_plain["status"] == "SUCCESS"
    assert res_plain["ipsae_min"] == res_named["ipsae_min"]
