"""Tests for the cached model/predictor factory.

These tests run with heavy deps ABSENT: no torch, no GPU, no streamlit needed.
We force the fallback dict-cache path (by simulating streamlit being missing)
so behavior is deterministic in CI, and exercise:

- same name -> same object (cache hit)
- distinct kwargs -> distinct cached objects
- capability() with a fake probe, including the unregistered / raising cases
- clear_cache() forcing a fresh load
"""

from __future__ import annotations

import importlib

import pytest

from protein_design_hub.predictors import factory


@pytest.fixture
def fresh_factory(monkeypatch):
    """Reset the factory registry/cache and force the no-streamlit path.

    This guarantees the dict-cache backend is used regardless of whether
    streamlit happens to be installed, so the tests assert the CI behavior.
    """
    monkeypatch.setattr(factory, "is_streamlit_available", lambda: False)
    monkeypatch.setattr(factory, "_REGISTRY", {}, raising=True)
    monkeypatch.setattr(factory, "_FALLBACK_CACHE", {}, raising=True)
    return factory


def test_same_name_returns_same_object(fresh_factory):
    """Repeated get_model() for one name must return the identical object."""
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return object()

    fresh_factory.register_loader("fake", loader)

    first = fresh_factory.get_model("fake")
    second = fresh_factory.get_model("fake")
    # Case-insensitive name resolution must hit the same cache entry.
    third = fresh_factory.get_model("FAKE")

    assert first is second is third
    assert calls["n"] == 1, "loader should run exactly once (cached)"


def test_distinct_kwargs_produce_distinct_objects(fresh_factory):
    """Different kwargs are different cache keys -> different objects."""
    calls = {"n": 0}

    def loader(device="cpu"):
        calls["n"] += 1
        return {"device": device, "id": calls["n"]}

    fresh_factory.register_loader("plm", loader)

    cpu = fresh_factory.get_model("plm", device="cpu")
    cuda = fresh_factory.get_model("plm", device="cuda")
    cpu_again = fresh_factory.get_model("plm", device="cpu")

    assert cpu is not cuda
    assert cpu is cpu_again
    assert calls["n"] == 2


def test_loader_never_called_until_get_model(fresh_factory):
    """Registration must be lazy: no loading happens at register time."""
    called = {"v": False}

    def loader():
        called["v"] = True
        return 42

    fresh_factory.register_loader("lazy", loader)
    assert called["v"] is False
    assert fresh_factory.get_model("lazy") == 42
    assert called["v"] is True


def test_register_duplicate_raises_without_replace(fresh_factory):
    fresh_factory.register_loader("dup", lambda: 1)
    with pytest.raises(ValueError):
        fresh_factory.register_loader("dup", lambda: 2)
    # replace=True overrides and clears the old cached object.
    fresh_factory.get_model("dup")
    fresh_factory.register_loader("dup", lambda: 99, replace=True)
    assert fresh_factory.get_model("dup") == 99


def test_capability_with_fake_probe(fresh_factory):
    """capability() returns the probe's dict, normalized to the contract."""

    def probe():
        return {"available": True, "reason": "weights present", "size_mb": 650}

    fresh_factory.register_loader("esm2", lambda: object(), capability=probe)

    cap = fresh_factory.capability("esm2")
    assert cap["available"] is True
    assert cap["reason"] == "weights present"
    assert cap["size_mb"] == 650
    # Calling capability must NOT trigger model loading.


def test_capability_unavailable_probe(fresh_factory):
    """A probe reporting unavailable (e.g. missing torch) is surfaced as-is."""

    def probe():
        return {"available": False, "reason": "torch not installed"}

    fresh_factory.register_loader("needs_torch", lambda: object(), capability=probe)
    cap = fresh_factory.capability("needs_torch")
    assert cap["available"] is False
    assert "torch" in cap["reason"]


def test_capability_default_when_no_probe(fresh_factory):
    fresh_factory.register_loader("noprobe", lambda: object())
    cap = fresh_factory.capability("noprobe")
    assert cap == {"available": True, "reason": "loader registered"}


def test_capability_unregistered_name(fresh_factory):
    cap = fresh_factory.capability("does_not_exist")
    assert cap["available"] is False
    assert cap["reason"] == "not registered"


def test_capability_probe_that_raises_is_handled(fresh_factory):
    def boom():
        raise RuntimeError("cuda exploded")

    fresh_factory.register_loader("boom", lambda: object(), capability=boom)
    cap = fresh_factory.capability("boom")
    assert cap["available"] is False
    assert "cuda exploded" in cap["reason"]


def test_get_model_unregistered_raises(fresh_factory):
    with pytest.raises(KeyError):
        fresh_factory.get_model("nope")


def test_available_models_sorted(fresh_factory):
    fresh_factory.register_loader("zeta", lambda: 1)
    fresh_factory.register_loader("alpha", lambda: 2)
    assert fresh_factory.available_models() == ["alpha", "zeta"]


def test_clear_cache_forces_reload(fresh_factory):
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return object()

    fresh_factory.register_loader("reload", loader)
    a = fresh_factory.get_model("reload")
    fresh_factory.clear_cache("reload")
    b = fresh_factory.get_model("reload")
    assert a is not b
    assert calls["n"] == 2


def test_clear_cache_all(fresh_factory):
    fresh_factory.register_loader("a", lambda: object())
    fresh_factory.register_loader("b", lambda: object())
    fresh_factory.get_model("a")
    fresh_factory.get_model("b")
    fresh_factory.clear_cache()
    # Cache is empty; next loads create fresh objects (no assertion on identity
    # of old objects needed, just that it does not error and reloads).
    assert fresh_factory.get_model("a") is not None
    assert fresh_factory.get_model("b") is not None


def test_module_imports_without_streamlit_or_torch():
    """The factory module must import cleanly even with heavy deps absent."""
    mod = importlib.import_module("protein_design_hub.predictors.factory")
    assert hasattr(mod, "get_model")
    assert hasattr(mod, "capability")
    # is_streamlit_available must be callable and return a bool.
    assert isinstance(mod.is_streamlit_available(), bool)
