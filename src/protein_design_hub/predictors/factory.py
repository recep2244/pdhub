"""Cached model/predictor factory.

This module centralizes expensive model loading (PLMs, structure predictors,
scorers) behind a single cache so that repeated calls for the same model name
return the *same* object instead of re-loading weights.

Design goals
------------
- **Streamlit-aware, but not Streamlit-dependent.** When ``streamlit`` is
  importable we wrap the loader with ``st.cache_resource`` so the Streamlit
  runtime owns the cached object (survives reruns, shared across sessions).
  When Streamlit is absent (CLI, batch jobs, CI) we transparently fall back to
  a process-level ``dict`` cache. Either way the public API is identical.
- **No heavy imports at module top level.** ``torch`` and friends are never
  imported here; loaders register themselves and are only invoked lazily.
- **Capability probing.** ``capability(name)`` returns a small
  ``{available, reason}`` dict describing whether a model can be loaded
  *without* actually loading it, so UI/CLI code can degrade gracefully.

Public API
----------
- ``register_loader(name, loader, *, capability=None, replace=False)``
- ``get_model(name, **kwargs)``
- ``capability(name) -> {available, reason, ...}``
- ``available_models() -> list[str]``
- ``clear_cache(name=None)``
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "register_loader",
    "get_model",
    "capability",
    "available_models",
    "clear_cache",
    "is_streamlit_available",
]

# A loader takes arbitrary keyword args and returns a (typically expensive)
# model object. It must be deterministic w.r.t. its kwargs for caching to be
# meaningful.
Loader = Callable[..., Any]
# A capability probe returns a {"available": bool, "reason": str, ...} dict.
# It must be cheap and must NOT import heavy deps unless they are already
# importable (use importlib.util.find_spec to check first).
CapabilityProbe = Callable[[], Dict[str, Any]]


def is_streamlit_available() -> bool:
    """Return True if Streamlit is importable in this environment.

    We check importability without keeping a hard dependency; the result is
    used only to decide *which* cache backend to use.
    """
    import importlib.util

    return importlib.util.find_spec("streamlit") is not None


class _Registration:
    """Bookkeeping for a single registered model loader."""

    __slots__ = ("name", "loader", "capability")

    def __init__(
        self,
        name: str,
        loader: Loader,
        capability: Optional[CapabilityProbe],
    ) -> None:
        self.name = name
        self.loader = loader
        self.capability = capability


# Module-level registry of name -> _Registration. Guarded by a lock because
# loaders may be called from multiple threads (e.g. the PyMOL worker thread or
# background jobs).
_REGISTRY: Dict[str, _Registration] = {}
# Fallback cache used only when Streamlit is not available. Keyed by
# (name, frozen-kwargs) so different kwargs yield different cached objects,
# matching st.cache_resource semantics.
_FALLBACK_CACHE: Dict[Any, Any] = {}
_LOCK = threading.RLock()


def _freeze_kwargs(kwargs: Dict[str, Any]) -> Any:
    """Produce a hashable, order-independent key for kwargs.

    Mirrors how ``st.cache_resource`` keys on arguments. Unhashable values are
    stringified so we never crash on, e.g., a list passed as a kwarg.
    """
    items = []
    for key in sorted(kwargs):
        value = kwargs[key]
        try:
            hash(value)
        except TypeError:
            value = repr(value)
        items.append((key, value))
    return tuple(items)


def register_loader(
    name: str,
    loader: Loader,
    *,
    capability: Optional[CapabilityProbe] = None,
    replace: bool = False,
) -> None:
    """Register a model loader under ``name``.

    Args:
        name: Case-insensitive model identifier.
        loader: Callable that builds and returns the model. Invoked lazily and
            at most once per distinct kwargs combination (results are cached).
        capability: Optional cheap probe returning ``{available, reason, ...}``.
            If omitted, the model is assumed available with reason
            ``"loader registered"``.
        replace: If False (default), re-registering an existing name raises
            ``ValueError``. Pass True to overwrite (also clears that name's
            cached objects).

    Raises:
        ValueError: If ``name`` is already registered and ``replace`` is False.
    """
    key = name.lower()
    with _LOCK:
        if key in _REGISTRY and not replace:
            raise ValueError(
                f"Loader '{name}' is already registered. "
                f"Pass replace=True to override."
            )
        if key in _REGISTRY and replace:
            clear_cache(key)
        _REGISTRY[key] = _Registration(key, loader, capability)


def _build_cached_loader(reg: _Registration) -> Loader:
    """Return a cached version of ``reg.loader``.

    Uses ``st.cache_resource`` when Streamlit is importable, otherwise a
    module-level dict cache. The returned callable has identical signature to
    the original loader.
    """
    if is_streamlit_available():
        import streamlit as st

        # st.cache_resource caches by the function identity + args. We wrap a
        # thin closure so each registered name gets its own cache entry; the
        # name is baked in as a default arg to keep the cache key stable.
        @st.cache_resource(show_spinner=False)
        def _cached(_name: str = reg.name, **kwargs: Any) -> Any:
            return _REGISTRY[_name].loader(**kwargs)

        return _cached

    # Fallback: process-level dict cache keyed by (name, frozen kwargs).
    def _cached(_name: str = reg.name, **kwargs: Any) -> Any:
        cache_key = (_name, _freeze_kwargs(kwargs))
        with _LOCK:
            if cache_key not in _FALLBACK_CACHE:
                _FALLBACK_CACHE[cache_key] = _REGISTRY[_name].loader(**kwargs)
            return _FALLBACK_CACHE[cache_key]

    return _cached


def get_model(name: str, **kwargs: Any) -> Any:
    """Return the (cached) model registered under ``name``.

    Repeated calls with the same name and kwargs return the *same* object.

    Args:
        name: Case-insensitive model identifier.
        **kwargs: Passed through to the registered loader. Different kwargs
            produce (and cache) distinct objects.

    Returns:
        The model object produced by the loader.

    Raises:
        KeyError: If no loader is registered under ``name``.
    """
    key = name.lower()
    with _LOCK:
        reg = _REGISTRY.get(key)
    if reg is None:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"No loader registered for '{name}'. Available: {available}")

    cached = _build_cached_loader(reg)
    # Pass name explicitly so the Streamlit cache key includes it.
    return cached(reg.name, **kwargs)


def capability(name: str) -> Dict[str, Any]:
    """Probe whether ``name`` can be loaded, without loading it.

    Args:
        name: Case-insensitive model identifier.

    Returns:
        A dict that always contains ``available`` (bool) and ``reason`` (str).
        Additional keys from the registered probe are preserved. Unknown names
        return ``{"available": False, "reason": "not registered"}``. A probe
        that raises is reported as unavailable with the exception text.
    """
    key = name.lower()
    with _LOCK:
        reg = _REGISTRY.get(key)
    if reg is None:
        return {"available": False, "reason": "not registered"}
    if reg.capability is None:
        return {"available": True, "reason": "loader registered"}
    try:
        result = reg.capability()
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": f"probe failed: {exc}"}
    # Normalize: guarantee the contract keys exist.
    if not isinstance(result, dict):
        return {"available": bool(result), "reason": "loader registered"}
    result.setdefault("available", False)
    result.setdefault("reason", "")
    return result


def available_models() -> List[str]:
    """Return the sorted list of registered model names."""
    with _LOCK:
        return sorted(_REGISTRY)


def clear_cache(name: Optional[str] = None) -> None:
    """Clear cached model objects.

    Args:
        name: If given, clear only that model's cached entries; otherwise clear
            everything. Affects both the Streamlit and fallback caches.
    """
    key = name.lower() if name is not None else None

    # Clear the fallback dict cache.
    with _LOCK:
        if key is None:
            _FALLBACK_CACHE.clear()
        else:
            for cache_key in [k for k in _FALLBACK_CACHE if k[0] == key]:
                del _FALLBACK_CACHE[cache_key]

    # Clear the Streamlit resource cache if present. st.cache_resource only
    # exposes a global clear, so a targeted clear degrades to a full clear.
    if is_streamlit_available():
        try:
            import streamlit as st

            st.cache_resource.clear()
        except Exception:  # pragma: no cover - streamlit runtime edge cases
            pass
