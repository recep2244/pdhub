"""Helpers to enforce Ollama GPU execution for LLM calls."""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

# ── TTL-cached GPU check ───────────────────────────────────────────
# Avoids shelling out to ``ollama ps`` on every single LLM call.
# Default TTL: 60 seconds.

_gpu_cache: Dict[str, str] = {}
_gpu_cache_ts: float = 0.0
_GPU_CACHE_TTL: float = 60.0


def get_ollama_processors(timeout: float = 3.0) -> Dict[str, str]:
    """Return model -> processor mapping from ``ollama ps``.

    Results are cached for 60 s to avoid excessive subprocess calls
    (a full pipeline can make 70+ LLM calls).

    Example processor values:
      - ``100% GPU``
      - ``100% CPU``
    """
    global _gpu_cache, _gpu_cache_ts

    now = time.monotonic()
    if _gpu_cache and (now - _gpu_cache_ts) < _GPU_CACHE_TTL:
        return _gpu_cache

    try:
        res = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except Exception:
        return _gpu_cache  # return stale cache on failure

    if res.returncode != 0 or not res.stdout.strip():
        return {}

    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return {}

    procs: Dict[str, str] = {}
    for ln in lines[1:]:
        cols = re.split(r"\s{2,}", ln.strip(), maxsplit=5)
        if len(cols) >= 4:
            procs[cols[0]] = cols[3]

    _gpu_cache = procs
    _gpu_cache_ts = now
    return procs


def invalidate_gpu_cache() -> None:
    """Force the next ``get_ollama_processors`` to re-check."""
    global _gpu_cache_ts
    _gpu_cache_ts = 0.0


# ── TTL-cached installed-model list ─────────────────────────────────
# Used by the meeting layer to transparently swap a missing model for an
# actually-installed one before the OpenAI-compatible call 404s.

_installed_cache: List[str] = []
_installed_cache_ts: float = 0.0
_INSTALLED_CACHE_TTL: float = 60.0


def get_installed_ollama_models(
    base_url: str = "http://localhost:11434",
    timeout: float = 2.0,
) -> List[str]:
    """Return list of model names installed locally via Ollama, 60 s cached.

    Reads from ``/api/tags`` (raw Ollama API, not the ``/v1`` compat layer)
    because that endpoint is cheap and returns the canonical name list.
    """
    global _installed_cache, _installed_cache_ts

    now = time.monotonic()
    if _installed_cache and (now - _installed_cache_ts) < _INSTALLED_CACHE_TTL:
        return _installed_cache

    # Strip ``/v1`` suffix if a base_url for the OpenAI compat layer was passed.
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    url = root + "/api/tags"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return _installed_cache  # stale on failure

    models = data.get("models", []) if isinstance(data, dict) else []
    names = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
    _installed_cache = names
    _installed_cache_ts = now
    return names


def invalidate_installed_cache() -> None:
    """Force the next ``get_installed_ollama_models`` to re-check."""
    global _installed_cache_ts
    _installed_cache_ts = 0.0


def resolve_installed_ollama_model(
    requested: str,
    base_url: str = "http://localhost:11434",
) -> str:
    """Return *requested* if installed, else a sensible installed fallback.

    Falls back in order: first model from ``OLLAMA_RECOMMENDED_MODELS`` that
    is installed, then any installed model. Returns *requested* unchanged
    when the installed-list probe fails (network/Ollama down) — the caller
    will surface the real backend error in that case.
    """
    installed = get_installed_ollama_models(base_url=base_url)
    if not installed:
        return requested
    if requested in installed:
        return requested

    # Try the project's recommended list, in priority order.
    try:
        from protein_design_hub.core.config import OLLAMA_RECOMMENDED_MODELS
        for rec, _desc in OLLAMA_RECOMMENDED_MODELS:
            if rec in installed:
                return rec
    except Exception:
        pass

    return installed[0]


def _match_model_processor(processors: Dict[str, str], model: Optional[str]) -> Optional[str]:
    if not processors:
        return None
    if model:
        if model in processors:
            return processors[model]
        model_base = model.split(":", 1)[0]
        for name, proc in processors.items():
            if name == model_base or name.startswith(model_base + ":"):
                return proc
    # Fallback: if only one loaded model, use it.
    if len(processors) == 1:
        return next(iter(processors.values()))
    return None


def ensure_ollama_gpu(provider: str, model: Optional[str]) -> None:
    """Raise if Ollama currently reports CPU-only processing for the target model."""
    if provider != "ollama":
        return
    processors = get_ollama_processors()
    if not processors:
        # No loaded model yet; cannot assert processor type at this point.
        return

    proc = _match_model_processor(processors, model)
    if proc is None:
        # If target model not found but all loaded models are CPU, fail fast.
        vals = list(processors.values())
        if vals and all("GPU" not in v.upper() for v in vals):
            raise RuntimeError(
                "Ollama is running CPU-only. Restart Ollama and ensure GPU is active "
                "(check with `ollama ps`)."
            )
        return

    if "GPU" not in proc.upper():
        raise RuntimeError(
            f"Ollama model '{model or 'active model'}' is on CPU ({proc}). "
            "GPU is required for this stage."
        )


def ollama_extra_body(provider: str, model: str = "") -> dict:
    """Return request extras that maximise Ollama GPU throughput.

    Performance options:
      - ``num_gpu: 999``       – offload all layers to GPU
      - ``num_ctx: 8192``      – context window large enough for long
        accumulated meeting prompts (honoured only via native /api/chat)
      - ``num_batch: 512``     – larger prompt eval batch → faster prefill
      - ``think: false``       – disable Qwen3 chain-of-thought token for
        fast non-reasoning responses (scientific pipeline uses reasoning
        selectively; disable by default for speed)

    For Qwen3 models, ``think: false`` is added to suppress the <think> token
    overhead (saves ~1-3 s per call in agentic pipelines).  Set to ``true``
    explicitly if you want full chain-of-thought on a specific call.
    """
    if provider != "ollama":
        return {}

    options: dict = {
        "num_gpu": 999,
        # 8192 gives headroom for long accumulated meeting prompts (~4-7K tokens)
        # so Ollama never sliding-window "K-shift" truncates them. Verified:
        # qwen2.5:14b-instruct-q3_K_M at num_ctx=8192 stays 100% GPU (~9.3 GiB)
        # on a 12 GB card. NOTE: only honoured via the native /api/chat path —
        # Ollama's OpenAI-compatible /v1 endpoint ignores `options` entirely.
        "num_ctx": 8192,
        "num_batch": 512,
    }

    # Qwen3: disable thinking tokens by default for pipeline speed.
    # The reasoning is still excellent without explicit <think> blocks.
    m = str(model or "").lower()
    if "qwen3" in m or "qwen-3" in m:
        options["think"] = False

    return {
        "extra_body": {
            "options": options,
            "keep_alive": "10m",
        },
    }
