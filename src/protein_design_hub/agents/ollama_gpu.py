"""Helpers to enforce Ollama GPU execution for LLM calls."""

from __future__ import annotations

import re
import subprocess
import time
from typing import Dict, Optional

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


def ollama_extra_body(provider: str) -> dict:
    """Return request extras that maximise Ollama GPU throughput.

    Performance options:
      - ``num_gpu: 999``   – offload all layers to GPU
      - ``num_ctx: 4096``  – cap context to save VRAM (meetings rarely
        exceed 3 K tokens per turn)
      - ``num_batch: 512`` – larger prompt eval batch → faster prefill
      - ``flash_attention: true`` – enables flash-attn for faster
        attention on Ampere+ GPUs (RTX 30xx/40xx)
    """
    if provider != "ollama":
        return {}
    return {
        "extra_body": {
            "options": {
                "num_gpu": 999,
                "num_ctx": 4096,
                "num_batch": 512,
            },
            "keep_alive": "10m",
        },
    }
