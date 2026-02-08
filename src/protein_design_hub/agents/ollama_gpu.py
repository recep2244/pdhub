"""Helpers to enforce Ollama GPU execution for LLM calls."""

from __future__ import annotations

import re
import subprocess
from typing import Dict, Optional


def get_ollama_processors(timeout: float = 3.0) -> Dict[str, str]:
    """Return model -> processor mapping from ``ollama ps``.

    Example processor values:
      - ``100% GPU``
      - ``100% CPU``
    """
    try:
        res = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except Exception:
        return {}

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
    return procs


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
    """Return request extras that hint Ollama to use GPU."""
    if provider != "ollama":
        return {}
    # Ollama accepts llama.cpp options via `options`.
    return {"extra_body": {"options": {"num_gpu": 999}}}

