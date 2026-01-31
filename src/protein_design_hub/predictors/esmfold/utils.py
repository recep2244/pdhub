"""Utilities for loading ESMFold checkpoints."""

from __future__ import annotations

from typing import Optional

import torch
from esm.esmfold.v1.esmfold import ESMFold


def load_esmfold_model(version: str = "v1", allow_missing: bool = False) -> ESMFold:
    """Load an ESMFold model checkpoint with optional missing-key tolerance."""
    model_name = "esmfold_3B_v0" if version == "v0" else "esmfold_3B_v1"
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")

    cfg = model_data["cfg"]["model"]
    # Older checkpoints may not include newer config keys.
    try:
        _ = cfg.use_esm_attn_map
    except Exception:
        if isinstance(cfg, dict):
            cfg["use_esm_attn_map"] = False
        else:
            setattr(cfg, "use_esm_attn_map", False)
    model_state = model_data["model"]
    model = ESMFold(esmfold_config=cfg)

    if not allow_missing:
        expected_keys = set(model.state_dict().keys())
        found_keys = set(model_state.keys())
        missing_essential_keys = [
            key for key in expected_keys - found_keys if not key.startswith("esm.")
        ]
        if missing_essential_keys:
            raise RuntimeError(
                "Keys '{}' are missing.".format(", ".join(sorted(missing_essential_keys))
                )
            )

    model.load_state_dict(model_state, strict=False)
    return model
