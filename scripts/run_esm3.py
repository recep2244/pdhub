"""Run ESM3 in an isolated environment.

Usage:
  python scripts/run_esm3.py --input_json /path/input.json \
      --output_pdb /path/out.pdb --output_json /path/out.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List


def _extract_plddt_from_pdb(pdb_path: Path) -> List[float]:
    plddt_values: List[float] = []
    seen_residues = set()
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            residue_id = (line[21], int(line[22:26]))
        except Exception:
            continue
        if residue_id in seen_residues:
            continue
        seen_residues.add(residue_id)
        try:
            plddt_values.append(float(line[60:66]))
        except Exception:
            continue
    return plddt_values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_pdb", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text())
    sequence = payload.get("sequence")
    model_name = payload.get("model_name", "esm3-open")
    num_steps = int(payload.get("num_steps", 8))

    if not sequence:
        raise ValueError("sequence is required")

    import esm
    from esm.sdk.api import ESMProtein, GenerationConfig

    hf_token = (
        payload.get("hf_token")
        or os.getenv("ESM3_HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
    )
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    token = payload.get("forge_token")
    if token:
        model = esm.sdk.client(model_name, token=token)
    else:
        from esm.models.esm3 import ESM3
        import torch

        requested_device = payload.get("device") or ""
        device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda"):
            try:
                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"
        model = ESM3.from_pretrained(model_name).to(device)

    protein = ESMProtein(sequence=sequence)
    gen_config = GenerationConfig(track="structure", num_steps=max(1, num_steps))
    protein = model.generate(protein, gen_config)

    output_pdb = Path(args.output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    protein.to_pdb(str(output_pdb))

    plddt_values = _extract_plddt_from_pdb(output_pdb)

    Path(args.output_json).write_text(
        json.dumps(
            {
                "plddt": plddt_values,
                "mean_plddt": (sum(plddt_values) / len(plddt_values)) if plddt_values else None,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
