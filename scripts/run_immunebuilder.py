"""Run ImmuneBuilder in an isolated environment.

Usage:
  python scripts/run_immunebuilder.py --input_json /path/input.json \
      --output_pdb /path/out.pdb --output_json /path/out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


def _extract_errors_from_pdb(pdb_path: Path, chain_id: Optional[str]) -> List[float]:
    errors: List[float] = []
    seen_residues = set()
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        if chain_id and line[21].strip() != chain_id:
            continue
        try:
            residue_id = (line[21], int(line[22:26]))
        except Exception:
            continue
        if residue_id in seen_residues:
            continue
        seen_residues.add(residue_id)
        try:
            errors.append(float(line[60:66]))
        except Exception:
            continue
    return errors


def _extract_errors_from_object(obj) -> Optional[List[float]]:
    for attr in [
        "errors",
        "error",
        "residue_errors",
        "residue_error",
        "predicted_errors",
        "predicted_error",
    ]:
        val = getattr(obj, attr, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_pdb", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_pdb = Path(args.output_pdb)
    output_json = Path(args.output_json)

    payload = json.loads(input_path.read_text())
    mode = payload.get("mode")
    active_chain_id = payload.get("active_chain_id")
    chain_a = payload.get("chain_a")
    chain_b = payload.get("chain_b")

    if not chain_a:
        raise ValueError("chain_a is required")

    if mode not in {"antibody", "nanobody", "tcr"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode in {"antibody", "tcr"} and not chain_b:
        raise ValueError("chain_b is required for this mode")

    if mode == "antibody":
        from ImmuneBuilder import ABodyBuilder2

        predictor = ABodyBuilder2()
        sequences = {"H": chain_a, "L": chain_b}
    elif mode == "nanobody":
        from ImmuneBuilder import NanoBodyBuilder2

        predictor = NanoBodyBuilder2()
        sequences = {"H": chain_a}
    else:
        from ImmuneBuilder import TCRBuilder2

        predictor = TCRBuilder2()
        sequences = {"A": chain_a, "B": chain_b}

    model = predictor.predict(sequences)
    model.save(str(output_pdb))

    errors = _extract_errors_from_object(model)
    if errors is None:
        errors = _extract_errors_from_pdb(output_pdb, active_chain_id)

    mean_error = (sum(errors) / len(errors)) if errors else None

    output_json.write_text(
        json.dumps(
            {
                "mean_error": mean_error,
                "per_residue_error": errors,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
