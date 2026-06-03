#!/usr/bin/env python3
"""Example: predict a dimer with ESMFold and score multimer quality.

Pipeline
--------
1. Query the ESMFold API for a *dimer* — the API is single-chain, so we use the
   poly-glycine linker trick (chain A + GGG…G + chain B), then split the linker
   back out into two chains. (Approximation; see analysis/esm_dimer.py. For a
   true multimer use ColabFold/AlphaFold-Multimer/Chai-1/Boltz.)
2. Score the predicted complex against a reference complex with OpenStructure
   compare-structures, reporting the multimeric metrics:
       DockQ, QS-score (global/best), ICS, IPS, interface-lDDT (+ global lDDT/TM).

Run:
    python examples/esm_dimer_multimer_metrics.py            # fold + (optional) score
    python examples/esm_dimer_multimer_metrics.py REF.pdb    # also score vs REF.pdb

Metric cheat-sheet (all 0–1, higher = better, vs the reference complex):
    DockQ  ≥0.80 high / 0.49–0.80 medium / 0.23–0.49 acceptable / <0.23 incorrect
    QS     quaternary-structure score (interface contact agreement, symmetric)
    ICS    interface contact similarity (F1 over conserved inter-chain contacts)
    IPS    interface patch similarity (F1 over interface residues; "jaccard"-like)
    ilddt  inter-chain lDDT
"""

import sys
from pathlib import Path

from protein_design_hub.analysis.esm_dimer import fold_dimer_esm_api
from protein_design_hub.evaluation.ost_runner import get_ost_runner

# Two short example chains (replace with your own). A homo/hetero-dimer both work.
SEQ_A = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER"   # e.g. a leucine-zipper-like helix
SEQ_B = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER"


def main() -> None:
    out = Path("dimer_model.pdb")
    print(f"Folding dimer ({len(SEQ_A)} + {len(SEQ_B)} aa) via ESMFold poly-G linker…")
    pdb = fold_dimer_esm_api(SEQ_A, SEQ_B, linker_len=25)
    out.write_text(pdb)
    n_chains = len({ln[21] for ln in pdb.splitlines() if ln.startswith("ATOM")})
    print(f"  ✓ wrote {out} ({n_chains} chains)")

    ref = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not ref or not ref.exists():
        print("\nNo reference complex given — skipping multimer scoring.")
        print("Provide one to compute DockQ/QS/ICS/IPS:")
        print("    python examples/esm_dimer_multimer_metrics.py reference_complex.pdb")
        return

    runner = get_ost_runner()
    if not runner.is_available():
        print("OpenStructure not available — cannot compute multimer metrics.")
        return
    print(f"\nScoring {out.name} vs reference {ref.name} (OST compare-structures)…")
    m = runner.compute_multimer_metrics(out, ref)
    if m.get("status") != "SUCCESS":
        print(f"  scoring failed: {m.get('error')}")
        return
    print("  Multimer metrics vs reference:")
    for k in ("dockq", "qs_global", "qs_best", "ics", "ips", "ilddt", "lddt", "tm_score"):
        v = m.get(k)
        print(f"    {k:10s}: {v:.3f}" if isinstance(v, (int, float)) else f"    {k:10s}: N/A")


if __name__ == "__main__":
    main()
