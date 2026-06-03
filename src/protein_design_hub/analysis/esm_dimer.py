"""Predict a pseudo-dimer with ESMFold and split it into two chains.

The ESM Atlas fold API (``/foldSequence/v1/pdb/``) is **single-chain only** — it
rejects ``:`` chain separators. To get a complex out of ESMFold you join the two
chains with a poly-glycine linker, fold the concatenation, then drop the linker
and relabel the halves as chains A/B.

⚠️ This is an APPROXIMATION: ESMFold has no inter-chain co-evolution signal and
the linker can bias the interface. For production multimer prediction use a true
multimer model (ColabFold / AlphaFold-Multimer / Chai-1 / Boltz) — all available
in this hub. Use this for cheap triage only.

Score the result against a reference complex with
``evaluation.ost_runner.get_ost_runner().compute_multimer_metrics()`` to get
DockQ, QS-score, ICS and IPS.
"""

from __future__ import annotations

import urllib.request

ESM_FOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"
_VALID = set("ACDEFGHIKLMNPQRSTVWY")


def fold_sequence_esm_api(sequence: str, timeout: float = 300.0) -> str:
    """POST a single sequence to the ESMFold API and return the PDB string."""
    seq = "".join(c for c in sequence.upper() if c in _VALID)
    req = urllib.request.Request(ESM_FOLD_API, data=seq.encode(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fold_dimer_esm_api(
    seq_a: str,
    seq_b: str,
    linker_len: int = 25,
    timeout: float = 300.0,
) -> str:
    """Predict a 2-chain pseudo-dimer via the poly-glycine linker trick.

    Returns a PDB string with chain A (``seq_a``) and chain B (``seq_b``); the
    linker residues are removed and chain B is renumbered from 1.
    """
    seq_a = "".join(c for c in seq_a.upper() if c in _VALID)
    seq_b = "".join(c for c in seq_b.upper() if c in _VALID)
    if not seq_a or not seq_b:
        raise ValueError("Both chains must be non-empty amino-acid sequences.")
    combined = seq_a + ("G" * linker_len) + seq_b
    raw_pdb = fold_sequence_esm_api(combined, timeout=timeout)
    return split_linker_pdb(raw_pdb, len(seq_a), linker_len, len(seq_b))


def split_linker_pdb(pdb: str, len_a: int, linker_len: int, len_b: int) -> str:
    """Split a linker-joined ESMFold PDB into chains A/B, dropping the linker.

    ESMFold numbers residues sequentially 1..N. Residues 1..len_a → chain A;
    the next ``linker_len`` residues (the poly-G) are removed; the final
    ``len_b`` residues → chain B, renumbered from 1.
    """
    a_end = len_a
    b_start = len_a + linker_len + 1
    out_lines = []
    wrote_a_ter = False
    for line in pdb.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            if resseq <= a_end:
                out_lines.append(line[:21] + "A" + line[22:])
            elif resseq >= b_start:
                if not wrote_a_ter:
                    out_lines.append("TER")
                    wrote_a_ter = True
                new_seq = resseq - (b_start - 1)
                out_lines.append(line[:21] + "B" + ("%4d" % new_seq) + line[26:])
            # else: linker residue → drop
    out_lines.append("TER")
    out_lines.append("END")
    return "\n".join(out_lines) + "\n"
