"""Phase-7 product-value validation gate.

A small, dependency-free release-gate harness that turns a corpus of pipeline
runs and an exported shortlist into the two product KPIs that govern whether
Protein Design Hub is ready to ship:

* **KPI-1 — Intent-to-shortlist.** The fraction of user runs that actually
  reached a non-empty, exported shortlist. This measures whether the product
  carries a user from "intent" all the way to a tangible, actionable result.
  Target: >= 0.70.
* **KPI-3 — Shortlist precision.** The precision of the exported shortlist when
  graded against *external* ground-truth labels (e.g. wet-lab or benchmark
  outcomes). This measures whether the things we surface are actually good.
  Target: >= 0.90.

The gate is deliberately pure Python with no heavy dependencies so it stays
import-safe in environments without torch/GPU/binaries, and can run in CI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

__all__ = [
    "kpi1_intent_to_shortlist",
    "kpi3_shortlist_precision",
    "evaluate_gate",
    "INSUFFICIENT_CORPUS_REASON",
    "DEFAULT_KPI1_TARGET",
    "DEFAULT_KPI3_TARGET",
]

#: Human-readable reason returned when the corpus is too thin to grade.
INSUFFICIENT_CORPUS_REASON = "insufficient corpus"

#: Default KPI targets (also exposed as ``evaluate_gate`` keyword defaults).
DEFAULT_KPI1_TARGET = 0.70
DEFAULT_KPI3_TARGET = 0.90


def _run_reached_shortlist(run: Mapping[str, Any]) -> bool:
    """Return ``True`` if a single run reached a non-empty exported shortlist.

    A run counts as a success when it both (a) was marked as exported, and
    (b) carries a non-empty shortlist payload. We are permissive about the
    exact schema so the gate can grade heterogeneous run records:

    * ``shortlist`` — a sequence of shortlisted candidates; non-empty wins.
    * ``shortlist_size`` — an integer count; ``> 0`` wins.
    * ``exported`` — an explicit boolean flag (only blocks when ``False``).
    """
    if not isinstance(run, Mapping):
        return False

    # An explicit exported flag, when present and falsey, vetoes the run.
    if "exported" in run and not run.get("exported"):
        return False

    shortlist = run.get("shortlist")
    if isinstance(shortlist, Sequence) and not isinstance(shortlist, (str, bytes)):
        if len(shortlist) > 0:
            return True

    size = run.get("shortlist_size")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        if size > 0:
            return True

    # Fall back to an explicit boolean only if a shortlist payload is absent.
    if shortlist is None and size is None and run.get("exported") is True:
        return True

    return False


def kpi1_intent_to_shortlist(runs: List[Dict[str, Any]]) -> float:
    """Fraction of runs that reached a non-empty exported shortlist.

    Args:
        runs: Sequence of run records. See :func:`_run_reached_shortlist` for
            the accepted per-run schema.

    Returns:
        Fraction in ``[0.0, 1.0]``. Returns ``0.0`` for an empty corpus.
    """
    if not runs:
        return 0.0
    reached = sum(1 for run in runs if _run_reached_shortlist(run))
    return reached / len(runs)


def _candidate_id(item: Any) -> Any:
    """Extract the identifier from a shortlist entry (dict or scalar)."""
    if isinstance(item, Mapping):
        return item.get("id")
    return item


def kpi3_shortlist_precision(
    shortlist: List[Dict[str, Any]],
    labels: Dict[Any, bool],
) -> float:
    """Precision of the shortlist against external labels.

    Each shortlist entry is matched to ``labels`` by its ``id`` (a dict with an
    ``"id"`` key, or a bare scalar id). An entry is a true positive when its
    label is truthy. Entries whose id is missing from ``labels`` are graded as
    negatives (we cannot credit an unverified hit).

    Args:
        shortlist: The exported shortlist to grade.
        labels: Mapping of candidate ``id`` -> positive (truthy) / negative.

    Returns:
        Precision in ``[0.0, 1.0]``. Returns ``0.0`` for an empty shortlist.
    """
    if not shortlist:
        return 0.0
    positives = sum(
        1 for item in shortlist if bool(labels.get(_candidate_id(item), False))
    )
    return positives / len(shortlist)


def evaluate_gate(
    runs: List[Dict[str, Any]],
    shortlist: List[Dict[str, Any]],
    labels: Dict[Any, bool],
    kpi1_target: float = DEFAULT_KPI1_TARGET,
    kpi3_target: float = DEFAULT_KPI3_TARGET,
) -> Dict[str, Any]:
    """Evaluate the Phase-7 release gate over a corpus + shortlist + labels.

    The gate passes (``release_ok=True``) only when *both* KPIs meet their
    targets. A degenerate corpus — no runs, an empty shortlist, or no external
    labels to grade against — cannot release and returns ``release_ok=False``
    with ``reason == INSUFFICIENT_CORPUS_REASON``.

    Args:
        runs: Pipeline run records for KPI-1.
        shortlist: Exported shortlist for KPI-3.
        labels: External ground-truth labels keyed by candidate id.
        kpi1_target: Minimum intent-to-shortlist fraction (default 0.70).
        kpi3_target: Minimum shortlist precision (default 0.90).

    Returns:
        A dict with both KPI values, per-KPI pass flags, targets, the corpus
        sizes, an overall ``release_ok`` boolean, and a human ``verdict``
        string. When the corpus is insufficient it also carries ``reason``.
    """
    insufficient = not runs or not shortlist or not labels

    kpi1 = kpi1_intent_to_shortlist(runs)
    kpi3 = kpi3_shortlist_precision(shortlist, labels)

    kpi1_pass = kpi1 >= kpi1_target
    kpi3_pass = kpi3 >= kpi3_target

    result: Dict[str, Any] = {
        "kpi1_intent_to_shortlist": kpi1,
        "kpi3_shortlist_precision": kpi3,
        "kpi1_target": kpi1_target,
        "kpi3_target": kpi3_target,
        "kpi1_pass": kpi1_pass,
        "kpi3_pass": kpi3_pass,
        "n_runs": len(runs),
        "n_shortlist": len(shortlist),
        "n_labels": len(labels),
    }

    if insufficient:
        result["release_ok"] = False
        result["reason"] = INSUFFICIENT_CORPUS_REASON
        result["verdict"] = (
            "BLOCKED: insufficient corpus to grade the release gate "
            f"(runs={len(runs)}, shortlist={len(shortlist)}, labels={len(labels)})."
        )
        return result

    release_ok = kpi1_pass and kpi3_pass
    result["release_ok"] = release_ok

    if release_ok:
        result["verdict"] = (
            "RELEASE OK: both KPIs met "
            f"(intent-to-shortlist {kpi1:.0%} >= {kpi1_target:.0%}, "
            f"precision {kpi3:.0%} >= {kpi3_target:.0%})."
        )
    else:
        failed = []
        if not kpi1_pass:
            failed.append(
                f"intent-to-shortlist {kpi1:.0%} < target {kpi1_target:.0%}"
            )
        if not kpi3_pass:
            failed.append(f"precision {kpi3:.0%} < target {kpi3_target:.0%}")
        result["verdict"] = "BLOCKED: " + "; ".join(failed) + "."

    return result
