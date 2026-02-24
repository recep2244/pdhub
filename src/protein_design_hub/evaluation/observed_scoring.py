"""Comprehensive observed-structure scoring: OpenStructure + MolProbity.

This module provides ObservedScoringRunner, which mirrors the
TMscoring-lddt-newmolprobility.sh pipeline from the restraint project,
running locally using:
  - OpenStructure (ost compare-structures) for lDDT, TM-score, QS-score, DockQ, etc.
  - MolProbity (phenix.molprobity + phenix.clashscore) for geometry quality metrics

Outputs:
  - pandas DataFrame with per-model rows and metric columns (ready for display)
  - JSON files per model in output_dir/openstructure/
  - MolProbity text files in output_dir/molprobity/
  - observed_scores.tsv (TSV summary)
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Default local paths (configurable in Settings page)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_OST_PREFIX = "/home/recep/.local/share/mamba/envs/ost"
_DEFAULT_MOL_ROOT = "/home/recep/Desktop/programs/scoring/Molprobity"

# Process-wide caches to avoid repeated disk probes
_OST_AVAILABLE: Optional[bool] = None
_MOL_AVAILABLE: Optional[bool] = None
_SINGLETON: Optional["ObservedScoringRunner"] = None


def get_observed_scorer(
    ost_prefix: Optional[str] = None,
    molprobity_root: Optional[str] = None,
) -> "ObservedScoringRunner":
    """Return the shared ObservedScoringRunner singleton."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = ObservedScoringRunner(
            ost_prefix=ost_prefix or _DEFAULT_OST_PREFIX,
            molprobity_root=molprobity_root or _DEFAULT_MOL_ROOT,
        )
    return _SINGLETON


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    """One model to score."""
    tool: str          # e.g. "boltz2", "chai1", "colabfold"
    model: str         # model identifier
    model_path: Path
    model_stage: str = "candidate"  # "candidate" or "final"


@dataclass
class ScoringResult:
    """Scoring result for one model."""
    tool: str
    model: str
    model_stage: str
    model_path: Path
    openstructure_status: str = "missing"
    ost_metrics: Dict[str, Any] = field(default_factory=dict)
    molprobity_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def lddt(self) -> Optional[float]:
        return self.ost_metrics.get("ost.lddt")

    @property
    def tm_score(self) -> Optional[float]:
        return self.ost_metrics.get("ost.tm_score")

    @property
    def qs_global(self) -> Optional[float]:
        return self.ost_metrics.get("ost.qs_global")

    @property
    def molprobity_score(self) -> Optional[float]:
        return self.molprobity_metrics.get("molprobity_score")

    @property
    def clashscore(self) -> Optional[float]:
        return self.molprobity_metrics.get("molprobity_clashscore") or \
               self.molprobity_metrics.get("phenix_clashscore")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class ObservedScoringRunner:
    """
    Score protein models against an observed structure using
    OpenStructure and MolProbity.

    Parameters
    ----------
    ost_prefix:
        Path to the conda/mamba environment containing OpenStructure.
    molprobity_root:
        Path to the MolProbity source checkout (must have build/setpaths.sh).
    """

    OST_SCORE_ARGS = [
        "--lddt", "--local-lddt", "--aa-local-lddt",
        "--bb-lddt", "--bb-local-lddt", "--ilddt",
        "--qs-score", "--dockq", "--ics", "--ips",
        "--rigid-scores", "--patch-scores", "--tm-score",
    ]

    def __init__(
        self,
        ost_prefix: str = _DEFAULT_OST_PREFIX,
        molprobity_root: str = _DEFAULT_MOL_ROOT,
    ):
        self.ost_prefix = Path(ost_prefix)
        self.molprobity_root = Path(molprobity_root)
        self._ost_ok: Optional[bool] = None
        self._mol_ok: Optional[bool] = None

    # ── availability ──────────────────────────────────────────────────────────

    def is_ost_available(self) -> bool:
        """Check if ost compare-structures is available."""
        global _OST_AVAILABLE
        if _OST_AVAILABLE is not None:
            return _OST_AVAILABLE
        ost_bin = self.ost_prefix / "bin" / "ost"
        if not ost_bin.exists():
            _OST_AVAILABLE = False
            return False
        try:
            r = subprocess.run(
                [str(ost_bin), "compare-structures", "--help"],
                capture_output=True, timeout=10,
            )
            _OST_AVAILABLE = (r.returncode in (0, 1))  # help exits 1 on some builds
        except Exception:
            _OST_AVAILABLE = False
        return _OST_AVAILABLE

    def is_molprobity_available(self) -> bool:
        """Check if MolProbity (phenix.molprobity) is available."""
        global _MOL_AVAILABLE
        if _MOL_AVAILABLE is not None:
            return _MOL_AVAILABLE
        setpaths = self.molprobity_root / "build" / "setpaths.sh"
        phenix = self.molprobity_root / "build" / "bin" / "phenix.molprobity"
        _MOL_AVAILABLE = setpaths.exists() and phenix.exists()
        return _MOL_AVAILABLE

    def availability_summary(self) -> str:
        """Human-readable availability string."""
        parts = []
        parts.append("OST " + ("✅" if self.is_ost_available() else "❌"))
        parts.append("MolProbity " + ("✅" if self.is_molprobity_available() else "❌"))
        return " | ".join(parts)

    # ── main entry point ──────────────────────────────────────────────────────

    def score(
        self,
        models: List[ModelEntry],
        reference: Path,
        output_dir: Path,
        jobs: int = 4,
    ) -> Tuple[List[ScoringResult], Path]:
        """
        Score all models against the reference structure.

        Parameters
        ----------
        models:
            List of ModelEntry objects to score.
        reference:
            Observed/reference structure path (PDB or CIF).
        output_dir:
            Directory to write per-model JSON and text files.
        jobs:
            Number of parallel scoring jobs.

        Returns
        -------
        (results_list, tsv_path)
        """
        output_dir = Path(output_dir)
        (output_dir / "openstructure").mkdir(parents=True, exist_ok=True)
        (output_dir / "molprobity").mkdir(parents=True, exist_ok=True)

        results = {}
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {
                pool.submit(self._score_one, m, reference, output_dir): m
                for m in models
            }
            for fut in as_completed(futs):
                m = futs[fut]
                try:
                    sr = fut.result()
                except Exception as e:
                    sr = ScoringResult(
                        tool=m.tool, model=m.model,
                        model_stage=m.model_stage, model_path=m.model_path,
                        openstructure_status="error",
                        ost_metrics={"error": str(e)},
                    )
                results[f"{m.tool}__{m.model}__{m.model_stage}"] = sr

        result_list = list(results.values())
        tsv_path = self._write_tsv(result_list, output_dir)
        return result_list, tsv_path

    def score_from_hub_output(
        self,
        run_root: Path,
        reference: Path,
        output_dir: Optional[Path] = None,
        jobs: int = 4,
    ) -> Tuple[List[ScoringResult], Path]:
        """
        Discover models from a hub pipeline output directory and score them.

        Looks for:
          <run_root>/boltz/**/*.cif  → boltz2 candidate models
          <run_root>/chai/pred.model_idx_*.cif  → chai1 candidate models
          <run_root>/cif_bundle/*.cif  → final models
        """
        models: List[ModelEntry] = []

        # Boltz candidate models
        for f in sorted((run_root / "boltz").glob("**/*.cif")) if (run_root / "boltz").exists() else []:
            models.append(ModelEntry(
                tool="boltz2", model=f.stem,
                model_path=f, model_stage="candidate",
            ))

        # Chai candidate models
        if (run_root / "chai").exists():
            for f in sorted((run_root / "chai").glob("pred.model_idx_*.cif")):
                name = f.stem.replace("pred.", "")
                models.append(ModelEntry(
                    tool="chai1", model=name,
                    model_path=f, model_stage="candidate",
                ))

        # Final models from cif_bundle
        if (run_root / "cif_bundle").exists():
            for tag in ["boltz2_model_0", "chai1_model_0"]:
                p = run_root / "cif_bundle" / f"{tag}.cif"
                if p.exists():
                    tool = tag.split("_")[0]
                    models.append(ModelEntry(
                        tool=tool, model=tag,
                        model_path=p, model_stage="final",
                    ))

        out = output_dir or (run_root / "observed_scores")
        return self.score(models, reference, out, jobs=jobs)

    def fetch_pdb(self, pdb_id: str, output_dir: Path) -> Path:
        """Download a CIF from RCSB PDB."""
        import urllib.request
        pdb_id = pdb_id.upper()
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        out_path = Path(output_dir) / f"{pdb_id}.cif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            urllib.request.urlretrieve(url, str(out_path))
        return out_path

    # ── per-model scoring ────────────────────────────────────────────────────

    def _score_one(
        self, model: ModelEntry, reference: Path, output_dir: Path
    ) -> ScoringResult:
        tag = f"{model.tool}__{model.model}__{model.model_stage}"
        ost_json = output_dir / "openstructure" / f"{tag}.json"
        mp_txt = output_dir / "molprobity" / f"{tag}.molprobity.txt"
        cl_txt = output_dir / "molprobity" / f"{tag}.clashscore.txt"

        ost_metrics = self._run_ost(model.model_path, reference, ost_json, tag)
        mol_metrics = self._run_molprobity(model.model_path, mp_txt, cl_txt)

        ost_status = ost_metrics.pop("_status", "missing")
        return ScoringResult(
            tool=model.tool,
            model=model.model,
            model_stage=model.model_stage,
            model_path=model.model_path,
            openstructure_status=ost_status,
            ost_metrics=ost_metrics,
            molprobity_metrics=mol_metrics,
        )

    def _run_ost(
        self, model_path: Path, reference: Path, out_json: Path, tag: str
    ) -> Dict[str, Any]:
        """Run OST compare-structures and return flattened metrics dict."""
        if not self.is_ost_available():
            return {"_status": "ost_unavailable"}

        ost_bin = str(self.ost_prefix / "bin" / "ost")
        tmp_dir = out_json.parent / f"{out_json.stem}.parts"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        metrics: Dict[str, Any] = {}
        ost_ok = False

        # Base metrics
        base_json = tmp_dir / "base.json"
        try:
            r = subprocess.run(
                [ost_bin, "compare-structures",
                 "-m", str(model_path),
                 "-r", str(reference),
                 *self.OST_SCORE_ARGS,
                 "-o", str(base_json)],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0 and base_json.exists():
                ost_ok = True
                data = json.loads(base_json.read_text())
                if data.get("status") == "SUCCESS":
                    _flatten_numeric("ost", data, metrics)
        except Exception:
            pass

        # DockQ capri
        self._run_ost_sub(
            ost_bin, model_path, reference,
            ["--dockq", "--dockq-capri-peptide"],
            tmp_dir / "dockq_capri.json", metrics,
        )

        # ICS/IPS trimmed
        self._run_ost_sub(ost_bin, model_path, reference, ["--ics-trimmed"],
                          tmp_dir / "ics_trimmed.json", metrics)
        self._run_ost_sub(ost_bin, model_path, reference, ["--ips-trimmed"],
                          tmp_dir / "ips_trimmed.json", metrics)

        # CAD score if available
        cad_exec = self._find_cad_exec()
        if cad_exec:
            self._run_ost_sub(
                ost_bin, model_path, reference,
                ["--cad-score", "--local-cad-score",
                 "--cad-exec", cad_exec, "--residue-number-alignment"],
                tmp_dir / "cad.json", metrics,
            )

        status = "SUCCESS" if ost_ok else "FAILURE"
        out_json.write_text(json.dumps({"status": status, **metrics}))
        metrics["_status"] = status
        return metrics

    def _run_ost_sub(
        self, ost_bin: str, model_path: Path, reference: Path,
        extra_args: List[str], out_json: Path, metrics: Dict[str, Any],
    ) -> None:
        try:
            r = subprocess.run(
                [ost_bin, "compare-structures",
                 "-m", str(model_path), "-r", str(reference),
                 *extra_args, "-o", str(out_json)],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0 and out_json.exists():
                data = json.loads(out_json.read_text())
                if data.get("status") == "SUCCESS":
                    _flatten_numeric("ost", data, metrics)
        except Exception:
            pass

    def _run_molprobity(
        self, model_path: Path, mp_txt: Path, cl_txt: Path
    ) -> Dict[str, Any]:
        """Run MolProbity and parse results."""
        if not self.is_molprobity_available():
            return {}

        setpaths = str(self.molprobity_root / "build" / "setpaths.sh")
        mol_root = str(self.molprobity_root)
        abs_model = str(model_path.resolve())  # absolute so cd doesn't break relative paths

        for out_file, cmd_part in [
            (mp_txt, f"phenix.molprobity '{abs_model}' nqh=False outliers_only=True"),
            (cl_txt, f"phenix.clashscore '{abs_model}'"),
        ]:
            try:
                with out_file.open("w") as fh:
                    subprocess.run(
                        ["bash", "-lc", f"cd '{mol_root}' && source {setpaths} && {cmd_part}"],
                        stdout=fh, stderr=subprocess.STDOUT,
                        timeout=120,
                    )
            except Exception:
                pass

        return _parse_molprobity(mp_txt, cl_txt)

    def _find_cad_exec(self) -> Optional[str]:
        for name in ["voronota-cadscore", "voronota_cadscore"]:
            p = self.ost_prefix / "bin" / name
            if p.exists():
                return str(p)
        return None

    # ── TSV output ─────────────────────────────────────────────────────────

    def _write_tsv(self, results: List[ScoringResult], output_dir: Path) -> Path:
        """Write observed_scores.tsv."""
        rows = []
        for sr in results:
            row: Dict[str, Any] = {
                "tool": sr.tool,
                "model": sr.model,
                "model_stage": sr.model_stage,
                "is_final_model": "yes" if sr.model_stage.startswith("final") else "no",
                "model_path": str(sr.model_path),
                "openstructure_status": sr.openstructure_status,
            }
            row.update(sr.ost_metrics)
            row.update(sr.molprobity_metrics)
            rows.append(row)

        # Collect all column names
        front = [
            "tool", "model", "model_stage", "is_final_model", "model_path",
            "openstructure_status", "phenix_clashscore",
            "molprobity_clashscore", "molprobity_all_atom_clashscore",
            "molprobity_score",
        ]
        all_cols: set = set()
        for r in rows:
            all_cols.update(r.keys())
        metric_cols = sorted(c for c in all_cols if c not in front)
        cols = [c for c in front if c in all_cols] + metric_cols

        tsv_path = output_dir / "observed_scores.tsv"
        with tsv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)

        return tsv_path

    def results_to_dataframe(self, results: List[ScoringResult]) -> pd.DataFrame:
        """Convert scoring results to a DataFrame for display."""
        rows = []
        for sr in results:
            row: Dict[str, Any] = {
                "Tool": sr.tool,
                "Model": sr.model,
                "Stage": sr.model_stage,
                "OST Status": sr.openstructure_status,
            }
            # Key metrics to show first
            key_metrics = [
                ("lDDT", "ost.lddt"),
                ("BB-lDDT", "ost.bb_lddt"),
                ("TM-score", "ost.tm_score"),
                ("QS-global", "ost.qs_global"),
                ("QS-best", "ost.qs_best"),
                ("DockQ (avg)", "ost.dockq_ave"),
                ("ICS", "ost.ics"),
                ("IPS", "ost.ips"),
                ("CAD-score", "ost.cad_score"),
                ("GDT-TS", "ost.oligo_gdtts"),
                ("GDT-HA", "ost.oligo_gdtha"),
            ]
            for label, key in key_metrics:
                v = sr.ost_metrics.get(key)
                row[label] = f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "")

            # MolProbity metrics
            for label, key in [
                ("MP Score", "molprobity_score"),
                ("MP Clashscore", "molprobity_clashscore"),
                ("Rama outliers %", "molprobity_rama_outliers_pct"),
                ("Rama favored %", "molprobity_rama_favored_pct"),
                ("Rotamer outliers %", "molprobity_rotamer_outliers_pct"),
            ]:
                v = sr.molprobity_metrics.get(key)
                row[label] = f"{v:.3f}" if isinstance(v, float) else ""
            rows.append(row)

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (
        isinstance(x, float) and math.isnan(x)
    )


def _collect_numeric(value: Any, out: list) -> None:
    if isinstance(value, dict):
        for v in value.values():
            _collect_numeric(v, out)
    elif isinstance(value, list):
        for v in value:
            _collect_numeric(v, out)
    elif _is_number(value):
        out.append(float(value))


def _flatten_numeric(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        if len(value) > 200:
            nums: list = []
            _collect_numeric(value, nums)
            out[f"{prefix}.count"] = float(len(value))
            if nums:
                out[f"{prefix}.mean"] = float(statistics.fmean(nums))
                out[f"{prefix}.min"] = float(min(nums))
                out[f"{prefix}.max"] = float(max(nums))
            return
        for k, v in value.items():
            if k in ("status", "error"):
                continue
            p = f"{prefix}.{k}" if prefix else str(k)
            _flatten_numeric(p, v, out)
        return
    if isinstance(value, list):
        nums2 = [float(v) for v in value if _is_number(v)]
        out[f"{prefix}.count"] = float(len(value))
        if nums2:
            out[f"{prefix}.mean"] = float(statistics.fmean(nums2))
            out[f"{prefix}.min"] = float(min(nums2))
            out[f"{prefix}.max"] = float(max(nums2))
            if len(nums2) <= 20:
                for i, v in enumerate(nums2):
                    out[f"{prefix}[{i}]"] = float(v)
        return
    if _is_number(value):
        out[prefix] = float(value)


def _parse_molprobity(txt_path: Path, clash_path: Path) -> Dict[str, Any]:
    """Parse MolProbity text output files."""
    metrics: Dict[str, Any] = {}

    def extract(pattern: str, text: str, key: str) -> None:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            metrics[key] = float(m.group(1))

    txt = txt_path.read_text(errors="ignore") if txt_path.exists() else ""
    clash_txt = clash_path.read_text(errors="ignore") if clash_path.exists() else ""

    extract(r"MolProbity\s+score\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt, "molprobity_score")
    extract(r"Clashscore\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt, "molprobity_clashscore")
    extract(r"All-atom\s+Clashscore\s*:\s*([0-9]+(?:\.[0-9]+)?)", txt, "molprobity_all_atom_clashscore")
    extract(r"Ramachandran\s+outliers\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt, "molprobity_rama_outliers_pct")
    extract(r"favored\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt, "molprobity_rama_favored_pct")
    extract(r"Rotamer\s+outliers\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt, "molprobity_rotamer_outliers_pct")
    extract(r"RMS\(bonds\)\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt, "molprobity_rms_bonds")
    extract(r"RMS\(angles\)\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt, "molprobity_rms_angles")
    extract(r"clashscore\s*=\s*([0-9]+(?:\.[0-9]+)?)", clash_txt, "phenix_clashscore")

    return metrics
