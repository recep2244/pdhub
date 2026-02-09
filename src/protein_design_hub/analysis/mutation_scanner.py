"""Mutation Scanner Module for rapid single-sequence mutation analysis.

This module provides functionality for:
1. Single-point saturation mutagenesis using structure predictors
2. Multi-point mutation scanning
3. Metric calculation (pLDDT, RMSD, Clash, SASA, TM-score)
4. Mutation ranking and beneficial variant recommendation
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np

from protein_design_hub.core.types import PredictionInput, Sequence

# Standard amino acids (single letter codes)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Amino acid properties for analysis
AA_PROPERTIES = {
    'A': {'name': 'Alanine', 'hydrophobicity': 1.8, 'charge': 0, 'size': 'small', 'polar': False},
    'C': {'name': 'Cysteine', 'hydrophobicity': 2.5, 'charge': 0, 'size': 'small', 'polar': True},
    'D': {'name': 'Aspartate', 'hydrophobicity': -3.5, 'charge': -1, 'size': 'small', 'polar': True},
    'E': {'name': 'Glutamate', 'hydrophobicity': -3.5, 'charge': -1, 'size': 'medium', 'polar': True},
    'F': {'name': 'Phenylalanine', 'hydrophobicity': 2.8, 'charge': 0, 'size': 'large', 'polar': False},
    'G': {'name': 'Glycine', 'hydrophobicity': -0.4, 'charge': 0, 'size': 'small', 'polar': False},
    'H': {'name': 'Histidine', 'hydrophobicity': -3.2, 'charge': 0.1, 'size': 'medium', 'polar': True},
    'I': {'name': 'Isoleucine', 'hydrophobicity': 4.5, 'charge': 0, 'size': 'medium', 'polar': False},
    'K': {'name': 'Lysine', 'hydrophobicity': -3.9, 'charge': 1, 'size': 'large', 'polar': True},
    'L': {'name': 'Leucine', 'hydrophobicity': 3.8, 'charge': 0, 'size': 'medium', 'polar': False},
    'M': {'name': 'Methionine', 'hydrophobicity': 1.9, 'charge': 0, 'size': 'large', 'polar': False},
    'N': {'name': 'Asparagine', 'hydrophobicity': -3.5, 'charge': 0, 'size': 'small', 'polar': True},
    'P': {'name': 'Proline', 'hydrophobicity': -1.6, 'charge': 0, 'size': 'small', 'polar': False},
    'Q': {'name': 'Glutamine', 'hydrophobicity': -3.5, 'charge': 0, 'size': 'medium', 'polar': True},
    'R': {'name': 'Arginine', 'hydrophobicity': -4.5, 'charge': 1, 'size': 'large', 'polar': True},
    'S': {'name': 'Serine', 'hydrophobicity': -0.8, 'charge': 0, 'size': 'small', 'polar': True},
    'T': {'name': 'Threonine', 'hydrophobicity': -0.7, 'charge': 0, 'size': 'small', 'polar': True},
    'V': {'name': 'Valine', 'hydrophobicity': 4.2, 'charge': 0, 'size': 'small', 'polar': False},
    'W': {'name': 'Tryptophan', 'hydrophobicity': -0.9, 'charge': 0, 'size': 'large', 'polar': False},
    'Y': {'name': 'Tyrosine', 'hydrophobicity': -1.3, 'charge': 0, 'size': 'large', 'polar': True},
}


@dataclass
class MutationResult:
    """Result of a single mutation prediction."""
    
    position: int  # 1-indexed position
    original_aa: str
    mutant_aa: str
    mutation_code: str  # e.g., "A42G"
    
    # pLDDT metrics
    mean_plddt: float
    plddt_per_residue: List[float]
    local_plddt: float  # pLDDT at the mutation site
    
    # Delta metrics (compared to wild-type/base)
    delta_mean_plddt: float = 0.0
    delta_local_plddt: float = 0.0
    
    # Additional metrics
    rmsd_to_base: Optional[float] = None
    tm_score_to_base: Optional[float] = None
    clash_score: Optional[float] = None
    sasa_total: Optional[float] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Structure path
    structure_path: Optional[Path] = None
    
    # Prediction metadata
    prediction_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def is_beneficial(self) -> bool:
        """Check if mutation improves pLDDT."""
        return self.delta_mean_plddt > 0 or self.delta_local_plddt > 0
    
    @property
    def improvement_score(self) -> float:
        """Calculate overall improvement score."""
        # Weighted combination of delta metrics
        # Could add penalties for bad RMSD or clashes here
        return 0.6 * self.delta_mean_plddt + 0.4 * self.delta_local_plddt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position": self.position,
            "original_aa": self.original_aa,
            "mutant_aa": self.mutant_aa,
            "mutation_code": self.mutation_code,
            "mean_plddt": self.mean_plddt,
            "local_plddt": self.local_plddt,
            "delta_mean_plddt": self.delta_mean_plddt,
            "delta_local_plddt": self.delta_local_plddt,
            "rmsd_to_base": self.rmsd_to_base,
            "tm_score_to_base": self.tm_score_to_base,
            "clash_score": self.clash_score,
            "sasa_total": self.sasa_total,
            "extra_metrics": self.extra_metrics,
            "is_beneficial": self.is_beneficial,
            "improvement_score": self.improvement_score,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class MultiMutationVariant:
    """Result of a multi-point mutation prediction."""

    positions: List[int]
    original_aas: List[str]
    mutant_aas: List[str]
    mutation_code: str  # e.g., "A42G+V50I"

    mean_plddt: float
    plddt_per_residue: List[float]
    local_plddt_mean: float
    local_plddt_min: float

    delta_mean_plddt: float = 0.0
    delta_local_plddt: float = 0.0

    rmsd_to_base: Optional[float] = None
    tm_score_to_base: Optional[float] = None
    clash_score: Optional[float] = None
    sasa_total: Optional[float] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    structure_path: Optional[Path] = None
    prediction_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    @property
    def improvement_score(self) -> float:
        return 0.6 * self.delta_mean_plddt + 0.4 * self.delta_local_plddt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": self.positions,
            "original_aas": self.original_aas,
            "mutant_aas": self.mutant_aas,
            "mutation_code": self.mutation_code,
            "mean_plddt": self.mean_plddt,
            "local_plddt_mean": self.local_plddt_mean,
            "local_plddt_min": self.local_plddt_min,
            "delta_mean_plddt": self.delta_mean_plddt,
            "delta_local_plddt": self.delta_local_plddt,
            "rmsd_to_base": self.rmsd_to_base,
            "tm_score_to_base": self.tm_score_to_base,
            "clash_score": self.clash_score,
            "sasa_total": self.sasa_total,
            "extra_metrics": self.extra_metrics,
            "structure_path": str(self.structure_path) if self.structure_path else None,
            "prediction_time": self.prediction_time,
            "success": self.success,
            "error_message": self.error_message,
            "improvement_score": self.improvement_score,
        }


@dataclass
class MultiMutationResult:
    """Result of multi-position mutation design pipeline."""

    predictor: str
    positions: List[int]
    original_aas: List[str]
    sequence: str

    base_mean_plddt: float
    base_local_plddt_mean: float
    base_local_plddt_min: float
    base_plddt_per_residue: List[float]
    base_structure_path: Optional[Path] = None
    base_clash_score: Optional[float] = None
    base_sasa_total: Optional[float] = None
    base_extra_metrics: Dict[str, Any] = field(default_factory=dict)

    variants: List[MultiMutationVariant] = field(default_factory=list)
    single_position_scans: List[SaturationMutagenesisResult] = field(default_factory=list)

    total_time: float = 0.0
    timestamp: str = ""

    @property
    def ranked_variants(self) -> List[MultiMutationVariant]:
        return sorted(
            [v for v in self.variants if v.success],
            key=lambda v: v.improvement_score,
            reverse=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictor": self.predictor,
            "positions": self.positions,
            "original_aas": self.original_aas,
            "sequence": self.sequence,
            "base_mean_plddt": self.base_mean_plddt,
            "base_local_plddt_mean": self.base_local_plddt_mean,
            "base_local_plddt_min": self.base_local_plddt_min,
            "base_clash_score": self.base_clash_score,
            "base_sasa_total": self.base_sasa_total,
            "base_extra_metrics": self.base_extra_metrics,
            "base_structure_path": str(self.base_structure_path) if self.base_structure_path else None,
            "variants": [v.to_dict() for v in self.variants],
            "single_position_scans": [s.to_dict() for s in self.single_position_scans],
            "total_time": self.total_time,
            "timestamp": self.timestamp,
        }


@dataclass
class SaturationMutagenesisResult:
    """Result of saturation mutagenesis at a single position."""

    predictor: str
    position: int
    original_aa: str
    sequence: str
    
    # Base structure metrics
    base_mean_plddt: float
    base_local_plddt: float
    base_plddt_per_residue: List[float]
    base_structure_path: Optional[Path] = None
    base_clash_score: Optional[float] = None
    base_sasa_total: Optional[float] = None
    base_extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Mutation results
    mutations: List[MutationResult] = field(default_factory=list)
    
    # Analysis metadata
    total_time: float = 0.0
    timestamp: str = ""
    
    @property
    def ranked_mutations(self) -> List[MutationResult]:
        """Get mutations ranked by improvement score."""
        return sorted(
            [m for m in self.mutations if m.success],
            key=lambda x: x.improvement_score,
            reverse=True
        )
    
    @property
    def beneficial_mutations(self) -> List[MutationResult]:
        """Get only beneficial mutations."""
        return [m for m in self.ranked_mutations if m.is_beneficial]
    
    @property
    def best_mutation(self) -> Optional[MutationResult]:
        """Get the best mutation recommendation."""
        ranked = self.ranked_mutations
        return ranked[0] if ranked else None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "predictor": self.predictor,
            "position": self.position,
            "original_aa": self.original_aa,
            "sequence": self.sequence,
            "base_mean_plddt": self.base_mean_plddt,
            "base_local_plddt": self.base_local_plddt,
            "base_clash_score": self.base_clash_score,
            "base_sasa_total": self.base_sasa_total,
            "base_extra_metrics": self.base_extra_metrics,
            "mutations": [m.to_dict() for m in self.mutations],
            "best_mutation": self.best_mutation.to_dict() if self.best_mutation else None,
            "beneficial_count": len(self.beneficial_mutations),
            "total_time": self.total_time,
            "timestamp": self.timestamp,
        }


class MutationScanner:
    """
    Scanner for rapid mutation analysis using ESMFold.
    """
    
    def __init__(
        self,
        predictor: Optional[str] = None,
        use_api: bool = True,
        esmfold_version: str = "v1",
        immunebuilder_mode: str = "antibody",
        immune_chain_a: Optional[str] = None,
        immune_chain_b: Optional[str] = None,
        immune_active_chain: str = "A",
        max_workers: int = 4,
        output_dir: Optional[Path] = None,
        evaluation_metrics: Optional[List[str]] = None,
        run_openstructure_comprehensive: bool = False,
    ):
        if predictor is None:
            predictor = "esmfold_api" if use_api else "esmfold_local"
        self.predictor = predictor
        self.use_api = use_api
        self.esmfold_version = esmfold_version
        self.immunebuilder_mode = immunebuilder_mode
        self.immune_chain_a = immune_chain_a
        self.immune_chain_b = immune_chain_b
        self.immune_active_chain = immune_active_chain
        self.max_workers = max_workers
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="mutation_scan_"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_metrics = evaluation_metrics or []
        self.run_openstructure_comprehensive = run_openstructure_comprehensive
        self._base_cache: Dict[Tuple[Any, ...], Tuple[str, List[float], Path]] = {}

        # Import metrics lazily
        try:
            from protein_design_hub.evaluation.metrics.clash_score import ClashScoreMetric
            from protein_design_hub.evaluation.metrics.sasa import SASAMetric
            from protein_design_hub.evaluation.metrics.rmsd import RMSDMetric
            from protein_design_hub.evaluation.metrics.tm_score import TMScoreMetric
            
            self._clash_metric = ClashScoreMetric()
            self._sasa_metric = SASAMetric()
            self._rmsd_metric = RMSDMetric(atoms="CA")
            self._tm_metric = TMScoreMetric()
            self._metrics_available = True
        except ImportError:
            self._metrics_available = False
            print("Warning: BioPython or other dependencies missing. Some metrics available.")
    
    def _predict_esmfold_api(self, sequence: str, output_path: Path) -> Tuple[str, List[float]]:
        import requests
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        sequence = "".join(c for c in sequence.upper().strip() if c in valid_aa)
        
        if len(sequence) > 400:
            raise ValueError(f"Sequence too long for API ({len(sequence)} > 400)")
        
        response = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            data=sequence,
            headers={"Content-Type": "text/plain"},
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(f"ESMFold API error: {response.status_code}")
        
        pdb_text = response.text
        output_path.write_text(pdb_text)
        plddt_values = self._extract_plddt_from_pdb(pdb_text)
        if plddt_values and max(plddt_values) <= 1.0:
            plddt_values = [v * 100.0 for v in plddt_values]
        return pdb_text, plddt_values
    
    def _predict_esmfold_local(
        self,
        sequence: str,
        output_path: Path,
        version: str = "v1",
    ) -> Tuple[str, List[float]]:
        try:
            import torch
            import esm
        except ImportError:
            raise ImportError("ESMFold local requires `fair-esm` and `torch`")
        if not hasattr(esm, "pretrained") or not hasattr(esm.pretrained, "esmfold_v1"):
            raise ImportError(
                "Detected a non-ESMFold `esm` package. The EvolutionaryScale ESM3 SDK uses the same import name "
                "and conflicts with `fair-esm`. Use a separate environment for ESM3 or reinstall `fair-esm`."
            )
        
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        sequence = "".join(c for c in sequence.upper().strip() if c in valid_aa)
        
        try:
            if version == "v0":
                if not hasattr(esm.pretrained, "esmfold_v0"):
                    raise ImportError("ESMFold v0 not available in installed `fair-esm`")
                model = esm.pretrained.esmfold_v0().eval()
            else:
                model = esm.pretrained.esmfold_v1().eval()
        except Exception as e:
            if "Keys" in str(e) and "missing" in str(e) or "use_esm_attn_map" in str(e):
                from protein_design_hub.predictors.esmfold.utils import load_esmfold_model
                model = load_esmfold_model(version=version, allow_missing=True).eval()
            else:
                raise
        if torch.cuda.is_available():
            model = model.cuda()
            model.set_chunk_size(128)
        
        with torch.no_grad():
            pdb_text = model.infer_pdb(sequence)
        
        output_path.write_text(pdb_text)
        plddt_values = self._extract_plddt_from_pdb(pdb_text)
        return pdb_text, plddt_values
    
    def _extract_plddt_from_pdb(self, pdb_text: str) -> List[float]:
        plddt_values = []
        seen_residues = set()
        for line in pdb_text.splitlines():
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
            try:
                residue_id = (line[21], int(line[22:26]))
            except: continue
            if residue_id in seen_residues: continue
            seen_residues.add(residue_id)
            try:
                plddt_values.append(float(line[60:66]))
            except: continue
        return plddt_values

    def _delta_score(self, mutated: float, base: float) -> float:
        if self.predictor == "immunebuilder":
            # ImmuneBuilder reports per-residue error (lower is better).
            return base - mutated
        return mutated - base

    def _base_cache_key(self, sequence: str) -> Tuple[Any, ...]:
        if self.predictor == "immunebuilder":
            return (
                self.predictor,
                sequence,
                self.immunebuilder_mode,
                self.immune_active_chain,
                self.immune_chain_a,
                self.immune_chain_b,
            )
        return (self.predictor, sequence)

    def _predict_esm3(self, sequence: str, output_path: Path) -> Tuple[str, List[float]]:
        from protein_design_hub.predictors.esm3.predictor import ESM3Predictor

        predictor = ESM3Predictor()
        input_data = PredictionInput(
            job_id=output_path.stem,
            sequences=[Sequence(id="query", sequence=sequence)],
            num_models=1,
            num_recycles=1,
            output_dir=output_path.parent,
        )
        result = predictor.predict(input_data)
        if not result.success or not result.structure_paths:
            raise RuntimeError(result.error_message or "ESM3 prediction failed")

        structure_path = result.structure_paths[0]
        if structure_path != output_path:
            output_path.write_text(structure_path.read_text())

        pdb_text = output_path.read_text()
        plddt_values = None
        if result.scores and result.scores[0].plddt_per_residue:
            plddt_values = result.scores[0].plddt_per_residue
        if not plddt_values:
            plddt_values = self._extract_plddt_from_pdb(pdb_text)
        return pdb_text, plddt_values

    def _predict_immunebuilder(self, sequence: str, output_path: Path) -> Tuple[str, List[float]]:
        import json
        import os
        import subprocess

        if not self.immune_chain_a:
            raise ValueError("ImmuneBuilder requires chain A sequence")
        if self.immunebuilder_mode in {"antibody", "tcr"} and not self.immune_chain_b:
            raise ValueError("ImmuneBuilder requires chain B sequence for this mode")

        if self.immune_active_chain not in {"A", "B"}:
            raise ValueError("ImmuneBuilder active chain must be 'A' or 'B'")

        chain_a = sequence if self.immune_active_chain == "A" else self.immune_chain_a
        chain_b = sequence if self.immune_active_chain == "B" else self.immune_chain_b

        if self.immunebuilder_mode == "nanobody" and not chain_a:
            raise ValueError("ImmuneBuilder nanobody mode requires chain A")
        if self.immunebuilder_mode == "nanobody" and self.immune_active_chain != "A":
            raise ValueError("ImmuneBuilder nanobody mode only supports chain A")

        # Map A/B to ImmuneBuilder chain ids in runner.
        active_chain_id = None
        if self.immunebuilder_mode in {"antibody", "nanobody"}:
            active_chain_id = "H" if self.immune_active_chain == "A" else "L"
        else:
            active_chain_id = self.immune_active_chain

        runner = Path(__file__).resolve().parents[3] / "scripts" / "run_immunebuilder.py"
        if not runner.exists():
            raise FileNotFoundError(f"ImmuneBuilder runner not found: {runner}")

        immu_python = os.getenv("IMMUNEBUILDER_PYTHON")
        if not immu_python:
            raise RuntimeError(
                "IMMUNEBUILDER_PYTHON is not set. Create a separate env with ImmuneBuilder and "
                "set IMMUNEBUILDER_PYTHON=/path/to/python."
            )

        payload = {
            "mode": self.immunebuilder_mode,
            "active_chain_id": active_chain_id,
            "chain_a": chain_a,
            "chain_b": chain_b,
        }
        input_json = output_path.with_suffix(".immunebuilder.json")
        output_json = output_path.with_suffix(".immunebuilder.out.json")
        input_json.write_text(json.dumps(payload))

        result = subprocess.run(
            [immu_python, str(runner), "--input_json", str(input_json),
             "--output_pdb", str(output_path), "--output_json", str(output_json)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ImmuneBuilder failed: {result.stderr.strip() or result.stdout.strip()}"
            )

        data = json.loads(output_json.read_text())
        per_residue_error = data.get("per_residue_error") or []
        if not per_residue_error:
            raise RuntimeError("ImmuneBuilder did not return per-residue error estimates")

        pdb_text = output_path.read_text()
        # Use error values as pLDDT surrogate (lower is better).
        return pdb_text, per_residue_error

    def calculate_biophysical_metrics(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        evaluation_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Calculate secondary metrics."""
        results = {}
        if self._metrics_available:
            # 1. Clash Score
            try:
                clash_res = self._clash_metric.compute(model_path)
                results['clash_score'] = clash_res.get('clash_score')
            except Exception:
                pass

            # 2. SASA
            try:
                sasa_res = self._sasa_metric.compute(model_path)
                results['sasa_total'] = sasa_res.get('sasa_total')
            except Exception:
                pass

            if reference_path:
                # 3. RMSD
                try:
                    rmsd_res = self._rmsd_metric.compute(model_path, reference_path)
                    results['rmsd'] = rmsd_res.get('rmsd')
                except Exception:
                    pass

                # 4. TM-score
                try:
                    tm_res = self._tm_metric.compute(model_path, reference_path)
                    results['tm_score'] = tm_res.get('tm_score')
                except Exception:
                    pass

        # Additional evaluation metrics via CompositeEvaluator
        extra_metrics = {}
        selected = evaluation_metrics if evaluation_metrics is not None else self.evaluation_metrics
        if selected:
            try:
                from protein_design_hub.evaluation.composite import CompositeEvaluator

                base_metric_names = {"clash_score", "sasa", "tm_score", "rmsd"}
                metrics_to_run = [m for m in selected if m not in base_metric_names]
                if metrics_to_run:
                    evaluator = CompositeEvaluator(metrics=metrics_to_run)
                    eval_result = evaluator.evaluate(model_path, reference_path)
                    extra_metrics = eval_result.metadata or {}
            except Exception as exc:
                extra_metrics = {"errors": [f"extra_metrics: {exc}"]}

        # Optional full OpenStructure comparison against baseline reference.
        # This is intentionally expensive and best used for shortlisted mutants.
        if reference_path and self.run_openstructure_comprehensive:
            try:
                from protein_design_hub.evaluation.composite import CompositeEvaluator

                ost_evaluator = CompositeEvaluator(metrics=["tm_score"])
                ost_result = ost_evaluator.evaluate_comprehensive(model_path, reference_path)
                extra_metrics["ost_comprehensive"] = ost_result
            except Exception as exc:
                errs = extra_metrics.get("errors", [])
                if not isinstance(errs, list):
                    errs = [str(errs)]
                errs.append(f"ost_comprehensive: {exc}")
                extra_metrics["errors"] = errs

        if extra_metrics:
            results["extra_metrics"] = extra_metrics
            
        return results

    def predict_single(self, sequence: str, name: str = "protein") -> Tuple[str, List[float], Path]:
        output_path = self.output_dir / f"{name}.pdb"
        predictor = self.predictor
        if predictor == "esmfold_api":
            pdb_text, plddt_values = self._predict_esmfold_api(sequence, output_path)
        elif predictor in ("esmfold_local", "esmfold_v1"):
            pdb_text, plddt_values = self._predict_esmfold_local(
                sequence,
                output_path,
                version=self.esmfold_version if predictor == "esmfold_local" else "v1",
            )
        elif predictor == "esmfold_v0":
            pdb_text, plddt_values = self._predict_esmfold_local(sequence, output_path, version="v0")
        elif predictor == "esm3":
            pdb_text, plddt_values = self._predict_esm3(sequence, output_path)
        elif predictor == "immunebuilder":
            pdb_text, plddt_values = self._predict_immunebuilder(sequence, output_path)
        else:
            raise ValueError(f"Unknown predictor: {predictor}")
        return pdb_text, plddt_values, output_path

    def scan_position(
        self,
        sequence: str,
        position: int,
        progress_callback: Optional[callable] = None,
    ) -> SaturationMutagenesisResult:
        import datetime
        start_time = time.time()
        
        if position < 1 or position > len(sequence):
            raise ValueError(f"Position {position} out of range")
        
        original_aa = sequence[position - 1]
        
        # 1. Base Structure
        base_key = self._base_cache_key(sequence)
        if base_key in self._base_cache:
            base_pdb, base_plddt, base_path = self._base_cache[base_key]
        else:
            base_pdb, base_plddt, base_path = self.predict_single(sequence, "base_wt")
            self._base_cache[base_key] = (base_pdb, base_plddt, base_path)
            
        base_mean_plddt = sum(base_plddt) / len(base_plddt) if base_plddt else 0
        base_local_plddt = base_plddt[position - 1] if position <= len(base_plddt) else 0
        
        # Base metrics
        base_metrics = self.calculate_biophysical_metrics(
            base_path,
            evaluation_metrics=self.evaluation_metrics,
        )
        
        if progress_callback:
            progress_callback(0, 20, f"Base (WT) calculated")
            
        # 2. Mutagenesis
        mutations_to_test = [aa for aa in AMINO_ACIDS if aa != original_aa]
        mutation_results = []
        
        for idx, mutant_aa in enumerate(mutations_to_test):
            mutation_code = f"{original_aa}{position}{mutant_aa}"
            if progress_callback:
                progress_callback(idx + 1, len(mutations_to_test) + 1, mutation_code)
                
            mutant_seq = sequence[:position - 1] + mutant_aa + sequence[position:]
            try:
                mut_start = time.time()
                mut_pdb, mut_plddt, mut_path = self.predict_single(mutant_seq, f"mut_{mutation_code}")
                
                mut_mean_plddt = sum(mut_plddt) / len(mut_plddt) if mut_plddt else 0
                mut_local_plddt = mut_plddt[position - 1] if position <= len(mut_plddt) else 0
                
                # Biophsyical metrics
                metrics = self.calculate_biophysical_metrics(
                    mut_path,
                    base_path,
                    evaluation_metrics=self.evaluation_metrics,
                )
                
                result = MutationResult(
                    position=position,
                    original_aa=original_aa,
                    mutant_aa=mutant_aa,
                    mutation_code=mutation_code,
                    mean_plddt=mut_mean_plddt,
                    plddt_per_residue=mut_plddt,
                    local_plddt=mut_local_plddt,
                    delta_mean_plddt=self._delta_score(mut_mean_plddt, base_mean_plddt),
                    delta_local_plddt=self._delta_score(mut_local_plddt, base_local_plddt),
                    rmsd_to_base=metrics.get('rmsd'),
                    tm_score_to_base=metrics.get('tm_score'),
                    clash_score=metrics.get('clash_score'),
                    sasa_total=metrics.get('sasa_total'),
                    extra_metrics=metrics.get("extra_metrics", {}),
                    structure_path=mut_path,
                    prediction_time=time.time() - mut_start,
                    success=True,
                )
            except Exception as e:
                result = MutationResult(
                    position=position,
                    original_aa=original_aa,
                    mutant_aa=mutant_aa,
                    mutation_code=mutation_code,
                    mean_plddt=0,
                    plddt_per_residue=[],
                    local_plddt=0,
                    success=False,
                    error_message=str(e),
                )
            mutation_results.append(result)
            
        return SaturationMutagenesisResult(
            predictor=self.predictor,
            position=position,
            original_aa=original_aa,
            sequence=sequence,
            base_mean_plddt=base_mean_plddt,
            base_local_plddt=base_local_plddt,
            base_plddt_per_residue=base_plddt,
            base_structure_path=base_path,
            base_clash_score=base_metrics.get('clash_score'),
            base_sasa_total=base_metrics.get('sasa_total'),
            base_extra_metrics=base_metrics.get("extra_metrics", {}),
            mutations=mutation_results,
            total_time=time.time() - start_time,
            timestamp=datetime.datetime.now().isoformat(),
        )

    def scan_positions(
        self,
        sequence: str,
        positions: List[int],
        top_k: int = 3,
        max_variants: int = 25,
        max_positions: int = 6,
        only_beneficial: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> MultiMutationResult:
        import datetime
        start_time = time.time()

        if not positions:
            raise ValueError("No positions provided")

        positions = sorted(set(positions))
        if max_positions is not None and len(positions) > max_positions:
            raise ValueError(f"Too many positions selected ({len(positions)} > {max_positions}).")
        if any(p < 1 or p > len(sequence) for p in positions):
            raise ValueError("One or more positions are out of range")

        # Base structure (cached)
        base_key = self._base_cache_key(sequence)
        if base_key in self._base_cache:
            base_pdb, base_plddt, base_path = self._base_cache[base_key]
        else:
            base_pdb, base_plddt, base_path = self.predict_single(sequence, "base_wt")
            self._base_cache[base_key] = (base_pdb, base_plddt, base_path)

        base_mean_plddt = sum(base_plddt) / len(base_plddt) if base_plddt else 0.0
        base_local_values = [base_plddt[p - 1] if p <= len(base_plddt) else 0 for p in positions]
        base_local_mean = sum(base_local_values) / len(base_local_values) if base_local_values else 0.0
        base_local_min = min(base_local_values) if base_local_values else 0.0

        base_metrics = self.calculate_biophysical_metrics(
            base_path,
            evaluation_metrics=self.evaluation_metrics,
        )

        # Run single-position scans
        scans: List[SaturationMutagenesisResult] = []
        for idx, pos in enumerate(positions):
            if progress_callback:
                progress_callback("scan", idx + 1, len(positions), f"Scanning position {pos}")
            scans.append(self.scan_position(sequence, pos))

        # Build candidate list per position
        candidate_lists: List[List[MutationResult]] = []
        for res in scans:
            candidates = res.ranked_mutations
            if only_beneficial:
                candidates = [m for m in candidates if m.is_beneficial]
            if not candidates:
                candidates = res.ranked_mutations
            candidate_lists.append(candidates[:max(1, top_k)])

        # Beam search to limit combinations
        beam: List[Tuple[float, List[MutationResult]]] = [(0.0, [])]
        for candidates in candidate_lists:
            new_beam: List[Tuple[float, List[MutationResult]]] = []
            for score, muts in beam:
                for m in candidates:
                    new_beam.append((score + m.improvement_score, muts + [m]))
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:max_variants]

        variants: List[MultiMutationVariant] = []
        for idx, (_, muts) in enumerate(beam):
            mutation_code = "+".join(m.mutation_code for m in muts)
            if progress_callback:
                progress_callback("variant", idx + 1, len(beam), mutation_code)

            mutant_seq = list(sequence)
            for m in muts:
                mutant_seq[m.position - 1] = m.mutant_aa
            mutant_seq = "".join(mutant_seq)

            try:
                mut_start = time.time()
                mut_pdb, mut_plddt, mut_path = self.predict_single(mutant_seq, f"multi_{idx+1}")
                mut_mean_plddt = sum(mut_plddt) / len(mut_plddt) if mut_plddt else 0.0
                local_vals = [mut_plddt[p - 1] if p <= len(mut_plddt) else 0 for p in positions]
                mut_local_mean = sum(local_vals) / len(local_vals) if local_vals else 0.0
                mut_local_min = min(local_vals) if local_vals else 0.0

                metrics = self.calculate_biophysical_metrics(
                    mut_path,
                    base_path,
                    evaluation_metrics=self.evaluation_metrics,
                )

                variants.append(MultiMutationVariant(
                    positions=positions,
                    original_aas=[sequence[p - 1] for p in positions],
                    mutant_aas=[m.mutant_aa for m in muts],
                    mutation_code=mutation_code,
                    mean_plddt=mut_mean_plddt,
                    plddt_per_residue=mut_plddt,
                    local_plddt_mean=mut_local_mean,
                    local_plddt_min=mut_local_min,
                    delta_mean_plddt=self._delta_score(mut_mean_plddt, base_mean_plddt),
                    delta_local_plddt=self._delta_score(mut_local_mean, base_local_mean),
                    rmsd_to_base=metrics.get('rmsd'),
                    tm_score_to_base=metrics.get('tm_score'),
                    clash_score=metrics.get('clash_score'),
                    sasa_total=metrics.get('sasa_total'),
                    extra_metrics=metrics.get("extra_metrics", {}),
                    structure_path=mut_path,
                    prediction_time=time.time() - mut_start,
                    success=True,
                ))
            except Exception as e:
                variants.append(MultiMutationVariant(
                    positions=positions,
                    original_aas=[sequence[p - 1] for p in positions],
                    mutant_aas=[m.mutant_aa for m in muts],
                    mutation_code=mutation_code,
                    mean_plddt=0.0,
                    plddt_per_residue=[],
                    local_plddt_mean=0.0,
                    local_plddt_min=0.0,
                    success=False,
                    error_message=str(e),
                ))

        return MultiMutationResult(
            predictor=self.predictor,
            positions=positions,
            original_aas=[sequence[p - 1] for p in positions],
            sequence=sequence,
            base_mean_plddt=base_mean_plddt,
            base_local_plddt_mean=base_local_mean,
            base_local_plddt_min=base_local_min,
            base_plddt_per_residue=base_plddt,
            base_structure_path=base_path,
            base_clash_score=base_metrics.get('clash_score'),
            base_sasa_total=base_metrics.get('sasa_total'),
            base_extra_metrics=base_metrics.get("extra_metrics", {}),
            variants=variants,
            single_position_scans=scans,
            total_time=time.time() - start_time,
            timestamp=datetime.datetime.now().isoformat(),
        )

def create_mutation_heatmap(result: SaturationMutagenesisResult) -> Dict[str, Any]:
    """Helper to create heatmap data."""
    # ... existing implementation or updated one ...
    # For now, we can just reuse the one in the web page or this one.
    # The one in web page does its own logic.
    pass
