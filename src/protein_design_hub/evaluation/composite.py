"""Composite evaluator combining multiple metrics."""

from pathlib import Path
from typing import Dict, List, Optional, Any

from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.evaluation.metrics.lddt import LDDTMetric
from protein_design_hub.evaluation.metrics.qs_score import QSScoreMetric
from protein_design_hub.evaluation.metrics.tm_score import TMScoreMetric
from protein_design_hub.evaluation.metrics.rmsd import RMSDMetric
from protein_design_hub.evaluation.metrics.lddt_pli import LDDTPLIMetric
from protein_design_hub.evaluation.metrics.clash_score import ClashScoreMetric
from protein_design_hub.evaluation.metrics.contact_energy import ContactEnergyMetric
from protein_design_hub.evaluation.metrics.rosetta_energy import RosettaEnergyMetric
from protein_design_hub.evaluation.metrics.sasa import SASAMetric
from protein_design_hub.evaluation.metrics.interface_bsa import InterfaceBSAMetric
from protein_design_hub.evaluation.metrics.salt_bridges import SaltBridgeMetric
from protein_design_hub.evaluation.metrics.openmm_gbsa import OpenMMGBSAMetric
from protein_design_hub.evaluation.metrics.rosetta_score_jd2 import RosettaScoreJd2Metric
from protein_design_hub.evaluation.metrics.sequence_recovery import SequenceRecoveryMetric
from protein_design_hub.evaluation.metrics.disorder import DisorderMetric
from protein_design_hub.evaluation.metrics.shape_complementarity import ShapeComplementarityMetric
from protein_design_hub.evaluation.metrics.voronota_cadscore import VoronotaCADScoreMetric
from protein_design_hub.evaluation.metrics.voronota_voromqa import VoronotaVoroMQAMetric
from protein_design_hub.core.types import EvaluationResult
from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.core.exceptions import EvaluationError


class CompositeEvaluator:
    """Evaluator that combines multiple structure quality metrics."""

    AVAILABLE_METRICS = {
        "lddt": LDDTMetric,
        "qs_score": QSScoreMetric,
        "tm_score": TMScoreMetric,
        "rmsd": RMSDMetric,
        "lddt_pli": LDDTPLIMetric,
        "clash_score": ClashScoreMetric,
        "contact_energy": ContactEnergyMetric,
        "rosetta_energy": RosettaEnergyMetric,
        "sasa": SASAMetric,
        "interface_bsa": InterfaceBSAMetric,
        "salt_bridges": SaltBridgeMetric,
        "openmm_gbsa": OpenMMGBSAMetric,
        "rosetta_score_jd2": RosettaScoreJd2Metric,
        "sequence_recovery": SequenceRecoveryMetric,
        "disorder": DisorderMetric,
        "shape_complementarity": ShapeComplementarityMetric,
        "cad_score": VoronotaCADScoreMetric,
        "voromqa": VoronotaVoroMQAMetric,
    }

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize composite evaluator.

        Args:
            metrics: List of metric names to use. Uses all if not specified.
            settings: Configuration settings.
        """
        self.settings = settings or get_settings()

        if metrics is None:
            metrics = self.settings.evaluation.metrics

        self.metrics: Dict[str, BaseMetric] = {}
        for metric_name in metrics:
            metric_name_lower = metric_name.lower().replace("-", "_")
            if metric_name_lower in self.AVAILABLE_METRICS:
                metric_class = self.AVAILABLE_METRICS[metric_name_lower]

                # Pass metric-specific settings
                if metric_name_lower == "lddt":
                    lddt_config = self.settings.evaluation.lddt
                    self.metrics[metric_name_lower] = metric_class(
                        inclusion_radius=lddt_config.inclusion_radius,
                        sequence_separation=lddt_config.sequence_separation,
                    )
                elif metric_name_lower == "tm_score":
                    tm_config = self.settings.evaluation.tm_score
                    self.metrics[metric_name_lower] = metric_class(
                        tmalign_path=tm_config.tmalign_path,
                    )
                else:
                    self.metrics[metric_name_lower] = metric_class()

    def evaluate(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
    ) -> EvaluationResult:
        """
        Evaluate a structure using all configured metrics.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure (required for most metrics).

        Returns:
            EvaluationResult with all computed metrics.
        """
        model_path = Path(model_path)
        if reference_path:
            reference_path = Path(reference_path)

        result = EvaluationResult(
            structure_path=model_path,
            reference_path=reference_path,
        )

        errors = []

        for metric_name, metric in self.metrics.items():
            # Skip metrics that require reference if not provided
            if metric.requires_reference and reference_path is None:
                continue

            if not metric.is_available():
                errors.append(f"{metric_name}: {metric.get_requirements()}")
                continue

            try:
                metric_result = metric.compute(model_path, reference_path)

                # Map results to EvaluationResult fields
                if metric_name == "lddt":
                    result.lddt = metric_result.get("lddt")
                    result.lddt_per_residue = metric_result.get("lddt_per_residue")
                elif metric_name == "qs_score":
                    result.qs_score = metric_result.get("qs_score")
                elif metric_name == "tm_score":
                    result.tm_score = metric_result.get("tm_score")
                    result.gdt_ts = metric_result.get("gdt_ts")
                    result.gdt_ha = metric_result.get("gdt_ha")
                    # Also get RMSD from TMalign if available
                    if result.rmsd is None:
                        result.rmsd = metric_result.get("rmsd")
                elif metric_name == "rmsd":
                    result.rmsd = metric_result.get("rmsd")
                elif metric_name == "clash_score":
                    result.clash_score = metric_result.get("clash_score")
                    result.clash_count = metric_result.get("clash_count")
                elif metric_name == "contact_energy":
                    result.contact_energy = metric_result.get("contact_energy")
                    result.contact_energy_per_residue = metric_result.get(
                        "contact_energy_per_residue"
                    )
                elif metric_name == "rosetta_energy":
                    result.rosetta_total_score = metric_result.get("rosetta_total_score")
                elif metric_name == "sasa":
                    result.sasa_total = metric_result.get("sasa_total")
                elif metric_name == "interface_bsa":
                    result.interface_bsa_total = metric_result.get("interface_bsa_total")
                elif metric_name == "salt_bridges":
                    result.salt_bridge_count = metric_result.get("salt_bridge_count")
                    result.salt_bridge_count_interchain = metric_result.get(
                        "salt_bridge_count_interchain"
                    )
                elif metric_name == "openmm_gbsa":
                    result.openmm_potential_energy_kj_mol = metric_result.get(
                        "openmm_potential_energy_kj_mol"
                    )
                    result.openmm_gbsa_energy_kj_mol = metric_result.get(
                        "openmm_gbsa_energy_kj_mol"
                    )
                elif metric_name == "cad_score":
                    result.cad_score = metric_result.get("cad_score")
                    result.cad_score_per_residue = metric_result.get("cad_score_per_residue")
                elif metric_name == "voromqa":
                    result.voromqa_score = metric_result.get("voromqa_score")
                    result.voromqa_per_residue = metric_result.get("voromqa_per_residue")
                    result.voromqa_residue_count = metric_result.get("voromqa_residue_count")
                    result.voromqa_atom_count = metric_result.get("voromqa_atom_count")
                elif metric_name == "rosetta_score_jd2":
                    # Keep both: generic field + explicit score_jd2 field.
                    result.rosetta_total_score = metric_result.get("rosetta_total_score")
                    result.rosetta_score_jd2_total_score = metric_result.get("rosetta_total_score")
                elif metric_name == "sequence_recovery":
                    result.sequence_recovery = metric_result.get("sequence_recovery")
                    result.sequence_recovery_per_residue = metric_result.get("per_residue_match")
                elif metric_name == "disorder":
                    result.disorder_fraction = metric_result.get("disorder_fraction")
                    result.disorder_per_residue = metric_result.get("per_residue_disorder")
                    result.disorder_regions = metric_result.get("disordered_regions")
                elif metric_name == "shape_complementarity":
                    result.shape_complementarity = metric_result.get("shape_complementarity")
                    result.interface_residues_a = metric_result.get("interface_residues_a")
                    result.interface_residues_b = metric_result.get("interface_residues_b")

                # Store full metric result in metadata
                if "metadata" not in result.__dict__ or result.metadata is None:
                    result.metadata = {}
                result.metadata[metric_name] = metric_result

            except EvaluationError as e:
                errors.append(f"{metric_name}: {str(e)}")
            except Exception as e:
                errors.append(f"{metric_name}: Unexpected error - {str(e)}")

        if errors:
            result.metadata["errors"] = errors

        return result

    def evaluate_batch(
        self,
        model_paths: List[Path],
        reference_path: Optional[Path] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple structures.

        Args:
            model_paths: List of paths to model structures.
            reference_path: Path to reference structure.

        Returns:
            List of EvaluationResult objects.
        """
        results = []
        for model_path in model_paths:
            result = self.evaluate(model_path, reference_path)
            results.append(result)
        return results

    def get_available_metrics(self) -> Dict[str, bool]:
        """
        Check which metrics are available.

        Returns:
            Dictionary mapping metric names to availability status.
        """
        return {name: metric.is_available() for name, metric in self.metrics.items()}

    def get_metric_requirements(self) -> Dict[str, str]:
        """
        Get requirements for unavailable metrics.

        Returns:
            Dictionary mapping metric names to requirement descriptions.
        """
        return {
            name: metric.get_requirements()
            for name, metric in self.metrics.items()
            if not metric.is_available()
        }

    @classmethod
    def list_all_metrics(cls) -> List[Dict[str, Any]]:
        """
        List all available metric types.

        Returns:
            List of metric information dictionaries.
        """
        metrics_info = []
        for name, metric_class in cls.AVAILABLE_METRICS.items():
            metric = metric_class()
            metrics_info.append(
                {
                    "name": name,
                    "description": metric.description,
                    "requires_reference": metric.requires_reference,
                    "available": metric.is_available(),
                    "requirements": metric.get_requirements()
                    if not metric.is_available()
                    else None,
                }
            )
        return metrics_info

    def evaluate_comprehensive(
        self,
        model_path: Path,
        reference_path: Path,
    ) -> Dict[str, Any]:
        """
        Compute all OpenStructure metrics at global, per-residue, per-chain, and interface levels.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.

        Returns:
            Comprehensive dictionary with all metrics at all levels.
        """
        model_path = Path(model_path)
        reference_path = Path(reference_path)

        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner

            runner = get_ost_runner()

            if not runner.is_available():
                raise EvaluationError(
                    "comprehensive",
                    "OpenStructure not available. Install with: micromamba create -n ost -c conda-forge -c bioconda openstructure",
                )

            # Get comprehensive OpenStructure metrics
            ost_metrics = runner.compute_all_metrics(model_path, reference_path)

            # Add TM-score from TMalign if available
            if "tm_score" in self.metrics and self.metrics["tm_score"].is_available():
                try:
                    tm_result = self.metrics["tm_score"].compute(model_path, reference_path)
                    ost_metrics["global"]["tm_score"] = tm_result.get("tm_score")
                    ost_metrics["global"]["tm_score_chain1"] = tm_result.get("tm_score_chain1")
                    ost_metrics["global"]["tm_score_chain2"] = tm_result.get("tm_score_chain2")
                    ost_metrics["global"]["gdt_ts"] = tm_result.get("gdt_ts")
                    ost_metrics["global"]["gdt_ha"] = tm_result.get("gdt_ha")
                    ost_metrics["global"]["aligned_length"] = tm_result.get("aligned_length")
                    ost_metrics["global"]["sequence_identity"] = tm_result.get("sequence_identity")
                except Exception as e:
                    ost_metrics["global"]["tm_error"] = str(e)

            return ost_metrics

        except ImportError:
            raise EvaluationError("comprehensive", "OpenStructure runner not available")
        except RuntimeError as e:
            raise EvaluationError("comprehensive", str(e))


class ComprehensiveEvaluationResult:
    """Container for comprehensive evaluation results at all levels."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self._global = data.get("global", {})
        self._per_residue = data.get("per_residue", {})
        self._per_chain = data.get("per_chain", {})
        self._interface = data.get("interface", {})

    # Global level properties
    @property
    def lddt(self) -> Optional[float]:
        return self._global.get("lddt")

    @property
    def lddt_mean(self) -> Optional[float]:
        return self._global.get("lddt_mean")

    @property
    def lddt_min(self) -> Optional[float]:
        return self._global.get("lddt_min")

    @property
    def lddt_max(self) -> Optional[float]:
        return self._global.get("lddt_max")

    @property
    def rmsd_ca(self) -> Optional[float]:
        return self._global.get("rmsd_ca")

    @property
    def rmsd_backbone(self) -> Optional[float]:
        return self._global.get("rmsd_backbone")

    @property
    def tm_score(self) -> Optional[float]:
        return self._global.get("tm_score")

    @property
    def qs_score(self) -> Optional[float]:
        return self._global.get("qs_score")

    @property
    def gdt_ts(self) -> Optional[float]:
        return self._global.get("gdt_ts")

    @property
    def gdt_ha(self) -> Optional[float]:
        return self._global.get("gdt_ha")

    @property
    def quality_categories(self) -> Dict[str, int]:
        return self._global.get("lddt_quality_categories", {})

    # Per-residue level
    @property
    def lddt_per_residue(self) -> List[float]:
        return self._per_residue.get("lddt", [])

    @property
    def lddt_residue_details(self) -> List[Dict]:
        return self._per_residue.get("lddt_details", [])

    # Per-chain level
    @property
    def chain_metrics(self) -> Dict[str, Dict]:
        return self._per_chain

    # Interface level
    @property
    def interface_metrics(self) -> Dict[str, Any]:
        return self._interface

    def to_dict(self) -> Dict[str, Any]:
        return self.data

    def summary(self) -> str:
        """Generate a text summary of evaluation results."""
        lines = ["=" * 50, "EVALUATION SUMMARY", "=" * 50, ""]

        # Global metrics
        lines.append("GLOBAL METRICS:")
        lines.append("-" * 30)
        if self.lddt is not None:
            lines.append(f"  lDDT:           {self.lddt:.4f}")
        if self.lddt_mean is not None:
            lines.append(f"  lDDT (mean):    {self.lddt_mean:.4f}")
        if self.lddt_min is not None and self.lddt_max is not None:
            lines.append(f"  lDDT (range):   {self.lddt_min:.4f} - {self.lddt_max:.4f}")
        if self.rmsd_ca is not None:
            lines.append(f"  RMSD (Cα):      {self.rmsd_ca:.4f} Å")
        if self.rmsd_backbone is not None:
            lines.append(f"  RMSD (BB):      {self.rmsd_backbone:.4f} Å")
        if self.tm_score is not None:
            lines.append(f"  TM-score:       {self.tm_score:.4f}")
        if self.qs_score is not None:
            lines.append(f"  QS-score:       {self.qs_score:.4f}")
        if self.gdt_ts is not None:
            lines.append(f"  GDT-TS:         {self.gdt_ts:.4f}")
        if self.gdt_ha is not None:
            lines.append(f"  GDT-HA:         {self.gdt_ha:.4f}")

        # Quality categories
        if self.quality_categories:
            lines.append("")
            lines.append("QUALITY DISTRIBUTION:")
            lines.append("-" * 30)
            total = sum(self.quality_categories.values())
            for cat, count in self.quality_categories.items():
                pct = (count / total * 100) if total > 0 else 0
                lines.append(f"  {cat}: {count} ({pct:.1f}%)")

        # Per-chain summary
        if self.chain_metrics:
            lines.append("")
            lines.append("PER-CHAIN METRICS:")
            lines.append("-" * 30)
            for chain, metrics in self.chain_metrics.items():
                if isinstance(metrics, dict) and "mean_lddt" in metrics:
                    lines.append(
                        f"  Chain {chain}: lDDT={metrics['mean_lddt']:.4f}, n={metrics.get('num_residues', 'N/A')}"
                    )

        return "\n".join(lines)
