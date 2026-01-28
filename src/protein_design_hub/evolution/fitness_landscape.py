"""Fitness landscape exploration and evaluation.

This module provides:
- Fitness functions for protein optimization
- Composite fitness combining multiple objectives
- Landscape exploration and visualization
- Epistasis analysis
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
import json


@dataclass
class FitnessResult:
    """Result of fitness evaluation."""
    sequence: str
    fitness: float
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "sequence": self.sequence,
            "fitness": self.fitness,
            "components": self.components,
            "metadata": self.metadata,
        }


class FitnessFunction(ABC):
    """Base class for fitness functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Function name."""
        pass

    @property
    def weight(self) -> float:
        """Weight for composite fitness."""
        return 1.0

    @property
    def maximize(self) -> bool:
        """Whether to maximize (True) or minimize (False)."""
        return True

    @abstractmethod
    def evaluate(self, sequence: str, **kwargs) -> float:
        """
        Evaluate fitness of a sequence.

        Args:
            sequence: Amino acid sequence.
            **kwargs: Additional parameters.

        Returns:
            Fitness score.
        """
        pass

    def evaluate_batch(self, sequences: List[str], **kwargs) -> List[float]:
        """Evaluate multiple sequences."""
        return [self.evaluate(seq, **kwargs) for seq in sequences]


class StructureFitness(FitnessFunction):
    """Fitness based on structure prediction quality."""

    def __init__(
        self,
        predictor_name: str = "esmfold_api",
        metric: str = "plddt",
        weight: float = 1.0,
    ):
        self.predictor_name = predictor_name
        self.metric = metric
        self._weight = weight

    @property
    def name(self) -> str:
        return f"Structure({self.predictor_name}/{self.metric})"

    @property
    def weight(self) -> float:
        return self._weight

    def evaluate(self, sequence: str, **kwargs) -> float:
        """Evaluate using structure prediction."""
        try:
            if self.predictor_name == "esmfold_api":
                import requests
                response = requests.post(
                    "https://api.esmatlas.com/foldSequence/v1/pdb/",
                    data=sequence,
                    headers={"Content-Type": "text/plain"},
                    timeout=120,
                )
                if response.status_code != 200:
                    return 0.0

                # Extract pLDDT from PDB B-factors
                plddt_values = []
                for line in response.text.split('\n'):
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        try:
                            plddt_values.append(float(line[60:66]))
                        except ValueError:
                            pass

                if plddt_values:
                    return sum(plddt_values) / len(plddt_values) / 100.0  # Normalize to 0-1
            else:
                # Use local predictor
                from protein_design_hub.predictors.registry import get_predictor
                from protein_design_hub.core.config import get_settings
                from protein_design_hub.core.types import PredictionInput, Sequence

                settings = get_settings()
                predictor = get_predictor(self.predictor_name, settings)

                pred_input = PredictionInput(
                    job_id="fitness_eval",
                    sequences=[Sequence(id="seq", sequence=sequence)],
                )
                result = predictor.predict(pred_input)

                if result.success and result.scores:
                    if self.metric == "plddt":
                        return result.scores[0].plddt / 100.0 if result.scores[0].plddt else 0.0
                    elif self.metric == "ptm":
                        return result.scores[0].ptm if result.scores[0].ptm else 0.0

            return 0.0
        except Exception:
            return 0.0


class SolubilityFitness(FitnessFunction):
    """Fitness based on predicted solubility."""

    def __init__(self, weight: float = 1.0):
        self._weight = weight

    @property
    def name(self) -> str:
        return "Solubility"

    @property
    def weight(self) -> float:
        return self._weight

    def evaluate(self, sequence: str, **kwargs) -> float:
        """Evaluate solubility."""
        from protein_design_hub.biophysics.solubility import calculate_solubility_score
        score = calculate_solubility_score(sequence)
        # Normalize to roughly 0-1 range
        return max(0, min(1, (score + 2) / 4))


class StabilityFitness(FitnessFunction):
    """Fitness based on predicted stability."""

    def __init__(self, weight: float = 1.0):
        self._weight = weight

    @property
    def name(self) -> str:
        return "Stability"

    @property
    def weight(self) -> float:
        return self._weight

    def evaluate(self, sequence: str, **kwargs) -> float:
        """Evaluate stability."""
        from protein_design_hub.biophysics.properties import calculate_instability_index
        instability = calculate_instability_index(sequence)
        # Convert to fitness (lower instability = higher fitness)
        return max(0, min(1, 1 - instability / 100))


class SequenceRecoveryFitness(FitnessFunction):
    """Fitness based on sequence recovery vs reference."""

    def __init__(self, reference: str, weight: float = 1.0):
        self.reference = reference.upper()
        self._weight = weight

    @property
    def name(self) -> str:
        return "SequenceRecovery"

    @property
    def weight(self) -> float:
        return self._weight

    def evaluate(self, sequence: str, **kwargs) -> float:
        """Calculate sequence identity to reference."""
        sequence = sequence.upper()
        if len(sequence) != len(self.reference):
            return 0.0

        matches = sum(1 for a, b in zip(sequence, self.reference) if a == b)
        return matches / len(self.reference)


class HydrophobicityFitness(FitnessFunction):
    """Fitness based on target hydrophobicity."""

    def __init__(self, target_gravy: float = 0.0, weight: float = 1.0):
        self.target_gravy = target_gravy
        self._weight = weight

    @property
    def name(self) -> str:
        return f"Hydrophobicity(target={self.target_gravy})"

    @property
    def weight(self) -> float:
        return self._weight

    def evaluate(self, sequence: str, **kwargs) -> float:
        """Evaluate deviation from target GRAVY."""
        from protein_design_hub.biophysics.properties import calculate_gravy
        gravy = calculate_gravy(sequence)
        deviation = abs(gravy - self.target_gravy)
        # Convert to fitness (lower deviation = higher fitness)
        return max(0, 1 - deviation / 4)


class ChargeFitness(FitnessFunction):
    """Fitness based on target net charge."""

    def __init__(self, target_charge: float = 0.0, pH: float = 7.0, weight: float = 1.0):
        self.target_charge = target_charge
        self.pH = pH
        self._weight = weight

    @property
    def name(self) -> str:
        return f"Charge(target={self.target_charge})"

    @property
    def weight(self) -> float:
        return self._weight

    def evaluate(self, sequence: str, **kwargs) -> float:
        """Evaluate deviation from target charge."""
        from protein_design_hub.biophysics.properties import calculate_charge
        charge = calculate_charge(sequence, self.pH)
        deviation = abs(charge - self.target_charge)
        # Convert to fitness
        return max(0, 1 - deviation / 20)


class CustomFitness(FitnessFunction):
    """Custom fitness function from user-provided callable."""

    def __init__(
        self,
        func: Callable[[str], float],
        name: str = "Custom",
        weight: float = 1.0,
        maximize: bool = True,
    ):
        self._func = func
        self._name = name
        self._weight = weight
        self._maximize = maximize

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def maximize(self) -> bool:
        return self._maximize

    def evaluate(self, sequence: str, **kwargs) -> float:
        return self._func(sequence)


class CompositeFitness:
    """Combine multiple fitness functions."""

    def __init__(self, functions: List[FitnessFunction]):
        """
        Initialize composite fitness.

        Args:
            functions: List of fitness functions to combine.
        """
        self.functions = functions

    def evaluate(self, sequence: str, **kwargs) -> FitnessResult:
        """
        Evaluate all fitness functions and combine.

        Args:
            sequence: Amino acid sequence.

        Returns:
            FitnessResult with combined fitness and components.
        """
        components: Dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for func in self.functions:
            score = func.evaluate(sequence, **kwargs)

            # Flip score if minimizing
            if not func.maximize:
                score = 1 - score

            components[func.name] = score
            weighted_sum += score * func.weight
            total_weight += func.weight

        # Compute weighted average
        combined = weighted_sum / total_weight if total_weight > 0 else 0.0

        return FitnessResult(
            sequence=sequence,
            fitness=combined,
            components=components,
        )

    def evaluate_batch(self, sequences: List[str], **kwargs) -> List[FitnessResult]:
        """Evaluate multiple sequences."""
        return [self.evaluate(seq, **kwargs) for seq in sequences]


@dataclass
class FitnessLandscape:
    """Fitness landscape for exploring sequence-fitness relationships."""

    parent_sequence: str
    fitness_function: CompositeFitness
    evaluated_sequences: Dict[str, FitnessResult] = field(default_factory=dict)

    def evaluate(self, sequence: str, **kwargs) -> FitnessResult:
        """Evaluate a sequence, caching result."""
        if sequence not in self.evaluated_sequences:
            result = self.fitness_function.evaluate(sequence, **kwargs)
            self.evaluated_sequences[sequence] = result
        return self.evaluated_sequences[sequence]

    def evaluate_batch(
        self,
        sequences: List[str],
        **kwargs
    ) -> List[FitnessResult]:
        """Evaluate multiple sequences."""
        results = []
        for seq in sequences:
            results.append(self.evaluate(seq, **kwargs))
        return results

    def get_top_sequences(self, n: int = 10) -> List[FitnessResult]:
        """Get top N sequences by fitness."""
        sorted_results = sorted(
            self.evaluated_sequences.values(),
            key=lambda x: x.fitness,
            reverse=True
        )
        return sorted_results[:n]

    def get_fitness_distribution(self) -> Dict[str, float]:
        """Get fitness distribution statistics."""
        if not self.evaluated_sequences:
            return {}

        fitnesses = [r.fitness for r in self.evaluated_sequences.values()]

        return {
            "count": len(fitnesses),
            "min": min(fitnesses),
            "max": max(fitnesses),
            "mean": sum(fitnesses) / len(fitnesses),
            "std": (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses)) ** 0.5,
        }

    def analyze_position_effects(self) -> Dict[int, Dict[str, float]]:
        """
        Analyze the effect of mutations at each position.

        Returns:
            Dictionary mapping positions to average fitness by amino acid.
        """
        from collections import defaultdict

        position_effects: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        parent = self.parent_sequence.upper()

        for seq, result in self.evaluated_sequences.items():
            seq = seq.upper()
            for i, (p, s) in enumerate(zip(parent, seq)):
                if s != p:  # Mutation at this position
                    position_effects[i + 1][s].append(result.fitness)

        # Average the effects
        averaged: Dict[int, Dict[str, float]] = {}
        for pos, aa_effects in position_effects.items():
            averaged[pos] = {
                aa: sum(fits) / len(fits)
                for aa, fits in aa_effects.items()
            }

        return averaged

    def calculate_epistasis(
        self,
        single_mutants: Dict[str, float],
        double_mutants: Dict[str, float],
    ) -> List[Dict]:
        """
        Calculate epistatic interactions between mutations.

        Args:
            single_mutants: Dict mapping single mutations (e.g., "A10G") to fitness.
            double_mutants: Dict mapping double mutations to fitness.

        Returns:
            List of epistasis calculations.
        """
        epistasis_results = []

        parent_fitness = self.evaluated_sequences.get(
            self.parent_sequence, FitnessResult(self.parent_sequence, 0.5)
        ).fitness

        for double_mut, double_fit in double_mutants.items():
            # Parse double mutation
            mutations = double_mut.split("/")
            if len(mutations) != 2:
                continue

            mut1, mut2 = mutations

            if mut1 not in single_mutants or mut2 not in single_mutants:
                continue

            single1 = single_mutants[mut1]
            single2 = single_mutants[mut2]

            # Expected additive effect
            delta1 = single1 - parent_fitness
            delta2 = single2 - parent_fitness
            expected = parent_fitness + delta1 + delta2

            # Epistasis = observed - expected
            epistasis = double_fit - expected

            epistasis_results.append({
                "mutation1": mut1,
                "mutation2": mut2,
                "single1_fitness": single1,
                "single2_fitness": single2,
                "double_fitness": double_fit,
                "expected_additive": expected,
                "epistasis": epistasis,
                "epistasis_type": "positive" if epistasis > 0.05 else "negative" if epistasis < -0.05 else "additive",
            })

        return epistasis_results

    def save(self, path: Path) -> None:
        """Save landscape to file."""
        data = {
            "parent_sequence": self.parent_sequence,
            "evaluated": {
                seq: result.to_dict()
                for seq, result in self.evaluated_sequences.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(
        cls,
        path: Path,
        fitness_function: CompositeFitness,
    ) -> "FitnessLandscape":
        """Load landscape from file."""
        with open(path) as f:
            data = json.load(f)

        landscape = cls(
            parent_sequence=data["parent_sequence"],
            fitness_function=fitness_function,
        )

        for seq, result_dict in data.get("evaluated", {}).items():
            landscape.evaluated_sequences[seq] = FitnessResult(
                sequence=result_dict["sequence"],
                fitness=result_dict["fitness"],
                components=result_dict.get("components", {}),
                metadata=result_dict.get("metadata", {}),
            )

        return landscape
