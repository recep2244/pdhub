"""Directed evolution workflow orchestration.

This module provides:
- Iterative evolution cycles
- Selection strategies
- Population management
- Evolution history tracking
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import json
import random
from datetime import datetime

from protein_design_hub.evolution.mutation_operators import (
    MutantSequence,
    MutationOperator,
    PointMutation,
    Recombination,
)
from protein_design_hub.evolution.fitness_landscape import (
    CompositeFitness,
    FitnessFunction,
    FitnessLandscape,
    FitnessResult,
)


class SelectionStrategy(Enum):
    """Selection strategies for directed evolution."""
    TRUNCATION = "truncation"  # Select top N
    TOURNAMENT = "tournament"  # Tournament selection
    ROULETTE = "roulette"  # Fitness-proportional selection
    RANK = "rank"  # Rank-based selection
    ELITE = "elite"  # Keep top performers + random


@dataclass
class EvolutionConfig:
    """Configuration for directed evolution."""
    # Population parameters
    population_size: int = 100
    elite_size: int = 5

    # Evolution parameters
    num_generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3

    # Selection
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT
    tournament_size: int = 3
    truncation_fraction: float = 0.2

    # Termination
    target_fitness: Optional[float] = None
    stagnation_generations: int = 5
    min_fitness_improvement: float = 0.01

    # Mutation operator settings
    num_mutations_per_variant: int = 1
    conservative_mutations: bool = False

    # Diversity
    diversity_threshold: float = 0.9  # Max sequence identity

    def to_dict(self) -> Dict:
        return {
            "population_size": self.population_size,
            "elite_size": self.elite_size,
            "num_generations": self.num_generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "selection_strategy": self.selection_strategy.value,
            "tournament_size": self.tournament_size,
            "truncation_fraction": self.truncation_fraction,
            "target_fitness": self.target_fitness,
            "stagnation_generations": self.stagnation_generations,
            "min_fitness_improvement": self.min_fitness_improvement,
            "num_mutations_per_variant": self.num_mutations_per_variant,
            "conservative_mutations": self.conservative_mutations,
            "diversity_threshold": self.diversity_threshold,
        }


@dataclass
class GenerationResult:
    """Result of a single generation."""
    generation: int
    population: List[FitnessResult]
    best_fitness: float
    mean_fitness: float
    diversity: float
    new_best: bool = False

    def to_dict(self) -> Dict:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "diversity": self.diversity,
            "new_best": self.new_best,
            "top_variants": [
                {"sequence": p.sequence[:50] + "...", "fitness": p.fitness}
                for p in sorted(self.population, key=lambda x: x.fitness, reverse=True)[:5]
            ],
        }


@dataclass
class EvolutionResult:
    """Result of complete evolution run."""
    config: EvolutionConfig
    parent_sequence: str
    generations: List[GenerationResult]
    best_variant: FitnessResult
    all_evaluated: Dict[str, FitnessResult]
    runtime_seconds: float = 0.0
    termination_reason: str = "completed"

    @property
    def num_generations(self) -> int:
        return len(self.generations)

    @property
    def improvement(self) -> float:
        if self.generations:
            return self.best_variant.fitness - self.generations[0].best_fitness
        return 0.0

    def get_fitness_trajectory(self) -> List[Tuple[int, float, float]]:
        """Get (generation, best_fitness, mean_fitness) trajectory."""
        return [
            (g.generation, g.best_fitness, g.mean_fitness)
            for g in self.generations
        ]

    def get_top_variants(self, n: int = 10) -> List[FitnessResult]:
        """Get top N variants across all generations."""
        sorted_variants = sorted(
            self.all_evaluated.values(),
            key=lambda x: x.fitness,
            reverse=True
        )
        return sorted_variants[:n]

    def to_dict(self) -> Dict:
        return {
            "config": self.config.to_dict(),
            "parent_sequence": self.parent_sequence,
            "num_generations": self.num_generations,
            "best_variant": {
                "sequence": self.best_variant.sequence,
                "fitness": self.best_variant.fitness,
                "components": self.best_variant.components,
            },
            "improvement": self.improvement,
            "runtime_seconds": self.runtime_seconds,
            "termination_reason": self.termination_reason,
            "fitness_trajectory": self.get_fitness_trajectory(),
        }

    def save(self, path: Path) -> None:
        """Save results to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class DirectedEvolution:
    """Directed evolution workflow."""

    def __init__(
        self,
        parent_sequence: str,
        fitness_function: CompositeFitness,
        config: Optional[EvolutionConfig] = None,
        mutation_operator: Optional[MutationOperator] = None,
    ):
        """
        Initialize directed evolution.

        Args:
            parent_sequence: Starting sequence.
            fitness_function: Composite fitness function.
            config: Evolution configuration.
            mutation_operator: Custom mutation operator.
        """
        self.parent_sequence = parent_sequence.upper()
        self.fitness_function = fitness_function
        self.config = config or EvolutionConfig()

        # Default mutation operator
        self.mutation_operator = mutation_operator or PointMutation(
            num_mutations=self.config.num_mutations_per_variant,
            conservative_only=self.config.conservative_mutations,
        )

        # Initialize fitness landscape
        self.landscape = FitnessLandscape(
            parent_sequence=self.parent_sequence,
            fitness_function=self.fitness_function,
        )

        # Callbacks
        self.callbacks: List[Callable[[GenerationResult], None]] = []

    def add_callback(self, callback: Callable[[GenerationResult], None]) -> None:
        """Add a callback for generation completion."""
        self.callbacks.append(callback)

    def run(self) -> EvolutionResult:
        """
        Run directed evolution.

        Returns:
            EvolutionResult with complete evolution history.
        """
        start_time = datetime.now()

        # Initialize population
        population = self._initialize_population()

        generations: List[GenerationResult] = []
        best_ever = population[0]
        stagnation_count = 0

        for gen in range(self.config.num_generations):
            # Evaluate population
            evaluated = self._evaluate_population(population)

            # Sort by fitness
            evaluated.sort(key=lambda x: x.fitness, reverse=True)

            # Track best
            gen_best = evaluated[0]
            new_best = gen_best.fitness > best_ever.fitness

            if new_best:
                best_ever = gen_best
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Calculate diversity
            diversity = self._calculate_diversity([e.sequence for e in evaluated])

            # Create generation result
            gen_result = GenerationResult(
                generation=gen,
                population=evaluated,
                best_fitness=gen_best.fitness,
                mean_fitness=sum(e.fitness for e in evaluated) / len(evaluated),
                diversity=diversity,
                new_best=new_best,
            )
            generations.append(gen_result)

            # Callbacks
            for callback in self.callbacks:
                callback(gen_result)

            # Check termination
            termination = self._check_termination(
                gen_result, stagnation_count, generations
            )
            if termination:
                break

            # Selection
            parents = self._select(evaluated)

            # Generate next population
            population = self._generate_offspring(parents, evaluated)

        runtime = (datetime.now() - start_time).total_seconds()

        termination_reason = "completed"
        if self.config.target_fitness and best_ever.fitness >= self.config.target_fitness:
            termination_reason = "target_reached"
        elif stagnation_count >= self.config.stagnation_generations:
            termination_reason = "stagnation"

        return EvolutionResult(
            config=self.config,
            parent_sequence=self.parent_sequence,
            generations=generations,
            best_variant=best_ever,
            all_evaluated=self.landscape.evaluated_sequences,
            runtime_seconds=runtime,
            termination_reason=termination_reason,
        )

    def _initialize_population(self) -> List[str]:
        """Initialize starting population."""
        population = [self.parent_sequence]

        # Generate initial variants
        while len(population) < self.config.population_size:
            variants = self.mutation_operator.mutate(
                self.parent_sequence,
                num_variants=self.config.population_size - len(population),
            )
            for v in variants:
                if v.sequence not in population:
                    population.append(v.sequence)
                    if len(population) >= self.config.population_size:
                        break

        return population

    def _evaluate_population(self, population: List[str]) -> List[FitnessResult]:
        """Evaluate fitness of all sequences."""
        return self.landscape.evaluate_batch(population)

    def _select(self, evaluated: List[FitnessResult]) -> List[str]:
        """Select parents for next generation."""
        strategy = self.config.selection_strategy

        if strategy == SelectionStrategy.TRUNCATION:
            n_select = int(len(evaluated) * self.config.truncation_fraction)
            return [e.sequence for e in evaluated[:n_select]]

        elif strategy == SelectionStrategy.TOURNAMENT:
            parents = []
            n_parents = self.config.population_size // 2

            for _ in range(n_parents):
                tournament = random.sample(evaluated, self.config.tournament_size)
                winner = max(tournament, key=lambda x: x.fitness)
                parents.append(winner.sequence)

            return parents

        elif strategy == SelectionStrategy.ROULETTE:
            # Fitness-proportional selection
            total_fitness = sum(max(0.01, e.fitness) for e in evaluated)
            parents = []
            n_parents = self.config.population_size // 2

            for _ in range(n_parents):
                r = random.uniform(0, total_fitness)
                cumsum = 0
                for e in evaluated:
                    cumsum += max(0.01, e.fitness)
                    if cumsum >= r:
                        parents.append(e.sequence)
                        break

            return parents

        elif strategy == SelectionStrategy.RANK:
            # Rank-based selection
            ranks = list(range(len(evaluated), 0, -1))
            total_rank = sum(ranks)
            parents = []
            n_parents = self.config.population_size // 2

            for _ in range(n_parents):
                r = random.uniform(0, total_rank)
                cumsum = 0
                for i, e in enumerate(evaluated):
                    cumsum += ranks[i]
                    if cumsum >= r:
                        parents.append(e.sequence)
                        break

            return parents

        else:  # ELITE
            # Keep elite + random selection
            elite = [e.sequence for e in evaluated[:self.config.elite_size]]
            rest = [e.sequence for e in evaluated[self.config.elite_size:]]
            random_select = random.sample(
                rest,
                min(len(rest), self.config.population_size // 2 - self.config.elite_size)
            )
            return elite + random_select

    def _generate_offspring(
        self,
        parents: List[str],
        current_pop: List[FitnessResult],
    ) -> List[str]:
        """Generate offspring from parents."""
        offspring = []

        # Keep elite
        elite = [e.sequence for e in current_pop[:self.config.elite_size]]
        offspring.extend(elite)

        # Generate new variants
        while len(offspring) < self.config.population_size:
            if random.random() < self.config.crossover_rate and len(parents) >= 2:
                # Recombination
                recomb = Recombination()
                variants = recomb.mutate(
                    parents[0],
                    parents=parents,
                    num_variants=1,
                )
                if variants:
                    offspring.append(variants[0].sequence)

            elif random.random() < self.config.mutation_rate:
                # Mutation
                parent = random.choice(parents)
                variants = self.mutation_operator.mutate(parent, num_variants=1)
                if variants:
                    offspring.append(variants[0].sequence)

            else:
                # Copy parent
                offspring.append(random.choice(parents))

        # Ensure diversity
        offspring = self._ensure_diversity(offspring)

        return offspring[:self.config.population_size]

    def _ensure_diversity(self, population: List[str]) -> List[str]:
        """Remove highly similar sequences to maintain diversity."""
        unique = []

        for seq in population:
            is_diverse = True
            for existing in unique:
                if self._sequence_identity(seq, existing) > self.config.diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                unique.append(seq)

        # Fill with random mutations if needed
        while len(unique) < len(population):
            variants = self.mutation_operator.mutate(
                random.choice(unique),
                num_variants=1,
            )
            if variants:
                unique.append(variants[0].sequence)

        return unique

    def _calculate_diversity(self, sequences: List[str]) -> float:
        """Calculate population diversity (average pairwise distance)."""
        if len(sequences) < 2:
            return 1.0

        total_distance = 0
        count = 0

        # Sample pairs for efficiency
        n_pairs = min(100, len(sequences) * (len(sequences) - 1) // 2)

        for _ in range(n_pairs):
            s1, s2 = random.sample(sequences, 2)
            total_distance += 1 - self._sequence_identity(s1, s2)
            count += 1

        return total_distance / count if count > 0 else 1.0

    @staticmethod
    def _sequence_identity(seq1: str, seq2: str) -> float:
        """Calculate sequence identity between two sequences."""
        if len(seq1) != len(seq2):
            return 0.0

        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)

    def _check_termination(
        self,
        current: GenerationResult,
        stagnation: int,
        history: List[GenerationResult],
    ) -> bool:
        """Check termination conditions."""
        # Target fitness reached
        if self.config.target_fitness and current.best_fitness >= self.config.target_fitness:
            return True

        # Stagnation
        if stagnation >= self.config.stagnation_generations:
            return True

        # Check improvement
        if len(history) > self.config.stagnation_generations:
            old_best = history[-self.config.stagnation_generations].best_fitness
            improvement = current.best_fitness - old_best
            if improvement < self.config.min_fitness_improvement:
                return True

        return False


def run_evolution_campaign(
    parent_sequence: str,
    fitness_functions: List[FitnessFunction],
    configs: Optional[List[EvolutionConfig]] = None,
    rounds: int = 3,
) -> List[EvolutionResult]:
    """
    Run multiple rounds of directed evolution.

    Args:
        parent_sequence: Starting sequence.
        fitness_functions: List of fitness functions.
        configs: Optional configs for each round.
        rounds: Number of evolution rounds.

    Returns:
        List of EvolutionResult for each round.
    """
    results = []
    current_seq = parent_sequence

    fitness = CompositeFitness(fitness_functions)

    for i in range(rounds):
        config = configs[i] if configs and i < len(configs) else EvolutionConfig()

        evolution = DirectedEvolution(
            parent_sequence=current_seq,
            fitness_function=fitness,
            config=config,
        )

        result = evolution.run()
        results.append(result)

        # Use best variant as parent for next round
        current_seq = result.best_variant.sequence

    return results
