"""
genetic_algo.py — Refined Genetic Algorithm for the UFLP.

Key improvements over the starter code:
  • Elitism: best individuals survive unchanged to next generation.
  • Adaptive mutation rate: starts high (exploration) and decays.
  • Smart initialisation: random shelter density ~ target_density.
  • Vectorised where possible (numpy operations on population matrix).
  • Stagnation detection: bumps mutation rate if fitness plateaus.
"""

import numpy as np
import random
from tqdm import tqdm


class GeneticAlgorithm:
    def __init__(
        self,
        n_genes: int,
        fitness_func,
        pop_size: int = 100,
        generations: int = 200,
        mutation_rate: float = 0.02,
        crossover_rate: float = 0.85,
        elitism_count: int = 2,
        tournament_size: int = 3,
        target_shelter_ratio: float = 0.01,
        adaptive_mutation: bool = True,
        stagnation_window: int = 15,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        n_genes              : number of candidate zip codes (chromosome length)
        fitness_func         : callable(chromosome) → float
        pop_size             : number of individuals per generation
        generations          : maximum number of generations
        mutation_rate        : base per-gene bit-flip probability
        crossover_rate       : probability of crossover per pair
        elitism_count        : top-k individuals copied unchanged
        tournament_size      : k for tournament selection
        target_shelter_ratio : expected fraction of genes =1 at init
        adaptive_mutation    : whether to use decay + stagnation bump
        stagnation_window    : generations without improvement before bump
        seed                 : random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.n_genes = n_genes
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate_base = mutation_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = min(elitism_count, pop_size)
        self.tournament_size = tournament_size
        self.adaptive_mutation = adaptive_mutation
        self.stagnation_window = stagnation_window

        # Smart initialisation: each gene is 1 with probability = target_shelter_ratio
        self.population = (
            np.random.rand(pop_size, n_genes) < target_shelter_ratio
        ).astype(np.int8)

        self.best_solution = None
        self.best_fitness = -np.inf
        self.history = []           # best fitness per generation
        self.avg_history = []       # average fitness per generation

    # ── Selection ────────────────────────────────────────────────────────────

    def _tournament_select(self, scores):
        """Tournament selection with configurable tournament size."""
        selected_idx = np.empty(self.pop_size, dtype=int)
        for i in range(self.pop_size):
            candidates = np.random.choice(self.pop_size, self.tournament_size, replace=False)
            winner = candidates[np.argmax(scores[candidates])]
            selected_idx[i] = winner
        return self.population[selected_idx].copy()

    # ── Crossover ────────────────────────────────────────────────────────────

    @staticmethod
    def _uniform_crossover(parent1, parent2, rate):
        """Uniform crossover: each gene independently swapped."""
        if np.random.rand() > rate:
            return parent1.copy(), parent2.copy()
        mask = np.random.randint(0, 2, len(parent1), dtype=np.int8)
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    # ── Mutation ─────────────────────────────────────────────────────────────

    def _mutate(self, individual):
        """Bit-flip mutation with current mutation rate."""
        flip_mask = np.random.rand(self.n_genes) < self.mutation_rate
        individual[flip_mask] = 1 - individual[flip_mask]
        return individual

    # ── Adaptive mutation helpers ────────────────────────────────────────────

    def _update_mutation_rate(self, gen, stagnation_counter):
        """
        • Linear decay from base rate to base_rate/5 over generations.
        • If stagnation detected, temporarily boost to 3× base.
        """
        if not self.adaptive_mutation:
            return

        decay = self.mutation_rate_base * (1.0 - 0.8 * gen / self.generations)
        self.mutation_rate = max(decay, self.mutation_rate_base / 5)

        if stagnation_counter >= self.stagnation_window:
            self.mutation_rate = min(self.mutation_rate_base * 3, 0.15)

    # ── Main evolution loop ──────────────────────────────────────────────────

    def evolve(self):
        """
        Run the evolutionary loop.

        Returns
        -------
        best_solution : 1-D int8 array
        best_fitness  : float
        """
        stagnation_counter = 0

        pbar = tqdm(range(self.generations), desc="GA Evolution", unit="gen")
        for gen in pbar:
            # Evaluate
            scores = np.array([self.fitness_func(ind) for ind in self.population])

            # Track best & average
            gen_best_idx = np.argmax(scores)
            gen_best_score = scores[gen_best_idx]
            gen_avg_score = scores.mean()

            if gen_best_score > self.best_fitness:
                self.best_fitness = gen_best_score
                self.best_solution = self.population[gen_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            self.history.append(self.best_fitness)
            self.avg_history.append(gen_avg_score)

            # Adaptive mutation rate
            self._update_mutation_rate(gen, stagnation_counter)

            # ── Create next generation ──

            # Elitism: keep top-k individuals
            elite_idx = np.argsort(scores)[-self.elitism_count:]
            elites = self.population[elite_idx].copy()

            # Selection
            selected = self._tournament_select(scores)

            # Crossover & Mutation
            next_pop = []
            for i in range(0, self.pop_size - self.elitism_count, 2):
                p1 = selected[i]
                p2 = selected[min(i + 1, len(selected) - 1)]
                c1, c2 = self._uniform_crossover(p1, p2, self.crossover_rate)
                next_pop.append(self._mutate(c1))
                if len(next_pop) < self.pop_size - self.elitism_count:
                    next_pop.append(self._mutate(c2))

            # Pad if odd
            while len(next_pop) < self.pop_size - self.elitism_count:
                next_pop.append(self._mutate(selected[np.random.randint(len(selected))].copy()))

            next_pop = np.array(next_pop, dtype=np.int8)
            self.population = np.vstack([elites, next_pop])

            # Progress bar update
            pbar.set_postfix({
                "best": f"{self.best_fitness:.4f}",
                "avg":  f"{gen_avg_score:.4f}",
                "mut":  f"{self.mutation_rate:.4f}",
                "stag": stagnation_counter,
            })

        return self.best_solution, self.best_fitness