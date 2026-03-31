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
        fixed_k: int | None = None,
        seed_solution: np.ndarray | None = None,
        seed_fraction: float = 0.25,
        seed_perturb_swaps: int = 4,
        local_search_elites: int = 3,
        local_search_steps: int = 12,
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
        fixed_k              : exact number of shelters per chromosome
        seed_solution        : optional greedy chromosome used for seeding
        seed_fraction        : population fraction initialised from seed
        seed_perturb_swaps   : average swaps to perturb seeded copies
        local_search_elites  : top-k individuals improved by local search
        local_search_steps   : number of random 1-swap trials per elite
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

        if fixed_k is None:
            fixed_k = int(round(target_shelter_ratio * n_genes))
        self.fixed_k = max(1, min(int(fixed_k), n_genes))

        self.seed_fraction = max(0.0, min(float(seed_fraction), 1.0))
        self.seed_perturb_swaps = max(1, int(seed_perturb_swaps))
        self.local_search_elites = max(0, min(int(local_search_elites), pop_size))
        self.local_search_steps = max(0, int(local_search_steps))

        # Fixed-K initialisation: every chromosome starts with exactly K shelters.
        self.population = np.zeros((pop_size, n_genes), dtype=np.int8)
        for i in range(pop_size):
            on_idx = np.random.choice(n_genes, self.fixed_k, replace=False)
            self.population[i, on_idx] = 1

        if seed_solution is not None:
            self._inject_seed_solution(seed_solution)

        self.best_solution = None
        self.best_fitness = -np.inf
        self.history = []           # best fitness per generation
        self.avg_history = []       # average fitness per generation

    # ── Fixed-K helpers ─────────────────────────────────────────────────────

    def _repair_fixed_k(self, individual):
        """Repair chromosome so it has exactly self.fixed_k selected sites."""
        ones = np.flatnonzero(individual == 1)
        n_ones = len(ones)

        if n_ones > self.fixed_k:
            to_drop = np.random.choice(ones, n_ones - self.fixed_k, replace=False)
            individual[to_drop] = 0
        elif n_ones < self.fixed_k:
            zeros = np.flatnonzero(individual == 0)
            to_add = np.random.choice(zeros, self.fixed_k - n_ones, replace=False)
            individual[to_add] = 1

        return individual

    def _swap_positions(self, individual, n_swaps):
        """Perform n selected<->unselected swaps to preserve fixed-K."""
        if n_swaps <= 0:
            return individual

        selected = np.flatnonzero(individual == 1)
        unselected = np.flatnonzero(individual == 0)
        n_swaps = min(int(n_swaps), len(selected), len(unselected))
        if n_swaps <= 0:
            return individual

        off_idx = np.random.choice(selected, size=n_swaps, replace=False)
        on_idx = np.random.choice(unselected, size=n_swaps, replace=False)

        individual[off_idx] = 0
        individual[on_idx] = 1
        return individual

    def _inject_seed_solution(self, seed_solution):
        """Inject greedy seed and perturbed variants into initial population."""
        seed = np.asarray(seed_solution, dtype=np.int8).copy()
        if seed.shape[0] != self.n_genes:
            raise ValueError(
                f"seed_solution length {seed.shape[0]} != n_genes {self.n_genes}"
            )

        seed = self._repair_fixed_k(seed)
        seeded_count = max(1, int(round(self.pop_size * self.seed_fraction)))
        seeded_count = min(seeded_count, self.pop_size)

        self.population[0] = seed
        for i in range(1, seeded_count):
            variant = seed.copy()
            n_swaps = np.random.poisson(self.seed_perturb_swaps)
            if n_swaps <= 0:
                n_swaps = 1
            self.population[i] = self._swap_positions(variant, n_swaps)

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

    def _uniform_crossover(self, parent1, parent2, rate):
        """Uniform crossover: each gene independently swapped."""
        if np.random.rand() > rate:
            return parent1.copy(), parent2.copy()
        mask = np.random.randint(0, 2, len(parent1), dtype=np.int8)
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        child1 = self._repair_fixed_k(child1)
        child2 = self._repair_fixed_k(child2)
        return child1, child2

    # ── Mutation ─────────────────────────────────────────────────────────────

    def _mutate(self, individual):
        """Swap mutation that preserves the fixed number of selected shelters."""
        n_swaps = np.random.binomial(self.fixed_k, self.mutation_rate)
        if n_swaps == 0 and np.random.rand() < self.mutation_rate:
            n_swaps = 1
        individual = self._swap_positions(individual, n_swaps)
        return self._repair_fixed_k(individual)

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

    def _local_search_swap(self, individual, start_score):
        """Randomised 1-swap hill-climb around an elite solution."""
        if self.local_search_steps <= 0:
            return individual, start_score

        best_individual = individual.copy()
        best_score = start_score

        for _ in range(self.local_search_steps):
            candidate = best_individual.copy()
            candidate = self._swap_positions(candidate, 1)
            candidate_score = self.fitness_func(candidate)
            if candidate_score > best_score:
                best_individual = candidate
                best_score = candidate_score

        return best_individual, best_score

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

            # Local improvement on current elites (memetic step)
            if self.local_search_elites > 0 and self.local_search_steps > 0:
                elite_ls_idx = np.argsort(scores)[-self.local_search_elites:]
                for idx in elite_ls_idx:
                    improved, improved_score = self._local_search_swap(
                        self.population[idx], scores[idx]
                    )
                    if improved_score > scores[idx]:
                        self.population[idx] = improved
                        scores[idx] = improved_score

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