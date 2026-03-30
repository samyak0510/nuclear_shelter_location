"""
fitness.py — Vectorized fitness evaluation for the UFLP Genetic Algorithm.

Uses the pre-computed sparse coverage matrix so each evaluation is O(nnz)
instead of O(N²).
"""

import numpy as np
from scipy import sparse


class FitnessFunction:
    """
    Evaluates a binary chromosome representing shelter placements.

    Fitness = w_cov * (covered_population / total_population)
            + w_infra * mean_infrastructure_score_of_selected_shelters
            - w_cost * (num_shelters / total_candidates)

    All components are normalised to [0, 1] so weights are directly
    interpretable.
    """

    def __init__(self, populations, coverage_matrix, infra_scores,
                 w_cov=0.7, w_infra=0.2, w_cost=0.1):
        """
        Parameters
        ----------
        populations     : 1-D array (N,) — population of each candidate ZIP
        coverage_matrix : sparse bool (N, N) — coverage[i,j]=True if zip i
                          is covered by a shelter at zip j
        infra_scores    : 1-D array (N,) — infrastructure score in [0,1]
        w_cov           : weight for population-coverage term
        w_infra         : weight for infrastructure-accessibility term
        w_cost          : weight for cost (shelter count) penalty
        """
        self.populations = populations.astype(np.float64)
        self.total_pop = populations.sum()
        self.coverage = coverage_matrix          # sparse (N, N)
        self.infra = infra_scores.astype(np.float64)
        self.n = len(populations)

        self.w_cov = w_cov
        self.w_infra = w_infra
        self.w_cost = w_cost

    def evaluate(self, chromosome):
        """
        Evaluate a single binary chromosome.

        Returns
        -------
        float — fitness score (higher is better)
        """
        selected = np.where(chromosome == 1)[0]
        n_shelters = len(selected)

        if n_shelters == 0:
            return 0.0

        # 1. Coverage: which ZIPs are covered by at least one selected shelter?
        # coverage[:, selected] is (N, k) — take any-per-row
        covered_mask = np.array(
            self.coverage[:, selected].sum(axis=1)
        ).flatten() > 0

        covered_pop = self.populations[covered_mask].sum()
        coverage_ratio = covered_pop / self.total_pop if self.total_pop > 0 else 0.0

        # 2. Infrastructure accessibility of selected sites
        infra_score = self.infra[selected].mean()

        # 3. Cost penalty (fewer shelters is better, all else equal)
        cost_ratio = n_shelters / self.n

        fitness = (self.w_cov   * coverage_ratio
                 + self.w_infra * infra_score
                 - self.w_cost  * cost_ratio)

        return fitness

    def evaluate_batch(self, population_matrix):
        """
        Evaluate an entire population at once.

        Parameters
        ----------
        population_matrix : 2-D array (pop_size, N)

        Returns
        -------
        scores : 1-D array (pop_size,)
        """
        return np.array([self.evaluate(ind) for ind in population_matrix])

    def detailed_report(self, chromosome):
        """Return a breakdown dict for a solution."""
        selected = np.where(chromosome == 1)[0]
        n_shelters = len(selected)

        if n_shelters == 0:
            return {"n_shelters": 0, "covered_pop": 0, "coverage_pct": 0,
                    "infra_score": 0, "cost_ratio": 0, "fitness": 0}

        covered_mask = np.array(
            self.coverage[:, selected].sum(axis=1)
        ).flatten() > 0
        covered_pop = self.populations[covered_mask].sum()
        coverage_pct = covered_pop / self.total_pop * 100

        infra_score = self.infra[selected].mean()
        cost_ratio = n_shelters / self.n

        fitness = (self.w_cov * (covered_pop / self.total_pop)
                 + self.w_infra * infra_score
                 - self.w_cost * cost_ratio)

        return {
            "n_shelters":   n_shelters,
            "covered_pop":  int(covered_pop),
            "total_pop":    int(self.total_pop),
            "coverage_pct": round(coverage_pct, 2),
            "infra_score":  round(infra_score, 4),
            "cost_ratio":   round(cost_ratio, 6),
            "fitness":      round(fitness, 6),
        }