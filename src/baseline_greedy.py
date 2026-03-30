"""
baseline.py — Greedy Heuristic Baseline for the UFLP.

Iteratively selects the safe ZIP code that covers the most
uncovered population until max_shelters is reached.

Uses sparse coverage matrix for vectorised marginal-gain calculation.
"""

import numpy as np
from scipy import sparse


def greedy_heuristic(populations, coverage_matrix, infra_scores,
                     max_shelters=100, w_cov=0.7, w_infra=0.2, w_cost=0.1):
    """
    Greedy Baseline for UFLP (vectorised).

    Parameters
    ----------
    populations      : 1-D array (N,) — population per candidate ZIP
    coverage_matrix  : sparse bool (N, N) — coverage adjacency
    infra_scores     : 1-D array (N,) — infrastructure scores in [0,1]
    max_shelters     : int — budget constraint
    w_cov, w_infra, w_cost : fitness weights (same as GA's fitness)

    Returns
    -------
    chromosome : 1-D int array (N,) — binary shelter placement
    fitness    : float — evaluated fitness score
    """
    n = len(populations)
    total_pop = populations.sum()
    chromosome = np.zeros(n, dtype=np.int8)

    print(f"Running Greedy Baseline (budget = {max_shelters} shelters)...")

    # Pre-compute: coverage_matrix.T @ pop gives "how much population
    # shelter j covers" — but we need *marginal* (uncovered) gain.
    # We'll work with a 'remaining_pop' vector and update it.

    remaining_pop = populations.copy()  # still-uncovered pop per ZIP
    selected_indices = []

    # Convert coverage to CSC for efficient column access
    cov_csc = coverage_matrix.tocsc()

    for k in range(max_shelters):
        # Marginal gain for each candidate = sum of remaining_pop covered
        # cov_csc[:, j].T @ remaining_pop = marginal pop gain of opening j
        # Vectorise: gains = cov_csc.T @ remaining_pop (shape: N,)
        gains = np.array(cov_csc.T.dot(remaining_pop)).flatten()

        # Zero out already-selected shelters
        if selected_indices:
            gains[selected_indices] = -1

        # Add small infrastructure tiebreaker
        gains += infra_scores * 10

        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]

        if best_gain <= 0:
            print(f"  Stopped early at iteration {k} (no improvement).")
            break

        selected_indices.append(best_idx)
        chromosome[best_idx] = 1

        # Zero out the population covered by this new shelter
        # so it doesn't count as marginal gain in future steps
        covered_by_j = np.array(cov_csc[:, best_idx].toarray()).flatten().astype(bool)
        remaining_pop[covered_by_j] = 0.0

        if (k + 1) % 50 == 0 or (k + 1) == max_shelters:
            covered_so_far = total_pop - remaining_pop.sum()
            cov_pct = covered_so_far / total_pop * 100
            print(f"  Greedy iter {k+1}: {cov_pct:.1f}% population covered")

    # Final fitness calculation (same formula as GA)
    selected = np.array(selected_indices)
    if len(selected) == 0:
        return chromosome, 0.0

    covered_mask = np.array(
        coverage_matrix[:, selected].sum(axis=1)
    ).flatten() > 0
    covered_pop = populations[covered_mask].sum()
    coverage_ratio = covered_pop / total_pop if total_pop > 0 else 0

    infra_avg = infra_scores[selected].mean()
    cost_ratio = len(selected) / n

    fitness = w_cov * coverage_ratio + w_infra * infra_avg - w_cost * cost_ratio

    print(f"  Greedy Done: {len(selected)} shelters, "
          f"coverage={coverage_ratio*100:.1f}%, fitness={fitness:.4f}")

    return chromosome, fitness