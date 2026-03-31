"""
main.py — Entry point: load data, preprocess, run GA with best Optuna params
(or defaults), run greedy baseline, compare, and save results.

Usage:
    1. First run Optuna tuning:  python src/optuna_tuning.py --n-trials 30
    2. Then run this:            python src/main.py
       (will auto-load best params from results/optuna_best_params.json)
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure 'src' is importable when running as `python src/main.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_census_data, load_nuclear_targets, load_urban_areas
from src.feature_engineering import features
from src.fitness import FitnessFunction
from src.genetic_algo import GeneticAlgorithm
from src.baseline_greedy import greedy_heuristic

# ── Output dir ───────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Default GA params (used if Optuna best params not found) ──
DEFAULT_GA_PARAMS = {
    "pop_size": 80,
    "generations": 200,
    "mutation_rate": 0.02,
    "crossover_rate": 0.85,
    "tournament_size": 2,
    "target_shelter_ratio": 0.01,
    "elitism_count": 1,
    "adaptive_mutation": True,
    "seed_fraction": 0.25,
    "seed_perturb_swaps": 4,
    "local_search_elites": 3,
    "local_search_steps": 12,
}


def load_best_params() -> dict:
    """Load best hyperparameters from Optuna results, or fall back to defaults."""
    params_path = os.path.join(RESULTS_DIR, "optuna_best_params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            data = json.load(f)
        params = data["best_params"]
        print(f"  Loaded Optuna best params (fitness={data['best_fitness']:.6f})")
        for k, v in params.items():
            print(f"    {k}: {v}")
        return params
    else:
        print("  No Optuna results found — using default GA parameters.")
        print("  Run `python src/optuna_tuning.py` first for tuned params.")
        return DEFAULT_GA_PARAMS.copy()


def ratio_to_fixed_k(n_genes: int, target_ratio: float) -> int:
    """Convert target ratio to an exact shelter budget K for fixed-K GA."""
    k = int(round(float(target_ratio) * n_genes))
    return max(1, min(n_genes, k))


# ═══════════════════════════════════════════════════════════════════════════════
#                                V I S U A L I S A T I O N S
# ═══════════════════════════════════════════════════════════════════════════════

def plot_convergence(ga_history, ga_avg_history, path):
    """Plot convergence curves for the final GA run."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    gens = range(len(ga_history))

    ax1.plot(gens, ga_history, color="#4C72B0", linewidth=1.5, label="Best Fitness")
    ax1.set_title("Best Fitness per Generation", fontsize=14)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(gens, ga_avg_history, color="#DD8452", linewidth=1.5, label="Avg Fitness")
    ax2.set_title("Average Fitness per Generation", fontsize=14)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Convergence plot -> {path}")


def plot_comparison(ga_report, greedy_report, path):
    """Bar chart comparing GA vs Greedy on key metrics."""
    metrics = ["coverage_pct", "infra_score", "fitness"]
    labels = ["Coverage %", "Infra Score", "Fitness"]
    ga_vals = [ga_report[m] for m in metrics]
    gr_vals = [greedy_report[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ga_vals, width, label="GA (Optuna-tuned)", color="#4C72B0")
    bars2 = ax.bar(x + width/2, gr_vals, width, label="Greedy Baseline", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("GA vs Greedy Baseline Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Comparison plot -> {path}")


def plot_shelter_map(zip_df, best_chromosome, path):
    """Scatter plot of all candidate ZIPs with shelters highlighted."""
    fig, ax = plt.subplots(figsize=(14, 9))

    selected = best_chromosome == 1
    not_selected = ~selected

    ax.scatter(zip_df.loc[not_selected, "lon"],
               zip_df.loc[not_selected, "lat"],
               s=1, alpha=0.15, c="gray", label="Candidate ZIPs")
    ax.scatter(zip_df.loc[selected, "lon"],
               zip_df.loc[selected, "lat"],
               s=20, alpha=0.8, c="red", marker="^", label="Selected Shelters")

    ax.set_title("Optimal Shelter Locations (GA)", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Shelter map -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#                                      M A I N
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("   NUCLEAR SHELTER LOCATION OPTIMIZER (UFLP)")
    print("=" * 60)

    # ── 1. Load Data ──
    census_df  = load_census_data()
    targets_df = load_nuclear_targets()
    urban_df   = load_urban_areas()

    # ── 2. Preprocess ──
    prep = features(census_df, targets_df, urban_df, service_radius=50.0)

    # ── 3. Create Fitness Function ──
    fitness_obj = FitnessFunction(
        populations=prep["populations"],
        coverage_matrix=prep["coverage_matrix"],
        infra_scores=prep["infra_scores"],
        w_cov=0.7,
        w_infra=0.2,
        w_cost=0.1,
    )

    # ── 4. Load best hyperparameters ──
    print("\n" + "=" * 60)
    print("   LOADING HYPERPARAMETERS")
    print("=" * 60)
    params = load_best_params()

    # ── 5. Run GA with best params ──
    print("\n" + "=" * 60)
    print("   RUNNING GENETIC ALGORITHM")
    print("=" * 60)

    target_ratio = params.get("target_shelter_ratio", 0.01)
    fixed_k = ratio_to_fixed_k(prep["n_genes"], target_ratio)
    print(f"  Fixed-K budget: {fixed_k} shelters ({target_ratio:.4%} of candidates)")

    print("  Building greedy seed solution for GA initialisation...")
    seed_t0 = time.time()
    greedy_seed_sol, _ = greedy_heuristic(
        prep["populations"],
        prep["coverage_matrix"],
        prep["infra_scores"],
        max_shelters=fixed_k,
    )
    greedy_seed_elapsed = time.time() - seed_t0

    ga_config = {
        "pop_size": int(params.get("pop_size", 80)),
        "generations": int(params.get("generations", 200)),
        "mutation_rate": float(params.get("mutation_rate", 0.02)),
        "crossover_rate": float(params.get("crossover_rate", 0.85)),
        "tournament_size": max(2, min(4, int(params.get("tournament_size", 2)))),
        "target_shelter_ratio": float(params.get("target_shelter_ratio", 0.01)),
        "elitism_count": max(1, min(3, int(params.get("elitism_count", 1)))),
        "adaptive_mutation": bool(params.get("adaptive_mutation", True)),
        "seed_fraction": float(params.get("seed_fraction", 0.25)),
        "seed_perturb_swaps": int(params.get("seed_perturb_swaps", 4)),
        "local_search_elites": int(params.get("local_search_elites", 3)),
        "local_search_steps": int(params.get("local_search_steps", 12)),
        "fixed_k": fixed_k,
    }

    t0 = time.time()
    ga = GeneticAlgorithm(
        n_genes=prep["n_genes"],
        fitness_func=fitness_obj.evaluate,
        pop_size=ga_config["pop_size"],
        generations=ga_config["generations"],
        mutation_rate=ga_config["mutation_rate"],
        crossover_rate=ga_config["crossover_rate"],
        tournament_size=ga_config["tournament_size"],
        target_shelter_ratio=ga_config["target_shelter_ratio"],
        elitism_count=ga_config["elitism_count"],
        adaptive_mutation=ga_config["adaptive_mutation"],
        fixed_k=ga_config["fixed_k"],
        seed_solution=greedy_seed_sol,
        seed_fraction=ga_config["seed_fraction"],
        seed_perturb_swaps=ga_config["seed_perturb_swaps"],
        local_search_elites=ga_config["local_search_elites"],
        local_search_steps=ga_config["local_search_steps"],
        seed=42,
    )
    best_sol, _best_fit = ga.evolve()
    ga_elapsed = time.time() - t0
    ga_report = fitness_obj.detailed_report(best_sol)

    # ── 6. Run Greedy Baseline ──
    print("\n" + "=" * 60)
    print("   GREEDY BASELINE")
    print("=" * 60)

    max_shelters_greedy = ga_report["n_shelters"]
    if max_shelters_greedy == fixed_k:
        print("  Reusing greedy seed run for baseline (same budget).")
        greedy_sol = greedy_seed_sol
        greedy_elapsed = greedy_seed_elapsed
    else:
        t0 = time.time()
        greedy_sol, _greedy_fit = greedy_heuristic(
            prep["populations"],
            prep["coverage_matrix"],
            prep["infra_scores"],
            max_shelters=max_shelters_greedy,
        )
        greedy_elapsed = time.time() - t0
    greedy_report = fitness_obj.detailed_report(greedy_sol)

    # ── 7. Results Summary ──
    print("\n" + "=" * 60)
    print("   FINAL RESULTS")
    print("=" * 60)

    print(f"\n  GA Configuration (effective):")
    for k, v in ga_config.items():
        print(f"    {k}: {v}")

    print(f"\n  GA Performance (time: {ga_elapsed:.1f}s):")
    for k, v in ga_report.items():
        print(f"    {k}: {v}")

    print(f"\n  Greedy Baseline (time: {greedy_elapsed:.1f}s):")
    for k, v in greedy_report.items():
        print(f"    {k}: {v}")

    print(f"\n  {'─'*40}")
    if ga_report["fitness"] > greedy_report["fitness"]:
        improvement = ((ga_report["fitness"] - greedy_report["fitness"])
                       / abs(greedy_report["fitness"]) * 100
                       if greedy_report["fitness"] != 0 else float("inf"))
        print(f"  ✓ GA outperformed Greedy by {improvement:.2f}%")
    else:
        print(f"  ✗ Greedy performed equally or better than GA.")

    # ── 8. Save Outputs ──
    print("\n  Saving outputs...")

    plot_convergence(ga.history, ga.avg_history,
                     os.path.join(RESULTS_DIR, "convergence_plot.png"))
    plot_comparison(ga_report, greedy_report,
                    os.path.join(RESULTS_DIR, "comparison_plot.png"))
    plot_shelter_map(prep["zip_codes"], best_sol,
                     os.path.join(RESULTS_DIR, "shelter_map.png"))

    # Save final results JSON
    final_results = {
        "ga_params": params,
        "ga_report": ga_report,
        "ga_time_sec": round(ga_elapsed, 1),
        "greedy_report": greedy_report,
        "greedy_time_sec": round(greedy_elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    # Convergence CSV
    conv_df = pd.DataFrame({
        "generation": range(len(ga.history)),
        "best_fitness": ga.history,
        "avg_fitness": ga.avg_history,
    })
    conv_df.to_csv(os.path.join(RESULTS_DIR, "convergence_data.csv"), index=False)

    print(f"\n{'='*60}")
    print("   ALL DONE — check the results/ directory")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()