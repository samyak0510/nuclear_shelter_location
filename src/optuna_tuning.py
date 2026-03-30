"""
optuna_tuning.py — Optuna-based hyperparameter optimisation for the GA.

Run separately from main.py:
    python src/optuna_tuning.py

Saves:
  - results/optuna_study.db          (SQLite study for resume/analysis)
  - results/optuna_best_params.json  (best hyperparameters)
  - results/optuna_report.txt        (human-readable summary)
  - results/optuna_importance.png    (hyperparameter importance plot)
  - results/optuna_history.png       (optimisation history plot)
"""

import sys
import os
import json
import time

# Ensure parent dir is on path so `src.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_census_data, load_nuclear_targets, load_urban_areas
from src.preprocessing import preprocess
from src.fitness import FitnessFunction
from src.genetic_algo import GeneticAlgorithm

# ── Output dir ───────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Global data (loaded once, reused across trials)
# ═══════════════════════════════════════════════════════════════════════════════
_PREP = None
_FITNESS_OBJ = None


def _load_data():
    """Load and preprocess data once."""
    global _PREP, _FITNESS_OBJ
    if _PREP is not None:
        return

    print("=" * 60)
    print("  Loading & preprocessing data for Optuna study...")
    print("=" * 60)

    census_df  = load_census_data()
    targets_df = load_nuclear_targets()
    urban_df   = load_urban_areas()

    _PREP = preprocess(census_df, targets_df, urban_df, service_radius=50.0)

    _FITNESS_OBJ = FitnessFunction(
        populations=_PREP["populations"],
        coverage_matrix=_PREP["coverage_matrix"],
        infra_scores=_PREP["infra_scores"],
        w_cov=0.7,
        w_infra=0.2,
        w_cost=0.1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ═══════════════════════════════════════════════════════════════════════════════

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.

    Samples GA hyperparameters, runs the GA for a fixed number of
    generations, and returns the best fitness achieved.
    """
    _load_data()

    # ── Sample hyperparameters ──
    pop_size = trial.suggest_int("pop_size", 40, 120, step=10)
    generations = trial.suggest_int("generations", 80, 300, step=20)
    mutation_rate = trial.suggest_float("mutation_rate", 0.005, 0.05, log=True)
    crossover_rate = trial.suggest_float("crossover_rate", 0.70, 0.95)
    tournament_size = trial.suggest_int("tournament_size", 2, 7)
    target_shelter_ratio = trial.suggest_float("target_shelter_ratio", 0.003, 0.02, log=True)
    elitism_count = trial.suggest_int("elitism_count", 1, 5)
    adaptive_mutation = trial.suggest_categorical("adaptive_mutation", [True, False])

    # ── Run GA ──
    ga = GeneticAlgorithm(
        n_genes=_PREP["n_genes"],
        fitness_func=_FITNESS_OBJ.evaluate,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        target_shelter_ratio=target_shelter_ratio,
        elitism_count=elitism_count,
        adaptive_mutation=adaptive_mutation,
        seed=trial.number,  # different seed per trial for diversity
    )

    best_sol, best_fit = ga.evolve()

    # Report detailed metrics as user attributes
    report = _FITNESS_OBJ.detailed_report(best_sol)
    trial.set_user_attr("n_shelters", report["n_shelters"])
    trial.set_user_attr("coverage_pct", report["coverage_pct"])
    trial.set_user_attr("infra_score", report["infra_score"])

    return best_fit


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_tuning(n_trials: int = 30, study_name: str = "ga_uflp_optuna"):
    """
    Run the Optuna study.

    Parameters
    ----------
    n_trials   : number of Optuna trials
    study_name : name of the study (also used for the SQLite DB)
    """
    db_path = os.path.join(RESULTS_DIR, f"{study_name}.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,   # resume from previous run
    )

    print(f"\n{'='*60}")
    print(f"  OPTUNA HYPERPARAMETER TUNING")
    print(f"  Trials: {n_trials} | Study: {study_name}")
    print(f"  DB: {db_path}")
    print(f"{'='*60}\n")

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    # ── Best trial ──
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  BEST TRIAL: #{best.number}")
    print(f"  Fitness:     {best.value:.6f}")
    print(f"  Coverage:    {best.user_attrs.get('coverage_pct', '?')}%")
    print(f"  Shelters:    {best.user_attrs.get('n_shelters', '?')}")
    print(f"{'='*60}")
    print(f"  Best parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Save best params ──
    best_params_path = os.path.join(RESULTS_DIR, "optuna_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({
            "best_fitness": best.value,
            "best_params": best.params,
            "coverage_pct": best.user_attrs.get("coverage_pct"),
            "n_shelters": best.user_attrs.get("n_shelters"),
            "infra_score": best.user_attrs.get("infra_score"),
            "n_trials": len(study.trials),
        }, f, indent=2)
    print(f"  Saved best params -> {best_params_path}")

    # ── Save human-readable report ──
    report_path = os.path.join(RESULTS_DIR, "optuna_report.txt")
    with open(report_path, "w") as f:
        f.write("OPTUNA HYPERPARAMETER TUNING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Study:         {study_name}\n")
        f.write(f"Total trials:  {len(study.trials)}\n")
        f.write(f"Total time:    {elapsed:.1f}s\n\n")
        f.write(f"BEST TRIAL (#{best.number})\n")
        f.write(f"  Fitness:    {best.value:.6f}\n")
        f.write(f"  Coverage:   {best.user_attrs.get('coverage_pct', '?')}%\n")
        f.write(f"  Shelters:   {best.user_attrs.get('n_shelters', '?')}\n")
        f.write(f"  Infra:      {best.user_attrs.get('infra_score', '?')}\n\n")
        f.write("  Parameters:\n")
        for k, v in best.params.items():
            f.write(f"    {k}: {v}\n")

        f.write(f"\n\nALL TRIALS (sorted by fitness):\n")
        f.write("-" * 50 + "\n")
        sorted_trials = sorted(study.trials,
                                key=lambda t: t.value if t.value is not None else -999,
                                reverse=True)
        for t in sorted_trials:
            val = f"{t.value:.6f}" if t.value is not None else "FAILED"
            cov = t.user_attrs.get("coverage_pct", "?")
            f.write(f"  Trial {t.number:>3}: fitness={val}  "
                    f"cov={cov}%  params={t.params}\n")

    print(f"  Saved report -> {report_path}")

    # ── Plots ──
    try:
        fig = plot_optimization_history(study)
        plt.tight_layout()
        hist_path = os.path.join(RESULTS_DIR, "optuna_history.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"  Saved history plot -> {hist_path}")
    except Exception as e:
        print(f"  Could not plot history: {e}")

    try:
        fig = plot_param_importances(study)
        plt.tight_layout()
        imp_path = os.path.join(RESULTS_DIR, "optuna_importance.png")
        plt.savefig(imp_path, dpi=150)
        plt.close()
        print(f"  Saved importance plot -> {imp_path}")
    except Exception as e:
        print(f"  Could not plot importances: {e}")

    return study


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optuna HP tuning for GA-UFLP")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--study-name", type=str, default="ga_uflp_optuna",
                        help="Study name (default: ga_uflp_optuna)")
    args = parser.parse_args()

    run_tuning(n_trials=args.n_trials, study_name=args.study_name)
