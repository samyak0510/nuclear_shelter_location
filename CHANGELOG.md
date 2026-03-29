# Changelog & Code Walkthrough — Nuclear Shelter Location Optimizer

## Quick Answer: Yes, We ARE Implementing GA


## What Changed (Before → After)

### 1. `data_loader.py` — **Complete Rewrite**

| Aspect | Before | After |
|---|---|---|
| Census data | Expected pre-aggregated CSV with `population`, `lat`, `lon` per ZIP | Raw data has 1.6M rows (one per age/gender combo) — now **aggregates** population per ZIP correctly |
| Coordinates | Assumed CSV had `lat`/`lon` columns | Census CSV has **no coordinates** — now uses `pgeocode` library to geocode all 33K ZIP codes |
| Nuclear targets | Loaded only name, lat, lon | Now also parses **Yield** (e.g. "500kt" → 500.0) and **Type** ("Air Burst" / "Surface Burst") |
| Caching | None | First-run results cached to `data/processed/census_processed.csv` — subsequent runs load instantly |

**Final data shapes:**
- Census: **32,832 ZIP codes** with `zip_code, population, lat, lon`
- Nuclear targets: **1,087 targets** with `name, lat, lon, yield_kt, burst_type, category`
- Urban areas: **3,601 areas** with `name, lat, lon`

---

### 2. `blast_radius.py` — **NEW FILE**

Uses the **Hopkinson-Cranz cube-root scaling law** from Glasstone & Dolan (1977):

```
R(Y) = R_ref × Y^(1/3) × burst_factor
```

| Feature | Detail |
|---|---|
| Reference radii | 5 psi (severe damage): 0.292 mi at 1kt; 1 psi (light damage): 0.932 mi at 1kt |
| Air burst multiplier | 1.25× (overpressure extends ~25% farther than surface burst) |
| `parse_yield()` | Parses "500kt", "1.2Mt", "800 Kt" into numeric kilotons |
| `normalise_burst_type()` | Handles non-breaking spaces in dataset ("Air\xa0Burst" → "Air Burst") |

**Result:** Instead of a **fixed 15-mile exclusion radius** for every target, each target now gets a **yield-specific radius**:
- 100 kt Air Burst → **1.69 miles**
- 500 kt Air Burst → **2.90 miles**
- 1000 kt Surface Burst → **2.92 miles**
- 2000 kt Air Burst → **4.60 miles**

This is far more realistic than the original 15-mile flat assumption.

---

### 3. `preprocessing.py` — **NEW FILE**

Centralises all preprocessing into one `preprocess()` function that produces everything the GA and Baseline need:

| Step | What it does |
|---|---|
| **Safety mask** | Uses `blast_radius.py` to compute per-target exclusion radii. Vectorised: builds a (32K × 1087) distance matrix, then checks `dist[i,j] < radius[j]` |
| **Filter** | Removes 1,870 ZIPs inside blast zones → **30,962 safe candidates** |
| **Infrastructure scores** | For each safe ZIP, computes `max(exp(-dist / 50mi))` over all 3,601 urban areas. Score in [0, 1]. Closer to urban area = higher score |
| **Coverage matrix** | Sparse boolean matrix (30,962 × 30,962). `coverage[i,j] = True` if ZIP i can be served by a shelter at ZIP j (within 50-mile service radius). Built in chunks to avoid 8GB memory blowup. ~4.8M non-zero entries (0.50% density) |

---

### 4. `utils.py` — **Added Vectorised Distance**

| Function | Purpose |
|---|---|
| `haversine_distance(lat1,lon1,lat2,lon2)` | Same as before — scalar/array haversine in miles |
| `haversine_distance_matrix(lats1,lons1,lats2,lons2)` | **NEW** — Computes full (N × M) pairwise distance matrix using NumPy broadcasting. Critical for performance |

---

### 5. `fitness.py` — **Complete Rewrite**

| Aspect | Before | After |
|---|---|---|
| Coverage calculation | O(N×M) nested Python loops (unusable for 30K zips) | Uses sparse coverage matrix: `coverage[:, selected].sum(axis=1) > 0` — near-instant |
| Normalisation | Raw population counts + arbitrary infra scaling | All components normalised to [0,1] with tunable weights |
| Fitness formula | `pop/1000 + infra*100` (arbitrary) | `0.7 × coverage_ratio + 0.2 × infra_score - 0.1 × cost_ratio` |
| Cost penalty | None | Penalises using too many shelters (all else equal, fewer is better) |
| `detailed_report()` | Did not exist | Returns breakdown: n_shelters, covered_pop, coverage_pct, infra_score, fitness |

**Fitness weights:**
- **w_cov = 0.7** — Population coverage is primary objective
- **w_infra = 0.2** — Infrastructure accessibility matters
- **w_cost = 0.1** — Slight penalty for building too many shelters

---

### 6. `genetic_algo.py` — **Major Refinements**

| Feature | Before | After |
|---|---|---|
| Population init | `np.random.randint(0,2)` — ~50% genes turned on (absurd: would build 15K shelters) | Smart init: each gene is 1 with probability `target_shelter_ratio` (~1%) |
| Selection | Tournament size fixed at 2 | Configurable tournament size (2–7) |
| Elitism | None — best solution could be lost | Top-k individuals (default 2) survive unchanged to next generation |
| Mutation | Fixed rate | **Adaptive**: decays linearly + stagnation boost (3× base rate if no improvement for 15 gens) |
| Data types | `int` arrays | `int8` arrays (4× less memory for 30K-length chromosomes) |
| Progress | `print()` per generation | `tqdm` progress bar with real-time best/avg/mutation/stagnation stats |
| Stagnation detection | None | Tracks generations without improvement; triggers mutation boost |

---

### 7. `optuna_tuning.py` — **NEW FILE**

Replaces the hardcoded sweep with **Optuna Bayesian optimisation**:

| Feature | Detail |
|---|---|
| Sampled params | `pop_size`, `generations`, `mutation_rate`, `crossover_rate`, `tournament_size`, `target_shelter_ratio`, `elitism_count`, `adaptive_mutation` |
| Data loading | Loads once globally, reused across all trials |
| Storage | SQLite DB (`results/ga_uflp_optuna.db`) — can resume from where you left off |
| Outputs | `optuna_best_params.json`, `optuna_report.txt`, `optuna_history.png`, `optuna_importance.png` |
| Usage | `python src/optuna_tuning.py --n-trials 30` |

**5-trial results (so far):**

| Trial | Fitness | Coverage | Shelters | Key insight |
|---|---|---|---|---|
| #4 (best) | **0.8522** | **99.77%** | 2,947 | Low mutation (0.005), tournament=5, moderate pop |
| #2 | 0.8514 | 99.65% | — | Large pop (110), few gens (120) |
| #1 | 0.8507 | 99.63% | — | Largest pop (120), many gens (260) |
| #0 | 0.8501 | 99.36% | — | Higher mutation with adaptive |
| #3 | 0.8492 | 99.75% | — | Tournament=2 (weakest selection pressure) |

---

### 8. `baseline.py` — **Vectorised Greedy**

| Aspect | Before | After |
|---|---|---|
| Marginal gain | O(N²) nested Python loops per iteration | `cov_csc.T @ remaining_pop` — one sparse matrix-vector multiply per iteration (~1000× faster) |
| Data source | Re-implemented safety check, used raw DataFrames | Takes same preprocessed arrays as GA (populations, coverage_matrix, infra_scores) |
| Fitness | Duplicated calculation | Uses same normalised formula as GA for fair comparison |

---

### 9. `main.py` — **Simplified**

| Aspect | Before | After |
|---|---|---|
| HP tuning | 8 hardcoded sweep configs baked into main | **Removed**. Now loads Optuna best params from JSON |
| Flow | Load → sweep → compare | Load → Preprocess → Load best params → Run GA → Run Greedy → Compare → Save plots/results |
| Fallback | — | If no Optuna JSON found, uses sensible defaults |

---

## Code Flow Diagram

```
1. python src/optuna_tuning.py --n-trials 30    (run FIRST, separately)
   │
   ├── Loads data once (census + targets + urban)
   ├── Preprocesses once (safety mask, infra scores, coverage matrix)
   ├── For each Optuna trial:
   │   ├── Samples 8 hyperparameters
   │   ├── Creates GeneticAlgorithm instance
   │   ├── Runs ga.evolve()
   │   └── Returns best_fitness to Optuna
   └── Saves: optuna_best_params.json, plots, DB

2. python src/main.py                           (run AFTER tuning)
   │
   ├── 1. Load Data
   │   ├── load_census_data()    → 32,832 ZIPs with pop + lat/lon
   │   ├── load_nuclear_targets() → 1,087 targets with yield + burst type
   │   └── load_urban_areas()    → 3,601 urban area centroids
   │
   ├── 2. Preprocess
   │   ├── Yield-scaled safety mask (blast_radius.py)
   │   ├── Filter to 30,962 safe candidates
   │   ├── Infrastructure scores (proximity to urban areas)
   │   └── Sparse coverage matrix (50-mi service radius)
   │
   ├── 3. Load best Optuna params from JSON
   │
   ├── 4. Run GA with those params
   │   ├── 220 generations, pop_size=40
   │   ├── Tournament selection (k=5)
   │   ├── Uniform crossover (rate=0.75)
   │   ├── Bit-flip mutation (rate=0.005)
   │   └── Returns: best_chromosome, best_fitness
   │
   ├── 5. Run Greedy Baseline (same # shelters as GA)
   │   └── Iteratively picks ZIP covering most uncovered population
   │
   ├── 6. Compare & Print Results
   │   ├── GA fitness vs Greedy fitness
   │   └── Coverage %, shelter count, infra score
   │
   └── 7. Save Outputs
       ├── convergence_plot.png
       ├── comparison_plot.png
       ├── shelter_map.png
       ├── final_results.json
       └── convergence_data.csv
```

---

## Current Status

| Item | Status |
|---|---|
| Data loading & preprocessing | **Done** — tested, working, cached |
| Blast radius calculator | **Done** — yield-scaled per target |
| Fitness function (vectorised) | **Done** — sparse matrix, normalised |
| Genetic Algorithm (refined) | **Done** — elitism, adaptive mutation, smart init |
| Optuna tuning | **Done** (5 trials) — best fitness **0.8522** (99.77% coverage) |
| Greedy baseline (vectorised) | **Done** — needs final run via `main.py` |
| Final comparison run | **Pending** — run `python src/main.py` |
| Road network data | **Skipped** (per your instruction) |

## Next Steps

1. **Run more Optuna trials** if desired: `python src/optuna_tuning.py --n-trials 25` (resumes from existing 5)
2. **Run final comparison**: `python src/main.py` (uses best Optuna params, generates all plots)
3. Results will be in `results/` directory
