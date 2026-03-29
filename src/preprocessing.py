"""
preprocessing.py — Preprocess loaded data into structures ready for
the Genetic Algorithm: yield-scaled safety mask, infrastructure scores,
sparse coverage matrix.
"""

import numpy as np
import pandas as pd
from src.utils import haversine_distance_matrix
from src.blast_radius import blast_radius_miles


def compute_safety_mask(zip_lats, zip_lons,
                        target_lats, target_lons,
                        target_yields_kt, target_burst_types,
                        threshold="5psi"):
    """
    Returns a boolean mask of shape (N_zips,).
    True  = ZIP is SAFE (outside all blast zones).
    False = ZIP is UNSAFE (within yield-scaled exclusion radius of any target).

    Uses per-target blast radius from the Hopkinson-Cranz cube-root scaling law
    instead of a fixed 15-mile exclusion.
    """
    print(f"  Computing safety mask (yield-scaled, threshold={threshold})...")

    # Compute per-target blast radii
    n_targets = len(target_lats)
    radii = np.array([
        blast_radius_miles(target_yields_kt[j], threshold, target_burst_types[j])
        for j in range(n_targets)
    ])
    print(f"    Blast radii range: {radii.min():.2f} – {radii.max():.2f} miles")

    # Distance matrix: (N_zips, N_targets)
    dist = haversine_distance_matrix(zip_lats, zip_lons, target_lats, target_lons)

    # A zip is unsafe if its distance to ANY target < that target's blast radius
    # dist[i, j] < radii[j]  →  broadcast radii as (1, M)
    inside_blast = dist < radii[None, :]   # (N_zips, N_targets)
    mask = ~inside_blast.any(axis=1)       # safe = not inside any blast zone

    n_unsafe = (~mask).sum()
    print(f"    {n_unsafe:,} ZIP codes inside blast zones (excluded).")
    print(f"    {mask.sum():,} ZIP codes are safe candidates.")
    return mask


def compute_infrastructure_scores(zip_lats, zip_lons, urban_lats, urban_lons,
                                  decay_miles=50.0):
    """
    Compute an infrastructure proximity score for each ZIP code in [0, 1].
    Score = max over all urban areas of exp(-dist / decay_miles).
    Closer to an urban area → higher score.
    """
    print("  Computing infrastructure accessibility scores...")

    # Distance matrix: (N_zips, N_urban)
    dist = haversine_distance_matrix(zip_lats, zip_lons, urban_lats, urban_lons)
    # Exponential decay — closest urban area dominates
    scores = np.exp(-dist / decay_miles).max(axis=1)
    print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    return scores


def compute_coverage_distances(zip_lats, zip_lons, service_radius_miles=50.0):
    """
    Build a sparse boolean coverage matrix:
        coverage[i, j] = True if zip i can be served by a shelter at zip j.

    Processes in chunks to avoid memory blow-up for ~30K+ zip codes.
    """
    from scipy import sparse

    n = len(zip_lats)
    print(f"  Building coverage adjacency (N={n:,}, radius={service_radius_miles} mi)...")

    # Process in chunks to avoid memory blow-up
    chunk_size = 2000
    rows, cols = [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # (chunk, N) distance sub-matrix
        dist_chunk = haversine_distance_matrix(
            zip_lats[start:end], zip_lons[start:end],
            zip_lats, zip_lons
        )
        r, c = np.where(dist_chunk <= service_radius_miles)
        rows.append(r + start)
        cols.append(c)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    coverage = sparse.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(n, n)
    )
    nnz = coverage.nnz
    density = nnz / (n * n) * 100
    print(f"    Coverage matrix: {nnz:,} non-zero entries ({density:.2f}% density).")
    return coverage


def preprocess(census_df, targets_df, urban_df, service_radius=50.0,
               blast_threshold="5psi"):
    """
    Master preprocessing function.

    Parameters
    ----------
    census_df       : DataFrame with zip_code, population, lat, lon
    targets_df      : DataFrame with name, lat, lon, yield_kt, burst_type
    urban_df        : DataFrame with name, lat, lon
    service_radius  : coverage radius in miles for the UFLP
    blast_threshold : overpressure threshold for blast zone exclusion

    Returns
    -------
    dict with keys:
        zip_codes       : DataFrame (filtered to safe only)
        populations     : 1-D array
        safety_mask     : boolean array (on original census)
        infra_scores    : 1-D array (for safe zips only)
        coverage_matrix : sparse bool matrix (safe-zip × safe-zip)
        n_genes         : int (number of candidate sites = safe zips)
    """
    print("\n=== PREPROCESSING ===")

    zip_lats = census_df["lat"].values.astype(np.float64)
    zip_lons = census_df["lon"].values.astype(np.float64)

    target_lats = targets_df["lat"].values.astype(np.float64)
    target_lons = targets_df["lon"].values.astype(np.float64)
    target_yields = targets_df["yield_kt"].values.astype(np.float64)
    target_burst_types = targets_df["burst_type"].tolist()

    urban_lats = urban_df["lat"].values.astype(np.float64)
    urban_lons = urban_df["lon"].values.astype(np.float64)

    # 1. Safety mask (yield-scaled per target)
    safety_mask = compute_safety_mask(
        zip_lats, zip_lons,
        target_lats, target_lons,
        target_yields, target_burst_types,
        threshold=blast_threshold,
    )

    # 2. Filter to safe ZIP codes only (GA candidates)
    safe_df = census_df[safety_mask].reset_index(drop=True)
    safe_lats = safe_df["lat"].values.astype(np.float64)
    safe_lons = safe_df["lon"].values.astype(np.float64)
    populations = safe_df["population"].values.astype(np.float64)

    # 3. Infrastructure scores for safe zips
    infra_scores = compute_infrastructure_scores(safe_lats, safe_lons,
                                                 urban_lats, urban_lons)

    # 4. Coverage adjacency matrix (among safe zips)
    coverage_matrix = compute_coverage_distances(safe_lats, safe_lons,
                                                 service_radius)

    n_genes = len(safe_df)
    total_pop = populations.sum()
    print(f"\n  Summary: {n_genes:,} candidate sites, "
          f"total reachable population = {total_pop:,.0f}")

    return {
        "zip_codes":       safe_df,
        "populations":     populations,
        "safety_mask":     safety_mask,
        "infra_scores":    infra_scores,
        "coverage_matrix": coverage_matrix,
        "n_genes":         n_genes,
    }
