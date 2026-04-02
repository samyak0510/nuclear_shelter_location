"""
data_loader.py — Load Census, Nuclear Target, and Urban Area data.

Delegates all cleaning, aggregation, and normalization to preprocessing.py.
"""

import pandas as pd
from pathlib import Path
from src.preprocessing import clean_census_data, clean_nuclear_targets, clean_urban_areas

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR     = BASE_DATA_DIR / "processed"

CENSUS_2010_PATH     = BASE_DATA_DIR / "raw/population_by_zip_2010.csv"
CENSUS_2000_PATH     = BASE_DATA_DIR / "raw/population_by_zip_2000.csv"
NUCLEAR_TARGETS_PATH = BASE_DATA_DIR / "raw/usa_nuclear_targets.csv"
URBAN_AREAS_PATH     = BASE_DATA_DIR / "raw/usa_urban_areas.csv"


# ── Public loaders ───────────────────────────────────────────────────────────

def load_census_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Loads 2010 ZIP-level population data, aggregates across age/gender rows,
    and attaches lat/lon coordinates via pgeocode.

    Returns
    -------
    DataFrame with columns: zip_code, population, lat, lon
    """
    census_cache_path = CACHE_DIR / "census_processed.csv"
    if use_cache and census_cache_path.exists():
        print("Loading Census Data from cache...")
        census_df = pd.read_csv(census_cache_path, dtype={"zip_code": str})
        print(f"  {len(census_df):,} ZIP codes loaded from cache.")
        return census_df

    print("Loading Census Data (this may take a minute on first run)...")

    raw_census = pd.read_csv(CENSUS_2010_PATH, dtype=str)
    census_df = clean_census_data(raw_census)

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    census_df.to_csv(census_cache_path, index=False)
    print(f"  Cached to {census_cache_path}")

    return census_df


def load_nuclear_targets() -> pd.DataFrame:
    """
    Loads US nuclear targets from the CSV file.

    Returns
    -------
    DataFrame with columns: name, lat, lon, yield_kt, burst_type
    """
    print("Loading Nuclear Targets...")

    raw_targets = pd.read_csv(NUCLEAR_TARGETS_PATH)
    targets_df = clean_nuclear_targets(raw_targets)

    print(f"  {len(targets_df):,} nuclear targets loaded.")
    print(f"  Yield range: {targets_df['yield_kt'].min():.0f} – {targets_df['yield_kt'].max():.0f} kt")
    print(f"  Burst types: {targets_df['burst_type'].value_counts().to_dict()}")
    return targets_df


def load_urban_areas() -> pd.DataFrame:
    """
    Loads urban area centroids and land area metadata.

    Returns
    -------
    DataFrame with columns: name, lat, lon
    """
    print("Loading Urban Areas...")

    raw_urban = pd.read_csv(URBAN_AREAS_PATH, dtype=str)
    urban_df = clean_urban_areas(raw_urban)

    print(f"  {len(urban_df):,} urban areas loaded.")
    return urban_df


def load_all() -> dict:
    """Loads all datasets and returns them in a single dict."""
    return {
        "census":      load_census_data(),
        "targets":     load_nuclear_targets(),
        "urban_areas": load_urban_areas(),
    }


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_all()

    print("\n── Census (first 5 rows) ──")
    print(data["census"].head())

    print("\n── Targets (first 5 rows) ──")
    print(data["targets"].head())

    print("\n── Urban Areas (first 5 rows) ──")
    print(data["urban_areas"].head())
