"""
data_loader.py — Load and preprocess Census, Nuclear Target, and Urban Area data.

Census CSVs are disaggregated by age/gender; we aggregate total population per
zip code and attach lat/lon coordinates via pgeocode.  Road-network data is
intentionally excluded.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pgeocode

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data/raw"
PROCESSED_DIR = DATA_DIR / "processed"

POP_2000_PATH    = DATA_DIR / "population_by_zip_2000.csv"
POP_2010_PATH    = DATA_DIR / "population_by_zip_2010.csv"
TARGETS_PATH     = DATA_DIR / "usa_nuclear_targets.csv"
URBAN_AREAS_PATH = DATA_DIR / "usa_urban_areas.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip all column names."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


# ── Public loaders ───────────────────────────────────────────────────────────

def load_census_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Loads 2010 ZIP-level population data, aggregates across age/gender rows,
    and attaches lat/lon coordinates via pgeocode.

    Returns
    -------
    DataFrame with columns:
        zip_code   (str, zero-padded 5-digit)
        population (int, total population in that ZIP)
        lat        (float)
        lon        (float)
    """
    cache_path = PROCESSED_DIR / "census_processed.csv"
    if use_cache and cache_path.exists():
        print("Loading Census Data from cache...")
        df = pd.read_csv(cache_path, dtype={"zip_code": str})
        print(f"  {len(df):,} ZIP codes loaded from cache.")
        return df

    print("Loading Census Data (this may take a minute on first run)...")

    # ── Read raw 2010 data ──
    raw = pd.read_csv(POP_2010_PATH, dtype=str)
    raw.columns = raw.columns.str.strip().str.lower()

    raw["population"] = pd.to_numeric(raw["population"], errors="coerce")
    raw["zipcode"] = raw["zipcode"].str.strip().str.zfill(5)

    # Aggregate: sum population across all age/gender rows per zip
    agg = (
        raw.groupby("zipcode", as_index=False)["population"]
        .sum()
        .rename(columns={"zipcode": "zip_code"})
    )

    # Drop rows with zero or null population
    agg = agg[agg["population"] > 0].copy()
    agg["population"] = agg["population"].astype(int)

    print(f"  Aggregated {len(agg):,} ZIP codes from raw data.")

    # ── Attach lat/lon via pgeocode ──
    print("  Geocoding ZIP codes (pgeocode)...")
    nomi = pgeocode.Nominatim("US")
    geo = nomi.query_postal_code(agg["zip_code"].tolist())
    agg["lat"] = geo["latitude"].values
    agg["lon"] = geo["longitude"].values

    # Drop ZIP codes that couldn't be geocoded
    before = len(agg)
    agg = agg.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"  {before - len(agg)} ZIP codes dropped (no coordinates).")
    print(f"  {len(agg):,} ZIP codes ready.")

    # ── Cache ──
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(cache_path, index=False)
    print(f"  Cached to {cache_path}")

    return agg


def load_nuclear_targets() -> pd.DataFrame:
    """
    Loads US nuclear targets from the CSV file.

    Returns
    -------
    DataFrame with columns: name, lat, lon, category, yield_kt, burst_type
    """
    from src.blast_radius import parse_yield, normalise_burst_type

    print("Loading Nuclear Targets...")

    df = pd.read_csv(TARGETS_PATH)
    df.columns = df.columns.str.strip().str.lower()

    # Rename
    rename_map = {}
    if "target" in df.columns:
        rename_map["target"] = "name"
    if "lng" in df.columns:
        rename_map["lng"] = "lon"
    df = df.rename(columns=rename_map)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Parse yield and burst type for blast radius calculations
    if "yield" in df.columns:
        df["yield_kt"] = df["yield"].apply(parse_yield)
    else:
        df["yield_kt"] = 500.0  # default assumption

    if "type" in df.columns:
        df["burst_type"] = df["type"].apply(normalise_burst_type)
    else:
        df["burst_type"] = "Surface Burst"

    keep = ["name", "lat", "lon", "yield_kt", "burst_type"]
    if "category" in df.columns:
        keep.append("category")

    print(f"  {len(df):,} nuclear targets loaded.")
    print(f"  Yield range: {df['yield_kt'].min():.0f} – {df['yield_kt'].max():.0f} kt")
    print(f"  Burst types: {df['burst_type'].value_counts().to_dict()}")
    return df[keep]


def load_urban_areas() -> pd.DataFrame:
    """
    Loads urban area centroids and land area metadata.

    Returns
    -------
    DataFrame with columns: name, lat, lon
    """
    print("Loading Urban Areas...")

    df = pd.read_csv(URBAN_AREAS_PATH, dtype=str)
    df.columns = df.columns.str.strip().str.lower()

    # Resolve column names
    col_map = {}
    for alias in ["name10", "namelsad10", "name"]:
        if alias in df.columns:
            col_map[alias] = "name"
            break
    for alias in ["intptlat10", "lat", "latitude"]:
        if alias in df.columns:
            col_map[alias] = "lat"
            break
    for alias in ["intptlon10", "lon", "lng", "longitude"]:
        if alias in df.columns:
            col_map[alias] = "lon"
            break

    df = df.rename(columns=col_map)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    print(f"  {len(df):,} urban areas loaded.")
    return df[["name", "lat", "lon"]]


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