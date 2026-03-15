print("Script started")

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

POP_2000_PATH    = DATA_DIR / "population_by_zip_2000.csv"
POP_2010_PATH    = DATA_DIR / "population_by_zip_2010.csv"
TARGETS_PATH     = DATA_DIR / "us_nuclear_targets.xlsx"
URBAN_AREAS_PATH = DATA_DIR / "Urban_Areas.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip all column names."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def _resolve_col(df: pd.DataFrame, aliases: list[str], target: str) -> pd.DataFrame:
    """Rename the first matching alias to target."""
    for alias in aliases:
        if alias in df.columns:
            return df.rename(columns={alias: target})
    return df


# ── Public loaders ────────────────────────────────────────────────────────────

def load_census_data() -> pd.DataFrame:
    """
    Loads and merges 2000 + 2010 ZIP-level population data.

    Returns
    -------
    DataFrame with columns:
        zip_code (str, zero-padded)
        population_2000 (float)
        population_2010 (float)
        lat (float)
        lon (float)
    """
    print("Loading Census Data...")

    # ── 2000 ──
    df00 = _normalise_cols(pd.read_csv(POP_2000_PATH, dtype=str))
    df00 = _resolve_col(df00, ["zipcode", "zip", "zcta"], "zip_code")
    df00 = _resolve_col(df00, ["population", "pop", "total_population"], "population_2000")
    df00["zip_code"]        = df00["zip_code"].str.zfill(5)
    df00["population_2000"] = pd.to_numeric(df00["population_2000"], errors="coerce")
    df00 = df00[["zip_code", "population_2000"]].dropna()

    # ── 2010 ──
    df10 = _normalise_cols(pd.read_csv(POP_2010_PATH, dtype=str))
    df10 = _resolve_col(df10, ["zipcode", "zip", "zcta"], "zip_code")
    df10 = _resolve_col(df10, ["population", "pop", "total_population"], "population_2010")
    df10 = _resolve_col(df10, ["lat", "latitude", "y"], "lat")
    df10 = _resolve_col(df10, ["lon", "lng", "longitude", "x"], "lon")
    df10["zip_code"]        = df10["zip_code"].str.zfill(5)
    df10["population_2010"] = pd.to_numeric(df10["population_2010"], errors="coerce")

    keep = ["zip_code", "population_2010"]
    for col in ["lat", "lon"]:
        if col in df10.columns:
            df10[col] = pd.to_numeric(df10[col], errors="coerce")
            keep.append(col)
    df10 = df10[keep].dropna(subset=["zip_code", "population_2010"])

    # ── Merge ──
    df = pd.merge(df00, df10, on="zip_code", how="outer")

    print(f"  {len(df):,} ZIP codes loaded (2000 + 2010 merged).")
    return df.reset_index(drop=True)


def load_urban_targets() -> list[dict]:
    """
    Loads US nuclear targets from Excel file.

    Returns
    -------
    List of dicts, each with keys: name, lat, lon
    (matches the original skeleton's return format exactly)
    """
    print("Loading Urban Targets...")

    df = _normalise_cols(pd.read_excel(TARGETS_PATH))
    df = _resolve_col(df, ["city", "target", "location", "place"], "name")
    df = _resolve_col(df, ["lat", "latitude", "y"], "lat")
    df = _resolve_col(df, ["lon", "lng", "longitude", "x"], "lon")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    targets = df[["name", "lat", "lon"]].to_dict(orient="records")

    print(f"  {len(targets):,} nuclear targets loaded.")
    return targets


def load_infrastructure_data() -> pd.DataFrame:
    """
    Loads urban area data (used as a proxy for infrastructure/accessibility).

    Returns
    -------
    DataFrame with columns:
        name (str)
        lat (float)
        lon (float)
    """
    print("Loading Infrastructure Data...")

    df = _normalise_cols(pd.read_csv(URBAN_AREAS_PATH, dtype=str))
    df = _resolve_col(df, ["name10", "namelsad10", "city", "urban_area", "place", "area_name"], "name")
    df = _resolve_col(df, ["intptlat10", "lat", "latitude", "y"], "lat")
    df = _resolve_col(df, ["intptlon10", "lon", "lng", "longitude", "x"], "lon")
    df = _resolve_col(df, ["population", "pop", "total_population"], "population")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    keep = ["name", "lat", "lon"]
    if "population" in df.columns:
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        keep.append("population")

    print(f"  {len(df):,} urban areas loaded.")
    return df[keep].reset_index(drop=True)


# ── Combined loader (optional convenience) ────────────────────────────────────

def load_all() -> dict:
    """Loads all datasets and returns them in a single dict."""
    return {
        "census":      load_census_data(),
        "targets":     load_urban_targets(),
        "urban_areas": load_infrastructure_data(),
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_all()

    print("\n── Census (first 5 rows) ──")
    print(data["census"].head())

    print("\n── Targets (first 5) ──")
    for t in data["targets"][:5]:
        print(t)

    print("\n── Urban Areas (first 5 rows) ──")
    print(data["urban_areas"].head())