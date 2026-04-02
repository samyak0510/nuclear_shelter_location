"""
preprocessing.py — Data cleaning and preprocessing utilities.

Handles parsing, normalization, geocoding, and aggregation of raw
Census, Nuclear Target, and Urban Area data before it is used by
the feature engineering or genetic algorithm stages.
"""

import pandas as pd
import pgeocode


# ── Column helpers ──────────────────────────────────────────────────────────

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip all column names."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


# ── Yield / burst-type parsing ─────────────────────────────────────────────

def parse_yield_kt(yield_str: str) -> float:
    """
    Parse the Yield column from the nuclear targets CSV into kilotons.

    Handles formats like '500kt', '1000 kt', '1.2Mt', '800 Kt'.

    Returns
    -------
    float
        Yield in kilotons.
    """
    if not isinstance(yield_str, str):
        return 0.0
    cleaned = yield_str.strip().lower().replace(" ", "").replace("\xa0", "")
    if cleaned.endswith("mt"):
        return float(cleaned[:-2]) * 1000
    if cleaned.endswith("kt"):
        return float(cleaned[:-2])
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def normalize_burst_type(burst_str: str) -> str:
    """
    Normalize the Type column (handles non-breaking spaces, case, etc.)
    Returns 'Air Burst' or 'Surface Burst'.
    """
    if not isinstance(burst_str, str):
        return "Surface Burst"
    normalized = burst_str.replace("\xa0", " ").strip().lower()
    if "air" in normalized:
        return "Air Burst"
    return "Surface Burst"


# ── Census preprocessing ───────────────────────────────────────────────────

def clean_census_data(raw_census: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw 2010 Census ZIP-level population data across age/gender
    rows and attach lat/lon coordinates via pgeocode.

    Parameters
    ----------
    raw_census : DataFrame
        Raw CSV with columns including 'zipcode' and 'population'.

    Returns
    -------
    DataFrame with columns: zip_code, population, lat, lon
    """
    raw_census.columns = raw_census.columns.str.strip().str.lower()

    raw_census["population"] = pd.to_numeric(raw_census["population"], errors="coerce")
    raw_census["zipcode"] = raw_census["zipcode"].str.strip().str.zfill(5)

    # Sum population across all age/gender rows per ZIP code
    population_by_zip = (
        raw_census.groupby("zipcode", as_index=False)["population"]
        .sum()
        .rename(columns={"zipcode": "zip_code"})
    )

    # Drop rows with zero or null population
    population_by_zip = population_by_zip[population_by_zip["population"] > 0].copy()
    population_by_zip["population"] = population_by_zip["population"].astype(int)

    print(f"  Aggregated {len(population_by_zip):,} ZIP codes from raw data.")

    # Attach lat/lon via pgeocode
    print("  Geocoding ZIP codes (pgeocode)...")
    geocoder = pgeocode.Nominatim("US")
    geocoded = geocoder.query_postal_code(population_by_zip["zip_code"].tolist())
    population_by_zip["lat"] = geocoded["latitude"].values
    population_by_zip["lon"] = geocoded["longitude"].values

    # Drop ZIP codes that couldn't be geocoded
    count_before = len(population_by_zip)
    population_by_zip = population_by_zip.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"  {count_before - len(population_by_zip)} ZIP codes dropped (no coordinates).")
    print(f"  {len(population_by_zip):,} ZIP codes ready.")

    return population_by_zip


# ── Nuclear targets preprocessing ──────────────────────────────────────────

def clean_nuclear_targets(raw_targets: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize raw nuclear targets data: resolve column names,
    parse yield strings, and normalize burst types.

    Parameters
    ----------
    raw_targets : DataFrame
        Raw CSV with columns like 'target', 'lat', 'lng', 'yield', 'type'.

    Returns
    -------
    DataFrame with columns: name, lat, lon, yield_kt, burst_type
        (plus 'category' if present in the original data)
    """
    raw_targets.columns = raw_targets.columns.str.strip().str.lower()

    # Rename columns to standard names
    column_rename_map = {}
    if "target" in raw_targets.columns:
        column_rename_map["target"] = "name"
    if "lng" in raw_targets.columns:
        column_rename_map["lng"] = "lon"
    raw_targets = raw_targets.rename(columns=column_rename_map)

    raw_targets["lat"] = pd.to_numeric(raw_targets["lat"], errors="coerce")
    raw_targets["lon"] = pd.to_numeric(raw_targets["lon"], errors="coerce")
    # Drop rows without valid coordinates
    raw_targets = raw_targets.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Parse yield and burst type
    if "yield" in raw_targets.columns:
        raw_targets["yield_kt"] = raw_targets["yield"].apply(parse_yield_kt)
    else:
        raw_targets["yield_kt"] = 500.0  # default assumption

    if "type" in raw_targets.columns:
        raw_targets["burst_type"] = raw_targets["type"].apply(normalize_burst_type)
    else:
        raw_targets["burst_type"] = "Surface Burst"

    output_columns = ["name", "lat", "lon", "yield_kt", "burst_type"]
    if "category" in raw_targets.columns:
        output_columns.append("category")

    return raw_targets[output_columns]


# ── Urban areas preprocessing ──────────────────────────────────────────────

def clean_urban_areas(raw_urban: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize raw urban areas data: resolve column name aliases,
    convert types, and drop invalid rows.

    Parameters
    ----------
    raw_urban : DataFrame
        Raw CSV with varying column names for name/lat/lon.

    Returns
    -------
    DataFrame with columns: name, lat, lon
    """
    raw_urban.columns = raw_urban.columns.str.strip().str.lower()

    # Resolve column name aliases
    column_rename_map = {}
    for alias in ["name10", "namelsad10", "name"]:
        if alias in raw_urban.columns:
            column_rename_map[alias] = "name"
            break
    for alias in ["intptlat10", "lat", "latitude"]:
        if alias in raw_urban.columns:
            column_rename_map[alias] = "lat"
            break
    for alias in ["intptlon10", "lon", "lng", "longitude"]:
        if alias in raw_urban.columns:
            column_rename_map[alias] = "lon"
            break

    raw_urban = raw_urban.rename(columns=column_rename_map)
    raw_urban["lat"] = pd.to_numeric(raw_urban["lat"], errors="coerce")
    raw_urban["lon"] = pd.to_numeric(raw_urban["lon"], errors="coerce")
    raw_urban = raw_urban.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    return raw_urban[["name", "lat", "lon"]]
