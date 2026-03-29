"""
Blast Radius Calculator for Nuclear Shelter Siting (UFLP)
==========================================================

Uses the Hopkinson-Cranz cube-root scaling law from Glasstone & Dolan,
"The Effects of Nuclear Weapons" (1977).

Core formula:
    R(Y) = R_ref * Y^(1/3) * burst_factor

Where:
    R_ref       = reference radius at 1 kt for a given overpressure (miles)
    Y           = weapon yield in kilotons
    burst_factor = 1.0 for surface burst, 1.25 for optimised air burst

Reference radii at 1 kt (surface burst), derived from declassified
FEMA/DHS/DTIC weapons-effects data:
    ┌──────────────┬───────────┬───────────┐
    │ Overpressure │   km      │   miles   │
    ├──────────────┼───────────┼───────────┤
    │ 20 psi       │ 0.22      │ 0.137     │  Total destruction
    │ 5 psi        │ 0.47      │ 0.292     │  Severe / residential collapse
    │ 1 psi        │ 1.50      │ 0.932     │  Light damage / glass breakage
    │ Fireball     │ 0.13      │ 0.081     │  Complete vaporisation
    └──────────────┴───────────┴───────────┘

Usage in the GA safety mask:
    - Parse each target's Yield (e.g. "500kt" → 500) and Type
    - Compute blast_radius_miles() for the chosen threshold
    - Exclude any ZIP code whose centroid is within that radius
      of any target (using Haversine distance)
"""

import math
import numpy as np
from typing import Literal

# ── Reference radii at 1 kt, surface burst (miles) ──────────────────────
REFERENCE_RADII_MILES = {
    "fireball":  0.081,   # ~130 m — complete vaporisation
    "20psi":     0.137,   # ~220 m — total structural destruction
    "5psi":      0.292,   # ~470 m — severe damage, residential collapse
    "1psi":      0.932,   # ~1500 m — light damage, window breakage
}

# Air-burst multiplier: an optimised air burst extends the overpressure
# contour ~20-30% farther than an equivalent surface burst.
AIR_BURST_FACTOR = 1.25
SURFACE_BURST_FACTOR = 1.0


def blast_radius_miles(
    yield_kt: float,
    threshold: Literal["fireball", "20psi", "5psi", "1psi"] = "5psi",
    burst_type: str = "Surface Burst",
) -> float:
    """
    Calculate the blast radius in miles using cube-root scaling.

    Parameters
    ----------
    yield_kt : float
        Weapon yield in kilotons (e.g. 500 for a 500 kt warhead).
    threshold : str
        Overpressure level defining the outer edge of the zone.
        One of: "fireball", "20psi", "5psi", "1psi".
        Recommended: "5psi" for shelter exclusion (residential collapse).
    burst_type : str
        "Air Burst" or "Surface Burst" (matches your dataset's Type column).

    Returns
    -------
    float
        Blast radius in miles.

    Examples
    --------
    >>> blast_radius_miles(500, "5psi", "Surface Burst")
    2.316...
    >>> blast_radius_miles(1000, "1psi", "Air Burst")
    11.647...
    """
    if yield_kt <= 0:
        return 0.0

    r_ref = REFERENCE_RADII_MILES[threshold]
    burst_factor = (
        AIR_BURST_FACTOR if "air" in burst_type.lower() else SURFACE_BURST_FACTOR
    )

    return r_ref * (yield_kt ** (1 / 3)) * burst_factor


def blast_radius_km(
    yield_kt: float,
    threshold: Literal["fireball", "20psi", "5psi", "1psi"] = "5psi",
    burst_type: str = "Surface Burst",
) -> float:
    """Same as blast_radius_miles but returns kilometres."""
    return blast_radius_miles(yield_kt, threshold, burst_type) * 1.60934


def parse_yield(yield_str: str) -> float:
    """
    Parse the Yield column from the nuclear targets CSV.

    Handles formats like '500kt', '1000 kt', '1.2Mt', '800 Kt'.

    Parameters
    ----------
    yield_str : str
        Raw yield string from the dataset.

    Returns
    -------
    float
        Yield in kilotons.
    """
    if not isinstance(yield_str, str):
        return 0.0
    s = yield_str.strip().lower().replace(" ", "").replace("\xa0", "")
    if s.endswith("mt"):
        return float(s[:-2]) * 1000
    elif s.endswith("kt"):
        return float(s[:-2])
    else:
        try:
            return float(s)
        except ValueError:
            return 0.0


def normalise_burst_type(burst_str: str) -> str:
    """
    Normalise the Type column (handles non-breaking spaces, case, etc.)
    Returns 'Air Burst' or 'Surface Burst'.
    """
    if not isinstance(burst_str, str):
        return "Surface Burst"
    s = burst_str.replace("\xa0", " ").strip().lower()
    if "air" in s:
        return "Air Burst"
    return "Surface Burst"


# ── Quick demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Blast radius examples (5 psi — severe damage zone):")
    print("=" * 55)
    for yt in [10, 100, 300, 500, 800, 1000]:
        r_surface = blast_radius_miles(yt, "5psi", "Surface Burst")
        r_air = blast_radius_miles(yt, "5psi", "Air Burst")
        print(
            f"  {yt:>5} kt  ->  Surface: {r_surface:6.2f} mi "
            f"({r_surface * 1.609:.2f} km)  |  "
            f"Air: {r_air:6.2f} mi ({r_air * 1.609:.2f} km)"
        )

    print()
    print("Blast radius examples (1 psi — light damage / safety zone):")
    print("=" * 55)
    for yt in [10, 100, 300, 500, 800, 1000]:
        r_surface = blast_radius_miles(yt, "1psi", "Surface Burst")
        r_air = blast_radius_miles(yt, "1psi", "Air Burst")
        print(
            f"  {yt:>5} kt  ->  Surface: {r_surface:6.2f} mi "
            f"({r_surface * 1.609:.2f} km)  |  "
            f"Air: {r_air:6.2f} mi ({r_air * 1.609:.2f} km)"
        )
