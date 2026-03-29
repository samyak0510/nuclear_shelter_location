"""
utils.py — Vectorized distance calculations and helper functions.
"""

import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance in miles between two points
    (or arrays of points) on Earth.

    Accepts scalars or numpy arrays.
    """
    R = 3959.0  # Earth radius in miles

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def haversine_distance_matrix(lats1, lons1, lats2, lons2):
    """
    Compute pairwise distances (miles) between two lists of coordinates.

    Parameters
    ----------
    lats1, lons1 : 1-D arrays of shape (N,)
    lats2, lons2 : 1-D arrays of shape (M,)

    Returns
    -------
    dist_matrix : ndarray of shape (N, M)
        dist_matrix[i, j] = haversine distance from point i to point j.
    """
    R = 3959.0
    phi1 = np.radians(lats1)[:, None]          # (N, 1)
    phi2 = np.radians(lats2)[None, :]          # (1, M)
    dphi = phi2 - phi1                          # (N, M)
    dlambda = np.radians(lons2)[None, :] - np.radians(lons1)[:, None]  # (N, M)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c