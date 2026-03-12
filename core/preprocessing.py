"""
preprocessing.py

Data preparation utilities for magnetotransport experiments.

Experimental magnetotransport data often contains:

• field sweep asymmetry
• measurement noise
• occasional outliers

These preprocessing functions help prepare data before
model fitting.

All functions operate directly on NumPy arrays and
return processed arrays.
"""

import numpy as np


# ============================================================
# SYMMETRIZATION OF LONGITUDINAL RESISTIVITY
# ============================================================

def symmetrize_rho_xx(B_field: np.ndarray, rho_xx: np.ndarray):
    """
    Symmetrize longitudinal resistivity.

    Physical principle:
    -------------------
    rho_xx(B) should be an even function of B.

        rho_xx(B) = rho_xx(-B)

    Symmetrization removes Hall contamination.

    Formula
    -------
    rho_sym(B) = [rho(B) + rho(-B)] / 2
    """

    B = np.asarray(B_field)
    rho = np.asarray(rho_xx)

    rho_sym = np.zeros_like(rho)

    for i, b in enumerate(B):
        idx = np.where(np.isclose(B, -b))[0]
        if len(idx) > 0:
            rho_sym[i] = (rho[i] + rho[idx[0]]) / 2
        else:
            rho_sym[i] = rho[i]

    return rho_sym


# ============================================================
# ANTISYMMETRIZATION OF HALL RESISTIVITY
# ============================================================

def antisymmetrize_rho_xy(B_field: np.ndarray, rho_xy: np.ndarray):
    """
    Antisymmetrize Hall resistivity.

    Physical principle:
    -------------------
    Hall resistivity is an odd function:

        rho_xy(B) = -rho_xy(-B)

    Formula
    -------
    rho_asym(B) = [rho(B) - rho(-B)] / 2
    """

    B = np.asarray(B_field)
    rho = np.asarray(rho_xy)

    rho_asym = np.zeros_like(rho)

    for i, b in enumerate(B):
        idx = np.where(np.isclose(B, -b))[0]
        if len(idx) > 0:
            rho_asym[i] = (rho[i] - rho[idx[0]]) / 2
        else:
            rho_asym[i] = rho[i]

    return rho_asym


# ============================================================
# OUTLIER REMOVAL
# ============================================================

def remove_outliers(data: np.ndarray, threshold: float = 3.0):
    """
    Remove statistical outliers using Z-score filtering.

    Parameters
    ----------
    data : np.ndarray
    threshold : float
        Z-score threshold

    Returns
    -------
    filtered_data : np.ndarray
    """

    data = np.asarray(data)

    mean = np.mean(data)
    std = np.std(data)

    z_scores = np.abs((data - mean) / std)

    mask = z_scores < threshold

    return data[mask]


# ============================================================
# DATA SMOOTHING
# ============================================================

def smooth_data(data: np.ndarray, window: int = 5):
    """
    Smooth data using moving average.

    Used for reducing experimental noise
    while preserving the overall trend of the data.

    Parameters
    ----------
    data : np.ndarray
        Input experimental data

    window : int
        Size of moving average window

    Returns
    -------
    smoothed_data : np.ndarray
        Smoothed signal
    """

    data = np.asarray(data)

    if window < 1:
        raise ValueError("window must be >= 1")

    # Moving average kernel
    kernel = np.ones(window) / window

    # Convolution performs the smoothing
    smoothed = np.convolve(data, kernel, mode="same")

    return smoothed