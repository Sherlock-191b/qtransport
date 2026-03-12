"""
qtransport.core.statistics
==========================

Statistical evaluation utilities for model fitting.

These functions compute goodness-of-fit metrics used to
evaluate magnetotransport model performance.

Metrics implemented:

• Chi-square
• Reduced chi-square
• Akaike Information Criterion (AIC)
• Bayesian Information Criterion (BIC)
• Parameter uncertainties from covariance matrix

All functions operate on NumPy arrays and integrate with
the FitResult dataclass defined in core/data_model.py.
"""

import numpy as np


# ============================================================
# CHI-SQUARE
# ============================================================

def chi_square(observed: np.ndarray, predicted: np.ndarray):
    """
    Compute chi-square statistic.

    χ² = Σ (observed - predicted)^2

    Assumes Gaussian noise with equal variance.
    """

    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    residuals = observed - predicted

    return np.sum(residuals ** 2)


# ============================================================
# REDUCED CHI-SQUARE
# ============================================================

def reduced_chi_square(chi2: float, n_points: int, n_params: int):
    """
    Reduced chi-square.

    χ²_red = χ² / (N - p)

    N = number of data points
    p = number of fitted parameters
    """

    dof = n_points - n_params

    if dof <= 0:
        return np.nan

    return chi2 / dof


# ============================================================
# AIC
# ============================================================

def akaike_information_criterion(chi2: float, n_params: int, n_points: int):
    """
    Akaike Information Criterion.

    AIC = N ln(χ²/N) + 2p
    """

    if n_points <= 0:
        return np.nan

    return n_points * np.log(chi2 / n_points) + 2 * n_params


# ============================================================
# BIC
# ============================================================

def bayesian_information_criterion(chi2: float, n_params: int, n_points: int):
    """
    Bayesian Information Criterion.

    BIC = N ln(χ²/N) + p ln(N)
    """

    if n_points <= 0:
        return np.nan

    return n_points * np.log(chi2 / n_points) + n_params * np.log(n_points)


# ============================================================
# PARAMETER UNCERTAINTIES
# ============================================================

def parameter_uncertainties(covariance_matrix: np.ndarray):
    """
    Compute standard errors of fitted parameters.

    Uncertainty = sqrt(diagonal elements of covariance matrix)
    """

    if covariance_matrix is None:
        return None

    return np.sqrt(np.diag(covariance_matrix))