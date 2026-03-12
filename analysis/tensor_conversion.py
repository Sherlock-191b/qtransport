"""
qtransport.analysis.tensor_conversion
=====================================

Tensor conversion utilities for magnetotransport analysis.

Magnetotransport experiments measure resistivity tensor
components (rho_xx, rho_xy). Many theoretical models
operate in conductivity space.

This module converts between:

• resistivity tensor (ρ)
• conductivity tensor (σ)

Using matrix inversion.

For a 2D Hall bar system the tensors are:

ρ tensor:

    | ρxx   ρxy |
    | -ρxy  ρxx |

σ tensor = ρ^{-1}

    | σxx   σxy |
    | -σxy  σxx |
"""

import numpy as np


# ============================================================
# RESISTIVITY → CONDUCTIVITY
# ============================================================

def resistivity_to_conductivity(rho_xx: np.ndarray, rho_xy: np.ndarray):
    """
    Convert resistivity tensor components to conductivity.

    σ = ρ^{-1}

    Parameters
    ----------
    rho_xx : np.ndarray
        Longitudinal resistivity

    rho_xy : np.ndarray
        Hall resistivity

    Returns
    -------
    sigma_xx : np.ndarray
    sigma_xy : np.ndarray
    """

    rho_xx = np.asarray(rho_xx)
    rho_xy = np.asarray(rho_xy)

    # determinant of resistivity tensor
    denom = rho_xx**2 + rho_xy**2

    sigma_xx = rho_xx / denom
    sigma_xy = -rho_xy / denom

    return sigma_xx, sigma_xy


# ============================================================
# CONDUCTIVITY → RESISTIVITY
# ============================================================

def conductivity_to_resistivity(sigma_xx: np.ndarray, sigma_xy: np.ndarray):
    """
    Convert conductivity tensor to resistivity tensor.

    ρ = σ^{-1}

    Parameters
    ----------
    sigma_xx : np.ndarray
    sigma_xy : np.ndarray

    Returns
    -------
    rho_xx : np.ndarray
    rho_xy : np.ndarray
    """

    sigma_xx = np.asarray(sigma_xx)
    sigma_xy = np.asarray(sigma_xy)

    denom = sigma_xx**2 + sigma_xy**2

    rho_xx = sigma_xx / denom
    rho_xy = -sigma_xy / denom

    return rho_xx, rho_xy


# ============================================================
# ALIAS FUNCTIONS (for convenience)
# ============================================================

def rho_to_sigma(rho_xx, rho_xy):
    """
    Alias for resistivity_to_conductivity.
    """
    return resistivity_to_conductivity(rho_xx, rho_xy)


def sigma_to_rho(sigma_xx, sigma_xy):
    """
    Alias for conductivity_to_resistivity.
    """
    return conductivity_to_resistivity(sigma_xx, sigma_xy)