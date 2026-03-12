"""
warnings.py

Centralized warning system for qtransport.

This module defines structured warning messages that are triggered
when potential problems are detected in experimental data or
during model fitting.

Warnings are NOT errors — they inform the user about potential
issues affecting analysis reliability.

Examples:
- insufficient magnetic field range
- weak signal amplitude
- fitting instability
- non-physical fitted parameters
"""

from dataclasses import dataclass
import numpy as np


# ============================================================
# WARNING DATA STRUCTURE
# ============================================================

@dataclass
class WarningMessage:
    """
    Structured warning message returned by analysis routines.

    Attributes
    ----------
    code : str
        Short identifier for the warning type

    message : str
        Human readable description

    severity : str
        Level of concern:
        "info", "warning", or "critical"
    """

    code: str
    message: str
    severity: str = "warning"


# ============================================================
# WARNING GENERATORS
# ============================================================

def check_field_range(B_field, minimum_range=0.5):
    """
    Check if magnetic field sweep range is large enough.

    Parameters
    ----------
    B_field : array-like
        Magnetic field values (Tesla)

    minimum_range : float
        Minimum acceptable field range (Tesla)

    Returns
    -------
    WarningMessage or None
    """

    B_field = np.asarray(B_field)

    field_range = np.max(B_field) - np.min(B_field)

    if field_range < minimum_range:
        return WarningMessage(
            code="LOW_FIELD_RANGE",
            message=(
                f"Magnetic field range ({field_range:.3f} T) is small. "
                "This may reduce reliability of transport parameter extraction."
            ),
            severity="warning",
        )

    return None


def check_low_signal(signal, threshold=1e-12):
    """
    Detect extremely small signals which may be dominated by noise.

    Parameters
    ----------
    signal : array-like
        Measured quantity (rho_xx or rho_xy)

    threshold : float
        Minimum amplitude threshold

    Returns
    -------
    WarningMessage or None
    """

    signal = np.asarray(signal)

    amplitude = np.max(np.abs(signal))

    if amplitude < threshold:
        return WarningMessage(
            code="LOW_SIGNAL",
            message=(
                "Signal amplitude is extremely small. "
                "Data may be dominated by measurement noise."
            ),
            severity="warning",
        )

    return None


def check_fit_instability(covariance_matrix):
    """
    Detect unstable fits using covariance matrix diagnostics.

    Parameters
    ----------
    covariance_matrix : ndarray

    Returns
    -------
    WarningMessage or None
    """

    if covariance_matrix is None:
        return WarningMessage(
            code="FIT_NO_COVARIANCE",
            message="Covariance matrix not available. Fit uncertainty cannot be estimated.",
            severity="warning",
        )

    diag = np.diag(covariance_matrix)

    if np.any(diag <= 0):
        return WarningMessage(
            code="FIT_UNSTABLE",
            message=(
                "Fit may be unstable. Non-positive parameter variances detected."
            ),
            severity="warning",
        )

    return None


def check_nonphysical_parameters(parameters):
    """
    Detect obviously non-physical parameter values.

    Example:
    Negative carrier density or mobility.

    Parameters
    ----------
    parameters : dict
        Fitted model parameters

    Returns
    -------
    list[WarningMessage]
    """

    warnings_list = []

    for name, value in parameters.items():

        # Carrier density must be positive
        if "n" in name.lower() and value < 0:
            warnings_list.append(
                WarningMessage(
                    code="NEGATIVE_DENSITY",
                    message=f"Parameter '{name}' has negative carrier density.",
                    severity="critical",
                )
            )

        # Mobility should generally be positive
        if "mu" in name.lower() and value < 0:
            warnings_list.append(
                WarningMessage(
                    code="NEGATIVE_MOBILITY",
                    message=f"Parameter '{name}' has negative mobility.",
                    severity="warning",
                )
            )

    return warnings_list