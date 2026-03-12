"""
validation.py

Utility functions for validating experimental magnetotransport datasets
before analysis or fitting.

These checks prevent numerical failures during preprocessing
and model fitting.
"""

import numpy as np


def validate_dataset_structure(B_field, rho_xx, rho_xy):
    """
    Validate that dataset arrays exist and have consistent lengths.

    Parameters
    ----------
    B_field : array-like
        Magnetic field values (Tesla)

    rho_xx : array-like
        Longitudinal resistivity (Ohm·m)

    rho_xy : array-like
        Hall resistivity (Ohm·m)

    Raises
    ------
    ValueError
        If arrays are not numeric or lengths do not match
    """

    # Convert inputs to numpy arrays
    B_field = np.asarray(B_field)
    rho_xx = np.asarray(rho_xx)
    rho_xy = np.asarray(rho_xy)

    # Ensure arrays are 1D
    if B_field.ndim != 1 or rho_xx.ndim != 1 or rho_xy.ndim != 1:
        raise ValueError("All input arrays must be one-dimensional.")

    # Ensure same length
    if not (len(B_field) == len(rho_xx) == len(rho_xy)):
        raise ValueError("B_field, rho_xx, and rho_xy must have the same length.")

    # Ensure numeric
    if not (
        np.issubdtype(B_field.dtype, np.number)
        and np.issubdtype(rho_xx.dtype, np.number)
        and np.issubdtype(rho_xy.dtype, np.number)
    ):
        raise ValueError("All dataset arrays must contain numeric values.")


def validate_no_nan(B_field, rho_xx, rho_xy):
    """
    Check that arrays contain no NaN values.

    Raises
    ------
    ValueError
        If NaN values are present
    """

    if np.isnan(B_field).any():
        raise ValueError("B_field contains NaN values.")

    if np.isnan(rho_xx).any():
        raise ValueError("rho_xx contains NaN values.")

    if np.isnan(rho_xy).any():
        raise ValueError("rho_xy contains NaN values.")


def validate_monotonic_field(B_field):
    """
    Validate that magnetic field axis is monotonic.

    Experimental sweeps should increase or decrease continuously.

    Raises
    ------
    ValueError
        If magnetic field is not monotonic
    """

    B_field = np.asarray(B_field)

    diff = np.diff(B_field)

    if not (np.all(diff >= 0) or np.all(diff <= 0)):
        raise ValueError("Magnetic field must be monotonic.")