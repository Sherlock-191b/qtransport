"""
unit_conversion.py

Handles unit conversions required in magnetotransport analysis.

Experimental data often comes in raw forms such as:

- Resistance (Ohm)
- Voltage (Volts)
- Magnetic field (Gauss or Tesla)

This module converts them into SI quantities required
for theoretical analysis:

- Resistivity (Ohm·m)
- Magnetic field (Tesla)

Hall bar geometry parameters are required for
resistance-to-resistivity conversion.
"""

import numpy as np


# ============================================================
# RESISTANCE → RESISTIVITY
# ============================================================

def convert_resistance_to_resistivity(
    resistance: np.ndarray,
    length: float,
    width: float,
    thickness: float
) -> np.ndarray:
    """
    Convert resistance (Ohm) to resistivity (Ohm·m).

    Formula
    -------
    rho = R * (A / L)

    where

    A = cross sectional area = width * thickness
    L = distance between voltage probes

    Parameters
    ----------
    resistance : np.ndarray
        Measured resistance values.

    length : float
        Probe separation (meters).

    width : float
        Sample width (meters).

    thickness : float
        Sample thickness (meters).

    Returns
    -------
    np.ndarray
        Resistivity values in Ohm·m.
    """

    area = width * thickness

    resistivity = resistance * (area / length)

    return resistivity


# ============================================================
# VOLTAGE → HALL RESISTIVITY
# ============================================================

def convert_voltage_to_resistivity(
    voltage: np.ndarray,
    current: float,
    thickness: float
) -> np.ndarray:
    """
    Convert Hall voltage to Hall resistivity.

    Hall resistivity formula:

        rho_xy = (V_H / I) * thickness

    Parameters
    ----------
    voltage : np.ndarray
        Hall voltage measurements.

    current : float
        Applied current (Amperes).

    thickness : float
        Sample thickness (meters).

    Returns
    -------
    np.ndarray
        Hall resistivity in Ohm·m.
    """

    rho_xy = (voltage / current) * thickness

    return rho_xy


# ============================================================
# MAGNETIC FIELD UNIT CONVERSION
# ============================================================

def convert_field_units(
    field: np.ndarray,
    input_unit: str = "T"
) -> np.ndarray:
    """
    Convert magnetic field units to Tesla.

    Supported input units:
    - T (Tesla)
    - G (Gauss)

    1 Tesla = 10,000 Gauss

    Parameters
    ----------
    field : np.ndarray
        Magnetic field values.

    input_unit : str
        Unit of the input data.

    Returns
    -------
    np.ndarray
        Magnetic field in Tesla.
    """

    if input_unit == "T":
        return field

    if input_unit == "G":
        return field / 10000.0

    raise ValueError("Unsupported field unit. Use 'T' or 'G'.")