"""
qtransport.utils.constants
==========================

Physical constants and unit conversion factors used throughout
the qtransport project.

These constants support magnetotransport analysis including
Drude transport, Hall effect analysis, quantum oscillations,
and weak localization models.

All constants are defined using SI units to ensure consistency
across the entire analysis pipeline.

NOTE:
No computational logic should exist in this module.
It only defines constants.
"""

import numpy as np


# ------------------------------------------------------------------
# Fundamental Physical Constants (SI Units)
# ------------------------------------------------------------------

# Electron charge (Coulombs)
ELECTRON_CHARGE = 1.602176634e-19

# Planck constant (Joule·second)
PLANCK_CONSTANT = 6.62607015e-34

# Reduced Planck constant (ħ = h / 2π)
REDUCED_PLANCK_CONSTANT = PLANCK_CONSTANT / (2 * np.pi)

# Boltzmann constant (Joule/Kelvin)
BOLTZMANN_CONSTANT = 1.380649e-23


# ------------------------------------------------------------------
# Quantum Transport Constants
# ------------------------------------------------------------------

# Magnetic flux quantum Φ0 = h / e
FLUX_QUANTUM = PLANCK_CONSTANT / ELECTRON_CHARGE

# Reduced flux quantum ħ / e
REDUCED_FLUX_QUANTUM = REDUCED_PLANCK_CONSTANT / ELECTRON_CHARGE


# ------------------------------------------------------------------
# Unit Conversion Factors
# ------------------------------------------------------------------

# Magnetic field conversions
TESLA_TO_GAUSS = 1e4
GAUSS_TO_TESLA = 1e-4

# Length conversions
METER_TO_CENTIMETER = 100.0
CENTIMETER_TO_METER = 0.01

# Resistivity conversions
OHM_M_TO_MILLI_OHM_CM = 1e5
MILLI_OHM_CM_TO_OHM_M = 1e-5

# Energy conversions
JOULE_TO_EV = 1.0 / ELECTRON_CHARGE
EV_TO_JOULE = ELECTRON_CHARGE


# ------------------------------------------------------------------
# Numerical Safety Constants
# ------------------------------------------------------------------

# Small numerical value to avoid division by zero
EPSILON = 1e-12

# Default tolerance used in numerical checks
NUMERICAL_TOLERANCE = 1e-9

# ------------------------------------------------------------------
# Compatibility aliases (used by physics models)
# ------------------------------------------------------------------

# Common physics shorthand
HBAR = REDUCED_PLANCK_CONSTANT
PI = np.pi
BOLTZMANN = BOLTZMANN_CONSTANT