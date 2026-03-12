"""
sdh.py

Shubnikov–de Haas oscillation model.

Implements Lifshitz-Kosevich formalism for
quantum oscillations in magnetoresistance.
"""

import numpy as np
from scipy.optimize import curve_fit
from .base_model import BaseModel
from utils.constants import ELECTRON_CHARGE, HBAR, BOLTZMANN


class SdHModel(BaseModel):

    def model_function(self, B, F, m_eff, T_D):

        e = ELECTRON_CHARGE
        hbar = HBAR
        kB = BOLTZMANN

        T = 2.0  # assumed constant measurement temp

        omega_c = e * B / m_eff

        R_T = (2 * np.pi**2 * kB * T / (hbar * omega_c)) / \
              np.sinh(2 * np.pi**2 * kB * T / (hbar * omega_c))

        R_D = np.exp(-2 * np.pi**2 * kB * T_D / (hbar * omega_c))

        oscillation = R_T * R_D * np.cos(2 * np.pi * F / B)

        return oscillation

    def initial_guess(self, B, data):

        return [100, 0.1, 5]

    def fit(self, B, data):

        p0 = self.initial_guess(B, data)

        popt, pcov = curve_fit(self.model_function, B, data, p0=p0)

        return popt, pcov