"""
hln.py

Hikami-Larkin-Nagaoka model for weak localization.

Describes magnetoconductivity corrections in
2D disordered systems.
"""

import numpy as np
from scipy.special import digamma
from scipy.optimize import curve_fit

from .base_model import BaseModel
from utils.constants import ELECTRON_CHARGE, HBAR


class HLNModel(BaseModel):

    def model_function(self, B, alpha, l_phi):

        e = ELECTRON_CHARGE
        hbar = HBAR

        B_phi = hbar / (4 * e * l_phi**2)

        x = B_phi / B

        delta_sigma = (
            -alpha * e**2 / (2 * np.pi**2 * hbar)
            * (digamma(0.5 + x) - np.log(x))
        )

        return delta_sigma

    def initial_guess(self, B, data):

        alpha = 0.5
        l_phi = 1e-6

        return [alpha, l_phi]

    def fit(self, B, data):

        p0 = self.initial_guess(B, data)

        popt, pcov = curve_fit(self.model_function, B, data, p0=p0)

        return popt, pcov