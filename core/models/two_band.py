"""
two_band.py

Classical two-band transport model.

Describes magnetotransport in materials with both
electron and hole carriers.

Reference:
Ashcroft & Mermin – Solid State Physics
"""

import numpy as np
from scipy.optimize import curve_fit
from .base_model import BaseModel
from utils.constants import ELECTRON_CHARGE


class TwoBandModel(BaseModel):

    def model_function(self, B, n_e, n_h, mu_e, mu_h):
        """
        Two-band resistivity tensor.

        Returns rho_xx and rho_xy.
        """

        e = ELECTRON_CHARGE

        sigma_xx = (
            e * (n_e * mu_e / (1 + (mu_e * B) ** 2)
            + n_h * mu_h / (1 + (mu_h * B) ** 2))
        )

        sigma_xy = (
            e * (n_h * mu_h**2 * B / (1 + (mu_h * B) ** 2)
            - n_e * mu_e**2 * B / (1 + (mu_e * B) ** 2))
        )

        denom = sigma_xx**2 + sigma_xy**2

        rho_xx = sigma_xx / denom
        rho_xy = sigma_xy / denom

        return rho_xx, rho_xy

    def initial_guess(self, B, data):

        n = 1e22
        mu = 0.1

        return [n, n, mu, mu]

    def fit(self, B, rho_xx):

        p0 = self.initial_guess(B, rho_xx)

        def model(B, n_e, n_h, mu_e, mu_h):
            rho_xx, _ = self.model_function(B, n_e, n_h, mu_e, mu_h)
            return rho_xx

        popt, pcov = curve_fit(model, B, rho_xx, p0=p0)

        return popt, pcov