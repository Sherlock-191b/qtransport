"""
Test script for physics models and preprocessing functions
"""

import numpy as np

from core.preprocessing import (
    symmetrize_rho_xx,
    antisymmetrize_rho_xy,
    remove_outliers,
    smooth_data
)

from core.models.two_band import TwoBandModel
from core.models.hln import HLNModel
from core.models.sdh import SdHModel


print("---- Testing Preprocessing ----")

B = np.array([-3,-2,-1,0,1,2,3])
rho_xx = np.array([10,9,8,7,8,9,10])
rho_xy = np.array([-3,-2,-1,0,1,2,3])

print("Symmetrized rho_xx:")
print(symmetrize_rho_xx(B,rho_xx))

print("Antisymmetrized rho_xy:")
print(antisymmetrize_rho_xy(B,rho_xy))

print("Outlier removal:")
data = np.array([1,1,1,1,100,1,1])
print(remove_outliers(data))

print("Smoothed data:")
print(smooth_data(rho_xx, window=3))


print("\n---- Testing Two Band Model ----")

B = np.linspace(-5,5,50)

two_band = TwoBandModel()

params = [1e26, 5e25, 0.5, 0.3]

rho_xx_pred, rho_xy_pred = two_band.model_function(B, *params)

print("rho_xx sample:", rho_xx_pred[:5])
print("rho_xy sample:", rho_xy_pred[:5])


print("\n---- Testing HLN Model ----")

hln = HLNModel()

B = np.linspace(-1,1,50)

alpha = -0.5
l_phi = 100e-9

delta_sigma = hln.model_function(B, alpha, l_phi)

print("HLN output sample:", delta_sigma[:5])


print("\n---- Testing SdH Model ----")

sdh = SdHModel()

B = np.linspace(1,10,100)

params = [50, 0.1, 5]  # frequency, mass, Dingle temperature

osc = sdh.model_function(B, *params)

print("SdH oscillation sample:", osc[:5])


print("\nAll model tests executed successfully.")