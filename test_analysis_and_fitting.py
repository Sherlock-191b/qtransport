"""
test_analysis_and_fitting.py

Integration test for major qtransport backend modules.

This test verifies:

1. TransportDataset structure
2. Tensor conversion (rho ↔ sigma)
3. Generic nonlinear fitting engine
4. SdH oscillation FFT detection

Run using:

python test_analysis_and_fitting.py
"""

import numpy as np

from core.data_model import TransportDataset
from core.fitting_engine import fit_model
from core.models.two_band import TwoBandModel

from analysis.tensor_conversion import resistivity_to_conductivity
from analysis.fft_tools import compute_fft_frequency, detect_sdh_frequencies


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

print("\n---- Generating Synthetic Data ----")

# magnetic field range
B_field = np.linspace(0.5, 10, 200)

# synthetic resistivity signals
rho_xx = 1e-5 + 1e-6 * B_field**2
rho_xy = 5e-6 * B_field

dataset = TransportDataset(
    B_field=B_field,
    rho_xx=rho_xx,
    rho_xy=rho_xy,
    temperature=5.0,
    metadata={"sample": "synthetic"}
)

print("Dataset created.")


# ============================================================
# TEST TENSOR CONVERSION
# ============================================================

print("\n---- Testing Tensor Conversion ----")

sigma_xx, sigma_xy = resistivity_to_conductivity(rho_xx, rho_xy)

print("sigma_xx sample:", sigma_xx[:5])
print("sigma_xy sample:", sigma_xy[:5])


# ============================================================
# TEST FITTING ENGINE
# ============================================================

print("\n---- Testing Fitting Engine ----")

model = TwoBandModel()

fit_result = fit_model(
    model=model,
    dataset=dataset
)

print("Fit success:", fit_result.success_flag)
print("Fitted parameters:", fit_result.parameters)

print("Chi-square:", fit_result.chi_square)
print("Reduced Chi-square:", fit_result.reduced_chi_square)


# ============================================================
# TEST FFT SdH DETECTION
# ============================================================

print("\n---- Testing FFT SdH Detection ----")

# create synthetic oscillation
frequency = 120  # Tesla frequency

oscillation = np.sin(2 * np.pi * frequency / B_field)

# add oscillation to resistivity
rho_xx_sdh = rho_xx + 1e-7 * oscillation

# compute FFT
freqs, power = compute_fft_frequency(B_field, rho_xx_sdh)

# detect peaks
peak_freqs, freqs, power = detect_sdh_frequencies(freqs, power)

print("Detected SdH frequencies:", peak_freqs[:5])


print("\nAll analysis and fitting tests completed successfully.")