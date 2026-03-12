"""
test_auto_detector_and_report.py

Integration test for:

analysis.auto_detector
report.figure_style
report.report_generator

This script generates synthetic magnetotransport data,
runs automatic physics detection, produces plots,
and generates a PDF report.

Run with:
python test_auto_detector_and_report.py
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from analysis.auto_detector import AutoDetector
from report.figure_style import apply_style
from report.report_generator import ReportGenerator


# ---------------------------------------------------------
# Mock data structures (to simulate project dataclasses)
# ---------------------------------------------------------

@dataclass
class TransportDataset:
    B_field: np.ndarray
    rho_xx: np.ndarray
    rho_xy: np.ndarray
    temperature: float
    metadata: dict


@dataclass
class FitResult:
    model_name: str
    parameters: dict
    covariance_matrix: np.ndarray
    parameter_errors: dict
    chi_square: float
    reduced_chi_square: float
    AIC: float
    BIC: float
    residuals: np.ndarray
    fitted_curve: np.ndarray
    success_flag: bool
    message: str


# ---------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------

print("\n---- Generating Synthetic Magnetotransport Data ----")

B = np.linspace(-8, 8, 500)

# Hall signal
rho_xy = 0.5 * B + np.random.normal(0, 0.02, len(B))

# Weak localization + background
rho_xx = 10 + 0.02 * B**2 - 0.1 * np.exp(-B**2)

# SdH oscillations
rho_xx += 0.02 * np.sin(2 * np.pi * B * 3)

print("Synthetic dataset created.")


# ---------------------------------------------------------
# Create dataset object
# ---------------------------------------------------------

dataset = TransportDataset(
    B_field=B,
    rho_xx=rho_xx,
    rho_xy=rho_xy,
    temperature=2.0,
    metadata={"sample": "test_sample"}
)


# ---------------------------------------------------------
# Test AutoDetector
# ---------------------------------------------------------

print("\n---- Testing AutoDetector ----")

detector = AutoDetector(B, rho_xx, rho_xy)

results = detector.run_detection()

print("Detection results:")
print(results)

print("\nExpected:")
print("hall_detected → True")
print("weak_localization_detected → True")
print("sdh_detected → True")


# ---------------------------------------------------------
# Mock fit result (simulate model fit)
# ---------------------------------------------------------

print("\n---- Creating Mock FitResult ----")

fitted_curve = rho_xx + np.random.normal(0, 0.01, len(B))

fit_result = FitResult(
    model_name="TwoBandModel",
    parameters={
        "n1": 5e22,
        "n2": 1e21,
        "mu1": 1500,
        "mu2": 800
    },
    covariance_matrix=np.eye(4),
    parameter_errors={
        "n1": 1e21,
        "n2": 5e20,
        "mu1": 50,
        "mu2": 40
    },
    chi_square=1e-6,
    reduced_chi_square=1e-8,
    AIC=12.3,
    BIC=15.1,
    residuals=rho_xx - fitted_curve,
    fitted_curve=fitted_curve,
    success_flag=True,
    message="Fit successful"
)

print("Mock FitResult created.")


# ---------------------------------------------------------
# Test figure style
# ---------------------------------------------------------

print("\n---- Testing Figure Style ----")

apply_style()

plt.figure()
plt.plot(B, rho_xx)
plt.title("Style Test")
plt.xlabel("B (T)")
plt.ylabel("rho_xx")

plt.savefig("style_test.png")
plt.close()

print("Style figure saved: style_test.png")


# ---------------------------------------------------------
# Test Report Generator
# ---------------------------------------------------------

print("\n---- Testing Report Generator ----")

report = ReportGenerator(dataset, fit_result)

# Save figure
report.save_figure("fit_plot_test.png")

print("Fit plot saved.")

# Save PDF report
report.save_report("analysis_report_test.pdf")

print("PDF report saved.")


print("\n---- All Tests Completed Successfully ----")