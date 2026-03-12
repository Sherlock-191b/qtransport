"""
qtransport.examples.synthetic_data_generator
===========================================

Generate realistic synthetic magnetotransport datasets for testing.

Experiments:
1. Two-band transport
2. Weak localization (HLN)
3. SdH oscillations

Includes realistic experimental artifacts:
- magnetic field offset
- contact misalignment mixing
- Gaussian measurement noise
- symmetry enforcement

Metadata included for reproducibility.
"""

import numpy as np
import pandas as pd
from scipy.special import psi

# -----------------------------------------------------------
# UTILITY: SAVE DATASET WITH METADATA
# -----------------------------------------------------------

def save_dataset_to_csv(df: pd.DataFrame, metadata: dict, filename: str):
    """
    Save dataset to CSV with metadata in header comments.
    """
    with open(filename, "w") as f:
        for key, val in metadata.items():
            f.write(f"# {key} = {val}\n")
        df.to_csv(f, index=False)

    print(f"\nDataset saved to {filename}")


# -----------------------------------------------------------
# UTILITY: EXPERIMENTAL ARTIFACTS
# -----------------------------------------------------------

def apply_measurement_artifacts(B_field, rho_xx, rho_xy):

    # Magnetic field offset (magnet calibration error)
    B_offset = np.random.uniform(-0.02, 0.02)
    B_meas = B_field + B_offset

    # Contact misalignment mixing
    mix = np.random.uniform(0.005, 0.02)

    rho_xy_meas = rho_xy + mix * rho_xx
    rho_xx_meas = rho_xx + mix * rho_xy

    return B_meas, rho_xx_meas, rho_xy_meas, B_offset, mix


# -----------------------------------------------------------
# UTILITY: SYMMETRY ENFORCEMENT
# -----------------------------------------------------------

def enforce_transport_symmetry(rho_xx, rho_xy):

    # enforce even symmetry
    rho_xx_sym = 0.5 * (rho_xx + rho_xx[::-1])

    # enforce odd symmetry
    rho_xy_sym = 0.5 * (rho_xy - rho_xy[::-1])

    return rho_xx_sym, rho_xy_sym


# -----------------------------------------------------------
# EXPERIMENT 1: TWO-BAND TRANSPORT
# -----------------------------------------------------------

def generate_two_band_dataset(B_field: np.ndarray, temperature: float):

    e_charge = 1.602e-19

    # realistic ranges
    n_e = np.random.uniform(5e22, 2e23)
    n_h = np.random.uniform(1e21, 5e22)

    mu_e = np.random.uniform(0.5, 2.0)
    mu_h = np.random.uniform(0.05, 0.5)

    noise_level = np.random.uniform(1e-12, 1e-11)

    sigma_xx = e_charge * (
        n_e * mu_e / (1 + (mu_e * B_field)**2) +
        n_h * mu_h / (1 + (mu_h * B_field)**2)
    )

    sigma_xy = e_charge * (
        n_e * mu_e**2 * B_field / (1 + (mu_e * B_field)**2) -
        n_h * mu_h**2 * B_field / (1 + (mu_h * B_field)**2)
    )

    denom = sigma_xx**2 + sigma_xy**2

    rho_xx = sigma_xx / denom
    rho_xy = -sigma_xy / denom

    # add noise
    rho_xx += np.random.normal(0, noise_level, size=B_field.shape)
    rho_xy += np.random.normal(0, noise_level, size=B_field.shape)

    # experimental artifacts
    B_meas, rho_xx, rho_xy, B_offset, mix = apply_measurement_artifacts(
        B_field, rho_xx, rho_xy
    )

    # enforce symmetry
    rho_xx, rho_xy = enforce_transport_symmetry(rho_xx, rho_xy)

    df = pd.DataFrame({
        "B_field": B_meas,
        "rho_xx": rho_xx,
        "rho_xy": rho_xy,
        "temperature": temperature
    })

    metadata = {
        "experiment": "two_band",
        "n_e": n_e,
        "n_h": n_h,
        "mu_e": mu_e,
        "mu_h": mu_h,
        "noise_level": noise_level,
        "B_offset": B_offset,
        "contact_mixing": mix,
        "temperature": temperature
    }

    print("\nTwo-band parameters:")
    for k,v in metadata.items():
        print(k,"=",v)

    return df, metadata


# -----------------------------------------------------------
# EXPERIMENT 2: WEAK LOCALIZATION (HLN)
# -----------------------------------------------------------

def generate_weak_localization_dataset(B_field: np.ndarray, temperature: float):

    alpha = np.random.uniform(0.5, 1.2)
    B_phi = np.random.uniform(0.01, 0.2)

    rho_0 = np.random.uniform(1e-3, 5e-3)
    noise_level = np.random.uniform(5e-5, 2e-4)

    hbar = 1.055e-34
    e = 1.602e-19

    B_safe = np.where(np.abs(B_field) < 1e-4, 1e-4, np.abs(B_field))

    delta_sigma = -alpha * e**2 / (2*np.pi**2*hbar) * (
        psi(0.5 + B_phi / B_safe) -
        np.log(B_phi / B_safe)
    )

    rho_xx = rho_0 - 5 * rho_0**2 * delta_sigma
    rho_xy = np.zeros_like(rho_xx)

    rho_xx += np.random.normal(0, noise_level, size=B_field.shape)

    B_meas, rho_xx, rho_xy, B_offset, mix = apply_measurement_artifacts(
        B_field, rho_xx, rho_xy
    )

    rho_xx, rho_xy = enforce_transport_symmetry(rho_xx, rho_xy)

    df = pd.DataFrame({
        "B_field": B_meas,
        "rho_xx": rho_xx,
        "rho_xy": rho_xy,
        "temperature": temperature
    })

    metadata = {
        "experiment": "weak_localization",
        "alpha": alpha,
        "B_phi": B_phi,
        "rho_0": rho_0,
        "noise_level": noise_level,
        "B_offset": B_offset,
        "contact_mixing": mix,
        "temperature": temperature
    }

    print("\nWeak localization parameters:")
    for k,v in metadata.items():
        print(k,"=",v)

    return df, metadata


# -----------------------------------------------------------
# EXPERIMENT 3: SdH OSCILLATIONS
# -----------------------------------------------------------

def generate_sdh_dataset(B_field: np.ndarray, temperature: float):

    A = np.random.uniform(0.001, 0.01)
    F = np.random.uniform(20, 60)

    lam = np.random.uniform(0.5, 3.0)
    phi = np.random.uniform(0, 2*np.pi)

    rho_0 = np.random.uniform(0.002, 0.01)
    noise_level = np.random.uniform(5e-5, 2e-4)

    B_safe = np.where(np.abs(B_field) < 0.05, 0.05, B_field)

    rho_xx = rho_0 + A * np.exp(-lam / np.abs(B_safe)) * \
             np.cos(2*np.pi*F/B_safe + phi)

    rho_xy = np.zeros_like(rho_xx)

    rho_xx += np.random.normal(0, noise_level, size=B_field.shape)
    rho_xy += np.random.normal(0, noise_level, size=B_field.shape)

    B_meas, rho_xx, rho_xy, B_offset, mix = apply_measurement_artifacts(
        B_field, rho_xx, rho_xy
    )

    rho_xx, rho_xy = enforce_transport_symmetry(rho_xx, rho_xy)

    df = pd.DataFrame({
        "B_field": B_meas,
        "rho_xx": rho_xx,
        "rho_xy": rho_xy,
        "temperature": temperature
    })

    metadata = {
        "experiment": "sdh",
        "A": A,
        "F": F,
        "lambda": lam,
        "phi": phi,
        "rho_0": rho_0,
        "noise_level": noise_level,
        "B_offset": B_offset,
        "contact_mixing": mix,
        "temperature": temperature
    }

    print("\nSdH parameters:")
    for k,v in metadata.items():
        print(k,"=",v)

    return df, metadata


# -----------------------------------------------------------
# MAIN CLI
# -----------------------------------------------------------

def main():

    print("=== Synthetic Magnetotransport Data Generator ===")
    print("Select experiment type:")
    print("1 → Two-band transport")
    print("2 → Weak localization (HLN)")
    print("3 → SdH oscillations")

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice not in {"1", "2", "3"}:
        print("Invalid choice")
        return

    num_points = int(input("Number of B-field points: "))
    B_min = float(input("Minimum B-field (T): "))
    B_max = float(input("Maximum B-field (T): "))
    temperature = float(input("Temperature (K): "))
    output_path = input("Enter output CSV filename: ")

    B_field = np.linspace(B_min, B_max, num_points)

    if choice == "1":
        df, metadata = generate_two_band_dataset(B_field, temperature)

    elif choice == "2":
        df, metadata = generate_weak_localization_dataset(B_field, temperature)

    elif choice == "3":
        df, metadata = generate_sdh_dataset(B_field, temperature)

    save_dataset_to_csv(df, metadata, output_path)


if __name__ == "__main__":
    main()
    
    
    
# run it by

# Bash
# cd C:\Users\sanju\OneDrive\Desktop\qtransport\examples

# Bash
# python synthetic_data_generator.py

# Follow the instructions
    # Select experiment type:
    # 1 → Two-band transport
    # 2 → Weak localization (HLN)
    # 3 → SdH oscillations
    # Enter 1, 2, or 3: 1
    # Number of B-field points: 200
    # Minimum B-field (T): -10
    # Maximum B-field (T): 10
    # Temperature (K): 5
    # Enter output CSV filename: two_band_test.csv