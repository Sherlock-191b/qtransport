"""
qtransport.analysis.fft_tools
=============================

Fast Fourier Transform tools for detecting Shubnikov–de Haas
oscillation frequencies.

SdH oscillations are periodic in inverse magnetic field (1/B).

Procedure:

1. Convert B → 1/B
2. Interpolate onto uniform grid
3. Remove background
4. Compute FFT
5. Detect dominant frequencies
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


# ============================================================
# FFT OF SdH OSCILLATIONS
# ============================================================

def compute_fft_frequency(B_field: np.ndarray, rho_xx: np.ndarray):
    """
    Compute FFT spectrum of SdH oscillations.

    Parameters
    ----------
    B_field : magnetic field
    rho_xx : longitudinal resistivity

    Returns
    -------
    freqs : frequency axis
    power : FFT power spectrum
    """

    B = np.asarray(B_field)
    rho = np.asarray(rho_xx)

    # -----------------------------------------
    # Convert to inverse magnetic field
    # -----------------------------------------

    invB = 1.0 / B

    # Sort increasing 1/B
    sort_idx = np.argsort(invB)

    invB = invB[sort_idx]
    rho = rho[sort_idx]

    # -----------------------------------------
    # Interpolate onto uniform 1/B grid
    # -----------------------------------------

    invB_uniform = np.linspace(invB.min(), invB.max(), len(invB))

    interp_func = interp1d(invB, rho, kind="linear")

    rho_uniform = interp_func(invB_uniform)

    # -----------------------------------------
    # Remove DC component
    # -----------------------------------------

    rho_uniform = rho_uniform - np.mean(rho_uniform)

    # -----------------------------------------
    # FFT
    # -----------------------------------------

    fft_vals = np.fft.fft(rho_uniform)

    power = np.abs(fft_vals) ** 2

    freqs = np.fft.fftfreq(
        len(invB_uniform),
        d=(invB_uniform[1] - invB_uniform[0])
    )

    return freqs, power


# ============================================================
# PEAK DETECTION
# ============================================================

def detect_sdh_frequencies(freqs: np.ndarray, power: np.ndarray):
    """
    Detect dominant SdH frequencies from FFT spectrum.

    Parameters
    ----------
    freqs : FFT frequency axis
    power : FFT power spectrum

    Returns
    -------
    peak_freqs : dominant oscillation frequencies
    freqs : frequency axis
    power : FFT spectrum
    """

    # Only positive frequencies are physically meaningful
    mask = freqs > 0

    freqs = freqs[mask]
    power = power[mask]

    # Find peaks
    peaks, _ = find_peaks(power)

    peak_freqs = freqs[peaks]

    return peak_freqs, freqs, power