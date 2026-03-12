"""
auto_detector.py

Automatic feature detection for magnetotransport datasets.

Detects:
- Hall signal
- Weak localization (WL)
- Shubnikov–de Haas oscillations (SdH)

Detection is purely statistical and does NOT perform fitting.
"""

import numpy as np
from scipy.signal import detrend
from scipy.fft import rfft, rfftfreq


class AutoDetector:
    """
    Automatically identify physical signatures in magnetotransport data.
    """

    def __init__(self, B_field, rho_xx, rho_xy):
        """
        Parameters
        ----------
        B_field : array-like
            Magnetic field values
        rho_xx : array-like
            Longitudinal resistivity
        rho_xy : array-like
            Hall resistivity
        """

        self.B = np.asarray(B_field)
        self.rho_xx = np.asarray(rho_xx)
        self.rho_xy = np.asarray(rho_xy)

    # ---------------------------------------------------------
    # Hall signal detection
    # ---------------------------------------------------------

    def detect_hall_signal(self, correlation_threshold=0.85):
        """
        Detect presence of Hall signal via correlation with magnetic field.

        Hall resistivity typically shows strong linear dependence on B.

        Returns
        -------
        bool
        """

        corr = np.corrcoef(self.B, self.rho_xy)[0, 1]

        return np.abs(corr) > correlation_threshold

    # ---------------------------------------------------------
    # Weak localization detection
    # ---------------------------------------------------------

    def detect_weak_localization(self, curvature_threshold=3.0):
        """
        Detect weak localization through low-field curvature.

        WL manifests as a sharp cusp near B = 0.

        Returns
        -------
        bool
        """

        # select low-field region
        mask = np.abs(self.B) < 0.5 * np.max(np.abs(self.B))

        B_low = self.B[mask]
        rho_low = self.rho_xx[mask]

        # second derivative estimate
        d2 = np.gradient(np.gradient(rho_low, B_low), B_low)

        curvature = np.max(np.abs(d2))

        # normalize by signal magnitude
        norm = np.std(rho_low)

        if norm == 0:
            return False

        score = curvature / norm

        return score > curvature_threshold

    # ---------------------------------------------------------
    # SdH detection
    # ---------------------------------------------------------

    def detect_sdh(self, peak_threshold=5.0):
        """
        Detect SdH oscillations using FFT peak detection.

        Returns
        -------
        bool
        """

        # remove background trend
        signal = detrend(self.rho_xx)

        N = len(signal)

        fft_vals = np.abs(rfft(signal))
        freqs = rfftfreq(N, d=np.mean(np.diff(self.B)))

        if len(fft_vals) < 5:
            return False

        # ignore DC component
        fft_vals[0] = 0

        peak = np.max(fft_vals)

        noise = np.median(fft_vals)

        if noise == 0:
            return False

        snr = peak / noise

        return snr > peak_threshold

    # ---------------------------------------------------------
    # full detection summary
    # ---------------------------------------------------------

    def run_detection(self):
        """
        Run all detectors.

        Returns
        -------
        dict
        """

        results = {
            "hall_detected": self.detect_hall_signal(),
            "weak_localization_detected": self.detect_weak_localization(),
            "sdh_detected": self.detect_sdh(),
        }

        return results