"""
data_model.py

Defines the core data structures used throughout qtransport.

This module contains two fundamental dataclasses:

1. TransportDataset
   Represents a single magnetotransport dataset.

2. FitResult
   Represents the output of a model fitting procedure.

These objects are passed across modules including:
- preprocessing
- fitting_engine
- analysis
- reporting

The structure strictly follows the API contract defined for qtransport.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
import pandas as pd


# ============================================================
# TRANSPORT DATASET
# ============================================================

@dataclass
class TransportDataset:
    """
    Container for magnetotransport experimental data.

    Attributes
    ----------
    B_field : np.ndarray
        Magnetic field values in Tesla.

    rho_xx : np.ndarray
        Longitudinal resistivity in Ohm·m.

    rho_xy : np.ndarray
        Hall resistivity in Ohm·m.

    temperature : float
        Measurement temperature in Kelvin.

    metadata : dict
        Additional experimental metadata
        (sample name, thickness, notes, etc).
    """

    B_field: np.ndarray
    rho_xx: np.ndarray
    rho_xy: np.ndarray
    temperature: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------
    # CREATE DATASET FROM DATAFRAME
    # --------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        temperature: float,
        metadata: Dict[str, Any] = None
    ):
        """
        Construct TransportDataset from a pandas DataFrame.

        Required columns in dataframe:
        - B_field
        - rho_xx
        - rho_xy

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing transport data.

        temperature : float
            Measurement temperature in Kelvin.

        metadata : dict, optional
            Additional experimental information.

        Returns
        -------
        TransportDataset
        """

        required_columns = ["B_field", "rho_xx", "rho_xy"]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return cls(
            B_field=df["B_field"].to_numpy(),
            rho_xx=df["rho_xx"].to_numpy(),
            rho_xy=df["rho_xy"].to_numpy(),
            temperature=float(temperature),
            metadata=metadata or {}
        )

    # --------------------------------------------------------
    # EXPORT DATASET TO DATAFRAME
    # --------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert dataset to pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
        """

        df = pd.DataFrame({
            "B_field": self.B_field,
            "rho_xx": self.rho_xx,
            "rho_xy": self.rho_xy
        })

        return df


# ============================================================
# FIT RESULT STRUCTURE
# ============================================================

@dataclass
class FitResult:
    """
    Result of a model fit.

    This structure stores both the fitted parameters and
    statistical diagnostics.

    Attributes
    ----------
    model_name : str
        Name of the model used.

    parameters : dict
        Best-fit parameter values.

    covariance_matrix : np.ndarray
        Covariance matrix returned from fitting algorithm.

    parameter_errors : dict
        Standard errors for fitted parameters.

    chi_square : float
        Chi-square statistic.

    reduced_chi_square : float
        Reduced chi-square statistic.

    AIC : float
        Akaike Information Criterion.

    BIC : float
        Bayesian Information Criterion.

    residuals : np.ndarray
        Difference between data and fitted model.

    fitted_curve : np.ndarray
        Model prediction for input B-field.

    success_flag : bool
        Whether fitting converged successfully.

    message : str
        Diagnostic message from fitting routine.
    """

    model_name: str
    parameters: Dict[str, float]
    covariance_matrix: np.ndarray
    parameter_errors: Dict[str, float]

    chi_square: float
    reduced_chi_square: float
    AIC: float
    BIC: float

    residuals: np.ndarray
    fitted_curve: np.ndarray

    success_flag: bool
    message: str

    # --------------------------------------------------------
    # EXPORT FIT RESULT AS DATAFRAME
    # --------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert fitted curve and residuals into DataFrame.

        Useful for exporting processed data.

        Returns
        -------
        pandas.DataFrame
        """

        df = pd.DataFrame({
            "fitted_curve": self.fitted_curve,
            "residuals": self.residuals
        })

        return df