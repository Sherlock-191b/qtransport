"""
qtransport.core.fitting_engine
==============================

Generic nonlinear fitting engine for magnetotransport models.

The fitting engine is model-agnostic. Any model inheriting
BaseModel can be optimized using nonlinear least squares.

Key responsibilities:

• perform parameter optimization
• compute residuals
• estimate covariance matrix
• compute fit statistics
• return standardized FitResult object
"""

import numpy as np
from scipy.optimize import least_squares

from core.data_model import FitResult
from core.statistics import (
    chi_square,
    reduced_chi_square,
    akaike_information_criterion,
    bayesian_information_criterion,
    parameter_uncertainties
)


# ============================================================
# FITTING ENGINE
# ============================================================

def fit_model(model, dataset):
    """
    Fit a BaseModel to a TransportDataset.

    Parameters
    ----------
    model : BaseModel
        Physics model to be fitted
    dataset : TransportDataset
        Experimental dataset

    Returns
    -------
    FitResult
    """

    B = dataset.B_field
    rho_xx = dataset.rho_xx

    # --------------------------------------------------------
    # Initial parameter guess from model
    # --------------------------------------------------------

    p0 = model.initial_guess(B, rho_xx)

    # --------------------------------------------------------
    # Residual function used by least_squares
    # --------------------------------------------------------

    def residuals(params):
        """
        Compute residual vector for least squares.

        Supports models that return either:
        - single observable (rho_xx)
        - two observables (rho_xx, rho_xy)
        """

        predicted = model.model_function(B, *params)

        # ------------------------------------------------
        # Case 1 : model returns only rho_xx
        # ------------------------------------------------
        if isinstance(predicted, np.ndarray):

            res = rho_xx - predicted
            return res

        # ------------------------------------------------
        # Case 2 : model returns rho_xx and rho_xy
        # ------------------------------------------------
        elif isinstance(predicted, tuple):

            rho_xx_model, rho_xy_model = predicted

            res_xx = dataset.rho_xx - rho_xx_model
            res_xy = dataset.rho_xy - rho_xy_model

            # flatten residuals into one vector
            return np.concatenate([res_xx, res_xy])

        else:
            raise ValueError("Model returned unsupported output format.")


    # --------------------------------------------------------
    # Perform nonlinear optimization
    # --------------------------------------------------------

    result = least_squares(residuals, p0)

    params_opt = result.x

    # --------------------------------------------------------
    # Compute fitted curve
    # --------------------------------------------------------

    prediction = model.model_function(B, *params_opt)

    # Some models return (rho_xx, rho_xy)
    if isinstance(prediction, tuple):
        fitted_curve = prediction[0]   # store rho_xx
    else:
        fitted_curve = prediction
    # --------------------------------------------------------
    # Compute covariance matrix approximation
    # --------------------------------------------------------

    try:
        J = result.jac
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = None

    # --------------------------------------------------------
    # Compute statistics
    # --------------------------------------------------------

    chi2 = chi_square(rho_xx, fitted_curve)

    n_points = len(rho_xx)
    n_params = len(params_opt)

    chi2_red = reduced_chi_square(chi2, n_points, n_params)

    AIC = akaike_information_criterion(chi2, n_params, n_points)
    BIC = bayesian_information_criterion(chi2, n_params, n_points)

    param_errors = parameter_uncertainties(cov)

    # --------------------------------------------------------
    # Construct FitResult
    # --------------------------------------------------------

    fit_result = FitResult(

        model_name=model.__class__.__name__,

        parameters=params_opt,

        covariance_matrix=cov,

        parameter_errors=param_errors,

        chi_square=chi2,

        reduced_chi_square=chi2_red,

        AIC=AIC,

        BIC=BIC,

        residuals=rho_xx - fitted_curve,

        fitted_curve=fitted_curve,

        success_flag=result.success,

        message=result.message

    )

    return fit_result