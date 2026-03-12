"""
base_model.py

Defines the abstract base class for all physics models
used in qtransport.

All transport models MUST inherit from this class.

This ensures the fitting engine can interact with
models in a uniform way.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for transport models.
    """

    def __init__(self):
        self.name = self.__class__.__name__

    # --------------------------------------------------------

    @abstractmethod
    def model_function(self, B: np.ndarray, *params):
        """
        Compute model prediction.

        Parameters
        ----------
        B : magnetic field array
        params : model parameters

        Returns
        -------
        predicted observable
        """
        pass

    # --------------------------------------------------------

    @abstractmethod
    def initial_guess(self, B: np.ndarray, data: np.ndarray):
        """
        Provide reasonable starting parameters for fitting.

        Parameters
        ----------
        B : magnetic field
        data : measured observable
        """
        pass

    # --------------------------------------------------------

    @abstractmethod
    def fit(self, B: np.ndarray, data: np.ndarray):
        """
        Fit model to data.

        Returns optimized parameters and covariance.
        """
        pass