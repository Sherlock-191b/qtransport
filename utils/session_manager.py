"""
session_manager.py

Manages the analysis session state for qtransport.

This module stores:
- loaded datasets
- model fit results

It allows:
- adding datasets
- storing fit results
- retrieving previous results
- clearing the session

Important:
This module contains NO Streamlit logic.
It is purely backend state management.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


# ============================================================
# SESSION DATA STRUCTURE
# ============================================================

@dataclass
class SessionRecord:
    """
    Represents one model analysis run.

    Attributes
    ----------
    dataset_name : str
        Identifier of dataset used

    model_name : str
        Model used for fitting

    result : object
        FitResult object (defined later in core/data_model)
    """

    dataset_name: str
    model_name: str
    result: object


# ============================================================
# SESSION MANAGER CLASS
# ============================================================

class SessionManager:
    """
    Central session storage for datasets and fit results.

    This class manages the analysis lifecycle during
    a user's working session.
    """

    def __init__(self):
        """Initialize empty session."""
        self.datasets: Dict[str, object] = {}
        self.results: List[SessionRecord] = []

    # --------------------------------------------------------
    # DATASET MANAGEMENT
    # --------------------------------------------------------

    def add_dataset(self, name: str, dataset: object):
        """
        Store dataset in session.

        Parameters
        ----------
        name : str
            Dataset identifier

        dataset : TransportDataset
            Dataset object
        """

        self.datasets[name] = dataset

    def get_dataset(self, name: str):
        """
        Retrieve dataset by name.

        Returns
        -------
        TransportDataset or None
        """

        return self.datasets.get(name)

    def list_datasets(self):
        """Return list of dataset names."""
        return list(self.datasets.keys())

    # --------------------------------------------------------
    # FIT RESULT MANAGEMENT
    # --------------------------------------------------------

    def add_result(self, dataset_name: str, model_name: str, result):
        """
        Store a model fit result.

        Parameters
        ----------
        dataset_name : str
        model_name : str
        result : FitResult
        """

        record = SessionRecord(
            dataset_name=dataset_name,
            model_name=model_name,
            result=result,
        )

        self.results.append(record)

    def get_results(self, dataset_name: Optional[str] = None):
        """
        Retrieve stored results.

        Parameters
        ----------
        dataset_name : optional filter

        Returns
        -------
        list of SessionRecord
        """

        if dataset_name is None:
            return self.results

        return [r for r in self.results if r.dataset_name == dataset_name]

    # --------------------------------------------------------
    # SESSION CONTROL
    # --------------------------------------------------------

    def clear_results(self):
        """Remove all stored fit results."""
        self.results = []

    def clear_all(self):
        """Reset entire session."""
        self.datasets = {}
        self.results = []