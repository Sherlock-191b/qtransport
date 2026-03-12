"""
report_generator.py

Generate analysis reports for magnetotransport fitting results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from report.figure_style import apply_style, set_transport_axes


class ReportGenerator:
    """
    Generates plots and parameter summaries for magnetotransport analysis.
    """

    def __init__(self, dataset, fit_result):
        """
        Parameters
        ----------
        dataset : TransportDataset
        fit_result : FitResult
        """

        self.dataset = dataset
        self.fit = fit_result

    # ---------------------------------------------------------
    # Plot measured vs fitted data
    # ---------------------------------------------------------

    def plot_fit(self):
        """
        Plot experimental data and fitted curve.
        """

        apply_style()

        B = self.dataset.B_field
        rho = self.dataset.rho_xx

        fitted = self.fit.fitted_curve

        fig, ax = plt.subplots()

        ax.scatter(B, rho, label="Data", s=15)

        if fitted is not None:
            ax.plot(B, fitted, label="Fit")

        set_transport_axes(ax)

        ax.legend()

        ax.set_title(self.fit.model_name)

        return fig

    # ---------------------------------------------------------
    # Parameter summary table
    # ---------------------------------------------------------

    def parameter_table(self):
        """
        Create dataframe summarizing fitted parameters.
        """

        params = self.fit.parameters
        errors = self.fit.parameter_errors

        data = {
            "parameter": list(params.keys()),
            "value": list(params.values()),
            "error": list(errors.values()),
        }

        return pd.DataFrame(data)

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    def save_report(self, filename="analysis_report.pdf"):
        """
        Export report as PDF.

        Includes:
        - Fit plot
        - Parameter table
        - Statistics
        """

        fig = self.plot_fit()
        table = self.parameter_table()

        with PdfPages(filename) as pdf:

            # plot page
            pdf.savefig(fig)
            plt.close(fig)

            # table page
            fig2, ax = plt.subplots()

            ax.axis("off")

            tbl = ax.table(
                cellText=table.values,
                colLabels=table.columns,
                loc="center"
            )

            tbl.scale(1, 1.5)

            ax.set_title("Fitted Parameters")

            pdf.savefig(fig2)

            plt.close(fig2)

    # ---------------------------------------------------------
    # Save figure only
    # ---------------------------------------------------------

    def save_figure(self, filename="fit_plot.png"):
        """
        Export only the main figure.
        """

        fig = self.plot_fit()
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)