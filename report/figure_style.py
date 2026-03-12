"""
figure_style.py

Defines matplotlib styles for publication-quality figures
used in magnetotransport analysis.
"""

import matplotlib.pyplot as plt


def apply_style():
    """
    Apply global matplotlib styling for all figures.
    """

    plt.rcParams.update({

        # figure
        "figure.figsize": (6, 4),
        "figure.dpi": 120,

        # fonts
        "font.size": 11,
        "font.family": "serif",

        # axis
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 1.2,

        # ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,

        # grid
        "axes.grid": False,

        # legend
        "legend.fontsize": 10,

        # line
        "lines.linewidth": 2

    })


def set_transport_axes(ax):
    """
    Apply standard labels for magnetotransport plots.

    Parameters
    ----------
    ax : matplotlib axis
    """

    ax.set_xlabel("Magnetic Field B (T)")
    ax.set_ylabel("Resistivity (Ω)")
    ax.tick_params(which="both", direction="in")