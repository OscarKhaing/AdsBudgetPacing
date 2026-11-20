"""Shared matplotlib style and colors for the report figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sim.config import FIGS_DIR  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AGENT_COLORS = {
    "even": SERIES[0], "adaptive": SERIES[1], "Q[hour,budget]": SERIES[2],
    "Q[hour]": SERIES[3], "Q[hour,budget,price]": SERIES[4], "random": MUTED,
}
AGENT_LABELS = {
    "even": "Even pacing", "adaptive": "Adaptive pacing",
    "Q[hour,budget]": "Q: hour+budget", "Q[hour]": "Q: hour",
    "Q[hour,budget,price]": "Q: hour+budget+price", "random": "Random α",
}
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
                   "#0d366b"]

TEXT_WIDTH = 5.5


def setup():
    """Apply a quiet, print-friendly style."""
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 4,
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 220,
    })


def save(fig, filename):
    """Save a figure into results/figs/ and close it."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGS_DIR / filename
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return figure_path
