"""Observation 1: how daily value and spend respond to the bid multiplier alpha.

FixedAlpha at alpha = 0, 0.25, ..., 3.0 on the 20 test days with no budget limit.
Run from the project root:  python experiments/exp1_market_response.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.fixed import FixedAlpha  # noqa: E402
from experiments import plotting  # noqa: E402
from experiments.runner import TEST_SEEDS, evaluate, t_interval  # noqa: E402
from sim.config import RESULTS_DIR  # noqa: E402
from sim.env import ensure_calibrated  # noqa: E402

ALPHAS = np.round(np.arange(0, 3.01, 0.25), 2)
NO_BUDGET = {"budget": float("inf")}


def run():
    """Sweep alpha with no budget limit; return mean daily value and spend per alpha."""
    sweep_rows = []
    for alpha in ALPHAS:
        episodes, _ = evaluate(FixedAlpha(alpha), NO_BUDGET, TEST_SEEDS)
        value_mean, value_ci = t_interval(episodes.total_reward)
        spend_mean, spend_ci = t_interval(episodes.total_spend)
        sweep_rows.append({"alpha": alpha, "value_mean": value_mean, "value_ci": value_ci,
                           "spend_mean": spend_mean, "spend_ci": spend_ci})
    sweep = pd.DataFrame(sweep_rows)
    sweep["marginal_value_per_fen"] = sweep.value_mean.diff() / sweep.spend_mean.diff()
    return sweep


def plot(sweep, budget):
    """Two panels: value vs. alpha, and spend vs. alpha with the daily budget line."""
    plotting.setup()
    import matplotlib.pyplot as plt

    fig, (value_axis, spend_axis) = plt.subplots(1, 2, figsize=(plotting.TEXT_WIDTH, 2.1))
    blue = plotting.SERIES[0]
    for axis, prefix in [(value_axis, "value"), (spend_axis, "spend")]:
        mean, half_width = sweep[f"{prefix}_mean"], sweep[f"{prefix}_ci"]
        axis.fill_between(sweep.alpha, mean - half_width, mean + half_width, color=blue,
                          alpha=0.18, lw=0)
        axis.plot(sweep.alpha, mean, "-o", color=blue)
        axis.set_xlim(-0.05, 3.05)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("bid multiplier α")
    value_axis.set(ylabel="expected clicks per day", title="(a) Value won")
    spend_axis.axhline(budget, color=plotting.INK2, ls="--", lw=1)
    spend_axis.text(3.0, budget, f"daily budget = {budget:.0f} fen", color=plotting.INK2,
                    ha="right", va="bottom", fontsize=6.5)
    spend_axis.set(ylabel="fen per day", title="(b) Spend")
    fig.tight_layout(w_pad=1.5)
    return plotting.save(fig, "obs1_market_response.png")


def main():
    config = ensure_calibrated()
    sweep = run()
    RESULTS_DIR.mkdir(exist_ok=True)
    sweep.to_csv(RESULTS_DIR / "obs1.csv", index=False)
    figure_path = plot(sweep, config["budget"])
    with pd.option_context("display.width", 160):
        print(sweep.round(4).to_string(index=False))
    print(f"saved {RESULTS_DIR / 'obs1.csv'} and {figure_path}")


if __name__ == "__main__":
    main()
