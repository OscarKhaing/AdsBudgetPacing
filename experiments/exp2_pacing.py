"""Observation 2: which pacing strategy gets the most value from the same budget.

Even pacing and adaptive pacing (both tuned on training days), Q-learning on
(hour, budget) with 5 runs, and a random-alpha sanity check, all on the 20 test days.
Run from the project root:  python experiments/exp2_pacing.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.adaptive import AdaptivePacing  # noqa: E402
from agents.fixed import EvenPacing  # noqa: E402
from agents.random_alpha import RandomAlpha  # noqa: E402
from experiments import plotting  # noqa: E402
from experiments.qtraining import (Q_CONFIGS, USED_CONFIG, evaluate_runs,  # noqa: E402
                                   save_tables, train_runs)
from experiments.runner import TEST_SEEDS, TUNE_SEEDS, evaluate, summarize  # noqa: E402
from sim.config import RESULTS_DIR  # noqa: E402
from sim.env import ensure_calibrated  # noqa: E402

OBS_VARS = ["hour", "budget"]
Q_NAME = "Q[hour,budget]"
AGENT_ORDER = ["even", "adaptive", Q_NAME, "random"]


def fit_baselines():
    even = EvenPacing().fit({}, TUNE_SEEDS)
    adaptive = AdaptivePacing().fit({}, TUNE_SEEDS)
    even.grid_.to_csv(RESULTS_DIR / "obs2_even_grid.csv", index=False)
    adaptive.grid_.to_csv(RESULTS_DIR / "obs2_adaptive_grid.csv", index=False)
    return even, adaptive


def train_q_settings():
    """Train every Q-learning setting (5 runs each); return test values and the used agents."""
    test_rows, used_agents, used_curve = [], None, None
    for config_name, config in Q_CONFIGS.items():
        agents, learning_curves = train_runs(OBS_VARS, config)
        test_episodes, _ = evaluate_runs(agents)
        for run, run_episodes in test_episodes.groupby("run"):
            test_rows.append({"config": config_name, "run": run, **config,
                              "test_value": run_episodes.total_reward.mean()})
        if config_name == USED_CONFIG:
            used_agents, used_curve = agents, learning_curves
        print(f"trained {config_name}")
    return pd.DataFrame(test_rows), used_agents, used_curve


def compare_agents(even, adaptive, q_agents):
    """Evaluate every agent on the test days; return (episodes, hourly)."""
    episode_tables, hourly_tables = [], []
    for agent in (even, adaptive, RandomAlpha(seed=0)):
        episodes, hourly = evaluate(agent, {}, TEST_SEEDS)
        episode_tables.append(episodes)
        hourly_tables.append(hourly)
    q_episodes, q_hourly = evaluate_runs(q_agents)
    episode_tables.append(q_episodes)
    hourly_tables.append(q_hourly)
    return (pd.concat(episode_tables, ignore_index=True),
            pd.concat(hourly_tables, ignore_index=True))


def cumulative_spend_by_hour(hourly):
    """Mean cumulative spend at the end of each hour, per agent (averaged over days and runs)."""
    hourly = hourly.assign(run=hourly["run"].fillna(-1))
    hourly = hourly.sort_values(["agent", "run", "seed", "hour"])
    hourly["cumulative_spend"] = hourly.groupby(["agent", "run", "seed"]).spend.cumsum()
    return hourly.groupby(["agent", "hour"]).cumulative_spend.mean().unstack("agent")


def plot_value(summary):
    plotting.setup()
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(2.6, 2.1))
    rows = summary.set_index("agent").loc[AGENT_ORDER[::-1]]
    positions = np.arange(len(rows))
    for position, (agent, row) in zip(positions, rows.iterrows()):
        axis.errorbar(row["mean"], position, xerr=[[row["mean"] - row.ci_low],
                                                   [row.ci_high - row["mean"]]],
                      fmt="o", color=plotting.AGENT_COLORS[agent], ms=6, capsize=2)
        axis.text(row["mean"], position + 0.22, f"{row['mean']:.2f}", ha="center",
                  va="bottom", fontsize=6.5, color=plotting.INK2)
    axis.set_yticks(positions, [plotting.AGENT_LABELS[agent] for agent in rows.index])
    axis.set_ylim(-0.5, len(rows) - 0.3)
    axis.set_xlim(0, rows["mean"].max() * 1.15)
    axis.set_xlabel("expected clicks per test day")
    axis.grid(axis="y", visible=False)
    fig.tight_layout()
    return plotting.save(fig, "obs2_pacing_value.png")


def plot_spend_over_day(cumulative_spend, budget):
    plotting.setup()
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(3.0, 2.1))
    hours_elapsed = np.arange(0, 25)
    axis.plot(hours_elapsed, budget * hours_elapsed / 24, color=plotting.INK2, ls="--", lw=1,
              label="even spending")
    for agent in AGENT_ORDER:
        cumulative = np.concatenate([[0.0], cumulative_spend[agent].to_numpy()])
        axis.plot(hours_elapsed, cumulative, color=plotting.AGENT_COLORS[agent],
                  label=plotting.AGENT_LABELS[agent])
    axis.set(xlabel="hours into the day", ylabel="spent so far (fen)", xlim=(0, 24),
             ylim=(0, budget * 1.05))
    axis.set_xticks([0, 6, 12, 18, 24])
    axis.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.5)
    fig.tight_layout()
    return plotting.save(fig, "obs2_spend_over_day.png")


def plot_learning_curve(learning_curve):
    plotting.setup()
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(2.6, 2.1))
    by_episode = learning_curve.groupby("episode").mean_reward
    color = plotting.AGENT_COLORS[Q_NAME]
    axis.fill_between(by_episode.min().index, by_episode.min(), by_episode.max(), color=color,
                      alpha=0.2, lw=0)
    axis.plot(by_episode.mean(), color=color)
    axis.set(xlabel="training day", ylabel="clicks per day (mean of 100)")
    fig.tight_layout()
    return plotting.save(fig, "obs2_learning_curve.png")


def main():
    config = ensure_calibrated()
    even, adaptive = fit_baselines()
    print(f"even alpha = {even.alpha}; adaptive eta = {adaptive.eta}, alpha0 = {adaptive.alpha0}")

    q_settings, q_agents, learning_curve = train_q_settings()
    save_tables(q_agents)
    q_settings.to_csv(RESULTS_DIR / "obs2_q_configs.csv", index=False)
    learning_curve.to_csv(RESULTS_DIR / "obs2_learning_curve.csv", index=False)

    episodes, hourly = compare_agents(even, adaptive, q_agents)
    summary = summarize(episodes).set_index("agent").loc[AGENT_ORDER].reset_index()
    q_runs = (episodes[episodes.agent == Q_NAME].groupby("run").total_reward.mean()
              .rename("mean_value").reset_index())
    cumulative_spend = cumulative_spend_by_hour(hourly)
    summary.to_csv(RESULTS_DIR / "obs2.csv", index=False)
    q_runs.to_csv(RESULTS_DIR / "obs2_q_runs.csv", index=False)

    plot_value(summary)
    plot_spend_over_day(cumulative_spend, config["budget"])
    plot_learning_curve(learning_curve)

    with pd.option_context("display.width", 160):
        print(summary.round(3).to_string(index=False))
        print(q_runs.round(3).to_string(index=False))
        print(q_settings.groupby("config", sort=False).test_value.agg(["mean", "std"]).round(3))
        print("share of budget spent after 6, 12, 18 hours:")
        print((cumulative_spend.loc[[5, 11, 17], AGENT_ORDER] / config["budget"]).round(2))


if __name__ == "__main__":
    main()
