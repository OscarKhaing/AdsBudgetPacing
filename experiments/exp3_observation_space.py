"""Observation 3: does changing what the Q-learner observes change what it learns?

Three Q-learners with the same settings as Observation 2 that differ only in their
state: (a) hour; (b) hour + budget, reusing Observation 2's Q-tables; (c) hour + budget
+ recent price.
Run from the project root:  python experiments/exp3_observation_space.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import plotting  # noqa: E402
from experiments.qtraining import (Q_CONFIGS, TRAIN_RUNS, USED_CONFIG,  # noqa: E402
                                   evaluate_runs, load_tables, save_tables, train_runs)
from experiments.runner import summarize  # noqa: E402
from sim.config import RESULTS_DIR  # noqa: E402

VARIANTS = {
    "Q[hour]": ["hour"],
    "Q[hour,budget]": ["hour", "budget"],
    "Q[hour,budget,price]": ["hour", "budget", "price"],
}
REUSED_VARIANT = "Q[hour,budget]"
PRICE_BIN_LABELS = ["low", "middle", "high"]


def train_or_load_variants(config):
    """Train (a) and (c) from scratch; load (b)'s tables from Observation 2."""
    agents_per_variant = {}
    for variant, obs_vars in VARIANTS.items():
        if variant == REUSED_VARIANT:
            agents_per_variant[variant] = load_tables(obs_vars, config)
            continue
        agents, _ = train_runs(obs_vars, config)
        save_tables(agents)
        agents_per_variant[variant] = agents
        print(f"trained {variant}")
    return agents_per_variant


def policy_disagreement(first_policy, second_policy):
    """Fraction of states visited by both policies where their greedy alphas differ."""
    both_visited = ~np.isnan(first_policy) & ~np.isnan(second_policy)
    return float((first_policy[both_visited] != second_policy[both_visited]).mean())


def disagreement_table(agents_per_variant):
    """(c) vs (b) per run, averaged over price bins; and (b) vs (b) across runs for scale."""
    hour_budget_policies = [agent.greedy_alpha() for agent in agents_per_variant[REUSED_VARIANT]]
    price_policies = [agent.greedy_alpha()
                      for agent in agents_per_variant["Q[hour,budget,price]"]]
    rows = []
    for run, (price_policy, hour_budget_policy) in enumerate(zip(price_policies,
                                                                 hour_budget_policies)):
        per_price_bin = [policy_disagreement(price_policy[:, :, price_bin], hour_budget_policy)
                         for price_bin in range(3)]
        rows.append({"comparison": "(c) vs (b), same run", "pair": str(run),
                     "disagreement": float(np.mean(per_price_bin))})
    for first_run, second_run in itertools.combinations(TRAIN_RUNS, 2):
        rows.append({"comparison": "(b) vs (b), different runs",
                     "pair": f"{first_run}-{second_run}",
                     "disagreement": policy_disagreement(hour_budget_policies[first_run],
                                                         hour_budget_policies[second_run])})
    return pd.DataFrame(rows)


def plot_value(summary, variants):
    plotting.setup()
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(2.6, 2.1))
    rows = summary.set_index("agent").loc[variants[::-1]]
    positions = np.arange(len(rows))
    for position, (variant, row) in zip(positions, rows.iterrows()):
        axis.errorbar(row["mean"], position,
                      xerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
                      fmt="o", ms=6, capsize=2, color=plotting.AGENT_COLORS[variant])
        axis.text(row["mean"], position + 0.22, f"{row['mean']:.2f}", ha="center",
                  va="bottom", fontsize=6.5, color=plotting.INK2)
    axis.set_yticks(positions, [plotting.AGENT_LABELS[variant] for variant in rows.index])
    axis.set_ylim(-0.5, len(rows) - 0.3)
    axis.set_xlim(0, rows["mean"].max() * 1.2)
    axis.set_xlabel("expected clicks per test day")
    axis.grid(axis="y", visible=False)
    fig.tight_layout()
    return plotting.save(fig, "obs3_value.png")


def plot_policies(agents_per_variant):
    """Greedy alpha of run 0: a line for (a), a heatmap for (b), one heatmap per price bin for (c)."""
    plotting.setup()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list("sequential_blue", plotting.SEQUENTIAL_BLUE)
    colormap.set_bad("#f0efec")
    fig, axes = plt.subplots(2, 3, figsize=(plotting.TEXT_WIDTH, 3.0))
    heatmap_style = {"cmap": colormap, "vmin": 0.0, "vmax": 2.0, "aspect": "auto",
                     "origin": "lower", "interpolation": "nearest"}

    hour_policy = agents_per_variant["Q[hour]"][0].greedy_alpha()
    axes[0, 0].step(np.arange(24), hour_policy, where="mid",
                    color=plotting.AGENT_COLORS["Q[hour]"])
    axes[0, 0].set(xlabel="hour", ylabel="greedy α", ylim=(0, 2.05), xlim=(-0.5, 23.5),
                   title="(a) hour")
    axes[0, 0].set_xticks([0, 6, 12, 18, 23])

    hour_budget_policy = agents_per_variant["Q[hour,budget]"][0].greedy_alpha()
    image = axes[0, 1].imshow(np.ma.masked_invalid(hour_budget_policy.T), **heatmap_style)
    axes[0, 1].set_title("(b) hour + budget")

    price_policy = agents_per_variant["Q[hour,budget,price]"][0].greedy_alpha()
    for price_bin, axis in enumerate(axes[1]):
        axis.imshow(np.ma.masked_invalid(price_policy[:, :, price_bin].T), **heatmap_style)
        axis.set_title(f"(c) {PRICE_BIN_LABELS[price_bin]} last-hour price")
    for axis in [axes[0, 1], *axes[1]]:
        axis.set_xlabel("hour")
        axis.set_xticks([0, 6, 12, 18, 23])
        axis.set_yticks([0, 9], ["0–10%", "90–100%"])
        axis.grid(False)
    for axis in (axes[0, 1], axes[1, 0]):
        axis.set_ylabel("budget left")
    axes[0, 2].axis("off")
    colorbar = fig.colorbar(image, ax=axes[0, 2], location="left", fraction=0.6, aspect=12)
    colorbar.set_label("greedy α (grey: never visited)", fontsize=7)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    return plotting.save(fig, "obs3_policies.png")


def main():
    config = Q_CONFIGS[USED_CONFIG]
    print(f"Q-learning setting from Observation 2: {USED_CONFIG} {config}")
    agents_per_variant = train_or_load_variants(config)

    episode_tables = [evaluate_runs(agents)[0] for agents in agents_per_variant.values()]
    episodes = pd.concat(episode_tables, ignore_index=True)
    variants = list(VARIANTS)
    summary = summarize(episodes).set_index("agent").loc[variants].reset_index()
    summary["n_states"] = [int(np.prod(agents_per_variant[v][0].state_shape)) for v in variants]
    run_means = (episodes.groupby(["agent", "run"]).total_reward.mean()
                 .unstack("agent")[variants].reset_index())
    disagreement = disagreement_table(agents_per_variant)

    summary.to_csv(RESULTS_DIR / "obs3.csv", index=False)
    run_means.to_csv(RESULTS_DIR / "obs3_q_runs.csv", index=False)
    disagreement.to_csv(RESULTS_DIR / "obs3_policy_disagreement.csv", index=False)
    plot_value(summary, variants)
    plot_policies(agents_per_variant)

    with pd.option_context("display.width", 160):
        print(summary.round(3).to_string(index=False))
        print(run_means.round(3).to_string(index=False))
        print(disagreement.groupby("comparison").disagreement.agg(["mean", "std", "min", "max"])
              .round(3))
    hour_only_policy = agents_per_variant["Q[hour]"][0].greedy_alpha()
    print("(a) run 0 greedy alpha by hour:", hour_only_policy.tolist())


if __name__ == "__main__":
    main()
