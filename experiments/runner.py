"""Run agents in the simulator and summarize their results.

Evaluation always uses the test split with TEST_SEEDS; baseline tuning uses the train
split with TUNE_SEEDS. A seed fixes a day's sampled auctions independently of the bids,
so all agents face the same auctions on a given test day (common random numbers).
"""

import numpy as np
import pandas as pd
from scipy import stats

from sim.env import AuctionEnv

TEST_SEEDS = list(range(100, 120))
TUNE_SEEDS = list(range(20_000, 20_050))


def run_episode(agent, env):
    """Play one day; return episode totals and a list of per-hour records."""
    obs = env.reset()
    agent.reset()
    hourly_records, done = [], False
    total_reward = total_spend = 0.0
    hour_budget_ran_out = np.nan
    while not done:
        alpha = agent.act(obs)
        next_obs, reward, done, info = env.step(alpha)
        agent.update(obs, alpha, reward, next_obs, done)
        budget_capped_this_hour = info["wins"] < info["wins_uncapped"]
        if np.isnan(hour_budget_ran_out) and budget_capped_this_hour:
            hour_budget_ran_out = obs["t"]
        hourly_records.append({"hour": obs["t"], "alpha": alpha, "reward": reward,
                               "spend": info["spend"],
                               "budget_left": next_obs["budget_left"]})
        total_reward += reward
        total_spend += info["spend"]
        obs = next_obs
    episode_totals = {"total_reward": total_reward, "total_spend": total_spend,
                      "utilization": total_spend / env.budget,
                      "hour_budget_ran_out": hour_budget_ran_out}
    return episode_totals, hourly_records


def evaluate(agent, env_kwargs=None, seeds=TEST_SEEDS, split="test", name=None):
    """Run `agent` for one day per seed; return (episodes, hourly) DataFrames."""
    name = name or agent.name
    episode_rows, hourly_rows = [], []
    for seed in seeds:
        env = AuctionEnv(split=split, seed=seed, **(env_kwargs or {}))
        episode_totals, hourly_records = run_episode(agent, env)
        episode_rows.append({"agent": name, "seed": seed, **episode_totals})
        hourly_rows += [{"agent": name, "seed": seed, **record} for record in hourly_records]
    return pd.DataFrame(episode_rows), pd.DataFrame(hourly_rows)


def t_interval(values, level=0.95):
    """Mean of `values` and the half-width of its Student-t confidence interval."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return values.mean(), np.nan
    standard_error = values.std(ddof=1) / np.sqrt(n)
    return values.mean(), stats.t.ppf(0.5 + level / 2, n - 1) * standard_error


def summarize(episodes, by="agent", value="total_reward"):
    """Per group: mean `value` with a 95% t-interval, plus mean spend, utilization and
    the mean hour in which the budget ran out (over days on which it ran out)."""
    summary_rows = []
    for group_name, group in episodes.groupby(by, sort=False):
        mean, half_width = t_interval(group[value])
        summary_rows.append({by: group_name, "n": len(group), "mean": mean,
                             "ci_low": mean - half_width, "ci_high": mean + half_width,
                             "mean_spend": group.total_spend.mean(),
                             "mean_utilization": group.utilization.mean(),
                             "mean_hour_ran_out": group.hour_budget_ran_out.mean()})
    return pd.DataFrame(summary_rows)
