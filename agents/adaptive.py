"""AdaptivePacing: a multiplicative feedback controller on the bid multiplier."""

import itertools

import numpy as np
import pandas as pd

from agents.base import Agent


class AdaptivePacing(Agent):
    """Steer each hour's spend toward an even split of the remaining budget.

    After each hour: target = budget_left_before / hours_left_before and
    alpha <- clip(alpha * exp(-eta * (spend - target) / target), 0, alpha_max).
    """

    ETAS = (0.1, 0.3, 1.0)
    ALPHA0S = (0.5, 1.0, 1.5)

    def __init__(self, alpha0=1.0, eta=0.3, T=24, alpha_max=3.0):
        self.alpha0, self.eta, self.T, self.alpha_max = alpha0, eta, T, alpha_max
        self.name = "adaptive"
        self.grid_ = None
        self.reset()

    def reset(self):
        self.alpha = self.alpha0

    def act(self, obs: dict) -> float:
        return self.alpha

    def update(self, obs, alpha, reward, next_obs, done):
        hours_left = self.T - obs["t"]
        target_spend = obs["budget_left"] / hours_left
        if not np.isfinite(target_spend) or target_spend <= 0:
            return
        spend = obs["budget_left"] - next_obs["budget_left"]
        relative_overspend = (spend - target_spend) / target_spend
        new_alpha = self.alpha * np.exp(-self.eta * relative_overspend)
        self.alpha = float(np.clip(new_alpha, 0.0, self.alpha_max))

    def fit(self, env_kwargs, seeds):
        """Keep the (eta, alpha0) with the highest mean reward on train-split days `seeds`."""
        from experiments.runner import evaluate

        grid_rows = []
        for eta, alpha0 in itertools.product(self.ETAS, self.ALPHA0S):
            candidate = AdaptivePacing(alpha0, eta, self.T, self.alpha_max)
            episodes, _ = evaluate(candidate, env_kwargs, seeds, split="train")
            grid_rows.append({"eta": eta, "alpha0": alpha0,
                              "mean_reward": episodes.total_reward.mean(),
                              "mean_utilization": episodes.utilization.mean()})
        self.grid_ = pd.DataFrame(grid_rows)
        best_row = self.grid_.loc[self.grid_.mean_reward.idxmax()]
        self.eta, self.alpha0 = float(best_row.eta), float(best_row.alpha0)
        self.reset()
        return self
