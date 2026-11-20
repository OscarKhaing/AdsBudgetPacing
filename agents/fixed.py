"""Constant-multiplier agents: FixedAlpha and EvenPacing."""

import numpy as np
import pandas as pd

from agents.base import Agent


class FixedAlpha(Agent):
    """Bids the same multiplier alpha in every hour."""

    def __init__(self, alpha: float, name: str | None = None):
        self.alpha = float(alpha)
        self.name = name or f"fixed({self.alpha:g})"

    def act(self, obs: dict) -> float:
        return self.alpha


class EvenPacing(FixedAlpha):
    """A single fixed alpha, grid-searched on training days to make the budget last."""

    GRID = np.round(np.arange(0.1, 3.01, 0.1), 2)

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha, name="even")
        self.grid_ = None

    def fit(self, env_kwargs, seeds):
        """Keep the alpha with the highest mean reward on train-split days `seeds`."""
        from experiments.runner import evaluate

        grid_rows = []
        for alpha in self.GRID:
            episodes, _ = evaluate(FixedAlpha(alpha), env_kwargs, seeds, split="train")
            grid_rows.append({"alpha": alpha, "mean_reward": episodes.total_reward.mean(),
                              "mean_spend": episodes.total_spend.mean(),
                              "frac_ran_out": episodes.hour_budget_ran_out.notna().mean()})
        self.grid_ = pd.DataFrame(grid_rows)
        self.alpha = float(self.grid_.loc[self.grid_.mean_reward.idxmax(), "alpha"])
        return self
