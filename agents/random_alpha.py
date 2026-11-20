"""RandomAlpha: a sanity-check agent that ignores the state."""

import numpy as np

from agents.base import Agent
from agents.qlearning import ALPHAS


class RandomAlpha(Agent):
    """Picks alpha uniformly from the Q-learner's action set every hour."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.name = "random"

    def act(self, obs: dict) -> float:
        return float(self.rng.choice(ALPHAS))
