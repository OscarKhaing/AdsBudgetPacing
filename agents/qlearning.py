"""Tabular Q-learning over the bid multiplier with a configurable state definition."""

from pathlib import Path

import numpy as np
import pandas as pd

from agents.base import Agent
from experiments.runner import run_episode
from sim.config import load_config
from sim.env import AuctionEnv

ALPHAS = np.round(np.arange(0, 2.01, 0.25), 2)
TRAIN_SEED_OFFSET = 10_000


class QAgent(Agent):
    """Epsilon-greedy tabular Q-learning; the state is built from `obs_vars`."""

    def __init__(self, obs_vars=("hour", "budget"), n_budget_bins=10, lr=0.1,
                 eps_start=1.0, eps_end=0.05, decay_frac=0.7, seed=0):
        self.obs_vars = list(obs_vars)
        self.n_budget_bins = n_budget_bins
        self.lr = lr
        self.eps_start, self.eps_end, self.decay_frac = eps_start, eps_end, decay_frac
        size_per_var = {"hour": 24, "budget": n_budget_bins, "price": 3}
        self.state_shape = tuple(size_per_var[var] for var in self.obs_vars)
        self.Q = np.zeros(self.state_shape + (len(ALPHAS),))
        self.visits = np.zeros(self.Q.shape, dtype=np.int64)
        self.price_bins = load_config()["price_bins"] if "price" in self.obs_vars else None
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.0
        self.greedy = False
        self.name = "Q[" + ",".join(self.obs_vars) + "]"

    def discretize(self, obs) -> tuple:
        """Map an observation to a tuple of integer state indices."""
        state = []
        for var in self.obs_vars:
            if var == "hour":
                state.append(int(obs["t"]))
            elif var == "budget":
                budget_fraction_left = obs["budget_left"] / obs["budget"]
                budget_bin = int(self.n_budget_bins * budget_fraction_left)
                state.append(min(budget_bin, self.n_budget_bins - 1))
            elif var == "price":
                last_prices = obs["last_prices"]
                if len(last_prices) == 0:
                    state.append(1)
                else:
                    state.append(int(np.searchsorted(self.price_bins, last_prices.mean(),
                                                     side="right")))
        return tuple(state)

    def act(self, obs: dict) -> float:
        """Epsilon-greedy alpha (epsilon = 0 in greedy mode); ties broken at random."""
        epsilon = 0.0 if self.greedy else self.epsilon
        if self.rng.random() < epsilon:
            action_index = int(self.rng.integers(len(ALPHAS)))
        else:
            q_values = self.Q[self.discretize(obs)]
            best_actions = np.flatnonzero(q_values == q_values.max())
            action_index = int(self.rng.choice(best_actions))
        return float(ALPHAS[action_index])

    def update(self, obs, alpha, reward, next_obs, done):
        """Q[s,a] += lr * (r + max_a' Q[s',a'] - Q[s,a]), with no future term at the last hour."""
        if self.greedy:
            return
        action_index = int(np.argmin(np.abs(ALPHAS - alpha)))
        state_action = self.discretize(obs) + (action_index,)
        future_value = 0.0 if done else self.Q[self.discretize(next_obs)].max()
        td_error = reward + future_value - self.Q[state_action]
        self.Q[state_action] += self.lr * td_error
        self.visits[state_action] += 1

    def epsilon_at(self, episode, n_episodes):
        """Linear decay from eps_start to eps_end over the first decay_frac of training."""
        decay_progress = min(episode / (self.decay_frac * n_episodes), 1.0)
        return self.eps_start + decay_progress * (self.eps_end - self.eps_start)

    def train(self, env_kwargs=None, n_episodes=5000, seed=0, log_every=100):
        """Train on the train split; log the mean episode reward of every `log_every` episodes."""
        env = AuctionEnv(split="train", seed=TRAIN_SEED_OFFSET + seed, **(env_kwargs or {}))
        log_rows, block_rewards = [], []
        for episode in range(n_episodes):
            self.epsilon = self.epsilon_at(episode, n_episodes)
            episode_totals, _ = run_episode(self, env)
            block_rewards.append(episode_totals["total_reward"])
            if (episode + 1) % log_every == 0:
                log_rows.append({"episode": episode + 1, "mean_reward": np.mean(block_rewards),
                                 "epsilon": self.epsilon})
                block_rewards = []
        self.greedy = True
        return pd.DataFrame(log_rows)

    def greedy_alpha(self):
        """Greedy alpha for every state; NaN where the state was never visited."""
        greedy_alphas = ALPHAS[self.Q.argmax(axis=-1)].astype(float)
        greedy_alphas[self.visits.sum(axis=-1) == 0] = np.nan
        return greedy_alphas

    def save(self, path):
        """Save the Q-table to `path` (.npy) and the visit counts beside it."""
        path = Path(path)
        np.save(path, self.Q)
        np.save(path.with_name(path.stem + "_visits.npy"), self.visits)

    def load(self, path):
        """Load a Q-table (and visit counts) saved by `save`; switch to greedy mode."""
        path = Path(path)
        self.Q = np.load(path)
        assert self.Q.shape == self.state_shape + (len(ALPHAS),), "Q-table shape mismatch"
        visits_path = path.with_name(path.stem + "_visits.npy")
        if visits_path.exists():
            self.visits = np.load(visits_path)
        self.greedy = True
        return self
