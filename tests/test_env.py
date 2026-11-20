"""Tests for the auction simulator (need data/processed/ and config.json)."""

import numpy as np
import pytest

from sim.env import AuctionEnv

SMALL_BUDGET = 50.0


def run_episode(env, alpha_schedule):
    """Run one day with a constant alpha or a per-hour schedule; return the step tuples."""
    env.reset()
    steps = []
    for hour in range(env.T):
        alpha = alpha_schedule[hour] if np.ndim(alpha_schedule) else alpha_schedule
        steps.append(env.step(alpha))
    return steps


def test_budget_never_negative():
    rng = np.random.default_rng(0)
    for seed in range(5):
        env = AuctionEnv(budget=SMALL_BUDGET, seed=seed)
        for obs, _, _, _ in run_episode(env, rng.uniform(0, 3, env.T)):
            assert obs["budget_left"] >= 0.0


def test_alpha_zero_spends_and_wins_nothing():
    env = AuctionEnv(budget=SMALL_BUDGET, seed=1)
    for _, reward, _, info in run_episode(env, 0.0):
        assert reward == 0.0 and info["spend"] == 0.0 and info["wins"] == 0
    assert env.budget_left == SMALL_BUDGET


def test_spend_non_decreasing_in_alpha():
    daily_spends = []
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        env = AuctionEnv(budget=float("inf"), seed=7)
        daily_spends.append(sum(info["spend"] for *_, info in run_episode(env, alpha)))
    assert all(lower <= higher for lower, higher in zip(daily_spends, daily_spends[1:]))
    assert daily_spends[-1] > 0


def test_same_seed_same_episode_different_seed_differs():
    def hourly_rewards(seed):
        env = AuctionEnv(budget=SMALL_BUDGET, seed=seed)
        return [reward for _, reward, _, _ in run_episode(env, 1.0)]
    assert hourly_rewards(3) == hourly_rewards(3)
    assert hourly_rewards(3) != hourly_rewards(4)


def test_done_after_exactly_T_steps():
    env = AuctionEnv(budget=SMALL_BUDGET, seed=0)
    dones = [done for _, _, done, _ in run_episode(env, 1.0)]
    assert dones == [False] * (env.T - 1) + [True]
    with pytest.raises(AssertionError):
        env.step(1.0)


def test_infinite_budget_never_caps_spend():
    env = AuctionEnv(budget=float("inf"), seed=2)
    for *_, info in run_episode(env, 3.0):
        assert info["wins"] == info["wins_uncapped"]
    env = AuctionEnv(budget=SMALL_BUDGET, seed=2)
    assert any(info["wins"] < info["wins_uncapped"] for *_, info in run_episode(env, 3.0))


def test_obs_contents():
    env = AuctionEnv(budget=SMALL_BUDGET, seed=0, n_per_step=100)
    obs = env.reset()
    assert obs["t"] == 0 and obs["budget_left"] == SMALL_BUDGET
    assert len(obs["last_prices"]) == 0
    obs, *_ = env.step(1.0)
    assert obs["t"] == 1 and len(obs["last_prices"]) == 100
