"""Unit tests for the pacing agents."""

from itertools import combinations

import numpy as np

from agents.adaptive import AdaptivePacing
from agents.qlearning import ALPHAS, QAgent
from agents.random_alpha import RandomAlpha

ALL_OBS_VAR_SETS = [list(var_set) for size in (1, 2, 3)
                    for var_set in combinations(["hour", "budget", "price"], size)]


def make_obs(t=0, budget_left=240.0, budget=240.0, last_prices=()):
    return {"t": t, "budget_left": budget_left, "budget": budget,
            "last_prices": np.asarray(last_prices, dtype=float)}


def test_adaptive_raises_alpha_after_underspending():
    agent = AdaptivePacing(alpha0=1.0, eta=0.3, T=24)
    agent.update(make_obs(t=0, budget_left=240.0), 1.0, 0.0,
                 make_obs(t=1, budget_left=235.0), False)
    assert agent.alpha > 1.0
    assert np.isclose(agent.alpha, np.exp(0.3 * 0.5))


def test_adaptive_lowers_alpha_after_overspending():
    agent = AdaptivePacing(alpha0=1.0, eta=0.3, T=24)
    agent.update(make_obs(t=0, budget_left=240.0), 1.0, 0.0,
                 make_obs(t=1, budget_left=220.0), False)
    assert agent.alpha < 1.0
    assert np.isclose(agent.alpha, np.exp(-0.3 * 1.0))


def test_adaptive_respects_bounds_and_resets():
    agent = AdaptivePacing(alpha0=2.9, eta=1.0, T=24)
    agent.update(make_obs(t=0, budget_left=240.0), 2.9, 0.0,
                 make_obs(t=1, budget_left=240.0), False)
    assert agent.alpha == 3.0
    agent.reset()
    assert agent.alpha == 2.9


def test_discretize_in_range_for_every_obs_vars_combination():
    rng = np.random.default_rng(0)
    for obs_vars in ALL_OBS_VAR_SETS:
        agent = QAgent(obs_vars)
        observations = [make_obs(t=0), make_obs(t=23, budget_left=0.0),
                        make_obs(t=12, budget_left=240.0, last_prices=[]),
                        make_obs(t=5, budget_left=239.99, last_prices=[0.0]),
                        make_obs(t=7, budget_left=1.0, last_prices=[10.0])]
        observations += [make_obs(t=int(rng.integers(24)),
                                  budget_left=float(rng.uniform(0, 240)),
                                  last_prices=rng.uniform(0, 0.3, 50)) for _ in range(200)]
        for obs in observations:
            state = agent.discretize(obs)
            assert len(state) == len(obs_vars)
            assert all(0 <= index < size for index, size in zip(state, agent.state_shape))
        if "price" in obs_vars:
            state = agent.discretize(make_obs(last_prices=[]))
            assert state[obs_vars.index("price")] == 1


def test_greedy_act_returns_argmax():
    agent = QAgent(["hour", "budget"], seed=0)
    obs = make_obs(t=3, budget_left=120.0)
    state = agent.discretize(obs)
    agent.Q[state] = np.linspace(1, 0, len(ALPHAS))
    agent.Q[state + (4,)] = 5.0
    agent.epsilon = 0.0
    assert all(agent.act(obs) == ALPHAS[4] for _ in range(20))
    agent.epsilon, agent.greedy = 1.0, True
    assert all(agent.act(obs) == ALPHAS[4] for _ in range(20))


def test_q_update_matches_hand_computation():
    agent = QAgent(["hour", "budget"], lr=0.5)
    obs, next_obs = make_obs(t=2, budget_left=200.0), make_obs(t=3, budget_left=150.0)
    state, next_state = agent.discretize(obs), agent.discretize(next_obs)
    action_index = 2
    agent.Q[state + (action_index,)] = 0.4
    agent.Q[next_state + (7,)] = 2.0
    agent.update(obs, ALPHAS[action_index], 1.0, next_obs, done=False)
    assert np.isclose(agent.Q[state + (action_index,)], 0.4 + 0.5 * (1.0 + 2.0 - 0.4))
    agent.Q[state + (action_index,)] = 0.4
    agent.update(obs, ALPHAS[action_index], 1.0, next_obs, done=True)
    assert np.isclose(agent.Q[state + (action_index,)], 0.4 + 0.5 * (1.0 - 0.4))


def test_epsilon_schedule():
    agent = QAgent()
    assert agent.epsilon_at(0, 1000) == 1.0
    assert np.isclose(agent.epsilon_at(350, 1000), 1.0 - 0.5 * 0.95)
    assert np.isclose(agent.epsilon_at(700, 1000), 0.05)
    assert np.isclose(agent.epsilon_at(999, 1000), 0.05)


def test_random_alpha_draws_from_action_set():
    agent = RandomAlpha(seed=0)
    chosen_alphas = {agent.act(make_obs()) for _ in range(500)}
    assert chosen_alphas == set(ALPHAS.tolist())
