"""Train, save, load and evaluate independent Q-learning runs (shared by Obs. 2 and 3).

The first setting (5,000 episodes, lr 0.1) was far behind the pacing rules, so we also
tried longer training and a smaller lr. We use 20,000 episodes with lr 0.1.
"""

from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from agents.qlearning import QAgent
from experiments.runner import TEST_SEEDS, evaluate
from sim.config import RESULTS_DIR

TRAIN_RUNS = [0, 1, 2, 3, 4]
Q_CONFIGS = {
    "first setting": {"n_episodes": 5_000, "lr": 0.1},
    "longer training": {"n_episodes": 20_000, "lr": 0.1},
    "smaller lr": {"n_episodes": 20_000, "lr": 0.05},
}
USED_CONFIG = "longer training"


def _train_one_run(job):
    obs_vars, config, run = job
    agent = QAgent(obs_vars, lr=config["lr"], seed=run)
    learning_curve = agent.train({}, n_episodes=config["n_episodes"], seed=run)
    return agent, learning_curve.assign(run=run)


def train_runs(obs_vars, config, runs=TRAIN_RUNS):
    """Train one agent per run seed in parallel; return (agents, learning curves)."""
    jobs = [(list(obs_vars), config, run) for run in runs]
    with ProcessPoolExecutor(max_workers=len(runs)) as executor:
        results = list(executor.map(_train_one_run, jobs))
    agents = [agent for agent, _ in results]
    learning_curves = pd.concat([curve for _, curve in results], ignore_index=True)
    return agents, learning_curves


def table_path(obs_vars, run):
    return RESULTS_DIR / f"q_{'_'.join(obs_vars)}_run{run}.npy"


def save_tables(agents, runs=TRAIN_RUNS):
    for agent, run in zip(agents, runs):
        agent.save(table_path(agent.obs_vars, run))


def load_tables(obs_vars, config, runs=TRAIN_RUNS):
    return [QAgent(obs_vars, lr=config["lr"], seed=run).load(table_path(obs_vars, run))
            for run in runs]


def evaluate_runs(agents, runs=TRAIN_RUNS, seeds=TEST_SEEDS):
    """Evaluate each trained agent greedily on the test days; add a `run` column."""
    episode_tables, hourly_tables = [], []
    for agent, run in zip(agents, runs):
        agent.greedy = True
        episodes, hourly = evaluate(agent, {}, seeds)
        episode_tables.append(episodes.assign(run=run))
        hourly_tables.append(hourly.assign(run=run))
    return pd.concat(episode_tables, ignore_index=True), pd.concat(hourly_tables, ignore_index=True)
