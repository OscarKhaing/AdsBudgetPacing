"""Stationary first-price auction simulator that resamples the iPinYou logs."""

from functools import lru_cache

import numpy as np
import pandas as pd

from sim.config import PROCESSED_DIR, load_config, update_config

CALIBRATION_SEED = 12345


@lru_cache(maxsize=None)
def load_hours(split: str, campaign: str = "1458"):
    """Return 24 (pctr, price) array pairs, one per hour of day."""
    logs = pd.read_parquet(PROCESSED_DIR / f"{campaign}_{split}.parquet")
    auctions_per_hour = []
    for hour in range(24):
        hour_rows = logs[logs.hour == hour]
        auctions_per_hour.append((hour_rows.pctr.to_numpy(np.float64),
                                  hour_rows.price.to_numpy(np.float64)))
    return auctions_per_hour


class AuctionEnv:
    """One day of hourly first-price auctions for an advertiser with a daily budget.

    Each hour, `n_per_step` logged auctions from that hour are drawn with replacement;
    the agent bids alpha * ecpc * pctr on each and pays its own bid when it beats the
    logged market price. budget=None uses budget_frac * ref_daily_spend from
    config.json; budget=float("inf") turns the budget limit off.
    """

    def __init__(self, split="train", T=24, n_per_step=500,
                 budget=None, budget_frac=0.3, seed=0):
        config = load_config()
        self.ecpc = config["ecpc"]
        self.auctions_per_hour = load_hours(split)
        self.T, self.n_per_step = T, n_per_step
        if budget is None:
            budget = budget_frac * config["ref_daily_spend"]
        self.budget = float(budget)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _obs(self, last_prices):
        return {"t": self.hour, "budget_left": self.budget_left, "budget": self.budget,
                "last_prices": last_prices}

    def reset(self) -> dict:
        self.hour = 0
        self.budget_left = self.budget
        return self._obs(np.empty(0))

    def step(self, alpha: float):
        """Run one hour of auctions at multiplier `alpha`; return (obs, reward, done, info)."""
        assert self.hour < self.T, "episode is over; call reset()"
        hour_pctr, hour_price = self.auctions_per_hour[self.hour]
        sample_indices = self.rng.integers(0, len(hour_price), self.n_per_step)
        pctr = hour_pctr[sample_indices]
        price = hour_price[sample_indices]

        bids = alpha * self.ecpc * pctr
        won_indices = np.flatnonzero(bids > price)
        cumulative_cost = np.cumsum(bids[won_indices])
        n_paid_wins = int(np.searchsorted(cumulative_cost, self.budget_left, side="right"))
        paid_indices = won_indices[:n_paid_wins]
        spend = float(cumulative_cost[n_paid_wins - 1]) if n_paid_wins else 0.0
        reward = float(pctr[paid_indices].sum())

        self.budget_left -= spend
        self.hour += 1
        done = self.hour == self.T
        info = {"spend": spend, "wins": n_paid_wins, "wins_uncapped": len(won_indices)}
        return self._obs(price), reward, done, info


def calibrate(n_episodes=200, budget_frac=0.3, seed=CALIBRATION_SEED):
    """Set the daily budget and price-bin cutoffs from alpha = 1 runs on the train split.

    budget = budget_frac * mean unconstrained daily spend; price bins = 33rd and 67th
    percentiles of the per-hour mean sampled price. Both are written to config.json.
    """
    env = AuctionEnv(split="train", budget=float("inf"), seed=seed)
    daily_spends, hourly_mean_prices = [], []
    for _ in range(n_episodes):
        env.reset()
        daily_spend = 0.0
        for _ in range(env.T):
            obs, _, _, info = env.step(1.0)
            daily_spend += info["spend"]
            hourly_mean_prices.append(obs["last_prices"].mean())
        daily_spends.append(daily_spend)
    ref_daily_spend = float(np.mean(daily_spends))
    price_cutoffs = np.percentile(hourly_mean_prices, [33, 67])
    return update_config(ref_daily_spend=ref_daily_spend, budget_frac=budget_frac,
                         budget=budget_frac * ref_daily_spend,
                         price_bins=[float(cutoff) for cutoff in price_cutoffs],
                         calibration={"alpha": 1.0, "n_episodes": n_episodes, "seed": seed,
                                      "daily_spend_std": float(np.std(daily_spends))})


def ensure_calibrated():
    """Run `calibrate` only if config.json lacks the calibrated values."""
    config = load_config()
    if not {"ref_daily_spend", "budget", "price_bins"} <= config.keys():
        config = calibrate()
    return config
