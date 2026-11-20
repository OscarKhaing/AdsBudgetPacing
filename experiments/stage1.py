"""Run all of Stage 1: data checks, calibration, then Observations 1, 2 and 3.

Needs data/processed/ (from `python sim/data_prep.py`). Regenerates every CSV in
results/ and every figure in results/figs/.
Run from the project root:  python experiments/stage1.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import (exp1_market_response, exp2_pacing,  # noqa: E402
                         exp3_observation_space)
from sim.config import PROCESSED_DIR, load_config  # noqa: E402
from sim.env import ensure_calibrated, load_hours  # noqa: E402

EXPERIMENTS = [
    ("Observation 1: market response", exp1_market_response),
    ("Observation 2: pacing comparison", exp2_pacing),
    ("Observation 3: observation space", exp3_observation_space),
]


def check_data():
    """Fail early if the processed data or the eCPC are missing."""
    for split in ("train", "test"):
        parquet_path = PROCESSED_DIR / f"1458_{split}.parquet"
        assert parquet_path.exists(), f"missing {parquet_path}; run python sim/data_prep.py"
        auctions_per_hour = load_hours(split)
        assert all(len(prices) > 0 for _, prices in auctions_per_hour), f"{split}: empty hour"
    assert "ecpc" in load_config(), "config.json has no ecpc; run python sim/data_prep.py"


def main():
    check_data()
    config = ensure_calibrated()
    print(f"eCPC = {config['ecpc']:.2f} fen/click, budget = {config['budget']:.2f} fen/day, "
          f"price bins = {[round(cutoff, 4) for cutoff in config['price_bins']]}")
    for title, experiment in EXPERIMENTS:
        print(f"\n=== {title} ===")
        start_time = time.time()
        experiment.main()
        print(f"--- finished in {time.time() - start_time:.0f} s")


if __name__ == "__main__":
    main()
