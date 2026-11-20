# When Does Q-Learning Beat Budget Pacing in Online Ad Auctions? Evidence from a Stationary Market

Oscar Khaing, Jacob Zhang, Jeffrey Yang, En-Yu Hsu, Justin Shiu
University of California, San Diego (CSE 150A/250A course project)

An advertiser has a fixed daily budget to spend across thousands of online ad auctions.
It does not know which ads will be clicked or what other advertisers will bid. Each
hour, it picks a bid multiplier α that scales all of its bids. We ask when Q-learning,
which learns to bid by trial and error, beats simple rules for spreading the budget over
the day. We model the day as a Markov decision process, simulate it with 3.1 million
real auctions from the iPinYou dataset, and value each ad with a logistic-regression
click model. This repository contains the code, results and figures behind our report,
[`report_final.pdf`](report_final.pdf).

## Main results

We study a stationary market, in which every simulated day follows the same
distribution. The table gives expected clicks per day on 20 held-out test days, with
95% confidence intervals.

| strategy | clicks/day |
| --- | --- |
| Adaptive pacing (adjusts α to stay on a spending schedule) | **3.12** [3.11, 3.13] |
| Even pacing (one tuned α all day) | 3.06 [3.04, 3.08] |
| Q-learning, state = hour + budget left | 2.24 [2.22, 2.26] |
| Q-learning, state = hour + budget left + last hour's price | 1.92 [1.89, 1.95] |
| Random α (sanity check) | 1.41 [1.35, 1.48] |
| Q-learning, state = hour only | 1.25 [1.24, 1.26] |

1. **How the market responds.** Higher bids buy more clicks, but each extra click costs
   more, so the budget is best spread over the whole day.
2. **Which pacing works best.** A simple feedback rule wins. Q-learning earns 28% fewer
   clicks, because it spends most of its budget in the morning.
3. **What the agent sees matters.** Without its remaining budget in the state,
   Q-learning does worse than random bidding. Adding last hour's price, which carries no
   information in this market, only slows learning.

The full result tables are the CSV files in [`results/`](results), and the figures are in
[`results/figs/`](results/figs).

## Reproducing the results

Requires Python 3.12 and about 6 GB of disk for the raw data. Every run is seeded, so
rerunning reproduces the committed CSV files and figures exactly.

1. **Install the dependencies.**

   ```bash
   python3.12 -m venv .venv          # or: uv venv .venv --python 3.12
   .venv/bin/pip install -r requirements.txt
   ```

2. **Download the data.** Get the iPinYou contest archive from
   [Kaggle](https://www.kaggle.com/datasets/lastsummer/ipinyou) and place it at the
   project root as `ipinyou.contest.dataset/`. See [`data/README.md`](data/README.md).

3. **Build the processed data** (about 80 s the first time).

   ```bash
   .venv/bin/python sim/data_prep.py
   ```

   This extracts campaign 1458, fits the click model, and writes `data/processed/` and
   the cost per click to `config.json`.

4. **Run the tests.**

   ```bash
   .venv/bin/python -m pytest
   ```

5. **Run all experiments** (about 1 minute on a laptop).

   ```bash
   .venv/bin/python experiments/stage1.py
   ```

   This regenerates every CSV file in `results/` and every figure in `results/figs/`.
   You can also run the observations one at a time with
   `experiments/exp1_market_response.py`, `exp2_pacing.py` and
   `exp3_observation_space.py`. The third reuses the Q-tables saved by the second.

`config.json` stores the cost per click, the daily budget (169.3 fen) and the price-level
cutoffs. If the budget or cutoffs are missing, `stage1.py` recomputes them, with the same
result.

## Method in brief

- **Simulator** ([`sim/env.py`](sim/env.py)). A day has 24 hourly steps. Each hour draws
  500 logged auctions from that hour of day, at random with replacement. The agent bids
  α × cost per click × P(click), wins when its bid beats the logged price, and pays its
  own bid (a first-price auction). The reward is the expected number of clicks won.
- **Agents** ([`agents/`](agents)). Even pacing uses one α tuned on training days.
  Adaptive pacing raises or lowers α after each hour to stay on an even spending
  schedule. Tabular Q-learning chooses among 9 values of α, with the hour and the budget
  left (in 10 bins) as its state; it is trained with ε-greedy exploration on 20,000
  simulated days, in 5 runs with different seeds. Random α is a sanity check.
- **Evaluation** ([`experiments/runner.py`](experiments/runner.py)). Training and tuning
  use 6–12 June 2013. Evaluation uses 20 simulated days built from 13–15 June. Every
  agent faces exactly the same auctions on each test day.

## Repository layout

| path | contents |
| --- | --- |
| `sim/data_prep.py` | extracts campaign 1458 from the raw logs, fits the click model |
| `sim/env.py` | auction simulator; budget and price-level calibration |
| `agents/` | even pacing, adaptive pacing, Q-learning, random α |
| `experiments/` | the three observations, the shared evaluation code, and `stage1.py` |
| `results/` | result tables (CSV), saved Q-tables, figures in `results/figs/` |
| `tests/` | tests for the simulator and the agents |
| `data/README.md` | how to obtain and process the data |
| `config.json` | cost per click, daily budget, price-level cutoffs |
| `report_final.pdf` | final report |
| `Project_Proposal.pdf` | original project proposal |

## Data and references

- W. Zhang, S. Yuan, J. Wang, X. Shen. *Real-Time Bidding Benchmarking with iPinYou
  Dataset.* arXiv:1407.7073, 2014.
- iPinYou dataset copy on Kaggle: <https://www.kaggle.com/datasets/lastsummer/ipinyou>.
- W. Zhang. *make-ipinyou-data*: <https://github.com/wnzhang/make-ipinyou-data>.
