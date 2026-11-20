# Data

We use campaign **1458** (an online retailer) from season 2 of the iPinYou RTB dataset
(Zhang et al., 2014, [arXiv:1407.7073](https://arxiv.org/abs/1407.7073)).

## Getting the raw data

Download the original contest archive from Kaggle
(<https://www.kaggle.com/datasets/lastsummer/ipinyou>, about 6 GB) and place it at the
project root as `ipinyou.contest.dataset/` (gitignored). We use:

| split | raw files | days |
| --- | --- | --- |
| train | `training2nd/imp.*.txt.bz2`, `training2nd/clk.*.txt.bz2` | 6–12 June 2013 |
| test | `testing2nd/leaderboard.test.data.20130613_15.txt.bz2` | 13–15 June 2013 |

## Building the processed data

`python sim/data_prep.py` runs two steps.

1. **Extract** (first run only, about 70 s). Rebuilds the per-campaign train and test
   logs with the same rules as
   [make-ipinyou-data](https://github.com/wnzhang/make-ipinyou-data). A training
   impression counts as clicked if it appears in a click log (matched on bid ID and
   creative ID), a test row if its click count is above 0, and the hour comes from the
   timestamp. The result is cached in `data/raw/`. It matches the dataset paper exactly:
   3,083,056 training impressions with 2,454 clicks. The test split has 614,638
   impressions, 515 of them clicked.
2. **Process** (about 5 s).
   - Keep the click label, the hour, five slot and user features (slot width, slot
     height, slot visibility, region, ad exchange) and the price. No values are missing.
   - Convert prices. They are recorded in Chinese fen per thousand impressions, so
     `price = payprice / 1000` gives fen per impression. All money in this project is in
     fen.
   - Fit a logistic regression for P(click | features) on the training split, with
     one-hot features (79 columns). Clicks are rare (0.08%), so scikit-learn's default
     solver stops before converging; we use Newton's method
     (`solver="newton-cholesky"`, `tol=1e-10`), which converges in 10 iterations. Test
     AUC is 0.55, a weak predictor, but its average predicted click rate matches the real
     one.
   - Compute eCPC = total train price / total train clicks = 86.55 fen and save it to
     `config.json`. Bids are `alpha * eCPC * pCTR`.
   - Save `data/processed/1458_{train,test}.parquet` with columns
     `hour, pctr, price, click`.

`python sim/data_prep.py --inspect` prints the columns and value counts of the extracted
logs.

## Caveats

- The logs only contain auctions that iPinYou won with its fairly high fixed bid, so
  higher market prices are never seen.
- Traffic varies a lot by hour, but the simulator gives every hour the same number of
  auctions.
- Prices on the test days are about 7% higher than on the training days.
