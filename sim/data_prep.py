"""Build the campaign-1458 data used by the simulator.

extract: rebuild make-ipinyou-data's train/test logs from the raw contest files.
process: fit the pCTR model, compute eCPC, save data/processed/{campaign}_{split}.parquet.
Run from the project root:  python sim/data_prep.py [--inspect]
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import PROCESSED_DIR, ROOT, update_config  # noqa: E402

CAMPAIGN = "1458"
RAW_DIR = ROOT / "ipinyou.contest.dataset"
CACHE_DIR = ROOT / "data" / "raw"

RAW_COLUMNS = {
    0: "bidid", 1: "timestamp", 6: "region", 7: "city", 8: "adexchange",
    13: "slotwidth", 14: "slotheight", 15: "slotvisibility", 16: "slotformat",
    17: "slotprice", 18: "creative", 19: "bidprice", 20: "payprice", 22: "advertiser",
}
ADVERTISER_COLUMN = 22
NCLICK_COLUMN = 24

FEATURES = ["hour", "slotwidth", "slotheight", "slotvisibility", "region", "adexchange"]
KEEP = ["click"] + FEATURES + ["price"]


def _read_raw(path, columns, chunksize=500_000):
    """Read selected columns of one raw .bz2 log, keeping only CAMPAIGN's rows."""
    reader = pd.read_csv(path, sep="\t", header=None, usecols=sorted(columns), dtype=str,
                         quoting=csv.QUOTE_NONE, chunksize=chunksize)
    campaign_chunks = [chunk[chunk[ADVERTISER_COLUMN] == CAMPAIGN] for chunk in reader]
    return pd.concat(campaign_chunks).rename(columns=columns)


def _read_impressions(path):
    return _read_raw(path, RAW_COLUMNS)


def _read_clicks(path):
    return _read_raw(path, {0: "bidid", 18: "creative", ADVERTISER_COLUMN: "advertiser"})


def _read_test(path):
    return _read_raw(path, {**RAW_COLUMNS, NCLICK_COLUMN: "nclick"})


def extract(raw_dir=RAW_DIR, workers=6):
    """Rebuild make-ipinyou-data's train.log / test.log for CAMPAIGN as parquet caches.

    train: season-2 impressions; click = 1 if (bidid, creative) is in a season-2 click log.
    test: season-2 leaderboard rows; click = 1 if nclick > 0.
    hour = characters 8-9 of the yyyyMMddHHmmssSSS timestamp.
    """
    raw_dir = Path(raw_dir)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    impression_files = sorted((raw_dir / "training2nd").glob("imp.*.txt.bz2"))
    click_files = sorted((raw_dir / "training2nd").glob("clk.*.txt.bz2"))
    test_file = next((raw_dir / "testing2nd").glob("leaderboard.test.data.*.txt.bz2"))
    print(f"extracting campaign {CAMPAIGN} from {len(impression_files)} impression files")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        test_job = executor.submit(_read_test, test_file)
        clicks = pd.concat(executor.map(_read_clicks, click_files))
        train = pd.concat(executor.map(_read_impressions, impression_files), ignore_index=True)
        test = test_job.result().reset_index(drop=True)

    clicked_keys = set(clicks.bidid + "-" + clicks.creative)
    train["click"] = (train.bidid + "-" + train.creative).isin(clicked_keys).astype(np.int8)
    test["click"] = (test.pop("nclick").astype(int) > 0).astype(np.int8)
    for logs in (train, test):
        logs["hour"] = logs.timestamp.str[8:10].astype(np.int8)
    train.to_parquet(CACHE_DIR / f"{CAMPAIGN}_train.log.parquet", index=False)
    test.to_parquet(CACHE_DIR / f"{CAMPAIGN}_test.log.parquet", index=False)
    print(f"train: {len(train):,} imps, {train.click.sum():,} clicks; "
          f"test: {len(test):,} imps, {test.click.sum():,} clicks")


def load_logs():
    """Load the cached train/test logs written by `extract`."""
    return (pd.read_parquet(CACHE_DIR / f"{CAMPAIGN}_train.log.parquet"),
            pd.read_parquet(CACHE_DIR / f"{CAMPAIGN}_test.log.parquet"))


def inspect(train, test, n_rows=5):
    """Print the columns, a few rows, and the value counts we rely on."""
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("columns:", list(train.columns))
        print(train.head(n_rows))
        for column in ["hour", "slotwidth", "slotheight", "slotvisibility", "adexchange"]:
            print(f"\n{column}:", train[column].value_counts(dropna=False).head(15).to_dict())
        print("\nregions:", train.region.nunique(), "| nulls:", train.isna().sum().to_dict())
        print("payprice:", pd.to_numeric(train.payprice).describe().round(1).to_dict())
        print("duplicate bidids (train, test):",
              train.bidid.duplicated().sum(), test.bidid.duplicated().sum())


def clean(logs):
    """Keep the model columns; price is per impression (payprice is per 1000 imps)."""
    logs = logs.copy()
    logs["price"] = pd.to_numeric(logs.payprice, errors="coerce") / 1000.0
    logs = logs[KEEP].replace({"null": np.nan}).dropna()
    return logs.reset_index(drop=True)


def fit_pctr(train, test):
    """Fit P(click | features) by one-hot logistic regression on train; add `pctr`.

    Uses Newton's method with a tight tolerance: with clicks this rare, the default solver
    stops after a few iterations, long before the fit has converged.
    """
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_train = encoder.fit_transform(train[FEATURES].astype(str))
    X_test = encoder.transform(test[FEATURES].astype(str))
    model = LogisticRegression(solver="newton-cholesky", tol=1e-10, max_iter=1000)
    model.fit(X_train, train.click)
    train["pctr"] = model.predict_proba(X_train)[:, 1]
    test["pctr"] = model.predict_proba(X_test)[:, 1]
    return {
        "n_features": int(X_train.shape[1]),
        "n_iter": int(model.n_iter_[0]),
        "train_auc": roc_auc_score(train.click, train.pctr),
        "test_auc": roc_auc_score(test.click, test.pctr),
        "test_ctr": test.click.mean(),
        "test_mean_pctr": test.pctr.mean(),
    }


def process():
    """Clean the cached logs, fit pCTR, save parquet files and eCPC to config.json."""
    train_raw, test_raw = load_logs()
    train, test = clean(train_raw), clean(test_raw)
    print(f"dropped rows with missing values: train {len(train_raw) - len(train)}, "
          f"test {len(test_raw) - len(test)}")
    pctr_stats = fit_pctr(train, test)
    for name, value in pctr_stats.items():
        print(f"  {name}: {value:.5g}")

    ecpc = train.price.sum() / train.click.sum()
    print(f"eCPC = {ecpc:.4f} (fen per click); train CTR = {train.click.mean():.5f}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_columns = ["hour", "pctr", "price", "click"]
    for split_name, logs in [("train", train), ("test", test)]:
        logs[output_columns].to_parquet(PROCESSED_DIR / f"{CAMPAIGN}_{split_name}.parquet",
                                        index=False)
    update_config(campaign=CAMPAIGN, ecpc=float(ecpc),
                  pctr_train_auc=float(pctr_stats["train_auc"]),
                  pctr_test_auc=float(pctr_stats["test_auc"]))


def main():
    parser = argparse.ArgumentParser(description="Build campaign data for the simulator.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--inspect", action="store_true", help="only print the logs")
    args = parser.parse_args()
    if not (CACHE_DIR / f"{CAMPAIGN}_test.log.parquet").exists():
        extract(args.raw_dir)
    if args.inspect:
        inspect(*load_logs())
        return
    process()


if __name__ == "__main__":
    main()
