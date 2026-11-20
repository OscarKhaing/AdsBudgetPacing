"""Project paths and helpers for reading/writing config.json."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGS_DIR = ROOT / "results" / "figs"


def load_config() -> dict:
    """Return the contents of config.json (empty dict if it does not exist yet)."""
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text())


def update_config(**values) -> dict:
    """Merge `values` into config.json and return the new config."""
    cfg = load_config()
    cfg.update(values)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg
