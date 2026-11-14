"""Project-wide constants: paths, sample split, seeds and defaults."""
from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_CSV: Path = ROOT / "data" / "sp500_price_volume.csv"
RESULTS_DIR: Path = ROOT / "results"
FIG_DIR: Path = ROOT / "paper" / "figures"
TAB_DIR: Path = ROOT / "paper" / "tables"

# Sample split: calibration ("real past") and evaluation ("real new").
FULL = ("2018-01-02", "2025-06-17")
TRAIN = ("2018-01-02", "2022-12-30")
TEST = ("2023-01-03", "2025-06-17")

# Main panel of tickers for univariate experiments.
N_PANEL = 100
PANEL_SEED = 20251114

# Testing defaults.
WINDOW_L = 20
ALPHA = 0.05
N_PERM = 1000
TRADING_DAYS = 252


def n_workers() -> int:
    """Number of worker processes for experiments (env ``SYNTHVAL_WORKERS``, default 4)."""
    try:
        return max(1, int(os.environ.get("SYNTHVAL_WORKERS", "4")))
    except ValueError:
        return 4
