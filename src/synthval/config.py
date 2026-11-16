"""Project-wide constants: paths, sample split, seeds and defaults."""
from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_CSV: Path = ROOT / "data" / "sp500_price_volume.csv"


def _dir_from_env(var: str, default: Path) -> Path:
    """``$var`` (absolute, or relative to ``ROOT``) if set, else ``default``."""
    v = os.environ.get(var)
    if not v:
        return default
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


# Output directories. A second protocol version (e.g. the GPU v2 runs, experiments/v2.py) sets
# SYNTHVAL_RESULTS_DIR / SYNTHVAL_FIG_DIR / SYNTHVAL_TAB_DIR so it never overwrites v1 outputs.
RESULTS_DIR: Path = _dir_from_env("SYNTHVAL_RESULTS_DIR", ROOT / "results")
FIG_DIR: Path = _dir_from_env("SYNTHVAL_FIG_DIR", ROOT / "paper" / "figures")
TAB_DIR: Path = _dir_from_env("SYNTHVAL_TAB_DIR", ROOT / "paper" / "tables")

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
