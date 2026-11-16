"""Condensed main-text version of the E3 likelihood table (Section 7, label tab:e3-likelihood).

Reads ``results/e3/e3_likelihood_raw.csv`` (written by the ``likelihood`` part of
``experiments/e3_sp500.py``) and writes ``paper/tables/manual_e3_likelihood_main.tex`` with the
quantities of the generated ``e3_likelihood.tex``, computed exactly as there: mean
out-of-sample predictive log-score over tickers (raw returns and returns divided by the TRAIN
standard deviation of the ticker), the number of tickers on which the model has the highest
log-score, and the share of tickers rejected at 5% by the Berkowitz and PIT-KS tests (Wilson 95%
interval) with the Benjamini-Hochberg count at 5%. Only the column headers are shortened.

Usage (from the repository root)::

    uv run python paper/scripts/e3_likelihood_main.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "e3" / "e3_likelihood_raw.csv"
OUT = ROOT / "paper" / "tables" / "manual_e3_likelihood_main.tex"
MODELS = ("GBM", "Merton", "GARCHt")
LABELS = {"GBM": "GBM", "Merton": "Merton", "GARCHt": "GARCH-$t$"}
ALPHA = 0.05


def wilson_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval, as ``synthval.pipeline.wilson_ci``."""
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def rate_ci(k: int, n: int, d: int = 2) -> str:
    lo, hi = wilson_ci(k, n)
    return f"{k / n:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def bh_count(p: np.ndarray, q: float = ALPHA) -> int:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    return int(np.sum(stats.false_discovery_control(p, method="bh") <= q)) if p.size else 0


def tnum(x: float, d: int = 3) -> str:
    s = f"{x:.{d}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def main() -> None:
    lk = pd.read_csv(SRC)
    best = lk.loc[lk.groupby("ticker")["pred_loglik"].idxmax(), "model"].value_counts()
    nt = lk["ticker"].nunique()
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Predictive log-score} & \multicolumn{2}{c}{Berkowitz}"
        r" & \multicolumn{2}{c}{PIT-KS} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Model & mean, $r$ & mean, $r/s_i$ & best on & rej.\ 5\% [95\% CI] & BH"
        r" & rej.\ 5\% [95\% CI] & BH \\",
        r"\midrule",
    ]
    for m in MODELS:
        g = lk[lk["model"] == m]
        pb, pk = g["berk_p"].to_numpy(float), g["ks_p"].to_numpy(float)
        cells = [
            tnum(g["pred_loglik"].mean()),
            tnum(g["pred_loglik_z"].mean()),
            f"{int(best.get(m, 0))}/{nt}",
            rate_ci(int(np.sum(pb <= ALPHA)), len(pb)),
            str(bh_count(pb)),
            rate_ci(int(np.sum(pk <= ALPHA)), len(pk)),
            str(bh_count(pk)),
        ]
        lines.append(LABELS[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
