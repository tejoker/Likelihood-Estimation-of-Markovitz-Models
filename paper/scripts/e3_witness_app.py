"""Appendix table of witness diagnostics (Appendix E, label tab:e3-witness-app).

Reads ``results/e3/e3_witness.csv`` (written by the ``witness`` part of
``experiments/e3_sp500.py``: critic values of the seed-0 pooled W-critic of each generator on its
1500 real and 1500 synthetic evaluation windows, with window features) and writes
``paper/tables/manual_e3_witness_app.tex``. Per generator:

* ``T``: mean critic value on real minus synthetic evaluation windows (the seed-0 statistic);
* Spearman correlation of the critic value with the window log realised volatility
  (``log_rv`` = log root mean square of the normalised returns of the window) and with the
  largest absolute normalised return of the window (``max_abs``), over all 3000 windows;
* share of ``T`` reproduced by a step function of ``log_rv``: the critic is replaced by its mean
  within 20 equal-count bins of ``log_rv`` (bins from the pooled windows of that generator), and
  the real-minus-synthetic difference of these bin means is divided by ``T``;
* median realised volatility (``exp(log_rv)``, in units of the TRAIN standard deviation of the
  ticker) of the real and of the synthetic evaluation windows.

Usage (from the repository root)::

    uv run python paper/scripts/e3_witness_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "e3" / "e3_witness.csv"
OUT = ROOT / "paper" / "tables" / "manual_e3_witness_app.tex"
MODELS = ("GBM", "Merton", "GARCHt", "Heston", "Bootstrap")
LABELS = {"GBM": "GBM", "Merton": "Merton", "GARCHt": "GARCH-$t$", "Heston": "Heston",
          "Bootstrap": r"Stat.\ bootstrap"}
N_BINS = 20


def tnum(x: float, d: int) -> str:
    s = f"{x:.{d}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return "$-$" + s[1:] if s.startswith("-") else s


def row(g: pd.DataFrame) -> dict[str, float]:
    real = (g["sample"] == "real").to_numpy()
    f = g["f"].to_numpy(float)
    lrv = g["log_rv"].to_numpy(float)
    T = f[real].mean() - f[~real].mean()
    edges = np.quantile(lrv, np.linspace(0.0, 1.0, N_BINS + 1))
    b = np.clip(np.searchsorted(edges, lrv, side="right") - 1, 0, N_BINS - 1)
    means = pd.Series(f).groupby(b).mean()
    fb = pd.Series(b).map(means).to_numpy(float)
    return {
        "T": T,
        "rho_rv": stats.spearmanr(f, lrv).statistic,
        "rho_max": stats.spearmanr(f, g["max_abs"].to_numpy(float)).statistic,
        "share": (fb[real].mean() - fb[~real].mean()) / T,
        "rv_real": float(np.exp(np.median(lrv[real]))),
        "rv_syn": float(np.exp(np.median(lrv[~real]))),
    }


def main() -> None:
    w = pd.read_csv(SRC)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Spearman corr.\ of $\fhat$ with} & Share of $T$"
        r" & \multicolumn{2}{c}{Median realised vol.} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
        r"Generator & $T$ & log realised vol. & $\max_t\abs{r_t}$ & from realised vol."
        r" & real & synthetic \\",
        r"\midrule",
    ]
    for m in MODELS:
        g = w[w["model"] == m]
        if g.empty:
            continue
        r = row(g)
        cells = [tnum(r["T"], 3), tnum(r["rho_rv"], 2), tnum(r["rho_max"], 2),
                 tnum(r["share"], 2), tnum(r["rv_real"], 2), tnum(r["rv_syn"], 2)]
        lines.append(LABELS[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
