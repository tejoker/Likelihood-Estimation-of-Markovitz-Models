"""Condensed main-text version of the E3 fit-diagnostics table (Section 6, label
tab:e3-fit-diagnostics).

Reads ``results/e3/e3_fit_summary.csv`` (written by the ``fit`` part of
``experiments/e3_sp500.py``, the same numbers as the generated ``e3_fit_diagnostics.tex``, which
goes to Appendix D in full) and writes ``paper/tables/manual_data_fit_main.tex``: the key
parameters of each univariate generator (median and IQR across the 100 panel tickers, number of
tickers at an optimiser bound), without the drift, fit-time and optimiser-success rows. Values
are copied as formatted by the experiment script, never recomputed.

Usage (from the repository root)::

    uv run python paper/scripts/data_fit_main.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "e3" / "e3_fit_summary.csv"
OUT = ROOT / "paper" / "tables" / "manual_data_fit_main.tex"

#: (model label in the CSV, label in the table, parameter label in the CSV, label in the table)
ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("GBM", "GBM", r"$\sigma\times 100$", r"$\sigma\times100$"),
    ("Merton", "Merton", r"$\sigma\times 100$", r"$\sigma\times100$"),
    ("Merton", "", r"$\lambda$", r"$\lambda_J$"),
    ("Merton", "", r"$\sigma_J\times 100$", r"$\sigma_J\times100$"),
    ("GARCH-t", r"GARCH-$t$", r"$\alpha+\beta$", r"$a+b$"),
    ("GARCH-t", "", r"$\nu$", r"$\nu$"),
    ("Heston", "Heston", r"$\kappa$", r"$\kappa_v$"),
    ("Heston", "", r"$\sqrt{\theta}\times 100$", r"$\sqrt{\theta_v}\times100$"),
    ("Heston", "", r"$\rho$", r"$\rho_v$"),
    ("Heston", "", r"$c=\xi/\sqrt{2\kappa\theta}$ (CV of $v$)",
     r"$c=\xi_v/\sqrt{2\kappa_v\theta_v}$"),
    ("Heston", "", "Feller condition holds (share)", r"share with $c\le1$ (Feller)"),
    ("Heston", "", "Tickers with a parameter at a bound", "tickers with a parameter at a bound"),
    ("Stat. bootstrap", "Stat.\\ bootstrap", "Mean block length", "mean block length (days)"),
)


def cell(v: object) -> str:
    return "" if pd.isna(v) else str(v)


def main() -> None:
    t = pd.read_csv(SRC, dtype=str)
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Parameter & Median & IQR & At bound \\",
        r"\midrule",
    ]
    for i, (model, mlab, par, plab) in enumerate(ROWS):
        hit = t[(t["Model"] == model) & (t["Parameter"] == par)]
        if hit.shape[0] != 1:
            raise RuntimeError(f"row {model!r} / {par!r} not found exactly once in {SRC}")
        r = hit.iloc[0]
        if mlab and i > 0:
            lines.append(r"\addlinespace")
        iqr = cell(r["IQR"])
        at = cell(r["At bound"])
        if par.startswith("Tickers with"):
            at = ""  # the count is in the Median column (of 100 tickers)
        lines.append(f"{mlab} & {plab} & {cell(r['Median'])} & {iqr if iqr != '--' else ''} & {at} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
