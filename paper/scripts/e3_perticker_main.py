"""Condensed main-text version of the E3 per-ticker table (Section 7, label tab:e3-perticker).

Reads ``results/e3/e3_perticker_summary.csv`` (written by ``experiments/e3_sp500.py``, parts
``perticker`` and ``perticker_null``) and writes ``paper/tables/manual_e3_perticker_main.tex``:
per generator and test, the share of the 100 tickers rejected at 5% on the calibrated p-values
(Wilson 95% interval), the Benjamini-Hochberg and Benjamini-Yekutieli rejection counts and
Storey's pi_0 on the calibrated p-values, and, for reference, the count of nominal rejections
at 5% and the median nominal p-value. Same numbers and rounding as the generated
``e3_perticker.tex``; the constant columns (p-value type, 100 tickers) move to the caption.

Usage (from the repository root)::

    uv run python paper/scripts/e3_perticker_main.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "e3" / "e3_perticker_summary.csv"
OUT = ROOT / "paper" / "tables" / "manual_e3_perticker_main.tex"
MODELS = ("GBM", "Merton", "GARCHt", "Heston", "Bootstrap")
LABELS = {"GBM": "GBM", "Merton": "Merton", "GARCHt": "GARCH-$t$", "Heston": "Heston",
          "Bootstrap": r"Stat.\ bootstrap"}
TESTS = {"critic": r"\WC{} (HAC)", "mmd": "MMD (wild)"}


def num(v: float, d: int) -> str:
    x = float(v)
    return "--" if not math.isfinite(x) else f"{x:.{d}f}"


def tp(p: float) -> str:
    x = float(p)
    if not math.isfinite(x):
        return "--"
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def main() -> None:
    s = pd.read_csv(SRC)
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{Calibrated $p$-values} & \multicolumn{2}{c}{Nominal $p$-values}"
        r" \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
        r"Generator & Test & Rej.\ 5\% [95\% CI] & BH & BY & Storey $\hat\pi_0$"
        r" & Rej.\ 5\% & Median $p$ \\",
        r"\midrule",
    ]
    first = True
    for m in MODELS:
        sm = s[s["model"] == m]
        if sm.empty:
            continue
        if not first:
            lines.append(r"\addlinespace")
        first = False
        for j, (t, tl) in enumerate(TESTS.items()):
            r = sm[sm["test"] == t]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = [
                LABELS[m] if j == 0 else "",
                tl,
                f"{num(r['share_cal'], 2)} [{num(r['share_cal_lo'], 2)}, {num(r['share_cal_hi'], 2)}]",
                str(int(r["n_reject_bh"])),
                str(int(r["n_reject_by"])),
                num(r["storey_pi0"], 2),
                str(int(r["n_reject_raw"])),
                tp(r["p_median"]),
            ]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
