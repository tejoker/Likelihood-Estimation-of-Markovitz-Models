"""Condensed main-text version of the E4 test table (Section 8, label tab:e4-tests).

Reads ``results/e4/e4_tests_summary.csv`` (written by ``experiments/e4_markowitz.py``) and writes
``paper/tables/manual_e4_tests_main.tex``: one row per generator (and the real-TRAIN reference),
rejection shares at the 5% level of the W-critic (Driscoll-Kraay HAC p-value) and of MMD
(wild-bootstrap p-value), nominal and size-adjusted, for the first synthetic seed and for
p_med = min(1, 2 x median p) over the seeds, plus the placebo size. Rates only; the generated
table ``e4_tests.tex`` (appendix) carries the Wilson 95% intervals. Numbers are rounded to two
decimals exactly as in the generated table.

Usage (from the repository root)::

    uv run python paper/scripts/e4_tests_main.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "e4" / "e4_tests_summary.csv"
OUT = ROOT / "paper" / "tables" / "manual_e4_tests_main.tex"

GENERATORS = ("MVGaussian", "MVStudentT", "CCCGARCHt", "MVBootstrap", "FactorGARCHt", "RealTRAIN")
LABELS = {
    "MVGaussian": "MV Gaussian", "MVStudentT": "MV Student-$t$", "CCCGARCHt": "CCC-GARCH-$t$",
    "MVBootstrap": "MV bootstrap", "FactorGARCHt": "Factor-GARCH-$t$",
    "RealTRAIN": r"Real \TRAIN{}, last 615 days",
}
PTYPE = {"critic": "hac", "mmd": "wild"}
NOMINAL, ADJ = "nominal", "size-adjusted (placebo)"
FIRST, MED = "first seed", "2 x median over seeds"


def fmt(v: float) -> str:
    return "--" if v is None or not math.isfinite(v) else f"{v:.2f}"


def rate(s: pd.DataFrame, g: str, t: str, calib: str, agg: str) -> float:
    d = s[(s["generator"] == g) & (s["test"] == t) & (s["pvalue_type"] == PTYPE[t])
          & (s["calibration"] == calib) & (s["aggregation"] == agg)]
    if not len(d) or not d["n_valid"].iloc[0]:
        return float("nan")
    return float(d["rej_0.05"].iloc[0])


def placebo(s: pd.DataFrame, g: str, t: str) -> float:
    d = s[(s["generator"] == g) & (s["test"] == t) & (s["pvalue_type"] == PTYPE[t])]
    if not len(d) or "placebo_rej_0.05" not in d.columns:
        return float("nan")
    return float(d["placebo_rej_0.05"].iloc[0])


def main() -> None:
    s = pd.read_csv(SRC)
    lines = [
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{5}{c}{\WC{} (HAC)} & \multicolumn{5}{c}{MMD (wild)} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        r" & \multicolumn{2}{c}{Nominal} & Placebo & \multicolumn{2}{c}{Size-adjusted}"
        r" & \multicolumn{2}{c}{Nominal} & Placebo & \multicolumn{2}{c}{Size-adjusted} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){10-11}",
        r"Generator & first & $\pmed$ & size & first & $\pmed$ & first & $\pmed$ & size & first"
        r" & $\pmed$ \\",
        r"\midrule",
    ]
    for g in GENERATORS:
        if g not in set(s["generator"]):
            continue
        if g == "RealTRAIN":
            lines.append(r"\midrule")
        cells = []
        for t in ("critic", "mmd"):
            cells += [fmt(rate(s, g, t, NOMINAL, FIRST)), fmt(rate(s, g, t, NOMINAL, MED)),
                      fmt(placebo(s, g, t)), fmt(rate(s, g, t, ADJ, FIRST)),
                      fmt(rate(s, g, t, ADJ, MED))]
        lines.append(LABELS[g] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
