"""Condensed main-text version of the E3 power-check table (Section 7, label tab:e3-power).

Reads ``results/e3/e3_power_summary.csv`` (design D1) and ``results/e3/e3_realvol_summary.csv``
(design D2), both written by ``experiments/e3_sp500.py``, and writes
``paper/tables/manual_e3_power_main.tex``. Same quantities and rounding as the generated
``e3_power.tex`` (power of the seed-combined rule with its Wilson 95% interval and one-sided
Clopper-Pearson lower bound, single-seed power, boundary sizes with Clopper-Pearson 95%
intervals, median Driscoll-Kraay standard error at the boundary), with short design labels so
that the table fits the text width.

Usage (from the repository root)::

    uv run python paper/scripts/e3_power_main.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = {"D1": ROOT / "results" / "e3" / "e3_power_summary.csv",
       "D2": ROOT / "results" / "e3" / "e3_realvol_summary.csv"}
OUT = ROOT / "paper" / "tables" / "manual_e3_power_main.tex"
LABELS = {"D1": r"D1: stationary GARCH-$t$ panel",
          "D2": r"D2: real \TRAIN{} volatility path"}


def num(v: float, d: int) -> str:
    return "--" if v is None or not math.isfinite(float(v)) else f"{float(v):.{d}f}"


def rate(r: pd.Series, col: str, lo: str, hi: str, d: int = 2) -> str:
    return f"{num(r[col], d)} [{num(r[lo], d)}, {num(r[hi], d)}]"


def main() -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Alternative: volatility $\times1.2$}"
        r" & \multicolumn{3}{c}{Boundary: $\Law(G)=\Law(\Rpast^*)$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Design & Power, rule & $\betaL$ & Power, 1 seed & Size, rule & Size, 1 seed"
        r" & $\widehat{\mathrm{se}}_{\mathrm{DK}}$ \\",
        r"\midrule",
    ]
    for d, path in SRC.items():
        s = pd.read_csv(path).set_index("case")
        alt, bd = s.loc["alternative"], s.loc["boundary"]
        cells = [
            rate(alt, "rate_rule", "rule_ci_lo", "rule_ci_hi"),
            num(alt["rule_cp_lower"], 2),
            rate(alt, "rate_single", "single_ci_lo", "single_ci_hi"),
            rate(bd, "rate_rule", "rule_cp_lo", "rule_cp_hi"),
            rate(bd, "rate_single", "single_cp_lo", "single_cp_hi"),
            num(bd["median_se_hac"], 3),
        ]
        lines.append(LABELS[d] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
