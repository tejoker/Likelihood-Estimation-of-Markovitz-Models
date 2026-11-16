"""Condensed main-text version of the E3 verdict table (Section 7, label tab:e3-verdicts).

Reads ``results/e3/e3_verdicts.csv`` (written by the ``verdict`` part of
``experiments/e3_sp500.py``) and writes ``paper/tables/manual_e3_verdicts_main.tex``: per
generator, twice the median over the synthetic seeds of the absolute W-critic HAC p-value and
the number of seeds rejecting at 5%, the anchored statistic Delta (median and range over the
seeds) and its seed-combined HAC p-value, and the verdict with the point power and with the
Clopper-Pearson variant. Same rounding as the generated ``e3_verdicts.tex``. The power and the
boundary size of the rule are the same for every generator; they are left to the caption and
to the power table (``manual_e3_power_main.tex``) so that the table fits the text width.

Usage (from the repository root)::

    uv run python paper/scripts/e3_verdicts_main.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "e3" / "e3_verdicts.csv"
OUT = ROOT / "paper" / "tables" / "manual_e3_verdicts_main.tex"
MODELS = ("GBM", "Merton", "GARCHt", "Heston", "Bootstrap")
LABELS = {"GBM": "GBM", "Merton": "Merton", "GARCHt": "GARCH-$t$", "Heston": "Heston",
          "Bootstrap": r"Stat.\ bootstrap"}
VERDICT = {"GOOD": r"\GOOD{}", "BAD": r"\BAD{}", "INCONCLUSIVE": r"\INCONCL{}",
           "INVALID": r"\INVALID{}"}
DDAGGER = r"$^{\ddagger}$"


def tnum(v: float, d: int = 3) -> str:
    x = float(v)
    if not math.isfinite(x):
        return "--"
    s = f"{x:.{d}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return "$-$" + s[1:] if s.startswith("-") else s


def tp(p: float) -> str:
    x = float(p)
    if not math.isfinite(x):
        return "--"
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def main() -> None:
    v = pd.read_csv(SRC).set_index("model")
    lines = [
        r"\begin{tabular}{lrrrrll}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Absolute \WC{}} & \multicolumn{2}{c}{Anchored relative}"
        r" & \multicolumn{2}{c}{Verdict} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Generator & $\pmed$ & seeds rej. & $\Danc$: median [range] & $\pmed$"
        r" & $\hat\beta\ge0.8$ & $\betaL\ge0.8$ \\",
        r"\midrule",
    ]
    for m in MODELS:
        if m not in v.index:
            continue
        r = v.loc[m]
        mark = "" if bool(r["rel_valid"]) else DDAGGER
        cells = [
            tp(r["abs_p_combined"]),
            f"{int(r['abs_n_reject'])}/{int(r['n_seeds'])}",
            f"{tnum(r['delta_median'])} [{tnum(r['delta_min'])}, {tnum(r['delta_max'])}]" + mark,
            tp(r["rel_p_combined"]) + mark,
            VERDICT.get(str(r["verdict"]), str(r["verdict"])),
            VERDICT.get(str(r["verdict_cp"]), str(r["verdict_cp"])),
        ]
        lines.append(LABELS[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
