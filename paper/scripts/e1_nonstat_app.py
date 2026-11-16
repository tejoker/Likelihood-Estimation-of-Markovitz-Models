"""Narrower appendix version of the E1(c) table (Appendix E, label tab:e1-nonstat).

Reads ``results/e1/e1_nonstat.csv`` (written by the ``nonstat`` part of
``experiments/e1_placebo.py``) and writes ``paper/tables/manual_e1_nonstat_app.tex``: absolute
tests of the real TEST windows against the real TRAIN windows over 10 test seeds (median
statistic, median p, twice the median p, share of seeds with p <= 0.05) next to the rejection
rate at 5% of the same comparison in the stationary simulated placebo (E1(a), cell ``joint``,
R_new* against R_past*, 200 replications, Wilson 95% interval). Same numbers and rounding as the
generated ``e1_nonstat.tex``; the units and median z_HAC columns are dropped (units go to the
caption) so that the table fits the text width.

Usage (from the repository root)::

    uv run python paper/scripts/e1_nonstat_app.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "e1" / "e1_nonstat.csv"
OUT = ROOT / "paper" / "tables" / "manual_e1_nonstat_app.tex"
TESTS = {"critic": r"\WC{}", "lrcritic": r"\LRC{}", "c2st": "C2ST", "mmd": "MMD",
         "sigmmd": r"\SigMMD{}", "ks": "KS"}
PTYPE = {"hac": r"HAC$^{\star}$", "hac_fixedb": r"HAC, fixed-$b$ (cons.)",
         "hac_fixedb_eq": r"HAC, fixed-$b$ (eq.\ var.)", "timeblock_permutation": "permutation",
         "wild": r"wild bootstrap$^{\star}$", "permutation": "permutation",
         "asymptotic": r"asymptotic i.i.d.\ (invalid)"}


def num(v: float, d: int = 3) -> str:
    x = float(v)
    if not math.isfinite(x):
        return "--"
    s = f"{x:.{d}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return "$-$" + s[1:] if s.startswith("-") else s


def main() -> None:
    d = pd.read_csv(SRC)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{Real \TEST{} vs real \TRAIN{} (10 test seeds)}"
        r" & Stationary placebo \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-7}",
        r"Test & $p$-value & Median stat. & Median $p$ & $2\times$median $p$"
        r" & Share $p\le0.05$ & size at 5\% [95\% CI] \\",
        r"\midrule",
    ]
    prev = None
    for _, r in d.iterrows():
        t = r["test"]
        if prev is not None and t != prev:
            lines.append(r"\addlinespace")
        cells = [
            TESTS.get(t, t) if t != prev else "",
            PTYPE.get(r["ptype"], str(r["ptype"])),
            num(r["statistic_median"]),
            num(r["p_median"]),
            num(min(1.0, float(r["p_2median"]))),
            num(r["share_reject_0.05"]),
            f"{num(r['placebo_rej_0.05'])} [{num(r['placebo_lo_0.05'])}, "
            f"{num(r['placebo_hi_0.05'])}]",
        ]
        lines.append(" & ".join(cells) + r" \\")
        prev = t
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
