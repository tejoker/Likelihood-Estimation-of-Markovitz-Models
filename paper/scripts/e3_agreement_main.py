"""Condensed main-text version of the E3 agreement table (Section 7, label tab:e3-agreement).

Reads ``results/e3/e3_agree_tau.csv`` and ``results/e3/e3_agree_kappa.csv`` (written by the
``agree`` part of ``experiments/e3_sp500.py``) and writes two bare tabulars with the numbers of
the generated ``e3_agreement.tex``:

* ``paper/tables/manual_e3_agreement_main_tau.tex``: Kendall tau between the ranking of GBM,
  Merton and GARCH-t by the critic statistic (small = good) and by the mean predictive
  log-score, pooled (W-critic, seed 0 and median over seeds) and per ticker (L = 5 statistics;
  mean tau and shares of tickers with tau = 1 and tau = -1; descriptive);
* ``paper/tables/manual_e3_agreement_main_kappa.tex``: Cohen kappa between per-ticker
  rejections of the test (calibrated p-values) and of the Berkowitz test, with BH at 5% for both
  and with unadjusted 5% decisions, per model and stratified by model (chance agreement within
  model), plus the BH cross-counts of the W-critic.

Usage (from the repository root)::

    uv run python paper/scripts/e3_agreement_main.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAU = ROOT / "results" / "e3" / "e3_agree_tau.csv"
KAPPA = ROOT / "results" / "e3" / "e3_agree_kappa.csv"
OUT_TAU = ROOT / "paper" / "tables" / "manual_e3_agreement_main_tau.tex"
OUT_KAPPA = ROOT / "paper" / "tables" / "manual_e3_agreement_main_kappa.tex"
MLAB = {"GBM": "GBM", "Merton": "Merton", "GARCHt": "GARCH-$t$"}
KROWS = (("GBM", "GBM"), ("Merton", "Merton"), ("GARCHt", "GARCH-$t$"),
         ("stratified", "All three (stratified)"))


def num(v: float, d: int = 2) -> str:
    x = float(v)
    if not math.isfinite(x):
        return "--"
    s = f"{x:.{d}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return "$-$" + s[1:] if s.startswith("-") else s


def ranking(s: str) -> str:
    return " $>$ ".join(MLAB.get(x, x) for x in str(s).split(">"))


def write_tau() -> None:
    t = pd.read_csv(TAU)
    lines = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Scope & Statistic & Ranking (critic; log-score) & $\tau$ & share $\tau=1$"
        r" & share $\tau=-1$ \\",
        r"\midrule",
    ]
    for _, r in t.iterrows():
        if r["scope"] == "pooled":
            rk = ranking(r["rank_critic"])
            both = rk if r["rank_critic"] == r["rank_loglik"] else (
                rk + "; " + ranking(r["rank_loglik"]))
            if r["rank_critic"] == r["rank_loglik"]:
                both += " (both)"
            stat = r"\WC{}, " + ("seed 0" if r["critic_statistic"] == "seed 0"
                                 else "median over seeds")
            lines.append(f"Pooled, $L=20$ & {stat} & {both} & {num(r['tau'])} & -- & -- \\\\")
        else:
            test = r"\WC{}" if r["test"] == "critic" else "MMD"
            lines.append(f"Per ticker, $L=5$ ({int(r['n'])}) & {test} & -- & {num(r['tau'])}"
                         f" & {num(r['share_tau_1'])} & {num(r['share_tau_m1'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TAU.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TAU}")


def write_kappa() -> None:
    k = pd.read_csv(KAPPA)

    def get(test: str, model: str, rule: str) -> pd.Series:
        return k[(k["test"] == test) & (k["model"] == model) & (k["rule"] == rule)].iloc[0]

    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{6}{c}{\WC{} (HAC), $L=5$} & \multicolumn{2}{c}{MMD (wild), $L=5$} \\",
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-9}",
        r" & \multicolumn{2}{c}{$\kappa$} & \multicolumn{4}{c}{BH decisions: tickers}"
        r" & \multicolumn{2}{c}{$\kappa$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-9}",
        r"Model & BH & 5\% & both & test only & Berk.\ only & neither & BH & 5\% \\",
        r"\midrule",
    ]
    for m, lab in KROWS:
        if m == "stratified":
            lines.append(r"\addlinespace")
        cb, cr = get("critic", m, "bh"), get("critic", m, "raw")
        mb, mr = get("mmd", m, "bh"), get("mmd", m, "raw")
        cells = [num(cb["kappa"]), num(cr["kappa"]), str(int(cb["both_reject"])),
                 str(int(cb["only_test"])), str(int(cb["only_berkowitz"])),
                 str(int(cb["neither"])), num(mb["kappa"]), num(mr["kappa"])]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_KAPPA.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_KAPPA}")


def main() -> None:
    OUT_TAU.parent.mkdir(parents=True, exist_ok=True)
    write_tau()
    write_kappa()


if __name__ == "__main__":
    main()
