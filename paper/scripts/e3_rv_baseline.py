"""Fixed realised-volatility statistic against the learned critic on the same windows (Section 7).

Writes ``paper/tables/manual_e3_rv_baseline.tex`` (label tab:e3-rv-baseline), a derived
computation on ``results/e3/e3_witness.csv``. That file holds, for each generator, the seed-0
absolute W-critic test of E3 re-run with the pooled test's random streams: the critic value
``f``, the window log realised volatility ``log_rv`` (log root mean square of the returns
divided by the TRAIN standard deviation of the ticker), the ticker and the date rank ``order`` of
each of the 1500 real TEST and 1500 synthetic evaluation windows.

For each generator the script computes, with the library function that produced the paper's
p-values (``synthval.twosample.critic.mean_diff_inference``: difference of means and
Driscoll-Kraay variance with date sums, Bartlett kernel and the Andrews AR(1) plug-in lag, one
sample at a time):

* the critic statistic ``T = mean f(real) - mean f(synthetic)`` and its Driscoll-Kraay ``z``
  (a check: both equal the seed-0 values of ``results/e3/e3_pooled_raw.csv``; the script stops
  if they do not);
* the same test with the critic replaced by the fixed, untrained function ``h(w) = -log_rv(w)``,
  oriented like the critic (positive when the synthetic windows are more volatile). Its
  statistic is ``Delta_RV = mean log_rv(synthetic) - mean log_rv(real)``; ``exp(Delta_RV)`` is
  the ratio of the geometric-mean realised volatilities of synthetic and real windows;
* the ratio of the two ``z``;
* descriptive statistics of the synthetic (and real) evaluation windows: standard deviation of
  ``log_rv`` and mean and median realised volatility ``exp(log_rv)``.

Only seed 0 and only the absolute test can be computed this way: the other seeds, the anchored
test and the power designs would need synthetic paths that the experiments did not store.

Usage (from the repository root)::

    uv run python paper/scripts/e3_rv_baseline.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from synthval.twosample.critic import mean_diff_inference

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "e3" / "e3_witness.csv"
POOLED = ROOT / "results" / "e3" / "e3_pooled_raw.csv"
OUT = ROOT / "paper" / "tables" / "manual_e3_rv_baseline.tex"
MODELS = ("GBM", "Merton", "GARCHt", "Heston", "Bootstrap")
LABELS = {"GBM": "GBM", "Merton": "Merton", "GARCHt": "GARCH-$t$", "Heston": "Heston",
          "Bootstrap": r"Stat.\ bootstrap"}


def tnum(x: float, d: int) -> str:
    if not math.isfinite(float(x)):
        return "--"
    s = f"{float(x):.{d}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return "$-$" + s[1:] if s.startswith("-") else s


def dk(gx: np.ndarray, gy: np.ndarray, ox: np.ndarray, oy: np.ndarray) -> dict[str, float]:
    # n_perm = 0: no permutation test, so the random generator is not used.
    return mean_diff_inference(gx, gy, rng=np.random.default_rng(0), n_perm=0,
                               order_x=ox, order_y=oy)


def main() -> None:
    w = pd.read_csv(SRC)
    pooled = pd.read_csv(POOLED)
    rows: list[dict[str, float | str]] = []
    real_desc: dict[str, float] | None = None
    for m in MODELS:
        g = w[w["model"] == m]
        if g.empty:
            continue
        real = g["sample"].eq("real").to_numpy()
        o = g["order"].to_numpy()
        f = g["f"].to_numpy(float)
        lrv = g["log_rv"].to_numpy(float)
        crit = dk(f[real], f[~real], o[real], o[~real])
        rv = dk(-lrv[real], -lrv[~real], o[real], o[~real])
        ref = pooled[(pooled["model"] == m) & (pooled["kind"] == "absolute")
                     & (pooled["test"] == "critic")].iloc[0]
        if not (np.isclose(crit["diff"], ref["statistic"], rtol=1e-9)
                and np.isclose(crit["z_hac"], ref["z_hac"], rtol=1e-9)):
            raise SystemExit(f"{m}: recomputed critic test differs from e3_pooled_raw.csv")
        rv_syn = np.exp(lrv[~real])
        rows.append({"model": m, "T": crit["diff"], "z_critic": crit["z_hac"],
                     "d_rv": rv["diff"], "ratio_gm": math.exp(rv["diff"]), "z_rv": rv["z_hac"],
                     "z_ratio": rv["z_hac"] / crit["z_hac"],
                     "sd_lrv": float(np.std(lrv[~real], ddof=1)),
                     "mean_rv": float(rv_syn.mean()), "median_rv": float(np.median(rv_syn))})
        desc = {"sd_lrv": float(np.std(lrv[real], ddof=1)),
                "mean_rv": float(np.exp(lrv[real]).mean()),
                "median_rv": float(np.median(np.exp(lrv[real])))}
        if real_desc is None:
            real_desc = desc
        elif not all(np.isclose(real_desc[k], desc[k]) for k in desc):
            raise SystemExit("the real evaluation windows differ across generators")

    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Learned critic} & \multicolumn{4}{c}{Fixed log realised volatility}"
        r" & \multicolumn{3}{c}{Evaluation windows} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-10}",
        r"Generator & $T$ & $\zhac$ & $\Delta_{\mathrm{RV}}$ & $e^{\Delta_{\mathrm{RV}}}$ & $\zhac$"
        r" & ratio of $\zhac$ & sd log RV & mean RV & median RV \\",
        r"\midrule",
    ]
    for r in rows:
        cells = [tnum(r["T"], 3), tnum(r["z_critic"], 1), tnum(r["d_rv"], 3),
                 tnum(r["ratio_gm"], 2), tnum(r["z_rv"], 1), tnum(r["z_ratio"], 2),
                 tnum(r["sd_lrv"], 2), tnum(r["mean_rv"], 2), tnum(r["median_rv"], 2)]
        lines.append(LABELS[str(r["model"])] + " & " + " & ".join(cells) + r" \\")
    assert real_desc is not None
    lines += [
        r"\midrule",
        r"Real \TEST{} & & & & & & & " + " & ".join(
            tnum(real_desc[k], 2) for k in ("sd_lrv", "mean_rv", "median_rv")) + r" \\",
        r"\bottomrule", r"\end{tabular}",
        "% generated by paper/scripts/e3_rv_baseline.py from results/e3/e3_witness.csv"
        " (critic T and z checked against results/e3/e3_pooled_raw.csv, seed 0)",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(pd.DataFrame(rows).to_string(index=False))
    print("real:", real_desc)


if __name__ == "__main__":
    main()
