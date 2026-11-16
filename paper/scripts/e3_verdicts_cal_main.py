"""Main-text E3 verdict table with the D2-calibrated p-values (Section 7, label tab:e3-verdicts).

Same content and rounding as ``e3_verdicts_main.py`` (which writes
``manual_e3_verdicts_main.tex``), plus one column: the Monte Carlo p-value of the observed
anchored ``p_med`` of each generator against the boundary replications of design D2 of the power
check,

    p_D2 = (1 + #{replications r : p_med^(r) <= p_med^obs}) / (1 + n_reps),

where ``p_med^(r) = min(1, 2 * median_s p_s^(r))`` is recomputed from the five single-seed
boundary p-values of replication ``r`` (``results/e3/e3_realvol_raw.csv``, ``case ==
"boundary"``). This is an approximate calibration: D2 replaces the real generators by a
Gaussian boundary generator on the real TRAIN volatility path, so its noise law and precision
differ from those of the real configuration. The bootstrap gets no value (its anchored test is
invalid).

The script also writes, as LaTeX comment lines at the end of the table file, the counts behind
these p-values in D2 and D1 (``results/e3/e3_power_raw.csv``) and the Holm-adjusted p-values over
the four parametric generators (GBM, Merton, GARCH-t, Heston), once with the nominal ``p_med``
and once with the D2-calibrated p-values. The text of Section 7 quotes them from there.

Inputs: ``results/e3/e3_verdicts.csv``, ``results/e3/e3_realvol_raw.csv``,
``results/e3/e3_power_raw.csv``. Output: ``paper/tables/manual_e3_verdicts_cal_main.tex``.

Usage (from the repository root)::

    uv run python paper/scripts/e3_verdicts_cal_main.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "e3"
OUT = ROOT / "paper" / "tables" / "manual_e3_verdicts_cal_main.tex"
MODELS = ("GBM", "Merton", "GARCHt", "Heston", "Bootstrap")
PARAMETRIC = ("GBM", "Merton", "GARCHt", "Heston")
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


def boundary_pmed(raw: pd.DataFrame) -> np.ndarray:
    """Rule p-value min(1, 2 median_s p_s) of every boundary replication."""
    b = raw[raw["case"] == "boundary"]
    n_seeds = b.groupby("rep")["seed_index"].nunique()
    if n_seeds.nunique() != 1:
        raise ValueError("unequal number of seeds across replications")
    med = b.groupby("rep")["pvalue"].median()
    return np.minimum(1.0, 2.0 * med.to_numpy(float))


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    keys = sorted(p, key=lambda k: p[k])
    m = len(keys)
    out: dict[str, float] = {}
    running = 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * p[k]))
        out[k] = running
    return out


def main() -> None:
    v = pd.read_csv(RES / "e3_verdicts.csv").set_index("model")
    bd2 = boundary_pmed(pd.read_csv(RES / "e3_realvol_raw.csv"))
    bd1 = boundary_pmed(pd.read_csv(RES / "e3_power_raw.csv"))
    obs = {m: float(v.loc[m, "rel_p_combined"]) for m in MODELS if m in v.index}
    cnt2 = {m: int(np.sum(bd2 <= p)) for m, p in obs.items()}
    cnt1 = {m: int(np.sum(bd1 <= p)) for m, p in obs.items()}
    mc2 = {m: (1 + c) / (1 + bd2.size) for m, c in cnt2.items()}
    holm_nom = holm({m: obs[m] for m in PARAMETRIC})
    holm_d2 = holm({m: mc2[m] for m in PARAMETRIC})

    lines = [
        r"\begin{tabular}{lrrrrrll}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Absolute \WC{}} & \multicolumn{3}{c}{Anchored relative}"
        r" & \multicolumn{2}{c}{Verdict} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        r"Generator & $\pmed$ & seeds rej. & $\Danc$: median [range] & $\pmed$"
        r" & D2-cal.\ $p$ & $\hat\beta\ge0.8$ & $\betaL\ge0.8$ \\",
        r"\midrule",
    ]
    for m in MODELS:
        if m not in v.index:
            continue
        r = v.loc[m]
        valid = bool(r["rel_valid"])
        mark = "" if valid else DDAGGER
        cells = [
            tp(r["abs_p_combined"]),
            f"{int(r['abs_n_reject'])}/{int(r['n_seeds'])}",
            f"{tnum(r['delta_median'])} [{tnum(r['delta_min'])}, {tnum(r['delta_max'])}]" + mark,
            tp(r["rel_p_combined"]) + mark,
            f"{mc2[m]:.3f}" if valid else "--",
            VERDICT.get(str(r["verdict"]), str(r["verdict"])),
            VERDICT.get(str(r["verdict_cp"]), str(r["verdict_cp"])),
        ]
        lines.append(LABELS[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [
        f"% D2 boundary replications: {bd2.size}; counts with p_med^bd <= observed: "
        + ", ".join(f"{m} {cnt2[m]}" for m in obs),
        f"% D1 boundary replications: {bd1.size}; counts: "
        + ", ".join(f"{m} {cnt1[m]}" for m in obs),
        "% Holm over GBM, Merton, GARCH-t, Heston, nominal p_med: "
        + ", ".join(f"{m} {holm_nom[m]:.4f}" for m in PARAMETRIC),
        "% Holm over GBM, Merton, GARCH-t, Heston, D2-calibrated p: "
        + ", ".join(f"{m} {holm_d2[m]:.4f}" for m in PARAMETRIC),
        "% generated by paper/scripts/e3_verdicts_cal_main.py from results/e3/e3_verdicts.csv,"
        " e3_realvol_raw.csv, e3_power_raw.csv",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print("\n".join(lines[-5:]))


if __name__ == "__main__":
    main()
