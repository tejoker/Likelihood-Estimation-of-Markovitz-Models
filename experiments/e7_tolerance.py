"""E7: certified tolerance of the decision rule (power curve of E3's rule over the size of the
volatility error; v2 protocol amendment of 2026-11-17, docs/PROTOCOL_v2.md).

E3's rule declares GOOD only if its power against the pre-specified deviation (synthetic
volatility x 1.2) is at least 0.8 under both power designs D1 and D2; on the S&P 500 panel the
D2 power is 0.41, so every generator the anchored test does not reject is INCONCLUSIVE. This
experiment measures the rule's power for other deviation sizes, with E3's own replication
functions (``e3_sp500.power_rep_rows_batched`` / ``realvol_rep_rows_batched``, critic banks),
E3's seeds, settings and fitted GARCH-t panel; only ``pipeline.POWER_VOL_FACTOR`` changes, and
only the ``alternative`` case is run (the boundary case does not depend on the factor; its size
is E3's). Replications are paired with E3's (same streams, so the curves are monotone up to
Monte Carlo noise).

Certified tolerance (decision, fixed before this experiment ran): for t in TOLERANCES, the rule
resolves a volatility error of relative size t if its power is at least 0.8 under D1 and D2 for
both the upward factor 1 + t and the downward factor 1 / (1 + t). A generator that the anchored
rule does not reject is GOOD at the smallest resolved t ("not detectably worse than the real
past, within a volatility error of t"); D1 at upward factors >= 1.2 is taken from E3 (power 1.00
at 1.2; power is non-decreasing in the factor).

    PYTHONPATH=src uv run --project envs/gpu python experiments/v2.py e7 --workers 1
    PYTHONPATH=src uv run --project envs/gpu python experiments/v2.py e7 --plots-only
"""
from __future__ import annotations

import glob
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import e3_sp500 as e3  # noqa: E402
from synthval import config  # noqa: E402
from synthval import pipeline as pl  # noqa: E402

logger = logging.getLogger("experiments.e7")

EXP = "e7tol"
PARTS = ("curve",)
TOLERANCES = (0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0)
DESIGN_FACTORS = {
    "D2": (1.3, 1.5, 2.0, 1 / 1.2, 1 / 1.3, 1 / 1.5, 1 / 2.0),
    "D1": (1 / 1.2, 1 / 1.3, 1 / 1.5, 1 / 2.0, 1.1, 1 / 1.1),
}
# Precision extension (2026-11-17 16:45; compute is not a constraint, exactness is): finer D2 grid
# where the power jumps (between x1.3 and x1.5), D1 at +-10%, and 400 replications on the cells
# that decide the tolerance (``PRECISION_CELLS``); 100 elsewhere (saturated or inside the null).
DESIGN_FACTORS["D2"] = DESIGN_FACTORS["D2"] + (1.35, 1.4, 1.45, 1 / 1.35, 1 / 1.4, 1 / 1.45)
DESIGN_FACTORS["D1"] = DESIGN_FACTORS["D1"] + (1 / 1.35, 1 / 1.4, 1 / 1.45)
PRECISION_CELLS = {("D2", 1.3), ("D2", 1.35), ("D2", 1.4), ("D2", 1.45), ("D2", 1.5),
                   ("D2", 1 / 1.5), ("D1", 1.1), ("D1", 1 / 1.1)}
PRECISION_REPS = 400
#: Grid of the first E7 run (2026-11-17 09:53-13:17), kept in the checkpoint key so its 900
#: finished blocks are reused; D1 at 1/1.5 and 1/2.0 were added afterwards (needed to evaluate
#: the amendment's rule at tolerances 0.5 and 1.0) and run as new cells of the same part.
DESIGNS_KEY = {"D2": [1.3, 1.5, 2.0, 1 / 1.2, 1 / 1.3, 1 / 1.5, 1 / 2.0], "D1": [1 / 1.2, 1 / 1.3]}
FULL = {"reps": 100, "precision_reps": PRECISION_REPS, "blocks": 1}
QUICK = {"reps": 1, "precision_reps": 2, "blocks": 1}
ALPHA = 0.05
POWER_MIN = 0.8
#: E3's pre-specified factor, captured at import: _rep changes pl.POWER_VOL_FACTOR in the process
#: that runs it (the main process when --workers 1), so it must not be read afterwards.
E3_VOL_FACTOR: float = float(pl.POWER_VOL_FACTOR)


def _rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    del rng  # E3's streams
    pl.POWER_VOL_FACTOR = float(cell["f"])
    panel = e3.get_panel(cell["_tickers"])
    if cell["design"] == "D1":
        rows = e3.power_rep_rows_batched(panel, e3.load_fits(cell["_fit_path"]), rep, cell["_cfg"])
    else:
        rows = e3.realvol_rep_rows_batched(panel, rep, cell["_cfg"])
    for r in rows:
        r["vol_factor"] = float(cell["f"])
    return rows


def e3_fit_path() -> str:
    paths = sorted(glob.glob(str(Path(config.RESULTS_DIR) / "e3" / "checkpoints" / "fit_GARCHt"
                                 / "*" / "fitted.pkl")))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one E3 GARCH-t fit under {config.RESULTS_DIR}, got {paths}")
    return paths[0]


def run_curve(ctx: common.ExpContext, cfg: dict) -> None:
    e3cfg = e3.QUICK if ctx.quick else e3.FULL
    tickers = e3.panel_tickers(e3cfg)
    c = {"base_kw": e3cfg["base_kw"], "n_boot": e3cfg["n_boot"], "n_seeds": e3cfg["n_seeds"],
         "cases": ("alternative",), "batched": True}
    fit = e3_fit_path()
    cells = [{"cell": f"{d}_f{f:.4f}", "design": d, "f": f, "_tickers": list(tickers),
              "_fit_path": fit, "_cfg": c}
             for d, fs in DESIGN_FACTORS.items() for f in fs]
    conf = {"part": "curve", "e3_root_seed": e3.ROOT_SEED, "n_seeds": c["n_seeds"],
            "pipeline": pl.pipeline_config(), "designs": DESIGNS_KEY,
            "fit": fit, "tickers": list(tickers)}
    def n_reps(cell: dict) -> int:
        precise = any(cell["design"] == d and math.isclose(cell["f"], f)
                      for d, f in PRECISION_CELLS)
        return int(cfg["precision_reps"] if precise else cfg["reps"])

    df = common.run_cells(_rep, cells, part="curve", ctx=ctx, root_seed=e3.ROOT_SEED,
                          n_reps=n_reps, config=conf, block_size=cfg["blocks"],
                          workers=ctx.workers)
    ctx.save_csv(df, "e7_curve_raw", config={**conf, "reps": cfg["reps"],
                                             "precision_reps": cfg["precision_reps"]})


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rule_power(df: pd.DataFrame) -> tuple[int, int]:
    per_rep = df.groupby("rep")["pvalue"].apply(
        lambda p: min(1.0, 2.0 * float(np.median(p.to_numpy(dtype=float)))))
    return int((per_rep <= ALPHA).sum()), int(per_rep.size)


def build_outputs(ctx: common.ExpContext) -> pd.DataFrame | None:
    path = ctx.results_dir / "e7_curve_raw.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    e3dir = Path(config.RESULTS_DIR) / ("e3_quick" if ctx.quick else "e3")
    rows = []
    for d, name in (("D1", "e3_power_raw.csv"), ("D2", "e3_realvol_raw.csv")):
        e = pd.read_csv(e3dir / name)
        for case, f in (("alternative", E3_VOL_FACTOR), ("boundary", 1.0)):
            k, n = rule_power(e[e["case"] == case])
            rows.append({"design": d, "vol_factor": f, "k": k, "n": n, "source": "E3"})
    for (d, f), g in raw.groupby(["design", "vol_factor"]):
        k, n = rule_power(g)
        rows.append({"design": d, "vol_factor": float(f), "k": k, "n": n, "source": "E7"})
    curve = pd.DataFrame(rows).sort_values(["design", "vol_factor"]).reset_index(drop=True)
    curve["power"] = curve["k"] / curve["n"]
    ci = [wilson(k, n) for k, n in zip(curve["k"], curve["n"])]
    curve["lo"], curve["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    ctx.save_csv(curve, "e7_curve", config={"part": "curve_summary"})

    def power_at(d: str, f: float) -> float:
        m = curve[(curve["design"] == d) & np.isclose(curve["vol_factor"], f)]
        if len(m):
            return float(m["power"].iloc[0])
        if d == "D1" and f >= E3_VOL_FACTOR:  # non-decreasing in the upward factor
            return power_at("D1", E3_VOL_FACTOR)
        return math.nan

    # Two readings. "prereg": amendment A1 as written (both directions in D1 and D2).
    # "directional" (post hoc, PROTOCOL_v2 change log 2026-11-17): in D2 the new sample is the real
    # TEST period, about 20% calmer than TRAIN, so a calmer generator moves towards R_new and is
    # inside the anchored null; only upward errors are alternatives there. D1 keeps both directions.
    def _min(vals: list[float]) -> float:
        return min(vals) if all(np.isfinite(vals)) else math.nan

    tol = []
    for t in TOLERANCES:
        p = {(d, lab): power_at(d, f) for d in ("D1", "D2")
             for lab, f in (("up", 1 + t), ("down", 1 / (1 + t)))}
        pre = _min(list(p.values()))
        dirn = _min([p[("D1", "up")], p[("D1", "down")], p[("D2", "up")]])
        tol.append({"tolerance": t, **{f"{d}_{lab}": v for (d, lab), v in p.items()},
                    "min_power_prereg": pre,
                    "resolved_prereg": bool(np.isfinite(pre) and pre >= POWER_MIN),
                    "min_power_directional": dirn,
                    "resolved_directional": bool(np.isfinite(dirn) and dirn >= POWER_MIN)})
    tol = pd.DataFrame(tol)
    ctx.save_csv(tol, "e7_tolerance", config={"part": "tolerance", "power_min": POWER_MIN})

    def _cert(col: str) -> float:
        r = tol[tol[col]]
        return float(r["tolerance"].min()) if len(r) else math.nan

    certified = _cert("resolved_prereg")
    certified_dir = _cert("resolved_directional")

    tab = Path(ctx.tab_dir)
    tab.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Volatility factor & D1 rule power [95\% CI] & reps & D2 rule power [95\% CI] & reps \\",
             r"\midrule"]
    for f in sorted(curve["vol_factor"].unique()):
        cells = []
        for d in ("D1", "D2"):
            m = curve[(curve["design"] == d) & np.isclose(curve["vol_factor"], f)]
            if len(m):
                r = m.iloc[0]
                cells += [f"{r['power']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]", f"{int(r['n'])}"]
            else:
                cells += ["--", "--"]
        lab = f"{f:.3f}" + (r" (boundary: size)" if np.isclose(f, 1.0) else "")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule"]
    for _, r in tol.iterrows():
        pre = "--" if not np.isfinite(r["min_power_prereg"]) else f"{r['min_power_prereg']:.2f}"
        dr = "--" if not np.isfinite(r["min_power_directional"]) else f"{r['min_power_directional']:.2f}"
        lines.append(f"Tolerance $\\pm{100 * r['tolerance']:.0f}\\%$: min.\\ power & "
                     f"\\multicolumn{{2}}{{l}}{{as fixed (A1): {pre}"
                     f"{' (resolved)' if r['resolved_prereg'] else ''}}} & "
                     f"\\multicolumn{{2}}{{l}}{{directional: {dr}"
                     f"{' (resolved)' if r['resolved_directional'] else ''}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tab / "e7_tolerance.tex").write_text("\n".join(lines) + "\n")
    (tab / "e7_tolerance_note.tex").write_text(
        r"Rule power: share of replications in which $\min(1, 2\times\mathrm{median})$ of the "
        r"anchored $p$-values over the synthetic seeds is at most 0.05; Wilson 95\% intervals. "
        r"Factor 1.2 and the boundary (1.0) are E3's runs; other factors: E7, paired with E3's "
        r"replications. A tolerance $t$ is resolved if the power is at least 0.8 under D1 and D2 "
        r"for the factors $1+t$ and $1/(1+t)$; D1 at upward factors $\ge 1.2$ is E3's value at 1.2. "
        f"Certified tolerance, rule as fixed (A1): {'none in the grid' if not np.isfinite(certified) else f'$\\pm{100 * certified:.0f}\\%$'}; "
        f"directional reading (post hoc; D2 upward only): {'none in the grid' if not np.isfinite(certified_dir) else f'$\\pm{100 * certified_dir:.0f}\\%$'}.\n")
    print(curve.to_string())
    print(tol.to_string())
    print("certified tolerance (A1 as fixed):", certified, "| directional (post hoc):", certified_dir)
    return curve


def main(argv: list[str] | None = None) -> None:
    ctx = common.parse_args(EXP, PARTS, description=__doc__, argv=argv)
    cfg = dict(QUICK if ctx.quick else FULL)
    ctx.write_meta({"quick": ctx.quick, "parts": ctx.parts, "cfg": cfg,
                    "designs": {k: list(v) for k, v in DESIGN_FACTORS.items()},
                    "tolerances": list(TOLERANCES), "power_min": POWER_MIN})
    if not ctx.plots_only:
        for part in ctx.parts:
            run_curve(ctx, cfg)
    build_outputs(ctx)


if __name__ == "__main__":
    main()
