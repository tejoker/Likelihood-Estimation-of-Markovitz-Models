"""E5 ablation: time-pooled CNN critic (``arch="pooled"``) vs the default MLP critic.

The only change is the critic network (:class:`synthval.twosample.PooledConvCritic`: two
``Conv1d(kernel 5)`` blocks, mean over time, dense head; optimiser, steps, split and inference
as in the default test). Every replication reuses the data and test streams of the experiment it
is compared with, so MLP and pooled results are paired replication by replication.

Parts
-----
sim    W-critic with ``arch="pooled"`` on
       * E2 designs (``e2_ground_truth``; n = 500 independent windows per sample): family a
         (volatility level) k = 0, 1, 2; b (tails) k = 2, 4; c (volatility clustering)
         k = 0, 1, 2; d (jumps) k = 2, 4 (size checked in a and c: the permutation p-value is
         exact for independent windows whatever the network); same windows and test stream as E2;
       * E2-dep designs (``e2_dependence``; n = 2000): mix k = 2, block k = 2 (marginal law
         fixed exactly, so only dependence separates the samples).
       ``sim_reps`` replications per alternative cell, ``h0_reps`` per H0 cell. Paired with the MLP W-critic (and KS, MMD) rows of
       the E2 / E2-dep raw CSVs at the same replication indices.
real   E3's real-data protocol with ``arch="pooled"`` via E5's machinery (``GRID`` = the pooled
       configuration only): absolute W-critic + anchored relative test per model at E3's five
       synthetic seeds (paired with E3's pooled rows), and the power designs D1, D2 (E3's
       replication functions; ``power_reps`` replications x 5 seeds, paired with E3's first
       ``power_reps`` replications); then the verdict rule.

Outputs: ``results/e5arch/`` (raw CSVs, summaries) and tables ``e5_arch_sim.tex``,
``e5_arch_real.tex`` (+ notes) in the paper's table directory.

    uv run python experiments/e5_architecture.py --part real --workers 4
    uv run python experiments/e5_architecture.py --part sim --workers 6
    uv run python experiments/e5_architecture.py --plots-only
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import e2_dependence as dep  # noqa: E402
import e2_ground_truth as e2  # noqa: E402
import e3_sp500 as e3  # noqa: E402
import e5_robustness as e5  # noqa: E402
from synthval.twosample import CriticTest  # noqa: E402

logger = logging.getLogger("experiments.e5arch")

EXP = "e5arch"
PARTS = ("sim", "real")
ARCH_KW = {"arch": "pooled"}
POOLED = {"config": "pooled", "factor": "Critic architecture", "setting": "time-pooled CNN",
          "L": 20, "kw": dict(ARCH_KW)}
E2_LEVELS = {"a": [0, 1, 2], "b": [2, 4], "c": [0, 1, 2], "d": [2, 4]}
DEP_LEVELS = {"mix": [2], "block": [2]}
FULL = {"sim_reps": 50, "h0_reps": 200, "e2_n": 500, "dep_n": 2000, "power_reps": 20, "n_perm": None,
        "blocks": {"sim": 5}}
QUICK = {"sim_reps": 2, "h0_reps": 3, "e2_n": 100, "dep_n": 200, "power_reps": 1, "n_perm": 100,
         "blocks": {"sim": 1}}
ALPHA = 0.05


# --------------------------------------------------------------------------------------------
# Part: sim
# --------------------------------------------------------------------------------------------
def _sim_rep(cell: dict, rep: int, rng: np.random.Generator) -> dict[str, Any]:
    del rng  # the streams of the paired experiment are used
    fam, n, k = cell["family"], int(cell["n"]), int(cell["k"])
    if cell["src"] == "e2":
        F = e2.FAMILIES[fam]
        X = e2.sim_windows(F["P"], n, common.keyed_rng(e2.ROOT_SEED, "main_X", fam, n, rep))
        Y = e2.sim_windows(F["Q"][k], n, common.keyed_rng(e2.ROOT_SEED, "main_Y", fam, n, k, rep))
        trng = common.keyed_rng(e2.ROOT_SEED, "main_test", fam, n, k, rep, "critic")
    else:
        X = dep.sim_P(n, common.keyed_rng(dep.ROOT_SEED, "X", fam, n, rep))
        Y = dep.sim_Q(fam, k, n, common.keyed_rng(dep.ROOT_SEED, "Y", fam, n, k, rep))
        trng = common.keyed_rng(dep.ROOT_SEED, "test", fam, n, k, rep, "critic")
    kw = dict(ARCH_KW)
    if cell["_n_perm"]:
        kw["n_perm"] = int(cell["_n_perm"])
    test = CriticTest(**kw)
    res, dt, err = common.safe_test(test, X, Y, rng=trng)
    return {"src": cell["src"], "family": fam, "n": n, "k": k, "test": "critic_pooled",
            "pvalue": float(res.pvalue) if res is not None else np.nan,
            "statistic": float(res.statistic) if res is not None else np.nan,
            "seconds": dt, "error": err}


def run_sim(ctx: common.ExpContext, cfg: dict) -> None:
    cells = [{"cell": f"e2_{f}_{cfg['e2_n']}_{k}", "src": "e2", "family": f, "n": cfg["e2_n"],
              "k": k, "_n_perm": cfg["n_perm"]} for f, ks in E2_LEVELS.items() for k in ks]
    cells += [{"cell": f"dep_{f}_{cfg['dep_n']}_{k}", "src": "dep", "family": f,
               "n": cfg["dep_n"], "k": k, "_n_perm": cfg["n_perm"]}
              for f, ks in DEP_LEVELS.items() for k in ks]
    conf = {"part": "sim", "arch": ARCH_KW, "e2_seed": e2.ROOT_SEED, "dep_seed": dep.ROOT_SEED,
            "n_perm": cfg["n_perm"], "code": e5.lib_code_hash()}
    df = common.run_cells(_sim_rep, cells, part="sim", ctx=ctx, root_seed=e2.ROOT_SEED,
                          n_reps=lambda c: (cfg["h0_reps"] if c["src"] == "e2" and c["k"] == 0
                                            else cfg["sim_reps"]),
                          config=conf, block_size=cfg["blocks"]["sim"],
                          workers=e5.part_workers(ctx, "sim"))
    ctx.save_csv(df, "e5arch_sim_raw", config={**conf, "sim_reps": cfg["sim_reps"],
                                                "h0_reps": cfg["h0_reps"]})


# --------------------------------------------------------------------------------------------
# Part: real (E5 grid + power designs with the pooled configuration only)
# --------------------------------------------------------------------------------------------
def run_real(ctx: common.ExpContext, cfg: dict) -> None:
    e5.GRID[:] = [POOLED]
    e5.GRID_BY_NAME.clear()
    e5.GRID_BY_NAME.update({"pooled": POOLED})
    c5 = dict(e5.QUICK if ctx.quick else e5.FULL)
    c5["power_reps"] = int(cfg["power_reps"])
    e5.run_grid(ctx, c5)
    e5.run_design_part(ctx, c5, "D1")
    e5.run_design_part(ctx, c5, "D2")


# --------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(a_only: int, b_only: int) -> float:
    m = a_only + b_only
    return float(min(1.0, 2 * stats.binom.cdf(min(a_only, b_only), m, 0.5))) if m else 1.0


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}]" if n else "--"


def rule_p(p: pd.Series) -> float:
    return float(min(1.0, 2 * np.median(p.to_numpy(dtype=float))))


def fmt_p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


E2_DESIGN = {"a": "volatility $\\times(1+\\delta)$, $\\delta$", "b": "tails, $\\nu$ of $Q$",
             "c": "clustering, persistence of $Q$", "d": "jumps, GBM fraction $\\delta$"}


def sim_table(ctx: common.ExpContext) -> pd.DataFrame | None:
    path = ctx.results_dir / "e5arch_sim_raw.csv"
    if not path.exists():
        return None
    pooled = pd.read_csv(path)
    raw = {"e2": pd.concat([pd.read_csv(Path(e2.__file__).resolve().parents[1] / "results" / "e2"
                                        / f"e2_main_{f}_raw.csv") for f in E2_LEVELS]),
           "dep": pd.concat([pd.read_csv(Path(dep.__file__).resolve().parents[1] / "results"
                                         / "e2dep" / f"e2dep_{f}_raw.csv") for f in DEP_LEVELS])}
    rows = []
    for (src, fam, n, k), g in pooled.groupby(["src", "family", "n", "k"], sort=False):
        g = g[g["error"].fillna("") == ""].set_index("rep")["pvalue"].dropna()
        r = raw[src]
        r = r[(r["family"] == fam) & (r["n"] == n) & (r["k"] == k)]
        ref = {t: r[r["test"] == t].set_index("rep")["pvalue"] for t in ("critic", "mmd", "ks")}
        reps = g.index.intersection(ref["critic"].dropna().index)
        rp, rm = g.loc[reps] <= ALPHA, ref["critic"].loc[reps] <= ALPHA
        level = (e2.FAMILIES[fam]["ticks"][k] if src == "e2"
                 else dep.FAMILIES[fam]["levels"][k])
        row = {"src": src, "family": fam, "n": int(n), "k": int(k), "level": level,
               "reps": len(reps), "rej_mlp": int(rm.sum()), "rej_pooled": int(rp.sum()),
               "pooled_only": int((rp & ~rm).sum()), "mlp_only": int((rm & ~rp).sum())}
        for t in ("mmd", "ks"):
            s = ref[t].reindex(reps)
            row[f"rej_{t}"] = int((s <= ALPHA).sum())
        row["mcnemar_p"] = mcnemar_exact(row["pooled_only"], row["mlp_only"])
        row["sec_pooled"] = float(pooled[(pooled.family == fam) & (pooled.k == k)]["seconds"].mean())
        row["sec_mlp"] = float(r[r["test"] == "critic"]["seconds"].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    ctx.save_csv(df, "e5arch_sim_summary", config={"part": "sim_summary"})
    fam_title = {**{f: e2.FAMILIES[f]["title"] for f in E2_LEVELS},
                 **{f: "E2-dep " + dep.FAMILIES[f]["title"] for f in DEP_LEVELS}}
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Family & Level & $n$ & Reps & W-critic, MLP & W-critic, pooled & McNemar $p$ & "
             r"MMD & KS \\", r"\midrule"]
    last = None
    for _, x in df.iterrows():
        fam = fam_title[x["family"]] if x["family"] != last else ""
        last = x["family"]
        lev = f"{x['level']}" + (" ($H_0$)" if (x["src"] == "e2" and x["k"] == 0) else "")
        ks = fmt_rate(x["rej_ks"], x["reps"]) + (r"$^{\dagger}$" if
                                                 (x["src"] == "dep" or x["family"] == "c") else "")
        lines.append(f"{fam} & {lev} & {x['n']} & {x['reps']} & {fmt_rate(x['rej_mlp'], x['reps'])}"
                     f" & {fmt_rate(x['rej_pooled'], x['reps'])} & {fmt_p(x['mcnemar_p'])} & "
                     f"{fmt_rate(x['rej_mmd'], x['reps'])} & {ks} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tab = Path(ctx.tab_dir)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "e5_arch_sim.tex").write_text("\n".join(lines) + "\n")
    sec_p = df["sec_pooled"].mean()
    sec_m = df["sec_mlp"].mean()
    note = (r"Rejection rates at $\alpha=0.05$ with Wilson 95\% intervals. Every replication uses "
            r"the windows and the test stream of the same replication of E2 (families a--d, "
            r"independent windows) or E2-dep (marginal law fixed exactly); only the critic "
            r"network differs. MLP: the default W-critic (3 hidden layers of width 128); pooled: "
            r"two Conv1d blocks (64 channels, kernel 5), mean over time, dense head. MMD and KS: "
            r"the same replications of E2 / E2-dep. McNemar: exact two-sided test on the "
            r"discordant replications. $^{\dagger}$ KS assumes i.i.d.\ returns and is not valid "
            r"within GARCH windows. Mean time per test (one thread): pooled "
            f"{sec_p:.0f} s, MLP {sec_m:.0f} s.")
    (tab / "e5_arch_sim_note.tex").write_text(note + "\n")
    return df


def _design_rule(df: pd.DataFrame, case: str, max_rep: int | None = None) -> tuple[int, int]:
    d = df[df["case"] == case]
    if max_rep is not None:
        d = d[d["rep"] < max_rep]
    per_rep = d.groupby("rep")["pvalue"].apply(rule_p)
    return int((per_rep <= ALPHA).sum()), int(per_rep.size)


def real_table(ctx: common.ExpContext) -> pd.DataFrame | None:
    gpath = ctx.results_dir / "e5_grid_raw.csv"
    if not gpath.exists():
        return None
    root = Path(e3.__file__).resolve().parents[1] / "results" / "e3"
    pooled = pd.read_csv(gpath)
    mlp = pd.read_csv(root / "e3_pooled_raw.csv")
    mlp = mlp.assign(seed_index=mlp["rep"])
    designs = {}
    for d, part, e3name in (("D1", "gridpower", "e3_power_raw.csv"),
                            ("D2", "gridrealvol", "e3_realvol_raw.csv")):
        p = ctx.results_dir / f"e5_{part}_raw.csv"
        if p.exists():
            pp = pd.read_csv(p)
            n_rep = int(pp["rep"].max()) + 1
            m = pd.read_csv(root / e3name)
            designs[d] = {"pooled": {c: _design_rule(pp, c) for c in ("alternative", "boundary")},
                          "mlp_paired": {c: _design_rule(m, c, n_rep)
                                         for c in ("alternative", "boundary")},
                          "mlp_full": {c: _design_rule(m, c) for c in ("alternative", "boundary")}}
    power_ok = {arch: all(designs.get(d, {}).get(arch, {}).get("alternative", (0, 1))[0]
                          / max(1, designs.get(d, {}).get(arch, {}).get("alternative", (0, 1))[1])
                          >= 0.8 for d in ("D1", "D2")) and len(designs) == 2
                for arch in ("pooled", "mlp_full")}
    rows = []
    for model in e3.MODELS:
        row: dict[str, Any] = {"model": model}
        for arch, df in (("mlp", mlp), ("pooled", pooled)):
            d = df[df["model"] == model]
            a = d[d["test"] == "critic"]["pvalue"]
            r = d[d["test"] == "relative_anchored"]["pvalue"]
            row[f"abs_{arch}"] = rule_p(a) if len(a) else np.nan
            row[f"abs_rej_{arch}"] = f"{int((a <= ALPHA).sum())}/{len(a)}"
            row[f"rel_{arch}"] = rule_p(r) if len(r) else np.nan
            if model in e3.REL_INVALID:
                v = "INVALID"
            elif row[f"rel_{arch}"] <= ALPHA:
                v = "BAD"
            else:
                v = "GOOD" if power_ok["pooled" if arch == "pooled" else "mlp_full"] \
                    else "INCONCLUSIVE"
            row[f"verdict_{arch}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    ctx.save_csv(df, "e5arch_real_summary", config={"part": "real_summary",
                                                     "designs": {d: {a: {c: list(v) for c, v in
                                                                         x.items()}
                                                                     for a, x in dd.items()}
                                                                 for d, dd in designs.items()}})
    names = {m: e3.MODEL_LABELS.get(m, m) for m in e3.MODELS}
    lines = [r"\begin{tabular}{lllllll}", r"\toprule",
             r" & \multicolumn{2}{c}{Absolute W-critic, $2\times$median $p$ (seeds rej.)} & "
             r"\multicolumn{2}{c}{Anchored relative, $2\times$median $p$} & "
             r"\multicolumn{2}{c}{Verdict} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
             r"Model & MLP & pooled & MLP & pooled & MLP & pooled \\", r"\midrule"]
    for _, x in df.iterrows():
        lines.append(f"{names[x['model']]} & {fmt_p(x['abs_mlp'])} ({x['abs_rej_mlp']}) & "
                     f"{fmt_p(x['abs_pooled'])} ({x['abs_rej_pooled']}) & {fmt_p(x['rel_mlp'])} & "
                     f"{fmt_p(x['rel_pooled'])} & {x['verdict_mlp']} & {x['verdict_pooled']} \\\\")
    lines.append(r"\midrule")
    for d in ("D1", "D2"):
        if d not in designs:
            continue
        for case, lab in (("alternative", "rule power"), ("boundary", "rule boundary size")):
            mf, mp, pl_ = (designs[d]["mlp_full"][case], designs[d]["mlp_paired"][case],
                           designs[d]["pooled"][case])
            lines.append(f"{d} {lab} & \\multicolumn{{2}}{{l}}{{MLP {fmt_rate(*mf)} ({mf[1]} reps); "
                         f"same {mp[1]} reps {fmt_rate(*mp)}}} & \\multicolumn{{4}}{{l}}{{pooled "
                         f"{fmt_rate(*pl_)} ({pl_[1]} reps)}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tab = Path(ctx.tab_dir)
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "e5_arch_real.tex").write_text("\n".join(lines) + "\n")
    note = (r"E3 protocol (100 tickers, time-block split with 10 blocks, Driscoll--Kraay HAC "
            r"inference, 5 synthetic seeds, $p$ combined as $\min(1, 2\times\mathrm{median})$) "
            r"with the default MLP critic (E3) and the time-pooled CNN critic; same fitted "
            r"models, seeds and test streams. Verdict: BAD if the anchored $p\le0.05$; GOOD if "
            r"not BAD and the rule power is at least 0.8 under both D1 and D2; INCONCLUSIVE "
            r"otherwise; INVALID: no relative test for the bootstrap. Power designs: E3's "
            r"replications (pooled: the first replications only, paired with the MLP row "
            r"``same reps''), Wilson 95\% intervals. The pooled architecture was chosen after "
            r"the real-data results of the MLP critic were known; these rows are post hoc.")
    (tab / "e5_arch_real_note.tex").write_text(note + "\n")
    return df


def build_outputs(ctx: common.ExpContext) -> None:
    s = sim_table(ctx)
    r = real_table(ctx)
    if s is not None:
        print(s[["family", "k", "level", "reps", "rej_mlp", "rej_pooled", "rej_mmd", "rej_ks",
                 "mcnemar_p"]].to_string())
    if r is not None:
        print(r.to_string())


def main(argv: list[str] | None = None) -> None:
    ctx = common.parse_args(EXP, PARTS, description=__doc__, argv=argv)
    cfg = dict(QUICK if ctx.quick else FULL)
    ctx.write_meta({"quick": ctx.quick, "parts": ctx.parts, "cfg": cfg, "arch": ARCH_KW,
                    "pooled_config": POOLED, "code": e5.lib_code_hash()})
    if not ctx.plots_only:
        for part in ctx.parts:
            {"sim": run_sim, "real": run_real}[part](ctx, cfg)
    build_outputs(ctx)


if __name__ == "__main__":
    main()
