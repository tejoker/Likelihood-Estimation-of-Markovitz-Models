"""E5: robustness of the E3 verdicts (DESIGN section 4, E5).

The E3 real-data pipeline (``synthval.pipeline`` and ``experiments/e3_sp500.py``: 100-ticker
panel, per-ticker fits on TRAIN, one synthetic path per ticker over the TEST horizon started from
the fitted end state, windows normalised with the real TRAIN std, dependent-window protocol:
time-block train/eval split and Driscoll-Kraay HAC p-values) is rerun under one-at-a-time changes
of its tuning choices and under an alternative TRAIN/TEST split; the critic's training settings
are also examined on simulated data with independent windows.

E5 applies E3's decision rule exactly (``e3_sp500`` docstring, part ``verdict``): the anchored
relative p-value is twice the median over ``n_seeds = 10`` synthetic seeds (E3's seeds,
``e3_sp500.synthetic_seed``); the power and boundary size are those of that rule, estimated with
E3's own replication functions run with the configuration's settings
(``e3_sp500.power_rep_rows``, design D1: per-ticker GARCH-t, R_past* from stationarity, R_new*
and G* = vol x 1.2 from the end-of-TRAIN state, boundary G0 from stationarity, i.e. exactly
Law(G0) = Law(R_past*); ``e3_sp500.realvol_rep_rows``, design D2: the real TRAIN volatility
path). GOOD needs the rule's power >= 0.8 under every design (the minimum is used). The
stationary bootstrap gets no verdict (``e3_sp500.REL_INVALID``: its synthetic windows copy R_past
windows that the critic saw as pool negatives); its relative p-values are kept, flagged, and
left out of every change count. Models and options (``e3_sp500.MODEL_KW``), fits (E3's pickles
when they exist), synthetic seeds and test streams are E3's, so the default configuration
reproduces E3's critic and anchored p-values, power replications and verdicts
(``compare_with_e3`` checks this against E3's CSVs when both full runs exist).

Grid (one-at-a-time, ``GRID``): window length L in {5, 10, 60} (default 20); critic width x depth
32x2 and 256x4 (default 128x3); critic steps 500 and 3000 (default 1000); learning rate 1e-4
(default 1e-3); gradient-penalty weight 1 (default 10); 4 time blocks (default 10); train
fraction 0.3 and 0.7 (default 0.5); Newey-West HAC lag rule (default Andrews AR(1) plug-in).
Every change applies to the absolute critic test, the anchored relative test, its power check
and the size check. The number of permutations is not a grid factor: under the dependent
protocol the reported p-values are HAC, the permutations run after training, so the critic and
the HAC p-value are identical for every n_perm; n_perm is examined in ``simpower``, where the
permutation p-value is the reported one.

Sub-experiments (``--part``, canonical order; alias ``power`` = gridpower + gridrealvol)
-------------------------------------------------------------------------------------
grid
    Pooled absolute critic test (real TEST windows vs synthetic windows) and anchored relative
    test (R_past = real TRAIN windows), per configuration and model, 10 synthetic seeds; per
    configuration and model the p-values are combined as ``min(1, 2 median)`` over the seeds
    (theory.md Remark 8.5). Rows record evaluation sizes, HAC lags, split type, training steps.
gridsize
    Size of the absolute critic test (one seed, HAC p-value) under each configuration, so that
    decision changes across configurations can be told apart from changes of size: X* and G*
    independent TEST-length panels from one law, tested exactly like E3. Designs ``garch``
    (E3's per-ticker GARCH-t fits, both from the fitted end-of-TRAIN state like E3's synthetic
    paths; tickers independent) and ``factor`` (E1's one-factor ``FactorGARCHt`` fitted on the
    real TRAIN panel, both from stationarity; common volatility regimes across tickers, as in
    real data). 200 replications per design; X*, G* and the test streams are the same in every
    configuration (common random numbers), so each configuration is compared with the default
    by an exact McNemar test on the paired rejections.
split
    Alternative split TRAIN 2018-01-02..2021-12-31 / TEST 2022-01-03..2025-06-17 (same 100
    tickers): the five models refitted on the alternative TRAIN; the full E3 battery (pooled
    absolute critic, lrcritic, c2st, mmd (wild), sigmmd (wild), ks (invalid under dependence);
    anchored (primary) and two-critic (secondary, anti-conservative at the boundary) relative
    tests), 10 seeds; the rule's power and boundary size at the alternative split's sizes
    (designs D1 and D2 with E3's functions, 50 replications each) for the verdict. The battery
    runs on at most ``MAX_WORKERS["split"]`` processes (about 0.9 GB per full battery).
simpower
    Power of the critic test (independent windows, exact permutation p-values) against two E2
    alternatives at n = 500 windows of L = 20 per sample: (a) Student-t(5) volatility x 1.1 and
    (c) GARCH(1,1)-t persistence 0.90 vs 0.98 (matched unconditional variance and nu); laws and
    simulators from ``e2_ground_truth``; lr in {1e-4, 1e-3} x critic steps in {500, 1000, 3000},
    plus 200 instead of 1000 permutations at the default lr and steps (same trained critic: only
    the Monte Carlo error of the permutation p-value changes); 100 replications per cell, with
    the same X, Y and test generator in every cell. It documents how power depends on the
    training settings (the defaults were fixed before any real-data run, DESIGN change log; a
    simulation cannot show that).
gridpower
    Design D1 of the rule's power check under each configuration (``e3_sp500.power_rep_rows``
    with the configuration's critic settings and window length: per replication R_past* and
    R_new* once, then 10 seeds of G* and of G0 with the same test streams). 50 replications per
    configuration (DESIGN minimum), 200 for the default (= E3's, so the default's power is E3's).
    Replication k uses E3's streams in every configuration (common random numbers).
gridrealvol
    Design D2 under each configuration (``e3_sp500.realvol_rep_rows``), same replication counts.

Verdict per configuration and model: BAD if the combined anchored p is <= 0.05; GOOD if not and
the minimum over the designs run for that configuration of the rule's power is >= 0.8;
INCONCLUSIVE otherwise; ``n/a`` for the bootstrap; ``verdict_cp`` uses the Clopper-Pearson lower
bound instead. Changes against the default are flagged separately for the relative decision
(reject or not at 0.05), for verdict changes caused only by the power estimate crossing 0.8, and
for the absolute decision (to be read next to the configuration's size from ``gridsize``).

Outputs (full mode): ``results/e5/*.csv`` + ``meta.json`` (+ ``meta_<part>.json``): replication
rows ``e5_grid_raw``, ``e5_gridsize_raw``, ``e5_gridpower_raw``, ``e5_gridrealvol_raw``,
``e5_split_raw``, ``e5_split_power_raw``, ``e5_simpower_raw``; summaries ``e5_sensitivity`` (per
configuration and model), ``e5_calibration`` (per configuration: rule power and boundary size
per design, absolute size per design with McNemar against the default), ``e5_power_summary``,
``e5_split``, ``e5_split_power``, ``e5_simpower``, ``e5_fits_main``, ``e5_fits_alt``,
``e5_e3_alignment``; fitted models ``results/e5/fits/``; tables ``paper/tables/e5_sensitivity.tex``,
``e5_calibration.tex``, ``e5_split.tex``, ``e5_simpower.tex``; figure
``paper/figures/e5_sensitivity.pdf``. ``--quick`` writes only to ``results/e5_quick/`` (8 tickers,
2 seeds, reduced critic steps / permutations / replications, cheaper Heston SMM), then times one
unit of each kind at the FULL settings and prints the estimated full-run wall time per part at
12 workers. ``--plots-only`` rebuilds tables and figure from the saved CSVs.

Usage (run the parts one after another: about 0.6 GB per worker, 1 GB for the split battery;
workers are capped by the available memory)::

    uv run python experiments/e5_robustness.py --quick
    uv run python experiments/e5_robustness.py --part grid,gridsize,split,simpower --workers 12
    uv run python experiments/e5_robustness.py --part gridpower --configs light --workers 12
    uv run python experiments/e5_robustness.py --part gridpower --configs heavy --workers 12
    uv run python experiments/e5_robustness.py --part gridrealvol --workers 12
    uv run python experiments/e5_robustness.py --plots-only

Run E3's ``--part fit`` first, so that E5 uses E3's fitted models (it fits identical ones
otherwise). ``--configs a,b`` restricts grid, gridsize, gridpower and gridrealvol to those
configurations (``light`` = all but cap_large and steps3000, ``heavy`` = those two); the results
of configurations computed earlier under the same settings are read back from the checkpoints,
so the CSVs and tables always cover every finished configuration. ``--power-reps`` and
``--power-reps-default`` override the replication counts of the power parts (finished
replications are reused).

Checkpoints are keyed by the part's settings, a hash of the library code the parts run
(``synthval`` config, data, pipeline, utils, models, twosample), the source of the E3 functions
they call and the fitted parameters, so a resumed run never mixes code versions or fits.
Replication counts are not part of the key: raising them later reuses the finished replications.
Every result row records its p-value type (``pvalue_type``: hac for the real-data learned and
relative tests, wild for MMD / SigMMD, asymptotic_iid for KS, permutation in simpower).
Rejection rates come with Wilson 95% intervals (boundary sizes also Clopper-Pearson).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import dataclasses  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import common  # noqa: E402  (experiments/ is on sys.path when run as a script)
import e3_sp500 as e3  # noqa: E402  (E3's seeds, model options, power designs and verdict rule)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import synthval  # noqa: E402
from synthval import config, data, pipeline, plotting, utils  # noqa: E402
from synthval.models import get_model, get_mv_model  # noqa: E402

logger = logging.getLogger("experiments.e5")

EXP = "e5"
ROOT_SEED = 20251117      # E5's own streams (gridsize, simpower)
PARTS = ("grid", "gridsize", "split", "simpower", "gridpower", "gridrealvol")
ALIASES = {"power": ["gridpower", "gridrealvol"]}
GRID_PARTS = ("grid", "gridsize", "gridpower", "gridrealvol")
ALPHA = pipeline.ALPHA
NAN = float("nan")

MODELS: tuple[str, ...] = tuple(e3.MODELS)
MODEL_LABELS: dict[str, str] = dict(e3.MODEL_LABELS)
#: Relative tests that are invalid for a model (E3's list): flagged, no verdict, no change count.
REL_INVALID: dict[str, str] = dict(e3.REL_INVALID)
#: Power designs of the decision rule (E3: D1 = ``power`` part, D2 = ``realvol`` part).
DESIGNS: tuple[str, ...] = tuple(e3.DESIGNS)
DESIGN_PART: dict[str, str] = {"D1": "gridpower", "D2": "gridrealvol"}
#: Size-check designs of ``gridsize``.
SIZE_DESIGNS: tuple[str, ...] = ("garch", "factor")
SIZE_LABELS: dict[str, str] = {"garch": r"GARCH-$t$", "factor": "factor"}
#: Quick mode: a much cheaper Heston SMM (fewer simulated paths, evaluations and bootstraps).
MODEL_KW_QUICK_OVERRIDE: dict[str, dict[str, Any]] = {
    "Heston": {"sim_obs": 5000, "min_paths": 3, "maxfev": 10, "n_nm_starts": 1, "n_boot": 30},
}

ALT_TRAIN = ("2018-01-02", "2021-12-31")
ALT_TEST = ("2022-01-03", "2025-06-17")
SPLITS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "main": (tuple(config.TRAIN), tuple(config.TEST)),
    "alt": (ALT_TRAIN, ALT_TEST),
}

# --------------------------------------------------------------------------------------------
# The one-at-a-time grid (full-run values; quick mode rescales n_steps)
# --------------------------------------------------------------------------------------------
GRID: list[dict[str, Any]] = [
    {"config": "default", "factor": "Default", "setting": "--", "L": 20, "kw": {}},
    {"config": "L5", "factor": "Window length $L$", "setting": "5", "L": 5, "kw": {}},
    {"config": "L10", "factor": "Window length $L$", "setting": "10", "L": 10, "kw": {}},
    {"config": "L60", "factor": "Window length $L$", "setting": "60", "L": 60, "kw": {}},
    {"config": "cap_small", "factor": r"Width$\times$depth", "setting": r"$32\times2$", "L": 20,
     "kw": {"hidden": 32, "depth": 2}},
    {"config": "cap_large", "factor": r"Width$\times$depth", "setting": r"$256\times4$",
     "L": 20, "kw": {"hidden": 256, "depth": 4}},
    {"config": "steps500", "factor": "Critic steps", "setting": "500", "L": 20,
     "kw": {"n_steps": 500}},
    {"config": "steps3000", "factor": "Critic steps", "setting": "3000", "L": 20,
     "kw": {"n_steps": 3000}},
    {"config": "lr1e-4", "factor": "Learning rate", "setting": r"$10^{-4}$", "L": 20,
     "kw": {"lr": 1e-4}},
    {"config": "gp1", "factor": r"GP weight $\lambda$", "setting": "1", "L": 20,
     "kw": {"gp_lambda": 1.0}},
    {"config": "tb4", "factor": "Time blocks", "setting": "4", "L": 20,
     "kw": {"time_blocks": 4}},
    {"config": "tf0.3", "factor": "Train fraction", "setting": "0.3", "L": 20,
     "kw": {"train_frac": 0.3}},
    {"config": "tf0.7", "factor": "Train fraction", "setting": "0.7", "L": 20,
     "kw": {"train_frac": 0.7}},
    {"config": "hacNW", "factor": "HAC lag rule", "setting": "Newey-West", "L": 20,
     "kw": {"hac_lags": "nw"}},
]
GRID_BY_NAME = {g["config"]: g for g in GRID}
#: Relative cost of one replication (default = 1), for scheduling and as the fallback of the
#: timing estimates (the quick-mode probes measure the PROBED configurations).
COST = {"cap_small": 0.25, "cap_large": 4.5, "steps500": 0.6, "steps3000": 3.5}
PROBED = ("default", "L5", "L60", "cap_small", "cap_large", "steps500", "steps3000")
#: The two expensive configurations (run separately with ``--configs`` if time is short).
HEAVY = ("cap_large", "steps3000")
#: Peak resident memory per worker (GB) used to cap the workers (full battery measured at about
#: 0.93 GB; critic + anchored at 0.46-0.56 GB, review), and hard caps.
WORKER_GB: dict[str, float] = {"split": 1.0}
DEFAULT_WORKER_GB = 0.6
MAX_WORKERS: dict[str, int] = {"split": 6}

# Simulated power (E2 alternatives)
SIM_FAMILIES = {"a": 0.10, "c": 0.90}   # E2 family -> level of the alternative Q
SIM_LABELS = {"a": r"Volatility $\times1.1$ ($t_5$)",
              "c": r"GARCH persistence 0.90 vs 0.98"}
SIM_LR = (1e-4, 1e-3)
SIM_STEPS = (500, 1000, 3000)
SIM_DEFAULT = (1e-3, 1000)              # (lr, steps) of the extra n_perm cell

FULL: dict[str, Any] = {
    "n_tickers": None,           # None: the default N_PANEL = 100 panel
    "n_seeds": 5,                # = e3_sp500.FULL["n_seeds"]: synthetic seeds, seeds per power rep
    "critic_base": {},           # pipeline defaults (CRITIC_KW)
    "steps_scale": 1.0,
    "power_reps": 50,            # per non-default configuration and design (DESIGN minimum)
    "power_reps_default": 100,   # = e3_sp500.FULL["power_reps"] / ["realvol_reps"]
    "size_reps": 200,            # per configuration and size design
    "size_designs": SIZE_DESIGNS,
    "factor_burn_in": 1000,      # E1's stationary start of the factor panel
    "split_power_reps": 25,
    "split_test_kw": {},
    "n_boot": pipeline.N_BOOT,
    "model_kw_override": {},
    "sim_n": 500, "sim_reps": 50, "sim_base": {}, "n_perm_small": 200,
    "blocks": {"grid": 1, "gridsize": 10, "power": 1, "split": 1, "simpower": 10},
}
QUICK: dict[str, Any] = {
    "n_tickers": 8,
    "n_seeds": 2,
    "critic_base": {"n_steps": 30, "n_perm": 100},
    "steps_scale": 0.02,
    "power_reps": 1,
    "power_reps_default": 2,     # > power_reps, as in the full run (exercises the paired rates)
    "size_reps": 2,
    "size_designs": SIZE_DESIGNS,
    "factor_burn_in": 250,
    "split_power_reps": 2,
    "split_test_kw": {"lrcritic": {"n_steps": 30, "n_perm": 100},
                      "c2st": {"n_steps": 30, "n_perm": 100},
                      "mmd": {"n_perm": 100}, "sigmmd": {"n_perm": 100}},
    "n_boot": 100,
    "model_kw_override": MODEL_KW_QUICK_OVERRIDE,
    "sim_n": 100, "sim_reps": 3, "sim_base": {"n_perm": 200}, "n_perm_small": 50,
    "blocks": {"grid": 1, "gridsize": 2, "power": 1, "split": 1, "simpower": 3},
}


def config_kw(g: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Critic overrides of grid configuration ``g`` (quick mode rescales the steps)."""
    kw = dict(cfg["critic_base"])
    for k, v in g["kw"].items():
        if k == "n_steps":
            v = max(1, int(round(v * cfg["steps_scale"])))
        kw[k] = v
    return kw


def model_kw(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Model options: E3's ``MODEL_KW`` (+ the quick-mode overrides)."""
    return {m: {**e3.MODEL_KW[m], **cfg["model_kw_override"].get(m, {})} for m in MODELS}


def synth_seed(s: int) -> int:
    """Synthetic seed index ``s`` -> E3's seed (shared by the models and the configurations)."""
    return e3.synthetic_seed(int(s))


# --------------------------------------------------------------------------------------------
# Code and alignment hashes
# --------------------------------------------------------------------------------------------
LIB_PATTERNS = ("config.py", "data.py", "pipeline.py", "utils.py", "models/*.py",
                "twosample/*.py")


@functools.lru_cache(maxsize=None)
def lib_code_hash() -> str:
    """Hash of the ``synthval`` sources E5 runs (fixed for the life of the process)."""
    root = Path(synthval.__file__).resolve().parent
    files = sorted({f for pat in LIB_PATTERNS for f in root.glob(pat)})
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(root).as_posix().encode() + b"\0"
                 + utils.file_sha256(f).encode() + b"\n")
    return h.hexdigest()[:16]


def source_hash(*objs: Any) -> str:
    return hashlib.sha256("".join(inspect.getsource(o) for o in objs).encode()).hexdigest()[:16]


def e3_key(kind: str) -> dict[str, Any]:
    """E3 root seed and the source hash of the E3 functions a part calls (checkpoint key)."""
    fns = {"grid": (e3.synthetic_seed, e3.seed_int),
           "D1": (e3.power_rep_rows, e3._anchored_row, e3.seed_int),
           "D2": (e3.realvol_rep_rows, e3.realvol_inputs, e3._anchored_row)}[kind]
    return {"e3_root_seed": e3.ROOT_SEED, "e3_source": source_hash(*fns)}


def check_e3_alignment() -> dict[str, bool]:
    """Log whether E5's full settings still match E3's (seeds per replication, reps, options)."""
    checks = {
        "n_seeds": FULL["n_seeds"] == e3.FULL["n_seeds"],
        "power_reps_default": FULL["power_reps_default"] == e3.FULL["power_reps"]
        == e3.FULL["realvol_reps"],
        "n_boot": FULL["n_boot"] == e3.FULL["n_boot"],
        "default critic settings": (FULL["critic_base"] == {} and e3.FULL["base_kw"] == {}
                                    and e3.FULL["test_kw"] == {}),
        "designs": DESIGNS == ("D1", "D2"),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        logger.warning("E5's full settings differ from e3_sp500 (%s): the default configuration "
                       "will not reproduce E3's verdict", ", ".join(bad))
    else:
        logger.info("E5 full settings match e3_sp500 (n_seeds %d, default power reps %d)",
                    FULL["n_seeds"], FULL["power_reps_default"])
    return checks


# --------------------------------------------------------------------------------------------
# Panels and fitted models (cached per process)
# --------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def tickers_of(n_tickers: int | None) -> tuple[str, ...]:
    return tuple(data.panel_tickers() if n_tickers is None else data.panel_tickers(n_tickers))


@functools.lru_cache(maxsize=None)
def _base_panel(data_split: str, n_tickers: int | None) -> pipeline.RealPanel:
    tickers = list(tickers_of(n_tickers))
    if data_split == "main":
        return pipeline.load_real_panel(tickers)
    train_p, test_p = SPLITS[data_split]
    rets = data.log_returns(data.clean_prices(data.load_prices(), tickers))
    train, test = data.split(rets, train_p), data.split(rets, test_p)
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError(f"missing returns in the {data_split} split")
    return pipeline.RealPanel(train=train, test=test, scale=data.train_vol(train))


def get_panel(data_split: str, n_tickers: int | None, L: int) -> pipeline.RealPanel:
    """The panel of a split with window length ``L`` (non-overlapping windows)."""
    return dataclasses.replace(_base_panel(data_split, n_tickers), L=int(L))


@functools.lru_cache(maxsize=None)
def _load_fits(path: str) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def fit_spec_hash(data_split: str, model: str, kw: dict, n_tickers: int | None) -> str:
    """Hash of what determines a panel fit (fits are deterministic given it)."""
    return common.config_hash({"model": model, "kw": kw, "split": SPLITS[data_split],
                               "tickers": list(tickers_of(n_tickers))})


def e3_fit_files(model: str, n_tickers: int | None) -> tuple[Path, Path]:
    """Where ``e3_sp500.ensure_fits`` stores the main-split fits (pickle, fit table)."""
    cfg = e3.fit_config(model, list(tickers_of(n_tickers)))
    d = Path(config.RESULTS_DIR) / "e3" / "checkpoints" / f"fit_{model}" / common.config_hash(cfg)
    return d / "fitted.pkl", d / "fit_table.csv"


#: Fit-table CSV of each fit pickle used in this process; pickles fitted in this process.
FIT_TABLES: dict[str, str] = {}
_FITTED_THIS_RUN: set[str] = set()


def _atomic_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def ensure_fits(ctx: common.ExpContext, data_split: str, models: tuple[str, ...],
                cfg: dict[str, Any], n_tickers: int | None) -> dict[str, str]:
    """Fit (or load) each model on the split's TRAIN returns; returns pickle paths.

    Main split of a full run: E3's pickle when it exists (read only; E3 owns it), so the default
    configuration uses exactly E3's fitted models. Otherwise E5's own pickle
    ``results/<exp_id>/fits/<split>_<model>_<spec hash>.pkl`` (with its fit table ``.csv``),
    fitted with ``pipeline.fit_panel`` when missing; ``--fresh`` refits it once per process.
    """
    mkw = model_kw(cfg)
    out: dict[str, str] = {}
    for m in models:
        kw = mkw[m]
        if data_split == "main" and not ctx.quick and kw == e3.MODEL_KW[m]:
            e3_pkl, e3_csv = e3_fit_files(m, n_tickers)
            if e3_pkl.exists() and e3_csv.exists():
                logger.info("using E3's fitted %s models: %s", m, e3_pkl)
                out[m], FIT_TABLES[str(e3_pkl)] = str(e3_pkl), str(e3_csv)
                continue
        path = ctx.results_dir / "fits" / (
            f"{data_split}_{m}_{fit_spec_hash(data_split, m, kw, n_tickers)}.pkl")
        csv = path.with_suffix(".csv")
        out[m], FIT_TABLES[str(path)] = str(path), str(csv)
        refit = ctx.fresh and not ctx.plots_only and str(path) not in _FITTED_THIS_RUN
        if path.exists() and csv.exists() and not refit:
            continue
        panel = _base_panel(data_split, n_tickers)
        t0 = time.perf_counter()
        # Only Heston (about 13 s per ticker) and Merton (0.1 s) are worth a process pool.
        heavy = m == "Heston" or (m == "Merton" and n_tickers is None)
        fitted, table = pipeline.fit_panel(m, panel.train, n_workers=ctx.workers if heavy else 1,
                                           **kw)
        logger.info("fit %s on %s TRAIN (%d tickers): %s", m, data_split, len(fitted),
                    common.fmt_seconds(time.perf_counter() - t0))
        csv.unlink(missing_ok=True)  # table last: a crash in between leaves no table => refit
        _atomic_pickle(fitted, path)
        table.to_csv(csv)
        _FITTED_THIS_RUN.add(str(path))
        _load_fits.cache_clear()
    return out


def fit_keys(paths: dict[str, str]) -> dict[str, str]:
    """Per model ``<pickle name>:<fitted-parameter hash>`` (downstream checkpoint keys): a refit
    with different parameters never reuses blocks computed from other fits."""
    return {m: f"{Path(p).parent.name}/{Path(p).name}:"
               f"{e3.fit_content_hash(Path(FIT_TABLES[p]))}" for m, p in paths.items()}


def ensure_factor_fit(ctx: common.ExpContext, n_tickers: int | None) -> tuple[str, str]:
    """Pickle path and parameter hash of E1's ``FactorGARCHt`` fitted on the main TRAIN panel."""
    spec = {"model": "FactorGARCHt", "split": SPLITS["main"],
            "tickers": list(tickers_of(n_tickers)), "code": lib_code_hash()}
    path = ctx.results_dir / "fits" / f"main_FactorGARCHt_{common.config_hash(spec)}.pkl"
    refit = ctx.fresh and not ctx.plots_only and str(path) not in _FITTED_THIS_RUN
    if not path.exists() or refit:
        panel = _base_panel("main", n_tickers)
        t0 = time.perf_counter()
        model = get_mv_model("FactorGARCHt").fit(panel.train.to_numpy(dtype=np.float64))
        logger.info("fit FactorGARCHt on main TRAIN (%d tickers): %s", panel.n_tickers,
                    common.fmt_seconds(time.perf_counter() - t0))
        _atomic_pickle(model, path)
        _FITTED_THIS_RUN.add(str(path))
        _load_fits.cache_clear()
    try:
        content = common.config_hash(_load_fits(str(path)).params())
    except Exception:  # noqa: BLE001 - params() of a multivariate model may not be JSON-able
        content = common.config_hash(spec)
    return str(path), content


def _rel(p: str) -> str:
    try:
        return str(Path(p).resolve().relative_to(config.ROOT))
    except ValueError:
        return str(p)


def fit_summary(paths: dict[str, str], data_split: str) -> pd.DataFrame:
    """One row per model: tickers, tickers with a parameter at a bound, fit seconds."""
    rows = []
    for m, p in paths.items():
        t = pd.read_csv(FIT_TABLES.get(p, str(Path(p).with_suffix(".csv"))))
        row = {"split": data_split, "model": m, "fit_file": _rel(p), "n_tickers": len(t),
               "n_any_at_bound": int((t.get("n_at_bound", pd.Series(0, index=t.index)) > 0).sum()),
               "fit_seconds_median": float(t["fit_seconds"].median()),
               "fit_seconds_total": float(t["fit_seconds"].sum())}
        if "at_bound" in t:
            counts: dict[str, int] = {}
            for s in t["at_bound"].dropna().astype(str):
                for name in filter(None, s.split(";")):
                    counts[name] = counts.get(name, 0) + 1
            row["at_bound_counts"] = ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Workers (module level: picklable for spawned processes)
# --------------------------------------------------------------------------------------------
def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _eval_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Synthetic seed ``rep`` of one model: ``pipeline.evaluate_model`` with the cell's settings.

    The seed is E3's ``synthetic_seed(rep)`` (shared by every model and configuration); each
    test's stream is keyed by its name, so with the default settings the critic and anchored rows
    are E3's pooled rows at that seed.
    """
    del rng
    panel = get_panel(cell["data_split"], cell["_n_tickers"], cell["L"])
    fitted = _load_fits(cell["_fits"])
    tests, methods = tuple(cell["_tests"]), tuple(cell["_methods"])
    kw = dict(cell["_kw"])
    test_kw = {t: dict(v) for t, v in cell["_test_kw"].items()}
    test_kw["critic"] = {**test_kw.get("critic", {}), **kw}
    res, dt, err = common.safe_test(
        pipeline.evaluate_model, panel, fitted, seed=synth_seed(rep), tests=tests,
        relative_methods=methods, test_kw=test_kw, base_kw=kw, n_boot=int(cell["_n_boot"]),
        keep_extra=True)
    if res is None:
        names = list(tests) + [f"relative_{m}" for m in methods]
        return [{"test": n, "kind": "relative" if n.startswith("relative_") else "absolute",
                 "pvalue": NAN, "statistic": NAN, "pvalue_type": "", "error": err}
                for n in names]
    rss = _max_rss_mb()
    for r in res:
        ex = r.pop("extra", None) or {}
        r.update(e3._train_steps(ex))
        if r["kind"] == "relative":
            # relative_tests does not pass these on; take them from the result's extra
            r.update(n_eval_past=ex.get("n_eval_past", NAN), n_eval_g=ex.get("n_eval_g", NAN),
                     split=str(ex.get("split", "")), hac_lags=str(ex.get("hac_lags", "")))
        r["seed"] = str(r["seed"])
        r["max_rss_mb"] = rss
        r["error"] = ""
    return res


def _design_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """One replication of power design D1 or D2: E3's own function with the cell's settings."""
    del rng  # E3's streams, keyed by the replication only (common to every configuration)
    panel = get_panel(cell["data_split"], cell["_n_tickers"], cell["L"])
    if cell["design"] == "D1":
        rows = e3.power_rep_rows(panel, _load_fits(cell["_fits"]), rep, cell["_c"])
    else:
        rows = e3.realvol_rep_rows(panel, rep, cell["_c"])
    rss = _max_rss_mb()
    for r in rows:
        r["max_rss_mb"] = rss
    return rows


def _size_rep(cell: dict, rep: int, rng: np.random.Generator) -> dict[str, Any]:
    """One size replication: absolute critic test on two independent panels from one law."""
    del rng  # X*, G* and the test stream are keyed by (design, rep): shared by every config
    design = cell["design"]
    panel = get_panel("main", cell["_n_tickers"], cell["L"])

    def key(what: str) -> np.random.SeedSequence:
        return common.keyed_seed_sequence(ROOT_SEED, "gridsize", design, what, int(rep))

    model = _load_fits(cell["_fits"])
    t0 = time.perf_counter()
    if design == "garch":   # E3's synthetic paths: per-ticker GARCH-t from the end of TRAIN
        X = pipeline.simulate_panel(model, panel.n_test, seed=key("X"), start_from_end=True)
        G = pipeline.simulate_panel(model, panel.n_test, seed=key("G"), start_from_end=True)
    else:                   # E1's factor panel, stationary start
        burn = int(cell["_burn"])
        X = pipeline.simulate_panel(model, panel.n_test, seed=key("X"), start_from_end=False,
                                    burn_in=burn)
        G = pipeline.simulate_panel(model, panel.n_test, seed=key("G"), start_from_end=False,
                                    burn_in=burn)
    sim_s = time.perf_counter() - t0
    Wx, _, ox = panel.synthetic_windows(X, "test")
    Wg, _, og = panel.synthetic_windows(G, "test")
    res, dt, err = common.safe_test(
        pipeline.absolute_tests, Wx, Wg, order_x=ox, order_y=og, tests=("critic",),
        protocol="dependent", rng=np.random.default_rng(key("test")),
        test_kw={"critic": dict(cell["_kw"])})
    if res is None:
        return {"pvalue": NAN, "statistic": NAN, "pvalue_type": "hac", "sim_seconds": sim_s,
                "error": err}
    r = res[0]
    return {"pvalue": r["pvalue"], "pvalue_type": r["pvalue_type"], "statistic": r["statistic"],
            "p_perm": r["p_perm"], "p_hac": r["p_hac"], "z_hac": r["z_hac"],
            "hac_lags": r["hac_lags"], "n_eval_x": r["n_eval_x"], "n_eval_y": r["n_eval_y"],
            "split": r["split"], "test_seconds": r["runtime_s"], "sim_seconds": sim_s,
            "max_rss_mb": _max_rss_mb(), "error": ""}


def _e2() -> Any:
    """``experiments/e2_ground_truth`` (imported lazily: only simpower needs it)."""
    import e2_ground_truth

    return e2_ground_truth


def sim_specs(family: str) -> tuple[dict, dict]:
    """(P, Q) specs of the E2 family ``family`` at the level in ``SIM_FAMILIES``."""
    F = _e2().FAMILIES[family]
    k = [round(float(v), 10) for v in F["levels"]].index(round(SIM_FAMILIES[family], 10))
    return F["P"], F["Q"][k]


def _simpower_rep(cell: dict, rep: int, rng: np.random.Generator) -> dict[str, Any]:
    """One replication: critic test (iid protocol) on X ~ P, Y ~ Q, both n windows."""
    del rng  # X, Y and the test stream are keyed by (family, n, rep): shared by all cells
    e2 = _e2()
    fam, n = cell["family"], int(cell["_n"])
    P, Q = sim_specs(fam)
    X = e2.sim_windows(P, n, common.keyed_rng(ROOT_SEED, "simpower_X", fam, n, rep))
    Y = e2.sim_windows(Q, n, common.keyed_rng(ROOT_SEED, "simpower_Y", fam, n, rep))
    test = pipeline.make_test("critic", "iid", **dict(cell["_kw"]))
    res, dt, err = common.safe_test(test, X, Y,
                                    rng=common.keyed_rng(ROOT_SEED, "simpower_test", fam, n, rep))
    ptype = "permutation" if test.n_perm > 0 else "normal"
    if res is None:
        return {"pvalue": NAN, "statistic": NAN, "pvalue_type": ptype, "n_perm": test.n_perm,
                "seconds": dt, "error": err}
    return {"pvalue": float(res.pvalue), "statistic": float(res.statistic), "pvalue_type": ptype,
            "n_perm": test.n_perm, "z": res.extra.get("z", NAN),
            "steps_run": res.extra.get("steps_run", NAN), "seconds": dt, "error": ""}


# --------------------------------------------------------------------------------------------
# Part runners
# --------------------------------------------------------------------------------------------
def _cost(g: dict[str, Any]) -> float:
    return COST.get(g["config"], 1.0)


def selected_configs(ctx: common.ExpContext) -> list[str]:
    """Configurations selected with ``--configs`` (default: all of ``GRID``; ``light`` = all but
    ``HEAVY``, ``heavy`` = ``HEAVY``)."""
    raw = getattr(ctx.args, "configs", None)
    if not raw:
        return [g["config"] for g in GRID]
    names: list[str] = []
    for s in (s.strip() for s in str(raw).split(",") if s.strip()):
        if s == "light":
            names += [g["config"] for g in GRID if g["config"] not in HEAVY]
        elif s == "heavy":
            names += list(HEAVY)
        else:
            names.append(s)
    bad = [n for n in names if n not in GRID_BY_NAME]
    if bad:
        raise SystemExit(f"unknown --configs {bad}; choose from {list(GRID_BY_NAME)}")
    return names


def part_workers(ctx: common.ExpContext, part: str) -> int:
    """Workers that fit in memory: ``min(workers, MAX_WORKERS, 0.85 MemAvailable / GB)``."""
    gb = WORKER_GB.get(part, DEFAULT_WORKER_GB)
    avail = e3.mem_available_gb()
    cap = int(0.85 * avail // gb) if np.isfinite(avail) else ctx.workers
    w = max(1, min(int(ctx.workers), MAX_WORKERS.get(part, int(ctx.workers)), cap))
    if w < ctx.workers:
        logger.warning("%s: %d workers instead of %d (about %.1f GB per worker, %.1f GB "
                       "available%s)", part, w, ctx.workers, gb, avail,
                       f", cap {MAX_WORKERS[part]}" if part in MAX_WORKERS else "")
    return w


def _blocks(n_reps: int, block_size: int) -> list[tuple[int, int]]:
    return [(s, min(n_reps, s + block_size)) for s in range(0, n_reps, block_size)]


def run_part(fn: Any, cells: list[dict], *, part: str, ctx: common.ExpContext, n_reps: Any,
             config: dict[str, Any], block_size: int, workers: int,
             filter_configs: bool = False) -> pd.DataFrame:
    """``common.run_cells`` on the selected cells, plus every complete unselected cell read back
    from the same checkpoint directory (same settings, so the CSV covers all finished cells)."""
    sel_names = set(selected_configs(ctx)) if filter_configs else None
    sel = [c for c in cells if sel_names is None or c["config"] in sel_names]
    rest = [c for c in cells if sel_names is not None and c["config"] not in sel_names]
    frames: list[pd.DataFrame] = []
    if sel:
        frames.append(common.run_cells(fn, sel, part=part, ctx=ctx, root_seed=ROOT_SEED,
                                       n_reps=n_reps, config=config, block_size=block_size,
                                       cost=lambda c: c.get("_cost", 1.0), workers=workers))
    if rest:
        ck = common.Checkpoint(ctx.ckpt_dir, part, {**dict(config), "block_size": int(block_size)})
        n_old = 0
        for c in rest:
            n = int(n_reps(c) if callable(n_reps) else n_reps)
            paths = [ck.block_path(c["cell"], s, e) for s, e in _blocks(n, block_size)]
            if all(p.exists() for p in paths):
                frames += [pd.read_csv(p) for p in paths]
                n_old += 1
        logger.info("[%s] %d of %d unselected cells read back from %s", part, n_old, len(rest),
                    ck.dir)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def part_config(cfg: dict, part: str, **extra: Any) -> dict[str, Any]:
    """What determines a part's replication rows (hashed into the checkpoint directory).

    Replication counts are left out on purpose (a replication's rows do not depend on them).
    """
    return {"root_seed": ROOT_SEED, "part": part, "code": lib_code_hash(),
            "pipeline": pipeline.pipeline_config(), "n_tickers": cfg["n_tickers"],
            "tickers": list(tickers_of(cfg["n_tickers"])), "critic_base": cfg["critic_base"],
            "steps_scale": cfg["steps_scale"], "grid": GRID, **extra}


def _eval_cell(name: str, g: dict, model: str, data_split: str, cfg: dict, fits: str, *,
               tests: tuple[str, ...], methods: tuple[str, ...], test_kw: dict,
               cost: float) -> dict[str, Any]:
    return {"cell": name, "config": g["config"], "factor": g["factor"], "setting": g["setting"],
            "model": model, "L": int(g["L"]), "data_split": data_split,
            "_n_tickers": cfg["n_tickers"], "_fits": fits, "_kw": config_kw(g, cfg),
            "_tests": list(tests), "_methods": list(methods), "_test_kw": test_kw,
            "_n_boot": cfg["n_boot"], "_cost": cost}


def _design_cell(name: str, g: dict, design: str, data_split: str, cfg: dict,
                 fits: str | None, *, n_seeds: int | None = None) -> dict[str, Any]:
    return {"cell": name, "config": g["config"], "factor": g["factor"], "setting": g["setting"],
            "design": design, "L": int(g["L"]), "data_split": data_split,
            "_n_tickers": cfg["n_tickers"], "_fits": fits,
            "_c": {"base_kw": config_kw(g, cfg), "n_boot": cfg["n_boot"],
                   "n_seeds": int(cfg["n_seeds"] if n_seeds is None else n_seeds)},
            "_cost": _cost(g)}


def _size_cell(g: dict, design: str, cfg: dict, fits: str) -> dict[str, Any]:
    return {"cell": f"size__{g['config']}__{design}", "config": g["config"],
            "factor": g["factor"], "setting": g["setting"], "design": design, "L": int(g["L"]),
            "_n_tickers": cfg["n_tickers"], "_fits": fits, "_kw": config_kw(g, cfg),
            "_burn": cfg["factor_burn_in"], "_cost": _cost(g)}


def run_grid(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    t0 = time.perf_counter()
    fits = ensure_fits(ctx, "main", MODELS, cfg, cfg["n_tickers"])
    t_fit = time.perf_counter() - t0
    ctx.save_csv(fit_summary(fits, "main"), "e5_fits_main", config={"model_kw": model_kw(cfg)})
    cells = [_eval_cell(f"{g['config']}__{m}", g, m, "main", cfg, fits[m], tests=("critic",),
                        methods=("anchored",), test_kw={}, cost=_cost(g))
             for g in GRID for m in MODELS]
    pc = part_config(cfg, "grid", model_kw=model_kw(cfg), fits=fit_keys(fits),
                     n_boot=cfg["n_boot"], **e3_key("grid"))
    df = run_part(_eval_rep, cells, part="grid", ctx=ctx, n_reps=cfg["n_seeds"], config=pc,
                  block_size=cfg["blocks"]["grid"], workers=part_workers(ctx, "grid"),
                  filter_configs=True)
    df = df.rename(columns={"rep": "seed_index"})
    df = df[df["seed_index"] < cfg["n_seeds"]] if not df.empty else df
    t_all = time.perf_counter() - t0
    ctx.save_csv(df, "e5_grid_raw", config={**pc, "n_seeds": cfg["n_seeds"]},
                 timing={"grid_fit_s": t_fit, "grid_s": t_all})
    ctx.write_part_meta("grid", {**pc, "n_seeds": cfg["n_seeds"]},
                        {"fit_s": t_fit, "total_s": t_all})
    return {"grid_s": t_all}


def run_gridsize(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    t0 = time.perf_counter()
    g_fit = ensure_fits(ctx, "main", ("GARCHt",), cfg, cfg["n_tickers"])
    fits: dict[str, str] = {}
    keys: dict[str, str] = {}
    if "garch" in cfg["size_designs"]:
        fits["garch"], keys["garch"] = g_fit["GARCHt"], fit_keys(g_fit)["GARCHt"]
    if "factor" in cfg["size_designs"]:
        fits["factor"], keys["factor"] = ensure_factor_fit(ctx, cfg["n_tickers"])
    cells = [_size_cell(g, d, cfg, fits[d]) for g in GRID for d in cfg["size_designs"]]
    pc = part_config(cfg, "gridsize", designs=list(cfg["size_designs"]), fits=keys,
                     factor_burn_in=cfg["factor_burn_in"],
                     streams="e5 keyed (gridsize, design, X|G|test, rep)")
    df = run_part(_size_rep, cells, part="gridsize", ctx=ctx, n_reps=cfg["size_reps"],
                  config=pc, block_size=cfg["blocks"]["gridsize"],
                  workers=part_workers(ctx, "gridsize"), filter_configs=True)
    df = df[df["rep"] < cfg["size_reps"]] if not df.empty else df
    t_all = time.perf_counter() - t0
    ctx.save_csv(df, "e5_gridsize_raw", config={**pc, "reps": cfg["size_reps"]},
                 timing={"gridsize_s": t_all})
    ctx.write_part_meta("gridsize", {**pc, "reps": cfg["size_reps"]}, {"total_s": t_all})
    return {"gridsize_s": t_all}


def _design_reps(cfg: dict) -> Any:
    def n(cell: dict) -> int:
        return int(cfg["power_reps_default"] if cell["config"] == "default"
                   else cfg["power_reps"])
    return n


def run_design_part(ctx: common.ExpContext, cfg: dict, design: str) -> dict[str, float]:
    """gridpower (D1) or gridrealvol (D2): E3's design under every configuration."""
    part = DESIGN_PART[design]
    t0 = time.perf_counter()
    fit = keys = None
    if design == "D1":
        f = ensure_fits(ctx, "main", (pipeline.POWER_MODEL,), cfg, cfg["n_tickers"])
        fit, keys = f[pipeline.POWER_MODEL], fit_keys(f)
    cells = [_design_cell(f"{design}__{g['config']}", g, design, "main", cfg, fit) for g in GRID]
    pc = part_config(cfg, part, design=design, n_seeds=cfg["n_seeds"], n_boot=cfg["n_boot"],
                     vol_factor=pipeline.POWER_VOL_FACTOR, fits=keys, **e3_key(design))
    nr = _design_reps(cfg)
    df = run_part(_design_rep, cells, part=part, ctx=ctx, n_reps=nr, config=pc,
                  block_size=cfg["blocks"]["power"], workers=part_workers(ctx, part),
                  filter_configs=True)
    if not df.empty:
        df = df[df["rep"] < df["config"].map(lambda c: nr({"config": c}))]
    t_all = time.perf_counter() - t0
    reps = {"power_reps": cfg["power_reps"], "power_reps_default": cfg["power_reps_default"]}
    ctx.save_csv(df, f"e5_{part}_raw", config={**pc, **reps}, timing={f"{part}_s": t_all})
    ctx.write_part_meta(part, {**pc, **reps}, {"total_s": t_all})
    return {f"{part}_s": t_all}


def run_gridpower(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    return run_design_part(ctx, cfg, "D1")


def run_gridrealvol(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    return run_design_part(ctx, cfg, "D2")


def run_split(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    t0 = time.perf_counter()
    fits = ensure_fits(ctx, "alt", MODELS, cfg, cfg["n_tickers"])
    t_fit = time.perf_counter() - t0
    ctx.save_csv(fit_summary(fits, "alt"), "e5_fits_alt", config={"model_kw": model_kw(cfg)})
    g0 = GRID_BY_NAME["default"]
    cells = [_eval_cell(f"split__{m}", g0, m, "alt", cfg, fits[m],
                        tests=pipeline.ABSOLUTE_TESTS, methods=pipeline.RELATIVE_METHODS,
                        test_kw=cfg["split_test_kw"], cost=3.0)
             for m in MODELS]
    pc = part_config(cfg, "split", model_kw=model_kw(cfg), fits=fit_keys(fits),
                     split_test_kw=cfg["split_test_kw"], n_boot=cfg["n_boot"],
                     alt_split=SPLITS["alt"], **e3_key("grid"))
    df = run_part(_eval_rep, cells, part="split", ctx=ctx, n_reps=cfg["n_seeds"], config=pc,
                  block_size=cfg["blocks"]["split"], workers=part_workers(ctx, "split"))
    df = df.rename(columns={"rep": "seed_index"})
    ctx.save_csv(df, "e5_split_raw", config={**pc, "n_seeds": cfg["n_seeds"]})
    gk = {pipeline.POWER_MODEL: fits[pipeline.POWER_MODEL]}
    dcells = [_design_cell(f"{d}__alt", g0, d, "alt", cfg,
                           fits[pipeline.POWER_MODEL] if d == "D1" else None) for d in DESIGNS]
    ppc = part_config(cfg, "split_power", n_seeds=cfg["n_seeds"], n_boot=cfg["n_boot"],
                      vol_factor=pipeline.POWER_VOL_FACTOR, fits=fit_keys(gk),
                      alt_split=SPLITS["alt"],
                      e3={d: e3_key(d) for d in DESIGNS})
    dp = run_part(_design_rep, dcells, part="split_power", ctx=ctx,
                  n_reps=cfg["split_power_reps"], config=ppc, block_size=cfg["blocks"]["power"],
                  workers=part_workers(ctx, "split_power"))
    t_all = time.perf_counter() - t0
    ctx.save_csv(dp, "e5_split_power_raw", config={**ppc, "reps": cfg["split_power_reps"]},
                 timing={"split_fit_s": t_fit, "split_s": t_all})
    ctx.write_part_meta("split", {"eval": {**pc, "n_seeds": cfg["n_seeds"]},
                                  "power": {**ppc, "reps": cfg["split_power_reps"]}},
                        {"fit_s": t_fit, "total_s": t_all})
    return {"split_s": t_all}


def _sim_cells(cfg: dict) -> list[dict[str, Any]]:
    cells = []

    def add(fam: str, lr: float, steps: int, n_perm: int | None) -> None:
        kw = {**cfg["sim_base"], "lr": lr,
              "n_steps": max(1, int(round(steps * cfg["steps_scale"])))}
        key = {"fam": fam, "lr": lr, "steps": steps}
        if n_perm is not None:
            kw["n_perm"] = int(n_perm)
            key["nperm"] = int(n_perm)
        cells.append({"cell": common.cell_key(**key), "family": fam,
                      "level": SIM_FAMILIES[fam], "lr": lr, "n_steps": steps,
                      "perm_cell": n_perm is not None, "n": cfg["sim_n"], "_n": cfg["sim_n"],
                      "_kw": kw, "_cost": steps / 1000.0})

    for fam in SIM_FAMILIES:
        for lr in SIM_LR:
            for steps in SIM_STEPS:
                add(fam, lr, steps, None)
        add(fam, SIM_DEFAULT[0], SIM_DEFAULT[1], cfg["n_perm_small"])
    return cells


def run_simpower(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    t0 = time.perf_counter()
    cells = _sim_cells(cfg)
    e2 = _e2()
    pc = {"root_seed": ROOT_SEED, "part": "simpower", "code": lib_code_hash(),
          "e2_source": source_hash(e2.sim_windows), "sim_n": cfg["sim_n"],
          "sim_base": cfg["sim_base"], "steps_scale": cfg["steps_scale"],
          "n_perm_small": cfg["n_perm_small"],
          "families": {f: list(sim_specs(f)) for f in SIM_FAMILIES}, "lr": list(SIM_LR),
          "steps": list(SIM_STEPS), "L": e2.L, "burn": e2.BURN, "critic": pipeline.CRITIC_KW}
    df = run_part(_simpower_rep, cells, part="simpower", ctx=ctx, n_reps=cfg["sim_reps"],
                  config=pc, block_size=cfg["blocks"]["simpower"],
                  workers=part_workers(ctx, "simpower"))
    t_all = time.perf_counter() - t0
    ctx.save_csv(df, "e5_simpower_raw", config={**pc, "reps": cfg["sim_reps"]},
                 timing={"simpower_s": t_all})
    ctx.write_part_meta("simpower", {**pc, "reps": cfg["sim_reps"]}, {"total_s": t_all})
    return {"simpower_s": t_all}


# --------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------
def combine_seeds(raw: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group: combined p-value ``min(1, 2 median)`` and per-seed rejections (Wilson CI)."""
    rows = []
    for key, g in raw.groupby(by, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        p = g["pvalue"].to_numpy(dtype=float)
        ok = np.isfinite(p)
        n = int(ok.sum())
        k = int(np.sum(p[ok] <= ALPHA))
        lo, hi = pipeline.wilson_ci(k, n) if n else (NAN, NAN)
        ptypes = sorted({str(x) for x in g["pvalue_type"].dropna() if str(x)})
        rows.append({**dict(zip(by, key)), "n_seeds": len(g), "n_valid": n,
                     "p_median": float(np.median(p[ok])) if n else NAN,
                     "p_comb": pipeline.median_pvalue(p), "n_reject": k,
                     "reject_share": k / n if n else NAN, "ci_lo": lo, "ci_hi": hi,
                     "pvalue_type": ";".join(ptypes),
                     "statistic_median": float(np.nanmedian(g["statistic"])) if n else NAN,
                     "p_seeds": ";".join(f"{x:.4g}" for x in p)})
    return pd.DataFrame(rows)


def design_summaries(raw: pd.DataFrame, by: list[str], common_reps: int) -> pd.DataFrame:
    """``e3_sp500.design_summary`` per group and design (alternative = power, boundary = size of
    the rule), plus the rule's rate over the replications ``< common_reps`` shared by every
    configuration (``*_common``, for paired comparisons with the default)."""
    out = []
    for key, g in raw.groupby(by + ["design"], sort=False):
        key = key if isinstance(key, tuple) else (key,)
        s = e3.design_summary(g, str(key[-1]))
        if s.empty:
            continue
        for col, v in reversed(list(zip(by, key[:-1]))):
            s.insert(0, col, v)
        gc = g[g["rep"] < common_reps]
        for i, case in zip(s.index, s["case"]):
            rc = e3._rule_counts(gc, case) if not gc.empty else {}
            s.loc[i, "n_reps_common"] = rc.get("n_reps", NAN)
            s.loc[i, "rate_rule_common"] = rc.get("rate_rule", NAN)
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_power_summaries(ctx: common.ExpContext, common_reps: int) -> pd.DataFrame:
    frames = []
    for d in DESIGNS:
        name = f"e5_{DESIGN_PART[d]}_raw"
        if ctx.has_csv(name):
            raw = ctx.load_csv(name)
            if not raw.empty:
                frames.append(design_summaries(raw, ["config"], common_reps))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Discordant counts (a only, b only) and the exact two-sided McNemar p-value."""
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    m = n10 + n01
    p = float(stats.binomtest(min(n10, n01), m, 0.5).pvalue) if m else 1.0
    return n10, n01, p


def size_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Per configuration and size design: rejection rate at ALPHA (Wilson CI) and the paired
    McNemar comparison with the default configuration (same X*, G* and test streams)."""
    rows = []
    for (c, d), g in raw.groupby(["config", "design"], sort=False):
        p = g.set_index("rep")["pvalue"].astype(float)
        p = p[np.isfinite(p)]
        n, k = int(p.size), int(np.sum(p <= ALPHA))
        lo, hi = pipeline.wilson_ci(k, n) if n else (NAN, NAN)
        row = {"config": c, "design": d, "size": k / n if n else NAN, "size_k": k, "size_n": n,
               "size_lo": lo, "size_hi": hi, "pvalue_type": ";".join(
                   sorted({str(x) for x in g["pvalue_type"].dropna()})),
               "hac_lags": ";".join(sorted({str(x) for x in g.get("hac_lags", pd.Series(
                   dtype=str)).dropna()}))[:60],
               "n_eval_x_median": float(np.nanmedian(g["n_eval_x"])) if "n_eval_x" in g else NAN,
               "n_error": int(g["error"].dropna().astype(str).str.len().gt(0).sum())
               if "error" in g else 0}
        d0 = raw[(raw["config"] == "default") & (raw["design"] == d)].set_index("rep")["pvalue"]
        both = p.index.intersection(d0[np.isfinite(d0.astype(float))].index)
        if c != "default" and both.size:
            a = p.loc[both].to_numpy(float) <= ALPHA
            b = d0.loc[both].to_numpy(float) <= ALPHA
            n10, n01, pm = mcnemar_exact(a, b)
            row.update(paired_n=int(both.size), only_config=n10, only_default=n01,
                       mcnemar_p=pm)
        else:
            row.update(paired_n=int(both.size), only_config=0, only_default=0, mcnemar_p=NAN)
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(ctx: common.ExpContext, cfg: dict) -> pd.DataFrame | None:
    """Per configuration: the rule's power and boundary size per design (Wilson / CP), the rule's
    power for the verdict (minimum over the designs run) and the absolute critic size per size
    design with the McNemar comparison against the default."""
    ps = load_power_summaries(ctx, int(cfg["power_reps"]))
    sz = (size_summary(ctx.load_csv("e5_gridsize_raw"))
          if ctx.has_csv("e5_gridsize_raw") else pd.DataFrame())
    if ps.empty and sz.empty:
        return None
    rows = []
    for g in GRID:
        c = g["config"]
        row: dict[str, Any] = {"config": c, "factor": g["factor"], "setting": g["setting"]}
        found = False
        for d in DESIGNS:
            for case, tag in (("alternative", "power"), ("boundary", "bd")):
                r = (ps[(ps["config"] == c) & (ps["design"] == d) & (ps["case"] == case)]
                     if not ps.empty else ps)
                if r.empty:
                    row[f"{tag}_{d}"] = NAN
                    continue
                r = r.iloc[0]
                found = True
                row.update({f"{tag}_{d}": float(r["rate_rule"]),
                            f"{tag}_{d}_k": int(r["n_reject_rule"]),
                            f"{tag}_{d}_n": int(r["n_reps"]),
                            f"{tag}_{d}_lo": float(r["rule_ci_lo"]),
                            f"{tag}_{d}_hi": float(r["rule_ci_hi"]),
                            f"{tag}_{d}_cp_lo": float(r["rule_cp_lo"]),
                            f"{tag}_{d}_cp_hi": float(r["rule_cp_hi"]),
                            f"{tag}_{d}_cp_lower": float(r["rule_cp_lower"]),
                            f"{tag}_{d}_single": float(r["rate_single"]),
                            f"{tag}_{d}_common": float(r["rate_rule_common"]),
                            f"{tag}_{d}_n_seeds": int(r["n_seeds"])})
        have = [d for d in DESIGNS if np.isfinite(row.get(f"power_{d}", NAN))]
        row["designs"] = "+".join(have)
        if have:
            dmin = min(have, key=lambda d: row[f"power_{d}"])
            row.update(power_rule=row[f"power_{dmin}"], power_rule_design=dmin,
                       power_rule_k=row[f"power_{dmin}_k"], power_rule_n=row[f"power_{dmin}_n"],
                       power_rule_cp=min(row[f"power_{d}_cp_lower"] for d in have))
        else:
            row.update(power_rule=NAN, power_rule_design="", power_rule_k=np.nan,
                       power_rule_n=np.nan, power_rule_cp=NAN)
        hb = [d for d in DESIGNS if np.isfinite(row.get(f"bd_{d}", NAN))]
        if hb:
            dmax = max(hb, key=lambda d: row[f"bd_{d}"])
            row.update(bd_rule=row[f"bd_{dmax}"], bd_rule_design=dmax)
        else:
            row.update(bd_rule=NAN, bd_rule_design="")
        for sd in SIZE_DESIGNS:
            r = sz[(sz["config"] == c) & (sz["design"] == sd)] if not sz.empty else sz
            if r.empty:
                row[f"abs_size_{sd}"] = NAN
                continue
            r = r.iloc[0]
            found = True
            row.update({f"abs_size_{sd}": float(r["size"]), f"abs_size_{sd}_k": int(r["size_k"]),
                        f"abs_size_{sd}_n": int(r["size_n"]),
                        f"abs_size_{sd}_lo": float(r["size_lo"]),
                        f"abs_size_{sd}_hi": float(r["size_hi"]),
                        f"abs_size_{sd}_only_config": int(r["only_config"]),
                        f"abs_size_{sd}_only_default": int(r["only_default"]),
                        f"abs_size_{sd}_mcnemar_p": float(r["mcnemar_p"])})
        if found:
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def _verdict(p: float, power: float, valid: bool) -> str:
    if not valid:
        return "n/a"
    if not np.isfinite(p):
        return "--"
    return pipeline.verdict(p, power)


def _reject(p: float) -> Any:
    return pd.NA if not np.isfinite(p) else bool(p <= ALPHA)


def _true(x: Any) -> bool:
    return False if pd.isna(x) else bool(x)


VERDICTS = ("BAD", "GOOD", "INCONCLUSIVE")
V_SHORT = {"BAD": "B", "GOOD": "G", "INCONCLUSIVE": "I", "n/a": "n/a", "--": "?"}


def sensitivity_table(ctx: common.ExpContext, cfg: dict,
                      cal: pd.DataFrame | None) -> pd.DataFrame:
    """Per configuration and model: combined p-values, decisions, verdict, change flags."""
    raw = ctx.load_csv("e5_grid_raw")
    agg = combine_seeds(raw, ["config", "model", "test"])
    calr = {} if cal is None else {r["config"]: r for _, r in cal.iterrows()}
    rows = []
    for g in GRID:
        cr = calr.get(g["config"], {})
        power = float(cr.get("power_rule", NAN)) if len(cr) else NAN
        power_cp = float(cr.get("power_rule_cp", NAN)) if len(cr) else NAN
        for m in MODELS:
            sub = agg[(agg["config"] == g["config"]) & (agg["model"] == m)]
            rel = sub[sub["test"] == "relative_anchored"]
            ab = sub[sub["test"] == "critic"]
            if rel.empty and ab.empty:
                continue
            r = rel.iloc[0] if not rel.empty else {}
            a = ab.iloc[0] if not ab.empty else {}
            valid = m not in REL_INVALID
            p_rel = float(r.get("p_comb", NAN)) if len(r) else NAN
            p_abs = float(a.get("p_comb", NAN)) if len(a) else NAN
            rows.append({
                "config": g["config"], "factor": g["factor"], "setting": g["setting"],
                "model": m, "L": g["L"], "rel_valid": valid, "rel_note": REL_INVALID.get(m, ""),
                "p_rel_comb": p_rel, "p_rel_seeds": r.get("p_seeds", "") if len(r) else "",
                "rel_n_reject": r.get("n_reject", np.nan) if len(r) else np.nan,
                "rel_n_valid": r.get("n_valid", np.nan) if len(r) else np.nan,
                "rel_delta_median": r.get("statistic_median", NAN) if len(r) else NAN,
                "rel_pvalue_type": r.get("pvalue_type", "") if len(r) else "",
                "rel_reject": _reject(p_rel),
                "power_rule": power, "power_rule_cp": power_cp,
                "power_rule_design": cr.get("power_rule_design", "") if len(cr) else "",
                "designs": cr.get("designs", "") if len(cr) else "",
                "verdict": _verdict(p_rel, power, valid),
                "verdict_cp": _verdict(p_rel, power_cp, valid),
                "p_abs_comb": p_abs, "p_abs_seeds": a.get("p_seeds", "") if len(a) else "",
                "abs_n_reject": a.get("n_reject", np.nan) if len(a) else np.nan,
                "abs_n_valid": a.get("n_valid", np.nan) if len(a) else np.nan,
                "abs_T_median": a.get("statistic_median", NAN) if len(a) else NAN,
                "abs_pvalue_type": a.get("pvalue_type", "") if len(a) else "",
                "abs_reject": _reject(p_abs),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ("rel_reject", "abs_reject"):
        out[col] = pd.array(out[col].tolist(), dtype="boolean")
    base = out[out["config"] == "default"].set_index("model")
    for col in ("verdict", "rel_reject", "abs_reject", "p_rel_comb", "p_abs_comb"):
        if not base.empty:
            out[f"{col}_default"] = out["model"].map(base[col])
        else:  # default configuration not run (--configs without it): no reference
            out[f"{col}_default"] = NAN if col.startswith("p_") else "--"
    for col in ("rel_reject_default", "abs_reject_default"):
        out[col] = pd.array([x if isinstance(x, (bool, np.bool_)) else pd.NA
                             for x in out[col].tolist()], dtype="boolean")
    valid = pd.array(out["rel_valid"].tolist(), dtype="boolean")
    rel_ch = out["rel_reject"] != out["rel_reject_default"]
    out["rel_decision_changed"] = rel_ch.where(valid, pd.NA)
    out["abs_decision_changed"] = out["abs_reject"] != out["abs_reject_default"]
    vd = [(v != v0) if (v in VERDICTS and v0 in VERDICTS) else pd.NA
          for v, v0 in zip(out["verdict"], out["verdict_default"])]
    out["verdict_changed"] = pd.array(vd, dtype="boolean")
    kind = []
    for rc, vc in zip(out["rel_decision_changed"], out["verdict_changed"]):
        kind.append("relative decision" if _true(rc) else "power" if _true(vc) else "")
    out["verdict_change_kind"] = kind
    return out


# --------------------------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------------------------
DAG, SECT, DDAG = r"$^\dagger$", r"$^\S$", r"$^\ddagger$"


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def fmt_rate(r: float, lo: float, hi: float, digits: int = 2) -> str:
    if not np.isfinite(r):
        return "--"
    return f"{r:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _g(row: Any, key: str) -> float:
    try:
        v = row[key]
    except (KeyError, TypeError):
        return NAN
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def write_sensitivity_tex(ctx: common.ExpContext, sens: pd.DataFrame,
                          cal: pd.DataFrame | None) -> None:
    """``e5_sensitivity.tex``: per configuration and model, 2 x median p of the anchored test
    with the verdict and of the absolute critic test; rule power k/n (minimum over designs).

    Marks: dagger = the decision (reject at 0.05) differs from the default; section = the verdict
    differs only because the power estimate crossed 0.8; double dagger = relative test invalid
    for this model (no verdict). Bold absolute p: rejection at 0.05.
    """
    cols: list[tuple[str, str]] = []
    for m in MODELS:
        cols += [(MODEL_LABELS[m], "rel."), (MODEL_LABELS[m], "abs.")]
    cols += [("Rule", "power")]
    calr = {} if cal is None else {r["config"]: r for _, r in cal.iterrows()}
    body, index = [], []
    for g in GRID:
        c = g["config"]
        sub = sens[sens["config"] == c].set_index("model")
        if sub.empty:
            continue
        cells = []
        for m in MODELS:
            if m not in sub.index:
                cells += ["--", "--"]
                continue
            r = sub.loc[m]
            if not bool(r["rel_valid"]):
                rel = fmt_p(r["p_rel_comb"]) + DDAG
            else:
                rel = f"{fmt_p(r['p_rel_comb'])} {V_SHORT.get(r['verdict'], '?')}"
                if c != "default" and _true(r["rel_decision_changed"]):
                    rel += DAG
                elif c != "default" and _true(r["verdict_changed"]):
                    rel += SECT
            ab = fmt_p(r["p_abs_comb"])
            if np.isfinite(r["p_abs_comb"]) and r["p_abs_comb"] <= ALPHA:
                ab = rf"\textbf{{{ab}}}"
            if c != "default" and _true(r["abs_decision_changed"]):
                ab += DAG
            cells += [rel, ab]
        cr = calr.get(c)
        if cr is not None and np.isfinite(_g(cr, "power_rule")):
            cells.append(f"{int(cr['power_rule_k'])}/{int(cr['power_rule_n'])}")
        else:
            cells.append("--")
        body.append(cells)
        index.append((g["factor"], g["setting"]))
    df = pd.DataFrame(body, columns=pd.MultiIndex.from_tuples(cols),
                      index=pd.MultiIndex.from_tuples(index, names=["Factor", "Setting"]))
    ctx.save_table(df, "e5_sensitivity", escape=False)


def write_calibration_tex(ctx: common.ExpContext, cal: pd.DataFrame) -> None:
    """``e5_calibration.tex``: rule power [Wilson] and boundary size [Clopper-Pearson] per design,
    single-seed absolute critic size [Wilson] per size design (asterisk: McNemar p < 0.05
    against the default), replications."""
    cols = ([("Rule power", d) for d in DESIGNS] + [("Rule bd. size", d) for d in DESIGNS]
            + [("Abs. size (1 seed)", SIZE_LABELS[s]) for s in SIZE_DESIGNS]
            + [("Reps", "power / size")])
    body, index = [], []
    for _, r in cal.iterrows():
        cells = [fmt_rate(_g(r, f"power_{d}"), _g(r, f"power_{d}_lo"), _g(r, f"power_{d}_hi"))
                 for d in DESIGNS]
        cells += [fmt_rate(_g(r, f"bd_{d}"), _g(r, f"bd_{d}_cp_lo"), _g(r, f"bd_{d}_cp_hi"))
                  for d in DESIGNS]
        for s in SIZE_DESIGNS:
            txt = fmt_rate(_g(r, f"abs_size_{s}"), _g(r, f"abs_size_{s}_lo"),
                           _g(r, f"abs_size_{s}_hi"))
            pm = _g(r, f"abs_size_{s}_mcnemar_p")
            if r["config"] != "default" and np.isfinite(pm) and pm < ALPHA:
                txt += "$^\\ast$"
            cells.append(txt)
        npw = sorted({int(_g(r, f"power_{d}_n")) for d in DESIGNS
                      if np.isfinite(_g(r, f"power_{d}_n"))})
        nsz = sorted({int(_g(r, f"abs_size_{s}_n")) for s in SIZE_DESIGNS
                      if np.isfinite(_g(r, f"abs_size_{s}_n"))})
        cells.append(f"{'/'.join(map(str, npw)) or '--'} / {'/'.join(map(str, nsz)) or '--'}")
        body.append(cells)
        index.append((r["factor"], r["setting"]))
    df = pd.DataFrame(body, columns=pd.MultiIndex.from_tuples(cols),
                      index=pd.MultiIndex.from_tuples(index, names=["Factor", "Setting"]))
    ctx.save_table(df, "e5_calibration", escape=False)


def plot_sensitivity(ctx: common.ExpContext, sens: pd.DataFrame, raw: pd.DataFrame,
                     cal: pd.DataFrame | None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plotting.setup()
    configs = [g for g in GRID if g["config"] in set(sens["config"])]
    n = len(configs)
    y_of = {g["config"]: n - 1 - i for i, g in enumerate(configs)}
    labels = [("Default" if g["config"] == "default" else f"{g['factor']}: {g['setting']}")
              for g in configs]
    ncols = 3 if cal is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(plotting.TWO_COL, 0.22 * n + 1.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.85][:ncols]})
    pmin = 1e-4
    off = {m: (j - (len(MODELS) - 1) / 2) * 0.14 for j, m in enumerate(MODELS)}
    for ax, test, title in ((axes[0], "relative_anchored", "Anchored relative test"),
                            (axes[1], "critic", "Absolute critic test")):
        for m in MODELS:
            col, mk = plotting.COLORS_MODEL[m], plotting.MARKERS_MODEL[m]
            faint = test == "relative_anchored" and m in REL_INVALID
            sub = raw[(raw["model"] == m) & (raw["test"] == test)]
            ys = sub["config"].map(y_of).to_numpy(dtype=float) + off[m]
            ax.scatter(np.clip(sub["pvalue"].to_numpy(dtype=float), pmin, 1.0), ys, s=7,
                       facecolors="none", edgecolors=col, marker=mk, linewidths=0.5,
                       alpha=0.25 if faint else 0.6)
            agg = sens[sens["model"] == m]
            pc = agg["p_rel_comb" if test != "critic" else "p_abs_comb"].to_numpy(dtype=float)
            ax.scatter(np.clip(pc, pmin, 1.0), agg["config"].map(y_of).to_numpy(dtype=float)
                       + off[m], s=16, color=col, marker=mk, linewidths=0,
                       alpha=0.35 if faint else 1.0)
        ax.axvline(ALPHA, color="0.3", ls="--", lw=0.7)
        ax.set_xscale("log")
        ax.set_xlim(pmin * 0.7, 1.3)
        ax.set_xlabel(r"$p$-value (HAC; $\leq 10^{-4}$ at left edge)")
        ax.set_title(title)
    crit = plotting.COLORS_TEST["critic"]
    if cal is not None:
        ax = axes[2]
        c = cal.set_index("config")
        specs = (("power_rule", "power_rule", +0.24, "o", "black", "black"),
                 ("bd_rule", "bd_rule", +0.08, "o", "white", "0.45"),
                 ("abs_size_garch", "abs_size_garch", -0.08, "^", crit, crit),
                 ("abs_size_factor", "abs_size_factor", -0.24, "v", "white", crit))
        for cfg_name, y in y_of.items():
            if cfg_name not in c.index:
                continue
            r = c.loc[cfg_name]
            for col, _, dy, mk, mfc, mec in specs:
                v = _g(r, col)
                if not np.isfinite(v):
                    continue
                if col == "power_rule":
                    d = r["power_rule_design"]
                    lo, hi = _g(r, f"power_{d}_lo"), _g(r, f"power_{d}_hi")
                elif col == "bd_rule":
                    d = r["bd_rule_design"]
                    lo, hi = _g(r, f"bd_{d}_cp_lo"), _g(r, f"bd_{d}_cp_hi")
                else:
                    lo, hi = _g(r, f"{col}_lo"), _g(r, f"{col}_hi")
                ax.errorbar(v, y + dy, xerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt=mk, ms=3,
                            mfc=mfc, mec=mec, color=mec, lw=0.6, capsize=1.2)
        ax.axvline(pipeline.POWER_MIN, color="black", ls=":", lw=0.7)
        ax.axvline(ALPHA, color="0.3", ls="--", lw=0.7)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("rejection rate (95% CI)")
        ax.set_title("Calibration")
    for ax in axes:
        prev = None
        for g in configs:
            if prev is not None and g["factor"] != prev:
                ax.axhline(y_of[g["config"]] + 0.5, color="0.85", lw=0.5, zorder=0)
            prev = g["factor"]
        ax.set_ylim(-0.6, n - 0.4)
    axes[0].set_yticks([n - 1 - i for i in range(n)])
    axes[0].set_yticklabels(labels)
    handles = [Line2D([], [], marker=plotting.MARKERS_MODEL[m], ls="", ms=4,
                      color=plotting.COLORS_MODEL[m], label=MODEL_LABELS[m]) for m in MODELS]
    handles += [Line2D([], [], marker="o", ls="", ms=4, color="0.4", label=r"$2\times$median"),
                Line2D([], [], marker="o", ls="", ms=3.5, mfc="none", mec="0.4",
                       label="per seed")]
    if cal is not None:
        handles += [Line2D([], [], marker="o", ls="", color="black", ms=3,
                           label=r"rule power (vol. $\times1.2$, min. over designs)"),
                    Line2D([], [], marker="o", ls="", mfc="white", color="0.45", ms=3,
                           label="rule boundary size (max.)"),
                    Line2D([], [], marker="^", ls="", color=crit, ms=3,
                           label=r"abs. size, GARCH-$t$"),
                    Line2D([], [], marker="v", ls="", mfc="white", color=crit, ms=3,
                           label="abs. size, factor")]
    # plotting.setup() turns on constrained layout: the legend goes above the axes
    fig.legend(handles=handles, loc="outside upper center", ncol=6, frameon=False,
               handletextpad=0.2, columnspacing=0.9)
    ctx.savefig(fig, "e5_sensitivity")


def build_grid_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    cal = calibration_table(ctx, cfg)
    if cal is not None:
        ctx.save_csv(cal, "e5_calibration")
        write_calibration_tex(ctx, cal)
        ps = load_power_summaries(ctx, int(cfg["power_reps"]))
        if not ps.empty:
            ctx.save_csv(ps, "e5_power_summary")
    if not ctx.has_csv("e5_grid_raw"):
        logger.info("grid outputs: e5_grid_raw.csv not found, sensitivity table skipped")
        return
    raw = ctx.load_csv("e5_grid_raw")
    if raw.empty:
        return
    sens = sensitivity_table(ctx, cfg, cal)
    ctx.save_csv(sens, "e5_sensitivity")
    write_sensitivity_tex(ctx, sens, cal)
    plot_sensitivity(ctx, sens, raw, cal)
    ch = sens[sens["config"] != "default"]
    n_rel = int(ch["rel_decision_changed"].sum())
    n_pow = int((ch["verdict_change_kind"] == "power").sum())
    n_abs = int(ch["abs_decision_changed"].sum())
    n_valid = int(ch["rel_valid"].sum())
    print(f"E5 grid vs default: {n_rel} relative-decision changes and {n_pow} power-only verdict "
          f"changes in {n_valid} (config, model) cells with a valid relative test; {n_abs} "
          f"absolute-decision changes in {len(ch)} cells", flush=True)
    calr = {} if cal is None else cal.set_index("config")
    for _, r in ch[ch["rel_decision_changed"].fillna(False) | ch["verdict_changed"].fillna(False)
                   | ch["abs_decision_changed"].fillna(False)].iterrows():
        sz = ""
        if r["config"] in getattr(calr, "index", ()):
            cr, c0 = calr.loc[r["config"]], calr.loc["default"] if "default" in calr.index else {}
            sz = "; abs size " + ", ".join(
                f"{s} {_g(c0, f'abs_size_{s}'):.2f}->{_g(cr, f'abs_size_{s}'):.2f}"
                f" (McNemar p {_g(cr, f'abs_size_{s}_mcnemar_p'):.2g})" for s in SIZE_DESIGNS)
        print(f"  changed: {r['config']:<10} {r['model']:<9} verdict {r['verdict_default']} -> "
              f"{r['verdict']} [{r['verdict_change_kind'] or '-'}] (p_rel "
              f"{r['p_rel_comb_default']:.3g} -> {r['p_rel_comb']:.3g}, power "
              f"{r['power_rule']:.2f}); abs reject {r['abs_reject_default']} -> {r['abs_reject']} "
              f"(p_abs {r['p_abs_comb']:.3g}){sz}", flush=True)


# --------------------------------------------------------------------------------------------
# Outputs: split
# --------------------------------------------------------------------------------------------
def build_split_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    if not ctx.has_csv("e5_split_raw"):
        logger.info("split outputs: e5_split_raw.csv not found, skipped")
        return
    raw = ctx.load_csv("e5_split_raw")
    agg = combine_seeds(raw, ["model", "test"])
    agg["valid"] = [e3.validity(m, t, True)[0] for m, t in zip(agg["model"], agg["test"])]
    agg["valid_note"] = [e3.validity(m, t, True)[1] for m, t in zip(agg["model"], agg["test"])]
    pw = (design_summaries(ctx.load_csv("e5_split_power_raw"), [], int(cfg["split_power_reps"]))
          if ctx.has_csv("e5_split_power_raw") else pd.DataFrame())

    def prate(df: pd.DataFrame, d: str, case: str) -> pd.Series | None:
        if df.empty:
            return None
        r = df[(df["design"] == d) & (df["case"] == case)]
        return None if r.empty else r.iloc[0]

    alt_pw = {d: prate(pw, d, "alternative") for d in DESIGNS}
    have = [d for d in DESIGNS if alt_pw[d] is not None]
    power_alt = min(float(alt_pw[d]["rate_rule"]) for d in have) if have else NAN
    verdicts = {}
    for m in MODELS:
        a = agg[(agg["model"] == m) & (agg["test"] == "relative_anchored")]
        verdicts[m] = _verdict(float(a["p_comb"].iloc[0]) if not a.empty else NAN, power_alt,
                               m not in REL_INVALID)
    agg["verdict_alt"] = agg["model"].map(verdicts)
    agg["power_rule_alt"] = power_alt
    agg["power_designs_alt"] = "+".join(have)
    cal = calibration_table(ctx, cfg)
    sens = (sensitivity_table(ctx, cfg, cal) if ctx.has_csv("e5_grid_raw") else None)
    main = (None if sens is None or sens.empty
            else sens[sens["config"] == "default"].set_index("model"))
    if main is not None:
        agg["verdict_main"] = agg["model"].map(main["verdict"])
        agg["p_rel_main"] = agg["model"].map(main["p_rel_comb"])
        agg["p_abs_main"] = agg["model"].map(main["p_abs_comb"])
    ctx.save_csv(agg, "e5_split")
    if not pw.empty:
        ctx.save_csv(pw, "e5_split_power")

    abs_cols = [("Absolute tests, alternative split", plotting.LABELS_TEST[t]
                 + ("$^\\ast$" if t == "ks" else "")) for t in pipeline.ABSOLUTE_TESTS]
    rel_cols = [("Relative tests, alternative split", "anchored"),
                ("Relative tests, alternative split", "two-critic" + DAG),
                ("Relative tests, alternative split", "verdict")]
    main_cols = [("Main split (default)", "anchored"), ("Main split (default)", "verdict")]
    cols = abs_cols + rel_cols + main_cols
    body, index = [], []
    for m in MODELS:
        sub = agg[agg["model"] == m].set_index("test")
        if sub.empty:
            continue
        inv = DDAG if m in REL_INVALID else ""
        row = []
        for t in pipeline.ABSOLUTE_TESTS:
            p = float(sub.loc[t, "p_comb"]) if t in sub.index else NAN
            s = fmt_p(p)
            row.append(rf"\textbf{{{s}}}" if np.isfinite(p) and p <= ALPHA and t != "ks" else s)
        for t in ("relative_anchored", "relative_two_critic"):
            row.append((fmt_p(float(sub.loc[t, "p_comb"])) + inv) if t in sub.index else "--")
        row.append(verdicts[m])
        if main is not None and m in main.index:
            row += [fmt_p(float(main.loc[m, "p_rel_comb"])) + inv, str(main.loc[m, "verdict"])]
        else:
            row += ["--", "--"]
        body.append(row)
        index.append(MODEL_LABELS[m])
    blank = ["" for _ in abs_cols]
    cmain = None
    if cal is not None and not cal.empty and (cal["config"] == "default").any():
        cmain = cal[cal["config"] == "default"].iloc[0]
    for case, tag, lab in (("alternative", "power", r"Rule power {d} (vol.\ $\times1.2$)"),
                           ("boundary", "bd", r"Rule bd.\ size {d}")):
        for d in DESIGNS:
            pa = prate(pw, d, case)
            if pa is None and cmain is None:
                continue
            if case == "alternative":
                alt = fmt_rate(pa["rate_rule"], pa["rule_ci_lo"], pa["rule_ci_hi"]) if pa is not None \
                    else "--"
                mn = (fmt_rate(_g(cmain, f"power_{d}"), _g(cmain, f"power_{d}_lo"),
                               _g(cmain, f"power_{d}_hi")) if cmain is not None else "--")
            else:
                alt = fmt_rate(pa["rate_rule"], pa["rule_cp_lo"], pa["rule_cp_hi"]) if pa is not None \
                    else "--"
                mn = (fmt_rate(_g(cmain, f"bd_{d}"), _g(cmain, f"bd_{d}_cp_lo"),
                               _g(cmain, f"bd_{d}_cp_hi")) if cmain is not None else "--")
            body.append(blank + [alt, "", "", mn, ""])
            index.append(lab.format(d=d))
    df = pd.DataFrame(body, columns=pd.MultiIndex.from_tuples(cols),
                      index=pd.Index(index, name="Model"))
    ctx.save_table(df, "e5_split", escape=False)


# --------------------------------------------------------------------------------------------
# Outputs: simulated power
# --------------------------------------------------------------------------------------------
def build_simpower_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    if not ctx.has_csv("e5_simpower_raw"):
        logger.info("simpower outputs: e5_simpower_raw.csv not found, skipped")
        return
    raw = ctx.load_csv("e5_simpower_raw")
    by = ["family", "lr", "n_steps", "n_perm", "pvalue_type"]
    tbl = common.rejection_table(raw, by)
    tbl["seconds_mean"] = [float(raw[(raw["family"] == r["family"]) & np.isclose(raw["lr"], r["lr"])
                                     & (raw["n_steps"] == r["n_steps"])
                                     & (raw["n_perm"] == r["n_perm"])]["seconds"].mean())
                           for _, r in tbl.iterrows()]
    # n_perm cell vs the same critic with the default permutations: p-value differences
    for fam in SIM_FAMILIES:
        a = raw[(raw["family"] == fam) & raw["perm_cell"].astype(bool)].set_index("rep")
        b = raw[(raw["family"] == fam) & ~raw["perm_cell"].astype(bool)
                & np.isclose(raw["lr"], SIM_DEFAULT[0])
                & (raw["n_steps"] == SIM_DEFAULT[1])].set_index("rep")
        both = a.index.intersection(b.index)
        if both.size:
            m = (tbl["family"] == fam) & (tbl["n_perm"] == a["n_perm"].iloc[0]) & np.isclose(
                tbl["lr"], SIM_DEFAULT[0]) & (tbl["n_steps"] == SIM_DEFAULT[1])
            tbl.loc[m, "max_abs_dstat_vs_default_nperm"] = float(np.nanmax(np.abs(
                a.loc[both, "statistic"].to_numpy(float) - b.loc[both, "statistic"].to_numpy(float))))
            tbl.loc[m, "max_abs_dp_vs_default_nperm"] = float(np.nanmax(np.abs(
                a.loc[both, "pvalue"].to_numpy(float) - b.loc[both, "pvalue"].to_numpy(float))))
    ctx.save_csv(tbl, "e5_simpower")
    n_def = int(raw.loc[~raw["perm_cell"].astype(bool), "n_perm"].mode().iloc[0])
    n_small = int(cfg["n_perm_small"])
    body, index = [], []
    for fam in SIM_FAMILIES:
        for lr in SIM_LR:
            cells = []
            for steps in SIM_STEPS:
                r = tbl[(tbl["family"] == fam) & np.isclose(tbl["lr"], lr)
                        & (tbl["n_steps"] == steps) & (tbl["n_perm"] == n_def)]
                cells.append("--" if r.empty else fmt_rate(r["rej_0.05"].iloc[0],
                                                           r["lo_0.05"].iloc[0],
                                                           r["hi_0.05"].iloc[0]))
            r = tbl[(tbl["family"] == fam) & np.isclose(tbl["lr"], lr)
                    & (tbl["n_steps"] == SIM_DEFAULT[1]) & (tbl["n_perm"] == n_small)]
            cells.append("--" if r.empty else fmt_rate(r["rej_0.05"].iloc[0], r["lo_0.05"].iloc[0],
                                                       r["hi_0.05"].iloc[0]))
            body.append(cells)
            lab = r"$10^{-4}$" if lr == 1e-4 else r"$10^{-3}$ (default)"
            index.append((SIM_LABELS[fam], lab))
    cols = [("Critic steps", f"{s}" + (" (default)" if s == SIM_DEFAULT[1] else ""))
            for s in SIM_STEPS] + [("Critic steps", f"{SIM_DEFAULT[1]}, {n_small} perm.")]
    df = pd.DataFrame(body, columns=pd.MultiIndex.from_tuples(cols),
                      index=pd.MultiIndex.from_tuples(index, names=["Alternative", "lr"]))
    ctx.save_table(df, "e5_simpower", escape=False)


# --------------------------------------------------------------------------------------------
# Alignment with E3's results
# --------------------------------------------------------------------------------------------
def compare_with_e3(ctx: common.ExpContext) -> None:
    """Full mode: compare the default configuration's rows with E3's CSVs (same seeds, streams,
    fits and settings, so they must be identical): grid vs ``e3_pooled_raw`` (critic, anchored),
    gridpower vs ``e3_power_raw``, gridrealvol vs ``e3_realvol_raw``."""
    if ctx.quick:
        return
    e3dir = Path(config.RESULTS_DIR) / "e3"
    specs = (("grid", "e5_grid_raw", "e3_pooled_raw", ["model", "seed_index", "test"]),
             ("D1", "e5_gridpower_raw", "e3_power_raw", ["rep", "seed_index", "case"]),
             ("D2", "e5_gridrealvol_raw", "e3_realvol_raw", ["rep", "seed_index", "case"]))
    rows = []
    for what, mine, theirs, keys in specs:
        if not ctx.has_csv(mine) or not (e3dir / f"{theirs}.csv").exists():
            continue
        a = ctx.load_csv(mine)
        a = a[a["config"] == "default"]
        b = pd.read_csv(e3dir / f"{theirs}.csv")
        if what == "grid":
            b = b.rename(columns={"rep": "seed_index"})
            a = a[a["test"].isin(["critic", "relative_anchored"])]
            b = b[b["test"].isin(["critic", "relative_anchored"])]
        mm = a[keys + ["pvalue"]].merge(b[keys + ["pvalue"]], on=keys, suffixes=("", "_e3"))
        d = np.abs(mm["pvalue"].to_numpy(float) - mm["pvalue_e3"].to_numpy(float))
        ok = d <= 1e-12 + 1e-9 * np.abs(mm["pvalue_e3"].to_numpy(float))
        rows.append({"what": what, "n_e5": len(a), "n_e3": len(b), "n_compared": len(mm),
                     "n_identical": int(ok.sum()),
                     "max_abs_diff": float(np.nanmax(d)) if d.size else NAN})
        print(f"E3 ALIGNMENT {what}: {int(ok.sum())}/{len(mm)} default-configuration p-values "
              f"identical to E3's {theirs}.csv (max |diff| {rows[-1]['max_abs_diff']:.3g})",
              flush=True)
    if rows:
        ctx.save_csv(pd.DataFrame(rows), "e5_e3_alignment")


def build_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    for fn in (build_grid_outputs, build_split_outputs, build_simpower_outputs):
        try:
            fn(ctx, cfg)
        except Exception:  # noqa: BLE001 - one broken output must not hide the others
            logger.exception("%s failed", fn.__name__)
    try:
        compare_with_e3(ctx)
    except Exception:  # noqa: BLE001
        logger.exception("compare_with_e3 failed")


# --------------------------------------------------------------------------------------------
# Timing probes (quick mode): one unit of each kind at the FULL settings (100 tickers)
# --------------------------------------------------------------------------------------------
def _probe_rep(cell: dict, rep: int, rng: np.random.Generator) -> dict[str, Any]:
    kind = cell["kind"]
    t0 = time.perf_counter()
    out: dict[str, Any] = {}
    if kind == "eval":
        for r in _eval_rep(cell, rep, rng):
            out[f"rt_{r['test']}"] = r.get("runtime_s", NAN)
            if r.get("error"):
                out["error"] = r["error"]
    elif kind == "design":
        _design_rep(cell, rep, rng)
    elif kind == "size":
        r = _size_rep(cell, rep, rng)
        out["rt_critic"] = r.get("test_seconds", NAN)
        if r.get("error"):
            out["error"] = r["error"]
    elif kind == "simpower":
        out["rt_sim"] = _simpower_rep(cell, rep, rng)["seconds"]
    elif kind == "heston_fit":
        panel = _base_panel("main", None)
        r = panel.train.iloc[:, 0].to_numpy(dtype=np.float64)
        get_model("Heston", **e3.MODEL_KW["Heston"]).fit(r)
    out["probe_seconds"] = time.perf_counter() - t0
    out["max_rss_mb"] = _max_rss_mb()
    return out


def run_probes(ctx: common.ExpContext) -> pd.DataFrame:
    F = FULL
    cells: list[dict] = []
    parts = set(ctx.parts)
    g0 = GRID_BY_NAME["default"]
    gridlike = bool(parts & set(GRID_PARTS))
    if gridlike:
        g_fit = ensure_fits(ctx, "main", ("GARCHt",), F, None)["GARCHt"]
        for c in PROBED:  # also gives the per-configuration cost ratios of the other parts
            g = GRID_BY_NAME[c]
            cells.append({**_eval_cell(f"grid_{c}", g, "GARCHt", "main", F, g_fit,
                                       tests=("critic",), methods=("anchored",), test_kw={},
                                       cost=_cost(g)), "kind": "eval"})
        if "gridsize" in parts:
            f_fit, _ = ensure_factor_fit(ctx, None)
            for d, fit in (("garch", g_fit), ("factor", f_fit)):
                cells.append({**_size_cell(g0, d, F, fit), "cell": f"size_{d}", "kind": "size"})
        if "gridpower" in parts:
            cells.append({**_design_cell("power_seed", g0, "D1", "main", F, g_fit, n_seeds=1),
                          "kind": "design"})
        if "gridrealvol" in parts:
            cells.append({**_design_cell("realvol_seed", g0, "D2", "main", F, None, n_seeds=1),
                          "kind": "design"})
    if "grid" in parts or "split" in parts:
        cells.append({"cell": "heston_fit", "kind": "heston_fit", "_cost": 2.0})
    if "split" in parts:
        a_fit = ensure_fits(ctx, "alt", ("GARCHt",), F, None)["GARCHt"]
        cells.append({**_eval_cell("split_battery", g0, "GARCHt", "alt", F, a_fit,
                                   tests=pipeline.ABSOLUTE_TESTS,
                                   methods=pipeline.RELATIVE_METHODS, test_kw={}, cost=3.0),
                      "kind": "eval"})
        cells.append({**_design_cell("power_alt_seed", g0, "D1", "alt", F, a_fit, n_seeds=1),
                      "kind": "design"})
        cells.append({**_design_cell("realvol_alt_seed", g0, "D2", "alt", F, None, n_seeds=1),
                      "kind": "design"})
    if "simpower" in parts:
        for c in _sim_cells(F):
            if c["family"] == "c" and c["lr"] == 1e-3 and not c["perm_cell"]:
                cells.append({**c, "cell": f"sim_{c['n_steps']}", "kind": "simpower"})
    if not cells:
        return pd.DataFrame()
    return common.run_cells(_probe_rep, cells, part="probe", ctx=ctx, root_seed=ROOT_SEED,
                            n_reps=1, config={"probe": True, "full": F, "grid": GRID,
                                              "code": lib_code_hash()},
                            block_size=1, cost=lambda c: c.get("_cost", 1.0), fresh=True)


def report_estimates(ctx: common.ExpContext, probe: pd.DataFrame,
                     workers: int = 12) -> dict[str, float]:
    """Print measured per-unit times and estimated full-run wall times per part."""
    F = FULL
    est: dict[str, float] = {}
    if probe.empty:
        return est
    t = probe.set_index("cell")

    def ps(name: str) -> float:
        return float(t.loc[name, "probe_seconds"]) if name in t.index else NAN

    heston = ps("heston_fit") if "heston_fit" in t.index else 0.0
    # Fits of the five models on one split: Heston dominates (parallel over tickers); the others
    # take seconds (Merton 0.1 s, GARCHt 0.02 s per ticker). Main split: E3's pickles if present.
    fit_wall = heston * math.ceil(config.N_PANEL / workers) + 30.0
    if heston:
        print(f"TIMING Heston SMM fit (full settings, 1 ticker): {heston:.1f} s; fits of the 5 "
              f"models on one split ~{common.fmt_seconds(fit_wall)} wall at {workers} workers "
              "(main split: 0 if E3's pickles exist)", flush=True)
    if "max_rss_mb" in probe:
        print(f"MEMORY peak RSS of a probe worker (full settings): "
              f"{float(probe['max_rss_mb'].max()):.0f} MB (split battery included if probed)",
              flush=True)

    def ratio(c: str, col: str) -> float:
        a = t.loc[f"grid_{c}", col] if f"grid_{c}" in t.index and col in t else NAN
        b = t.loc["grid_default", col] if "grid_default" in t.index and col in t else NAN
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            return float(a / b)
        return COST.get(c, 1.0)

    def per_config(fn: Any) -> dict[str, list[float]]:
        return {g["config"]: fn(g) for g in GRID}

    def report(label: str, tasks_by_cfg: dict[str, list[float]], extra: float = 0.0) -> float:
        tasks = [x for v in tasks_by_cfg.values() for x in v]
        wall = common.print_wall_estimate(label, tasks, workers=workers, extra_s=extra)
        light = [x for c, v in tasks_by_cfg.items() if c not in HEAVY for x in v]
        heavy = [x for c, v in tasks_by_cfg.items() if c in HEAVY for x in v]
        if heavy:
            print(f"  of which --configs without {','.join(HEAVY)}: "
                  f"{common.fmt_seconds(common.estimate_wall(light, workers) + extra)}; "
                  f"--configs {','.join(HEAVY)}: "
                  f"{common.fmt_seconds(common.estimate_wall(heavy, workers))}", flush=True)
        return wall

    if "grid" in ctx.parts:
        def grid_tasks(g: dict) -> list[float]:
            c = g["config"]
            s = ps(f"grid_{c}") if c in PROBED else ps("grid_default") * COST.get(c, 1.0)
            if c in PROBED:
                common.print_rep_timing(f"grid {c}", s,
                                        note="one model x seed, critic + anchored, full panel")
            return common.block_tasks(s, F["n_seeds"], F["blocks"]["grid"]) * len(MODELS)
        est["grid"] = report("grid", per_config(grid_tasks), fit_wall)
    if "gridsize" in ctx.parts:
        s_d = {d: ps(f"size_{d}") for d in F["size_designs"]}
        for d, s in s_d.items():
            common.print_rep_timing(f"gridsize {d} default", s,
                                    note="2 panel draws + absolute critic test")

        def size_tasks(g: dict) -> list[float]:
            r = ratio(g["config"], "rt_critic")
            return [x for d, s in s_d.items()
                    for x in common.block_tasks(s * r, F["size_reps"], F["blocks"]["gridsize"])]
        est["gridsize"] = report("gridsize", per_config(size_tasks), 10.0)
    for d, part, pname in (("D1", "gridpower", "power_seed"), ("D2", "gridrealvol",
                                                                "realvol_seed")):
        if part not in ctx.parts:
            continue
        s1 = ps(pname)
        common.print_rep_timing(f"{part} default, one seed", s1,
                                note="alternative + boundary anchored tests")
        common.print_rep_timing(f"{part} default replication ({F['n_seeds']} seeds, "
                                "extrapolated)", s1 * F["n_seeds"])

        def dtasks(g: dict, s1: float = s1) -> list[float]:
            c = g["config"]
            reps = F["power_reps_default"] if c == "default" else F["power_reps"]
            per = F["n_seeds"] * s1 * ratio(c, "rt_relative_anchored")
            return common.block_tasks(per, reps, F["blocks"]["power"])
        est[part] = report(part, per_config(dtasks), 10.0)
    if "split" in ctx.parts:
        s = ps("split_battery")
        common.print_rep_timing("split battery", s, note="one model x seed, full E3 battery "
                                "(6 absolute + anchored + two-critic)")
        nb = min(workers, MAX_WORKERS["split"])
        battery = common.block_tasks(s, F["n_seeds"], F["blocks"]["split"]) * len(MODELS)
        battery_wall = common.estimate_wall(battery, nb)
        print(f"TIMING split battery phase: {common.fmt_seconds(sum(battery))} CPU, "
              f"~{common.fmt_seconds(battery_wall)} wall at {nb} workers (memory cap)",
              flush=True)
        tasks = []
        for pname, lab in (("power_alt_seed", "D1"), ("realvol_alt_seed", "D2")):
            s1 = ps(pname)
            common.print_rep_timing(f"split power {lab}, one seed", s1,
                                    note="alternative + boundary at the alternative split")
            tasks += common.block_tasks(s1 * F["n_seeds"], F["split_power_reps"],
                                        F["blocks"]["power"])
        est["split"] = common.print_wall_estimate(
            "split (power phase; + fits and battery phase as precompute)", tasks,
            workers=workers, extra_s=fit_wall + battery_wall)
    if "simpower" in ctx.parts:
        tasks = []
        for steps in SIM_STEPS:
            s = ps(f"sim_{steps}")
            common.print_rep_timing(f"simpower n_steps={steps}", s,
                                    note=f"critic test, n={F['sim_n']} windows per sample")
            n_cells = len(SIM_FAMILIES) * (len(SIM_LR) + (steps == SIM_DEFAULT[1]))
            tasks += common.block_tasks(s, F["sim_reps"], F["blocks"]["simpower"]) * n_cells
        est["simpower"] = common.print_wall_estimate("simpower", tasks, workers=workers)
    return est


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------
def _add_args(parser: Any) -> None:
    parser.add_argument("--configs", default=None, metavar="A,B",
                        help="grid parts: only these configurations "
                             f"({', '.join(GRID_BY_NAME)}; 'light' = all but "
                             f"{','.join(HEAVY)}, 'heavy' = those two); default all")
    parser.add_argument("--power-reps", type=int, default=None, metavar="N",
                        help="override the power replications per non-default configuration "
                             "and design (full default 50, the DESIGN minimum)")
    parser.add_argument("--power-reps-default", type=int, default=None, metavar="N",
                        help="override the default configuration's power replications (full "
                             "default 200 = E3's)")
    parser.add_argument("--no-probe", action="store_true",
                        help="quick mode: skip the full-settings timing probes")


def main(argv: list[str] | None = None) -> None:
    ctx = common.parse_args(EXP, PARTS, description=__doc__, aliases=ALIASES,
                            add_arguments=_add_args, argv=argv)
    cfg = dict(QUICK if ctx.quick else FULL)
    for opt, key in (("power_reps", "power_reps"), ("power_reps_default", "power_reps_default")):
        v = getattr(ctx.args, opt, None)
        if v is not None:
            if v < 1:
                raise SystemExit(f"--{opt.replace('_', '-')} must be >= 1")
            cfg[key] = int(v)
            logger.warning("%s overridden: %d (not the full configuration)", key, int(v))
    selected_configs(ctx)  # validate --configs early
    sw = common.Stopwatch()
    timing: dict[str, float] = {}
    alignment = check_e3_alignment()
    ctx.write_meta({"quick": ctx.quick, "parts": ctx.parts, "plots_only": ctx.plots_only,
                    "configs": selected_configs(ctx), "cfg": cfg, "grid": GRID,
                    "models": list(MODELS), "model_kw": model_kw(cfg),
                    "alt_split": SPLITS["alt"], "pipeline": pipeline.pipeline_config(),
                    "root_seed": ROOT_SEED, "e3_root_seed": e3.ROOT_SEED,
                    "code": lib_code_hash(), "e3_alignment": alignment,
                    "rel_invalid": REL_INVALID})
    if not ctx.plots_only:
        runners = {"grid": run_grid, "gridsize": run_gridsize, "split": run_split,
                   "simpower": run_simpower, "gridpower": run_gridpower,
                   "gridrealvol": run_gridrealvol}
        for part in ctx.parts:
            timing.update(runners[part](ctx, cfg))
            sw.lap(part)
    build_outputs(ctx, cfg)
    sw.lap("outputs")
    if ctx.quick and not ctx.plots_only and not ctx.args.no_probe:
        probe = run_probes(ctx)
        sw.lap("probes")
        est = report_estimates(ctx, probe)
        timing.update({f"estimate_wall_s_{k}": v for k, v in est.items()})
    timing.update({f"wall_{k}_s": v for k, v in sw.laps.items()})
    ctx.write_meta(None, timing=timing)
    for k, v in sw.laps.items():
        print(f"TIMING wall {k} (this run, {ctx.workers} workers): {common.fmt_seconds(v)}",
              flush=True)
    print(f"TIMING total wall (this run, {ctx.workers} workers): {common.fmt_seconds(sw.total())}",
          flush=True)


if __name__ == "__main__":
    main()
