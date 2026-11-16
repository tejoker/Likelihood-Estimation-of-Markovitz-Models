"""E6: neural generators under the E3 protocol (docs/PROTOCOL_v2.md, "Planned additions").

The neural generators are test subjects of the paper's method, evaluated exactly like the five
classical generators of E3 (``experiments/e3_sp500.py``): same 100-ticker panel, TRAIN/TEST
split, windows (L = 20), per-ticker TRAIN-std normalisation, synthetic seeds, tests and decision
rule. This script only adds what is specific to the neural models (one pooled training run)
and reuses E3's code for everything else (it imports ``e3_sp500``).

Models
------
QuantGAN    ``synthval.models.neural.QuantGAN`` (Wiese, Knobloch, Korn & Kretschmer 2020, "pure
            TCN" generator: TCN with skip connections, receptive field 127, 3-dimensional noise,
            Lambert W preprocessing; WGAN-GP training). Trained ONCE on the pooled TRAIN
            log-returns of all panel tickers, each standardised by its own TRAIN mean and std;
            the path of ticker ``i`` is a generated standardised path times ``std_i`` plus
            ``mean_i``. Deviations from the paper: class docstring.

Parts (``--part``; canonical run order)
---------------------------------------
fit       ``pipeline.fit_panel("QuantGAN", panel.train, fit_seed=...)`` on the E3 panel (on
          ``$SYNTHVAL_DEVICE``; CPU threads = ``--workers``). The fitted per-ticker views are
          pickled under ``results/<exp>/checkpoints/fit_<model>/<config hash>/`` and reused by
          every later part. ``e6_fit_<model>.csv`` (per ticker: mean, std and the shared training
          record), ``e6_train_curve_<model>.csv``; table ``e6_fit.tex`` (training record and
          TRAIN diagnostics, real vs a simulated TRAIN-sized sample: never used for selection),
          figure ``e6_train_curve.pdf``. ``--fresh`` retrains only when this part is selected.
pooled    per model and synthetic seed ``s = 0 .. n_seeds - 1``: ``pipeline.evaluate_model``
          with E3's pooled settings (``e3.synthetic_seed(s)``, so synthetic draws and test
          streams are those of E3's models at the same seed: paired comparisons; dependent
          protocol, time blocks + HAC; ``start_from_end`` is irrelevant for QuantGAN). Default
          tests: the absolute W-critic and the anchored relative test (``FULL["tests"]``,
          ``FULL["relative_methods"]``). ``e6_pooled_raw.csv``, ``e6_pooled_summary.csv``,
          table ``e6_pooled_tests.tex``, figure ``e6_pvalues.pdf`` (E3's table code).
stylized  E3's stylized-fact summary (``stylized.panel_summary`` per ticker, ACF of |r| lags
          1-50, pooled tail of normalised |r|) of the real TEST returns and of each model's
          headline synthetic TEST path (seed 0, as in E3). ``e6_stylized_raw.csv``,
          ``e6_stylized_acf.csv``, ``e6_stylized_tail.csv``. Table ``e6_stylized.tex`` and figure
          ``e6_stylized.pdf`` show Real, E3's classical models (read from E3's CSVs in the same
          results tree, when present) and the neural models, with E3's code.
verdict   (derived) E3's decision rule (``e3.compute_verdict``) on E6's pooled results with the
          power designs D1 / D2 recomputed (``e3.design_summary``) from E3's
          ``e3_power_raw.csv`` and ``e3_realvol_raw.csv`` in the same results tree
          (``config.RESULTS_DIR / "e3"``; quick mode: ``e3_quick``; ``--e3-dir`` overrides). The
          power check does not depend on the generator, so it applies when E6 runs with E3's
          settings (checked: number of seeds). ``e6_verdicts.csv``, ``e6_power_designs.csv``,
          table ``e6_verdicts.tex``. Rerun by every invocation and by ``--plots-only``.

E3's table and verdict functions read the model list from ``e3_sp500.MODELS``; this script
calls them on a small context adapter (:class:`E3View`: E6 inputs and outputs, E3's power
designs) with the model list swapped for the duration of the call (:func:`e3_models`).

Settings: ``n_tickers``, ``n_seeds``, ``test_kw``, ``base_kw``, ``n_boot``, ``acf_lags`` and
``tail_points`` are read from E3's ``FULL`` / ``QUICK`` at run time (the v2 runner's E3 overrides
apply: 10 seeds under v2). ``FULL`` / ``QUICK`` below hold the model settings and the tests.

Usage::

    uv run python experiments/e6_neural.py --quick --workers 1        # smoke test (CPU, tiny)
    PYTHONPATH=src uv run --project envs/gpu python experiments/v2.py e6 --part fit
    PYTHONPATH=src uv run --project envs/gpu python experiments/v2.py e6 --part pooled,stylized --workers 4
    PYTHONPATH=src uv run --project envs/gpu python experiments/v2.py e6 --plots-only

Checkpoints: the fit is stored under a hash of its configuration (model settings, tickers,
TRAIN dates, fit seed; not the device); downstream checkpoint keys include a hash of the fitted
table, so a retrain never reuses stale blocks. A change of ``synthval/models/neural.py`` after
the fit is detected (warning: rerun ``--part fit --fresh``).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from collections.abc import Iterator, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import common  # noqa: E402  (experiments/ is on sys.path when run as a script)
import e3_sp500 as e3  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from synthval import config, plotting, stylized  # noqa: E402
from synthval import pipeline as pl  # noqa: E402
from synthval.models import neural  # noqa: E402
from synthval.twosample.nets import get_device  # noqa: E402

logger = logging.getLogger("experiments.e6")

EXP = "e6"
ROOT_SEED = 20261116  # fit seeds; synthetic seeds and test streams are E3's
PARTS = ("fit", "pooled", "stylized", "verdict")
DERIVED_PARTS = ("verdict",)
MODELS: tuple[str, ...] = ("QuantGAN",)
E3_MODELS: tuple[str, ...] = tuple(e3.MODELS)  # E3's classical generators (comparison rows)
MODEL_LABELS: dict[str, str] = {"QuantGAN": "QuantGAN"}
MODEL_COLORS: dict[str, str] = {"QuantGAN": plotting.OKABE_ITO["skyblue"]}
MODEL_MARKERS: dict[str, str] = {"QuantGAN": "h"}
#: Settings shared with E3 (read from ``e3.FULL`` / ``e3.QUICK`` at run time).
E3_SHARED: tuple[str, ...] = ("n_tickers", "n_seeds", "test_kw", "base_kw", "n_boot",
                              "acf_lags", "tail_points")
#: Peak memory per pooled worker (critic + anchored test, panel, QuantGAN generator), GB.
WORKER_GB = 0.6

FULL: dict[str, Any] = {
    "model_kw": {"QuantGAN": {}},          # class defaults (neural.QuantGAN docstring)
    "tests": ["critic"],                   # absolute tests (E3 runs all six)
    "relative_methods": ["anchored"],      # E3 also runs the two-critic test (secondary)
    "blocks": {"pooled": 1},
}
QUICK: dict[str, Any] = {
    "model_kw": {"QuantGAN": {**neural.TINY_KW, "n_steps": 30, "check": True}},
    "tests": ["critic"],
    "relative_methods": ["anchored"],
    "blocks": {"pooled": 1},
}


def run_config(quick: bool) -> dict[str, Any]:
    """E6 configuration: E3's shared settings (current values) updated with ``FULL``/``QUICK``."""
    base = e3.QUICK if quick else e3.FULL
    cfg = {k: copy.deepcopy(base[k]) for k in E3_SHARED}
    cfg.update(copy.deepcopy(QUICK if quick else FULL))
    return cfg


def seed_int(*keys: Any) -> int:
    return int(common.keyed_seed_sequence(ROOT_SEED, *keys).generate_state(1, np.uint64)[0])


# --------------------------------------------------------------------------------------------
# Part: fit (one pooled training run per model)
# --------------------------------------------------------------------------------------------
_FITTED_THIS_RUN: set[str] = set()
_SOURCE_WARNED: set[str] = set()


def fit_config(model: str, cfg: dict, tickers: list[str]) -> dict[str, Any]:
    return {"model": model, "model_kw": cfg["model_kw"][model], "tickers": list(tickers),
            "train": list(config.TRAIN), "fit_seed": seed_int("fit", model)}


def fit_paths(ctx: common.ExpContext, model: str, cfg: dict, tickers: list[str]
              ) -> tuple[Path, Path, Path, str]:
    """Pickle, fit table, training curve and configuration hash of a model's fit."""
    ck = common.Checkpoint(ctx.ckpt_dir, f"fit_{model}", fit_config(model, cfg, tickers),
                           fresh=ctx.fresh)
    return (ck.file("fitted.pkl"), ck.file("fit_table.csv"), ck.file("train_curve.csv"),
            ck.hash)


def model_source_hash() -> str:
    return hashlib.sha256(inspect.getsource(neural).encode()).hexdigest()[:16]


def _check_fit_source(pkl: Path) -> None:
    if str(pkl) in _SOURCE_WARNED:
        return
    _SOURCE_WARNED.add(str(pkl))
    sfile = pkl.with_name("model_source.sha")
    old = sfile.read_text().strip() if sfile.exists() else ""
    cur = model_source_hash()
    if old != cur:
        logger.warning("synthval/models/neural.py changed since the fit in %s was made (%s): "
                       "rerun '--part fit --fresh' if the change affects training or simulation",
                       pkl.parent, "no record" if not old else f"{old} -> {cur}")


def ensure_fit(ctx: common.ExpContext, model: str, cfg: dict, tickers: list[str]
               ) -> tuple[str, pd.DataFrame, float]:
    """Path of the pickled per-ticker views of ``model`` (training it if needed), fit table and
    wall seconds spent training in this call (0 when loaded from the checkpoint)."""
    pkl, csv, curve_csv, _ = fit_paths(ctx, model, cfg, tickers)
    refit = (ctx.fresh and "fit" in ctx.parts and not ctx.plots_only
             and str(pkl) not in _FITTED_THIS_RUN)
    if pkl.exists() and csv.exists() and not refit:
        if str(pkl) not in _FITTED_THIS_RUN:
            _check_fit_source(pkl)
        return str(pkl), pd.read_csv(csv, index_col="ticker"), 0.0
    import torch

    panel = e3.get_panel(tickers)
    old_threads = torch.get_num_threads()
    if get_device().type == "cpu":
        torch.set_num_threads(max(1, ctx.workers))
    t0 = time.perf_counter()
    try:
        fitted, table = pl.fit_panel(model, panel.train, fit_seed=seed_int("fit", model),
                                     **cfg["model_kw"][model])
    finally:
        torch.set_num_threads(old_threads)
    wall = time.perf_counter() - t0
    core = next(iter(fitted.values())).core
    csv.unlink(missing_ok=True)  # table last: a crash in between leaves no table (=> refit)
    tmp = pkl.with_name(f".{pkl.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(fitted, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, pkl)
    e3._atomic_write(pkl.with_name("model_source.sha"), model_source_hash() + "\n")
    curve = pd.DataFrame(core.fit_info_.get("curve", {}))
    e3._atomic_write(curve_csv, curve.to_csv(index=False))
    e3._atomic_write(csv, table.to_csv())
    _FITTED_THIS_RUN.add(str(pkl))
    e3._FITS.pop(str(pkl), None)
    for k in [k for k in e3._SIMS if k[0] == str(pkl)]:
        e3._SIMS.pop(k)
    logger.info("trained %s on %d tickers in %s (%s)", model, len(tickers),
                common.fmt_seconds(wall), core.fit_info_.get("device"))
    return str(pkl), pd.read_csv(csv, index_col="ticker"), wall


def fit_hashes(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, str]:
    """Per model ``<configuration hash>:<fitted-table hash>`` (downstream checkpoint keys)."""
    out = {}
    for m in MODELS:
        _, csv, _, h = fit_paths(ctx, m, cfg, tickers)
        out[m] = f"{h}:{e3.fit_content_hash(csv)}" if csv.exists() else h
    return out


def run_fit(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    timing: dict[str, float] = {}
    for m in MODELS:
        path, table, wall = ensure_fit(ctx, m, cfg, tickers)
        timing[f"fit_{m}_wall_s"] = wall
        conf = {"part": "fit", **fit_config(m, cfg, tickers)}
        ctx.save_csv(table.reset_index(), f"e6_fit_{m}", config=conf)
        curve_csv = fit_paths(ctx, m, cfg, tickers)[2]
        if curve_csv.exists():
            ctx.save_csv(pd.read_csv(curve_csv), f"e6_train_curve_{m}", config=conf)
    ctx.write_part_meta("fit", {"models": list(MODELS), "model_kw": cfg["model_kw"],
                                "tickers": tickers, "device": str(get_device())}, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part: pooled (E3's pooled protocol, critic + anchored by default)
# --------------------------------------------------------------------------------------------
def _pooled_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Tests of one model at synthetic seed index ``rep`` (``rng`` unused: E3's seeds)."""
    panel = e3.get_panel(cell["_tickers"])
    fitted = e3.load_fits(cell["_fit_path"])
    c = cell["_cfg"]
    rows = pl.evaluate_model(panel, fitted, seed=e3.synthetic_seed(rep), tests=tuple(c["tests"]),
                             relative_base="critic",
                             relative_methods=tuple(c["relative_methods"]),
                             protocol="dependent", test_kw=c["test_kw"], base_kw=c["base_kw"],
                             n_boot=c["n_boot"], start_from_end=True, all_pvalues=True,
                             keep_extra=True)
    for r in rows:
        r.update(e3._train_steps(r.pop("extra", None) or {}))
        r["seed"] = str(r["seed"])
        r["max_rss_mb"] = e3._max_rss_mb()
    return rows


def pooled_config(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, Any]:
    return {"tickers": tickers, "fit_hash": fit_hashes(ctx, cfg, tickers),
            "tests": list(cfg["tests"]), "methods": list(cfg["relative_methods"]),
            "test_kw": cfg["test_kw"], "base_kw": cfg["base_kw"], "n_boot": cfg["n_boot"],
            "protocol": "dependent", "synthetic_root_seed": e3.ROOT_SEED,
            "pipeline": pl.pipeline_config(), "train_steps": True}


def pooled_workers(ctx: common.ExpContext) -> int:
    avail = e3.mem_available_gb()
    cap = int(0.85 * avail // WORKER_GB) if np.isfinite(avail) else ctx.workers
    w = max(1, min(ctx.workers, cap))
    if w < ctx.workers:
        logger.warning("pooled: %d workers instead of %d (memory: about %.1f GB per worker, "
                       "%.1f GB available)", w, ctx.workers, WORKER_GB, avail)
    return w


def run_pooled(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    t0 = time.perf_counter()
    paths = {m: ensure_fit(ctx, m, cfg, tickers)[0] for m in MODELS}
    conf = pooled_config(ctx, cfg, tickers)
    c = {k: cfg[k] for k in ("tests", "relative_methods", "test_kw", "base_kw", "n_boot")}
    cells = [{"cell": f"model={m}", "model": m, "_tickers": list(tickers), "_fit_path": paths[m],
              "_cfg": c} for m in MODELS]
    df = common.run_cells(_pooled_rep, cells, part="pooled", ctx=ctx, root_seed=ROOT_SEED,
                          n_reps=cfg["n_seeds"], config=conf,
                          block_size=cfg["blocks"]["pooled"], workers=pooled_workers(ctx))
    ctx.save_csv(df, "e6_pooled_raw", config={"part": "pooled", **conf,
                                              "n_seeds": cfg["n_seeds"]})
    timing = {"pooled_wall_s": time.perf_counter() - t0}
    ctx.write_part_meta("pooled", {**conf, "n_seeds": cfg["n_seeds"]}, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part: stylized (E3's summary on the headline synthetic path)
# --------------------------------------------------------------------------------------------
def stylized_rows(sources: dict[str, pd.DataFrame], scale: pd.Series, acf_lags: int,
                  tail_points: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-ticker stylized facts, ACF of |r| quantiles per lag and the pooled tail of normalised
    |r| per source: the computation of ``e3.run_stylized``."""
    per, acf_rows, tail_rows = [], [], []
    nl = int(acf_lags)
    for src, df in sources.items():
        ps = stylized.panel_summary(df, per_ticker=True)
        ps = ps.reset_index().rename(columns={"index": "ticker"})
        ps.insert(0, "source", src)
        per.append(ps)
        acfs = np.vstack([stylized.acf(np.abs(df[t].to_numpy()), nl)[1:] for t in df.columns])
        for k in range(nl):
            col = acfs[:, k]
            acf_rows.append({"source": src, "lag": k + 1, "median": float(np.nanmedian(col)),
                             "q25": float(np.nanquantile(col, 0.25)),
                             "q75": float(np.nanquantile(col, 0.75))})
        z = np.sort(np.abs((df / scale.loc[df.columns]).to_numpy()).ravel())[::-1]
        n = z.size
        ranks = np.unique(np.geomspace(1, n, int(tail_points)).astype(int))
        for r in ranks:
            tail_rows.append({"source": src, "x": float(z[r - 1]), "ccdf": r / n})
    return pd.concat(per, ignore_index=True), pd.DataFrame(acf_rows), pd.DataFrame(tail_rows)


def run_stylized(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    paths = {m: ensure_fit(ctx, m, cfg, tickers)[0] for m in MODELS}
    t0 = time.perf_counter()
    panel = e3.get_panel(tickers)
    sources: dict[str, pd.DataFrame] = {"Real": panel.test}
    for m in MODELS:
        sim = e3.headline_sim(paths[m], panel)  # seed 0, exactly as in the pooled part
        sources[m] = pd.DataFrame(sim, index=panel.test.index, columns=panel.tickers)
    raw, acf, tail = stylized_rows(sources, panel.scale, cfg["acf_lags"], cfg["tail_points"])
    conf = {"part": "stylized", "tickers": tickers, "fit_hash": fit_hashes(ctx, cfg, tickers),
            "synthetic_seed_index": 0, "synthetic_root_seed": e3.ROOT_SEED}
    ctx.save_csv(raw, "e6_stylized_raw", config=conf)
    ctx.save_csv(acf, "e6_stylized_acf", config=conf)
    ctx.save_csv(tail, "e6_stylized_tail", config=conf)
    timing = {"stylized_wall_s": time.perf_counter() - t0}
    ctx.write_part_meta("stylized", conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Reuse of E3's table and verdict code
# --------------------------------------------------------------------------------------------
@contextlib.contextmanager
def e3_models(models: Sequence[str]) -> Iterator[None]:
    """Run E3 functions with ``e3.MODELS = models`` (labels, colours and markers of the E6
    models added); everything is restored on exit."""
    old = e3.MODELS
    added = [(d, {m: v[m] for m in models if m in v and m not in d})
             for d, v in ((e3.MODEL_LABELS, MODEL_LABELS),
                          (plotting.COLORS_MODEL, MODEL_COLORS),
                          (plotting.MARKERS_MODEL, MODEL_MARKERS))]
    e3.MODELS = tuple(models)
    for d, new in added:
        d.update(new)
    try:
        yield
    finally:
        e3.MODELS = old
        for d, new in added:
            for k in new:
                d.pop(k, None)


#: E3 inputs that E6 reads from E3's results directory: summary name -> (raw CSV, design).
E3_DESIGN_INPUTS: dict[str, tuple[str, str]] = {
    f"e3_{part}_summary": (f"e3_{part}_raw", d) for d, part in e3.DESIGNS.items()}
#: E3 stylized outputs merged with E6's (classical-model rows only).
E3_STYLIZED: tuple[str, ...] = ("e3_stylized_raw", "e3_stylized_acf", "e3_stylized_tail")


class E3View:
    """The context E3's output functions expect, mapped onto E6.

    * ``e3_<x>`` CSVs, tables and figures are E6's ``e6_<x>`` (read and written in ``ctx``);
    * ``e3_power_summary`` / ``e3_realvol_summary`` are recomputed with ``e3.design_summary``
      from E3's raw power CSVs in ``e3_dir``;
    * with ``merge_e3_stylized``, the stylized CSVs are E6's rows plus the classical-model rows
      of E3's CSVs in ``e3_dir`` (when present).
    """

    def __init__(self, ctx: common.ExpContext, e3_dir: Path, *,
                 merge_e3_stylized: bool = False) -> None:
        self.ctx = ctx
        self.e3_dir = Path(e3_dir)
        self.merge_e3_stylized = merge_e3_stylized

    @staticmethod
    def e6_name(name: str) -> str:
        return "e6_" + name[3:] if name.startswith("e3_") else name

    def _input(self, name: str) -> pd.DataFrame | None:
        if name in E3_DESIGN_INPUTS:
            raw_name, design = E3_DESIGN_INPUTS[name]
            path = self.e3_dir / f"{raw_name}.csv"
            return e3.design_summary(pd.read_csv(path), design) if path.exists() else None
        own = (self.ctx.load_csv(self.e6_name(name)) if self.ctx.has_csv(self.e6_name(name))
               else None)
        if self.merge_e3_stylized and name in E3_STYLIZED and own is not None:
            path = self.e3_dir / f"{name}.csv"
            if path.exists():
                other = pd.read_csv(path)
                other = other[other["source"].isin(E3_MODELS)]  # E3's classical models only
                own = pd.concat([own, other], ignore_index=True)
        return own

    def has_csv(self, name: str) -> bool:
        return self._input(name) is not None

    def load_csv(self, name: str, **kw: Any) -> pd.DataFrame:
        df = self._input(name)
        if df is None:
            raise FileNotFoundError(f"{name}: no E6 or E3 input (E3 directory {self.e3_dir})")
        return df

    def save_csv(self, df: pd.DataFrame, name: str, config: dict | None = None,
                 timing: dict | None = None) -> Path:
        conf = {**(config or {}), "e3_dir": str(self.e3_dir)}
        return self.ctx.save_csv(df, self.e6_name(name), config=conf, timing=timing)

    def save_table(self, df: pd.DataFrame, name: str, **kw: Any) -> Path:
        return self.ctx.save_table(df, self.e6_name(name), **kw)

    def savefig(self, fig: Any, name: str) -> Path:
        return self.ctx.savefig(fig, self.e6_name(name))


def e3_dir_for(ctx: common.ExpContext) -> Path:
    """E3's results directory in the same results tree (``--e3-dir`` overrides)."""
    if ctx.args.e3_dir:
        p = Path(ctx.args.e3_dir)
        return p if p.is_absolute() else config.ROOT / p
    return ctx.results_root / (f"{e3.EXP}_quick" if ctx.quick else e3.EXP)


def compute_verdict(ctx: common.ExpContext) -> bool:
    """E3's decision rule on E6's pooled results with E3's power designs (see module doc)."""
    e3_dir = e3_dir_for(ctx)
    view = E3View(ctx, e3_dir)
    des = []
    for name, (raw_name, d) in E3_DESIGN_INPUTS.items():
        s = view._input(name)
        if s is None:
            logger.warning("verdict: %s not found in %s (run E3 --part %s first): no GOOD "
                           "verdict possible", raw_name, e3_dir, e3.DESIGNS[d])
            continue
        des.append(s.assign(source=str(e3_dir / f"{raw_name}.csv")))
    if des:
        ctx.save_csv(pd.concat(des, ignore_index=True), "e6_power_designs",
                     config={"part": "verdict", "e3_dir": str(e3_dir)})
    with e3_models(MODELS):
        ok = e3.compute_verdict(view)
    if ok and des:
        v = ctx.load_csv("e6_verdicts")
        n_design = {int(n) for s in des for n in s["n_seeds"]}
        for _, r in v.iterrows():
            if n_design != {int(r["n_seeds"])}:
                logger.warning("verdict %s: %d synthetic seeds, but the power designs used %s "
                               "seeds per replication: the rule's power does not apply",
                               r["model"], int(r["n_seeds"]), sorted(n_design))
    return ok


# --------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------
FIT_ROWS: tuple[tuple[str, str, int], ...] = (
    # (fit_info key, label, decimals; -1: string / integer)
    ("device", "Device", -1),
    ("loss", "Loss", -1),
    ("receptive_field", "Receptive field (days)", -1),
    ("n_weights_g", "Generator weights", -1),
    ("n_weights_d", "Discriminator weights", -1),
    ("n_series", "Tickers (pooled)", -1),
    ("n_windows", "Training windows", -1),
    ("n_steps", "Generator steps", -1),
    ("n_critic", "Critic steps per generator step", -1),
    ("batch_size", "Batch size", -1),
    ("train_seconds", "Training time (min)", 1),
    ("lw_delta", r"Lambert W $\delta$", 3),
    ("lw_sigma", r"Lambert W $\sigma$", 3),
    ("final_wdist", r"Critic $\hat W$, last 10\% of steps", 3),
    ("final_penalty", r"Penalty, last 10\% of steps", 3),
)
CHECK_ROWS: tuple[tuple[str, str, int], ...] = (
    ("std", "Std.\\ dev.", 3), ("kurt", "Excess kurtosis", 2),
    ("acf_abs_1", r"ACF $|z|$, lag 1 (median)", 3),
    ("acf_abs_20", r"ACF $|z|$, lag 20 (median)", 3), ("max_abs", r"$\max |z|$", 1),
    ("finite_share", "Finite share", 3),
)


def build_fit_outputs(ctx: common.ExpContext) -> None:
    """``e6_fit.tex``: training record and TRAIN diagnostics; ``e6_train_curve.pdf``."""
    cols: dict[str, dict[str, str]] = {}
    curves: dict[str, pd.DataFrame] = {}
    for m in MODELS:
        if not ctx.has_csv(f"e6_fit_{m}"):
            continue
        r = ctx.load_csv(f"e6_fit_{m}").iloc[0]
        col: dict[str, str] = {}
        for key, label, d in FIT_ROWS:
            v = r.get(key, np.nan)
            if key == "train_seconds":
                v = float(v) / 60.0
            col[label] = (str(v) if d < 0 else e3.tnum(v, d))
        for key, label, d in CHECK_ROWS:
            col[f"TRAIN check: {label}, real / sim."] = (
                f"{e3.tnum(r.get(f'check_real_{key}', np.nan), d)} / "
                f"{e3.tnum(r.get(f'check_sim_{key}', np.nan), d)}")
        cols[MODEL_LABELS[m]] = col
        if ctx.has_csv(f"e6_train_curve_{m}"):
            curves[m] = ctx.load_csv(f"e6_train_curve_{m}")
    if cols:
        tab = pd.DataFrame(cols)
        tab.index.name = "Quantity"
        ctx.save_table(tab, "e6_fit", escape=False)
    curves = {m: c for m, c in curves.items() if not c.empty}
    if curves:
        import matplotlib.pyplot as plt

        plotting.setup()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=plotting.figsize("two", 0.32))
        for m, c in curves.items():
            a1.plot(c["step"], c["wdist"], color=MODEL_COLORS[m], lw=0.9, label=MODEL_LABELS[m])
            a2.plot(c["step"], c["penalty"], color=MODEL_COLORS[m], lw=0.9)
        a1.set_xlabel("generator step")
        a1.set_ylabel(r"critic $\hat W$ (batch mean)")
        a2.set_xlabel("generator step")
        a2.set_ylabel("gradient penalty")
        a1.legend(loc="upper right", fontsize=6.5)
        ctx.savefig(fig, "e6_train_curve")


def build_outputs(ctx: common.ExpContext) -> None:
    build_fit_outputs(ctx)
    e3_dir = e3_dir_for(ctx)
    with e3_models(MODELS):
        e3.build_pooled_outputs(E3View(ctx, e3_dir))
        e3.build_verdict_table(E3View(ctx, e3_dir))
    with e3_models(E3_MODELS + MODELS):
        e3.build_stylized_outputs(E3View(ctx, e3_dir, merge_e3_stylized=True))


# --------------------------------------------------------------------------------------------
# Quick mode: full-settings timing estimates
# --------------------------------------------------------------------------------------------
def report_estimates(ctx: common.ExpContext, tickers: list[str], workers: int = 4) -> None:
    """Training time per generator step at the FULL model settings on this device (2 steps on
    the quick panel), and the pooled cost per seed from E3's full pooled runtimes."""
    full = run_config(False)
    avail = e3.mem_available_gb()
    if np.isfinite(avail) and avail < 3.0:
        print(f"ESTIMATE skipped: {avail:.1f} GB available (the full-size training probe needs "
              "about 1 GB)", flush=True)
    else:
        panel = e3.get_panel(tickers)
        for m in MODELS:
            kw = {**full["model_kw"][m], "n_steps": 2, "check": False, "log_every": 1}
            core = neural.QuantGAN(**kw).fit(panel.train.to_numpy(),
                                             rng=np.random.default_rng(0))
            sps = float(core.fit_info_["seconds_per_step"])
            n = int(neural.QuantGAN(**full["model_kw"][m]).n_steps)
            common.print_rep_timing(f"fit {m}", sps, note=f"per generator step at the FULL "
                                    f"settings on {core.fit_info_['device']}, "
                                    f"{core.fit_info_['n_critic']} critic steps each")
            common.print_wall_estimate(f"fit {m} ({n} steps, {core.fit_info_['device']})",
                                       [sps * n], workers=1, overhead_s=60.0)
    src = ctx.results_root / e3.EXP / "e3_pooled_raw.csv"
    if src.exists():
        raw = pd.read_csv(src)
        tests = list(full["tests"]) + [f"relative_{m}" for m in full["relative_methods"]]
        per = raw[raw["test"].isin(tests)].groupby(["model", "rep"])["runtime_s"].sum()
        s = float(per.median()) + 5.0  # + simulation and windows
        common.print_rep_timing("pooled", s, note=f"one model x seed, from E3's full pooled "
                                f"runtimes of {', '.join(tests)} ({src})")
        common.print_wall_estimate("pooled", [s] * (len(MODELS) * int(full["n_seeds"])),
                                   workers=workers)
    else:
        print(f"ESTIMATE pooled skipped: {src} not found", flush=True)


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------
def _add_args(parser: Any) -> None:
    parser.add_argument("--no-probe", action="store_true",
                        help="quick mode: skip the full-settings timing probe")
    parser.add_argument("--e3-dir", default=None,
                        help="E3 results directory with e3_power_raw.csv / e3_realvol_raw.csv "
                             "(default: <results>/e3, quick: <results>/e3_quick)")


def main(argv: list[str] | None = None) -> None:
    ctx = common.parse_args(EXP, PARTS, description=__doc__, add_arguments=_add_args, argv=argv)
    cfg = run_config(ctx.quick)
    tickers = e3.panel_tickers(cfg)
    sw = common.Stopwatch()
    timing: dict[str, float] = {}
    ctx.write_meta({"quick": ctx.quick, "parts": ctx.parts, "plots_only": ctx.plots_only,
                    "cfg": cfg, "root_seed": ROOT_SEED, "synthetic_root_seed": e3.ROOT_SEED,
                    "models": list(MODELS), "tickers": tickers, "e3_dir": str(e3_dir_for(ctx)),
                    "pipeline": pl.pipeline_config()})
    runners = {"fit": run_fit, "pooled": run_pooled, "stylized": run_stylized}
    if not ctx.plots_only:
        for part in ctx.parts:
            if part in runners:
                t0 = time.perf_counter()
                timing.update(runners[part](ctx, cfg, tickers))
                logger.info("part %s done in %s", part,
                            common.fmt_seconds(time.perf_counter() - t0))
    t0 = time.perf_counter()
    if compute_verdict(ctx):
        timing["verdict_wall_s"] = time.perf_counter() - t0
    sw.lap("compute")
    build_outputs(ctx)
    sw.lap("outputs")
    if ctx.quick and not ctx.plots_only and not ctx.args.no_probe:
        report_estimates(ctx, tickers)
        sw.lap("probes")
    timing.update({f"wall_{k}_s": v for k, v in sw.laps.items()})
    ctx.write_meta(None, timing=timing)
    print(f"TIMING total wall (this run, {ctx.workers} workers): {common.fmt_seconds(sw.total())}",
          flush=True)


if __name__ == "__main__":
    main()
