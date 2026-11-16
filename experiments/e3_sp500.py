"""E3: S&P 500 univariate models: critic verdicts, likelihood benchmarks, stylized facts.

DESIGN section 4 (E3); decision rule in section 1; inference protocol in section 3.7.

Data, models, synthetic paths
-----------------------------
Panel: ``data.load_panel()`` (``N_PANEL = 100`` tickers, ``PANEL_SEED``); TRAIN 2018-01-02 ..
2022-12-30 (1258 days, the "real past"), TEST 2023-01-03 .. 2025-06-16 (615 days, the "real
new"). Five univariate models are fitted per ticker on TRAIN with ``pipeline.fit_panel``: GBM,
Merton, GARCH(1,1)-t (variance targeting, persistence cap 0.995), Heston (Andersen QE, simulated
method of moments) and the stationary bootstrap. Fitted models are pickled under
``results/<exp>/checkpoints/fit_<model>/<config hash>/`` and reused by every later part (and by
parallel processes running other parts).

Synthetic data (``pipeline.evaluate_model``): one path per ticker over the TEST horizon, started
from the fitted end-of-TRAIN state (``start_from_end=True``), divided by the real TRAIN std of
the ticker and given the TEST date ranks; windows of L = 20, 3000 per side. Synthetic seed
``s = 0..9`` (``s = 0`` is the headline) is ``seed_int("synthetic", s)`` and is shared by the five
models: common random numbers, so at a given seed the test-side streams (e.g. which time blocks
train the critic) are the same for every model and model comparisons are paired. The headline
path (``s = 0``) is also the synthetic path of the per-ticker tests, of the stylized facts and of
the critic-witness figure.

Inference on the anchored relative test (after review)
-----------------------------------------------------
On real data a single-seed anchored p-value is **not calibrated**: R_past (real TRAIN) is
non-stationary (COVID 2020, the 2022 drawdown) and the 10-block time split compares the critic
with a random half of the TRAIN history, so a generator that reproduces the pooled TRAIN law
exactly is rejected far more often than alpha (the ``realvol`` part measures this). The only
anchored inference reported is therefore the seed-combined p-value ``pipeline.median_pvalue``
(twice the median over the synthetic seeds, theory.md Remark 8.5), whose size and power are
estimated for exactly that rule (``power`` and ``realvol`` parts, ``n_seeds`` seeds per
replication). Seed-0 p-values, per-seed rejection counts and the seed-0 HAC interval stay in the
CSVs (labelled uncalibrated) but not in the tables. The anchored (and two-critic) relative test
is **not valid for the stationary bootstrap** (its synthetic windows copy R_past windows that the
critic saw as pool negatives); its relative results are flagged and it gets no verdict.

Parts (``--part``; canonical run order)
---------------------------------------
fit         fit the 5 models per ticker; ``e3_fit_<model>.csv`` (parameters, ``at_bound``,
            scalar ``fit_info_``, fit time), ``e3_heston_moments_raw.csv`` (Heston SMM moments,
            data vs simulated, per ticker). Tables ``e3_fit_diagnostics.tex``,
            ``e3_heston_moments.tex``. ``--fresh`` refits only when this part is selected.
power       pre-specified power check of the decision rule (DESIGN section 1 item 3; run before
            looking at real-data test results), reference design "D1": the E3 GARCHt fits,
            R_past* (TRAIN length, stationary start) and R_new* (TEST length, end-of-TRAIN
            start) drawn once per replication, then ``n_seeds`` synthetic seeds, each with
            G* = an independent TEST-length draw with volatility x 1.2 (end-of-TRAIN start) and
            the boundary G0 = an independent draw with the law of R_past* (stationary start:
            exactly Law(G) = Law(R_past)); anchored relative test with the E3 settings and the
            same test streams for G* and G0. The rule's p-value is twice the median over the
            seeds, as in the verdict. 200 replications (DESIGN minimum 50; theory.md 11.7
            recommends >= 200). Seed 0 of replication ``k`` reproduces
            ``pipeline.power_check(n_reps=1, seed=seed_int("power", k))`` (alternative only; the
            boundary draw starts from stationarity here). ``e3_power_raw.csv``,
            ``e3_power_summary.csv``.
realvol     the same rule under a design "D2" with the real non-stationarity: R_past* window
            ``(i, k)`` = real TRAIN realised volatility of ticker i in window k (normalised
            returns) times fresh N(0, I_L) noise (one draw per replication, keeps the 2020 and
            2022 regimes and their co-movement across tickers); R_new = the real TEST windows;
            boundary G0 window = a volatility drawn uniformly from the ticker's TRAIN windows
            times N(0, I_L) noise, so G0 has exactly the *pooled* TRAIN window law; alternative
            G* = 1.2 x G0. Its boundary is the non-stationary placebo of the anchored test.
            ``e3_realvol_raw.csv``, ``e3_realvol_summary.csv``; with ``power``: table
            ``e3_power.tex``.
pooled      per model and synthetic seed: pooled absolute tests (critic, lrcritic, c2st, mmd,
            sigmmd, ks) with the dependent protocol (time_blocks = 10 + HAC for learned tests,
            wild bootstrap for kernel tests, KS flagged invalid) and the anchored (primary) and
            two-critic (secondary, invalid at the boundary) relative tests with R_past = real
            TRAIN windows (HAC p-values); training steps of the learned tests are recorded.
            ``e3_pooled_raw.csv``, ``e3_pooled_summary.csv`` (twice-median p, median p,
            rejection share over seeds, validity flags). Table ``e3_pooled_tests.tex``, figure
            ``e3_pvalues.pdf``. Workers are capped by memory (about 1 GB per worker).
perticker   per ticker at L = 5 (123 windows per side, TEST): critic (time blocks + HAC) and MMD
            (wild bootstrap) on the headline synthetic path. The test generator is keyed by the
            ticker (shared by the models). ``e3_perticker_raw.csv``.
perticker_null  size and calibration of the per-ticker tests: per ticker, independent pairs
            (X*, Y*) of GARCHt TEST-length paths from the fitted end-of-TRAIN state, tested
            exactly like ``perticker`` (10 pairs per ticker, 1000 in all). The nominal p-values
            are oversized (review: about 14% critic / 22% MMD at 5%), so the per-ticker
            p-values are calibrated against this pooled parametric null,
            ``p_cal = (1 + #{p0 <= p}) / (1 + n0)``; Benjamini-Hochberg (primary),
            Benjamini-Yekutieli and Storey pi0 (lambda = 0.5) use ``p_cal``.
            ``e3_perticker_null_raw.csv``; with ``perticker``: ``e3_perticker_summary.csv``,
            ``e3_perticker_null_size.csv``, tables ``e3_perticker.tex``,
            ``e3_perticker_null.tex``.
witness     what the headline pooled critic picks up: the seed-0 critic of each model is
            re-trained with exactly the random streams of the pooled test (same split, same
            initialisation; its statistic is checked against the pooled one) and its values f
            on the evaluation windows are stored with window features (log realised
            volatility, max |r|). ``e3_witness.csv``, figure ``e3_critic_witness.pdf``.
stylized    stylized facts (``stylized.panel_summary``) of the real TEST returns and of the
            headline synthetic TEST path of each model: panel medians and IQR; ACF of |r|
            (lags 1-50) and the pooled tail of normalised |r|. ``e3_stylized_raw.csv``,
            ``e3_stylized_summary.csv``, ``e3_stylized_acf.csv``, ``e3_stylized_tail.csv``;
            table ``e3_stylized.tex``, figure ``e3_stylized.pdf``.
likelihood  GBM / Merton / GARCHt: mean out-of-sample predictive log-likelihood on TEST given
            TRAIN, Berkowitz and PIT-KS tests per ticker (BH over tickers), pairwise
            Amisano-Giacomini tests (``vuong_test(hac=True)``, k = 0) per ticker and pooled
            (cross-sectional mean of the daily log-score differences, Newey-West variance:
            Driscoll-Kraay). ``e3_likelihood_raw.csv``, ``e3_ag_raw.csv``; tables
            ``e3_likelihood.tex``, ``e3_likelihood_ag.tex``.
agree       (from the CSVs of pooled, perticker, perticker_null, likelihood) Kendall tau between
            the model rankings by critic statistic and by predictive log-likelihood (3
            likelihood models; pooled, and per ticker with the L = 5 statistics; descriptive:
            tickers are cross-sectionally dependent, so no test against random rankings), Cohen
            kappa between per-ticker test and Berkowitz rejections per model and stratified by
            model (chance agreement within model, so model main effects do not count as
            agreement), on the calibrated per-ticker p-values. ``e3_agree_tau.csv``,
            ``e3_agree_kappa.csv``, table ``e3_agreement.tex``.
verdict     (from the CSVs of pooled, power, realvol) the decision rule per model: absolute critic
            twice-median p, anchored Delta (median and range over seeds) and twice-median p,
            power and boundary size of that rule, verdict. BAD if the twice-median anchored p
            is <= 0.05; GOOD if not and the rule's power is >= 0.8 under every power design
            (D1 and D2: the minimum is used); INCONCLUSIVE otherwise; INVALID (no verdict) for
            the bootstrap.
            The Clopper-Pearson verdict (beta_L >= 0.8, theory.md 11.7) is shown alongside.
            ``e3_verdicts.csv``, table ``e3_verdicts.tex``.

``agree`` and ``verdict`` only read CSVs; they are recomputed by every run (and by
``--plots-only``), so tables never mix old derived results with new inputs. When parts run in
parallel processes, finish with ``--plots-only``.

Usage::

    uv run python experiments/e3_sp500.py --quick                 # smoke test + timing estimate
    uv run python experiments/e3_sp500.py --part fit --workers 12  # run first: caches the fits
    uv run python experiments/e3_sp500.py --part power --workers 12
    uv run python experiments/e3_sp500.py --part pooled --workers 5  # memory: ~1 GB per worker
    uv run python experiments/e3_sp500.py --plots-only            # rebuild tables and figures

Every rejection rate is reported with a Wilson 95% interval (boundary sizes also with the
Clopper-Pearson interval of theory.md Definition 8.1) and every p-value row records its type
(``pvalue_type``: hac, wild, permutation, asymptotic_iid, calibrated; Berkowitz / PIT-KS / AG:
asymptotic).

Critic banks (opt-in): with ``SYNTHVAL_BATCHED=1`` the ``power`` and ``realvol`` parts (and E5's
power designs, which call the same functions) train the 2 x ``n_seeds`` anchored critics of a
replication as one bank (``synthval.batched``; ``power_rep_rows_batched``,
``realvol_rep_rows_batched``): same random streams, splits, initial weights, mini-batches and
permutations as the sequential functions, statistics equal up to floating-point rounding
amplified by training (of the order of the difference made by changing the number of CPU threads,
which at the full settings can reach the order of se_hac for a single test: single p-values are
not reproducible across threads, CPU/GPU or banks; rejection rates are). One GPU process then keeps
the GPU busy. The batched mode adds a ``batched`` entry to the parts' checkpoint key, so it never
mixes with sequential blocks; unset, keys and results are unchanged. ``SYNTHVAL_BANK_MAX`` caps
the critics per bank (GPU memory).

Checkpoints: fits are stored under a hash of the fit configuration (E5 reads them there); every
downstream part's checkpoint key includes a hash of the fitted parameter table, so a refit with
different results never reuses stale blocks. Library changes to a model class are detected (a
warning asks for ``--part fit --fresh``); other library changes need ``--fresh``.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import hashlib  # noqa: E402
import inspect  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import common  # noqa: E402  (experiments/ is on sys.path when run as a script)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from synthval import config, data, plotting, stylized  # noqa: E402
from synthval import likelihood as lik  # noqa: E402
from synthval import pipeline as pl  # noqa: E402
from synthval.batched import (AnchoredJob, anchored_relative_tests, bank_max,  # noqa: E402
                              batched_enabled, batched_key)
from synthval.models import get_model  # noqa: E402
from synthval.twosample.critic import time_block_labels, train_eval_split  # noqa: E402

logger = logging.getLogger("experiments.e3")

EXP = "e3"
ROOT_SEED = 20251116
PARTS = ("fit", "power", "realvol", "pooled", "perticker", "perticker_null", "witness",
         "stylized", "likelihood", "agree", "verdict")
DERIVED_PARTS = ("agree", "verdict")
MODELS: tuple[str, ...] = ("GBM", "Merton", "GARCHt", "Heston", "Bootstrap")
LIK_MODELS: tuple[str, ...] = ("GBM", "Merton", "GARCHt")
#: Constructor options of each model (GARCHt: the DESIGN 3.5 experiment settings, which are also
#: the class defaults; written out so that the checkpoint hash records them).
MODEL_KW: dict[str, dict[str, Any]] = {
    "GBM": {}, "Merton": {}, "GARCHt": {"variance_targeting": True, "persist_max": 0.995},
    "Heston": {}, "Bootstrap": {},
}
#: Models fitted with a process pool (the others take < 0.1 s per ticker: serial).
POOLED_FITS: tuple[str, ...] = ("Heston", "Merton")
MODEL_LABELS: dict[str, str] = {"GBM": "GBM", "Merton": "Merton", "GARCHt": "GARCH-t",
                                "Heston": "Heston", "Bootstrap": "Stat. bootstrap",
                                "Real": "Real"}
ABS_TESTS: tuple[str, ...] = tuple(pl.ABSOLUTE_TESTS)
REL_TESTS: tuple[str, ...] = tuple(f"relative_{m}" for m in pl.RELATIVE_METHODS)
ALL_TESTS: tuple[str, ...] = ABS_TESTS + REL_TESTS
TEST_LABELS: dict[str, str] = {**plotting.LABELS_TEST, "relative_anchored": "Anchored rel.",
                               "relative_two_critic": "Two-critic rel."}
TEST_COLORS: dict[str, str] = {**plotting.COLORS_TEST,
                               "relative_anchored": plotting.OKABE_ITO["black"],
                               "relative_two_critic": plotting.OKABE_ITO["grey"]}
TEST_MARKERS: dict[str, str] = {**plotting.MARKERS_TEST, "relative_anchored": "P",
                                "relative_two_critic": "X"}
AG_PAIRS: tuple[tuple[str, str], ...] = (("GARCHt", "GBM"), ("GARCHt", "Merton"),
                                         ("Merton", "GBM"))
ALPHA = pl.ALPHA
P_FLOOR = 1e-10  # display floor of p-values in figures
#: One-sided normal quantile of the lower confidence bound Delta - z se (level 1 - ALPHA).
Z_ONE_SIDED = float(stats.norm.ppf(1.0 - ALPHA))
#: Power designs of the decision rule (``power`` and ``realvol`` parts).
DESIGNS: dict[str, str] = {"D1": "power", "D2": "realvol"}
#: Cases of a power-design replication: generator off by vol x 1.2, and at the boundary.
CASES: tuple[str, ...] = ("alternative", "boundary")
DESIGN_LABELS: dict[str, str] = {
    "D1": r"D1: GARCH-$t$ per ticker, stationary $R_{\mathrm{past}}^*$ (DESIGN)",
    "D2": r"D2: real TRAIN vol.\ path, non-stationary $R_{\mathrm{past}}^*$",
}
#: Relative tests that are not valid for a model, with the reason (flag in tables, no verdict).
#: Verdict entry of a model whose relative test is invalid (not "n/a": pandas reads that as NaN).
NO_VERDICT = "INVALID"
REL_INVALID: dict[str, str] = {
    "Bootstrap": "G copies R_past windows seen by the critic as pool negatives (train/eval "
                 "leakage)",
}
#: Peak resident memory per worker process (GB) at the full settings. Pooled: measured 0.92 GB
#: (MMD / SigMMD kernel matrices of 2 x 3000 windows); critic-only parts measured in the
#: quick-mode probes (printed as MEMORY lines). Used to cap the workers of each part.
WORKER_GB: dict[str, float] = {"pooled": 1.0, "witness": 0.8}
DEFAULT_WORKER_GB = 0.5  # critic-only parts: measured 0.47 GB
#: Hard cap of the pooled / witness workers whatever the free memory (review: 12 x 0.92 GB
#: exceeds the machine's free RAM).
MAX_WORKERS: dict[str, int] = {"pooled": 5, "witness": 5}

# Full configuration (DESIGN section 4, E3) and the smoke-test configuration.
FULL: dict[str, Any] = {
    "n_tickers": None,                 # None: config.N_PANEL tickers (data.panel_tickers())
    "n_seeds": 5,                      # synthetic seeds; also seeds per power replication (10 -> 5: compute budget)
    "power_reps": 100,
    "realvol_reps": 100,
    "perticker_L": 5,
    "perticker_tests": ["critic", "mmd"],
    "perticker_null_draws": 10,        # null pairs per ticker (1000 in all)
    "test_kw": {},                     # pipeline defaults (DESIGN 3.7)
    "base_kw": {},
    "n_boot": pl.N_BOOT,
    "acf_lags": 50,
    "tail_points": 400,
    "blocks": {"pooled": 1, "power": 1, "realvol": 1, "perticker": 10, "perticker_null": 2,
               "witness": 1},
}
_SHORT = {"n_steps": 100, "n_perm": 100}
QUICK: dict[str, Any] = {
    "n_tickers": 6,
    "n_seeds": 2,
    "power_reps": 3,
    "realvol_reps": 3,
    "perticker_L": 5,
    "perticker_tests": ["critic", "mmd"],
    "perticker_null_draws": 2,
    "test_kw": {"critic": dict(_SHORT), "lrcritic": dict(_SHORT), "c2st": dict(_SHORT),
                "mmd": {"n_perm": 100}, "sigmmd": {"n_perm": 100}},
    "base_kw": dict(_SHORT),
    "n_boot": 100,
    "acf_lags": 50,
    "tail_points": 200,
    "blocks": {"pooled": 1, "power": 1, "realvol": 1, "perticker": 5, "perticker_null": 1,
               "witness": 1},
}


# --------------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------------
def seed_int(*keys: Any) -> int:
    """Integer seed determined by ``ROOT_SEED`` and the keys (``SeedSequence`` entropy)."""
    return int(common.keyed_seed_sequence(ROOT_SEED, *keys).generate_state(1, np.uint64)[0])


def synthetic_seed(s: int) -> int:
    """Seed of synthetic seed index ``s`` (shared by all models: common random numbers)."""
    return seed_int("synthetic", int(s))


# --------------------------------------------------------------------------------------------
# Panel and fitted models (per-process caches; workers are spawned, so each loads once)
# --------------------------------------------------------------------------------------------
_PANELS: dict[tuple[str, ...], pl.RealPanel] = {}
_FITS: dict[str, dict[str, Any]] = {}
_SIMS: dict[tuple[str, int], np.ndarray] = {}
_FITTED_THIS_RUN: set[str] = set()


def panel_tickers(cfg: dict) -> list[str]:
    return data.panel_tickers(cfg["n_tickers"] or config.N_PANEL)


def get_panel(tickers: tuple[str, ...] | list[str]) -> pl.RealPanel:
    key = tuple(tickers)
    if key not in _PANELS:
        _PANELS[key] = pl.load_real_panel(list(key))
    return _PANELS[key]


def load_fits(path: str) -> dict[str, Any]:
    if path not in _FITS:
        with open(path, "rb") as fh:
            _FITS[path] = pickle.load(fh)
    return _FITS[path]


def headline_sim(fit_path: str, panel: pl.RealPanel) -> np.ndarray:
    """The headline (seed 0) synthetic TEST path of every ticker, exactly as in the pooled part.

    ``pipeline.evaluate_model(seed=s)`` simulates with child 0 of ``SeedSequence(s)``.
    """
    key = (fit_path, panel.n_test)
    if key not in _SIMS:
        ss_sim, _ = np.random.SeedSequence(synthetic_seed(0)).spawn(2)
        _SIMS[key] = pl.simulate_panel(load_fits(fit_path), panel.n_test, seed=ss_sim,
                                       start_from_end=True)
    return _SIMS[key]


def fit_config(model: str, tickers: list[str]) -> dict[str, Any]:
    return {"model": model, "model_kw": MODEL_KW[model], "tickers": list(tickers),
            "train": list(config.TRAIN), "fit_seed": None}


def fit_paths(ctx: common.ExpContext, model: str, tickers: list[str]) -> tuple[Path, Path, str]:
    """Pickle, fit table and configuration hash of the fits (E5 reads the same location)."""
    ck = common.Checkpoint(ctx.ckpt_dir, f"fit_{model}", fit_config(model, tickers),
                           fresh=ctx.fresh)
    return ck.file("fitted.pkl"), ck.file("fit_table.csv"), ck.hash


def model_source_hash(model: str) -> str:
    """Hash of the source of the model class (and its synthval base classes)."""
    cls = type(get_model(model, **MODEL_KW[model]))
    src = "".join(inspect.getsource(c) for c in cls.__mro__
                  if c.__module__.startswith("synthval"))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


_SOURCE_WARNED: set[str] = set()


def _check_fit_source(model: str, pkl: Path) -> None:
    """Warn (once per process) if the model class changed since the cached fits were made."""
    if str(pkl) in _SOURCE_WARNED:
        return
    _SOURCE_WARNED.add(str(pkl))
    sfile = pkl.with_name("model_source.sha")
    try:
        cur = model_source_hash(model)
    except (OSError, TypeError) as exc:  # pragma: no cover - source not available
        logger.debug("no source hash for %s: %s", model, exc)
        return
    old = sfile.read_text().strip() if sfile.exists() else ""
    if old != cur:
        logger.warning("the %s model class changed since its fits were cached (%s): rerun "
                       "'--part fit --fresh' if the change affects fitting or simulation",
                       model, "no record" if not old else f"{old} -> {cur}")


def ensure_fits(ctx: common.ExpContext, model: str, tickers: list[str]
                ) -> tuple[str, pd.DataFrame, float]:
    """Path of the pickled per-ticker fits of ``model`` (fitting them if needed), fit table,
    wall seconds spent fitting in this call (0 when loaded from the cache).

    ``--fresh`` refits only when the ``fit`` part is selected (once per process), so parallel
    processes running other parts with ``--fresh`` never refit.
    """
    pkl, csv, _ = fit_paths(ctx, model, tickers)
    refit = (ctx.fresh and "fit" in ctx.parts and not ctx.plots_only
             and str(pkl) not in _FITTED_THIS_RUN)
    if pkl.exists() and csv.exists() and not refit:
        if str(pkl) not in _FITTED_THIS_RUN:
            _check_fit_source(model, pkl)
        return str(pkl), pd.read_csv(csv, index_col="ticker"), 0.0
    panel = get_panel(tickers)
    t0 = time.perf_counter()
    # Only Heston (~20 s per ticker) and Merton (~0.2 s) are worth a process pool.
    pool = model in POOLED_FITS and len(tickers) > 1
    fitted, table = pl.fit_panel(model, panel.train, n_workers=ctx.workers if pool else 1,
                                 **MODEL_KW[model])
    wall = time.perf_counter() - t0
    # The table goes first so that a crash in between leaves no table (=> refit next time).
    csv.unlink(missing_ok=True)
    tmp = pkl.with_name(f".{pkl.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(fitted, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, pkl)
    _atomic_write(pkl.with_name("model_source.sha"), model_source_hash(model) + "\n")
    _atomic_write(csv, table.to_csv())
    _FITTED_THIS_RUN.add(str(pkl))
    _FITS.pop(str(pkl), None)
    for k in [k for k in _SIMS if k[0] == str(pkl)]:
        _SIMS.pop(k)
    logger.info("fitted %s on %d tickers in %s", model, len(tickers), common.fmt_seconds(wall))
    return str(pkl), pd.read_csv(csv, index_col="ticker"), wall


def all_fit_paths(ctx: common.ExpContext, tickers: list[str], models=MODELS) -> dict[str, str]:
    return {m: ensure_fits(ctx, m, tickers)[0] for m in models}


def fit_content_hash(csv: Path) -> str:
    """Hash of the fitted parameter table (fit timings excluded)."""
    t = pd.read_csv(csv, index_col="ticker")
    t = t.drop(columns=[c for c in t.columns if "second" in c])
    return hashlib.sha256(t.to_csv(float_format="%.10g").encode()).hexdigest()[:12]


def fit_hashes(ctx: common.ExpContext, tickers: list[str], models=MODELS) -> dict[str, str]:
    """Per model ``<configuration hash>:<fitted-content hash>`` (downstream checkpoint keys).

    The content part changes whenever a refit gives different parameters, so downstream parts
    never reuse blocks computed from other fits. Call after :func:`ensure_fits`.
    """
    out = {}
    for m in models:
        _, csv, h = fit_paths(ctx, m, tickers)
        out[m] = f"{h}:{fit_content_hash(csv)}" if csv.exists() else h
    return out


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def mem_available_gb() -> float:
    """``MemAvailable`` of ``/proc/meminfo`` in GB (NaN if unknown)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0 ** 2
    except (OSError, ValueError):
        pass
    return float("nan")


def memory_workers(part: str, workers: int, avail_gb: float | None = None) -> int:
    """Workers that fit in memory for ``part``: ``min(workers, MAX_WORKERS[part],
    0.85 x MemAvailable / WORKER_GB[part])`` (at least 1)."""
    gb = WORKER_GB.get(part, DEFAULT_WORKER_GB)
    avail = mem_available_gb() if avail_gb is None else avail_gb
    cap = int(0.85 * avail // gb) if np.isfinite(avail) else workers
    return max(1, min(int(workers), MAX_WORKERS.get(part, int(workers)), cap))


def part_workers(ctx: common.ExpContext, part: str) -> int:
    w = memory_workers(part, ctx.workers)
    if w < ctx.workers:
        logger.warning("%s: %d workers instead of %d (memory: about %.1f GB per worker, "
                       "%.1f GB available%s)", part, w, ctx.workers,
                       WORKER_GB.get(part, DEFAULT_WORKER_GB), mem_available_gb(),
                       f", cap {MAX_WORKERS[part]}" if part in MAX_WORKERS else "")
    return w


# --------------------------------------------------------------------------------------------
# Part: fit
# --------------------------------------------------------------------------------------------
def heston_moment_rows(fit_path: str) -> list[dict[str, Any]]:
    rows = []
    for ticker, mdl in load_fits(fit_path).items():
        info = mdl.fit_info_
        names = list(info["moment_names"])
        for j, nm in enumerate(names):
            rows.append({"ticker": ticker, "moment": nm, "data": float(info["m_data"][j]),
                         "sim": float(info["m_sim"][j]), "t": float(info["t"][j]),
                         "weight": float(info["weights"][j])})
        rows.append({"ticker": ticker, "moment": "kurtosis (not targeted)",
                     "data": float(info["kurt_data"]), "sim": float(info["kurt_sim"]),
                     "t": float("nan"), "weight": 0.0})
    return rows


def run_fit(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    timing: dict[str, float] = {}
    for m in sorted(MODELS, key=lambda x: x != "Heston"):  # longest first
        path, table, wall = ensure_fits(ctx, m, tickers)
        timing[f"fit_{m}_wall_s"] = wall
        timing[f"fit_{m}_cpu_s"] = float(table["fit_seconds"].sum())
        ctx.save_csv(table.reset_index(), f"e3_fit_{m}",
                     config={"part": "fit", **fit_config(m, tickers)})
        if m == "Heston":
            ctx.save_csv(pd.DataFrame(heston_moment_rows(path)), "e3_heston_moments_raw",
                         config={"part": "fit", **fit_config(m, tickers)})
    ctx.write_part_meta("fit", {"models": list(MODELS), "model_kw": MODEL_KW,
                                "tickers": tickers}, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part: power (pre-specified power check)
# --------------------------------------------------------------------------------------------
def _anchored_row(Wn: np.ndarray, Wg: np.ndarray, Wp: np.ndarray, orders: tuple, *,
                  seed: np.random.SeedSequence, c: dict) -> dict[str, Any]:
    """Anchored relative test with the E3 settings; the test streams come from ``seed``."""
    r = pl.relative_tests(Wn, Wg, Wp, orders=orders, base="critic", methods=("anchored",),
                          protocol="dependent", rng=np.random.default_rng(seed),
                          base_kw=c["base_kw"], n_boot=c["n_boot"], keep_extra=True)[0]
    ts = (r.pop("extra", None) or {}).get("train_summary") or {}
    return {"pvalue": r["pvalue"], "pvalue_type": r["pvalue_type"], "statistic": r["statistic"],
            "p_perm": r["p_perm"], "p_hac": r["p_hac"], "z_hac": r["z_hac"],
            "se_hac": r["se_hac"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
            "steps_run": ts.get("steps_run", np.nan), "test_seconds": r["runtime_s"]}


def _power_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Power design D1, replication ``rep``: one (R_past*, R_new*), ``n_seeds`` synthetic seeds.

    Streams: ``ss = SeedSequence(seed_int("power", rep)).spawn(1)[0]`` and its five children
    ``(s_past, s_new, s_g, s_test, s_bd)`` are those of ``pipeline.power_check(n_reps=1,
    seed=seed_int("power", rep))`` (and of E5's power replications), so seed 0 of the
    alternative reproduces that call bit for bit. Seeds ``s >= 1`` use streams keyed by
    ``("power", rep, "seed", s)``. Per seed the alternative G* (vol x 1.2, end-of-TRAIN start)
    and the boundary G0 (stationary start, the law of R_past*) are tested with the same test
    streams (common random numbers).
    """
    del rng
    fn = power_rep_rows_batched if cell["_cfg"].get("batched") else power_rep_rows
    return fn(get_panel(cell["_tickers"]), load_fits(cell["_fit_path"]), rep, cell["_cfg"])


def power_rep_rows(panel: pl.RealPanel, fitted: dict, rep: int, c: dict
                   ) -> list[dict[str, Any]]:
    """Body of :func:`_power_rep` (also used by the timing probe)."""
    sc, L, burn = panel.scale_array(), panel.L, pl.BURN_IN
    n_tr, n_te = panel.n_train, panel.n_test
    ss = np.random.SeedSequence(seed_int("power", rep)).spawn(1)[0]
    s_past, s_new, s_g, s_test, s_bd = ss.spawn(5)
    t0 = time.perf_counter()
    Rp = pl.simulate_panel(fitted, n_tr, seed=s_past, start_from_end=False, burn_in=burn)
    Rn = pl.simulate_panel(fitted, n_te, seed=s_new, start_from_end=True, burn_in=burn)
    t_rep = time.perf_counter() - t0
    Wp, _, op = pl.panel_windows(Rp, L, panel.step, scale=sc, date_offset=0)
    Wn, _, on = pl.panel_windows(Rn, L, panel.step, scale=sc, date_offset=n_tr)
    rows: list[dict[str, Any]] = []
    for s in range(int(c["n_seeds"])):
        if s == 0:
            sg, st, sb = s_g, s_test, s_bd
        else:
            sg, st, sb = common.keyed_seed_sequence(ROOT_SEED, "power", rep, "seed", s).spawn(3)
        t0 = time.perf_counter()
        G = pl.POWER_VOL_FACTOR * pl.simulate_panel(fitted, n_te, seed=sg, start_from_end=True,
                                                    burn_in=burn)
        G0 = pl.simulate_panel(fitted, n_te, seed=sb, start_from_end=False, burn_in=burn)
        t_seed = time.perf_counter() - t0
        for case, sim in (("alternative", G), ("boundary", G0)):
            if case not in c.get("cases", CASES):
                continue
            Wg, _, og = pl.panel_windows(sim, L, panel.step, scale=sc, date_offset=n_tr)
            rows.append({"design": "D1", "seed_index": s, "case": case,
                         **_anchored_row(Wn, Wg, Wp, (on, og, op), seed=st, c=c),
                         "sim_seconds_rep": t_rep, "sim_seconds_seed": t_seed})
    return rows


def power_config(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, Any]:
    return {"tickers": tickers, "fit_hash": fit_hashes(ctx, tickers, (pl.POWER_MODEL,)),
            "base_kw": cfg["base_kw"], "n_boot": cfg["n_boot"], "root_seed": ROOT_SEED,
            "n_seeds": cfg["n_seeds"], "vol_factor": pl.POWER_VOL_FACTOR,
            "boundary_start": "stationary", "alternative_start": "end_of_train",
            "method": "anchored", "protocol": "dependent", "design": "D1",
            "pipeline": pl.pipeline_config(), **batched_key()}


def _rule_counts(raw: pd.DataFrame, case: str) -> dict[str, Any]:
    """Rejection counts of one case: single seed (seed 0, one test per replication) and the
    decision rule (twice the median p over the seeds of a replication)."""
    g = raw[raw["case"] == case]
    if g.empty:
        return {}
    p0 = g.loc[g["seed_index"] == 0, "pvalue"].to_numpy(float)
    p0 = p0[np.isfinite(p0)]
    comb = g.groupby("rep")["pvalue"].apply(lambda s: pl.median_pvalue(s.to_numpy(float)))
    comb = comb.to_numpy(float)
    comb = comb[np.isfinite(comb)]
    n0, k0, k0h = p0.size, int(np.sum(p0 <= ALPHA)), int(np.sum(p0 <= ALPHA / 2))
    n, k = comb.size, int(np.sum(comb <= ALPHA))
    lo0, hi0 = pl.wilson_ci(k0, n0)
    lo, hi = pl.wilson_ci(k, n)
    cplo, cphi = clopper_pearson_ci(k, n)
    cplo0, cphi0 = clopper_pearson_ci(k0, n0)
    return {"n_reps": n, "n_seeds": int(g.groupby("rep")["seed_index"].nunique().median()),
            "n_reject_rule": k, "rate_rule": k / n if n else np.nan, "rule_ci_lo": lo,
            "rule_ci_hi": hi, "rule_cp_lo": cplo, "rule_cp_hi": cphi,
            "rule_cp_lower": pl.clopper_pearson_lower(k, n),
            "n_single": n0, "n_reject_single": k0, "rate_single": k0 / n0 if n0 else np.nan,
            "single_ci_lo": lo0, "single_ci_hi": hi0, "single_cp_lo": cplo0,
            "single_cp_hi": cphi0, "rate_single_half_alpha": k0h / n0 if n0 else np.nan,
            "rate_all_seeds": float(np.mean(g["pvalue"].to_numpy(float) <= ALPHA)),
            "median_se_hac": float(np.nanmedian(g["se_hac"])),
            "median_statistic": float(np.nanmedian(g["statistic"])),
            "median_steps_run": float(np.nanmedian(g["steps_run"]))}


def clopper_pearson_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Two-sided Clopper-Pearson (1934) interval for ``k / n``."""
    if n <= 0:
        return float("nan"), float("nan")
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def design_summary(raw: pd.DataFrame, design: str) -> pd.DataFrame:
    """One row per case (alternative = power, boundary = size) of a power design.

    ``*_rule`` columns: the decision rule's p-value (twice the median over the seeds of a
    replication), the quantity the verdict uses; ``*_single``: one seed per replication (seed
    0), the uncalibrated single-seed test; ``rate_single_half_alpha`` = P(p <= alpha / 2), the
    review's stop-gap approximation of the rule's power.
    """
    rows = []
    for case in ("alternative", "boundary"):
        d = _rule_counts(raw, case)
        if d:
            rows.append({"design": design, "case": case, "alpha": ALPHA,
                         "vol_factor": pl.POWER_VOL_FACTOR if case == "alternative" else 1.0,
                         "method": "anchored", "base": "critic", "pvalue_type": "hac",
                         "rule": "2*median p over seeds", **d,
                         "median_rep_seconds": float(raw.groupby("rep")["rep_seconds"].first()
                                                     .median())
                         if "rep_seconds" in raw else np.nan})
    return pd.DataFrame(rows)


def _log_design(name: str, summ: pd.DataFrame) -> None:
    for _, s in summ.iterrows():
        logger.info("%s %s: rule %d/%d = %.3f [%.3f, %.3f] (CP lower %.3f); single seed "
                    "%d/%d = %.3f [%.3f, %.3f]; median se_hac %.4f", name, s["case"],
                    s["n_reject_rule"], s["n_reps"], s["rate_rule"], s["rule_ci_lo"],
                    s["rule_ci_hi"], s["rule_cp_lower"], s["n_reject_single"], s["n_single"],
                    s["rate_single"], s["single_ci_lo"], s["single_ci_hi"], s["median_se_hac"])


def run_power(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    t0 = time.perf_counter()
    fpath = ensure_fits(ctx, pl.POWER_MODEL, tickers)[0]
    conf = power_config(ctx, cfg, tickers)
    cell = {"cell": "power", "_tickers": list(tickers), "_fit_path": fpath,
            "_cfg": {"base_kw": cfg["base_kw"], "n_boot": cfg["n_boot"],
                     "n_seeds": cfg["n_seeds"], **_batched_flag()}}
    df = common.run_cells(_power_rep, [cell], part="power", ctx=ctx, root_seed=ROOT_SEED,
                          n_reps=cfg["power_reps"], config=conf,
                          block_size=cfg["blocks"]["power"], workers=part_workers(ctx, "power"))
    ctx.save_csv(df, "e3_power_raw", config={"part": "power", **conf})
    summ = design_summary(df, "D1")
    ctx.save_csv(summ, "e3_power_summary", config={"part": "power", **conf})
    _log_design("power D1", summ)
    timing = {"power_wall_s": time.perf_counter() - t0}
    ctx.write_part_meta("power", {**conf, "n_reps": cfg["power_reps"]}, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part: realvol (power design D2: the real TRAIN volatility path; non-stationary placebo)
# --------------------------------------------------------------------------------------------
def realvol_inputs(panel: pl.RealPanel) -> dict[str, np.ndarray]:
    """Real TEST windows, TRAIN window orders and the TRAIN volatility path ``v`` (N, K_train).

    ``v[i, k]`` is the realised volatility (root mean square) of the normalised real TRAIN
    window ``k`` of ticker ``i``; the windows of :func:`pipeline.panel_windows` are
    ticker-major, so windows ``i * K .. (i + 1) * K - 1`` belong to ticker ``i``.
    """
    Xp, gp, op = panel.windows("train")
    Xn, gn, on = panel.windows("test")
    N = panel.n_tickers
    v = np.sqrt(np.mean(Xp * Xp, axis=1)).reshape(N, -1)
    if not np.array_equal(gp, np.repeat(np.arange(N), v.shape[1])):
        raise RuntimeError("TRAIN windows are not ticker-major")
    return {"Xn": Xn, "on": on, "op": op, "v": v, "k_test": Xn.shape[0] // N}


def _realvol_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Design D2, replication ``rep``: one R_past* draw, ``n_seeds`` seeds of (G0, 1.2 G0).

    R_past* window (i, k) = v[i, k] z (z ~ N(0, I_L)); G0 window (i, j) = v[i, U] z' with U
    uniform over ticker i's TRAIN windows (the pooled TRAIN window law of ticker i), at the TEST
    date ranks; R_new = real TEST windows. Streams keyed by ``("realvol", rep)`` (R_past*) and
    ``("realvol", rep, "seed", s)`` (G0, test).
    """
    del rng
    fn = realvol_rep_rows_batched if cell["_cfg"].get("batched") else realvol_rep_rows
    return fn(get_panel(cell["_tickers"]), rep, cell["_cfg"])


def realvol_rep_rows(panel: pl.RealPanel, rep: int, c: dict) -> list[dict[str, Any]]:
    """Body of :func:`_realvol_rep` (also used by the timing probe)."""
    inp = realvol_inputs(panel)
    v, L, kt = inp["v"], panel.L, inp["k_test"]
    N, K = v.shape
    rp = np.random.default_rng(common.keyed_seed_sequence(ROOT_SEED, "realvol", rep))
    Rp = (v[:, :, None] * rp.standard_normal((N, K, L))).reshape(N * K, L)
    rows: list[dict[str, Any]] = []
    for s in range(int(c["n_seeds"])):
        sg, st = common.keyed_seed_sequence(ROOT_SEED, "realvol", rep, "seed", s).spawn(2)
        rg = np.random.default_rng(sg)
        vg = np.take_along_axis(v, rg.integers(0, K, size=(N, kt)), axis=1)
        G0 = (vg[:, :, None] * rg.standard_normal((N, kt, L))).reshape(N * kt, L)
        orders = (inp["on"], inp["on"], inp["op"])
        for case, G in (("alternative", pl.POWER_VOL_FACTOR * G0), ("boundary", G0)):
            if case not in c.get("cases", CASES):
                continue
            rows.append({"design": "D2", "seed_index": s, "case": case,
                         **_anchored_row(inp["Xn"], G, Rp, orders, seed=st, c=c)})
    return rows


def realvol_config(cfg: dict, tickers: list[str]) -> dict[str, Any]:
    return {"tickers": tickers, "base_kw": cfg["base_kw"], "n_boot": cfg["n_boot"],
            "root_seed": ROOT_SEED, "n_seeds": cfg["n_seeds"],
            "vol_factor": pl.POWER_VOL_FACTOR, "method": "anchored", "protocol": "dependent",
            "design": "D2 (real TRAIN vol path x N(0, I_L); G0 = pooled TRAIN window law)",
            "pipeline": pl.pipeline_config(), **batched_key()}


def run_realvol(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    t0 = time.perf_counter()
    conf = realvol_config(cfg, tickers)
    cell = {"cell": "realvol", "_tickers": list(tickers),
            "_cfg": {"base_kw": cfg["base_kw"], "n_boot": cfg["n_boot"],
                     "n_seeds": cfg["n_seeds"], **_batched_flag()}}
    df = common.run_cells(_realvol_rep, [cell], part="realvol", ctx=ctx, root_seed=ROOT_SEED,
                          n_reps=cfg["realvol_reps"], config=conf,
                          block_size=cfg["blocks"]["realvol"],
                          workers=part_workers(ctx, "realvol"))
    ctx.save_csv(df, "e3_realvol_raw", config={"part": "realvol", **conf})
    summ = design_summary(df, "D2")
    ctx.save_csv(summ, "e3_realvol_summary", config={"part": "realvol", **conf})
    _log_design("realvol D2", summ)
    timing = {"realvol_wall_s": time.perf_counter() - t0}
    ctx.write_part_meta("realvol", {**conf, "n_reps": cfg["realvol_reps"]}, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Power designs with a critic bank (opt-in: SYNTHVAL_BATCHED=1; synthval.batched)
# --------------------------------------------------------------------------------------------
# The sequential functions above stay the default and are not edited: E5 keys its checkpoints on
# their source. The batched versions below give the same rows (same streams, splits, initial
# weights, mini-batches and permutations; statistics equal up to the rounding of the batched
# training) but train the 2 x n_seeds critics of a replication as one bank, which keeps a GPU
# busy from one process. tests/test_batched.py checks them against the sequential functions.
def _batched_flag() -> dict[str, bool]:
    """``{"batched": True}`` for a part's ``_cfg`` when ``SYNTHVAL_BATCHED`` is on, else ``{}``."""
    return {"batched": True} if batched_enabled() else {}


def _anchored_rows_batched(jobs: list[tuple[dict, tuple, dict]], c: dict
                           ) -> list[dict[str, Any]]:
    """Rows ``{**head, **_anchored_row(Wn, Wg, Wp, orders, seed=seed, c=c), **tail}`` of every
    job ``(head, (Wn, Wg, Wp, orders, seed), tail)``, with all critics trained as one bank
    (at most ``$SYNTHVAL_BANK_MAX`` critics per bank). ``test_seconds`` is the test's share of
    the bank time plus its own split and inference time."""
    if not jobs:
        return []
    res = anchored_relative_tests(
        [AnchoredJob(Wn, Wg, Wp, rng=np.random.default_rng(seed), orders=orders)
         for _, (Wn, Wg, Wp, orders, seed), _ in jobs],
        base="critic", protocol="dependent", base_kw=c["base_kw"], n_boot=c["n_boot"],
        keep_extra=True, max_bank=bank_max())
    rows = []
    for (head, _, tail), r in zip(jobs, res):
        ts = (r.pop("extra", None) or {}).get("train_summary") or {}
        rows.append({**head, "pvalue": r["pvalue"], "pvalue_type": r["pvalue_type"],
                     "statistic": r["statistic"], "p_perm": r["p_perm"], "p_hac": r["p_hac"],
                     "z_hac": r["z_hac"], "se_hac": r["se_hac"], "ci_lo": r["ci_lo"],
                     "ci_hi": r["ci_hi"], "steps_run": ts.get("steps_run", np.nan),
                     "test_seconds": r["runtime_s"], **tail})
    return rows


def power_rep_rows_batched(panel: pl.RealPanel, fitted: dict, rep: int, c: dict
                           ) -> list[dict[str, Any]]:
    """:func:`power_rep_rows` with the replication's 2 x ``n_seeds`` critics as one bank."""
    sc, L, burn = panel.scale_array(), panel.L, pl.BURN_IN
    n_tr, n_te = panel.n_train, panel.n_test
    ss = np.random.SeedSequence(seed_int("power", rep)).spawn(1)[0]
    s_past, s_new, s_g, s_test, s_bd = ss.spawn(5)
    t0 = time.perf_counter()
    Rp = pl.simulate_panel(fitted, n_tr, seed=s_past, start_from_end=False, burn_in=burn)
    Rn = pl.simulate_panel(fitted, n_te, seed=s_new, start_from_end=True, burn_in=burn)
    t_rep = time.perf_counter() - t0
    Wp, _, op = pl.panel_windows(Rp, L, panel.step, scale=sc, date_offset=0)
    Wn, _, on = pl.panel_windows(Rn, L, panel.step, scale=sc, date_offset=n_tr)
    jobs: list[tuple[dict, tuple, dict]] = []
    for s in range(int(c["n_seeds"])):
        if s == 0:
            sg, st, sb = s_g, s_test, s_bd
        else:
            sg, st, sb = common.keyed_seed_sequence(ROOT_SEED, "power", rep, "seed", s).spawn(3)
        t0 = time.perf_counter()
        G = pl.POWER_VOL_FACTOR * pl.simulate_panel(fitted, n_te, seed=sg, start_from_end=True,
                                                    burn_in=burn)
        G0 = pl.simulate_panel(fitted, n_te, seed=sb, start_from_end=False, burn_in=burn)
        t_seed = time.perf_counter() - t0
        for case, sim in (("alternative", G), ("boundary", G0)):
            if case not in c.get("cases", CASES):
                continue
            Wg, _, og = pl.panel_windows(sim, L, panel.step, scale=sc, date_offset=n_tr)
            jobs.append(({"design": "D1", "seed_index": s, "case": case},
                         (Wn, Wg, Wp, (on, og, op), st),
                         {"sim_seconds_rep": t_rep, "sim_seconds_seed": t_seed}))
    return _anchored_rows_batched(jobs, c)


def realvol_rep_rows_batched(panel: pl.RealPanel, rep: int, c: dict) -> list[dict[str, Any]]:
    """:func:`realvol_rep_rows` with the replication's 2 x ``n_seeds`` critics as one bank."""
    inp = realvol_inputs(panel)
    v, L, kt = inp["v"], panel.L, inp["k_test"]
    N, K = v.shape
    rp = np.random.default_rng(common.keyed_seed_sequence(ROOT_SEED, "realvol", rep))
    Rp = (v[:, :, None] * rp.standard_normal((N, K, L))).reshape(N * K, L)
    jobs: list[tuple[dict, tuple, dict]] = []
    for s in range(int(c["n_seeds"])):
        sg, st = common.keyed_seed_sequence(ROOT_SEED, "realvol", rep, "seed", s).spawn(2)
        rg = np.random.default_rng(sg)
        vg = np.take_along_axis(v, rg.integers(0, K, size=(N, kt)), axis=1)
        G0 = (vg[:, :, None] * rg.standard_normal((N, kt, L))).reshape(N * kt, L)
        orders = (inp["on"], inp["on"], inp["op"])
        for case, G in (("alternative", pl.POWER_VOL_FACTOR * G0), ("boundary", G0)):
            if case not in c.get("cases", CASES):
                continue
            jobs.append(({"design": "D2", "seed_index": s, "case": case},
                         (inp["Xn"], G, Rp, orders, st), {}))
    return _anchored_rows_batched(jobs, c)


# --------------------------------------------------------------------------------------------
# Part: pooled
# --------------------------------------------------------------------------------------------
def _pooled_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Absolute + relative tests of one model at synthetic seed index ``rep``."""
    panel = get_panel(cell["_tickers"])
    fitted = load_fits(cell["_fit_path"])
    c = cell["_cfg"]
    rows = pl.evaluate_model(panel, fitted, seed=synthetic_seed(rep), tests=ABS_TESTS,
                             relative_base="critic", relative_methods=pl.RELATIVE_METHODS,
                             protocol="dependent", test_kw=c["test_kw"], base_kw=c["base_kw"],
                             n_boot=c["n_boot"], start_from_end=True, all_pvalues=True,
                             keep_extra=True)
    for r in rows:
        r.update(_train_steps(r.pop("extra", None) or {}))
        r["seed"] = str(r["seed"])
        r["max_rss_mb"] = _max_rss_mb()
    return rows


def _train_steps(ex: dict) -> dict[str, float]:
    """Optimiser steps run (and best step under early stopping) of a learned test's critic(s)."""
    nan = float("nan")
    if "steps_run" in ex:  # absolute learned tests (fit_witness info merged into extra)
        return {"steps_run": float(ex["steps_run"]), "best_step": float(ex.get("best_step", nan)),
                "steps_run_2": nan}
    if "train_summary" in ex:  # anchored relative test
        ts = ex["train_summary"] or {}
        return {"steps_run": float(ts.get("steps_run") or nan),
                "best_step": float(ts.get("best_step", nan)), "steps_run_2": nan}
    if "train_info_g" in ex:  # two-critic: critic (R_new, G) and critic (R_new, R_past)
        a, b = ex.get("train_info_g") or {}, ex.get("train_info_past") or {}
        return {"steps_run": float(a.get("steps_run", nan)),
                "best_step": float(a.get("best_step", nan)),
                "steps_run_2": float(b.get("steps_run", nan))}
    return {"steps_run": nan, "best_step": nan, "steps_run_2": nan}


def pooled_config(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, Any]:
    return {"tickers": tickers, "fit_hash": fit_hashes(ctx, tickers), "tests": list(ABS_TESTS),
            "methods": list(pl.RELATIVE_METHODS), "test_kw": cfg["test_kw"],
            "base_kw": cfg["base_kw"], "n_boot": cfg["n_boot"], "protocol": "dependent",
            "root_seed": ROOT_SEED, "pipeline": pl.pipeline_config(), "train_steps": True}


def run_pooled(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    t0 = time.perf_counter()
    paths = all_fit_paths(ctx, tickers)
    conf = pooled_config(ctx, cfg, tickers)
    cells = [{"cell": f"model={m}", "model": m, "_tickers": list(tickers), "_fit_path": paths[m],
              "_cfg": cfg} for m in MODELS]
    df = common.run_cells(_pooled_rep, cells, part="pooled", ctx=ctx, root_seed=ROOT_SEED,
                          n_reps=cfg["n_seeds"], config=conf,
                          block_size=cfg["blocks"]["pooled"], workers=part_workers(ctx, "pooled"))
    ctx.save_csv(df, "e3_pooled_raw", config={"part": "pooled", **conf,
                                              "n_seeds": cfg["n_seeds"]})
    ctx.save_csv(pooled_summary(df), "e3_pooled_summary", config={"part": "pooled", **conf})
    timing = {"pooled_wall_s": time.perf_counter() - t0}
    ctx.write_part_meta("pooled", {**conf, "n_seeds": cfg["n_seeds"]}, timing)
    return timing


def validity(model: str, test: str, valid_flag: bool) -> tuple[bool, str]:
    """Whether a pooled result is a valid test, and why not (tables flag invalid rows)."""
    if test == "ks":
        return False, "KS assumes i.i.d. returns"
    if test == "relative_two_critic":
        return False, "two-critic relative test is anti-conservative at the boundary (secondary)"
    if test.startswith("relative_") and model in REL_INVALID:
        return False, REL_INVALID[model]
    return bool(valid_flag), ""


def pooled_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Per model and test, over synthetic seeds: ``p_combined`` (twice the median, the valid
    combination), median p and statistic, per-seed rejection share (descriptive; for the
    anchored test the single-seed p-values are not calibrated on real data), seed-0 values."""
    rows = []
    for (m, t), g in raw.groupby(["model", "test"], sort=False):
        g = g.sort_values("rep")
        p = g["pvalue"].to_numpy(float)
        ok = np.isfinite(p)
        n, k = int(ok.sum()), int(np.sum(p[ok] <= ALPHA))
        lo, hi = pl.wilson_ci(k, n)
        g0 = g[g["rep"] == 0]
        r0 = g0.iloc[0] if len(g0) else None
        valid, note = validity(m, t, bool(g["valid"].iloc[0]))
        st = g["statistic"].to_numpy(float)
        rows.append({
            "model": m, "test": t, "kind": g["kind"].iloc[0],
            "pvalue_type": g["pvalue_type"].iloc[0], "valid": valid, "valid_note": note,
            "n_seeds": n, "statistic_seed0": float(r0["statistic"]) if r0 is not None else np.nan,
            "statistic_median": float(np.nanmedian(st)),
            "statistic_min": float(np.nanmin(st)), "statistic_max": float(np.nanmax(st)),
            "p_seed0": float(r0["pvalue"]) if r0 is not None else np.nan,
            "p_median": float(np.nanmedian(p)) if n else np.nan,
            "p_combined": pl.median_pvalue(p), "n_reject": k,
            "share_reject": k / n if n else np.nan, "share_lo": lo, "share_hi": hi,
            "ci_lo_seed0": float(r0["ci_lo"]) if r0 is not None and "ci_lo" in r0 else np.nan,
            "ci_hi_seed0": float(r0["ci_hi"]) if r0 is not None and "ci_hi" in r0 else np.nan,
            "p_perm_seed0": float(r0["p_perm"]) if r0 is not None else np.nan,
            "p_wild_seed0": float(r0["p_wild"]) if r0 is not None and "p_wild" in r0 else np.nan,
            "median_se_hac": (float(np.nanmedian(g["se_hac"])) if "se_hac" in g
                              and g["se_hac"].notna().any() else np.nan),
            "median_steps_run": (float(np.nanmedian(g["steps_run"])) if "steps_run" in g
                                 and g["steps_run"].notna().any() else np.nan),
            "median_runtime_s": float(np.nanmedian(g["runtime_s"])),
        })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODELS, ordered=True)
    out["test"] = pd.Categorical(out["test"], ALL_TESTS, ordered=True)
    return out.sort_values(["model", "test"]).reset_index(drop=True).astype(
        {"model": str, "test": str})


# --------------------------------------------------------------------------------------------
# Part: per-ticker tests at L = 5
# --------------------------------------------------------------------------------------------
def _ticker_tests(X: np.ndarray, Y: np.ndarray, ox: np.ndarray, oy: np.ndarray, c: dict,
                  seed: Any, what: str) -> list[dict[str, Any]]:
    """The per-ticker tests (dependent protocol) on one pair of window samples.

    Every test gets a fresh generator from ``seed`` (as one joint ``absolute_tests`` call would
    give); a failure (small samples) is recorded as a NaN row.
    """
    rows: list[dict[str, Any]] = []
    for name in c["tests"]:
        kw = c["test_kw"].get(name)
        t0 = time.perf_counter()
        try:
            r = pl.absolute_tests(X, Y, order_x=ox, order_y=oy, tests=(name,),
                                  protocol="dependent", rng=np.random.default_rng(seed),
                                  test_kw={name: kw} if kw else None, all_pvalues=True)[0]
            r["error"] = ""
        except Exception as exc:  # noqa: BLE001 - small samples: record and continue
            logger.warning("per-ticker %s %s failed: %s", what, name, exc)
            r = {"test": name, "statistic": np.nan, "pvalue": np.nan,
                 "pvalue_type": "", "runtime_s": time.perf_counter() - t0,
                 "error": f"{type(exc).__name__}: {exc}"}
        rows.append(r)
    return rows


def _ticker_windows(panel: pl.RealPanel, ticker: str, r: np.ndarray, L: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """L-windows of one ticker's TEST-horizon returns (real scale, TEST date ranks)."""
    sc = panel.scale.loc[[ticker]].to_numpy(dtype=np.float64)
    W, _, o = pl.panel_windows(np.asarray(r, dtype=np.float64).reshape(-1, 1), L, scale=sc,
                               date_offset=panel.n_train)
    return W, o


def _perticker_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Critic and MMD for ticker number ``rep`` (headline synthetic path, dependent protocol)."""
    del rng
    panel = get_panel(cell["_tickers"])
    c = cell["_cfg"]
    sim = headline_sim(cell["_fit_path"], panel)
    ticker = panel.tickers[rep]
    L = int(c["L"])
    X, ox = _ticker_windows(panel, ticker, panel.test[ticker].to_numpy(), L)
    Y, oy = _ticker_windows(panel, ticker, sim[:, rep], L)
    seed = common.keyed_seed_sequence(ROOT_SEED, "perticker", ticker)
    rows = _ticker_tests(X, Y, ox, oy, c, seed, f"{cell['model']} {ticker}")
    for r in rows:
        r["ticker"] = ticker
        r["L"] = L
    return rows


def perticker_config(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, Any]:
    return {"tickers": tickers, "fit_hash": fit_hashes(ctx, tickers),
            "tests": cfg["perticker_tests"], "L": cfg["perticker_L"],
            "test_kw": {k: v for k, v in cfg["test_kw"].items() if k in cfg["perticker_tests"]},
            "protocol": "dependent", "synthetic_seed_index": 0, "root_seed": ROOT_SEED,
            "pipeline": pl.pipeline_config()}


def run_perticker(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    t0 = time.perf_counter()
    paths = all_fit_paths(ctx, tickers)
    conf = perticker_config(ctx, cfg, tickers)
    wcfg = {"tests": conf["tests"], "L": conf["L"], "test_kw": conf["test_kw"]}
    cells = [{"cell": f"model={m}", "model": m, "_tickers": list(tickers), "_fit_path": paths[m],
              "_cfg": wcfg} for m in MODELS]
    df = common.run_cells(_perticker_rep, cells, part="perticker", ctx=ctx,
                          root_seed=ROOT_SEED, n_reps=len(tickers), config=conf,
                          block_size=cfg["blocks"]["perticker"],
                          workers=part_workers(ctx, "perticker"))
    ctx.save_csv(df, "e3_perticker_raw", config={"part": "perticker", **conf})
    timing = {"perticker_wall_s": time.perf_counter() - t0}
    ctx.write_part_meta("perticker", conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part: perticker_null (size and calibration of the per-ticker tests)
# --------------------------------------------------------------------------------------------
def _perticker_null_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Null pairs of ticker number ``rep``: X*, Y* independent GARCHt TEST-horizon paths from
    the fitted end-of-TRAIN state (as the real TEST data and the synthetic paths start), tested
    exactly like :func:`_perticker_rep`. Pair ``b`` uses streams keyed by
    ``("perticker_null", ticker, b)``."""
    del rng
    panel = get_panel(cell["_tickers"])
    c = cell["_cfg"]
    ticker = panel.tickers[rep]
    model = {ticker: load_fits(cell["_fit_path"])[ticker]}
    L = int(c["L"])
    prm = model[ticker].params()
    persistence = float(prm.get("alpha", np.nan) + prm.get("beta", np.nan))
    rows: list[dict[str, Any]] = []
    for b in range(int(c["n_draws"])):
        sx, sy, st = common.keyed_seed_sequence(ROOT_SEED, "perticker_null", ticker, b).spawn(3)
        x = pl.simulate_panel(model, panel.n_test, seed=sx, start_from_end=True)[:, 0]
        y = pl.simulate_panel(model, panel.n_test, seed=sy, start_from_end=True)[:, 0]
        X, ox = _ticker_windows(panel, ticker, x, L)
        Y, oy = _ticker_windows(panel, ticker, y, L)
        for r in _ticker_tests(X, Y, ox, oy, c, st, f"null {ticker} {b}"):
            r.update(ticker=ticker, draw=b, L=L, persistence=persistence)
            rows.append(r)
    return rows


def perticker_null_config(ctx: common.ExpContext, cfg: dict, tickers: list[str]
                          ) -> dict[str, Any]:
    return {"tickers": tickers, "fit_hash": fit_hashes(ctx, tickers, (pl.POWER_MODEL,)),
            "null_model": pl.POWER_MODEL, "n_draws": cfg["perticker_null_draws"],
            "tests": cfg["perticker_tests"], "L": cfg["perticker_L"],
            "test_kw": {k: v for k, v in cfg["test_kw"].items() if k in cfg["perticker_tests"]},
            "protocol": "dependent", "start_from_end": True, "root_seed": ROOT_SEED,
            "pipeline": pl.pipeline_config()}


def run_perticker_null(ctx: common.ExpContext, cfg: dict, tickers: list[str]
                       ) -> dict[str, float]:
    t0 = time.perf_counter()
    fpath = ensure_fits(ctx, pl.POWER_MODEL, tickers)[0]
    conf = perticker_null_config(ctx, cfg, tickers)
    cell = {"cell": "null", "_tickers": list(tickers), "_fit_path": fpath,
            "_cfg": {"tests": conf["tests"], "L": conf["L"], "test_kw": conf["test_kw"],
                     "n_draws": conf["n_draws"]}}
    df = common.run_cells(_perticker_null_rep, [cell], part="perticker_null", ctx=ctx,
                          root_seed=ROOT_SEED, n_reps=len(tickers), config=conf,
                          block_size=cfg["blocks"]["perticker_null"],
                          workers=part_workers(ctx, "perticker_null"))
    ctx.save_csv(df, "e3_perticker_null_raw", config={"part": "perticker_null", **conf})
    timing = {"perticker_null_wall_s": time.perf_counter() - t0}
    ctx.write_part_meta("perticker_null", conf, timing)
    return timing


def bh_reject(p: np.ndarray, q: float = ALPHA, method: str = "bh") -> np.ndarray:
    """Benjamini-Hochberg (``"bh"``) / Benjamini-Yekutieli (``"by"``) rejections (NaN: False)."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(p.size, dtype=bool)
    ok = np.isfinite(p)
    if ok.any():
        adj = stats.false_discovery_control(p[ok], method=method)
        out[ok] = adj <= q
    return out


def storey_pi0(p: np.ndarray, lam: float = 0.5) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    return float(min(1.0, np.mean(p > lam) / (1.0 - lam))) if p.size else float("nan")


def calibrate_pvalues(p: np.ndarray, p_null: np.ndarray) -> np.ndarray:
    """Monte Carlo calibration ``(1 + #{p0 <= p}) / (1 + n0)`` against null p-values ``p0``.

    Valid (conservative with ties) when the observed p-value has the null law of the ``p0``
    (here: the pooled GARCHt null over tickers); NaN stays NaN.
    """
    p = np.asarray(p, dtype=float)
    p0 = np.sort(np.asarray(p_null, dtype=float)[np.isfinite(p_null)])
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if p0.size:
        out[ok] = (1.0 + np.searchsorted(p0, p[ok], side="right")) / (1.0 + p0.size)
    return out


def perticker_calibrated(raw: pd.DataFrame, null: pd.DataFrame | None) -> pd.DataFrame:
    """``raw`` with ``p_cal`` (calibrated against the per-test null p-values; NaN without
    them) and ``p_use`` (``p_cal`` when available, else the nominal p-value)."""
    out = raw.copy()
    out["p_cal"] = np.nan
    for t in out["test"].unique():
        if null is None or null.empty:
            continue
        p0 = null.loc[null["test"] == t, "pvalue"].to_numpy(float)
        sel = out["test"] == t
        out.loc[sel, "p_cal"] = calibrate_pvalues(out.loc[sel, "pvalue"].to_numpy(float), p0)
    out["p_use"] = out["p_cal"].where(out["p_cal"].notna(), out["pvalue"])
    out["p_use_type"] = np.where(out["p_cal"].notna(), "calibrated", "nominal (oversized)")
    return out


def perticker_null_size(null: pd.DataFrame) -> pd.DataFrame:
    """Size of the nominal per-ticker p-values under the GARCHt null, per test and level, with
    Wilson intervals (draws of a ticker share its law, so the intervals are approximate), and
    the calibrated 5% critical value (5% quantile of the null p-values)."""
    tab = common.rejection_table(null, ["test", "pvalue_type"])
    q = null.groupby("test")["pvalue"].quantile(ALPHA)
    tab["crit_p_5pct"] = tab["test"].map(q)
    tab["n_tickers"] = tab["test"].map(null.groupby("test")["ticker"].nunique())
    low = null[null["persistence"] < 0.98] if "persistence" in null else null.iloc[:0]
    tab["rej_0.05_persistence_lt_0.98"] = tab["test"].map(
        low.groupby("test")["pvalue"].apply(lambda s: float(np.mean(s <= ALPHA)))
        if len(low) else pd.Series(dtype=float))
    return tab


def perticker_summary(raw: pd.DataFrame, null: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per model and test: nominal and calibrated rejections at 5% (Wilson), BH / BY counts and
    Storey pi0 on the calibrated p-values (nominal ones, flagged, if the null is missing)."""
    cal = perticker_calibrated(raw, null)
    rows = []
    for (m, t), g in cal.groupby(["model", "test"], sort=False):
        p = g["pvalue"].to_numpy(float)
        pu = g["p_use"].to_numpy(float)
        ok = np.isfinite(p)
        n, k = int(ok.sum()), int(np.sum(p[ok] <= ALPHA))
        kc = int(np.sum(pu[np.isfinite(pu)] <= ALPHA))
        lo, hi = pl.wilson_ci(k, n)
        clo, chi = pl.wilson_ci(kc, n)
        rows.append({"model": m, "test": t, "pvalue_type": str(g["pvalue_type"].dropna().iloc[0])
                     if g["pvalue_type"].notna().any() else "",
                     "fdr_pvalue_type": str(g["p_use_type"].iloc[0]), "n_tickers": n,
                     "n_failed": int((~ok).sum()), "n_reject_raw": k,
                     "share_raw": k / n if n else np.nan, "share_lo": lo, "share_hi": hi,
                     "n_reject_cal": kc, "share_cal": kc / n if n else np.nan,
                     "share_cal_lo": clo, "share_cal_hi": chi,
                     "n_reject_bh": int(bh_reject(pu, ALPHA, "bh").sum()),
                     "n_reject_by": int(bh_reject(pu, ALPHA, "by").sum()),
                     "storey_pi0": storey_pi0(pu), "p_median": float(np.nanmedian(p)),
                     "p_cal_median": float(np.nanmedian(g["p_cal"]))
                     if g["p_cal"].notna().any() else np.nan,
                     "n_null": int(np.isfinite(null.loc[null["test"] == t, "pvalue"]).sum())
                     if null is not None and not null.empty else 0,
                     "median_runtime_s": float(np.nanmedian(g["runtime_s"]))})
    out = pd.DataFrame(rows)
    out["order"] = [MODELS.index(m) * 10 + ABS_TESTS.index(t) for m, t in
                    zip(out["model"], out["test"])]
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Part: witness (what the headline critic picks up)
# --------------------------------------------------------------------------------------------
def _sub_rng(base_seed: int, key: str) -> np.random.Generator:
    """Per-test generator of ``pipeline.absolute_tests`` (``pipeline._sub_rng``)."""
    fn = getattr(pl, "_sub_rng", None)
    if fn is not None:
        return fn(base_seed, key)
    return np.random.default_rng(np.random.SeedSequence([base_seed, zlib.crc32(key.encode())]))


def _window_features(W: np.ndarray) -> dict[str, np.ndarray]:
    return {"log_rv": 0.5 * np.log(np.mean(W * W, axis=1) + 1e-12),
            "max_abs": np.max(np.abs(W), axis=1), "sum": W.sum(axis=1)}


def _witness_rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Re-train the seed-0 pooled critic of one model with the pooled test's random streams.

    ``evaluate_model`` -> ``absolute_tests(rng=default_rng(child 1))`` draws one base seed, and
    the critic uses ``_sub_rng(base_seed, "critic")`` for its time-block split and training
    (``LearnedTest.__call__``); the same calls here give the same critic, so
    ``mean f(X_ev) - mean f(Y_ev)`` equals the reported seed-0 statistic (checked in the
    outputs).
    """
    panel = get_panel(cell["_tickers"])
    fitted = load_fits(cell["_fit_path"])
    c = cell["_cfg"]
    ss_sim, ss_test = np.random.SeedSequence(synthetic_seed(0)).spawn(2)
    sim = pl.simulate_panel(fitted, panel.n_test, seed=ss_sim, start_from_end=True)
    X, gx, ox = panel.windows("test")
    Y, gy, oy = panel.synthetic_windows(sim, "test")
    base_seed = int(np.random.default_rng(ss_test).integers(0, 2**63 - 1))
    sub = _sub_rng(base_seed, "critic")
    test = pl.make_test("critic", "dependent", **c["test_kw"].get("critic", {}))
    labs = [time_block_labels(ox, test.time_blocks), time_block_labels(oy, test.time_blocks)]
    (tx, ex), (ty, ey) = train_eval_split([X.shape[0], Y.shape[0]], rng=sub,
                                          frac_train=test.train_frac, groups=labs)
    t0 = time.perf_counter()
    witness, info = test.fit_witness(X[tx], Y[ty], rng=sub)
    train_s = time.perf_counter() - t0
    fx, fy = np.ravel(witness(X[ex])), np.ravel(witness(Y[ey]))
    T = float(fx.mean() - fy.mean())
    rows: list[dict[str, Any]] = []
    for sample, W, f, g, o in (("real", X[ex], fx, gx[ex], ox[ex]),
                               ("synthetic", Y[ey], fy, gy[ey], oy[ey])):
        feats = _window_features(W)
        for i in range(W.shape[0]):
            rows.append({"sample": sample, "f": float(f[i]), "ticker_idx": int(g[i]),
                         "order": int(o[i]), "log_rv": float(feats["log_rv"][i]),
                         "max_abs": float(feats["max_abs"][i]), "sum": float(feats["sum"][i]),
                         "T_witness": T, "train_seconds": train_s,
                         "steps_run": int(info.get("steps_run", test.n_steps))})
    return rows


def run_witness(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    t0 = time.perf_counter()
    paths = all_fit_paths(ctx, tickers)
    conf = {**pooled_config(ctx, cfg, tickers), "synthetic_seed_index": 0}
    cells = [{"cell": f"model={m}", "model": m, "_tickers": list(tickers), "_fit_path": paths[m],
              "_cfg": {"test_kw": cfg["test_kw"]}} for m in MODELS]
    df = common.run_cells(_witness_rep, cells, part="witness", ctx=ctx, root_seed=ROOT_SEED,
                          n_reps=1, config=conf, block_size=1,
                          workers=part_workers(ctx, "witness"))
    ctx.save_csv(df, "e3_witness", config={"part": "witness", **conf})
    timing = {"witness_wall_s": time.perf_counter() - t0}
    ctx.write_part_meta("witness", conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part: stylized facts
# --------------------------------------------------------------------------------------------
STYL_ROWS: tuple[tuple[str, str, float, int], ...] = (
    # (statistic, LaTeX label, display factor, decimals)
    ("std", r"Std.\ dev.\ ($\times 100$)", 100.0, 2),
    ("skew", "Skewness", 1.0, 2),
    ("kurt", "Excess kurtosis", 1.0, 1),
    ("acf_r_1", r"ACF $r$, lag 1", 1.0, 3),
    ("acf_abs_1", r"ACF $|r|$, lag 1", 1.0, 3),
    ("acf_abs_5", r"ACF $|r|$, lag 5", 1.0, 3),
    ("acf_abs_20", r"ACF $|r|$, lag 20", 1.0, 3),
    ("acf_abs_50", r"ACF $|r|$, lag 50", 1.0, 3),
    ("acf_sq_1", r"ACF $r^2$, lag 1", 1.0, 3),
    ("lev_mean", r"Leverage, mean lags 1--10", 1.0, 3),
    ("hill", r"Hill index of $|r|$ (5\%)", 1.0, 2),
    ("kurt_agg_5", "Excess kurtosis, 5-day sums", 1.0, 2),
    ("kurt_agg_20", "Excess kurtosis, 20-day sums", 1.0, 2),
    ("max_drawdown", "Max.\\ drawdown (log)", 1.0, 2),
)


def run_stylized(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    t0 = time.perf_counter()
    paths = all_fit_paths(ctx, tickers)
    t_fit = time.perf_counter() - t0
    panel = get_panel(tickers)
    sources: dict[str, pd.DataFrame] = {"Real": panel.test}
    for m in MODELS:
        sim = headline_sim(paths[m], panel)
        sources[m] = pd.DataFrame(sim, index=panel.test.index, columns=panel.tickers)
    per, acf_rows, tail_rows = [], [], []
    nl = int(cfg["acf_lags"])
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
        z = np.sort(np.abs((df / panel.scale.loc[df.columns]).to_numpy()).ravel())[::-1]
        n = z.size
        ranks = np.unique(np.geomspace(1, n, int(cfg["tail_points"])).astype(int))
        for r in ranks:
            tail_rows.append({"source": src, "x": float(z[r - 1]), "ccdf": r / n})
    raw = pd.concat(per, ignore_index=True)
    conf = {"part": "stylized", "tickers": tickers, "fit_hash": fit_hashes(ctx, tickers),
            "synthetic_seed_index": 0, "root_seed": ROOT_SEED}
    ctx.save_csv(raw, "e3_stylized_raw", config=conf)
    ctx.save_csv(pd.DataFrame(acf_rows), "e3_stylized_acf", config=conf)
    ctx.save_csv(pd.DataFrame(tail_rows), "e3_stylized_tail", config=conf)
    timing = {"stylized_wall_s": time.perf_counter() - t0 - t_fit}
    ctx.write_part_meta("stylized", conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part: likelihood benchmark
# --------------------------------------------------------------------------------------------
def run_likelihood(ctx: common.ExpContext, cfg: dict, tickers: list[str]) -> dict[str, float]:
    paths = all_fit_paths(ctx, tickers, LIK_MODELS)
    t0 = time.perf_counter()
    panel = get_panel(tickers)
    rows: list[dict[str, Any]] = []
    ll: dict[str, np.ndarray] = {m: np.empty((panel.n_test, len(tickers))) for m in LIK_MODELS}
    for m in LIK_MODELS:
        fitted = load_fits(paths[m])
        for j, t in enumerate(tickers):
            mdl = fitted[t]
            rtr = panel.train[t].to_numpy(dtype=np.float64)
            rte = panel.test[t].to_numpy(dtype=np.float64)
            llo = np.asarray(mdl.loglik_obs(rte, r_history=rtr), dtype=np.float64)
            ll[m][:, j] = llo
            u = np.asarray(mdl.pit(rte, r_history=rtr), dtype=np.float64)
            bk, ks = lik.berkowitz_test(u), lik.pit_ks_test(u)
            rows.append({"model": m, "ticker": t,
                         "pred_loglik": lik.predictive_loglik(mdl, rte, rtr),
                         "pred_loglik_z": float(np.mean(llo) + math.log(panel.scale[t])),
                         "berk_stat": bk.statistic, "berk_p": bk.pvalue,
                         "berk_p_ind": bk.extra["p_ind"], "berk_mean": bk.extra["mean"],
                         "berk_var": bk.extra["var"], "berk_rho": bk.extra["rho"],
                         "berk_n_clipped": bk.extra["n_clipped"], "ks_stat": ks.statistic,
                         "ks_p": ks.pvalue, "pvalue_type": "asymptotic"})
    ag_rows: list[dict[str, Any]] = []
    for a, b in AG_PAIRS:
        for j, t in enumerate(tickers):
            v = lik.vuong_test(ll[a][:, j], ll[b][:, j], k_a=0, k_b=0, hac=True)
            ag_rows.append({"pair": f"{a} vs {b}", "model_a": a, "model_b": b, "ticker": t,
                            "z": v.statistic, "p": v.pvalue, "p_a_better": v.extra["p_a_better"],
                            "p_b_better": v.extra["p_b_better"], "mean_d": v.extra["mean_d"],
                            "lags": v.extra["lags"], "pvalue_type": "hac_normal"})
        # pooled: cross-sectional mean of the daily differences, Newey-West (Driscoll-Kraay)
        v = lik.vuong_test(ll[a].mean(axis=1), ll[b].mean(axis=1), k_a=0, k_b=0, hac=True)
        ag_rows.append({"pair": f"{a} vs {b}", "model_a": a, "model_b": b, "ticker": "POOLED",
                        "z": v.statistic, "p": v.pvalue, "p_a_better": v.extra["p_a_better"],
                        "p_b_better": v.extra["p_b_better"], "mean_d": v.extra["mean_d"],
                        "lags": v.extra["lags"], "pvalue_type": "hac_normal (Driscoll-Kraay)"})
    conf = {"part": "likelihood", "tickers": tickers,
            "fit_hash": fit_hashes(ctx, tickers, LIK_MODELS), "models": list(LIK_MODELS),
            "pairs": [list(p) for p in AG_PAIRS], "vuong": {"hac": True, "k": 0}}
    ctx.save_csv(pd.DataFrame(rows), "e3_likelihood_raw", config=conf)
    ctx.save_csv(pd.DataFrame(ag_rows), "e3_ag_raw", config=conf)
    timing = {"likelihood_wall_s": time.perf_counter() - t0}
    ctx.write_part_meta("likelihood", conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Derived parts: agreement and verdict (from CSVs)
# --------------------------------------------------------------------------------------------
def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen (1960) kappa of two binary raters (NaN when chance agreement is 1)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.size == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def stratified_kappa(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Kappa pooled over strata with chance agreement computed *within* each stratum:
    ``sum_s n_s (po_s - pe_s) / sum_s n_s (1 - pe_s)``.

    Pooling raters' decisions over models with one overall chance agreement counts model main
    effects (e.g. GBM rejected by both raters almost everywhere) as agreement; this version
    does not.
    """
    num = den = 0.0
    for a, b in pairs:
        a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
        if a.size == 0:
            continue
        po = float(np.mean(a == b))
        pa, pb = float(a.mean()), float(b.mean())
        pe = pa * pb + (1 - pa) * (1 - pb)
        num += a.size * (po - pe)
        den += a.size * (1 - pe)
    return num / den if den > 0 else float("nan")


def _kendall(x: np.ndarray, y: np.ndarray) -> float:
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return float("nan")
    return float(stats.kendalltau(x, y).statistic)


def compute_agree(ctx: common.ExpContext) -> bool:
    """Kendall tau (pooled; per ticker, descriptive) and Cohen kappa (per model, stratified).

    Per-ticker p-values are the calibrated ones (``perticker_null``) when available, else the
    nominal ones (flagged in ``pvalue_basis``). The per-ticker tau is descriptive: tickers share
    dates and test seeds, so no test against random rankings is reported (the earlier z-test
    assumed independent tickers); the ticker-bootstrap interval in the CSV ignores that
    dependence too.
    """
    need = ("e3_pooled_raw", "e3_perticker_raw", "e3_likelihood_raw")
    missing = [n for n in need if not ctx.has_csv(n)]
    if missing:
        logger.info("agree: skipped (missing %s)", missing)
        return False
    pooled = ctx.load_csv("e3_pooled_raw")
    null = ctx.load_csv("e3_perticker_null_raw") if ctx.has_csv("e3_perticker_null_raw") else None
    pt = perticker_calibrated(ctx.load_csv("e3_perticker_raw"), null)
    lk = ctx.load_csv("e3_likelihood_raw")
    lm = list(LIK_MODELS)
    llm = lk.groupby("model")["pred_loglik"].mean().reindex(lm)
    crit = pooled[(pooled["kind"] == "absolute") & (pooled["test"] == "critic")]
    tau_rows = []
    for label, stat in (("seed 0", crit[crit["rep"] == 0].groupby("model")["statistic"].first()),
                        ("median over seeds", crit.groupby("model")["statistic"].median())):
        s = stat.reindex(lm).to_numpy(float)
        tau_rows.append({"scope": "pooled", "critic_statistic": label, "test": "critic",
                         "tau": _kendall(-s, llm.to_numpy(float)), "n": 1,
                         "rank_critic": ">".join(np.array(lm)[np.argsort(s)]),
                         "rank_loglik": ">".join(np.array(lm)[np.argsort(-llm.to_numpy())])})
    llt = lk.pivot(index="ticker", columns="model", values="pred_loglik")[lm]
    rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, *common.key_words("agree")]))
    for test in sorted(pt["test"].unique(), key=lambda t: ABS_TESTS.index(t)):
        st = pt[pt["test"] == test].pivot(index="ticker", columns="model",
                                          values="statistic").reindex(columns=lm)
        common_t = st.index.intersection(llt.index)
        taus = np.array([_kendall(-st.loc[t].to_numpy(float), llt.loc[t].to_numpy(float))
                         for t in common_t])
        taus = taus[np.isfinite(taus)]
        n = taus.size
        if n == 0:
            continue
        boot = rng.choice(taus, size=(2000, n), replace=True).mean(axis=1)
        tau_rows.append({"scope": "per ticker (L=5)", "critic_statistic": "per-ticker", "test":
                         test, "tau": float(taus.mean()), "n": n,
                         "tau_ci_lo": float(np.quantile(boot, 0.025)),
                         "tau_ci_hi": float(np.quantile(boot, 0.975)),
                         "ci_note": "ticker bootstrap; ignores cross-sectional dependence",
                         "share_tau_1": float(np.mean(taus > 0.99)),
                         "share_tau_m1": float(np.mean(taus < -0.99))})
    kappa_rows = []
    berk = lk.pivot(index="ticker", columns="model", values="berk_p")
    basis = str(pt["p_use_type"].iloc[0]) if len(pt) else ""
    for test in sorted(pt["test"].unique(), key=lambda t: ABS_TESTS.index(t)):
        pv = pt[pt["test"] == test].pivot(index="ticker", columns="model", values="p_use")
        for rule in ("bh", "raw"):
            pairs = []
            for m in lm:
                idx = pv.index.intersection(berk.index)
                pa, pb = pv.loc[idx, m].to_numpy(float), berk.loc[idx, m].to_numpy(float)
                if rule == "bh":
                    a, b = bh_reject(pa), bh_reject(pb)
                else:
                    a, b = pa <= ALPHA, pb <= ALPHA
                pairs.append((a, b))
                kappa_rows.append(_kappa_row(test, m, rule, a, b, cohen_kappa(a, b), basis))
            a, b = np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs])
            kappa_rows.append(_kappa_row(test, "stratified", rule, a, b,
                                         stratified_kappa(pairs), basis))
            kappa_rows.append(_kappa_row(test, "pooled (model effects)", rule, a, b,
                                         cohen_kappa(a, b), basis))
    conf = {"part": "agree", "inputs": list(need), "tau_bootstrap": 2000,
            "pvalue_basis": basis}
    ctx.save_csv(pd.DataFrame(tau_rows), "e3_agree_tau", config=conf)
    ctx.save_csv(pd.DataFrame(kappa_rows), "e3_agree_kappa", config=conf)
    return True


def _kappa_row(test: str, model: str, rule: str, a: np.ndarray, b: np.ndarray, kappa: float,
               basis: str) -> dict[str, Any]:
    return {"test": test, "model": model, "rule": rule, "pvalue_basis": basis,
            "n": int(a.size), "kappa": kappa,
            "agreement": float(np.mean(a == b)) if a.size else np.nan,
            "both_reject": int(np.sum(a & b)), "only_test": int(np.sum(a & ~b)),
            "only_berkowitz": int(np.sum(~a & b)), "neither": int(np.sum(~a & ~b))}


def _verdict(p: float, power: float) -> str:
    """``pipeline.verdict``, or ``"--"`` when the relative p-value is missing (failed test)."""
    return pl.verdict(p, power) if np.isfinite(p) else "--"


def load_designs(ctx: common.ExpContext) -> dict[str, pd.DataFrame]:
    """Summaries of the power designs that exist (``D1``: power part, ``D2``: realvol part).

    A ``e3_power_summary.csv`` in the pre-review single-seed format (no ``design`` column) is
    ignored with a warning.
    """
    out = {}
    for d, part in DESIGNS.items():
        name = f"e3_{part}_summary"
        if not ctx.has_csv(name):
            continue
        s = ctx.load_csv(name)
        if "design" not in s or "rate_rule" not in s:
            logger.warning("%s.csv has the old single-seed format: rerun --part %s", name, part)
            continue
        out[d] = s.set_index("case")
    return out


def compute_verdict(ctx: common.ExpContext) -> bool:
    """Decision rule per model from the seed-combined anchored p-value and the rule's power.

    ``power_rule`` is the minimum over the power designs (D1, D2) of the power of the rule
    actually applied (twice the median p over ``n_seeds`` seeds), NaN if a design is missing;
    GOOD needs it to be >= 0.8 (``verdict_cp``: its Clopper-Pearson lower bound). ``boundary_rule`` is the maximum
    boundary size of the rule over the designs. Bootstrap: ``NO_VERDICT`` (relative test
    invalid).
    """
    if not ctx.has_csv("e3_pooled_raw"):
        logger.info("verdict: skipped (no pooled results)")
        return False
    pooled = ctx.load_csv("e3_pooled_raw")
    des = load_designs(ctx)
    if not des:
        logger.warning("verdict: no power design results: every verdict is INCONCLUSIVE or BAD")
    nan = float("nan")

    def dget(d: str, case: str, col: str) -> float:
        try:
            return float(des[d].loc[case, col])
        except KeyError:
            return nan

    pw = {d: dget(d, "alternative", "rate_rule") for d in DESIGNS}
    pw_cp = {d: dget(d, "alternative", "rule_cp_lower") for d in DESIGNS}
    bd = {d: dget(d, "boundary", "rate_rule") for d in DESIGNS}
    have = [d for d in DESIGNS if np.isfinite(pw[d])]
    d_min = min(have, key=lambda d: pw[d]) if have else ""
    if len(have) == len(DESIGNS):
        power_rule, power_rule_cp = pw[d_min], min(pw_cp[d] for d in have)
    else:  # GOOD needs the power under every design
        power_rule = power_rule_cp = nan
        if have:
            logger.warning("verdict: power design(s) %s missing (run --part %s): no GOOD "
                           "verdict possible", [d for d in DESIGNS if d not in have],
                           ",".join(DESIGNS[d] for d in DESIGNS if d not in have))
    have_bd = [d for d in DESIGNS if np.isfinite(bd[d])]
    d_bd = max(have_bd, key=lambda d: bd[d]) if have_bd else ""
    rows = []
    for m in MODELS:
        a = pooled[(pooled["model"] == m) & (pooled["test"] == "critic")].sort_values("rep")
        r = pooled[(pooled["model"] == m)
                   & (pooled["test"] == "relative_anchored")].sort_values("rep")
        if a.empty or r.empty:
            continue
        a0, r0 = a[a["rep"] == 0].iloc[0], r[r["rep"] == 0].iloc[0]
        pa, pr = a["pvalue"].to_numpy(float), r["pvalue"].to_numpy(float)
        p_comb = pl.median_pvalue(pr)
        d = r["statistic"].to_numpy(float)
        se = r["se_hac"].to_numpy(float) if "se_hac" in r else np.full(d.size, nan)
        valid = m not in REL_INVALID
        row = {
            "model": m, "n_seeds": int(len(r)),
            "abs_pvalue_type": a0["pvalue_type"],
            "abs_stat_median": float(np.nanmedian(a["statistic"])),
            "abs_p_combined": pl.median_pvalue(pa), "abs_n_reject": int(np.sum(pa <= ALPHA)),
            "abs_stat_seed0": float(a0["statistic"]), "abs_p_seed0": float(a0["pvalue"]),
            "rel_pvalue_type": r0["pvalue_type"], "rel_valid": valid,
            "rel_note": REL_INVALID.get(m, ""),
            "delta_median": float(np.nanmedian(d)), "delta_min": float(np.nanmin(d)),
            "delta_max": float(np.nanmax(d)),
            "delta_lb95_median": float(np.nanmedian(d - Z_ONE_SIDED * se)),
            "rel_p_combined": p_comb, "rel_p_median": float(np.nanmedian(pr)),
            "se_hac_real_median": float(np.nanmedian(se)),
            # single-seed quantities: not calibrated on real data (see the module docstring)
            "uncal_delta_seed0": float(r0["statistic"]), "uncal_rel_p_seed0": float(r0["pvalue"]),
            "uncal_delta_ci_lo_seed0": float(r0["ci_lo"]),
            "uncal_delta_ci_hi_seed0": float(r0["ci_hi"]),
            "uncal_rel_n_reject": int(np.sum(pr <= ALPHA)),
            "power_rule": power_rule, "power_rule_cp_lower": power_rule_cp,
            "power_rule_design": d_min,
            "boundary_rule": bd[d_bd] if d_bd else nan, "boundary_rule_design": d_bd,
        }
        for dn in DESIGNS:
            row.update({
                f"power_{dn}": pw[dn], f"power_{dn}_ci_lo": dget(dn, "alternative", "rule_ci_lo"),
                f"power_{dn}_ci_hi": dget(dn, "alternative", "rule_ci_hi"),
                f"power_{dn}_cp_lower": pw_cp[dn],
                f"power_{dn}_single": dget(dn, "alternative", "rate_single"),
                f"boundary_{dn}": bd[dn],
                f"boundary_{dn}_cp_lo": dget(dn, "boundary", "rule_cp_lo"),
                f"boundary_{dn}_cp_hi": dget(dn, "boundary", "rule_cp_hi"),
                f"boundary_{dn}_n_reps": dget(dn, "boundary", "n_reps"),
                f"boundary_{dn}_n_reject": dget(dn, "boundary", "n_reject_rule"),
                f"boundary_{dn}_single": dget(dn, "boundary", "rate_single"),
                f"se_hac_{dn}_median": dget(dn, "boundary", "median_se_hac"),
                f"se_ratio_real_{dn}": float(np.nanmedian(se)) / dget(dn, "boundary",
                                                                      "median_se_hac"),
            })
        row["verdict"] = _verdict(p_comb, power_rule) if valid else NO_VERDICT
        row["verdict_cp"] = _verdict(p_comb, power_rule_cp) if valid else NO_VERDICT
        # pre-review rule (single-seed D1 power) for traceability only
        row["verdict_design_power"] = (_verdict(p_comb, row["power_D1_single"]) if valid
                                       else NO_VERDICT)
        rows.append(row)
    for row in rows:
        logger.info("verdict %s: 2*median anchored p %.3f, median Delta %.3f, se_hac real %.4f "
                    "(x%.1f D1, x%.1f D2), rule power %.2f (%s) -> %s", row["model"],
                    row["rel_p_combined"], row["delta_median"], row["se_hac_real_median"],
                    row["se_ratio_real_D1"], row["se_ratio_real_D2"], row["power_rule"],
                    row["power_rule_design"] or "none", row["verdict"])
    ctx.save_csv(pd.DataFrame(rows), "e3_verdicts",
                 config={"part": "verdict", "alpha": ALPHA, "power_min": pl.POWER_MIN,
                         "rule": "verdict(2*median anchored HAC p over seeds, min over designs "
                                 "of the power of that rule)",
                         "designs": list(des), "invalid_relative": REL_INVALID})
    return True


# --------------------------------------------------------------------------------------------
# LaTeX formatting helpers
# --------------------------------------------------------------------------------------------
def tnum(v: Any, d: int = 3) -> str:
    """Number with a math minus; ``--`` for NaN."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(x):
        return "--"
    s = f"{x:.{d}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return "$-$" + s[1:] if s.startswith("-") else s


def tp(p: Any) -> str:
    """p-value: 3 decimals, ``$<$0.001`` below."""
    try:
        x = float(p)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(x):
        return "--"
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def tex(s: Any) -> str:
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("$", r"\$")):
        out = out.replace(a, b)
    return out


def rate_ci(k: int, n: int, d: int = 2) -> str:
    if not n:
        return "--"
    lo, hi = pl.wilson_ci(int(k), int(n))
    return f"{k / n:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def _nanmedian(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def med_iqr(x: np.ndarray, d: int, factor: float = 1.0) -> tuple[str, str]:
    x = np.asarray(x, dtype=float) * factor
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "--", "--"
    q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
    return tnum(q50, d), f"[{tnum(q25, d)}, {tnum(q75, d)}]"


PTYPE_LABEL = {"hac": "HAC", "wild": "wild", "permutation": "perm.",
               "asymptotic_iid": "asy.\\ i.i.d.", "normal": "normal"}
#: Row markers: dagger = not a valid test here (KS under dependence; two-critic relative test at
#: the boundary, secondary); double dagger = relative test invalid for the model (bootstrap).
DAGGER, DDAGGER = r"$^{\dagger}$", r"$^{\ddagger}$"


# --------------------------------------------------------------------------------------------
# Tables and figures
# --------------------------------------------------------------------------------------------
#: (fit-table column, LaTeX label, display factor, decimals) of the fitted parameters per model.
PARAM_ROWS: dict[str, tuple[tuple[str, str, float, int], ...]] = {
    "GBM": (("mu", r"$\mu\times 10^4$", 1e4, 2), ("sigma", r"$\sigma\times 100$", 100.0, 2)),
    "Merton": (("mu", r"$\mu\times 10^4$", 1e4, 2), ("sigma", r"$\sigma\times 100$", 100.0, 2),
               ("lam", r"$\lambda$", 1.0, 3), ("muJ", r"$\mu_J\times 100$", 100.0, 2),
               ("sJ", r"$\sigma_J\times 100$", 100.0, 2)),
    "GARCHt": (("mu", r"$\mu\times 10^4$", 1e4, 2), ("alpha", r"$\alpha$", 1.0, 3),
               ("beta", r"$\beta$", 1.0, 3), ("persistence", r"$\alpha+\beta$", 1.0, 3),
               ("nu", r"$\nu$", 1.0, 2), ("omega", r"$\omega\times 10^6$", 1e6, 2)),
    "Heston": (("kappa", r"$\kappa$", 1.0, 3), ("theta", r"$\sqrt{\theta}\times 100$", 100.0, 2),
               ("xi", r"$\xi\times 100$", 100.0, 2), ("rho", r"$\rho$", 1.0, 2),
               ("c", r"$c=\xi/\sqrt{2\kappa\theta}$ (CV of $v$)", 1.0, 2)),
    "Bootstrap": (("block_length", "Mean block length", 1.0, 1),),
}
#: Names used in ``fit_info_["at_bound"]`` that differ from the fit-table column.
AT_BOUND_ALIAS = {"persistence": "persistence"}


def build_fit_tables(ctx: common.ExpContext) -> None:
    if not all(ctx.has_csv(f"e3_fit_{m}") for m in MODELS):
        logger.info("fit tables: skipped (missing fit CSVs)")
        return
    rows = []
    for m in MODELS:
        ft = ctx.load_csv(f"e3_fit_{m}").set_index("ticker")
        if m == "Heston" and "theta" in ft:
            ft = ft.assign(theta=np.sqrt(ft["theta"].clip(lower=0)))
        ab = [s if isinstance(s, str) else "" for s in ft["at_bound"]]  # all-empty -> NaN
        counts: dict[str, int] = {}
        for s in ab:
            for nm in filter(None, s.split(";")):
                counts[nm] = counts.get(nm, 0) + 1
        shown = set()
        for col, label, fac, d in PARAM_ROWS[m]:
            if col not in ft:
                continue
            med, iqr = med_iqr(ft[col].to_numpy(float), d, fac)
            rows.append({"Model": MODEL_LABELS[m], "Parameter": label, "Median": med,
                         "IQR": iqr, "At bound": str(counts.get(col, 0))})
            shown.add(col)
        for nm, cnt in sorted(counts.items()):
            if nm not in shown:
                rows.append({"Model": MODEL_LABELS[m], "Parameter": tex(nm) + " (at bound)",
                             "Median": "--", "IQR": "--", "At bound": str(cnt)})
        extra = []
        if m == "Heston" and "feller_ok" in ft:
            extra.append(("Feller condition holds (share)",
                          tnum(ft["feller_ok"].astype(float).mean(), 2), "--"))
        if m == "Bootstrap" and "fallback" in ft:
            extra.append(("Block-rule fallback (tickers)", str(int(ft["fallback"].sum())), "--"))
        if "success" in ft:
            extra.append(("Optimiser success (share)", tnum(ft["success"].astype(float).mean(), 2),
                          "--"))
        med, iqr = med_iqr(ft["fit_seconds"].to_numpy(float), 2)
        extra.append(("Fit time (s)", med, iqr))
        for label, med, iqr in extra:
            rows.append({"Model": MODEL_LABELS[m], "Parameter": label, "Median": med,
                         "IQR": iqr, "At bound": ""})
        rows.append({"Model": MODEL_LABELS[m], "Parameter": "Tickers with a parameter at a bound",
                     "Median": str(int((ft["n_at_bound"] > 0).sum())), "IQR": "",
                     "At bound": f"of {len(ft)}"})
    tab = pd.DataFrame(rows)
    ctx.save_csv(tab, "e3_fit_summary")
    ctx.save_table(tab.set_index(["Model", "Parameter"]), "e3_fit_diagnostics", escape=False)
    if ctx.has_csv("e3_heston_moments_raw"):
        hm = ctx.load_csv("e3_heston_moments_raw")
        names = list(dict.fromkeys(hm["moment"]))
        labels = {"var": r"Variance ($\times 10^4$)", "tail_qr": "Tail quantile ratio",
                  "acf_sq_1": r"ACF $r^2$, lag 1", "acf_sq_5": r"ACF $r^2$, lag 5",
                  "acf_sq_10": r"ACF $r^2$, lag 10", "acf_sq_20": r"ACF $r^2$, lag 20",
                  "lev_1": r"corr$(r_t, r_{t+1}^2)$", "kurtosis (not targeted)":
                  "Kurtosis (not targeted)"}
        hrows = []
        for nm in names:
            g = hm[hm["moment"] == nm]
            fac = 1e4 if nm == "var" else 1.0
            d = 2 if nm in ("var", "tail_qr", "kurtosis (not targeted)") else 3
            md, iqd = med_iqr(g["data"].to_numpy(float), d, fac)
            ms, iqs = med_iqr(g["sim"].to_numpy(float), d, fac)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(g["sim"].to_numpy(float) / g["data"].to_numpy(float) - 1.0)
            hrows.append({"Moment": labels.get(nm, tex(nm)), "Data": md, "Data IQR": iqd,
                          "Simulated": ms, "Sim.\\ IQR": iqs,
                          "Median $|$rel.\\ err.$|$": tnum(np.nanmedian(rel), 2),
                          "Median $|t|$": tnum(_nanmedian(np.abs(g["t"].to_numpy(float))), 2)})
        htab = pd.DataFrame(hrows)
        ctx.save_csv(htab, "e3_heston_moments_summary")
        ctx.save_table(htab.set_index("Moment"), "e3_heston_moments", escape=False)


def cp_rate(k: Any, n: Any, d: int = 3) -> str:
    """``rate [Clopper-Pearson 95%]``."""
    try:
        k, n = int(k), int(n)
    except (TypeError, ValueError):
        return "--"
    if n <= 0:
        return "--"
    lo, hi = clopper_pearson_ci(k, n)
    return f"{k / n:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def build_power_table(ctx: common.ExpContext) -> None:
    """``e3_power.tex``: power and boundary size of the decision rule, per design."""
    des = load_designs(ctx)
    if not des:
        return
    rows = []
    for d, s in des.items():
        alt = s.loc["alternative"] if "alternative" in s.index else None
        bd = s.loc["boundary"] if "boundary" in s.index else None
        ref = alt if alt is not None else bd
        rows.append({
            "Design": DESIGN_LABELS[d],
            r"Reps $\times$ seeds": f"{int(ref['n_reps'])} $\\times$ {int(ref['n_seeds'])}",
            r"Power, rule [95\% CI]": rate_ci(alt["n_reject_rule"], alt["n_reps"])
            if alt is not None else "--",
            r"$\beta_L$ (rule)": tnum(alt["rule_cp_lower"], 2) if alt is not None else "--",
            r"Power, 1 seed [95\% CI]": rate_ci(alt["n_reject_single"], alt["n_single"])
            if alt is not None else "--",
            r"Boundary size, rule [CP 95\%]": cp_rate(bd["n_reject_rule"], bd["n_reps"], 2)
            if bd is not None else "--",
            r"Boundary size, 1 seed [CP 95\%]": cp_rate(bd["n_reject_single"], bd["n_single"], 2)
            if bd is not None else "--",
            r"Median $\widehat{\mathrm{se}}_{\mathrm{HAC}}$": tnum(bd["median_se_hac"], 3)
            if bd is not None else "--",
        })
    ctx.save_table(pd.DataFrame(rows).set_index("Design"), "e3_power", escape=False)


def build_pooled_outputs(ctx: common.ExpContext) -> None:
    """``e3_pooled_tests.tex``: per test and model, twice the median p over the synthetic seeds
    and (absolute tests) the share of seeds rejecting at 5% (descriptive; relative tests show
    no share because their single-seed p-values are not calibrated on real data)."""
    if not ctx.has_csv("e3_pooled_raw"):
        return
    raw = ctx.load_csv("e3_pooled_raw")
    summ = pooled_summary(raw)
    ctx.save_csv(summ, "e3_pooled_summary")
    models = [m for m in MODELS if m in set(summ["model"])]
    rows = []
    for t in ALL_TESTS:
        st = summ[summ["test"] == t]
        if st.empty:
            continue
        label = TEST_LABELS[t]
        if t == "ks" or t == "relative_two_critic":
            label += DAGGER
        if t == "relative_two_critic":
            label += " (secondary)"
        row = {"Test": label,
               "$p$-value": PTYPE_LABEL.get(st["pvalue_type"].iloc[0], st["pvalue_type"].iloc[0])}
        for m in models:
            r = st[st["model"] == m]
            if r.empty:
                row[MODEL_LABELS[m]] = "--"
                continue
            r = r.iloc[0]
            cell = tp(r["p_combined"])
            if r["kind"] == "absolute":
                cell += f" ({int(r['n_reject'])}/{int(r['n_seeds'])})"
            elif m in REL_INVALID:
                cell += DDAGGER
            row[MODEL_LABELS[m]] = cell
        rows.append(row)
    ctx.save_table(pd.DataFrame(rows).set_index("Test"), "e3_pooled_tests", escape=False)
    plot_pvalues(ctx, raw)


def plot_pvalues(ctx: common.ExpContext, raw: pd.DataFrame) -> None:
    """Per-seed p-values (points) and twice their median (black bar) per test and model; tests
    that are not valid for the model are drawn faded."""
    import matplotlib.pyplot as plt

    plotting.setup()
    models = [m for m in MODELS if m in set(raw["model"])]
    tests = [t for t in ALL_TESTS if t in set(raw["test"])]
    fig, axes = plt.subplots(1, len(models), figsize=plotting.figsize("two", 0.40), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, models):
        sub = raw[raw["model"] == m]
        for j, t in enumerate(tests):
            s = sub[sub["test"] == t].sort_values("rep")
            if s.empty:
                continue
            ok = validity(m, t, True)[0]
            p = np.clip(s["pvalue"].to_numpy(float), P_FLOOR, 1.0)
            n = p.size
            x = j + (np.linspace(-0.28, 0.28, n) if n > 1 else np.zeros(1))
            ax.scatter(x, p, s=9, color=TEST_COLORS[t], marker=TEST_MARKERS[t],
                       alpha=0.75 if ok else 0.3,
                       linewidths=0.4 if TEST_MARKERS[t] in "xX+" else 0)
            pc = max(pl.median_pvalue(s["pvalue"].to_numpy(float)), P_FLOOR)
            ax.plot([j - 0.34, j + 0.34], [pc, pc], color=plotting.OKABE_ITO["black"],
                    lw=1.2 if ok else 0.6, ls="-" if ok else ":", zorder=3)
        ax.axhline(ALPHA, color=plotting.OKABE_ITO["grey"], lw=0.7, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(P_FLOOR / 3, 1.6)
        ax.set_xlim(-0.6, len(tests) - 0.4)
        ax.set_xticks(range(len(tests)))
        ax.set_xticklabels([TEST_LABELS[t] for t in tests], rotation=90, fontsize=6.5)
        ax.set_title(MODEL_LABELS[m])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(rf"$p$-value (floored at $10^{{{round(math.log10(P_FLOOR))}}}$)")
    ctx.savefig(fig, "e3_pvalues")


def build_perticker_table(ctx: common.ExpContext) -> None:
    """``e3_perticker.tex`` (calibrated rejections, BH / BY / Storey on calibrated p-values,
    nominal rejections for reference) and ``e3_perticker_null.tex`` (size of the nominal
    per-ticker p-values under the GARCHt null)."""
    null = (ctx.load_csv("e3_perticker_null_raw") if ctx.has_csv("e3_perticker_null_raw")
            else None)
    if null is not None and not null.empty:
        size = perticker_null_size(null)
        ctx.save_csv(size, "e3_perticker_null_size")
        nrows = []
        for _, r in size.iterrows():
            nrows.append({"Test": TEST_LABELS[r["test"]],
                          "$p$-value": PTYPE_LABEL.get(r["pvalue_type"], tex(r["pvalue_type"])),
                          "Null pairs": f"{int(r['n_valid'])} ({int(r['n_tickers'])} tickers)",
                          r"Size at 1\% [95\% CI]": rate_ci(round(r["rej_0.01"] * r["n_valid"]),
                                                            r["n_valid"]),
                          r"Size at 5\% [95\% CI]": rate_ci(round(r["rej_0.05"] * r["n_valid"]),
                                                            r["n_valid"]),
                          r"Size at 10\% [95\% CI]": rate_ci(round(r["rej_0.10"] * r["n_valid"]),
                                                             r["n_valid"]),
                          r"Calibrated 5\% critical $p$": tp(r["crit_p_5pct"])})
        ctx.save_table(pd.DataFrame(nrows).set_index("Test"), "e3_perticker_null", escape=False)
    if not ctx.has_csv("e3_perticker_raw"):
        return
    raw = ctx.load_csv("e3_perticker_raw")
    summ = perticker_summary(raw, null)
    ctx.save_csv(summ, "e3_perticker_summary")
    ctx.save_csv(perticker_calibrated(raw, null), "e3_perticker_calibrated")
    calibrated = null is not None and not null.empty
    if not calibrated:
        logger.warning("per-ticker table: no null calibration (run --part perticker_null); BH / "
                       "BY / Storey use the oversized nominal p-values")
    tag = "cal." if calibrated else r"nominal$^{\dagger}$"
    rows = []
    for _, r in summ.iterrows():
        rows.append({"Model": MODEL_LABELS[r["model"]], "Test": TEST_LABELS[r["test"]],
                     "$p$-value": PTYPE_LABEL.get(r["pvalue_type"], tex(r["pvalue_type"])),
                     "Tickers": str(int(r["n_tickers"])),
                     rf"Rej.\ 5\%, {tag} [95\% CI]": rate_ci(int(r["n_reject_cal"]),
                                                            int(r["n_tickers"])),
                     f"BH ({tag})": str(int(r["n_reject_bh"])),
                     f"BY ({tag})": str(int(r["n_reject_by"])),
                     rf"Storey $\hat\pi_0$ ({tag})": tnum(r["storey_pi0"], 2),
                     r"Rej.\ 5\%, nominal$^{\dagger}$": f"{int(r['n_reject_raw'])}",
                     "Median nominal $p$": tp(r["p_median"])})
    ctx.save_table(pd.DataFrame(rows).set_index(["Model", "Test"]), "e3_perticker",
                   escape=False)


def build_witness_figure(ctx: common.ExpContext) -> None:
    if not ctx.has_csv("e3_witness"):
        return
    import matplotlib.pyplot as plt

    w = ctx.load_csv("e3_witness")
    # consistency with the reported seed-0 statistic
    if ctx.has_csv("e3_pooled_raw"):
        pr = ctx.load_csv("e3_pooled_raw")
        ref = pr[(pr["test"] == "critic") & (pr["rep"] == 0)].set_index("model")["statistic"]
        chk = []
        for m, g in w.groupby("model"):
            tw = float(g["T_witness"].iloc[0])
            tr = float(ref.get(m, np.nan))
            chk.append({"model": m, "T_witness": tw, "T_pooled_seed0": tr,
                        "abs_diff": abs(tw - tr)})
        chk = pd.DataFrame(chk)
        ctx.save_csv(chk, "e3_witness_check")
        bad = chk[~(chk["abs_diff"] <= 1e-6 * (1 + chk["T_pooled_seed0"].abs()))]
        if len(bad):
            logger.warning("witness critic differs from the pooled seed-0 critic:\n%s",
                           bad.to_string())
        else:
            logger.info("witness critics reproduce the pooled seed-0 statistics (max |diff| "
                        "%.2e)", chk["abs_diff"].max())
    plotting.setup()
    models = [m for m in MODELS if m in set(w["model"])]
    fig, axes = plt.subplots(2, len(models), figsize=plotting.figsize("two", 0.55),
                             squeeze=False)
    black = plotting.OKABE_ITO["black"]
    for j, m in enumerate(models):
        g = w[w["model"] == m]
        fr, fs = g.loc[g["sample"] == "real", "f"], g.loc[g["sample"] == "synthetic", "f"]
        lo, hi = np.quantile(g["f"], [0.005, 0.995])
        bins = np.linspace(lo, hi, 31)
        ax = axes[0, j]
        ax.hist(fr, bins=bins, density=True, histtype="step", color=black, lw=1.0, label="Real")
        ax.hist(fs, bins=bins, density=True, histtype="step", color=plotting.COLORS_MODEL[m],
                lw=1.0, label="Synthetic")
        ax.set_title(f"{MODEL_LABELS[m]}\n$T$ = {fr.mean() - fs.mean():.3f}")
        ax.set_xlabel("critic value $f$")
        if j == 0:
            ax.set_ylabel("density")
            ax.legend(loc="upper left", fontsize=6.5)
        ax = axes[1, j]
        edges = np.quantile(g["log_rv"], np.linspace(0, 1, 11))
        for sample, col, mk in (("real", black, "o"), ("synthetic", plotting.COLORS_MODEL[m],
                                                       plotting.MARKERS_MODEL[m])):
            s = g[g["sample"] == sample]
            idx = np.clip(np.searchsorted(edges, s["log_rv"], side="right") - 1, 0, 9)
            means = s.groupby(idx).agg(x=("log_rv", "median"), f=("f", "mean"))
            ax.plot(means["x"], means["f"], color=col, marker=mk, ms=3, lw=0.9)
        ax.set_xlabel("window log realised vol.")
        if j == 0:
            ax.set_ylabel("mean $f$ (decile bins)")
    ctx.savefig(fig, "e3_critic_witness")


def build_stylized_outputs(ctx: common.ExpContext) -> None:
    if not ctx.has_csv("e3_stylized_raw"):
        return
    raw = ctx.load_csv("e3_stylized_raw")
    raw = raw[~raw["ticker"].isin(["median", "q25", "q75", "iqr"])]
    sources = ["Real"] + [m for m in MODELS if m in set(raw["source"])]
    rows, summ = [], []
    for stat, label, fac, d in STYL_ROWS:
        if stat not in raw:
            continue
        row = {"Statistic: panel median (IQR)": label}
        for src in sources:
            x = raw.loc[raw["source"] == src, stat].to_numpy(float) * fac
            x = x[np.isfinite(x)]
            if x.size == 0:
                row[MODEL_LABELS[src]] = "--"
                continue
            q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
            row[MODEL_LABELS[src]] = f"{tnum(q50, d)} ({tnum(q75 - q25, d)})"
            summ.append({"statistic": stat, "source": src, "median": q50 / fac,
                         "q25": q25 / fac, "q75": q75 / fac, "n": int(x.size)})
        rows.append(row)
    ctx.save_csv(pd.DataFrame(summ), "e3_stylized_summary")
    ctx.save_table(pd.DataFrame(rows).set_index("Statistic: panel median (IQR)"), "e3_stylized",
                   escape=False)
    if ctx.has_csv("e3_stylized_acf") and ctx.has_csv("e3_stylized_tail"):
        plot_stylized(ctx, ctx.load_csv("e3_stylized_acf"), ctx.load_csv("e3_stylized_tail"))


def plot_stylized(ctx: common.ExpContext, acf: pd.DataFrame, tail: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plotting.setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=plotting.figsize("two", 0.36))
    sources = ["Real"] + [m for m in MODELS if m in set(acf["source"])]
    for src in sources:
        col = plotting.COLORS_MODEL[src]
        lw = 1.6 if src == "Real" else 1.0
        s = acf[acf["source"] == src]
        a1.plot(s["lag"], s["median"], color=col, lw=lw, label=MODEL_LABELS[src],
                marker=plotting.MARKERS_MODEL[src], markevery=10, ms=3)
        if src == "Real":
            a1.fill_between(s["lag"], s["q25"], s["q75"], color=col, alpha=0.12, lw=0)
        t = tail[tail["source"] == src]
        a2.loglog(t["x"], t["ccdf"], color=col, lw=lw, label=MODEL_LABELS[src])
    a1.axhline(0, color=plotting.OKABE_ITO["grey"], lw=0.6)
    a1.set_xlabel("lag (days)")
    a1.set_ylabel("ACF of $|r|$ (median over tickers)")
    a2.set_xlabel(r"$|r| / \hat\sigma_{\mathrm{TRAIN}}$")
    a2.set_ylabel("empirical CCDF (pooled)")
    a2.set_xlim(left=1.0)
    a2.legend(loc="lower left", fontsize=6.5)
    ctx.savefig(fig, "e3_stylized")


def build_likelihood_tables(ctx: common.ExpContext) -> None:
    if not ctx.has_csv("e3_likelihood_raw"):
        return
    lk = ctx.load_csv("e3_likelihood_raw")
    best = lk.loc[lk.groupby("ticker")["pred_loglik"].idxmax(), "model"].value_counts()
    nt = lk["ticker"].nunique()
    rows = []
    for m in LIK_MODELS:
        g = lk[lk["model"] == m]
        pb, pk = g["berk_p"].to_numpy(float), g["ks_p"].to_numpy(float)
        rows.append({
            "Model": MODEL_LABELS[m],
            "Mean log-score": tnum(g["pred_loglik"].mean(), 3),
            r"Mean log-score, $r/\hat\sigma$": tnum(g["pred_loglik_z"].mean(), 3),
            "Best (tickers)": f"{int(best.get(m, 0))}/{nt}",
            r"Berkowitz rej.\ 5\% [95\% CI]": rate_ci(int(np.sum(pb <= ALPHA)), len(pb)),
            "Berkowitz BH": str(int(bh_reject(pb).sum())),
            r"PIT-KS rej.\ 5\% [95\% CI]": rate_ci(int(np.sum(pk <= ALPHA)), len(pk)),
            "PIT-KS BH": str(int(bh_reject(pk).sum())),
        })
    ctx.save_table(pd.DataFrame(rows).set_index("Model"), "e3_likelihood", escape=False)
    if not ctx.has_csv("e3_ag_raw"):
        return
    ag = ctx.load_csv("e3_ag_raw")
    arows = []
    for a, b in AG_PAIRS:
        g = ag[(ag["model_a"] == a) & (ag["model_b"] == b)]
        pool = g[g["ticker"] == "POOLED"]
        per = g[g["ticker"] != "POOLED"]
        rej = bh_reject(per["p"].to_numpy(float))
        z = per["z"].to_numpy(float)
        n = len(per)
        arows.append({
            "Comparison": f"{MODEL_LABELS[a]} vs {MODEL_LABELS[b]}",
            r"Pooled mean $d_t$": tnum(pool["mean_d"].iloc[0], 4) if len(pool) else "--",
            "Pooled $z$ (DK)": tnum(pool["z"].iloc[0], 2) if len(pool) else "--",
            "Pooled $p$": tp(pool["p"].iloc[0]) if len(pool) else "--",
            f"{MODEL_LABELS[a]} better (BH)": f"{int(np.sum(rej & (z > 0)))}/{n}",
            f"{MODEL_LABELS[b]} better (BH)": f"{int(np.sum(rej & (z < 0)))}/{n}",
            r"$z>0$ (tickers)": f"{int(np.sum(z > 0))}/{n}",
        })
    # columns differ per pair; build a uniform table with generic names
    gen = []
    for r, (a, b) in zip(arows, AG_PAIRS):
        gen.append({"Comparison (a vs b)": r["Comparison"],
                    r"Pooled mean $d_t$": r[r"Pooled mean $d_t$"],
                    "Pooled $z$ (DK)": r["Pooled $z$ (DK)"], "Pooled $p$": r["Pooled $p$"],
                    "a better (BH)": r[f"{MODEL_LABELS[a]} better (BH)"],
                    "b better (BH)": r[f"{MODEL_LABELS[b]} better (BH)"],
                    r"$z>0$ (tickers)": r[r"$z>0$ (tickers)"]})
    ctx.save_table(pd.DataFrame(gen).set_index("Comparison (a vs b)"), "e3_likelihood_ag",
                   escape=False)


def _rank_labels(s: Any) -> str:
    """``"GARCHt>Merton>GBM"`` -> ``"GARCH-t $>$ Merton $>$ GBM"``."""
    return " $>$ ".join(tex(MODEL_LABELS.get(x, x)) for x in str(s).split(">"))


def build_agree_table(ctx: common.ExpContext) -> None:
    """``e3_agreement.tex``: Kendall tau (descriptive) and Cohen kappa per model and
    stratified by model (calibrated per-ticker p-values when available)."""
    if not (ctx.has_csv("e3_agree_tau") and ctx.has_csv("e3_agree_kappa")):
        return
    tau = ctx.load_csv("e3_agree_tau")
    kap = ctx.load_csv("e3_agree_kappa")
    if "pvalue_basis" not in kap or "share_tau_m1" not in tau:
        logger.warning("agreement table: CSVs in the old format: rerun the agree part")
        return
    rows = []
    for _, r in tau.iterrows():
        if r["scope"] == "pooled":
            rows.append({"Measure": r"Kendall $\tau$, pooled (critic " + tex(r["critic_statistic"])
                         + ")", "Model / test": "3 likelihood models",
                         "Value": tnum(r["tau"], 2),
                         "Detail": f"critic: {_rank_labels(r['rank_critic'])}; log-score: "
                                   f"{_rank_labels(r['rank_loglik'])}"})
        else:
            rows.append({"Measure": r"Kendall $\tau$, per ticker (mean; descriptive)",
                         "Model / test": TEST_LABELS[r["test"]] + r", $L=5$",
                         "Value": tnum(r["tau"], 2),
                         "Detail": f"$\\tau=1$: {r['share_tau_1']:.2f}, $\\tau=-1$: "
                                   f"{r['share_tau_m1']:.2f} (random rankings: 0.17 each); "
                                   f"{int(r['n'])} tickers"})
    basis = "cal." if str(kap["pvalue_basis"].iloc[0]) == "calibrated" else "nominal"
    order = list(LIK_MODELS) + ["stratified"]
    for test in [t for t in ABS_TESTS if t in set(kap["test"])]:
        for m in order:
            r = kap[(kap["rule"] == "bh") & (kap["test"] == test) & (kap["model"] == m)]
            rr = kap[(kap["rule"] == "raw") & (kap["test"] == test) & (kap["model"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            raw_k = tnum(rr["kappa"].iloc[0], 2) if len(rr) else "--"
            who = ("3 models, within-model chance agreement" if m == "stratified"
                   else MODEL_LABELS[m])
            rows.append({"Measure": rf"Cohen $\kappa$ vs Berkowitz (BH; 5\%), {basis} $p$",
                         "Model / test": f"{who}, {TEST_LABELS[test]}",
                         "Value": f"{tnum(r['kappa'], 2)}; {raw_k}",
                         "Detail": f"both {int(r['both_reject'])}, test only "
                                   f"{int(r['only_test'])}, Berkowitz only "
                                   f"{int(r['only_berkowitz'])}, neither {int(r['neither'])}"})
    ctx.save_table(pd.DataFrame(rows).set_index("Measure"), "e3_agreement", escape=False)


def build_verdict_table(ctx: common.ExpContext) -> None:
    """``e3_verdicts.tex``: the decision rule per model with the rule's power and boundary size
    next to every verdict (theory.md 11.7). Single-seed quantities are not shown."""
    if not ctx.has_csv("e3_verdicts"):
        return
    v = ctx.load_csv("e3_verdicts")
    if "power_rule" not in v:
        logger.warning("verdict table: e3_verdicts.csv in the old format: rerun the verdict part")
        return
    rows = []
    for _, r in v.iterrows():
        n = int(r["n_seeds"])
        mark = DDAGGER if not bool(r["rel_valid"]) else ""
        dbd = str(r["boundary_rule_design"]) if isinstance(r["boundary_rule_design"], str) else ""
        dpw = str(r["power_rule_design"]) if isinstance(r["power_rule_design"], str) else ""
        if dpw and np.isfinite(r["power_rule"]):
            pw = (f"{tnum(r['power_rule'], 2)} [{tnum(r[f'power_{dpw}_ci_lo'], 2)}, "
                  f"{tnum(r[f'power_{dpw}_ci_hi'], 2)}] ({dpw})")
        else:
            pw = "--"
        if dbd:
            bd = (f"{tnum(r['boundary_rule'], 3)} [{tnum(r[f'boundary_{dbd}_cp_lo'], 3)}, "
                  f"{tnum(r[f'boundary_{dbd}_cp_hi'], 3)}] ({dbd})")
        else:
            bd = "--"
        rows.append({
            ("", "Model"): MODEL_LABELS[r["model"]],
            ("Absolute W-critic", r"$2\times$median $p$"): tp(r["abs_p_combined"]),
            ("Absolute W-critic", "seeds rej."): f"{int(r['abs_n_reject'])}/{n}",
            ("Anchored relative", r"$\Delta$ median [range]"):
                f"{tnum(r['delta_median'], 3)} [{tnum(r['delta_min'], 3)}, "
                f"{tnum(r['delta_max'], 3)}]" + mark,
            ("Anchored relative", r"$2\times$median $p$"): tp(r["rel_p_combined"]) + mark,
            ("Rule", r"power $\hat\beta$ [95\% CI]"): pw,
            ("Rule", r"boundary size [CP 95\%]"): bd,
            ("Verdict", r"$\hat\beta\ge0.8$"): r["verdict"],
            ("Verdict", r"$\beta_L\ge0.8$"): r["verdict_cp"],
        })
    tab = pd.DataFrame(rows)
    tab.columns = pd.MultiIndex.from_tuples(tab.columns)
    tab = tab.set_index(("", "Model"))
    tab.index.name = "Model"
    ctx.save_table(tab, "e3_verdicts", escape=False)


def build_outputs(ctx: common.ExpContext) -> None:
    build_fit_tables(ctx)
    build_power_table(ctx)
    build_pooled_outputs(ctx)
    build_perticker_table(ctx)
    build_witness_figure(ctx)
    build_stylized_outputs(ctx)
    build_likelihood_tables(ctx)
    build_agree_table(ctx)
    build_verdict_table(ctx)


# --------------------------------------------------------------------------------------------
# Timing probes (quick mode): heavy units at the FULL settings on the full 100-ticker panel
# --------------------------------------------------------------------------------------------
def _probe_rep(cell: dict, rep: int, rng: np.random.Generator) -> dict[str, Any]:
    kind = cell["kind"]
    F = FULL
    panel = get_panel(cell["_tickers"])
    out: dict[str, Any] = {}
    # power / realvol: one seed, boundary case only (the alternative costs the same: x 2)
    c1 = {"base_kw": F["base_kw"], "n_boot": F["n_boot"], "n_seeds": 1, "cases": ("boundary",)}
    if kind in ("pooled", "pooled_rel"):  # one model x seed, split in two probes (wall time)
        fitted, _ = pl.fit_panel("GBM", panel.train, n_workers=1)
        absolute = kind == "pooled"
        t0 = time.perf_counter()
        rows = pl.evaluate_model(panel, fitted, seed=synthetic_seed(0),
                                 tests=ABS_TESTS if absolute else (),
                                 relative_base=None if absolute else "critic",
                                 relative_methods=pl.RELATIVE_METHODS,
                                 protocol="dependent", test_kw=F["test_kw"],
                                 base_kw=F["base_kw"], n_boot=F["n_boot"], start_from_end=True,
                                 keep_extra=True)
        out["probe_seconds"] = time.perf_counter() - t0
        for r in rows:
            out[f"rt_{r['test']}"] = float(r["runtime_s"])
            out[f"steps_{r['test']}"] = _train_steps(r.get("extra") or {})["steps_run"]
    elif kind in ("power", "realvol"):  # one seed, boundary test only
        t0 = time.perf_counter()
        if kind == "power":
            fitted, _ = pl.fit_panel(pl.POWER_MODEL, panel.train, n_workers=1,
                                     **MODEL_KW[pl.POWER_MODEL])
            t0 = time.perf_counter()
            r = power_rep_rows(panel, fitted, 0, c1)[0]
        else:
            r = realvol_rep_rows(panel, 0, c1)[0]
        total = time.perf_counter() - t0
        t_test = float(r["test_seconds"])
        t_rep = float(r.get("sim_seconds_rep", 0.0))
        t_seed = float(r.get("sim_seconds_seed", 0.0))
        t_other = max(0.0, total - t_test - t_rep - t_seed)  # windows, inputs (per replication)
        # one replication: per-replication simulations + K seeds x (seed simulations + 2 tests)
        out["probe_seconds"] = t_rep + t_other + F["n_seeds"] * (t_seed + 2.0 * t_test)
        out.update(t_test=t_test, t_sim_rep=t_rep, t_sim_seed=t_seed, t_other=t_other)
    elif kind == "perticker":  # also the cost of one perticker_null pair
        tk = panel.tickers[:2]
        fitted, _ = pl.fit_panel("GBM", panel.train[tk], n_workers=1)
        sim = pl.simulate_panel(fitted, panel.n_test, seed=1)
        L = F["perticker_L"]
        c = {"tests": F["perticker_tests"], "test_kw": F["test_kw"]}
        t0 = time.perf_counter()
        for j, t in enumerate(tk):
            X, ox = _ticker_windows(panel, t, panel.test[t].to_numpy(), L)
            Y, oy = _ticker_windows(panel, t, sim[:, j], L)
            _ticker_tests(X, Y, ox, oy, c, common.keyed_seed_sequence(ROOT_SEED, "probe", t),
                          "probe")
        out["probe_seconds"] = (time.perf_counter() - t0) / len(tk)
    out["max_rss_mb"] = _max_rss_mb()
    return out


#: Parts whose cost each probe measures.
PROBE_PARTS: dict[str, set[str]] = {"pooled": {"pooled", "witness"},
                                    "pooled_rel": {"pooled"}, "power": {"power"},
                                    "realvol": {"realvol"},
                                    "perticker": {"perticker", "perticker_null"}}


def run_probes(ctx: common.ExpContext) -> pd.DataFrame:
    full_tickers = panel_tickers(FULL)
    cells = [{"cell": k, "kind": k, "_tickers": full_tickers} for k, parts in PROBE_PARTS.items()
             if parts & set(ctx.parts)]
    if not cells:
        return pd.DataFrame()
    cost = {"pooled": 3.0, "pooled_rel": 2.5, "power": 2.0, "realvol": 1.5, "perticker": 1.0}
    return common.run_cells(_probe_rep, cells, part="probe", ctx=ctx, root_seed=ROOT_SEED,
                            n_reps=1, config={"probe": True, "full": FULL}, block_size=1,
                            cost=lambda c: cost[c["kind"]], fresh=True)


def report_estimates(ctx: common.ExpContext, probe: pd.DataFrame, quick_timing: dict[str, float],
                     n_quick: int, workers: int = 12) -> dict[str, float]:
    """Print measured per-replication times and estimated full-run wall times per part, at
    ``workers`` workers and, where memory allows fewer, at the memory-feasible count."""
    F = FULL
    n_full = config.N_PANEL
    scale = n_full / n_quick
    K = int(F["n_seeds"])
    est: dict[str, float] = {}
    avail = mem_available_gb()
    t = {r["cell"]: r for _, r in probe.iterrows()} if not probe.empty else {}
    rss_gb: dict[str, float] = {}
    for k, r in t.items():
        mb = float(r["max_rss_mb"])
        for part in PROBE_PARTS[k]:
            rss_gb[part] = max(rss_gb.get(part, 0.0), mb / 1024.0)
        print(f"MEMORY {k} probe: peak RSS {mb:.0f} MB per worker process (about "
              f"{mb * workers / 1024:.1f} GB at {workers} workers; {avail:.1f} GB available "
              "now; a worker that ran two probes reports the larger)", flush=True)

    def estimate(part: str, tasks: list[float], **kw: Any) -> float:
        wall = common.print_wall_estimate(part, tasks, workers=workers, **kw)
        gb = WORKER_GB.get(part, DEFAULT_WORKER_GB)  # what part_workers() uses at run time
        wm = memory_workers(part, workers, avail)
        if part in rss_gb and wm < workers:
            wall = common.print_wall_estimate(f"{part} (memory-feasible, {gb:.2f} GB/worker)",
                                              tasks, workers=wm, **kw)
            print(f"WORKERS {part}: run with --workers {wm} (or the script caps it)", flush=True)
        return wall

    if "fit" in ctx.parts:
        tasks: list[float] = []
        for m in MODELS:
            if not ctx.has_csv(f"e3_fit_{m}"):
                continue
            fs = ctx.load_csv(f"e3_fit_{m}")["fit_seconds"].to_numpy(float)
            common.print_rep_timing(f"fit {m}", float(np.mean(fs)),
                                    note=f"per ticker, measured on {fs.size} tickers")
            tasks += list(np.resize(fs, n_full))
        est["fit"] = common.print_wall_estimate("fit", tasks, workers=workers,
                                                extra_s=3.0 * len(MODELS))
    for part, reps_key in (("power", "power_reps"), ("realvol", "realvol_reps")):
        if part in t and part in ctx.parts:
            r = t[part]
            s = float(r["probe_seconds"])
            common.print_rep_timing(part, s, note=f"full settings, {K} seeds x (vol x1.2 + "
                                    f"boundary): one test {float(r['t_test']):.1f} s, "
                                    f"simulations {float(r['t_sim_rep']):.1f} s per rep + "
                                    f"{float(r['t_sim_seed']):.1f} s per seed")
            est[part] = estimate(part, common.block_tasks(s, F[reps_key], F["blocks"][part]))
    if "pooled" in t:
        pr = [t[k] for k in ("pooled", "pooled_rel") if k in t]
        s = sum(float(r["probe_seconds"]) for r in pr)
        rts = {k[3:]: float(v) for r in pr for k, v in r.items()
               if str(k).startswith("rt_") and np.isfinite(v)}
        steps = {k[6:]: float(v) for r in pr for k, v in r.items()
                 if str(k).startswith("steps_") and np.isfinite(v)}
        common.print_rep_timing("pooled", s, note="full settings, one model x seed: "
                                + ", ".join(f"{k} {v:.1f}s" for k, v in rts.items()))
        if steps:
            print("STEPS pooled probe (optimiser steps run): "
                  + ", ".join(f"{k} {v:.0f}" for k, v in steps.items()), flush=True)
        if "pooled" in ctx.parts:
            est["pooled"] = estimate("pooled", [s] * (len(MODELS) * K))
        if "witness" in ctx.parts:
            sw = rts.get("critic", s / 8)
            common.print_rep_timing("witness", sw, note="one critic fit per model")
            est["witness"] = estimate("witness", [sw] * len(MODELS))
    if "perticker" in t:
        s = float(t["perticker"]["probe_seconds"])
        if "perticker" in ctx.parts:
            common.print_rep_timing("perticker", s, note="full settings, critic + MMD, "
                                    "one ticker")
            tasks = []
            for _ in MODELS:
                tasks += common.block_tasks(s, n_full, F["blocks"]["perticker"])
            est["perticker"] = estimate("perticker", tasks)
        if "perticker_null" in ctx.parts:
            B = int(F["perticker_null_draws"])
            common.print_rep_timing("perticker_null", s * B, note=f"full settings, one ticker "
                                    f"x {B} null pairs (critic + MMD)")
            est["perticker_null"] = estimate(
                "perticker_null", common.block_tasks(s * B, n_full,
                                                     F["blocks"]["perticker_null"]))
    for part in ("stylized", "likelihood"):
        if f"{part}_wall_s" in quick_timing:
            s = quick_timing[f"{part}_wall_s"]
            common.print_rep_timing(part, s / n_quick, note="per ticker, serial")
            est[part] = common.print_wall_estimate(part, [s * scale], workers=workers,
                                                   overhead_s=5.0)
    for part in DERIVED_PARTS:
        if f"{part}_wall_s" in quick_timing:
            s = quick_timing[f"{part}_wall_s"]
            common.print_rep_timing(part, s, note="reads CSVs only")
            est[part] = common.print_wall_estimate(part, [s], workers=workers, overhead_s=5.0)
    if est:
        tot = sum(est.values())
        print(f"ESTIMATED FULL-RUN WALL TIME all parts sequentially (memory-capped where "
              f"needed, else {workers} workers): {common.fmt_seconds(tot)} ({tot / 3600:.2f} h)",
              flush=True)
    return est


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------
def _add_args(parser: Any) -> None:
    parser.add_argument("--no-probe", action="store_true",
                        help="quick mode: skip the full-settings timing probes")


def main(argv: list[str] | None = None) -> None:
    ctx = common.parse_args(EXP, PARTS, description=__doc__, add_arguments=_add_args, argv=argv)
    cfg = QUICK if ctx.quick else FULL
    tickers = panel_tickers(cfg)
    sw = common.Stopwatch()
    timing: dict[str, float] = {}
    ctx.write_meta({"quick": ctx.quick, "parts": ctx.parts, "plots_only": ctx.plots_only,
                    "cfg": cfg, "root_seed": ROOT_SEED, "models": list(MODELS),
                    "model_kw": MODEL_KW, "tickers": tickers,
                    "pipeline": pl.pipeline_config()})
    runners = {"fit": run_fit, "power": run_power, "realvol": run_realvol, "pooled": run_pooled,
               "perticker": run_perticker, "perticker_null": run_perticker_null,
               "witness": run_witness, "stylized": run_stylized, "likelihood": run_likelihood}
    if not ctx.plots_only:
        for part in ctx.parts:
            if part in runners:
                t0 = time.perf_counter()
                timing.update(runners[part](ctx, cfg, tickers))
                logger.info("part %s done in %s", part,
                            common.fmt_seconds(time.perf_counter() - t0))
    # Derived parts are cheap and always recomputed from the current CSVs, so their tables
    # never mix stale derived results with new inputs.
    for part, fn in (("agree", compute_agree), ("verdict", compute_verdict)):
        t0 = time.perf_counter()
        if fn(ctx):
            timing[f"{part}_wall_s"] = time.perf_counter() - t0
    sw.lap("compute")
    build_outputs(ctx)
    sw.lap("outputs")
    if ctx.quick and not ctx.plots_only and not ctx.args.no_probe:
        probe = run_probes(ctx)
        sw.lap("probes")
        est = report_estimates(ctx, probe, timing, len(tickers))
        timing.update({f"estimate_wall_s_{k}": v for k, v in est.items()})
    timing.update({f"wall_{k}_s": v for k, v in sw.laps.items()})
    ctx.write_meta(None, timing=timing)
    print(f"TIMING total wall (this run, {ctx.workers} workers): {common.fmt_seconds(sw.total())}",
          flush=True)


if __name__ == "__main__":
    main()
