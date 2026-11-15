"""E4: Markowitz utility of synthetic data and its link with two-sample tests (DESIGN section 4, E4).

Universes. 30 random universes of N = 30 assets drawn from the complete universe of 479 S&P 500
names (``data.complete_universe``); universe ``u`` is ``sorted(rng_u.choice(universe, 30))`` with
``rng_u = keyed_rng(ROOT_SEED, "universe", u)``, so the first universes are the same whatever the
number drawn (quick mode uses the first three). Universes are drawn independently and may share
tickers. Returns are daily log-returns, TRAIN 2018-2022 (1258 days) and TEST 2023-01-03 to
2025-06-16 (615 days).

Generators (``synthval.models``, fitted on real TRAIN, default options): MVGaussian, MVStudentT,
CCCGARCHt (GARCHt margins with variance targeting and persistence cap 0.995, Gaussian copula),
MVBootstrap (row-wise stationary bootstrap) and FactorGARCHt (one GARCHt market factor + GARCHt
idiosyncratic terms).

Sub-experiments (``--part``):

utility
    Per universe and generator: B = 20 synthetic TRAIN-length samples (1258 days, stationary start
    with a 250-day burn-in). On each sample, moments are estimated with the sample covariance and
    with Ledoit-Wolf (identity target; sample mean in both cases) and six portfolios are built:
    GMV unconstrained (``gmv``), GMV long-only (``gmv_lo``), long-only mean-variance with
    gamma in {1, 5, 20} (``mv1``, ``mv5``, ``mv20``) and 1/N (``ew``). Each is evaluated on real
    TEST (``portfolio.utility_gap``: daily-rebalanced simple returns) against the same rule
    estimated on real TRAIN (TRTR). Utility losses (positive = synthetic-trained portfolio worse):
    ``var_ratio = var_synth / var_real``, ``sharpe_loss = Sharpe_real - Sharpe_synth``
    (annualised), ``ce_loss = CE_real - CE_synth`` (daily; CE with the portfolio's own gamma for
    mean-variance rules and gamma = 5 for GMV and 1/N). Aggregations (theory.md note 11.6):
    ``replica`` (mean over the B replica-wise portfolios: the DESIGN measure; it includes the
    estimation-noise term of theory Cor. 9.3(iii)), ``pooled`` (portfolio from the moments of the
    B samples stacked: isolates the covariance misspecification, Cor. 9.3(i)/(iii)) and ``wavg``
    (weights averaged over replicas, Proposition 9.4). Diagnostics: mean pairwise correlation
    (``portfolio.average_correlation``) and mean pairwise correlation of rolling 20-day
    log-volatilities, real vs each generator; ``kappa = cond(S_te^{-1} S_syn)`` and its
    Kantorovich bound ``(1 + kappa)^2 / (4 kappa)`` on the GMV variance ratio to the ex-post
    optimum (Proposition 9.2(b); checked replica by replica); GARCH components at the persistence
    cap 0.995 of the GARCHt fit (CCCGARCHt: 30 margins per universe; FactorGARCHt: factor + 30
    residuals). The cap is the near-unit-root diagnostic: no fitted component can exceed it, so a
    count of ``alpha + beta >= 0.999`` would be zero by construction. 1/N is the same rule for
    both samples, so its gaps are zero by construction; it is reported in the TRTR table only.
tests
    Per universe and generator: one synthetic sample over the TEST horizon (615 days, simulated
    with ``start_from_end=True``: GARCH filters continue from the end of TRAIN) against the real
    TEST sample, both divided by the per-ticker real TRAIN standard deviation and cut into
    non-overlapping multivariate windows of shape (N, L) = (30, 5) (123 windows each).
    Dependent-window protocol (``pipeline.absolute_tests(protocol="dependent")``): W-critic with
    a 10-block time split and the Driscoll-Kraay HAC p-value (``pvalue_type = "hac"``), MMD with
    the wild bootstrap (``"wild"``); the permutation p-values are also recorded (``p_perm``,
    invalid under dependence). ``order_*`` = date rank of each window's first day (TEST ranks
    1258 + 5k for real and synthetic windows alike). Common scale within a universe: the test
    generator is keyed by (universe, seed) only, so every generator of a universe gets the same
    time-block train/eval split, critic initialisation and mini-batch stream, and the MMD uses a
    fixed bandwidth per universe (median heuristic on the real TEST windows) instead of the pooled
    median. The critic still standardises its inputs with the pooled training half (a library
    choice; this scale depends on the generator). Repeated over 10 synthetic seeds (the E3
    convention; ``--test-seeds`` overrides and extends without recomputing): rejection shares for
    the first seed (one test per universe) and for twice the median p-value over seeds; the link
    uses seed means. Reference row ``RealTRAIN``: the last 615 TRAIN days (123 windows,
    2020-07 to 2022-12, date ranks 643 + 5k) in place of the synthetic sample, the same size as
    every generator row (how far the recent past is from the future).
placebo
    Size of exactly the tests-part call under H0, per generator: "real" X* and "synthetic" Y* are
    two independent TEST-length draws (``start_from_end=True``) from the generator fitted on a
    universe, normalised, windowed and ordered as real TEST windows, with the MMD bandwidth from
    X* (the rule applied to the real sample in part tests). 200 replications per generator,
    replication ``r`` on universe ``r mod 30``. Gives the empirical size of the HAC / wild (and
    permutation) p-values with Wilson 95% intervals, the mean and sd of the studentised
    statistic (``z_hac``; for MMD the wild-standardised V-statistic), a KS test of p-value
    uniformity, and the placebo null used for size-adjusted rejection shares in part tests:
    ``p_cal = (1 + #{placebo score >= score}) / (1 + n_placebo)`` against the placebo scores of
    the same generator and test, pooled over universes (score: ``z_hac`` for the critic, minus the
    wild p-value for MMD). This is a Monte Carlo test of H0 "the real TEST sample is a draw from
    the fitted generator" calibrated on the mixture over universes (approximate per universe).
link
    Spearman correlation between each test quantity (W-critic statistic T and HAC z, MMD^2 and its
    wild-standardised statistic, the p-values, and, when part placebo has run, the z
    standardised by the generator's placebo mean and sd; plus the absolute average-correlation
    error as a non-test benchmark) and each utility loss (sample and LW moments; replica, pooled
    and wavg aggregations). Four views, because there are only 5 generators:
    *pooled* over the 150 (universe, generator) points and *within-universe* (Spearman across
    the 5 generators of each universe, averaged), both with universe-bootstrap 95% CIs and, for
    within-universe, a permutation p-value from permuting generator labels within each universe;
    both mostly rank the 5 generators. *Within-generator*: Spearman across universes for each
    generator, averaged over generators (universe-bootstrap CI and a permutation p-value from
    permuting universes within each generator); it is insensitive to generator-level offsets of
    a statistic (such as the placebo bias of the critic). *Generator-level*: Spearman of the 5
    generator means with its exact permutation p-value over the 5! orderings: the only view whose
    sampling unit is the generator, n = 5. Conclusions about RQ5 are conditional on these
    generators.

Outputs (full mode; ``--quick`` writes all of them under ``results/e4_quick/`` only):
``results/e4/{universes, e4_utility_reps, e4_utility, e4_trtr, e4_diagnostics, e4_tests,
e4_tests_summary, e4_placebo, e4_placebo_size, e4_link_points, e4_link,
e4_link_by_generator}.csv`` and ``meta*.json``; tables ``paper/tables/e4_{utility, utility_agg,
trtr, correlation, tests, placebo, link, link_pooled}.tex`` with notes ``e4_{tests, placebo,
link}_note.tex``; figures ``paper/figures/e4_utility_vs_distance.pdf`` and
``e4_placebo_size.pdf``.

Usage::

    uv run python experiments/e4_markowitz.py --quick            # smoke test + timing estimate
    uv run python experiments/e4_markowitz.py --part utility --workers 12
    uv run python experiments/e4_markowitz.py --part tests --workers 12
    uv run python experiments/e4_markowitz.py --part placebo --workers 12
    uv run python experiments/e4_markowitz.py --part link        # after utility, tests, placebo
    uv run python experiments/e4_markowitz.py --plots-only

Caveats recorded with the results: universes overlap in tickers and all share the TEST dates, so
the Wilson intervals of rejection shares across universes (which assume independent universes)
and the universe bootstrap understate the uncertainty due to common market shocks; rejection
shares are descriptive. The nominal HAC / wild p-values are oversized in this configuration
(part placebo measures by how much); read the size-adjusted shares.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import functools  # noqa: E402
import itertools  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from typing import Any  # noqa: E402

import common  # noqa: E402  (experiments/ is on sys.path when run as a script)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import linalg, stats  # noqa: E402

from synthval import config, data, pipeline, plotting  # noqa: E402
from synthval import portfolio as pf  # noqa: E402
from synthval.models import get_mv_model  # noqa: E402
from synthval.twosample.mmd import median_heuristic  # noqa: E402

logger = logging.getLogger("experiments.e4")

EXP = "e4"
ROOT_SEED = 20251118
PARTS = ("utility", "tests", "placebo", "link")

GENERATORS: tuple[str, ...] = ("MVGaussian", "MVStudentT", "CCCGARCHt", "MVBootstrap",
                               "FactorGARCHt")
#: Reference "generator" of the tests part: the last T1 days of real TRAIN.
REF_GEN = "RealTRAIN"
#: Pseudo-generator of the utility part holding the TRTR (real-trained) results.
REAL = "Real"
ESTIMATORS: tuple[str, ...] = ("sample", "lw")
PORTFOLIOS: tuple[str, ...] = ("gmv", "gmv_lo", "mv1", "mv5", "mv20", "ew")
PORT_GAMMA: dict[str, float] = {"mv1": 1.0, "mv5": 5.0, "mv20": 20.0}
#: Risk aversion of the certainty equivalent of the GMV and 1/N portfolios.
CE_GAMMA_DEFAULT = 5.0
AGGS: tuple[str, ...] = ("replica", "pooled", "wavg")
TESTS: tuple[str, ...] = ("critic", "mmd")
#: p-value type of each test under the dependent-window protocol.
TEST_PTYPE: dict[str, str] = {"critic": "hac", "mmd": "wild"}
LOSSES: tuple[str, ...] = ("var_ratio", "sharpe_loss", "ce_loss")
#: Persistence cap of the GARCHt fits (``GARCHt(persist_max=0.995)``, the experiment default).
PERSIST_CAP = 0.995
LOGVOL_WINDOW = 20
TD = config.TRADING_DAYS
ALPHAS: tuple[float, ...] = (0.01, 0.05, 0.10)

FULL: dict[str, Any] = {
    "n_universes": 30, "n_assets": 30, "n_replicas": 20, "burn_in": pipeline.BURN_IN,
    "L": 5, "test_seeds": 10, "test_kw": {}, "n_placebo": 200, "n_boot_link": 2000,
    "n_perm_link": 2000, "alpha": pipeline.ALPHA,
}
QUICK: dict[str, Any] = {
    "n_universes": 3, "n_assets": 30, "n_replicas": 3, "burn_in": pipeline.BURN_IN,
    "L": 5, "test_seeds": 1, "test_kw": {"critic": {"n_steps": 150}, "mmd": {"n_perm": 200}},
    "n_placebo": 3, "n_boot_link": 200, "n_perm_link": 200, "alpha": pipeline.ALPHA,
}

GEN_LABELS: dict[str, str] = {
    "MVGaussian": "MV Gaussian", "MVStudentT": "MV Student-t", "CCCGARCHt": "CCC-GARCH-t",
    "MVBootstrap": "MV bootstrap", "FactorGARCHt": "Factor-GARCH-t",
    REF_GEN: "Real TRAIN, last 615 days (reference)", REAL: "Real TRAIN",
}
TEST_LABELS: dict[str, str] = {"critic": "W-critic (HAC)", "mmd": "MMD (wild)"}
PORT_LABELS: dict[str, str] = {
    "gmv": "GMV", "gmv_lo": "GMV long-only", "mv1": r"MV $\gamma=1$", "mv5": r"MV $\gamma=5$",
    "mv20": r"MV $\gamma=20$", "ew": "$1/N$",
}
EST_LABELS: dict[str, str] = {"sample": "Sample", "lw": "LW"}
#: plotting.COLORS_MODEL / MARKERS_MODEL have no FactorGARCHt entry yet (fallback here).
_FALLBACK_COLOR = {"FactorGARCHt": plotting.OKABE_ITO["purple"]}
_FALLBACK_MARKER = {"FactorGARCHt": "d"}

PERM_LABEL = "permutation (invalid under dependence)"
AGG_FIRST = "first seed"
AGG_MEDIAN = "2 x median over seeds"
CAL_NOMINAL = "nominal"
CAL_ADJ = "size-adjusted (placebo)"
CI_NOTE = ("Wilson 95% CI across universes assumes independent universes; universes share every "
           "TEST date and overlap in tickers, so it is descriptive")
CAVEATS: tuple[str, ...] = (
    CI_NOTE,
    "nominal HAC / wild p-values are oversized in this configuration; part placebo measures "
    "the size and gives size-adjusted shares (placebo pooled over universes)",
    "critic inputs are standardised with the pooled training half (generator-dependent scale; "
    "no library option for a fixed scale)",
    "the link has 5 generators: pooled and within-universe correlations mostly rank them; the "
    "generator-level correlation has n = 5",
)


def gen_color(g: str) -> str:
    return plotting.COLORS_MODEL.get(g, _FALLBACK_COLOR.get(g, plotting.OKABE_ITO["grey"]))


def gen_marker(g: str) -> str:
    return plotting.MARKERS_MODEL.get(g, _FALLBACK_MARKER.get(g, "o"))


def _num(v: Any) -> float:
    """Float or NaN."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------------------------
# Data, universes and generators (cached per worker process)
# --------------------------------------------------------------------------------------------
def draw_universes(n_universes: int, n_assets: int) -> tuple[list[tuple[str, ...]], int]:
    """Universe ``u``: ``n_assets`` names of the complete universe drawn with its own keyed rng.

    Returns the universes and the size of the complete universe they are drawn from.
    """
    names = data.complete_universe(data.load_prices())
    out = []
    for u in range(n_universes):
        rng = common.keyed_rng(ROOT_SEED, "universe", u)
        out.append(tuple(sorted(rng.choice(names, n_assets, replace=False).tolist())))
    return out, len(names)


def universe_config(universes: list[tuple[str, ...]], universe_size: int) -> dict[str, Any]:
    """Checkpoint-hash entries identifying the tickers of every universe."""
    return {"universes_hash": common.config_hash({"universes": [list(t) for t in universes]}),
            "universe_size": int(universe_size)}


@functools.lru_cache(maxsize=64)
def load_universe(tickers: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(R_train (T0, N), R_test (T1, N), scale (N,))``; scale = real TRAIN std per ticker."""
    tr, te = data.load_panel(list(tickers))
    scale = data.train_vol(tr).to_numpy(dtype=np.float64)
    return tr.to_numpy(dtype=np.float64), te.to_numpy(dtype=np.float64), scale


@functools.lru_cache(maxsize=8)
def fit_generator(gen: str, tickers: tuple[str, ...]) -> tuple[Any, float]:
    """Fitted generator on the universe's real TRAIN returns and the fit time (deterministic)."""
    R = load_universe(tickers)[0]
    t0 = time.perf_counter()
    model = get_mv_model(gen).fit(R)
    return model, time.perf_counter() - t0


def _quiet_worker_logging() -> None:
    # GARCHt logs one warning per margin at the persistence cap; the counts are recorded in the
    # diagnostics table instead.
    logging.getLogger("synthval.models").setLevel(logging.ERROR)


# --------------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------------
def logvol_corr(X: np.ndarray, w: int = LOGVOL_WINDOW) -> float:
    """Mean pairwise correlation of rolling ``w``-day log-volatilities (same definition as the
    ``resid_logvol_corr_mean`` diagnostic of FactorGARCHt); NaN if undefined."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2 * w or X.shape[1] < 2:
        return float("nan")
    sd = np.lib.stride_tricks.sliding_window_view(X, w, axis=0).std(axis=-1, ddof=1)
    if not np.all(sd > 0):
        return float("nan")
    C = np.corrcoef(np.log(sd), rowvar=False)
    N = C.shape[0]
    return float((C.sum() - np.trace(C)) / (N * (N - 1)))


def kappa_kantorovich(S_hat: np.ndarray, S_te: np.ndarray) -> tuple[float, float]:
    """``kappa = cond(S_te^{-1} S_hat)`` (generalised eigenvalues) and ``(1+kappa)^2/(4 kappa)``."""
    ev = linalg.eigh(S_hat, S_te, eigvals_only=True)
    k = float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")
    return k, float((1.0 + k) ** 2 / (4.0 * k)) if np.isfinite(k) else float("inf")


def moments(est: str, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, S = pf.sample_moments(R)
    if est == "lw":
        S = pf.ledoit_wolf(R)
    return mu, S


def build_weights(port: str, mu: np.ndarray, S: np.ndarray) -> np.ndarray:
    if port == "gmv":
        return pf.gmv_weights(S)
    if port == "gmv_lo":
        return pf.gmv_weights(S, long_only=True)
    if port in PORT_GAMMA:
        return pf.mean_variance_weights(mu, S, PORT_GAMMA[port], long_only=True)
    if port == "ew":
        return pf.equal_weights(mu.size)
    raise ValueError(port)


def all_weights(R: np.ndarray) -> tuple[dict[tuple[str, str], np.ndarray | None], int]:
    """Weights of every (estimator, portfolio) rule on sample ``R``; failures give ``None``."""
    out: dict[tuple[str, str], np.ndarray | None] = {}
    n_fail = 0
    for est in ESTIMATORS:
        mu, S = moments(est, R)
        for port in PORTFOLIOS:
            try:
                w = np.asarray(build_weights(port, mu, S), dtype=np.float64)
                if not np.all(np.isfinite(w)):
                    raise FloatingPointError("non-finite weights")
            except Exception as exc:  # noqa: BLE001 - one NaN row instead of a lost run
                logger.warning("portfolio %s/%s failed: %s", est, port, exc)
                w, n_fail = None, n_fail + 1
            out[(est, port)] = w
    return out, n_fail


def ce_gamma(port: str) -> float:
    return PORT_GAMMA.get(port, CE_GAMMA_DEFAULT)


_GAP_KEYS = ("mean", "vol", "var", "sharpe", "max_drawdown", "ce")


def gap_metrics(w_syn: np.ndarray | None, w_real: np.ndarray | None, R_te: np.ndarray,
                port: str, S_te: np.ndarray, v_opt: float) -> dict[str, float]:
    """Out-of-sample metrics of the synthetic- and real-trained portfolios and the losses."""
    nan = float("nan")
    if w_syn is None or w_real is None:
        row = {f"{k}_{s}": nan for k in _GAP_KEYS for s in ("synth", "real")}
        row.update(var_ratio=nan, sharpe_loss=nan, ce_loss=nan, weight_l1=nan, gross_synth=nan,
                   v_rel_opt=nan)
        return row
    g = pf.utility_gap(w_syn, w_real, R_te, gamma=ce_gamma(port))
    row = {f"{k}_{s}": g[f"{k}_{s}"] for k in _GAP_KEYS for s in ("synth", "real")}
    row.update(var_ratio=g["var_ratio"], sharpe_loss=g["sharpe_real"] - g["sharpe_synth"],
               ce_loss=g["ce_loss"], weight_l1=g["weight_l1"],
               gross_synth=float(np.abs(w_syn).sum()),
               v_rel_opt=float(w_syn @ S_te @ w_syn / v_opt) if port == "gmv" else nan)
    return row


def garch_counts(gen: str, model: Any) -> dict[str, Any]:
    """At-cap counts of the GARCH components of CCCGARCHt and FactorGARCHt.

    ``n_at_cap`` (the fit hit the persistence cap ``PERSIST_CAP``) is the near-unit-root
    diagnostic; a threshold above the cap (e.g. 0.999) would count zero by construction.
    """
    if gen == "CCCGARCHt":
        pers = np.asarray(model.persistence_, dtype=float)
        bounds = model.margin_at_bound_
    elif gen == "FactorGARCHt":
        pers = np.asarray(model.persistence_, dtype=float)
        bounds = model.fit_info_["at_bound"]
    else:
        return {}
    return {
        "n_comp": int(pers.size),
        "n_at_cap": int(sum("persistence" in b for b in bounds)),
        "n_any_bound": int(sum(bool(b) for b in bounds)),
        "persist_median": float(np.median(pers)), "persist_max": float(np.max(pers)),
    }


def model_extras(gen: str, model: Any) -> dict[str, Any]:
    if gen == "MVStudentT":
        return {"nu": float(model.nu_)}
    if gen == "MVBootstrap":
        return {"mean_block": float(model.mean_block_)}
    if gen == "FactorGARCHt":
        fi = model.fit_info_
        return {"r2_mean": float(fi["r2_mean"]), "corr_mean_implied": float(fi["corr_mean_implied"]),
                "resid_corr_rms": float(fi["resid_corr_rms"])}
    if gen == "CCCGARCHt":
        P = model.R_
        N = P.shape[0]
        return {"copula_corr_mean": float((P.sum() - N) / (N * (N - 1)))}
    return {}


# --------------------------------------------------------------------------------------------
# Part "utility"
# --------------------------------------------------------------------------------------------
def _real_task(tickers: tuple[str, ...]) -> list[dict[str, Any]]:
    """TRTR performance and real-data diagnostics of one universe."""
    R, R_te, _ = load_universe(tickers)
    _, S_te = pf.sample_moments(R_te)
    w_opt = pf.gmv_weights(S_te)
    v_opt = float(w_opt @ S_te @ w_opt)
    W, n_fail = all_weights(R)
    rows: list[dict[str, Any]] = []
    for (est, port), w in W.items():
        row: dict[str, Any] = {"kind": "trtr", "est": est, "port": port}
        if w is None:
            row.update({k: float("nan") for k in _GAP_KEYS})
        else:
            ev = pf.evaluate(w, R_te, gamma=ce_gamma(port))
            row.update({k: ev[k] for k in _GAP_KEYS})
            row["gross"] = float(np.abs(w).sum())
            row["v_rel_opt"] = float(w @ S_te @ w / v_opt) if port == "gmv" else float("nan")
        rows.append(row)
    _, S_tr = pf.sample_moments(R)
    k, kb = kappa_kantorovich(S_tr, S_te)
    rows.append({"kind": "diag", "avg_corr_train": pf.average_correlation(R),
                 "avg_corr_test": pf.average_correlation(R_te), "logvol_corr_train": logvol_corr(R),
                 "logvol_corr_test": logvol_corr(R_te), "kappa": k, "kantorovich": kb,
                 "n_fail": n_fail})
    return rows


def _utility_task(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """All B replicas of one (universe, generator) cell (rng streams keyed by replica)."""
    _quiet_worker_logging()
    tickers, u, gen, cfg = cell["_tickers"], cell["universe"], cell["generator"], cell["_cfg"]
    if gen == REAL:
        return _real_task(tickers)
    R, R_te, _ = load_universe(tickers)
    T0 = R.shape[0]
    _, S_te = pf.sample_moments(R_te)
    w_opt = pf.gmv_weights(S_te)
    v_opt = float(w_opt @ S_te @ w_opt)
    W_real, _ = all_weights(R)
    model, fit_s = fit_generator(gen, tickers)
    rows: list[dict[str, Any]] = []
    samples, w_sum, w_cnt = [], {}, {}
    corr, lvc, kap, kant, kant_ok = [], [], [], [], []
    n_fail, sim_s = 0, 0.0
    for b in range(int(cfg["n_replicas"])):
        rng_b = common.keyed_rng(ROOT_SEED, "utility", u, gen, b)
        t0 = time.perf_counter()
        Y = np.asarray(model.simulate(T0, rng=rng_b, burn_in=int(cfg["burn_in"])), dtype=float)
        sim_s += time.perf_counter() - t0
        if not np.all(np.isfinite(Y)):
            raise FloatingPointError(f"{gen} produced non-finite returns (universe {u}, rep {b})")
        samples.append(Y)
        corr.append(pf.average_correlation(Y))
        lvc.append(logvol_corr(Y))
        k, kb = kappa_kantorovich(pf.sample_moments(Y)[1], S_te)
        kap.append(k)
        kant.append(kb)
        W_syn, nf = all_weights(Y)
        n_fail += nf
        for (est, port), w in W_syn.items():
            m = gap_metrics(w, W_real[(est, port)], R_te, port, S_te, v_opt)
            rows.append({"kind": "rep", "replica": b, "est": est, "port": port, **m})
            if est == "sample" and port == "gmv" and np.isfinite(m["v_rel_opt"]):
                kant_ok.append(m["v_rel_opt"] <= kb * (1 + 1e-9))
            if w is not None:
                w_sum[(est, port)] = w_sum.get((est, port), 0.0) + w
                w_cnt[(est, port)] = w_cnt.get((est, port), 0) + 1
    # Pooled moments (B x T0 rows) and averaged weights.
    Y_pool = np.vstack(samples)
    W_pool, nf = all_weights(Y_pool)
    n_fail += nf
    for (est, port), w in W_pool.items():
        m = gap_metrics(w, W_real[(est, port)], R_te, port, S_te, v_opt)
        rows.append({"kind": "pooled", "est": est, "port": port, **m})
    for est in ESTIMATORS:
        for port in PORTFOLIOS:
            n = w_cnt.get((est, port), 0)
            w = w_sum[(est, port)] / n if n else None
            m = gap_metrics(w, W_real[(est, port)], R_te, port, S_te, v_opt)
            rows.append({"kind": "wavg", "est": est, "port": port, **m})
    k_pool, kb_pool = kappa_kantorovich(pf.sample_moments(Y_pool)[1], S_te)
    diag = {"kind": "diag", "avg_corr_syn": float(np.mean(corr)),
            "avg_corr_syn_sd": float(np.std(corr, ddof=1)) if len(corr) > 1 else float("nan"),
            "avg_corr_pooled": pf.average_correlation(Y_pool),
            "logvol_corr_syn": float(np.nanmean(lvc)), "kappa": float(np.median(kap)),
            "kantorovich": float(np.median(kant)), "kappa_pooled": k_pool,
            "kantorovich_pooled": kb_pool,
            "kant_check": float(np.mean(kant_ok)) if kant_ok else float("nan"),
            "fit_seconds": fit_s, "sim_seconds": sim_s, "n_fail": n_fail,
            **garch_counts(gen, model), **model_extras(gen, model)}
    rows.append(diag)
    return rows


def utility_config(cfg: dict, uconf: dict[str, Any]) -> dict[str, Any]:
    return {"n_universes": cfg["n_universes"], "n_assets": cfg["n_assets"],
            "n_replicas": cfg["n_replicas"], "burn_in": cfg["burn_in"],
            "generators": list(GENERATORS), "estimators": list(ESTIMATORS),
            "portfolios": list(PORTFOLIOS), "port_gamma": PORT_GAMMA,
            "ce_gamma_default": CE_GAMMA_DEFAULT, "persist_cap": PERSIST_CAP,
            "root_seed": ROOT_SEED, "train": list(config.TRAIN), "test": list(config.TEST),
            "evaluate_kind": "simple", "logvol_window": LOGVOL_WINDOW, **uconf}


def utility_cells(universes: list[tuple[str, ...]], cfg: dict,
                  gens: tuple[str, ...] = (REAL,) + GENERATORS) -> list[dict]:
    return [{"cell": f"u{u:02d}_{g}", "universe": u, "generator": g, "_tickers": t, "_cfg": cfg}
            for u, t in enumerate(universes) for g in gens]


def run_utility(ctx: common.ExpContext, cfg: dict, universes: list,
                uconf: dict[str, Any]) -> dict[str, float]:
    t0 = time.perf_counter()
    conf = utility_config(cfg, uconf)
    df = common.run_cells(_utility_task, utility_cells(universes, cfg), part="utility", ctx=ctx,
                          root_seed=ROOT_SEED, n_reps=1, config=conf, block_size=1,
                          cost=lambda c: 0.2 if c["generator"] == REAL else
                          (3.0 if c["generator"] in ("CCCGARCHt", "FactorGARCHt") else 1.0))
    reps = df[df["kind"] == "rep"].dropna(axis=1, how="all")
    keys = ["universe", "generator", "est", "port"]
    metric_cols = [c for c in reps.columns if c not in keys + ["kind", "replica", "rep",
                                                                "rep_seconds", "cell"]]
    agg_rep = reps.groupby(keys, sort=False)[metric_cols].mean().reset_index()
    agg_rep["agg"] = "replica"
    agg_rep["n_valid"] = reps.groupby(keys, sort=False)["var_ratio"].count().to_numpy()
    other = df[df["kind"].isin(["pooled", "wavg"])].rename(columns={"kind": "agg"})
    util = pd.concat([agg_rep, other[keys + ["agg"] + metric_cols]], ignore_index=True)
    trtr = (df[df["kind"] == "trtr"].dropna(axis=1, how="all").drop(columns=["kind", "rep"])
            .reset_index(drop=True))
    diag = (df[df["kind"] == "diag"].dropna(axis=1, how="all").drop(columns=["kind", "rep"])
            .reset_index(drop=True))
    reps = reps.reset_index(drop=True)
    wall = time.perf_counter() - t0
    timing = {"utility_wall_s": wall,
              "utility_task_mean_s": float(df.groupby("cell")["rep_seconds"].first().mean())}
    ctx.save_csv(reps.drop(columns=["kind", "rep"]), "e4_utility_reps", config=conf)
    ctx.save_csv(util, "e4_utility", config=conf)
    ctx.save_csv(trtr, "e4_trtr", config=conf)
    ctx.save_csv(diag, "e4_diagnostics", config=conf, timing=timing)
    ctx.write_part_meta("utility", conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Tests: the one call shared by parts "tests" and "placebo"
# --------------------------------------------------------------------------------------------
def mv_windows(X: np.ndarray, L: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping (n, N, L) windows of a normalised panel and their date ranks."""
    W = data.make_windows(X, L)
    return W, (int(offset) + np.arange(W.shape[0], dtype=np.int64) * L)


def mmd_bandwidth(W: np.ndarray) -> float:
    """Median-heuristic bandwidth of the (flattened) windows of one sample.

    Applied to the real sample only, so that every generator of a universe is tested with the
    same kernel (the pooled median heuristic depends on the generator). Deterministic for
    samples of at most 2000 windows (the rng only subsamples larger ones).
    """
    Z = np.asarray(W, dtype=np.float64).reshape(W.shape[0], -1)
    return float(median_heuristic(Z, rng=common.keyed_rng(ROOT_SEED, "mmd_bandwidth")))


def run_absolute_tests(Wx: np.ndarray, ox: np.ndarray, Wy: np.ndarray, oy: np.ndarray, *,
                       cfg: dict, rng: np.random.Generator,
                       bandwidth: float) -> list[dict[str, Any]]:
    """W-critic (HAC) and MMD (wild, fixed bandwidth) under the dependent-window protocol.

    Adds to each row of ``pipeline.absolute_tests``: ``zstat`` (studentised statistic: ``z_hac``
    for the critic; ``(mmd2_v - perm_mean) / perm_std`` for MMD, where ``mmd2_v`` is the
    V-statistic the wild bootstrap refers to and ``perm_mean`` / ``perm_std`` are the mean / sd
    of its wild-bootstrap replicates), ``cal_score`` (score of the placebo calibration: ``z_hac``
    for the critic, ``-p_wild`` for MMD; larger = more evidence against H0), ``se_hac`` and
    ``mean_diff`` (critic), ``mmd2_v``, ``perm_mean``, ``perm_std``, ``bandwidth`` (MMD).
    """
    kw = {k: dict(v) for k, v in cfg["test_kw"].items()}
    kw.setdefault("mmd", {})["bandwidth"] = float(bandwidth)
    rows = pipeline.absolute_tests(Wx, Wy, order_x=ox, order_y=oy, tests=TESTS,
                                   protocol="dependent", rng=rng, test_kw=kw, all_pvalues=True,
                                   keep_extra=True)
    nan = float("nan")
    for r in rows:
        ex = r.pop("extra", {}) or {}
        r.update(se_hac=nan, mean_diff=nan, mmd2_v=nan, perm_mean=nan, perm_std=nan,
                 bandwidth=nan)
        if r["test"] == "critic":
            r.update(se_hac=_num(ex.get("se_hac")), mean_diff=_num(ex.get("mean_diff")))
            r["zstat"] = _num(r["z_hac"])
            r["cal_score"] = r["zstat"]
        elif r["test"] == "mmd":
            v, m, s = _num(ex.get("mmd2_v")), _num(ex.get("perm_mean")), _num(ex.get("perm_std"))
            bws = ex.get("bandwidths") or [nan]
            r.update(mmd2_v=v, perm_mean=m, perm_std=s, bandwidth=_num(bws[0]))
            r["zstat"] = (v - m) / s if np.isfinite(s) and s > 0 else nan
            r["cal_score"] = -_num(r["pvalue"])
    return rows


# --------------------------------------------------------------------------------------------
# Part "tests"
# --------------------------------------------------------------------------------------------
def _tests_task(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Critic and MMD for one (universe, generator, synthetic seed ``rep``)."""
    _quiet_worker_logging()
    tickers, u, gen, cfg = cell["_tickers"], cell["universe"], cell["generator"], cell["_cfg"]
    R, R_te, scale = load_universe(tickers)
    T0, T1, L = R.shape[0], R_te.shape[0], int(cfg["L"])
    Wx, ox = mv_windows(R_te / scale, L, T0)
    t0 = time.perf_counter()
    if gen == REF_GEN:
        # Same size as a synthetic sample: the last T1 TRAIN days, with their own date ranks.
        Y, offset, fit_s = R[T0 - T1:], T0 - T1, 0.0
    else:
        model, fit_s = fit_generator(gen, tickers)
        Y = pipeline.simulate_panel(model, T1, seed=common.keyed_rng(ROOT_SEED, "tests_sim", u,
                                                                     gen, rep),
                                    start_from_end=True, burn_in=int(cfg["burn_in"]))
        offset = T0
    sim_s = time.perf_counter() - t0
    Wy, oy = mv_windows(Y / scale, L, offset)
    # Test stream keyed by (universe, seed) only: same split / init / batches for every generator.
    rows = run_absolute_tests(Wx, ox, Wy, oy, cfg=cfg,
                              rng=common.keyed_rng(ROOT_SEED, "tests_test", u, rep),
                              bandwidth=mmd_bandwidth(Wx))
    extra = {"seed": rep, "avg_corr_y": pf.average_correlation(Y),
             "avg_corr_x": pf.average_correlation(R_te), "fit_seconds": fit_s,
             "sim_seconds": sim_s, "L": L, "y_first_rank": int(oy[0])}
    return [{**r, **extra} for r in rows]


def tests_config(cfg: dict, uconf: dict[str, Any]) -> dict[str, Any]:
    """Hashed configuration of part tests (``test_seeds`` excluded: blocks are per seed)."""
    return {"n_universes": cfg["n_universes"], "n_assets": cfg["n_assets"], "L": cfg["L"],
            "test_kw": cfg["test_kw"], "tests": list(TESTS),
            "protocol": "dependent", "burn_in": cfg["burn_in"], "start_from_end": True,
            "generators": list(GENERATORS) + [REF_GEN], "root_seed": ROOT_SEED,
            "reference": "last T1 TRAIN days, date ranks T0 - T1 + L k",
            "mmd_bandwidth": "median heuristic on the real TEST windows",
            "test_rng": "keyed by (universe, seed)",
            "pipeline": pipeline.pipeline_config(), **uconf}


def tests_cells(universes: list[tuple[str, ...]], cfg: dict,
                gens: tuple[str, ...] = GENERATORS + (REF_GEN,)) -> list[dict]:
    return [{"cell": f"u{u:02d}_{g}", "universe": u, "generator": g, "_tickers": t, "_cfg": cfg}
            for u, t in enumerate(universes) for g in gens]


def run_tests(ctx: common.ExpContext, cfg: dict, universes: list,
              uconf: dict[str, Any]) -> dict[str, float]:
    t0 = time.perf_counter()
    conf = tests_config(cfg, uconf)
    df = common.run_cells(_tests_task, tests_cells(universes, cfg), part="tests", ctx=ctx,
                          root_seed=ROOT_SEED, n_reps=int(cfg["test_seeds"]), config=conf,
                          block_size=1)
    df = df.drop(columns=["rep"])
    timing = {"tests_wall_s": time.perf_counter() - t0,
              "tests_rep_mean_s": float(df.groupby(["cell", "seed"])["rep_seconds"].first().mean())}
    meta = {**conf, "test_seeds": int(cfg["test_seeds"]), "caveats": list(CAVEATS)}
    ctx.save_csv(df, "e4_tests", config=meta)
    write_tests_summary(ctx, timing)
    ctx.write_part_meta("tests", meta, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Part "placebo"
# --------------------------------------------------------------------------------------------
def placebo_reps(u: int, n_placebo: int, n_universes: int) -> int:
    """Replications of universe ``u`` when replication ``r`` uses universe ``r mod n_universes``."""
    return len(range(u, int(n_placebo), int(n_universes)))


def _placebo_task(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """H0 placebo: two independent TEST-length draws of one fitted generator, tests-part call."""
    _quiet_worker_logging()
    tickers, u, gen, cfg = cell["_tickers"], cell["universe"], cell["generator"], cell["_cfg"]
    R, R_te, scale = load_universe(tickers)
    T0, T1, L = R.shape[0], R_te.shape[0], int(cfg["L"])
    model, fit_s = fit_generator(gen, tickers)
    t0 = time.perf_counter()
    sims = [pipeline.simulate_panel(model, T1, seed=common.keyed_rng(ROOT_SEED, "placebo_sim", u,
                                                                     gen, rep, side),
                                    start_from_end=True, burn_in=int(cfg["burn_in"]))
            for side in ("x", "y")]
    sim_s = time.perf_counter() - t0
    Wx, ox = mv_windows(sims[0] / scale, L, T0)
    Wy, oy = mv_windows(sims[1] / scale, L, T0)
    rows = run_absolute_tests(Wx, ox, Wy, oy, cfg=cfg,
                              rng=common.keyed_rng(ROOT_SEED, "placebo_test", u, gen, rep),
                              bandwidth=mmd_bandwidth(Wx))
    extra = {"cycle": rep, "placebo_rep": rep * int(cfg["n_universes"]) + u,
             "avg_corr_x": pf.average_correlation(sims[0]),
             "avg_corr_y": pf.average_correlation(sims[1]), "fit_seconds": fit_s,
             "sim_seconds": sim_s, "L": L}
    return [{**r, **extra} for r in rows]


def placebo_config(cfg: dict, uconf: dict[str, Any]) -> dict[str, Any]:
    """Hashed configuration of part placebo (``n_placebo`` excluded: blocks are per rep)."""
    return {"n_universes": cfg["n_universes"], "n_assets": cfg["n_assets"], "L": cfg["L"],
            "test_kw": cfg["test_kw"], "tests": list(TESTS), "protocol": "dependent",
            "burn_in": cfg["burn_in"], "start_from_end": True, "generators": list(GENERATORS),
            "root_seed": ROOT_SEED, "design": "X*, Y* independent TEST-length draws of the "
            "fitted generator; replication r on universe r mod n_universes",
            "mmd_bandwidth": "median heuristic on the X* windows",
            "pipeline": pipeline.pipeline_config(), **uconf}


def placebo_cells(universes: list[tuple[str, ...]], cfg: dict,
                  gens: tuple[str, ...] = GENERATORS) -> list[dict]:
    nu = len(universes)
    return [{"cell": f"u{u:02d}_{g}", "universe": u, "generator": g, "_tickers": t, "_cfg": cfg}
            for g in gens for u, t in enumerate(universes)
            if placebo_reps(u, cfg["n_placebo"], nu) > 0]


def run_placebo(ctx: common.ExpContext, cfg: dict, universes: list,
                uconf: dict[str, Any]) -> dict[str, float]:
    t0 = time.perf_counter()
    conf = placebo_config(cfg, uconf)
    nu = len(universes)
    df = common.run_cells(_placebo_task, placebo_cells(universes, cfg), part="placebo", ctx=ctx,
                          root_seed=ROOT_SEED,
                          n_reps=lambda c: placebo_reps(c["universe"], cfg["n_placebo"], nu),
                          config=conf, block_size=1)
    df = df.drop(columns=["rep"])
    size = placebo_size(df)
    timing = {"placebo_wall_s": time.perf_counter() - t0,
              "placebo_rep_mean_s": float(df.groupby(["cell", "cycle"])["rep_seconds"]
                                          .first().mean())}
    meta = {**conf, "n_placebo": int(cfg["n_placebo"])}
    ctx.save_csv(df, "e4_placebo", config=meta)
    ctx.save_csv(size, "e4_placebo_size", config=meta, timing=timing)
    write_tests_summary(ctx, timing)
    ctx.write_part_meta("placebo", meta, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Rejection shares, placebo size and size adjustment
# --------------------------------------------------------------------------------------------
def placebo_size(pl: pd.DataFrame) -> pd.DataFrame:
    """Placebo rejection rates (Wilson 95% CIs) per generator, test and p-value type, with the
    mean / sd / se of the studentised statistic and a KS test of p-value uniformity (protocol
    p-value rows only; the KS is approximate for the discrete wild p-value)."""
    d1 = pl[["generator", "test", "pvalue_type", "pvalue"]].rename(columns={"pvalue": "p"})
    d2 = (pl[["generator", "test", "p_perm"]].rename(columns={"p_perm": "p"})
          .assign(pvalue_type=PERM_LABEL))
    tab = common.rejection_table(pd.concat([d1, d2], ignore_index=True),
                                 ["generator", "test", "pvalue_type"], pcol="p", alphas=ALPHAS)
    rows = []
    for (g, t, pt), d in pl.groupby(["generator", "test", "pvalue_type"]):
        z = d["zstat"].to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        p = d["pvalue"].to_numpy(dtype=float)
        p = p[np.isfinite(p)]
        rows.append({"generator": g, "test": t, "pvalue_type": pt,
                     "z_mean": float(z.mean()) if z.size else np.nan,
                     "z_sd": float(z.std(ddof=1)) if z.size > 1 else np.nan,
                     "z_se": float(z.std(ddof=1) / np.sqrt(z.size)) if z.size > 1 else np.nan,
                     "stat_mean": float(d["statistic"].mean()),
                     "ks_uniform_p": (float(stats.kstest(p, "uniform").pvalue) if p.size >= 2
                                      else np.nan),
                     "n_universes": int(d["universe"].nunique())})
    mom = pd.DataFrame(rows)
    out = tab.merge(mom, on=["generator", "test", "pvalue_type"], how="left")
    order = {g: i for i, g in enumerate(GENERATORS)}
    return (out.assign(_o=out["generator"].map(order), _p=out["pvalue_type"].eq(PERM_LABEL))
            .sort_values(["_o", "test", "_p"]).drop(columns=["_o", "_p"]).reset_index(drop=True))


def calibrated_pvalues(df: pd.DataFrame, pl: pd.DataFrame) -> np.ndarray:
    """Size-adjusted p-values ``(1 + #{placebo score >= score}) / (1 + n_placebo)``.

    Placebo scores (``cal_score``) of the same generator and test, pooled over universes; ties
    count against H0 (conservative). NaN for rows without a placebo (the RealTRAIN reference).
    """
    out = np.full(len(df), np.nan)
    pos = np.arange(len(df))
    for (g, t), d in df.groupby(["generator", "test"]):
        ref = pl.loc[(pl["generator"] == g) & (pl["test"] == t), "cal_score"].to_numpy(float)
        ref = np.sort(ref[np.isfinite(ref)])
        if ref.size == 0:
            continue
        s = d["cal_score"].to_numpy(dtype=float)
        n_ge = ref.size - np.searchsorted(ref, s, side="left")
        p = (1.0 + n_ge) / (1.0 + ref.size)
        idx = pos[df.index.get_indexer(d.index)]
        out[idx] = np.where(np.isfinite(s), p, np.nan)
    return out


def _median_over_seeds(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (df.groupby(["universe", "generator", "test", "pvalue_type"])[col]
            .apply(lambda p: pipeline.median_pvalue(p.to_numpy())).reset_index()
            .rename(columns={col: "p"}).assign(aggregation=AGG_MEDIAN))


def tests_summary(df: pd.DataFrame, pl: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rejection shares across universes with Wilson 95% CIs (descriptive, see ``CI_NOTE``).

    Per generator, test and p-value type: the first synthetic seed (one test per universe) and
    twice the median p-value over seeds; nominal p-values (HAC / wild, and the permutation
    p-value, invalid under dependence) and, when the placebo is available, size-adjusted
    p-values (:func:`calibrated_pvalues`) together with the placebo size of the same test
    (``placebo_rej_<a>`` etc.).
    """
    df = df.reset_index(drop=True)
    first = df[df["seed"] == 0]
    cols = ["universe", "generator", "test", "pvalue_type"]
    long = []
    for col, label in (("pvalue", None), ("p_perm", PERM_LABEL)):
        d = first[cols + [col]].rename(columns={col: "p"})
        if label is not None:
            d = d.assign(pvalue_type=label)
        long.append(d.assign(aggregation=AGG_FIRST, calibration=CAL_NOMINAL))
    long.append(_median_over_seeds(df, "pvalue").assign(calibration=CAL_NOMINAL))
    size = None
    if pl is not None and len(pl):
        df = df.assign(p_cal=calibrated_pvalues(df, pl))
        first = df[df["seed"] == 0]
        long.append(first[cols + ["p_cal"]].rename(columns={"p_cal": "p"})
                    .assign(aggregation=AGG_FIRST, calibration=CAL_ADJ))
        long.append(_median_over_seeds(df, "p_cal").assign(calibration=CAL_ADJ))
        size = placebo_size(pl)
    allp = pd.concat(long, ignore_index=True)
    tab = common.rejection_table(allp, ["generator", "test", "pvalue_type", "calibration",
                                        "aggregation"], pcol="p", alphas=ALPHAS)
    stat = (df.groupby(["generator", "test"])
            .agg(stat_mean=("statistic", "mean"), stat_median=("statistic", "median"),
                 zstat_mean=("zstat", "mean")).reset_index())
    tab = tab.merge(stat, on=["generator", "test"], how="left")
    if size is not None:
        ps = size[size["pvalue_type"] != PERM_LABEL]
        keep = {"n_valid": "placebo_n"}
        for a in ALPHAS:
            tag = f"{a:.2f}"
            keep.update({f"rej_{tag}": f"placebo_rej_{tag}", f"lo_{tag}": f"placebo_lo_{tag}",
                         f"hi_{tag}": f"placebo_hi_{tag}"})
        ps = ps[["generator", "test"] + list(keep)].rename(columns=keep)
        tab = tab.merge(ps, on=["generator", "test"], how="left")
    tab["ci_note"] = CI_NOTE
    return tab


def write_tests_summary(ctx: common.ExpContext, timing: dict[str, float] | None = None) -> None:
    """Write ``e4_tests_summary.csv`` from the raw tests (and placebo, if present) CSVs.

    Called by parts tests and placebo, so whichever finishes last (also when run in parallel
    processes) writes the summary with the size adjustment.
    """
    if not ctx.has_csv("e4_tests"):
        return
    pl = ctx.load_csv("e4_placebo") if ctx.has_csv("e4_placebo") else None
    summ = tests_summary(ctx.load_csv("e4_tests"), pl)
    ctx.save_csv(summ, "e4_tests_summary",
                 config={"size_adjusted": pl is not None, "ci_note": CI_NOTE,
                         "n_placebo_rows": 0 if pl is None else int(len(pl))}, timing=timing)


# --------------------------------------------------------------------------------------------
# Part "link"
# --------------------------------------------------------------------------------------------
QUANTITIES: dict[str, str] = {
    "critic_stat": r"W-critic $T$", "critic_z": r"W-critic $z_{\rm HAC}$",
    "critic_p": r"W-critic $p$ (HAC)", "critic_zadj": r"W-critic $z$, placebo-std.",
    "mmd_stat": r"MMD$^2$", "mmd_z": r"MMD wild-std.\ stat.", "mmd_p": r"MMD $p$ (wild)",
    "mmd_zadj": r"MMD wild-std., placebo-std.",
    "corr_gap": r"$|\Delta$ avg.\ corr.$|$",
}
#: Quantities shown in the LaTeX tables (all in the CSV).
TABLE_QUANTITIES: tuple[str, ...] = ("critic_stat", "critic_z", "mmd_stat", "mmd_z", "corr_gap")
#: p-value type behind each p-value quantity (dependent-window protocol).
QUANTITY_PTYPE: dict[str, str] = {"critic_p": "hac", "mmd_p": "wild"}
#: Utility losses shown in the LaTeX tables (sample moments; all combinations in the CSV).
LINK_TABLE_LOSSES: tuple[tuple[str, str, str], ...] = (
    ("var_ratio", "sample", "gmv"), ("var_ratio", "sample", "gmv_lo"),
    ("ce_loss", "sample", "mv1"), ("ce_loss", "sample", "mv5"), ("ce_loss", "sample", "mv20"),
    ("sharpe_loss", "sample", "mv5"),
)
LOSS_LABELS = {"var_ratio": "var.\\ ratio", "ce_loss": "CE loss", "sharpe_loss": "Sharpe loss"}


def placebo_moments(pl: pd.DataFrame | None) -> pd.DataFrame | None:
    """Placebo mean and sd of the studentised statistic per generator and test."""
    if pl is None or not len(pl):
        return None
    return (pl.groupby(["generator", "test"])["zstat"]
            .agg(z_mean="mean", z_sd=lambda z: z.std(ddof=1)).reset_index())


def link_points(util: pd.DataFrame, tests: pd.DataFrame, diag: pd.DataFrame,
                pl_mom: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (universe, generator): test quantities (seed means of the statistics and
    studentised statistics, seed medians of the p-values; placebo-standardised z when the
    placebo moments are given) and utility losses ``<loss>__<est>__<port>__<agg>``."""
    t = tests[tests["generator"] != REF_GEN]
    g = t.groupby(["universe", "generator", "test"])
    q = pd.concat([g["statistic"].mean().rename("stat"), g["zstat"].mean().rename("z"),
                   g["pvalue"].median().rename("p")], axis=1).reset_index()
    values = ["stat", "z", "p"]
    if pl_mom is not None:
        q = q.merge(pl_mom, on=["generator", "test"], how="left")
        q["zadj"] = (q["z"] - q["z_mean"]) / q["z_sd"]
        values.append("zadj")
    wide = q.pivot_table(index=["universe", "generator"], columns="test", values=values)
    wide.columns = [f"{test}_{v}" for v, test in wide.columns]
    wide = wide.reset_index()
    u = util[util["port"] != "ew"]
    lw = u.pivot_table(index=["universe", "generator"], columns=["est", "port", "agg"],
                       values=list(LOSSES))
    lw.columns = [f"{m}__{e}__{p}__{a}" for m, e, p, a in lw.columns]
    lw = lw.reset_index()
    real = diag[diag["generator"] == REAL][["universe", "avg_corr_train"]]
    gd = diag[diag["generator"] != REAL][["universe", "generator", "avg_corr_syn"]]
    gd = gd.merge(real, on="universe")
    gd["corr_gap"] = (gd["avg_corr_syn"] - gd["avg_corr_train"]).abs()
    pts = wide.merge(lw, on=["universe", "generator"], how="inner")
    pts = pts.merge(gd[["universe", "generator", "corr_gap"]], on=["universe", "generator"],
                    how="left")
    order = {g: i for i, g in enumerate(GENERATORS)}
    return (pts.assign(_o=pts["generator"].map(order)).sort_values(["universe", "_o"])
            .drop(columns="_o").reset_index(drop=True))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation on the pairwise-complete observations (NaN if < 3 or constant)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman correlations of every column of ``X`` with every column of ``Y`` (average ranks
    for ties; pairwise-complete observations for columns with missing values)."""
    out = np.full((X.shape[1], Y.shape[1]), np.nan)
    okx, oky = np.isfinite(X).all(axis=0), np.isfinite(Y).all(axis=0)
    if X.shape[0] >= 3 and okx.any() and oky.any():
        rx = stats.rankdata(X[:, okx], axis=0)
        ry = stats.rankdata(Y[:, oky], axis=0)
        rx -= rx.mean(axis=0)
        ry -= ry.mean(axis=0)
        nx, ny = np.linalg.norm(rx, axis=0), np.linalg.norm(ry, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            C = (rx.T @ ry) / np.outer(nx, ny)
        C[nx == 0, :] = np.nan
        C[:, ny == 0] = np.nan
        out[np.ix_(okx, oky)] = C
    for i in np.flatnonzero(~okx):
        out[i] = [_spearman(X[:, i], Y[:, j]) for j in range(Y.shape[1])]
    for j in np.flatnonzero(~oky):
        out[okx, j] = [_spearman(X[:, i], Y[:, j]) for i in np.flatnonzero(okx)]
    return out


def _unit_ranks(A: np.ndarray) -> np.ndarray:
    """Centred average ranks of each column scaled to unit norm (so ``U_x.T @ U_y`` is the
    Spearman matrix); NaN columns if a column has missing values, is constant, or n < 3."""
    A = np.asarray(A, dtype=np.float64)
    out = np.full(A.shape, np.nan)
    ok = np.isfinite(A).all(axis=0)
    if A.shape[0] >= 3 and ok.any():
        r = stats.rankdata(A[:, ok], axis=0)
        r -= r.mean(axis=0)
        nrm = np.linalg.norm(r, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = r / nrm
        r[:, nrm == 0] = np.nan
        out[:, ok] = r
    return out


class _nanmean_quiet:
    """Silence the 'Mean of empty slice' / all-NaN warnings of nanmean / nanquantile."""

    def __enter__(self) -> None:
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc: Any) -> None:
        self._cm.__exit__(*exc)


def _perm_null_mean(UX: list[np.ndarray], UY: list[np.ndarray], n_perm: int,
                    rng: np.random.Generator, chunk: int = 250) -> np.ndarray:
    """Null draws of ``mean_k Spearman(X_k, Y_k)`` when the rows of each ``Y_k`` are permuted
    independently (``UX``/``UY``: unit-rank matrices of the groups); shape ``(n_perm, q, l)``."""
    nq, nl = UX[0].shape[1], UY[0].shape[1]
    tot = np.zeros((n_perm, nq, nl))
    cnt = np.zeros((n_perm, nq, nl))
    for a, b in zip(UX, UY):
        m = a.shape[0]
        for s in range(0, n_perm, chunk):
            e = min(n_perm, s + chunk)
            P = rng.permuted(np.tile(np.arange(m), (e - s, 1)), axis=1)
            r = np.einsum("iq,pil->pql", a, b[P])
            ok = np.isfinite(r)
            tot[s:e] += np.where(ok, r, 0.0)
            cnt[s:e] += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / cnt


def _perm_p(null: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Two-sided Monte Carlo p-value ``(1 + #{|null| >= |obs|}) / (1 + B)`` per entry."""
    ok = np.isfinite(null)
    ge = (np.abs(np.where(ok, null, 0.0)) >= np.abs(obs) - 1e-12) & ok
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = (1.0 + ge.sum(axis=0)) / (1.0 + n)
    return np.where(np.isfinite(obs) & (n > 0), p, np.nan)


def link_statistics(pts: pd.DataFrame, qcols: list[str], lcols: list[str], n_boot: int,
                    n_perm: int, rng_boot: np.random.Generator,
                    rng_perm: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled, within-universe, within-generator and generator-level Spearman correlations of
    every (quantity, loss) pair.

    * *Pooled*: over all (universe, generator) points; universe-bootstrap 95% percentile CI
      (universes drawn with replacement, all generators of a drawn universe kept together; the
      same draws serve every view and pair).
    * *Within-universe* ``rho_u``: Spearman across the generators of each universe, averaged;
      bootstrap CI and a two-sided permutation p-value from permuting the generator labels of
      the losses within each universe (null: no agreement between the orderings of these
      generators; conditional on the generator set).
    * *Within-generator* ``rho_g``: Spearman across universes for each generator, averaged over
      generators; bootstrap CI and a permutation p-value from permuting universes within each
      generator. Invariant to generator-level offsets of either variable.
    * *Generator-level*: Spearman of the generator means over universes (n = number of
      generators) with an exact two-sided permutation p-value over all orderings.

    Returns the summary table and the per-generator correlations.
    """
    X, Y = pts[qcols].to_numpy(dtype=float), pts[lcols].to_numpy(dtype=float)
    uni, gen = pts["universe"].to_numpy(), pts["generator"].to_numpy()
    us = np.unique(uni)
    gs = [g for g in GENERATORS if g in set(gen)]
    idx_u = [np.flatnonzero(uni == u) for u in us]
    idx_g = [np.flatnonzero(gen == g) for g in gs]
    pos_g = [{int(uni[i]): int(i) for i in ii} for ii in idx_g]
    nq, nl = X.shape[1], Y.shape[1]

    pooled = spearman_matrix(X, Y)
    UXu, UYu = [_unit_ranks(X[i]) for i in idx_u], [_unit_ranks(Y[i]) for i in idx_u]
    rho_u = np.stack([a.T @ b for a, b in zip(UXu, UYu)])
    UXg, UYg = [_unit_ranks(X[i]) for i in idx_g], [_unit_ranks(Y[i]) for i in idx_g]
    rho_g = np.stack([a.T @ b for a, b in zip(UXg, UYg)])
    with _nanmean_quiet():
        within_u = np.nanmean(rho_u, axis=0)
        within_g = np.nanmean(rho_g, axis=0)
        Xm = np.stack([np.nanmean(X[i], axis=0) for i in idx_g])
        Ym = np.stack([np.nanmean(Y[i], axis=0) for i in idx_g])
    UXm, UYm = _unit_ranks(Xm), _unit_ranks(Ym)
    gl = UXm.T @ UYm
    k = len(gs)
    perms = (list(itertools.permutations(range(k))) if k <= 7
             else [rng_perm.permutation(k) for _ in range(n_perm)])
    null_gl = np.stack([UXm.T @ UYm[list(p)] for p in perms])
    p_gl = np.where(np.isfinite(gl),
                    np.mean(np.abs(null_gl) >= np.abs(gl) - 1e-12, axis=0), np.nan)

    bp = np.empty((n_boot, nq, nl))
    bu = np.empty((n_boot, nq, nl))
    bg = np.empty((n_boot, nq, nl))
    for b in range(n_boot):
        draw = rng_boot.integers(0, us.size, us.size)
        ii = np.concatenate([idx_u[d] for d in draw])
        bp[b] = spearman_matrix(X[ii], Y[ii])
        drawn = [int(u) for u in us[draw]]
        rg = []
        for pg in pos_g:
            sel = np.array([pg[u] for u in drawn if u in pg], dtype=int)
            rg.append(spearman_matrix(X[sel], Y[sel]) if sel.size >= 3
                      else np.full((nq, nl), np.nan))
        with _nanmean_quiet():
            bu[b] = np.nanmean(rho_u[draw], axis=0)
            bg[b] = np.nanmean(np.stack(rg), axis=0)
    with _nanmean_quiet():
        q_p = np.nanquantile(bp, [0.025, 0.975], axis=0)
        q_u = np.nanquantile(bu, [0.025, 0.975], axis=0)
        q_g = np.nanquantile(bg, [0.025, 0.975], axis=0)
    p_u = _perm_p(_perm_null_mean(UXu, UYu, n_perm, rng_perm), within_u)
    p_g = _perm_p(_perm_null_mean(UXg, UYg, n_perm, rng_perm), within_g)

    rows, by_gen = [], []
    for i, q in enumerate(qcols):
        n_ok = np.isfinite(X[:, i])
        for j, lcol in enumerate(lcols):
            m, est, port, agg = lcol.split("__")
            key = {"quantity": q, "loss": m, "est": est, "port": port, "agg": agg}
            rows.append({**key,
                         "rho_pooled": pooled[i, j], "lo_pooled": q_p[0, i, j],
                         "hi_pooled": q_p[1, i, j],
                         "rho_within_u": within_u[i, j], "lo_within_u": q_u[0, i, j],
                         "hi_within_u": q_u[1, i, j], "p_perm_within_u": p_u[i, j],
                         "rho_within_g": within_g[i, j], "lo_within_g": q_g[0, i, j],
                         "hi_within_g": q_g[1, i, j], "p_perm_within_g": p_g[i, j],
                         "rho_gen_level": gl[i, j], "p_exact_gen_level": p_gl[i, j],
                         "n_points": int(np.sum(n_ok & np.isfinite(Y[:, j]))),
                         "n_universes": int(us.size), "n_generators": k,
                         "n_gen_perms": len(perms), "n_boot": int(n_boot),
                         "n_perm": int(n_perm)})
            for gi, g in enumerate(gs):
                by_gen.append({**key, "generator": g, "rho": rho_g[gi, i, j],
                               "n_universes": int(idx_g[gi].size),
                               "mean_quantity": Xm[gi, i], "mean_loss": Ym[gi, j]})
    return pd.DataFrame(rows), pd.DataFrame(by_gen)


def run_link(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    t0 = time.perf_counter()
    util, tests, diag = (ctx.load_csv("e4_utility"), ctx.load_csv("e4_tests"),
                         ctx.load_csv("e4_diagnostics"))
    pl = ctx.load_csv("e4_placebo") if ctx.has_csv("e4_placebo") else None
    if pl is None:
        logger.warning("e4_placebo.csv missing: link without the placebo-standardised z "
                       "(run part placebo first for the full link)")
    pts = link_points(util, tests, diag, placebo_moments(pl))
    loss_cols = [c for c in pts.columns if c.split("__")[0] in LOSSES]
    qcols = [q for q in QUANTITIES if q in pts.columns]
    link, by_gen = link_statistics(pts, qcols, loss_cols, int(cfg["n_boot_link"]),
                                   int(cfg["n_perm_link"]),
                                   common.keyed_rng(ROOT_SEED, "link_bootstrap"),
                                   common.keyed_rng(ROOT_SEED, "link_perm"))
    link.insert(1, "pvalue_type", link["quantity"].map(QUANTITY_PTYPE).fillna(""))
    conf = {"n_boot_link": cfg["n_boot_link"], "n_perm_link": cfg["n_perm_link"],
            "quantities": qcols, "bootstrap": "universe-level, percentile 95%",
            "test_aggregation": "seed-mean statistic and studentised statistic, seed-median "
            "p-value", "utility_aggregations": list(AGGS), "placebo_standardised": pl is not None,
            "views": ["pooled", "within-universe (perm. of generator labels)",
                      "within-generator (perm. of universes)",
                      "generator-level (exact perm., n = number of generators)"],
            "caveats": list(CAVEATS), "root_seed": ROOT_SEED}
    timing = {"link_wall_s": time.perf_counter() - t0}
    ctx.save_csv(pts, "e4_link_points", config=conf)
    ctx.save_csv(by_gen, "e4_link_by_generator", config=conf)
    ctx.save_csv(link, "e4_link", config=conf, timing=timing)
    ctx.write_part_meta("link", conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Tables and figures
# --------------------------------------------------------------------------------------------
def _ann_pct(x: pd.Series | float) -> Any:
    """Daily return quantity -> annualised percent (x 252 x 100)."""
    return x * TD * 100.0


def save_note(ctx: common.ExpContext, name: str, text: str) -> None:
    """Write a table note ``<tab_dir>/<name>.tex`` (plain LaTeX text, generated from the data)."""
    path = ctx.tab_dir / f"{name}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(text.split()) + "\n", encoding="utf-8")


def build_utility_outputs(ctx: common.ExpContext) -> None:
    if not (ctx.has_csv("e4_utility") and ctx.has_csv("e4_trtr") and ctx.has_csv("e4_diagnostics")):
        logger.warning("utility CSVs missing: skipping the utility tables")
        return
    util, trtr, diag = (ctx.load_csv("e4_utility"), ctx.load_csv("e4_trtr"),
                        ctx.load_csv("e4_diagnostics"))
    gens = [g for g in GENERATORS if g in set(util["generator"])]
    ports = [p for p in PORTFOLIOS if p != "ew"]
    metric_hdr = {"var_ratio": "Var.\\ ratio", "sharpe_loss": "Sharpe loss",
                  "ce_loss": "CE loss (\\% p.a.)"}

    def cell_value(d: pd.DataFrame, m: str) -> float:
        v = float(d[m].mean()) if len(d) else float("nan")
        return _ann_pct(v) if m == "ce_loss" else v

    # e4_utility.tex: replica-wise, mean over universes.
    rep = util[util["agg"] == "replica"]
    rows, idx = [], []
    for g in gens:
        for p in ports:
            row = {}
            for m in LOSSES:
                for e in ESTIMATORS:
                    d = rep[(rep["generator"] == g) & (rep["port"] == p) & (rep["est"] == e)]
                    row[(metric_hdr[m], EST_LABELS[e])] = cell_value(d, m)
            rows.append(row)
            idx.append((GEN_LABELS[g], PORT_LABELS[p]))
    tab = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["Generator",
                                                                         "Portfolio"]))
    tab.columns = pd.MultiIndex.from_tuples(tab.columns)
    ctx.save_table(tab, "e4_utility", escape=False,
                   digits={c: (2 if c[0].startswith("CE") else 3) for c in tab.columns})

    # e4_utility_agg.tex: replica-wise vs pooled moments vs averaged weights (sample moments).
    agg_lbl = {"replica": "Replica", "pooled": "Pooled", "wavg": "Avg.\\ $w$"}
    rows, idx = [], []
    for g in gens:
        for p in ports:
            row = {}
            for m in ("var_ratio", "ce_loss"):
                for a in AGGS:
                    d = util[(util["generator"] == g) & (util["port"] == p)
                             & (util["est"] == "sample") & (util["agg"] == a)]
                    row[(metric_hdr[m], agg_lbl[a])] = cell_value(d, m)
            rows.append(row)
            idx.append((GEN_LABELS[g], PORT_LABELS[p]))
    tab = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["Generator",
                                                                         "Portfolio"]))
    tab.columns = pd.MultiIndex.from_tuples(tab.columns)
    ctx.save_table(tab, "e4_utility_agg", escape=False,
                   digits={c: (2 if c[0].startswith("CE") else 3) for c in tab.columns})

    # e4_trtr.tex: real-trained out-of-sample performance (mean over universes).
    rows, idx = [], []
    for p in PORTFOLIOS:
        row = {}
        for e in ESTIMATORS:
            d = trtr[(trtr["port"] == p) & (trtr["est"] == e)]
            row[("Vol.\\ (\\% p.a.)", EST_LABELS[e])] = float((d["vol"] * np.sqrt(TD) * 100).mean())
            row[("Sharpe", EST_LABELS[e])] = float(d["sharpe"].mean())
            row[("CE (\\% p.a.)", EST_LABELS[e])] = float(_ann_pct(d["ce"]).mean())
        rows.append(row)
        idx.append(PORT_LABELS[p])
    tab = pd.DataFrame(rows, index=pd.Index(idx, name="Portfolio"))
    tab.columns = pd.MultiIndex.from_tuples(tab.columns)
    tab = tab[[c for m in ("Vol.\\ (\\% p.a.)", "Sharpe", "CE (\\% p.a.)")
               for c in tab.columns if c[0] == m]]
    ctx.save_table(tab, "e4_trtr", escape=False, digits=2)

    # e4_correlation.tex ("sd (universes)": across universes; the at-cap count is the
    # near-unit-root diagnostic, the persistence cap being 0.995).
    real = diag[diag["generator"] == REAL]
    rows, idx = [], []
    cap_col = f"At cap ({PERSIST_CAP:g})"
    sd_col = "sd (universes)"

    def counts(d: pd.DataFrame, col: str) -> str:
        if col not in d.columns or d[col].isna().all():
            return "--"
        return f"{int(d[col].sum())}/{int(d['n_comp'].sum())}"

    rows.append({"Avg.\\ corr.": real["avg_corr_train"].mean(),
                 sd_col: real["avg_corr_train"].std(ddof=1),
                 "Log-vol corr.": real["logvol_corr_train"].mean(),
                 "$\\kappa$": real["kappa"].median(), "Kant.\\ bound": real["kantorovich"].median(),
                 cap_col: "--"})
    idx.append("Real TRAIN")
    rows.append({"Avg.\\ corr.": real["avg_corr_test"].mean(),
                 sd_col: real["avg_corr_test"].std(ddof=1),
                 "Log-vol corr.": real["logvol_corr_test"].mean(), "$\\kappa$": np.nan,
                 "Kant.\\ bound": np.nan, cap_col: "--"})
    idx.append("Real TEST")
    for g in gens:
        d = diag[diag["generator"] == g]
        rows.append({"Avg.\\ corr.": d["avg_corr_syn"].mean(), sd_col: d["avg_corr_syn"].std(ddof=1),
                     "Log-vol corr.": d["logvol_corr_syn"].mean(),
                     "$\\kappa$": d["kappa"].median(), "Kant.\\ bound": d["kantorovich"].median(),
                     cap_col: counts(d, "n_at_cap")})
        idx.append(GEN_LABELS[g])
    tab = pd.DataFrame(rows, index=pd.Index(idx, name=""))
    ctx.save_table(tab, "e4_correlation", escape=False,
                   digits={"Avg.\\ corr.": 3, sd_col: 3, "Log-vol corr.": 3, "$\\kappa$": 2,
                           "Kant.\\ bound": 3})


def _fmt_num(v: float, digits: int) -> str:
    if not np.isfinite(v):
        return "--"
    return f"{v:.{digits}f}".replace("-", "$-$")


def build_tests_outputs(ctx: common.ExpContext) -> None:
    if not ctx.has_csv("e4_tests"):
        logger.warning("e4_tests.csv missing: skipping the tests table")
        return
    df = ctx.load_csv("e4_tests")
    pl = ctx.load_csv("e4_placebo") if ctx.has_csv("e4_placebo") else None
    s = tests_summary(df, pl)
    n_u = int(df["universe"].nunique())
    n_days_ref = int(df["n_y"].iloc[0]) * int(df["L"].iloc[0])

    def rate(g: str, t: str, calib: str, agg: str) -> str:
        r = s[(s["generator"] == g) & (s["test"] == t) & (s["pvalue_type"] == TEST_PTYPE[t])
              & (s["calibration"] == calib) & (s["aggregation"] == agg)]
        if not len(r) or not r["n_valid"].iloc[0]:
            return "--"
        r = r.iloc[0]
        return common.fmt_rate_ci(r["rej_0.05"], r["lo_0.05"], r["hi_0.05"], 2)

    rows, idx = [], []
    for g in list(GENERATORS) + [REF_GEN]:
        if g not in set(s["generator"]):
            continue
        for t in TESTS:
            d = s[(s["generator"] == g) & (s["test"] == t)]
            if not len(d):
                continue
            size = "--"
            if "placebo_rej_0.05" in d.columns and np.isfinite(d["placebo_rej_0.05"].iloc[0]):
                size = common.fmt_rate_ci(d["placebo_rej_0.05"].iloc[0],
                                          d["placebo_lo_0.05"].iloc[0],
                                          d["placebo_hi_0.05"].iloc[0], 2)
            rows.append({
                "Rej.\\ (first seed)": rate(g, t, CAL_NOMINAL, AGG_FIRST),
                "Rej.\\ ($2\\times$med.)": rate(g, t, CAL_NOMINAL, AGG_MEDIAN),
                "Placebo size": size,
                "Size-adj.\\ (first seed)": rate(g, t, CAL_ADJ, AGG_FIRST),
                "Size-adj.\\ ($2\\times$med.)": rate(g, t, CAL_ADJ, AGG_MEDIAN),
                "Mean stat.": _fmt_num(float(d["stat_mean"].iloc[0]), 3 if t == "critic" else 4),
            })
            idx.append((GEN_LABELS[g], TEST_LABELS[t]))
    tab = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["Generator", "Test"]))
    ctx.save_table(tab, "e4_tests", escape=False)
    n_pl = int(pl["placebo_rep"].nunique()) if pl is not None and len(pl) else 0
    save_note(ctx, "e4_tests_note", f"""
        Rejection shares at the 5\\% level across the {n_u} universes (W-critic: Driscoll-Kraay
        HAC p-value, {pipeline.TIME_BLOCKS}-block time split; MMD: wild-bootstrap p-value with a
        fixed bandwidth per universe), for the first synthetic
        seed and for twice the median p-value over {int(df['seed'].nunique())} seeds. Bracketed
        intervals are Wilson 95\\% intervals that assume independent universes; universes share
        every TEST date and overlap in tickers, so the effective number of independent
        comparisons is smaller and the shares are descriptive. Placebo size: rejection rate of
        the same test when both samples are independent draws from the fitted generator (H0
        true; {n_pl} replications per generator cycling over universes; Wilson 95\\% interval).
        Size-adjusted: rejection when the p-value is below the 5\\% quantile of the placebo
        p-values of the same generator and test (Monte Carlo p-value against the placebo,
        pooled over universes). The reference row compares real TEST with the last
        {n_days_ref} TRAIN days (same number of windows); it has no placebo. Mean stat.: mean
        over universes and
        seeds of the critic statistic $T$ or of MMD$^2_u$.""")


def build_placebo_outputs(ctx: common.ExpContext) -> None:
    if not ctx.has_csv("e4_placebo"):
        logger.warning("e4_placebo.csv missing: skipping the placebo table and figure")
        return
    pl = ctx.load_csv("e4_placebo")
    size = placebo_size(pl)
    ptype_lbl = {"hac": "HAC", "wild": "wild", PERM_LABEL: "permutation"}
    rows, idx = [], []
    for _, r in size.iterrows():
        row = {f"$\\alpha={a:.2f}$": common.fmt_rate_ci(r[f"rej_{a:.2f}"], r[f"lo_{a:.2f}"],
                                                         r[f"hi_{a:.2f}"], 3) for a in ALPHAS}
        proto = r["pvalue_type"] != PERM_LABEL
        row["Mean $z$"] = _fmt_num(r["z_mean"], 2) if proto else "--"
        row["SD $z$"] = _fmt_num(r["z_sd"], 2) if proto else "--"
        row["KS $p$"] = _fmt_num(r["ks_uniform_p"], 3) if proto else "--"
        rows.append(row)
        idx.append((GEN_LABELS[r["generator"]], TEST_LABELS[r["test"]].split(" (")[0],
                    ptype_lbl.get(r["pvalue_type"], r["pvalue_type"])))
    tab = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        idx, names=["Generator", "Test", "$p$-value"]))
    ctx.save_table(tab, "e4_placebo", escape=False)
    n_pl = int(pl["placebo_rep"].nunique())
    n_u = int(pl["universe"].nunique())
    save_note(ctx, "e4_placebo_note", f"""
        Placebo size of the E4 tests: both samples are independent TEST-length draws (start from
        the end of TRAIN) from the generator fitted on one universe, windowed, normalised and
        ordered exactly as the real TEST sample, and tested with the E4 call. {n_pl}
        replications per generator, cycling over {n_u} universes; rejection rates with Wilson
        95\\% intervals. $z$: the studentised statistic ($z_{{\\rm HAC}}$ for the W-critic; the
        V-statistic standardised by the mean and sd of its wild-bootstrap replicates for MMD);
        under a correctly sized test its mean is about 0 and its sd about 1. KS $p$: KS test of
        uniformity of the HAC / wild p-values (approximate for the discrete wild p-value). The
        permutation p-values assume independent windows and are shown for comparison only.""")
    plot_placebo_size(ctx, pl)


def plot_placebo_size(ctx: common.ExpContext, pl: pd.DataFrame) -> None:
    """Empirical CDF of the placebo p-values (HAC / wild) against the nominal level."""
    import matplotlib.pyplot as plt

    plotting.setup()
    fig, axes = plt.subplots(1, 2, figsize=plotting.figsize("two", 0.38), sharey=True)
    grid = np.linspace(0.0, 0.20, 201)
    for ax, t in zip(axes, TESTS):
        for g in GENERATORS:
            p = pl.loc[(pl["generator"] == g) & (pl["test"] == t), "pvalue"].to_numpy(float)
            p = p[np.isfinite(p)]
            if not p.size:
                continue
            ecdf = np.searchsorted(np.sort(p), grid, side="right") / p.size
            ax.step(grid, ecdf, where="post", color=gen_color(g), lw=1.0, label=GEN_LABELS[g])
        ax.plot(grid, grid, color=plotting.OKABE_ITO["grey"], lw=0.7, ls="--", zorder=0)
        ax.axvline(0.05, color=plotting.OKABE_ITO["grey"], lw=0.5, ls=":", zorder=0)
        ax.set_xlim(0.0, 0.20)
        ax.set_xlabel(r"nominal level $\alpha$")
        ax.set_title(TEST_LABELS[t], fontsize=8)
    axes[0].set_ylabel(r"placebo rejection rate")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside upper center", ncol=len(handles), fontsize=7)
    ctx.savefig(fig, "e4_placebo_size")


def _fmt_rho(r: float, lo: float, hi: float) -> str:
    if not np.isfinite(r):
        return "--"
    f = lambda v: f"{v:.2f}".replace("-", "$-$")  # noqa: E731
    return f"{f(r)} [{f(lo)}, {f(hi)}]"


def _fmt_p(p: float) -> str:
    return "--" if not np.isfinite(p) else f"{p:.3f}"


def build_link_outputs(ctx: common.ExpContext) -> None:
    if not (ctx.has_csv("e4_link") and ctx.has_csv("e4_link_points")):
        logger.warning("link CSVs missing: skipping the link tables and figure")
        return
    link, pts = ctx.load_csv("e4_link"), ctx.load_csv("e4_link_points")
    for agg, name in (("replica", "e4_link"), ("pooled", "e4_link_pooled")):
        rows, idx = [], []
        for m, e, p in LINK_TABLE_LOSSES:
            for q in TABLE_QUANTITIES:
                d = link[(link["quantity"] == q) & (link["loss"] == m) & (link["est"] == e)
                         & (link["port"] == p) & (link["agg"] == agg)]
                if not len(d):
                    continue
                r = d.iloc[0]
                rows.append({
                    "Pooled $\\rho$": _fmt_rho(r["rho_pooled"], r["lo_pooled"], r["hi_pooled"]),
                    "Within-univ.\\ $\\bar\\rho$": _fmt_rho(r["rho_within_u"], r["lo_within_u"],
                                                            r["hi_within_u"]),
                    "$p_u$ (perm.)": _fmt_p(r["p_perm_within_u"]),
                    "Within-gen.\\ $\\bar\\rho$": _fmt_rho(r["rho_within_g"], r["lo_within_g"],
                                                           r["hi_within_g"]),
                    "$p_g$ (perm.)": _fmt_p(r["p_perm_within_g"]),
                    "Gen.-level $\\rho$": _fmt_num(r["rho_gen_level"], 2),
                    "$p$ (exact)": _fmt_p(r["p_exact_gen_level"]),
                })
                idx.append((f"{PORT_LABELS[p]} {LOSS_LABELS[m]}", QUANTITIES[q]))
        if rows:
            tab = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
                idx, names=["Utility loss", "Test quantity"]))
            ctx.save_table(tab, name, escape=False)
    n_u = int(link["n_universes"].max()) if len(link) else 0
    n_g = int(link["n_generators"].max()) if len(link) else 0
    n_perms = int(link["n_gen_perms"].max()) if len(link) else 0
    save_note(ctx, "e4_link_note", f"""
        Spearman correlations between test quantities and utility losses (``e4\\_link'':
        replica-wise portfolios, the DESIGN measure, which includes the estimation-noise term;
        ``e4\\_link\\_pooled'': portfolios from the moments of the pooled replicas, which isolate
        the covariance misspecification). Test quantities are seed means; the studentised
        statistics replace the p-values, which lose resolution for strong rejections. Pooled:
        over the {n_u * n_g} (universe, generator) points. Within-universe: across the {n_g}
        generators of each universe, averaged; the permutation $p$ permutes generator labels
        within universes and is conditional on these generators. Within-generator: across the
        {n_u} universes of each generator, averaged; the permutation $p$ permutes universes
        within generators; it is insensitive to generator-level offsets of a statistic (such as
        the placebo bias of the critic). Brackets: universe-bootstrap 95\\% percentile
        intervals (universes share dates, so they are optimistic). Generator-level: Spearman
        of the {n_g} generator means, exact two-sided permutation $p$ over the {n_perms}
        orderings; this is the only view whose sampling unit is the generator, so the evidence
        across generators rests on n = {n_g}, and conclusions are conditional on these
        generators.""")
    plot_utility_vs_distance(ctx, pts, link)


def plot_utility_vs_distance(ctx: common.ExpContext, pts: pd.DataFrame, link: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plotting.setup()
    ys = (("var_ratio__sample__gmv__replica", "GMV variance ratio (TSTR / TRTR)", 1.0),
          ("ce_loss__sample__mv5__replica", r"MV $\gamma=5$ CE loss (% p.a.)", TD * 100.0))
    xs = (("critic_stat", r"W-critic statistic $T$"), ("mmd_stat", r"MMD$^2_u$"))
    fig, axes = plt.subplots(2, 2, figsize=plotting.figsize("two", 0.62), sharex="col")
    for i, (ycol, ylab, yscale) in enumerate(ys):
        for j, (xcol, xlab) in enumerate(xs):
            ax = axes[i, j]
            if ycol not in pts.columns or xcol not in pts.columns:
                ax.set_visible(False)
                continue
            for g in GENERATORS:
                d = pts[pts["generator"] == g]
                if not len(d):
                    continue
                ax.scatter(d[xcol], d[ycol] * yscale, s=12, color=gen_color(g),
                           marker=gen_marker(g), label=GEN_LABELS[g], alpha=0.8, linewidths=0.4)
            m, e, p, a = ycol.split("__")
            r = link[(link["quantity"] == xcol) & (link["loss"] == m) & (link["est"] == e)
                     & (link["port"] == p) & (link["agg"] == a)]
            if len(r):
                r = r.iloc[0]
                ax.text(0.03, 0.95, rf"pooled $\rho$ = {r['rho_pooled']:.2f} "
                        rf"[{r['lo_pooled']:.2f}, {r['hi_pooled']:.2f}]" "\n"
                        rf"within-gen. $\bar\rho$ = {r['rho_within_g']:.2f} "
                        rf"[{r['lo_within_g']:.2f}, {r['hi_within_g']:.2f}]",
                        transform=ax.transAxes, va="top", ha="left", fontsize=6.5)
            if i == 1:
                ax.set_xlabel(xlab)
            if j == 0:
                ax.set_ylabel(ylab)
    if ys[0][0] in pts.columns:
        for ax in axes[0]:
            ax.axhline(1.0, color=plotting.OKABE_ITO["grey"], lw=0.6, ls="--", zorder=0)
    for ax in axes[1]:
        ax.axhline(0.0, color=plotting.OKABE_ITO["grey"], lw=0.6, ls="--", zorder=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside upper center", ncol=len(handles), fontsize=7)
    ctx.savefig(fig, "e4_utility_vs_distance")


def build_outputs(ctx: common.ExpContext) -> None:
    parts = set(PARTS) if ctx.plots_only else set(ctx.parts)
    if "utility" in parts:
        build_utility_outputs(ctx)
    if "tests" in parts or "placebo" in parts:
        build_tests_outputs(ctx)
    if "placebo" in parts:
        build_placebo_outputs(ctx)
    if "link" in parts:
        build_link_outputs(ctx)


# --------------------------------------------------------------------------------------------
# Timing probes (quick mode): full-configuration cells, fresh, timing only
# --------------------------------------------------------------------------------------------
PROBE_TEST_GENS = ("MVGaussian", "CCCGARCHt", "FactorGARCHt")
PROBE_PLACEBO_GENS = ("FactorGARCHt", "MVBootstrap")


def _probe_task(cell: dict, rep: int, rng: np.random.Generator) -> dict[str, Any]:
    t0 = time.perf_counter()
    if cell["kind"] == "utility":
        _utility_task(cell, 0, rng)
    elif cell["kind"] == "tests":
        _tests_task(cell, 0, rng)
    else:
        _placebo_task(cell, 0, rng)
    return {"probe_seconds": time.perf_counter() - t0}


def run_probes(ctx: common.ExpContext, universes: list, parts: list[str]) -> pd.DataFrame:
    F = FULL
    t = universes[0]
    cells: list[dict] = []
    if "utility" in parts:
        cells += [{**c, "cell": f"util_{c['generator']}", "kind": "utility"}
                  for c in utility_cells([t], F)]
    if "tests" in parts:
        cells += [{**c, "cell": f"tests_{c['generator']}", "kind": "tests"}
                  for c in tests_cells([t], F, gens=PROBE_TEST_GENS)]
    if "placebo" in parts:
        cells += [{**c, "cell": f"placebo_{c['generator']}", "kind": "placebo"}
                  for c in placebo_cells([t], {**F, "n_universes": 1},
                                         gens=PROBE_PLACEBO_GENS)]
    if not cells:
        return pd.DataFrame()
    return common.run_cells(_probe_task, cells, part="probe", ctx=ctx, root_seed=ROOT_SEED,
                            n_reps=1, config={"probe": True, "full": F}, block_size=1,
                            cost=lambda c: 1.0 if c["kind"] == "utility" else 10.0, fresh=True)


def report_estimates(probe: pd.DataFrame, parts: list[str], link_quick_s: float | None,
                     quick_cfg: dict, workers: int = 12) -> dict[str, float]:
    """Print measured per-replication times and estimated full-run wall times per part."""
    F = FULL
    est: dict[str, float] = {}
    if probe.empty:
        return est
    t = {r["cell"]: float(r["probe_seconds"]) for _, r in probe.iterrows()}
    nu = F["n_universes"]
    if "utility" in parts:
        tasks = []
        for g in (REAL,) + GENERATORS:
            s = t[f"util_{g}"]
            note = ("TRTR portfolios + diagnostics" if g == REAL else
                    f"fit + B={F['n_replicas']} replicas x 2 estimators x 6 portfolios + pooled/wavg")
            common.print_rep_timing(f"utility {g}", s, note=note)
            tasks += [s] * nu
        est["utility"] = common.print_wall_estimate("utility", tasks, workers=workers)
    for part, gens_probed, gens_all, n_each, what in (
            ("tests", PROBE_TEST_GENS, GENERATORS + (REF_GEN,), nu * F["test_seeds"],
             "critic 1000 steps + MMD wild/perm, one seed"),
            ("placebo", PROBE_PLACEBO_GENS, GENERATORS, F["n_placebo"],
             "2 simulated panels + critic 1000 steps + MMD wild/perm")):
        if part not in parts:
            continue
        probed = [t[f"{part}_{g}"] for g in gens_probed]
        mean_s = float(np.mean(probed))
        tasks = []
        for g in gens_all:
            s = t.get(f"{part}_{g}", mean_s)
            measured = f"{part}_{g}" in t
            common.print_rep_timing(f"{part} {g}", s, note=(what if measured else
                                                             "not probed: mean of probed generators"))
            tasks += [s] * n_each
        est[part] = common.print_wall_estimate(part, tasks, workers=workers)
    if "link" in parts and link_quick_s is not None:
        n_pts_ratio = F["n_universes"] / quick_cfg["n_universes"]
        draws_ratio = ((F["n_boot_link"] + F["n_perm_link"])
                       / (quick_cfg["n_boot_link"] + quick_cfg["n_perm_link"]))
        s = link_quick_s * draws_ratio * n_pts_ratio
        print(f"TIMING link (quick run): {link_quick_s:.1f} s for "
              f"{quick_cfg['n_universes']} universes, {quick_cfg['n_boot_link']} bootstrap draws "
              f"and {quick_cfg['n_perm_link']} permutations", flush=True)
        est["link"] = common.print_wall_estimate("link (serial, main process)", [s],
                                                 workers=1, overhead_s=5.0)
    return est


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------
def _add_args(parser: Any) -> None:
    parser.add_argument("--no-probe", action="store_true",
                        help="quick mode: skip the full-configuration timing probes")
    parser.add_argument("--test-seeds", type=int, default=None,
                        help="override the number of synthetic seeds of the tests part "
                             "(finished seeds are reused)")
    parser.add_argument("--placebo-reps", type=int, default=None,
                        help="override the number of placebo replications per generator "
                             "(finished replications are reused)")


def main(argv: list[str] | None = None) -> None:
    ctx = common.parse_args(EXP, PARTS, description=__doc__, add_arguments=_add_args, argv=argv)
    cfg = dict(QUICK if ctx.quick else FULL)
    if ctx.args.test_seeds is not None:
        cfg["test_seeds"] = int(ctx.args.test_seeds)
    if ctx.args.placebo_reps is not None:
        cfg["n_placebo"] = int(ctx.args.placebo_reps)
    sw = common.Stopwatch()
    timing: dict[str, float] = {}
    universes, universe_size = draw_universes(cfg["n_universes"], cfg["n_assets"])
    uconf = universe_config(universes, universe_size)
    ctx.write_meta({"quick": ctx.quick, "parts": ctx.parts, "plots_only": ctx.plots_only,
                    "cfg": cfg, "root_seed": ROOT_SEED, "caveats": list(CAVEATS), **uconf})
    if not ctx.plots_only:
        uni_df = pd.DataFrame({"universe": range(len(universes)),
                               "tickers": [" ".join(t) for t in universes]})
        ctx.save_csv(uni_df, "universes", config={"n_universes": cfg["n_universes"],
                                                  "n_assets": cfg["n_assets"],
                                                  "root_seed": ROOT_SEED, **uconf})
        if "utility" in ctx.parts:
            timing.update(run_utility(ctx, cfg, universes, uconf))
            sw.lap("utility")
        if "tests" in ctx.parts:
            timing.update(run_tests(ctx, cfg, universes, uconf))
            sw.lap("tests")
        if "placebo" in ctx.parts:
            timing.update(run_placebo(ctx, cfg, universes, uconf))
            sw.lap("placebo")
        if "link" in ctx.parts:
            timing.update(run_link(ctx, cfg))
            sw.lap("link")
    build_outputs(ctx)
    sw.lap("outputs")
    if ctx.quick and not ctx.plots_only and not ctx.args.no_probe:
        probe = run_probes(ctx, universes, ctx.parts)
        sw.lap("probes")
        est = report_estimates(probe, ctx.parts, timing.get("link_wall_s"), cfg)
        timing.update({f"estimate_wall_s_{k}": v for k, v in est.items()})
    timing.update({f"wall_{k}_s": v for k, v in sw.laps.items()})
    ctx.write_meta(None, timing=timing)
    print(f"TIMING total wall (this run, {ctx.workers} workers): {common.fmt_seconds(sw.total())}",
          flush=True)


if __name__ == "__main__":
    main()
