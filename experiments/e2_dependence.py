"""E2-dep: dependence-only alternatives, marginal held fixed (complement to DESIGN section 4, E2).

Why. In E2 main every alternative (volatility level, tails, GARCH persistence at matched
variance, jumps) also changes the one-dimensional marginal law of a daily return, and pooled KS
on the n * L flattened returns reaches power 1 there. This experiment is the complementary case
that matters in practice: a generator that reproduces the marginal *exactly* but gets the
temporal dependence inside a window wrong (e.g. a block bootstrap with too-short blocks, which
destroys volatility clustering). KS sees only the marginal and has no power by construction;
window-based tests (critic, lrcritic, c2st, mmd, sigmmd) can detect the change.

P (the "real" law) is the E2 base GARCH(1,1)-t process of family (c), imported from
``e2_ground_truth.py``: alpha = 0.08, beta = 0.90 (persistence 0.98), nu = 5, unconditional daily
volatility 0.02, mean 3e-4 per day. Windows of L = 20 days, each from its own path started at the
unconditional variance with a 250-day burn-in (exactly as in E2 main), so windows are i.i.d. and
permutation p-values are exact (protocol "iid", DESIGN 3.7).

Families (grid index k = 0..4; k = 0 is H0, Q = P):

mix    Q_delta, a window-level mixture: with probability delta a window consists of L independent
       returns, the return at position t taken from position t of its own independent P window
       (one P window per return); otherwise it is a P window. delta in {0, 0.25, 0.5, 0.75, 1}.
       Window density (1 - delta) p(w) + delta m(w), with m(w) = prod_t f_t(w_t) the product of
       P's position marginals.
block  Q_b: each window is the concatenation of L / b independent P blocks of length b; block j
       (positions j*b .. (j+1)*b - 1) is cut at the same positions from its own independent P
       window. b in {20 (Q = P, H0), 10, 5, 2, 1}; b = 1 is the i.i.d. marginal (the same law as
       mix delta = 1). This mirrors the block length of a block bootstrap. Window density
       prod_j p_j(w_j), p_j the joint density of P's returns at the positions of block j.

Marginal identity. In both families and at every level, every individual return of a Q window
has exactly the law of the return at the same position of a P window, i.e. P's (stationary,
after the burn-in) marginal: by construction it *is* the return at that position of an
independent P window. Within a block the joint law is P's too; only the dependence between
returns of one window changes. Each run (also ``--quick``) verifies this numerically: two-sample
KS between 10^5 (full run: 10^6) returns of P and of Q at the extreme level (delta = 1, b = 1).
The P returns are taken one per independent window (position cycling through 1..L), because
pooled returns of one GARCH window are dependent and would invalidate the KS p-value; the
extreme-level Q returns are all independent. ``e2dep_check_<family>.csv`` also records
dependence diagnostics per level (within-window lag-1 autocorrelation of |r - mu| and the
cross-window sd of the log realised variance), which do change. Kurtosis is not reported: this
GARCH-t has an infinite fourth moment (0.08^2 * 9 + 2 * 0.08 * 0.90 + 0.90^2 > 1).

KS. Since the marginal is identical, KS has no power by construction. Within-window GARCH
dependence also makes its nominal p-value invalid (E2 family (c): about 8% at nominal 5%), so KS
is marked with a dagger in the tables and faded in the figure; its rejection rates here measure
that size distortion, not power.

Design. n in {500, 2000} windows per sample; 200 replications at k = 0 and 100 otherwise. The X
(P) sample of replication r is shared by the levels of a (family, n) cell (common random numbers,
as in E2). The H0 cells of the two families use independent seeds (two independent size
estimates).

Tests: critic, lrcritic, c2st, mmd, sigmmd (permutation p-values, exact for independent windows)
and ks (asymptotic; invalid here, see above), configured exactly as in E2 main
(``e2_ground_truth.make_tests``: the TEST_REGISTRY defaults), plus an **approximate (Monte
Carlo)** likelihood-ratio oracle: the Monte Carlo test of H0: Y ~ P against H1: Y ~ Q with
statistic T = sum_i log q(y_i) / p(y_i), where

* p(w) = ``GARCHt.loglik_windows(w, sigma2_init=draws)``, with K = 512 draws of the conditional
  variance at the window start from the same law as the simulation (250 recursions from the
  unconditional variance; these are E2 family (c)'s draws, ``e2_ground_truth._sigma2_draws``;
  ``loglik_windows(w, n_init=K)`` would draw from the fully stationary law instead);
* f_t (marginal of the return at position t) = Monte Carlo mixture of scaled t(nu) densities over
  Km = 2048 draws of the conditional variance at position t (the start draws propagated t steps
  through the variance recursion), tabulated once per process on 1000 points of
  u = asinh(|r - mu| / 0.002) and evaluated with a cubic spline (beyond |r - mu| = 2 the t(nu)
  power tail); the run reports the interpolation error against the direct Monte Carlo mixture;
* p_j (block density) = ``GARCHt.loglik_windows`` on the block's returns with K draws at the
  block's start position.

The null distribution of T does not depend on the replication, so it is computed once per
(family, level, n) cell from B = 1000 samples of 2000 P windows (n = 2000: one sum per sample;
n = 500: four disjoint sums per sample, i.e. 4000 null values; both families use the same null
windows, so p and the marginal are computed once per window), and the oracle p-value is the
Monte Carlo p-value with randomised ties. Statistic and null use the same fixed draws, so the
test is exact; only its optimality is approximate (K, Km finite), so it is labelled "approximate
(Monte Carlo)" in every CSV, table and figure and is not a strict power bound. At k = 0, T = 0
identically and the p-value is uniform. Its cost (null + per-replication densities) is printed
by the quick run (target: at most about 20% of the run). ``--no-oracle`` drops it.

Usage::

    uv run python experiments/e2_dependence.py --quick       # smoke test + full-run estimate
    uv run python experiments/e2_dependence.py --workers 4   # full run, both families
    uv run python experiments/e2_dependence.py --part mix --workers 4
    uv run python experiments/e2_dependence.py --plots-only  # rebuild figures and tables

Run both families in one process (the default) so they share the oracle null windows and one
worker pool; if they run in separate processes, finish with ``--plots-only``.

Outputs (full mode; ``--quick`` writes everything, figure and tables included, under
``results/e2dep_quick/`` only):

* ``results/e2dep/e2dep_<family>_raw.csv`` (one row per replication and test),
  ``e2dep_rejection.csv`` (rates at alpha in {0.01, 0.05, 0.10} with Wilson 95% intervals),
  ``e2dep_size.csv``, ``e2dep_check_<family>.csv`` (marginal check and dependence diagnostics),
  ``e2dep_oracle_check.json``, ``meta.json``, ``meta_<family>.json``;
* ``paper/figures/e2dep_power.pdf``: panel (a) mix vs delta, panel (b) block vs b (categorical
  axis, b decreasing to the right); one line per test (colours, markers and labels of
  ``e2_power.pdf`` from ``synthval.plotting``), Wilson 95% error bars, solid n = 500, dashed
  n = 2000, the oracle in black;
* ``paper/tables/e2dep_size.tex`` (rates under H0, Wilson CIs, N valid replications, p-value
  type), ``e2dep_power.tex`` (rate at 5% with Wilson CI per level, N, p-value type) and the
  generated notes ``e2dep_size_note.tex`` / ``e2dep_power_note.tex``. Everything is built from
  the CSVs (``--plots-only``).

Checkpoints: blocks of 10 replications under ``results/e2dep/checkpoints/main/<hash>/`` and the
oracle null chunks under ``results/e2dep/checkpoints/oracle_<family>/<hash>/``. The hashes cover
the configuration, the library files used (``synthval/models``, ``synthval/twosample``,
``synthval/config.py``, via ``e2_ground_truth.code_keys``) and the source of every function of
this script and of ``e2_ground_truth.py`` that determines the results, so a resumed run never
mixes code versions. Seeds are keyed by (part, family, n, level, replication[, test]), so results
do not depend on the number of workers or on which family runs.

Memory: like E2 main, about 0.6 GB per worker at n = 2000.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import functools  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import common  # noqa: E402  (experiments/ is on sys.path when run as a script)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

import e2_ground_truth as e2  # noqa: E402  (base process, tests and helpers of E2 main)
from synthval import plotting  # noqa: E402
from synthval.twosample.critic import perm_pvalues  # noqa: E402

logger = logging.getLogger("experiments.e2dep")

EXP = "e2dep"
ROOT_SEED = 20251116
PARTS = ("mix", "block")

L = e2.L              # 20
BURN = e2.BURN        # 250
P_SPEC = e2.spec_garch(0.98)   # alpha 0.08, beta 0.90, nu 5, vol 0.02, mu 3e-4 (E2 family c)
ALPHAS = e2.ALPHAS
TESTS = e2.TESTS      # critic, lrcritic, c2st, mmd, sigmmd, ks
ALL_TESTS = TESTS + ("oracle",)
ORACLE_TYPE = "approximate (Monte Carlo)"
ORACLE_LABEL = "LR oracle, approximate (Monte Carlo)"

FAMILIES: dict[str, dict[str, Any]] = {
    "mix": {"title": "(a) window mixture", "levels": [0.0, 0.25, 0.5, 0.75, 1.0],
            "ticks": ["0", "0.25", "0.5", "0.75", "1"],
            "xlabel": r"fraction $\delta$ of i.i.d.-marginal windows in $Q$ ($\delta=0$: $Q=P$)",
            "tex_ticks": ["0", "0.25", "0.5", "0.75", "1"], "param": r"$\delta$"},
    "block": {"title": "(b) block length", "levels": [20, 10, 5, 2, 1],
              "ticks": ["20", "10", "5", "2", "1"],
              "xlabel": r"block length $b$ of $Q$ ($b=20$: $Q=P$; $b=1$: i.i.d. marginal)",
              "tex_ticks": ["20", "10", "5", "2", "1"], "param": r"$b$"},
}
EXTREME_K = 4         # delta = 1 / b = 1: the i.i.d. marginal

FULL = {
    "n_list": [2000], "reps_h0": 100, "reps_h1": 50, "tests": {},  # compute budget (was [500, 2000], 200, 100)
    "oracle": True, "oracle_B": 1000, "oracle_K": 512, "oracle_Km": 2048, "oracle_G": 1000,
    "oracle_chunk": 10, "block": 10, "check_n": 1_000_000, "diag_windows": 20_000,
}
QUICK = {
    "n_list": [100, 200], "reps_h0": 3, "reps_h1": 3,
    "tests": {"n_steps": 200, "n_perm": 200, "mmd_n_perm": 200},
    "oracle": True, "oracle_B": 100, "oracle_K": 64, "oracle_Km": 512, "oracle_G": 1000,
    "oracle_chunk": 25, "block": 3, "check_n": 100_000, "diag_windows": 5_000,
}
PROBE_K = {"mix": 2, "block": 3}   # levels timed by the quick-run probes (delta 0.5, b = 2)

MARG_C = 0.002        # scale of the asinh grid of the tabulated marginal (0.1 x daily vol)
MARG_AMAX = 2.0       # end of the grid in |r - mu|; beyond it the t(nu) power tail is used


# --------------------------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------------------------
def p_model() -> Any:
    """The GARCHt model of P (cached per process by e2_ground_truth)."""
    return e2.model_of(P_SPEC)


def p_params() -> dict[str, float]:
    return p_model().params()


def sim_P(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` independent P windows ``(n, L)``, each from its own path (250-day burn-in)."""
    return e2.sim_windows(P_SPEC, int(n), rng)


def sim_block(n: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` Q_b windows: block j (positions j*b..(j+1)*b-1) cut at the same positions from its
    own independent P window, so every block has exactly P's joint law at its positions."""
    b = int(b)
    if b < 1 or L % b:
        raise ValueError(f"block length {b} must divide L = {L}")
    m = L // b
    if m == 1:
        return sim_P(n, rng)
    W = sim_P(n * m, rng).reshape(n, m, L)
    out = np.empty((n, L))
    for j in range(m):
        out[:, j * b:(j + 1) * b] = W[:, j, j * b:(j + 1) * b]
    return out


def sim_mix(n: int, delta: float, rng: np.random.Generator) -> np.ndarray:
    """``n`` Q_delta windows: each, with probability ``delta``, L independent returns (position t
    from position t of its own P window, i.e. ``sim_block(., 1)``), otherwise a P window."""
    is_iid = rng.random(n) < float(delta)
    ni = int(is_iid.sum())
    out = np.empty((n, L))
    if ni:
        out[is_iid] = sim_block(ni, 1, rng)
    if ni < n:
        out[~is_iid] = sim_P(n - ni, rng)
    return out


def sim_Q(fam: str, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    lev = FAMILIES[fam]["levels"][int(k)]
    if fam == "mix":
        return sim_mix(n, float(lev), rng)
    if fam == "block":
        return sim_block(n, int(lev), rng)
    raise ValueError(f"unknown family {fam!r}")


# --------------------------------------------------------------------------------------------
# Oracle densities (approximate, Monte Carlo)
# --------------------------------------------------------------------------------------------
LW_ROWS = 500         # windows per loglik_windows call (keeps the (rows, K) arrays in cache:
#                       about 3x faster at K = 512 than one call on 2000 windows; same values)


def lw(W: np.ndarray, sigma2_init: np.ndarray) -> np.ndarray:
    """``GARCHt.loglik_windows(W, sigma2_init=...)`` of P, evaluated in row chunks (row-wise
    independent, so the values are those of a single call)."""
    m = p_model()
    return np.concatenate([m.loglik_windows(W[i:i + LW_ROWS], sigma2_init=sigma2_init)
                           for i in range(0, W.shape[0], LW_ROWS)])


@functools.lru_cache(maxsize=None)
def sigma2_paths(K: int) -> np.ndarray:
    """``(K, L)`` draws of the conditional variance at each window position.

    Column 0: E2 family (c)'s draws (law after the 250-day burn-in, as in the simulation);
    column t: column t-1 propagated one step through the variance recursion with fresh
    innovations, so column t has exactly the law of the variance at position t. Keyed, hence
    identical in every process."""
    s0 = e2._sigma2_draws(e2.spec_key(P_SPEC), int(K))
    if s0 is None:
        raise RuntimeError("GARCHt.stationary_variance_draws is missing: no Monte Carlo oracle")
    p = p_params()
    S = np.empty((int(K), L))
    S[:, 0] = s0
    rng = common.keyed_rng(ROOT_SEED, "sigma2_path", int(K))
    zs = math.sqrt((p["nu"] - 2.0) / p["nu"])
    for t in range(1, L):
        z = rng.standard_t(p["nu"], size=int(K)) * zs
        S[:, t] = p["omega"] + (p["alpha"] * z * z + p["beta"]) * S[:, t - 1]
    return S


@functools.lru_cache(maxsize=None)
def marg_table(Km: int, G: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid ``u`` (G,) of asinh(|r - mu| / MARG_C) and ``log f_t`` on it, shape ``(L, G)``:
    the Monte Carlo mixture over the Km variance draws at position t (loglik_windows on
    one-return windows)."""
    u = np.linspace(0.0, math.asinh(MARG_AMAX / MARG_C), int(G))
    a = MARG_C * np.sinh(u)
    S = sigma2_paths(int(Km))
    mu = p_params()["mu"]
    logf = np.stack([lw((mu + a)[:, None], S[:, t]) for t in range(L)])
    return u, logf


@functools.lru_cache(maxsize=None)
def _marg_splines(Km: int, G: int) -> list[CubicSpline]:
    u, logf = marg_table(Km, G)
    return [CubicSpline(u, logf[t]) for t in range(L)]


def logm(W: np.ndarray, Km: int, G: int) -> np.ndarray:
    """``log m(w) = sum_t log f_t(w_t)`` per window (tabulated Monte Carlo marginals)."""
    u_grid, logf = marg_table(Km, G)
    spl = _marg_splines(Km, G)
    p = p_params()
    a = np.abs(np.asarray(W, dtype=float) - p["mu"])
    u = np.arcsinh(a / MARG_C)
    out = np.zeros(a.shape[0])
    for t in range(L):
        val = spl[t](np.minimum(u[:, t], u_grid[-1]))
        hi = u[:, t] > u_grid[-1]
        if hi.any():  # power tail of the t(nu) mixture beyond the grid
            val[hi] = logf[t, -1] - (p["nu"] + 1.0) * (np.log(a[hi, t]) - math.log(MARG_AMAX))
        out += val
    return out


def logm_direct(W: np.ndarray, Km: int) -> np.ndarray:
    """Untabulated Monte Carlo marginal (used only to check the interpolation)."""
    S = sigma2_paths(int(Km))
    return sum(lw(W[:, t:t + 1], S[:, t]) for t in range(L))


def logp(W: np.ndarray, K: int) -> np.ndarray:
    """P window log-density, Monte Carlo over K start variances (E2 family (c)'s method)."""
    return lw(W, sigma2_paths(int(K))[:, 0])


def logblock(W: np.ndarray, b: int, K: int, Km: int, G: int) -> np.ndarray:
    """Q_b window log-density: sum of the P block densities (K start variances per block)."""
    b = int(b)
    if b == L:
        return logp(W, K)
    if b == 1:
        return logm(W, Km, G)
    S = sigma2_paths(int(K))
    return sum(lw(W[:, j * b:(j + 1) * b], S[:, j * b]) for j in range(L // b))


def llr_windows(fam: str, k: int, W: np.ndarray, oc: dict,
                memo: dict | None = None) -> np.ndarray:
    """Per-window oracle log-likelihood ratio ``log q_k(w) - log p(w)`` (0 at k = 0)."""
    if int(k) == 0:
        return np.zeros(W.shape[0])
    memo = {} if memo is None else memo
    K, Km, G = oc["K"], oc["Km"], oc["G"]

    def get(name: str, fn: Any) -> np.ndarray:
        if name not in memo:
            memo[name] = fn()
        return memo[name]

    lp = get("p", lambda: logp(W, K))
    lev = FAMILIES[fam]["levels"][int(k)]
    if fam == "mix":
        d = float(lev)
        dm = get("m", lambda: logm(W, Km, G)) - lp
        if d >= 1.0:
            return dm
        return np.logaddexp(math.log1p(-d), math.log(d) + dm)
    b = int(lev)
    name = "m" if b == 1 else f"b{b}"
    return get(name, lambda: logblock(W, b, K, Km, G)) - lp


def oracle_cfg(cfg: dict) -> dict:
    return {"K": cfg["oracle_K"], "Km": cfg["oracle_Km"], "G": cfg["oracle_G"]}


# --------------------------------------------------------------------------------------------
# Code fingerprints (checkpoint keys)
# --------------------------------------------------------------------------------------------
_ORACLE_FUNCS = ("p_model", "p_params", "sim_P", "lw", "sigma2_paths", "marg_table",
                 "_marg_splines", "logm", "logp", "logblock", "llr_windows", "_null_chunk")
_MAIN_FUNCS = _ORACLE_FUNCS + ("sim_block", "sim_mix", "sim_Q", "_rep")
_E2_FUNCS = ("sim_windows", "_model", "model_of", "spec_garch", "spec_key", "_persistent",
             "_sigma2_draws", "make_tests", "_test_row", "train_diagnostics", "_ptype")


@functools.lru_cache(maxsize=None)
def code_fingerprint(kind: str) -> str:
    """SHA-256 (16 hex) of the source of the functions (of this script and of
    e2_ground_truth.py) that determine the results of ``kind`` ("oracle" or "main"), plus the
    E2 constants used."""
    own = _ORACLE_FUNCS if kind == "oracle" else _MAIN_FUNCS
    h = hashlib.sha256()
    for name in own:
        h.update(inspect.getsource(globals()[name]).encode() + b"\0")
    for name in _E2_FUNCS:
        h.update(inspect.getsource(getattr(e2, name)).encode() + b"\0")
    h.update(repr((e2.L, e2.BURN, e2.MU, e2.VOL, e2.GARCH_ALPHA, e2.GARCH_NU, e2.ROOT_SEED,
                   e2.TESTS, e2.HAS_MC_INIT)).encode())
    return h.hexdigest()[:16]


def base_config() -> dict:
    return {"seed": ROOT_SEED, "L": L, "burn": BURN, "P": P_SPEC, "P_params": p_params(),
            "marg_grid": [MARG_C, MARG_AMAX]}


# --------------------------------------------------------------------------------------------
# Oracle null distributions (once per cell, shared windows for both families)
# --------------------------------------------------------------------------------------------
def _save_npz(path: str, arrs: dict[str, np.ndarray]) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrs)
    os.replace(tmp, path)


def _null_chunk(payload: dict, rng: np.random.Generator) -> dict[str, dict[str, np.ndarray]]:
    """Null T statistics of null samples b0..b1-1 for each family in ``payload["fams"]``.

    Null sample b = ``n_max`` P windows from ``keyed_rng(ROOT_SEED, "oracle_null", b)`` (the
    same windows for both families). Returns ``{fam: {"n<n>": array (5, nb, n_max // n)}}``:
    for each n, the sums of the per-window log-likelihood ratios over the n_max // n disjoint
    consecutive groups of n windows. ``fams`` maps family -> chunk file ("" = do not save).
    """
    del rng
    b0, b1, n_max, n_list = payload["b0"], payload["b1"], payload["n_max"], payload["n_list"]
    oc, fams = payload["oc"], payload["fams"]
    out: dict[str, dict[str, np.ndarray]] = {}
    todo = {}
    for f, path in fams.items():
        if path and Path(path).exists() and not payload.get("fresh", False):
            with np.load(path) as z:
                out[f] = {k: z[k] for k in z.files}
        else:
            todo[f] = path
    if not todo:
        return out
    nb = b1 - b0
    W = np.concatenate([sim_P(n_max, common.keyed_rng(ROOT_SEED, "oracle_null", b))
                        for b in range(b0, b1)], axis=0)
    memo: dict = {}
    for f, path in todo.items():
        llr = np.stack([llr_windows(f, k, W, oc, memo) for k in range(5)]).reshape(5, nb, n_max)
        arrs = {f"n{n}": llr.reshape(5, nb, n_max // n, n).sum(axis=-1) for n in n_list}
        if path:
            _save_npz(path, arrs)
        out[f] = arrs
    return out


def oracle_config(cfg: dict, fam: str) -> dict:
    return {**base_config(), "family": fam, "levels": FAMILIES[fam]["levels"],
            "B": cfg["oracle_B"], **oracle_cfg(cfg), "n_list": cfg["n_list"],
            "chunk": cfg["oracle_chunk"], **e2.code_keys("models"),
            "code_e2dep": code_fingerprint("oracle")}


def compute_nulls(ctx: common.ExpContext, cfg: dict, fams: list[str]) -> dict[str, str]:
    """Compute (or load) each family's oracle null distribution; returns ``{fam: CSV path}``."""
    n_list = list(cfg["n_list"])
    n_max = max(n_list)
    if any(n_max % n for n in n_list):
        raise ValueError("every n must divide max(n_list) (disjoint null sub-samples)")
    dirs = {f: common.Checkpoint(ctx.ckpt_dir, f"oracle_{f}", oracle_config(cfg, f)).dir
            for f in fams}
    paths = {f: str(dirs[f] / "null.csv") for f in fams}
    todo = [f for f in fams if ctx.fresh or not Path(paths[f]).exists()]
    if not todo:
        return paths
    B, ch = cfg["oracle_B"], cfg["oracle_chunk"]
    items = [(f"null_{b0}", {"b0": b0, "b1": min(B, b0 + ch), "n_max": n_max, "n_list": n_list,
                             "oc": oracle_cfg(cfg), "fresh": ctx.fresh,
                             "fams": {f: str(dirs[f] / f"chunk_{b0:05d}_{ch}.npz")
                                      for f in todo}})
             for b0 in range(0, B, ch)]
    t0 = time.perf_counter()
    res = common.run_items(_null_chunk, items, part="oracle", root_seed=ROOT_SEED,
                           workers=ctx.workers)
    logger.info("oracle nulls for %s: %s", todo, common.fmt_seconds(time.perf_counter() - t0))
    for f in todo:
        frames = []
        for (_key, pl), r in zip(items, res):
            for n in n_list:
                T = r[f][f"n{n}"]  # (5, nb, groups)
                kk, bb, jj = np.meshgrid(np.arange(5), np.arange(pl["b0"], pl["b1"]),
                                         np.arange(T.shape[2]), indexing="ij")
                frames.append(pd.DataFrame({"family": f, "k": kk.ravel(), "n": n,
                                            "b": bb.ravel(), "j": jj.ravel(), "T": T.ravel()}))
        df = pd.concat(frames, ignore_index=True)
        tmp = f"{paths[f]}.{os.getpid()}.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, paths[f])
    return paths


@functools.lru_cache(maxsize=None)
def load_null(path: str) -> dict[tuple[int, int], np.ndarray]:
    df = pd.read_csv(path)
    return {(int(k), int(n)): g["T"].to_numpy() for (k, n), g in df.groupby(["k", "n"])}


# --------------------------------------------------------------------------------------------
# One replication
# --------------------------------------------------------------------------------------------
def _rep(cell: dict, rep: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    del rng  # streams are keyed explicitly below
    fam, n, k = cell["family"], int(cell["n"]), int(cell["k"])
    cfg = cell["_cfg"]
    X = sim_P(n, common.keyed_rng(ROOT_SEED, "X", fam, n, rep))
    Y = sim_Q(fam, k, n, common.keyed_rng(ROOT_SEED, "Y", fam, n, k, rep))
    rows = []
    for name, test in e2.make_tests(cfg["tests"]).items():
        trng = common.keyed_rng(ROOT_SEED, "test", fam, n, k, rep, name)
        res, dt, err = common.safe_test(test, X, Y, rng=trng)
        rows.append(e2._test_row(name, test, res, dt, err))
    if cfg["oracle"]:
        oc = cfg["oc"]
        t0 = time.perf_counter()
        T = float(llr_windows(fam, k, Y, oc).sum())
        if cell.get("_null_path"):
            null = load_null(cell["_null_path"])[(k, n)]
            pv = perm_pvalues(T, null, rng=common.keyed_rng(ROOT_SEED, "oracle_u", fam, n, k, rep))
            pval, n_null = float(pv["randomized"]), int(null.size)
        else:  # timing probe: statistic only
            pval, n_null = np.nan, 0
        rows.append({"test": "oracle", "ptype": "monte_carlo",
                     "seconds": time.perf_counter() - t0, "error": "", "pvalue": pval,
                     "statistic": T, "n_null": n_null, "oracle_type": ORACLE_TYPE,
                     "oracle_K": oc["K"], "oracle_Km": oc["Km"]})
    return rows


def main_config(cfg: dict) -> dict:
    """Everything (apart from cells and reps) that determines main results (both families, so
    the hash does not depend on which family runs)."""
    return {**base_config(), "families": {f: F["levels"] for f, F in FAMILIES.items()},
            "tests": e2.test_settings(cfg["tests"]), "oracle": cfg["oracle"],
            "oracle_B": cfg["oracle_B"] if cfg["oracle"] else None,
            "oracle_cfg": oracle_cfg(cfg) if cfg["oracle"] else None,
            **e2.code_keys("models", "tests"), "code_e2dep": code_fingerprint("main")}


def cell_cfg(cfg: dict) -> dict:
    return {"tests": cfg["tests"], "oracle": cfg["oracle"], "oc": oracle_cfg(cfg)}


def make_cells(cfg: dict, fams: list[str], null_paths: dict[str, str]) -> list[dict]:
    cells = []
    for fam in fams:
        F = FAMILIES[fam]
        for n in cfg["n_list"]:
            for k in range(5):
                cells.append({"cell": common.cell_key(fam=fam, n=n, k=k), "family": fam,
                              "n": n, "k": k, "level": F["levels"][k],
                              "level_label": F["ticks"][k], "_cfg": cell_cfg(cfg),
                              "_null_path": null_paths.get(fam, "")})
    return cells


def run_main(ctx: common.ExpContext, cfg: dict, fams: list[str],
             null_paths: dict[str, str]) -> dict[str, float]:
    t0 = time.perf_counter()
    df = common.run_cells(
        _rep, make_cells(cfg, fams, null_paths), part="main", ctx=ctx, root_seed=ROOT_SEED,
        n_reps=lambda c: cfg["reps_h0"] if c["k"] == 0 else cfg["reps_h1"],
        config=main_config(cfg), block_size=cfg["block"], cost=lambda c: 1.0 + c["n"] / 4000.0)
    timing = {"main_s": time.perf_counter() - t0}
    conf = {**main_config(cfg), "n_list": cfg["n_list"], "reps_h0": cfg["reps_h0"],
            "reps_h1": cfg["reps_h1"]}
    for fam in fams:
        ctx.save_csv(df[df["family"] == fam], f"e2dep_{fam}_raw", config={"part": fam, **conf})
        ctx.write_part_meta(fam, conf, timing)
    return timing


# --------------------------------------------------------------------------------------------
# Marginal check and dependence diagnostics
# --------------------------------------------------------------------------------------------
def _diagnostics(W: np.ndarray, mu: float) -> dict[str, float]:
    a = np.abs(W - mu)
    x, y = a[:, :-1].ravel(), a[:, 1:].ravel()
    rv = np.mean((W - mu) ** 2, axis=1)
    return {"acf_abs_lag1": float(np.corrcoef(x, y)[0, 1]),
            "sd_log_rv": float(np.std(np.log(rv))), "var": float(np.var(W))}


def run_check(ctx: common.ExpContext, cfg: dict, fam: str) -> pd.DataFrame:
    """KS between P and the extreme level of ``fam`` (marginal identity) + diagnostics."""
    n_ret, chunk = int(cfg["check_n"]), 100_000
    mu = p_params()["mu"]
    xs = []
    for c, s in enumerate(range(0, n_ret, chunk)):
        m = min(chunk, n_ret - s)
        W = sim_P(m, common.keyed_rng(ROOT_SEED, "check_P", c))
        xs.append(W[np.arange(m), np.arange(s, s + m) % L])  # one return per window
    x = np.concatenate(xs)
    ys, nw, cw = [], n_ret // L, chunk // L
    for c, s in enumerate(range(0, nw, cw)):
        m = min(cw, nw - s)
        ys.append(sim_Q(fam, EXTREME_K, m, common.keyed_rng(ROOT_SEED, "check_Q", fam, c)).ravel())
    y = np.concatenate(ys)
    ks = stats.ks_2samp(x, y, method="asymp")
    rows = []
    nd = int(cfg["diag_windows"])
    for k in range(5):
        W = sim_Q(fam, k, nd, common.keyed_rng(ROOT_SEED, "diag", fam, k))
        row = {"family": fam, "k": k, "level": FAMILIES[fam]["levels"][k],
               "level_label": FAMILIES[fam]["ticks"][k], "diag_windows": nd,
               **_diagnostics(W, mu)}
        if k == EXTREME_K:
            row.update(ks_n_P=x.size, ks_n_Q=y.size, ks_stat=float(ks.statistic),
                       ks_pvalue=float(ks.pvalue), var_P_ks=float(np.var(x)),
                       var_Q_ks=float(np.var(y)),
                       ks_design="P: one return per independent window (position cycling); "
                                 "Q: all returns of extreme-level windows (independent)")
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"MARGINAL CHECK {fam} (k={EXTREME_K}, level {FAMILIES[fam]['ticks'][EXTREME_K]}): "
          f"KS on {x.size} P vs {y.size} Q returns: D = {ks.statistic:.5f}, "
          f"p = {ks.pvalue:.3f}; var ratio Q/P = {np.var(y) / np.var(x):.4f}", flush=True)
    for r in rows:
        print(f"  DEPENDENCE {fam} k={r['k']} ({r['level_label']}): within-window "
              f"lag-1 acf|r| = {r['acf_abs_lag1']:.3f}, sd log RV = {r['sd_log_rv']:.3f}, "
              f"var x 1e4 = {r['var'] * 1e4:.3f}", flush=True)
    ctx.save_csv(df, f"e2dep_check_{fam}", config={"part": f"check_{fam}", **base_config(),
                                                   "check_n": n_ret, "diag_windows": nd})
    return df


def run_oracle_check(ctx: common.ExpContext, cfg: dict) -> dict[str, float]:
    """Interpolation error of the tabulated marginal vs the direct Monte Carlo mixture."""
    oc = oracle_cfg(cfg)
    W = np.concatenate([sim_P(1000, common.keyed_rng(ROOT_SEED, "oracle_check", 0)),
                        sim_block(1000, 1, common.keyed_rng(ROOT_SEED, "oracle_check", 1))])
    W[0, 0] = p_params()["mu"] + 2.5  # one point beyond the grid (tail extrapolation)
    t0 = time.perf_counter()
    tab = logm(W, oc["Km"], oc["G"])
    dt_tab = time.perf_counter() - t0
    direct = logm_direct(W, oc["Km"])
    err = np.abs(tab - direct)
    out = {"max_abs_err_window": float(err[1:].max()), "max_abs_err_with_tail": float(err.max()),
           "median_abs_err_window": float(np.median(err)), "Km": oc["Km"], "G": oc["G"],
           "n_windows": int(W.shape[0]), "seconds_tabulated_2000_windows": dt_tab}
    print(f"ORACLE CHECK tabulated marginal (Km={oc['Km']}, G={oc['G']}): max |error| per "
          f"window (sum of {L} log-densities) = {out['max_abs_err_window']:.2e} "
          f"(with one beyond-grid return: {out['max_abs_err_with_tail']:.2e})", flush=True)
    ctx.save_json(out, "e2dep_oracle_check", config={"oracle_cfg": oc})
    return out


# --------------------------------------------------------------------------------------------
# Figures and tables (from the saved CSVs only)
# --------------------------------------------------------------------------------------------
TEST_ORDER = {t: i for i, t in enumerate(ALL_TESTS)}
FAM_ORDER = {f: i for i, f in enumerate(FAMILIES)}
MARK_KS, MARK_NAN = r"$^{\dagger}$", r"$^{\ast}$"
ALPHA_COLS = {a: rf"$\alpha={a:g}$" for a in ALPHAS}


def test_label(test: str, tex: bool = True) -> str:
    if test == "oracle":
        return ORACLE_LABEL
    return plotting.LABELS_TEST[test] + (MARK_KS if (tex and test == "ks") else "")


def save_note(ctx: common.ExpContext, name: str, sentences: list[str]) -> Path:
    """Write ``<tab_dir>/<name>.tex``: a generated table note (no hand-typed numbers)."""
    ctx.tab_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.tab_dir / f"{name}.tex"
    text = ("% generated by experiments/e2_dependence.py\n"
            + "\n".join(s for s in sentences if s) + "\n")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def _ks_sentence(tbl: pd.DataFrame) -> str:
    parts = []
    for fam in FAMILIES:
        s = tbl[(tbl["family"] == fam) & (tbl["k"] == 0) & (tbl["test"] == "ks")]
        for _, r in s.sort_values("n").iterrows():
            parts.append(f"{fam}, $n={int(r['n'])}$: "
                         + common.fmt_rate_ci(r["rej_0.05"], r["lo_0.05"], r["hi_0.05"]))
    msg = (f"{MARK_KS} Every return of $Q$ has exactly the marginal law of $P$, so KS (pooled "
           "returns, marginal only) has no power by construction; the returns of one GARCH "
           "window share the conditional variance, so its nominal $p$-value is not valid either")
    if parts:
        msg += " (rejection rate under $H_0$ at nominal 5\\%: " + "; ".join(parts) + ")"
    return msg + ". Its rejection rates are size distortions, not power."


def _ptype_sentence(df: pd.DataFrame) -> str:
    mc = "Monte Carlo"
    o = df[(df["test"] == "oracle") & (df["n_null"] > 0)] if "n_null" in df else df.iloc[:0]
    if len(o):
        nn = o.groupby("n")["n_null"].max()
        mc = ("Monte Carlo with " + ", ".join(f"{int(v)} null values ($n={int(n)}$)"
                                              for n, v in nn.items()))
    return ("$p$-values: permutation of the held-out evaluation labels with the network fixed "
            "(W-critic, LR-critic, C2ST; exact for independent windows), permutation (MMD, "
            f"Sig-MMD), asymptotic (KS), {mc} and randomised ties (LR oracle).")


def _oracle_sentence(df: pd.DataFrame) -> str:
    o = df[df["test"] == "oracle"]
    if o.empty:
        return ""
    K, Km = int(o["oracle_K"].max()), int(o["oracle_Km"].max())
    return ("The LR oracle is approximate (Monte Carlo): the $P$ window and block densities "
            f"integrate the conditional variance at the window or block start over $K={K}$ "
            f"draws of its law, the position marginals over $K_m={Km}$ draws; statistic and "
            "null use the same draws, so the oracle test is exact but only approximately most "
            "powerful (not a strict power bound).")


def _check_sentence(ctx: common.ExpContext) -> str:
    parts = []
    for fam in FAMILIES:
        if not ctx.has_csv(f"e2dep_check_{fam}"):
            continue
        c = ctx.load_csv(f"e2dep_check_{fam}")
        r = c[c["k"] == EXTREME_K].iloc[0]
        parts.append(f"{fam} {FAMILIES[fam]['param']}$\\,=\\,${FAMILIES[fam]['tex_ticks'][EXTREME_K]}: "
                     f"$D={r['ks_stat']:.4f}$, $p={r['ks_pvalue']:.2f}$ "
                     f"({int(r['ks_n_P'])} vs {int(r['ks_n_Q'])} returns)")
    if not parts:
        return ""
    return ("Marginal check (two-sample KS between independent returns of $P$, one per window, "
            "and of $Q$ at the i.i.d.-marginal level): " + "; ".join(parts) + ".")


def build_outputs(ctx: common.ExpContext, cfg: dict) -> None:
    raws = [ctx.load_csv(f"e2dep_{f}_raw", dtype={"level_label": str})
            for f in FAMILIES if ctx.has_csv(f"e2dep_{f}_raw")]
    if not raws:
        logger.info("no results yet: skipping e2dep figures and tables")
        return
    df = pd.concat(raws, ignore_index=True)
    df["level_label"] = [FAMILIES[f]["ticks"][int(k)] for f, k in zip(df["family"], df["k"])]
    by = ["family", "n", "k", "level_label", "test", "ptype"]
    tbl = common.rejection_table(df, by, alphas=ALPHAS)
    tbl["oracle_type"] = np.where(tbl["test"] == "oracle", ORACLE_TYPE, "")
    tbl["ks_invalid"] = tbl["test"] == "ks"
    tbl["_o"] = [(FAM_ORDER[f], n, k, TEST_ORDER[t])
                 for f, n, k, t in zip(tbl["family"], tbl["n"], tbl["k"], tbl["test"])]
    tbl = tbl.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    n_fail = e2._warn_failures(tbl, "e2dep", ["family", "n", "k", "test"])
    ctx.save_csv(tbl, "e2dep_rejection")
    nan_note = [f"{MARK_NAN} Some replications failed (NaN $p$-value) and are excluded; see "
                "\\texttt{n\\_nan} in \\texttt{e2dep\\_rejection.csv}."] if n_fail else []

    # Size table (k = 0; every test except the oracle, whose statistic is identically 0 there).
    size = tbl[(tbl["k"] == 0) & (tbl["test"] != "oracle")].copy()
    ctx.save_csv(size.reset_index(drop=True), "e2dep_size")
    size["Family"] = size["family"].map(lambda f: FAMILIES[f]["title"])
    size["$n$"] = size["n"]
    size["Test"] = [test_label(t) + (MARK_NAN if nn else "")
                    for t, nn in zip(size["test"], size["n_nan"])]
    st = common.rate_ci_table(size, ["Family", "$n$", "Test"], ALPHAS,
                              label=lambda a: ALPHA_COLS[a])
    st.insert(0, "$N$", size["n_valid"].astype(int).to_numpy())
    st.insert(0, "$p$-value", [e2.PTYPE_SHORT.get(p, p) for p in size["ptype"]])
    ctx.save_table(st, "e2dep_size", escape=False)
    save_note(ctx, "e2dep_size_note", [
        "Rejection rates under $H_0$ ($Q=P$: mix $\\delta=0$ and block $b=20$, independent "
        "seeds) with Wilson 95\\% intervals; $N$ = valid replications; independent windows.",
        _ptype_sentence(df), _ks_sentence(tbl)] + nan_note)

    # Power table: rate at alpha = 0.05 with its Wilson interval, per level.
    rows = []
    for (fam, n, test), g in tbl.groupby(["family", "n", "test"]):
        F = FAMILIES[fam]
        g = g.sort_values("k")
        n0 = g.loc[g["k"] == 0, "n_valid"]
        n1 = g.loc[g["k"] > 0, "n_valid"]
        row = {"Family": f"{F['title']} ({F['param']}: " + ", ".join(F["tex_ticks"]) + ")",
               "$n$": int(n), "Test": test_label(test),
               "$p$-value": e2.PTYPE_SHORT.get(g["ptype"].iloc[0], g["ptype"].iloc[0]),
               "$N$": (f"{int(n0.iloc[0]) if len(n0) else 0}/"
                       f"{int(n1.min()) if len(n1) else 0}")}
        for _, r in g.iterrows():
            col = f"$k={int(r['k'])}$" + (" ($H_0$)" if r["k"] == 0 else "")
            row[col] = (common.fmt_rate_ci(r["rej_0.05"], r["lo_0.05"], r["hi_0.05"], 2)
                        + (MARK_NAN if r["n_nan"] else ""))
        rows.append(((FAM_ORDER[fam], int(n), TEST_ORDER[test]), row))
    pw = pd.DataFrame([r for _, r in sorted(rows, key=lambda t: t[0])])
    kcols = sorted([c for c in pw.columns if c.startswith("$k=")],
                   key=lambda c: int(c.split("=")[1].split("$")[0]))
    pw = pw.set_index(["Family", "$n$", "Test"])[["$p$-value", "$N$"] + kcols].fillna("--")
    ctx.save_table(pw, "e2dep_power", escape=False)
    save_note(ctx, "e2dep_power_note", [
        "Rejection rates at $\\alpha=0.05$ with Wilson 95\\% intervals; $N$ = valid "
        "replications under $H_0$ / per alternative (minimum); independent windows. Family "
        "(a): a fraction $\\delta$ of the $Q$ windows has $L$ independent returns; family (b): "
        "$Q$ windows are concatenations of independent $P$ blocks of length $b$. Every return "
        "of $Q$ has exactly the marginal law of $P$.",
        _ptype_sentence(df), _ks_sentence(tbl), _oracle_sentence(df), _check_sentence(ctx)]
        + nan_note)
    plot_power(ctx, tbl)


def plot_power(ctx: common.ExpContext, tbl: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plotting.setup()
    fig, axes = plt.subplots(1, 2, figsize=plotting.figsize("two", 0.46), sharey=True)
    n_vals = sorted(tbl["n"].unique())
    styles = {n: s for n, s in zip(n_vals, ["-", "--", ":", "-."])}
    tests = [t for t in ALL_TESTS if t in set(tbl["test"])]
    # Shift each (test, n) series slightly along x so that the Wilson bars do not overlap; the
    # oracle (the benchmark) stays on the grid.
    series = [(t, n) for t in tests if t != "oracle" for n in n_vals]
    width = 0.3
    dodge = {sn: (i - (len(series) - 1) / 2) * width / max(1, len(series) - 1)
             for i, sn in enumerate(series)}
    dodge.update({("oracle", n): 0.0 for n in n_vals})
    for ax, (fam, F) in zip(axes, FAMILIES.items()):
        sub = tbl[tbl["family"] == fam]
        ax.axhline(0.05, color=plotting.OKABE_ITO["grey"], lw=0.6, ls=":")
        if sub.empty:
            ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center")
        for test in tests:
            col = plotting.COLORS_TEST[test]
            a = 0.35 if test == "ks" else 1.0   # KS: not valid here (faded)
            is_or = test == "oracle"
            for n in n_vals:
                s = sub[(sub["test"] == test) & (sub["n"] == n)].sort_values("k")
                if s.empty:
                    continue
                x = s["k"].to_numpy(dtype=float) + dodge[(test, n)]
                y = s["rej_0.05"].to_numpy(dtype=float)
                yerr = np.vstack([y - s["lo_0.05"].to_numpy(dtype=float),
                                  s["hi_0.05"].to_numpy(dtype=float) - y])
                ax.errorbar(x, y, yerr=np.clip(yerr, 0, None), fmt="none", ecolor=col,
                            elinewidth=0.6, capsize=0, alpha=0.5 * a, zorder=1)
                ax.plot(x, y, color=col, marker=plotting.MARKERS_TEST[test], ls=styles[n],
                        lw=1.4 if is_or else 1.0, ms=4.5 if is_or else 3.2, alpha=a,
                        zorder=3 if is_or else 2)
        if not sub.empty:
            ax.set_title("faded KS: not valid (marginal identical,\nwithin-window dependence)",
                         loc="right", fontsize=6.5)
        ax.set_xticks(range(5), F["ticks"])
        ax.set_xlabel(F["xlabel"])
        ax.set_title(F["title"], loc="left")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.3, 4.3)
    axes[0].set_ylabel(r"rejection rate at $\alpha=0.05$")
    handles = [Line2D([], [], color=plotting.COLORS_TEST[t], marker=plotting.MARKERS_TEST[t],
                      ls="-", label=test_label(t, tex=False)) for t in tests]
    handles += [Line2D([], [], color="black", ls=styles[n], label=f"$n={n}$") for n in n_vals]
    handles += [Line2D([], [], color=plotting.OKABE_ITO["grey"], lw=0.6, marker="|", ms=6,
                       ls="", label="Wilson 95% CI")]
    ncol = math.ceil(len(handles) / 2)
    rows = [handles[:ncol], handles[ncol:]]
    order = [h for j in range(ncol) for h in (rows[0][j:j + 1] + rows[1][j:j + 1])]
    fig.legend(handles=order, loc="outside lower center", ncol=ncol, frameon=False)
    ctx.savefig(fig, "e2dep_power")


# --------------------------------------------------------------------------------------------
# Timing probes (quick mode): a few replications at the FULL settings
# --------------------------------------------------------------------------------------------
def _probe(cell: dict, rep: int, rng: np.random.Generator) -> dict[str, Any]:
    oc = cell["_oc"]
    if oc is not None:  # warm the per-process caches (draws, marginal table) outside the timing
        sigma2_paths(oc["K"])
        _marg_splines(oc["Km"], oc["G"])
    t0 = time.perf_counter()
    if cell["kind"] == "oracle":
        _null_chunk({"b0": 0, "b1": cell["chunk"], "n_max": max(FULL["n_list"]),
                     "n_list": FULL["n_list"], "oc": oc,
                     "fams": {f: "" for f in cell["fams"].split("+")}}, rng)
        return {"probe_seconds": time.perf_counter() - t0}
    rows = _rep(cell, rep, rng)
    return {"probe_seconds": time.perf_counter() - t0,
            "oracle_seconds": sum(r["seconds"] for r in rows if r["test"] == "oracle"),
            "tests_seconds": sum(r["seconds"] for r in rows if r["test"] != "oracle")}


def run_probes(ctx: common.ExpContext, cfg_full: dict, fams: list[str]) -> pd.DataFrame:
    oc = oracle_cfg(cfg_full) if cfg_full["oracle"] else None
    cells: list[dict] = []
    if oc is not None:
        cells.append({"cell": "oracle", "kind": "oracle", "fams": "+".join(fams), "chunk": 2,
                      "_oc": oc})
    for fam in fams:
        k = PROBE_K[fam]
        for n in cfg_full["n_list"]:
            cells.append({"cell": f"{fam}_{n}", "kind": "main", "family": fam, "n": n, "k": k,
                          "_cfg": cell_cfg(cfg_full), "_null_path": "", "_oc": oc})
    return common.run_cells(_probe, cells, part="probe", ctx=ctx, root_seed=ROOT_SEED, n_reps=1,
                            config={"probe": True, "full": cfg_full}, block_size=1,
                            cost=lambda c: 2.0 if c["kind"] == "oracle" else 1.0 + c["n"] / 2000,
                            fresh=True)


def report_estimates(probe: pd.DataFrame, cfg_full: dict, fams: list[str]) -> dict[str, float]:
    F = cfg_full
    t = {r["cell"]: r for _, r in probe.iterrows()}
    null_cpu = 0.0
    if F["oracle"] and "oracle" in t:
        null_cpu = float(t["oracle"]["probe_seconds"]) / 2 * F["oracle_B"]
        print(f"TIMING oracle null ({'+'.join(fams)}): {null_cpu:.0f} s CPU for "
              f"B={F['oracle_B']} samples of {max(F['n_list'])} windows (K={F['oracle_K']}, "
              f"Km={F['oracle_Km']}, tabulated marginal)", flush=True)
    est: dict[str, float] = {}
    tasks_all: list[float] = []
    cpu_tests = cpu_oracle = 0.0
    for fam in fams:
        tasks: list[float] = []
        for n in F["n_list"]:
            r = t[f"{fam}_{n}"]
            s, so = float(r["probe_seconds"]), float(r.get("oracle_seconds", 0.0) or 0.0)
            common.print_rep_timing(f"{fam} n={n}", s,
                                    note=f"full settings, 6 tests + oracle {so:.2f} s, "
                                         f"level k={PROBE_K[fam]}")
            for k in range(5):
                reps = F["reps_h0"] if k == 0 else F["reps_h1"]
                per = s - so if k == 0 else s  # the oracle statistic is 0 at k = 0
                tasks += common.block_tasks(per, reps, F["block"])
                cpu_tests += (s - so) * reps
                cpu_oracle += (0.0 if k == 0 else so) * reps
        tasks_all += tasks
        for w in (4, 12):
            est[f"{fam}_w{w}"] = common.print_wall_estimate(
                f"{fam} (+ oracle null of both families)", tasks, workers=w,
                extra_s=null_cpu / w)
    for w in (4, 12):
        est[f"all_w{w}"] = common.print_wall_estimate(
            f"{'+'.join(fams)} (one process, shared oracle null)", tasks_all, workers=w,
            extra_s=null_cpu / w)
    if F["oracle"]:
        share = (null_cpu + cpu_oracle) / max(cpu_tests, 1e-9)
        print(f"ORACLE COST: {common.fmt_seconds(null_cpu + cpu_oracle)} CPU (null "
              f"{common.fmt_seconds(null_cpu)} + per-replication {common.fmt_seconds(cpu_oracle)})"
              f" = {100 * share:.0f}% of the tests' {common.fmt_seconds(cpu_tests)} CPU",
              flush=True)
        est["oracle_share"] = share
    return est


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------
def _add_args(parser: Any) -> None:
    parser.add_argument("--no-probe", action="store_true",
                        help="quick mode: skip the full-settings timing probes")
    parser.add_argument("--no-oracle", action="store_true",
                        help="drop the approximate (Monte Carlo) LR oracle")


def main(argv: list[str] | None = None) -> None:
    ctx = common.parse_args(EXP, PARTS, description=__doc__, add_arguments=_add_args, argv=argv)
    cfg = dict(QUICK if ctx.quick else FULL)
    full = dict(FULL)
    if ctx.args.no_oracle:
        cfg["oracle"] = full["oracle"] = False
    if cfg["oracle"] and not e2.HAS_MC_INIT:
        raise RuntimeError("GARCHt.stationary_variance_draws missing: run with --no-oracle")
    fams = list(ctx.parts)
    sw = common.Stopwatch()
    timing: dict[str, float] = {}
    ctx.write_meta({"quick": ctx.quick, "parts": fams, "plots_only": ctx.plots_only, "cfg": cfg,
                    **base_config()})
    if not ctx.plots_only:
        for fam in fams:
            run_check(ctx, cfg, fam)
        if cfg["oracle"]:
            run_oracle_check(ctx, cfg)
        sw.lap("checks")
        null_paths: dict[str, str] = {}
        if cfg["oracle"]:
            null_paths = compute_nulls(ctx, cfg, fams)
            sw.lap("oracle_null")
        timing.update(run_main(ctx, cfg, fams, null_paths))
        sw.lap("main")
    build_outputs(ctx, cfg)
    sw.lap("outputs")
    if ctx.quick and not ctx.plots_only and not ctx.args.no_probe:
        probe = run_probes(ctx, full, fams)
        sw.lap("probes")
        est = report_estimates(probe, full, fams)
        timing.update({f"estimate_{k}": v for k, v in est.items()})
    timing.update({f"wall_{k}_s": v for k, v in sw.laps.items()})
    ctx.write_meta(None, timing=timing)
    for k, v in sw.laps.items():
        print(f"TIMING {k}: {common.fmt_seconds(v)}", flush=True)
    print(f"TIMING total wall (this run, {ctx.workers} workers): {common.fmt_seconds(sw.total())}",
          flush=True)


if __name__ == "__main__":
    main()
