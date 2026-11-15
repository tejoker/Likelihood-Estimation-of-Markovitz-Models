"""Wasserstein-critic two-sample test (the paper's method) and helpers shared by learned tests.

The test (DESIGN section 3.7):

1. Split the real sample ``X`` and the synthetic sample ``Y`` into training and evaluation
   halves (by group when group labels are given, so that no group is in both halves).
2. On the training halves, train a critic ``f`` alone (no generator) with the WGAN-GP objective
   ``max_f mean f(X_tr) - mean f(Y_tr) - lambda * E[(||grad f(x_hat)||_2 - 1)^2]``, where
   ``x_hat`` are random interpolates between real-train and synthetic-train points
   (Gulrajani et al. 2017). Inputs are standardised with statistics of the pooled training half.
3. Freeze ``f`` and compute ``T = mean f(X_ev) - mean f(Y_ev)``. For an (approximately)
   1-Lipschitz ``f`` this estimates a lower bound on the Wasserstein-1 distance between the
   two laws (in the standardised representation).
4. p-value by permuting the labels of the pooled evaluation set with ``f`` fixed (one-sided,
   ``(1 + #{T_b >= T}) / (1 + B)``).

**Validity of the permutation p-value.** The permutation test has exact finite-sample level
*under the assumption that the windows are i.i.d.* (more generally: that, conditionally on the
training half, the pooled evaluation windows are exchangeable under H0). With i.i.d. windows
the frozen ``f`` is independent of the evaluation points, so exchangeability holds whatever the
training did. It **fails when windows are serially dependent**, e.g. consecutive windows cut
from one persistent-volatility series (real data, or one long simulated path per sample). Two
things then go wrong: (i) the evaluation scores are autocorrelated within each sample, so the
permutation null is too narrow; (ii) with a *random* train/eval split the critic learns
path-specific volatility regimes from training windows that persist into the neighbouring
evaluation windows, which biases ``T`` upward under H0. Monte Carlo (two independent
GARCH(1,1)-t5 paths, a + b = 0.98, 500 consecutive L = 20 windows each, H0 true, 5% level):
fixed witness, permutation 15.0%, HAC with the Andrews lag 6.8%; trained critic, random split:
permutation 11.0%, HAC 9.0% (the bias (ii) is not a variance problem); trained critic,
10-block time split: permutation 8.0%, HAC 6.5% [3.5, 10.9] (200 replications). For such
data pass ``order_x`` / ``order_y``, set ``time_blocks`` (contiguous time-block train/eval
split, :func:`time_block_labels`) and ``inference="hac"`` (the HAC-studentised p-value
``extra["p_hac"]`` becomes the reported p-value; asymptotic, not exact). In simulation studies,
draw independent windows (one path or one burn-in per window) so that the exact permutation
test applies.

Asymptotic companions: ``extra["z"]`` (difference of means over its i.i.d. standard error) and
``extra["z_hac"]`` (same with a Driscoll-Kraay / Newey-West long-run variance over windows
ordered in time, when ``order_x`` / ``order_y`` are given; Bartlett kernel, data-driven
Andrews (1991) AR(1) plug-in bandwidth by default, see :func:`hac_var_mean`).

The helpers :func:`train_eval_split`, :func:`hac_var_mean`, :func:`perm_mean_diff`,
:func:`perm_pvalues` and :func:`mean_diff_inference` and the base class :class:`LearnedTest`
are reused by :mod:`synthval.twosample.c2st`, :mod:`synthval.twosample.mmd` and
:mod:`synthval.twosample.relative`.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from scipy import stats

from ..config import N_PERM
from .base import TestResult, TwoSampleTest
from .nets import Arch, Standardizer, Witness, build_network, to_tensor, torch_seed

logger = logging.getLogger(__name__)

TieBreak = Literal["conservative", "mid", "randomized"]
HacLags = int | Literal["andrews", "nw"] | None

__all__ = [
    "CriticTest",
    "LearnedTest",
    "train_eval_split",
    "time_block_labels",
    "time_grid",
    "andrews_alpha",
    "andrews_hac_lags",
    "nw_hac_lags",
    "hac_var_mean",
    "perm_mean_diff",
    "perm_pvalues",
    "mean_diff_inference",
    "check_samples",
]


# ----------------------------------------------------------------------------------------------
# Input checks
# ----------------------------------------------------------------------------------------------
def check_samples(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate two samples: float64, at least 2-D, same per-sample shape, finite values."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[1:] != Y.shape[1:]:
        raise ValueError(f"per-sample shapes differ: {X.shape[1:]} vs {Y.shape[1:]}")
    if X.shape[0] < 2 or Y.shape[0] < 2:
        raise ValueError("each sample needs at least 2 observations")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise ValueError("samples contain NaN or inf")
    return X, Y


def _check_aux(a: np.ndarray | None, n: int, what: str) -> np.ndarray | None:
    if a is None:
        return None
    a = np.asarray(a).ravel()
    if a.shape[0] != n:
        raise ValueError(f"{what} has length {a.shape[0]}, expected {n}")
    return a


# ----------------------------------------------------------------------------------------------
# Train / evaluation split
# ----------------------------------------------------------------------------------------------
def train_eval_split(sizes: Sequence[int], *, rng: np.random.Generator, frac_train: float = 0.5,
                     groups: Sequence[np.ndarray | None] | None = None,
                     max_tries: int = 100) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split several samples into training and evaluation index sets.

    Group labels of all samples that have them live in **one shared namespace** (e.g. ticker
    ids for real and synthetic windows): each label is assigned as a whole to the training or to
    the evaluation half for *every* sample, so no group appears in both halves of any sample nor
    across samples. Groups are shuffled and the cut is placed where the cumulative number of
    windows is closest to ``frac_train`` of the total. Samples without labels are split at
    random, with ``round(frac_train * n)`` training points.

    Parameters
    ----------
    sizes : sequence of int
        Number of observations in each sample.
    rng : numpy.random.Generator
        Random generator.
    frac_train : float, default 0.5
        Target fraction of observations in the training half, in (0, 1).
    groups : sequence of (array or None), optional
        Group labels per sample (aligned with that sample's rows).
    max_tries : int, default 100
        Number of reshuffles allowed to obtain non-empty halves for every grouped sample.

    Returns
    -------
    list of (train_idx, eval_idx)
        Sorted integer index arrays, one pair per sample.

    Raises
    ------
    ValueError
        If fewer than two groups exist, or no grouped split leaves both halves non-empty.
    """
    if not 0.0 < frac_train < 1.0:
        raise ValueError("frac_train must be in (0, 1)")
    k = len(sizes)
    groups = list(groups) if groups is not None else [None] * k
    if len(groups) != k:
        raise ValueError("groups must have one entry per sample")
    out: list[tuple[np.ndarray, np.ndarray] | None] = [None] * k

    grouped = [i for i in range(k) if groups[i] is not None]
    if grouped:
        labs = [np.asarray(groups[i]).ravel() for i in grouped]
        for i, g in zip(grouped, labs):
            if g.shape[0] != sizes[i]:
                raise ValueError(f"groups[{i}] has length {g.shape[0]}, expected {sizes[i]}")
        uniq = np.unique(np.concatenate(labs))
        if uniq.size < 2:
            raise ValueError("a group split needs at least two distinct groups")
        codes = [np.searchsorted(uniq, g) for g in labs]
        counts = sum(np.bincount(c, minlength=uniq.size) for c in codes)
        for _ in range(max_tries):
            perm = rng.permutation(uniq.size)
            cum = np.cumsum(counts[perm])
            cut = int(np.argmin(np.abs(cum[:-1] - frac_train * cum[-1])))
            in_train = np.zeros(uniq.size, dtype=bool)
            in_train[perm[:cut + 1]] = True
            ok = True
            pairs = []
            for c in codes:
                mask = in_train[c]
                tr, ev = np.flatnonzero(mask), np.flatnonzero(~mask)
                if tr.size == 0 or ev.size == 0:
                    ok = False
                    break
                pairs.append((tr, ev))
            if ok:
                for i, pr in zip(grouped, pairs):
                    out[i] = pr
                break
        else:
            raise ValueError("could not find a group split with non-empty halves for every sample")

    for i in range(k):
        if out[i] is None:
            n = int(sizes[i])
            if n < 2:
                raise ValueError("each sample needs at least 2 observations to split")
            n_tr = min(max(int(round(frac_train * n)), 1), n - 1)
            p = rng.permutation(n)
            out[i] = (np.sort(p[:n_tr]), np.sort(p[n_tr:]))
    return out  # type: ignore[return-value]


def time_block_labels(order: np.ndarray, n_blocks: int) -> np.ndarray:
    """Label each observation with the index of its contiguous time block.

    The sample's own time range ``[min(order), max(order)]`` is cut into ``n_blocks`` intervals
    of equal length (labels ``0 .. n_blocks - 1``). Used as group labels, these make the
    train/evaluation split of learned tests a split by contiguous periods: with serially
    dependent windows a *random* split lets the network learn path-specific volatility regimes
    from training windows that persist into neighbouring evaluation windows, which biases the
    statistic upward under H0 (a bias that no variance correction can remove). Labels are
    computed per sample, so block ``k`` of ``X`` and block ``k`` of ``Y`` (same label, same side
    of the split) are the ``k``-th periods of their own time ranges.

    Parameters
    ----------
    order : array (n,)
        Time indices.
    n_blocks : int
        Number of blocks (>= 2).

    Returns
    -------
    numpy.ndarray of int64 (n,)
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    o = np.asarray(order, dtype=np.float64).ravel()
    lo, hi = float(o.min()), float(o.max())
    if hi <= lo:
        return np.zeros(o.size, dtype=np.int64)
    lab = np.floor((o - lo) / (hi - lo) * n_blocks).astype(np.int64)
    return np.minimum(lab, n_blocks - 1)


# ----------------------------------------------------------------------------------------------
# Variance of a mean: i.i.d. and HAC (Driscoll-Kraay / Newey-West)
# ----------------------------------------------------------------------------------------------
def time_grid(order: np.ndarray, max_grid: int = 1_000_000) -> tuple[np.ndarray, int]:
    """Map time indices to positions on a regular grid (shared by the HAC and wild bootstrap).

    Integer orders are placed on their actual grid with step equal to the greatest common
    divisor of the gaps between distinct values (so window start days ``0, 20, 40, ...`` give
    a grid in units of one window, and missing windows leave holes). Non-integer orders, or
    integer grids longer than ``max_grid``, use the ranks of the distinct values.

    Parameters
    ----------
    order : array (n,)
        Time index of each observation (ties allowed: same date).
    max_grid : int, default 1_000_000
        Largest dense grid built from integer orders.

    Returns
    -------
    pos : numpy.ndarray of int64 (n,)
        Grid position of each observation.
    n_grid : int
        Grid length.
    """
    order = np.asarray(order).ravel()
    if order.size == 0:
        return np.zeros(0, dtype=np.int64), 0
    if np.issubdtype(order.dtype, np.integer):
        u = np.unique(order)
        step = int(np.gcd.reduce(np.diff(u))) if u.size > 1 else 1
        step = max(step, 1)
        span = int((u[-1] - u[0]) // step) + 1
        if span <= max_grid:
            return ((order - u[0]) // step).astype(np.int64), span
    _, pos = np.unique(order, return_inverse=True)
    pos = pos.astype(np.int64).ravel()
    return pos, int(pos.max()) + 1


def _masked_ar1(S: np.ndarray, mask: np.ndarray, min_pairs: int = 10) -> tuple[float, float] | None:
    """Lag-1 autocorrelation and variance of a series observed on ``mask`` (holes are not zeros).

    Uses only pairs of adjacent grid points that are both observed; returns ``None`` when fewer
    than ``min_pairs`` such pairs exist.
    """
    both = mask[1:] & mask[:-1]
    n_pairs = int(both.sum())
    if n_pairs < min_pairs or mask.sum() < 2:
        return None
    gamma0 = float(np.sum(S[mask] ** 2) / mask.sum())
    if gamma0 <= 0.0:
        return 0.0, 0.0
    gamma1 = float(np.sum((S[1:] * S[:-1])[both]) / n_pairs)
    return gamma1 / gamma0, gamma0


def andrews_alpha(S: np.ndarray, mask: np.ndarray | None = None, rho_max: float = 0.97,
                  min_pairs: int = 10) -> tuple[float, np.ndarray] | None:
    """Andrews (1991) AR(1) plug-in constant ``alpha(1)`` for one or several series.

    For univariate AR(1) approximations with coefficients ``rho_a`` and innovation variances
    ``sigma_a^2`` (equal weights),

        alpha(1) = sum_a 4 rho_a^2 sigma_a^4 / ((1 - rho_a)^6 (1 + rho_a)^2)
                   / sum_a sigma_a^4 / (1 - rho_a)^4.

    ``rho_a`` is the lag-1 autocorrelation over pairs of adjacent, *observed* grid points
    (``mask``), so a series observed on a random subset of dates is not mistaken for a less
    persistent one; it is clipped to ``[-rho_max, rho_max]``.

    Parameters
    ----------
    S : array (T,) or (T, k)
        Series on a regular grid (approximately mean zero).
    mask : bool array (T,), optional
        Observed grid points (default: all).
    rho_max : float, default 0.97
        Clip for the AR(1) coefficients.
    min_pairs : int, default 10
        Minimum number of adjacent observed pairs.

    Returns
    -------
    (alpha, rho) or None
        ``None`` when there are too few adjacent observed pairs.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim == 1:
        S = S[:, None]
    mask = np.ones(S.shape[0], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    num = den = 0.0
    rhos = []
    for a in range(S.shape[1]):
        r = _masked_ar1(S[:, a], mask, min_pairs)
        if r is None:
            return None
        rho, gamma0 = r
        rho = float(np.clip(rho, -rho_max, rho_max))
        s4 = (gamma0 * (1.0 - rho**2)) ** 2
        num += 4.0 * rho**2 * s4 / ((1.0 - rho) ** 6 * (1.0 + rho) ** 2)
        den += s4 / (1.0 - rho) ** 4
        rhos.append(rho)
    alpha = num / den if den > 0 else 0.0
    return float(alpha), np.asarray(rhos)


def andrews_hac_lags(S: np.ndarray, mask: np.ndarray | None = None) -> int | None:
    """Bartlett truncation lag from the Andrews (1991) AR(1) plug-in.

    Bandwidth ``S_T = 1.1447 (alpha(1) T)^(1/3)`` (``T`` = grid length); the Newey-West weights
    ``1 - j / (lags + 1)`` are used with ``lags = ceil(S_T) - 1``. Returns ``None`` if the
    AR(1) coefficient cannot be estimated (too few adjacent observed dates).
    """
    res = andrews_alpha(S, mask)
    if res is None:
        return None
    alpha, _ = res
    T = np.asarray(S).shape[0]
    bw = 1.1447 * (alpha * T) ** (1.0 / 3.0)
    return int(min(max(int(np.ceil(bw)) - 1, 0), max(T - 1, 0)))


def nw_hac_lags(n_time: int) -> int:
    """Newey-West (1994) rule of thumb ``floor(4 (T / 100)^(2/9))`` (not data-driven)."""
    return int(np.floor(4.0 * (max(n_time, 1) / 100.0) ** (2.0 / 9.0)))


def hac_var_mean(g: np.ndarray, order: np.ndarray | None = None, lags: HacLags = None,
                 max_grid: int = 1_000_000) -> tuple[float, int]:
    """Variance of ``mean(g)``, i.i.d. or robust to serial and cross-sectional dependence.

    With ``order`` (time index of each observation) this is the Driscoll-Kraay (1998) estimator:
    centred values are summed within each time index (absorbing any dependence between
    observations at the same date, e.g. windows of different tickers), and a Bartlett-kernel
    (Newey-West 1987) long-run variance is computed on the resulting time series. With one
    observation per date it is the usual Newey-West estimator. Lags are in units of the grid of
    :func:`time_grid` (one step = greatest common gap of the integer time index; missing dates
    count as zero).

    The default lag is data-driven: the Andrews (1991) AR(1) plug-in for the Bartlett kernel,
    with the AR(1) coefficient estimated on adjacent observed dates (:func:`andrews_hac_lags`).
    If too few adjacent dates are observed (sparse irregular grid) the series is compressed to
    the ranks of the observed dates and the lag is chosen there.

    Parameters
    ----------
    g : array (n,)
        Observations.
    order : array (n,), optional
        Time index of each observation. ``None`` gives the i.i.d. variance ``s^2 / n``.
    lags : int, "andrews", "nw" or None, default None
        Bartlett truncation lag. ``None`` / ``"andrews"``: Andrews (1991) AR(1) plug-in;
        ``"nw"``: Newey-West (1994) rule ``floor(4 (T/100)^(2/9))``; an int: fixed lag.
    max_grid : int, default 1_000_000
        Largest dense time grid built from integer orders.

    Returns
    -------
    var : float
        Estimated variance of the sample mean.
    lags : int
        Truncation lag used (0 for the i.i.d. case).
    """
    g = np.asarray(g, dtype=np.float64).ravel()
    n = g.size
    if n < 2:
        return float("nan"), 0
    e = g - g.mean()
    if order is None:
        return float(e @ e / (n - 1) / n), 0
    order = np.asarray(order).ravel()
    if order.shape[0] != n:
        raise ValueError("order must have the same length as g")
    pos, n_grid = time_grid(order, max_grid)
    S = np.bincount(pos, weights=e, minlength=n_grid)
    if lags is None or lags == "andrews":
        mask = np.bincount(pos, minlength=n_grid) > 0
        chosen = andrews_hac_lags(S, mask)
        if chosen is None:
            # Sparse irregular grid: work on consecutive observed dates instead.
            _, pos = np.unique(pos, return_inverse=True)
            S = np.bincount(pos.ravel(), weights=e)
            chosen = andrews_hac_lags(S)
            if chosen is None:
                chosen = nw_hac_lags(S.size)
        lags = chosen
    elif lags == "nw":
        lags = nw_hac_lags(int(np.unique(pos).size))
    elif isinstance(lags, str):
        raise ValueError(f"unknown lags option {lags!r}")
    lags = int(max(0, min(int(lags), S.size - 1)))
    lrv = float(S @ S)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        lrv += 2.0 * w * float(S[lag:] @ S[:-lag])
    # Small-sample correction n/(n-1) to match the i.i.d. estimator when lags = 0 and dates are distinct.
    lrv *= n / (n - 1.0)
    return max(lrv, 0.0) / n**2, lags


# ----------------------------------------------------------------------------------------------
# Permutation null and inference for a difference of means of a fixed function
# ----------------------------------------------------------------------------------------------
def perm_mean_diff(g: np.ndarray, n1: int, n_perm: int, *, rng: np.random.Generator,
                   block: int = 256) -> np.ndarray:
    """Permutation distribution of ``mean(g[first group]) - mean(g[second group])``.

    ``g`` holds the (fixed) witness values on the pooled evaluation set, the first ``n1`` being
    the real sample. Labels are re-drawn ``n_perm`` times; the work is vectorised in blocks of
    ``block`` permutations (memory ``O(block * len(g))``).

    Parameters
    ----------
    g : array (m,)
        Witness values on the pooled sample.
    n1 : int
        Size of the first sample.
    n_perm : int
        Number of permutations.
    rng : numpy.random.Generator
        Random generator.
    block : int, default 256
        Permutations processed per vectorised block.

    Returns
    -------
    numpy.ndarray (n_perm,)
        Permuted statistics.
    """
    g = np.asarray(g, dtype=np.float64).ravel()
    m = g.size
    n2 = m - n1
    if not 0 < n1 < m:
        raise ValueError("both groups must be non-empty")
    total = g.sum()
    base = np.zeros(m, dtype=np.float64)
    base[:n1] = 1.0
    out = np.empty(n_perm, dtype=np.float64)
    for start in range(0, n_perm, block):
        b = min(block, n_perm - start)
        mask = rng.permuted(np.tile(base, (b, 1)), axis=1)
        s1 = mask @ g
        out[start:start + b] = s1 / n1 - (total - s1) / n2
    return out


def perm_pvalues(t_obs: float, t_perm: np.ndarray, *, rng: np.random.Generator | None = None,
                 rtol: float = 1e-12) -> dict[str, float]:
    """One-sided Monte Carlo permutation p-values with three treatments of ties.

    With ``B`` permuted statistics, ``G = #{T_b > T}`` and ``E = #{T_b = T}`` (equality up to
    ``rtol * max(1, |T|)``, which absorbs float round-off):

    * ``"conservative"``: ``(1 + G + E) / (1 + B)`` (the DESIGN formula ``(1 + #{T_b >= T}) /
      (1 + B)``; valid, conservative when the statistic is discrete);
    * ``"mid"``: ``(G + (1 + E) / 2) / (1 + B)`` (mid-p; not guaranteed valid, nominal on
      average);
    * ``"randomized"``: ``(G + U (1 + E)) / (1 + B)`` with ``U ~ U(0, 1)``: under exchangeability
      the observed statistic's rank among the ``B + 1`` values, with ties broken at random, is
      uniform, and this p-value is then exactly ``U(0, 1)`` (exact size for every alpha, even for
      discrete statistics such as a classification accuracy). Only computed when ``rng`` is
      given.

    For continuous statistics ties have probability zero and the three coincide up to
    ``1 / (1 + B)``.

    Returns
    -------
    dict
        ``conservative``, ``mid``, ``randomized`` (NaN without ``rng``), ``n_ties`` (``E``).
    """
    t_perm = np.asarray(t_perm, dtype=np.float64).ravel()
    B = t_perm.size
    tol = rtol * max(1.0, abs(float(t_obs)))
    gt = int(np.sum(t_perm > t_obs + tol))
    eq = int(np.sum(np.abs(t_perm - t_obs) <= tol))
    out = {"conservative": (1.0 + gt + eq) / (1.0 + B),
           "mid": (gt + 0.5 * (1.0 + eq)) / (1.0 + B),
           "randomized": float("nan"), "n_ties": float(eq)}
    if rng is not None:
        out["randomized"] = (gt + float(rng.uniform()) * (1.0 + eq)) / (1.0 + B)
    return out


def _z_and_p(diff: float, se: float) -> tuple[float, float]:
    """One-sided normal z and p-value for H1: mean difference > 0."""
    if not np.isfinite(se) or se <= 0.0:
        if diff == 0.0 or not np.isfinite(diff):
            return 0.0, 0.5
        z = float(np.sign(diff) * np.inf)
    else:
        z = float(diff / se)
    return z, float(stats.norm.sf(z))


def mean_diff_inference(gx: np.ndarray, gy: np.ndarray, *, rng: np.random.Generator,
                        n_perm: int, order_x: np.ndarray | None = None,
                        order_y: np.ndarray | None = None,
                        hac_lags: HacLags = None,
                        tie_break: TieBreak = "conservative") -> dict[str, Any]:
    """Permutation, i.i.d.-z and HAC-z inference for ``D = mean(gx) - mean(gy)``.

    The permutation p-value is exact only if the pooled evaluation points are exchangeable
    under H0 (i.i.d. windows); the HAC z is the asymptotic alternative for serially dependent
    windows.

    Parameters
    ----------
    gx, gy : arrays
        Values of a fixed function (frozen critic / logit / indicator) on the evaluation points
        of the real and the synthetic sample.
    rng : numpy.random.Generator
        Random generator (permutations, randomized tie-breaking).
    n_perm : int
        Number of label permutations (0 skips the permutation test; ``p_perm`` is then NaN).
    order_x, order_y : arrays, optional
        Time indices of the evaluation points, for the HAC variance.
    hac_lags : int, "andrews", "nw" or None
        Bartlett truncation lag (default: Andrews AR(1) plug-in per sample), see
        :func:`hac_var_mean`.
    tie_break : {"conservative", "mid", "randomized"}, default "conservative"
        Which permutation p-value is returned as ``p_perm`` (see :func:`perm_pvalues`).

    Returns
    -------
    dict
        ``diff``, ``p_perm``, ``p_perm_conservative``, ``p_perm_mid``, ``perm_ties``, ``z``,
        ``p_z``, ``z_hac``, ``p_hac``, ``hac`` (bool), ``hac_lags``, ``se``, ``se_hac``.
    """
    if tie_break not in ("conservative", "mid", "randomized"):
        raise ValueError("tie_break must be 'conservative', 'mid' or 'randomized'")
    gx = np.asarray(gx, dtype=np.float64).ravel()
    gy = np.asarray(gy, dtype=np.float64).ravel()
    diff = float(gx.mean() - gy.mean())
    if n_perm > 0:
        t_perm = perm_mean_diff(np.concatenate([gx, gy]), gx.size, n_perm, rng=rng)
        pv = perm_pvalues(diff, t_perm, rng=rng if tie_break == "randomized" else None)
        p_perm = pv[tie_break]
        p_cons, p_mid, ties = pv["conservative"], pv["mid"], int(pv["n_ties"])
    else:
        p_perm = p_cons = p_mid = float("nan")
        ties = 0
    vx, _ = hac_var_mean(gx)
    vy, _ = hac_var_mean(gy)
    se = float(np.sqrt(vx + vy))
    z, p_z = _z_and_p(diff, se)
    hac = order_x is not None or order_y is not None
    if hac:
        hx, lx = hac_var_mean(gx, order_x, hac_lags)
        hy, ly = hac_var_mean(gy, order_y, hac_lags)
        se_hac = float(np.sqrt(hx + hy))
        z_hac, p_hac = _z_and_p(diff, se_hac)
        lags_used: tuple[int, int] | None = (lx, ly)
    else:
        se_hac, z_hac, p_hac, lags_used = se, z, p_z, None
    return {"diff": diff, "p_perm": p_perm, "p_perm_conservative": p_cons, "p_perm_mid": p_mid,
            "perm_ties": ties, "z": z, "p_z": p_z, "se": se, "z_hac": z_hac, "p_hac": p_hac,
            "se_hac": se_hac, "hac": hac, "hac_lags": lags_used}


# ----------------------------------------------------------------------------------------------
# Learned tests: common machinery
# ----------------------------------------------------------------------------------------------
def _time_block_groups(orders: Sequence[np.ndarray | None], groups: Sequence[np.ndarray | None],
                       n_blocks: int) -> list[np.ndarray]:
    """Time-block group labels for each sample (see :func:`time_block_labels`)."""
    if any(g is not None for g in groups):
        raise ValueError("pass either groups_* or time_blocks, not both (for a panel, time "
                         "blocks shared by all tickers are the robust choice)")
    if any(o is None for o in orders):
        raise ValueError("time_blocks requires the time indices order_* of every sample")
    return [time_block_labels(o, n_blocks) for o in orders]  # type: ignore[arg-type]


class LearnedTest(TwoSampleTest):
    """Base class for tests that learn a witness function on a training half.

    Subclasses implement :meth:`_train` (fit a network on standardised training data) and may
    override :meth:`_witness_values` and :meth:`_stat_from_diff`. The statistic must be an affine
    function ``stat_offset + stat_scale * (mean g(X_ev) - mean g(Y_ev))`` of a difference of
    means of a fixed function ``g`` (this is what makes the vectorised permutation test, the HAC
    z and the paired bootstrap of :mod:`synthval.twosample.relative` possible).

    Parameters
    ----------
    n_steps : int
        Number of optimiser steps.
    batch_size : int
        Mini-batch size *per sample*.
    lr : float
        Adam learning rate.
    betas : tuple of float
        Adam betas.
    weight_decay : float
        Adam weight decay (L2).
    arch : {"auto", "mlp", "conv"}
        Architecture, see :func:`synthval.twosample.nets.build_network`.
    hidden : int, optional
        Hidden width (default 128 for the MLP, 64 for the CNN).
    depth : int
        Number of MLP hidden layers.
    layer_norm : bool
        LayerNorm in the MLP hidden layers.
    train_frac : float
        Fraction of observations used for training.
    n_perm : int
        Number of evaluation-label permutations.
    early_stopping : bool
        Hold out ``val_frac`` of the training half, evaluate the validation objective every
        ``eval_every`` steps, keep the best weights and stop after ``patience`` evaluations
        without improvement.
    val_frac, eval_every, patience :
        Early-stopping settings.
    min_steps : int
        With early stopping, checkpoints before ``min_steps`` are neither selected nor counted
        towards ``patience``.
    curve_points : int
        Maximum number of points kept in the returned training curve.
    hac_lags : int, "andrews", "nw" or None
        Bartlett truncation lag for ``z_hac`` (default: Andrews AR(1) plug-in, see
        :func:`hac_var_mean`).
    inference : {"permutation", "hac"}
        Which p-value is reported as ``pvalue``. ``"permutation"``: the held-out permutation
        p-value (exact for i.i.d. windows; the normal p-value of ``z`` when ``n_perm == 0``).
        ``"hac"``: the normal p-value of the HAC-studentised ``z_hac`` (asymptotic, robust to
        serial dependence between windows; pass ``order_x`` / ``order_y``, otherwise it equals
        the i.i.d. normal p-value). Recommended for real data cut into consecutive windows.
    tie_break : {"conservative", "mid", "randomized"}
        Treatment of ties in the permutation p-value (:func:`perm_pvalues`); only matters for
        discrete statistics.
    time_blocks : int, optional
        If set, the train/eval split is by ``time_blocks`` contiguous time blocks built from
        ``order_x`` / ``order_y`` (:func:`time_block_labels`; required then, and ``groups_*``
        must not be given). Recommended (e.g. 10) together with ``inference="hac"`` for serially
        dependent windows: a random split leaks volatility regimes from training to evaluation
        windows and biases the statistic under H0. For a panel, block labels are shared by all
        tickers, which also prevents leakage through common (market) shocks at the same dates.
    """

    name = "learned"
    stat_offset: float = 0.0
    stat_scale: float = 1.0

    def __init__(self, *, n_steps: int, batch_size: int, lr: float,
                 betas: tuple[float, float] = (0.9, 0.999), weight_decay: float = 0.0,
                 arch: Arch = "auto", hidden: int | None = None, depth: int = 3,
                 layer_norm: bool = False, train_frac: float = 0.5, n_perm: int = N_PERM,
                 early_stopping: bool = False, val_frac: float = 0.2, eval_every: int = 50,
                 patience: int = 5, min_steps: int = 0, curve_points: int = 100,
                 hac_lags: HacLags = None,
                 inference: Literal["permutation", "hac"] = "permutation",
                 tie_break: TieBreak = "conservative", time_blocks: int | None = None) -> None:
        if n_steps < 1 or batch_size < 1 or lr <= 0:
            raise ValueError("n_steps, batch_size and lr must be positive")
        if n_perm < 0:
            raise ValueError("n_perm must be >= 0")
        if inference not in ("permutation", "hac"):
            raise ValueError("inference must be 'permutation' or 'hac'")
        if tie_break not in ("conservative", "mid", "randomized"):
            raise ValueError("tie_break must be 'conservative', 'mid' or 'randomized'")
        if min_steps < 0:
            raise ValueError("min_steps must be >= 0")
        if time_blocks is not None and time_blocks < 2:
            raise ValueError("time_blocks must be >= 2")
        self.n_steps = int(n_steps)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.weight_decay = float(weight_decay)
        self.arch: Arch = arch
        self.hidden = hidden
        self.depth = int(depth)
        self.layer_norm = bool(layer_norm)
        self.train_frac = float(train_frac)
        self.n_perm = int(n_perm)
        self.early_stopping = bool(early_stopping)
        self.val_frac = float(val_frac)
        self.eval_every = int(eval_every)
        self.patience = int(patience)
        self.min_steps = int(min_steps)
        self.curve_points = int(curve_points)
        self.hac_lags = hac_lags
        self.inference = inference
        self.tie_break: TieBreak = tie_break
        self.time_blocks = None if time_blocks is None else int(time_blocks)

    # -- hooks ------------------------------------------------------------------------------
    def _train(self, net: torch.nn.Module, Xt: torch.Tensor, Yt: torch.Tensor,
               Xv: torch.Tensor | None, Yv: torch.Tensor | None) -> dict[str, Any]:
        """Train ``net`` in place on standardised tensors; return curve information."""
        raise NotImplementedError

    def _witness_values(self, out: np.ndarray) -> np.ndarray:
        """Map raw network outputs on evaluation points to the function ``g`` whose means are compared."""
        return out

    def _stat_from_diff(self, diff: float) -> float:
        return self.stat_offset + self.stat_scale * diff

    # -- shared pieces ----------------------------------------------------------------------
    def _subsample_curve(self, curve: dict[str, list[float]]) -> dict[str, list[float]]:
        if not curve or not curve.get("step"):
            return curve
        n = len(curve["step"])
        if n <= self.curve_points:
            return curve
        idx = np.unique(np.linspace(0, n - 1, self.curve_points).round().astype(int))
        return {k: [v[i] for i in idx] for k, v in curve.items()}

    def fit_witness(self, X_tr: np.ndarray, Y_tr: np.ndarray, *,
                    rng: np.random.Generator) -> tuple[Witness, dict[str, Any]]:
        """Fit the network on training samples and return the frozen witness.

        Inputs are standardised with statistics of the pooled ``X_tr`` and ``Y_tr``; the returned
        :class:`~synthval.twosample.nets.Witness` applies the same transform to raw inputs.

        Parameters
        ----------
        X_tr, Y_tr : arrays (n, ...)
            Real and synthetic training samples (same per-sample shape).
        rng : numpy.random.Generator
            Random generator (network initialisation, mini-batches, validation split).

        Returns
        -------
        witness : Witness
            Numpy callable returning the raw network output (critic value / logit).
        info : dict
            ``train_curve``, ``val_curve`` (if early stopping), ``steps_run``, ``train_time_s``,
            ``arch``.
        """
        X_tr, Y_tr = check_samples(X_tr, Y_tr)
        t0 = time.perf_counter()
        std = Standardizer().fit(np.concatenate([X_tr, Y_tr], axis=0))
        Xs, Ys = std.transform(X_tr), std.transform(Y_tr)
        Xv = Yv = None
        if self.early_stopping:
            (ix_t, ix_v), (iy_t, iy_v) = train_eval_split(
                [Xs.shape[0], Ys.shape[0]], rng=rng, frac_train=1.0 - self.val_frac)
            Xs, Xv_np = Xs[ix_t], Xs[ix_v]
            Ys, Yv_np = Ys[iy_t], Ys[iy_v]
            Xv, Yv = to_tensor(Xv_np), to_tensor(Yv_np)
        with torch_seed(rng):
            net = build_network(X_tr.shape[1:], self.arch, hidden=self.hidden, depth=self.depth,
                                layer_norm=self.layer_norm)
            info = self._train(net, to_tensor(Xs), to_tensor(Ys), Xv, Yv)
        info["train_curve"] = self._subsample_curve(info.get("train_curve", {}))
        if "val_curve" in info:
            info["val_curve"] = self._subsample_curve(info["val_curve"])
        info["train_time_s"] = time.perf_counter() - t0
        info["arch"] = type(net).__name__
        for p in net.parameters():
            p.requires_grad_(False)
        return Witness(net, std), info

    def _early_stop_loop_state(self) -> dict[str, Any]:
        return {"best": -np.inf, "best_state": None, "bad": 0, "best_step": 0}

    def __call__(self, X: np.ndarray, Y: np.ndarray, *, rng: np.random.Generator,
                 groups_x: np.ndarray | None = None, groups_y: np.ndarray | None = None,
                 order_x: np.ndarray | None = None,
                 order_y: np.ndarray | None = None) -> TestResult:
        """Run the test. See the module docstring; ``X`` real, ``Y`` synthetic."""
        X, Y = check_samples(X, Y)
        gx_lab = _check_aux(groups_x, X.shape[0], "groups_x")
        gy_lab = _check_aux(groups_y, Y.shape[0], "groups_y")
        ox = _check_aux(order_x, X.shape[0], "order_x")
        oy = _check_aux(order_y, Y.shape[0], "order_y")
        split = "group" if (gx_lab is not None or gy_lab is not None) else "random"
        if self.time_blocks is not None:
            gx_lab, gy_lab = _time_block_groups([ox, oy], [gx_lab, gy_lab], self.time_blocks)
            split = "time_block"
        (tx, ex), (ty, ey) = train_eval_split([X.shape[0], Y.shape[0]], rng=rng,
                                              frac_train=self.train_frac,
                                              groups=[gx_lab, gy_lab])
        witness, info = self.fit_witness(X[tx], Y[ty], rng=rng)
        out_x, out_y = witness(X[ex]), witness(Y[ey])
        gx, gy = self._witness_values(out_x), self._witness_values(out_y)
        inf = mean_diff_inference(gx, gy, rng=rng, n_perm=self.n_perm,
                                  order_x=None if ox is None else ox[ex],
                                  order_y=None if oy is None else oy[ey],
                                  hac_lags=self.hac_lags, tie_break=self.tie_break)
        stat = self._stat_from_diff(inf["diff"])
        if self.inference == "hac":
            pvalue = inf["p_hac"]
        elif self.n_perm > 0:
            pvalue = inf["p_perm"]
        else:
            pvalue = self._asymptotic_pvalue(gx, gy, inf, rng=rng)
        extra: dict[str, Any] = {
            "z": inf["z"], "p_z": inf["p_z"], "z_hac": inf["z_hac"], "p_hac": inf["p_hac"],
            "hac": inf["hac"], "hac_lags": inf["hac_lags"], "se": inf["se"],
            "se_hac": inf["se_hac"], "mean_diff": inf["diff"], "p_perm": inf["p_perm"],
            "p_perm_conservative": inf["p_perm_conservative"], "p_perm_mid": inf["p_perm_mid"],
            "perm_ties": inf["perm_ties"], "inference": self.inference,
            "tie_break": self.tie_break,
            "n_train_x": int(tx.size), "n_train_y": int(ty.size),
            "n_eval_x": int(ex.size), "n_eval_y": int(ey.size),
            "split": split, "n_perm": self.n_perm,
        }
        extra.update(info)
        extra.update(self._extra_stats(out_x, out_y, gx, gy, stat))
        return TestResult(name=self.name, statistic=float(stat), pvalue=float(pvalue),
                          n_x=int(X.shape[0]), n_y=int(Y.shape[0]), extra=extra)

    def _asymptotic_pvalue(self, gx: np.ndarray, gy: np.ndarray, inf: dict[str, Any], *,
                           rng: np.random.Generator) -> float:
        """p-value used when ``n_perm == 0`` (normal approximation by default)."""
        return float(inf["p_z"])

    def _extra_stats(self, out_x: np.ndarray, out_y: np.ndarray, gx: np.ndarray,
                     gy: np.ndarray, stat: float) -> dict[str, Any]:
        return {}


# ----------------------------------------------------------------------------------------------
# Wasserstein critic test
# ----------------------------------------------------------------------------------------------
class CriticTest(LearnedTest):
    """Wasserstein-critic two-sample test (WGAN-GP critic trained alone, held-out permutation).

    Parameters
    ----------
    n_steps : int, default 1000
        Critic optimiser steps.
    batch_size : int, default 128
        Mini-batch size per sample.
    lr : float, default 1e-3
        Adam learning rate. **Deviation from DESIGN (1e-4):** with 1000 steps, lr = 1e-4
        under-trains the critic; against variance-matched t(3) vs N(0, 1) windows (n = 500,
        L = 20) the pooled rejection counts at 5% were 9/22 (1e-4) vs 19/22 (1e-3). Size is
        unaffected (held-out evaluation). Fixed before any E2 run; include lr in E5.
    betas : tuple, default (0.5, 0.9)
        Adam betas (Gulrajani et al. 2017).
    gp_lambda : float, default 10
        Gradient-penalty weight.
    arch, hidden, depth, layer_norm :
        Architecture (:func:`synthval.twosample.nets.build_network`).
    train_frac : float, default 0.5
        Fraction of each sample used to train the critic.
    n_perm : int, default ``config.N_PERM`` (1000)
        Permutations of the evaluation labels (0: use the normal p-value of ``extra["z"]``).
    early_stopping : bool, default False
        Early stopping on a held-out slice of the training half, monitoring the validation
        WGAN-GP objective ``mean f(X_v) - mean f(Y_v) - lambda * GP_v`` (maximised).
    val_frac, eval_every, patience, min_steps, curve_points :
        See :class:`LearnedTest`.
    hac_lags, inference, tie_break, time_blocks :
        See :class:`LearnedTest`. For serially dependent windows use ``inference="hac"`` (the
        permutation p-value assumes i.i.d. windows) together with ``time_blocks`` (e.g. 10) and
        ``order_x`` / ``order_y``.

    Notes
    -----
    ``statistic`` is ``T = mean f(X_ev) - mean f(Y_ev)`` and ``extra["W1_lower"] = T``. ``extra``
    also holds ``z``/``p_z`` (i.i.d. normal), ``z_hac``/``p_hac`` (Driscoll-Kraay / Newey-West
    when ``order_x``/``order_y`` are given, else equal to ``z``/``p_z``), ``p_perm``, the split
    sizes and the sub-sampled training curve ``train_curve = {"step", "wdist", "gp"}`` (chunk
    averages of the mini-batch estimates of ``mean f(X) - mean f(Y)`` and of the gradient
    penalty).
    """

    name = "critic"

    def __init__(self, *, n_steps: int = 1000, batch_size: int = 128, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.5, 0.9), gp_lambda: float = 10.0,
                 arch: Arch = "auto", hidden: int | None = None, depth: int = 3,
                 layer_norm: bool = False, train_frac: float = 0.5, n_perm: int = N_PERM,
                 early_stopping: bool = False, val_frac: float = 0.2, eval_every: int = 50,
                 patience: int = 5, min_steps: int = 0, curve_points: int = 100,
                 hac_lags: HacLags = None,
                 inference: Literal["permutation", "hac"] = "permutation",
                 tie_break: TieBreak = "conservative", time_blocks: int | None = None) -> None:
        super().__init__(n_steps=n_steps, batch_size=batch_size, lr=lr, betas=betas,
                         weight_decay=0.0, arch=arch, hidden=hidden, depth=depth,
                         layer_norm=layer_norm, train_frac=train_frac, n_perm=n_perm,
                         early_stopping=early_stopping, val_frac=val_frac,
                         eval_every=eval_every, patience=patience, min_steps=min_steps,
                         curve_points=curve_points, hac_lags=hac_lags, inference=inference,
                         tie_break=tie_break, time_blocks=time_blocks)
        if gp_lambda < 0:
            raise ValueError("gp_lambda must be >= 0")
        self.gp_lambda = float(gp_lambda)

    @staticmethod
    def _gp_terms(net: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor,
                  eps: torch.Tensor, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor,
                                                                    torch.Tensor]:
        """Return ``(mean f(xb) - mean f(yb), gradient penalty, f(x_hat))`` in one forward pass."""
        shape = (-1,) + (1,) * (xb.dim() - 1)
        x_hat = (eps.view(shape) * xb + (1.0 - eps.view(shape)) * yb).requires_grad_(True)
        b = xb.shape[0]
        out = net(torch.cat([xb, yb, x_hat], dim=0))
        f_x, f_y, f_hat = out[:b], out[b:2 * b], out[2 * b:]
        grad = torch.autograd.grad(f_hat.sum(), x_hat, create_graph=create_graph)[0]
        gp = ((grad.reshape(b, -1).norm(2, dim=1) - 1.0) ** 2).mean()
        return f_x.mean() - f_y.mean(), gp, f_hat

    def _train(self, net: torch.nn.Module, Xt: torch.Tensor, Yt: torch.Tensor,
               Xv: torch.Tensor | None, Yv: torch.Tensor | None) -> dict[str, Any]:
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, betas=self.betas)
        nx, ny, b = Xt.shape[0], Yt.shape[0], self.batch_size
        chunk = max(1, self.n_steps // max(1, self.curve_points * 2))
        curve: dict[str, list[float]] = {"step": [], "wdist": [], "gp": []}
        val_curve: dict[str, list[float]] = {"step": [], "objective": []}
        acc_w = acc_gp = 0.0
        acc_n = 0
        es = self._early_stop_loop_state()
        if Xv is not None and Yv is not None:
            # Fixed validation pairs and interpolation weights (paired random indices).
            m_v = max(Xv.shape[0], Yv.shape[0])
            vix = torch.randint(0, Xv.shape[0], (m_v,))
            viy = torch.randint(0, Yv.shape[0], (m_v,))
            veps = torch.rand(m_v)
        step = 0
        net.train()
        for step in range(1, self.n_steps + 1):
            # Pair random indices: sizes may differ, each interpolate uses one real and one synthetic point.
            ix = torch.randint(0, nx, (b,))
            iy = torch.randint(0, ny, (b,))
            eps = torch.rand(b)
            wd, gp, _ = self._gp_terms(net, Xt[ix], Yt[iy], eps, create_graph=True)
            loss = -wd + self.gp_lambda * gp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            acc_w += float(wd.detach())
            acc_gp += float(gp.detach())
            acc_n += 1
            if step % chunk == 0 or step == self.n_steps:
                curve["step"].append(step)
                curve["wdist"].append(acc_w / acc_n)
                curve["gp"].append(acc_gp / acc_n)
                acc_w = acc_gp = 0.0
                acc_n = 0
            if Xv is not None and Yv is not None and (step % self.eval_every == 0):
                wd_v, gp_v, _ = self._gp_terms(net, Xv[vix], Yv[viy], veps, create_graph=False)
                obj = float(wd_v.detach()) - self.gp_lambda * float(gp_v.detach())
                val_curve["step"].append(step)
                val_curve["objective"].append(obj)
                if step < self.min_steps:
                    continue
                if obj > es["best"]:
                    es.update(best=obj, bad=0, best_step=step,
                              best_state={k: v.detach().clone() for k, v in net.state_dict().items()})
                else:
                    es["bad"] += 1
                    if es["bad"] >= self.patience:
                        break
        info: dict[str, Any] = {"train_curve": curve, "steps_run": step}
        if Xv is not None:
            if es["best_state"] is not None:
                net.load_state_dict(es["best_state"])
            info["val_curve"] = val_curve
            info["best_step"] = es["best_step"]
        net.eval()
        return info

    def fit_critic(self, X_tr: np.ndarray, Y_tr: np.ndarray, *,
                   rng: np.random.Generator) -> tuple[Witness, dict[str, Any]]:
        """Train a critic on ``(X_tr, Y_tr)`` and return ``(f, info)``; ``f`` maps raw inputs to ``(n,)``."""
        return self.fit_witness(X_tr, Y_tr, rng=rng)

    def _extra_stats(self, out_x: np.ndarray, out_y: np.ndarray, gx: np.ndarray,
                     gy: np.ndarray, stat: float) -> dict[str, Any]:
        return {"W1_lower": float(stat), "gp_lambda": self.gp_lambda}
