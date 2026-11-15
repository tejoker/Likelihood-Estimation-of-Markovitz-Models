"""Gaussian-kernel MMD two-sample test (Gretton et al. 2012).

Statistic: the unbiased estimate

    MMD_u^2 = sum_{i != j} k(x_i, x_j) / (n (n - 1)) + sum_{i != j} k(y_i, y_j) / (m (m - 1))
              - 2 sum_{i, j} k(x_i, y_j) / (n m)

with the Gaussian kernel ``k(a, b) = exp(-||a - b||^2 / (2 h^2))``. The bandwidth ``h`` is the
median heuristic (median pairwise Euclidean distance on the pooled sample); optionally the
kernel is the sum over ``h * {0.25, 0.5, 1, 2, 4}``. Both choices depend on the pooled sample
only, so they are invariant to the labels.

Two kinds of p-value (``inference``):

* ``"permutation"`` (default): the pooled kernel matrix ``K`` is computed once; for a block of
  label indicator vectors ``A`` (one column per permutation), ``K @ A`` gives all within- and
  between-sample kernel sums at once. **Exact only if the windows are i.i.d.** (exchangeable
  under H0). With serially dependent windows (consecutive windows of one persistent-volatility
  series) near-in-time windows are more similar than independent ones, ``MMD_u^2`` has a
  positive mean under H0 that the permutation null does not reproduce, and the test
  over-rejects (reviewer check: 25% at nominal 5% for GARCH(1,1)-t windows, a + b = 0.98).
* ``"wild"``: the wild bootstrap for degenerate kernel statistics of Chwialkowski, Sejdinovic &
  Gretton (2014), robust to serial dependence (asymptotic, not exact). It is applied to the
  V-statistic ``MMD_b^2 = || mean phi(X) - mean phi(Y) ||^2`` (whose null mean includes the
  long-run variance of the mean embeddings), with bootstrap replicates
  ``|| sum_i w~_i phi(x_i) / n_x - sum_j w~_j phi(y_j) / n_y ||^2 = a' K a``, where ``w`` is a
  stationary Gaussian AR(1) multiplier process with ``Cov(w_s, w_t) = exp(-|s - t| / l)`` on
  the time grid, ``w~`` is ``w`` centred within each sample, and ``a = (w~_x / n_x, -w~_y /
  n_y)``. Observations with the same time index share one multiplier, in both samples
  (Driscoll-Kraay-type), so dependence between windows of the same date (several tickers, or
  real-vs-real placebo halves) is reproduced too. The block length ``l`` defaults to an
  Andrews (1991)-type AR(1) plug-in ``l = (alpha(1) T)^(1/3)`` (the optimal rate constant for
  the kernel ``exp(-|x|)``, ``int k^2 = 1``) computed on the date-aggregated scores of the
  leading kernel principal components (:func:`wild_block_length`); a fixed ``wild_block`` can
  be passed instead. Chwialkowski et al. leave ``l`` as a tuning parameter, so the plug-in is a
  heuristic of ours.

Samples larger than ``max_n`` are subsampled (without replacement) for memory: ``K`` takes
``8 (n_x + n_y)^2`` bytes (288 MB at the default ``max_n = 3000``).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from scipy.signal import lfilter
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, LinearOperator, eigsh

from .base import TestResult, TwoSampleTest, permutation_pvalue
from .critic import andrews_alpha, check_samples, time_grid

logger = logging.getLogger(__name__)

__all__ = [
    "MMDTest",
    "median_heuristic",
    "gaussian_kernel_matrix",
    "gaussian_kernel_cross",
    "mmd2_unbiased",
    "mmd2_biased",
    "mmd2_permutations",
    "mmd2_wild_bootstrap",
    "wild_block_length",
    "ar1_multipliers",
    "subsample",
    "MULTI_BANDWIDTH_FACTORS",
]

MULTI_BANDWIDTH_FACTORS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def subsample_idx(n: int, max_n: int | None, *, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices of a random subset of at most ``max_n`` of ``n`` rows (all rows if small)."""
    if max_n is None or n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def subsample(Z: np.ndarray, max_n: int | None, *, rng: np.random.Generator) -> np.ndarray:
    """Random subset of at most ``max_n`` rows (all rows, in order, if already small enough)."""
    if max_n is None or Z.shape[0] <= max_n:
        return Z
    return Z[subsample_idx(Z.shape[0], max_n, rng=rng)]


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of ``A`` and ``B`` (clipped at 0)."""
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def median_heuristic(Z: np.ndarray, *, rng: np.random.Generator, max_points: int = 2000) -> float:
    """Median of the pairwise Euclidean distances between distinct rows of ``Z``.

    Computed on a random subset of at most ``max_points`` rows (the median of ~2e6 distances is
    an accurate estimate and costs O(max_points^2)). Returns 1.0 if all points coincide.

    Parameters
    ----------
    Z : array (n, d)
        Pooled sample.
    rng : numpy.random.Generator
        Random generator (subset selection).
    max_points : int, default 2000
        Maximum number of rows used.

    Returns
    -------
    float
    """
    Z = np.asarray(Z, dtype=np.float64)
    Z = subsample(Z.reshape(Z.shape[0], -1), max_points, rng=rng)
    d2 = _sqdist(Z, Z)
    iu = np.triu_indices(Z.shape[0], k=1)
    med = float(np.sqrt(np.median(d2[iu])))
    return med if med > 0 and np.isfinite(med) else 1.0


def _bandwidths(h: float, multi: bool) -> tuple[float, ...]:
    return tuple(h * f for f in MULTI_BANDWIDTH_FACTORS) if multi else (h,)


def gaussian_kernel_cross(A: np.ndarray, B: np.ndarray, bandwidths: Sequence[float], *,
                          block: int = 512) -> np.ndarray:
    """Kernel matrix ``sum_h exp(-||a_i - b_j||^2 / (2 h^2))`` between rows of ``A`` and ``B``.

    Built in row blocks so the peak extra memory is ``O(block * len(B))``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    K = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
    bb = (B * B).sum(1)
    for i in range(0, A.shape[0], block):
        a = A[i:i + block]
        d2 = (a * a).sum(1)[:, None] + bb[None, :] - 2.0 * (a @ B.T)
        np.maximum(d2, 0.0, out=d2)
        kb = np.zeros_like(d2)
        for h in bandwidths:
            kb += np.exp(-d2 / (2.0 * h * h))
        K[i:i + block] = kb
    return K


def gaussian_kernel_matrix(Z: np.ndarray, bandwidths: Sequence[float], *,
                           block: int = 512) -> np.ndarray:
    """Symmetric kernel matrix on the rows of ``Z`` (see :func:`gaussian_kernel_cross`)."""
    K = gaussian_kernel_cross(Z, Z, bandwidths, block=block)
    np.fill_diagonal(K, float(len(bandwidths)))  # exact diagonal, k(z, z) = number of kernels
    return K


def mmd2_unbiased(K: np.ndarray, n_x: int) -> float:
    """Unbiased MMD^2 from the pooled kernel matrix (first ``n_x`` rows = sample X)."""
    m = K.shape[0]
    n_y = m - n_x
    Kxx = K[:n_x, :n_x]
    Kyy = K[n_x:, n_x:]
    Kxy = K[:n_x, n_x:]
    sxx = (Kxx.sum() - np.trace(Kxx)) / (n_x * (n_x - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (n_y * (n_y - 1))
    return float(sxx + syy - 2.0 * Kxy.sum() / (n_x * n_y))


def mmd2_permutations(K: np.ndarray, n_x: int, n_perm: int, *, rng: np.random.Generator,
                      block: int = 100) -> np.ndarray:
    """Unbiased MMD^2 under ``n_perm`` random relabellings of the pooled sample.

    For a block of 0/1 indicator columns ``A`` (``n_x`` ones each), ``KA = K @ A`` gives
    ``s_xx = colsum(A * KA)``, ``s_xy = colsum(KA) - s_xx`` and ``s_yy = 1'K1 - 2 s_xy - s_xx``.
    Memory ``O(m * block)`` on top of ``K``.

    Parameters
    ----------
    K : array (m, m)
        Pooled kernel matrix.
    n_x : int
        Size of the first sample.
    n_perm : int
        Number of permutations.
    rng : numpy.random.Generator
        Random generator.
    block : int, default 100
        Permutations per matrix product.

    Returns
    -------
    numpy.ndarray (n_perm,)
    """
    m = K.shape[0]
    n_y = m - n_x
    diag = np.diag(K).copy()
    rows = K.sum(1)
    total = rows.sum()
    base = np.zeros(m, dtype=np.float64)
    base[:n_x] = 1.0
    out = np.empty(n_perm, dtype=np.float64)
    for start in range(0, n_perm, block):
        b = min(block, n_perm - start)
        A = rng.permuted(np.tile(base, (b, 1)), axis=1).T  # (m, b)
        KA = K @ A
        sxx = (A * KA).sum(0)
        sxy = rows @ A - sxx
        syy = total - 2.0 * sxy - sxx
        dx = diag @ A
        dy = diag.sum() - dx
        out[start:start + b] = ((sxx - dx) / (n_x * (n_x - 1)) + (syy - dy) / (n_y * (n_y - 1))
                                - 2.0 * sxy / (n_x * n_y))
    return out


def mmd2_biased(K: np.ndarray, n_x: int) -> float:
    """V-statistic ``|| mean phi(X) - mean phi(Y) ||^2`` from the pooled kernel matrix."""
    m = K.shape[0]
    a = np.empty(m)
    a[:n_x] = 1.0 / n_x
    a[n_x:] = -1.0 / (m - n_x)
    return float(a @ K @ a)


def ar1_multipliers(n_grid: int, block: float, size: int, *,
                    rng: np.random.Generator) -> np.ndarray:
    """Stationary Gaussian AR(1) multiplier processes on a grid, ``Cov(w_s, w_t) = exp(-|s-t|/l)``.

    ``w_0 ~ N(0, 1)``, ``w_t = phi w_{t-1} + sqrt(1 - phi^2) e_t`` with ``phi = exp(-1 / l)``
    (Chwialkowski, Sejdinovic & Gretton 2014). ``block <= 0`` gives i.i.d. N(0, 1) multipliers.

    Returns
    -------
    numpy.ndarray (n_grid, size)
    """
    e = rng.standard_normal((n_grid, size))
    if block <= 0.0 or n_grid < 2:
        return e
    phi = float(np.exp(-1.0 / block))
    if phi == 0.0:
        return e
    w = np.empty_like(e)
    w[0] = e[0]
    w[1:] = lfilter([np.sqrt(1.0 - phi * phi)], [1.0, -phi], e[1:], axis=0,
                    zi=(phi * e[0])[None, :])[0]
    return w


def mmd2_wild_bootstrap(K: np.ndarray, n_x: int, pos: np.ndarray, n_grid: int, block: float,
                        n_boot: int, *, rng: np.random.Generator,
                        chunk: int = 100) -> np.ndarray:
    """Wild-bootstrap replicates of the MMD V-statistic under serial dependence.

    Replicate ``b``: ``a_b' K a_b`` with ``a_b = (w~_x / n_x, -w~_y / n_y)``, where ``w`` is an
    AR(1) multiplier process on the time grid (:func:`ar1_multipliers`), evaluated at each
    observation's grid position ``pos`` (shared by all observations at the same position, in
    both samples), and centred within each sample (Chwialkowski, Sejdinovic & Gretton 2014,
    centred version).

    Parameters
    ----------
    K : array (m, m)
        Pooled kernel matrix (first ``n_x`` rows = sample X).
    n_x : int
        Size of the first sample.
    pos : int array (m,)
        Grid position of each pooled observation (see :func:`synthval.twosample.critic.time_grid`).
    n_grid : int
        Grid length.
    block : float
        Multiplier correlation length ``l``.
    n_boot : int
        Number of replicates.
    rng : numpy.random.Generator
        Random generator.
    chunk : int, default 100
        Replicates per matrix product.

    Returns
    -------
    numpy.ndarray (n_boot,)
    """
    m = K.shape[0]
    n_y = m - n_x
    pos = np.asarray(pos, dtype=np.int64)
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        W = ar1_multipliers(n_grid, block, b, rng=rng)[pos]  # (m, b)
        Ax = W[:n_x] - W[:n_x].mean(0)
        Ay = W[n_x:] - W[n_x:].mean(0)
        A = np.concatenate([Ax / n_x, -Ay / n_y], axis=0)
        out[start:start + b] = (A * (K @ A)).sum(0)
    return out


def wild_block_length(K: np.ndarray, n_x: int, pos: np.ndarray, n_grid: int, *,
                      rng: np.random.Generator, n_pc: int = 5, max_points: int = 3000) -> float:
    """Data-driven multiplier correlation length for :func:`mmd2_wild_bootstrap`.

    The leading ``n_pc`` principal components of the centred pooled kernel matrix give scalar
    score series (coordinates of ``phi(z_i)`` in the main directions of the null distribution
    of the MMD); each is centred within its sample, summed within each date, and the Andrews
    (1991) AR(1) plug-in ``alpha(1)`` is computed on these series (adjacent observed dates
    only, see :func:`synthval.twosample.critic.andrews_alpha`). The block length is
    ``l = (alpha(1) T)^(1/3)``, ``T`` the grid length: the MSE-optimal rate for the implied
    kernel ``exp(-|x|)`` (``k_1 = 1``, ``int k^2 = 1``). Returns 0 (i.i.d. multipliers) when
    no dependence is detected.

    Parameters
    ----------
    K : array (m, m)
        Pooled kernel matrix.
    n_x : int
        Size of the first sample.
    pos : int array (m,)
        Grid positions.
    n_grid : int
        Grid length.
    rng : numpy.random.Generator
        Random generator (Lanczos start vector, subset selection).
    n_pc : int, default 5
        Number of kernel principal components.
    max_points : int, default 3000
        The eigen-decomposition uses a random subset of at most this many points.

    Returns
    -------
    float
        Block length ``l >= 0`` in grid units.
    """
    m = K.shape[0]
    idx = subsample_idx(m, max_points, rng=rng)
    Ks = K[np.ix_(idx, idx)] if idx.size < m else K
    ms = Ks.shape[0]
    k = int(min(n_pc, ms - 2))
    if k < 1:
        return 0.0

    def matvec(v: np.ndarray) -> np.ndarray:
        v = np.ravel(v)
        v = v - v.mean()
        u = Ks @ v
        return u - u.mean()

    op = LinearOperator((ms, ms), matvec=matvec, dtype=np.float64)
    try:
        vals, vecs = eigsh(op, k=k, which="LA", v0=rng.standard_normal(ms), tol=1e-3,
                           maxiter=50 * ms)
    except (ArpackNoConvergence, ArpackError) as err:  # pragma: no cover - rare
        logger.warning("kernel PCA for the wild-bootstrap block length failed (%s); "
                       "using i.i.d. multipliers", err)
        return 0.0
    keep = vals > 1e-12 * max(1.0, float(vals.max()))
    if not np.any(keep):
        return 0.0
    vals, vecs = vals[keep], vecs[:, keep]
    # Scores of all m points on the principal axes (Nystrom extension when subsampled).
    if idx.size < m:
        Kc = K[:, idx]  # one (m, max_points) copy, centred in place
        Kc -= Kc.mean(0, keepdims=True)
        Kc -= Kc.mean(1, keepdims=True)
        scores = Kc @ vecs / np.sqrt(vals)
        del Kc
    else:
        scores = vecs * np.sqrt(vals)
    in_x = np.zeros(m, dtype=bool)
    in_x[:n_x] = True
    scores[in_x] -= scores[in_x].mean(0)
    scores[~in_x] -= scores[~in_x].mean(0)
    S = np.stack([np.bincount(pos, weights=scores[:, a], minlength=n_grid)
                  for a in range(scores.shape[1])], axis=1)
    mask = np.bincount(pos, minlength=n_grid) > 0
    res = andrews_alpha(S, mask)
    if res is None:
        # Sparse irregular grid: consecutive observed dates as the time unit, mapped back to
        # grid units with the mean spacing.
        obs = np.flatnonzero(mask)
        res = andrews_alpha(S[obs])
        if res is None:
            return 0.0
        spacing = (obs[-1] - obs[0]) / max(obs.size - 1, 1)
        return float((res[0] * obs.size) ** (1.0 / 3.0) * spacing)
    return float((res[0] * n_grid) ** (1.0 / 3.0))


def standardize_pooled(*samples: np.ndarray, eps: float = 1e-12) -> list[np.ndarray]:
    """Standardise each column with the mean / std of the pooled samples (label invariant)."""
    Z = np.concatenate(samples, axis=0)
    mu = Z.mean(0)
    sd = Z.std(0)
    sd = np.where(sd < eps, 1.0, sd)
    return [(s - mu) / sd for s in samples]


class MMDTest(TwoSampleTest):
    """Gaussian-kernel MMD^2 test: permutation p-value (i.i.d. windows) or wild bootstrap.

    Parameters
    ----------
    n_perm : int, default 500
        Number of permutations (``inference="permutation"``) or wild-bootstrap replicates
        (``inference="wild"``).
    max_n : int or None, default 3000
        Each sample is subsampled to at most ``max_n`` rows (memory ``8 (2 max_n)^2`` bytes).
    multi_bandwidth : bool, default False
        Sum of Gaussian kernels with bandwidths ``median * {0.25, 0.5, 1, 2, 4}``.
    bandwidth : float, optional
        Fixed base bandwidth instead of the median heuristic.
    standardize : bool, default False
        Standardise each feature with pooled (label-invariant) statistics first.
    perm_block : int, default 100
        Permutations (or bootstrap replicates) per vectorised block.
    inference : {"permutation", "wild"}, default "permutation"
        ``"permutation"``: exact for i.i.d. windows, over-sized under serial dependence.
        ``"wild"``: wild bootstrap of the V-statistic (Chwialkowski, Sejdinovic & Gretton 2014),
        asymptotically valid for serially dependent windows; uses ``order_x`` / ``order_y``
        (time indices; if omitted, rows are taken to be in time order).
    wild_block : float, optional
        Multiplier correlation length ``l`` in grid units (default: data-driven,
        :func:`wild_block_length`; ``0`` gives i.i.d. multipliers).

    Notes
    -----
    Inputs of shape ``(n, L)`` or ``(n, N, L)`` are flattened to vectors. ``statistic`` is always
    the unbiased ``MMD_u^2``. ``extra`` holds the bandwidth(s), the subsample sizes, the mean/std
    of the null replicates, and with ``inference="wild"`` the V-statistic ``mmd2_v`` the
    bootstrap p-value refers to and the block length ``wild_block``.
    """

    name = "mmd"

    def __init__(self, *, n_perm: int = 500, max_n: int | None = 3000,
                 multi_bandwidth: bool = False, bandwidth: float | None = None,
                 standardize: bool = False, perm_block: int = 100,
                 inference: Literal["permutation", "wild"] = "permutation",
                 wild_block: float | None = None) -> None:
        if n_perm < 1:
            raise ValueError("n_perm must be >= 1")
        if inference not in ("permutation", "wild"):
            raise ValueError("inference must be 'permutation' or 'wild'")
        if wild_block is not None and wild_block < 0:
            raise ValueError("wild_block must be >= 0")
        self.n_perm = int(n_perm)
        self.max_n = max_n
        self.multi_bandwidth = bool(multi_bandwidth)
        self.bandwidth = bandwidth
        self.standardize = bool(standardize)
        self.perm_block = int(perm_block)
        self.inference = inference
        self.wild_block = wild_block

    def prepare(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map the inputs to the feature vectors the kernel acts on (flattening by default)."""
        X, Y = check_samples(X, Y)
        return X.reshape(X.shape[0], -1), Y.reshape(Y.shape[0], -1)

    def bandwidths_for(self, Z: np.ndarray, *, rng: np.random.Generator) -> tuple[float, ...]:
        """Bandwidth(s) for the pooled feature sample ``Z``."""
        h = float(self.bandwidth) if self.bandwidth is not None else median_heuristic(Z, rng=rng)
        return _bandwidths(h, self.multi_bandwidth)

    def test_features(self, Fx: np.ndarray, Fy: np.ndarray, *, rng: np.random.Generator,
                      order_x: np.ndarray | None = None,
                      order_y: np.ndarray | None = None) -> tuple[float, float, dict[str, Any]]:
        """MMD test on feature matrices; returns ``(MMD_u^2, p-value, extra)``.

        ``order_x`` / ``order_y`` (time indices aligned with the rows) are used only with
        ``inference="wild"``; by default the row numbers are used.
        """
        ix = subsample_idx(Fx.shape[0], self.max_n, rng=rng)
        iy = subsample_idx(Fy.shape[0], self.max_n, rng=rng)
        Fx, Fy = Fx[ix], Fy[iy]
        if self.standardize:
            Fx, Fy = standardize_pooled(Fx, Fy)
        Z = np.concatenate([Fx, Fy], axis=0)
        n_x = Fx.shape[0]
        bws = self.bandwidths_for(Z, rng=rng)
        K = gaussian_kernel_matrix(Z, bws)
        stat = mmd2_unbiased(K, n_x)
        extra: dict[str, Any] = {"bandwidths": list(bws), "n_used_x": int(n_x),
                                 "n_used_y": int(Fy.shape[0]), "n_perm": self.n_perm,
                                 "multi_bandwidth": self.multi_bandwidth,
                                 "inference": self.inference}
        if self.inference == "permutation":
            t_null = mmd2_permutations(K, n_x, self.n_perm, rng=rng, block=self.perm_block)
            p = permutation_pvalue(stat - 1e-12 * max(1.0, abs(stat)), t_null)
        else:
            # Time index of the retained rows (row numbers when no order is given).
            ox = ix if order_x is None else np.asarray(order_x).ravel()[ix]
            oy = iy if order_y is None else np.asarray(order_y).ravel()[iy]
            pos, n_grid = time_grid(np.concatenate([ox, oy]))
            if self.wild_block is None:
                block = wild_block_length(K, n_x, pos, n_grid, rng=rng)
            else:
                block = float(self.wild_block)
            v_stat = mmd2_biased(K, n_x)
            t_null = mmd2_wild_bootstrap(K, n_x, pos, n_grid, block, self.n_perm, rng=rng,
                                         chunk=self.perm_block)
            p = permutation_pvalue(v_stat - 1e-12 * max(1.0, abs(v_stat)), t_null)
            extra.update(mmd2_v=v_stat, wild_block=block, n_grid=int(n_grid))
        extra.update(perm_mean=float(t_null.mean()), perm_std=float(t_null.std()))
        return stat, p, extra

    def __call__(self, X: np.ndarray, Y: np.ndarray, *, rng: np.random.Generator,
                 groups_x: np.ndarray | None = None, groups_y: np.ndarray | None = None,
                 order_x: np.ndarray | None = None,
                 order_y: np.ndarray | None = None) -> TestResult:
        """Run the test (``groups_*`` are ignored; ``order_*`` are used by the wild bootstrap)."""
        Fx, Fy = self.prepare(X, Y)
        ox = _check_order(order_x, Fx.shape[0], "order_x")
        oy = _check_order(order_y, Fy.shape[0], "order_y")
        stat, p, extra = self.test_features(Fx, Fy, rng=rng, order_x=ox, order_y=oy)
        return TestResult(name=self.name, statistic=stat, pvalue=p, n_x=int(Fx.shape[0]),
                          n_y=int(Fy.shape[0]), extra=extra)


def _check_order(order: np.ndarray | None, n: int, what: str) -> np.ndarray | None:
    if order is None:
        return None
    order = np.asarray(order).ravel()
    if order.shape[0] != n:
        raise ValueError(f"{what} has length {order.shape[0]}, expected {n}")
    return order
