"""Truncated path signatures (Chen's identity, vectorised numpy) and the signature-MMD test.

Signature of a path ``X: [0, 1] -> R^d`` truncated at depth ``M``: the collection of iterated
integrals ``S^{i_1...i_k} = int_{0 < t_1 < ... < t_k < 1} dX^{i_1}_{t_1} ... dX^{i_k}_{t_k}`` for
``1 <= k <= M``. Level ``k`` is stored flattened in row-major order of ``(i_1, ..., i_k)`` (first
index most significant), and the levels are concatenated, so the feature dimension is
``d + d^2 + ... + d^M`` (the constant level 0, equal to 1, is omitted).

For a linear segment with increment ``D`` the signature is the tensor exponential
``(D, D^{(x)2}/2!, ..., D^{(x)M}/M!)``; for a piecewise-linear path the signature is the
truncated tensor product of the segment signatures (Chen's identity):
``S(X * Y)_k = S(X)_k + S(Y)_k + sum_{i=1}^{k-1} S(X)_i (x) S(Y)_{k-i}``.

A window of daily returns ``(r_1, ..., r_L)`` is turned into the time-augmented path through the
points ``(k / L, r_1 + ... + r_k)``, ``k = 0..L`` (optionally with the lead-lag transform of the
cumulative-return coordinate, whose signature sees the quadratic variation). Chevyrev &
Oberhauser (2022).

The signature-MMD test inherits the inference options of :class:`~synthval.twosample.mmd.MMDTest`:
the default permutation p-value is exact **only for i.i.d. windows** (exchangeable under H0) and
over-rejects when windows are serially dependent (reviewer check: 14% at nominal 5% for
consecutive GARCH(1,1)-t windows, a + b = 0.98); ``inference="wild"`` uses the
dependence-robust wild bootstrap of Chwialkowski, Sejdinovic & Gretton (2014) instead.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .base import TestResult, TwoSampleTest
from .critic import check_samples
from .mmd import MMDTest, _check_order, standardize_pooled

__all__ = [
    "sig_dim",
    "signature",
    "signature_product",
    "segment_signature",
    "sig_levels",
    "windows_to_paths",
    "lead_lag",
    "signature_features",
    "SigMMDTest",
]


def sig_dim(d: int, depth: int) -> int:
    """Number of signature features ``d + d^2 + ... + d^depth``."""
    return int(sum(d**k for k in range(1, depth + 1)))


def _tensor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched tensor product of flattened tensors: (n, p) x (n, q) -> (n, p q)."""
    return (a[:, :, None] * b[:, None, :]).reshape(a.shape[0], -1)


def _exp_levels(delta: np.ndarray, depth: int) -> list[np.ndarray]:
    """Levels 1..depth of the tensor exponential of increments ``delta`` (n, d)."""
    levels = [delta]
    for k in range(2, depth + 1):
        levels.append(_tensor(levels[-1], delta) / k)
    return levels


def _chen(A: list[np.ndarray], B: list[np.ndarray], depth: int) -> list[np.ndarray]:
    """Truncated tensor product of two signatures given as level lists (levels 1..depth)."""
    out = []
    for k in range(1, depth + 1):
        c = A[k - 1] + B[k - 1]
        for i in range(1, k):
            c = c + _tensor(A[i - 1], B[k - i - 1])
        out.append(c)
    return out


def sig_levels(s: np.ndarray, d: int, depth: int) -> list[np.ndarray]:
    """Split flattened signatures ``(n, sig_dim)`` (or ``(sig_dim,)``) into level arrays.

    Level ``k`` is returned with shape ``(n, d, ..., d)`` (``k`` trailing axes).
    """
    s = np.asarray(s, dtype=np.float64)
    single = s.ndim == 1
    if single:
        s = s[None]
    if s.shape[1] != sig_dim(d, depth):
        raise ValueError("signature dimension does not match (d, depth)")
    out, start = [], 0
    for k in range(1, depth + 1):
        size = d**k
        lev = s[:, start:start + size].reshape((s.shape[0],) + (d,) * k)
        out.append(lev[0] if single else lev)
        start += size
    return out


def _split_flat(s: np.ndarray, d: int, depth: int) -> list[np.ndarray]:
    out, start = [], 0
    for k in range(1, depth + 1):
        out.append(s[:, start:start + d**k])
        start += d**k
    return out


def segment_signature(increments: np.ndarray, depth: int) -> np.ndarray:
    """Signature of straight-line segments with the given increments ``(n, d)`` (or ``(d,)``)."""
    inc = np.asarray(increments, dtype=np.float64)
    single = inc.ndim == 1
    if single:
        inc = inc[None]
    out = np.concatenate(_exp_levels(inc, depth), axis=1)
    return out[0] if single else out


def signature_product(s1: np.ndarray, s2: np.ndarray, d: int, depth: int) -> np.ndarray:
    """Chen product of two (batches of) flattened truncated signatures.

    ``signature(concat(X, Y)) == signature_product(signature(X), signature(Y))`` when the
    end point of ``X`` is the start point of ``Y``.
    """
    a = np.asarray(s1, dtype=np.float64)
    b = np.asarray(s2, dtype=np.float64)
    single = a.ndim == 1
    if single:
        a, b = a[None], b[None]
    out = np.concatenate(_chen(_split_flat(a, d, depth), _split_flat(b, d, depth), depth), axis=1)
    return out[0] if single else out


def signature(paths: np.ndarray, depth: int = 4) -> np.ndarray:
    """Truncated signature of piecewise-linear paths through the given points.

    Parameters
    ----------
    paths : array (n, P, d) or (P, d)
        ``P`` points in ``R^d`` per path; the path interpolates linearly between them.
    depth : int, default 4
        Truncation depth ``M >= 1``.

    Returns
    -------
    numpy.ndarray (n, sig_dim(d, depth)) or (sig_dim(d, depth),)
        Levels 1..M, each flattened row-major, concatenated.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    X = np.asarray(paths, dtype=np.float64)
    single = X.ndim == 2
    if single:
        X = X[None]
    if X.ndim != 3:
        raise ValueError("paths must have shape (n, P, d) or (P, d)")
    n, P, d = X.shape
    if P < 2:
        out = np.zeros((n, sig_dim(d, depth)))
        return out[0] if single else out
    inc = np.diff(X, axis=1)
    levels = _exp_levels(inc[:, 0], depth)
    for s in range(1, P - 1):
        levels = _chen(levels, _exp_levels(inc[:, s], depth), depth)
    out = np.concatenate(levels, axis=1)
    return out[0] if single else out


def lead_lag(X: np.ndarray) -> np.ndarray:
    """Lead-lag transform of point sequences ``(n, P, c)`` -> ``(n, 2P - 1, 2c)``.

    Points ``(X_0, X_0), (X_1, X_0), (X_1, X_1), (X_2, X_1), ...``: the lead coordinates move
    first, then the lag coordinates catch up. Then
    ``S^{lead, lag} - S^{lag, lead} = sum_k (X_k - X_{k-1})^2`` (quadratic variation).
    """
    X = np.asarray(X, dtype=np.float64)
    P = X.shape[1]
    j = np.arange(2 * P - 1)
    lead = X[:, (j + 1) // 2]
    lag = X[:, j // 2]
    return np.concatenate([lead, lag], axis=2)


def windows_to_paths(windows: np.ndarray, *, time_aug: bool = True,
                     lead_lag_transform: bool = False) -> np.ndarray:
    """Return windows -> piecewise-linear path points.

    Parameters
    ----------
    windows : array (n, L) or (n, N, L)
        Daily log-returns per window (N assets for multivariate windows).
    time_aug : bool, default True
        Prepend the time coordinate ``t in [0, 1]`` (``k / L``, or ``j / (2L)`` after lead-lag).
    lead_lag_transform : bool, default False
        Apply the lead-lag transform to the cumulative-return coordinates.

    Returns
    -------
    numpy.ndarray (n, P, d)
        ``P = L + 1`` points (``2L + 1`` with lead-lag), ``d = N (x2 with lead-lag) (+1 with time)``.
    """
    W = np.asarray(windows, dtype=np.float64)
    if W.ndim == 2:
        W = W[:, None, :]
    if W.ndim != 3:
        raise ValueError("windows must have shape (n, L) or (n, N, L)")
    n, N, L = W.shape
    cum = np.concatenate([np.zeros((n, N, 1)), np.cumsum(W, axis=2)], axis=2)  # (n, N, L+1)
    X = np.transpose(cum, (0, 2, 1))  # (n, L+1, N)
    if lead_lag_transform:
        X = lead_lag(X)
    if time_aug:
        P = X.shape[1]
        t = np.broadcast_to(np.linspace(0.0, 1.0, P)[None, :, None], (n, P, 1))
        X = np.concatenate([t, X], axis=2)
    return X


def signature_features(windows: np.ndarray, depth: int = 4, *, time_aug: bool = True,
                       lead_lag_transform: bool = False, max_features: int = 50_000) -> np.ndarray:
    """Signature features of return windows (see :func:`windows_to_paths`, :func:`signature`).

    Raises
    ------
    ValueError
        If the feature dimension would exceed ``max_features`` (reduce ``depth``).
    """
    paths = windows_to_paths(windows, time_aug=time_aug, lead_lag_transform=lead_lag_transform)
    D = sig_dim(paths.shape[2], depth)
    if D > max_features:
        raise ValueError(f"signature dimension {D} exceeds max_features={max_features}; "
                         "reduce depth")
    return signature(paths, depth)


class SigMMDTest(TwoSampleTest):
    """Signature-MMD test: Gaussian-kernel MMD on standardised truncated signatures.

    Parameters
    ----------
    depth : int, default 4
        Signature truncation depth.
    time_aug : bool, default True
        Time-augment the cumulative-return path.
    lead_lag : bool, default False
        Lead-lag transform of the cumulative-return coordinates.
    n_perm : int, default 500
        Number of permutations.
    max_n : int or None, default 3000
        Per-sample subsample size for the kernel matrix.
    multi_bandwidth : bool, default False
        Sum of five Gaussian kernels (median x {0.25, 0.5, 1, 2, 4}).
    standardize : bool, default True
        Standardise each signature feature with pooled (label-invariant) mean/std.
    inference : {"permutation", "wild"}, default "permutation"
        Permutation p-value (exact for i.i.d. windows) or dependence-robust wild bootstrap
        (uses ``order_x`` / ``order_y``), see :class:`~synthval.twosample.mmd.MMDTest`.
    wild_block : float, optional
        Wild-bootstrap multiplier correlation length (default: data-driven).

    Notes
    -----
    Inputs are return windows ``(n, L)`` or ``(n, N, L)``; signatures are computed before
    subsampling to ``max_n``.
    """

    name = "sigmmd"

    def __init__(self, *, depth: int = 4, time_aug: bool = True, lead_lag: bool = False,
                 n_perm: int = 500, max_n: int | None = 3000, multi_bandwidth: bool = False,
                 standardize: bool = True,
                 inference: Literal["permutation", "wild"] = "permutation",
                 wild_block: float | None = None) -> None:
        self.depth = int(depth)
        self.time_aug = bool(time_aug)
        self.lead_lag = bool(lead_lag)
        self.standardize = bool(standardize)
        self.mmd = MMDTest(n_perm=n_perm, max_n=max_n, multi_bandwidth=multi_bandwidth,
                           standardize=False, inference=inference, wild_block=wild_block)

    @property
    def n_perm(self) -> int:
        return self.mmd.n_perm

    @property
    def max_n(self) -> int | None:
        return self.mmd.max_n

    @property
    def inference(self) -> str:
        return self.mmd.inference

    def features(self, Z: np.ndarray) -> np.ndarray:
        """Signature features of a window sample (not standardised)."""
        return signature_features(Z, self.depth, time_aug=self.time_aug,
                                  lead_lag_transform=self.lead_lag)

    def prepare(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signature features of both samples, standardised with pooled statistics."""
        X, Y = check_samples(X, Y)
        Fx, Fy = self.features(X), self.features(Y)
        if self.standardize:
            Fx, Fy = standardize_pooled(Fx, Fy)
        return Fx, Fy

    def __call__(self, X: np.ndarray, Y: np.ndarray, *, rng: np.random.Generator,
                 groups_x: np.ndarray | None = None, groups_y: np.ndarray | None = None,
                 order_x: np.ndarray | None = None,
                 order_y: np.ndarray | None = None) -> TestResult:
        """Run the test (``groups_*`` ignored; ``order_*`` used by the wild bootstrap)."""
        Fx, Fy = self.prepare(X, Y)
        ox = _check_order(order_x, Fx.shape[0], "order_x")
        oy = _check_order(order_y, Fy.shape[0], "order_y")
        stat, p, extra = self.mmd.test_features(Fx, Fy, rng=rng, order_x=ox, order_y=oy)
        extra: dict[str, Any] = {**extra, "depth": self.depth, "sig_dim": int(Fx.shape[1]),
                                 "lead_lag": self.lead_lag, "time_aug": self.time_aug}
        return TestResult(name=self.name, statistic=stat, pvalue=p, n_x=int(Fx.shape[0]),
                          n_y=int(Fy.shape[0]), extra=extra)
