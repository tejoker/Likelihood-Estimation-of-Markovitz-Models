"""Moment estimators, Markowitz portfolios and out-of-sample evaluation (DESIGN section 3.10).

Inputs are daily log-returns ``R`` of shape ``(T, N)`` and daily moments; nothing is annualised
except the Sharpe ratio reported by :func:`evaluate` (``x sqrt(252)``).

Optimisation problems are solved in the common form::

    minimise 0.5 x'Qx - c'x   subject to   a'x = b   [and x >= 0],

analytically (KKT linear system) when there is no sign constraint, and with
``scipy.optimize.minimize(method="SLSQP")`` with analytic gradients when ``long_only=True``.
If ``Q`` is singular (e.g. a sample covariance with ``T <= N``) the unconstrained problem is
solved by the null-space method on the budget hyperplane, which enforces ``a'x = b`` exactly
and returns the minimum-norm minimiser; if the problem has no minimiser (a zero-variance,
zero-cost portfolio with non-zero expected return) a ``ValueError`` is raised.
The SLSQP solution is then polished by a primal active-set refinement: the equality-constrained
problem is solved exactly on the estimated support and the full KKT conditions (primal
feasibility, ``x >= 0``, and dual feasibility of the bound multipliers) are verified. Since the
problems are convex, a point that passes the check is a global optimum; if the check fails the
(clipped, renormalised) SLSQP point is returned and a warning is logged.

References
----------
DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification: how
    inefficient is the 1/N portfolio strategy? *Review of Financial Studies* 22(5), 1915-1953.
Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance
    matrices. *Journal of Multivariate Analysis* 88(2), 365-411.
Cornuejols, G. and Tutuncu, R. (2007). *Optimization Methods in Finance*. Cambridge University
    Press (section 8.2: maximum-Sharpe portfolio as a convex QP).
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy import linalg, optimize

from .config import TRADING_DAYS

logger = logging.getLogger(__name__)

__all__ = [
    "sample_moments",
    "ledoit_wolf",
    "average_correlation",
    "gmv_weights",
    "mean_variance_weights",
    "tangency_weights",
    "equal_weights",
    "portfolio_returns",
    "max_drawdown",
    "evaluate",
    "utility_gap",
]

Method = Literal["auto", "analytic", "slsqp"]


# ----------------------------------------------------------------------------------------------
# Moment estimators
# ----------------------------------------------------------------------------------------------
def _check_panel(R: np.ndarray, min_obs: int = 2) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"returns must have shape (T, N); got ndim={R.ndim}")
    if R.shape[0] < min_obs or R.shape[1] < 1:
        raise ValueError(f"returns of shape {R.shape} are too small")
    if not np.all(np.isfinite(R)):
        raise ValueError("returns contain NaN or inf")
    return R


def sample_moments(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean and unbiased sample covariance of daily returns.

    Parameters
    ----------
    R : ndarray, shape (T, N)
        Daily log-returns.

    Returns
    -------
    mu : ndarray, shape (N,)
    Sigma : ndarray, shape (N, N)
        Covariance with ``ddof = 1``.
    """
    R = _check_panel(R)
    return R.mean(axis=0), np.atleast_2d(np.cov(R, rowvar=False, ddof=1))


def ledoit_wolf(
    R: np.ndarray, *, return_shrinkage: bool = False
) -> np.ndarray | tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage covariance towards the scaled identity.

    This is the "well-conditioned estimator" of Ledoit and Wolf (2004, JMVA), with target
    ``F = m I``, ``m = tr(S) / N``::

        Sigma_LW = (b2 / d2) m I + (1 - b2 / d2) S,
        d2  = ||S - m I||^2,
        b2  = min(d2, n^{-2} sum_t ||x_t x_t' - S||^2),

    where ``||A||^2 = tr(A A') / N`` and ``S = n^{-1} sum_t x_t x_t'`` is the covariance of the
    demeaned returns ``x_t`` with divisor ``n`` (as in the paper and in
    ``sklearn.covariance.LedoitWolf``). The identity target is chosen over the
    constant-correlation target of Ledoit and Wolf (2004, JPM) because it needs no structural
    assumption on correlations and keeps the estimator well conditioned for any ``N / T``.

    Parameters
    ----------
    R : ndarray, shape (T, N)
        Daily returns.
    return_shrinkage : bool, default False
        Also return the shrinkage intensity ``b2 / d2`` in ``[0, 1]``.

    Returns
    -------
    Sigma : ndarray, shape (N, N)
    shrinkage : float
        Only if ``return_shrinkage``.
    """
    R = _check_panel(R)
    n, p = R.shape
    X = R - R.mean(axis=0)
    S = X.T @ X / n
    m = np.trace(S) / p
    d2 = (np.sum(S**2) - 2.0 * m * np.trace(S) + p * m**2) / p
    b2_bar = (np.sum(np.sum(X**2, axis=1) ** 2) / n - np.sum(S**2)) / (n * p)
    b2 = min(max(b2_bar, 0.0), d2)
    shrinkage = float(b2 / d2) if d2 > 0.0 else 0.0
    Sigma = (1.0 - shrinkage) * S
    Sigma[np.diag_indices(p)] += shrinkage * m
    return (Sigma, shrinkage) if return_shrinkage else Sigma


def average_correlation(X: np.ndarray, *, is_cov: bool = False) -> float:
    """Mean pairwise (off-diagonal) correlation of a return panel or covariance matrix.

    A one-number summary of cross-sectional dependence, used to compare real and synthetic
    samples (e.g. to attribute a portfolio-utility gap to a generator that under- or
    over-states correlations).

    Parameters
    ----------
    X : ndarray
        Returns of shape ``(T, N)``, or a covariance matrix ``(N, N)`` if ``is_cov``.
    is_cov : bool, default False
        Interpret ``X`` as a covariance (or correlation) matrix.

    Returns
    -------
    float
        ``2 / (N (N - 1)) sum_{i<j} rho_ij``; ``nan`` if ``N = 1``.
    """
    if is_cov:
        S = _check_cov(X)
    else:
        S = np.atleast_2d(np.cov(_check_panel(X), rowvar=False, ddof=1))
    N = S.shape[0]
    if N < 2:
        return float("nan")
    d = np.sqrt(np.diag(S))
    if not np.all(d > 0):
        raise ValueError("average_correlation: zero variance on the diagonal")
    C = S / np.outer(d, d)
    return float(C[np.triu_indices(N, 1)].mean())


# ----------------------------------------------------------------------------------------------
# Quadratic programs
# ----------------------------------------------------------------------------------------------
def _check_cov(Sigma: np.ndarray) -> np.ndarray:
    S = np.atleast_2d(np.asarray(Sigma, dtype=np.float64))
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"Sigma must be square, got shape {S.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("Sigma contains NaN or inf")
    if not np.allclose(S, S.T, rtol=1e-10, atol=1e-14 * np.max(np.abs(S))):
        raise ValueError("Sigma must be symmetric")
    return 0.5 * (S + S.T)


def _check_mu(mu: np.ndarray, N: int) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).ravel()
    if mu.size != N:
        raise ValueError(f"mu has length {mu.size}, Sigma is {N}x{N}")
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu contains NaN or inf")
    return mu


def _eq_qp(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    """Solve ``min 0.5 x'Qx - c'x s.t. a'x = b``: ``x = Q^{-1}(c + lam a)``.

    Raises ``LinAlgError`` if ``Q`` is not positive definite or ``a'Q^{-1}a = 0``.
    """
    cf = linalg.cho_factor(Q, lower=True, check_finite=False)
    Qa = linalg.cho_solve(cf, a, check_finite=False)
    Qc = linalg.cho_solve(cf, c, check_finite=False)
    denom = float(a @ Qa)
    if not np.isfinite(denom) or abs(denom) < 1e-300:
        raise linalg.LinAlgError("degenerate equality-constrained QP (a'Q^{-1}a = 0)")
    lam = (b - float(a @ Qc)) / denom
    return Qc + lam * Qa, lam


def _eq_qp_nullspace(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    """Null-space solution of ``min 0.5 x'Qx - c'x s.t. a'x = b`` for positive semi-definite ``Q``.

    ``x = x_p + Z y`` with ``x_p = b a / a'a`` and ``Z`` an orthonormal basis of
    ``null(a')``, so the budget holds exactly whatever ``y``; ``y`` is the minimum-norm
    least-squares solution of ``(Z'QZ) y = Z'(c - Q x_p)``. Raises ``ValueError`` when that
    reduced system is inconsistent (``Z'c`` has a component in the null space of ``Z'QZ``: a
    zero-variance self-financing portfolio with non-zero expected return, so the objective is
    unbounded below) or the result fails the stationarity check.
    """
    N = a.size
    aa = float(a @ a)
    if not aa > 0.0:
        raise ValueError("degenerate budget constraint (a = 0)")
    x_p = (b / aa) * a
    if N == 1:
        return x_p
    basis = np.linalg.qr(a[:, None], mode="complete")[0]
    Z = basis[:, 1:]  # orthonormal basis of the hyperplane a'x = 0
    H = Z.T @ Q @ Z
    H = 0.5 * (H + H.T)
    g = Z.T @ (c - Q @ x_p)
    y = np.linalg.lstsq(H, g, rcond=None)[0]
    x = x_p + Z @ y
    resid = float(np.linalg.norm(Z.T @ (Q @ x - c)))
    scale = float(np.linalg.norm(Z.T @ c)) + float(np.linalg.norm(Q, 2)) * float(
        np.linalg.norm(x)) + 1e-300
    if not (np.all(np.isfinite(x)) and resid <= 1e-8 * scale):
        raise ValueError(
            "the equality-constrained quadratic program has no (numerically reliable) solution: "
            "the covariance matrix is singular on the budget hyperplane and the expected-return "
            f"vector is not in its range (stationarity residual {resid / scale:.2e} relative), "
            "i.e. a zero-variance self-financing portfolio has non-zero expected return. Use a "
            "well-conditioned covariance estimate such as ledoit_wolf()."
        )
    return x


def _budget_ok(x: np.ndarray, a: np.ndarray, b: float, tol: float = 1e-9) -> bool:
    """``a'x = b`` up to a relative tolerance, and ``x`` finite."""
    if not np.all(np.isfinite(x)):
        return False
    return abs(float(a @ x) - b) <= tol * (abs(b) + float(np.abs(a) @ np.abs(x)))


def _kkt_polish(
    Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float, x0: np.ndarray
) -> np.ndarray | None:
    """Primal active-set refinement of an approximate solution of the non-negative QP.

    Starting from the support of ``x0``, solve the equality-constrained QP on the support, drop
    the most negative component or add the most dual-infeasible excluded component, until the
    KKT conditions hold. Returns ``None`` if they cannot be verified.
    """
    N = x0.size
    scale_x = max(float(np.max(np.abs(x0))), 1e-300)
    support = x0 > 1e-9 * scale_x
    if not support.any():
        support[np.argmax(x0)] = True
    tol_g = 1e-9 * (np.max(np.abs(Q)) * scale_x + np.max(np.abs(c)) + 1e-300)
    for _ in range(4 * N + 10):
        idx = np.flatnonzero(support)
        try:
            xs, lam = _eq_qp(Q[np.ix_(idx, idx)], c[idx], a[idx], b)
        except linalg.LinAlgError:
            return None
        if xs.min() < -1e-12 * max(float(np.max(np.abs(xs))), 1e-300):
            support[idx[np.argmin(xs)]] = False
            if not support.any():
                return None
            continue
        x = np.zeros(N)
        x[idx] = np.maximum(xs, 0.0)
        g = Q @ x - c - lam * a  # bound multipliers: must be >= 0 off the support
        g[support] = np.inf
        j = int(np.argmin(g))
        if g[j] < -tol_g:
            support[j] = True
            continue
        return x
    return None


def _solve_qp(
    Q: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    b: float,
    *,
    nonneg: bool,
    method: Method = "auto",
) -> np.ndarray:
    """``argmin 0.5 x'Qx - c'x`` s.t. ``a'x = b`` (and ``x >= 0`` if ``nonneg``)."""
    if method not in ("auto", "analytic", "slsqp"):
        raise ValueError(f"unknown method {method!r}")
    if nonneg and method == "analytic":
        raise ValueError("method='analytic' is only available without the long-only constraint")
    # Rescale so that SLSQP's absolute tolerances are meaningful (argmin unchanged).
    s = float(np.mean(np.abs(np.diag(Q))))
    s = s if s > 0 else 1.0
    Qs, cs = Q / s, c / s
    if not nonneg and method != "slsqp":
        ev = np.linalg.eigvalsh(Qs)
        if ev[0] > a.size * np.finfo(np.float64).eps * ev[-1]:
            try:
                x = _eq_qp(Qs, cs, a, b)[0]
                if _budget_ok(x, a, b):
                    return x
                logger.warning("ill-conditioned covariance: the Cholesky KKT solution violates "
                               "the budget constraint; using the null-space method")
            except linalg.LinAlgError:
                logger.warning("covariance not numerically positive definite: using the "
                               "null-space method")
        else:
            logger.warning("singular covariance (eigenvalue ratio %.1e): using the null-space "
                           "(minimum-norm) solution", ev[0] / ev[-1] if ev[-1] > 0 else 0.0)
        return _eq_qp_nullspace(Qs, cs, a, b)

    N = a.size
    if nonneg:
        ap = np.maximum(a, 0.0)
        x0 = b * ap / float(ap @ a)
    else:
        x0 = b * a / float(a @ a)
    res = optimize.minimize(
        lambda x: 0.5 * x @ Qs @ x - cs @ x,
        x0,
        jac=lambda x: Qs @ x - cs,
        method="SLSQP",
        bounds=[(0.0, None)] * N if nonneg else None,
        constraints=[{"type": "eq", "fun": lambda x: a @ x - b, "jac": lambda x: a}],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    x = np.asarray(res.x, dtype=np.float64)
    if not nonneg:
        if not res.success:
            logger.warning("SLSQP did not converge: %s", res.message)
        return x
    polished = _kkt_polish(Qs, cs, a, b, x)
    if polished is not None:
        return polished
    logger.warning("long-only QP: KKT polish failed (SLSQP status: %s); returning clipped "
                   "SLSQP solution", res.message)
    x = np.maximum(x, 0.0)
    return x * (b / float(a @ x))


# ----------------------------------------------------------------------------------------------
# Portfolios
# ----------------------------------------------------------------------------------------------
def gmv_weights(Sigma: np.ndarray, long_only: bool = False, *, method: Method = "auto") -> np.ndarray:
    """Global minimum-variance portfolio ``argmin w'Sigma w`` s.t. ``sum w = 1``.

    Parameters
    ----------
    Sigma : ndarray, shape (N, N)
        Covariance matrix.
    long_only : bool, default False
        Add ``w >= 0`` (SLSQP + KKT polish). Otherwise the analytic solution
        ``Sigma^{-1} 1 / (1' Sigma^{-1} 1)``.
    method : {"auto", "analytic", "slsqp"}, default "auto"
        ``"slsqp"`` forces the numerical solver also when unconstrained (used for checks).

    Returns
    -------
    ndarray, shape (N,)
        Weights summing to one.
    """
    S = _check_cov(Sigma)
    N = S.shape[0]
    ones = np.ones(N)
    return _solve_qp(S, np.zeros(N), ones, 1.0, nonneg=long_only, method=method)


def mean_variance_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    gamma: float,
    long_only: bool = True,
    *,
    method: Method = "auto",
) -> np.ndarray:
    """Mean-variance portfolio ``argmax w'mu - (gamma / 2) w'Sigma w`` s.t. ``sum w = 1``.

    Parameters
    ----------
    mu : ndarray, shape (N,)
        Expected daily returns.
    Sigma : ndarray, shape (N, N)
        Daily covariance.
    gamma : float
        Risk aversion (> 0). It is unit-free when ``mu`` and ``Sigma`` share the same horizon.
    long_only : bool, default True
        Add ``w >= 0``.
    method : {"auto", "analytic", "slsqp"}, default "auto"

    Returns
    -------
    ndarray, shape (N,)
    """
    if not gamma > 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    S = _check_cov(Sigma)
    m = _check_mu(mu, S.shape[0])
    return _solve_qp(gamma * S, m, np.ones(S.shape[0]), 1.0, nonneg=long_only, method=method)


def tangency_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    long_only: bool = True,
    *,
    rf: float = 0.0,
    method: Method = "auto",
) -> np.ndarray:
    """Maximum-Sharpe (tangency) portfolio, fully invested.

    Unconstrained: ``w = Sigma^{-1}(mu - rf) / 1'Sigma^{-1}(mu - rf)``, defined when the
    denominator is positive. Long-only: the standard convex reformulation (Cornuejols and
    Tutuncu, 2007) ``min y'Sigma y`` s.t. ``(mu - rf)'y = 1``, ``y >= 0``, then
    ``w = y / sum(y)``; it requires at least one asset with ``mu_i > rf``.

    Parameters
    ----------
    mu : ndarray, shape (N,)
        Expected daily returns.
    Sigma : ndarray, shape (N, N)
        Daily covariance.
    long_only : bool, default True
        Add ``w >= 0``.
    rf : float, default 0.0
        Daily risk-free rate.
    method : {"auto", "analytic", "slsqp"}, default "auto"

    Returns
    -------
    ndarray, shape (N,)

    Raises
    ------
    ValueError
        If no portfolio in the admissible set has a positive expected excess return.
    """
    S = _check_cov(Sigma)
    m = _check_mu(mu, S.shape[0]) - float(rf)
    N = S.shape[0]
    scale = float(np.max(np.abs(m)))
    if scale == 0.0:
        raise ValueError("all expected excess returns are zero: tangency portfolio undefined")
    a = m / scale  # same constraint direction, better conditioned
    if long_only:
        if not np.any(a > 0):
            raise ValueError("no asset has a positive expected excess return: the long-only "
                             "tangency portfolio is undefined")
        y = _solve_qp(S, np.zeros(N), a, 1.0, nonneg=True, method=method)
    else:
        y = _solve_qp(S, np.zeros(N), a, 1.0, nonneg=False, method=method)
        if float(np.sum(y)) <= 0.0:
            raise ValueError("1' Sigma^{-1} (mu - rf) <= 0: the normalised portfolio lies on the "
                             "inefficient branch (no tangency portfolio)")
    return y / float(np.sum(y))


def equal_weights(N: int) -> np.ndarray:
    """The 1/N portfolio."""
    N = int(N)
    if N < 1:
        raise ValueError("N must be >= 1")
    return np.full(N, 1.0 / N)


# ----------------------------------------------------------------------------------------------
# Out-of-sample evaluation
# ----------------------------------------------------------------------------------------------
def portfolio_returns(
    w: np.ndarray, R: np.ndarray, *, kind: Literal["simple", "log"] = "simple"
) -> np.ndarray:
    """Daily returns of a fixed-weight portfolio from asset log-returns ``R`` (T, N).

    ``kind="simple"``: simple return of a portfolio rebalanced daily to ``w``,
    ``sum_i w_i (exp(r_it) - 1)`` (exact). ``kind="log"``: ``w' r_t``, the linear
    approximation in log-returns consistent with moment estimates computed on log-returns.
    """
    R = _check_panel(R, min_obs=1)
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size != R.shape[1]:
        raise ValueError(f"w has length {w.size}, R has {R.shape[1]} columns")
    if kind == "simple":
        return np.expm1(R) @ w
    if kind == "log":
        return R @ w
    raise ValueError(f"unknown kind {kind!r}")


def max_drawdown(rp: np.ndarray, *, kind: Literal["simple", "log"] = "simple") -> float:
    """Maximum drawdown ``max_t (1 - W_t / max_{s<=t} W_s)`` of the wealth path, ``W_0 = 1``.

    ``W_t = prod (1 + rp)`` for simple returns, ``exp(cumsum rp)`` for log-returns.
    """
    rp = np.asarray(rp, dtype=np.float64).ravel()
    if kind == "simple":
        wealth = np.cumprod(1.0 + rp)
    elif kind == "log":
        wealth = np.exp(np.cumsum(rp))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    wealth = np.concatenate([[1.0], wealth])
    peak = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / peak))


def evaluate(
    w: np.ndarray,
    R_test: np.ndarray,
    *,
    gamma: float = 5.0,
    rf: float = 0.0,
    kind: Literal["simple", "log"] = "simple",
) -> dict[str, float]:
    """Out-of-sample performance of fixed weights ``w`` on daily log-returns ``R_test``.

    Parameters
    ----------
    w : ndarray, shape (N,)
        Portfolio weights.
    R_test : ndarray, shape (T, N)
        Daily log-returns of the test period, ``T >= 2``.
    gamma : float, default 5.0
        Risk aversion of the certainty-equivalent return.
    rf : float, default 0.0
        Daily risk-free rate.
    kind : {"simple", "log"}, default "simple"
        How portfolio returns are formed (see :func:`portfolio_returns`).

    Returns
    -------
    dict
        ``mean``: daily mean return; ``vol``: daily standard deviation (``ddof=1``); ``var``:
        daily variance; ``sharpe``: **annualised** Sharpe ratio
        ``sqrt(252) (mean - rf) / vol`` (the only annualised quantity); ``max_drawdown``: as a
        positive fraction; ``ce``: daily certainty-equivalent return
        ``(mean - rf) - gamma / 2 * var`` (DeMiguel, Garlappi and Uppal, 2009); ``n_obs``.
    """
    rp = portfolio_returns(w, R_test, kind=kind)
    if rp.size < 2:
        raise ValueError("evaluate needs at least 2 test observations")
    mean = float(np.mean(rp))
    var = float(np.var(rp, ddof=1))
    vol = float(np.sqrt(var))
    excess = mean - float(rf)
    sharpe = float(np.sqrt(TRADING_DAYS) * excess / vol) if vol > 0 else float("nan")
    return {
        "mean": mean,
        "vol": vol,
        "var": var,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(rp, kind=kind),
        "ce": excess - 0.5 * float(gamma) * var,
        "n_obs": float(rp.size),
    }


_GAP_METRICS = ("mean", "vol", "var", "sharpe", "max_drawdown", "ce")


def utility_gap(
    w_synth: np.ndarray,
    w_real: np.ndarray,
    R_test: np.ndarray,
    *,
    gamma: float = 5.0,
    rf: float = 0.0,
    kind: Literal["simple", "log"] = "simple",
) -> dict[str, float]:
    """Out-of-sample differences between a synthetic-trained and a real-trained portfolio.

    Both weight vectors are evaluated with :func:`evaluate` on the same real test returns.

    Returns
    -------
    dict
        For each metric ``m`` in ``mean, vol, var, sharpe, max_drawdown, ce``: ``m_synth``,
        ``m_real`` and ``m_gap = m_synth - m_real``. Also ``var_ratio = var_synth / var_real``,
        ``ce_loss = ce_real - ce_synth`` (positive when the synthetic-trained portfolio is worse)
        and ``weight_l1 = ||w_synth - w_real||_1``.
    """
    ws = np.asarray(w_synth, dtype=np.float64).ravel()
    wr = np.asarray(w_real, dtype=np.float64).ravel()
    if ws.shape != wr.shape:
        raise ValueError("w_synth and w_real must have the same length")
    es = evaluate(ws, R_test, gamma=gamma, rf=rf, kind=kind)
    er = evaluate(wr, R_test, gamma=gamma, rf=rf, kind=kind)
    out: dict[str, float] = {}
    for m in _GAP_METRICS:
        out[f"{m}_synth"] = es[m]
        out[f"{m}_real"] = er[m]
        out[f"{m}_gap"] = es[m] - er[m]
    out["var_ratio"] = es["var"] / er["var"] if er["var"] > 0 else float("nan")
    out["ce_loss"] = er["ce"] - es["ce"]
    out["weight_l1"] = float(np.sum(np.abs(ws - wr)))
    return out
