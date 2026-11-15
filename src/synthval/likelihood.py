"""Likelihood-based evaluation of return models (DESIGN section 3.8).

Functions
---------
predictive_loglik
    Mean out-of-sample conditional log-likelihood per observation.
berkowitz_test
    Berkowitz (2001) likelihood-ratio test on the probability integral transform.
pit_ks_test
    Kolmogorov-Smirnov test of PIT uniformity (Diebold, Gunther and Tay 1998).
vuong_test
    Vuong (1989) test for non-nested models, optional Newey-West variance and BIC/AIC correction.
window_loglik
    Log-density of each window of returns under a model (used by the LR oracle).
lr_oracle_test
    Likelihood-ratio oracle with a Monte Carlo null distribution: the Neyman-Pearson test when
    window densities are exact (i.i.d. models, or GARCH with ``n_init > 0``), an approximate
    likelihood-ratio benchmark for GARCH with the default fixed-start window density.

All ``TestResult`` objects follow :class:`synthval.twosample.base.TestResult`.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np
from scipy import optimize, stats
from scipy.special import ndtri

from .twosample.base import TestResult, permutation_pvalue

logger = logging.getLogger(__name__)

__all__ = [
    "predictive_loglik",
    "berkowitz_test",
    "pit_ks_test",
    "vuong_test",
    "newey_west_variance",
    "window_loglik",
    "lr_oracle_test",
]

_LOG_2PI = math.log(2.0 * math.pi)


def _vec(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf")
    return a


# --------------------------------------------------------------------------------------------
# predictive log-likelihood
# --------------------------------------------------------------------------------------------
def predictive_loglik(model: Any, r_test: np.ndarray, r_train: np.ndarray | None = None) -> float:
    """Mean per-observation out-of-sample conditional log-likelihood.

    Parameters
    ----------
    model : ReturnModel
        A fitted model with a tractable likelihood (usually fitted on ``r_train``).
    r_test : ndarray, shape (T,)
        Evaluation returns.
    r_train : ndarray, optional
        Returns preceding ``r_test``; filtered models (GARCH) start the test-period filter from
        the end of this series. Ignored by i.i.d. models.

    Returns
    -------
    float
        ``mean_t log p(r_t | r_{<t})`` over the test period.
    """
    r_test = _vec(r_test, "r_test")
    hist = None if r_train is None else _vec(r_train, "r_train")
    return float(np.mean(model.loglik_obs(r_test, r_history=hist)))


# --------------------------------------------------------------------------------------------
# PIT tests
# --------------------------------------------------------------------------------------------
def _ar1_exact_loglik(z: np.ndarray, c: float, rho: float, sigma: float) -> float:
    """Exact Gaussian AR(1) log-likelihood (first observation from the stationary law)."""
    s2 = sigma * sigma
    m1 = c / (1.0 - rho)
    v1 = s2 / (1.0 - rho * rho)
    ll = -0.5 * (_LOG_2PI + math.log(v1) + (z[0] - m1) ** 2 / v1)
    e = z[1:] - c - rho * z[:-1]
    ll += -0.5 * (e.size * (_LOG_2PI + math.log(s2)) + np.sum(e * e) / s2)
    return float(ll)


def berkowitz_test(u: np.ndarray, *, clip: float = 1e-10) -> TestResult:
    """Berkowitz (2001) LR test of ``z_t = Phi^{-1}(u_t)`` iid N(0, 1) against Gaussian AR(1).

    The alternative ``z_t - m = rho (z_{t-1} - m) + sigma e_t`` is fitted by exact Gaussian
    maximum likelihood (first observation from the stationary distribution, as in Berkowitz
    2001); ``LR = 2 (l(c, rho, sigma) - l(0, 0, 1))`` is asymptotically chi-squared with 3 degrees
    of freedom (mean, variance, AR(1) coefficient) under correct specification.

    Parameters
    ----------
    u : ndarray, shape (n,)
        PIT values ``u_t = F_t(r_t | past)`` in time order.
    clip : float
        ``u`` is clipped to ``[clip, 1 - clip]`` before the inverse normal transform.

    Returns
    -------
    TestResult
        ``statistic`` = LR, ``pvalue`` from chi2(3). ``extra`` holds the fitted ``mean``,
        ``var`` (stationary), ``rho``, ``sigma``, the 1-df independence statistic ``lr_ind`` and
        its p-value ``p_ind`` (H0: rho = 0, mean and variance free), and ``n_clipped``.
    """
    u = _vec(u, "u")
    n = u.size
    if n < 3:
        raise ValueError("berkowitz_test needs at least 3 observations")
    n_clipped = int(np.sum((u < clip) | (u > 1.0 - clip)))
    z = ndtri(np.clip(u, clip, 1.0 - clip))
    ll0 = float(-0.5 * (n * _LOG_2PI + np.sum(z * z)))

    # start from conditional OLS
    y, xl = z[1:], z[:-1]
    X = np.column_stack([np.ones(n - 1), xl])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c0, rho0 = float(beta[0]), float(np.clip(beta[1], -0.99, 0.99))
    s0 = float(np.sqrt(max(np.mean((y - X @ beta) ** 2), 1e-12)))

    def nll(th: np.ndarray) -> float:
        val = -_ar1_exact_loglik(z, th[0], th[1], math.exp(th[2]))
        return val if np.isfinite(val) else 1e300

    candidates = [np.array([c0, rho0, math.log(s0)]), np.array([0.0, 0.0, 0.0])]
    best_th, best_ll = None, -np.inf
    for th0 in candidates:
        res = optimize.minimize(nll, th0, method="L-BFGS-B",
                                bounds=[(None, None), (-0.999, 0.999), (-20.0, 20.0)])
        for th in (res.x, th0):
            ll = -nll(th)
            if ll > best_ll:
                best_th, best_ll = th, ll
    c, rho, sigma = float(best_th[0]), float(best_th[1]), math.exp(float(best_th[2]))
    lr = max(2.0 * (best_ll - ll0), 0.0)
    # restricted model rho = 0 (iid N(m, s^2), closed form) for the independence LR
    s2r = float(np.var(z))
    ll_r = float(-0.5 * n * (_LOG_2PI + math.log(max(s2r, 1e-300)) + 1.0))
    lr_ind = max(2.0 * (best_ll - ll_r), 0.0)
    return TestResult(
        name="berkowitz",
        statistic=float(lr),
        pvalue=float(stats.chi2.sf(lr, 3)),
        n_x=n,
        extra={
            "mean": c / (1.0 - rho),
            "var": sigma * sigma / (1.0 - rho * rho),
            "rho": rho,
            "sigma": sigma,
            "lr_ind": float(lr_ind),
            "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "n_clipped": n_clipped,
            "df": 3,
        },
    )


def pit_ks_test(u: np.ndarray) -> TestResult:
    """Kolmogorov-Smirnov test of PIT values against U(0, 1) (Diebold, Gunther and Tay 1998).

    The KS null distribution assumes i.i.d. PIT values; serial dependence of ``u`` is tested by
    :func:`berkowitz_test`, not here.
    """
    u = _vec(u, "u")
    res = stats.kstest(u, "uniform")
    return TestResult(name="pit_ks", statistic=float(res.statistic), pvalue=float(res.pvalue),
                      n_x=u.size, extra={})


# --------------------------------------------------------------------------------------------
# Vuong test
# --------------------------------------------------------------------------------------------
def newey_west_variance(d: np.ndarray, lags: int | None = None) -> tuple[float, int]:
    """Newey-West (1987) long-run variance of ``d`` with Bartlett weights.

    ``lags`` defaults to ``floor(4 (n / 100)^{2/9})``. Returns ``(variance, lags)``.
    """
    d = _vec(d, "d")
    n = d.size
    if lags is None:
        lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = int(max(0, min(lags, n - 1)))
    e = d - d.mean()
    lrv = float(e @ e) / n
    for j in range(1, lags + 1):
        lrv += 2.0 * (1.0 - j / (lags + 1.0)) * float(e[j:] @ e[:-j]) / n
    return lrv, lags


def vuong_test(ll_a: np.ndarray, ll_b: np.ndarray, k_a: int = 0, k_b: int = 0, *,
               hac: bool = False, lags: int | None = None,
               correction: str = "bic") -> TestResult:
    """Vuong (1989) test for non-nested (or overlapping) models.

    ``d_t = ll_a[t] - ll_b[t]``; statistic ``Z = (sum_t d_t - C) / (sqrt(n) * omega)`` which, for
    ``C = 0``, is ``sqrt(n) mean(d) / sd(d)``. ``omega^2`` is the sample variance of ``d``
    (divisor n) or, with ``hac=True``, its Newey-West long-run variance (needed for out-of-sample
    conditional log-likelihoods, which can be serially dependent). ``C`` is the Schwarz
    correction ``(k_a - k_b) log(n) / 2`` (``correction="bic"``), the Akaike correction
    ``k_a - k_b`` (``"aic"``) or 0 (``"none"``); with the default ``k_a = k_b = 0`` there is no
    correction.

    Returns
    -------
    TestResult
        ``statistic`` = Z (positive favours model a), ``pvalue`` two-sided (H0: equal fit).
        ``extra``: ``p_a_better`` = P(N(0,1) >= Z) (one-sided, H1: a better), ``p_b_better``,
        ``mean_d``, ``sd``, ``lr`` (corrected sum), ``hac``, ``lags``, ``correction``.
    """
    a = _vec(ll_a, "ll_a")
    b = _vec(ll_b, "ll_b")
    if a.size != b.size:
        raise ValueError("ll_a and ll_b must have the same length")
    if correction not in ("bic", "aic", "none"):
        raise ValueError("correction must be 'bic', 'aic' or 'none'")
    d = a - b
    n = d.size
    if correction == "bic":
        corr = (k_a - k_b) * math.log(n) / 2.0
    elif correction == "aic":
        corr = float(k_a - k_b)
    else:
        corr = 0.0
    lr = float(d.sum()) - corr
    if hac:
        omega2, used_lags = newey_west_variance(d, lags)
        if not omega2 > 0:
            logger.warning("vuong_test: non-positive HAC variance, using the iid variance")
            omega2 = float(np.var(d))
    else:
        omega2, used_lags = float(np.var(d)), 0
    if omega2 > 0:
        z = lr / (math.sqrt(n) * math.sqrt(omega2))
    else:  # identical (up to a constant) log-likelihoods
        z = 0.0 if lr == 0 else math.copysign(np.inf, lr)
    return TestResult(
        name="vuong",
        statistic=float(z),
        pvalue=float(2.0 * stats.norm.sf(abs(z))),
        n_x=n,
        n_y=n,
        extra={
            "p_a_better": float(stats.norm.sf(z)),
            "p_b_better": float(stats.norm.cdf(z)),
            "mean_d": float(d.mean()),
            "sd": float(math.sqrt(max(omega2, 0.0))),
            "lr": lr,
            "hac": bool(hac),
            "lags": used_lags,
            "correction": correction,
        },
    )


# --------------------------------------------------------------------------------------------
# window log-likelihoods and the LR oracle
# --------------------------------------------------------------------------------------------
def window_loglik(model: Any, windows: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Log-density of each window of returns under ``model``.

    Each window is scored as a stretch of a stationary path. For i.i.d. models (GBM, Merton)
    the result is the exact window density (sum of the per-observation log-densities). For
    :class:`~synthval.models.univariate.GARCHt` the exact density integrates over the
    stationary law of the conditional variance at the window's first observation, which has
    no closed form: by default each window's filter starts at the unconditional variance
    ``omega / (1 - alpha - beta)``, an **approximation**; passing
    ``sigma2_init=model.stationary_variance_draws(K, rng=rng)`` integrates over ``K`` draws
    from the stationary law instead (exact up to Monte Carlo error).

    Parameters
    ----------
    model : ReturnModel
        Fitted model with a likelihood. If it provides ``loglik_windows`` that method is used
        (with ``**kwargs``); otherwise ``model.loglik_obs`` is applied to each window
        separately.
    windows : ndarray, shape (n, L)
    **kwargs
        Passed to ``model.loglik_windows`` (e.g. ``sigma2_init`` for GARCHt).

    Returns
    -------
    ndarray, shape (n,)
    """
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 1:
        w = w[None, :]
    if w.ndim != 2:
        raise ValueError(f"windows must have shape (n, L), got {w.shape}")
    if hasattr(model, "loglik_windows"):
        return np.asarray(model.loglik_windows(w, **kwargs), dtype=np.float64)
    if kwargs:
        raise TypeError(f"{getattr(model, 'name', model)} has no loglik_windows; "
                        f"cannot pass {sorted(kwargs)}")
    if not getattr(model, "has_likelihood", False):
        raise NotImplementedError(f"{getattr(model, 'name', model)}: likelihood is intractable")
    return np.array([float(np.sum(model.loglik_obs(row))) for row in w])


def _null_sampler(source: Any, n: int, L: int) -> Callable[[np.random.Generator], np.ndarray]:
    """Build ``rng -> (n, L)`` sampler of windows under the null hypothesis."""
    if isinstance(source, np.ndarray):
        pool = np.asarray(source, dtype=np.float64)
        if pool.ndim != 2 or pool.shape[1] != L:
            raise ValueError(f"null window pool must have shape (m, {L}), got {pool.shape}")

        def from_pool(rng: np.random.Generator) -> np.ndarray:
            return pool[rng.integers(0, pool.shape[0], size=n)]

        return from_pool
    if hasattr(source, "simulate"):
        def from_model(rng: np.random.Generator) -> np.ndarray:
            return source.simulate(L, n, rng=rng)

        return from_model
    if callable(source):
        def from_callable(rng: np.random.Generator) -> np.ndarray:
            out = np.asarray(source(rng), dtype=np.float64)
            if out.shape != (n, L):
                raise ValueError(f"null sampler returned shape {out.shape}, expected {(n, L)}")
            return out

        return from_callable
    raise TypeError("Y_model_true must be None, a model with .simulate, a callable or an array")


def lr_oracle_test(X: np.ndarray, Y_model_true: Any = None, model_null: Any = None,
                   model_alt: Any = None, *, rng: np.random.Generator, n_mc: int = 500,
                   alpha: float = 0.05, n_init: int = 0) -> TestResult:
    """Likelihood-ratio oracle for ground-truth studies (Monte Carlo null distribution).

    Tests H0: the windows ``X`` are drawn from ``model_null`` against H1: they are drawn from
    ``model_alt``, both fully specified. The statistic is
    ``T = sum_i [log p_alt(x_i) - log p_null(x_i)]`` over the windows of ``X`` (window densities
    from :func:`window_loglik`). The null distribution of T is obtained by Monte Carlo:
    ``n_mc`` samples of the same shape as ``X`` are drawn under H0 and
    ``p = (1 + #{T_b >= T}) / (1 + n_mc)``. The test has exact level for any statistic, provided
    the null samples reproduce how ``X`` is drawn under H0 (see ``Y_model_true``).

    Optimality. When the window densities are exact, the test is the Neyman-Pearson test of
    this simple-vs-simple problem: it is most powerful at its level, and its power bounds that
    of any two-sample test (a two-sample test of X against a synthetic sample Y drawn under H0
    independently of X is a randomised test based on X alone). This holds for i.i.d. models
    (GBM, Merton, Student-t). For GARCH it holds only approximately: with ``n_init=0``
    (default) each window is filtered from the unconditional variance, which is not the exact
    window density, so the test is an *approximate likelihood-ratio benchmark*, valid but not
    necessarily most powerful (for GARCH(0.08, 0.90)-t(6) against matched alternatives of
    persistence 0.8-0.95, the exact density separates the hypotheses 1.7-3 times more per
    window than the fixed start). With ``n_init = K > 0``, models
    that provide ``stationary_variance_draws`` (GARCHt) integrate over ``K`` draws from the
    stationary law of the initial conditional variance; the draws are made once per model and
    shared by ``X`` and all Monte Carlo samples, so ``T`` stays a fixed function of the data
    and the Monte Carlo p-value stays exact, while ``T`` converges to the exact log-likelihood
    ratio as ``K`` grows (cost proportional to ``K``).

    Parameters
    ----------
    X : ndarray, shape (n, L)
        Windows of the sample under test (the "real" sample of a ground-truth study).
    Y_model_true : optional
        How to draw samples under H0; it **must reproduce the sampling protocol of** ``X``.
        One of: ``None`` (use ``model_null.simulate(L, n, rng=rng)``: ``n`` *independent*
        stationary windows; valid only if ``X`` was also generated as independent windows, not
        cut from one long path), an object with ``simulate`` (called the same way), a callable
        ``rng -> (n, L) array`` (e.g. ``lambda g: null.simulate(n * L, 1, rng=g).reshape(n, L)``
        when ``X`` is cut from one long path: for GARCH, windows from one path are dependent and
        the independent-window null understates the variance of ``T``, so the test
        over-rejects, e.g. 11% instead of 5% at n = 500, L = 20, persistence 0.98), or an array
        of pre-simulated null windows ``(m, L)`` from which samples of size ``n`` are drawn with
        replacement (independent windows).
    model_null, model_alt : ReturnModel
        Fully specified null and alternative models with likelihoods.
    rng : numpy.random.Generator
    n_mc : int
        Monte Carlo replicates for the null distribution.
    alpha : float
        Level at which the Monte Carlo critical value is reported in ``extra``.
    n_init : int
        Number of stationary initial-variance draws for models with
        ``stationary_variance_draws`` (see Optimality); 0 keeps the unconditional-variance
        start. Ignored for models without that method.

    Returns
    -------
    TestResult
        ``statistic`` = T, ``pvalue`` Monte Carlo p-value; ``extra``: ``crit`` (Monte Carlo
        (1 - alpha)-quantile of T under H0), ``t_null_mean``, ``t_null_sd``, ``z`` (normal
        approximation), ``n_mc``, ``llr_mean`` (mean per-window log-LR on X), ``n_init``.
    """
    if model_null is None or model_alt is None:
        raise ValueError("lr_oracle_test needs model_null and model_alt")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, L), got {X.shape}")
    n, L = X.shape
    kw_null: dict[str, Any] = {}
    kw_alt: dict[str, Any] = {}
    if int(n_init) > 0:
        for mdl, kw in ((model_null, kw_null), (model_alt, kw_alt)):
            if hasattr(mdl, "stationary_variance_draws"):
                kw["sigma2_init"] = mdl.stationary_variance_draws(int(n_init), rng=rng)

    def stat(W: np.ndarray) -> np.ndarray:
        return window_loglik(model_alt, W, **kw_alt) - window_loglik(model_null, W, **kw_null)

    llr_x = stat(X)
    t_obs = float(np.sum(llr_x))
    sampler = _null_sampler(model_null if Y_model_true is None else Y_model_true, n, L)
    t_null = np.empty(int(n_mc))
    for b in range(int(n_mc)):
        t_null[b] = float(np.sum(stat(sampler(rng))))
    sd = float(np.std(t_null, ddof=1)) if n_mc > 1 else float("nan")
    z = (t_obs - float(np.mean(t_null))) / sd if sd > 0 else float("nan")
    return TestResult(
        name="lr_oracle",
        statistic=t_obs,
        pvalue=permutation_pvalue(t_obs, t_null),
        n_x=n,
        n_y=int(np.asarray(Y_model_true).shape[0]) if isinstance(Y_model_true, np.ndarray) else 0,
        extra={
            "crit": float(np.quantile(t_null, 1.0 - alpha, method="higher")) if n_mc > 0 else float("nan"),
            "alpha": float(alpha),
            "t_null_mean": float(np.mean(t_null)) if n_mc > 0 else float("nan"),
            "t_null_sd": sd,
            "z": float(z),
            "n_mc": int(n_mc),
            "llr_mean": float(np.mean(llr_x)),
            "n_init": int(n_init),
        },
    )
