"""Univariate models of daily log-returns (DESIGN sections 3.4-3.5).

All models work in **daily** units under the real-world measure with ``dt = 1`` trading day.

Classes
-------
GBM
    i.i.d. Gaussian log-returns; closed-form MLE; exact likelihood.
Merton
    Merton (1976) jump-diffusion; MLE on the truncated Poisson mixture; exact likelihood.
GARCHt
    Constant-mean GARCH(1,1) with standardised Student-t innovations; Student-t MLE, by default
    with variance targeting (Engle and Mezrich 1996); exact filtered likelihood
    (Bollerslev 1987).
Heston
    Heston (1993) stochastic volatility; variance simulated by full-truncation Euler with
    ``n_substeps`` (default 20) sub-steps per day (Lord, Koekkoek and van Dijk 2010) and daily
    returns generated from the integrated variance; calibrated by simulated method of moments
    (Duffie and Singleton 1993); likelihood intractable.
StationaryBootstrap
    Politis and Romano (1994) stationary bootstrap of the training returns, mean block length
    from the Politis and White (2004) rule with the Patton, Politis and White (2009) correction;
    likelihood intractable.

Every parametric model can be fitted with ``fit(r)`` or built from known parameters, either
through the constructor (``GBM(mu=0.0, sigma=0.01)``) or ``Model.from_params(**params)``.
Models with a tractable likelihood also expose ``loglik_windows(windows)`` (log-density of each
window of shape ``(n, L)``), used by :func:`synthval.likelihood.window_loglik`.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
from scipy import optimize, stats
from scipy.signal import lfilter
from scipy.special import digamma, gammaincinv, gammaln, logsumexp, ndtr

from .base import ReturnModel

logger = logging.getLogger(__name__)

__all__ = [
    "GBM",
    "Merton",
    "GARCHt",
    "Heston",
    "StationaryBootstrap",
    "optimal_block_length",
    "stationary_bootstrap_indices",
    "smm_moments",
]

_LOG_2PI = math.log(2.0 * math.pi)
_U_LO = np.finfo(np.float64).tiny
_U_HI = 1.0 - np.finfo(np.float64).epsneg


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _as_series(r: Any, name: str = "r", min_len: int = 1) -> np.ndarray:
    """Return ``r`` as a finite float64 vector of length >= ``min_len``."""
    x = np.asarray(r, dtype=np.float64)
    if x.ndim != 1:
        x = np.squeeze(x)
        if x.ndim == 0:
            x = x.reshape(1)
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {np.shape(r)}")
    if x.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} observations, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or inf")
    return x


def _as_windows(windows: Any) -> np.ndarray:
    """Return windows as a finite float64 array of shape ``(n, L)``."""
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 1:
        w = w[None, :]
    if w.ndim != 2:
        raise ValueError(f"windows must have shape (n, L), got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("windows contain NaN or inf")
    return w


def _check_sim_args(n_steps: int, n_paths: int, rng: Any) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if int(n_steps) < 1 or int(n_paths) < 1:
        raise ValueError("n_steps and n_paths must be positive")


def _clip_unit(u: np.ndarray) -> np.ndarray:
    """Clip CDF values into the open unit interval."""
    return np.clip(u, _U_LO, _U_HI)


class _ParamMixin:
    """Shared constructor logic: all parameters given (fitted) or none (unfitted)."""

    _PARAM_NAMES: tuple[str, ...] = ()

    def _init_params(self, values: dict[str, float | None]) -> None:
        given = [v is not None for v in values.values()]
        if any(given) and not all(given):
            missing = [k for k, v in values.items() if v is None]
            raise ValueError(f"{type(self).__name__}: give all parameters or none; missing {missing}")
        self._p: dict[str, float] | None = (
            {k: float(v) for k, v in values.items()} if given and all(given) else None
        )
        if self._p is not None:
            self._validate()

    def _validate(self) -> None:  # pragma: no cover - overridden
        pass

    def _require_fit(self) -> dict[str, float]:
        if self._p is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return self._p

    def params(self) -> dict[str, float]:
        return dict(self._require_fit())

    @classmethod
    def from_params(cls, **params: float):
        """Build a fitted model from known parameters (same names as :meth:`params`)."""
        return cls(**params)


# --------------------------------------------------------------------------------------------
# GBM
# --------------------------------------------------------------------------------------------
class GBM(_ParamMixin, ReturnModel):
    """Geometric Brownian motion: daily log-returns ``r_t ~ iid N(mu, sigma^2)``.

    Parameters
    ----------
    mu, sigma : float, optional
        Daily drift and volatility of log-returns. Give both to build a fitted model.
    """

    name = "GBM"
    has_likelihood = True
    _PARAM_NAMES = ("mu", "sigma")

    def __init__(self, mu: float | None = None, sigma: float | None = None) -> None:
        self._init_params({"mu": mu, "sigma": sigma})

    def _validate(self) -> None:
        if not self._p["sigma"] > 0:
            raise ValueError("sigma must be positive")

    def fit(self, r: np.ndarray) -> "GBM":
        """Closed-form MLE: sample mean and standard deviation with divisor ``T``."""
        x = _as_series(r, min_len=2)
        sigma = float(x.std())
        if sigma <= 0:
            raise ValueError("GBM.fit: constant series")
        self._p = {"mu": float(x.mean()), "sigma": sigma}
        return self

    def simulate(self, n_steps: int, n_paths: int, *, rng: np.random.Generator,
                 burn_in: int = 250, start_from_end: bool = False) -> np.ndarray:
        """Simulate ``(n_paths, n_steps)`` i.i.d. returns (``burn_in``/``start_from_end`` unused)."""
        p = self._require_fit()
        _check_sim_args(n_steps, n_paths, rng)
        return p["mu"] + p["sigma"] * rng.standard_normal((int(n_paths), int(n_steps)))

    def loglik_obs(self, r: np.ndarray, r_history: np.ndarray | None = None) -> np.ndarray:
        """Gaussian log-density of each observation (``r_history`` is ignored: i.i.d. model)."""
        p = self._require_fit()
        x = _as_series(r)
        z = (x - p["mu"]) / p["sigma"]
        return -0.5 * _LOG_2PI - math.log(p["sigma"]) - 0.5 * z * z

    def pit(self, r: np.ndarray, r_history: np.ndarray | None = None) -> np.ndarray:
        p = self._require_fit()
        x = _as_series(r)
        return _clip_unit(ndtr((x - p["mu"]) / p["sigma"]))

    def loglik_windows(self, windows: np.ndarray) -> np.ndarray:
        """Log-density of each window, shape ``(n,)``."""
        w = _as_windows(windows)
        return self.loglik_obs(w.ravel()).reshape(w.shape).sum(axis=1)


# --------------------------------------------------------------------------------------------
# Merton jump-diffusion
# --------------------------------------------------------------------------------------------
_MERTON_LAM_BOUNDS = (1e-4, 0.5)
_MERTON_SJ_MIN = 1e-4
_MERTON_SIGMA_MIN = 1e-5


def _poisson_nmax(lam: float, tol: float = 1e-12, n_min: int = 10, n_cap: int = 200) -> int:
    """Smallest ``n >= n_min`` with ``P(N > n) < tol`` for ``N ~ Poisson(lam)``."""
    n = n_min
    while n < n_cap and stats.poisson.sf(n, lam) >= tol:
        n += 1
    return n


def _merton_core(x: np.ndarray, mu: float, sigma: float, lam: float, muJ: float, sJ: float,
                 n_max: int, grad: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-observation Merton log-density (log-sum-exp over ``n = 0..n_max`` jumps) and gradient.

    The gradient is the sum over observations of d log f / d(mu, sigma, lam, muJ, sJ).
    """
    n = np.arange(n_max + 1, dtype=np.float64)
    logw = -lam + n * math.log(lam) - gammaln(n + 1.0)
    m = mu + n * muJ
    v = sigma * sigma + n * sJ * sJ
    d = x[:, None] - m
    a = logw - 0.5 * np.log(2.0 * math.pi * v) - 0.5 * d * d / v
    ll = logsumexp(a, axis=1)
    if not grad:
        return ll, None
    p = np.exp(a - ll[:, None])  # posterior jump-count probabilities
    dv = d / v
    h = dv * dv - 1.0 / v
    g = np.array([
        np.sum(p * dv),
        sigma * np.sum(p * h),
        np.sum(p * (n / lam - 1.0)),
        np.sum((p * dv) @ n),
        sJ * np.sum((p * h) @ n),
    ])
    return ll, g


class Merton(_ParamMixin, ReturnModel):
    """Merton (1976) jump-diffusion for daily log-returns.

    ``r_t = mu + sigma e_t + sum_{k <= N_t} J_k`` with ``e_t ~ N(0, 1)``, ``N_t ~ Poisson(lam)``
    and ``J_k ~ N(muJ, sJ^2)``, all independent; ``lam`` is the jump intensity per day.

    The density is the Poisson mixture ``sum_n Pois(n; lam) N(mu + n muJ, sigma^2 + n sJ^2)``,
    truncated at ``n_max = max(10, min{n : P(N > n) < 1e-12})`` and evaluated with log-sum-exp.

    Parameters
    ----------
    mu, sigma, lam, muJ, sJ : float, optional
        Give all five to build a fitted model.
    n_starts : int
        Maximum number of moment-based starting points for the multistart MLE.
    """

    name = "Merton"
    has_likelihood = True
    _PARAM_NAMES = ("mu", "sigma", "lam", "muJ", "sJ")

    def __init__(self, mu: float | None = None, sigma: float | None = None,
                 lam: float | None = None, muJ: float | None = None, sJ: float | None = None,
                 *, n_starts: int = 6) -> None:
        self.n_starts = int(n_starts)
        self.fit_info_: dict[str, Any] = {}
        self._init_params({"mu": mu, "sigma": sigma, "lam": lam, "muJ": muJ, "sJ": sJ})

    def _validate(self) -> None:
        p = self._p
        if not (p["sigma"] > 0 and p["lam"] > 0 and p["sJ"] > 0):
            raise ValueError("Merton: sigma, lam and sJ must be positive")

    def _nmax(self) -> int:
        return _poisson_nmax(self._require_fit()["lam"])

    @staticmethod
    def _starting_points(z: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
        """Moment-based starts on standardised data (mean 0, variance 1).

        For a grid of intensities, the jump variance is chosen to match the excess kurtosis
        (``kappa_4 = 3 lam sJ^4`` when ``muJ = 0``) and the jump mean to match the skewness
        (``kappa_3 ~ 3 lam muJ sJ^2``); the diffusion variance takes the rest.
        """
        sk = float(stats.skew(z))
        ek = max(float(stats.kurtosis(z)), 0.05)
        starts = []
        for lam in (0.005, 0.02, 0.08, 0.3):
            share = float(np.clip(math.sqrt(ek * lam / 3.0), 0.02, 0.85))
            sJ2 = share / lam
            muJ = float(np.clip(sk / (3.0 * share), -math.sqrt(sJ2), math.sqrt(sJ2)))
            jump_var = lam * (sJ2 + muJ * muJ)
            if jump_var > 0.9:
                f = 0.9 / jump_var
                sJ2 *= f
                muJ *= math.sqrt(f)
                jump_var = 0.9
            sigma = math.sqrt(max(1.0 - jump_var, 0.05))
            starts.append(np.array([-lam * muJ, sigma, lam, muJ, math.sqrt(sJ2)]))
        starts.append(np.array([0.0, 0.98, 1e-3, 0.0, 2.0]))  # near-Gaussian start
        starts.append(np.array([0.0, 0.7, 0.4, 0.0, 1.0]))  # frequent small jumps
        return [np.clip(s, lo, hi) for s in starts]

    def fit(self, r: np.ndarray) -> "Merton":
        """Maximum likelihood with bounds and multistart (L-BFGS-B, analytic gradient).

        Bounds: ``lam in [1e-4, 0.5]`` per day, ``sigma >= 1e-5``, ``sJ >= 1e-4``. The optimiser
        works on standardised returns ``z = (r - mean) / sd``; parameters are mapped back exactly
        (location-scale equivariance of the model).
        """
        t0 = time.perf_counter()
        x = _as_series(r, min_len=10)
        loc, scale = float(x.mean()), float(x.std())
        if scale <= 0:
            raise ValueError("Merton.fit: constant series")
        z = (x - loc) / scale
        n_max = _poisson_nmax(_MERTON_LAM_BOUNDS[1])
        lo = np.array([-5.0, _MERTON_SIGMA_MIN / scale, _MERTON_LAM_BOUNDS[0], -20.0,
                       _MERTON_SJ_MIN / scale])
        hi = np.array([5.0, 5.0, _MERTON_LAM_BOUNDS[1], 20.0, 30.0])
        hi = np.maximum(hi, lo * 10)
        T = z.size

        def nll(th: np.ndarray) -> tuple[float, np.ndarray]:
            ll, g = _merton_core(z, *th, n_max=n_max, grad=True)
            return -float(ll.sum()) / T, -g / T

        best = None
        starts = self._starting_points(z, lo, hi)[: max(1, self.n_starts)]
        for th0 in starts:
            try:
                res = optimize.minimize(nll, th0, jac=True, method="L-BFGS-B",
                                        bounds=list(zip(lo, hi)),
                                        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
            except (FloatingPointError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Merton start %s failed: %s", th0, exc)
                continue
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        if best is None:  # pragma: no cover - defensive
            raise RuntimeError("Merton.fit: all starts failed")
        mu_z, sig_z, lam, muJ_z, sJ_z = best.x
        self._p = {
            "mu": loc + scale * mu_z,
            "sigma": scale * sig_z,
            "lam": float(lam),
            "muJ": scale * muJ_z,
            "sJ": scale * sJ_z,
        }
        self._p = {k: float(v) for k, v in self._p.items()}
        at_bound = [k for k, v, a, b in zip(self._PARAM_NAMES, best.x, lo, hi)
                    if np.isclose(v, a, rtol=1e-6, atol=0) or np.isclose(v, b, rtol=1e-6, atol=0)]
        self.fit_info_ = {
            "success": bool(best.success),
            "message": str(best.message),
            "loglik": float(-best.fun * T - T * math.log(scale)),
            "n_starts": len(starts),
            "at_bound": at_bound,
            "seconds": time.perf_counter() - t0,
        }
        return self

    def simulate(self, n_steps: int, n_paths: int, *, rng: np.random.Generator,
                 burn_in: int = 250, start_from_end: bool = False) -> np.ndarray:
        """Exact simulation of i.i.d. returns (``burn_in``/``start_from_end`` unused)."""
        p = self._require_fit()
        _check_sim_args(n_steps, n_paths, rng)
        shape = (int(n_paths), int(n_steps))
        nj = rng.poisson(p["lam"], size=shape)
        e = rng.standard_normal(shape)
        ej = rng.standard_normal(shape)
        return p["mu"] + p["sigma"] * e + p["muJ"] * nj + p["sJ"] * np.sqrt(nj) * ej

    def loglik_obs(self, r: np.ndarray, r_history: np.ndarray | None = None) -> np.ndarray:
        """Exact (truncated-mixture) log-density of each observation; ``r_history`` ignored."""
        p = self._require_fit()
        x = _as_series(r)
        ll, _ = _merton_core(x, p["mu"], p["sigma"], p["lam"], p["muJ"], p["sJ"], self._nmax())
        return ll

    def pit(self, r: np.ndarray, r_history: np.ndarray | None = None) -> np.ndarray:
        p = self._require_fit()
        x = _as_series(r)
        n = np.arange(self._nmax() + 1, dtype=np.float64)
        w = np.exp(-p["lam"] + n * math.log(p["lam"]) - gammaln(n + 1.0))
        w /= w.sum()
        m = p["mu"] + n * p["muJ"]
        s = np.sqrt(p["sigma"] ** 2 + n * p["sJ"] ** 2)
        return _clip_unit(ndtr((x[:, None] - m) / s) @ w)

    def loglik_windows(self, windows: np.ndarray) -> np.ndarray:
        """Log-density of each window, shape ``(n,)``."""
        w = _as_windows(windows)
        return self.loglik_obs(w.ravel()).reshape(w.shape).sum(axis=1)


# --------------------------------------------------------------------------------------------
# GARCH(1,1) with standardised Student-t innovations
# --------------------------------------------------------------------------------------------
_GARCH_PERSIST_MAX = 0.9999
_GARCH_NU_BOUNDS = (2.05, 100.0)


def _std_t_const(nu: float) -> float:
    """Log normalising constant of the unit-variance Student-t density."""
    return float(gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * math.log(math.pi * (nu - 2.0)))


def _std_t_const_grad(nu: float) -> float:
    return float(0.5 * digamma(0.5 * (nu + 1.0)) - 0.5 * digamma(0.5 * nu) - 0.5 / (nu - 2.0))


def _garch_filter(e: np.ndarray, omega: float, alpha: float, beta: float,
                  s2_0: float) -> np.ndarray:
    """Conditional variances ``s2[0] = s2_0``, ``s2[t] = omega + alpha e[t-1]^2 + beta s2[t-1]``."""
    T = e.size
    s2 = np.empty(T)
    if T == 0:
        return s2
    s2[0] = s2_0
    if T > 1:
        u = omega + alpha * e[:-1] ** 2
        s2[1:] = lfilter([1.0], [1.0, -beta], u, zi=[beta * s2_0])[0]
    return s2


def _std_t_loglik_obs(e: np.ndarray, s2: np.ndarray, nu: float) -> np.ndarray:
    """Log-density of ``e`` with conditional variance ``s2`` under a unit-variance t(nu)."""
    return _std_t_const(nu) - 0.5 * np.log(s2) - 0.5 * (nu + 1.0) * np.log1p(e * e / ((nu - 2.0) * s2))


def _garch_nll_grad(theta: np.ndarray, x: np.ndarray, s2_0: float) -> tuple[float, np.ndarray]:
    """Mean negative log-likelihood of GARCH(1,1)-t and its analytic gradient.

    ``theta = (mu, omega, alpha, beta, nu)``; the initial variance ``s2_0`` is held fixed.
    """
    mu, omega, alpha, beta, nu = (float(v) for v in theta)
    T = x.size
    e = x - mu
    s2 = _garch_filter(e, omega, alpha, beta, s2_0)
    if not np.all(s2 > 0) or not np.all(np.isfinite(s2)):
        return 1e10, np.zeros(5)
    nu2 = nu - 2.0
    q = e * e / (nu2 * s2)
    l1q = np.log1p(q)
    ll = T * _std_t_const(nu) - 0.5 * np.sum(np.log(s2)) - 0.5 * (nu + 1.0) * np.sum(l1q)
    dl_ds2 = (-0.5 + 0.5 * (nu + 1.0) * q / (1.0 + q)) / s2
    dl_de = -(nu + 1.0) * e / (nu2 * s2 * (1.0 + q))
    D = np.zeros((4, T))
    if T > 1:
        X = np.vstack([np.ones(T - 1), e[:-1] ** 2, s2[:-1], -2.0 * alpha * e[:-1]])
        D[:, 1:] = lfilter([1.0], [1.0, -beta], X, axis=1)
    g_omega, g_alpha, g_beta, g_mu_s2 = D @ dl_ds2
    g_mu = -np.sum(dl_de) + g_mu_s2
    g_nu = T * _std_t_const_grad(nu) - 0.5 * np.sum(l1q) + 0.5 * (nu + 1.0) * np.sum(q / (1.0 + q)) / nu2
    grad = np.array([g_mu, g_omega, g_alpha, g_beta, g_nu])
    return -float(ll) / T, -grad / T


def _garch_nll_grad_vt(theta: np.ndarray, x: np.ndarray, s2_0: float,
                       var_x: float) -> tuple[float, np.ndarray]:
    """Variance-targeted objective: ``theta = (mu, alpha, beta, nu)``, ``omega = var_x (1 - a - b)``.

    The gradient follows from :func:`_garch_nll_grad` by the chain rule
    (``d omega / d alpha = d omega / d beta = -var_x``).
    """
    mu, a, b, nu = (float(v) for v in theta)
    f, g = _garch_nll_grad(np.array([mu, var_x * (1.0 - a - b), a, b, nu]), x, s2_0)
    return f, np.array([g[0], g[2] - var_x * g[1], g[3] - var_x * g[1], g[4]])


class GARCHt(_ParamMixin, ReturnModel):
    """GARCH(1,1) with constant mean and standardised (unit-variance) Student-t innovations.

    ``r_t = mu + eps_t``, ``eps_t = s_t z_t``, ``s_t^2 = omega + alpha eps_{t-1}^2 + beta s_{t-1}^2``,
    ``z_t ~ t(nu) * sqrt((nu - 2) / nu)`` i.i.d.

    Estimation. By default (``variance_targeting=True``) the intercept is tied to the sample
    variance, ``omega = (1 - alpha - beta) * var(r)`` (variance targeting: Engle and Mezrich
    1996; asymptotic theory in Francq, Horvath and Zakoian 2011), and ``(mu, alpha, beta, nu)``
    maximise the Student-t likelihood. The unconditional variance of the fitted model then
    equals the sample variance, so simulated paths have the right volatility level. The
    unrestricted MLE (``variance_targeting=False``) can reach the persistence bound
    ``alpha + beta = 0.9999`` on samples with a volatility regime shift (e.g. 2020 for cruise
    lines), where ``omega / (1 - alpha - beta)`` is not identified and was 70-260 times the
    sample variance on S&P 500 stocks, which makes simulated paths explode. Either way, a fit at
    the persistence bound is logged as a warning and listed in ``fit_info_["at_bound"]``.

    Filter initialisation (documented convention):

    * in-sample (``fit`` and ``loglik_obs(r)`` without ``r_history``): ``s_1^2`` is the sample
      variance (divisor ``T``) of the series being filtered;
    * out-of-sample (``loglik_obs(r_test, r_history=r_train)``): the filter is run on ``r_train``
      with the in-sample convention and continued, so the first test variance is the one-step
      forecast ``omega + alpha eps_T^2 + beta s_T^2`` from the end of ``r_train``;
    * an explicit ``sigma2_0`` overrides both (used e.g. for windows, which start at the
      unconditional variance ``omega / (1 - alpha - beta)``).

    Parameters
    ----------
    mu, omega, alpha, beta, nu : float, optional
        Give all five to build a fitted model.
    n_starts : int
        Number of starting points of the multistart optimiser.
    variance_targeting : bool, default True
        Tie ``omega`` to the sample variance in :meth:`fit` (see above); ``False`` gives the
        unrestricted maximum likelihood estimator.
    """

    name = "GARCHt"
    has_likelihood = True
    _PARAM_NAMES = ("mu", "omega", "alpha", "beta", "nu")
    _STARTS = ((0.05, 0.90, 8.0), (0.10, 0.85, 5.0), (0.15, 0.70, 6.0), (0.03, 0.96, 10.0),
               (0.08, 0.60, 4.0))

    def __init__(self, mu: float | None = None, omega: float | None = None,
                 alpha: float | None = None, beta: float | None = None, nu: float | None = None,
                 *, n_starts: int = 5, variance_targeting: bool = True) -> None:
        self.n_starts = int(n_starts)
        self.variance_targeting = bool(variance_targeting)
        self.fit_info_: dict[str, Any] = {}
        self._end: tuple[float, float] | None = None  # (eps_T, s2_T) at the end of the fit sample
        self._init_params({"mu": mu, "omega": omega, "alpha": alpha, "beta": beta, "nu": nu})

    def _validate(self) -> None:
        p = self._p
        if not (p["omega"] > 0 and p["alpha"] >= 0 and p["beta"] >= 0):
            raise ValueError("GARCHt: need omega > 0, alpha >= 0, beta >= 0")
        if not p["alpha"] + p["beta"] < 1:
            raise ValueError("GARCHt: need alpha + beta < 1 (covariance stationarity)")
        if not p["nu"] > 2:
            raise ValueError("GARCHt: need nu > 2")

    # -- fitting -------------------------------------------------------------------------
    def fit(self, r: np.ndarray) -> "GARCHt":
        """Maximum likelihood (SLSQP, analytic gradient) with multistart.

        Constraints: ``omega > 0``, ``alpha, beta >= 0``, ``alpha + beta <= 0.9999``,
        ``nu in [2.05, 100]``; with ``variance_targeting=True`` (default) also
        ``omega = (1 - alpha - beta) var(r)`` (sample variance, divisor ``T``). The optimiser runs
        on ``r / sd(r)`` (scale equivariance: ``mu`` and ``sqrt(omega)`` scale with the data,
        ``alpha, beta, nu`` are invariant).
        """
        t0 = time.perf_counter()
        r = _as_series(r, min_len=20)
        scale = float(r.std())
        if scale <= 0:
            raise ValueError("GARCHt.fit: constant series")
        x = r / scale
        s2_0 = float(x.var())
        T = x.size
        m0 = float(x.mean())
        vt = self.variance_targeting
        if vt:  # theta = (mu, alpha, beta, nu)
            fun, args = _garch_nll_grad_vt, (x, s2_0, s2_0)
            bounds = [(m0 - 1.0, m0 + 1.0), (0.0, _GARCH_PERSIST_MAX), (0.0, _GARCH_PERSIST_MAX),
                      _GARCH_NU_BOUNDS]
            ia, ib, jac = 1, 2, np.array([0.0, -1.0, -1.0, 0.0])
        else:  # theta = (mu, omega, alpha, beta, nu)
            fun, args = _garch_nll_grad, (x, s2_0)
            bounds = [(m0 - 1.0, m0 + 1.0), (1e-8, 10.0), (0.0, _GARCH_PERSIST_MAX),
                      (0.0, _GARCH_PERSIST_MAX), _GARCH_NU_BOUNDS]
            ia, ib, jac = 2, 3, np.array([0.0, 0.0, -1.0, -1.0, 0.0])
        cons = ({"type": "ineq", "fun": lambda th: _GARCH_PERSIST_MAX - th[ia] - th[ib],
                 "jac": lambda th: jac},)
        best = None
        n_used = 0
        for a0, b0, nu0 in self._STARTS[: max(1, self.n_starts)]:
            th0 = (np.array([m0, a0, b0, nu0]) if vt
                   else np.array([m0, s2_0 * (1.0 - a0 - b0), a0, b0, nu0]))
            n_used += 1
            try:
                res = optimize.minimize(fun, th0, args=args, jac=True, method="SLSQP",
                                        bounds=bounds, constraints=cons,
                                        options={"maxiter": 500, "ftol": 1e-11})
            except (ValueError, FloatingPointError) as exc:  # pragma: no cover - defensive
                logger.debug("GARCHt start failed: %s", exc)
                continue
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        if best is None:  # pragma: no cover - defensive
            raise RuntimeError("GARCHt.fit: all starts failed")
        if vt:
            mu_x, a, b, nu = best.x
        else:
            mu_x, om_x, a, b, nu = best.x
        a, b = max(float(a), 0.0), max(float(b), 0.0)
        if a + b > _GARCH_PERSIST_MAX:  # SLSQP may violate the linear constraint by rounding
            f = _GARCH_PERSIST_MAX / (a + b)
            a, b = a * f, b * f
        if vt:
            om_x = s2_0 * (1.0 - a - b)
        self._p = {"mu": float(mu_x * scale), "omega": float(om_x * scale ** 2), "alpha": a,
                   "beta": b, "nu": float(np.clip(nu, *_GARCH_NU_BOUNDS))}
        s2 = self.conditional_variance(r)
        self._end = (float(r[-1] - self._p["mu"]), float(s2[-1]))
        at_bound = [nm for nm, hit in (
            ("persistence", a + b >= _GARCH_PERSIST_MAX - 1e-6),
            ("alpha", a <= 1e-8), ("beta", b <= 1e-8),
            ("nu", self._p["nu"] <= _GARCH_NU_BOUNDS[0] + 1e-6
             or self._p["nu"] >= _GARCH_NU_BOUNDS[1] - 1e-6)) if hit]
        uncond_ratio = self.unconditional_variance() / float(r.var())
        if "persistence" in at_bound:
            logger.warning(
                "GARCHt.fit: alpha + beta at the bound %.4f (unconditional / sample variance = "
                "%.3g, variance_targeting=%s); simulated paths are close to integrated GARCH",
                _GARCH_PERSIST_MAX, uncond_ratio, vt)
        self.fit_info_ = {
            "success": bool(best.success),
            "message": str(best.message),
            "loglik": float(np.sum(self.loglik_obs(r))),
            "n_starts": n_used,
            "variance_targeting": vt,
            "persistence": a + b,
            "uncond_ratio": float(uncond_ratio),
            "at_bound": at_bound,
            "seconds": time.perf_counter() - t0,
        }
        return self

    # -- filtering -----------------------------------------------------------------------
    def unconditional_variance(self) -> float:
        p = self._require_fit()
        return p["omega"] / (1.0 - p["alpha"] - p["beta"])

    def _initial_variance(self, r: np.ndarray, r_history: np.ndarray | None,
                          sigma2_0: float | None) -> float:
        p = self._require_fit()
        if sigma2_0 is not None:
            if r_history is not None:
                raise ValueError("give r_history or sigma2_0, not both")
            if not sigma2_0 > 0:
                raise ValueError("sigma2_0 must be positive")
            return float(sigma2_0)
        if r_history is not None:
            h = _as_series(r_history, name="r_history")
            if h.size > 0:
                s2h = self.conditional_variance(h)
                e_last = h[-1] - p["mu"]
                return p["omega"] + p["alpha"] * e_last ** 2 + p["beta"] * float(s2h[-1])
        v = float(r.var()) if r.size >= 2 else 0.0
        return v if v > 0 else self.unconditional_variance()

    def conditional_variance(self, r: np.ndarray, r_history: np.ndarray | None = None, *,
                             sigma2_0: float | None = None) -> np.ndarray:
        """Filtered conditional variances ``s_t^2`` of ``r`` (see class docstring for the start)."""
        p = self._require_fit()
        x = _as_series(r)
        s2_0 = self._initial_variance(x, r_history, sigma2_0)
        return _garch_filter(x - p["mu"], p["omega"], p["alpha"], p["beta"], s2_0)

    def std_resid(self, r: np.ndarray, r_history: np.ndarray | None = None, *,
                  sigma2_0: float | None = None) -> np.ndarray:
        """Standardised residuals ``z_t = (r_t - mu) / s_t`` (unit variance under the model)."""
        x = _as_series(r)
        s2 = self.conditional_variance(x, r_history, sigma2_0=sigma2_0)
        return (x - self._p["mu"]) / np.sqrt(s2)

    def forecast_variance(self) -> float:
        """One-step-ahead variance after the end of the fit sample (unconditional if unknown)."""
        p = self._require_fit()
        if self._end is None:
            return self.unconditional_variance()
        e, s2 = self._end
        return p["omega"] + p["alpha"] * e * e + p["beta"] * s2

    def loglik_obs(self, r: np.ndarray, r_history: np.ndarray | None = None, *,
                   sigma2_0: float | None = None) -> np.ndarray:
        """Per-observation conditional log-densities.

        With ``r_history`` (the training returns) the filter continues from the end of
        ``r_history``: ``loglik_obs(r_test, r_history=r_train)`` equals the last ``len(r_test)``
        entries of ``loglik_obs(concat(r_train, r_test), sigma2_0=var(r_train))``.
        """
        p = self._require_fit()
        x = _as_series(r)
        s2 = self.conditional_variance(x, r_history, sigma2_0=sigma2_0)
        return _std_t_loglik_obs(x - p["mu"], s2, p["nu"])

    def loglik(self, r: np.ndarray, r_history: np.ndarray | None = None, *,
               sigma2_0: float | None = None) -> float:
        return float(np.sum(self.loglik_obs(r, r_history, sigma2_0=sigma2_0)))

    def pit(self, r: np.ndarray, r_history: np.ndarray | None = None, *,
            sigma2_0: float | None = None) -> np.ndarray:
        """Conditional CDF values ``u_t = F_nu(z_t)`` (same filter conventions as loglik_obs)."""
        nu = self._require_fit()["nu"]
        z = self.std_resid(r, r_history, sigma2_0=sigma2_0)
        return _clip_unit(stats.t.cdf(z * math.sqrt(nu / (nu - 2.0)), nu))

    def loglik_windows(self, windows: np.ndarray,
                       sigma2_init: np.ndarray | None = None) -> np.ndarray:
        """Log-density of each window of a stationary path, shape ``(n,)``.

        The exact density of a window ``(x_1, ..., x_L)`` cut from a stationary path is
        ``E_pi[prod_t f(x_t | s_t^2)]``, the expectation being over the stationary law ``pi`` of
        the conditional variance ``s_1^2`` at the first observation (independent of the
        window's innovations); it has no closed form.

        * ``sigma2_init=None`` (default): ``s_1^2`` is fixed at the unconditional variance
          ``omega / (1 - alpha - beta)``. This is an **approximation** of the window density
          (it ignores the dispersion of ``s_1^2``).
        * ``sigma2_init`` an array of ``K`` draws from ``pi`` (see
          :meth:`stationary_variance_draws`): Monte Carlo integration,
          ``log((1/K) sum_k prod_t f(x_t | s_t^2; s_1^2 = sigma2_init[k]))``, which converges
          to the exact log-density as ``K`` grows.
        """
        p = self._require_fit()
        w = _as_windows(windows)
        mu, om, a, b, nu = (p[k] for k in self._PARAM_NAMES)
        if sigma2_init is None:
            s2 = np.full(w.shape[0], self.unconditional_variance())
            ll = np.zeros(w.shape[0])
            for t in range(w.shape[1]):
                e = w[:, t] - mu
                ll += _std_t_loglik_obs(e, s2, nu)
                s2 = om + a * e * e + b * s2
            return ll
        s2k = np.asarray(sigma2_init, dtype=np.float64).ravel()
        if s2k.size == 0 or not np.all(np.isfinite(s2k)) or not np.all(s2k > 0):
            raise ValueError("sigma2_init must be a non-empty array of positive variances")
        out = np.empty(w.shape[0])
        chunk = max(1, 2_000_000 // s2k.size)  # bound memory: chunk x K arrays
        for i in range(0, w.shape[0], chunk):
            wc = w[i: i + chunk]
            s2 = np.broadcast_to(s2k, (wc.shape[0], s2k.size)).copy()
            ll = np.zeros_like(s2)
            for t in range(w.shape[1]):
                e = (wc[:, t] - mu)[:, None]
                ll += _std_t_loglik_obs(e, s2, nu)
                s2 = om + a * e * e + b * s2
            out[i: i + chunk] = logsumexp(ll, axis=1) - math.log(s2k.size)
        return out

    def stationary_variance_draws(self, n: int, *, rng: np.random.Generator,
                                  burn_in: int | None = None) -> np.ndarray:
        """Draws from the stationary law of the conditional variance ``s_t^2``, shape ``(n,)``.

        ``n`` independent variance recursions start at the unconditional variance and run for
        ``burn_in`` steps (default ``max(500, log(1e-8) / log(alpha + beta))``, at most 50,000),
        so the start is forgotten up to a factor ``(alpha + beta)^burn_in <= 1e-8`` in mean.
        """
        p = self._require_fit()
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        n = int(n)
        if n < 1:
            raise ValueError("n must be positive")
        pers = p["alpha"] + p["beta"]
        if burn_in is None:
            burn_in = 500 if pers <= 0 else int(min(50_000, max(500, math.ceil(
                math.log(1e-8) / math.log(pers)))))
        nu = p["nu"]
        zscale = math.sqrt((nu - 2.0) / nu)
        s2 = np.full(n, self.unconditional_variance())
        for _ in range(int(burn_in)):
            z = rng.standard_t(nu, size=n) * zscale
            s2 = p["omega"] + (p["alpha"] * z * z + p["beta"]) * s2
        return s2

    # -- simulation ----------------------------------------------------------------------
    def simulate(self, n_steps: int, n_paths: int, *, rng: np.random.Generator,
                 burn_in: int = 250, start_from_end: bool = False) -> np.ndarray:
        """Simulate ``(n_paths, n_steps)`` returns.

        ``start_from_end=True`` starts every path at the one-step-ahead variance forecast from
        the end of the fit sample (no burn-in); otherwise paths start at the unconditional
        variance and the first ``burn_in`` steps are discarded.
        """
        p = self._require_fit()
        _check_sim_args(n_steps, n_paths, rng)
        n_steps, n_paths = int(n_steps), int(n_paths)
        if start_from_end and self._end is None:
            logger.warning("GARCHt.simulate: no fitted end state, starting from stationarity")
        from_end = bool(start_from_end and self._end is not None)
        burn = 0 if from_end else max(int(burn_in), 0)
        s2 = np.full(n_paths, self.forecast_variance() if from_end else self.unconditional_variance())
        mu, om, a, b, nu = (p[k] for k in self._PARAM_NAMES)
        zscale = math.sqrt((nu - 2.0) / nu)
        out = np.empty((n_steps, n_paths))
        for t in range(burn + n_steps):  # innovations drawn step by step (bounded memory)
            eps = np.sqrt(s2) * (rng.standard_t(nu, size=n_paths) * zscale)
            if t >= burn:
                out[t - burn] = mu + eps
            s2 = om + a * eps * eps + b * s2
        return np.ascontiguousarray(out.T)


# --------------------------------------------------------------------------------------------
# Stationary bootstrap and block length selection
# --------------------------------------------------------------------------------------------
def optimal_block_length(x: np.ndarray, *, kind: str = "stationary",
                         fallback: float = 20.0) -> float:
    """Politis-White (2004) automatic block length with the Patton-Politis-White (2009) correction.

    ``b = (2 G^2 / D)^{1/3} n^{1/3}`` with ``G = sum_{|k|<=M} lambda(k/M) |k| R(k)``,
    ``g = sum_{|k|<=M} lambda(k/M) R(k)``, ``D_SB = 2 g^2`` (stationary bootstrap, corrected
    constant) or ``D_CB = 4/3 g^2`` (circular block bootstrap), flat-top kernel
    ``lambda(t) = min(1, 2(1 - |t|))_+``, and ``M = 2 m_hat`` where ``m_hat`` is the smallest
    lag after which ``K_n = max(5, ceil(sqrt(log10 n)))`` consecutive sample autocorrelations lie
    inside ``+-2 sqrt(log10(n) / n)`` (``M <= m_max = ceil(sqrt(n)) + K_n``). The result is
    capped at ``ceil(min(3 sqrt(n), n / 3))``.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Series whose dependence the bootstrap must preserve.
    kind : {"stationary", "circular"}
    fallback : float
        Returned when the rule is degenerate (too few observations, ``g <= 0``, ``G = 0`` or a
        non-finite value).

    Returns
    -------
    float
        Mean (stationary) or fixed (circular) block length, at least 1.
    """
    x = _as_series(x, name="x")
    n = x.size
    if kind not in ("stationary", "circular"):
        raise ValueError("kind must be 'stationary' or 'circular'")
    if n < 20:
        return float(fallback)
    e = x - x.mean()
    kn = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    m_max = min(int(math.ceil(math.sqrt(n))) + kn, n - 1)
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    acov = np.array([e[k:] @ e[: n - k] / n for k in range(m_max + 1)])
    if not acov[0] > 0:
        return float(fallback)
    rho = acov / acov[0]
    crit = 2.0 * math.sqrt(math.log10(n) / n)
    insig = np.abs(rho[1:]) < crit  # lags 1..m_max
    m_hat = None
    for j in range(1, m_max - kn + 2):  # run of kn insignificant lags starting at lag j
        if np.all(insig[j - 1: j - 1 + kn]):
            m_hat = j - 1
            break
    M = m_max if m_hat is None else min(2 * max(m_hat, 1), m_max)
    k = np.arange(1, M + 1)
    lam = np.minimum(1.0, 2.0 * (1.0 - k / M))
    G = 2.0 * np.sum(lam * k * acov[1: M + 1])
    g = acov[0] + 2.0 * np.sum(lam * acov[1: M + 1])
    D = 2.0 * g * g if kind == "stationary" else (4.0 / 3.0) * g * g
    if not (g > 0 and G != 0 and np.isfinite(G) and np.isfinite(D)):
        return float(fallback)
    b = (2.0 * G * G / D) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    if not np.isfinite(b):
        return float(fallback)
    return float(min(max(b, 1.0), b_max))


def stationary_bootstrap_indices(T: int, n_steps: int, n_paths: int, mean_block: float, *,
                                 rng: np.random.Generator) -> np.ndarray:
    """Indices of Politis-Romano (1994) stationary-bootstrap paths with circular wrapping.

    Each path starts at a uniform index; at every later step, with probability
    ``1 / mean_block`` a new block starts at a uniform index, otherwise the next index
    ``(i + 1) mod T`` is taken. Returns an int array of shape ``(n_paths, n_steps)``.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if T < 1 or mean_block < 1:
        raise ValueError("need T >= 1 and mean_block >= 1")
    p = 1.0 / float(mean_block)
    shape = (int(n_paths), int(n_steps))
    starts = rng.integers(0, T, size=shape)
    new = rng.random(shape) < p
    new[:, 0] = True
    t = np.arange(shape[1])
    t_start = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    return (np.take_along_axis(starts, t_start, axis=1) + (t - t_start)) % T


def _block_series(r: np.ndarray, how: str) -> np.ndarray:
    if how == "r":
        return r
    if how == "abs":
        return np.abs(r - r.mean())
    if how == "sq":
        return (r - r.mean()) ** 2
    raise ValueError("block_series must be 'r', 'abs' or 'sq'")


class StationaryBootstrap(ReturnModel):
    """Politis-Romano (1994) stationary bootstrap of the training returns (circular wrapping).

    Parameters
    ----------
    block_length : float, optional
        Mean block length. If ``None`` it is chosen by :func:`optimal_block_length` (Politis-White
        2004 rule with the Patton-Politis-White 2009 correction), falling back to 20 when the
        rule is degenerate.
    block_series : {"abs", "r", "sq"}
        Series on which the automatic rule is applied. The default ``"abs"`` (absolute demeaned
        returns) targets volatility clustering: daily returns are nearly serially uncorrelated,
        so the rule applied to ``r`` itself returns blocks of one to three days that destroy the
        dependence in ``|r|`` the bootstrap is meant to keep.
    min_block, max_block : float
        Bounds applied to the automatic block length.
    """

    name = "Bootstrap"
    has_likelihood = False

    def __init__(self, block_length: float | None = None, *, block_series: str = "abs",
                 min_block: float = 1.0, max_block: float = np.inf) -> None:
        if block_length is not None and not block_length >= 1:
            raise ValueError("block_length must be >= 1")
        _block_series(np.zeros(2), block_series)  # validate
        self.block_length = None if block_length is None else float(block_length)
        self.block_series = block_series
        self.min_block, self.max_block = float(min_block), float(max_block)
        self._r: np.ndarray | None = None
        self.block_length_: float | None = None

    def fit(self, r: np.ndarray) -> "StationaryBootstrap":
        r = _as_series(r, min_len=2)
        self._r = r.copy()
        if self.block_length is not None:
            b = self.block_length
        else:
            b = optimal_block_length(_block_series(r, self.block_series), kind="stationary")
            b = float(np.clip(b, self.min_block, self.max_block))
        self.block_length_ = float(min(max(b, 1.0), r.size))
        return self

    def simulate(self, n_steps: int, n_paths: int, *, rng: np.random.Generator,
                 burn_in: int = 250, start_from_end: bool = False) -> np.ndarray:
        """Resample ``(n_paths, n_steps)`` returns (stationary by construction: ``burn_in`` and
        ``start_from_end`` are ignored)."""
        if self._r is None:
            raise RuntimeError("StationaryBootstrap is not fitted")
        _check_sim_args(n_steps, n_paths, rng)
        idx = stationary_bootstrap_indices(self._r.size, n_steps, n_paths, self.block_length_,
                                           rng=rng)
        return self._r[idx]

    def params(self) -> dict[str, float]:
        if self.block_length_ is None:
            raise RuntimeError("StationaryBootstrap is not fitted")
        return {"block_length": self.block_length_}

    def n_params(self) -> int:
        return 1


# --------------------------------------------------------------------------------------------
# Heston: sub-stepped full-truncation Euler, simulated method of moments
# --------------------------------------------------------------------------------------------
SMM_MOMENT_NAMES = ("var", "mad_ratio", "log_kurt", "acf_sq_1", "acf_sq_5", "acf_sq_10",
                    "acf_sq_20", "lev_1")
_ACF_LAGS = (1, 5, 10, 20)


def smm_moments(R: np.ndarray) -> np.ndarray:
    """Moments used to calibrate :class:`Heston`, computed path by path.

    For each row of ``R`` (shape ``(S, T)`` or ``(T,)``), with ``x = r - mean(r)``:

    * ``var``: ``mean(x^2)``;
    * ``mad_ratio``: ``mean|x| / sd(x)`` (``sqrt(2 / pi) = 0.798`` under normality). It measures
      the mass in the centre of the distribution and so identifies the share of quiet days,
      which kurtosis does not: for a Gaussian scale mixture ``r = sqrt(V) Z`` it depends on
      ``E[V^{1/2}] / E[V]^{1/2}``, the kurtosis on ``E[V^2] / E[V]^2``;
    * ``log_kurt``: log of the (non-excess) kurtosis ``mean(x^4) / var^2``. On heavy-tailed
      series the raw kurtosis has a bootstrap standard deviation of the order of its value
      (a few extreme days dominate it), so a ``1 / Var*`` weight would all but ignore it; the
      log has a much less skewed sampling distribution;
    * ``acf_sq_k``: autocorrelation of ``x^2`` at lags 1, 5, 10, 20;
    * ``lev_1``: leverage correlation ``corr(x_t, x_{t+1}^2)``.

    Returns shape ``(S, 8)`` (or ``(8,)`` for 1-d input); see ``SMM_MOMENT_NAMES``.
    """
    R = np.asarray(R, dtype=np.float64)
    one = R.ndim == 1
    R = np.atleast_2d(R)
    if R.shape[1] <= max(_ACF_LAGS) + 1:
        raise ValueError("series too short for the SMM moments")
    with np.errstate(divide="ignore", invalid="ignore"):
        x = R - R.mean(axis=1, keepdims=True)
        s = x * x
        var = s.mean(axis=1)
        mad = np.abs(x).mean(axis=1) / np.sqrt(var)
        log_kurt = np.log((s * s).mean(axis=1) / var ** 2)
        sd = s - var[:, None]
        den = np.einsum("ij,ij->i", sd, sd)
        acfs = [np.einsum("ij,ij->i", sd[:, k:], sd[:, :-k]) / den for k in _ACF_LAGS]
        a = x[:, :-1] - x[:, :-1].mean(axis=1, keepdims=True)
        b = s[:, 1:] - s[:, 1:].mean(axis=1, keepdims=True)
        lev = np.einsum("ij,ij->i", a, b) / np.sqrt(np.einsum("ij,ij->i", a, a)
                                                     * np.einsum("ij,ij->i", b, b))
    out = np.column_stack([var, mad, log_kurt, *acfs, lev])
    return out[0] if one else out


def _cir_unit_integrals(kappa: float | np.ndarray, c: float | np.ndarray, z: np.ndarray,
                        v0: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily integrals of a unit-mean CIR variance, full-truncation Euler with sub-steps.

    Unit-mean process ``dv = kappa (1 - v) dt + c sqrt(2 kappa) sqrt(v) dW`` (``c`` is the
    coefficient of variation of its stationary Gamma law). Each day is split into
    ``M = z.shape[1]`` steps of length ``h = 1 / M`` with full truncation (Lord, Koekkoek and
    van Dijk 2010): ``v_{k+1} = v_k + kappa (1 - v_k^+) h + c sqrt(2 kappa h) sqrt(v_k^+) z_k``.

    Parameters
    ----------
    kappa, c : float or ndarray of shape (P,)
        Parameter sets simulated side by side (they share the shocks ``z``).
    z : ndarray, shape (n_days, M, S)
        Standard normal variance shocks.
    v0 : float or ndarray broadcastable to (P, S)
        Initial unit-scale variance.

    Returns
    -------
    I : ndarray, shape (n_days, P, S)
        Integrated variance of each day, ``sum_k v_k^+ h``.
    J : ndarray, shape (n_days, P, S)
        ``sum_k sqrt(v_k^+ h) z_k``, the Euler sum for ``int sqrt(v) dW`` over each day.
    v : ndarray, shape (P, S)
        Variance at the end of the last day.
    """
    kappa = np.atleast_1d(np.asarray(kappa, dtype=np.float64)).reshape(-1, 1)
    c = np.atleast_1d(np.asarray(c, dtype=np.float64)).reshape(-1, 1)
    n_days, M, S = z.shape
    P = max(kappa.shape[0], c.shape[0])
    h = 1.0 / M
    kh = np.broadcast_to(kappa * h, (P, 1)).copy()
    xs = np.broadcast_to(c * np.sqrt(2.0 * kappa * h), (P, 1)).copy()
    v = np.array(np.broadcast_to(np.asarray(v0, dtype=np.float64), (P, S)))
    I = np.empty((n_days, P, S))
    J = np.empty((n_days, P, S))
    vp, sq = np.empty((P, S)), np.empty((P, S))
    Ia, Ja = np.empty((P, S)), np.empty((P, S))
    sqrt_h = math.sqrt(h)
    for t in range(n_days):
        Ia.fill(0.0)
        Ja.fill(0.0)
        zt = z[t]
        for k in range(M):
            np.maximum(v, 0.0, out=vp)
            Ia += vp
            np.sqrt(vp, out=sq)
            sq *= zt[k]          # sqrt(v+) z
            Ja += sq
            sq *= xs             # xi sqrt(h) sqrt(v+) z (unit scale)
            vp *= kh             # kappa h v+
            v -= vp
            v += sq
            v += kh              # kappa theta h with theta = 1
        np.multiply(Ia, h, out=I[t])
        np.multiply(Ja, sqrt_h, out=J[t])
    return I, J, v


def _heston_returns(I: np.ndarray, J: np.ndarray, A: np.ndarray, theta: float, rho: float,
                    mu: float = 0.0) -> np.ndarray:
    """Daily log-returns ``mu - theta I / 2 + sqrt(theta) (rho J + sqrt(1 - rho^2) A)``.

    ``I, J`` are unit-scale integrals from :func:`_cir_unit_integrals` and ``A = sqrt(I) Z``
    with ``Z`` independent standard normals (the orthogonal Brownian part, which is exactly
    ``N(0, I)`` given the variance path).
    """
    return mu - 0.5 * theta * I + math.sqrt(theta) * (rho * J + math.sqrt(1.0 - rho * rho) * A)


def _stationary_unit_variance(c: float | np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF draws ``(P, S)`` from the unit-mean stationary Gamma(1/c^2, c^2) law."""
    c = np.atleast_1d(np.asarray(c, dtype=np.float64)).reshape(-1, 1)
    u = np.asarray(u, dtype=np.float64).reshape(1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = c * c * gammaincinv(1.0 / (c * c), u)
    return np.where(c < 1e-6, 1.0, out)


def _ewma_variance(r: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA variance at the end of ``r`` (started at the sample variance)."""
    e = r - r.mean()
    v = lfilter([1.0 - lam], [1.0, -lam], e * e, zi=[lam * float(e.var())])[0]
    return float(v[-1])


class Heston(_ParamMixin, ReturnModel):
    """Heston (1993) stochastic volatility for daily log-returns (``dt = 1`` trading day).

    ``d log S = (mu - v / 2) dt + sqrt(v) dW1``, ``dv = kappa (theta - v) dt + xi sqrt(v) dW2``,
    ``d<W1, W2> = rho dt``. All parameters are daily (``theta`` a daily variance, ``kappa`` a
    per-day rate). ``c = xi / sqrt(2 kappa theta)`` is the coefficient of variation of the
    stationary Gamma law of ``v``; the Feller ratio is ``2 kappa theta / xi^2 = 1 / c^2``.

    Simulation (same scheme in :meth:`fit` and :meth:`simulate`). Each day is split into
    ``n_substeps`` full-truncation Euler steps for the variance (Lord, Koekkoek and van Dijk
    2010), and the daily log-return is generated from the variance path's integrals,
    ``r_t = mu - I_t / 2 + rho J_t + sqrt((1 - rho^2) I_t) Z_t`` with ``I_t = int v`` and
    ``J_t = int sqrt(v) dW2`` over the day (Euler sums) and ``Z_t`` an independent N(0, 1): given
    the variance path, the orthogonal Brownian part integrates exactly to ``N(0, I_t)``
    (the decomposition used by Broadie and Kaya 2006). Paths start from the exact stationary
    Gamma law of ``v``. A one-step daily Euler scheme is **not** used: away from the Feller
    condition its truncated variance sticks at zero for days or weeks, and the returns are then
    exactly constant (36-52% of days on fitted S&P 500 models). Against the exact
    (non-central chi-square) transition, 20 sub-steps give a Kolmogorov distance of 0.01-0.04
    for the law of the daily integrated variance and a share of days with zero integrated
    variance of 0.04% at ``c = 2`` and 0.4-0.6% at ``c = 2.5`` (the upper bound of ``c``).

    Calibration: simulated method of moments (Duffie and Singleton 1993) on
    :func:`smm_moments` (variance, ``mean|x| / sd``, log kurtosis, ACF of ``r^2`` at lags 1, 5,
    10, 20, ``corr(r_t, r_{t+1}^2)``):

    * simulated moments are averages over ``S`` paths of the same length as the data (so
      finite-sample biases of the estimators are matched), with common random numbers;
    * the weighting matrix is diagonal, ``W = diag(1 / Var*(m_j))``, from stationary-bootstrap
      variances of the data moments (mean block length ``max(20, PW(|r|))``);
    * profiling: by the scaling property of the CIR process (``v = theta v1`` with ``v1`` of
      unit mean), the simulated variance paths depend on ``(kappa, c)`` only, and ``theta`` and
      ``rho`` enter only when the returns are assembled. The objective is minimised over
      ``(log theta, rho)`` (bounded Nelder-Mead, cheap) for each ``(kappa, c)``, and the profile
      over ``(log kappa, log c)``: a batched 8 x 8 grid (``kappa`` 0.003-0.4,
      ``c`` 0.3-2.5), then bounded Nelder-Mead from the ``n_nm_starts`` best grid points and a
      restart from the best solution;
    * ``mu`` is then set so that the simulated mean return equals the sample mean.

    Bounds: ``theta in [0.25, 4] var(r)``, ``kappa in [1e-3, 0.5]``, ``c in [0.05, 2.5]``,
    ``rho in [-0.95, 0.95]``. The Feller condition (``c <= 1``) is not imposed.
    ``fit_info_`` reports the data and simulated moments (``m_data``, ``m_sim``, and the
    weighted deviations ``t``), ``frac_zero_var`` (share of simulated days with zero
    integrated variance, i.e. exactly constant returns: a discretisation artefact that should
    be ~0) and ``frac_low_var`` (share of days whose integrated variance is below 1% of
    ``theta``: a property of the fitted model, large when ``c`` is large).

    ``simulate(start_from_end=True)`` starts from a RiskMetrics EWMA (lambda = 0.94) estimate of
    the variance at the end of the fit sample (the Heston variance is latent; this is a
    documented proxy), otherwise from the stationary law with ``burn_in`` discarded days.

    Parameters
    ----------
    mu, kappa, theta, xi, rho : float, optional
        Give all five to build a fitted model.
    n_substeps : int
        Euler sub-steps per day (fit and simulation).
    sim_obs : int
        Target number of simulated observations per objective evaluation; the number of
        simulated paths is ``clip(sim_obs // T, min_paths, max_paths)``.
    burn_in_smm : int
        Simulated days discarded before each SMM path (paths already start from the
        stationary law, so the default is 0).
    n_boot : int
        Bootstrap replicates for the weighting matrix.
    maxfev : int
        Maximum profile evaluations of each outer Nelder-Mead run.
    n_nm_starts : int
        Number of outer Nelder-Mead runs, started from the best grid points.
    seed : int
        Seed of the default generator used by :meth:`fit` when no ``rng`` is passed.
    """

    name = "Heston"
    has_likelihood = False
    _PARAM_NAMES = ("mu", "kappa", "theta", "xi", "rho")
    KAPPA_BOUNDS = (1e-3, 0.5)
    C_BOUNDS = (0.05, 2.5)
    RHO_BOUNDS = (-0.95, 0.95)
    THETA_BOUNDS_REL = (0.25, 4.0)
    _KAPPA_GRID = tuple(np.geomspace(0.003, 0.4, 8))
    _C_GRID = (0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5)
    _RHO_SCAN = (-0.8, -0.5, -0.2, 0.1, 0.4)

    def __init__(self, mu: float | None = None, kappa: float | None = None,
                 theta: float | None = None, xi: float | None = None, rho: float | None = None,
                 *, n_substeps: int = 20, sim_obs: int = 50_000, min_paths: int = 10,
                 max_paths: int = 50, burn_in_smm: int = 0, n_boot: int = 200,
                 maxfev: int = 120, n_nm_starts: int = 2, seed: int = 20251114) -> None:
        if int(n_substeps) < 1:
            raise ValueError("n_substeps must be >= 1")
        self.n_substeps = int(n_substeps)
        self.n_nm_starts = int(n_nm_starts)
        self.sim_obs, self.min_paths, self.max_paths = int(sim_obs), int(min_paths), int(max_paths)
        self.burn_in_smm, self.n_boot, self.maxfev, self.seed = (int(burn_in_smm), int(n_boot),
                                                                  int(maxfev), int(seed))
        self.fit_info_: dict[str, Any] = {}
        self.v_end_: float | None = None
        self._init_params({"mu": mu, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho})

    def _validate(self) -> None:
        p = self._p
        if not (p["kappa"] > 0 and p["theta"] > 0 and p["xi"] >= 0 and -1 < p["rho"] < 1):
            raise ValueError("Heston: need kappa > 0, theta > 0, xi >= 0, |rho| < 1")
        if p["kappa"] / self.n_substeps > 0.5:
            raise ValueError("Heston: need kappa / n_substeps <= 0.5 (Euler stability)")

    @staticmethod
    def _xi(theta: float, kappa: float, c: float) -> float:
        """Vol-of-variance ``xi = c sqrt(2 kappa theta)`` for a stationary coefficient of variation c."""
        return c * math.sqrt(2.0 * kappa * theta)

    def cv(self) -> float:
        """Coefficient of variation ``c = xi / sqrt(2 kappa theta)`` of the stationary variance."""
        p = self._require_fit()
        return p["xi"] / math.sqrt(2.0 * p["kappa"] * p["theta"])

    # -- calibration ---------------------------------------------------------------------
    def smm_problem(self, r: np.ndarray, *,
                    rng: np.random.Generator | None = None) -> "_HestonSMM":
        """The SMM objective of :meth:`fit` for the series ``r`` (same random numbers as ``fit``
        with the same ``rng``); exposed to check the optimiser against brute force."""
        r = _as_series(r, min_len=100)
        rng = np.random.default_rng(self.seed) if rng is None else rng
        return _HestonSMM(self, r, rng)

    def fit(self, r: np.ndarray, *, rng: np.random.Generator | None = None) -> "Heston":
        """Calibrate by SMM (see class docstring). ``rng`` drives the common random numbers and
        the bootstrap; by default ``np.random.default_rng(self.seed)`` (deterministic)."""
        t0 = time.perf_counter()
        r = _as_series(r, min_len=100)
        prob = self.smm_problem(r, rng=rng)
        lo, hi = prob.lo, prob.hi
        # batched grid over (kappa, c): theta matches the variance, rho from a coarse scan
        pts = np.clip(np.log([(k, c) for k in self._KAPPA_GRID for c in self._C_GRID]), lo, hi)
        grid_vals = prob.grid_values(pts)
        steps = np.array([0.35, 0.15])

        def outer_nm(u0: np.ndarray, step: np.ndarray) -> optimize.OptimizeResult:
            u0 = np.clip(u0, lo, hi)
            simplex = np.vstack([u0] + [np.clip(u0 + np.eye(2)[i] * step[i] *
                                                (1 if u0[i] + step[i] <= hi[i] else -1), lo, hi)
                                        for i in range(2)])
            return optimize.minimize(prob.profile, u0, method="Nelder-Mead",
                                     bounds=list(zip(lo, hi)),
                                     options={"maxfev": self.maxfev, "initial_simplex": simplex,
                                              "xatol": 0.01, "fatol": 1e-3})

        order = np.argsort(grid_vals)
        runs = [outer_nm(pts[i], steps) for i in order[: max(1, self.n_nm_starts)]]
        res = min(runs, key=lambda o: o.fun)
        res_restart = outer_nm(res.x, steps / 3.0)
        if res_restart.fun <= res.fun:
            res = res_restart
        u = np.clip(res.x, lo, hi)
        obj, lth, rho = prob.profile_full(u)
        kappa, c, theta = math.exp(u[0]), math.exp(u[1]), math.exp(lth)
        xi = self._xi(theta, kappa, c)
        I, J = prob.unit_paths(np.array([kappa]), np.array([c]))
        I, J = I[:, 0], J[:, 0]
        ret = _heston_returns(I, J, np.sqrt(I) * prob.zr, theta, rho)
        mu = float(r.mean() - ret.mean())  # ret was simulated with mu = 0
        self._p = {"mu": mu, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho}
        self.v_end_ = _ewma_variance(r)
        m_sim = smm_moments(ret.T).mean(axis=0)
        at_bound = [nm for nm, v, a, bb in (
            ("theta", lth, prob.lth_lo, prob.lth_hi), ("kappa", u[0], lo[0], hi[0]),
            ("c", u[1], lo[1], hi[1]), ("rho", rho, *self.RHO_BOUNDS))
            if abs(v - a) < 1e-6 or abs(v - bb) < 1e-6]
        ib = int(order[0])
        self.fit_info_ = {
            "success": bool(res.success),
            "message": str(res.message),
            "objective": float(obj),
            "n_eval": len(prob.cache),
            "n_inner_eval": prob.n_inner,
            "n_paths": prob.S,
            "n_substeps": self.n_substeps,
            "block_length_boot": prob.block_length,
            "moment_names": SMM_MOMENT_NAMES,
            "m_data": prob.m_data,
            "m_sim": m_sim,
            "weights": prob.w,
            "t": np.sqrt(prob.w) * (m_sim - prob.m_data),
            "c": c,
            "feller_ratio": 1.0 / (c * c),
            "feller_ok": bool(c <= 1.0),
            "frac_zero_var": float(np.mean(I <= 0.0)),
            "frac_low_var": float(np.mean(I < 0.01)),
            "grid_best": {"objective": float(grid_vals[ib]), "kappa": float(np.exp(pts[ib, 0])),
                          "c": float(np.exp(pts[ib, 1]))},
            "at_bound": at_bound,
            "seconds": time.perf_counter() - t0,
        }
        if at_bound:
            logger.info("Heston.fit: parameters at a bound: %s", at_bound)
        if self.fit_info_["frac_zero_var"] > 0.005:
            logger.warning("Heston.fit: %.2f%% of simulated days have zero integrated variance; "
                           "increase n_substeps", 100 * self.fit_info_["frac_zero_var"])
        return self

    # -- simulation ----------------------------------------------------------------------
    def simulate(self, n_steps: int, n_paths: int, *, rng: np.random.Generator,
                 burn_in: int = 250, start_from_end: bool = False) -> np.ndarray:
        """Simulate ``(n_paths, n_steps)`` daily log-returns (sub-stepped scheme, class docstring).

        Without ``start_from_end`` the variance starts from its stationary Gamma law and the
        first ``burn_in`` days are discarded.
        """
        p = self._require_fit()
        _check_sim_args(n_steps, n_paths, rng)
        n_steps, n_paths = int(n_steps), int(n_paths)
        from_end = bool(start_from_end and self.v_end_ is not None)
        if start_from_end and not from_end:
            logger.warning("Heston.simulate: no end-of-sample variance, starting from stationarity")
        burn = 0 if from_end else max(int(burn_in), 0)
        theta, kappa, rho = p["theta"], p["kappa"], p["rho"]
        c = self.cv()
        if from_end:
            v = np.full((1, n_paths), self.v_end_ / theta)
        elif c < 1e-6:
            v = np.ones((1, n_paths))
        else:
            v = rng.gamma(1.0 / (c * c), c * c, size=(1, n_paths))
        n_days, M = burn + n_steps, self.n_substeps
        I = np.empty((n_days, n_paths))
        J = np.empty((n_days, n_paths))
        chunk = max(1, 2_000_000 // (M * n_paths))  # bound the memory of the shocks
        for s in range(0, n_days, chunk):
            nd = min(chunk, n_days - s)
            Ic, Jc, v = _cir_unit_integrals(kappa, c, rng.standard_normal((nd, M, n_paths)), v)
            I[s: s + nd], J[s: s + nd] = Ic[:, 0], Jc[:, 0]
        I, J = I[burn:], J[burn:]
        A = np.sqrt(I) * rng.standard_normal((n_steps, n_paths))
        return np.ascontiguousarray(_heston_returns(I, J, A, theta, rho, p["mu"]).T)


class _HestonSMM:
    """SMM objective of :class:`Heston` for one series (data moments, weights, common randoms).

    Attributes ``m_data``, ``w`` (diagonal weights), ``S`` (simulated paths), ``zr`` (return
    shocks), ``lo``/``hi`` (bounds of ``(log kappa, log c)``), ``lth_lo``/``lth_hi`` (bounds of
    ``log theta``); ``cache`` maps evaluated ``(log kappa, log c)`` to
    ``(objective, log theta, rho)`` and ``n_inner`` counts objective evaluations.
    """

    def __init__(self, model: "Heston", r: np.ndarray, rng: np.random.Generator) -> None:
        T = r.size
        self.model = model
        self.m_data = smm_moments(r)
        if not np.all(np.isfinite(self.m_data)):
            raise ValueError("Heston.fit: moments of r are not finite (constant series?)")
        # weighting matrix from stationary-bootstrap variances of the data moments
        b = max(20.0, optimal_block_length(np.abs(r - r.mean()), kind="stationary"))
        n_boot = max(2, model.n_boot)
        m_boot = np.vstack([  # in chunks of 50 replicates to bound memory
            smm_moments(r[stationary_bootstrap_indices(T, T, min(50, n_boot - i), b, rng=rng)])
            for i in range(0, n_boot, 50)])
        vb = np.var(m_boot, axis=0, ddof=1)
        ok = np.isfinite(vb) & (vb > 0)
        vb = np.where(ok, vb, np.max(vb[ok]) if np.any(ok) else 1.0)
        self.block_length = b
        self.w = 1.0 / vb
        # common random numbers
        self.S = int(np.clip(model.sim_obs // T, model.min_paths, model.max_paths))
        self.nb = model.burn_in_smm
        self.z = rng.standard_normal((self.nb + T, model.n_substeps, self.S))
        self.zr = rng.standard_normal((T, self.S))
        self.u0 = rng.random(self.S)
        self.var0 = float(r.var())
        self.lth_lo = math.log(model.THETA_BOUNDS_REL[0] * self.var0)
        self.lth_hi = math.log(model.THETA_BOUNDS_REL[1] * self.var0)
        self.rho_bounds = model.RHO_BOUNDS
        self.lo = np.log([model.KAPPA_BOUNDS[0], model.C_BOUNDS[0]])
        self.hi = np.log([model.KAPPA_BOUNDS[1], model.C_BOUNDS[1]])
        self.cache: dict[tuple[float, float], tuple[float, float, float]] = {}
        self.n_inner = 0

    def unit_paths(self, kappa: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unit-scale daily integrals ``(I, J)``, shape ``(T, P, S)``, for parameter arrays."""
        I, J, _ = _cir_unit_integrals(kappa, c, self.z, _stationary_unit_variance(c, self.u0))
        return I[self.nb:], J[self.nb:]

    def q(self, I: np.ndarray, J: np.ndarray, A: np.ndarray, lth: float, rho: float) -> float:
        """Weighted squared moment distance at ``theta = exp(lth)``, ``rho`` (paths ``(T, S)``)."""
        self.n_inner += 1
        lth = min(max(lth, self.lth_lo), self.lth_hi)
        rho = min(max(rho, self.rho_bounds[0]), self.rho_bounds[1])
        g = smm_moments(_heston_returns(I, J, A, math.exp(lth), rho).T).mean(axis=0) - self.m_data
        val = float(np.sum(self.w * g * g))
        return val if np.isfinite(val) else 1e12

    def start_theta_rho(self, I: np.ndarray, J: np.ndarray,
                        A: np.ndarray) -> tuple[float, float, float]:
        """Variance-matching ``log theta`` and the best ``rho`` of a coarse scan."""
        lth = math.log(self.var0 / max(float(I.mean()), 1e-300))
        lth = min(max(lth, self.lth_lo), self.lth_hi)
        val, rho = min((self.q(I, J, A, lth, rh), rh) for rh in self.model._RHO_SCAN)
        return val, lth, rho

    def inner(self, I: np.ndarray, J: np.ndarray) -> tuple[float, float, float]:
        """Minimum over ``(log theta, rho)`` for fixed variance paths: ``(value, lth, rho)``."""
        A = np.sqrt(I) * self.zr
        _, lth0, rho0 = self.start_theta_rho(I, J, A)
        simplex = np.array([[lth0, rho0], [lth0 + 0.1, rho0],
                            [lth0, rho0 + (0.15 if rho0 < 0.7 else -0.15)]])
        res = optimize.minimize(lambda u: self.q(I, J, A, u[0], u[1]), simplex[0],
                                method="Nelder-Mead",
                                bounds=[(self.lth_lo, self.lth_hi), self.rho_bounds],
                                options={"initial_simplex": simplex, "xatol": 1e-4,
                                         "fatol": 1e-6, "maxfev": 400})
        lth = min(max(float(res.x[0]), self.lth_lo), self.lth_hi)
        rho = min(max(float(res.x[1]), self.rho_bounds[0]), self.rho_bounds[1])
        return float(res.fun), lth, rho

    def profile_full(self, u: np.ndarray) -> tuple[float, float, float]:
        """Profile objective at ``u = (log kappa, log c)``: ``(value, log theta, rho)``."""
        u = np.clip(np.asarray(u, dtype=np.float64), self.lo, self.hi)
        key = (float(u[0]), float(u[1]))
        if key not in self.cache:
            I, J = self.unit_paths(np.exp(u[:1]), np.exp(u[1:]))
            self.cache[key] = self.inner(I[:, 0], J[:, 0])
        return self.cache[key]

    def profile(self, u: np.ndarray) -> float:
        """Profile objective value at ``u = (log kappa, log c)``."""
        return self.profile_full(u)[0]

    def grid_values(self, pts: np.ndarray, chunk: int = 16) -> np.ndarray:
        """Cheap objective (variance-matching theta, scanned rho) at points ``(log kappa, log c)``,
        simulated in batches of ``chunk`` parameter sets that share the random numbers."""
        vals = np.empty(len(pts))
        for i in range(0, len(pts), chunk):
            P = pts[i: i + chunk]
            I, J = self.unit_paths(np.exp(P[:, 0]), np.exp(P[:, 1]))
            for j in range(P.shape[0]):
                vals[i + j] = self.start_theta_rho(I[:, j], J[:, j], np.sqrt(I[:, j]) * self.zr)[0]
        return vals
