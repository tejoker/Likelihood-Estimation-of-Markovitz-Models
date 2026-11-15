"""Multivariate models of daily log-returns (DESIGN section 3.6).

All models work with daily log-returns of ``N`` assets under the real-world measure, arrays of
shape ``(T, N)``, ``dt = 1`` trading day, and no annualisation anywhere.

Models
------
``MVGaussian``
    i.i.d. ``N(mu, Sigma)`` with the sample mean and the (unbiased) sample covariance.
``MVStudentT``
    i.i.d. multivariate Student-t ``t_nu(mu, Sigma)`` (``Sigma`` is the scatter matrix,
    ``Cov = nu / (nu - 2) * Sigma``), fitted by ECME (Liu and Rubin, 1995) with
    ``nu`` restricted to ``[2.1, 200]``.
``CCCGARCHt``
    Constant-conditional-correlation GARCH(1,1) (Bollerslev, 1990) whose margins are the
    univariate ``GARCHt`` models of :mod:`synthval.models.univariate`, coupled by a Gaussian
    copula on the standardised innovations (see the class docstring for the reason).
``MVStationaryBootstrap``
    Politis and Romano (1994) stationary bootstrap applied to whole rows of the training
    panel (the same random blocks for every asset), which preserves the cross-sectional
    dependence exactly. Its block-length rule and index generator are those of the univariate
    ``StationaryBootstrap`` (one implementation, one convention for E3 and E4).

References
----------
Bollerslev, T. (1990). Modelling the coherence in short-run nominal exchange rates: a
    multivariate generalized ARCH model. *Review of Economics and Statistics* 72(3), 498-505.
Joe, H. (2005). Asymptotic efficiency of the two-stage estimation method for copula-based
    models. *Journal of Multivariate Analysis* 94(2), 401-419.
Liu, C. and Rubin, D. B. (1995). ML estimation of the t distribution using EM and its
    extensions, ECM and ECME. *Statistica Sinica* 5(1), 19-39.
Patton, A., Politis, D. N. and White, H. (2009). Correction to "Automatic block-length
    selection for the dependent bootstrap". *Econometric Reviews* 28(4), 372-375.
Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American
    Statistical Association* 89(428), 1303-1313.
Politis, D. N. and White, H. (2004). Automatic block-length selection for the dependent
    bootstrap. *Econometric Reviews* 23(1), 53-70.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
from scipy import linalg, optimize, special

from .base import MultiReturnModel, ReturnModel

logger = logging.getLogger(__name__)

#: Admissible range of the Student-t degrees of freedom in ``MVStudentT``.
NU_BOUNDS: tuple[float, float] = (2.1, 200.0)

_LOG_2PI = float(np.log(2.0 * np.pi))
_TINY = 1e-300

__all__ = [
    "MVGaussian",
    "MVStudentT",
    "CCCGARCHt",
    "MVStationaryBootstrap",
    "stationary_bootstrap_indices",
    "NU_BOUNDS",
]


# ----------------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------------
def _check_returns(R: np.ndarray, min_obs: int = 2) -> np.ndarray:
    """Validate a ``(T, N)`` panel of returns and return it as float64."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"returns must have shape (T, N); got ndim={R.ndim}")
    if R.shape[1] < 1:
        raise ValueError("returns must have at least one column")
    if R.shape[0] < min_obs:
        raise ValueError(f"need at least {min_obs} observations, got {R.shape[0]}")
    if not np.all(np.isfinite(R)):
        raise ValueError("returns contain NaN or inf")
    return R


def _check_n_steps(n_steps: int) -> int:
    n = int(n_steps)
    if n < 1 or n != n_steps:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    return n


def _cholesky(S: np.ndarray, what: str = "matrix") -> np.ndarray:
    """Lower Cholesky factor of a symmetric positive-definite matrix (clear error otherwise)."""
    try:
        return linalg.cholesky(S, lower=True)
    except linalg.LinAlgError as exc:
        raise linalg.LinAlgError(f"{what} is not positive definite") from exc


def _mahalanobis_sq(X: np.ndarray, mu: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distances ``(x_t - mu)' (L L')^{-1} (x_t - mu)``, shape ``(T,)``."""
    Z = linalg.solve_triangular(L, (X - mu).T, lower=True, check_finite=False)
    return np.einsum("ij,ij->j", Z, Z)


def _logdet_from_chol(L: np.ndarray) -> float:
    return float(2.0 * np.sum(np.log(np.diag(L))))


def _require_fitted(model: object, attr: str) -> None:
    if not hasattr(model, attr):
        raise RuntimeError(f"{type(model).__name__} is not fitted; call fit() first")


# ----------------------------------------------------------------------------------------------
# MVGaussian
# ----------------------------------------------------------------------------------------------
class MVGaussian(MultiReturnModel):
    """i.i.d. multivariate Gaussian returns ``r_t ~ N(mu, Sigma)``.

    Parameters
    ----------
    ddof : int, default 1
        Delta degrees of freedom of the covariance estimator (1: unbiased sample covariance,
        0: Gaussian MLE).
    """

    name = "MVGaussian"
    has_likelihood = True

    def __init__(self, ddof: int = 1) -> None:
        self.ddof = int(ddof)

    def fit(self, R: np.ndarray) -> "MVGaussian":
        """Estimate the sample mean and covariance from ``R`` of shape ``(T, N)``."""
        R = _check_returns(R, min_obs=2)
        self.n_assets_ = R.shape[1]
        self.mu_ = R.mean(axis=0)
        self.Sigma_ = np.atleast_2d(np.cov(R, rowvar=False, ddof=self.ddof))
        self._L = _cholesky(self.Sigma_, "sample covariance (T <= N or collinear assets?)")
        return self

    def simulate(
        self,
        n_steps: int,
        *,
        rng: np.random.Generator,
        burn_in: int = 250,
        start_from_end: bool = False,
    ) -> np.ndarray:
        """Simulate ``(n_steps, N)`` i.i.d. Gaussian returns (``burn_in``/``start_from_end`` unused)."""
        _require_fitted(self, "mu_")
        n = _check_n_steps(n_steps)
        Z = rng.standard_normal((n, self.n_assets_))
        return self.mu_ + Z @ self._L.T

    def loglik_obs(self, R: np.ndarray, R_history: np.ndarray | None = None) -> np.ndarray:
        """Per-observation log-densities, shape ``(T,)`` (``R_history`` unused: i.i.d. model)."""
        _require_fitted(self, "mu_")
        R = _check_returns(R, min_obs=1)
        d2 = _mahalanobis_sq(R, self.mu_, self._L)
        return -0.5 * (self.n_assets_ * _LOG_2PI + _logdet_from_chol(self._L) + d2)

    def loglik(self, R: np.ndarray, R_history: np.ndarray | None = None) -> float:
        """Total log-likelihood of ``R``."""
        return float(np.sum(self.loglik_obs(R, R_history)))

    def params(self) -> dict:
        """Fitted parameters ``{"mu": (N,), "Sigma": (N, N)}``."""
        _require_fitted(self, "mu_")
        return {"mu": self.mu_.copy(), "Sigma": self.Sigma_.copy()}

    def n_params(self) -> int:
        N = self.n_assets_
        return N + N * (N + 1) // 2


# ----------------------------------------------------------------------------------------------
# MVStudentT
# ----------------------------------------------------------------------------------------------
def _t_profile_nu(nu: float, d2: np.ndarray, p: int) -> float:
    """Mean multivariate-t log-density as a function of ``nu`` (terms free of ``nu`` dropped)."""
    return float(
        special.gammaln(0.5 * (nu + p))
        - special.gammaln(0.5 * nu)
        - 0.5 * p * np.log(nu)
        - 0.5 * (nu + p) * np.mean(np.log1p(d2 / nu))
    )


def _t_logpdf(d2: np.ndarray, nu: float, p: int, logdet: float) -> np.ndarray:
    """Multivariate-t log-density given squared Mahalanobis distances ``d2``."""
    return (
        special.gammaln(0.5 * (nu + p))
        - special.gammaln(0.5 * nu)
        - 0.5 * p * np.log(nu * np.pi)
        - 0.5 * logdet
        - 0.5 * (nu + p) * np.log1p(d2 / nu)
    )


class MVStudentT(MultiReturnModel):
    """i.i.d. multivariate Student-t returns ``r_t ~ t_nu(mu, Sigma)``.

    Density ``f(x) = Gamma((nu+p)/2) / (Gamma(nu/2) (nu pi)^{p/2} |Sigma|^{1/2})
    * (1 + (x-mu)' Sigma^{-1} (x-mu) / nu)^{-(nu+p)/2}``; ``Cov(r_t) = nu/(nu-2) Sigma``.

    Fitted by ECME (Liu and Rubin, 1995). With the latent scale ``tau_t ~ Gamma(nu/2, nu/2)``,
    each iteration does

    1. E-step: ``w_t = (nu + p) / (nu + delta_t)`` with ``delta_t`` the squared Mahalanobis
       distance under the current ``(mu, Sigma)``;
    2. CM-step 1 (EM): ``mu = sum w_t x_t / sum w_t`` and
       ``Sigma = n^{-1} sum w_t (x_t - mu)(x_t - mu)'``;
    3. CM-step 2 (ECME): ``nu`` maximises the *observed* log-likelihood given the new
       ``(mu, Sigma)`` (one-dimensional bounded Brent search on ``log nu`` over ``nu_bounds``).

    Every step weakly increases the observed log-likelihood (the ``nu`` step keeps the old
    value if the search does not improve on it), so ``loglik_path_`` is non-decreasing.

    Parameters
    ----------
    nu_bounds : tuple of float, default (2.1, 200.0)
        Admissible range of ``nu``. The lower bound keeps the covariance finite.
    fix_nu : float or None, default None
        If given, ``nu`` is held fixed and only ``mu`` and ``Sigma`` are estimated (plain EM).
    tol : float, default 1e-12
        Convergence when the log-likelihood increase is below ``tol * (1 + |loglik|)``.
    max_iter : int, default 2000
        Maximum number of ECME iterations.
    """

    name = "MVStudentT"
    has_likelihood = True

    def __init__(
        self,
        nu_bounds: tuple[float, float] = NU_BOUNDS,
        fix_nu: float | None = None,
        tol: float = 1e-12,
        max_iter: int = 2000,
    ) -> None:
        lo, hi = float(nu_bounds[0]), float(nu_bounds[1])
        if not 2.0 < lo < hi:
            raise ValueError(f"nu_bounds must satisfy 2 < lo < hi, got {nu_bounds}")
        if fix_nu is not None and not lo <= fix_nu <= hi:
            raise ValueError(f"fix_nu={fix_nu} outside nu_bounds={nu_bounds}")
        self.nu_bounds = (lo, hi)
        self.fix_nu = None if fix_nu is None else float(fix_nu)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    # -- estimation --------------------------------------------------------------------------
    def _initial_nu(self, X: np.ndarray) -> float:
        if self.fix_nu is not None:
            return self.fix_nu
        D = X - X.mean(axis=0)
        k = float(np.mean(np.mean(D**4, axis=0) / np.mean(D**2, axis=0) ** 2 - 3.0))
        nu0 = 4.0 + 6.0 / k if k > 0.06 else 100.0  # moment match: excess kurtosis 6/(nu-4)
        return float(np.clip(nu0, max(3.0, self.nu_bounds[0]), min(100.0, self.nu_bounds[1])))

    def _update_nu(self, nu_old: float, d2: np.ndarray, p: int) -> float:
        lo, hi = self.nu_bounds
        res = optimize.minimize_scalar(
            lambda s: -_t_profile_nu(float(np.exp(s)), d2, p),
            bounds=(np.log(lo), np.log(hi)),
            method="bounded",
            options={"xatol": 1e-10},
        )
        candidates = [nu_old, lo, hi]
        if np.isfinite(res.x):
            candidates.append(float(np.clip(np.exp(res.x), lo, hi)))
        values = [_t_profile_nu(c, d2, p) for c in candidates]
        best = int(np.argmax(values))
        # keep the old value on ties so that the iteration is monotone
        if values[best] <= values[0]:
            return nu_old
        return candidates[best]

    def fit(self, R: np.ndarray) -> "MVStudentT":
        """Fit ``(mu, Sigma, nu)`` by ECME on ``R`` of shape ``(T, N)``."""
        X = _check_returns(R, min_obs=2)
        n, p = X.shape
        if n <= p + 1:
            raise ValueError(f"MVStudentT needs T > N + 1 observations, got T={n}, N={p}")
        nu = self._initial_nu(X)
        mu = X.mean(axis=0)
        S = np.atleast_2d(np.cov(X, rowvar=False, ddof=0))
        Sigma = S * (nu - 2.0) / nu
        L = _cholesky(Sigma, "initial covariance")
        d2 = _mahalanobis_sq(X, mu, L)
        ll = float(np.sum(_t_logpdf(d2, nu, p, _logdet_from_chol(L))))
        path = [ll]
        converged = False
        it = 0
        for it in range(1, self.max_iter + 1):
            w = (nu + p) / (nu + d2)  # E-step
            mu = (w @ X) / np.sum(w)  # CM-step 1
            D = X - mu
            Sigma = (D * w[:, None]).T @ D / n
            Sigma = 0.5 * (Sigma + Sigma.T)
            L = _cholesky(Sigma, "scatter matrix")
            d2 = _mahalanobis_sq(X, mu, L)
            if self.fix_nu is None:  # CM-step 2 (ECME: observed likelihood)
                nu = self._update_nu(nu, d2, p)
            ll_new = float(np.sum(_t_logpdf(d2, nu, p, _logdet_from_chol(L))))
            path.append(ll_new)
            if abs(ll_new - ll) <= self.tol * (1.0 + abs(ll_new)):
                ll = ll_new
                converged = True
                break
            ll = ll_new
        if not converged:
            logger.warning("MVStudentT: ECME did not converge in %d iterations", self.max_iter)
        self.n_assets_ = p
        self.mu_ = mu
        self.Sigma_ = Sigma
        self.nu_ = float(nu)
        self._L = L
        self.loglik_ = ll
        self.loglik_path_ = np.asarray(path)
        self.n_iter_ = it
        self.converged_ = converged
        return self

    # -- simulation and likelihood -----------------------------------------------------------
    def simulate(
        self,
        n_steps: int,
        *,
        rng: np.random.Generator,
        burn_in: int = 250,
        start_from_end: bool = False,
    ) -> np.ndarray:
        """Simulate ``(n_steps, N)`` i.i.d. multivariate-t returns (normal / sqrt(chi2/nu))."""
        _require_fitted(self, "mu_")
        n = _check_n_steps(n_steps)
        Z = rng.standard_normal((n, self.n_assets_)) @ self._L.T
        g = rng.chisquare(self.nu_, size=n) / self.nu_
        return self.mu_ + Z / np.sqrt(g)[:, None]

    def loglik_obs(self, R: np.ndarray, R_history: np.ndarray | None = None) -> np.ndarray:
        """Per-observation log-densities, shape ``(T,)`` (``R_history`` unused: i.i.d. model)."""
        _require_fitted(self, "mu_")
        R = _check_returns(R, min_obs=1)
        d2 = _mahalanobis_sq(R, self.mu_, self._L)
        return _t_logpdf(d2, self.nu_, self.n_assets_, _logdet_from_chol(self._L))

    def loglik(self, R: np.ndarray, R_history: np.ndarray | None = None) -> float:
        """Total log-likelihood of ``R``."""
        return float(np.sum(self.loglik_obs(R, R_history)))

    def params(self) -> dict:
        """Fitted parameters: ``mu``, scatter ``Sigma``, ``nu`` and the implied ``cov``."""
        _require_fitted(self, "mu_")
        return {
            "mu": self.mu_.copy(),
            "Sigma": self.Sigma_.copy(),
            "nu": self.nu_,
            "cov": self.Sigma_ * self.nu_ / (self.nu_ - 2.0),
        }

    def n_params(self) -> int:
        N = self.n_assets_
        return N + N * (N + 1) // 2 + (0 if self.fix_nu is not None else 1)


# ----------------------------------------------------------------------------------------------
# CCC-GARCH(1,1)-t with a Gaussian copula
# ----------------------------------------------------------------------------------------------
#: Accepted names for each GARCHt parameter in ``margin.params()``.
_GARCH_KEYS: dict[str, tuple[str, ...]] = {
    "mu": ("mu", "mean", "const", "Const"),
    "omega": ("omega", "w", "Omega"),
    "alpha": ("alpha", "a", "alpha1", "alpha[1]", "alpha_1"),
    "beta": ("beta", "b", "beta1", "beta[1]", "beta_1"),
    "nu": ("nu", "df", "dof", "eta"),
}


def _garch_params(p: Mapping[str, float]) -> dict[str, float]:
    """Map a GARCHt ``params()`` dict onto ``mu, omega, alpha, beta, nu`` (alias tolerant)."""
    out: dict[str, float] = {}
    for key, aliases in _GARCH_KEYS.items():
        for a in aliases:
            if a in p:
                out[key] = float(p[a])
                break
        else:
            raise KeyError(
                f"GARCHt margin params() has no entry for {key!r} (looked for {aliases}); "
                f"got keys {sorted(p)}"
            )
    if out["omega"] <= 0 or out["alpha"] < 0 or out["beta"] < 0 or out["nu"] <= 2:
        raise ValueError(f"invalid GARCHt margin parameters: {out}")
    return out


def _default_margin() -> ReturnModel:
    from . import get_model  # lazy: univariate.py is only needed when CCCGARCHt is fitted

    return get_model("GARCHt")


def _garch_filter(
    eps: np.ndarray, omega: np.ndarray, alpha: np.ndarray, beta: np.ndarray, s2_0: np.ndarray
) -> np.ndarray:
    """Conditional variances ``s2_t = omega + alpha eps_{t-1}^2 + beta s2_{t-1}``, ``s2_0`` given."""
    T = eps.shape[0]
    s2 = np.empty_like(eps)
    s2[0] = s2_0
    e2 = eps**2
    for t in range(1, T):
        s2[t] = omega + alpha * e2[t - 1] + beta * s2[t - 1]
    return s2


def _std_t_to_normal(Z: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Normal scores ``Phi^{-1}(F_nu(z))`` of standardised-t (unit variance) values.

    Computed through the lower tail of ``-|z|`` and symmetry, so it stays accurate in both tails.
    """
    zt = Z * np.sqrt(nu / (nu - 2.0))
    lower = np.maximum(special.stdtr(nu, -np.abs(zt)), _TINY)
    return np.sign(zt) * (-special.ndtri(lower))


def _normal_to_std_t(X: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_std_t_to_normal`: ``F_nu^{-1}(Phi(x))`` rescaled to unit variance."""
    lower = np.maximum(special.ndtr(-np.abs(X)), _TINY)
    t = -special.stdtrit(nu, lower)
    return np.sign(X) * t * np.sqrt((nu - 2.0) / nu)


def _std_t_logpdf(Z: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Log-density of the unit-variance Student-t(nu)."""
    return (
        special.gammaln(0.5 * (nu + 1.0))
        - special.gammaln(0.5 * nu)
        - 0.5 * np.log(np.pi * (nu - 2.0))
        - 0.5 * (nu + 1.0) * np.log1p(Z**2 / (nu - 2.0))
    )


def _nearest_correlation(C: np.ndarray, eig_floor: float = 1e-8) -> np.ndarray:
    """Symmetrise, clip eigenvalues at ``eig_floor`` and rescale to unit diagonal."""
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    if vals.min() >= eig_floor:
        out = C
    else:
        out = (vecs * np.maximum(vals, eig_floor)) @ vecs.T
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def _sample_correlation(A: np.ndarray) -> np.ndarray:
    """Sample correlation matrix of the columns of ``A`` (made a valid correlation matrix)."""
    if A.shape[1] == 1:
        return np.ones((1, 1))
    return _nearest_correlation(np.corrcoef(A, rowvar=False))


class CCCGARCHt(MultiReturnModel):
    """CCC-GARCH(1,1) with standardised-t margins coupled by a Gaussian copula.

    Model, for asset ``i``::

        r_it = mu_i + eps_it,   eps_it = s_it z_it,
        s_it^2 = omega_i + alpha_i eps_{i,t-1}^2 + beta_i s_{i,t-1}^2,
        z_it = F_{nu_i}^{-1}(Phi(x_it)) (unit-variance Student-t margin),
        x_t ~ N(0, P) i.i.d.,  P a correlation matrix (constant over time).

    The standardised innovations ``z_t`` therefore have exactly the univariate GARCHt margins
    (each with its own ``nu_i``) and a Gaussian copula with correlation ``P``.

    Choice of the innovation law. We use a Gaussian copula rather than multivariate-t
    innovations because the univariate GARCHt fits give *heterogeneous* tail indices ``nu_i``;
    a multivariate t forces one common ``nu`` and so would change every margin away from its
    fitted univariate law, whereas the copula keeps each margin equal to the fitted GARCHt model
    and adds only the dependence. The price is zero tail dependence (joint crashes are no more
    likely than under Gaussian dependence with the same ``P``), which is a stated limitation.

    Second limitation: attenuated return correlations. Each conditional variance is driven only
    by the asset's own past shocks, so volatilities co-move only through the contemporaneous
    correlation of the shocks (for jointly Gaussian scores ``corr(x_i^2, x_j^2) = P_ij^2``),
    whereas real volatilities share a strong common component. In addition ``P`` is estimated
    from standardised residuals, which gives the high-volatility, high-correlation periods the
    same weight as calm periods. Both effects make the *unconditional* correlations of
    simulated returns markedly lower than the sample correlations of the data, although the
    copula and the margins are reproduced exactly. On 30 S&P 500 stocks (TRAIN 2018-2022) the
    mean pairwise return correlation is 0.43 in the data, 0.28 on a long CCC path and 0.31 on
    average over TRAIN-length CCC paths (0.41 for the row bootstrap); the mean of ``P`` is 0.36,
    and the correlation of log conditional volatilities is 0.64 in the data against 0.12 in
    simulations. Synthetic covariance matrices are therefore biased towards
    low correlation, which matters for minimum-variance and mean-variance weights; compare
    :func:`synthval.portfolio.average_correlation` of real and simulated samples when
    interpreting portfolio results. A factor (common-volatility) specification would address
    the volatility co-movement, and DCC the time variation of correlations; neither is
    implemented.

    Estimation is two-stage (inference functions for margins, Joe 2005): each margin is a
    univariate ``GARCHt`` fitted on its own column; ``P`` is then estimated from the
    standardised residuals ``z_it = (r_it - mu_i) / s_it``. With
    ``corr_method="normal_scores"`` (default), ``P`` is the sample correlation of the normal
    scores ``Phi^{-1}(F_{nu_i}(z_it))``, the consistent estimator of the Gaussian-copula
    parameter; ``corr_method="pearson"`` uses the Pearson correlation of ``z_it`` itself
    (Bollerslev's CCC estimator; under t margins it is slightly attenuated relative to ``P``).
    Both matrices are stored (``R_`` is the one used, ``R_pearson_`` and ``R_normal_scores_``).

    The conditional-variance filter is run here from the margins' parameters, so only ``fit``
    and ``params`` of the margin are required. It uses the same initialisation as the univariate
    ``GARCHt``: in sample, ``s2_0`` is the sample variance (divisor ``T``) of the series being
    filtered; out of sample, the filter runs through ``R_history`` and continues from its end
    state. When the margin also implements ``loglik_obs``, the fit checks that the
    margin's own filtered log-densities agree with ours on the second half of the sample (this
    catches mismatched parameter names or units).

    Near-unit-root margins. The GARCHt fit may hit its persistence bound
    (``alpha + beta = 0.9999``), typically for stocks hit hard in 2020. The implied
    unconditional variance ``omega / (1 - alpha - beta)`` is then poorly determined and can be
    tens of times the sample variance, and a burn-in of 250 days does not wash out the start
    (``0.9999^250 = 0.975``). Margins with ``alpha + beta >= near_unit_root`` are flagged in
    ``near_unit_root_`` (with a logged warning; the margins' own ``fit_info_["at_bound"]``
    lists are kept in ``margin_at_bound_``), and ``simulate`` starts their variance recursion
    at the training sample variance ``sigma2_init_`` on the first returned day, while the
    other margins start at their unconditional variance and are burnt in. On the ten S&P 500
    stocks at the bound this reduces the 95% quantile of (simulated / training variance) over
    TRAIN-length paths from 5-60 to 2-7, leaving the median close to one.

    Parameters
    ----------
    margin_factory : callable or None, default None
        Zero-argument callable returning an unfitted univariate margin model; ``None`` uses
        ``synthval.models.get_model("GARCHt")``. Its ``params()`` must contain the mean,
        ``omega``, ``alpha``, ``beta`` and ``nu`` (common aliases accepted, see ``_GARCH_KEYS``).
    corr_method : {"normal_scores", "pearson"}, default "normal_scores"
        Estimator of the copula correlation ``P`` (see above).
    check_margins : bool, default True
        Cross-check our variance filter against the margin's ``loglik_obs``.
    near_unit_root : float, default 0.999
        Persistence ``alpha + beta`` from which a margin is treated as near-integrated (see
        above). ``near_unit_root > 1`` disables the special start, so every margin starts at
        its unconditional variance as in the univariate ``GARCHt.simulate``.

    Attributes
    ----------
    R_, R_pearson_, R_normal_scores_ : ndarray, shape (N, N)
        Copula correlation used, and both estimators.
    persistence_ : ndarray, shape (N,)
        ``alpha_i + beta_i``.
    near_unit_root_ : ndarray of bool, shape (N,)
        ``persistence_ >= near_unit_root``.
    margin_at_bound_ : list of list of str
        Per margin, the parameters at a bound reported by the margin's ``fit_info_``
        (empty lists when the margin does not report it).
    variance_ratio_ : ndarray, shape (N,)
        Unconditional / training sample variance (``inf`` if ``alpha + beta >= 1``).
    sigma2_init_, sigma2_next_ : ndarray, shape (N,)
        Training sample variance (in-sample filter start) and one-step-ahead variance at the
        end of the training sample.
    """

    name = "CCCGARCHt"
    has_likelihood = True

    def __init__(
        self,
        margin_factory: Callable[[], ReturnModel] | None = None,
        corr_method: Literal["normal_scores", "pearson"] = "normal_scores",
        check_margins: bool = True,
        near_unit_root: float = 0.999,
    ) -> None:
        if corr_method not in ("normal_scores", "pearson"):
            raise ValueError(f"unknown corr_method {corr_method!r}")
        if not float(near_unit_root) > 0.0:
            raise ValueError(f"near_unit_root must be positive, got {near_unit_root}")
        self.margin_factory = margin_factory
        self.corr_method = corr_method
        self.check_margins = bool(check_margins)
        self.near_unit_root = float(near_unit_root)

    # -- estimation --------------------------------------------------------------------------
    def _check_margin(self, i: int, margin: ReturnModel, r: np.ndarray, ours: np.ndarray) -> None:
        try:
            theirs = np.asarray(margin.loglik_obs(r), dtype=float)
        except NotImplementedError:
            return
        if theirs.shape != ours.shape:
            logger.warning("CCCGARCHt: margin %d loglik_obs has shape %s", i, theirs.shape)
            return
        half = slice(r.size // 2, None)
        gap = float(np.median(np.abs(theirs[half] - ours[half])))
        if gap > 0.05:
            raise RuntimeError(
                f"CCCGARCHt: margin {i}: filtered log-densities from params() differ from the "
                f"margin's loglik_obs by {gap:.3g} per observation; parameter names or units "
                "of the GARCHt margin do not match (mu, omega, alpha, beta, nu in daily units)"
            )
        if gap > 1e-3:
            logger.warning("CCCGARCHt: margin %d: filter mismatch %.3g per observation", i, gap)

    def fit(self, R: np.ndarray) -> "CCCGARCHt":
        """Fit the GARCHt margins column by column, then the copula correlation."""
        R = _check_returns(R, min_obs=20)
        T, N = R.shape
        factory = self.margin_factory if self.margin_factory is not None else _default_margin
        margins: list[ReturnModel] = []
        theta = {k: np.empty(N) for k in _GARCH_KEYS}
        for i in range(N):
            m = factory()
            m.fit(R[:, i].copy())
            p = _garch_params(m.params())
            for k in _GARCH_KEYS:
                theta[k][i] = p[k]
            margins.append(m)
        mu, omega, alpha, beta, nu = (theta[k] for k in ("mu", "omega", "alpha", "beta", "nu"))
        eps = R - mu
        s2_0 = R.var(axis=0)
        s2 = _garch_filter(eps, omega, alpha, beta, s2_0)
        Z = eps / np.sqrt(s2)
        if self.check_margins:
            ll_marg = _std_t_logpdf(Z, nu) - 0.5 * np.log(s2)
            for i, m in enumerate(margins):
                self._check_margin(i, m, R[:, i], ll_marg[:, i])
        X = _std_t_to_normal(Z, nu)
        self.R_pearson_ = _sample_correlation(Z)
        self.R_normal_scores_ = _sample_correlation(X)
        self.R_ = self.R_normal_scores_ if self.corr_method == "normal_scores" else self.R_pearson_
        self._LP = _cholesky(self.R_, "copula correlation")

        self.n_assets_ = N
        self.margins_ = margins
        self.mu_, self.omega_, self.alpha_, self.beta_, self.nu_ = mu, omega, alpha, beta, nu
        self.sigma2_init_ = s2_0
        self.sigma2_fit_ = s2
        self.std_resid_ = Z
        #: one-step-ahead conditional variance at the end of the training sample
        self.sigma2_next_ = omega + alpha * eps[-1] ** 2 + beta * s2[-1]
        self._diagnose_margins()
        return self

    def _diagnose_margins(self) -> None:
        """Persistence / bound diagnostics of the fitted margins (logged when relevant)."""
        persist = self.alpha_ + self.beta_
        self.persistence_ = persist
        self.near_unit_root_ = persist >= self.near_unit_root
        self.margin_at_bound_ = [
            [str(b) for b in (getattr(m, "fit_info_", None) or {}).get("at_bound", [])]
            for m in self.margins_
        ]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = self.omega_ / (1.0 - persist) / self.sigma2_init_
        self.variance_ratio_ = np.where(persist < 1.0, ratio, np.inf)
        if np.any(self.near_unit_root_):
            idx = np.flatnonzero(self.near_unit_root_)
            logger.warning(
                "CCCGARCHt: %d margin(s) near a unit root (alpha+beta >= %g): columns %s, "
                "persistence %s, unconditional/sample variance %s; simulate() starts them at "
                "the training variance", idx.size, self.near_unit_root, idx.tolist(),
                np.round(persist[idx], 5).tolist(), np.round(self.variance_ratio_[idx], 1).tolist())
        other = [i for i, b in enumerate(self.margin_at_bound_)
                 if b and not self.near_unit_root_[i]]
        if other:
            logger.info("CCCGARCHt: margins with parameters at a bound: %s",
                        {i: self.margin_at_bound_[i] for i in other})

    # -- helpers -----------------------------------------------------------------------------
    def unconditional_variance(self) -> np.ndarray:
        """``omega / (1 - alpha - beta)`` per asset (training variance if non-stationary)."""
        _require_fitted(self, "mu_")
        persist = self.alpha_ + self.beta_
        with np.errstate(divide="ignore", invalid="ignore"):
            v = self.omega_ / (1.0 - persist)
        return np.where(persist < 1.0, v, self.sigma2_init_)

    def _s2_start(self, X: np.ndarray) -> np.ndarray:
        """In-sample filter start: sample variance of ``X`` (unconditional if degenerate)."""
        v = X.var(axis=0) if X.shape[0] >= 2 else np.zeros(X.shape[1])
        return np.where(v > 0, v, self.unconditional_variance())

    def innovations(self, n: int, *, rng: np.random.Generator) -> np.ndarray:
        """Draw ``(n, N)`` standardised innovations from the fitted Gaussian copula / t margins."""
        _require_fitted(self, "mu_")
        X = rng.standard_normal((n, self.n_assets_)) @ self._LP.T
        return _normal_to_std_t(X, self.nu_)

    # -- simulation and likelihood -----------------------------------------------------------
    def simulate(
        self,
        n_steps: int,
        *,
        rng: np.random.Generator,
        burn_in: int = 250,
        start_from_end: bool = False,
    ) -> np.ndarray:
        """Simulate ``(n_steps, N)`` returns.

        Parameters
        ----------
        n_steps : int
            Number of simulated days.
        rng : numpy.random.Generator
            Random generator.
        burn_in : int, default 250
            Discarded initial days when ``start_from_end`` is False. The recursion then starts
            at the unconditional variance, except for the near-unit-root margins
            (``near_unit_root_``), whose variance is set to the training variance
            ``sigma2_init_`` on the first returned day.
        start_from_end : bool, default False
            Start from the fitted end-of-sample state: the first simulated day has conditional
            variance ``sigma2_next_`` for every asset (no burn-in).
        """
        _require_fitted(self, "mu_")
        n = _check_n_steps(n_steps)
        burn = 0 if start_from_end else max(int(burn_in), 0)
        total = n + burn
        Z = self.innovations(total, rng=rng)
        s2 = self.sigma2_next_.copy() if start_from_end else self.unconditional_variance()
        nur = self.near_unit_root_
        reset = None if start_from_end or not np.any(nur) else nur
        out = np.empty((total, self.n_assets_))
        for t in range(total):
            if reset is not None and t == burn:
                s2 = np.where(reset, self.sigma2_init_, s2)
            e = np.sqrt(s2) * Z[t]
            out[t] = e
            s2 = self.omega_ + self.alpha_ * e**2 + self.beta_ * s2
        return self.mu_ + out[burn:]

    def filter(
        self, R: np.ndarray, R_history: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conditional variances and standardised residuals of ``R``.

        Same convention as the univariate ``GARCHt``: with ``R_history`` (e.g. the training
        returns) the filter runs through the history first, starting from the history's sample
        variance as in ``fit``, so the first variance of ``R`` is the one-step forecast from the
        end of the history; without it, the filter starts from the sample variance of ``R``
        (the unconditional variance if ``R`` has a single row).

        Returns
        -------
        s2, Z : ndarray
            Both of shape ``(T, N)``.
        """
        _require_fitted(self, "mu_")
        R = _check_returns(R, min_obs=1)
        eps = R - self.mu_
        if R_history is not None:
            H = _check_returns(R_history, min_obs=1)
            if H.shape[1] != R.shape[1]:
                raise ValueError("R_history and R must have the same number of columns")
            full = np.vstack([H - self.mu_, eps])
            s2 = _garch_filter(full, self.omega_, self.alpha_, self.beta_, self._s2_start(H))
            s2 = s2[H.shape[0]:]
        else:
            s2 = _garch_filter(eps, self.omega_, self.alpha_, self.beta_, self._s2_start(R))
        return s2, eps / np.sqrt(s2)

    def loglik_obs(self, R: np.ndarray, R_history: np.ndarray | None = None) -> np.ndarray:
        """Per-observation joint conditional log-densities, shape ``(T,)``.

        ``log f(r_t | F_{t-1}) = sum_i [log f_{nu_i}(z_it) - log s_it] + log c_P(x_t)`` with the
        Gaussian-copula density ``log c_P(x) = -log|P|/2 - x'(P^{-1} - I)x / 2``.
        """
        s2, Z = self.filter(R, R_history)
        marg = np.sum(_std_t_logpdf(Z, self.nu_) - 0.5 * np.log(s2), axis=1)
        X = _std_t_to_normal(Z, self.nu_)
        Y = linalg.solve_triangular(self._LP, X.T, lower=True, check_finite=False)
        quad = np.einsum("ij,ij->j", Y, Y) - np.sum(X**2, axis=1)
        return marg - 0.5 * _logdet_from_chol(self._LP) - 0.5 * quad

    def loglik(self, R: np.ndarray, R_history: np.ndarray | None = None) -> float:
        """Total conditional log-likelihood of ``R``."""
        return float(np.sum(self.loglik_obs(R, R_history)))

    def params(self) -> dict:
        """Margin parameters (arrays of length N), copula correlation, start and end-of-sample
        variances and the near-unit-root diagnostics (see the class docstring)."""
        _require_fitted(self, "mu_")
        return {
            "mu": self.mu_.copy(),
            "omega": self.omega_.copy(),
            "alpha": self.alpha_.copy(),
            "beta": self.beta_.copy(),
            "nu": self.nu_.copy(),
            "R": self.R_.copy(),
            "R_pearson": self.R_pearson_.copy(),
            "sigma2_init": self.sigma2_init_.copy(),
            "sigma2_next": self.sigma2_next_.copy(),
            "persistence": self.persistence_.copy(),
            "near_unit_root": self.near_unit_root_.copy(),
            "variance_ratio": self.variance_ratio_.copy(),
            "margin_at_bound": [list(b) for b in self.margin_at_bound_],
            "corr_method": self.corr_method,
        }

    def n_params(self) -> int:
        N = self.n_assets_
        return 5 * N + N * (N - 1) // 2


# ----------------------------------------------------------------------------------------------
# Multivariate stationary bootstrap
# ----------------------------------------------------------------------------------------------
_BLOCK_SERIES = ("abs", "r", "sq")


def _block_series(x: np.ndarray, how: str) -> np.ndarray:
    """Series on which the block-length rule is applied (same definitions as the univariate
    ``StationaryBootstrap``: ``"r"`` the returns, ``"abs"`` ``|r - mean|``, ``"sq"``
    ``(r - mean)^2``)."""
    if how == "r":
        return x
    if how == "abs":
        return np.abs(x - x.mean())
    if how == "sq":
        return (x - x.mean()) ** 2
    raise ValueError(f"block_series must be one of {_BLOCK_SERIES}, got {how!r}")


def stationary_bootstrap_indices(
    T: int, n: int, mean_block: float, *, rng: np.random.Generator
) -> np.ndarray:
    """Row indices of one Politis-Romano stationary-bootstrap path with wrap-around.

    The first index is uniform on ``{0, ..., T-1}``; at each later step a new block starts (at a
    fresh uniform index) with probability ``1 / mean_block``, otherwise the next index is
    ``(previous + 1) mod T``. Block lengths are therefore geometric with mean ``mean_block``.

    This is a thin wrapper around
    :func:`synthval.models.univariate.stationary_bootstrap_indices` with ``n_paths = 1``, so the
    univariate and multivariate bootstraps share one implementation and, for the same ``rng``
    state, the same indices.

    Parameters
    ----------
    T : int
        Length of the original sample.
    n : int
        Length of the bootstrap path.
    mean_block : float
        Mean block length, ``>= 1``.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    ndarray of int, shape (n,)
    """
    from .univariate import stationary_bootstrap_indices as _sb_indices

    if not mean_block >= 1.0:
        raise ValueError(f"mean_block must be >= 1, got {mean_block}")
    return _sb_indices(int(T), int(n), 1, float(mean_block), rng=rng)[0]


class MVStationaryBootstrap(MultiReturnModel):
    """Stationary bootstrap of whole rows of the training panel (Politis and Romano, 1994).

    All assets share the same random blocks, so every simulated day is an actual training day
    and the contemporaneous cross-sectional dependence is reproduced exactly; serial dependence
    (in particular volatility clustering) is preserved within blocks only.

    Block-length convention (identical to the univariate ``StationaryBootstrap`` used in E3).
    With ``mean_block="auto"`` the Politis-White (2004) rule with the Patton-Politis-White
    (2009) correction, :func:`synthval.models.univariate.optimal_block_length` (the single
    implementation of the rule in the package), is applied column by column to
    ``block_series`` (default ``|r_i - mean(r_i)|``) and the per-asset lengths are aggregated
    by their median. Applied to the returns themselves (``block_series="r"``), the rule sees
    almost no serial correlation and picks blocks of one to a few days, which destroys the
    dependence in ``|r|`` that the bootstrap is meant to keep (complete S&P 500 universe, TRAIN
    2018-2022: median per-stock length 3.2 days on ``r`` against 53 days on ``|r|``). The
    median rather than the mean is used because it is robust to the few assets whose length is
    far from the others (on ``r`` the per-stock lengths of a 30-stock panel range from 1 to 45
    days, median 2.7, mean 5.6).

    Parameters
    ----------
    mean_block : float, "auto" or None, default "auto"
        Mean block length. ``"auto"`` (or ``None``) uses the rule above; a number fixes it.
    default_block : float, default 20.0
        Used when the automatic rule cannot be applied (fewer than 20 observations, or a
        degenerate series), as in the univariate model.
    block_series : {"abs", "r", "sq"}, default "abs"
        Series on which the automatic rule is applied, per asset: ``|r - mean|``, ``r`` or
        ``(r - mean)^2`` (same names and definitions as the univariate model).
    aggregate : {"median", "mean"}, default "median"
        How the per-asset automatic lengths are combined into one common block length.

    Attributes
    ----------
    mean_block_ : float
        Mean block length used by ``simulate`` (clipped to ``[1, T]``).
    block_lengths_ : ndarray, shape (N,)
        Per-asset automatic lengths (only with ``mean_block="auto"``).
    """

    name = "MVBootstrap"
    has_likelihood = False

    def __init__(
        self,
        mean_block: float | Literal["auto"] | None = "auto",
        default_block: float = 20.0,
        *,
        block_series: Literal["abs", "r", "sq"] = "abs",
        aggregate: Literal["median", "mean"] = "median",
    ) -> None:
        if mean_block not in ("auto", None) and not float(mean_block) >= 1.0:  # type: ignore[arg-type]
            raise ValueError(f"mean_block must be >= 1, got {mean_block}")
        if block_series not in _BLOCK_SERIES:
            raise ValueError(f"block_series must be one of {_BLOCK_SERIES}, got {block_series!r}")
        if aggregate not in ("median", "mean"):
            raise ValueError(f"aggregate must be 'median' or 'mean', got {aggregate!r}")
        if not float(default_block) >= 1.0:
            raise ValueError(f"default_block must be >= 1, got {default_block}")
        self.mean_block = mean_block
        self.block_series = block_series
        self.aggregate = aggregate
        self.default_block = float(default_block)

    def fit(self, R: np.ndarray) -> "MVStationaryBootstrap":
        """Store the training panel and choose the mean block length."""
        R = _check_returns(R, min_obs=2)
        T = R.shape[0]
        self.R_ = R.copy()
        if self.mean_block in ("auto", None):
            from .univariate import optimal_block_length

            per_asset = np.array([
                optimal_block_length(_block_series(R[:, i], self.block_series),
                                     kind="stationary", fallback=self.default_block)
                for i in range(R.shape[1])
            ])
            self.block_lengths_ = per_asset
            agg = np.median if self.aggregate == "median" else np.mean
            b = float(agg(per_asset))
        else:
            b = float(self.mean_block)  # type: ignore[arg-type]
        self.mean_block_ = float(np.clip(b, 1.0, float(T)))
        return self

    def simulate(
        self,
        n_steps: int,
        *,
        rng: np.random.Generator,
        burn_in: int = 250,
        start_from_end: bool = False,
    ) -> np.ndarray:
        """Resample ``(n_steps, N)`` rows (stationary from the start: ``burn_in`` and
        ``start_from_end`` are ignored)."""
        _require_fitted(self, "R_")
        n = _check_n_steps(n_steps)
        idx = stationary_bootstrap_indices(self.R_.shape[0], n, self.mean_block_, rng=rng)
        return self.R_[idx]

    def loglik_obs(self, R: np.ndarray, R_history: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError(f"{self.name}: likelihood is intractable")

    def params(self) -> dict:
        """``{"mean_block", "block_series", "T", "N"}``."""
        _require_fitted(self, "R_")
        return {"mean_block": self.mean_block_,
                "block_series": None if self.mean_block not in ("auto", None)
                else self.block_series,
                "T": int(self.R_.shape[0]), "N": int(self.R_.shape[1])}
