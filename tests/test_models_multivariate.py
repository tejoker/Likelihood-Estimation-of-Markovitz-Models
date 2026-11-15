"""Tests for synthval.models.multivariate (DESIGN section 3.6)."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy import optimize, stats

from synthval.models import MULTIVARIATE_MODELS, get_mv_model
from synthval.models.base import ReturnModel
from synthval.models.multivariate import (
    CCCGARCHt,
    MVGaussian,
    MVStationaryBootstrap,
    MVStudentT,
    _normal_to_std_t,
    _std_t_to_normal,
    stationary_bootstrap_indices,
)

try:  # the real univariate GARCHt is written concurrently by another module owner
    from synthval.models.univariate import GARCHt  # noqa: F401

    HAVE_GARCHT = True
    GARCHT_REASON = ""
except Exception as exc:  # pragma: no cover - depends on the state of the repository
    HAVE_GARCHT = False
    GARCHT_REASON = (f"synthval.models.univariate.GARCHt is not importable ({exc!r}); "
                     "CCC logic is covered by the stub-margin tests")


# ----------------------------------------------------------------------------------------------
# Fixtures and independent reference implementations
# ----------------------------------------------------------------------------------------------
def _random_cov(rng: np.random.Generator, N: int, scale: float = 1e-4) -> np.ndarray:
    A = rng.standard_normal((N, N))
    return scale * (A @ A.T / N + 0.5 * np.eye(N))


def _sim_mvt(rng, mu, Sigma, nu, n):
    L = np.linalg.cholesky(Sigma)
    g = rng.chisquare(nu, n) / nu
    return mu + (rng.standard_normal((n, len(mu))) @ L.T) / np.sqrt(g)[:, None]


def _std_t_logpdf_ref(z, nu):
    """Unit-variance Student-t log-density through scipy.stats."""
    k = np.sqrt(nu / (nu - 2.0))
    return stats.t.logpdf(z * k, nu) + np.log(k)


def _garch_filter_ref(eps, omega, alpha, beta, s2_0):
    s2 = np.empty_like(eps)
    s2[0] = s2_0
    for t in range(1, eps.shape[0]):
        s2[t] = omega + alpha * eps[t - 1] ** 2 + beta * s2[t - 1]
    return s2


TRUE_MARGINS = [
    dict(mu=3e-4, omega=2e-6, alpha=0.08, beta=0.88, nu=5.0),
    dict(mu=-1e-4, omega=4e-6, alpha=0.10, beta=0.85, nu=8.0),
    dict(mu=5e-4, omega=1e-6, alpha=0.05, beta=0.92, nu=12.0),
]
TRUE_P = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]])


def simulate_ccc_ref(T, P, margins, rng, burn=500):
    """Independent CCC-GARCH-t / Gaussian-copula simulator (scipy.stats transforms)."""
    N = len(margins)
    get = lambda k: np.array([m[k] for m in margins])  # noqa: E731
    mu, om, al, be, nu = (get(k) for k in ("mu", "omega", "alpha", "beta", "nu"))
    X = rng.standard_normal((T + burn, N)) @ np.linalg.cholesky(P).T
    Z = stats.t.ppf(stats.norm.cdf(X), nu) * np.sqrt((nu - 2) / nu)
    s2 = om / (1 - al - be)
    out = np.empty((T + burn, N))
    for t in range(T + burn):
        e = np.sqrt(s2) * Z[t]
        out[t] = mu + e
        s2 = om + al * e**2 + be * s2
    return out[burn:]


class StubGARCHt(ReturnModel):
    """Oracle stand-in for the univariate GARCHt: ``fit`` keeps known parameters.

    ``loglik_obs`` is written independently (scipy.stats), with ``s2_0`` = sample variance of the
    series (the GARCHt convention). ``keys`` renames the parameters to test alias handling;
    ``omega_scale`` corrupts omega in ``params()`` to test the margin consistency check.
    """

    name = "GARCHt"
    has_likelihood = True

    def __init__(self, theta, keys=("mu", "omega", "alpha", "beta", "nu"), omega_scale=1.0,
                 with_loglik=True, at_bound=None):
        self.theta = dict(theta)
        self.keys = keys
        self.omega_scale = omega_scale
        self.with_loglik = with_loglik
        self.at_bound = at_bound

    def fit(self, r):
        self.r_ = np.asarray(r, dtype=float)
        if self.at_bound is not None:  # mimic GARCHt.fit_info_
            self.fit_info_ = {"at_bound": list(self.at_bound)}
        return self

    def params(self):
        th = dict(self.theta)
        th["omega"] = th["omega"] * self.omega_scale
        return {new: th[old] for old, new in
                zip(("mu", "omega", "alpha", "beta", "nu"), self.keys)}

    def loglik_obs(self, r, r_history=None):
        if not self.with_loglik:
            raise NotImplementedError
        th = self.theta
        eps = np.asarray(r, dtype=float) - th["mu"]
        s2 = _garch_filter_ref(eps, th["omega"], th["alpha"], th["beta"], np.var(r))
        return _std_t_logpdf_ref(eps / np.sqrt(s2), th["nu"]) - 0.5 * np.log(s2)


def stub_factory(thetas, **kw):
    """Zero-argument factory returning one stub per asset, in column order."""
    it = iter([StubGARCHt(th, **kw) for th in thetas])
    return lambda: next(it)


@pytest.fixture(scope="module")
def ccc_data():
    return simulate_ccc_ref(4000, TRUE_P, TRUE_MARGINS, np.random.default_rng(11))


# ----------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------
def test_registry_instantiates_all_models():
    for name in MULTIVARIATE_MODELS:
        m = get_mv_model(name)
        assert hasattr(m, "fit") and hasattr(m, "simulate")
    assert get_mv_model("MVBootstrap").name == "MVBootstrap"


@pytest.mark.parametrize("cls", [MVGaussian, MVStudentT, MVStationaryBootstrap])
def test_simulate_shape_and_reproducibility(cls):
    rng = np.random.default_rng(0)
    R = _sim_mvt(rng, np.zeros(4), _random_cov(rng, 4), 6.0, 800)
    m = cls().fit(R)
    a = m.simulate(300, rng=np.random.default_rng(42))
    b = m.simulate(300, rng=np.random.default_rng(42))
    assert a.shape == (300, 4) and a.dtype == np.float64
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        m.simulate(0, rng=rng)
    m2 = pickle.loads(pickle.dumps(m))  # fitted models travel through parallel_map
    np.testing.assert_array_equal(m2.simulate(300, rng=np.random.default_rng(42)), a)


def test_input_validation():
    with pytest.raises(ValueError):
        MVGaussian().fit(np.zeros(10))
    with pytest.raises(ValueError):
        MVGaussian().fit(np.array([[0.0, np.nan], [1.0, 2.0], [0.5, 0.1]]))
    with pytest.raises(RuntimeError):
        MVStudentT().simulate(5, rng=np.random.default_rng(0))


# ----------------------------------------------------------------------------------------------
# MVGaussian
# ----------------------------------------------------------------------------------------------
def test_mvgaussian_fit_and_loglik():
    rng = np.random.default_rng(1)
    Sigma = _random_cov(rng, 3)
    R = rng.multivariate_normal([1e-3, 0.0, -1e-3], Sigma, size=2000)
    m = MVGaussian().fit(R)
    np.testing.assert_allclose(m.mu_, R.mean(0), rtol=1e-12)
    np.testing.assert_allclose(m.Sigma_, np.cov(R, rowvar=False), rtol=1e-12)
    ref = stats.multivariate_normal(m.mu_, m.Sigma_).logpdf(R[:50])
    np.testing.assert_allclose(m.loglik_obs(R[:50]), ref, rtol=1e-10)
    S = m.simulate(100_000, rng=rng)
    np.testing.assert_allclose(np.cov(S, rowvar=False), m.Sigma_, atol=0.02 * Sigma.max())
    assert m.n_params() == 3 + 6


# ----------------------------------------------------------------------------------------------
# MVStudentT
# ----------------------------------------------------------------------------------------------
def test_mvt_em_recovers_df_and_scatter():
    """n = 5000, N = 5, nu = 5: ECME recovers nu, mu and the scatter matrix."""
    rng = np.random.default_rng(2024)
    N, n, nu = 5, 5000, 5.0
    Sigma = _random_cov(rng, N)
    mu = rng.normal(0, 5e-4, N)
    R = _sim_mvt(rng, mu, Sigma, nu, n)
    m = MVStudentT().fit(R)
    assert m.converged_
    assert abs(m.nu_ - nu) < 1.0, m.nu_
    d = np.sqrt(np.diag(Sigma))
    assert np.max(np.abs(m.Sigma_ - Sigma) / np.outer(d, d)) < 0.08
    assert np.max(np.abs(m.mu_ - mu) / d) < 0.05
    # ECME is monotone in the observed log-likelihood
    path = m.loglik_path_
    assert np.all(np.diff(path) >= -1e-9 * np.abs(path[1:]))
    np.testing.assert_allclose(m.params()["cov"], m.Sigma_ * m.nu_ / (m.nu_ - 2), rtol=1e-12)


def test_mvt_em_reaches_the_mle():
    """ECME log-likelihood is not beaten by a generic optimiser (N = 2, 6 parameters)."""
    rng = np.random.default_rng(7)
    R = _sim_mvt(rng, np.array([1e-3, -5e-4]), _random_cov(rng, 2), 4.0, 2000)
    m = MVStudentT().fit(R)

    def negll(th):
        L = np.array([[np.exp(th[2]), 0.0], [th[3], np.exp(th[4])]])
        return -stats.multivariate_t(th[:2], L @ L.T, df=np.exp(th[5])).logpdf(R).sum()

    Lc = np.linalg.cholesky(m.Sigma_)
    th0 = np.r_[m.mu_, np.log(Lc[0, 0]), Lc[1, 0], np.log(Lc[1, 1]), np.log(m.nu_)]
    start = th0 + np.r_[2e-4, -2e-4, 0.05, 2e-3, -0.05, 0.3]
    res = optimize.minimize(negll, start, method="Nelder-Mead",
                            options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
    assert m.loglik_ >= -res.fun - 1e-6
    assert abs(np.exp(res.x[5]) - m.nu_) < 0.02
    np.testing.assert_allclose(m.loglik_, m.loglik(R), rtol=1e-12)


def test_mvt_loglik_matches_scipy():
    rng = np.random.default_rng(3)
    R = _sim_mvt(rng, np.zeros(3), _random_cov(rng, 3), 6.0, 1500)
    m = MVStudentT().fit(R)
    ref = stats.multivariate_t(m.mu_, m.Sigma_, df=m.nu_).logpdf(R[:100])
    np.testing.assert_allclose(m.loglik_obs(R[:100]), ref, rtol=1e-10)


def test_mvt_df_bounds():
    rng = np.random.default_rng(4)
    Sigma = _random_cov(rng, 3)
    gauss = rng.multivariate_normal(np.zeros(3), Sigma, size=4000)
    m_g = MVStudentT().fit(gauss)
    assert 30.0 < m_g.nu_ <= 200.0
    heavy = _sim_mvt(rng, np.zeros(3), Sigma, 1.2, 4000)  # MLE of nu below 2.1
    m_h = MVStudentT().fit(heavy)
    assert m_h.nu_ == pytest.approx(2.1)
    m_f = MVStudentT(fix_nu=7.0).fit(heavy)
    assert m_f.nu_ == 7.0 and m_f.n_params() == 3 + 6
    with pytest.raises(ValueError):
        MVStudentT(nu_bounds=(1.5, 10.0))


def test_mvt_simulation_moments():
    rng = np.random.default_rng(5)
    Sigma = _random_cov(rng, 3)
    m = MVStudentT().fit(_sim_mvt(rng, np.zeros(3), Sigma, 8.0, 3000))
    S = m.simulate(200_000, rng=rng)
    cov = m.Sigma_ * m.nu_ / (m.nu_ - 2)
    d = np.sqrt(np.diag(cov))
    assert np.max(np.abs(np.cov(S, rowvar=False) - cov) / np.outer(d, d)) < 0.03
    kurt = stats.kurtosis(S[:, 0])
    assert kurt == pytest.approx(6.0 / (m.nu_ - 4.0), rel=0.35)


# ----------------------------------------------------------------------------------------------
# CCCGARCHt (stub margins: always run)
# ----------------------------------------------------------------------------------------------
def test_normal_t_transforms():
    nu = np.array([3.0, 5.0, 30.0])
    x = np.linspace(-8, 0, 81)[:, None] * np.ones(3)  # lower tail: scipy.stats is accurate here
    z = _normal_to_std_t(x, nu)
    ref = stats.t.ppf(stats.norm.cdf(x), nu) * np.sqrt((nu - 2) / nu)
    np.testing.assert_allclose(z, ref, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(_normal_to_std_t(-x, nu), -z, rtol=1e-14)  # exact symmetry
    np.testing.assert_allclose(_std_t_to_normal(z, nu), x, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(_std_t_to_normal(-z, nu), -x, rtol=1e-8, atol=1e-10)
    assert np.all(np.isfinite(_normal_to_std_t(np.array([[-30.0, 30.0, 37.0]]), nu)))


def test_ccc_recovers_correlations(ccc_data):
    m = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS)).fit(ccc_data)
    np.testing.assert_allclose(m.R_, TRUE_P, atol=0.04)
    np.testing.assert_allclose(m.R_pearson_, TRUE_P, atol=0.06)
    assert m.n_assets_ == 3 and len(m.margins_) == 3
    np.testing.assert_allclose(np.diag(m.R_), 1.0)
    p = m.params()
    np.testing.assert_allclose(p["nu"], [5.0, 8.0, 12.0])
    assert p["R"].shape == (3, 3)
    m_p = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS), corr_method="pearson").fit(ccc_data)
    np.testing.assert_array_equal(m_p.R_, m_p.R_pearson_)


def test_ccc_filter_and_end_state(ccc_data):
    m = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS)).fit(ccc_data)
    mu = np.array([t["mu"] for t in TRUE_MARGINS])
    eps = ccc_data - mu
    for i, th in enumerate(TRUE_MARGINS):
        s2 = _garch_filter_ref(eps[:, i], th["omega"], th["alpha"], th["beta"],
                               np.var(ccc_data[:, i]))
        np.testing.assert_allclose(m.sigma2_fit_[:, i], s2, rtol=1e-12)
        nxt = th["omega"] + th["alpha"] * eps[-1, i] ** 2 + th["beta"] * s2[-1]
        assert m.sigma2_next_[i] == pytest.approx(nxt, rel=1e-12)
    # standardised residuals: unit variance (approximately)
    np.testing.assert_allclose(m.std_resid_.var(axis=0), 1.0, atol=0.1)


def test_ccc_simulate_starts_from_end_state(ccc_data):
    """start_from_end=True continues the recursion from sigma2_next_ (exact identity)."""
    m = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS)).fit(ccc_data)
    r = m.simulate(4, rng=np.random.default_rng(9), start_from_end=True)
    Z = m.innovations(4, rng=np.random.default_rng(9))
    s2 = m.sigma2_next_.copy()
    for t in range(4):
        e = np.sqrt(s2) * Z[t]
        np.testing.assert_allclose(r[t], m.mu_ + e, rtol=1e-12)
        s2 = m.omega_ + m.alpha_ * e**2 + m.beta_ * s2


def test_ccc_start_from_end_variance():
    """After a large shock, first simulated day has variance sigma2_next_ >> unconditional."""
    rng = np.random.default_rng(12)
    R = simulate_ccc_ref(1500, TRUE_P, TRUE_MARGINS, rng)
    R[-1] = np.array([t["mu"] for t in TRUE_MARGINS]) + np.array([-0.08, 0.07, -0.06])
    m = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS)).fit(R)
    assert np.all(m.sigma2_next_ > 3 * m.unconditional_variance())
    first = np.array([m.simulate(1, rng=rng, start_from_end=True)[0] for _ in range(4000)])
    np.testing.assert_allclose(first.var(axis=0), m.sigma2_next_, rtol=0.12)


def test_ccc_simulation_round_trip(ccc_data):
    """Long simulated path: unconditional variances, copula correlation and tails reproduced."""
    m = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS)).fit(ccc_data)
    S = m.simulate(40_000, rng=np.random.default_rng(13))
    assert np.all(np.isfinite(S))
    np.testing.assert_allclose(S.var(axis=0), m.unconditional_variance(), rtol=0.15)
    m2 = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS)).fit(S)
    np.testing.assert_allclose(m2.R_, m.R_, atol=0.03)
    assert np.all(stats.kurtosis(S, axis=0) > 1.0)  # GARCH-t returns are leptokurtic


def test_ccc_loglik_matches_reference(ccc_data):
    m = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS)).fit(ccc_data[:3000])
    R_test, R_hist = ccc_data[3000:], ccc_data[:3000]
    ll = m.loglik_obs(R_test, R_history=R_hist)
    eps = np.vstack([R_hist, R_test]) - m.mu_
    s2 = np.column_stack([
        _garch_filter_ref(eps[:, i], m.omega_[i], m.alpha_[i], m.beta_[i],
                          np.var(R_hist[:, i])) for i in range(3)])[3000:]
    Z = eps[3000:] / np.sqrt(s2)
    marg = (_std_t_logpdf_ref(Z, m.nu_) - 0.5 * np.log(s2)).sum(axis=1)
    U = stats.t.cdf(Z * np.sqrt(m.nu_ / (m.nu_ - 2)), m.nu_)
    X = stats.norm.ppf(U)
    cop = stats.multivariate_normal(np.zeros(3), m.R_).logpdf(X) - stats.norm.logpdf(X).sum(axis=1)
    np.testing.assert_allclose(ll, marg + cop, rtol=1e-9)
    # conditioning on the history matters only at the start of the test sample
    ll0 = m.loglik_obs(R_test)
    assert ll0.shape == ll.shape and np.all(np.isfinite(ll0))


def test_ccc_parameter_aliases(ccc_data):
    alias = ("const", "w", "a", "b", "df")
    m1 = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS)).fit(ccc_data)
    m2 = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS, keys=alias)).fit(ccc_data)
    np.testing.assert_array_equal(m1.R_, m2.R_)
    np.testing.assert_array_equal(m1.sigma2_next_, m2.sigma2_next_)
    with pytest.raises(KeyError):
        CCCGARCHt(margin_factory=stub_factory(
            TRUE_MARGINS, keys=("mu", "omega", "alpha", "beta", "shape"))).fit(ccc_data)


def test_ccc_detects_margin_unit_mismatch(ccc_data):
    """params() in different units from the margin's own likelihood is caught."""
    with pytest.raises(RuntimeError, match="units"):
        CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS, omega_scale=1e4)).fit(ccc_data)
    # without loglik_obs the check is skipped
    CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS, omega_scale=1.0,
                                          with_loglik=False)).fit(ccc_data)


def _near_unit_root_margins(R):
    """Margin 1 at the persistence bound with unconditional variance 60x its sample variance."""
    th = [dict(t) for t in TRUE_MARGINS]
    th[1].update(alpha=0.1, beta=0.8999)
    th[1]["omega"] = 60.0 * R[:, 1].var() * (1.0 - 0.9999)
    return th


def test_ccc_near_unit_root_diagnostics(ccc_data, caplog):
    th = _near_unit_root_margins(ccc_data)
    with caplog.at_level("WARNING", logger="synthval.models.multivariate"):
        m = CCCGARCHt(margin_factory=stub_factory(th, with_loglik=False)).fit(ccc_data)
    assert any("near a unit root" in r.getMessage() for r in caplog.records)
    np.testing.assert_array_equal(m.near_unit_root_, [False, True, False])
    np.testing.assert_allclose(m.persistence_, [0.96, 0.9999, 0.97])
    assert m.variance_ratio_[1] == pytest.approx(60.0, rel=1e-10)
    p = m.params()
    for k in ("persistence", "near_unit_root", "variance_ratio", "margin_at_bound", "sigma2_init"):
        assert k in p
    assert p["margin_at_bound"] == [[], [], []]  # the stub reports no fit_info_
    # bound flags reported by the margins are passed through
    flags = [None, ["persistence"], ["nu"]]
    it = iter([StubGARCHt(t, with_loglik=False, at_bound=f) for t, f in zip(th, flags)])
    m2 = CCCGARCHt(margin_factory=lambda: next(it)).fit(ccc_data)
    assert m2.margin_at_bound_ == [[], ["persistence"], ["nu"]]
    # no warning and no flag for ordinary margins
    caplog.clear()
    with caplog.at_level("WARNING", logger="synthval.models.multivariate"):
        m3 = CCCGARCHt(margin_factory=stub_factory(TRUE_MARGINS)).fit(ccc_data)
    assert not caplog.records and not m3.near_unit_root_.any()


@pytest.mark.parametrize("burn", [0, 7])
def test_ccc_near_unit_root_start_is_training_variance(ccc_data, burn):
    """Exact recursion: flagged margin restarts at sigma2_init_ on the first returned day."""
    th = _near_unit_root_margins(ccc_data)
    m = CCCGARCHt(margin_factory=stub_factory(th, with_loglik=False)).fit(ccc_data)
    r = m.simulate(5, rng=np.random.default_rng(4), burn_in=burn)
    Z = m.innovations(5 + burn, rng=np.random.default_rng(4))
    s2 = m.unconditional_variance()
    for t in range(5 + burn):
        if t == burn:
            s2 = np.where([False, True, False], m.sigma2_init_, s2)
        e = np.sqrt(s2) * Z[t]
        if t >= burn:
            np.testing.assert_allclose(r[t - burn], m.mu_ + e, rtol=1e-12)
        s2 = m.omega_ + m.alpha_ * e**2 + m.beta_ * s2
    # near_unit_root > 1 disables the rule: every margin starts at its unconditional variance
    m_off = CCCGARCHt(margin_factory=stub_factory(th, with_loglik=False),
                      near_unit_root=1.1).fit(ccc_data)
    assert not m_off.near_unit_root_.any()
    r_off = m_off.simulate(1, rng=np.random.default_rng(4), burn_in=0)
    Z0 = m_off.innovations(1, rng=np.random.default_rng(4))[0]
    np.testing.assert_allclose(r_off[0], m.mu_ + np.sqrt(m.unconditional_variance()) * Z0,
                               rtol=1e-12)


def test_ccc_near_unit_root_simulated_variance_is_contained(ccc_data):
    """TRAIN-length paths of the flagged margin: the upper tail of (simulated / training
    variance) is much smaller than with the unconditional (60x) start. The median is close to
    one in both cases (alpha = 0.1: the median variance path decays fast), so the rule acts on
    the dispersion, as on the real S&P 500 margins at the bound."""
    th = _near_unit_root_margins(ccc_data)
    m = CCCGARCHt(margin_factory=stub_factory(th, with_loglik=False)).fit(ccc_data)
    m_off = CCCGARCHt(margin_factory=stub_factory(th, with_loglik=False),
                      near_unit_root=1.1).fit(ccc_data)
    rng = np.random.default_rng(8)
    on = np.array([m.simulate(1000, rng=rng)[:, 1].var() for _ in range(80)])
    off = np.array([m_off.simulate(1000, rng=rng)[:, 1].var() for _ in range(80)])
    on, off = on / m.sigma2_init_[1], off / m.sigma2_init_[1]
    assert np.quantile(on, 0.75) < 3.0 < np.quantile(off, 0.75)
    assert np.mean(np.log(on)) < 0.6 < np.mean(np.log(off))


# ----------------------------------------------------------------------------------------------
# CCCGARCHt with the real univariate GARCHt
# ----------------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_GARCHT, reason=GARCHT_REASON)
def test_ccc_with_real_garcht_margins():
    rng = np.random.default_rng(21)
    R = simulate_ccc_ref(2500, TRUE_P, TRUE_MARGINS, rng)
    R_tr, R_te = R[:2300], R[2300:]
    m = CCCGARCHt().fit(R_tr)  # default factory: get_model("GARCHt"); also runs the unit check
    np.testing.assert_allclose(m.R_, TRUE_P, atol=0.05)
    persist = np.array([t["alpha"] + t["beta"] for t in TRUE_MARGINS])
    np.testing.assert_allclose(m.alpha_ + m.beta_, persist, atol=0.05)
    S = m.simulate(1000, rng=rng, start_from_end=True)
    assert S.shape == (1000, 3) and np.all(np.isfinite(S))
    # our filter reproduces each margin's own conditional log-densities, in and out of sample
    s2, Z = m.filter(R_te, R_history=R_tr)
    for i, g in enumerate(m.margins_):
        np.testing.assert_allclose(_std_t_logpdf_ref(Z[:, i], m.nu_[i]) - 0.5 * np.log(s2[:, i]),
                                   g.loglik_obs(R_te[:, i], r_history=R_tr[:, i]), rtol=1e-9)
        if hasattr(g, "forecast_variance"):
            assert m.sigma2_next_[i] == pytest.approx(g.forecast_variance(), rel=1e-9)
    assert np.isfinite(m.loglik(R_te, R_history=R_tr))
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m2.simulate(50, rng=np.random.default_rng(3)),
                                  m.simulate(50, rng=np.random.default_rng(3)))


# ----------------------------------------------------------------------------------------------
# MVStationaryBootstrap
# ----------------------------------------------------------------------------------------------
def test_mv_bootstrap_preserves_cross_correlation():
    rng = np.random.default_rng(31)
    C = np.array([[1, 0.7, 0.3, -0.2], [0.7, 1, 0.4, 0.0], [0.3, 0.4, 1, 0.5],
                  [-0.2, 0.0, 0.5, 1]])
    R = _sim_mvt(rng, np.zeros(4), 1e-4 * C, 5.0, 1500)
    m = MVStationaryBootstrap(mean_block=10).fit(R)
    S = m.simulate(60_000, rng=rng)
    np.testing.assert_allclose(np.corrcoef(S, rowvar=False), np.corrcoef(R, rowvar=False),
                               atol=0.03)


def test_mv_bootstrap_same_blocks_for_all_assets():
    rng = np.random.default_rng(32)
    R = rng.standard_normal((300, 5))
    m = MVStationaryBootstrap(mean_block=7.5).fit(R)
    S = m.simulate(1000, rng=np.random.default_rng(1))
    idx = stationary_bootstrap_indices(300, 1000, 7.5, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(S, R[idx])  # whole rows: one index per day for all assets


def test_stationary_bootstrap_indices_block_law_and_wraparound():
    rng = np.random.default_rng(33)
    T, n, b = 1000, 200_000, 10.0
    idx = stationary_bootstrap_indices(T, n, b, rng=rng)
    assert idx.min() >= 0 and idx.max() < T
    breaks = idx[1:] != (idx[:-1] + 1) % T
    assert breaks.mean() == pytest.approx(1.0 / b, abs=0.004)
    # stationarity: every row equally likely (counts aggregated over 10 bins of 100 rows)
    counts = np.bincount(idx, minlength=T).reshape(10, -1).sum(axis=1)
    np.testing.assert_allclose(counts / n, 0.1, atol=0.01)
    small = stationary_bootstrap_indices(50, 500, 1000.0, rng=rng)
    steps = np.c_[small[:-1], small[1:]]
    assert np.any((steps[:, 0] == 49) & (steps[:, 1] == 0))  # wraps around
    assert np.all(small < 50)
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(50, 10, 0.5, rng=rng)


def _garch_panel(rng, T=2000, N=3):
    """Independent GARCH(1,1)-t columns (strong volatility clustering, no autocorrelation)."""
    margins = [dict(mu=0.0, omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)] * N
    return simulate_ccc_ref(T, np.eye(N), margins, rng)


def test_mv_bootstrap_auto_block_matches_univariate_convention():
    """Per-asset lengths are exactly those of the univariate StationaryBootstrap (E3 = E4)."""
    from synthval.models.univariate import StationaryBootstrap, optimal_block_length

    rng = np.random.default_rng(34)
    R = _garch_panel(rng)
    R[:, 0] = R[:, 0] + 3.0 * np.convolve(rng.standard_normal(2009), np.ones(10) / 10,
                                          mode="valid") * 1e-3
    m = MVStationaryBootstrap().fit(R)
    uni = [StationaryBootstrap().fit(R[:, i]).block_length_ for i in range(3)]
    np.testing.assert_allclose(m.block_lengths_, uni, rtol=1e-12)
    assert m.mean_block_ == pytest.approx(np.median(uni))
    assert m.params()["block_series"] == "abs"
    # options: the rule on the returns themselves, and the mean across assets
    m_r = MVStationaryBootstrap(block_series="r", aggregate="mean").fit(R)
    ref_r = [optimal_block_length(R[:, i]) for i in range(3)]
    np.testing.assert_allclose(m_r.block_lengths_, ref_r, rtol=1e-12)
    assert m_r.mean_block_ == pytest.approx(np.mean(ref_r))
    m_sq = MVStationaryBootstrap(block_series="sq").fit(R)
    uni_sq = [StationaryBootstrap(block_series="sq").fit(R[:, i]).block_length_ for i in range(3)]
    np.testing.assert_allclose(m_sq.block_lengths_, uni_sq, rtol=1e-12)
    # GARCH columns 1, 2: |r| needs long blocks, r alone gives short ones
    assert np.all(m.block_lengths_[1:] > 3.0 * m_r.block_lengths_[1:])
    assert m.mean_block_ > 10.0
    # fixed length, short-sample default and errors
    assert MVStationaryBootstrap(mean_block=20).fit(R).params() == {
        "mean_block": 20.0, "block_series": None, "T": 2000, "N": 3}
    assert MVStationaryBootstrap().fit(R[:8]).mean_block_ == 8.0  # default 20, clipped to T
    assert MVStationaryBootstrap(default_block=5.0).fit(R[:15]).mean_block_ == 5.0
    with pytest.raises(ValueError):
        MVStationaryBootstrap(block_series="log")
    with pytest.raises(ValueError):
        MVStationaryBootstrap(aggregate="max")
    with pytest.raises(ValueError):
        MVStationaryBootstrap(mean_block=0.5)
    with pytest.raises(NotImplementedError):
        m.loglik_obs(R)


def test_mv_bootstrap_one_asset_equals_univariate_bootstrap():
    """Same rule and same index generator: with N = 1 the paths coincide draw for draw."""
    from synthval.models.univariate import StationaryBootstrap

    r = _garch_panel(np.random.default_rng(35), T=1500, N=1)[:, 0]
    mv = MVStationaryBootstrap().fit(r[:, None])
    uv = StationaryBootstrap().fit(r)
    assert mv.mean_block_ == pytest.approx(uv.block_length_, rel=1e-12)
    a = mv.simulate(800, rng=np.random.default_rng(6))[:, 0]
    b = uv.simulate(800, 1, rng=np.random.default_rng(6))[0]
    np.testing.assert_array_equal(a, b)


def test_mv_bootstrap_auto_block_keeps_volatility_clustering():
    """With the |r| rule the bootstrap keeps the ACF of |r| (the r rule destroys it)."""
    rng = np.random.default_rng(36)
    R = _garch_panel(rng, T=2500, N=2)

    def acf_abs(X, lag):
        A = np.abs(X - X.mean(axis=0))
        A = A - A.mean(axis=0)
        return float(np.mean(np.sum(A[lag:] * A[:-lag], axis=0) / np.sum(A * A, axis=0)))

    real = acf_abs(R, 10)
    sims = {}
    for series in ("abs", "r"):
        m = MVStationaryBootstrap(block_series=series).fit(R)
        sims[series] = np.mean([acf_abs(m.simulate(2500, rng=rng), 10) for _ in range(20)])
    assert real > 0.1
    assert abs(sims["abs"] - real) < 0.5 * real
    assert sims["r"] < 0.5 * real
