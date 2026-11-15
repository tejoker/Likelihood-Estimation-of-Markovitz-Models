"""Tests for synthval.models.univariate (GBM, Merton, GARCHt, Heston, StationaryBootstrap)."""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pytest
from scipy import optimize, stats

from synthval.models import UNIVARIATE_MODELS, get_model
from synthval.models.univariate import (
    SMM_MOMENT_NAMES,
    GARCHt,
    GBM,
    Heston,
    Merton,
    StationaryBootstrap,
    _cir_unit_integrals,
    _garch_nll_grad,
    _garch_nll_grad_vt,
    _heston_returns,
    _merton_core,
    _stationary_unit_variance,
    optimal_block_length,
    smm_moments,
    stationary_bootstrap_indices,
)

MERTON_TRUE = dict(mu=3e-4, sigma=0.01, lam=0.05, muJ=-0.01, sJ=0.03)
GARCH_TRUE = dict(mu=5e-4, omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)
HESTON_TRUE = dict(mu=3e-4, kappa=0.03, theta=1e-4, xi=Heston._xi(1e-4, 0.03, 1.0), rho=-0.6)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _garch_loop(r, mu, omega, alpha, beta, s2_0):
    """Naive GARCH(1,1) filter used as an independent reference."""
    s2 = np.empty(r.size)
    s2[0] = s2_0
    for t in range(1, r.size):
        s2[t] = omega + alpha * (r[t - 1] - mu) ** 2 + beta * s2[t - 1]
    return s2


# --------------------------------------------------------------------------------------------
# registry and common interface
# --------------------------------------------------------------------------------------------
def test_registry_names_and_flags():
    expected = {"GBM": True, "Merton": True, "GARCHt": True, "Heston": False, "Bootstrap": False}
    assert set(UNIVARIATE_MODELS) == set(expected)
    for name, lik in expected.items():
        m = get_model(name)
        assert m.name == name
        assert m.has_likelihood is lik


def test_params_constructor_validation():
    with pytest.raises(ValueError):
        GBM(mu=0.0)  # partial parameters
    with pytest.raises(ValueError):
        GARCHt(mu=0, omega=1e-6, alpha=0.5, beta=0.6, nu=5)  # alpha + beta >= 1
    with pytest.raises(RuntimeError):
        GBM().params()
    m = Merton.from_params(**MERTON_TRUE)
    assert m.params() == pytest.approx(MERTON_TRUE)
    assert m.n_params() == 5 and GBM(0, 1).n_params() == 2 and GARCHt(**GARCH_TRUE).n_params() == 5


@pytest.mark.parametrize("model", [GBM(mu=0, sigma=0.01), Merton(**MERTON_TRUE),
                                   GARCHt(**GARCH_TRUE), Heston(**HESTON_TRUE)])
def test_simulate_shape_and_reproducibility(model):
    a = model.simulate(30, 7, rng=_rng(5))
    b = model.simulate(30, 7, rng=_rng(5))
    assert a.shape == (7, 30) and a.dtype == np.float64
    assert np.all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)
    with pytest.raises(TypeError):
        model.simulate(30, 7, rng=5)


# --------------------------------------------------------------------------------------------
# GBM
# --------------------------------------------------------------------------------------------
def test_gbm_closed_form_and_loglik():
    r = _rng(0).normal(4e-4, 0.015, 5000)
    m = GBM().fit(r)
    assert m.params()["mu"] == pytest.approx(r.mean(), rel=1e-12)
    assert m.params()["sigma"] == pytest.approx(r.std(ddof=0), rel=1e-12)
    ref = stats.norm.logpdf(r, loc=r.mean(), scale=r.std())
    np.testing.assert_allclose(m.loglik_obs(r), ref, rtol=1e-12)
    assert m.loglik(r) == pytest.approx(ref.sum(), rel=1e-12)
    np.testing.assert_allclose(m.pit(r), stats.norm.cdf(r, r.mean(), r.std()), atol=1e-14)
    # recovery
    assert abs(m.params()["mu"] - 4e-4) < 4 * 0.015 / math.sqrt(5000)
    assert m.params()["sigma"] == pytest.approx(0.015, rel=0.05)
    # the MLE maximises the likelihood
    assert m.loglik(r) >= GBM(mu=4e-4, sigma=0.015).loglik(r)


def test_gbm_pit_uniform_under_true_model():
    m = GBM(mu=1e-4, sigma=0.02)
    u = m.pit(m.simulate(3000, 1, rng=_rng(1))[0])
    assert stats.kstest(u, "uniform").pvalue > 0.01


# --------------------------------------------------------------------------------------------
# Merton
# --------------------------------------------------------------------------------------------
def _merton_bruteforce(r, mu, sigma, lam, muJ, sJ, n_terms=50):
    n = np.arange(n_terms)
    w = stats.poisson.pmf(n, lam)
    dens = stats.norm.pdf(r[:, None], loc=mu + n * muJ, scale=np.sqrt(sigma ** 2 + n * sJ ** 2))
    cdf = stats.norm.cdf(r[:, None], loc=mu + n * muJ, scale=np.sqrt(sigma ** 2 + n * sJ ** 2))
    return np.log(dens @ w), cdf @ w


@pytest.mark.parametrize("pars", [MERTON_TRUE, dict(mu=0.0, sigma=0.008, lam=0.5, muJ=0.004,
                                                     sJ=0.01)])
def test_merton_loglik_vs_bruteforce_mixture(pars):
    m = Merton(**pars)
    r = m.simulate(2000, 1, rng=_rng(2))[0]
    ll_ref, cdf_ref = _merton_bruteforce(r, **pars)
    np.testing.assert_allclose(m.loglik_obs(r), ll_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(m.pit(r), cdf_ref, atol=1e-10)
    # far tails (10-25 sd): the truncation at P(N > n) < 1e-12 bounds the absolute, not the
    # relative, density error; the log-density error stays tiny
    ext = np.array([-0.2, -0.1, 0.1])
    np.testing.assert_allclose(m.loglik_obs(ext), _merton_bruteforce(ext, **pars)[0], atol=5e-3)


def test_merton_gradient():
    x = _rng(3).standard_t(4, 400)
    th = np.array([0.02, 0.8, 0.1, -0.4, 1.5])
    f = lambda t: float(np.sum(_merton_core(x, *t, n_max=11)[0]))
    g = _merton_core(x, *th, n_max=11, grad=True)[1]
    num = optimize.approx_fprime(th, f, 1e-7)
    np.testing.assert_allclose(g, num, rtol=1e-4, atol=1e-3)


def test_merton_parameter_recovery():
    true = Merton(**MERTON_TRUE)
    r = true.simulate(5000, 1, rng=_rng(4))[0]
    m = Merton().fit(r)
    p = m.params()
    assert m.fit_info_["success"]
    assert p["sigma"] == pytest.approx(0.01, rel=0.08)
    assert p["lam"] == pytest.approx(0.05, rel=0.5)
    assert p["sJ"] == pytest.approx(0.03, rel=0.3)
    assert abs(p["muJ"] - (-0.01)) < 0.01
    assert abs(p["mu"] - 3e-4) < 5e-4
    # the MLE beats the truth in-sample, and the reported loglik is consistent
    assert m.loglik(r) >= true.loglik(r) - 1e-6
    assert m.fit_info_["loglik"] == pytest.approx(m.loglik(r), abs=1e-6)
    # implied variance matches the sample variance
    var = p["sigma"] ** 2 + p["lam"] * (p["muJ"] ** 2 + p["sJ"] ** 2)
    assert var == pytest.approx(r.var(), rel=0.05)


def test_merton_fit_time_and_bounds():
    r = Merton(**MERTON_TRUE).simulate(1250, 1, rng=_rng(6))[0]
    t0 = time.perf_counter()
    m = Merton().fit(r)
    assert time.perf_counter() - t0 < 5.0
    p = m.params()
    assert 1e-4 <= p["lam"] <= 0.5 and p["sJ"] >= 1e-4 and p["sigma"] >= 1e-5
    # Gaussian data: still a valid fit, likelihood at least that of the GBM MLE
    g = _rng(7).normal(0, 0.01, 1250)
    mg = Merton().fit(g)
    assert mg.loglik(g) >= GBM().fit(g).loglik(g) - 1e-6


def test_merton_simulation_moments_and_pit():
    true = Merton(**MERTON_TRUE)
    x = true.simulate(200_000, 1, rng=_rng(8))[0]
    p = MERTON_TRUE
    mean = p["mu"] + p["lam"] * p["muJ"]
    var = p["sigma"] ** 2 + p["lam"] * (p["muJ"] ** 2 + p["sJ"] ** 2)
    assert x.mean() == pytest.approx(mean, abs=4 * math.sqrt(var / x.size))
    assert x.var() == pytest.approx(var, rel=0.03)
    u = true.pit(x[:3000])
    assert stats.kstest(u, "uniform").pvalue > 0.01


# --------------------------------------------------------------------------------------------
# GARCH(1,1)-t
# --------------------------------------------------------------------------------------------
def test_garch_gradient():
    x = _rng(9).standard_t(5, 600) * 0.8
    th = np.array([0.05, 0.05, 0.1, 0.85, 6.0])
    f = lambda t: _garch_nll_grad(t, x, 1.0)[0]
    num = optimize.approx_fprime(th, f, 1e-7)
    np.testing.assert_allclose(_garch_nll_grad(th, x, 1.0)[1], num, rtol=1e-4, atol=1e-6)
    # variance-targeted objective: omega = var_x (1 - alpha - beta), chain rule
    th4 = np.array([0.05, 0.1, 0.85, 6.0])
    vx = float(x.var())
    f4 = lambda t: _garch_nll_grad_vt(t, x, 1.0, vx)[0]
    assert f4(th4) == pytest.approx(f(np.array([0.05, vx * 0.05, 0.1, 0.85, 6.0])), rel=1e-14)
    np.testing.assert_allclose(_garch_nll_grad_vt(th4, x, 1.0, vx)[1],
                               optimize.approx_fprime(th4, f4, 1e-7), rtol=1e-4, atol=1e-6)


def test_garch_filter_matches_naive_loop_and_density():
    m = GARCHt(**GARCH_TRUE)
    r = m.simulate(800, 1, rng=_rng(10))[0]
    s2 = m.conditional_variance(r)
    ref = _garch_loop(r, GARCH_TRUE["mu"], GARCH_TRUE["omega"], GARCH_TRUE["alpha"],
                      GARCH_TRUE["beta"], r.var())
    np.testing.assert_allclose(s2, ref, rtol=1e-12)
    nu = GARCH_TRUE["nu"]
    scale = np.sqrt(ref * (nu - 2) / nu)  # unit-variance t has scale sqrt((nu-2)/nu)
    ll_ref = stats.t.logpdf(r, df=nu, loc=GARCH_TRUE["mu"], scale=scale)
    np.testing.assert_allclose(m.loglik_obs(r), ll_ref, rtol=1e-10)
    np.testing.assert_allclose(m.pit(r), stats.t.cdf(r, df=nu, loc=GARCH_TRUE["mu"], scale=scale),
                               atol=1e-12)


def test_garch_history_continuation_no_reset_no_offbyone():
    m = GARCHt(**GARCH_TRUE)
    x = m.simulate(1500, 1, rng=_rng(11))[0]
    r_train, r_test = x[:1000], x[1000:]
    full = np.concatenate([r_train, r_test])
    ll_cont = m.loglik_obs(r_test, r_history=r_train)
    ll_full = m.loglik_obs(full, sigma2_0=r_train.var())[-r_test.size:]
    np.testing.assert_allclose(ll_cont, ll_full, rtol=1e-12)
    u_full = m.pit(full, sigma2_0=r_train.var())[-r_test.size:]
    np.testing.assert_allclose(m.pit(r_test, r_history=r_train), u_full, atol=1e-14)
    # first test-period variance is the one-step forecast from the end of training
    s2_train = _garch_loop(r_train, *(GARCH_TRUE[k] for k in ("mu", "omega", "alpha", "beta")),
                           r_train.var())
    s2_first = (GARCH_TRUE["omega"] + GARCH_TRUE["alpha"] * (r_train[-1] - GARCH_TRUE["mu"]) ** 2
                + GARCH_TRUE["beta"] * s2_train[-1])
    assert m.conditional_variance(r_test, r_history=r_train)[0] == pytest.approx(s2_first, rel=1e-12)
    with pytest.raises(ValueError):
        m.loglik_obs(r_test, r_history=r_train, sigma2_0=1e-4)


def test_garch_parameter_recovery_and_speed():
    true = GARCHt(**GARCH_TRUE)
    r = true.simulate(5000, 1, rng=_rng(12))[0]
    m = GARCHt().fit(r)  # default: variance targeting
    mle = GARCHt(variance_targeting=False).fit(r)
    for fitted in (m, mle):
        p = fitted.params()
        assert fitted.fit_info_["success"]
        assert abs(p["alpha"] - 0.08) < 0.03
        assert abs(p["beta"] - 0.90) < 0.04
        assert p["nu"] == pytest.approx(6.0, rel=0.35)
        assert fitted.unconditional_variance() == pytest.approx(true.unconditional_variance(),
                                                                rel=0.3)
        assert abs(p["mu"] - 5e-4) < 5e-4
        assert fitted.fit_info_["loglik"] == pytest.approx(fitted.loglik(r), rel=1e-12)
    # variance targeting: the unconditional variance is the sample variance, exactly
    assert m.fit_info_["variance_targeting"] and not mle.fit_info_["variance_targeting"]
    assert m.unconditional_variance() == pytest.approx(r.var(), rel=1e-9)
    assert m.fit_info_["uncond_ratio"] == pytest.approx(1.0, rel=1e-9)
    # each estimator maximises its own (restricted / unrestricted) likelihood
    a, b = GARCH_TRUE["alpha"], GARCH_TRUE["beta"]
    true_vt = GARCHt(**{**GARCH_TRUE, "omega": r.var() * (1 - a - b)})
    assert m.loglik(r) >= true_vt.loglik(r) - 1e-6
    assert mle.loglik(r) >= true.loglik(r) - 1e-6
    assert mle.loglik(r) >= m.loglik(r) - 1e-6
    assert mle.loglik(r) - m.loglik(r) < 3.0  # the restriction costs little at the truth
    # per-ticker fit time on a TRAIN-length series
    r1 = true.simulate(1250, 1, rng=_rng(13))[0]
    t0 = time.perf_counter()
    GARCHt().fit(r1)
    assert time.perf_counter() - t0 < 2.0


def _arch_backcast(y: np.ndarray) -> float:
    """arch's default backcast: EWMA (0.94) of the first 75 squared demeaned observations."""
    e = y - y.mean()
    tau = min(75, y.size)
    w = 0.94 ** np.arange(tau)
    return float(np.sum(e[:tau] ** 2 * w / w.sum()))


def test_garch_matches_arch():
    arch = pytest.importorskip("arch")
    true = GARCHt(**GARCH_TRUE)
    r = true.simulate(5000, 1, rng=_rng(14))[0]
    T = r.size
    am = arch.arch_model(100 * r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off")
    mu, om, a, b, nu = (float(v) for v in res.params.values)
    arch_model_raw = GARCHt(mu=mu / 100, omega=om / 1e4, alpha=a, beta=b, nu=nu)
    bc = _arch_backcast(100 * r)
    s2_0_arch = lambda mod: (mod.params()["omega"] * 1e4
                             + (mod.params()["alpha"] + mod.params()["beta"]) * bc) / 1e4
    # 1) same log-likelihood at arch's parameters with arch's initialisation (scale-converted)
    ll_arch = res.loglikelihood + T * math.log(100.0)
    assert arch_model_raw.loglik(r, sigma2_0=s2_0_arch(arch_model_raw)) == pytest.approx(
        ll_arch, abs=1e-6)
    # 2) parameters agree (the two fits differ only through the initial variance)
    mine = GARCHt(variance_targeting=False).fit(r)  # arch: unrestricted MLE
    p = mine.params()
    assert abs(p["alpha"] - a) < 0.01
    assert abs(p["beta"] - b) < 0.015
    assert p["nu"] == pytest.approx(nu, rel=0.1)
    assert p["mu"] == pytest.approx(mu / 100, abs=1e-4)
    assert mine.unconditional_variance() == pytest.approx(arch_model_raw.unconditional_variance(),
                                                          rel=0.05)
    # 3) each optimiser attains the maximum of its own objective
    assert mine.loglik(r) >= arch_model_raw.loglik(r) - 1e-4
    assert ll_arch >= mine.loglik(r, sigma2_0=s2_0_arch(mine)) - 1e-2


def test_garch_pit_uniform_under_true_model():
    m = GARCHt(**GARCH_TRUE)
    x = m.simulate(4000, 1, rng=_rng(15))[0]
    u = m.pit(x[2000:], r_history=x[:2000])
    assert stats.kstest(u, "uniform").pvalue > 0.01


def test_garch_simulate_start_states():
    m = GARCHt(**GARCH_TRUE)
    r = m.simulate(1000, 1, rng=_rng(16))[0]
    m_fit = GARCHt().fit(r)
    s2_next = m_fit.forecast_variance()
    s2_train = m_fit.conditional_variance(r)
    p = m_fit.params()
    assert s2_next == pytest.approx(p["omega"] + p["alpha"] * (r[-1] - p["mu"]) ** 2
                                    + p["beta"] * s2_train[-1], rel=1e-12)
    first = m_fit.simulate(1, 100_000, rng=_rng(17), start_from_end=True)[:, 0]
    assert first.var() == pytest.approx(s2_next, rel=0.05)
    stat = m_fit.simulate(1, 20_000, rng=_rng(18), burn_in=300)[:, 0]
    assert stat.var() == pytest.approx(m_fit.unconditional_variance(), rel=0.15)
    # from_params model has no end state: falls back to stationarity without error
    assert m.simulate(5, 3, rng=_rng(19), start_from_end=True).shape == (3, 5)


def test_garch_window_loglik_starts_at_unconditional_variance():
    m = GARCHt(**GARCH_TRUE)
    W = m.simulate(20, 50, rng=_rng(20))
    ref = np.array([m.loglik(w, sigma2_0=m.unconditional_variance()) for w in W])
    np.testing.assert_allclose(m.loglik_windows(W), ref, rtol=1e-12)
    # Monte Carlo integration over initial variances: log-mean-exp of the filtered densities
    s2k = np.array([0.5, 1.0, 3.0]) * m.unconditional_variance()
    lk = np.array([[m.loglik(w, sigma2_0=s) for s in s2k] for w in W])
    ref_mc = np.log(np.mean(np.exp(lk - lk.max(axis=1, keepdims=True)), axis=1)) + lk.max(axis=1)
    np.testing.assert_allclose(m.loglik_windows(W, sigma2_init=s2k), ref_mc, rtol=1e-12)
    np.testing.assert_allclose(m.loglik_windows(W, sigma2_init=np.full(4, m.unconditional_variance())),
                               ref, rtol=1e-12)
    with pytest.raises(ValueError):
        m.loglik_windows(W, sigma2_init=np.array([1e-4, -1.0]))


def test_garch_stationary_variance_draws():
    m = GARCHt(**GARCH_TRUE)
    s2 = m.stationary_variance_draws(20_000, rng=_rng(30))
    assert s2.shape == (20_000,) and np.all(s2 > 0)
    # mean of the stationary law is the unconditional variance; its dispersion is large
    assert s2.mean() == pytest.approx(m.unconditional_variance(), rel=0.05)
    assert 0.2 < s2.std() / s2.mean() < 1.5
    # same law as the conditional variance along one long stationary path
    x = m.simulate(200_000, 1, rng=_rng(31), burn_in=2000)[0]
    s2_path = m.conditional_variance(x, sigma2_0=m.unconditional_variance())[1000:]
    assert stats.ks_2samp(s2, s2_path[::10]).pvalue > 0.001
    iid = GARCHt(mu=0.0, omega=1e-4, alpha=0.0, beta=0.0, nu=5.0)
    np.testing.assert_allclose(iid.stationary_variance_draws(10, rng=_rng(32)), 1e-4)


def test_garch_variance_targeting_bounds_near_integrated_fit():
    """A volatility regime shift makes the unrestricted MLE (near-)integrated with an inflated
    unconditional variance; variance targeting keeps simulated paths at the sample variance."""
    rng = _rng(33)
    garch = lambda scale: GARCHt(mu=0.0, omega=2e-6 * scale, alpha=0.06, beta=0.90, nu=4.0)
    # calm, a crisis with 25 times the variance, then 5 times the calm variance (like 2018-2022)
    r = np.concatenate([garch(1).simulate(1000, 1, rng=rng)[0],
                        garch(25).simulate(150, 1, rng=rng)[0],
                        garch(5).simulate(100, 1, rng=rng)[0]])
    mle = GARCHt(variance_targeting=False).fit(r)
    vt = GARCHt().fit(r)
    assert "persistence" in mle.fit_info_["at_bound"]  # spurious integrated GARCH
    assert mle.fit_info_["uncond_ratio"] > 10.0
    assert vt.fit_info_["uncond_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert "persistence" not in vt.fit_info_["at_bound"]
    assert vt.fit_info_["persistence"] < 0.999
    y_vt = vt.simulate(615, 1000, rng=_rng(34))
    y_mle = mle.simulate(615, 1000, rng=_rng(34))
    pv_vt, pv_mle = y_vt.var(axis=1) / r.var(), y_mle.var(axis=1) / r.var()
    assert 0.2 < np.median(pv_vt) < 2.0
    assert np.quantile(pv_vt, 0.99) < 20.0
    assert np.quantile(pv_mle, 0.99) > 5 * np.quantile(pv_vt, 0.99)
    # returns stay bounded: rare large days only (heavy t(4)-like tails), far below the MLE's
    max_vt, max_mle = np.abs(y_vt).max(axis=1), np.abs(y_mle).max(axis=1)
    assert np.mean(max_vt > 0.5) < 0.02
    assert max_vt.max() < 0.25 * max_mle.max()


# --------------------------------------------------------------------------------------------
# Heston
# --------------------------------------------------------------------------------------------
def test_heston_intractable_and_simulation():
    m = Heston(**HESTON_TRUE)
    assert m.cv() == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(NotImplementedError):
        m.loglik(np.zeros(10))
    with pytest.raises(NotImplementedError):
        m.pit(np.zeros(10))
    with pytest.raises(ValueError):
        Heston(mu=0.0, kappa=15.0, theta=1e-4, xi=1e-3, rho=0.0)  # kappa h > 0.5
    x = m.simulate(20_000, 4, rng=_rng(21))
    assert np.all(np.isfinite(x))
    # Var[r] = E[int v] = theta up to the tiny -v/2 drift term
    assert x.var() == pytest.approx(HESTON_TRUE["theta"], rel=0.1)
    # c = 1: Var(v) = theta^2, so the kurtosis of r is about 3 (1 + c^2) = 6
    assert 1.5 < stats.kurtosis(x.ravel()) < 5.0
    # no day has exactly zero variance (the artefact of a one-step daily Euler scheme)
    assert np.mean(x == HESTON_TRUE["mu"]) == 0.0
    # leverage: corr(r_t, r_{t+1}^2) has the sign of rho
    assert smm_moments(x).mean(axis=0)[-1] < -0.05


def test_smm_moments_reference():
    x = _rng(22).standard_t(5, 3000) * 0.01
    m = smm_moments(x)
    assert m.shape == (len(SMM_MOMENT_NAMES),) == (8,)
    d = x - x.mean()
    s = d ** 2
    sd = s - s.mean()
    assert m[0] == pytest.approx(np.var(x), rel=1e-12)
    assert m[1] == pytest.approx(np.mean(np.abs(d)) / np.std(x), rel=1e-12)
    assert m[2] == pytest.approx(math.log(stats.kurtosis(x, fisher=False)), rel=1e-10)
    for j, k in enumerate((1, 5, 10, 20)):
        assert m[3 + j] == pytest.approx(np.sum(sd[k:] * sd[:-k]) / np.sum(sd ** 2), rel=1e-10)
    assert m[7] == pytest.approx(np.corrcoef(d[:-1], d[1:] ** 2)[0, 1], rel=1e-10)
    np.testing.assert_allclose(smm_moments(np.vstack([x, x]))[1], m)
    g = _rng(1).standard_normal(20_000)
    assert smm_moments(g)[1] == pytest.approx(math.sqrt(2 / math.pi), abs=0.02)


def _exact_cir_daily_integral(kappa, c, n_days, n_paths, n_sub, rng):
    """Reference: unit-mean CIR with exact non-central chi-square transitions on a fine grid,
    daily integrated variance by the trapezoidal rule."""
    xi = c * math.sqrt(2 * kappa)
    h = 1.0 / n_sub
    e = math.exp(-kappa * h)
    cst = xi * xi * (1 - e) / (4 * kappa)
    df = 4 * kappa / xi ** 2
    v = rng.gamma(1 / c ** 2, c ** 2, size=n_paths)
    out = np.empty((n_days, n_paths))
    for t in range(n_days):
        acc = 0.5 * v
        for k in range(n_sub):
            v = cst * rng.noncentral_chisquare(df, np.maximum(v * e / cst, 1e-300))
            acc += v if k < n_sub - 1 else 0.5 * v
        out[t] = acc * h
    return out.ravel()


def test_cir_substep_scheme_against_exact_transition():
    """20 full-truncation sub-steps reproduce the law of the daily integrated variance far from
    the Feller condition (c = 2, Feller ratio 0.25); one daily step does not."""
    kappa, c = 0.05, 2.0
    ref = _exact_cir_daily_integral(kappa, c, 150, 400, 50, _rng(40))
    rng = _rng(41)
    v0 = rng.gamma(1 / c ** 2, c ** 2, size=(1, 400))
    I20 = _cir_unit_integrals(kappa, c, rng.standard_normal((150, 20, 400)), v0)[0].ravel()
    I1 = _cir_unit_integrals(kappa, c, rng.standard_normal((150, 1, 400)), v0)[0].ravel()
    assert np.mean(I20 == 0.0) < 0.002
    assert stats.ks_2samp(I20, ref).statistic < 0.03
    assert np.mean(I1 == 0.0) > 0.15  # the daily scheme sticks at zero variance
    assert stats.ks_2samp(I1, ref).statistic > 0.1
    # stationary mean 1 and coefficient of variation c of the unit-mean process
    v = _stationary_unit_variance(np.array([0.5, 1.0, 2.0]), _rng(42).random(200_000))
    np.testing.assert_allclose(v.mean(axis=1), 1.0, rtol=0.03)
    np.testing.assert_allclose(v.std(axis=1), [0.5, 1.0, 2.0], rtol=0.05)


def test_cir_integrals_identity_and_returns():
    """Euler bookkeeping: sum of variance increments = kappa (1 - I) + xi1 J over each day."""
    kappa, c = np.array([0.02, 0.3]), np.array([1.5, 0.7])
    rng = _rng(43)
    z = rng.standard_normal((30, 20, 64))
    v0 = _stationary_unit_variance(c, rng.random(64))
    I, J, v_end = _cir_unit_integrals(kappa, c, z, v0)
    assert I.shape == J.shape == (30, 2, 64) and v_end.shape == (2, 64)
    # rerun day by day to recover the daily start/end variances
    v = v0
    for t in range(30):
        It, Jt, v_next = _cir_unit_integrals(kappa, c, z[t: t + 1], v)
        np.testing.assert_allclose(It[0], I[t], rtol=1e-12)
        xi1 = (c * np.sqrt(2 * kappa))[:, None]
        np.testing.assert_allclose(v_next - v, kappa[:, None] * (1 - I[t]) + xi1 * J[t],
                                   rtol=1e-9, atol=1e-12)
        v = v_next
    np.testing.assert_allclose(v, v_end, rtol=1e-12)
    # returns: conditional variance given the path is theta I (exact orthogonal part)
    A = np.sqrt(I[:, 0]) * rng.standard_normal((30, 64))
    r0 = _heston_returns(I[:, 0], J[:, 0], A, 2e-4, 0.0, mu=1e-3)
    np.testing.assert_allclose(r0, 1e-3 - 1e-4 * I[:, 0] + math.sqrt(2e-4) * A, rtol=1e-12)


def _heavy_tailed_series(T: int, seed: int) -> np.ndarray:
    """A GARCH-t(4) series: kurtosis and volatility clustering similar to daily stock returns."""
    return GARCHt(mu=4e-4, omega=8e-6, alpha=0.10, beta=0.86, nu=4.0).simulate(
        T, 1, rng=_rng(seed))[0]


def test_heston_fit_heavy_tailed_no_constant_returns_and_optimum():
    r = _heavy_tailed_series(1250, 44)
    t0 = time.perf_counter()
    m = Heston().fit(r)
    elapsed = time.perf_counter() - t0
    fi = m.fit_info_
    p = m.params()
    assert elapsed < 60.0
    assert fi["m_sim"].shape == fi["m_data"].shape == fi["t"].shape == (8,)
    assert fi["moment_names"] == SMM_MOMENT_NAMES
    assert Heston.C_BOUNDS[0] <= fi["c"] <= Heston.C_BOUNDS[1]
    assert p["xi"] == pytest.approx(Heston._xi(p["theta"], p["kappa"], fi["c"]), rel=1e-12)
    assert fi["feller_ratio"] == pytest.approx(2 * p["kappa"] * p["theta"] / p["xi"] ** 2)
    # the variance moment pins theta near the sample variance
    assert p["theta"] == pytest.approx(r.var(), rel=0.3)
    # discretisation artefact gone: no day of zero integrated variance, no constant returns
    assert fi["frac_zero_var"] < 0.002
    y = m.simulate(615, 200, rng=_rng(45))
    assert np.mean(y == p["mu"]) < 0.002
    assert y.mean() == pytest.approx(r.mean(), abs=3 * r.std() / math.sqrt(y.size) + 2e-4)
    # tails and bulk: weighted deviations of the shape moments are moderate
    assert np.all(np.abs(fi["t"][1:3]) < 3.0)
    # the optimiser beats a brute-force scan of the profile objective (same random numbers)
    prob = m.smm_problem(r)
    assert prob.profile(np.log([p["kappa"], fi["c"]])) == pytest.approx(fi["objective"], rel=1e-9)
    scan = [prob.profile(np.log([k, c])) for k in (0.01, 0.05, 0.2) for c in (0.5, 1.2, 2.0)]
    assert fi["objective"] <= min(scan) + 1e-9
    assert fi["objective"] <= fi["grid_best"]["objective"] + 1e-9
    # start_from_end uses the EWMA end-of-sample variance
    assert m.v_end_ is not None and m.v_end_ > 0
    first = m.simulate(1, 20_000, rng=_rng(46), start_from_end=True)[:, 0]
    # E[int_0^1 v] = theta + (v_0 - theta) (1 - exp(-kappa)) / kappa from v_0 = v_end
    k, th = p["kappa"], p["theta"]
    assert first.var() == pytest.approx(th + (m.v_end_ - th) * (1 - math.exp(-k)) / k, rel=0.1)


@pytest.mark.slow
def test_heston_parameter_recovery():
    true = Heston(**HESTON_TRUE)
    r = true.simulate(5000, 1, rng=_rng(23))[0]
    m = Heston().fit(r)
    p = m.params()
    # theta is pinned by the variance moment: close to the realised variance, and to the truth
    # up to the (large, kappa = 0.03) sampling error of a 5000-day variance
    assert p["theta"] == pytest.approx(r.var(), rel=0.15)
    assert p["theta"] == pytest.approx(HESTON_TRUE["theta"], rel=0.4)
    assert HESTON_TRUE["kappa"] / 3 < p["kappa"] < HESTON_TRUE["kappa"] * 3
    assert 0.6 < m.fit_info_["c"] < 1.6  # true coefficient of variation of v is 1
    # rho is identified by corr(r_t, r_{t+1}^2) only, whose sampling sd over 5000 days (0.024)
    # is a third of its mean under the truth (-0.064): only the sign is reliably recovered
    assert -0.95 <= p["rho"] < -0.05
    assert m.fit_info_["m_sim"][-1] == pytest.approx(m.fit_info_["m_data"][-1], abs=0.03)
    assert abs(p["mu"] - HESTON_TRUE["mu"]) < 5e-4
    assert m.fit_info_["objective"] < 20.0
    assert m.fit_info_["frac_zero_var"] == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("ticker", ["TDG", "ROK"])
def test_heston_real_data_no_constant_returns(ticker):
    """On the S&P 500 tickers where the daily Euler scheme gave 39-53% constant-return days,
    the fitted model has (almost) no day of zero integrated variance."""
    from synthval import config
    if not config.DATA_CSV.exists():
        pytest.skip("data file not available")
    data = pytest.importorskip("synthval.data")
    try:
        train, _ = data.load_panel([ticker])
    except (ValueError, KeyError) as exc:
        pytest.skip(f"{ticker} not loadable: {exc}")
    r = train[ticker].to_numpy()
    m = Heston().fit(r)
    assert m.fit_info_["frac_zero_var"] < 0.002
    y = m.simulate(615, 200, rng=_rng(47))
    assert np.mean(y == m.params()["mu"]) < 0.002
    # windows of 20 days are never constant
    assert not np.any(np.all(np.diff(y[:, :600].reshape(-1, 20), axis=1) == 0, axis=1))


# --------------------------------------------------------------------------------------------
# stationary bootstrap
# --------------------------------------------------------------------------------------------
def _ar1(phi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(n + 200)
    x = np.zeros_like(z)
    for t in range(1, z.size):
        x[t] = phi * x[t - 1] + z[t]
    return x[200:]


def test_optimal_block_length_ar1_theory_and_arch():
    x = _ar1(0.5, 5000, _rng(25))
    b = optimal_block_length(x)
    # theory for AR(1): b_SB = (2 phi / (1 - phi^2))^{2/3} n^{1/3} = 20.7 at phi = 0.5, n = 5000
    theory = (2 * 0.5 / (1 - 0.25)) ** (2 / 3) * 5000 ** (1 / 3)
    assert theory * 0.65 < b < theory * 1.5
    arch_bs = pytest.importorskip("arch.bootstrap")
    ref = arch_bs.optimal_block_length(x)
    assert b == pytest.approx(float(ref["stationary"].iloc[0]), rel=0.15)
    assert optimal_block_length(x, kind="circular") == pytest.approx(
        float(ref["circular"].iloc[0]), rel=0.15)
    # circular blocks are longer than stationary mean blocks by (3/2)^{1/3}
    assert optimal_block_length(x, kind="circular") / b == pytest.approx(1.5 ** (1 / 3), rel=1e-9)


def test_optimal_block_length_degenerate_fallback():
    assert optimal_block_length(np.ones(500)) == 20.0
    assert optimal_block_length(np.arange(10.0)) == 20.0
    b = optimal_block_length(_rng(26).standard_normal(2000))
    assert 1.0 <= b < 10.0  # white noise: short blocks


def test_stationary_bootstrap_indices_structure():
    T, b = 300, 8.0
    idx = stationary_bootstrap_indices(T, 2000, 50, b, rng=_rng(27))
    assert idx.shape == (50, 2000) and idx.min() >= 0 and idx.max() < T
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % T)
    # a block continues with probability 1 - 1/b (up to the rare uniform restart at i + 1)
    assert cont.mean() == pytest.approx(1 - 1 / b + 1 / (b * T), abs=0.01)
    # circular wrap: T-1 is followed by 0 inside blocks
    wrap = idx[:, :-1] == T - 1
    assert np.mean(idx[:, 1:][wrap] == 0) == pytest.approx(1 - 1 / b, abs=0.05)
    # every index is used about equally often (stationarity of the circular scheme)
    counts = np.bincount(idx.ravel(), minlength=T)
    assert counts.min() > 0.6 * idx.size / T and counts.max() < 1.4 * idx.size / T


def test_stationary_bootstrap_model():
    r = GARCHt(**GARCH_TRUE).simulate(1250, 1, rng=_rng(28))[0]
    m = StationaryBootstrap().fit(r)
    assert m.has_likelihood is False
    b = m.params()["block_length"]
    assert 1.0 <= b <= math.ceil(min(3 * math.sqrt(1250), 1250 / 3))
    assert b == pytest.approx(optimal_block_length(np.abs(r - r.mean())), rel=1e-12)
    y = m.simulate(500, 20, rng=_rng(29))
    assert y.shape == (20, 500)
    assert np.isin(y, r).all()
    assert y.var() == pytest.approx(r.var(), rel=0.2)
    fixed = StationaryBootstrap(block_length=20).fit(r)
    assert fixed.params()["block_length"] == 20.0
    with pytest.raises(NotImplementedError):
        m.loglik(r)
