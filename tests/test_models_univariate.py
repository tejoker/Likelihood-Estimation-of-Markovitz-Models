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
    _garch_nll_grad,
    _garch_nll_grad_vt,
    _heston_returns,
    _merton_core,
    _qe_integrals,
    _qe_unit_variance,
    _stationary_unit_variance,
    optimal_block_length,
    smm_moments,
    stationary_bootstrap_indices,
    tail_quantile_ratio,
)

MERTON_TRUE = dict(mu=3e-4, sigma=0.01, lam=0.05, muJ=-0.01, sJ=0.03)
GARCH_TRUE = dict(mu=5e-4, omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)
HESTON_TRUE = dict(mu=3e-4, kappa=0.03, theta=1e-4, xi=Heston._xi(1e-4, 0.03, 1.0), rho=-0.6)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _load_train(tickers):
    """TRAIN returns of real tickers (skips the test when the data file is unavailable)."""
    from synthval import config
    if not config.DATA_CSV.exists():
        pytest.skip("data file not available")
    from synthval import data
    try:
        train, _ = data.load_panel(list(tickers))
    except (ValueError, KeyError) as exc:
        pytest.skip(f"{tickers} not loadable: {exc}")
    return train


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


def test_merton_intensity_at_bound_is_reported():
    """Daily jump intensity above the 0.5 bound (frequent small jumps, as for GOOG/MTB/NVDA):
    the fit stops at the bound and says so."""
    r = Merton(mu=0.0, sigma=0.008, lam=2.0, muJ=0.0, sJ=0.006).simulate(2000, 1, rng=_rng(60))[0]
    m = Merton().fit(r)
    assert m.params()["lam"] == pytest.approx(0.5, rel=1e-6)
    assert m.fit_info_["at_bound"] == ["lam"]
    assert Merton().fit(Merton(**MERTON_TRUE).simulate(3000, 1, rng=_rng(61))[0]).fit_info_[
        "at_bound"] == []
    assert GBM().fit(r).fit_info_["at_bound"] == []  # closed form: no bounds


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
    m = GARCHt().fit(r)  # default: variance targeting, persistence <= 0.995
    mle = GARCHt(variance_targeting=False, persist_max=0.9999).fit(r)
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
    assert m.fit_info_["persist_max"] == 0.995 and mle.fit_info_["persist_max"] == 0.9999
    assert m.fit_info_["at_bound"] == [] and mle.fit_info_["at_bound"] == []
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
    mine = GARCHt(variance_targeting=False, persist_max=0.9999).fit(r)  # arch: unrestricted MLE
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


def test_garch_persist_max_option_and_bounds():
    r = GARCHt(**GARCH_TRUE).simulate(3000, 1, rng=_rng(62))[0]
    capped = GARCHt(persist_max=0.9).fit(r)  # binding cap (true persistence 0.98)
    assert capped.fit_info_["persistence"] == pytest.approx(0.9, abs=1e-6)
    assert "persistence" in capped.fit_info_["at_bound"]
    assert capped.unconditional_variance() == pytest.approx(r.var(), rel=1e-9)
    free = GARCHt(variance_targeting=False, persist_max=0.9).fit(r)
    assert free.params()["alpha"] + free.params()["beta"] <= 0.9 + 1e-9
    assert "persistence" in free.fit_info_["at_bound"]
    with pytest.raises(ValueError):
        GARCHt(persist_max=1.0)
    with pytest.raises(ValueError):
        GARCHt(persist_max=0.0)
    # construction from known parameters is not restricted by the cap
    assert GARCHt(mu=0.0, omega=1e-7, alpha=0.05, beta=0.949, nu=5.0).unconditional_variance() > 0


def test_garch_rcl_variance_targeting_no_explosive_paths():
    """RCL (2020 regime shift): the unrestricted MLE sits at alpha + beta = 0.9999 with an
    unconditional variance about 70 times the sample variance and explosive paths; the default
    (variance targeting, persistence <= 0.995) keeps the volatility level of the data."""
    r = _load_train(["RCL"])["RCL"].to_numpy()
    mle = GARCHt(variance_targeting=False, persist_max=0.9999).fit(r)
    m = GARCHt().fit(r)
    assert "persistence" in mle.fit_info_["at_bound"] and mle.fit_info_["uncond_ratio"] > 20
    assert m.fit_info_["uncond_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert m.fit_info_["persistence"] <= 0.995 + 1e-9
    y = m.simulate(615, 2000, rng=_rng(63))
    y_mle = mle.simulate(615, 2000, rng=_rng(63))
    mx, mx_mle = np.abs(y).max(axis=1), np.abs(y_mle).max(axis=1)
    pv = y.var(axis=1) / r.var()
    assert 0.3 < np.median(pv) < 1.5  # paths at the volatility level of the data
    assert np.mean(mx > 1.0) < 0.025 < np.mean(mx_mle > 1.0)  # measured 1.35% vs 5.85%
    assert np.median(mx) < 0.3 and mx.max() < 0.5 * mx_mle.max()
    # Not tested: max |r| < 1 on every path. 0.7-1.4% of the 615-day paths still exceed 1:
    # the power-law tail of a stationary GARCH-t with alpha ~ 0.10 and nu ~ 4.6
    # (E[(alpha z^2 + beta)^2] > 1, infinite fourth moment), which a lower persistence cap
    # does not remove (the likelihood then raises alpha).
    assert np.isfinite(y).all()


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
    mle = GARCHt(variance_targeting=False, persist_max=0.9999).fit(r)
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
        Heston(mu=0.0, kappa=0.03, theta=1e-4, xi=1e-3, rho=1.0)
    x = m.simulate(20_000, 4, rng=_rng(21))
    assert np.all(np.isfinite(x))
    # Var[r] = E[int v] = theta up to the tiny -v/2 drift term
    assert x.var() == pytest.approx(HESTON_TRUE["theta"], rel=0.1)
    # c = 1: Var(v) = theta^2, so the kurtosis of r is about 3 (1 + c^2) = 6
    assert 1.5 < stats.kurtosis(x.ravel()) < 5.0
    # no day has exactly zero variance (c = 1 <= sqrt(psi_c): QE quadratic branch only)
    assert np.mean(x == HESTON_TRUE["mu"]) == 0.0
    assert np.mean(np.diff(x, axis=1) == 0.0) == 0.0
    # leverage: corr(r_t, r_{t+1}^2) has the sign of rho
    assert smm_moments(x).mean(axis=0)[-1] < -0.05
    # simulation in blocks of paths (bounded memory) is stationary in every block
    y = m.simulate(5, 3000, rng=_rng(22), burn_in=1500)
    assert y.shape == (3000, 5)
    assert y[:1000].var() == pytest.approx(y[2000:].var(), rel=0.25)


def test_smm_moments_reference():
    x = _rng(22).standard_t(5, 3000) * 0.01
    m = smm_moments(x)
    assert m.shape == (len(SMM_MOMENT_NAMES),) == (7,)
    assert SMM_MOMENT_NAMES[1] == "tail_qr"
    d = x - x.mean()
    s = d ** 2
    sd = s - s.mean()
    assert m[0] == pytest.approx(np.var(x), rel=1e-12)
    q01, q25, q75, q99 = np.quantile(x, [0.01, 0.25, 0.75, 0.99])
    assert m[1] == pytest.approx((q99 - q01) / (q75 - q25), rel=1e-12)
    assert m[1] == pytest.approx(tail_quantile_ratio(x)[0], rel=1e-12)
    for j, k in enumerate((1, 5, 10, 20)):
        assert m[2 + j] == pytest.approx(np.sum(sd[k:] * sd[:-k]) / np.sum(sd ** 2), rel=1e-10)
    assert m[6] == pytest.approx(np.corrcoef(d[:-1], d[1:] ** 2)[0, 1], rel=1e-10)
    np.testing.assert_allclose(smm_moments(np.vstack([x, x]))[1], m)
    # tail ratio: 3.45 for a Gaussian, 5.0 for a t(4); scale and location free
    g = _rng(1).standard_normal((20, 5000))
    assert tail_quantile_ratio(g).mean() == pytest.approx(3.449, abs=0.05)
    assert tail_quantile_ratio(_rng(2).standard_t(4, (20, 5000))).mean() == pytest.approx(5.0, abs=0.15)
    assert tail_quantile_ratio(3.0 * x + 1.0)[0] == pytest.approx(m[1], rel=1e-12)


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


def _andersen_qe_reference(v0, kappa, theta, xi, z, psi_c=1.5):
    """Andersen (2008) QE variance step written out for one path and general (theta, xi)."""
    e = math.exp(-kappa)
    v = [float(v0)]
    for zt in z:
        vt = v[-1]
        m = theta + (vt - theta) * e
        s2 = vt * xi ** 2 * e * (1 - e) / kappa + theta * xi ** 2 * (1 - e) ** 2 / (2 * kappa)
        psi = s2 / m ** 2
        if psi <= psi_c:
            b2 = 2 / psi - 1 + math.sqrt(2 / psi) * math.sqrt(2 / psi - 1)
            v.append(m / (1 + b2) * (math.sqrt(b2) + zt) ** 2)
        else:
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m
            u = stats.norm.cdf(zt)
            v.append(0.0 if u <= p else math.log((1 - p) / (1 - u)) / beta)
    return np.array(v)


def test_qe_scheme_matches_andersen_and_scales():
    """The vectorised unit-mean QE step equals Andersen's formulas for general (theta, xi)
    divided by theta (scaling property), in both branches; the returns are Andersen's
    log-price step with K0..K4 and gamma1 = gamma2 = 1/2."""
    rng = _rng(43)
    kappa, theta = 0.04, 2e-4
    for c in (0.8, 2.0):  # quadratic branch only / both branches
        xi = Heston._xi(theta, kappa, c)
        z = rng.standard_normal((300, 3))
        v0 = np.array([0.05, 1.0, 3.0])
        V = _qe_unit_variance(kappa, c, z, v0)
        assert V.shape == (301, 1, 3)
        for j in range(3):
            ref = _andersen_qe_reference(theta * v0[j], kappa, theta, xi, z[:, j])
            np.testing.assert_allclose(theta * V[:, 0, j], ref, rtol=1e-9, atol=1e-18)
        if c > math.sqrt(1.5):
            assert np.mean(V == 0.0) > 0.05  # exponential branch: point mass at zero
        else:
            assert np.all(V > 0.0)
        # Andersen's log-price step with gamma1 = gamma2 = 1/2 (dt = 1)
        I, J = _qe_integrals(V, kappa, c)
        Zr = rng.standard_normal(I.shape)
        rho, mu = -0.6, 3e-4
        r = _heston_returns(I, J, np.sqrt(I) * Zr, theta, rho, mu)
        K0 = -rho * kappa * theta / xi
        K1 = 0.5 * (kappa * rho / xi - 0.5) - rho / xi
        K2 = 0.5 * (kappa * rho / xi - 0.5) + rho / xi
        K3 = K4 = 0.5 * (1 - rho ** 2)
        Vt, Vn = theta * V[:-1], theta * V[1:]
        r_ref = mu + K0 + K1 * Vt + K2 * Vn + np.sqrt(K3 * Vt + K4 * Vn) * Zr
        np.testing.assert_allclose(r, r_ref, rtol=1e-9, atol=1e-15)
    # several parameter sets side by side share the shocks
    z = rng.standard_normal((50, 4))
    Vb = _qe_unit_variance(np.array([0.02, 0.3]), np.array([1.1, 0.5]), z, 1.0)
    np.testing.assert_allclose(Vb[:, 1], _qe_unit_variance(0.3, 0.5, z, 1.0)[:, 0], rtol=1e-14)


def test_qe_one_step_moments_and_daily_integral_law():
    """One QE step has the exact conditional mean and variance of the CIR transition, and the
    daily QE integrated variance is close in law to the exact one at c = 1.2 (Feller ratio
    0.69); above sqrt(psi_c) the scheme puts mass at zero (why Heston.fit bounds c)."""
    rng = _rng(44)
    for kappa, c, v in ((0.05, 1.2, 0.3), (0.5, 0.7, 2.0), (0.05, 2.0, 0.05)):
        e = math.exp(-kappa)
        m = 1 + (v - 1) * e
        s2 = 2 * c * c * (v * e * (1 - e) + 0.5 * (1 - e) ** 2)
        x = _qe_unit_variance(kappa, c, rng.standard_normal((1, 400_000)), v)[1, 0]
        assert x.mean() == pytest.approx(m, rel=0.01)
        assert x.var() == pytest.approx(s2, rel=0.03)
    for kappa, c in ((0.05, 1.2), (0.5, 1.2)):
        ref = _exact_cir_daily_integral(kappa, c, 150, 400, 50, _rng(40))
        rng = _rng(41)
        v0 = rng.gamma(1 / c ** 2, c ** 2, size=400)
        V = _qe_unit_variance(kappa, c, rng.standard_normal((150, 400)), v0)
        I = _qe_integrals(V, kappa, c)[0].ravel()
        assert np.all(V > 0.0)
        assert stats.ks_2samp(I, ref).statistic < 0.05
    V2 = _qe_unit_variance(0.05, 2.0, _rng(45).standard_normal((300, 400)),
                           _rng(46).gamma(0.25, 4.0, size=400))
    I2 = _qe_integrals(V2, 0.05, 2.0)[0]
    assert np.mean(I2 == 0.0) > 0.1  # constant-return days with c = 2
    # stationary mean 1 and coefficient of variation c of the unit-mean process
    v = _stationary_unit_variance(np.array([0.5, 1.0, 2.0]), _rng(42).random(200_000))
    np.testing.assert_allclose(v.mean(axis=1), 1.0, rtol=0.03)
    np.testing.assert_allclose(v.std(axis=1), [0.5, 1.0, 2.0], rtol=0.05)


def test_heston_above_sqrt_psi_c_has_zero_variance_days(caplog):
    """A model built with c > sqrt(psi_c) (allowed from parameters) simulates days with zero
    variance, and says so."""
    theta, kappa = 1e-4, 0.05
    m = Heston(mu=1e-4, kappa=kappa, theta=theta, xi=Heston._xi(theta, kappa, 2.0), rho=-0.5)
    with caplog.at_level("WARNING", logger="synthval.models.univariate"):
        y = m.simulate(1000, 50, rng=_rng(47))
    assert "zero variance" in caplog.text
    assert np.mean(np.diff(y, axis=1) == 0.0) > 0.05  # runs of constant returns


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
    assert fi["m_sim"].shape == fi["m_data"].shape == fi["t"].shape == (7,)
    assert fi["moment_names"] == SMM_MOMENT_NAMES
    assert fi["scheme"] == "QE" and fi["psi_c"] == 1.5
    assert Heston.C_BOUNDS[0] <= fi["c"] <= math.sqrt(1.5) + 1e-12
    assert p["xi"] == pytest.approx(Heston._xi(p["theta"], p["kappa"], fi["c"]), rel=1e-12)
    assert fi["feller_ratio"] == pytest.approx(2 * p["kappa"] * p["theta"] / p["xi"] ** 2)
    # the variance moment pins theta near the sample variance
    assert p["theta"] == pytest.approx(r.var(), rel=0.3)
    # no day of zero variance, no constant return
    assert fi["frac_zero_var"] == 0.0 and fi["frac_const"] == 0.0
    y = m.simulate(615, 200, rng=_rng(45))
    assert np.mean(y == p["mu"]) == 0.0 and np.mean(np.diff(y, axis=1) == 0.0) == 0.0
    assert y.mean() == pytest.approx(r.mean(), abs=3 * r.std() / math.sqrt(y.size) + 2e-4)
    # the tail moment carries weight: its weighted deviation is moderate
    j = SMM_MOMENT_NAMES.index("tail_qr")
    assert abs(fi["t"][j]) < 3.0
    assert fi["weights"][j] >= 1.0 / (0.1 * fi["m_data"][j]) ** 2 * (1 - 1e-12)
    assert fi["kurt_data"] > 3.0 and fi["kurt_sim"] > 3.0
    # the reported optimum is the best profile evaluation, and beats a brute-force scan of the
    # profile objective and the best grid point (same random numbers: deterministic objective)
    prob = m.smm_problem(r)
    assert prob.profile(np.log([p["kappa"], fi["c"]])) == pytest.approx(fi["objective"], rel=1e-9)
    scan = [prob.profile(np.log([k, c])) for k in (0.01, 0.05, 0.2) for c in (0.5, 0.9, 1.2)]
    assert fi["objective"] <= min(scan) + 1e-9
    assert fi["objective"] <= fi["grid_best"]["objective"] + 1e-9
    # start_from_end uses the EWMA end-of-sample variance
    assert m.v_end_ is not None and m.v_end_ > 0
    first = m.simulate(1, 20_000, rng=_rng(46), start_from_end=True)[:, 0]
    # E[(v_0 + v_1) / 2] with E[v_1 | v_0] = theta + (v_0 - theta) e^{-kappa} (trapezoid step)
    k, th = p["kappa"], p["theta"]
    assert first.var() == pytest.approx(th + (m.v_end_ - th) * (1 + math.exp(-k)) / 2, rel=0.1)


def _check_heston_real(ticker, seconds_max=None):
    r = _load_train([ticker])[ticker].to_numpy()
    t0 = time.perf_counter()
    m = Heston().fit(r)
    elapsed = time.perf_counter() - t0
    fi = m.fit_info_
    if seconds_max is not None:
        assert elapsed < seconds_max
    # days with v == 0 in the fitted SMM paths, constant returns in simulated paths
    assert fi["frac_zero_var"] < 0.001 and fi["frac_const"] < 0.001
    mu = m.params()["mu"]
    for sfe in (False, True):
        y = m.simulate(615, 200, rng=_rng(47), start_from_end=sfe)
        assert np.mean(y == mu) < 0.001
        assert np.mean(np.diff(y, axis=1) == 0.0) < 0.001
    return m


def test_heston_real_data_no_zero_variance_days():
    """TDG: the daily full-truncation Euler scheme gave 36-53% constant-return days on this and
    similar tickers; with QE (c <= sqrt(psi_c)) there are none."""
    _check_heston_real("TDG", seconds_max=60.0)


@pytest.mark.slow
@pytest.mark.parametrize("ticker", ["ORLY", "ROK", "CMS"])
def test_heston_real_data_no_zero_variance_days_more(ticker):
    _check_heston_real(ticker)


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
    assert 0.6 < m.fit_info_["c"] <= math.sqrt(1.5) + 1e-12  # true c = 1
    # rho is identified by corr(r_t, r_{t+1}^2) only, whose sampling sd over 5000 days (0.024)
    # is a third of its mean under the truth (-0.064): only the sign is reliably recovered
    assert -0.95 <= p["rho"] < -0.05
    assert m.fit_info_["m_sim"][-1] == pytest.approx(m.fit_info_["m_data"][-1], abs=0.03)
    assert abs(p["mu"] - HESTON_TRUE["mu"]) < 5e-4
    assert m.fit_info_["objective"] < 20.0
    assert m.fit_info_["frac_zero_var"] == 0.0


# --------------------------------------------------------------------------------------------
# stationary bootstrap
# --------------------------------------------------------------------------------------------
def _ar1(phi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(n + 200)
    x = np.zeros_like(z)
    for t in range(1, z.size):
        x[t] = phi * x[t - 1] + z[t]
    return x[200:]


def _arch_block_lengths(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arch_bs = pytest.importorskip("arch.bootstrap")
    ref = arch_bs.optimal_block_length(np.asarray(X).reshape(len(X), -1))
    return ref["stationary"].to_numpy(), ref["circular"].to_numpy()


def test_optimal_block_length_ar1_theory_and_arch():
    x = _ar1(0.5, 5000, _rng(25))
    b = optimal_block_length(x)
    # theory for AR(1): b_SB = (2 phi / (1 - phi^2))^{2/3} n^{1/3} = 20.7 at phi = 0.5, n = 5000
    theory = (2 * 0.5 / (1 - 0.25)) ** (2 / 3) * 5000 ** (1 / 3)
    assert theory * 0.65 < b < theory * 1.5
    # circular blocks are longer than stationary mean blocks by (3/2)^{1/3}
    assert optimal_block_length(x, kind="circular") / b == pytest.approx(1.5 ** (1 / 3), rel=1e-9)
    # identical to arch (same lag cut-off, same autocorrelation normalisation), for series with
    # short, long and no dependence, and for |x| and x^2 transforms
    rng = _rng(64)
    g = GARCHt(**GARCH_TRUE).simulate(1258, 1, rng=rng)[0]
    series = [x, _ar1(0.9, 1258, rng), _ar1(-0.4, 800, rng), rng.standard_normal(615), g,
              np.abs(g - g.mean()), (g - g.mean()) ** 2, rng.standard_t(3, 2000)]
    for s in series:
        sb, cb = _arch_block_lengths(s)
        assert optimal_block_length(s) == pytest.approx(max(sb[0], 1.0), rel=1e-12)
        assert optimal_block_length(s, kind="circular") == pytest.approx(max(cb[0], 1.0),
                                                                         rel=1e-12)


def test_optimal_block_length_matches_arch_on_sp500():
    """Exact agreement with arch on every series of the 479-ticker universe (TRAIN), for
    |r - mean| (the StationaryBootstrap default) and r; arch's values below 1 are floored."""
    from synthval import config
    if not config.DATA_CSV.exists():
        pytest.skip("data file not available")
    from synthval import data
    prices = data.load_prices()
    tickers = data.complete_universe(prices)
    train = data.split(data.log_returns(data.clean_prices(prices, tickers)), config.TRAIN)
    R = train.to_numpy()
    assert R.shape[1] == 479
    for X in (np.abs(R - R.mean(axis=0)), R):
        sb, cb = _arch_block_lengths(X)
        mine = np.array([optimal_block_length(X[:, i]) for i in range(X.shape[1])])
        np.testing.assert_allclose(mine, np.maximum(sb, 1.0), rtol=1e-12, atol=0)
        mine_c = np.array([optimal_block_length(X[:, i], kind="circular")
                           for i in range(X.shape[1])])
        np.testing.assert_allclose(mine_c, np.maximum(cb, 1.0), rtol=1e-12, atol=0)


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
    assert m.fit_info_["at_bound"] == [] and m.fit_info_["fallback"] is False
    assert m.fit_info_["block_length_rule"] == pytest.approx(b, rel=1e-12)
    # a bound is reported: the floor of 1 (white noise for which the rule gives 0.85) and a
    # user bound
    floored = StationaryBootstrap(block_series="r").fit(_rng(68).standard_normal(600))
    assert floored.block_length_ == 1.0 and floored.fit_info_["at_bound"] == ["block_length"]
    assert floored.fit_info_["block_length_rule"] < 1.0
    assert StationaryBootstrap(max_block=5.0).fit(r).fit_info_["at_bound"] == ["block_length"]
    assert StationaryBootstrap().fit(r[:10]).fit_info_["fallback"] is True
    y = m.simulate(500, 20, rng=_rng(29))
    assert y.shape == (20, 500)
    assert np.isin(y, r).all()
    assert y.var() == pytest.approx(r.var(), rel=0.2)
    fixed = StationaryBootstrap(block_length=20).fit(r)
    assert fixed.params()["block_length"] == 20.0
    with pytest.raises(NotImplementedError):
        m.loglik(r)
