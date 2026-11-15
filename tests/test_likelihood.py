"""Tests for synthval.likelihood."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from synthval.likelihood import (
    berkowitz_test,
    lr_oracle_test,
    newey_west_variance,
    pit_ks_test,
    predictive_loglik,
    vuong_test,
    window_loglik,
)
from synthval.models.base import ReturnModel
from synthval.models.univariate import GARCHt, GBM, Heston, Merton

GARCH_TRUE = dict(mu=5e-4, omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------------------------
# predictive log-likelihood
# --------------------------------------------------------------------------------------------
def test_predictive_loglik_gbm_and_garch():
    rng = _rng(0)
    r_train, r_test = rng.normal(0, 0.01, 1000), rng.normal(0, 0.01, 300)
    m = GBM().fit(r_train)
    ref = stats.norm.logpdf(r_test, m.params()["mu"], m.params()["sigma"]).mean()
    assert predictive_loglik(m, r_test, r_train) == pytest.approx(ref, rel=1e-12)
    g = GARCHt(**GARCH_TRUE)
    x = g.simulate(1300, 1, rng=rng)[0]
    pl = predictive_loglik(g, x[1000:], x[:1000])
    assert pl == pytest.approx(g.loglik_obs(x[1000:], r_history=x[:1000]).mean(), rel=1e-12)
    # conditioning on the history differs from restarting the filter
    assert pl != pytest.approx(g.loglik_obs(x[1000:]).mean(), rel=1e-9)


# --------------------------------------------------------------------------------------------
# Berkowitz and PIT-KS
# --------------------------------------------------------------------------------------------
def test_berkowitz_size_under_iid_uniforms():
    rng = _rng(1)
    pvals = np.array([berkowitz_test(rng.random(500)).pvalue for _ in range(200)])
    rate = np.mean(pvals <= 0.05)
    # binomial(200, 0.05): sd = 0.0154
    assert 0.015 <= rate <= 0.095
    assert stats.kstest(pvals, "uniform").pvalue > 0.01


def test_berkowitz_power_against_misscaled_model():
    rng = _rng(2)
    # true sd 1.3 while the model says 1 (and the reverse)
    rej_wide = np.mean([berkowitz_test(stats.norm.cdf(1.3 * rng.standard_normal(500))).pvalue
                        <= 0.05 for _ in range(50)])
    rej_narrow = np.mean([berkowitz_test(stats.norm.cdf(0.8 * rng.standard_normal(500))).pvalue
                          <= 0.05 for _ in range(50)])
    assert rej_wide >= 0.95 and rej_narrow >= 0.95


def test_berkowitz_estimates_and_exact_likelihood():
    rng = _rng(3)
    n, phi, m, s = 2000, 0.3, 0.2, 1.1
    z = np.empty(n)
    z[0] = m + s * rng.standard_normal() / math.sqrt(1 - phi ** 2)
    for t in range(1, n):
        z[t] = m + phi * (z[t - 1] - m) + s * rng.standard_normal()
    res = berkowitz_test(stats.norm.cdf(z))
    assert res.pvalue < 1e-6 and res.extra["p_ind"] < 1e-6
    assert res.extra["rho"] == pytest.approx(phi, abs=0.06)
    assert res.extra["mean"] == pytest.approx(m, abs=0.1)
    assert res.extra["sigma"] == pytest.approx(s, rel=0.05)
    # exact Gaussian AR(1) maximum likelihood agrees with statsmodels' Kalman-filter ARIMA(1,0,0)
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        llf = ARIMA(z, order=(1, 0, 0), trend="c").fit().llf
    ll0 = -0.5 * (n * math.log(2 * math.pi) + np.sum(z ** 2))
    assert res.statistic / 2 + ll0 == pytest.approx(llf, abs=0.05)
    assert res.statistic >= 0


def test_pit_ks_matches_scipy():
    u = _rng(4).random(400)
    res = pit_ks_test(u)
    ref = stats.kstest(u, "uniform")
    assert res.statistic == pytest.approx(ref.statistic) and res.pvalue == pytest.approx(ref.pvalue)
    assert pit_ks_test(u ** 2).pvalue < 1e-6


def test_pit_tests_under_true_garch_and_wrong_gbm():
    g = GARCHt(**GARCH_TRUE)
    x = g.simulate(3000, 1, rng=_rng(5))[0]
    u_true = g.pit(x[1500:], r_history=x[:1500])
    assert pit_ks_test(u_true).pvalue > 0.01
    assert berkowitz_test(u_true).pvalue > 0.01
    # a GBM with half the volatility is rejected
    bad = GBM(mu=0.0, sigma=0.5 * x.std())
    assert berkowitz_test(bad.pit(x[1500:])).pvalue < 1e-6


# --------------------------------------------------------------------------------------------
# Vuong
# --------------------------------------------------------------------------------------------
def test_vuong_direction_and_formula():
    rng = _rng(6)
    x = rng.standard_t(5, 3000) * 0.01  # 5 df: d_t has a finite variance
    ll_t = stats.t.logpdf(x, df=5, scale=0.01)
    g = GBM().fit(x)
    ll_n = g.loglik_obs(x)
    res = vuong_test(ll_t, ll_n)
    d = ll_t - ll_n
    assert res.statistic == pytest.approx(math.sqrt(d.size) * d.mean() / d.std(), rel=1e-12)
    assert res.statistic > 2 and res.extra["p_a_better"] < 0.025 and res.pvalue < 0.05
    rev = vuong_test(ll_n, ll_t)
    assert rev.statistic == pytest.approx(-res.statistic) and rev.pvalue == pytest.approx(res.pvalue)
    assert rev.extra["p_b_better"] == pytest.approx(res.extra["p_a_better"])
    # BIC correction penalises the larger model
    k = vuong_test(ll_t, ll_n, k_a=3, k_b=1)
    expected = (d.sum() - (3 - 1) * math.log(d.size) / 2) / (math.sqrt(d.size) * d.std())
    assert k.statistic == pytest.approx(expected, rel=1e-12)
    aic = vuong_test(ll_t, ll_n, k_a=3, k_b=1, correction="aic")
    assert aic.statistic == pytest.approx((d.sum() - 2) / (math.sqrt(d.size) * d.std()), rel=1e-12)
    # identical models
    same = vuong_test(ll_n, ll_n)
    assert same.statistic == 0.0 and same.pvalue == 1.0


def test_vuong_hac():
    rng = _rng(7)
    d_iid = rng.standard_normal(2000) * 0.1 + 0.002
    a = vuong_test(d_iid, np.zeros_like(d_iid), hac=True)
    b = vuong_test(d_iid, np.zeros_like(d_iid))
    assert a.statistic == pytest.approx(b.statistic, rel=0.15)
    # positively autocorrelated differences: HAC sd is larger, statistic smaller
    e = rng.standard_normal(2000)
    d_ar = np.empty_like(e)
    d_ar[0] = e[0]
    for t in range(1, e.size):
        d_ar[t] = 0.6 * d_ar[t - 1] + e[t]
    d_ar = 0.1 * d_ar + 0.01
    h = vuong_test(d_ar, np.zeros_like(d_ar), hac=True, lags=20)
    i = vuong_test(d_ar, np.zeros_like(d_ar))
    assert abs(h.statistic) < abs(i.statistic) and h.extra["lags"] == 20
    # Newey-West formula
    lrv, L = newey_west_variance(d_ar, 3)
    c = d_ar - d_ar.mean()
    ref = c @ c / c.size + sum(2 * (1 - j / 4) * (c[j:] @ c[:-j]) / c.size for j in (1, 2, 3))
    assert L == 3 and lrv == pytest.approx(ref, rel=1e-12)


# --------------------------------------------------------------------------------------------
# window log-likelihood
# --------------------------------------------------------------------------------------------
class _IIDLaplace(ReturnModel):
    """Minimal model without ``loglik_windows`` to exercise the generic fallback."""

    name = "laplace"
    has_likelihood = True

    def loglik_obs(self, r, r_history=None):
        return stats.laplace.logpdf(np.asarray(r), scale=0.01)

    def params(self):
        return {}


def test_window_loglik():
    W = _rng(8).normal(0, 0.01, (40, 20))
    g = GBM(mu=0.0, sigma=0.01)
    np.testing.assert_allclose(window_loglik(g, W),
                               stats.norm.logpdf(W, 0, 0.01).sum(axis=1), rtol=1e-12)
    mt = Merton(mu=0, sigma=0.008, lam=0.05, muJ=0, sJ=0.03)
    np.testing.assert_allclose(window_loglik(mt, W),
                               np.array([mt.loglik(w) for w in W]), rtol=1e-12)
    gt = GARCHt(**GARCH_TRUE)
    np.testing.assert_allclose(window_loglik(gt, W), np.array(
        [gt.loglik(w, sigma2_0=gt.unconditional_variance()) for w in W]), rtol=1e-12)
    np.testing.assert_allclose(window_loglik(_IIDLaplace(), W),
                               stats.laplace.logpdf(W, scale=0.01).sum(axis=1), rtol=1e-12)
    assert window_loglik(g, W[0]).shape == (1,)
    with pytest.raises(NotImplementedError):
        window_loglik(Heston(mu=0, kappa=0.03, theta=1e-4, xi=1e-3, rho=-0.5), W)
    # keyword arguments reach loglik_windows (GARCH: Monte Carlo over initial variances)
    s2k = gt.stationary_variance_draws(50, rng=_rng(80))
    np.testing.assert_allclose(window_loglik(gt, W, sigma2_init=s2k),
                               gt.loglik_windows(W, sigma2_init=s2k), rtol=1e-12)
    with pytest.raises(TypeError):
        window_loglik(_IIDLaplace(), W, sigma2_init=s2k)
    # n_init: K stationary draws from a fixed seed (cached), log-mean-exp over them
    ref = gt.loglik_windows(W, sigma2_init=gt.stationary_variance_draws(
        60, rng=np.random.default_rng(7)))
    np.testing.assert_allclose(window_loglik(gt, W, n_init=60, seed=7), ref, rtol=1e-12)
    np.testing.assert_allclose(window_loglik(gt, W, n_init=60, seed=7), ref, rtol=1e-12)  # cache
    assert len(gt._s2init_cache) == 1
    assert not np.allclose(window_loglik(gt, W, n_init=60, seed=8), ref)
    np.testing.assert_allclose(window_loglik(gt, W, n_init=0), window_loglik(gt, W), rtol=1e-12)


# --------------------------------------------------------------------------------------------
# LR oracle
# --------------------------------------------------------------------------------------------
def test_lr_oracle_rejects_clear_alternative():
    rng = _rng(9)
    null, alt = GBM(mu=0.0, sigma=0.01), GBM(mu=0.0, sigma=0.012)
    res = lr_oracle_test(alt.simulate(20, 200, rng=rng), None, null, alt, rng=rng, n_mc=199)
    assert res.pvalue <= 0.01 and res.statistic > res.extra["crit"]
    assert res.name == "lr_oracle" and res.n_x == 200
    # jumps: Merton alternative against a variance-matched GBM null
    mt = Merton(mu=0.0, sigma=0.008, lam=0.05, muJ=0.0, sJ=0.03)
    gbm = GBM(mu=0.0, sigma=math.sqrt(0.008 ** 2 + 0.05 * 0.03 ** 2))
    res = lr_oracle_test(mt.simulate(20, 200, rng=rng), None, gbm, mt, rng=rng, n_mc=99)
    assert res.pvalue <= 0.02
    # volatility clustering, windows scored from the unconditional variance
    g_alt = GARCHt(mu=0.0, omega=1e-4 * 0.02, alpha=0.1, beta=0.88, nu=30.0)
    g_null = GARCHt(mu=0.0, omega=1e-4, alpha=0.0, beta=0.0, nu=30.0)
    res = lr_oracle_test(g_alt.simulate(20, 300, rng=rng), None, g_null, g_alt, rng=rng, n_mc=99)
    assert res.pvalue <= 0.02


def test_lr_oracle_null_sources():
    rng = _rng(10)
    null, alt = GBM(mu=0.0, sigma=0.01), GBM(mu=0.0, sigma=0.012)
    X = null.simulate(20, 100, rng=rng)
    pool = null.simulate(20, 5000, rng=rng)
    r1 = lr_oracle_test(X, pool, null, alt, rng=_rng(11), n_mc=99)
    r2 = lr_oracle_test(X, lambda g: null.simulate(20, 100, rng=g), null, alt, rng=_rng(11),
                        n_mc=99)
    r3 = lr_oracle_test(X, null, null, alt, rng=_rng(11), n_mc=99)
    r4 = lr_oracle_test(X, None, null, alt, rng=_rng(11), n_mc=99)
    assert r1.statistic == r2.statistic == r3.statistic == r4.statistic
    assert r3.pvalue == r4.pvalue == r2.pvalue  # same draws through the same sampler
    assert r1.n_y == 5000
    with pytest.raises(ValueError):
        lr_oracle_test(X, None, null, None, rng=rng)
    with pytest.raises(ValueError):
        lr_oracle_test(X, null.simulate(10, 50, rng=rng), null, alt, rng=rng, n_mc=5)


def test_lr_oracle_pvalues_uniform_under_null():
    rng = _rng(12)
    null, alt = GBM(mu=0.0, sigma=0.01), GBM(mu=0.0, sigma=0.011)
    pvals = np.array([lr_oracle_test(null.simulate(20, 100, rng=rng), None, null, alt, rng=rng,
                                     n_mc=99).pvalue for _ in range(100)])
    assert stats.kstest(pvals, "uniform").pvalue > 0.01
    assert 0.0 <= np.mean(pvals <= 0.05) <= 0.12


def _garch_pair(pers_alt: float) -> tuple[GARCHt, GARCHt]:
    """Null GARCH(0.08, 0.90)-t(6) with unit variance and an alternative of lower persistence
    at the same unconditional variance and tails."""
    null = GARCHt(mu=0.0, omega=0.02, alpha=0.08, beta=0.90, nu=6.0)
    a = 0.08 if pers_alt > 0.08 else 0.0
    alt = GARCHt(mu=0.0, omega=1.0 - pers_alt, alpha=a, beta=pers_alt - a, nu=6.0)
    return null, alt


def test_lr_oracle_garch_exact_window_density_is_more_powerful():
    """The fixed-start window density only approximates the GARCH likelihood ratio; the Monte
    Carlo integral over the stationary initial variance separates the hypotheses more."""
    null, alt = _garch_pair(0.90)
    rng = _rng(13)
    WP, WQ = null.simulate(20, 3000, rng=rng), alt.simulate(20, 3000, rng=rng)
    s2_null = null.stationary_variance_draws(400, rng=rng)
    s2_alt = alt.stationary_variance_draws(400, rng=rng)

    def gap(kw_null, kw_alt):
        llr = lambda W: (window_loglik(null, W, **kw_null) - window_loglik(alt, W, **kw_alt))
        return llr(WP).mean() - llr(WQ).mean()

    fixed = gap({}, {})
    exact = gap({"sigma2_init": s2_null}, {"sigma2_init": s2_alt})
    assert fixed > 0 and exact > 1.5 * fixed
    # the oracle with n_init: valid p-value, rejects the alternative, records n_init
    res = lr_oracle_test(alt.simulate(20, 200, rng=rng), None, null, alt, rng=rng, n_mc=49,
                         n_init=200)
    assert res.pvalue <= 0.04 and res.extra["n_init"] == 200
    # i.i.d. models have no initial state: n_init is ignored
    g0, g1 = GBM(mu=0.0, sigma=0.01), GBM(mu=0.0, sigma=0.012)
    X = g1.simulate(20, 50, rng=rng)
    a = lr_oracle_test(X, None, g0, g1, rng=_rng(14), n_mc=19)
    b = lr_oracle_test(X, None, g0, g1, rng=_rng(14), n_mc=19, n_init=100)
    assert a.statistic == b.statistic and a.pvalue == b.pvalue


@pytest.mark.slow
def test_lr_oracle_size_windows_cut_from_paths_with_matching_null_sampler():
    """X cut from long GARCH paths (dependent windows): valid when the null sampler reproduces
    that protocol (the default independent-window sampler would over-reject)."""
    null, alt = _garch_pair(0.95)
    n, L, per_path = 200, 20, 25  # 8 paths of 500 days, 25 consecutive windows each
    rng = _rng(15)
    from_paths = lambda g: null.simulate(per_path * L, n // per_path, rng=g).reshape(n, L)
    pvals = np.array([lr_oracle_test(from_paths(rng), from_paths, null, alt, rng=rng,
                                     n_mc=99).pvalue for _ in range(150)])
    # binomial(150, 0.05): sd 0.018
    assert np.mean(pvals <= 0.05) <= 0.10
    assert stats.kstest(pvals, "uniform").pvalue > 0.01
