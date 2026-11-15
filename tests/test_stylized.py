"""Tests for synthval.stylized."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf

from synthval import stylized as sf


def simulate_garch(n: int, *, omega: float, alpha: float, beta: float, gamma: float = 0.0,
                   rng: np.random.Generator, burn: int = 1000) -> np.ndarray:
    """Gaussian GJR-GARCH(1,1): s2_t = w + (a + g 1[r_{t-1}<0]) r_{t-1}^2 + b s2_{t-1}."""
    z = rng.standard_normal(n + burn)
    r = np.empty(n + burn)
    s2 = omega / (1.0 - alpha - beta - gamma / 2.0)
    prev = 0.0
    for t in range(n + burn):
        s2 = omega + (alpha + gamma * (prev < 0)) * prev**2 + beta * s2
        prev = np.sqrt(s2) * z[t]
        r[t] = prev
    return r[burn:]


@pytest.fixture(scope="module")
def gauss() -> np.ndarray:
    return np.random.default_rng(1).normal(0.0005, 0.01, size=40_000)


@pytest.fixture(scope="module")
def garch() -> np.ndarray:
    return simulate_garch(40_000, omega=1e-6, alpha=0.08, beta=0.90,
                          rng=np.random.default_rng(2))


# ---------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------
def test_acf_matches_statsmodels():
    x = np.random.default_rng(0).standard_t(5, size=1500)
    ref = sm_acf(x, nlags=50, fft=False)
    np.testing.assert_allclose(sf.acf(x, 50), ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sf.acf(np.abs(x), 20), sm_acf(np.abs(x), nlags=20), atol=1e-12)


def test_acf_edge_cases():
    out = sf.acf(np.arange(5.0), 8)
    assert out.shape == (9,) and out[0] == 1.0
    assert np.isnan(out[5:]).all() and np.isfinite(out[:5]).all()
    assert np.isnan(sf.acf(np.ones(10), 3)[1:]).all()
    with pytest.raises(ValueError):
        sf.acf(np.array([1.0, np.nan, 2.0]), 1)


def test_hill_recovers_pareto_alpha():
    rng = np.random.default_rng(3)
    alpha = 3.0
    n = 20_000
    mag = (1.0 - rng.random(n)) ** (-1.0 / alpha)       # Pareto(alpha), x_m = 1
    r = mag * rng.choice([-1.0, 1.0], size=n)           # symmetric returns
    a_hat = sf.hill(np.abs(r))
    assert abs(a_hat - alpha) / alpha < 0.15
    s = sf.summary(r)
    assert abs(s["hill"] - alpha) / alpha < 0.15
    assert abs(s["hill_left"] - alpha) / alpha < 0.15
    assert abs(s["hill_right"] - alpha) / alpha < 0.15


def test_hill_formula_small_example():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 0.5, 0.25, 0.125, 0.1, 0.05])
    # fraction 0.3 -> k = 3 largest {16, 8, 4}, threshold 2
    expected = 1.0 / np.mean(np.log([16 / 2, 8 / 2, 4 / 2]))
    assert sf.hill(x, fraction=0.3) == pytest.approx(expected)
    assert np.isnan(sf.hill(x, fraction=0.1))  # k = 1 < 2


def test_max_drawdown_example():
    r = np.array([0.1, -0.2, -0.1, 0.05, 0.3])
    # log-price 0, .1, -.1, -.2, -.15, .15 -> peak .1, trough -.2
    assert sf.max_drawdown(r) == pytest.approx(0.3)
    assert sf.max_drawdown(np.array([0.01, 0.02])) == 0.0
    assert sf.max_drawdown(np.array([-0.05])) == pytest.approx(0.05)  # from p_0 = 0


def test_aggregated_kurtosis_uses_nonoverlapping_sums():
    x = np.random.default_rng(4).standard_t(4, size=103)
    k = sf.aggregated_kurtosis(x, (1, 5))
    assert k[1] == pytest.approx(stats.kurtosis(x, bias=False))
    agg = x[:100].reshape(20, 5).sum(axis=1)
    assert k[5] == pytest.approx(stats.kurtosis(agg, bias=False))
    assert np.isnan(sf.aggregated_kurtosis(x[:15], (5,))[5])  # 3 blocks < 4


def test_leverage_sign():
    rng = np.random.default_rng(5)
    gjr = simulate_garch(30_000, omega=1e-6, alpha=0.0, beta=0.90, gamma=0.15, rng=rng)
    lev = sf.leverage(gjr)
    assert lev.shape == (10,)
    assert np.mean(lev) < -0.01  # theory: negative at every lag (falls raise volatility)
    sym = sf.leverage(simulate_garch(30_000, omega=1e-6, alpha=0.075, beta=0.90,
                                     rng=np.random.default_rng(5)))
    assert np.mean(lev) < np.mean(sym)
    x = np.arange(6.0)
    assert sf.leverage(x, 4)[3] != sf.leverage(x, 4)[3]  # 2 pairs < 3 -> NaN


# ---------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------
def test_summary_keys(gauss):
    s = sf.summary(gauss)
    expected = (["n", "mean", "std", "skew", "kurt"]
                + [f"acf_r_{k}" for k in range(1, 6)]
                + [f"acf_abs_{k}" for k in (1, 5, 10, 20, 50)]
                + [f"acf_sq_{k}" for k in (1, 5, 10, 20, 50)]
                + [f"lev_{k}" for k in range(1, 11)] + ["lev_mean"]
                + ["hill", "hill_left", "hill_right"]
                + ["kurt_agg_1", "kurt_agg_5", "kurt_agg_20", "max_drawdown"])
    assert list(s) == expected
    assert all(isinstance(v, float) and np.isfinite(v) for v in s.values())
    assert s["kurt_agg_1"] == pytest.approx(s["kurt"])
    assert s["lev_mean"] == pytest.approx(np.mean([s[f"lev_{k}"] for k in range(1, 11)]))


def test_iid_gaussian_has_no_stylized_facts(gauss):
    s = sf.summary(gauss)
    n = gauss.size
    assert s["mean"] == pytest.approx(0.0005, abs=4 * 0.01 / np.sqrt(n))
    assert s["std"] == pytest.approx(0.01, rel=0.02)
    assert abs(s["skew"]) < 5 * np.sqrt(6 / n)
    assert abs(s["kurt"]) < 5 * np.sqrt(24 / n)
    band = 5 / np.sqrt(n)  # 5 standard errors of a white-noise ACF
    for key in [k for k in s if k.startswith(("acf_r_", "acf_abs_", "acf_sq_", "lev_"))]:
        assert abs(s[key]) < band, key
    for h in (5, 20):
        assert abs(s[f"kurt_agg_{h}"]) < 5 * np.sqrt(24 / (n // h))


def test_garch_has_volatility_clustering(garch):
    s = sf.summary(garch)
    assert s["acf_sq_1"] > 0.1
    assert s["acf_abs_1"] > 0.1
    assert s["acf_abs_1"] > s["acf_abs_20"] > s["acf_abs_50"] > 0  # slow decay, positive
    assert abs(s["acf_r_1"]) < 5 / np.sqrt(garch.size)         # returns themselves uncorrelated
    assert s["kurt"] > 0.5                                       # GARCH implies fat tails
    # NB: for persistent GARCH the kurtosis of h-day sums can *rise* with h at first, so
    # aggregational Gaussianity is checked on iid Laplace data instead (below).


def test_aggregational_gaussianity_iid_laplace():
    x = np.random.default_rng(9).laplace(size=40_000)  # excess kurtosis 3, of h-sums 3/h
    s = sf.summary(x)
    assert s["kurt_agg_1"] > 2.3
    assert s["kurt_agg_20"] < 0.6
    assert s["kurt_agg_1"] > s["kurt_agg_5"] > s["kurt_agg_20"]


def test_summary_short_series_gives_nan_not_error():
    s = sf.summary(np.random.default_rng(6).normal(size=30))
    assert np.isnan(s["acf_abs_50"]) and np.isnan(s["kurt_agg_20"])
    assert np.isfinite(s["acf_abs_20"]) and np.isfinite(s["kurt"])
    with pytest.raises(ValueError):
        sf.summary(np.zeros((3, 3)))


# ---------------------------------------------------------------------------------------------
# panel_summary
# ---------------------------------------------------------------------------------------------
def test_panel_summary_structure_and_aggregates():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.standard_t(5, size=(500, 6)) * 0.01,
                      columns=[f"T{i}" for i in range(6)])
    df.iloc[:40, 2] = np.nan   # late listing: leading NaN dropped, not an error
    out = sf.panel_summary(df)
    assert list(out.index) == list(df.columns) + ["median", "q25", "q75", "iqr"]
    assert out.attrs["tickers"] == list(df.columns)
    assert out.loc["T2", "n"] == 460
    per = out.loc[list(df.columns)]
    for col in ("std", "kurt", "acf_sq_1", "hill"):
        assert out.loc["median", col] == pytest.approx(per[col].median())
        assert out.loc["iqr", col] == pytest.approx(per[col].quantile(0.75) - per[col].quantile(0.25))
    assert out.loc["T0", "kurt"] == pytest.approx(sf.summary(df["T0"].to_numpy())["kurt"])
    agg = sf.panel_summary(df, per_ticker=False)
    assert list(agg.index) == ["median", "q25", "q75", "iqr"]
    pd.testing.assert_frame_equal(agg, out.loc[["median", "q25", "q75", "iqr"]])
    df.iloc[100, 3] = np.nan
    with pytest.raises(ValueError, match="internal"):
        sf.panel_summary(df)


def test_panel_summary_accepts_arrays():
    x = np.random.default_rng(8).normal(size=(200, 3))
    out = sf.panel_summary(x)
    assert list(out.index[:3]) == [0, 1, 2]
