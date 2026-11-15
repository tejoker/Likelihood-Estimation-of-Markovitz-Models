"""Tests for synthval.portfolio (DESIGN section 3.10)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize

from synthval.portfolio import (
    average_correlation,
    equal_weights,
    evaluate,
    gmv_weights,
    ledoit_wolf,
    max_drawdown,
    mean_variance_weights,
    portfolio_returns,
    sample_moments,
    tangency_weights,
    utility_gap,
)


# ----------------------------------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------------------------------
def _cov_with_shorts() -> tuple[np.ndarray, np.ndarray]:
    """Daily covariance and means for which unconstrained portfolios hold short positions."""
    vols = np.array([0.010, 0.012, 0.020, 0.030, 0.015, 0.025, 0.018, 0.022])
    C = np.full((8, 8), 0.3)
    C[:4, :4] = 0.8
    C[4:, 4:] = 0.5
    np.fill_diagonal(C, 1.0)
    Sigma = C * np.outer(vols, vols)
    mu = np.array([2e-4, 5e-4, -1e-4, 9e-4, 3e-4, 6e-4, -2e-4, 4e-4])
    return mu, Sigma


def _random_cov(rng: np.random.Generator, N: int) -> np.ndarray:
    A = rng.standard_normal((N, N))
    return 1e-4 * (A @ A.T / N + 0.2 * np.eye(N))


def assert_kkt(Q, c, a, b, x, tol=1e-8):
    """KKT conditions of min 0.5 x'Qx - c'x s.t. a'x = b, x >= 0 (scaled by mean diag Q)."""
    s = np.mean(np.diag(Q))
    Q, c = Q / s, c / s
    assert x.min() >= 0.0
    assert a @ x == pytest.approx(b, abs=1e-12)
    g = Q @ x - c
    on = x > 1e-12
    assert on.any()
    lam = np.mean(g[on] / a[on])
    np.testing.assert_allclose(g[on], lam * a[on], atol=tol)  # stationarity on the support
    assert np.all(g[~on] - lam * a[~on] >= -tol)  # bound multipliers are non-negative


def _sharpe(w, mu, Sigma):
    return float(w @ mu / np.sqrt(w @ Sigma @ w))


def _max_sharpe_direct(mu, Sigma):
    """Long-only max-Sharpe by SLSQP on the Sharpe ratio itself (independent check)."""
    N = mu.size

    def f(w):
        s = np.sqrt(w @ Sigma @ w)
        return -(w @ mu) / s, -(mu / s - (w @ mu) * (Sigma @ w) / s**3)

    best = None
    for start in [np.full(N, 1 / N), np.eye(N)[np.argmax(mu)] * 0.9 + 0.1 / N]:
        res = optimize.minimize(f, start, jac=True, method="SLSQP", bounds=[(0, 1)] * N,
                                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                                options={"ftol": 1e-15, "maxiter": 1000})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


# ----------------------------------------------------------------------------------------------
# Moments
# ----------------------------------------------------------------------------------------------
def test_sample_moments():
    R = np.random.default_rng(0).standard_normal((300, 4)) * 0.01
    mu, S = sample_moments(R)
    np.testing.assert_allclose(mu, R.mean(0), rtol=1e-14)
    np.testing.assert_allclose(S, np.cov(R, rowvar=False, ddof=1), rtol=1e-14)
    with pytest.raises(ValueError):
        sample_moments(np.ones(5))


@pytest.mark.parametrize("shape", [(500, 10), (60, 40), (20, 50), (100, 1), (1250, 30)])
def test_ledoit_wolf_matches_sklearn(shape):
    sk = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(sum(shape))
    T, N = shape
    R = 0.01 * rng.standard_normal((T, N)) @ (np.eye(N) + 0.3 * rng.standard_normal((N, N)))
    R += rng.normal(0, 1e-3, N)  # non-zero means: both estimators demean
    S, delta = ledoit_wolf(R, return_shrinkage=True)
    ref = sk.LedoitWolf().fit(R)
    scale = np.max(np.abs(ref.covariance_))
    np.testing.assert_allclose(S, ref.covariance_, rtol=1e-8, atol=1e-8 * scale)
    assert delta == pytest.approx(ref.shrinkage_, abs=1e-8)
    assert 0.0 <= delta <= 1.0
    assert np.all(np.linalg.eigvalsh(S) > 0)  # well conditioned even when N > T
    np.testing.assert_array_equal(ledoit_wolf(R), S)


def test_average_correlation():
    C = np.full((5, 5), 0.3)
    np.fill_diagonal(C, 1.0)
    vols = np.array([0.01, 0.02, 0.015, 0.03, 0.012])
    assert average_correlation(C * np.outer(vols, vols), is_cov=True) == pytest.approx(0.3)
    R = np.random.default_rng(9).standard_normal((400, 6)) @ (np.eye(6) + 0.5)
    Cr = np.corrcoef(R, rowvar=False)
    assert average_correlation(R) == pytest.approx(Cr[np.triu_indices(6, 1)].mean(), rel=1e-12)
    assert np.isnan(average_correlation(np.array([[2e-4]]), is_cov=True))
    with pytest.raises(ValueError):
        average_correlation(np.diag([1e-4, 0.0]), is_cov=True)


# ----------------------------------------------------------------------------------------------
# Global minimum variance
# ----------------------------------------------------------------------------------------------
def test_gmv_unconstrained_analytic_equals_slsqp():
    mu, Sigma = _cov_with_shorts()
    w = gmv_weights(Sigma)
    ones = np.ones(len(w))
    ref = np.linalg.solve(Sigma, ones)
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-12)
    assert w.min() < 0  # this case really has short positions
    w_num = gmv_weights(Sigma, method="slsqp")
    np.testing.assert_allclose(w_num, w, atol=1e-7)
    assert w_num @ Sigma @ w_num == pytest.approx(w @ Sigma @ w, rel=1e-9)


def test_gmv_long_only_kkt_and_feasibility():
    mu, Sigma = _cov_with_shorts()
    w = gmv_weights(Sigma, long_only=True)
    assert np.all(w >= 0) and w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.sum(w == 0) >= 1  # some constraints bind
    assert_kkt(Sigma, np.zeros(8), np.ones(8), 1.0, w)
    rng = np.random.default_rng(1)  # no random long-only portfolio has lower variance
    W = rng.dirichlet(np.ones(8) * 0.5, size=5000)
    assert np.min(np.einsum("ij,jk,ik->i", W, Sigma, W)) >= w @ Sigma @ w


def test_gmv_long_only_equals_unconstrained_when_interior():
    Sigma = np.diag([1e-4, 2e-4, 4e-4])
    w = gmv_weights(Sigma, long_only=True)
    np.testing.assert_allclose(w, gmv_weights(Sigma), atol=1e-12)
    np.testing.assert_allclose(w, np.array([4, 2, 1]) / 7, atol=1e-12)


def test_gmv_scale_invariance_and_edge_cases():
    rng = np.random.default_rng(2)
    Sigma = _random_cov(rng, 12)
    w = gmv_weights(Sigma, long_only=True)
    np.testing.assert_allclose(gmv_weights(Sigma * 1e-6, long_only=True), w, atol=1e-10)
    np.testing.assert_allclose(gmv_weights(Sigma * 1e4, long_only=True), w, atol=1e-10)
    np.testing.assert_allclose(gmv_weights(np.array([[4e-4]])), [1.0])
    with pytest.raises(ValueError):
        gmv_weights(np.array([[1.0, 0.5], [0.4, 1.0]]))  # not symmetric
    with pytest.raises(ValueError):
        gmv_weights(Sigma, long_only=True, method="analytic")


@pytest.mark.parametrize("seed", range(5))
def test_gmv_long_only_kkt_on_estimated_covariances(seed, caplog):
    """Realistic case: N = 40 assets, Ledoit-Wolf on T = 60 days; many weights are zero."""
    rng = np.random.default_rng(100 + seed)
    N, T = 40, 60
    B = rng.normal(1.0, 0.3, N)
    R = 0.01 * (np.outer(rng.standard_normal(T), B) + rng.standard_normal((T, N)) * rng.uniform(0.5, 2, N))
    Sigma = ledoit_wolf(R)
    with caplog.at_level("WARNING", logger="synthval.portfolio"):
        w = gmv_weights(Sigma, long_only=True)
        mu = R.mean(axis=0)
        w_mv = mean_variance_weights(mu, Sigma, 5.0)
        w_tg = tangency_weights(mu, Sigma)
    assert not caplog.records  # the KKT polish succeeded: no fallback to the raw SLSQP point
    assert_kkt(Sigma, np.zeros(N), np.ones(N), 1.0, w)
    assert_kkt(5.0 * Sigma, mu, np.ones(N), 1.0, w_mv)
    y = w_tg / (mu @ w_tg)
    k = np.max(np.abs(mu))
    assert_kkt(Sigma, np.zeros(N), mu / k, 1.0 / k, y, tol=1e-7)


# ----------------------------------------------------------------------------------------------
# Singular covariance matrices (unconstrained problems)
# ----------------------------------------------------------------------------------------------
def _duplicate_asset_cov():
    """Asset 3 is an exact copy of asset 0: Sigma is singular (rank 3)."""
    S = np.zeros((4, 4))
    S[:3, :3] = np.diag([1e-4, 2e-4, 3e-4])
    S[3, :3] = S[0, :3]
    S[:3, 3] = S[:3, 0]
    S[3, 3] = S[0, 0]
    return S


def test_singular_cov_duplicate_asset(caplog):
    S = _duplicate_asset_cov()
    with caplog.at_level("WARNING", logger="synthval.portfolio"):
        w = gmv_weights(S)
    assert any("singular" in r.getMessage() for r in caplog.records)
    # minimiser set: w0 + w3 = 6/11 with w1 = 3/11, w2 = 2/11; minimum norm splits w0 = w3
    np.testing.assert_allclose(w, np.array([3, 3, 2, 3]) / 11, atol=1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    mu = np.array([1e-3, 5e-4, 2e-4, 1e-3])  # same mean for the duplicate: bounded problem
    w_mv = mean_variance_weights(mu, S, 5.0, long_only=False)
    assert w_mv[0] == pytest.approx(w_mv[3], abs=1e-12) and w_mv.sum() == pytest.approx(1.0)
    # reference: merge the duplicate into one asset (non-singular 3x3 problem), split evenly
    w3 = mean_variance_weights(mu[:3], S[:3, :3], 5.0, long_only=False)
    np.testing.assert_allclose(w_mv, [w3[0] / 2, w3[1], w3[2], w3[0] / 2], atol=1e-10)
    # different means for identical assets: riskless arbitrage, objective unbounded -> error
    with pytest.raises(ValueError, match="no \\(numerically reliable\\) solution"):
        mean_variance_weights(np.array([1e-3, 5e-4, 2e-4, 2e-3]), S, 5.0, long_only=False)
    # long-only problems stay bounded
    w_lo = mean_variance_weights(np.array([1e-3, 5e-4, 2e-4, 2e-3]), S, 5.0)
    assert np.all(w_lo >= 0) and w_lo.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_rank_deficient_sample_covariance_keeps_budget(seed):
    """T = N = 30: sample covariance of rank 29. The budget holds exactly and the solution
    satisfies stationarity on the budget hyperplane (previous least-squares KKT fallback
    could return weights summing to about -0.86)."""
    rng = np.random.default_rng(200 + seed)
    N = T = 30
    beta = rng.normal(1.0, 0.3, N)
    R = 0.01 * (np.outer(rng.standard_normal(T), beta) + rng.standard_normal((T, N)))
    mu, S = sample_moments(R)
    assert np.linalg.matrix_rank(S) == N - 1
    ones = np.ones(N)
    Zb = np.linalg.svd(np.eye(N) - np.outer(ones, ones) / N)[0][:, : N - 1]  # other basis
    for w, c in ((gmv_weights(S), np.zeros(N)),
                 (mean_variance_weights(mu, S, 5.0, long_only=False), mu / 5.0)):
        assert w.sum() == pytest.approx(1.0, abs=1e-10)
        g = Zb.T @ (S @ w - c)
        assert np.linalg.norm(g) <= 1e-9 * (np.linalg.norm(S, 2) * np.linalg.norm(w)
                                            + np.linalg.norm(c))
        # unique minimiser here (Sigma is positive definite on the hyperplane): same as the
        # null-space solution computed independently with the SVD basis
        x_p = ones / N
        y = np.linalg.solve(Zb.T @ S @ Zb, Zb.T @ (c - S @ x_p))
        np.testing.assert_allclose(w, x_p + Zb @ y, rtol=1e-6, atol=1e-8 * np.abs(w).max())


# ----------------------------------------------------------------------------------------------
# Mean-variance
# ----------------------------------------------------------------------------------------------
def test_mean_variance_unconstrained_and_long_only():
    mu, Sigma = _cov_with_shorts()
    gamma = 3.0
    w = mean_variance_weights(mu, Sigma, gamma, long_only=False)
    Si1 = np.linalg.solve(Sigma, np.ones(8))
    Sim = np.linalg.solve(Sigma, mu)
    w_gmv = Si1 / Si1.sum()
    ref = w_gmv + (Sim - Sim.sum() * w_gmv) / gamma
    np.testing.assert_allclose(w, ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mean_variance_weights(mu, Sigma, gamma, long_only=False,
                                                     method="slsqp"), w, atol=1e-6)
    wl = mean_variance_weights(mu, Sigma, gamma)  # long_only=True is the default
    assert np.all(wl >= 0) and wl.sum() == pytest.approx(1.0, abs=1e-12)
    assert_kkt(gamma * Sigma, mu, np.ones(8), 1.0, wl)
    obj = lambda v: v @ mu - 0.5 * gamma * v @ Sigma @ v  # noqa: E731
    W = np.random.default_rng(3).dirichlet(np.ones(8) * 0.3, size=5000)
    assert obj(wl) >= np.max(W @ mu - 0.5 * gamma * np.einsum("ij,jk,ik->i", W, Sigma, W))


def test_mean_variance_limits():
    mu, Sigma = _cov_with_shorts()
    np.testing.assert_allclose(mean_variance_weights(mu, Sigma, 1e9),
                               gmv_weights(Sigma, long_only=True), atol=1e-6)
    w_lowrisk = mean_variance_weights(mu, Sigma, 1e-6)  # tiny gamma: all in the best asset
    np.testing.assert_allclose(w_lowrisk, np.eye(8)[np.argmax(mu)], atol=1e-9)
    with pytest.raises(ValueError):
        mean_variance_weights(mu, Sigma, 0.0)


# ----------------------------------------------------------------------------------------------
# Tangency
# ----------------------------------------------------------------------------------------------
def test_tangency_unconstrained():
    mu, Sigma = _cov_with_shorts()
    w = tangency_weights(mu, Sigma, long_only=False)
    ref = np.linalg.solve(Sigma, mu)
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-10)
    np.testing.assert_allclose(tangency_weights(mu, Sigma, long_only=False, method="slsqp"), w,
                               atol=1e-6)
    W = np.random.default_rng(4).normal(size=(5000, 8))
    W /= W.sum(axis=1, keepdims=True)
    sr = (W @ mu) / np.sqrt(np.einsum("ij,jk,ik->i", W, Sigma, W))
    assert np.max(sr) <= _sharpe(w, mu, Sigma) + 1e-12


def test_tangency_long_only_matches_direct_sharpe_maximisation():
    mu, Sigma = _cov_with_shorts()
    w = tangency_weights(mu, Sigma)
    assert np.all(w >= 0) and w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.sum(w == 0) >= 1
    w_direct = _max_sharpe_direct(mu, Sigma)
    assert _sharpe(w, mu, Sigma) >= _sharpe(w_direct, mu, Sigma) - 1e-10
    np.testing.assert_allclose(w, w_direct, atol=1e-4)
    # KKT of the reformulated QP: min 0.5 y'Sigma y s.t. mu'y = 1, y >= 0, with y = w / (mu'w)
    y = w / (mu @ w)
    assert_kkt(Sigma, np.zeros(8), mu / np.max(np.abs(mu)), 1.0 / np.max(np.abs(mu)), y,
               tol=1e-7)


def test_tangency_risk_free_and_errors():
    mu, Sigma = _cov_with_shorts()
    rf = 1e-4
    w = tangency_weights(mu, Sigma, rf=rf)
    w_direct = _max_sharpe_direct(mu - rf, Sigma)
    assert _sharpe(w, mu - rf, Sigma) >= _sharpe(w_direct, mu - rf, Sigma) - 1e-10
    with pytest.raises(ValueError, match="positive expected excess"):
        tangency_weights(-np.abs(mu), Sigma)
    with pytest.raises(ValueError, match="inefficient"):
        tangency_weights(-np.ones(8) * 1e-4, np.diag(np.full(8, 1e-4)), long_only=False)


def test_equal_weights():
    np.testing.assert_allclose(equal_weights(4), [0.25] * 4)
    with pytest.raises(ValueError):
        equal_weights(0)


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def _hand_series():
    """Two assets whose 50/50 daily-rebalanced portfolio has simple returns .1, -.2, .25, -.1."""
    simple_a = np.array([0.20, -0.30, 0.30, -0.10])
    simple_b = np.array([0.00, -0.10, 0.20, -0.10])
    R = np.log1p(np.column_stack([simple_a, simple_b]))
    return R, np.array([0.5, 0.5])


def test_evaluate_hand_computed():
    R, w = _hand_series()
    rp = np.array([0.10, -0.20, 0.25, -0.10])
    np.testing.assert_allclose(portfolio_returns(w, R), rp, atol=1e-15)
    out = evaluate(w, R, gamma=2.0)
    # mean = 0.05 / 4; deviations .0875, -.2125, .2375, -.1125 -> sum of squares .121875
    assert out["mean"] == pytest.approx(0.0125, abs=1e-15)
    assert out["var"] == pytest.approx(0.121875 / 3, abs=1e-15)
    assert out["vol"] == pytest.approx(np.sqrt(0.040625), abs=1e-15)
    assert out["sharpe"] == pytest.approx(np.sqrt(252) * 0.0125 / np.sqrt(0.040625), rel=1e-13)
    # wealth 1.1, 0.88, 1.1, 0.99 -> worst drawdown 0.88 / 1.1 - 1 = -20%
    assert out["max_drawdown"] == pytest.approx(0.2, abs=1e-14)
    assert out["ce"] == pytest.approx(0.0125 - 0.040625, abs=1e-15)
    assert out["n_obs"] == 4
    out_rf = evaluate(w, R, gamma=2.0, rf=0.0025)
    assert out_rf["sharpe"] == pytest.approx(np.sqrt(252) * 0.01 / np.sqrt(0.040625), rel=1e-13)
    assert out_rf["ce"] == pytest.approx(0.01 - 0.040625, abs=1e-15)


def test_evaluate_log_kind_and_drawdown():
    R = np.array([[0.01], [0.02], [-0.05], [0.01], [0.03]])
    out = evaluate(np.array([1.0]), R, kind="log", gamma=4.0)
    assert out["mean"] == pytest.approx(0.004, abs=1e-15)
    assert out["var"] == pytest.approx(np.var(R, ddof=1), rel=1e-13)
    # log wealth .01, .03, -.02, -.01, .02: peak .03 -> trough -.02
    assert out["max_drawdown"] == pytest.approx(1 - np.exp(-0.05), rel=1e-13)
    assert max_drawdown(np.full(10, 0.01)) == 0.0
    assert max_drawdown(np.array([-0.5, 0.0])) == pytest.approx(0.5)  # W_0 = 1 counts as a peak
    with pytest.raises(ValueError):
        evaluate(np.array([0.5, 0.5]), R)
    with pytest.raises(ValueError):
        portfolio_returns(np.array([1.0]), R, kind="arith")


def test_utility_gap():
    rng = np.random.default_rng(5)
    R = rng.normal(3e-4, 0.01, size=(500, 3))
    w_real = np.array([0.5, 0.3, 0.2])
    w_synth = np.array([0.2, 0.2, 0.6])
    gap = utility_gap(w_synth, w_real, R, gamma=5.0)
    es, er = evaluate(w_synth, R, gamma=5.0), evaluate(w_real, R, gamma=5.0)
    for m in ("mean", "vol", "var", "sharpe", "max_drawdown", "ce"):
        assert gap[f"{m}_gap"] == pytest.approx(es[m] - er[m], abs=1e-15)
        assert gap[f"{m}_synth"] == es[m] and gap[f"{m}_real"] == er[m]
    assert gap["ce_loss"] == pytest.approx(er["ce"] - es["ce"], abs=1e-15)
    assert gap["var_ratio"] == pytest.approx(es["var"] / er["var"])
    assert gap["weight_l1"] == pytest.approx(0.8)  # .3 + .1 + .4
    same = utility_gap(w_real, w_real, R)
    assert all(same[k] == 0.0 for k in same if k.endswith("_gap"))
    assert same["var_ratio"] == 1.0 and same["ce_loss"] == 0.0
