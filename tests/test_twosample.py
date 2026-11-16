"""Tests for synthval.twosample: architectures, splits, HAC, permutation algebra, size and power."""
from __future__ import annotations

import itertools
import time

import numpy as np
import pytest
import torch
from scipy import stats

from synthval.twosample import (C2ST, TEST_REGISTRY, ConvCritic, CriticTest, KSTest, MLPCritic,
                                MMDTest, SigMMDTest, TestResult, andrews_hac_lags,
                                build_network, hac_var_mean, mean_diff_inference,
                                median_heuristic, mmd2_biased, mmd2_unbiased,
                                mmd2_wild_bootstrap, nw_hac_lags, perm_mean_diff, perm_pvalues,
                                time_grid, train_eval_split)
from synthval.twosample.mmd import (ar1_multipliers, gaussian_kernel_matrix, mmd2_permutations,
                                    wild_block_length)


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def max_rejections(reps: int, alpha: float = 0.05, tail: float = 0.005) -> int:
    """Largest rejection count compatible with level ``alpha`` (binomial upper ``tail`` quantile)."""
    return int(stats.binom.ppf(1.0 - tail, reps, alpha))


def gaussian_windows(rng, n, L=20, scale=1.0):
    return scale * rng.standard_normal((n, L))


def garch_t_path(rng, T, a=0.08, b=0.90, nu=5, burn=500):
    """One GARCH(1,1)-t path with unit unconditional variance (daily returns)."""
    w = 1.0 - a - b
    z = rng.standard_t(nu, T + burn) * np.sqrt((nu - 2) / nu)
    out = np.empty(T)
    s2, e = 1.0, 0.0
    for t in range(T + burn):
        s2 = w + a * e * e + b * s2
        e = np.sqrt(s2) * z[t]
        if t >= burn:
            out[t - burn] = e
    return out


def dependent_windows(rng, n, L=20):
    """``n`` consecutive non-overlapping windows of one persistent GARCH-t path.

    Such windows are serially dependent, hence not exchangeable.
    """
    return garch_t_path(rng, n * L).reshape(n, L)


# ----------------------------------------------------------------------------------------------
# Architectures
# ----------------------------------------------------------------------------------------------
def test_networks_shapes_and_no_batchnorm():
    x = torch.randn(7, 20)
    mlp = MLPCritic(20, hidden=16, depth=2)
    assert mlp(x).shape == (7,)
    conv = ConvCritic(3, 20, hidden=8)
    assert conv(torch.randn(7, 3, 20)).shape == (7,)
    conv1 = ConvCritic(1, 20, hidden=8)
    assert conv1(x).shape == (7,)  # (n, L) treated as (n, 1, L)
    for net in (mlp, conv, conv1, MLPCritic(20, layer_norm=True)):
        assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in net.modules())
    assert any(isinstance(m, torch.nn.LayerNorm) for m in conv.modules())


def test_build_network_dispatch():
    assert isinstance(build_network((20,)), MLPCritic)
    assert isinstance(build_network((4, 5)), ConvCritic)
    net = build_network((20,), "conv")
    assert isinstance(net, ConvCritic) and net.in_channels == 1
    assert isinstance(build_network((4, 5), "mlp"), MLPCritic)
    with pytest.raises(ValueError):
        build_network((2, 3, 4))


def test_pooled_conv_critic_shapes_dispatch_and_shift_invariance():
    from synthval.twosample import PooledConvCritic
    net = build_network((20,), "pooled")
    assert isinstance(net, PooledConvCritic) and net.in_channels == 1
    x = torch.randn(7, 20, dtype=torch.float64)
    assert net.double()(x).shape == (7,)
    mv = build_network((3, 20), "pooled", hidden=8)
    assert isinstance(mv, PooledConvCritic) and mv(torch.randn(5, 3, 20)).shape == (5,)
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in net.modules())
    assert any(isinstance(m, torch.nn.LayerNorm)
               for m in build_network((20,), "pooled", layer_norm=True).modules())
    # identical inputs map to identical outputs (no batch coupling)
    z = torch.zeros(3, 20, dtype=torch.float64)
    out = net(z)
    assert torch.allclose(out, out[0].expand(3))
    with pytest.raises(ValueError):
        build_network((2, 3, 4), "pooled")


def test_critic_test_pooled_arch_runs_and_detects_scale():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20))
    Y = 1.5 * rng.standard_normal((300, 20))
    res = CriticTest(arch="pooled", n_steps=150, n_perm=100)(X, Y, rng=np.random.default_rng(1))
    assert res.extra["arch"] == "PooledConvCritic"
    assert res.pvalue < 0.05


def test_witness_standardises_with_pooled_training_stats():
    rng = np.random.default_rng(0)
    Xtr = 3.0 + 2.0 * rng.standard_normal((100, 5))
    Ytr = -1.0 + 0.5 * rng.standard_normal((80, 5))
    w, info = CriticTest(n_steps=5, n_perm=10).fit_critic(Xtr, Ytr, rng=rng)
    pooled = np.concatenate([Xtr, Ytr])
    np.testing.assert_allclose(w.standardizer.mean_.ravel(), pooled.mean(0))
    np.testing.assert_allclose(w.standardizer.scale_.ravel(), pooled.std(0))
    assert w(Xtr).shape == (100,)
    assert info["steps_run"] == 5


# ----------------------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------------------
def test_group_split_never_shares_a_group():
    rng = np.random.default_rng(1)
    gx = rng.integers(0, 20, 400)
    gy = rng.integers(10, 30, 300)  # labels 10..19 appear in both samples
    for _ in range(20):
        (tx, ex), (ty, ey) = train_eval_split([400, 300], rng=rng, groups=[gx, gy])
        train_labels = set(gx[tx]) | set(gy[ty])
        eval_labels = set(gx[ex]) | set(gy[ey])
        assert not train_labels & eval_labels
        assert sorted(np.concatenate([tx, ex]).tolist()) == list(range(400))
        assert sorted(np.concatenate([ty, ey]).tolist()) == list(range(300))
        frac = (tx.size + ty.size) / 700
        assert 0.3 < frac < 0.7


def test_group_split_one_sample_grouped_and_random_split():
    rng = np.random.default_rng(2)
    gx = np.repeat(np.arange(10), 10)
    (tx, ex), (ty, ey) = train_eval_split([100, 51], rng=rng, groups=[gx, None])
    assert not set(gx[tx]) & set(gx[ex])
    assert ty.size == 26 and ey.size == 25 and not set(ty) & set(ey)
    with pytest.raises(ValueError):
        train_eval_split([10, 10], rng=rng, groups=[np.zeros(10, int), np.zeros(10, int)])


def test_critic_uses_group_split():
    rng = np.random.default_rng(3)
    n = 200
    X, Y = gaussian_windows(rng, n, 10), gaussian_windows(rng, n, 10)
    g = np.repeat(np.arange(20), n // 20)
    (tx, ex), (ty, ey) = train_eval_split([n, n], rng=np.random.default_rng(11), groups=[g, g])
    assert not (set(g[tx]) | set(g[ty])) & (set(g[ex]) | set(g[ey]))
    res = CriticTest(n_steps=20, n_perm=50)(X, Y, rng=np.random.default_rng(11), groups_x=g,
                                            groups_y=g)
    assert res.extra["split"] == "group"
    assert res.extra["n_train_x"] == tx.size and res.extra["n_eval_y"] == ey.size


# ----------------------------------------------------------------------------------------------
# HAC variance and permutation helpers
# ----------------------------------------------------------------------------------------------
def test_hac_var_mean_iid_ar1_and_panel():
    rng = np.random.default_rng(4)
    g = rng.standard_normal(500)
    v, lags = hac_var_mean(g)
    assert lags == 0 and np.isclose(v, g.var(ddof=1) / g.size)
    # AR(1), phi = 0.5: long-run variance = 1 / (1 - phi)^2 = 4 (unit innovations).
    phi, T = 0.5, 20000
    e = rng.standard_normal(T)
    x = np.empty(T)
    x[0] = e[0]
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    v_hac, lags = hac_var_mean(x, np.arange(T), lags=40)
    assert lags == 40
    assert 3.4 < v_hac * T < 4.6
    # Panel: k identical "tickers" per date -> variance of the mean equals that of one series.
    k = 5
    y = rng.standard_normal(400)
    panel = np.repeat(y, k)
    order = np.repeat(np.arange(400), k)
    v_panel, _ = hac_var_mean(panel, order, lags=0)
    assert np.isclose(v_panel, y.var(ddof=0) / 400, rtol=0.02)
    assert v_panel > 3 * hac_var_mean(panel)[0]


def test_time_grid_gcd_and_ranks():
    pos, n = time_grid(np.array([40, 0, 20, 100, 20]))
    assert n == 6 and pos.tolist() == [2, 0, 1, 5, 1]  # step 20: window units, holes kept
    pos, n = time_grid(np.array([0.5, 0.1, 0.5, 2.0]))
    assert n == 3 and pos.tolist() == [1, 0, 1, 2]  # non-integer: ranks of distinct values


def _ar1(rng, phi, T):
    e = rng.standard_normal(T)
    x = np.empty(T)
    x[0] = e[0] / np.sqrt(1 - phi**2)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_andrews_lags_data_driven_and_robust_to_holes():
    rng = np.random.default_rng(40)
    T = 4000
    # i.i.d.: (almost) no lags; AR(1) phi = 0.6: Andrews formula 1.1447 (alpha T)^(1/3) - 1.
    assert andrews_hac_lags(rng.standard_normal(T)) <= 2
    x = _ar1(rng, 0.6, T)
    alpha = 4 * 0.36 / (0.4**2 * 1.6**2)
    target = 1.1447 * (alpha * T) ** (1 / 3) - 1
    lag_full = andrews_hac_lags(x - x.mean())
    assert abs(lag_full - target) <= 0.25 * target
    # Observing a random half of the dates must not halve the estimated persistence.
    keep = np.sort(rng.choice(T, T // 2, replace=False))
    _, lags_half = hac_var_mean(x[keep], keep)
    assert abs(lags_half - lag_full) <= 0.25 * lag_full
    # Default lag for one observation per date equals the Andrews choice; "nw" is the old rule.
    _, lags_def = hac_var_mean(x, np.arange(T))
    assert abs(lags_def - lag_full) <= 1
    assert hac_var_mean(x, np.arange(T), lags="nw")[1] == nw_hac_lags(T)
    with pytest.raises(ValueError):
        hac_var_mean(x, np.arange(T), lags="bogus")
    # Window start days 0, 20, 40, ...: lags are counted in windows, not days.
    v_days, l_days = hac_var_mean(x[:500], 20 * np.arange(500))
    v_idx, l_idx = hac_var_mean(x[:500], np.arange(500))
    assert l_days == l_idx and np.isclose(v_days, v_idx)


def test_hac_size_fixed_witness_dependent_windows():
    """Fixed witness (log realised variance) on consecutive GARCH-t windows: the permutation and
    i.i.d. z over-reject (exchangeability fails) while the default (Andrews) HAC p-value is
    much closer to the level. Reviewer's design (600 reps there: perm 15%, NW lag 9.2%,
    Andrews lag 6.8%)."""
    rng = np.random.default_rng(41)
    reps, n, L = 200, 500, 20
    rej = {"perm": 0, "hac": 0}
    lags = []
    for _ in range(reps):
        X, Y = dependent_windows(rng, n, L), dependent_windows(rng, n, L)
        ex = np.sort(rng.permutation(n)[:n // 2])
        ey = np.sort(rng.permutation(n)[:n // 2])
        inf = mean_diff_inference(np.log((X[ex] ** 2).mean(1)), np.log((Y[ey] ** 2).mean(1)),
                                  rng=rng, n_perm=200, order_x=ex, order_y=ey)
        rej["perm"] += inf["p_perm"] <= 0.05
        rej["hac"] += inf["p_hac"] <= 0.05
        lags += list(inf["hac_lags"])
    assert rej["perm"] >= 18  # >= 9%: documents the failure of exchangeability
    assert rej["hac"] <= max_rejections(reps, alpha=0.07)
    assert np.mean(lags) >= 8 > nw_hac_lags(n)  # data-driven lag is longer than the NW rule (4)


def test_perm_pvalues_ties():
    t_perm = np.array([0.0, 1.0, 1.0, 2.0, -1.0])
    pv = perm_pvalues(1.0, t_perm, rng=np.random.default_rng(0))
    assert pv["n_ties"] == 2
    assert np.isclose(pv["conservative"], (1 + 1 + 2) / 6)
    assert np.isclose(pv["mid"], (1 + 0.5 * 3) / 6)
    assert 1 / 6 <= pv["randomized"] <= 4 / 6
    assert np.isnan(perm_pvalues(1.0, t_perm)["randomized"])


def test_randomized_perm_pvalue_exact_for_discrete_statistic():
    """Discrete witness (0/1): the conservative p-value is undersized, the randomized one is
    uniform (exact size at every alpha)."""
    rng = np.random.default_rng(42)
    reps, n = 2000, 30
    p_cons, p_rand = np.empty(reps), np.empty(reps)
    for r in range(reps):
        g = (rng.uniform(size=2 * n) < 0.3).astype(float)
        inf = mean_diff_inference(g[:n], g[n:], rng=rng, n_perm=199, tie_break="randomized")
        p_cons[r], p_rand[r] = inf["p_perm_conservative"], inf["p_perm"]
    assert np.mean(p_cons <= 0.05) < 0.04
    for a in (0.05, 0.1, 0.5):
        assert abs(np.mean(p_rand <= a) - a) < 3.5 * np.sqrt(a * (1 - a) / reps)
    assert stats.kstest(p_rand, "uniform").pvalue > 0.001


def test_perm_mean_diff_matches_exact_enumeration():
    rng = np.random.default_rng(5)
    g = rng.standard_normal(8)
    n1 = 4
    t_obs = g[:n1].mean() - g[n1:].mean()
    exact = []
    for idx in itertools.combinations(range(8), n1):
        mask = np.zeros(8, bool)
        mask[list(idx)] = True
        exact.append(g[mask].mean() - g[~mask].mean())
    p_exact = np.mean(np.array(exact) >= t_obs - 1e-12)
    t_perm = perm_mean_diff(g, n1, 20000, rng=rng)
    assert abs(np.mean(t_perm >= t_obs - 1e-12) - p_exact) < 0.015
    assert set(np.round(np.unique(t_perm), 10)) <= set(np.round(exact, 10))


# ----------------------------------------------------------------------------------------------
# MMD algebra
# ----------------------------------------------------------------------------------------------
def _naive_mmd2(X, Y, bws):
    def k(a, b):
        return sum(np.exp(-np.sum((a - b) ** 2) / (2 * h * h)) for h in bws)
    n, m = len(X), len(Y)
    sxx = sum(k(X[i], X[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    syy = sum(k(Y[i], Y[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    sxy = sum(k(X[i], Y[j]) for i in range(n) for j in range(m)) / (n * m)
    return sxx + syy - 2 * sxy


@pytest.mark.parametrize("bws", [(1.3,), (0.4, 0.9, 2.0)])
def test_mmd2_unbiased_matches_naive(bws):
    rng = np.random.default_rng(6)
    X, Y = rng.standard_normal((12, 3)), 1.2 * rng.standard_normal((9, 3))
    K = gaussian_kernel_matrix(np.concatenate([X, Y]), bws, block=5)
    assert np.isclose(mmd2_unbiased(K, 12), _naive_mmd2(X, Y, bws))


def test_mmd_vectorised_permutations_match_recomputation():
    rng = np.random.default_rng(7)
    Z = rng.standard_normal((25, 4))
    K = gaussian_kernel_matrix(Z, (1.5,))
    n_x = 10
    out = mmd2_permutations(K, n_x, 7, rng=np.random.default_rng(42), block=3)
    base = np.zeros(25)
    base[:n_x] = 1.0
    rng2 = np.random.default_rng(42)
    masks = np.concatenate([rng2.permuted(np.tile(base, (b, 1)), axis=1) for b in (3, 3, 1)])
    for t, mask in zip(out, masks):
        order = np.concatenate([np.flatnonzero(mask), np.flatnonzero(mask == 0)])
        assert np.isclose(t, mmd2_unbiased(K[np.ix_(order, order)], n_x))


def test_median_heuristic_matches_pdist():
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(8)
    Z = rng.standard_normal((300, 6))
    assert np.isclose(median_heuristic(Z, rng=rng, max_points=1000), np.median(pdist(Z)))


def test_ar1_multipliers_covariance():
    rng = np.random.default_rng(30)
    W = ar1_multipliers(400, 3.0, 2000, rng=rng)
    assert abs(W.var() - 1.0) < 0.03
    for lag in (1, 3):
        r = np.mean(W[lag:] * W[:-lag]) / W.var()
        assert abs(r - np.exp(-lag / 3.0)) < 0.03
    iid = ar1_multipliers(400, 0.0, 500, rng=rng)
    assert abs(np.mean(iid[1:] * iid[:-1])) < 0.01


def test_wild_bootstrap_conditional_mean_exact():
    """E*[a'Ka] = tr(K C Sigma_w C') for the centred, scaled multipliers (shared positions)."""
    rng = np.random.default_rng(31)
    nx, ny, block = 40, 30, 2.5
    Z = rng.standard_normal((nx + ny, 3))
    K = gaussian_kernel_matrix(Z, (1.7,))
    order = np.concatenate([rng.permutation(60)[:nx], rng.permutation(60)[:ny]])
    pos, n_grid = time_grid(order)
    sw = np.exp(-np.abs(pos[:, None] - pos[None, :]) / block)
    C = np.zeros((nx + ny, nx + ny))
    C[:nx, :nx] = (np.eye(nx) - 1.0 / nx) / nx
    C[nx:, nx:] = -(np.eye(ny) - 1.0 / ny) / ny
    exact = np.trace(K @ C @ sw @ C.T)
    reps = mmd2_wild_bootstrap(K, nx, pos, n_grid, block, 20000, rng=rng)
    assert abs(reps.mean() - exact) < 4 * reps.std() / np.sqrt(reps.size)
    # V-statistic identity.
    a = np.r_[np.full(nx, 1 / nx), np.full(ny, -1 / ny)]
    assert np.isclose(mmd2_biased(K, nx), a @ K @ a)


def test_wild_block_length_detects_dependence():
    rng = np.random.default_rng(32)
    n = 400
    Xi, Yi = gaussian_windows(rng, n, 10), gaussian_windows(rng, n, 10)
    Xd, Yd = dependent_windows(rng, n, 10), dependent_windows(rng, n, 10)
    blocks = {}
    for name, (X, Y) in {"iid": (Xi, Yi), "dep": (Xd, Yd)}.items():
        Z = np.concatenate([X, Y])
        K = gaussian_kernel_matrix(Z, (median_heuristic(Z, rng=rng),))
        pos, n_grid = time_grid(np.r_[np.arange(n), np.arange(n)])
        blocks[name] = wild_block_length(K, n, pos, n_grid, rng=rng)
    assert blocks["iid"] < 2.5 < blocks["dep"]


def test_mmd_wild_interface():
    rng = np.random.default_rng(33)
    X, Y = rng.standard_normal((150, 8)), rng.standard_normal((120, 8))
    res = MMDTest(inference="wild", n_perm=100)(X, Y, rng=rng)
    assert res.extra["inference"] == "wild" and res.extra["wild_block"] >= 0
    assert np.isfinite(res.extra["mmd2_v"]) and 0 < res.pvalue <= 1
    res = MMDTest(inference="wild", n_perm=100, wild_block=4.0, max_n=100)(
        X, Y, rng=rng, order_x=20 * np.arange(150), order_y=20 * np.arange(120))
    assert res.extra["wild_block"] == 4.0 and res.extra["n_used_x"] == 100
    assert res.extra["n_grid"] <= 150  # day indices 0, 20, ... are mapped to window units
    res = SigMMDTest(inference="wild", n_perm=100, depth=3)(X, Y, rng=rng)
    assert res.extra["inference"] == "wild" and "wild_block" in res.extra
    with pytest.raises(ValueError):
        MMDTest(inference="wild")(X, Y, rng=rng, order_x=np.arange(3))
    with pytest.raises(ValueError):
        MMDTest(inference="bootstrap")


def test_size_mmd_wild_iid():
    reps = 100
    t = MMDTest(n_perm=200, inference="wild")
    assert _rejections(t, reps, 100, 5, seed=106) <= max_rejections(reps)


def test_mmd_subsamples_above_max_n():
    rng = np.random.default_rng(9)
    res = MMDTest(max_n=50, n_perm=20)(rng.standard_normal((120, 5)),
                                       rng.standard_normal((80, 5)), rng=rng)
    assert res.extra["n_used_x"] == 50 and res.extra["n_used_y"] == 50
    res = MMDTest(n_perm=20, multi_bandwidth=True)(rng.standard_normal((30, 5)),
                                                   rng.standard_normal((30, 5)), rng=rng)
    assert len(res.extra["bandwidths"]) == 5


# ----------------------------------------------------------------------------------------------
# Learned tests: interface details
# ----------------------------------------------------------------------------------------------
def test_critic_extra_fields_and_hac():
    rng = np.random.default_rng(10)
    n = 200
    X, Y = gaussian_windows(rng, n, 10), gaussian_windows(rng, n, 10, 1.3)
    res = CriticTest(n_steps=50, n_perm=100)(X, Y, rng=rng)
    assert isinstance(res, TestResult) and res.name == "critic"
    for key in ("z", "p_z", "z_hac", "p_hac", "W1_lower", "train_curve", "n_eval_x"):
        assert key in res.extra
    assert res.extra["W1_lower"] == res.statistic
    assert res.extra["z_hac"] == res.extra["z"] and res.extra["hac"] is False
    assert len(res.extra["train_curve"]["step"]) <= 100
    assert 1 / 101 <= res.pvalue <= 1
    order = np.arange(n)
    res2 = CriticTest(n_steps=50, n_perm=100)(X, Y, rng=rng, order_x=order, order_y=order)
    assert res2.extra["hac"] is True and np.isfinite(res2.extra["z_hac"])
    assert 0 <= res2.extra["p_hac"] <= 1


def test_critic_unequal_sizes_conv_and_multivariate():
    rng = np.random.default_rng(11)
    res = CriticTest(n_steps=30, n_perm=50)(rng.standard_normal((300, 10)),
                                            rng.standard_normal((150, 10)), rng=rng)
    assert res.n_x == 300 and res.n_y == 150 and res.extra["arch"] == "MLPCritic"
    res = CriticTest(n_steps=30, n_perm=50, arch="conv")(rng.standard_normal((100, 10)),
                                                         rng.standard_normal((100, 10)), rng=rng)
    assert res.extra["arch"] == "ConvCritic"
    res = CriticTest(n_steps=30, n_perm=50)(rng.standard_normal((100, 3, 5)),
                                            rng.standard_normal((100, 3, 5)), rng=rng)
    assert res.extra["arch"] == "ConvCritic"


def test_critic_early_stopping_and_determinism():
    rng = np.random.default_rng(12)
    X, Y = gaussian_windows(rng, 200, 10), gaussian_windows(rng, 200, 10)
    t = CriticTest(n_steps=2000, n_perm=50, early_stopping=True, eval_every=10, patience=2)
    res = t(X, Y, rng=np.random.default_rng(0))
    assert res.extra["steps_run"] < 2000 and "val_curve" in res.extra
    res2 = t(X, Y, rng=np.random.default_rng(0))
    assert res.statistic == res2.statistic and res.pvalue == res2.pvalue


def test_c2st_statistics_and_binomial_pvalue():
    rng = np.random.default_rng(13)
    X, Y = gaussian_windows(rng, 200, 10), gaussian_windows(rng, 200, 10, 1.5)
    acc = C2ST(n_steps=100, n_perm=0, tie_break="conservative")(X, Y,
                                                                rng=np.random.default_rng(1))
    assert acc.name == "c2st" and 0 <= acc.statistic <= 1
    assert np.isclose(acc.statistic, acc.extra["accuracy"])
    m = acc.extra["n_eval_x"] + acc.extra["n_eval_y"]
    correct = round(acc.statistic * m)
    assert np.isclose(acc.pvalue, stats.binom.sf(correct - 1, m, 0.5))
    # Default for the accuracy: randomized tie-breaking within the atom at the observed count.
    rnd = C2ST(n_steps=100, n_perm=0)(X, Y, rng=np.random.default_rng(1))
    assert rnd.extra["tie_break"] == "randomized" and rnd.statistic == acc.statistic
    assert stats.binom.sf(correct, m, 0.5) <= rnd.pvalue <= acc.pvalue
    lr = C2ST("logit", n_steps=100, n_perm=100)(X, Y, rng=np.random.default_rng(1))
    assert lr.name == "lrcritic" and np.isclose(lr.statistic, lr.extra["mean_logit_diff"])
    assert lr.extra["tie_break"] == "conservative"


def test_c2st_permutation_pvalue_variants():
    rng = np.random.default_rng(16)
    X, Y = gaussian_windows(rng, 200, 10), gaussian_windows(rng, 200, 10)
    res = C2ST(n_steps=60, n_perm=200)(X, Y, rng=rng)
    ex = res.extra
    assert ex["tie_break"] == "randomized" and ex["p_perm"] == res.pvalue
    lo = ex["p_perm_conservative"] - (1 + ex["perm_ties"]) / 201
    assert lo - 1e-12 <= res.pvalue <= ex["p_perm_conservative"] + 1e-12


def test_learned_inference_hac_and_options():
    rng = np.random.default_rng(17)
    n = 200
    X, Y = gaussian_windows(rng, n, 10), gaussian_windows(rng, n, 10)
    order = np.arange(n)
    res = CriticTest(n_steps=30, n_perm=50, inference="hac")(X, Y, rng=rng, order_x=order,
                                                              order_y=order)
    assert res.pvalue == res.extra["p_hac"] and res.extra["inference"] == "hac"
    assert res.extra["hac"] is True and 0 < res.extra["p_perm"] <= 1
    res = CriticTest(n_steps=30, n_perm=50, inference="hac", hac_lags="nw")(
        X, Y, rng=rng, order_x=order, order_y=order)
    assert res.extra["hac_lags"] == (nw_hac_lags(res.extra["n_eval_x"]),
                                     nw_hac_lags(res.extra["n_eval_y"]))
    with pytest.raises(ValueError):
        CriticTest(inference="wild")
    with pytest.raises(ValueError):
        C2ST(tie_break="up")
    assert CriticTest().lr == 1e-3  # deviation from DESIGN (1e-4), see CriticTest docstring


def test_time_block_split():
    from synthval.twosample.critic import time_block_labels

    lab = time_block_labels(20 * np.arange(100), 4)
    assert lab.tolist() == np.repeat(np.arange(4), 25).tolist()
    assert time_block_labels(np.array([3.0, 3.0]), 3).tolist() == [0, 0]
    rng = np.random.default_rng(19)
    n = 200
    X, Y = dependent_windows(rng, n, 10), dependent_windows(rng, n, 10)
    order = np.arange(n)
    t = CriticTest(n_steps=30, n_perm=50, time_blocks=5, inference="hac")
    res = t(X, Y, rng=rng, order_x=order, order_y=order)
    assert res.extra["split"] == "time_block"
    assert res.extra["n_train_x"] % 40 == 0 and res.extra["n_eval_y"] % 40 == 0
    assert res.extra["n_train_x"] == res.extra["n_train_y"]  # shared block labels
    with pytest.raises(ValueError):
        t(X, Y, rng=rng)  # needs order_*
    with pytest.raises(ValueError):
        t(X, Y, rng=rng, order_x=order, order_y=order, groups_x=order % 7, groups_y=order % 7)
    with pytest.raises(ValueError):
        CriticTest(time_blocks=1)


def test_early_stopping_min_steps():
    rng = np.random.default_rng(18)
    X, Y = gaussian_windows(rng, 200, 10), gaussian_windows(rng, 200, 10)
    res = C2ST(n_steps=600, n_perm=20, eval_every=25, patience=2, min_steps=200)(X, Y, rng=rng)
    assert res.extra["best_step"] >= 200 and res.extra["steps_run"] >= 250
    res = CriticTest(n_steps=600, n_perm=20, early_stopping=True, eval_every=25, patience=2,
                     min_steps=150)(X, Y, rng=rng)
    assert res.extra["best_step"] >= 150


def test_c2st_fit_logit_recovers_log_likelihood_ratio():
    rng = np.random.default_rng(14)
    mu = np.array([1.0, 0.5])
    n = 3000
    X = rng.standard_normal((n, 2)) + mu
    Y = rng.standard_normal((n, 2))
    logit = C2ST(n_steps=1500, depth=2, hidden=32, early_stopping=False).fit_logit(X, Y, rng=rng)
    Z = rng.standard_normal((2000, 2)) * 1.5 + mu / 2
    true = Z @ mu - mu @ mu / 2
    est = logit(Z)
    assert np.corrcoef(est, true)[0, 1] > 0.95
    slope = np.polyfit(true, est, 1)[0]
    assert 0.7 < slope < 1.3
    assert "train_curve" in logit.info


def test_registry():
    assert set(TEST_REGISTRY) == {"critic", "lrcritic", "c2st", "mmd", "sigmmd", "ks"}
    rng = np.random.default_rng(15)
    X, Y = gaussian_windows(rng, 60, 8), gaussian_windows(rng, 60, 8)
    fast = {"critic": dict(n_steps=10, n_perm=20), "lrcritic": dict(n_steps=10, n_perm=20),
            "c2st": dict(n_steps=10, n_perm=20), "mmd": dict(n_perm=20),
            "sigmmd": dict(n_perm=20, depth=3), "ks": {}}
    for name, factory in TEST_REGISTRY.items():
        test = factory(**fast[name])
        assert test.name == name
        res = test(X, Y, rng=rng)
        assert 0 < res.pvalue <= 1 and np.isfinite(res.statistic)


# ----------------------------------------------------------------------------------------------
# Size under H0 (i.i.d. N(0, 1) windows in both samples)
# ----------------------------------------------------------------------------------------------
def _rejections(test, reps, n, L, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(reps):
        res = test(gaussian_windows(rng, n, L), gaussian_windows(rng, n, L), rng=rng)
        count += res.pvalue <= alpha
    return count


def test_size_mmd():
    reps = 100
    assert _rejections(MMDTest(n_perm=100), reps, 60, 5, seed=100) <= max_rejections(reps)


def test_size_ks():
    reps = 100
    assert _rejections(KSTest(), reps, 60, 5, seed=101) <= max_rejections(reps)


def test_size_critic():
    reps = 30
    t = CriticTest(n_steps=40, batch_size=64, n_perm=100)
    assert _rejections(t, reps, 100, 10, seed=102) <= max_rejections(reps)


@pytest.mark.parametrize("statistic", ["accuracy", "logit"])
def test_size_c2st(statistic):
    reps = 30
    t = C2ST(statistic, n_steps=40, batch_size=64, n_perm=100, early_stopping=False)
    assert _rejections(t, reps, 100, 10, seed=103) <= max_rejections(reps)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["critic", "c2st", "lrcritic"])
def test_size_learned_slow(name):
    reps = 100
    t = TEST_REGISTRY[name](n_steps=300, n_perm=200)
    assert _rejections(t, reps, 300, 20, seed=104) <= max_rejections(reps)


@pytest.mark.slow
@pytest.mark.parametrize("cls", [MMDTest, SigMMDTest])
def test_size_wild_dependent_windows_slow(cls):
    """Consecutive GARCH-t windows (H0): the permutation test over-rejects badly (reviewer: MMD
    25%, SigMMD 14%); the wild bootstrap removes most of the distortion (our Monte Carlo, 400
    reps: MMD about 8%, SigMMD about 10% at nominal 5%; asymptotic, not exact)."""
    reps, n = 100, 500
    rng = np.random.default_rng(107)
    perm, wild = cls(n_perm=200), cls(n_perm=200, inference="wild")
    order = np.arange(n)
    rej_p = rej_w = 0
    for _ in range(reps):
        X, Y = dependent_windows(rng, n), dependent_windows(rng, n)
        rej_p += perm(X, Y, rng=rng).pvalue <= 0.05
        rej_w += wild(X, Y, rng=rng, order_x=order, order_y=order).pvalue <= 0.05
    assert rej_w <= max_rejections(reps, alpha=0.10)
    assert rej_w < rej_p


@pytest.mark.slow
def test_size_hac_pvalue_slow():
    """Normal HAC p-value of the critic stays close to its level on independent windows."""
    reps, n = 100, 300
    rng = np.random.default_rng(105)
    t = CriticTest(n_steps=100, n_perm=50)
    order = np.arange(n)
    count = 0
    for _ in range(reps):
        res = t(gaussian_windows(rng, n, 10), gaussian_windows(rng, n, 10), rng=rng,
                order_x=order, order_y=order)
        count += res.extra["p_hac"] <= 0.05
    assert count <= max_rejections(reps)


# ----------------------------------------------------------------------------------------------
# Power
# ----------------------------------------------------------------------------------------------
POWER_TESTS = {
    "critic": lambda: CriticTest(n_steps=500, n_perm=200),
    "c2st": lambda: C2ST(n_steps=300, n_perm=200),
    "mmd": lambda: MMDTest(n_perm=200),
    "sigmmd": lambda: SigMMDTest(n_perm=200),
    "ks": lambda: KSTest(),
}


@pytest.mark.parametrize("name", list(POWER_TESTS))
def test_power_variance_ratio(name):
    rng = np.random.default_rng(200)
    X, Y = gaussian_windows(rng, 300, 20), gaussian_windows(rng, 300, 20, np.sqrt(1.5))
    assert POWER_TESTS[name]()(X, Y, rng=rng).pvalue <= 0.01


@pytest.mark.parametrize("name", list(POWER_TESTS))
def test_power_student_t3(name):
    """Student-t(3) returns vs Gaussian returns (unit-scale t, as drawn)."""
    rng = np.random.default_rng(201)
    X, Y = gaussian_windows(rng, 300, 20), rng.standard_t(3, (300, 20))
    assert POWER_TESTS[name]()(X, Y, rng=rng).pvalue <= 0.01


@pytest.mark.parametrize("name", ["mmd", "sigmmd", "ks"])
def test_power_variance_matched_t3(name):
    """Variance-matched t(3): only the shape (tails) differs."""
    rng = np.random.default_rng(202)
    X, Y = gaussian_windows(rng, 500, 20), rng.standard_t(3, (500, 20)) / np.sqrt(3.0)
    assert POWER_TESTS[name]()(X, Y, rng=rng).pvalue <= 0.05


@pytest.mark.slow
@pytest.mark.parametrize("name", ["critic", "c2st", "lrcritic"])
def test_power_variance_matched_t3_slow(name):
    reps, n = 10, 2000
    rng = np.random.default_rng(203)
    t = TEST_REGISTRY[name](n_perm=200)
    rej = sum(t(gaussian_windows(rng, n, 20), rng.standard_t(3, (n, 20)) / np.sqrt(3.0),
                rng=rng).pvalue <= 0.05 for _ in range(reps))
    assert rej >= 7


@pytest.mark.slow
def test_critic_speed_target():
    """n = 2000 windows per sample, L = 20, default settings, one thread: under 25 s."""
    rng = np.random.default_rng(300)
    X, Y = gaussian_windows(rng, 2000, 20), gaussian_windows(rng, 2000, 20)
    t0 = time.perf_counter()
    CriticTest()(X, Y, rng=rng)
    assert time.perf_counter() - t0 < 25.0


def test_default_arch_env(monkeypatch):
    from synthval.twosample import PooledConvCritic
    from synthval.twosample.nets import default_arch
    monkeypatch.delenv("SYNTHVAL_ARCH_1D", raising=False)
    monkeypatch.delenv("SYNTHVAL_ARCH_2D", raising=False)
    assert default_arch(1) == "mlp" and default_arch(2) == "conv"
    assert isinstance(build_network((20,)), MLPCritic)
    monkeypatch.setenv("SYNTHVAL_ARCH_1D", "pooled")
    monkeypatch.setenv("SYNTHVAL_ARCH_2D", "pooled")
    assert isinstance(build_network((20,)), PooledConvCritic)
    assert isinstance(build_network((3, 20)), PooledConvCritic)
    assert isinstance(build_network((20,), "mlp"), MLPCritic)  # explicit arch wins
    monkeypatch.setenv("SYNTHVAL_ARCH_1D", "transformer")
    with pytest.raises(ValueError):
        build_network((20,))


def test_device_default_cpu(monkeypatch):
    from synthval.twosample.nets import get_device, to_tensor
    monkeypatch.delenv("SYNTHVAL_DEVICE", raising=False)
    assert get_device().type == "cpu"
    assert to_tensor(np.zeros((2, 3))).device.type == "cpu"
