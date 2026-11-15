"""Tests for the relative tests: anchored (primary), two-critic (secondary, paired bootstrap) and
MMD (Bounliphone et al. 2016)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import stats

from synthval.models.univariate import GARCHt
from synthval.twosample import (C2ST, CriticTest, KSTest, MMDTest, SigMMDTest, median_heuristic,
                                relative_test, train_eval_split)
from synthval.twosample.critic import _draw_indices
from synthval.twosample.relative import (_mmd_bootstrap, perm_studentized_pvalue,
                                         relative_test_anchored, relative_test_two_critic)


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def max_rejections(reps: int, alpha: float = 0.05, tail: float = 0.005) -> int:
    return int(stats.binom.ppf(1.0 - tail, reps, alpha))


def binom_interval(reps: int, alpha: float = 0.05, level: float = 0.99) -> tuple[int, int]:
    """Central ``level`` interval of the number of rejections of an exact level-alpha test."""
    a = (1.0 - level) / 2.0
    return int(stats.binom.ppf(a, reps, alpha)), int(stats.binom.ppf(1.0 - a, reps, alpha))


def samples(rng, n=400, L=20, s_g=1.0, s_past=np.sqrt(1.5)):
    R_new = rng.standard_normal((n, L))
    G = s_g * rng.standard_normal((n, L))
    R_past = s_past * rng.standard_normal((n, L))
    return R_new, G, R_past


def t_windows(rng, n, L=20, nu=5.0, s=1.0):
    """i.i.d. windows of unit-variance Student-t returns (times ``s``)."""
    return s * rng.standard_t(nu, (n, L)) / np.sqrt(nu / (nu - 2.0))


def garch_windows(rng, n, L=20, scale=1.0, persistence=0.98, alpha=0.08, nu=5.0):
    """Independent GARCH(1,1)-t windows: every window is its own path with a 250-step burn-in."""
    beta = persistence - alpha
    m = GARCHt(mu=0.0, omega=scale**2 * (1.0 - persistence), alpha=alpha, beta=beta, nu=nu)
    return m.simulate(L, n, rng=rng, burn_in=250)


BASES = {
    "critic": lambda: CriticTest(n_steps=400, n_perm=100),
    "lrcritic": lambda: C2ST("logit", n_steps=300, n_perm=100),
    "c2st": lambda: C2ST(n_steps=300, n_perm=100),
    "mmd": lambda: MMDTest(),
    "sigmmd": lambda: SigMMDTest(),
}
LEARNED = ("critic", "lrcritic", "c2st")
CASES = ([(b, "anchored") for b in BASES] + [(b, "two_critic") for b in LEARNED])


# ----------------------------------------------------------------------------------------------
# Behaviour common to every version
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("base,method", CASES)
def test_good_generator_not_rejected(base, method):
    """G has the law of R_new while R_past differs: Delta < 0, no rejection."""
    rng = np.random.default_rng(1)
    for _ in range(2):
        res = relative_test(*samples(rng), BASES[base](), rng=rng, n_boot=300, method=method)
        assert res.pvalue > 0.05
        assert res.extra["D_past"] > res.extra["D_g"]
        assert np.isclose(res.statistic, res.extra["D_g"] - res.extra["D_past"])


@pytest.mark.parametrize("base,method", CASES)
def test_bad_generator_rejected(base, method):
    """G (scale 2) is clearly further from R_new than R_past (scale 1.1) is: reject."""
    rng = np.random.default_rng(2)
    for _ in range(2):
        res = relative_test(*samples(rng, s_g=2.0, s_past=1.1), BASES[base](), rng=rng,
                            n_boot=300, method=method)
        assert res.pvalue <= 0.01
        lo, hi = res.extra["ci"]
        assert 0 < lo <= res.statistic <= hi


def test_method_dispatch():
    rng = np.random.default_rng(12)
    Rn, G, Rp = samples(rng, n=120, L=10)
    base = CriticTest(n_steps=20, n_perm=20)
    assert relative_test(Rn, G, Rp, base, rng=rng).extra["relative_method"] == "anchored"
    res = relative_test(Rn, G, Rp, base, rng=rng, method="two_critic", n_boot=50)
    assert res.extra["relative_method"] == "two_critic" and "p_boot" in res.extra
    assert relative_test_two_critic is not None
    # MMD bases have no learned witness: Bounliphone whatever the method.
    for m in ("anchored", "two_critic"):
        assert relative_test(Rn, G, Rp, "mmd", rng=rng, method=m).extra["relative_method"] \
            == "bounliphone"
    with pytest.raises(ValueError):
        relative_test(Rn, G, Rp, base, rng=rng, method="nope")
    with pytest.raises(TypeError):
        relative_test_anchored(Rn, G, Rp, MMDTest(), rng=rng)


# ----------------------------------------------------------------------------------------------
# Anchored relative test
# ----------------------------------------------------------------------------------------------
def test_anchored_fields_and_definitions():
    rng = np.random.default_rng(13)
    n_new, n_g, n_p = 150, 240, 160
    Rn, G, Rp = rng.standard_normal((n_new, 10)), rng.standard_normal((n_g, 10)), \
        1.3 * rng.standard_normal((n_p, 10))
    res = relative_test(Rn, G, Rp, CriticTest(n_steps=60, n_perm=99), rng=rng)
    e = res.extra
    assert res.name == "relative_critic" and e["method"] == "anchored"
    assert e["pvalue_type"] == "permutation" and res.pvalue == e["p_perm"]
    assert np.isclose(res.statistic, e["mean_f_past_ev"] - e["mean_f_g_ev"])
    assert np.isclose(res.statistic, e["D_g"] - e["D_past"])
    # R_new is used only for training, all of it; G and R_past are split (train_frac 0.5).
    assert e["n_train_new"] == n_new and "n_eval_new" not in e
    assert e["n_train_g"] + e["n_eval_g"] == n_g and e["n_eval_g"] == n_g // 2
    assert e["n_train_past"] + e["n_eval_past"] == n_p and e["n_eval_past"] == n_p // 2
    # Normal CI from the i.i.d. se (no order given), centred at Delta.
    lo, hi = e["ci"]
    assert np.isclose(0.5 * (lo + hi), res.statistic)
    assert np.isclose(hi - lo, 2 * 1.959964 * e["se"], rtol=1e-4)
    assert e["hac"] is False and e["ci_hac"] == e["ci_iid"]
    assert 0 < e["p_perm_stud"] <= 1 and (1 / 100) <= res.pvalue <= 1
    s = e["train_summary"]
    assert s["steps_run"] == 60 and "wdist_last" in s and "gp_last" in s
    assert e["train_info"]["weighted"] is True
    for key in ("z", "p_z", "z_hac", "p_hac", "se_hac", "mean_f_new", "p_perm_conservative"):
        assert key in e
    # n_perm = 0: normal p-value of the i.i.d. z.
    res0 = relative_test(Rn, G, Rp, CriticTest(n_steps=20, n_perm=0), rng=rng)
    assert res0.extra["pvalue_type"] == "normal" and res0.pvalue == res0.extra["p_z"]
    assert np.isnan(res0.extra["p_perm_stud"])


class _RecordingCritic(CriticTest):
    """CriticTest that records the arguments of ``fit_witness``."""

    def fit_witness(self, X_tr, Y_tr, *, rng, weights_x=None, weights_y=None):
        self.seen = (np.asarray(X_tr).shape[0], np.asarray(Y_tr).shape[0], weights_x, weights_y)
        return super().fit_witness(X_tr, Y_tr, rng=rng, weights_x=weights_x, weights_y=weights_y)


def test_anchored_pool_has_equal_total_weights():
    """With unequal sizes, G_tr and R_past_tr each get total weight 1/2 in the negative class."""
    rng = np.random.default_rng(14)
    Rn, G, Rp = rng.standard_normal((100, 5)), rng.standard_normal((300, 5)), \
        rng.standard_normal((80, 5))
    base = _RecordingCritic(n_steps=10, n_perm=10)
    res = relative_test(Rn, G, Rp, base, rng=rng)
    n_x, n_y, wx, wy = base.seen
    ng_tr, np_tr = res.extra["n_train_g"], res.extra["n_train_past"]
    assert n_x == 100 and wx is None and n_y == ng_tr + np_tr
    assert np.isclose(wy[:ng_tr].sum(), 0.5) and np.isclose(wy[ng_tr:].sum(), 0.5)
    assert np.allclose(wy[:ng_tr], wy[0]) and np.allclose(wy[ng_tr:], wy[-1])


def test_weighted_training_rebalances_pool():
    """Weighted training: a rare pool component gets the same influence as the frequent one."""
    rng = np.random.default_rng(15)
    X = rng.standard_normal((600, 5))
    A = rng.standard_normal((900, 5))          # same law as X: uninformative
    B = 2.0 * rng.standard_normal((100, 5))    # rare, informative component
    Y = np.concatenate([A, B])
    w = np.concatenate([np.full(900, 0.5 / 900), np.full(100, 0.5 / 100)])
    B_new = 2.0 * rng.standard_normal((2000, 5))
    base = C2ST("logit", n_steps=300, early_stopping=False)
    f_unw, _ = base.fit_witness(X, Y, rng=np.random.default_rng(1))
    f_w, info = base.fit_witness(X, Y, rng=np.random.default_rng(1), weights_y=w)
    assert info["weighted"] is True
    # The weighted classifier sees B in half of the negative mini-batch points (10% unweighted),
    # so its logit on fresh B-like windows is clearly more negative.
    assert f_w(B_new).mean() < f_unw(B_new).mean() - 0.5
    # Early stopping path (validation weights) runs too.
    _, info_es = C2ST("logit", n_steps=100, eval_every=20).fit_witness(
        X, Y, rng=np.random.default_rng(2), weights_y=w)
    assert info_es["weighted"] is True and len(info_es["val_curve"]["bce"]) > 0
    _, info_c = CriticTest(n_steps=40, early_stopping=True, eval_every=10).fit_witness(
        X, Y, rng=np.random.default_rng(3), weights_y=w)
    assert info_c["weighted"] is True and len(info_c["val_curve"]["objective"]) > 0
    # Index sampler: zero-weight points are never drawn; None is uniform randint.
    wt = torch.tensor([0.0, 1.0, 0.0, 3.0], dtype=torch.float64)
    idx = _draw_indices(4, 1000, wt)
    assert set(idx.tolist()) <= {1, 3} and (idx == 3).float().mean() > 0.6
    assert _draw_indices(4, 50, None).max() < 4
    with pytest.raises(ValueError):
        base.fit_witness(X, Y, rng=rng, weights_y=-w)
    with pytest.raises(ValueError):
        base.fit_witness(X, Y, rng=rng, weights_y=w[:10])


def test_anchored_group_split():
    """G and R_past are split by group with one shared label namespace; R_new is not split."""
    rng = np.random.default_rng(5)
    n = 300
    Rn, G, Rp = samples(rng, n=n, L=10)
    g = np.repeat(np.arange(15), n // 15)
    splits = train_eval_split([n, n], rng=np.random.default_rng(9), groups=[g, g])
    assert not set(g[splits[0][0]]) & set(g[splits[1][1]])
    res = relative_test(Rn, G, Rp, CriticTest(n_steps=30, n_perm=20),
                        rng=np.random.default_rng(9), groups_new=g, groups_g=g, groups_past=g)
    assert res.extra["split"] == "group"
    assert res.extra["n_eval_g"] == splits[0][1].size
    assert res.extra["n_eval_past"] == splits[1][1].size
    assert res.extra["n_train_new"] == n


def test_anchored_hac_time_blocks_dependent_windows():
    """(iv) HAC path with time-block split on serially dependent windows (long GARCH paths)."""
    rng = np.random.default_rng(16)
    L, n = 20, 200
    m = GARCHt(mu=0.0, omega=0.02, alpha=0.08, beta=0.9, nu=5.0)

    def long_windows(scale=1.0):
        return scale * m.simulate(n * L, 1, rng=rng)[0].reshape(n, L)

    Rn, G, Rp = long_windows(), long_windows(1.1), long_windows()
    order = np.arange(n)
    kw = dict(order_new=order, order_g=order, order_past=order)
    base = CriticTest(n_steps=60, n_perm=50, time_blocks=10, inference="hac")
    res = relative_test(Rn, G, Rp, base, rng=rng, **kw)
    e = res.extra
    assert e["split"] == "time_block" and e["inference"] == "hac" and e["pvalue_type"] == "hac"
    assert res.pvalue == e["p_hac"] and e["hac"] is True
    assert len(e["hac_lags"]) == 2 and all(lag >= 0 for lag in e["hac_lags"])
    assert e["n_eval_g"] % (n // 10) == 0 and e["n_eval_past"] % (n // 10) == 0
    lo, hi = e["ci"]
    assert e["ci"] == e["ci_hac"] and np.isclose(0.5 * (lo + hi), res.statistic)
    assert np.isclose(hi - lo, 2 * 1.959964 * e["se_hac"], rtol=1e-4)
    assert np.isfinite(e["z_hac"]) and np.isfinite(e["p_perm"])
    # Explicit inference="default" with orders: permutation p-value, HAC still reported.
    res = relative_test(Rn, G, Rp, CriticTest(n_steps=30, n_perm=50, time_blocks=10), rng=rng,
                        inference="default", **kw)
    assert res.pvalue == res.extra["p_perm"] and res.extra["hac"] is True
    assert res.extra["ci"] == res.extra["ci_iid"]
    # order_new is not needed for the anchored split; order_g / order_past are.
    res = relative_test(Rn, G, Rp, base, rng=rng, order_g=order, order_past=order)
    assert res.extra["split"] == "time_block"
    with pytest.raises(ValueError):
        relative_test(Rn, G, Rp, base, rng=rng, order_g=order)
    with pytest.raises(ValueError):
        relative_test(Rn, G, Rp, base, rng=rng, groups_g=order % 3, groups_past=order % 3,
                      **kw)


@pytest.mark.parametrize("base", ["critic", "lrcritic"])
def test_anchored_exact_size_fast(base):
    """(i) Law(G) = Law(R_past) = Law(R_new), i.i.d. Student-t windows: level respected."""
    rng = np.random.default_rng(17)
    make = {"critic": lambda: CriticTest(n_steps=80, n_perm=200),
            "lrcritic": lambda: C2ST("logit", n_steps=100, n_perm=200)}[base]
    reps, n = 40, 150
    rej = 0
    for _ in range(reps):
        res = relative_test(t_windows(rng, n), t_windows(rng, n), t_windows(rng, n), make(),
                            rng=rng)
        rej += res.pvalue <= 0.05
    assert rej <= max_rejections(reps)


@pytest.mark.slow
@pytest.mark.parametrize("base,config", [("critic", "equal"), ("critic", "new_differs"),
                                         ("lrcritic", "new_differs")])
def test_anchored_exact_size_slow(base, config):
    """(i) Boundary Law(G) = Law(R_past), independent GARCH(1,1)-t windows (L = 20), 300 reps.

    ``equal``: R_new has the same law too; ``new_differs``: R_new has 1.25x the volatility (the
    critic then learns a non-trivial function, the case where the two-critic test fails). The
    5% rejection count must lie in the central 99% binomial interval of an exact test."""
    rng = np.random.default_rng({"equal": 18, "new_differs": 19}[config] + len(base))
    make = {"critic": lambda: CriticTest(n_steps=150, n_perm=200),
            "lrcritic": lambda: C2ST("logit", n_steps=200, n_perm=200)}[base]
    s_new = 1.0 if config == "equal" else 1.25
    reps, n = 300, 200
    rej = 0
    for _ in range(reps):
        res = relative_test(garch_windows(rng, n, scale=s_new), garch_windows(rng, n),
                            garch_windows(rng, n), make(), rng=rng)
        rej += res.pvalue <= 0.05
    lo, hi = binom_interval(reps)
    assert lo <= rej <= hi, (rej, lo, hi)


@pytest.mark.parametrize("base", ["critic", "lrcritic"])
def test_anchored_power(base):
    """(ii) R_past = R_new in law (Student-t), G Gaussian with 1.5x the volatility: reject."""
    rng = np.random.default_rng(20)
    make = {"critic": lambda: CriticTest(n_steps=300, n_perm=200),
            "lrcritic": lambda: C2ST("logit", n_steps=300, n_perm=200)}[base]
    for _ in range(2):
        n = 400
        res = relative_test(t_windows(rng, n), 1.5 * rng.standard_normal((n, 20)),
                            t_windows(rng, n), make(), rng=rng)
        assert res.pvalue <= 0.01 and res.extra["p_perm_stud"] <= 0.01
        assert res.extra["ci"][0] > 0


@pytest.mark.parametrize("base", ["critic", "lrcritic"])
def test_anchored_interior_of_null(base):
    """(iii) G = R_new in law, R_past farther (Gaussian, 1.5x vol): Delta < 0, no rejection."""
    rng = np.random.default_rng(22)
    make = {"critic": lambda: CriticTest(n_steps=300, n_perm=200),
            "lrcritic": lambda: C2ST("logit", n_steps=300, n_perm=200)}[base]
    rej = 0
    for _ in range(4):
        n = 400
        res = relative_test(t_windows(rng, n), t_windows(rng, n),
                            1.5 * rng.standard_normal((n, 20)), make(), rng=rng)
        assert res.statistic < 0
        rej += res.pvalue <= 0.05
    assert rej == 0


def test_perm_studentized_pvalue():
    rng = np.random.default_rng(23)
    # Exchangeable values (unequal sizes): approximately uniform, level respected.
    ps = np.array([perm_studentized_pvalue(rng.standard_normal(30), rng.standard_normal(60), 199,
                                           rng=rng) for _ in range(300)])
    assert np.sum(ps <= 0.05) <= max_rejections(300) and 0.4 < ps.mean() < 0.6
    assert perm_studentized_pvalue(rng.standard_normal(50) + 3, rng.standard_normal(50), 99,
                                   rng=rng) == pytest.approx(0.01)
    # Constant values: no evidence.
    assert perm_studentized_pvalue(np.ones(5), np.ones(5), 50, rng=rng) == 1.0


# ----------------------------------------------------------------------------------------------
# Two-critic relative test (secondary) and MMD (Bounliphone)
# ----------------------------------------------------------------------------------------------
def test_mmd_bootstrap_method():
    rng = np.random.default_rng(3)
    good = relative_test(*samples(rng), "mmd", rng=rng, mmd_method="bootstrap", n_boot=300)
    bad = relative_test(*samples(rng, s_g=2.0, s_past=1.1), "mmd", rng=rng,
                        mmd_method="bootstrap", n_boot=300)
    assert good.pvalue > 0.05 and bad.pvalue <= 0.01
    assert bad.extra["method"] == "bootstrap"
    assert 0.7 < bad.extra["se_boot"] / bad.extra["se"] < 1.4


@pytest.mark.parametrize("ng,nz", [(2000, 300), (300, 2000), (400, 400)])
def test_mmd_multinomial_bootstrap_is_centred(ng, nz):
    """E*[Delta*] = Delta for any sample sizes (the old version was biased when n_g != n_past,
    by about 0.7 asymptotic s.e. at n_g = 2000, n_past = 300)."""
    rng = np.random.default_rng(10)
    d = 5
    Fx = rng.standard_normal((400, d))
    Fy = 1.15 * rng.standard_normal((ng, d))
    Fz = 1.15 * rng.standard_normal((nz, d))
    h = median_heuristic(np.concatenate([Fx, Fy, Fz]), rng=rng)
    res = relative_test(Fx, Fy, Fz, MMDTest(bandwidth=h), rng=rng)
    boot = _mmd_bootstrap(Fx, Fy, Fz, (h,), 2000, rng=rng)
    mc_se = boot.std(ddof=1) / np.sqrt(boot.size)
    assert abs(boot.mean() - res.statistic) < 4 * mc_se
    assert abs(boot.mean() - res.statistic) < 0.15 * res.extra["se"]


def test_inference_hac_option():
    rng = np.random.default_rng(11)
    n = 200
    Rn, G, Rp = samples(rng, n=n, L=10)
    order = np.arange(n)
    kw = dict(order_new=order, order_g=order, order_past=order)
    res = relative_test(Rn, G, Rp, CriticTest(n_steps=50, n_perm=20), rng=rng, n_boot=100,
                        inference="hac", method="two_critic", **kw)
    assert res.pvalue == res.extra["p_hac"] and res.extra["method"] == "hac"
    lo, hi = res.extra["ci"]
    assert np.isclose(0.5 * (lo + hi), res.statistic)
    assert np.isclose(hi - lo, 2 * 1.959964 * res.extra["se_hac"], rtol=1e-4)
    assert "ci_boot" in res.extra and "p_boot" in res.extra
    # None follows the base test's own inference setting (both learned methods).
    for method in ("two_critic", "anchored"):
        res = relative_test(Rn, G, Rp, CriticTest(n_steps=50, n_perm=20, inference="hac"),
                            rng=rng, n_boot=100, method=method, **kw)
        assert res.extra["inference"] == "hac" and res.pvalue == res.extra["p_hac"]
    res = relative_test(Rn, G, Rp, MMDTest(inference="wild"), rng=rng, **kw)
    assert res.extra["inference"] == "hac" and res.pvalue == res.extra["p_hac"]
    res = relative_test(Rn, G, Rp, "mmd", rng=rng, mmd_method="bootstrap", n_boot=100,
                        inference="hac", **kw)
    assert res.pvalue == res.extra["p_hac"] and "ci_bootstrap" in res.extra
    res = relative_test(Rn, G, Rp, "mmd", rng=rng, **kw)
    assert res.extra["inference"] == "default" and res.pvalue == res.extra["p_z"]
    with pytest.raises(ValueError):
        relative_test(Rn, G, Rp, "mmd", rng=rng, inference="wild")
    # Time-block split for the two-critic version (serially dependent windows).
    res = relative_test(Rn, G, Rp, CriticTest(n_steps=30, n_perm=20, time_blocks=4,
                                               inference="hac"), rng=rng, n_boot=50,
                        method="two_critic", **kw)
    assert res.extra["split"] == "time_block" and res.extra["n_eval_new"] % 50 == 0
    for method in ("two_critic", "anchored"):
        with pytest.raises(ValueError):
            relative_test(Rn, G, Rp, CriticTest(n_steps=30, time_blocks=4), rng=rng,
                          method=method)


def test_result_fields_string_base_and_hac():
    rng = np.random.default_rng(4)
    n = 200
    Rn, G, Rp = samples(rng, n=n, L=10)
    order = np.arange(n)
    res = relative_test(Rn, G, Rp, CriticTest(n_steps=100, n_perm=50), rng=rng, n_boot=200,
                        order_new=order, order_g=order, order_past=order, method="two_critic")
    assert res.name == "relative_critic"
    assert np.isclose(res.statistic, res.extra["D_g"] - res.extra["D_past"])
    lo, hi = res.extra["ci"]
    assert lo < hi
    for key in ("z", "p_z", "z_hac", "p_hac", "se_boot", "p_abs_g", "p_abs_past"):
        assert key in res.extra
    assert res.extra["hac"] is True
    # Paired bootstrap SE agrees with the analytic SE of the same linear statistic.
    assert 0.75 < res.extra["se_boot"] / res.extra["se"] < 1.3
    res = relative_test(Rn, G, Rp, "mmd", rng=rng, order_new=order, order_g=order,
                        order_past=order)
    assert res.name == "relative_mmd" and res.extra["hac"] is True
    with pytest.raises(TypeError):
        relative_test(Rn, G, Rp, KSTest(), rng=rng)
    with pytest.raises(KeyError):
        relative_test(Rn, G, Rp, "nope", rng=rng)


def test_group_split_shared_across_three_samples():
    rng = np.random.default_rng(5)
    n = 300
    Rn, G, Rp = samples(rng, n=n, L=10)
    g = np.repeat(np.arange(15), n // 15)
    splits = train_eval_split([n, n, n], rng=np.random.default_rng(9), groups=[g, g, g])
    train_labels = set().union(*(set(g[tr]) for tr, _ in splits))
    eval_labels = set().union(*(set(g[ev]) for _, ev in splits))
    assert not train_labels & eval_labels
    res = relative_test(Rn, G, Rp, CriticTest(n_steps=50, n_perm=20),
                        rng=np.random.default_rng(9), n_boot=100, groups_new=g, groups_g=g,
                        groups_past=g, method="two_critic")
    assert res.extra["split"] == "group"
    assert res.extra["n_eval_new"] == splits[0][1].size
    assert res.extra["n_train_past"] == splits[2][0].size


def test_bounliphone_variance_matches_monte_carlo():
    """Asymptotic SE of Delta (Hoeffding projections) vs Monte Carlo SD, fixed bandwidth."""
    rng = np.random.default_rng(6)
    base = MMDTest(bandwidth=3.0)
    n, d, reps = 150, 5, 200
    shift_g, shift_p = np.zeros(d), np.zeros(d)
    shift_g[0], shift_p[0] = 0.6, -0.3
    deltas, ses = [], []
    for _ in range(reps):
        Rn = rng.standard_normal((n, d))
        G = rng.standard_normal((n, d)) + shift_g
        Rp = rng.standard_normal((n, d)) + shift_p
        res = relative_test(Rn, G, Rp, base, rng=rng)
        deltas.append(res.statistic)
        ses.append(res.extra["se"])
    ratio = np.sqrt(np.mean(np.square(ses))) / np.std(deltas, ddof=1)
    assert 0.8 < ratio < 1.25


def test_boundary_size_mmd():
    """Delta = 0 by symmetry (G shifted by +delta, R_past by -delta): level is respected."""
    rng = np.random.default_rng(7)
    reps, n, d = 100, 200, 10
    shift = np.zeros(d)
    shift[0] = 0.4
    rej = 0
    for _ in range(reps):
        Rn = rng.standard_normal((n, d))
        res = relative_test(Rn, rng.standard_normal((n, d)) + shift,
                            rng.standard_normal((n, d)) - shift, "mmd", rng=rng)
        rej += res.pvalue <= 0.05
    assert rej <= max_rejections(reps)


@pytest.mark.slow
def test_two_critic_anticonservative_at_boundary_slow():
    """theory.md Prop. 6.8: at Law(G) = Law(R_past) != Law(R_new) the two-critic test is
    anti-conservative (its delta is symmetric with a training-noise spread, so the unconditional
    rejection rate exceeds alpha; it tends to 1/2 for fixed critics and growing evaluation sets),
    while the anchored test on the same data keeps its level. A small training fraction (200
    training and 1800 evaluation windows per sample) makes the excess large: in a pilot the
    two-critic test rejected 8/30 at 5% with sd(z) = 2.1 (1 for a calibrated test)."""
    rng = np.random.default_rng(8)
    reps, n, d = 60, 2000, 10
    base = CriticTest(n_steps=200, n_perm=200, train_frac=0.1)
    rej_two = rej_anc = 0
    z_two = []
    for _ in range(reps):
        Rn = rng.standard_normal((n, d))
        G, Rp = 1.3 * rng.standard_normal((n, d)), 1.3 * rng.standard_normal((n, d))
        two = relative_test(Rn, G, Rp, base, rng=rng, method="two_critic", n_boot=300)
        anc = relative_test(Rn, G, Rp, base, rng=rng, method="anchored")
        rej_two += two.pvalue <= 0.05
        rej_anc += anc.pvalue <= 0.05
        z_two.append(two.extra["z"])
    assert rej_two > max_rejections(reps), rej_two
    assert np.std(z_two) > 1.4
    assert rej_anc <= max_rejections(reps), rej_anc
