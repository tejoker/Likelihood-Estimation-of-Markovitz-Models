"""Tests for the relative-similarity test (critic / C2ST paired bootstrap, MMD Bounliphone)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import stats

from synthval.twosample import (C2ST, CriticTest, KSTest, MMDTest, SigMMDTest, median_heuristic,
                                relative_test, train_eval_split)
from synthval.twosample.relative import _mmd_bootstrap


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def max_rejections(reps: int, alpha: float = 0.05, tail: float = 0.005) -> int:
    return int(stats.binom.ppf(1.0 - tail, reps, alpha))


def samples(rng, n=400, L=20, s_g=1.0, s_past=np.sqrt(1.5)):
    R_new = rng.standard_normal((n, L))
    G = s_g * rng.standard_normal((n, L))
    R_past = s_past * rng.standard_normal((n, L))
    return R_new, G, R_past


BASES = {
    "critic": lambda: CriticTest(n_steps=400, n_perm=100),
    "lrcritic": lambda: C2ST("logit", n_steps=300, n_perm=100),
    "c2st": lambda: C2ST(n_steps=300, n_perm=100),
    "mmd": lambda: MMDTest(),
    "sigmmd": lambda: SigMMDTest(),
}


@pytest.mark.parametrize("base", list(BASES))
def test_good_generator_not_rejected(base):
    """G has the law of R_new while R_past differs: Delta < 0, no rejection."""
    rng = np.random.default_rng(1)
    for _ in range(2):
        res = relative_test(*samples(rng), BASES[base](), rng=rng, n_boot=300)
        assert res.pvalue > 0.05
        assert res.extra["D_past"] > res.extra["D_g"]


@pytest.mark.parametrize("base", list(BASES))
def test_bad_generator_rejected(base):
    """G (scale 2) is clearly further from R_new than R_past (scale 1.1) is: reject."""
    rng = np.random.default_rng(2)
    for _ in range(2):
        res = relative_test(*samples(rng, s_g=2.0, s_past=1.1), BASES[base](), rng=rng,
                            n_boot=300)
        assert res.pvalue <= 0.01
        lo, hi = res.extra["ci"]
        assert 0 < lo <= res.statistic <= hi


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
                        inference="hac", **kw)
    assert res.pvalue == res.extra["p_hac"] and res.extra["method"] == "hac"
    lo, hi = res.extra["ci"]
    assert np.isclose(0.5 * (lo + hi), res.statistic)
    assert np.isclose(hi - lo, 2 * 1.959964 * res.extra["se_hac"], rtol=1e-4)
    assert "ci_boot" in res.extra and "p_boot" in res.extra
    # None follows the base test's own inference setting.
    res = relative_test(Rn, G, Rp, CriticTest(n_steps=50, n_perm=20, inference="hac"), rng=rng,
                        n_boot=100, **kw)
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
    # Time-block split for the learned version (serially dependent windows).
    res = relative_test(Rn, G, Rp, CriticTest(n_steps=30, n_perm=20, time_blocks=4,
                                               inference="hac"), rng=rng, n_boot=50, **kw)
    assert res.extra["split"] == "time_block" and res.extra["n_eval_new"] % 50 == 0
    with pytest.raises(ValueError):
        relative_test(Rn, G, Rp, CriticTest(n_steps=30, time_blocks=4), rng=rng)


def test_result_fields_string_base_and_hac():
    rng = np.random.default_rng(4)
    n = 200
    Rn, G, Rp = samples(rng, n=n, L=10)
    order = np.arange(n)
    res = relative_test(Rn, G, Rp, CriticTest(n_steps=100, n_perm=50), rng=rng, n_boot=200,
                        order_new=order, order_g=order, order_past=order)
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
                        groups_past=g)
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
def test_boundary_size_critic_slow():
    rng = np.random.default_rng(8)
    reps, n, d = 40, 400, 10
    shift = np.zeros(d)
    shift[0] = 0.4
    base = CriticTest(n_steps=200, n_perm=20)
    rej = 0
    for _ in range(reps):
        Rn = rng.standard_normal((n, d))
        res = relative_test(Rn, rng.standard_normal((n, d)) + shift,
                            rng.standard_normal((n, d)) - shift, base, rng=rng, n_boot=300)
        rej += res.pvalue <= 0.05
    assert rej <= max_rejections(reps)
