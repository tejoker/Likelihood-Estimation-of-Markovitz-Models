"""Tests for the shared real-data pipeline (synthval.pipeline)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from synthval import config, pipeline as pl
from synthval.models import MULTIVARIATE_MODELS, get_model, get_mv_model

TINY_LEARNED = {"n_steps": 40, "n_perm": 50}
TINY_TEST_KW = {"critic": TINY_LEARNED, "lrcritic": TINY_LEARNED, "c2st": TINY_LEARNED,
                "mmd": {"n_perm": 50}, "sigmmd": {"n_perm": 50}}
HAS_ANCHORED = pl.relative_method_available("anchored")
REL_METHODS = ("anchored", "two_critic") if HAS_ANCHORED else ("two_critic",)


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _garch_panel(seed: int = 0, n: int = 3, t_train: int = 800, t_test: int = 600):
    rng = np.random.default_rng(seed)
    g = get_model("GARCHt", mu=2e-4, omega=2e-6, alpha=0.08, beta=0.9, nu=6.0)
    cols = [f"T{i}" for i in range(n)]
    vol = np.linspace(0.5, 2.0, n)  # heterogeneous ticker scales
    tr = pd.DataFrame(g.simulate(t_train, n, rng=rng).T * vol, columns=cols)
    te = pd.DataFrame(g.simulate(t_test, n, rng=rng).T * vol, columns=cols)
    return tr, te


@pytest.fixture(scope="module")
def small_panel():
    tr, te = _garch_panel()
    return pl.RealPanel(train=tr, test=te, scale=tr.std(ddof=1))


@pytest.fixture(scope="module")
def garch_fits(small_panel):
    return pl.fit_panel("GARCHt", small_panel.train, n_workers=1)


@pytest.fixture(scope="module")
def three_samples(small_panel, garch_fits):
    fitted, _ = garch_fits
    sim = pl.simulate_panel(fitted, small_panel.n_test, seed=11)
    Xn, _, on = small_panel.windows("test")
    Y, _, oy = small_panel.synthetic_windows(sim, "test")
    Xp, _, op = small_panel.windows("train")
    return Xn, on, Y, oy, Xp, op


# ----------------------------------------------------------------------------------------------
# Windows and panel
# ----------------------------------------------------------------------------------------------
def test_panel_windows_shapes_and_orders():
    T, N, L = 105, 3, 20
    a = np.arange(T * N, dtype=float).reshape(T, N) + 1.0
    scale = np.array([1.0, 2.0, 4.0])
    W, g, o = pl.panel_windows(a, L, scale=scale, date_offset=7)
    n = (T - L) // L + 1
    assert W.shape == (N * n, L) and g.shape == o.shape == (N * n,)
    assert g.dtype == np.int64 and o.dtype == np.int64
    np.testing.assert_array_equal(g, np.repeat(np.arange(N), n))
    np.testing.assert_array_equal(o, np.tile(7 + L * np.arange(n), N))
    # window k of ticker j = rows k*L .. k*L+L-1 of column j, divided by its scale
    np.testing.assert_allclose(W[n + 2], a[2 * L:3 * L, 1] / 2.0)
    # overlapping windows: the order is still the date rank of the first day
    W2, _, o2 = pl.panel_windows(a, L, step=5, scale=None)
    assert W2.shape[0] == N * ((T - L) // 5 + 1)
    np.testing.assert_array_equal(o2[:4], [0, 5, 10, 15])


def test_panel_windows_scale_invariance_and_alignment():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.standard_normal((200, 4)) * 0.01, columns=list("ABCD"))
    c = pd.Series([0.5, 3.0, 10.0, 0.1], index=list("ABCD"))
    W1, g1, o1 = pl.panel_windows(df, 20, scale=df.std())
    W2, g2, o2 = pl.panel_windows(df * c, 20, scale=(df * c).std())
    np.testing.assert_allclose(W1, W2, rtol=1e-12)
    np.testing.assert_array_equal(o1, o2)
    # a Series scale is aligned on the DataFrame columns, not used by position
    shuffled = df.std().loc[["D", "B", "A", "C"]]
    W3, _, _ = pl.panel_windows(df, 20, scale=shuffled)
    np.testing.assert_allclose(W1, W3)
    # same constant scale for real and synthetic: ndarray input uses the Series order
    W4, _, _ = pl.panel_windows(df.to_numpy(), 20, scale=df.std())
    np.testing.assert_allclose(W1, W4)
    with pytest.raises(ValueError):
        pl.panel_windows(df, 20, scale=np.ones(3))
    with pytest.raises(KeyError):
        pl.panel_windows(df, 20, scale=pd.Series(1.0, index=list("ABC")))
    with pytest.raises(ValueError):
        pl.panel_windows(df, 20, scale=None, date_offset=-1)
    with pytest.raises(TypeError):
        pl.panel_windows(df, 20)  # scale is required (pass None explicitly)


def test_real_panel_train_test_orders(small_panel, garch_fits):
    Xp, gp, op = small_panel.windows("train")
    Xn, gn, on = small_panel.windows("test")
    assert small_panel.test_offset == small_panel.n_train
    assert op.min() == 0 and on.min() == small_panel.n_train
    assert op.max() < on.min()  # TRAIN and TEST ranks are distinct, TEST follows TRAIN
    assert not set(op) & set(on)
    assert on.max() + small_panel.L <= small_panel.n_train + small_panel.n_test
    # real windows are normalised by the TRAIN std of each ticker
    j = 2
    np.testing.assert_allclose(Xn[gn == j][0],
                               small_panel.test.iloc[:20, j] / small_panel.scale.iloc[j])
    np.testing.assert_allclose(np.std(Xp, ddof=0), 1.0, atol=0.15)
    # synthetic windows over the TEST horizon carry exactly the real TEST ranks and groups
    fitted, _ = garch_fits
    sim = pl.simulate_panel(fitted, small_panel.n_test, seed=0)
    Y, gy, oy = small_panel.synthetic_windows(sim, "test")
    np.testing.assert_array_equal(oy, on)
    np.testing.assert_array_equal(gy, gn)
    np.testing.assert_allclose(Y[gy == j][0], sim[:20, j] / small_panel.scale.iloc[j])
    with pytest.raises(ValueError):
        small_panel.synthetic_windows(sim[:, :2], "test")


@pytest.mark.skipif(not config.DATA_CSV.exists(), reason="data file not available")
def test_load_real_panel_default():
    panel = pl.load_real_panel()
    assert panel.n_tickers == config.N_PANEL
    assert (panel.n_train, panel.n_test) == (1258, 615)
    Xn, gn, on = panel.windows("test")
    Xp, gp, op = panel.windows("train")
    assert Xn.shape == (100 * 30, 20) and Xp.shape == (100 * 62, 20)
    assert op.max() < on.min() == panel.n_train
    np.testing.assert_allclose(panel.scale.to_numpy(), panel.train.std(ddof=1).to_numpy())


# ----------------------------------------------------------------------------------------------
# Fits and simulation
# ----------------------------------------------------------------------------------------------
def test_fit_panel_table(small_panel, garch_fits):
    fitted, table = garch_fits
    assert list(fitted) == small_panel.tickers
    assert list(table.index) == small_panel.tickers and table.index.name == "ticker"
    for col in ("mu", "omega", "alpha", "beta", "nu", "at_bound", "n_at_bound", "fit_seconds",
                "persistence"):
        assert col in table.columns
    assert table.attrs["model"] == "GARCHt"
    assert (table["fit_seconds"] > 0).all()
    assert table["n_at_bound"].dtype.kind in "iu"
    p0 = fitted[small_panel.tickers[0]].params()
    assert table.iloc[0]["alpha"] == pytest.approx(p0["alpha"])
    with pytest.raises(ValueError):
        pl.fit_panel(MULTIVARIATE_MODELS[0], small_panel.train)


def test_fit_panel_parallel_matches_serial(small_panel):
    tr = small_panel.train
    f1, t1 = pl.fit_panel("Bootstrap", tr, n_workers=1)
    f2, t2 = pl.fit_panel("Bootstrap", tr, n_workers=2)
    pd.testing.assert_series_equal(t1["block_length"], t2["block_length"])
    s1 = pl.simulate_panel(f1, 50, seed=3)
    s2 = pl.simulate_panel(f2, 50, seed=3)
    np.testing.assert_array_equal(s1, s2)


def test_simulate_panel_deterministic(small_panel, garch_fits):
    fitted, _ = garch_fits
    a = pl.simulate_panel(fitted, 300, seed=5)
    b = pl.simulate_panel(fitted, 300, seed=5)
    c = pl.simulate_panel(fitted, 300, seed=6)
    assert a.shape == (300, small_panel.n_tickers) and a.dtype == np.float64
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    # the path of ticker j depends only on the seed and its position
    first_two = dict(list(fitted.items())[:2])
    np.testing.assert_array_equal(pl.simulate_panel(first_two, 300, seed=5), a[:, :2])
    # stationary start (with burn-in) differs from the end-of-TRAIN start, both deterministic
    d = pl.simulate_panel(fitted, 300, seed=5, start_from_end=False)
    np.testing.assert_array_equal(d, pl.simulate_panel(fitted, 300, seed=5,
                                                       start_from_end=False))
    assert not np.allclose(a, d)
    # models without an end state ignore start_from_end
    gbm, _ = pl.fit_panel("GBM", small_panel.train, n_workers=1)
    np.testing.assert_array_equal(pl.simulate_panel(gbm, 100, seed=1),
                                  pl.simulate_panel(gbm, 100, seed=1, start_from_end=False))


def test_simulate_panel_multivariate(small_panel):
    name = "FactorGARCHt" if "FactorGARCHt" in MULTIVARIATE_MODELS else "MVGaussian"
    mv = get_mv_model(name).fit(small_panel.train.to_numpy())
    a = pl.simulate_panel(mv, 120, seed=2)
    assert a.shape == (120, small_panel.n_tickers)
    np.testing.assert_array_equal(a, pl.simulate_panel(mv, 120, seed=2))
    assert not np.allclose(a, pl.simulate_panel(mv, 120, seed=3))


# ----------------------------------------------------------------------------------------------
# Tests under the two protocols
# ----------------------------------------------------------------------------------------------
def test_make_test_protocol_settings():
    t = pl.make_test("critic", "dependent")
    assert (t.time_blocks, t.inference) == (pl.TIME_BLOCKS, "hac")
    assert (t.n_steps, t.lr, t.batch_size, t.betas, t.gp_lambda) == (1000, 1e-3, 128,
                                                                     (0.5, 0.9), 10.0)
    t = pl.make_test("critic", "iid")
    assert (t.time_blocks, t.inference) == (None, "permutation")
    assert pl.make_test("lrcritic", "dependent").statistic == "logit"
    assert pl.make_test("mmd", "dependent").inference == "wild"
    assert pl.make_test("sigmmd", "iid").inference == "permutation"
    assert pl.make_test("mmd", "iid").n_perm == pl.N_PERM_KERNEL
    t = pl.make_test("critic", "dependent", time_blocks=4, lr=1e-4)  # E5 overrides win
    assert (t.time_blocks, t.lr, t.inference) == (4, 1e-4, "hac")
    with pytest.raises(ValueError):
        pl.make_test("critic", "nope")
    with pytest.raises(KeyError):
        pl.make_test("nope")


def _check_pvalue(p):
    assert 0.0 < p <= 1.0


@pytest.mark.parametrize("protocol", ["dependent", "iid"])
def test_absolute_tests_protocols(three_samples, protocol):
    Xn, on, Y, oy, _, _ = three_samples
    rows = pl.absolute_tests(Xn, Y, order_x=on, order_y=oy, protocol=protocol,
                             rng=np.random.default_rng(0), test_kw=TINY_TEST_KW)
    by = {r["test"]: r for r in rows}
    assert list(by) == list(pl.ABSOLUTE_TESTS)
    for r in rows:
        _check_pvalue(r["pvalue"])
        assert r["protocol"] == protocol and r["runtime_s"] >= 0
        assert (r["n_x"], r["n_y"]) == (Xn.shape[0], Y.shape[0])
    for name in pl.LEARNED_TESTS:
        r = by[name]
        _check_pvalue(r["p_perm"])
        _check_pvalue(r["p_hac"])  # orders given: HAC p-value under both protocols
        assert np.isfinite(r["z"]) and np.isfinite(r["z_hac"]) and r["valid"]
        if protocol == "dependent":
            assert r["pvalue_type"] == "hac" and r["pvalue"] == r["p_hac"]
            assert r["split"] == "time_block"
            assert r["n_eval_x"] < Xn.shape[0]
        else:
            assert r["pvalue_type"] == "permutation" and r["pvalue"] == r["p_perm"]
            assert r["split"] == "random"
    for name in pl.KERNEL_TESTS:
        r = by[name]
        _check_pvalue(r["p_perm"])
        _check_pvalue(r["p_wild"])
        assert math.isnan(r["p_hac"]) and r["valid"]
        if protocol == "dependent":
            assert r["pvalue_type"] == "wild" and r["pvalue"] == r["p_wild"]
        else:
            assert r["pvalue_type"] == "permutation" and r["pvalue"] == r["p_perm"]
        assert np.isfinite(r["wild_block"])
    ks = by["ks"]
    assert ks["pvalue_type"] == "asymptotic_iid"
    assert ks["valid"] is (protocol == "iid")  # KS flagged invalid under dependence


def test_absolute_tests_options(three_samples):
    Xn, on, Y, oy, _, _ = three_samples
    # each test has its own random stream: its result does not depend on the other tests run
    a = pl.absolute_tests(Xn, Y, order_x=on, order_y=oy, tests=("mmd",),
                          rng=np.random.default_rng(4), test_kw=TINY_TEST_KW)[0]
    b = pl.absolute_tests(Xn, Y, order_x=on, order_y=oy, tests=("ks", "mmd"),
                          rng=np.random.default_rng(4), test_kw=TINY_TEST_KW)[1]
    assert a["test"] == b["test"] == "mmd"
    assert (a["statistic"], a["p_wild"], a["p_perm"]) == (b["statistic"], b["p_wild"],
                                                          b["p_perm"])
    # without the second kernel call only the protocol's p-value is computed
    c = pl.absolute_tests(Xn, Y, order_x=on, order_y=oy, tests=("mmd",), all_pvalues=False,
                          rng=np.random.default_rng(4), test_kw=TINY_TEST_KW)[0]
    assert c["p_wild"] == a["p_wild"] and math.isnan(c["p_perm"])
    d = pl.absolute_tests(Xn, Y, order_x=on, order_y=oy, tests=("critic",), keep_extra=True,
                          rng=np.random.default_rng(4), test_kw=TINY_TEST_KW)[0]
    assert "train_curve" in d["extra"]
    with pytest.raises(ValueError):
        pl.absolute_tests(Xn, Y, tests=("ks",), rng=np.random.default_rng(0))
    with pytest.raises(KeyError):
        pl.absolute_tests(Xn, Y, order_x=on, order_y=oy, tests=("nope",),
                          rng=np.random.default_rng(0))
    with pytest.raises(KeyError):
        pl.absolute_tests(Xn, Y, order_x=on, order_y=oy, tests=("ks",),
                          rng=np.random.default_rng(0), test_kw={"nope": {}})


@pytest.mark.parametrize("protocol", ["dependent", "iid"])
def test_relative_tests_protocols(three_samples, protocol):
    Xn, on, Y, oy, Xp, op = three_samples
    rows = pl.relative_tests(Xn, Y, Xp, orders=(on, oy, op), methods=REL_METHODS,
                             protocol=protocol, rng=np.random.default_rng(1),
                             base_kw=TINY_LEARNED, n_boot=100)
    assert [r["method"] for r in rows] == list(REL_METHODS)
    for r in rows:
        _check_pvalue(r["pvalue"])
        assert r["base"] == "critic" and r["protocol"] == protocol
        assert (r["n_new"], r["n_g"], r["n_past"]) == (Xn.shape[0], Y.shape[0], Xp.shape[0])
        assert r["ci_lo"] <= r["statistic"] <= r["ci_hi"]
        assert r["statistic"] == pytest.approx(r["D_g"] - r["D_past"], abs=1e-9)
        _check_pvalue(r["p_hac"])
        if protocol == "dependent":
            assert r["pvalue_type"] == "hac" and r["pvalue"] == r["p_hac"]
        elif r["method"] == "anchored":
            assert r["pvalue_type"] == "permutation" and r["pvalue"] == r["p_perm"]
        else:
            assert r["pvalue_type"] == "bootstrap" and r["pvalue"] == r["p_boot"]
        assert r["valid"] is (r["method"] == "anchored")
    # kernel bases have a single (Bounliphone) version whatever `methods` says
    k = pl.relative_tests(Xn, Y, Xp, orders=(on, oy, op), base="mmd", protocol=protocol,
                          rng=np.random.default_rng(1))
    assert len(k) == 1 and k[0]["base"] == "mmd"
    _check_pvalue(k[0]["pvalue"])
    assert k[0]["pvalue_type"] == ("hac" if protocol == "dependent" else "normal")


def test_relative_tests_checks(three_samples):
    Xn, on, Y, oy, Xp, op = three_samples
    with pytest.raises(ValueError):
        pl.relative_tests(Xn, Y, Xp, rng=np.random.default_rng(0))  # dependent needs orders
    with pytest.raises(ValueError):
        pl.relative_tests(Xn, Y, Xp, orders=(on, oy, op), methods=("nope",),
                          rng=np.random.default_rng(0))


def test_evaluate_model_rows(small_panel, garch_fits):
    fitted, _ = garch_fits
    rows = pl.evaluate_model(small_panel, fitted, seed=3, tests=("critic", "ks"),
                             relative_methods=REL_METHODS[:1], test_kw=TINY_TEST_KW,
                             base_kw=TINY_LEARNED)
    assert [(r["kind"], r["test"]) for r in rows] == [
        ("absolute", "critic"), ("absolute", "ks"), ("relative", f"relative_{REL_METHODS[0]}")]
    assert all(r["seed"] == 3 for r in rows)
    again = pl.evaluate_model(small_panel, fitted, seed=3, tests=("ks",), relative_base=None)
    assert again[0]["statistic"] == rows[1]["statistic"]  # same synthetic paths for one seed


# ----------------------------------------------------------------------------------------------
# Decision rule
# ----------------------------------------------------------------------------------------------
def test_verdict_logic():
    assert pl.verdict(0.01, 0.95) == "BAD"
    assert pl.verdict(0.05, 0.95) == "BAD"  # p <= alpha rejects
    assert pl.verdict(0.01, 0.10) == "BAD"  # rejection does not need power
    assert pl.verdict(0.30, 0.80) == "GOOD"
    assert pl.verdict(0.30, 0.79) == "INCONCLUSIVE"
    assert pl.verdict(0.30, None) == "INCONCLUSIVE"
    assert pl.verdict(0.30, float("nan")) == "INCONCLUSIVE"
    assert pl.verdict(0.08, 0.9, alpha=0.10) == "BAD"
    assert pl.verdict(0.30, 0.85, power_min=0.9) == "INCONCLUSIVE"
    with pytest.raises(ValueError):
        pl.verdict(float("nan"), 0.9)


def test_binomial_intervals_and_median_pvalue():
    lo, hi = pl.wilson_ci(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    lo, hi = pl.wilson_ci(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1611, abs=1e-4)
    assert pl.wilson_ci(20, 20)[1] == 1.0
    assert pl.wilson_ci(0, 3)[0] == 0.0 and pl.wilson_ci(3, 3)[1] == 1.0
    assert pl.clopper_pearson_lower(0, 10) == 0.0
    assert pl.clopper_pearson_lower(10, 10) == pytest.approx(0.05 ** 0.1)
    assert pl.clopper_pearson_lower(40, 50) < 40 / 50
    assert pl.median_pvalue([0.01, 0.2, 0.3]) == pytest.approx(0.4)
    assert pl.median_pvalue([0.7, 0.9]) == 1.0
    with pytest.raises(ValueError):
        pl.wilson_ci(11, 10)


def test_power_check_tiny(small_panel, garch_fits):
    fitted, _ = garch_fits
    kw = dict(n_reps=2, seed=7, n_workers=1, base_kw={"n_steps": 30, "n_perm": 30},
              method=REL_METHODS[0])
    res = pl.power_check(small_panel.train, small_panel.test, fitted=fitted, boundary=True, **kw)
    assert res.n_reps == 2 and res.pvalues.shape == (2,)
    assert np.all((res.pvalues > 0) & (res.pvalues <= 1))
    assert res.power == res.n_reject / 2
    assert 0.0 <= res.ci[0] <= res.power <= res.ci[1] <= 1.0
    assert 0.0 <= res.cp_lower <= res.power
    assert res.vol_factor == pl.POWER_VOL_FACTOR and res.method == REL_METHODS[0]
    assert {"rep", "pvalue", "statistic", "p_hac", "boundary_pvalue"} <= set(res.reps.columns)
    assert np.isfinite(res.boundary_size)
    assert set(res.summary()) >= {"power", "ci_lo", "ci_hi", "cp_lower", "boundary_size"}
    assert pl.verdict(0.5, res.power) in ("GOOD", "INCONCLUSIVE")
    # deterministic given the seed; fits done inside when `fitted` is not given
    res2 = pl.power_check(small_panel.train, small_panel.test, **kw)
    np.testing.assert_allclose(res2.pvalues, res.pvalues)
