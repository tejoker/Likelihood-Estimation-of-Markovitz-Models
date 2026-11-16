"""Tests for the QuantGAN-style neural generator (synthval.models.neural) and its pipeline path."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from synthval import pipeline as pl
from synthval.models import PANEL_MODELS, UNIVARIATE_MODELS, get_panel_model
from synthval.models.neural import (TCN, TINY_KW, QuantGAN, QuantGANSeries, fit_lambertw,
                                    lambertw_gaussianize, lambertw_heavytail, lambertw_logpdf)

#: Tiny TCN that learns the scale within a couple of seconds on one core (see test_scale).
SCALE_KW = {**TINY_KW, "n_steps": 150, "lr_g": 2e-3, "lr_d": 2e-3}
TINY_LEARNED = {"n_steps": 30, "n_perm": 50}


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _panel_returns(seed: int = 0, T: int = 300, vols=(0.5, 1.0, 2.0)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = 0.01 * rng.standard_t(5, size=(T, len(vols))) * np.asarray(vols) + 3e-4
    return pd.DataFrame(a, columns=[f"T{i}" for i in range(len(vols))])


@pytest.fixture(scope="module")
def tiny_fit():
    R = _panel_returns()
    return R, QuantGAN(**TINY_KW).fit(R.to_numpy(), rng=np.random.default_rng(1))


@pytest.fixture(scope="module")
def scale_fit():
    R = _panel_returns(seed=5)
    return R, pl.fit_panel("QuantGAN", R, fit_seed=3, **SCALE_KW)


# ----------------------------------------------------------------------------------------------
# Lambert W x Gaussian
# ----------------------------------------------------------------------------------------------
def test_lambertw_roundtrip_and_density():
    rng = np.random.default_rng(0)
    u = rng.standard_normal(1000)
    y = lambertw_heavytail(u, 0.3, 0.2, 1.7)
    np.testing.assert_allclose(lambertw_gaussianize(y, 0.3, 0.2, 1.7), u, atol=1e-10)
    np.testing.assert_allclose(lambertw_gaussianize(y, 0.0), y)  # delta = 0: identity
    grid = np.linspace(-200, 200, 400_001)
    mass = np.trapezoid(np.exp(lambertw_logpdf(grid, 0.2, 1.7, 0.3)), grid)
    assert mass == pytest.approx(1.0, abs=2e-3)  # heavy tail: a little mass beyond the grid


def test_fit_lambertw_recovers_parameters():
    rng = np.random.default_rng(1)
    y = lambertw_heavytail(rng.standard_normal(20_000), 0.15, -0.1, 0.8)
    f = fit_lambertw(y)
    assert f["success"] and f["at_bound"] == []
    assert f["delta"] == pytest.approx(0.15, abs=0.02)
    assert f["mu"] == pytest.approx(-0.1, abs=0.03)
    assert f["sigma"] == pytest.approx(0.8, abs=0.03)
    g = fit_lambertw(rng.standard_normal(20_000))  # Gaussian data: delta near 0
    assert g["delta"] < 0.02


# ----------------------------------------------------------------------------------------------
# TCN
# ----------------------------------------------------------------------------------------------
def test_tcn_receptive_field_shapes_and_causality():
    assert TCN(3, 1, 8, 7).receptive_field == 127  # WKKK Table 3
    torch.manual_seed(0)
    net = TCN(3, 1, 8, 4)  # receptive field 1 + 2 * 7 = 15
    rf = net.receptive_field
    assert rf == 15
    x = torch.randn(2, 3, 40)
    y = net(x)
    assert y.shape == (2, 1, 40 - rf + 1)
    x2 = x.clone()
    x2[:, :, 20] += 5.0  # input 20 feeds outputs j with j <= 20 <= j + rf - 1
    d = (net(x2) - y).abs().amax(dim=(0, 1)).detach().numpy()
    changed = np.flatnonzero(d > 0)
    assert changed.min() == 20 - rf + 1 and changed.max() == 20
    with pytest.raises(ValueError):
        net(torch.randn(1, 3, rf - 1))


# ----------------------------------------------------------------------------------------------
# QuantGAN
# ----------------------------------------------------------------------------------------------
def test_registry():
    assert "QuantGAN" in PANEL_MODELS and "QuantGAN" not in UNIVARIATE_MODELS
    m = get_panel_model("QuantGAN", **TINY_KW)
    assert isinstance(m, QuantGAN) and m.receptive_field == 7


def test_preprocessing_roundtrip(tiny_fit):
    R, _ = tiny_fit
    a = R.to_numpy()
    m2 = QuantGAN(**TINY_KW)
    X = m2._preprocess(a)
    assert X.shape == a.shape
    assert X.mean() == pytest.approx(0.0, abs=1e-12) and X.std() == pytest.approx(1.0)
    for j in range(a.shape[1]):
        np.testing.assert_allclose(m2.postprocess(X[:, j], series=j), a[:, j], rtol=1e-9,
                                   atol=1e-12)
    np.testing.assert_allclose(m2.loc_, a.mean(axis=0))
    np.testing.assert_allclose(m2.scale_, a.std(axis=0, ddof=1))
    assert m2.lw_["delta"] > 0.02  # Student-t(5) data: heavier tails than Gaussian


def test_simulate_shapes_and_errors(tiny_fit):
    R, m = tiny_fit
    s = m.simulate(50, 4, rng=np.random.default_rng(2), series=1)
    assert s.shape == (4, 50) and np.all(np.isfinite(s))
    z = m.simulate_standardised(33, 2, rng=np.random.default_rng(2))
    assert z.shape == (2, 33)
    with pytest.raises(ValueError):
        m.simulate(10, 1, rng=np.random.default_rng(0))  # panel fit needs series=
    single = QuantGAN(**TINY_KW).fit(R["T0"].to_numpy(), rng=np.random.default_rng(0))
    assert single.simulate(10, 3, rng=np.random.default_rng(0)).shape == (3, 10)
    with pytest.raises(ValueError):
        QuantGAN(**TINY_KW).fit(R.to_numpy()[:7])  # fewer than receptive_field + 1 days
    assert "curve" in m.fit_info_ and len(m.fit_info_["curve"]["step"]) == 4  # 20 / 5


def test_determinism(tiny_fit):
    R, m = tiny_fit
    m_same = QuantGAN(**TINY_KW).fit(R.to_numpy(), rng=np.random.default_rng(1))
    for k, v in m._g_state.items():
        assert torch.equal(v, m_same._g_state[k])
    a = m.simulate(40, 3, rng=np.random.default_rng(9), series=2)
    b = m_same.simulate(40, 3, rng=np.random.default_rng(9), series=2)
    np.testing.assert_array_equal(a, b)
    c = m.simulate(40, 3, rng=np.random.default_rng(10), series=2)
    assert not np.allclose(a, c)
    m_other = QuantGAN(**TINY_KW).fit(R.to_numpy(), rng=np.random.default_rng(2))
    assert not np.allclose(m_other.simulate(40, 3, rng=np.random.default_rng(9), series=2), a)
    # default rng: the constructor seed
    d1 = QuantGAN(**TINY_KW, seed=4).fit(R.to_numpy())
    d2 = QuantGAN(**TINY_KW, seed=4).fit(R.to_numpy())
    np.testing.assert_array_equal(d1.simulate_standardised(20, 1, rng=np.random.default_rng(0)),
                                  d2.simulate_standardised(20, 1, rng=np.random.default_rng(0)))


def test_gan_r1_loss_runs():
    R = _panel_returns(T=120)
    m = QuantGAN(**{**TINY_KW, "n_steps": 5, "loss": "gan-r1"}).fit(
        R.to_numpy(), rng=np.random.default_rng(0))
    assert np.all(np.isfinite(m.simulate(20, 2, rng=np.random.default_rng(1), series=0)))
    with pytest.raises(ValueError):
        QuantGAN(loss="vanilla")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------------
# Pipeline integration
# ----------------------------------------------------------------------------------------------
def test_fit_panel_and_simulate_panel(scale_fit):
    R, (fitted, table) = scale_fit
    assert list(fitted) == list(R.columns)
    assert all(isinstance(v, QuantGANSeries) for v in fitted.values())
    core = fitted["T0"].core
    assert all(v.core is core for v in fitted.values())  # one shared generator
    assert list(table.index) == list(R.columns) and table.attrs["pooled"]
    np.testing.assert_allclose(table["std"].to_numpy(), R.std(ddof=1).to_numpy())
    assert {"lw_delta", "final_wdist", "fit_seconds", "at_bound", "device"} <= set(table.columns)
    sim = pl.simulate_panel(fitted, 400, seed=7)
    assert sim.shape == (400, 3) and np.all(np.isfinite(sim))
    np.testing.assert_array_equal(sim, pl.simulate_panel(fitted, 400, seed=7))
    assert not np.allclose(sim, pl.simulate_panel(fitted, 400, seed=8))
    # roughly the right scale per ticker after a short training (GAN: factor-of-3 band)
    ratio = sim.std(axis=0, ddof=1) / R.std(ddof=1).to_numpy()
    assert np.all((ratio > 1 / 3) & (ratio < 3)), ratio
    # ticker scales are ordered like the real ones (per-ticker rescaling)
    assert np.all(np.diff(sim.std(axis=0)) > 0)


def test_fit_panel_seed_reproducible():
    R = _panel_returns(T=150)
    f1, t1 = pl.fit_panel("QuantGAN", R, fit_seed=11, **TINY_KW)
    f2, _ = pl.fit_panel("QuantGAN", R, fit_seed=11, **TINY_KW)
    np.testing.assert_array_equal(pl.simulate_panel(f1, 60, seed=1),
                                  pl.simulate_panel(f2, 60, seed=1))


def test_pickle_roundtrip(scale_fit):
    _, (fitted, _) = scale_fit
    sim = pl.simulate_panel(fitted, 120, seed=3)
    back = pickle.loads(pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL))
    assert back["T0"].core is back["T2"].core
    np.testing.assert_array_equal(pl.simulate_panel(back, 120, seed=3), sim)


def test_evaluate_model_runs(scale_fit):
    R, (fitted, _) = scale_fit
    test = _panel_returns(seed=6, T=200)
    panel = pl.RealPanel(train=R, test=test, scale=R.std(ddof=1))
    rows = pl.evaluate_model(panel, fitted, seed=1, tests=("critic",), relative_base="critic",
                             relative_methods=("anchored",), test_kw={"critic": TINY_LEARNED},
                             base_kw=TINY_LEARNED, n_boot=20)
    assert [r["test"] for r in rows] == ["critic", "relative_anchored"]
    assert all(0.0 <= r["pvalue"] <= 1.0 for r in rows)
