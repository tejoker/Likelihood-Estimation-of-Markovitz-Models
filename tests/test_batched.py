"""Tests for synthval.batched: critic banks against the sequential critic tests.

(a) the bank draws exactly the sequential random streams (initial weights, mini-batch indices,
interpolation weights) per problem; (b) ``impl="loop"`` reproduces the sequential results bit for
bit, and the grouped bank agrees up to the rounding that training amplifies (compared with the
effect of changing the number of CPU threads in the sequential code); (c) batched permutation
p-values are uniform under H0 / at the anchored boundary.
"""
from __future__ import annotations

import copy
import functools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

import synthval.batched.critic as bcrit
import synthval.twosample.critic as crit
from synthval import pipeline as pl
from synthval.batched import (AnchoredJob, MLPCriticBank, PooledConvCriticBank, RelativeProblem,
                              TwoSampleProblem, WitnessProblem, anchored_relative_tests,
                              batched_enabled, batched_key, build_bank, critic_tests,
                              fit_witnesses, relative_tests_anchored)
from synthval.batched.critic import bank_gp_terms
from synthval.models import get_model
from synthval.twosample import C2ST, CriticTest, build_network
from synthval.twosample.relative import relative_test_anchored

ROOT = Path(__file__).resolve().parents[1]
#: Fields of a result's ``extra`` that the batched path adds or measures differently.
BANK_FIELDS = {"batched", "bank_size", "bank_impl", "bank_train_time_s", "train_time_s"}


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _windows(seed: int, n: int, L: int = 10, scale: float = 1.0) -> np.ndarray:
    return scale * np.random.default_rng(seed).standard_normal((n, L))


def _strip(d: dict) -> dict:
    """``extra`` without the bank fields and timings (recursively in ``train_info``)."""
    out = {}
    for k, v in d.items():
        if k in BANK_FIELDS:
            continue
        if isinstance(v, dict):
            v = _strip(v)
        out[k] = v
    return out


def _assert_same(a, b) -> None:
    """Equal, NaN-aware, for nested dicts / lists / tuples / floats."""
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b)
    else:
        assert a == b


# ----------------------------------------------------------------------------------------------
# Banks
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape,arch,kw", [
    ((12,), "pooled", {}),
    ((3, 12), "pooled", {"layer_norm": True}),
    ((12,), "mlp", {"layer_norm": True}),
    ((2, 6), "mlp", {}),
])
def test_bank_forward_matches_single_networks(shape, arch, kw):
    torch.manual_seed(0)
    nets = [build_network(shape, arch, hidden=8, **kw) for _ in range(3)]
    x = torch.randn(17, 3, *shape)
    for impl in ("grouped", "loop"):
        bank = build_bank(nets, impl)
        out = bank(x)
        assert out.shape == (3, 17)
        for k, net in enumerate(nets):
            ref = net(x[:, k].contiguous())
            if impl == "loop":
                assert torch.equal(out[k], ref)
            else:
                torch.testing.assert_close(out[k], ref, rtol=1e-5, atol=1e-6)
    # store / load round trip
    bank = build_bank(nets)
    clone = copy.deepcopy(nets[1])
    for p in clone.parameters():
        p.data.zero_()
    bank.store_critic(1, clone)
    for p, q in zip(clone.parameters(), nets[1].parameters()):
        assert torch.equal(p, q)


@pytest.mark.parametrize("arch", ["pooled", "mlp"])
def test_bank_gradients_are_separated(arch):
    """The gradient of critic k's WGAN-GP loss is zero outside slice k of every bank parameter
    and equals the single network's gradient inside it."""
    torch.manual_seed(1)
    nets = [build_network((10,), arch, hidden=8) for _ in range(3)]
    bank = build_bank(nets)
    xb, yb, eps = torch.randn(9, 3, 10), torch.randn(9, 3, 10), torch.rand(9, 3)
    wd, gp = bank_gp_terms(bank, xb, yb, eps)
    grads = dict(zip([id(p) for p in bank.parameters()],
                     torch.autograd.grad(-wd[1] + 10.0 * gp[1], list(bank.parameters()))))
    wd1, gp1, _ = CriticTest._gp_terms(nets[1], xb[:, 1], yb[:, 1], eps[:, 1], True)
    ref = torch.autograd.grad(-wd1 + 10.0 * gp1, list(nets[1].parameters()))
    ref_of = {id(p): r for p, r in zip(nets[1].parameters(), ref)}
    for bank_p, p in bank._pairs(nets[1]):
        g = grads[id(bank_p)]
        assert torch.count_nonzero(g[0]) == 0 and torch.count_nonzero(g[2]) == 0
        torch.testing.assert_close(g[1].reshape(p.shape), ref_of[id(p)], rtol=1e-4, atol=1e-6)


def test_bank_checks():
    torch.manual_seed(2)
    with pytest.raises(NotImplementedError):
        build_bank([build_network((10,), "conv", hidden=8)])
    with pytest.raises(ValueError):
        build_bank([build_network((10,), "pooled", hidden=8),
                    build_network((10,), "pooled", hidden=16)])
    with pytest.raises(TypeError):
        build_bank([build_network((10,), "pooled", hidden=8), build_network((10,), "mlp")])
    assert isinstance(build_bank([build_network((10,), "mlp")]), MLPCriticBank)
    assert isinstance(build_bank([build_network((10,), "pooled")]), PooledConvCriticBank)


def test_unsupported_bases_and_shared_generators():
    rng = np.random.default_rng(0)
    p = [WitnessProblem(_windows(0, 30), _windows(1, 30), np.random.default_rng(1))]
    with pytest.raises(TypeError):
        fit_witnesses(C2ST(n_steps=5), p)
    with pytest.raises(NotImplementedError):
        fit_witnesses(CriticTest(n_steps=5, early_stopping=True), p)
    shared = [WitnessProblem(_windows(0, 30), _windows(1, 30), rng) for _ in range(2)]
    with pytest.raises(ValueError):
        fit_witnesses(CriticTest(n_steps=5, arch="mlp"), shared)
    with pytest.raises(ValueError):
        critic_tests(CriticTest(n_steps=5), [TwoSampleProblem(_windows(0, 30), _windows(1, 30),
                                                               rng)] * 2)


def test_env_switch(monkeypatch):
    monkeypatch.delenv("SYNTHVAL_BATCHED", raising=False)
    assert not batched_enabled() and batched_key() == {}
    monkeypatch.setenv("SYNTHVAL_BATCHED", "1")
    assert batched_enabled() and batched_key()["batched"]["mode"] == "critic_bank"
    assert "code" in batched_key(code=True)["batched"]
    monkeypatch.setenv("SYNTHVAL_BATCHED", "maybe")
    with pytest.raises(ValueError):
        batched_enabled()


# ----------------------------------------------------------------------------------------------
# (a) random streams
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arch", ["pooled", "mlp"])
def test_rng_streams_and_initial_weights_match_sequential(arch, monkeypatch):
    """Per problem, the bank uses the sequential initial weights and draws exactly the
    sequential mini-batch indices (uniform and weighted) and interpolation weights."""
    base = CriticTest(n_steps=6, batch_size=16, arch=arch, hidden=8)
    data = [(_windows(0, 40), _windows(1, 55), None, None),
            (_windows(2, 33), _windows(3, 60), None, np.linspace(1.0, 3.0, 60)),
            (_windows(4, 50), _windows(5, 41), np.arange(1.0, 51.0), np.ones(41))]

    def problems():
        return [WitnessProblem(X, Y, np.random.default_rng(10 + i), wx, wy)
                for i, (X, Y, wx, wy) in enumerate(data)]

    # Sequential: record initial weights and every draw of CriticTest._train.
    seq_init, seq_idx, seq_eps = [], [], []
    real_build, real_draw, real_rand = crit.build_network, crit._draw_indices, torch.rand

    def build(*a, **k):
        net = real_build(*a, **k)
        seq_init.append(copy.deepcopy(net.state_dict()))
        return net

    def draw(*a, **k):
        out = real_draw(*a, **k)
        seq_idx.append(out.clone())
        return out

    def rand(*a, **k):
        out = real_rand(*a, **k)
        seq_eps.append(out.clone())
        return out

    with monkeypatch.context() as m:
        m.setattr(crit, "build_network", build)
        m.setattr(crit, "_draw_indices", draw)
        m.setattr(torch, "rand", rand)
        seq_states = []
        for p in problems():
            base.fit_witness(p.X_tr, p.Y_tr, rng=p.rng, weights_x=p.weights_x,
                             weights_y=p.weights_y)
            seq_states.append(p.rng.bit_generator.state)

    # Batched: record initial weights and the traced draws.
    bat_init = []
    real_bbuild = bcrit.build_network

    def bbuild(*a, **k):
        net = real_bbuild(*a, **k)
        bat_init.append(copy.deepcopy(net.state_dict()))
        return net

    trace: dict[int, list] = {}
    monkeypatch.setattr(bcrit, "build_network", bbuild)
    probs = problems()
    fit_witnesses(base, probs, chunk_steps=4,
                  trace=lambda k, step, ix, iy, e: trace.setdefault(k, []).append(
                      (step, ix.clone(), iy.clone(), e.clone())))
    assert [p.rng.bit_generator.state for p in probs] == seq_states
    T = base.n_steps
    for k in range(len(data)):
        for name in seq_init[k]:
            assert torch.equal(seq_init[k][name], bat_init[k][name])
        assert [t[0] for t in trace[k]] == list(range(1, T + 1))
        for s, (_, ix, iy, e) in enumerate(trace[k]):
            assert torch.equal(ix, seq_idx[2 * (k * T + s)])
            assert torch.equal(iy, seq_idx[2 * (k * T + s) + 1])
            assert torch.equal(e, seq_eps[k * T + s])


# ----------------------------------------------------------------------------------------------
# (b) numerical equivalence
# ----------------------------------------------------------------------------------------------
def _abs_problems(n_prob: int = 3) -> list[TwoSampleProblem]:
    return [TwoSampleProblem(_windows(20 + s, 90 + 7 * s), _windows(40 + s, 80, scale=1.1),
                             np.random.default_rng(60 + s)) for s in range(n_prob)]


def _rel_problems(n_prob: int = 3) -> list[RelativeProblem]:
    return [RelativeProblem(_windows(70 + s, 80, scale=1.2), _windows(80 + s, 70 + 5 * s),
                            _windows(90 + s, 100), np.random.default_rng(110 + s))
            for s in range(n_prob)]


def _seq_abs(base, probs):
    return [base(p.X, p.Y, rng=copy.deepcopy(p.rng)) for p in probs]


def _seq_rel(base, probs):
    return [relative_test_anchored(p.R_new, p.G, p.R_past, base, rng=copy.deepcopy(p.rng))
            for p in probs]


@pytest.mark.parametrize("arch", ["pooled", "mlp"])
def test_loop_bank_is_bit_identical_to_sequential(arch):
    """With the single networks' own operations the bank reproduces the sequential results
    exactly: same statistic, p-values and extra fields (absolute and weighted anchored)."""
    base = CriticTest(n_steps=25, batch_size=32, arch=arch, hidden=16, n_perm=99)
    for seq_fn, bat_fn, probs in ((_seq_abs, critic_tests, _abs_problems()),
                                  (_seq_rel, None, _rel_problems())):
        seq = seq_fn(base, probs)
        if bat_fn is None:
            bat = relative_tests_anchored(probs, base, impl="loop")
        else:
            bat = bat_fn(base, probs, impl="loop")
        for a, b in zip(seq, bat):
            assert a.statistic == b.statistic and a.pvalue == b.pvalue
            _assert_same(_strip(a.extra), _strip(b.extra))


@pytest.mark.parametrize("arch", ["pooled", "mlp"])
def test_grouped_bank_matches_sequential_up_to_rounding(arch):
    """The grouped bank equals the sequential path up to rounding amplified by training.
    Observed here (torch 2.15 CPU): |dstat| <= 1.2e-4 with se 0.07-0.22, the same order as
    running the sequential code on 4 CPU threads instead of 1 (<= 7.5e-5); identical p-values.
    Asserted with margin: |dstat| <= 0.02 se, |dp| <= 2 / (1 + B); all other fields equal."""
    base = CriticTest(n_steps=60, batch_size=32, arch=arch, hidden=16, n_perm=99)
    for kind, probs in (("abs", _abs_problems()), ("rel", _rel_problems())):
        if kind == "abs":
            seq, bat = _seq_abs(base, probs), critic_tests(base, probs)
        else:
            seq, bat = _seq_rel(base, probs), relative_tests_anchored(probs, base)
        for a, b in zip(seq, bat):
            assert abs(a.statistic - b.statistic) <= 0.02 * a.extra["se"]
            assert abs(a.pvalue - b.pvalue) <= 2.0 / (1 + base.n_perm) + 1e-12
            ea, eb = _strip(a.extra), _strip(b.extra)
            num = {"W1_lower", "mean_diff", "z", "p_z", "se", "z_hac", "p_hac", "se_hac",
                   "p_perm", "p_perm_conservative", "p_perm_mid", "p_perm_stud", "ci",
                   "ci_iid", "ci_hac", "mean_f_new", "mean_f_past_ev", "mean_f_g_ev", "D_g",
                   "D_past", "train_curve", "train_info", "train_summary"}
            _assert_same({k: v for k, v in ea.items() if k not in num},
                         {k: v for k, v in eb.items() if k not in num})


def test_pipeline_helper_matches_relative_tests():
    """``anchored_relative_tests`` gives the rows of ``pipeline.relative_tests`` (dependent
    protocol: time blocks + HAC), bit for bit with ``impl="loop"``; each job's generator ends
    in the sequential end state."""
    r = np.random.default_rng(5)
    rp, rn = 0.01 * r.standard_normal((400, 3)), 0.01 * r.standard_normal((200, 3))
    Wp, gp, op = pl.panel_windows(rp, 20, scale=0.01, date_offset=0)
    Wn, gn, on = pl.panel_windows(rn, 20, scale=0.01, date_offset=400)
    jobs, seq = [], []
    kw = dict(base="critic", protocol="dependent", base_kw={"n_steps": 15, "n_perm": 50,
                                                            "batch_size": 16, "hidden": 8,
                                                            "arch": "pooled"}, n_boot=50)
    for s in range(3):
        G = 1.2 ** s * pl.panel_windows(0.01 * r.standard_normal((200, 3)), 20, scale=0.01,
                                        date_offset=400)[0]
        rng = np.random.default_rng(300 + s)
        jobs.append(AnchoredJob(Wn, G, Wp, rng=rng, orders=(on, on, op)))
        seq_rng = copy.deepcopy(rng)
        seq.append((pl.relative_tests(Wn, G, Wp, orders=(on, on, op), methods=("anchored",),
                                      rng=seq_rng, keep_extra=True, **kw)[0],
                    seq_rng.bit_generator.state))
    rows = anchored_relative_tests(jobs, keep_extra=True, impl="loop", **kw)
    for (a, state), b, j in zip(seq, rows, jobs):
        assert j.rng.bit_generator.state == state
        ea, eb = a.pop("extra"), b.pop("extra")
        _assert_same({k: v for k, v in a.items() if k != "runtime_s"},
                     {k: v for k, v in b.items() if k != "runtime_s"})
        _assert_same(_strip(ea), _strip(eb))
        assert eb["train_info"]["bank_size"] == 3 and eb["pvalue_type"] == "hac"


# ----------------------------------------------------------------------------------------------
# E3 power designs (experiments/e3_sp500.py): batched rows equal the sequential rows
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e3_module():
    saved = dict(os.environ)
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import e3_sp500 as e3
    except Exception as exc:  # pragma: no cover - optional experiment dependencies
        pytest.skip(f"experiments/e3_sp500.py not importable: {exc!r}")
    finally:
        sys.path.remove(str(ROOT / "experiments"))
        os.environ.clear()
        os.environ.update(saved)
    return e3


@pytest.fixture(scope="module")
def tiny_panel():
    rng = np.random.default_rng(3)
    g = get_model("GARCHt", mu=2e-4, omega=2e-6, alpha=0.08, beta=0.9, nu=6.0)
    cols = ["A", "B", "C"]
    tr = pd.DataFrame(g.simulate(400, 3, rng=rng).T, columns=cols)
    te = pd.DataFrame(g.simulate(200, 3, rng=rng).T, columns=cols)
    panel = pl.RealPanel(train=tr, test=te, scale=tr.std(ddof=1))
    fitted, _ = pl.fit_panel("GARCHt", tr, n_workers=1)
    return panel, fitted


def test_e3_batched_power_rows_match_sequential(e3_module, tiny_panel, monkeypatch):
    e3 = e3_module
    panel, fitted = tiny_panel
    c = {"base_kw": {"n_steps": 10, "n_perm": 40, "batch_size": 16, "hidden": 8,
                     "arch": "pooled"}, "n_boot": 40, "n_seeds": 2}
    monkeypatch.setattr(e3, "anchored_relative_tests",
                        functools.partial(anchored_relative_tests, impl="loop"))
    timing = {"test_seconds", "sim_seconds_rep", "sim_seconds_seed"}
    for seq_fn, bat_fn, args in ((e3.power_rep_rows, e3.power_rep_rows_batched,
                                  (panel, fitted, 1, c)),
                                 (e3.realvol_rep_rows, e3.realvol_rep_rows_batched,
                                  (panel, 1, c))):
        seq, bat = seq_fn(*args), bat_fn(*args)
        assert len(seq) == len(bat) == 4
        for a, b in zip(seq, bat):
            assert list(a) == list(b)  # same columns in the same order
            _assert_same({k: v for k, v in a.items() if k not in timing},
                         {k: v for k, v in b.items() if k not in timing})


def test_e3_e5_workers_dispatch_on_the_batched_flag(e3_module, tiny_panel, monkeypatch):
    """E3's ``_power_rep`` / ``_realvol_rep`` and E5's ``_design_rep`` run the batched functions
    only when the cell's configuration carries ``batched=True`` (set from SYNTHVAL_BATCHED)."""
    e3 = e3_module
    panel, fitted = tiny_panel
    calls: list[str] = []
    for name in ("power_rep_rows", "power_rep_rows_batched", "realvol_rep_rows",
                 "realvol_rep_rows_batched"):
        monkeypatch.setattr(e3, name, lambda *a, _n=name, **k: calls.append(_n) or [{}])
    monkeypatch.setattr(e3, "get_panel", lambda tickers: panel)
    monkeypatch.setattr(e3, "load_fits", lambda path: fitted)
    cell = {"_tickers": ["A"], "_fit_path": "", "_cfg": {}}
    e3._power_rep(cell, 0, None)
    e3._realvol_rep(cell, 0, None)
    e3._power_rep({**cell, "_cfg": {"batched": True}}, 0, None)
    e3._realvol_rep({**cell, "_cfg": {"batched": True}}, 0, None)
    assert calls == ["power_rep_rows", "realvol_rep_rows", "power_rep_rows_batched",
                     "realvol_rep_rows_batched"]
    monkeypatch.delenv("SYNTHVAL_BATCHED", raising=False)
    assert e3._batched_flag() == {}
    monkeypatch.setenv("SYNTHVAL_BATCHED", "1")
    assert e3._batched_flag() == {"batched": True}

    saved = dict(os.environ)
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import e5_robustness as e5
    except Exception as exc:  # pragma: no cover - optional experiment dependencies
        pytest.skip(f"experiments/e5_robustness.py not importable: {exc!r}")
    finally:
        sys.path.remove(str(ROOT / "experiments"))
        os.environ.clear()
        os.environ.update(saved)
    monkeypatch.setattr(e5, "get_panel", lambda *a: panel)
    monkeypatch.setattr(e5, "_load_fits", lambda path: fitted)
    calls.clear()
    for design in ("D1", "D2"):
        for flag in ({}, {"batched": True}):
            e5._design_rep({"data_split": "main", "_n_tickers": 3, "L": 20, "design": design,
                            "_fits": "", "_c": flag}, 0, None)
    assert calls == ["power_rep_rows", "power_rep_rows_batched", "realvol_rep_rows",
                     "realvol_rep_rows_batched"]
    assert "batched" in e5.batched_design_key("D1")
    monkeypatch.delenv("SYNTHVAL_BATCHED")
    assert e5.batched_design_key("D1", "D2") == {}


# ----------------------------------------------------------------------------------------------
# (c) exactness of the batched permutation p-values
# ----------------------------------------------------------------------------------------------
def test_batched_anchored_pvalues_uniform_at_boundary():
    """Anchored test at Law(G) = Law(R_past) with i.i.d. windows, 60 tests in one bank: the
    permutation p-values are uniform (exact level whatever the trained critic)."""
    base = CriticTest(n_steps=20, batch_size=32, arch="mlp", hidden=16, n_perm=199)
    probs = [RelativeProblem(_windows(1000 + s, 60, scale=1.3), _windows(2000 + s, 60),
                             _windows(3000 + s, 60), np.random.default_rng(4000 + s))
             for s in range(60)]
    res = relative_tests_anchored(probs, base)
    p = np.array([r.pvalue for r in res])
    assert res[0].extra["train_info"]["bank_size"] == 60
    assert stats.kstest(p, "uniform").pvalue > 1e-3
    assert np.sum(p <= 0.05) <= int(stats.binom.ppf(0.995, p.size, 0.05))
