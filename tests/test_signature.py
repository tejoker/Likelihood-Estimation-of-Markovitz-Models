"""Tests for truncated signatures (Chen's identity) and the signature-MMD test."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from synthval.twosample.signature import (SigMMDTest, lead_lag, segment_signature, sig_dim,
                                          sig_levels, signature, signature_features,
                                          signature_product, windows_to_paths)


def max_rejections(reps: int, alpha: float = 0.05, tail: float = 0.005) -> int:
    return int(stats.binom.ppf(1.0 - tail, reps, alpha))


# ----------------------------------------------------------------------------------------------
# Brute-force iterated integrals of a piecewise-linear path (depth <= 3)
# ----------------------------------------------------------------------------------------------
def brute_force_signature(points: np.ndarray, depth: int) -> list[np.ndarray]:
    """Exact iterated integrals by summing over ordered segment indices (no Chen recursion).

    For increments D_s, with ties within one segment contributing D^i D^j / 2 (two indices) or
    D^i D^j D^k / 6 (three indices):
      S^i = sum_s D_s^i
      S^ij = sum_{s<t} D_s^i D_t^j + sum_s D_s^i D_s^j / 2
      S^ijk = sum_{s<t<u} D_s^i D_t^j D_u^k + sum_{s<u} (D_s^i D_s^j / 2) D_u^k
              + sum_{s<u} D_s^i (D_u^j D_u^k / 2) + sum_s D_s^i D_s^j D_s^k / 6
    """
    D = np.diff(points, axis=0)
    S, d = D.shape
    out = [D.sum(0)]
    if depth >= 2:
        L2 = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                v = sum(D[s, i] * D[t, j] for s in range(S) for t in range(s + 1, S))
                v += sum(D[s, i] * D[s, j] / 2 for s in range(S))
                L2[i, j] = v
        out.append(L2)
    if depth >= 3:
        L3 = np.zeros((d, d, d))
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    v = 0.0
                    for s in range(S):
                        v += D[s, i] * D[s, j] * D[s, k] / 6
                        for u in range(s + 1, S):
                            v += D[s, i] * D[s, j] / 2 * D[u, k]
                            v += D[s, i] * D[u, j] * D[u, k] / 2
                            for t in range(s + 1, u):
                                v += D[s, i] * D[t, j] * D[u, k]
                    L3[i, j, k] = v
        out.append(L3)
    return out


def fine_grid_level2(points: np.ndarray, refine: int = 50) -> np.ndarray:
    """Level 2 by the trapezoid rule on a refined grid (exact for piecewise-linear paths)."""
    fine = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        for r in range(1, refine + 1):
            fine.append(a + (b - a) * r / refine)
    fine = np.array(fine)
    inc = np.diff(fine, axis=0)
    level1 = fine - fine[0]
    mid = 0.5 * (level1[:-1] + level1[1:])
    return mid.T @ inc


@pytest.mark.parametrize("d,P", [(1, 4), (2, 2), (2, 5), (3, 6)])
def test_signature_matches_brute_force(d, P):
    rng = np.random.default_rng(d * 10 + P)
    pts = rng.standard_normal((P, d))
    sig = signature(pts, depth=3)
    levels = sig_levels(sig, d, 3)
    for got, want in zip(levels, brute_force_signature(pts, 3)):
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(levels[1], fine_grid_level2(pts), rtol=1e-9, atol=1e-12)


def test_signature_batch_equals_loop():
    rng = np.random.default_rng(1)
    paths = rng.standard_normal((7, 9, 3))
    batch = signature(paths, depth=4)
    assert batch.shape == (7, sig_dim(3, 4))
    for i in range(7):
        np.testing.assert_allclose(batch[i], signature(paths[i], depth=4))


def test_chen_identity():
    rng = np.random.default_rng(2)
    paths = rng.standard_normal((5, 12, 3))
    for cut in (1, 4, 10):
        whole = signature(paths, depth=4)
        first = signature(paths[:, :cut + 1], depth=4)
        second = signature(paths[:, cut:], depth=4)
        np.testing.assert_allclose(signature_product(first, second, 3, 4), whole,
                                   rtol=1e-10, atol=1e-12)


def test_straight_line_exponential_and_reparametrisation():
    rng = np.random.default_rng(3)
    a, v = rng.standard_normal(3), rng.standard_normal(3)
    ts = np.sort(np.concatenate([[0.0, 1.0], rng.uniform(size=6)]))
    line = a + ts[:, None] * v  # collinear points, uneven spacing
    levels = sig_levels(signature(line, depth=4), 3, 4)
    tensor = np.array(1.0)
    for k, lev in enumerate(levels, start=1):
        tensor = np.multiply.outer(tensor, v)
        np.testing.assert_allclose(lev, tensor / math.factorial(k), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(signature(line, 4), segment_signature(v, 4), rtol=1e-10,
                               atol=1e-12)


def test_shuffle_identity_and_translation_invariance():
    rng = np.random.default_rng(4)
    pts = rng.standard_normal((10, 3))
    l1, l2 = sig_levels(signature(pts, 2), 3, 2)
    inc = pts[-1] - pts[0]
    np.testing.assert_allclose(l1, inc)
    np.testing.assert_allclose(l2 + l2.T, np.outer(inc, inc), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(signature(pts + 5.0, 4), signature(pts, 4), atol=1e-10)


def test_lead_lag_area_is_quadratic_variation():
    rng = np.random.default_rng(5)
    r = rng.standard_normal((4, 15))
    paths = windows_to_paths(r, time_aug=False, lead_lag_transform=True)
    assert paths.shape == (4, 31, 2)
    l2 = sig_levels(signature(paths, 2), 2, 2)[1]
    np.testing.assert_allclose(l2[:, 0, 1] - l2[:, 1, 0], (r**2).sum(1), rtol=1e-10)


def test_windows_to_paths_layout():
    r = np.arange(1.0, 6.0)[None, :]  # one window, L = 5
    p = windows_to_paths(r)
    assert p.shape == (1, 6, 2)
    np.testing.assert_allclose(p[0, :, 0], np.linspace(0, 1, 6))
    np.testing.assert_allclose(p[0, :, 1], [0, 1, 3, 6, 10, 15])
    pll = windows_to_paths(r, lead_lag_transform=True)
    assert pll.shape == (1, 11, 3)
    mv = windows_to_paths(np.ones((2, 3, 4)))
    assert mv.shape == (2, 5, 4)
    np.testing.assert_allclose(lead_lag(np.array([[[0.0], [1.0], [3.0]]]))[0],
                               [[0, 0], [1, 0], [1, 1], [3, 1], [3, 3]])


def test_signature_features_dimension_and_guard():
    rng = np.random.default_rng(6)
    F = signature_features(rng.standard_normal((10, 20)), depth=4)
    assert F.shape == (10, 30)
    F = signature_features(rng.standard_normal((10, 20)), depth=4, lead_lag_transform=True)
    assert F.shape == (10, 120)
    with pytest.raises(ValueError):
        signature_features(rng.standard_normal((3, 30, 5)), depth=4)  # 31^4 features


def test_sigmmd_size():
    reps = 100
    rng = np.random.default_rng(7)
    t = SigMMDTest(n_perm=100)
    rej = sum(t(rng.standard_normal((60, 10)), rng.standard_normal((60, 10)), rng=rng).pvalue
              <= 0.05 for _ in range(reps))
    assert rej <= max_rejections(reps)


@pytest.mark.parametrize("lead_lag_flag", [False, True])
def test_sigmmd_power(lead_lag_flag):
    rng = np.random.default_rng(8)
    t = SigMMDTest(n_perm=200, lead_lag=lead_lag_flag, depth=3 if lead_lag_flag else 4)
    X = rng.standard_normal((300, 20))
    assert t(X, np.sqrt(1.5) * rng.standard_normal((300, 20)), rng=rng).pvalue <= 0.01
    res = t(X, rng.standard_t(3, (300, 20)), rng=rng)
    assert res.pvalue <= 0.01 and res.extra["depth"] == (3 if lead_lag_flag else 4)


def test_sigmmd_wild_size_iid_and_power():
    """Wild-bootstrap signature MMD (dependence-robust option): level on i.i.d. windows, power."""
    reps = 60
    rng = np.random.default_rng(9)
    t = SigMMDTest(n_perm=200, depth=3, inference="wild")
    rej = sum(t(rng.standard_normal((80, 10)), rng.standard_normal((80, 10)), rng=rng).pvalue
              <= 0.05 for _ in range(reps))
    assert rej <= max_rejections(reps)
    X = rng.standard_normal((300, 20))
    res = t(X, np.sqrt(1.5) * rng.standard_normal((300, 20)), rng=rng)
    assert res.pvalue <= 0.01 and res.extra["inference"] == "wild"
