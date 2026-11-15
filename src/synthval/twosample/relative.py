"""Relative-similarity test: is the synthetic sample at least as close to the new real data as
the real past is?

Hypotheses (DESIGN section 1, decision rule step 2)::

    H0: D(G, R_new) <= D(R_past, R_new)     vs     H1: D(G, R_new) > D(R_past, R_new)

with statistic ``Delta = D_hat(G, R_new) - D_hat(R_past, R_new)``; large ``Delta`` is evidence
that the generator ``G`` is *worse* than simply replaying history.

Two versions:

* **Learned (critic / C2ST).** ``R_new``, ``G`` and ``R_past`` are split into train/eval halves
  (by group when labels are given, with one shared label namespace so a ticker is on the same
  side in all three samples). One critic per pair is trained on the training halves,
  ``f1`` on ``(R_new, G)`` and ``f2`` on ``(R_new, R_past)``, and frozen. On the evaluation
  halves ``D1 = mean f1(R_new) - mean f1(G)`` and ``D2 = mean f2(R_new) - mean f2(R_past)``, so

      Delta = mean_{R_new}(f1 - f2) - mean_G f1 + mean_{R_past} f2,

  a sum of three independent sample means given the training half. The **paired bootstrap**
  resamples the shared ``R_new`` evaluation windows once per replicate for both distances
  (this is what keeps the positive correlation between ``D1`` and ``D2``), and ``G`` and
  ``R_past`` independently. p-value: ``(1 + #{Delta*_b - Delta >= Delta}) / (1 + B)`` (bootstrap
  distribution centred at the estimate, Hall & Wilson 1991); CI: percentile. The analytic
  i.i.d. z and, with ``order_*``, the Driscoll-Kraay HAC z are reported as well.

* **MMD / signature MMD** (Bounliphone et al. 2016). One Gaussian kernel (median heuristic on the
  pooled three samples) for both distances; ``Delta = MMD_u^2(R_new, G) - MMD_u^2(R_new,
  R_past)``. Default ``mmd_method="asymptotic"``: the two unbiased estimates are jointly
  asymptotically normal (U-statistic theory), and the variance of their difference is the
  variance of its first-order Hoeffding projection, which for ``X = R_new``, ``Y = G``,
  ``Z = R_past`` and kernel mean embeddings ``mu_P(.) = E_P k(., W)`` is

      Var(Delta) ~ 4 Var_X[mu_Z(x) - mu_Y(x)] / n_x + 4 Var_Y[mu_Y(y) - mu_X(y)] / n_y
                   + 4 Var_Z[mu_X(z) - mu_Z(z)] / n_z,

  the unequal-sample-size version of Bounliphone et al.'s covariance formula (the ``U_xx`` term
  common to both distances cancels exactly). p-value ``1 - Phi(Delta / se)``, CI
  ``Delta +- 1.96 se``. ``mmd_method="bootstrap"`` instead resamples (paired over ``R_new``) the
  U-statistics through multinomial weights on stored kernel matrices; within-sample terms drop
  pairs of resampled positions that share an original index, which makes each resampled
  U-statistic unbiased for the original one (so the bootstrap distribution is centred at
  ``Delta`` whatever the sample sizes). **Limitation:** both approximations rely on a
  non-degenerate projection; when ``G``, ``R_new`` and ``R_past`` all have the same law the
  projection vanishes and neither the normal approximation nor the bootstrap is reliable (the
  test is then typically conservative).

**Serial dependence.** The paired bootstrap (learned version), the multinomial bootstrap and the
i.i.d. asymptotic variance (MMD version) all treat windows as i.i.d.; with consecutive windows
of persistent-volatility series they under-estimate the variance of ``Delta`` and the test
over-rejects. Pass ``order_*`` (time indices) and ``inference="hac"``: the p-value is then the
normal p-value of the Driscoll-Kraay / Newey-West studentised ``Delta`` (``extra["p_hac"]``,
Andrews AR(1) plug-in lag) and the CI is ``Delta +- z se_hac``.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import stats

from .base import TestResult, TwoSampleTest
from .c2st import C2ST
from .critic import (CriticTest, LearnedTest, _check_aux, _time_block_groups, _z_and_p,
                     hac_var_mean, mean_diff_inference, train_eval_split)
from .mmd import MMDTest, gaussian_kernel_cross, standardize_pooled
from .signature import SigMMDTest

__all__ = ["relative_test", "relative_test_learned", "relative_test_mmd"]


def _as_windows(Z: np.ndarray, what: str) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[0] < 4:
        raise ValueError(f"{what} needs at least 4 observations")
    if not np.all(np.isfinite(Z)):
        raise ValueError(f"{what} contains NaN or inf")
    return Z


def _resolve_base(base: Any) -> TwoSampleTest:
    if isinstance(base, str):
        from . import TEST_REGISTRY  # local import: the registry lives in the package __init__

        if base not in TEST_REGISTRY:
            raise KeyError(f"unknown base test {base!r}; options: {sorted(TEST_REGISTRY)}")
        return TEST_REGISTRY[base]()
    if isinstance(base, type):
        return base()
    return base


def _percentile_ci(boot: np.ndarray, level: float) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [a, 1.0 - a])
    return float(lo), float(hi)


def _centred_boot_pvalue(delta: float, boot: np.ndarray) -> float:
    """``(1 + #{Delta*_b - Delta >= Delta}) / (1 + B)`` for H1: Delta > 0."""
    return float((1 + np.sum(boot - delta >= delta)) / (1 + boot.size))


def _resolve_inference(inference: str | None, base: Any) -> str:
    """``None`` follows the base test: ``"hac"`` if it was configured with ``inference="hac"``
    (learned tests) or ``"wild"`` (MMD tests), else ``"default"``."""
    if inference is None:
        return "hac" if getattr(base, "inference", "permutation") in ("hac", "wild") else "default"
    if inference not in ("default", "hac"):
        raise ValueError("inference must be 'default', 'hac' or None")
    return inference


def _normal_ci(delta: float, se: float, level: float) -> tuple[float, float]:
    q = stats.norm.ppf(0.5 + level / 2.0)
    return float(delta - q * se), float(delta + q * se)


# ----------------------------------------------------------------------------------------------
# Learned (critic / C2ST) version
# ----------------------------------------------------------------------------------------------
def relative_test_learned(R_new: np.ndarray, G: np.ndarray, R_past: np.ndarray,
                          base: LearnedTest, *, rng: np.random.Generator, n_boot: int = 500,
                          groups_new: np.ndarray | None = None,
                          groups_g: np.ndarray | None = None,
                          groups_past: np.ndarray | None = None,
                          order_new: np.ndarray | None = None,
                          order_g: np.ndarray | None = None,
                          order_past: np.ndarray | None = None,
                          ci_level: float = 0.95, boot_block: int = 100,
                          inference: Literal["default", "hac"] | None = None) -> TestResult:
    """Critic-based relative test (see module docstring); ``base`` is a ``CriticTest`` or ``C2ST``.

    Parameters
    ----------
    R_new, G, R_past : arrays (n, ...)
        Held-out real windows, synthetic windows, past real windows (same per-sample shape).
    base : LearnedTest
        Configured learned test; its training settings, ``train_frac``, ``hac_lags``,
        ``tie_break`` and ``time_blocks`` (time-block split from ``order_*``, recommended for
        serially dependent windows) are used, and its ``n_perm`` for the two absolute tests
        reported in ``extra``.
    rng : numpy.random.Generator
        Random generator.
    n_boot : int, default 500
        Bootstrap replicates.
    groups_new, groups_g, groups_past : arrays, optional
        Group labels (shared namespace, e.g. ticker ids) for the group-based split.
    order_new, order_g, order_past : arrays, optional
        Time indices for the HAC z.
    ci_level : float, default 0.95
        Confidence level of the interval.
    boot_block : int, default 100
        Bootstrap replicates per vectorised block.
    inference : {"default", "hac"} or None, default None
        ``"default"``: centred paired-bootstrap p-value and percentile CI (i.i.d. windows).
        ``"hac"``: normal p-value of the HAC-studentised ``Delta`` and ``Delta +- z se_hac``
        (serially dependent windows; pass ``order_*``). ``None``: ``"hac"`` if ``base`` was
        built with ``inference="hac"``, else ``"default"``.

    Returns
    -------
    TestResult
        ``statistic = Delta``, ``pvalue`` = centred paired-bootstrap p-value (or ``p_hac``);
        ``extra`` has ``ci``, ``ci_boot``, ``D_g``, ``D_past``, ``se_boot``, ``z``, ``p_z``,
        ``z_hac``, ``p_hac``, ``p_boot``, ``p_abs_g`` / ``p_abs_past`` (permutation p-values of
        the two absolute tests on the same split, valid for i.i.d. windows), split sizes and
        training information.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    inference = _resolve_inference(inference, base)
    R_new = _as_windows(R_new, "R_new")
    G = _as_windows(G, "G")
    R_past = _as_windows(R_past, "R_past")
    if not (R_new.shape[1:] == G.shape[1:] == R_past.shape[1:]):
        raise ValueError("R_new, G and R_past must have the same per-sample shape")
    sizes = [R_new.shape[0], G.shape[0], R_past.shape[0]]
    grp = [_check_aux(groups_new, sizes[0], "groups_new"), _check_aux(groups_g, sizes[1], "groups_g"),
           _check_aux(groups_past, sizes[2], "groups_past")]
    ords = [_check_aux(order_new, sizes[0], "order_new"), _check_aux(order_g, sizes[1], "order_g"),
            _check_aux(order_past, sizes[2], "order_past")]
    split = "group" if any(g is not None for g in grp) else "random"
    if base.time_blocks is not None:
        grp = _time_block_groups(ords, grp, base.time_blocks)
        split = "time_block"
    (tn, en), (tg, eg), (tp, ep) = train_eval_split(sizes, rng=rng, frac_train=base.train_frac,
                                                    groups=grp)
    w1, info1 = base.fit_witness(R_new[tn], G[tg], rng=rng)
    w2, info2 = base.fit_witness(R_new[tn], R_past[tp], rng=rng)
    a1 = base._witness_values(w1(R_new[en]))
    b1 = base._witness_values(w1(G[eg]))
    a2 = base._witness_values(w2(R_new[en]))
    c2 = base._witness_values(w2(R_past[ep]))
    scale = base.stat_scale
    d1 = base._stat_from_diff(float(a1.mean() - b1.mean()))
    d2 = base._stat_from_diff(float(a2.mean() - c2.mean()))
    h = a1 - a2
    delta = scale * float(h.mean() - b1.mean() + c2.mean())

    boot = np.empty(n_boot)
    nn, ng, npst = h.size, b1.size, c2.size
    for start in range(0, n_boot, boot_block):
        b = min(boot_block, n_boot - start)
        mh = h[rng.integers(0, nn, (b, nn))].mean(1)  # shared R_new resample (paired)
        mg = b1[rng.integers(0, ng, (b, ng))].mean(1)
        mp = c2[rng.integers(0, npst, (b, npst))].mean(1)
        boot[start:start + b] = scale * (mh - mg + mp)
    p_boot = _centred_boot_pvalue(delta, boot)

    # Analytic i.i.d. and HAC standard errors of the same linear statistic.
    on, og, op = (None if o is None else o[e] for o, e in zip(ords, (en, eg, ep)))
    v_iid = sum(hac_var_mean(v)[0] for v in (h, b1, c2))
    se = scale * float(np.sqrt(v_iid))
    z, p_z = _z_and_p(delta, se)
    hac = any(o is not None for o in (on, og, op))
    if hac:
        parts = [hac_var_mean(v, o, base.hac_lags) for v, o in zip((h, b1, c2), (on, og, op))]
        se_hac = scale * float(np.sqrt(sum(p[0] for p in parts)))
        z_hac, p_hac = _z_and_p(delta, se_hac)
        lags = [p[1] for p in parts]
    else:
        se_hac, z_hac, p_hac, lags = se, z, p_z, None

    # The two absolute tests on the same split (permutation p-values, exact for i.i.d. windows).
    abs_g = mean_diff_inference(a1, b1, rng=rng, n_perm=base.n_perm, tie_break=base.tie_break)
    abs_p = mean_diff_inference(a2, c2, rng=rng, n_perm=base.n_perm, tie_break=base.tie_break)

    ci_boot = _percentile_ci(boot, ci_level)
    if inference == "hac":
        pvalue, ci, method = p_hac, _normal_ci(delta, se_hac, ci_level), "hac"
    else:
        pvalue, ci, method = p_boot, ci_boot, "paired_bootstrap"
    extra: dict[str, Any] = {
        "ci": ci, "ci_boot": ci_boot, "ci_level": ci_level, "D_g": d1, "D_past": d2,
        "se_boot": float(boot.std(ddof=1)) if n_boot > 1 else float("nan"),
        "p_boot": p_boot, "z": z, "p_z": p_z, "se": se, "z_hac": z_hac, "p_hac": p_hac,
        "se_hac": se_hac, "hac": hac, "hac_lags": lags,
        "p_abs_g": abs_g["p_perm"], "p_abs_past": abs_p["p_perm"],
        "n_eval_new": int(nn), "n_eval_g": int(ng), "n_eval_past": int(npst),
        "n_train_new": int(tn.size), "n_train_g": int(tg.size), "n_train_past": int(tp.size),
        "split": split, "method": method, "inference": inference, "n_boot": n_boot,
        "base": base.name,
        "train_info_g": info1, "train_info_past": info2,
    }
    return TestResult(name=f"relative_{base.name}", statistic=delta, pvalue=float(pvalue),
                      n_x=sizes[0], n_y=sizes[1], extra=extra)


# ----------------------------------------------------------------------------------------------
# MMD version (Bounliphone et al. 2016)
# ----------------------------------------------------------------------------------------------
def _cross_sums(A: np.ndarray, B: np.ndarray, bws: tuple[float, ...],
                block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Row and column sums of the kernel matrix ``K(A, B)`` without storing it."""
    rows = np.empty(A.shape[0])
    cols = np.zeros(B.shape[0])
    for i in range(0, A.shape[0], block):
        Kb = gaussian_kernel_cross(A[i:i + block], B, bws, block=block)
        rows[i:i + block] = Kb.sum(1)
        cols += Kb.sum(0)
    return rows, cols


def _sub_idx(n: int, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def relative_test_mmd(R_new: np.ndarray, G: np.ndarray, R_past: np.ndarray,
                      base: MMDTest | SigMMDTest, *, rng: np.random.Generator,
                      method: str = "asymptotic", n_boot: int = 500,
                      order_new: np.ndarray | None = None, order_g: np.ndarray | None = None,
                      order_past: np.ndarray | None = None, ci_level: float = 0.95,
                      boot_block: int = 50,
                      inference: Literal["default", "hac"] | None = None) -> TestResult:
    """MMD-based relative test (Bounliphone et al. 2016); see module docstring.

    Parameters
    ----------
    R_new, G, R_past : arrays (n, ...)
        Reference (held-out real), candidate (synthetic) and past real samples.
    base : MMDTest or SigMMDTest
        Supplies the feature map (flattening or signatures), ``max_n``, bandwidth options and
        standardisation (pooled over the three samples).
    rng : numpy.random.Generator
        Random generator (subsampling, median heuristic, bootstrap).
    method : {"asymptotic", "bootstrap"}, default "asymptotic"
        Joint asymptotic normality of the two U-statistics, or paired multinomial bootstrap.
    n_boot : int, default 500
        Bootstrap replicates (``method="bootstrap"`` only).
    order_new, order_g, order_past : arrays, optional
        Time indices; with ``method="asymptotic"`` they give ``z_hac`` (Driscoll-Kraay variance
        of the Hoeffding projections).
    ci_level : float, default 0.95
        Confidence level.
    boot_block : int, default 50
        Bootstrap replicates per vectorised block.
    inference : {"default", "hac"} or None, default None
        ``"default"``: the ``method`` p-value and CI (i.i.d. windows). ``"hac"``: normal p-value
        of ``Delta / se_hac`` (Driscoll-Kraay variance of the Hoeffding projections; serially
        dependent windows, pass ``order_*``) and ``Delta +- z se_hac``. ``None``: ``"hac"`` if
        ``base`` was built with ``inference="wild"``, else ``"default"``.

    Returns
    -------
    TestResult
        ``statistic = Delta = MMD_u^2(R_new, G) - MMD_u^2(R_new, R_past)``; ``extra`` holds
        ``ci``, ``D_g``, ``D_past``, ``se``, ``z``, ``p_z``, ``z_hac``, ``p_hac``, the standard
        errors of the two MMD estimates and their correlation, bandwidths and sample sizes.
    """
    if method not in ("asymptotic", "bootstrap"):
        raise ValueError("method must be 'asymptotic' or 'bootstrap'")
    inference = _resolve_inference(inference, base)
    R_new = _as_windows(R_new, "R_new")
    G = _as_windows(G, "G")
    R_past = _as_windows(R_past, "R_past")
    if not (R_new.shape[1:] == G.shape[1:] == R_past.shape[1:]):
        raise ValueError("R_new, G and R_past must have the same per-sample shape")
    sizes = [R_new.shape[0], G.shape[0], R_past.shape[0]]
    ords = [_check_aux(order_new, sizes[0], "order_new"), _check_aux(order_g, sizes[1], "order_g"),
            _check_aux(order_past, sizes[2], "order_past")]

    if isinstance(base, SigMMDTest):
        feats = [base.features(Z) for Z in (R_new, G, R_past)]
        do_std, mmd = base.standardize, base.mmd
    else:
        feats = [Z.reshape(Z.shape[0], -1) for Z in (R_new, G, R_past)]
        do_std, mmd = base.standardize, base
    if do_std:
        feats = standardize_pooled(*feats)
    idx = [_sub_idx(f.shape[0], mmd.max_n, rng) for f in feats]
    Fx, Fy, Fz = (f[i] for f, i in zip(feats, idx))
    ox, oy, oz = (None if o is None else o[i] for o, i in zip(ords, idx))
    nx, ny, nz = Fx.shape[0], Fy.shape[0], Fz.shape[0]
    bws = mmd.bandwidths_for(np.concatenate([Fx, Fy, Fz], axis=0), rng=rng)
    kdiag = float(len(bws))

    rs_xx, _ = _cross_sums(Fx, Fx, bws)
    rs_yy, _ = _cross_sums(Fy, Fy, bws)
    rs_zz, _ = _cross_sums(Fz, Fz, bws)
    rs_xy, cs_xy = _cross_sums(Fx, Fy, bws)
    rs_xz, cs_xz = _cross_sums(Fx, Fz, bws)
    mu_x_x = (rs_xx - kdiag) / (nx - 1)   # leave-one-out embeddings at own points
    mu_y_y = (rs_yy - kdiag) / (ny - 1)
    mu_z_z = (rs_zz - kdiag) / (nz - 1)
    mu_y_x, mu_z_x = rs_xy / ny, rs_xz / nz
    mu_x_y, mu_x_z = cs_xy / nx, cs_xz / nx
    u_xx, u_yy, u_zz = mu_x_x.mean(), mu_y_y.mean(), mu_z_z.mean()
    u_xy, u_xz = mu_y_x.mean(), mu_z_x.mean()
    d_g = float(u_xx + u_yy - 2.0 * u_xy)
    d_p = float(u_xx + u_zz - 2.0 * u_xz)
    delta = d_g - d_p

    # First-order Hoeffding projections (times 2).
    phi_x = 2.0 * (mu_z_x - mu_y_x)
    phi_y = 2.0 * (mu_y_y - mu_x_y)
    phi_z = 2.0 * (mu_x_z - mu_z_z)
    v = hac_var_mean(phi_x)[0] + hac_var_mean(phi_y)[0] + hac_var_mean(phi_z)[0]
    se = float(np.sqrt(max(v, 0.0)))
    z, p_z = _z_and_p(delta, se)
    # Standard errors of each MMD estimate and their correlation (Bounliphone et al. Thm 2).
    px1, py1 = 2.0 * (mu_x_x - mu_y_x), 2.0 * (mu_y_y - mu_x_y)
    px2, pz2 = 2.0 * (mu_x_x - mu_z_x), 2.0 * (mu_z_z - mu_x_z)
    var1 = hac_var_mean(px1)[0] + hac_var_mean(py1)[0]
    var2 = hac_var_mean(px2)[0] + hac_var_mean(pz2)[0]
    cov12 = float(np.cov(px1, px2)[0, 1] / nx)
    corr = cov12 / np.sqrt(var1 * var2) if var1 > 0 and var2 > 0 else float("nan")
    hac = any(o is not None for o in (ox, oy, oz))
    if hac:
        parts = [hac_var_mean(p, o) for p, o in zip((phi_x, phi_y, phi_z), (ox, oy, oz))]
        se_hac = float(np.sqrt(max(sum(p[0] for p in parts), 0.0)))
        z_hac, p_hac = _z_and_p(delta, se_hac)
        lags = [p[1] for p in parts]
    else:
        se_hac, z_hac, p_hac, lags = se, z, p_z, None

    q = stats.norm.ppf(0.5 + ci_level / 2.0)
    extra: dict[str, Any] = {
        "D_g": d_g, "D_past": d_p, "se": se, "z": z, "p_z": p_z, "z_hac": z_hac, "p_hac": p_hac,
        "se_hac": se_hac, "hac": hac, "hac_lags": lags, "se_D_g": float(np.sqrt(max(var1, 0))),
        "se_D_past": float(np.sqrt(max(var2, 0))), "corr_D": float(corr),
        "bandwidths": list(bws), "n_used_new": nx, "n_used_g": ny, "n_used_past": nz,
        "ci_level": ci_level, "base": base.name, "method": method, "inference": inference,
    }
    if method == "asymptotic":
        extra["ci"] = (float(delta - q * se), float(delta + q * se))
        pvalue = p_z
    else:
        boot = _mmd_bootstrap(Fx, Fy, Fz, bws, n_boot, rng=rng, block=boot_block)
        extra["ci"] = _percentile_ci(boot, ci_level)
        extra["se_boot"] = float(boot.std(ddof=1)) if n_boot > 1 else float("nan")
        extra["n_boot"] = n_boot
        pvalue = _centred_boot_pvalue(delta, boot)
        extra["p_boot"] = pvalue
    if inference == "hac":
        extra["ci_" + method] = extra["ci"]
        extra["ci"] = _normal_ci(delta, se_hac, ci_level)
        pvalue = p_hac
    return TestResult(name=f"relative_{base.name}", statistic=delta, pvalue=float(pvalue),
                      n_x=sizes[0], n_y=sizes[1], extra=extra)


def _mmd_bootstrap(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray, bws: tuple[float, ...],
                   n_boot: int, *, rng: np.random.Generator, block: int = 50) -> np.ndarray:
    """Paired multinomial bootstrap of ``Delta = U_yy - U_zz - 2 U_xy + 2 U_xz``.

    With multinomial bootstrap counts ``c`` (``sum c = n``), the within-sample term sums the
    kernel over pairs of resampled positions whose *original indices differ*:
    ``U* = (c' K c - sum_i c_i^2 K_ii) / (n - 1)^2``. Since ``E[c_i c_j] = (n - 1) / n`` for
    ``i != j`` this is unbiased for the original ``U`` (summing over distinct *positions*
    instead, ``(c' K c - n k_d) / (n (n - 1))``, has mean ``U (n - 1) / n + k_d / n``, a bias
    that does not cancel in ``Delta`` when ``n_g != n_past``). Cross terms
    ``c_x' K_xy c_y / (n_x n_y)`` are unbiased as they are; the ``R_new`` counts are shared by
    ``U_xy`` and ``U_xz`` (paired). Hence ``E*[Delta*] = Delta``. Stores four kernel matrices
    (``8 n^2`` bytes each).
    """
    nx, ny, nz = Fx.shape[0], Fy.shape[0], Fz.shape[0]
    kd = float(len(bws))
    Kyy = gaussian_kernel_cross(Fy, Fy, bws)
    Kzz = gaussian_kernel_cross(Fz, Fz, bws)
    Kxy = gaussian_kernel_cross(Fx, Fy, bws)
    Kxz = gaussian_kernel_cross(Fx, Fz, bws)
    np.fill_diagonal(Kyy, kd)
    np.fill_diagonal(Kzz, kd)
    out = np.empty(n_boot)
    px, py, pz = np.full(nx, 1.0 / nx), np.full(ny, 1.0 / ny), np.full(nz, 1.0 / nz)
    for start in range(0, n_boot, block):
        b = min(block, n_boot - start)
        Cx = rng.multinomial(nx, px, size=b).T.astype(np.float64)
        Cy = rng.multinomial(ny, py, size=b).T.astype(np.float64)
        Cz = rng.multinomial(nz, pz, size=b).T.astype(np.float64)
        uyy = ((Cy * (Kyy @ Cy)).sum(0) - kd * (Cy * Cy).sum(0)) / (ny - 1) ** 2
        uzz = ((Cz * (Kzz @ Cz)).sum(0) - kd * (Cz * Cz).sum(0)) / (nz - 1) ** 2
        uxy = (Cx * (Kxy @ Cy)).sum(0) / (nx * ny)
        uxz = (Cx * (Kxz @ Cz)).sum(0) / (nx * nz)
        out[start:start + b] = uyy - uzz - 2.0 * uxy + 2.0 * uxz
    return out


# ----------------------------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------------------------
def relative_test(R_new: np.ndarray, G: np.ndarray, R_past: np.ndarray,
                  base: Any = "critic", *, rng: np.random.Generator, n_boot: int = 500,
                  groups_new: np.ndarray | None = None, groups_g: np.ndarray | None = None,
                  groups_past: np.ndarray | None = None, order_new: np.ndarray | None = None,
                  order_g: np.ndarray | None = None, order_past: np.ndarray | None = None,
                  mmd_method: str = "asymptotic", ci_level: float = 0.95,
                  inference: Literal["default", "hac"] | None = None) -> TestResult:
    """Relative-similarity test ``H0: D(G, R_new) <= D(R_past, R_new)`` vs ``H1: >``.

    Parameters
    ----------
    R_new : array (n, ...)
        Held-out real windows (the reference).
    G : array (n_g, ...)
        Synthetic windows.
    R_past : array (n_p, ...)
        Real windows from the calibration period.
    base : str, class or instance, default "critic"
        ``"critic"``, ``"lrcritic"``, ``"c2st"`` (learned: paired bootstrap), ``"mmd"``,
        ``"sigmmd"`` (Bounliphone et al.), or a configured ``CriticTest`` / ``C2ST`` /
        ``MMDTest`` / ``SigMMDTest`` instance.
    rng : numpy.random.Generator
        Random generator.
    n_boot : int, default 500
        Bootstrap replicates.
    groups_new, groups_g, groups_past : arrays, optional
        Group labels for the train/eval split of learned bases (ignored by MMD bases, which do
        not learn anything).
    order_new, order_g, order_past : arrays, optional
        Time indices for ``extra["z_hac"]``.
    mmd_method : {"asymptotic", "bootstrap"}, default "asymptotic"
        Inference for MMD bases.
    ci_level : float, default 0.95
        Confidence level of ``extra["ci"]``.
    inference : {"default", "hac"} or None, default None
        ``"hac"`` reports the HAC-studentised p-value and normal CI (serially dependent windows;
        pass ``order_*``); ``"default"`` the bootstrap / asymptotic i.i.d. inference; ``None``
        follows the base test (``"hac"`` if it was built with ``inference="hac"`` or
        ``"wild"``).

    Returns
    -------
    TestResult
        ``statistic = Delta = D(G, R_new) - D(R_past, R_new)``, one-sided ``pvalue`` for
        ``H1: Delta > 0``, ``extra["ci"]``.
    """
    b = _resolve_base(base)
    if isinstance(b, (CriticTest, C2ST, LearnedTest)):
        return relative_test_learned(R_new, G, R_past, b, rng=rng, n_boot=n_boot,
                                     groups_new=groups_new, groups_g=groups_g,
                                     groups_past=groups_past, order_new=order_new,
                                     order_g=order_g, order_past=order_past, ci_level=ci_level,
                                     inference=inference)
    if isinstance(b, (MMDTest, SigMMDTest)):
        return relative_test_mmd(R_new, G, R_past, b, rng=rng, method=mmd_method,
                                 n_boot=n_boot, order_new=order_new, order_g=order_g,
                                 order_past=order_past, ci_level=ci_level, inference=inference)
    raise TypeError(f"relative test not available for base {type(b).__name__}")
