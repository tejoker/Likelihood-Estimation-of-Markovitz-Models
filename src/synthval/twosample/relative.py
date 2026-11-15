"""Relative-similarity tests: is the synthetic sample at least as close to the new real data as
the real past is?

Three samples: ``R_new`` (held-out real windows), ``G`` (synthetic windows from a generator
calibrated on the past) and ``R_past`` (real windows of the calibration period). Large values of
the statistic ``Delta`` are evidence that ``G`` is *worse* than simply replaying history
(DESIGN section 1, decision rule step 2). :func:`relative_test` dispatches on the base test and
on ``method``:

=========================  ===================  =============================================
base                       ``method``           test
=========================  ===================  =============================================
critic / lrcritic / c2st   ``"anchored"``       one critic, exact at ``Law(G) = Law(R_past)``
(learned)                  (default, primary)   (:func:`relative_test_anchored`)
critic / lrcritic / c2st   ``"two_critic"``     one critic per pair, paired bootstrap
                           (secondary)          (:func:`relative_test_learned`)
mmd / sigmmd               ignored              Bounliphone et al. (2016)
                                                (:func:`relative_test_mmd`)
=========================  ===================  =============================================

**Anchored relative test** (``method="anchored"``, DESIGN section 3.7, theory.md Remark 6.9(4)).
``G`` and ``R_past`` are split into train/eval parts (contiguous time blocks built from
``order_g`` / ``order_past`` when the base test has ``time_blocks`` set; by group with a shared
label namespace when ``groups_g`` / ``groups_past`` are given; else at random with the base
test's ``train_frac``). ``R_new`` is used *only for training, all of it*. One learned critic
``f`` (same architecture and optimiser as the base test) is trained to separate ``R_new``
(positive class) from the pool ``{G_tr, R_past_tr}`` (negative class) with **weighted
training**: inside the negative class ``G_tr`` and ``R_past_tr`` get equal total weight (1/2
each, whatever their sizes) and the positive class gets the same total weight as the negative
class. For the WGAN critic the objective is ``weighted mean f(R_new) - weighted mean f(pool) -
lambda GP`` with the gradient penalty on interpolates between ``R_new`` and pool points drawn
with those weights; for the C2ST logit it is the weighted BCE (implemented by drawing
mini-batch points with the weights, so every mini-batch loss is an unbiased estimate of the
weighted loss; see :meth:`LearnedTest.fit_witness`; with early stopping the pool's inner
training / validation subsets are renormalised, so the 1/2-1/2 balance then holds up to the
random composition of those subsets). The weights only affect power, never the level. With
frozen ``f``,

    Delta = mean f(R_past_ev) - mean f(G_ev),        H1: Delta > 0

("the critic that scores R_new-likeness finds the real past more R_new-like than G"). The
``R_new`` terms of the two absolute discrepancies cancel, so ``R_new`` never enters the
evaluation. **Exactness:** at the boundary ``Law(G) = Law(R_past)`` with i.i.d. windows (and
``G`` independent of ``R_past``), the pooled ``G_ev`` / ``R_past_ev`` windows are i.i.d. given
the training data and the training randomness, so the permutation of their labels with ``f``
fixed has exact finite-sample level (theory.md Proposition 3.1), for every critic, whatever
``R_new`` is and whatever the training did. The null actually tested is
``E_G f <= E_past f`` for the trained ``f`` (theory.md Remark 6.9(4)); in the interior of the
null with unequal eval sizes the plain permutation is only approximately calibrated, and
``extra["p_perm_stud"]`` gives the studentised permutation p-value (exact at the boundary,
asymptotically valid for the weaker null, Janssen 1997; Chung & Romano 2013). For serially
dependent windows (real data, long simulated paths) the permutation p-value is not valid: pass
``order_*`` and ``inference="hac"`` (and ``time_blocks`` on the base test); the reported
p-value is then the normal p-value of the Driscoll-Kraay / Newey-West studentised ``Delta``
(``extra["p_hac"]``, :func:`~synthval.twosample.critic.mean_diff_inference`). Caveats: if
``G`` contains copies of ``R_past`` windows (stationary bootstrap generator), ``G_ev`` can
duplicate pool training points and exchangeability fails; on real data the last ``R_past``
block borders the first ``R_new`` dates, so with volatility clustering the critic can learn a
regime shared by ``R_new`` and the end of ``R_past`` (small, but in the direction of
rejection).

**Two-critic relative test** (``method="two_critic"``, secondary). ``R_new``, ``G`` and
``R_past`` are split into train/eval halves (by group when labels are given, with one shared
label namespace so a ticker is on the same side in all three samples; by time blocks with
``time_blocks``). One critic per pair is trained on the training halves, ``f1`` on
``(R_new, G)`` and ``f2`` on ``(R_new, R_past)``, and frozen. On the evaluation halves
``D1 = mean f1(R_new) - mean f1(G)`` and ``D2 = mean f2(R_new) - mean f2(R_past)``, so

    Delta = D1 - D2 = mean_{R_new}(f1 - f2) - mean_G f1 + mean_{R_past} f2,

a sum of three independent sample means given the training half. The **paired bootstrap**
resamples the shared ``R_new`` evaluation windows once per replicate for both distances (this
keeps the positive correlation between ``D1`` and ``D2``), and ``G`` and ``R_past``
independently. p-value: ``(1 + #{Delta*_b - Delta >= Delta}) / (1 + B)`` (bootstrap
distribution centred at the estimate, Hall & Wilson 1991); CI: percentile. The analytic i.i.d.
z and, with ``order_*``, the Driscoll-Kraay HAC z are reported as well. **This version is
anti-conservative at the boundary** ``Law(G) = Law(R_past)`` (theory.md Proposition 6.8): it
is valid only conditionally on the two trained critics (null ``delta <= 0`` for that pair),
and at the boundary ``delta`` is the difference of two exchangeable replicates of a
training-dependent quantity, symmetric around 0 with a spread of the order of the evaluation
noise, so the unconditional rejection rate exceeds ``alpha`` (it tends to 1/2 for fixed
critics and growing evaluation sets). The anchored version is exact there.

**MMD / signature MMD** (Bounliphone et al. 2016). There is no learned witness, so ``method``
does not apply (the anchored construction needs a trained critic) and the MMD bases always use
this version. One Gaussian kernel (median heuristic on the pooled three samples) for both
distances; ``Delta = MMD_u^2(R_new, G) - MMD_u^2(R_new, R_past)``. Default
``mmd_method="asymptotic"``: the two unbiased estimates are jointly asymptotically normal
(U-statistic theory), and the variance of their difference is the variance of its first-order
Hoeffding projection, which for ``X = R_new``, ``Y = G``, ``Z = R_past`` and kernel mean
embeddings ``mu_P(.) = E_P k(., W)`` is

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

**Serial dependence.** The permutation p-value (anchored), the paired bootstrap (two-critic),
the multinomial bootstrap and the i.i.d. asymptotic variance (MMD) all treat windows as i.i.d.;
with consecutive windows of persistent-volatility series they under-estimate the variance of
``Delta`` and the test over-rejects. Pass ``order_*`` (time indices in window units or date
ranks) and ``inference="hac"``: the p-value is then the normal p-value of the Driscoll-Kraay /
Newey-West studentised ``Delta`` (``extra["p_hac"]``, Andrews AR(1) plug-in lag) and the CI is
``Delta +- z se_hac``.
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

__all__ = ["relative_test", "relative_test_anchored", "relative_test_learned",
           "relative_test_two_critic", "relative_test_mmd", "perm_studentized_pvalue"]


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


def _check_three(R_new: np.ndarray, G: np.ndarray,
                 R_past: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_new = _as_windows(R_new, "R_new")
    G = _as_windows(G, "G")
    R_past = _as_windows(R_past, "R_past")
    if not (R_new.shape[1:] == G.shape[1:] == R_past.shape[1:]):
        raise ValueError("R_new, G and R_past must have the same per-sample shape")
    return R_new, G, R_past


# ----------------------------------------------------------------------------------------------
# Anchored relative test (one critic: R_new vs the balanced pool {G_tr, R_past_tr})
# ----------------------------------------------------------------------------------------------
def perm_studentized_pvalue(gx: np.ndarray, gy: np.ndarray, n_perm: int, *,
                            rng: np.random.Generator, block: int = 256) -> float:
    """One-sided permutation p-value of the studentised difference of means (Welch t).

    ``t = (mean gx - mean gy) / sqrt(s_x^2 / n_x + s_y^2 / n_y)`` is recomputed for each of
    ``n_perm`` random relabellings of the pooled values; p-value ``(1 + #{t_b >= t}) / (1 + B)``.
    Exact when the pooled values are exchangeable; asymptotically valid for equality of means
    with unequal laws and unequal sizes (Janssen 1997; Chung & Romano 2013).

    Parameters
    ----------
    gx, gy : arrays
        Values of a fixed function on the two samples (H1: ``mean gx > mean gy``).
    n_perm : int
        Number of permutations (>= 1).
    rng : numpy.random.Generator
        Random generator.
    block : int, default 256
        Permutations per vectorised block.

    Returns
    -------
    float
    """
    gx = np.asarray(gx, dtype=np.float64).ravel()
    gy = np.asarray(gy, dtype=np.float64).ravel()
    n1, n2 = gx.size, gy.size
    if n1 < 2 or n2 < 2 or n_perm < 1:
        raise ValueError("need >= 2 values per sample and n_perm >= 1")
    g = np.concatenate([gx, gy])
    g = g - g.mean()  # centring avoids cancellation in the variance updates
    g2 = g * g
    tot, tot2 = g.sum(), g2.sum()

    def welch(s1: np.ndarray, q1: np.ndarray) -> np.ndarray:
        m1, m2 = s1 / n1, (tot - s1) / n2
        v1 = np.maximum(q1 - n1 * m1 * m1, 0.0) / (n1 - 1)
        v2 = np.maximum((tot2 - q1) - n2 * m2 * m2, 0.0) / (n2 - 1)
        se = np.sqrt(v1 / n1 + v2 / n2)
        d = m1 - m2
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, d / np.where(se > 0, se, 1.0), np.sign(d) * np.inf)
        return np.where(np.isnan(t), 0.0, t)

    t_obs = float(welch(np.array([g[:n1].sum()]), np.array([g2[:n1].sum()]))[0])
    base = np.zeros(n1 + n2)
    base[:n1] = 1.0
    count = 0
    tol = 1e-12 * max(1.0, abs(t_obs)) if np.isfinite(t_obs) else 0.0
    for start in range(0, n_perm, block):
        b = min(block, n_perm - start)
        mask = rng.permuted(np.tile(base, (b, 1)), axis=1)
        t_b = welch(mask @ g, mask @ g2)
        count += int(np.sum(t_b >= t_obs - tol))
    return float((1 + count) / (1 + n_perm))


def _train_summary(info: dict[str, Any]) -> dict[str, Any]:
    """Compact summary of a training curve: first and last 10% averages of each tracked series."""
    out: dict[str, Any] = {"steps_run": info.get("steps_run"),
                           "train_time_s": info.get("train_time_s"), "arch": info.get("arch")}
    if "best_step" in info:
        out["best_step"] = info["best_step"]
    for name in ("train_curve", "val_curve"):
        curve = info.get(name) or {}
        steps = curve.get("step") or []
        if not steps:
            continue
        k = max(1, len(steps) // 10)
        prefix = "" if name == "train_curve" else "val_"
        for key, vals in curve.items():
            if key == "step" or not vals:
                continue
            out[f"{prefix}{key}_first"] = float(np.mean(vals[:k]))
            out[f"{prefix}{key}_last"] = float(np.mean(vals[-k:]))
    return out


def relative_test_anchored(R_new: np.ndarray, G: np.ndarray, R_past: np.ndarray,
                           base: LearnedTest, *, rng: np.random.Generator,
                           groups_new: np.ndarray | None = None,
                           groups_g: np.ndarray | None = None,
                           groups_past: np.ndarray | None = None,
                           order_new: np.ndarray | None = None,
                           order_g: np.ndarray | None = None,
                           order_past: np.ndarray | None = None,
                           ci_level: float = 0.95,
                           inference: Literal["default", "hac"] | None = None,
                           studentized: bool = True) -> TestResult:
    """Anchored relative test (DESIGN section 3.7; see the module docstring).

    Parameters
    ----------
    R_new, G, R_past : arrays (n, ...)
        Held-out real windows, synthetic windows, past real windows (same per-sample shape).
        ``R_new`` is used only to train the critic (all of it).
    base : LearnedTest
        Configured ``CriticTest`` (WGAN-GP critic) or ``C2ST`` (``statistic="logit"``: the
        likelihood-ratio critic; ``"accuracy"`` also works, with ``f`` the indicator of being
        classified as ``R_new``). Its architecture, optimiser, ``n_steps``, early stopping,
        ``train_frac`` (share of ``G`` and ``R_past`` used for training), ``time_blocks``,
        ``n_perm``, ``tie_break`` and ``hac_lags`` are used.
    rng : numpy.random.Generator
        Random generator (split, training, permutations).
    groups_new, groups_g, groups_past : arrays, optional
        Group labels (shared namespace, e.g. ticker ids). ``groups_g`` / ``groups_past`` give a
        group split of ``G`` and ``R_past``; ``groups_new`` is only validated (``R_new`` is not
        split).
    order_new, order_g, order_past : arrays, optional
        Time indices (window units or date ranks). ``order_g`` / ``order_past`` are required
        when ``base.time_blocks`` is set (contiguous time-block split, shared label namespace)
        and give the HAC variance of ``Delta``; ``order_new`` is only validated.
    ci_level : float, default 0.95
        Confidence level of ``extra["ci"]``.
    inference : {"default", "hac"} or None, default None
        ``"default"``: permutation p-value of the ``G_ev`` / ``R_past_ev`` labels with ``f``
        fixed (exact at ``Law(G) = Law(R_past)`` for i.i.d. windows; normal p-value of the i.i.d.
        z if ``base.n_perm == 0``), CI with the i.i.d. standard error. ``"hac"``: the normal
        p-value of the Driscoll-Kraay / Newey-West studentised ``Delta`` (``p_hac``) and a CI
        with the HAC standard error (serially dependent windows; pass ``order_*``). ``None``:
        ``"hac"`` if ``base`` was built with ``inference="hac"``, else ``"default"``.
    studentized : bool, default True
        Also compute the studentised permutation p-value ``extra["p_perm_stud"]`` (needs
        ``base.n_perm > 0``).

    Returns
    -------
    TestResult
        ``statistic = Delta = mean f(R_past_ev) - mean f(G_ev)``, one-sided ``pvalue`` for
        ``H1: Delta > 0``. ``extra``: ``method="anchored"``, ``pvalue_type``
        (``"permutation"``, ``"hac"`` or ``"normal"``), ``ci`` (normal, HAC se under
        ``inference="hac"``, else i.i.d. se), ``ci_iid``, ``ci_hac``, ``p_perm`` (+ conservative
        / mid versions), ``p_perm_stud``, ``z``, ``p_z``, ``se``, ``z_hac``, ``p_hac``,
        ``se_hac``, ``hac``, ``hac_lags``, ``mean_f_new`` (mean of ``f`` on the ``R_new``
        windows, **in-sample**: they are the training positives), ``mean_f_past_ev``,
        ``mean_f_g_ev``, ``D_g = mean_f_new - mean_f_g_ev`` and
        ``D_past = mean_f_new - mean_f_past_ev`` (diagnostic absolute discrepancies, biased
        upward by the in-sample ``R_new`` term, which cancels in ``Delta = D_g - D_past``),
        split sizes, ``split``, ``train_info`` (curves) and ``train_summary``.
    """
    if not isinstance(base, LearnedTest):
        raise TypeError("the anchored relative test needs a learned base test (critic / C2ST)")
    inference = _resolve_inference(inference, base)
    R_new, G, R_past = _check_three(R_new, G, R_past)
    n_new, n_g, n_p = R_new.shape[0], G.shape[0], R_past.shape[0]
    _check_aux(groups_new, n_new, "groups_new")
    _check_aux(order_new, n_new, "order_new")
    gg = _check_aux(groups_g, n_g, "groups_g")
    gp = _check_aux(groups_past, n_p, "groups_past")
    og = _check_aux(order_g, n_g, "order_g")
    op = _check_aux(order_past, n_p, "order_past")
    split = "group" if (gg is not None or gp is not None) else "random"
    if base.time_blocks is not None:
        gg, gp = _time_block_groups([og, op], [gg, gp], base.time_blocks)
        split = "time_block"
    (tg, eg), (tp, ep) = train_eval_split([n_g, n_p], rng=rng, frac_train=base.train_frac,
                                          groups=[gg, gp])
    pool = np.concatenate([G[tg], R_past[tp]], axis=0)
    w_pool = np.concatenate([np.full(tg.size, 0.5 / tg.size), np.full(tp.size, 0.5 / tp.size)])
    witness, info = base.fit_witness(R_new, pool, rng=rng, weights_y=w_pool)
    f_new = base._witness_values(witness(R_new))
    f_g = base._witness_values(witness(G[eg]))
    f_p = base._witness_values(witness(R_past[ep]))
    inf = mean_diff_inference(f_p, f_g, rng=rng, n_perm=base.n_perm,
                              order_x=None if op is None else op[ep],
                              order_y=None if og is None else og[eg],
                              hac_lags=base.hac_lags, tie_break=base.tie_break)
    delta = float(inf["diff"])
    p_stud = (perm_studentized_pvalue(f_p, f_g, base.n_perm, rng=rng)
              if studentized and base.n_perm > 0 else float("nan"))
    if inference == "hac":
        pvalue, ptype = inf["p_hac"], "hac"
    elif base.n_perm > 0:
        pvalue, ptype = inf["p_perm"], "permutation"
    else:
        pvalue, ptype = inf["p_z"], "normal"
    ci_iid = _normal_ci(delta, inf["se"], ci_level)
    ci_hac = _normal_ci(delta, inf["se_hac"], ci_level) if inf["hac"] else ci_iid
    m_new, m_p, m_g = float(f_new.mean()), float(f_p.mean()), float(f_g.mean())
    extra: dict[str, Any] = {
        "method": "anchored", "relative_method": "anchored", "pvalue_type": ptype,
        "inference": inference,
        "ci": ci_hac if inference == "hac" else ci_iid, "ci_iid": ci_iid, "ci_hac": ci_hac,
        "ci_level": ci_level,
        "p_perm": inf["p_perm"], "p_perm_conservative": inf["p_perm_conservative"],
        "p_perm_mid": inf["p_perm_mid"], "perm_ties": inf["perm_ties"], "p_perm_stud": p_stud,
        "z": inf["z"], "p_z": inf["p_z"], "se": inf["se"], "z_hac": inf["z_hac"],
        "p_hac": inf["p_hac"], "se_hac": inf["se_hac"], "hac": inf["hac"],
        "hac_lags": inf["hac_lags"],
        "mean_f_new": m_new, "mean_f_past_ev": m_p, "mean_f_g_ev": m_g,
        "D_g": m_new - m_g, "D_past": m_new - m_p,
        "n_train_new": int(n_new), "n_train_g": int(tg.size), "n_train_past": int(tp.size),
        "n_eval_g": int(eg.size), "n_eval_past": int(ep.size), "split": split,
        "pool_weight_g": 0.5, "pool_weight_past": 0.5,
        "n_perm": base.n_perm, "tie_break": base.tie_break, "base": base.name,
        "train_info": info, "train_summary": _train_summary(info),
    }
    return TestResult(name=f"relative_{base.name}", statistic=delta, pvalue=float(pvalue),
                      n_x=int(n_new), n_y=int(n_g), extra=extra)


# ----------------------------------------------------------------------------------------------
# Two-critic learned version (secondary; anti-conservative at Law(G) = Law(R_past))
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
    """Two-critic relative test (``method="two_critic"``, secondary; see module docstring).

    ``base`` is a ``CriticTest`` or ``C2ST``. **Anti-conservative at the boundary**
    ``Law(G) = Law(R_past)`` (theory.md Proposition 6.8): valid only conditionally on the two
    trained critics; prefer :func:`relative_test_anchored`, which is exact there.

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
        "split": split, "method": method, "relative_method": "two_critic",
        "inference": inference, "n_boot": n_boot,
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
        "relative_method": "bounliphone",
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
relative_test_two_critic = relative_test_learned  # explicit alias for method="two_critic"


def relative_test(R_new: np.ndarray, G: np.ndarray, R_past: np.ndarray,
                  base: Any = "critic", *, rng: np.random.Generator,
                  method: Literal["anchored", "two_critic"] = "anchored", n_boot: int = 500,
                  groups_new: np.ndarray | None = None, groups_g: np.ndarray | None = None,
                  groups_past: np.ndarray | None = None, order_new: np.ndarray | None = None,
                  order_g: np.ndarray | None = None, order_past: np.ndarray | None = None,
                  mmd_method: str = "asymptotic", ci_level: float = 0.95,
                  inference: Literal["default", "hac"] | None = None) -> TestResult:
    """Relative test: is ``G`` detectably further from ``R_new`` than ``R_past`` is? (H1).

    Parameters
    ----------
    R_new : array (n, ...)
        Held-out real windows (the reference).
    G : array (n_g, ...)
        Synthetic windows.
    R_past : array (n_p, ...)
        Real windows from the calibration period.
    base : str, class or instance, default "critic"
        ``"critic"``, ``"lrcritic"``, ``"c2st"`` (learned), ``"mmd"``, ``"sigmmd"`` (Bounliphone
        et al.), or a configured ``CriticTest`` / ``C2ST`` / ``MMDTest`` / ``SigMMDTest``
        instance.
    rng : numpy.random.Generator
        Random generator.
    method : {"anchored", "two_critic"}, default "anchored"
        Learned bases only. ``"anchored"`` (primary): one critic trained on ``R_new`` vs the
        balanced pool ``{G_tr, R_past_tr}``, ``Delta = mean f(R_past_ev) - mean f(G_ev)``,
        permutation p-value exact at ``Law(G) = Law(R_past)`` for i.i.d. windows
        (:func:`relative_test_anchored`). ``"two_critic"`` (secondary, anti-conservative at that
        boundary, theory.md Prop. 6.8): ``Delta = D(G, R_new) - D(R_past, R_new)`` with one
        critic per pair and a paired bootstrap (:func:`relative_test_learned`). **Ignored for
        MMD / SigMMD bases**, which have no learned witness and always use the Bounliphone et
        al. (2016) difference of unbiased MMD^2 estimates (:func:`relative_test_mmd`);
        ``extra["relative_method"]`` records which version ran.
    n_boot : int, default 500
        Bootstrap replicates (two-critic; MMD with ``mmd_method="bootstrap"``). The anchored
        test uses ``base.n_perm`` permutations instead.
    groups_new, groups_g, groups_past : arrays, optional
        Group labels for the train/eval split of learned bases (ignored by MMD bases, which do
        not learn anything). The anchored test splits only ``G`` and ``R_past``.
    order_new, order_g, order_past : arrays, optional
        Time indices (window units or date ranks) for ``extra["z_hac"]`` / ``extra["p_hac"]``
        and, with ``base.time_blocks``, for the time-block split.
    mmd_method : {"asymptotic", "bootstrap"}, default "asymptotic"
        Inference for MMD bases.
    ci_level : float, default 0.95
        Confidence level of ``extra["ci"]``.
    inference : {"default", "hac"} or None, default None
        ``"hac"`` reports the HAC-studentised p-value and normal CI (serially dependent windows;
        pass ``order_*``); ``"default"`` the i.i.d. inference (anchored: permutation; two-critic:
        paired bootstrap; MMD: asymptotic or bootstrap); ``None`` follows the base test
        (``"hac"`` if it was built with ``inference="hac"`` or ``"wild"``).

    Returns
    -------
    TestResult
        Statistic ``Delta`` (positive: ``G`` worse than history), one-sided ``pvalue`` for
        ``H1: Delta > 0``, ``extra["ci"]``, ``extra["relative_method"]``. For every version
        ``Delta = extra["D_g"] - extra["D_past"]``.
    """
    if method not in ("anchored", "two_critic"):
        raise ValueError("method must be 'anchored' or 'two_critic'")
    b = _resolve_base(base)
    if isinstance(b, (CriticTest, C2ST, LearnedTest)):
        kw = dict(groups_new=groups_new, groups_g=groups_g, groups_past=groups_past,
                  order_new=order_new, order_g=order_g, order_past=order_past,
                  ci_level=ci_level, inference=inference)
        if method == "anchored":
            return relative_test_anchored(R_new, G, R_past, b, rng=rng, **kw)
        return relative_test_learned(R_new, G, R_past, b, rng=rng, n_boot=n_boot, **kw)
    if isinstance(b, (MMDTest, SigMMDTest)):
        return relative_test_mmd(R_new, G, R_past, b, rng=rng, method=mmd_method,
                                 n_boot=n_boot, order_new=order_new, order_g=order_g,
                                 order_past=order_past, ci_level=ci_level, inference=inference)
    raise TypeError(f"relative test not available for base {type(b).__name__}")
