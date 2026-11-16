"""Two-sample tests for synthetic-vs-real return windows (DESIGN section 3.7).

Every test is a callable ``test(X, Y, *, rng, groups_x=None, groups_y=None, order_x=None,
order_y=None) -> TestResult`` with ``X`` the real and ``Y`` the synthetic sample.

``TEST_REGISTRY`` maps short names to zero-argument factories (classes or partials) with the
paper's default settings::

    critic    Wasserstein-critic test (WGAN-GP critic, held-out permutation)
    lrcritic  likelihood-ratio critic (C2ST with the mean-logit statistic)
    c2st      classifier two-sample test (balanced accuracy)
    mmd       Gaussian-kernel MMD, median heuristic
    sigmmd    MMD on truncated path signatures
    ks        KS on pooled returns (baseline)

Validity under serial dependence: the default permutation p-values (critic, lrcritic, c2st,
mmd, sigmmd) are exact only for i.i.d. (exchangeable) windows. For consecutive windows cut from
persistent-volatility series pass ``order_x`` / ``order_y`` and use ``inference="hac"``
(learned tests: HAC-studentised p-value) or ``inference="wild"`` (MMD / signature MMD: wild
bootstrap of Chwialkowski, Sejdinovic & Gretton 2014); ``relative_test(..., inference="hac")``
likewise.
"""
from __future__ import annotations

from functools import partial

from .base import TestResult, TwoSampleTest, permutation_pvalue
from .c2st import C2ST
from .critic import (CriticTest, LearnedTest, andrews_hac_lags, hac_var_mean,
                     mean_diff_inference, nw_hac_lags, perm_mean_diff, perm_pvalues, time_grid,
                     train_eval_split)
from .ks import KSTest
from .mmd import (MMDTest, median_heuristic, mmd2_biased, mmd2_unbiased, mmd2_wild_bootstrap,
                  wild_block_length)
from .nets import ConvCritic, MLPCritic, PooledConvCritic, build_network
from .relative import relative_test, relative_test_learned, relative_test_mmd
from .signature import SigMMDTest, signature, signature_features, windows_to_paths

TEST_REGISTRY = {
    "critic": CriticTest,
    "lrcritic": partial(C2ST, statistic="logit"),
    "c2st": C2ST,
    "mmd": MMDTest,
    "sigmmd": SigMMDTest,
    "ks": KSTest,
}

__all__ = [
    "TEST_REGISTRY",
    "TestResult",
    "TwoSampleTest",
    "permutation_pvalue",
    "CriticTest",
    "C2ST",
    "LearnedTest",
    "MMDTest",
    "SigMMDTest",
    "KSTest",
    "MLPCritic",
    "ConvCritic",
    "PooledConvCritic",
    "build_network",
    "relative_test",
    "relative_test_learned",
    "relative_test_mmd",
    "train_eval_split",
    "hac_var_mean",
    "andrews_hac_lags",
    "nw_hac_lags",
    "time_grid",
    "mean_diff_inference",
    "perm_mean_diff",
    "perm_pvalues",
    "median_heuristic",
    "mmd2_unbiased",
    "mmd2_biased",
    "mmd2_wild_bootstrap",
    "wild_block_length",
    "signature",
    "signature_features",
    "windows_to_paths",
]
