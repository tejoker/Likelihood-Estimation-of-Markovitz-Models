"""Two-sample Kolmogorov-Smirnov test on pooled, flattened returns (baseline).

All returns of all windows (and all assets for multivariate windows) are pooled into one
univariate sample per side. The test therefore only sees the one-dimensional marginal law of a
daily return: it ignores serial dependence (volatility clustering), cross-sectional dependence
and the window structure, and its nominal p-value assumes i.i.d. observations, which is not true
for overlapping or dependent returns (the test is then over-sized). It is kept as the baseline
practitioners use.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats

from .base import TestResult, TwoSampleTest
from .critic import check_samples

__all__ = ["KSTest"]


class KSTest(TwoSampleTest):
    """Two-sample KS test on pooled flattened returns.

    Parameters
    ----------
    method : {"auto", "exact", "asymp"}, default "asymp"
        Passed to :func:`scipy.stats.ks_2samp`. The asymptotic distribution is accurate for the
        sample sizes used here (thousands of returns) and much faster than the exact one.
    """

    name = "ks"

    def __init__(self, method: Literal["auto", "exact", "asymp"] = "asymp") -> None:
        self.method = method

    def __call__(self, X: np.ndarray, Y: np.ndarray, *, rng: np.random.Generator | None = None,
                 groups_x: np.ndarray | None = None, groups_y: np.ndarray | None = None,
                 order_x: np.ndarray | None = None,
                 order_y: np.ndarray | None = None) -> TestResult:
        """Run the test (deterministic; ``rng``, ``groups_*``, ``order_*`` are ignored)."""
        X, Y = check_samples(X, Y)
        x, y = X.ravel(), Y.ravel()
        res = stats.ks_2samp(x, y, alternative="two-sided", method=self.method)
        return TestResult(name=self.name, statistic=float(res.statistic),
                          pvalue=float(res.pvalue), n_x=int(X.shape[0]), n_y=int(Y.shape[0]),
                          extra={"n_obs_x": int(x.size), "n_obs_y": int(y.size),
                                 "method": self.method})
