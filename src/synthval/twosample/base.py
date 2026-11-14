"""Common types for two-sample tests."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TestResult:
    """Outcome of a statistical test."""

    __test__ = False  # keep pytest from collecting this class

    name: str
    statistic: float
    pvalue: float
    n_x: int = 0
    n_y: int = 0
    extra: dict = field(default_factory=dict)

    def reject(self, alpha: float = 0.05) -> bool:
        return bool(self.pvalue <= alpha)


class TwoSampleTest:
    """Interface: ``test(X, Y, rng=rng, groups_x=None, groups_y=None) -> TestResult``.

    ``X`` is the real sample and ``Y`` the synthetic one, shapes ``(n, L)``, ``(n, N, L)``
    or ``(n, d)``. ``groups_*`` are optional integer labels (e.g. tickers); learned tests split
    train/eval by group when they are given. ``order_*`` are optional time indices used for
    HAC-studentised statistics robust to serial dependence between windows.
    """

    __test__ = False
    name: str = "base"

    def __call__(self, X: np.ndarray, Y: np.ndarray, *, rng: np.random.Generator,
                 groups_x: np.ndarray | None = None,
                 groups_y: np.ndarray | None = None,
                 order_x: np.ndarray | None = None,
                 order_y: np.ndarray | None = None) -> TestResult:
        raise NotImplementedError


def permutation_pvalue(t_obs: float, t_perm: np.ndarray) -> float:
    """One-sided permutation p-value ``(1 + #{T_b >= T}) / (1 + B)`` (large T = evidence against H0)."""
    t_perm = np.asarray(t_perm, dtype=float)
    return float((1 + np.sum(t_perm >= t_obs)) / (1 + t_perm.size))
