"""Base classes for generative return models (daily log-returns, real-world measure)."""
from __future__ import annotations

import numpy as np


class ReturnModel:
    """Univariate model of daily log-returns.

    Subclasses implement ``fit`` and ``simulate``; models with a tractable likelihood
    (``has_likelihood = True``) also implement ``loglik_obs`` and ``pit``.
    """

    name: str = "base"
    has_likelihood: bool = False

    def fit(self, r: np.ndarray) -> "ReturnModel":
        raise NotImplementedError

    def simulate(
        self,
        n_steps: int,
        n_paths: int,
        *,
        rng: np.random.Generator,
        burn_in: int = 250,
        start_from_end: bool = False,
    ) -> np.ndarray:
        """Simulate ``(n_paths, n_steps)`` daily log-returns."""
        raise NotImplementedError

    def loglik_obs(self, r: np.ndarray, r_history: np.ndarray | None = None) -> np.ndarray:
        """Per-observation conditional log-densities, shape ``(T,)``.

        ``r_history`` (e.g. the training returns) initialises filtered models (GARCH).
        """
        raise NotImplementedError(f"{self.name}: likelihood is intractable")

    def loglik(self, r: np.ndarray, r_history: np.ndarray | None = None) -> float:
        return float(np.sum(self.loglik_obs(r, r_history=r_history)))

    def pit(self, r: np.ndarray, r_history: np.ndarray | None = None) -> np.ndarray:
        """Probability integral transform u_t = F_t(r_t | past), values in (0, 1)."""
        raise NotImplementedError(f"{self.name}: conditional CDF is intractable")

    def params(self) -> dict[str, float]:
        raise NotImplementedError

    def n_params(self) -> int:
        return len(self.params())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        try:
            p = ", ".join(f"{k}={v:.4g}" for k, v in self.params().items())
        except Exception:
            p = "unfitted"
        return f"{type(self).__name__}({p})"


class MultiReturnModel:
    """Multivariate model of daily log-returns of N assets."""

    name: str = "mv_base"

    def fit(self, R: np.ndarray) -> "MultiReturnModel":
        """Fit on returns ``R`` of shape ``(T, N)``."""
        raise NotImplementedError

    def simulate(self, n_steps: int, *, rng: np.random.Generator, burn_in: int = 250) -> np.ndarray:
        """Simulate ``(n_steps, N)`` daily log-returns."""
        raise NotImplementedError

    def params(self) -> dict:
        raise NotImplementedError
