"""Batched counterpart of :func:`synthval.pipeline.relative_tests` for the anchored test.

The power checks of E3 and E5 run thousands of independent anchored relative tests, each
training one critic. :func:`anchored_relative_tests` takes a list of jobs (one
``pipeline.relative_tests(..., methods=("anchored",))`` call each) and trains all their critics
as one critic bank (:mod:`synthval.batched.critic`); every other step (test construction,
per-method streams, split, inference, the output row) is the sequential pipeline code itself.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import pipeline as pl
from .critic import _distinct_rngs, replayable, run_staged
from .nets import BankImpl

__all__ = ["AnchoredJob", "anchored_relative_tests"]


@dataclass
class AnchoredJob:
    """Arguments of one ``pipeline.relative_tests(R_new, G, R_past, ...)`` call.

    ``rng`` is the generator passed to that call (one object per job); it is left in the state
    the sequential call leaves it in.
    """

    R_new: np.ndarray
    G: np.ndarray
    R_past: np.ndarray
    rng: np.random.Generator
    orders: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] = (None, None, None)
    groups: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] = (None, None, None)


def anchored_relative_tests(jobs: Sequence[AnchoredJob], *, base: str | Any = "critic",
                            protocol: pl.Protocol = "dependent",
                            base_kw: Mapping[str, Any] | None = None, n_boot: int = pl.N_BOOT,
                            keep_extra: bool = False, impl: BankImpl = "grouped",
                            max_bank: int | None = None) -> list[dict[str, Any]]:
    """Anchored relative tests of many jobs with their critics trained as one bank.

    Equals ``[pipeline.relative_tests(j.R_new, j.G, j.R_past, orders=j.orders, base=base,
    methods=("anchored",), protocol=protocol, rng=j.rng, base_kw=base_kw, n_boot=n_boot,
    groups=j.groups, keep_extra=keep_extra)[0] for j in jobs]`` up to the floating-point
    rounding of the batched training (same streams, splits, initial weights, mini-batches and
    permutations; :func:`synthval.batched.critic.run_staged`).

    Parameters
    ----------
    jobs : sequence of AnchoredJob
        One relative test each (sizes may differ; one generator object per job).
    base : str or CriticTest, default "critic"
        Base test name (built with :func:`synthval.pipeline.make_test`, the protocol settings
        and ``base_kw``, as ``relative_tests`` does) or a configured ``CriticTest``. Must be a
        WGAN-GP critic without early stopping.
    protocol, base_kw, n_boot, keep_extra :
        As :func:`synthval.pipeline.relative_tests`.
    impl : {"grouped", "loop"}, default "grouped"
        Bank evaluation (:class:`synthval.batched.nets.CriticBank`).
    max_bank : int, optional
        Largest number of critics per bank (default: all jobs in one bank).

    Returns
    -------
    list of dict
        One ``relative_tests`` row per job. ``runtime_s`` is the job's own wall time (its
        staging and final runs) plus an equal share of the bank's training time. With
        ``keep_extra`` the ``extra["train_info"]`` dict also holds ``batched``, ``bank_size``,
        ``bank_impl`` and ``bank_train_time_s``.
    """
    jobs = list(jobs)
    _distinct_rngs([j.rng for j in jobs])
    b = pl.make_test(base, protocol, **dict(base_kw or {})) if isinstance(base, str) else base

    def make(j: AnchoredJob) -> Any:
        return replayable(lambda bb, r: pl.relative_tests(
            j.R_new, j.G, j.R_past, orders=j.orders, base=bb, methods=("anchored",),
            protocol=protocol, rng=r, base_kw=base_kw, n_boot=n_boot, groups=j.groups,
            keep_extra=keep_extra)[0], j.rng)

    rows, seconds = run_staged(b, [make(j) for j in jobs], impl=impl, max_bank=max_bank)
    for row, s in zip(rows, seconds):
        row["runtime_s"] = s
    return rows
