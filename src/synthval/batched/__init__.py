"""Batched learned tests: many independent WGAN-GP critics trained in one process as one bank.

On a GPU each experiment worker process costs about 1.8 GB of host RAM for the CUDA libraries,
so only a few workers fit in memory and a small critic (1000 Adam steps, batch 128) leaves the GPU
mostly idle. This package trains the ``K`` critics of ``K`` independent tests (e.g. the 2 x
``n_seeds`` anchored relative tests of one power-check replication) as one network, a *critic
bank*, with the same results as the sequential code up to floating-point rounding:

* :mod:`synthval.batched.nets`: :class:`PooledConvCriticBank` (grouped ``Conv1d``) and
  :class:`MLPCriticBank` (``bmm``); input ``(n, K, *sample_shape)``, output ``(K, n)``.
* :mod:`synthval.batched.critic`: :func:`fit_witnesses` (batched ``fit_witness``: same initial
  weights and per-problem random streams as the sequential path), :func:`run_staged` (batch
  whole sequential procedures), :func:`critic_tests` (absolute ``CriticTest``) and
  :func:`relative_tests_anchored` (anchored relative test).
* :mod:`synthval.batched.pipeline`: :func:`anchored_relative_tests`, the batched
  ``pipeline.relative_tests(..., methods=("anchored",))`` used by E3's and E5's power designs.

The sequential code paths are untouched and remain the default. The experiments opt in with the
environment variable ``SYNTHVAL_BATCHED=1`` (:func:`batched_enabled`); ``SYNTHVAL_BANK_MAX``
caps the critics per bank (:func:`bank_max`). The package lives outside ``synthval.twosample``
and ``synthval.pipeline`` on purpose: E2 and E5 key their checkpoints on the sources of those
modules, which therefore stay unchanged (E1's key hashes every ``synthval`` source file, so it
changes with this package as with any addition to the library; E1 never calls it).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from ..utils import file_sha256
from .critic import (RelativeProblem, TwoSampleProblem, WitnessProblem, check_bankable,
                     critic_tests, fit_witnesses, relative_tests_anchored, replayable,
                     run_staged)
from .nets import CriticBank, MLPCriticBank, PooledConvCriticBank, build_bank
from .pipeline import AnchoredJob, anchored_relative_tests

__all__ = [
    "batched_enabled", "bank_max", "batched_key", "batched_code_hash",
    "CriticBank", "PooledConvCriticBank", "MLPCriticBank", "build_bank",
    "WitnessProblem", "TwoSampleProblem", "RelativeProblem", "check_bankable", "fit_witnesses",
    "run_staged", "replayable", "critic_tests", "relative_tests_anchored",
    "AnchoredJob", "anchored_relative_tests",
]

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("", "0", "false", "no", "off")


def batched_enabled() -> bool:
    """Whether the experiments train independent critics as banks: ``$SYNTHVAL_BATCHED``
    (``1``/``true``/``yes``/``on``; default off)."""
    v = os.environ.get("SYNTHVAL_BATCHED", "").strip().lower()
    if v in _FALSE:
        return False
    if v in _TRUE:
        return True
    raise ValueError(f"$SYNTHVAL_BATCHED={v!r}: use 1 or 0")


def bank_max() -> int | None:
    """Largest number of critics per bank, ``$SYNTHVAL_BANK_MAX`` (unset or 0: no cap)."""
    v = os.environ.get("SYNTHVAL_BANK_MAX", "").strip()
    if not v:
        return None
    n = int(v)
    if n < 0:
        raise ValueError("$SYNTHVAL_BANK_MAX must be >= 0")
    return n or None


def batched_code_hash() -> str:
    """SHA-256 (16 hex digits) of the sources of this package."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(root.glob("*.py")):
        h.update(f.name.encode() + b"\0" + file_sha256(f).encode() + b"\n")
    return h.hexdigest()[:16]


def batched_key(code: bool = False) -> dict[str, Any]:
    """Checkpoint-key entry of the batched mode: ``{}`` when it is off (keys unchanged),
    ``{"batched": {...}}`` when it is on (batched results are rounding-level different from
    sequential ones, like GPU from CPU results, so they get their own checkpoints).

    Parameters
    ----------
    code : bool, default False
        Include :func:`batched_code_hash` (for experiments whose keys hash the library code).
    """
    if not batched_enabled():
        return {}
    entry: dict[str, Any] = {"mode": "critic_bank"}
    if code:
        entry["code"] = batched_code_hash()
    return {"batched": entry}
