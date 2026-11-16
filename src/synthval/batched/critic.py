"""Batched WGAN-GP critic tests: many independent critics trained as one critic bank.

Sequential reference: :meth:`synthval.twosample.CriticTest.fit_witness` (one critic per call).
:func:`fit_witnesses` trains the critics of ``K`` problems together and returns, for every
problem, the witness that the sequential call would return up to floating-point rounding:

* **Initial weights.** For problem ``k`` the seed is drawn from the problem's generator by
  :func:`~synthval.twosample.nets.torch_seed`, and the single network is built under that seed
  with :func:`~synthval.twosample.nets.build_network`, exactly as ``fit_witness`` does; its
  parameters are copied into slice ``k`` of the bank (:mod:`synthval.batched.nets`).
* **Random streams.** Right after the initialisation, inside the forked torch RNG, the global
  CPU generator state is copied into a dedicated ``torch.Generator`` of problem ``k``. Every
  mini-batch index (``randint`` or, with weights, ``multinomial``) and gradient-penalty
  interpolation weight (``rand``) of problem ``k`` is then drawn from that generator in the
  sequential order (step by step: real indices, synthetic indices, interpolation weights), so
  each problem sees exactly the draws of the sequential loop. Problems may have different
  sample sizes and weights.
* **Optimisation.** The bank's loss is the sum of the ``K`` sequential losses; critic ``k``
  depends only on its own inputs and parameters, so its gradient is exactly its own, and Adam
  (elementwise, same hyperparameters, same step count) is ``K`` independent Adams. The trained
  slices are copied back into the single networks, which are then frozen and wrapped in a
  :class:`~synthval.twosample.nets.Witness` (same evaluation code as the sequential path).

The only difference from the sequential path is rounding (grouped vs single convolutions,
``bmm`` vs ``addmm``), which training then propagates; ``impl="loop"`` replaces the grouped pass
by the single networks' own operations (slow, for validation).

Whole tests are batched by **staging** the unchanged sequential procedure (:func:`run_staged`):
each procedure (e.g. :func:`synthval.twosample.relative.relative_test_anchored` on one problem)
runs once with a ``fit_witness`` that records its inputs and stops it; all recorded problems are
trained as one bank; then each procedure runs again from the start with a ``fit_witness`` that
checks the inputs and the generator state and returns the trained witness. Splits, statistics,
permutation / HAC inference and the ``extra`` fields are therefore computed by the sequential
code itself, and every generator ends in the state the sequential call leaves it in.

Limitations: only :class:`~synthval.twosample.CriticTest` (the WGAN-GP critic; not C2ST /
lrcritic), without early stopping; :class:`~synthval.twosample.nets.ConvCritic` has no bank
(``arch="pooled"`` or ``"mlp"``).
"""
from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import numpy as np
import torch

from ..twosample.base import TestResult
from ..twosample.critic import CriticTest, LearnedTest, _check_weights, check_samples
from ..twosample.nets import (Standardizer, Witness, build_network, get_device, to_tensor,
                              torch_seed)
from ..twosample.relative import relative_test_anchored
from .nets import BankImpl, CriticBank, build_bank

R = TypeVar("R")
#: ``trace(problem_index, step, ix, iy, eps)``: called with the draws of every problem and step.
TraceFn = Callable[[int, int, torch.Tensor, torch.Tensor, torch.Tensor], None]

__all__ = ["WitnessProblem", "TwoSampleProblem", "RelativeProblem", "check_bankable",
           "fit_witnesses", "bank_gp_terms", "run_staged", "replayable", "critic_tests",
           "relative_tests_anchored"]


@dataclass
class WitnessProblem:
    """Arguments of one :meth:`~synthval.twosample.LearnedTest.fit_witness` call.

    ``rng`` is in the state in which ``fit_witness`` would receive it; it is advanced as
    ``fit_witness`` advances it (one seed draw). Each problem needs its own generator object.
    """

    X_tr: np.ndarray
    Y_tr: np.ndarray
    rng: np.random.Generator
    weights_x: np.ndarray | None = None
    weights_y: np.ndarray | None = None


@dataclass
class TwoSampleProblem:
    """Arguments of one absolute critic test ``base(X, Y, rng=rng, ...)``."""

    X: np.ndarray
    Y: np.ndarray
    rng: np.random.Generator
    groups_x: np.ndarray | None = None
    groups_y: np.ndarray | None = None
    order_x: np.ndarray | None = None
    order_y: np.ndarray | None = None


@dataclass
class RelativeProblem:
    """Arguments of one anchored relative test :func:`relative_test_anchored`."""

    R_new: np.ndarray
    G: np.ndarray
    R_past: np.ndarray
    rng: np.random.Generator
    groups_new: np.ndarray | None = None
    groups_g: np.ndarray | None = None
    groups_past: np.ndarray | None = None
    order_new: np.ndarray | None = None
    order_g: np.ndarray | None = None
    order_past: np.ndarray | None = None


# ----------------------------------------------------------------------------------------------
# Batched fit_witness
# ----------------------------------------------------------------------------------------------
def check_bankable(base: Any) -> None:
    """Raise unless ``base`` is a :class:`CriticTest` whose training a critic bank reproduces.

    Raises
    ------
    TypeError
        ``base`` is not a ``CriticTest`` or overrides its training or witness fitting.
    NotImplementedError
        ``base.early_stopping`` is set (sequential only).
    """
    if not isinstance(base, CriticTest):
        raise TypeError("critic banks train WGAN-GP critics: base must be a CriticTest, got "
                        f"{type(base).__name__}")
    cls = type(base)
    if (cls._train is not CriticTest._train or cls._gp_terms is not CriticTest._gp_terms
            or cls.fit_witness is not LearnedTest.fit_witness
            or "fit_witness" in vars(base)):
        raise TypeError(f"{cls.__name__} overrides the critic training; no batched version")
    if base.early_stopping:
        raise NotImplementedError("early stopping is not available in batched mode; run the "
                                  "sequential test")


@dataclass
class _Prepared:
    """One problem after standardisation and initialisation (what ``fit_witness`` holds)."""

    Xs: np.ndarray
    Ys: np.ndarray
    wx: torch.Tensor | None
    wy: torch.Tensor | None
    std: Standardizer
    net: torch.nn.Module
    gen: torch.Generator
    weighted: bool
    rng_state: dict[str, Any]
    prep_s: float


def _prepare(base: CriticTest, p: WitnessProblem) -> _Prepared:
    """The steps of ``fit_witness`` before training (no early stopping), for one problem."""
    t0 = time.perf_counter()
    X_tr, Y_tr = check_samples(p.X_tr, p.Y_tr)
    wx = _check_weights(p.weights_x, X_tr.shape[0], "weights_x")
    wy = _check_weights(p.weights_y, Y_tr.shape[0], "weights_y")
    weighted = wx is not None or wy is not None
    std = Standardizer().fit(np.concatenate([X_tr, Y_tr], axis=0))
    Xs, Ys = std.transform(X_tr), std.transform(Y_tr)
    with torch_seed(p.rng):
        net = build_network(X_tr.shape[1:], base.arch, hidden=base.hidden, depth=base.depth,
                            layer_norm=base.layer_norm).to(get_device())
        gen = torch.Generator()
        gen.set_state(torch.get_rng_state())  # the global state the sequential loop draws from
    tw = [None if w is None else torch.as_tensor(w, dtype=torch.float64) for w in (wx, wy)]
    return _Prepared(Xs=Xs, Ys=Ys, wx=tw[0], wy=tw[1], std=std, net=net, gen=gen,
                     weighted=weighted, rng_state=p.rng.bit_generator.state,
                     prep_s=time.perf_counter() - t0)


def bank_gp_terms(bank: CriticBank, xb: torch.Tensor, yb: torch.Tensor,
                  eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched :meth:`CriticTest._gp_terms`: per-critic ``mean f(xb) - mean f(yb)`` and GP.

    Parameters
    ----------
    bank : CriticBank
        ``K`` critics.
    xb, yb : tensors (b, K, ...)
        Real and synthetic mini-batches (column ``k`` for critic ``k``).
    eps : tensor (b, K)
        Interpolation weights.

    Returns
    -------
    wd, gp : tensors (K,)
    """
    b, K = xb.shape[0], xb.shape[1]
    shape = (b, K) + (1,) * (xb.dim() - 2)
    e = eps.to(device=xb.device, dtype=xb.dtype).view(shape)
    x_hat = (e * xb + (1.0 - e) * yb).requires_grad_(True)
    out = bank(torch.cat([xb, yb, x_hat], dim=0))                                # (K, 3b)
    f_x, f_y, f_hat = out[:, :b], out[:, b:2 * b], out[:, 2 * b:]
    grad = torch.autograd.grad(f_hat.sum(), x_hat, create_graph=True)[0]
    gp = ((grad.reshape(b, K, -1).norm(2, dim=2) - 1.0) ** 2).t().contiguous().mean(dim=1)
    return f_x.mean(dim=1) - f_y.mean(dim=1), gp


def _draw_chunk(n_steps: int, start: int, b: int, nx: list[int], ny: list[int],
                wx: list[torch.Tensor | None], wy: list[torch.Tensor | None],
                gens: list[torch.Generator], trace: TraceFn | None,
                ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draws of steps ``start + 1 .. start + n_steps`` of every problem, shape (steps, b, K).

    Problem ``k`` draws from its own generator in the order of ``CriticTest._train``: for each
    step the real indices, the synthetic indices, then the interpolation weights.
    """
    K = len(gens)
    IX = torch.empty(n_steps, b, K, dtype=torch.int64)
    IY = torch.empty(n_steps, b, K, dtype=torch.int64)
    EPS = torch.empty(n_steps, b, K, dtype=torch.float32)
    for k in range(K):
        g = gens[k]
        xs, ys, es = [], [], []
        for s in range(n_steps):
            ix = (torch.randint(0, nx[k], (b,), generator=g) if wx[k] is None
                  else torch.multinomial(wx[k], b, replacement=True, generator=g))
            iy = (torch.randint(0, ny[k], (b,), generator=g) if wy[k] is None
                  else torch.multinomial(wy[k], b, replacement=True, generator=g))
            e = torch.rand(b, generator=g)
            if trace is not None:
                trace(ids[k], start + s + 1, ix, iy, e)
            xs.append(ix)
            ys.append(iy)
            es.append(e)
        IX[:, :, k] = torch.stack(xs)
        IY[:, :, k] = torch.stack(ys)
        EPS[:, :, k] = torch.stack(es)
    return IX, IY, EPS


def _curve(base: CriticTest, wd: list[float], gp: list[float]) -> dict[str, list[float]]:
    """Training curve of one critic, accumulated exactly as ``CriticTest._train`` does."""
    T = base.n_steps
    chunk = max(1, T // max(1, base.curve_points * 2))
    curve: dict[str, list[float]] = {"step": [], "wdist": [], "gp": []}
    acc_w = acc_gp = 0.0
    acc_n = 0
    for step in range(1, T + 1):
        acc_w += wd[step - 1]
        acc_gp += gp[step - 1]
        acc_n += 1
        if step % chunk == 0 or step == T:
            curve["step"].append(step)
            curve["wdist"].append(acc_w / acc_n)
            curve["gp"].append(acc_gp / acc_n)
            acc_w = acc_gp = 0.0
            acc_n = 0
    return curve


def _train_bank(base: CriticTest, preps: list[_Prepared], impl: BankImpl, chunk_steps: int,
                trace: TraceFn | None, ids: list[int]) -> list[tuple[Witness, dict[str, Any]]]:
    """Train the critics of ``preps`` as one bank; return witnesses and ``fit_witness`` infos."""
    t0 = time.perf_counter()
    K, T, b = len(preps), base.n_steps, base.batch_size
    bank = build_bank([p.net for p in preps], impl)
    nx = [p.Xs.shape[0] for p in preps]
    ny = [p.Ys.shape[0] for p in preps]
    Xall = to_tensor(np.concatenate([p.Xs for p in preps], axis=0))
    Yall = to_tensor(np.concatenate([p.Ys for p in preps], axis=0))
    dev = Xall.device
    offx = torch.as_tensor(np.cumsum([0] + nx[:-1]), dtype=torch.int64)
    offy = torch.as_tensor(np.cumsum([0] + ny[:-1]), dtype=torch.int64)
    wx, wy, gens = [p.wx for p in preps], [p.wy for p in preps], [p.gen for p in preps]
    opt = torch.optim.Adam(bank.parameters(), lr=base.lr, betas=base.betas)
    wd_hist = torch.empty(T, K, device=dev)
    gp_hist = torch.empty(T, K, device=dev)
    bank.train()
    for start in range(0, T, chunk_steps):
        S = min(chunk_steps, T - start)
        IX, IY, EPS = _draw_chunk(S, start, b, nx, ny, wx, wy, gens, trace, ids)
        IX += offx
        IY += offy
        if dev.type == "cuda":  # pinned host memory: the copies overlap with queued GPU work
            IX, IY, EPS = (t.pin_memory().to(dev, non_blocking=True) for t in (IX, IY, EPS))
        elif dev.type != "cpu":
            IX, IY, EPS = (t.to(dev) for t in (IX, IY, EPS))
        for s in range(S):
            wd, gp = bank_gp_terms(bank, Xall[IX[s]], Yall[IY[s]], EPS[s])
            loss = (-wd + base.gp_lambda * gp).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            wd_hist[start + s] = wd.detach()
            gp_hist[start + s] = gp.detach()
    bank.eval()
    wd_rows, gp_rows = wd_hist.cpu().tolist(), gp_hist.cpu().tolist()
    for k, p in enumerate(preps):
        bank.store_critic(k, p.net)
        for q in p.net.parameters():
            q.requires_grad_(False)
    elapsed = time.perf_counter() - t0
    out = []
    for k, p in enumerate(preps):
        info: dict[str, Any] = {"train_curve": _curve(base, [r[k] for r in wd_rows],
                                                      [r[k] for r in gp_rows]),
                                "steps_run": T}
        if p.weighted:
            info["weighted"] = True
        info["train_curve"] = base._subsample_curve(info["train_curve"])
        info["train_time_s"] = p.prep_s + elapsed / K
        info["arch"] = type(p.net).__name__
        info.update(batched=True, bank_size=K, bank_impl=impl, bank_train_time_s=elapsed)
        out.append((Witness(p.net, p.std), info))
    return out


def _fit_all(base: CriticTest, problems: Sequence[WitnessProblem], *, impl: BankImpl,
             max_bank: int | None, chunk_steps: int, trace: TraceFn | None
             ) -> list[tuple[Witness, dict[str, Any], dict[str, Any]]]:
    """:func:`fit_witnesses` plus the generator state after the seed draw of each problem."""
    check_bankable(base)
    if impl not in ("grouped", "loop"):
        raise ValueError("impl must be 'grouped' or 'loop'")
    if max_bank is not None and int(max_bank) < 1:
        raise ValueError("max_bank must be >= 1")
    if chunk_steps < 1:
        raise ValueError("chunk_steps must be >= 1")
    probs = list(problems)
    if len({id(p.rng) for p in probs}) != len(probs):
        raise ValueError("every problem needs its own generator (a shared generator makes the "
                         "sequential results depend on the order of the problems)")
    preps = [_prepare(base, p) for p in probs]
    by_shape: dict[tuple[int, ...], list[int]] = {}
    for i, p in enumerate(preps):
        by_shape.setdefault(tuple(p.Xs.shape[1:]), []).append(i)
    out: list[Any] = [None] * len(probs)
    for idx in by_shape.values():
        size = len(idx) if max_bank is None else int(max_bank)
        for s in range(0, len(idx), size):
            chunk = idx[s:s + size]
            res = _train_bank(base, [preps[i] for i in chunk], impl, chunk_steps, trace, chunk)
            for i, (w, info) in zip(chunk, res):
                out[i] = (w, info, preps[i].rng_state)
    return out


def fit_witnesses(base: CriticTest, problems: Sequence[WitnessProblem], *,
                  impl: BankImpl = "grouped", max_bank: int | None = None,
                  chunk_steps: int = 50, trace: TraceFn | None = None
                  ) -> list[tuple[Witness, dict[str, Any]]]:
    """Batched :meth:`~synthval.twosample.LearnedTest.fit_witness` of a :class:`CriticTest`.

    Equivalent, up to floating-point rounding, to ``[base.fit_witness(p.X_tr, p.Y_tr,
    rng=p.rng, weights_x=p.weights_x, weights_y=p.weights_y) for p in problems]`` (same initial
    weights, same mini-batch indices and interpolation weights per problem, independent Adams;
    see the module docstring), with every generator advanced as by that call.

    Parameters
    ----------
    base : CriticTest
        Configured critic test (architecture, optimiser, ``n_steps``, ``batch_size``,
        ``gp_lambda``); no early stopping.
    problems : sequence of WitnessProblem
        Training samples, generator and optional sampling weights of each problem (sizes and
        weights may differ; one generator object per problem).
    impl : {"grouped", "loop"}, default "grouped"
        Bank evaluation (:class:`synthval.batched.nets.CriticBank`).
    max_bank : int, optional
        Largest number of critics per bank (memory cap); default: one bank per sample shape.
    chunk_steps : int, default 50
        Steps whose random draws are made (and moved to the device) at once.
    trace : callable, optional
        ``trace(problem_index, step, ix, iy, eps)`` receives every draw (local indices).

    Returns
    -------
    list of (Witness, dict)
        Per problem, as ``fit_witness``; ``info`` also has ``batched=True``, ``bank_size``,
        ``bank_impl`` and ``bank_train_time_s`` (wall time of the whole bank; ``train_time_s``
        is the problem's own preparation time plus its share ``bank_train_time_s / bank_size``).
    """
    return [(w, info) for w, info, _ in _fit_all(base, problems, impl=impl, max_bank=max_bank,
                                                 chunk_steps=chunk_steps, trace=trace)]


# ----------------------------------------------------------------------------------------------
# Staging: batch whole sequential procedures that fit one critic each
# ----------------------------------------------------------------------------------------------
class _Captured(BaseException):
    """Stops a staged procedure at its critic fit (BaseException: not caught by ``except
    Exception`` handlers inside the procedure)."""


def _arrays_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _capturing(base: CriticTest, box: list[WitnessProblem]) -> CriticTest:
    proxy = copy.copy(base)

    def fit_witness(X_tr: np.ndarray, Y_tr: np.ndarray, *, rng: np.random.Generator,
                    weights_x: np.ndarray | None = None,
                    weights_y: np.ndarray | None = None) -> tuple[Witness, dict[str, Any]]:
        box.append(WitnessProblem(X_tr, Y_tr, rng, weights_x, weights_y))
        raise _Captured

    proxy.fit_witness = fit_witness  # type: ignore[method-assign]
    return proxy


def _replaying(base: CriticTest, prob: WitnessProblem, witness: Witness, info: dict[str, Any],
               rng_state: dict[str, Any]) -> CriticTest:
    proxy = copy.copy(base)
    used: list[bool] = []

    def fit_witness(X_tr: np.ndarray, Y_tr: np.ndarray, *, rng: np.random.Generator,
                    weights_x: np.ndarray | None = None,
                    weights_y: np.ndarray | None = None) -> tuple[Witness, dict[str, Any]]:
        if used:
            raise RuntimeError("a staged procedure must fit exactly one critic")
        used.append(True)
        if not (_arrays_equal(X_tr, prob.X_tr) and _arrays_equal(Y_tr, prob.Y_tr)
                and _arrays_equal(weights_x, prob.weights_x)
                and _arrays_equal(weights_y, prob.weights_y)):
            raise RuntimeError("replay: the procedure's training data differ from the staged "
                               "run (the procedure is not repeatable)")
        with torch_seed(rng):  # advance rng (and seed torch) exactly as fit_witness does
            pass
        if rng.bit_generator.state != rng_state:
            raise RuntimeError("replay: the generator state differs from the staged run")
        return witness, dict(info)

    proxy.fit_witness = fit_witness  # type: ignore[method-assign]
    return proxy


def run_staged(base: CriticTest, procedures: Sequence[Callable[[CriticTest], R]], *,
               impl: BankImpl = "grouped", max_bank: int | None = None,
               chunk_steps: int = 50) -> tuple[list[R], list[float]]:
    """Run procedures that each fit one critic with ``base``, training all critics as a bank.

    ``procedures[k](b)`` must run a sequential procedure with ``b`` in place of ``base`` and
    call ``b.fit_witness`` exactly once; it is called twice (staging, then replay) and must
    therefore be repeatable: the second call must start from the random state of the first
    (see :func:`replayable`). The result of the second call is returned; it equals the
    sequential procedure's result up to the rounding of the batched training.

    Parameters
    ----------
    base : CriticTest
        The critic test the procedures use (see :func:`check_bankable`).
    procedures : sequence of callables
        One per problem.
    impl, max_bank, chunk_steps :
        See :func:`fit_witnesses`.

    Returns
    -------
    results : list
        Procedure results, in order.
    seconds : list of float
        Wall time attributable to each procedure: its two runs plus an equal share of the bank.
    """
    check_bankable(base)
    problems: list[WitnessProblem] = []
    t_stage: list[float] = []
    for proc in procedures:
        box: list[WitnessProblem] = []
        t0 = time.perf_counter()
        try:
            proc(_capturing(base, box))
        except _Captured:
            pass
        else:
            raise RuntimeError("a staged procedure returned without fitting a critic")
        t_stage.append(time.perf_counter() - t0)
        problems.append(box[0])
    if not problems:
        return [], []
    t0 = time.perf_counter()
    fitted = _fit_all(base, problems, impl=impl, max_bank=max_bank, chunk_steps=chunk_steps,
                      trace=None)
    share = (time.perf_counter() - t0) / len(problems)
    results: list[R] = []
    seconds: list[float] = []
    for proc, prob, (w, info, state), ts in zip(procedures, problems, fitted, t_stage):
        t0 = time.perf_counter()
        results.append(proc(_replaying(base, prob, w, info, state)))
        seconds.append(ts + share + time.perf_counter() - t0)
    return results, seconds


def replayable(fn: Callable[[CriticTest, np.random.Generator], R],
               rng: np.random.Generator) -> Callable[[CriticTest], R]:
    """Procedure for :func:`run_staged` from ``fn(base, rng)``: the staging call gets a copy of
    ``rng``, the replay gets ``rng`` itself (left in the state of the sequential call)."""
    calls = [0]

    def proc(b: CriticTest) -> R:
        r = copy.deepcopy(rng) if calls[0] == 0 else rng
        calls[0] += 1
        return fn(b, r)

    return proc


def _distinct_rngs(rngs: Sequence[np.random.Generator]) -> None:
    if len({id(r) for r in rngs}) != len(rngs):
        raise ValueError("every problem needs its own generator (a shared generator makes the "
                         "sequential results depend on the order of the problems)")


def critic_tests(base: CriticTest, problems: Sequence[TwoSampleProblem], *,
                 impl: BankImpl = "grouped", max_bank: int | None = None) -> list[TestResult]:
    """Batched absolute critic tests: ``[base(p.X, p.Y, rng=p.rng, ...) for p in problems]``.

    Every result equals the sequential one up to the rounding of the batched training (same
    split, initial weights, mini-batches, permutations; see :func:`run_staged`); ``extra`` also
    holds the bank fields of :func:`fit_witnesses`.
    """
    problems = list(problems)
    _distinct_rngs([p.rng for p in problems])

    def make(p: TwoSampleProblem) -> Callable[[CriticTest], TestResult]:
        return replayable(lambda b, r: b(p.X, p.Y, rng=r, groups_x=p.groups_x,
                                         groups_y=p.groups_y, order_x=p.order_x,
                                         order_y=p.order_y), p.rng)

    return run_staged(base, [make(p) for p in problems], impl=impl, max_bank=max_bank)[0]


def relative_tests_anchored(problems: Sequence[RelativeProblem], base: CriticTest, *,
                            ci_level: float = 0.95,
                            inference: Literal["default", "hac"] | None = None,
                            studentized: bool = True, impl: BankImpl = "grouped",
                            max_bank: int | None = None) -> list[TestResult]:
    """Batched :func:`~synthval.twosample.relative.relative_test_anchored` over problems.

    Equals ``[relative_test_anchored(p.R_new, p.G, p.R_past, base, rng=p.rng, ...) for p in
    problems]`` up to the rounding of the batched training (each problem's critic separates its
    ``R_new`` from its weighted pool ``{G_tr, R_past_tr}``; one bank trains them all).
    """
    problems = list(problems)
    _distinct_rngs([p.rng for p in problems])

    def make(p: RelativeProblem) -> Callable[[CriticTest], TestResult]:
        return replayable(lambda b, r: relative_test_anchored(
            p.R_new, p.G, p.R_past, b, rng=r, groups_new=p.groups_new, groups_g=p.groups_g,
            groups_past=p.groups_past, order_new=p.order_new, order_g=p.order_g,
            order_past=p.order_past, ci_level=ci_level, inference=inference,
            studentized=studentized), p.rng)

    return run_staged(base, [make(p) for p in problems], impl=impl, max_bank=max_bank)[0]
