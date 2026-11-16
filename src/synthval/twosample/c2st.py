"""Classifier two-sample test (C2ST) and the likelihood-ratio critic.

A binary classifier ``c(x) = sigmoid(s(x))`` is trained with the cross-entropy loss to
separate real (label 1) from synthetic (label 0) training windows, with **balanced
mini-batches** (the same number of real and synthetic points in every batch, whatever the
sample sizes). With balanced classes the Bayes-optimal logit is the log density ratio
``s*(x) = log p_real(x) / p_synth(x)`` (Goodfellow et al. 2014), so the learned logit is an
estimate of the log-likelihood ratio.

Statistics on the held-out evaluation half, with the classifier frozen:

* ``statistic="accuracy"`` (default; Lopez-Paz & Oquab 2017): the *balanced* accuracy
  ``(TPR + TNR) / 2`` of the rule ``s(x) > 0`` (equal to the plain accuracy when the evaluation
  halves have equal sizes). Writing ``h = 1{s > 0}``, ``acc = 1/2 + (mean h(X_ev) - mean
  h(Y_ev)) / 2``, a difference of means of a fixed function.
* ``statistic="logit"`` (the **likelihood-ratio critic**): ``mean s(X_ev) - mean s(Y_ev)``, a
  learned estimate of ``KL(P||Q) + KL(Q||P)`` (cf. Cheng & Cloninger 2022).

p-values: permutation of the pooled evaluation labels with the classifier fixed. As for the
critic (:mod:`synthval.twosample.critic`), this is exact **only if the windows are i.i.d.**
(exchangeable evaluation points under H0); with serially dependent windows (consecutive windows
of one persistent-volatility series) it over-rejects, and ``inference="hac"`` reports the
HAC-studentised p-value instead (pass ``order_x`` / ``order_y``).

**Discreteness of the accuracy.** The accuracy takes few values, so the conservative
permutation p-value ``(1 + #{T_b >= T}) / (1 + B)`` has size well below alpha: with 250 + 250
evaluation points the exact size at nominal 5% is about 4.0%, and Monte Carlo rejection rates
of about 2% were observed under H0 with the default early stopping. The default for the
accuracy is therefore ``tie_break="randomized"``: ties with the observed value are broken at
random, which makes the p-value exactly uniform under H0 (exact size, see
:func:`~synthval.twosample.critic.perm_pvalues`); ``extra`` also holds the conservative and
mid-p versions. The logit statistic is continuous and keeps the DESIGN formula.

With ``n_perm=0`` the accuracy uses the one-sided binomial test of ``#correct`` against
``Bin(m, 1/2)`` when the evaluation halves have equal size (conservative under H0 beyond
discreteness, because ``Var(#correct) = m q (1 - q) <= m / 4``; the same ``tie_break`` is
applied to the atom at the observed count), otherwise the normal bound ``Var(acc) <= (1/n_x +
1/n_y) / 16``; the logit uses the normal p-value of ``extra["z"]``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from scipy import stats

from ..config import N_PERM
from .critic import HacLags, LearnedTest, TieBreak, _draw_indices
from .nets import Arch

__all__ = ["C2ST"]


class C2ST(LearnedTest):
    """Classifier two-sample test; ``statistic="logit"`` gives the likelihood-ratio critic.

    Parameters
    ----------
    statistic : {"accuracy", "logit"}, default "accuracy"
        Evaluation statistic (see module docstring). ``name`` is ``"c2st"`` for accuracy and
        ``"lrcritic"`` for the logit.
    n_steps : int, default 1000
        Optimiser steps.
    batch_size : int, default 128
        Mini-batch size per class (balanced batches).
    lr : float, default 1e-3
        Adam learning rate.
    betas : tuple, default (0.9, 0.999)
        Adam betas.
    weight_decay : float, default 1e-4
        Adam L2 weight decay (mild regularisation against over-fitting the training half).
    arch, hidden, depth, layer_norm :
        Architecture (:func:`synthval.twosample.nets.build_network`), same as the critic.
    train_frac : float, default 0.5
        Fraction of each sample used for training.
    n_perm : int, default ``config.N_PERM`` (1000)
        Permutations of the evaluation labels.
    early_stopping : bool, default True
        Hold out ``val_frac`` of the training half, monitor the class-balanced validation
        cross-entropy every ``eval_every`` steps, keep the best weights.
    val_frac, eval_every, patience, min_steps, curve_points, hac_lags, inference, time_blocks :
        See :class:`~synthval.twosample.critic.LearnedTest` (for serially dependent windows:
        ``inference="hac"``, ``time_blocks`` and ``order_x`` / ``order_y``).
    tie_break : {"conservative", "mid", "randomized"}, optional
        Treatment of ties in the permutation (or binomial) p-value. Default: ``"randomized"``
        for the accuracy (exact size despite discreteness), ``"conservative"`` for the logit.
    """

    def __init__(self, statistic: Literal["accuracy", "logit"] = "accuracy", *,
                 n_steps: int = 1000, batch_size: int = 128, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), weight_decay: float = 1e-4,
                 arch: Arch = "auto", hidden: int | None = None, depth: int = 3,
                 layer_norm: bool = False, train_frac: float = 0.5, n_perm: int = N_PERM,
                 early_stopping: bool = True, val_frac: float = 0.2, eval_every: int = 50,
                 patience: int = 5, min_steps: int = 0, curve_points: int = 100,
                 hac_lags: HacLags = None,
                 inference: Literal["permutation", "hac"] = "permutation",
                 tie_break: TieBreak | None = None, time_blocks: int | None = None) -> None:
        if statistic not in ("accuracy", "logit"):
            raise ValueError("statistic must be 'accuracy' or 'logit'")
        if tie_break is None:
            tie_break = "randomized" if statistic == "accuracy" else "conservative"
        super().__init__(n_steps=n_steps, batch_size=batch_size, lr=lr, betas=betas,
                         weight_decay=weight_decay, arch=arch, hidden=hidden, depth=depth,
                         layer_norm=layer_norm, train_frac=train_frac, n_perm=n_perm,
                         early_stopping=early_stopping, val_frac=val_frac,
                         eval_every=eval_every, patience=patience, min_steps=min_steps,
                         curve_points=curve_points, hac_lags=hac_lags, inference=inference,
                         tie_break=tie_break, time_blocks=time_blocks)
        self.statistic = statistic
        self.name = "c2st" if statistic == "accuracy" else "lrcritic"
        if statistic == "accuracy":
            self.stat_offset, self.stat_scale = 0.5, 0.5
        else:
            self.stat_offset, self.stat_scale = 0.0, 1.0

    # -- training ---------------------------------------------------------------------------
    @staticmethod
    def _balanced_bce(net: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor,
                      wx: torch.Tensor | None = None,
                      wy: torch.Tensor | None = None) -> torch.Tensor:
        """Class-balanced binary cross-entropy (real = 1, synthetic = 0).

        ``wx`` / ``wy`` (optional, each summing to one) replace the within-class means by
        weighted means (weighted BCE; used on the full validation set in weighted training).
        """
        out = net(torch.cat([xb, yb], dim=0))
        s_x, s_y = out[:xb.shape[0]], out[xb.shape[0]:]
        # BCE(label 1) = softplus(-s), BCE(label 0) = softplus(s).
        lx = torch.nn.functional.softplus(-s_x)
        ly = torch.nn.functional.softplus(s_y)
        mx = lx.mean() if wx is None else (wx.to(device=lx.device, dtype=lx.dtype) * lx).sum()
        my = ly.mean() if wy is None else (wy.to(device=ly.device, dtype=ly.dtype) * ly).sum()
        return 0.5 * (mx + my)

    def _train(self, net: torch.nn.Module, Xt: torch.Tensor, Yt: torch.Tensor,
               Xv: torch.Tensor | None, Yv: torch.Tensor | None, *,
               wx: torch.Tensor | None = None, wy: torch.Tensor | None = None,
               wxv: torch.Tensor | None = None,
               wyv: torch.Tensor | None = None) -> dict[str, Any]:
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, betas=self.betas,
                               weight_decay=self.weight_decay)
        nx, ny, b = Xt.shape[0], Yt.shape[0], self.batch_size
        chunk = max(1, self.n_steps // max(1, self.curve_points * 2))
        curve: dict[str, list[float]] = {"step": [], "bce": []}
        val_curve: dict[str, list[float]] = {"step": [], "bce": []}
        acc, acc_n = 0.0, 0
        es = self._early_stop_loop_state()
        step = 0
        net.train()
        for step in range(1, self.n_steps + 1):
            # Weighted training (wx / wy given): indices drawn with those probabilities, so the
            # mini-batch loss is an unbiased estimate of the weighted BCE.
            ix = _draw_indices(nx, b, wx, Xt.device)
            iy = _draw_indices(ny, b, wy, Yt.device)
            loss = self._balanced_bce(net, Xt[ix], Yt[iy])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            acc += float(loss.detach())
            acc_n += 1
            if step % chunk == 0 or step == self.n_steps:
                curve["step"].append(step)
                curve["bce"].append(acc / acc_n)
                acc, acc_n = 0.0, 0
            if Xv is not None and Yv is not None and step % self.eval_every == 0:
                with torch.no_grad():
                    v = float(self._balanced_bce(net, Xv, Yv, wxv, wyv))
                val_curve["step"].append(step)
                val_curve["bce"].append(v)
                if step < self.min_steps:
                    continue
                if -v > es["best"]:
                    es.update(best=-v, bad=0, best_step=step,
                              best_state={k: t.detach().clone() for k, t in net.state_dict().items()})
                else:
                    es["bad"] += 1
                    if es["bad"] >= self.patience:
                        break
        info: dict[str, Any] = {"train_curve": curve, "steps_run": step}
        if Xv is not None:
            if es["best_state"] is not None:
                net.load_state_dict(es["best_state"])
            info["val_curve"] = val_curve
            info["best_step"] = es["best_step"]
        net.eval()
        return info

    # -- statistic --------------------------------------------------------------------------
    def _witness_values(self, out: np.ndarray) -> np.ndarray:
        if self.statistic == "accuracy":
            return (out > 0.0).astype(np.float64)
        return out

    def _asymptotic_pvalue(self, gx: np.ndarray, gy: np.ndarray, inf: dict[str, Any], *,
                           rng: np.random.Generator) -> float:
        if self.statistic == "logit":
            return float(inf["p_z"])
        nx, ny = gx.size, gy.size
        if nx == ny:
            correct = int(round(gx.sum() + (ny - gy.sum())))
            m = nx + ny
            above = float(stats.binom.sf(correct, m, 0.5))  # P(#correct > c)
            atom = float(stats.binom.pmf(correct, m, 0.5))
            if self.tie_break == "conservative":
                return above + atom
            if self.tie_break == "mid":
                return above + 0.5 * atom
            return above + float(rng.uniform()) * atom
        acc = 0.5 + 0.5 * inf["diff"]
        se = 0.25 * np.sqrt(1.0 / nx + 1.0 / ny)
        return float(stats.norm.sf((acc - 0.5) / se))

    def _extra_stats(self, out_x: np.ndarray, out_y: np.ndarray, gx: np.ndarray,
                     gy: np.ndarray, stat: float) -> dict[str, Any]:
        tpr = float(np.mean(out_x > 0.0))
        tnr = float(np.mean(out_y <= 0.0))
        return {"accuracy": 0.5 * (tpr + tnr), "acc_real": tpr, "acc_synth": tnr,
                "mean_logit_diff": float(out_x.mean() - out_y.mean()),
                "statistic_type": self.statistic}

    # -- public helpers ---------------------------------------------------------------------
    def fit_logit(self, X_tr: np.ndarray, Y_tr: np.ndarray, *,
                  rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        """Train the classifier on ``(X_tr, Y_tr)`` and return the learned logit as a function.

        The returned callable maps raw inputs ``(n, ...)`` to ``(n,)`` logits, an estimate of
        ``log p_real(x) / p_synth(x)`` (balanced training batches make the class prior 1/2), for
        comparison with the true log-likelihood ratio in ground-truth studies.

        Parameters
        ----------
        X_tr, Y_tr : arrays (n, ...)
            Real and synthetic training samples.
        rng : numpy.random.Generator
            Random generator.

        Returns
        -------
        callable
            ``logit(Z) -> np.ndarray`` of shape ``(n,)``. Its ``info`` attribute holds the
            training information (curves, steps).
        """
        witness, info = self.fit_witness(X_tr, Y_tr, rng=rng)
        witness.info = info  # type: ignore[attr-defined]
        return witness
