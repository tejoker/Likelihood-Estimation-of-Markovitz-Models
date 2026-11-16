"""Neural return generators: a QuantGAN-style TCN generator trained on a pooled panel.

Reference: M. Wiese, R. Knobloch, R. Korn, P. Kretschmer (2020), "Quant GANs: deep generation of
financial time series", Quantitative Finance 20(9), 1419-1440 (arXiv:1907.06673, cited below as
"WKKK"). The model is a *test subject* of the paper's evaluation method (like GBM or GARCH-t),
so it follows WKKK as closely as the panel setting allows; every deviation is listed in the
:class:`QuantGAN` docstring.

Contents
--------
* Heavy-tail Lambert W x Gaussian transform (Goerg 2015): :func:`lambertw_gaussianize`, its
  inverse :func:`lambertw_heavytail`, the log-density :func:`lambertw_logpdf` and the maximum
  likelihood fit :func:`fit_lambertw` (WKKK section 5.3 and section 6, step 3).
* :class:`TemporalBlock` and :class:`TCN`: temporal convolutional network with skip connections
  and unpadded ("valid") dilated causal convolutions (WKKK Definitions 3.10, 3.15, B.1 and
  Table 3).
* :class:`QuantGAN`: the generator, its preprocessing / post-processing and the adversarial
  training. One model is trained on all series of a panel (pooled); :meth:`QuantGAN.series_models`
  returns one :class:`QuantGANSeries` per series (a :class:`~synthval.models.base.ReturnModel`
  view sharing the generator), which is what ``pipeline.fit_panel("QuantGAN", ...)`` returns, so
  ``pipeline.simulate_panel`` and ``pipeline.evaluate_model`` use it like per-ticker fits.

Randomness: every random draw comes from the numpy ``Generator`` passed in (initial weights via
:func:`synthval.twosample.nets.torch_seed`; mini-batch indices, noise and gradient-penalty
weights drawn with numpy on the CPU). Networks run on :func:`synthval.twosample.nets.get_device`
(``$SYNTHVAL_DEVICE``, default CPU). A CPU run is bit-reproducible; a GPU run sees the same draws
and differs only by floating-point rounding.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from scipy import optimize, special, stats
from torch import nn

from .. import stylized
from ..twosample.nets import get_device, torch_seed
from .base import ReturnModel

logger = logging.getLogger(__name__)

__all__ = [
    "lambertw_gaussianize", "lambertw_heavytail", "lambertw_logpdf", "fit_lambertw",
    "TemporalBlock", "TCN", "QuantGAN", "QuantGANSeries", "TINY_KW",
]

Loss = Literal["wgan-gp", "gan-r1"]

#: Cheap settings for unit tests and smoke runs (tiny TCN, a few steps): seconds on one CPU core.
TINY_KW: dict[str, Any] = {"hidden": 8, "n_blocks": 3, "n_steps": 20, "n_critic": 2,
                           "batch_size": 16, "log_every": 5, "check": False}

_LOG_2PI = math.log(2.0 * math.pi)


# ----------------------------------------------------------------------------------------------
# Lambert W x Gaussian (heavy-tail) transform, Goerg (2015)
# ----------------------------------------------------------------------------------------------
def lambertw_gaussianize(y: np.ndarray, delta: float, mu: float = 0.0,
                         sigma: float = 1.0) -> np.ndarray:
    """Inverse heavy-tail Lambert W transform ``u = W_delta((y - mu) / sigma)``.

    ``W_delta(z) = sign(z) sqrt(W(delta z^2) / delta)`` with ``W`` the principal branch of the
    Lambert W function (Goerg 2015, eq. 9). For data that are Lambert W x Gaussian with
    parameters ``(mu, sigma, delta)``, ``u`` is standard normal. ``delta = 0`` is the identity.

    Parameters
    ----------
    y : ndarray
        Observations (any shape).
    delta : float
        Tail parameter, ``>= 0``.
    mu, sigma : float
        Location and scale of the latent Gaussian.

    Returns
    -------
    ndarray
        Gaussianised values ``u`` (same shape as ``y``), in latent standard units.
    """
    z = (np.asarray(y, dtype=np.float64) - mu) / sigma
    if delta <= 0.0:
        return z
    w = special.lambertw(delta * z * z).real
    return np.sign(z) * np.sqrt(w / delta)


def lambertw_heavytail(u: np.ndarray, delta: float, mu: float = 0.0,
                       sigma: float = 1.0) -> np.ndarray:
    """Heavy-tail Lambert W transform ``y = u exp(delta u^2 / 2) sigma + mu`` (WKKK eq. 3).

    The inverse of :func:`lambertw_gaussianize`. Values with ``delta u^2 / 2 > 709`` overflow
    to ``inf`` (no clipping: an exploding generator must show up as such).
    """
    u = np.asarray(u, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return u * np.exp(0.5 * delta * u * u) * sigma + mu


def lambertw_logpdf(y: np.ndarray, mu: float, sigma: float, delta: float) -> np.ndarray:
    """Log-density of the location-scale heavy-tail Lambert W x Gaussian law (Goerg 2015, eq. 13).

    ``log f(y) = log phi(u) + log(u / z) - log(1 + W(delta z^2)) - log(sigma)`` with
    ``z = (y - mu) / sigma`` and ``u = W_delta(z)``.
    """
    z = (np.asarray(y, dtype=np.float64) - mu) / sigma
    if delta <= 0.0:
        return -0.5 * z * z - 0.5 * _LOG_2PI - math.log(sigma)
    x = delta * z * z
    w = special.lambertw(x).real
    # u / z = sqrt(W(x) / x) -> 1 as x -> 0
    ratio = np.where(x > 1e-12, np.sqrt(w / np.maximum(x, 1e-300)), 1.0)
    u2 = w / delta
    return -0.5 * u2 - 0.5 * _LOG_2PI + np.log(ratio) - np.log1p(w) - math.log(sigma)


def fit_lambertw(y: np.ndarray, *, delta_max: float = 1.0) -> dict[str, Any]:
    """Maximum likelihood fit of ``(mu, sigma, delta)`` of the heavy-tail Lambert W x Gaussian law.

    WKKK (section 5.3) estimate the transform's parameters by (quasi) maximum likelihood, citing
    Goerg (2010, section 4.1); this is that estimator (L-BFGS-B on ``(mu, log sigma, delta)``,
    ``0 <= delta <= delta_max``).

    Parameters
    ----------
    y : ndarray
        Observations (flattened; non-finite values are not allowed).
    delta_max : float, default 1.0
        Upper bound of ``delta`` (``1 / delta`` is the tail index; 1 means no finite mean).

    Returns
    -------
    dict
        ``mu``, ``sigma``, ``delta``, ``loglik`` (mean per observation), ``success``,
        ``at_bound`` (list with ``"delta"`` if the estimate is at a bound), ``n``.
    """
    x = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 10 or not np.all(np.isfinite(x)):
        raise ValueError("fit_lambertw needs at least 10 finite observations")
    q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
    s0 = max((q75 - q25) / 1.349, 1e-8)

    def nll(theta: np.ndarray) -> float:
        mu, log_s, d = theta
        v = -float(np.mean(lambertw_logpdf(x, mu, math.exp(log_s), d)))
        return v if np.isfinite(v) else 1e10

    x0 = np.array([q50, math.log(s0), min(0.1, delta_max)])
    bounds = [(q50 - 5 * s0, q50 + 5 * s0), (math.log(s0) - 3.0, math.log(s0) + 3.0),
              (0.0, delta_max)]
    res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    mu, log_s, d = (float(v) for v in res.x)
    at = ["delta"] if (d <= 1e-6 or d >= delta_max - 1e-6) else []
    return {"mu": mu, "sigma": math.exp(log_s), "delta": d, "loglik": -float(res.fun),
            "success": bool(res.success), "at_bound": at, "n": int(x.size)}


# ----------------------------------------------------------------------------------------------
# Temporal convolutional network with skip connections
# ----------------------------------------------------------------------------------------------
class TemporalBlock(nn.Module):
    """Temporal block of WKKK Definition B.1 with a residual shortcut and a skip output.

    ``h = PReLU(conv2(PReLU(conv1(x))))``, two dilated causal convolutions without padding, so
    the time dimension shrinks by ``2 * dilation * (kernel_size - 1)``. Returns
    ``(h + shortcut(x), h)``: the input of the next block (residual connection of Bai, Kolter &
    Koltun 2018, the TCN that WKKK extend; ``shortcut`` is the identity or a 1x1 convolution when
    the channel counts differ, cropped to the last time steps) and the skip output ``h``
    (WKKK Definition 3.15).

    Parameters
    ----------
    n_in, n_hidden, n_out : int
        Input, hidden and output channels.
    kernel_size, dilation : int
        Kernel size ``K`` and dilation ``D`` of both convolutions.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, kernel_size: int,
                 dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_in, n_hidden, kernel_size, dilation=dilation)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv1d(n_hidden, n_out, kernel_size, dilation=dilation)
        self.act2 = nn.PReLU()
        self.shortcut = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.shrink = 2 * dilation * (kernel_size - 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act2(self.conv2(self.act1(self.conv1(x))))
        res = x if self.shortcut is None else self.shortcut(x)
        return h + res[..., self.shrink:], h


class TCN(nn.Module):
    """TCN with skip connections and unpadded dilated causal convolutions (WKKK Table 3).

    Blocks: block 1 has kernel size 1 and dilation 1; block ``l = 2..n_blocks`` has kernel size 2
    and dilation ``2^(l - 2)``. The skip outputs of all blocks (cropped to the output length) are
    summed and mapped to ``n_out`` channels by a 1x1 convolution. With ``n_blocks = 7`` this is
    WKKK's TCN: dilations 1, 1, 2, 4, 8, 16, 32 and receptive field ``1 + 2 * 63 = 127``.

    Parameters
    ----------
    n_in, n_out : int
        Input and output channels.
    hidden : int, default 80
        Channels of every block (WKKK Table 4: 80).
    n_blocks : int, default 7
        Number of temporal blocks (>= 1).

    Notes
    -----
    Input ``(B, n_in, T)`` with ``T >= receptive_field`` gives output
    ``(B, n_out, T - receptive_field + 1)``: output ``t`` depends exactly on inputs
    ``t .. t + receptive_field - 1`` (no padding, WKKK Definition 3.10).
    """

    def __init__(self, n_in: int, n_out: int, hidden: int = 80, n_blocks: int = 7) -> None:
        super().__init__()
        if min(n_in, n_out, hidden, n_blocks) < 1:
            raise ValueError("n_in, n_out, hidden and n_blocks must be positive")
        blocks = []
        for i in range(n_blocks):
            k, d = (1, 1) if i == 0 else (2, 2 ** (i - 1))
            blocks.append(TemporalBlock(n_in if i == 0 else hidden, hidden, hidden, k, d))
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv1d(hidden, n_out, 1)
        self.receptive_field = 1 + sum(b.shrink for b in self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_out = x.shape[-1] - self.receptive_field + 1
        if t_out < 1:
            raise ValueError(f"input length {x.shape[-1]} < receptive field "
                             f"{self.receptive_field}")
        skip: torch.Tensor | None = None
        for blk in self.blocks:
            x, h = blk(x)
            h = h[..., -t_out:]
            skip = h if skip is None else skip + h
        return self.out(skip)


def _n_weights(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def _set_requires_grad(net: nn.Module, flag: bool) -> None:
    for p in net.parameters():
        p.requires_grad_(flag)


# ----------------------------------------------------------------------------------------------
# QuantGAN
# ----------------------------------------------------------------------------------------------
class QuantGAN(ReturnModel):
    """QuantGAN "pure TCN" generator (WKKK section 7.2), trained on one series or a pooled panel.

    **Model.** Standardised log-returns are Gaussianised with the inverse Lambert W transform and
    standardised again; the result is modelled as the neural process
    ``X_t = g(Z_{t-126}, ..., Z_t)`` with ``Z_t ~ N(0, I_3)`` i.i.d. and ``g`` a TCN with skip
    connections and receptive field 127 (:class:`TCN`, WKKK Tables 3 and 4: 7 temporal blocks of
    80 channels with PReLUs, dilations 1, 1, 2, 4, 8, 16, 32, kernel size 1 then 2, 1x1 output
    convolution). The discriminator is the same TCN with one input channel; it sees windows of
    127 days (its receptive field), so it returns one value per window.

    **Preprocessing (WKKK section 6, Figure 9).** (1) log-returns (given); (2) standardise each
    series by its own mean and standard deviation (``ddof = 1``, the scale ``data.train_vol``
    uses); (3) inverse Lambert W x Gaussian transform with ``(mu, sigma, delta)`` fitted by
    maximum likelihood on the pooled standardised returns (:func:`fit_lambertw`); (4) standardise
    the pooled result; (5) rolling windows of length 127 and stride 1 *within* each series.
    **Post-processing** inverts (4), (3) and (2): a simulated path of series ``i`` is
    ``mean_i + std_i * LambertW_forward(s4 * g + m4)``.

    **Simulation.** A path of ``n`` days is ``g`` applied to ``n + 126`` noise vectors drawn from
    the caller's ``rng``; every output sees a full receptive field, so no burn-in is needed and
    the process is stationary by construction (``burn_in`` and ``start_from_end`` are accepted
    and ignored: there is no state to start from).

    **Deviations from WKKK, and why.**

    * *Pooled panel.* WKKK train on one series (S&P 500 index, 2413 days). Here one generator is
      trained on all series of the panel, each standardised by its own TRAIN mean and std (one
      Lambert W ``delta`` for the pooled standardised returns): 1258 TRAIN days per ticker are
      too few for a 2 x 160k-weight GAN. The generator is therefore the same for every ticker up
      to the per-ticker mean and std; tail heaviness and dependence are pooled.
    * *Loss.* WGAN-GP (Gulrajani et al. 2017: Wasserstein critic without sigmoid, gradient
      penalty ``lambda = 10``, ``n_critic = 5`` critic steps per generator step, Adam
      ``lr = 1e-4``, ``betas = (0, 0.9)``, their Algorithm 1 defaults) instead of WKKK's
      Algorithm 1 (the original GAN objective, sigmoid discriminator) with the regulariser of
      Mescheder et al. (2018). Reasons: WKKK report that their training "was very irregular and
      did not converge"; the WGAN-GP critic loss is a usable progress signal; it was fixed in
      docs/PROTOCOL_v2.md before any run. ``loss="gan-r1"`` gives WKKK's objective (non-saturating
      generator loss, R1 penalty ``r1_gamma / 2 * E|grad D(x_real)|^2``) for comparison.
    * *No checkpoint selection.* WKKK saved several parameter sets and kept the one with the best
      evaluation metrics (EMD, DY, ACF and leverage scores). That is an analyst's choice with no
      fixed rule; here the generator after a fixed number of steps (``n_steps``) is used, fixed
      before the evaluation. Diagnostics on a TRAIN-sized simulated sample are recorded in
      ``fit_info_`` (``check_*``), never used for selection.
    * *Unreported training settings.* WKKK give no learning rate, batch size or number of
      iterations; the defaults here are the WGAN-GP ones above, batch 64 and ``n_steps = 10_000``
      generator steps (50 000 critic steps, about 28 passes over the 113 000 TRAIN windows of the
      100-ticker panel).
    * *Uniform window sampling.* Windows are drawn uniformly over (series, start), so the first
      and last 126 days of each series are under-sampled (WKKK Remark 6.1 mention a weighted
      scheme without using it).
    * *Residual shortcut* inside each temporal block (Bai et al. 2018, whose TCN WKKK extend;
      WKKK Definition B.1 does not mention it) and PyTorch's default weight initialisation (not
      reported by WKKK).

    Parameters
    ----------
    hidden : int, default 80
        Channels of the generator and discriminator TCNs.
    n_blocks : int, default 7
        Temporal blocks per TCN; receptive field ``1 + 2 (2^(n_blocks - 1) - 1)`` (127 for 7).
        The training windows have this length.
    noise_dim : int, default 3
        Dimension of the noise ``Z_t`` (WKKK: 3).
    loss : {"wgan-gp", "gan-r1"}, default "wgan-gp"
        Training objective (see above).
    n_steps : int, default 10000
        Generator updates.
    n_critic : int, default 5
        Discriminator updates per generator update.
    batch_size : int, default 64
        Windows per mini-batch (real and generated alike).
    lr_g, lr_d : float, default 1e-4
        Adam learning rates of the generator and the discriminator.
    betas : tuple of float, default (0.0, 0.9)
        Adam betas of both optimisers.
    gp_lambda : float, default 10.0
        Gradient-penalty weight (``loss="wgan-gp"``).
    r1_gamma : float, default 10.0
        R1 penalty weight (``loss="gan-r1"``).
    lambert_w : bool, default True
        Apply the Lambert W preprocessing (False: steps 3-4 skipped).
    check : bool, default True
        After training, simulate a TRAIN-sized standardised sample and record stylized-fact
        diagnostics in ``fit_info_``.
    log_every : int, default 100
        Generator steps per point of the training curve (``fit_info_["curve"]``).
    seed : int, default 20251116
        Seed of ``fit`` when no ``rng`` is passed (deterministic default, as for Heston).

    Attributes
    ----------
    loc_, scale_ : ndarray (N,)
        Per-series mean and standard deviation of the training returns.
    lw_ : dict
        Lambert W fit (``mu``, ``sigma``, ``delta``, ...) on the pooled standardised returns.
    post_mean_, post_std_ : float
        Mean and standard deviation of the pooled Gaussianised returns (step 4).
    receptive_field : int
        Receptive field of the TCNs (window length of the discriminator).
    fit_info_ : dict
        Scalars (Lambert W fit, sizes, training time, device, final losses, ``check_*``
        diagnostics, ``at_bound``) and ``curve`` (training curve, dict of lists).
    """

    name = "QuantGAN"
    has_likelihood = False

    def __init__(self, *, hidden: int = 80, n_blocks: int = 7, noise_dim: int = 3,
                 loss: Loss = "wgan-gp", n_steps: int = 10_000, n_critic: int = 5,
                 batch_size: int = 64, lr_g: float = 1e-4, lr_d: float = 1e-4,
                 betas: tuple[float, float] = (0.0, 0.9), gp_lambda: float = 10.0,
                 r1_gamma: float = 10.0, lambert_w: bool = True, check: bool = True,
                 log_every: int = 100, seed: int = 20251116) -> None:
        if loss not in ("wgan-gp", "gan-r1"):
            raise ValueError(f"loss must be 'wgan-gp' or 'gan-r1', got {loss!r}")
        if min(hidden, n_blocks, noise_dim, n_steps, n_critic, batch_size, log_every) < 1:
            raise ValueError("hidden, n_blocks, noise_dim, n_steps, n_critic, batch_size and "
                             "log_every must be positive")
        self.hidden, self.n_blocks, self.noise_dim = int(hidden), int(n_blocks), int(noise_dim)
        self.loss = loss
        self.n_steps, self.n_critic, self.batch_size = int(n_steps), int(n_critic), int(batch_size)
        self.lr_g, self.lr_d = float(lr_g), float(lr_d)
        self.betas = (float(betas[0]), float(betas[1]))
        self.gp_lambda, self.r1_gamma = float(gp_lambda), float(r1_gamma)
        self.lambert_w, self.check = bool(lambert_w), bool(check)
        self.log_every, self.seed = int(log_every), int(seed)
        self.receptive_field = 1 + 2 * (2 ** (self.n_blocks - 1) - 1)
        self.loc_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.lw_: dict[str, Any] = {}
        self.post_mean_ = 0.0
        self.post_std_ = 1.0
        self.fit_info_: dict[str, Any] = {}
        self._g_state: dict[str, torch.Tensor] | None = None
        self._net_cache: dict[str, nn.Module] = {}

    # -- pickling: the device-specific generator cache is rebuilt on demand -------------------
    def __getstate__(self) -> dict[str, Any]:
        st = self.__dict__.copy()
        st["_net_cache"] = {}
        return st

    def __setstate__(self, st: dict[str, Any]) -> None:
        self.__dict__.update(st)
        self._net_cache = {}

    # -- networks ------------------------------------------------------------------------------
    def build_generator(self) -> TCN:
        """An untrained generator ``TCN(noise_dim, 1, hidden, n_blocks)`` (on the CPU)."""
        return TCN(self.noise_dim, 1, self.hidden, self.n_blocks)

    def build_discriminator(self) -> TCN:
        """An untrained discriminator ``TCN(1, 1, hidden, n_blocks)`` (on the CPU)."""
        return TCN(1, 1, self.hidden, self.n_blocks)

    def generator(self) -> TCN:
        """The trained generator on :func:`get_device`, in eval mode (cached per device)."""
        if self._g_state is None:
            raise RuntimeError("QuantGAN is not fitted")
        dev = get_device()
        key = str(dev)
        if key not in self._net_cache:
            g = self.build_generator()
            g.load_state_dict(self._g_state)
            self._net_cache[key] = g.to(dev).eval()
        return self._net_cache[key]  # type: ignore[return-value]

    # -- pre- and post-processing --------------------------------------------------------------
    def _preprocess(self, R: np.ndarray) -> np.ndarray:
        """Steps 2-4 of WKKK Figure 9 on ``(T, N)`` returns (sets the transform parameters)."""
        self.loc_ = R.mean(axis=0)
        self.scale_ = R.std(axis=0, ddof=1)
        if not np.all(np.isfinite(self.scale_) & (self.scale_ > 0)):
            raise ValueError("every series needs a finite positive standard deviation")
        Z = (R - self.loc_) / self.scale_
        if self.lambert_w:
            self.lw_ = fit_lambertw(Z)
            U = lambertw_gaussianize(Z, self.lw_["delta"], self.lw_["mu"], self.lw_["sigma"])
        else:
            self.lw_ = {"mu": 0.0, "sigma": 1.0, "delta": 0.0, "loglik": float("nan"),
                        "success": True, "at_bound": [], "n": int(Z.size)}
            U = Z
        self.post_mean_, self.post_std_ = float(U.mean()), float(U.std())
        return (U - self.post_mean_) / self.post_std_

    def postprocess(self, X: np.ndarray, series: int | None = None) -> np.ndarray:
        """Map generator output to returns: invert steps 4, 3 and (if ``series`` given) 2.

        ``series=None`` returns standardised returns (zero mean, unit variance units of the
        training series).
        """
        if self.loc_ is None or self.scale_ is None:
            raise RuntimeError("QuantGAN is not fitted")
        U = np.asarray(X, dtype=np.float64) * self.post_std_ + self.post_mean_
        Z = lambertw_heavytail(U, self.lw_["delta"], self.lw_["mu"], self.lw_["sigma"])
        if series is None:
            return Z
        return self.loc_[series] + self.scale_[series] * Z

    # -- fit -------------------------------------------------------------------------------------
    def fit(self, r: np.ndarray, *, rng: np.random.Generator | None = None) -> "QuantGAN":
        """Train on one series ``(T,)`` or on the pooled columns of a panel ``(T, N)``.

        Parameters
        ----------
        r : ndarray, shape (T,) or (T, N)
            Daily log-returns (finite), one column per series; ``T >= receptive_field``.
        rng : numpy Generator, optional
            Source of every random draw; default ``np.random.default_rng(self.seed)``.

        Returns
        -------
        QuantGAN
            ``self``, fitted.
        """
        R = np.asarray(r, dtype=np.float64)
        if R.ndim == 1:
            R = R[:, None]
        if R.ndim != 2 or R.shape[1] < 1:
            raise ValueError(f"r must have shape (T,) or (T, N), got {np.shape(r)}")
        if not np.all(np.isfinite(R)):
            raise ValueError("r must be finite (no NaN)")
        T, N = R.shape
        rf = self.receptive_field
        if T < rf + 1:
            raise ValueError(f"need at least receptive_field + 1 = {rf + 1} observations, got {T}")
        rng = np.random.default_rng(self.seed) if rng is None else rng
        t0 = time.perf_counter()
        X = self._preprocess(R)  # (T, N), standardised Gaussianised returns
        t_pre = time.perf_counter() - t0
        info = self._train(X, rng)
        info["preprocess_seconds"] = t_pre
        lw = self.lw_
        self.fit_info_ = {
            "lw_mu": lw["mu"], "lw_sigma": lw["sigma"], "lw_delta": lw["delta"],
            "lw_loglik": lw["loglik"], "lw_success": lw["success"],
            "post_mean": self.post_mean_, "post_std": self.post_std_,
            "receptive_field": rf, "n_series": N, "n_obs": T, "n_windows": N * (T - rf + 1),
            "loss": self.loss, **info, "at_bound": list(lw.get("at_bound", [])),
        }
        if self.check:
            self.fit_info_.update(self._check(R, rng))
        self.fit_info_["seconds"] = time.perf_counter() - t0
        logger.info("QuantGAN fitted on %d series x %d days in %.1f s (%s, delta %.3f)", N, T,
                    self.fit_info_["seconds"], info["device"], lw["delta"])
        return self

    def _train(self, X: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
        """Adversarial training on the windows of ``X`` (T, N); returns the training record."""
        dev = get_device()
        T, N = X.shape
        rf, B, nz = self.receptive_field, self.batch_size, self.noise_dim
        n_start = T - rf + 1
        series = torch.from_numpy(np.ascontiguousarray(X.T, dtype=np.float32)).to(dev)
        windows = series.unfold(1, rf, 1)  # (N, n_start, rf) view, no copy
        with torch_seed(rng):
            G = self.build_generator().to(dev)
            D = self.build_discriminator().to(dev)
        opt_g = torch.optim.Adam(G.parameters(), lr=self.lr_g, betas=self.betas)
        opt_d = torch.optim.Adam(D.parameters(), lr=self.lr_d, betas=self.betas)
        t_noise = 2 * rf - 1  # generator input length for one window of rf outputs

        def noise(b: int) -> torch.Tensor:
            return torch.from_numpy(rng.standard_normal((b, nz, t_noise), dtype=np.float32)).to(dev)

        def real_batch(b: int) -> torch.Tensor:
            k = rng.integers(0, N * n_start, size=b)
            i = torch.from_numpy(k // n_start).to(dev)
            j = torch.from_numpy(k % n_start).to(dev)
            return windows[i, j].unsqueeze(1)  # (b, 1, rf)

        curve: dict[str, list[float]] = {"step": [], "d_loss": [], "g_loss": [], "wdist": [],
                                         "penalty": []}
        acc = np.zeros(4)
        n_acc = 0
        G.train()
        D.train()
        t0 = time.perf_counter()
        for step in range(1, self.n_steps + 1):
            _set_requires_grad(D, True)
            for _ in range(self.n_critic):
                xr = real_batch(B)
                with torch.no_grad():
                    xf = G(noise(B))
                if self.loss == "wgan-gp":
                    eps = torch.from_numpy(rng.random((B, 1, 1), dtype=np.float32)).to(dev)
                    xh = (eps * xr + (1.0 - eps) * xf).requires_grad_(True)
                    out = D(torch.cat([xr, xf, xh], dim=0)).reshape(-1)
                    f_r, f_f, f_h = out[:B], out[B:2 * B], out[2 * B:]
                    grad = torch.autograd.grad(f_h.sum(), xh, create_graph=True)[0]
                    pen = ((grad.reshape(B, -1).norm(2, dim=1) - 1.0) ** 2).mean()
                    wd = f_r.mean() - f_f.mean()
                    d_loss = -wd + self.gp_lambda * pen
                else:  # gan-r1: original GAN objective + R1 penalty on real data
                    xr = xr.detach().requires_grad_(True)
                    f_r = D(xr).reshape(-1)
                    f_f = D(xf).reshape(-1)
                    grad = torch.autograd.grad(f_r.sum(), xr, create_graph=True)[0]
                    pen = grad.reshape(B, -1).pow(2).sum(dim=1).mean()
                    wd = f_r.mean() - f_f.mean()
                    d_loss = (nn.functional.softplus(-f_r).mean()
                              + nn.functional.softplus(f_f).mean() + 0.5 * self.r1_gamma * pen)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
            _set_requires_grad(D, False)
            f_g = D(G(noise(B))).reshape(-1)
            g_loss = (-f_g.mean() if self.loss == "wgan-gp"
                      else nn.functional.softplus(-f_g).mean())
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            acc += [float(d_loss.detach()), float(g_loss.detach()), float(wd.detach()),
                    float(pen.detach())]
            n_acc += 1
            if step % self.log_every == 0 or step == self.n_steps:
                m = acc / n_acc
                curve["step"].append(step)
                for k, v in zip(("d_loss", "g_loss", "wdist", "penalty"), m):
                    curve[k].append(float(v))
                acc[:] = 0.0
                n_acc = 0
        train_s = time.perf_counter() - t0
        G.eval()
        self._g_state = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}
        self._net_cache = {}
        tail = max(1, len(curve["step"]) // 10)  # last 10% of the curve
        return {"device": str(dev), "n_steps": self.n_steps, "n_critic": self.n_critic,
                "batch_size": B, "n_weights_g": _n_weights(G), "n_weights_d": _n_weights(D),
                "train_seconds": train_s, "seconds_per_step": train_s / self.n_steps,
                "final_d_loss": float(np.mean(curve["d_loss"][-tail:])),
                "final_g_loss": float(np.mean(curve["g_loss"][-tail:])),
                "final_wdist": float(np.mean(curve["wdist"][-tail:])),
                "final_penalty": float(np.mean(curve["penalty"][-tail:])), "curve": curve}

    def _check(self, R: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        """Stylized-fact diagnostics: real vs a simulated sample of the same size, standardised
        units (recorded only; never used to select or tune the model)."""
        T, N = R.shape
        Zr = (R - self.loc_) / self.scale_
        Zs = self.simulate_standardised(T, N, rng=rng).T  # (T, N)
        out: dict[str, float] = {}
        for tag, Z in (("real", Zr), ("sim", Zs)):
            z = Z.ravel()
            fin = np.isfinite(z)
            out[f"check_{tag}_finite_share"] = float(fin.mean())
            z = z[fin]
            out[f"check_{tag}_std"] = float(z.std(ddof=1))
            out[f"check_{tag}_kurt"] = float(stats.kurtosis(z, fisher=True, bias=False))
            out[f"check_{tag}_max_abs"] = float(np.max(np.abs(z)))
            for lag in (1, 20):
                acfs = [stylized.acf(np.abs(Z[:, j]), lag)[lag] for j in range(N)
                        if np.all(np.isfinite(Z[:, j]))]
                out[f"check_{tag}_acf_abs_{lag}"] = float(np.median(acfs)) if acfs else np.nan
        return out

    # -- simulation ------------------------------------------------------------------------------
    def simulate_standardised(self, n_steps: int, n_paths: int, *,
                              rng: np.random.Generator) -> np.ndarray:
        """Standardised returns ``(n_paths, n_steps)`` (zero-mean, unit-variance units).

        Path ``k`` is the generator applied to ``n_steps + receptive_field - 1`` noise vectors
        ``rng.standard_normal((noise_dim, ...))`` (float32), drawn path by path in chunks.
        """
        n_steps, n_paths = int(n_steps), int(n_paths)
        if n_steps < 1 or n_paths < 1:
            raise ValueError("n_steps and n_paths must be >= 1")
        G = self.generator()
        dev = get_device()
        t_in = n_steps + self.receptive_field - 1
        chunk = max(1, int(4_000_000 // (self.hidden * t_in)))  # ~16 MB per activation
        out = np.empty((n_paths, n_steps), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, n_paths, chunk):
                b = min(chunk, n_paths - s)
                z = rng.standard_normal((b, self.noise_dim, t_in), dtype=np.float32)
                g = G(torch.from_numpy(z).to(dev))
                out[s:s + b] = g[:, 0, :].double().cpu().numpy()
        return self.postprocess(out)

    def simulate(self, n_steps: int, n_paths: int, *, rng: np.random.Generator,
                 burn_in: int = 0, start_from_end: bool = False,
                 series: int | None = None) -> np.ndarray:
        """Simulate ``(n_paths, n_steps)`` daily log-returns of training series ``series``.

        ``series`` defaults to 0 for a single-series fit and is required for a panel fit
        (:meth:`simulate_standardised` gives the pooled standardised units). ``burn_in`` and
        ``start_from_end`` are ignored (stationary generator, no state; see the class docstring).
        """
        del burn_in, start_from_end
        if self.loc_ is None:
            raise RuntimeError("QuantGAN is not fitted")
        if series is None:
            if self.loc_.size != 1:
                raise ValueError("panel fit: pass series=i (or use series_models / "
                                 "simulate_standardised)")
            series = 0
        Z = self.simulate_standardised(n_steps, n_paths, rng=rng)
        return self.loc_[series] + self.scale_[series] * Z

    def series_models(self, names: Sequence[str] | None = None) -> dict[str, "QuantGANSeries"]:
        """One :class:`QuantGANSeries` per training series (sharing this generator).

        Parameters
        ----------
        names : sequence of str, optional
            Labels in column order (default ``"0", "1", ...``).
        """
        if self.loc_ is None:
            raise RuntimeError("QuantGAN is not fitted")
        n = self.loc_.size
        names = [str(i) for i in range(n)] if names is None else [str(s) for s in names]
        if len(names) != n:
            raise ValueError(f"{len(names)} names for {n} series")
        return {s: QuantGANSeries(self, i, s) for i, s in enumerate(names)}

    def params(self) -> dict[str, float]:
        """Shared scalar parameters of the transforms and the network size."""
        if self.loc_ is None:
            raise RuntimeError("QuantGAN is not fitted")
        return {"lw_mu": float(self.lw_["mu"]), "lw_sigma": float(self.lw_["sigma"]),
                "lw_delta": float(self.lw_["delta"]), "post_mean": self.post_mean_,
                "post_std": self.post_std_, "receptive_field": float(self.receptive_field)}

    def config(self) -> dict[str, Any]:
        """Constructor arguments (JSON-friendly)."""
        return {"hidden": self.hidden, "n_blocks": self.n_blocks, "noise_dim": self.noise_dim,
                "loss": self.loss, "n_steps": self.n_steps, "n_critic": self.n_critic,
                "batch_size": self.batch_size, "lr_g": self.lr_g, "lr_d": self.lr_d,
                "betas": list(self.betas), "gp_lambda": self.gp_lambda,
                "r1_gamma": self.r1_gamma, "lambert_w": self.lambert_w, "check": self.check,
                "log_every": self.log_every, "seed": self.seed}


class QuantGANSeries(ReturnModel):
    """One series of a fitted (pooled) :class:`QuantGAN`: the shared generator rescaled by the
    series' training mean and standard deviation.

    This is the per-ticker object that ``pipeline.fit_panel("QuantGAN", ...)`` returns, so that
    ``pipeline.simulate_panel`` (one path per ticker, ticker ``i`` with its own child generator)
    treats the pooled model like per-ticker fits. All views of one fit share ``core`` (pickled
    once with them).
    """

    name = "QuantGAN"
    has_likelihood = False

    def __init__(self, core: QuantGAN, index: int, label: str) -> None:
        self.core = core
        self.index = int(index)
        self.label = str(label)

    @property
    def fit_info_(self) -> dict[str, Any]:
        """The shared training record of the pooled fit."""
        return self.core.fit_info_

    def fit(self, r: np.ndarray) -> "QuantGANSeries":
        raise NotImplementedError("a QuantGANSeries is a view of a pooled fit: fit QuantGAN on "
                                  "the panel (pipeline.fit_panel('QuantGAN', ...))")

    def simulate(self, n_steps: int, n_paths: int, *, rng: np.random.Generator,
                 burn_in: int = 0, start_from_end: bool = False) -> np.ndarray:
        """``(n_paths, n_steps)`` log-returns of this series (see :meth:`QuantGAN.simulate`)."""
        return self.core.simulate(n_steps, n_paths, rng=rng, burn_in=burn_in,
                                  start_from_end=start_from_end, series=self.index)

    def params(self) -> dict[str, float]:
        """Training mean and standard deviation of this series (the rest is shared)."""
        return {"mean": float(self.core.loc_[self.index]),
                "std": float(self.core.scale_[self.index])}
