"""Critic / classifier architectures and small torch helpers for the learned two-sample tests.

Design rules (see ``docs/DESIGN.md`` section 3.7):

* **No BatchNorm.** The WGAN-GP gradient penalty is a per-sample quantity; BatchNorm couples
  the samples of a batch and makes ``grad_x f(x)`` ill defined. LayerNorm (per-sample) is
  allowed and is used in :class:`ConvCritic`.
* Every network maps a batch of inputs to one real number per input (shape ``(n,)``): the
  critic value ``f(x)`` for the Wasserstein test, the logit for the classifier test.
* Inputs are standardised outside the network with statistics of the pooled *training* half
  (:class:`Standardizer`), and the fitted network is wrapped in a :class:`Witness`, a plain
  numpy callable that applies the same transform.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Literal

import numpy as np
import torch
from torch import nn

Arch = Literal["auto", "mlp", "conv"]

__all__ = [
    "MLPCritic",
    "ConvCritic",
    "build_network",
    "Standardizer",
    "Witness",
    "torch_seed",
    "to_tensor",
]


class MLPCritic(nn.Module):
    """Fully connected critic for flat inputs ``(n, d)`` (or any ``(n, ...)``, flattened).

    Parameters
    ----------
    in_dim : int
        Number of input features after flattening (``L`` for univariate windows).
    hidden : int, default 128
        Width of every hidden layer.
    depth : int, default 3
        Number of hidden layers.
    layer_norm : bool, default False
        Insert a LayerNorm after every hidden linear layer (BatchNorm is never used).
    negative_slope : float, default 0.2
        Slope of the LeakyReLU activations.
    """

    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 3, *,
                 layer_norm: bool = False, negative_slope: float = 0.2) -> None:
        super().__init__()
        if in_dim < 1 or hidden < 1 or depth < 1:
            raise ValueError("in_dim, hidden and depth must be positive")
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.LeakyReLU(negative_slope))
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.in_dim = in_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one scalar per input, shape ``(n,)``."""
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


class ConvCritic(nn.Module):
    """1D convolutional critic for inputs ``(n, C, L)`` (or ``(n, L)`` when ``in_channels == 1``).

    Architecture: ``n_conv`` blocks of ``Conv1d(kernel_size, padding="same") -> LayerNorm([hidden,
    L]) -> LeakyReLU``, then ``Flatten -> Linear(hidden * L, hidden) -> LeakyReLU -> Linear(hidden,
    1)``. Flattening (rather than global pooling) keeps the position information inside the
    window, so the critic can see e.g. volatility clustering within the window.

    Parameters
    ----------
    in_channels : int
        Number of channels ``C`` (assets for multivariate windows, 1 for univariate).
    L : int
        Window length.
    hidden : int, default 64
        Number of convolution channels and width of the dense head.
    n_conv : int, default 2
        Number of convolution blocks.
    kernel_size : int, default 5
        Convolution kernel size.
    layer_norm : bool, default True
        Use per-sample LayerNorm after each convolution (never BatchNorm).
    negative_slope : float, default 0.2
        Slope of the LeakyReLU activations.
    """

    def __init__(self, in_channels: int, L: int, hidden: int = 64, *, n_conv: int = 2,
                 kernel_size: int = 5, layer_norm: bool = True,
                 negative_slope: float = 0.2) -> None:
        super().__init__()
        if in_channels < 1 or L < 1 or hidden < 1 or n_conv < 1:
            raise ValueError("in_channels, L, hidden and n_conv must be positive")
        k = max(1, min(kernel_size, L))
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(n_conv):
            layers.append(nn.Conv1d(c, hidden, kernel_size=k, padding="same"))
            if layer_norm:
                layers.append(nn.LayerNorm([hidden, L]))
            layers.append(nn.LeakyReLU(negative_slope))
            c = hidden
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * L, hidden),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden, 1),
        )
        self.in_channels = in_channels
        self.L = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one scalar per input, shape ``(n,)``."""
        if x.dim() == 2:
            if self.in_channels != 1:
                raise ValueError("(n, L) input requires in_channels == 1")
            x = x.unsqueeze(1)
        return self.head(self.features(x)).squeeze(-1)


def build_network(input_shape: tuple[int, ...], arch: Arch = "auto", *,
                  hidden: int | None = None, depth: int = 3, layer_norm: bool = False,
                  kernel_size: int = 5) -> nn.Module:
    """Choose an architecture from the per-sample input shape.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of one sample: ``(d,)`` / ``(L,)`` or ``(C, L)``.
    arch : {"auto", "mlp", "conv"}
        ``"auto"``: :class:`MLPCritic` for rank-1 samples, :class:`ConvCritic` for rank-2 samples.
        ``"conv"`` with a rank-1 sample treats it as ``(1, L)``. ``"mlp"`` flattens anything.
    hidden : int, optional
        Hidden width (default 128 for the MLP, 64 for the CNN).
    depth : int, default 3
        Number of hidden layers of the MLP (number of conv blocks is fixed at 2).
    layer_norm : bool, default False
        LayerNorm in the MLP hidden layers (the CNN always uses LayerNorm).
    kernel_size : int, default 5
        CNN kernel size.

    Returns
    -------
    torch.nn.Module
    """
    shape = tuple(int(s) for s in input_shape)
    if arch == "auto":
        if len(shape) == 1:
            arch = "mlp"
        elif len(shape) == 2:
            arch = "conv"
        else:
            raise ValueError(f"cannot choose an architecture for sample shape {shape}")
    if arch == "mlp":
        return MLPCritic(int(np.prod(shape)), hidden=hidden or 128, depth=depth,
                         layer_norm=layer_norm)
    if arch == "conv":
        if len(shape) == 1:
            c, L = 1, shape[0]
        elif len(shape) == 2:
            c, L = shape
        else:
            raise ValueError(f"ConvCritic needs (L,) or (C, L) samples, got {shape}")
        return ConvCritic(c, L, hidden=hidden or 64, kernel_size=kernel_size)
    raise ValueError(f"unknown arch {arch!r}")


class Standardizer:
    """Affine standardisation fitted on the pooled training half and applied to every sample.

    For ``(n, d)`` / ``(n, L)`` inputs the statistics are per coordinate; for ``(n, C, L)``
    inputs they are per channel (pooled over the window), which keeps the transform
    translation-equivariant along the time axis as a convolution expects.

    Parameters
    ----------
    eps : float, default 1e-12
        Standard deviations below ``eps`` are replaced by 1 (constant coordinates).
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, Z: np.ndarray) -> "Standardizer":
        """Estimate location and scale on ``Z`` (the pooled training sample)."""
        Z = np.asarray(Z, dtype=np.float64)
        axes: tuple[int, ...] = (0,) if Z.ndim == 2 else (0, *range(2, Z.ndim))
        mean = Z.mean(axis=axes, keepdims=True)
        scale = Z.std(axis=axes, keepdims=True)
        scale = np.where(scale < self.eps, 1.0, scale)
        self.mean_, self.scale_ = mean, scale
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Apply the fitted transform."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted")
        return (np.asarray(Z, dtype=np.float64) - self.mean_) / self.scale_


def to_tensor(Z: np.ndarray) -> torch.Tensor:
    """float32 torch tensor (networks run in single precision on CPU)."""
    return torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32))


class Witness:
    """Frozen network + input standardisation, evaluated as a numpy function.

    Calling ``w(Z)`` on raw (unstandardised) inputs of shape ``(n, ...)`` returns the network
    output (critic value or logit) as a ``float64`` array of shape ``(n,)``.

    Parameters
    ----------
    net : torch.nn.Module
        Trained network (put in eval mode, gradients disabled at call time).
    standardizer : Standardizer
        Transform fitted on the pooled training half.
    batch_size : int, default 8192
        Evaluation batch size.
    """

    def __init__(self, net: nn.Module, standardizer: Standardizer, batch_size: int = 8192) -> None:
        self.net = net.eval()
        self.standardizer = standardizer
        self.batch_size = batch_size

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        Zs = self.standardizer.transform(Z)
        out = np.empty(Zs.shape[0], dtype=np.float64)
        with torch.no_grad():
            for i in range(0, Zs.shape[0], self.batch_size):
                out[i:i + self.batch_size] = self.net(to_tensor(Zs[i:i + self.batch_size])).double().numpy()
        return out


@contextlib.contextmanager
def torch_seed(rng: np.random.Generator) -> Iterator[None]:
    """Seed torch from ``rng`` inside a forked RNG state (global torch state is restored on exit).

    Follows the project convention ``torch.manual_seed(int(rng.integers(2**31 - 1)))`` without
    leaking the seed to the caller's torch state.
    """
    seed = int(rng.integers(2**31 - 1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield
