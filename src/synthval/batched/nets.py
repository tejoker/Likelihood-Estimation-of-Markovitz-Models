"""Critic banks: K independent critics of one architecture held in one network.

A bank stores the parameters of ``K`` critics stacked along a leading axis and evaluates all of
them in one pass: grouped ``Conv1d(groups=K)`` for the time-pooled CNN, batched matrix products
(``baddbmm``) for the dense layers. Input ``x`` has shape ``(n, K, *sample_shape)`` (slice
``x[:, k]`` is the batch of critic ``k``) and the output has shape ``(K, n)``.

**Independence.** Critic ``k``'s output depends only on ``x[:, k]`` and on slice ``k`` of every
parameter (groups of a grouped convolution never mix; ``bmm`` multiplies matching batch
entries), so the gradient of ``sum_k loss_k`` with respect to slice ``k`` is exactly the gradient
of ``loss_k`` alone, and Adam (elementwise) acts as ``K`` independent Adams. The only difference
from training the ``K`` critics one by one is floating-point rounding (grouped and single
convolutions, ``bmm`` and ``addmm`` may sum in different orders).

``impl="loop"`` evaluates the critics one after the other with exactly the operations of the
single networks (:class:`~synthval.twosample.nets.PooledConvCritic`,
:class:`~synthval.twosample.nets.MLPCritic`); it is slow and exists to validate the bank
machinery (random streams, gathering, gradient separation, optimiser) against the sequential
code.

Supported architectures: :class:`~synthval.twosample.nets.PooledConvCritic` (univariate ``(L,)``
and multivariate ``(C, L)`` windows) and :class:`~synthval.twosample.nets.MLPCritic` (any sample
shape, flattened). :class:`~synthval.twosample.nets.ConvCritic` has no bank.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from ..twosample.nets import MLPCritic, PooledConvCritic

BankImpl = Literal["grouped", "loop"]

__all__ = ["BankImpl", "CriticBank", "PooledConvCriticBank", "MLPCriticBank", "build_bank"]


class CriticBank(nn.Module):
    """Base class: ``K`` critics; ``forward`` maps ``(n, K, *sample_shape)`` to ``(K, n)``.

    Parameters are allocated with zeros: a bank is meaningful only after every slice has been
    loaded from a single network with :meth:`load_critic` (see :func:`build_bank`).

    Parameters
    ----------
    K : int
        Number of critics.
    impl : {"grouped", "loop"}
        ``"grouped"``: one grouped / batched pass (fast). ``"loop"``: one critic after the other
        with the single networks' operations (reference for tests).
    """

    single_type: type[nn.Module] = nn.Module

    def __init__(self, K: int, impl: BankImpl = "grouped") -> None:
        super().__init__()
        if int(K) < 1:
            raise ValueError("K must be >= 1")
        if impl not in ("grouped", "loop"):
            raise ValueError("impl must be 'grouped' or 'loop'")
        self.K = int(K)
        self.impl: BankImpl = impl

    # -- correspondence with the single networks ---------------------------------------------
    @classmethod
    def spec_of(cls, net: nn.Module) -> dict[str, Any]:
        """Hyperparameters of a single network (the bank constructor's keyword arguments)."""
        raise NotImplementedError

    def spec(self) -> dict[str, Any]:
        """Hyperparameters shared by the ``K`` critics."""
        raise NotImplementedError

    def _pairs(self, net: nn.Module) -> list[tuple[nn.Parameter, torch.Tensor]]:
        """``(bank parameter, single-network parameter)`` pairs (slice ``k`` <-> the network)."""
        raise NotImplementedError

    def _check_net(self, net: nn.Module) -> None:
        if type(net) is not self.single_type:
            raise TypeError(f"{type(self).__name__} holds {self.single_type.__name__} critics, "
                            f"got {type(net).__name__}")
        if self.spec_of(net) != self.spec():
            raise ValueError(f"network hyperparameters {self.spec_of(net)} differ from the "
                             f"bank's {self.spec()}")

    @torch.no_grad()
    def load_critic(self, k: int, net: nn.Module) -> None:
        """Copy the parameters of the single network ``net`` into slice ``k``."""
        self._check_net(net)
        for bank_p, p in self._pairs(net):
            bank_p[k].copy_(p.detach().reshape(bank_p.shape[1:]))

    @torch.no_grad()
    def store_critic(self, k: int, net: nn.Module) -> None:
        """Copy slice ``k`` into the parameters of the single network ``net`` (in place)."""
        self._check_net(net)
        for bank_p, p in self._pairs(net):
            p.copy_(bank_p[k].detach().reshape(p.shape))

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() < 3 or x.shape[1] != self.K:
            raise ValueError(f"bank input must have shape (n, K={self.K}, ...), got "
                             f"{tuple(x.shape)}")


class PooledConvCriticBank(CriticBank):
    """``K`` :class:`~synthval.twosample.nets.PooledConvCritic` critics (grouped convolutions).

    Parameters
    ----------
    K : int
        Number of critics.
    in_channels, hidden, n_conv, kernel_size, layer_norm, negative_slope :
        As :class:`~synthval.twosample.nets.PooledConvCritic`.
    ln_eps : float, default 1e-5
        LayerNorm epsilon (``nn.LayerNorm`` default).
    impl : {"grouped", "loop"}, default "grouped"
        See :class:`CriticBank`.

    Notes
    -----
    Input ``(n, K, L)`` (requires ``in_channels == 1``) or ``(n, K, C, L)``. The grouped pass
    reshapes it to ``(n, K * C, L)``; convolution ``j`` is ``Conv1d(K * c_j, K * hidden,
    groups=K)`` with weight ``conv_w[j].reshape(K * hidden, c_j, kernel_size)``; the head uses
    ``baddbmm`` on ``(K, n, hidden)``.
    """

    single_type = PooledConvCritic

    def __init__(self, K: int, in_channels: int, hidden: int = 64, *, n_conv: int = 2,
                 kernel_size: int = 5, layer_norm: bool = False, negative_slope: float = 0.2,
                 ln_eps: float = 1e-5, impl: BankImpl = "grouped") -> None:
        super().__init__(K, impl)
        if in_channels < 1 or hidden < 1 or n_conv < 1 or kernel_size < 1:
            raise ValueError("in_channels, hidden, n_conv and kernel_size must be positive")
        self.in_channels, self.hidden, self.n_conv = int(in_channels), int(hidden), int(n_conv)
        self.kernel_size, self.layer_norm = int(kernel_size), bool(layer_norm)
        self.negative_slope, self.ln_eps = float(negative_slope), float(ln_eps)
        self.conv_w = nn.ParameterList()
        self.conv_b = nn.ParameterList()
        self.ln_w = nn.ParameterList()
        self.ln_b = nn.ParameterList()
        c = self.in_channels
        for _ in range(self.n_conv):
            self.conv_w.append(nn.Parameter(torch.zeros(self.K, hidden, c, kernel_size)))
            self.conv_b.append(nn.Parameter(torch.zeros(self.K, hidden)))
            if self.layer_norm:
                self.ln_w.append(nn.Parameter(torch.ones(self.K, hidden)))
                self.ln_b.append(nn.Parameter(torch.zeros(self.K, hidden)))
            c = hidden
        self.head_w1 = nn.Parameter(torch.zeros(self.K, hidden, hidden))
        self.head_b1 = nn.Parameter(torch.zeros(self.K, hidden))
        self.head_w2 = nn.Parameter(torch.zeros(self.K, 1, hidden))
        self.head_b2 = nn.Parameter(torch.zeros(self.K, 1))

    @classmethod
    def spec_of(cls, net: nn.Module) -> dict[str, Any]:
        if not isinstance(net, PooledConvCritic):
            raise TypeError(f"expected PooledConvCritic, got {type(net).__name__}")
        conv0 = net.convs[0]
        ln = isinstance(net.norms[0], nn.LayerNorm)
        for conv in net.convs:
            if conv.stride != (1,) or conv.dilation != (1,) or conv.groups != 1 or \
                    conv.padding != "same" or conv.padding_mode != "zeros":
                raise ValueError("unsupported convolution settings in PooledConvCritic")
        return {"in_channels": int(net.in_channels), "hidden": int(conv0.out_channels),
                "n_conv": len(net.convs), "kernel_size": int(conv0.kernel_size[0]),
                "layer_norm": ln, "negative_slope": float(net.act.negative_slope),
                "ln_eps": float(net.norms[0].eps) if ln else 1e-5}

    def spec(self) -> dict[str, Any]:
        return {"in_channels": self.in_channels, "hidden": self.hidden, "n_conv": self.n_conv,
                "kernel_size": self.kernel_size, "layer_norm": self.layer_norm,
                "negative_slope": self.negative_slope, "ln_eps": self.ln_eps}

    def _pairs(self, net: nn.Module) -> list[tuple[nn.Parameter, torch.Tensor]]:
        pairs: list[tuple[nn.Parameter, torch.Tensor]] = []
        for j, conv in enumerate(net.convs):
            pairs += [(self.conv_w[j], conv.weight), (self.conv_b[j], conv.bias)]
            if self.layer_norm:
                pairs += [(self.ln_w[j], net.norms[j].weight), (self.ln_b[j], net.norms[j].bias)]
        pairs += [(self.head_w1, net.head[0].weight), (self.head_b1, net.head[0].bias),
                  (self.head_w2, net.head[2].weight), (self.head_b2, net.head[2].bias)]
        return pairs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Critic values ``(K, n)`` of inputs ``(n, K, L)`` or ``(n, K, C, L)``."""
        self._check_input(x)
        if x.dim() == 3:
            if self.in_channels != 1:
                raise ValueError("(n, K, L) input requires in_channels == 1")
            x = x.unsqueeze(2)
        if x.dim() != 4 or x.shape[2] != self.in_channels:
            raise ValueError(f"expected (n, K, {self.in_channels}, L) input, got "
                             f"{tuple(x.shape)}")
        if self.impl == "loop":
            return self._forward_loop(x)
        n, K, C, L = x.shape
        H, slope = self.hidden, self.negative_slope
        h = x.reshape(n, K * C, L)
        for j in range(self.n_conv):
            w = self.conv_w[j]
            h = F.conv1d(h, w.reshape(K * H, w.shape[2], w.shape[3]), self.conv_b[j].reshape(-1),
                         padding="same", groups=K)
            if self.layer_norm:
                h4 = F.layer_norm(h.view(n, K, H, L).transpose(2, 3), (H,), eps=self.ln_eps)
                h4 = h4 * self.ln_w[j][:, None, :] + self.ln_b[j][:, None, :]
                h = h4.transpose(2, 3).reshape(n, K * H, L)
            h = F.leaky_relu(h, slope)
        z = h.view(n, K, H, L).mean(dim=-1).transpose(0, 1)                       # (K, n, H)
        z = F.leaky_relu(torch.baddbmm(self.head_b1.unsqueeze(1), z,
                                       self.head_w1.transpose(1, 2)), slope)
        out = torch.baddbmm(self.head_b2.unsqueeze(1), z, self.head_w2.transpose(1, 2))
        return out.squeeze(-1)

    def _forward_loop(self, x: torch.Tensor) -> torch.Tensor:
        slope = self.negative_slope
        outs = []
        for k in range(self.K):
            h = x[:, k].contiguous()
            for j in range(self.n_conv):
                h = F.conv1d(h, self.conv_w[j][k], self.conv_b[j][k], padding="same")
                if self.layer_norm:
                    h = F.layer_norm(h.transpose(1, 2), (self.hidden,), self.ln_w[j][k],
                                     self.ln_b[j][k], self.ln_eps).transpose(1, 2)
                h = F.leaky_relu(h, slope)
            z = F.leaky_relu(F.linear(h.mean(dim=-1), self.head_w1[k], self.head_b1[k]), slope)
            outs.append(F.linear(z, self.head_w2[k], self.head_b2[k]).squeeze(-1))
        return torch.stack(outs)


class MLPCriticBank(CriticBank):
    """``K`` :class:`~synthval.twosample.nets.MLPCritic` critics (batched matrix products).

    Parameters
    ----------
    K : int
        Number of critics.
    in_dim, hidden, depth, layer_norm, negative_slope :
        As :class:`~synthval.twosample.nets.MLPCritic`.
    ln_eps : float, default 1e-5
        LayerNorm epsilon.
    impl : {"grouped", "loop"}, default "grouped"
        See :class:`CriticBank`.
    """

    single_type = MLPCritic

    def __init__(self, K: int, in_dim: int, hidden: int = 128, depth: int = 3, *,
                 layer_norm: bool = False, negative_slope: float = 0.2, ln_eps: float = 1e-5,
                 impl: BankImpl = "grouped") -> None:
        super().__init__(K, impl)
        if in_dim < 1 or hidden < 1 or depth < 1:
            raise ValueError("in_dim, hidden and depth must be positive")
        self.in_dim, self.hidden, self.depth = int(in_dim), int(hidden), int(depth)
        self.layer_norm, self.negative_slope = bool(layer_norm), float(negative_slope)
        self.ln_eps = float(ln_eps)
        self.lin_w = nn.ParameterList()
        self.lin_b = nn.ParameterList()
        self.ln_w = nn.ParameterList()
        self.ln_b = nn.ParameterList()
        d = self.in_dim
        for _ in range(self.depth):
            self.lin_w.append(nn.Parameter(torch.zeros(self.K, hidden, d)))
            self.lin_b.append(nn.Parameter(torch.zeros(self.K, hidden)))
            if self.layer_norm:
                self.ln_w.append(nn.Parameter(torch.ones(self.K, hidden)))
                self.ln_b.append(nn.Parameter(torch.zeros(self.K, hidden)))
            d = hidden
        self.lin_w.append(nn.Parameter(torch.zeros(self.K, 1, d)))
        self.lin_b.append(nn.Parameter(torch.zeros(self.K, 1)))

    @staticmethod
    def _layers(net: nn.Module) -> tuple[list[nn.Linear], list[nn.LayerNorm], list[nn.Module]]:
        mods = list(net.net)
        return ([m for m in mods if isinstance(m, nn.Linear)],
                [m for m in mods if isinstance(m, nn.LayerNorm)],
                [m for m in mods if isinstance(m, nn.LeakyReLU)])

    @classmethod
    def spec_of(cls, net: nn.Module) -> dict[str, Any]:
        if not isinstance(net, MLPCritic):
            raise TypeError(f"expected MLPCritic, got {type(net).__name__}")
        lins, lns, acts = cls._layers(net)
        return {"in_dim": int(net.in_dim), "hidden": int(lins[0].out_features),
                "depth": len(lins) - 1, "layer_norm": bool(lns),
                "negative_slope": float(acts[0].negative_slope),
                "ln_eps": float(lns[0].eps) if lns else 1e-5}

    def spec(self) -> dict[str, Any]:
        return {"in_dim": self.in_dim, "hidden": self.hidden, "depth": self.depth,
                "layer_norm": self.layer_norm, "negative_slope": self.negative_slope,
                "ln_eps": self.ln_eps}

    def _pairs(self, net: nn.Module) -> list[tuple[nn.Parameter, torch.Tensor]]:
        lins, lns, _ = self._layers(net)
        pairs: list[tuple[nn.Parameter, torch.Tensor]] = []
        for j, lin in enumerate(lins):
            pairs += [(self.lin_w[j], lin.weight), (self.lin_b[j], lin.bias)]
        for j, ln in enumerate(lns):
            pairs += [(self.ln_w[j], ln.weight), (self.ln_b[j], ln.bias)]
        return pairs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Critic values ``(K, n)`` of inputs ``(n, K, *sample_shape)`` (flattened per critic)."""
        self._check_input(x)
        n, K = x.shape[0], self.K
        x = x.reshape(n, K, -1)
        if x.shape[2] != self.in_dim:
            raise ValueError(f"flattened sample size {x.shape[2]} != in_dim {self.in_dim}")
        slope = self.negative_slope
        if self.impl == "loop":
            outs = []
            for k in range(K):
                h = x[:, k].contiguous()
                for j in range(self.depth):
                    h = F.linear(h, self.lin_w[j][k], self.lin_b[j][k])
                    if self.layer_norm:
                        h = F.layer_norm(h, (self.hidden,), self.ln_w[j][k], self.ln_b[j][k],
                                         self.ln_eps)
                    h = F.leaky_relu(h, slope)
                outs.append(F.linear(h, self.lin_w[-1][k], self.lin_b[-1][k]).squeeze(-1))
            return torch.stack(outs)
        h = x.transpose(0, 1)                                                     # (K, n, d)
        for j in range(self.depth):
            h = torch.baddbmm(self.lin_b[j].unsqueeze(1), h, self.lin_w[j].transpose(1, 2))
            if self.layer_norm:
                h = F.layer_norm(h, (self.hidden,), eps=self.ln_eps)
                h = h * self.ln_w[j][:, None, :] + self.ln_b[j][:, None, :]
            h = F.leaky_relu(h, slope)
        out = torch.baddbmm(self.lin_b[-1].unsqueeze(1), h, self.lin_w[-1].transpose(1, 2))
        return out.squeeze(-1)


_BANKS: dict[type, type[CriticBank]] = {PooledConvCritic: PooledConvCriticBank,
                                        MLPCritic: MLPCriticBank}


def build_bank(nets: Sequence[nn.Module], impl: BankImpl = "grouped") -> CriticBank:
    """Bank holding copies of the single networks ``nets`` (same class and hyperparameters).

    Parameters
    ----------
    nets : sequence of nn.Module
        ``K >= 1`` :class:`~synthval.twosample.nets.PooledConvCritic` or
        :class:`~synthval.twosample.nets.MLPCritic` networks on one device.
    impl : {"grouped", "loop"}, default "grouped"
        See :class:`CriticBank`.

    Returns
    -------
    CriticBank
        On the device of ``nets[0]``; slice ``k`` holds the parameters of ``nets[k]``.
    """
    if len(nets) == 0:
        raise ValueError("need at least one network")
    cls = _BANKS.get(type(nets[0]))
    if cls is None:
        raise NotImplementedError(f"no critic bank for {type(nets[0]).__name__} (supported: "
                                  "PooledConvCritic, MLPCritic)")
    device = next(nets[0].parameters()).device
    bank = cls(len(nets), **cls.spec_of(nets[0]), impl=impl).to(device)
    for k, net in enumerate(nets):
        bank.load_critic(k, net)
    return bank
