"""Drop-in replacements for nn.Linear built from compressed factors.

Both layers keep their factors as trainable Parameters, so the compressed model
can be fine-tuned ("healed") afterwards. In eval mode the dense weight is
rebuilt once and cached, so accuracy is measured exactly; this experiment
measures size vs. accuracy, not speed (real speedups need contraction kernels).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tn_compress.decompose import mpo_decompose, mpo_reconstruct, svd_lowrank


class LowRankLinear(nn.Module):
    """y = x @ (A @ B)^T + b   with A (out, r), B (r, in)."""

    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        A, B = svd_lowrank(linear.weight.data, rank)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.B), self.A, self.bias)   # (.., in) -> (.., r) -> (.., out)

    def n_params(self) -> int:
        return self.A.numel() + self.B.numel()


class MPOLinear(nn.Module):
    """Linear layer whose weight is an MPO / tensor train with bond dimension <= chi."""

    def __init__(self, linear: nn.Linear, out_modes: list[int], in_modes: list[int], chi: int):
        super().__init__()
        cores = mpo_decompose(linear.weight.data, out_modes, in_modes, chi)
        self.cores = nn.ParameterList(nn.Parameter(c) for c in cores)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self._cached: torch.Tensor | None = None

    def weight(self) -> torch.Tensor:
        if not self.training and self._cached is not None:
            return self._cached
        W = mpo_reconstruct(list(self.cores))
        self._cached = None if self.training else W.detach()
        return W

    def train(self, mode: bool = True):
        self._cached = None
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight().to(x.dtype), self.bias)

    def n_params(self) -> int:
        return sum(c.numel() for c in self.cores)
