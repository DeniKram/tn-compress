"""Matrix factorizations used for compression.

Two methods, both applied to a weight matrix W of shape (out, in):

1. SVD low-rank:  W (out, in) ~= A (out, r) @ B (r, in)
2. MPO / tensor train ("tensor network"):
   out = o_1 * ... * o_d,  in = i_1 * ... * i_d
   W is reshaped to a 2d-dimensional tensor, the indices are paired as
   (o_1 i_1)(o_2 i_2)...(o_d i_d), and TT-SVD splits it into d cores:
       core_k has shape (r_{k-1}, o_k, i_k, r_k),  r_0 = r_d = 1
   The bond dimension chi caps every r_k and controls size vs. accuracy.
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------- SVD
def svd_lowrank(W: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-r approximation of W (Eckart-Young). Returns A (out, r), B (r, in)."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    r = min(rank, S.numel())
    sqrt_s = S[:r].sqrt()
    A = U[:, :r] * sqrt_s            # (out, r)
    B = sqrt_s[:, None] * Vh[:r]     # (r, in)
    return A, B


def svd_params(out_f: int, in_f: int, rank: int) -> int:
    return rank * (out_f + in_f)


# ---------------------------------------------------------------- MPO
def mpo_decompose(
    W: torch.Tensor, out_modes: list[int], in_modes: list[int], max_rank: int
) -> list[torch.Tensor]:
    """TT-SVD of W (out, in) into MPO cores of shape (r_{k-1}, o_k, i_k, r_k)."""
    out_f, in_f = W.shape
    d = len(out_modes)
    assert len(in_modes) == d, "out_modes and in_modes must have the same length"
    assert math.prod(out_modes) == out_f, f"prod(out_modes) != {out_f}"
    assert math.prod(in_modes) == in_f, f"prod(in_modes) != {in_f}"

    # (out, in) -> (o_1..o_d, i_1..i_d) -> (o_1, i_1, o_2, i_2, ..., o_d, i_d)
    T = W.float().reshape(*out_modes, *in_modes)
    perm = [p for k in range(d) for p in (k, d + k)]
    T = T.permute(*perm).contiguous()

    cores: list[torch.Tensor] = []
    r_prev = 1
    C = T.reshape(1, -1)
    for k in range(d - 1):
        n_k = out_modes[k] * in_modes[k]
        C = C.reshape(r_prev * n_k, -1)                  # (r_{k-1} * o_k * i_k, rest)
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        r = min(max_rank, S.numel())
        cores.append(U[:, :r].reshape(r_prev, out_modes[k], in_modes[k], r))
        C = S[:r, None] * Vh[:r]                         # (r_k, rest)
        r_prev = r
    cores.append(C.reshape(r_prev, out_modes[-1], in_modes[-1], 1))
    return cores


def mpo_reconstruct(cores: list[torch.Tensor]) -> torch.Tensor:
    """Contract MPO cores back into a dense (out, in) matrix."""
    out_modes = [c.shape[1] for c in cores]
    in_modes = [c.shape[2] for c in cores]
    d = len(cores)

    T = cores[0]                                         # (1, o_1, i_1, r_1)
    for c in cores[1:]:
        # (..., r) x (r, o, i, r') -> (..., o, i, r')
        T = torch.tensordot(T, c, dims=([T.dim() - 1], [0]))
    T = T.squeeze(0).squeeze(-1)                         # (o_1, i_1, ..., o_d, i_d)
    perm = [2 * k for k in range(d)] + [2 * k + 1 for k in range(d)]
    T = T.permute(*perm).contiguous()                    # (o_1..o_d, i_1..i_d)
    return T.reshape(math.prod(out_modes), math.prod(in_modes))


def mpo_params(cores: list[torch.Tensor]) -> int:
    return sum(c.numel() for c in cores)


def relative_error(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    return (torch.linalg.norm(W.float() - W_hat) / torch.linalg.norm(W.float())).item()
