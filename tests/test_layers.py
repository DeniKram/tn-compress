import torch
import torch.nn as nn

from tn_compress.layers import LowRankLinear, MPOLinear

torch.manual_seed(0)


def test_lowrank_full_rank_matches_linear():
    lin = nn.Linear(32, 48)
    layer = LowRankLinear(lin, rank=32)
    x = torch.randn(5, 32)
    assert torch.allclose(layer(x), lin(x), atol=1e-4)


def test_mpo_full_rank_matches_linear_and_is_trainable():
    lin = nn.Linear(4 * 3 * 2, 2 * 5 * 3)          # in 24 = 4*3*2, out 30 = 2*5*3
    layer = MPOLinear(lin, out_modes=[2, 5, 3], in_modes=[4, 3, 2], chi=10_000)
    x = torch.randn(7, 24)
    layer.eval()
    assert torch.allclose(layer(x), lin(x), atol=1e-4)
    layer.train()
    layer(x).sum().backward()
    assert all(c.grad is not None for c in layer.cores)
