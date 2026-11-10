import math

import torch

from tn_compress.decompose import (mpo_decompose, mpo_params, mpo_reconstruct,
                                   relative_error, svd_lowrank)

torch.manual_seed(0)


def test_svd_full_rank_is_exact():
    W = torch.randn(48, 32)
    A, B = svd_lowrank(W, rank=32)
    assert relative_error(W, A @ B) < 1e-5


def test_svd_error_decreases_with_rank():
    W = torch.randn(64, 64)
    errs = [relative_error(W, (lambda a, b: a @ b)(*svd_lowrank(W, r))) for r in (4, 16, 64)]
    assert errs[0] > errs[1] > errs[2]


def test_mpo_full_rank_is_exact():
    W = torch.randn(4 * 6 * 4, 3 * 2 * 5)
    cores = mpo_decompose(W, [4, 6, 4], [3, 2, 5], max_rank=10_000)
    assert relative_error(W, mpo_reconstruct(cores)) < 1e-5


def test_mpo_core_shapes_and_params():
    W = torch.randn(3072, 768)
    cores = mpo_decompose(W, [4, 8, 8, 12], [4, 6, 4, 8], max_rank=16)
    shapes = [tuple(c.shape) for c in cores]
    assert shapes[0][0] == 1 and shapes[-1][-1] == 1
    for (a, *_ , r), (r2, *_ ) in zip(shapes, shapes[1:]):
        assert r == r2
    assert all(s[3] <= 16 for s in shapes)
    assert mpo_params(cores) < W.numel() / 10
    assert mpo_reconstruct(cores).shape == (3072, 768)


def test_mpo_error_decreases_with_chi():
    W = torch.randn(4 * 6 * 4 * 8, 4 * 8 * 8 * 12) * 0.02
    errs = [relative_error(W, mpo_reconstruct(mpo_decompose(W, [4, 6, 4, 8], [4, 8, 8, 12], c)))
            for c in (4, 16, 64)]
    assert errs[0] > errs[1] > errs[2]
