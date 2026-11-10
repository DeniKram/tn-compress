"""Mean relative reconstruction error of the FFN matrices for each setting.

If error falls steadily with rank/chi, the chi=192 accuracy dip comes from how
errors propagate through the network, not from a bug in the decomposition.
"""
import torch

from run_experiment import MPO_CHIS, SVD_RANKS
from tn_compress.decompose import mpo_decompose, mpo_reconstruct, relative_error, svd_lowrank
from tn_compress.model import MODES_768, MODES_3072, ffn_layers, load_base


def main() -> None:
    _, model = load_base()
    layers = [lin for _, _, lin in ffn_layers(model)]  # 12 Linear layers

    for r in SVD_RANKS:
        errs = []
        for lin in layers:
            A, B = svd_lowrank(lin.weight.data, r)
            errs.append(relative_error(lin.weight.data, A @ B))
        print(f"svd rank={r:>3}: mean rel. error = {sum(errs) / len(errs):.4f}")

    for chi in MPO_CHIS:
        errs = []
        for lin in layers:
            out_f, in_f = lin.weight.shape
            om = MODES_3072 if out_f == 3072 else MODES_768
            im = MODES_3072 if in_f == 3072 else MODES_768
            W_hat = mpo_reconstruct(mpo_decompose(lin.weight.data, om, im, chi))
            errs.append(relative_error(lin.weight.data, W_hat))
        print(f"mpo chi={chi:>3}:  mean rel. error = {sum(errs) / len(errs):.4f}")


if __name__ == "__main__":
    main()