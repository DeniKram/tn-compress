"""Layer sensitivity: compress ONE FFN matrix at a time and measure the accuracy drop.

Answers two questions:
  1. Which of the 12 FFN matrices are fragile, and which can be compressed for free?
     (CompactifAI does a similar layer-sensitivity profiling before tensorizing.)
  2. Why is MPO chi=192 less accurate than chi=96 before healing?
     If one or two layers show a much bigger drop at chi=192 than at chi=96,
     the dip comes from those layers, not from the method as a whole.

No healing here: each run is a single forward pass over the 872 SST-2 validation
sentences, so 12 layers x 4 settings = 48 evaluations (a few minutes on a GPU).

Usage:
    python layer_sensitivity.py
    python layer_sensitivity.py --settings mpo:96 mpo:192 svd:96 svd:192
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch

from tn_compress.data import sst2_loader
from tn_compress.decompose import mpo_decompose, mpo_reconstruct, relative_error, svd_lowrank
from tn_compress.evaluate import accuracy
from tn_compress.model import MODES_768, MODES_3072, compress, ffn_layers, load_base


def layer_error(lin: torch.nn.Linear, method: str, value: int) -> float:
    """Relative Frobenius error of one compressed weight matrix."""
    W = lin.weight.data
    if method == "svd":
        A, B = svd_lowrank(W, value)
        return relative_error(W, A @ B)
    out_f, in_f = W.shape
    om = MODES_3072 if out_f == 3072 else MODES_768
    im = MODES_3072 if in_f == 3072 else MODES_768
    return relative_error(W, mpo_reconstruct(mpo_decompose(W, om, im, value)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--settings", nargs="+", default=["mpo:96", "mpo:192", "svd:96", "svd:192"],
                   help="method:value pairs, e.g. mpo:192 svd:96")
    p.add_argument("--out", default="layer_sensitivity.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    tok, base = load_base()
    val = sst2_loader(tok, "validation", batch_size=64)
    base_acc = accuracy(base, val, device)
    print(f"original acc = {base_acc:.4f}\n")

    layers = ffn_layers(base)
    names = [f"block{i // 2}.{name}" for i, (_, name, _) in enumerate(layers)]

    rows = []
    for setting in args.settings:
        method, value = setting.split(":")
        value = int(value)
        for idx, (_, _, lin) in enumerate(layers):
            model = compress(base, method, value, only={idx})
            acc = accuracy(model, val, device)
            err = layer_error(lin, method, value)
            rows.append(dict(setting=setting, layer=names[idx], rel_error=err,
                             acc=acc, drop_pp=100 * (base_acc - acc)))
            print(f"{setting:>8}  {names[idx]:<12} rel.err={err:.4f}  acc={acc:.4f}  "
                  f"drop={100 * (base_acc - acc):5.2f} pp")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\nAccuracy drop in percentage points (rows = layer, columns = setting):")
    print(df.pivot(index="layer", columns="setting", values="drop_pp")
            .reindex(names).to_markdown(floatfmt=".2f"))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
