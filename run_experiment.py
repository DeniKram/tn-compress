"""Compress DistilBERT's FFN layers with SVD and MPO and measure size vs. accuracy.

Usage:
    python run_experiment.py                 # compression only (~2-5 min on a GPU)
    python run_experiment.py --heal          # + short fine-tune of each compressed model
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import torch

from tn_compress.data import sst2_loader
from tn_compress.evaluate import accuracy
from tn_compress.heal import heal
from tn_compress.model import compress, ffn_params, load_base

SVD_RANKS = [384, 192, 96, 48, 24]          # FFN 1.6x, 3.2x, 6.4x, 12.8x, 25.6x smaller
MPO_CHIS = [384, 192, 96, 64, 48]           # FFN 1.6x, 3.2x, 6.2x, 12.6x, 20.4x smaller (same budgets)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--heal", action="store_true", help="fine-tune each compressed model briefly")
    p.add_argument("--heal-samples", type=int, default=8000)
    p.add_argument("--out", default="results.csv")
    p.add_argument("--seed", type=int, default=0, help="seed for healing (shuffle order, dropout)")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    tok, base = load_base()
    val = sst2_loader(tok, "validation", batch_size=64)
    train = sst2_loader(tok, "train", batch_size=32, limit=args.heal_samples, shuffle=True) if args.heal else None

    base_ffn = ffn_params(base)
    base_total = sum(p.numel() for p in base.parameters())
    t0 = time.time()
    base_acc = accuracy(base, val, device)
    rows = [dict(method="original", setting="-", ffn_params=base_ffn, ffn_ratio=1.0,
                 model_params=base_total, acc=base_acc, acc_healed=None)]
    print(f"original: acc={base_acc:.4f}  FFN params={base_ffn:,}  ({time.time() - t0:.0f}s)")

    configs = [("svd", r) for r in SVD_RANKS] + [("mpo", c) for c in MPO_CHIS]
    for method, value in configs:
        model = compress(base, method, value)
        f = ffn_params(model)
        total = base_total - base_ffn + f
        acc = accuracy(model, val, device)
        row = dict(method=method, setting=f"{'rank' if method == 'svd' else 'chi'}={value}",
                   ffn_params=f, ffn_ratio=round(base_ffn / f, 1), model_params=total,
                   acc=acc, acc_healed=None)
        if args.heal:
            heal(model, train, device)
            row["acc_healed"] = accuracy(model, val, device)
        rows.append(row)
        healed = f"  healed={row['acc_healed']:.4f}" if args.heal else ""
        print(f"{method} {row['setting']:>9}: FFN {f:>10,} ({row['ffn_ratio']}x smaller)  acc={acc:.4f}{healed}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n" + df.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
