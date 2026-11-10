"""Plot accuracy vs. FFN parameter count from results.csv -> results.png.

Usage:
    python plot_results.py                   # reads results.csv, writes results.png
    python plot_results.py --csv other.csv --out other.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"svd": "#2a78d6", "mpo": "#eb6834"}
LABELS = {"svd": "SVD low-rank", "mpo": "MPO / tensor train"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results.csv")
    p.add_argument("--out", default="results.png")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    base = df[df.method == "original"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    for method in ("svd", "mpo"):
        d = df[df.method == method].sort_values("ffn_params")
        x = d.ffn_params / 1e6
        c = COLORS[method]
        ax.plot(x, 100 * d.acc_healed, color=c, lw=2, marker="o", ms=6,
                label=f"{LABELS[method]}, healed")
        ax.plot(x, 100 * d.acc, color=c, lw=2, ls="--", marker="o", ms=6, mfc="white",
                label=f"{LABELS[method]}, no healing")

    ax.axhline(100 * base.acc, color=MUTED, lw=1, ls=":")
    ax.text(base.ffn_params / 1e6, 100 * base.acc + 0.6, f"original {100 * base.acc:.1f}%",
            ha="right", va="bottom", color=MUTED, fontsize=9)
    ax.axhline(50, color=GRID, lw=1)
    ax.text(1.05, 50.6, "chance (50%)", color=MUTED, fontsize=9)

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 9, 18, 28.3])
    ax.set_xticklabels(["1M", "2M", "4M", "9M", "18M", "28.3M\n(original)"])
    ax.set_xlim(1.0, 32)
    ax.set_ylim(45, 95)
    ax.set_xlabel("FFN parameters (log scale)", color=INK)
    ax.set_ylabel("SST-2 validation accuracy, %", color=INK)
    ax.set_title("DistilBERT FFN compression: accuracy vs. size (one seed)", color=INK,
                 loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
