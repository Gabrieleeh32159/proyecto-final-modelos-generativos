"""Plot the training loss curve from loss.csv.

Usage:
    python world_model/plot_loss.py                      # -> world_model/results/loss.png
    python world_model/plot_loss.py --window 50
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"  # dataviz series-1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="world_model/checkpoints/loss.csv")
    p.add_argument("--out", default="world_model/results/loss.png")
    p.add_argument("--window", type=int, default=20,
                   help="rolling-mean window, in logged rows")
    args = p.parse_args()

    with open(args.csv) as f:
        rows = [(int(s), float(l)) for s, l in csv.reader(f)]
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    steps, losses = map(np.array, zip(*sorted(rows)))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    has_smooth = len(losses) >= args.window
    ax.plot(steps, losses, color=BLUE, alpha=0.25 if has_smooth else 1.0,
            linewidth=1 if has_smooth else 2, label="loss")
    if has_smooth:
        smooth = np.convolve(losses, np.ones(args.window) / args.window, mode="valid")
        ax.plot(steps[args.window - 1:], smooth, color=BLUE, linewidth=2,
                label=f"rolling mean ({args.window})")
        ax.legend(frameon=False)
    ax.set_xlabel("training step")
    ax.set_ylabel("v-prediction MSE")
    ax.set_title(f"training loss — step {steps[-1]:,}, last {losses[-1]:.4f}")
    ax.grid(alpha=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}  ({len(rows)} points, last loss {losses[-1]:.4f})")


if __name__ == "__main__":
    main()
