#!/usr/bin/env python3
"""Vẽ đường cong loss và Dice từ runs/<tên>/log.csv (có thể truyền nhiều file để so sánh)."""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--out", default="figures/training_curves.png")
args = ap.parse_args()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for p in args.logs:
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    name = os.path.basename(os.path.dirname(p))
    ax[0].plot([int(r["epoch"]) for r in rows], [float(r["train_loss"]) for r in rows], label=name)
    v = [r for r in rows if r["dice_mean"]]
    for k, ls in (("dice_mean", "-"), ("dice_ET", ":")):
        ax[1].plot([int(r["epoch"]) for r in v], [float(r[k]) for r in v], ls, label=f"{name} {k}")
ax[0].set(title="Train loss", xlabel="epoch"); ax[1].set(title="Val Dice", xlabel="epoch", ylim=(0, 1))
for a in ax:
    a.legend(fontsize=8); a.grid(alpha=.3)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.tight_layout(); fig.savefig(args.out, dpi=120)
print("saved", args.out)
