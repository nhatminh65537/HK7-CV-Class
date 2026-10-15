#!/usr/bin/env python3
"""
So sánh kết quả đánh giá của nhiều lượt trên CÙNG các ca (đọc metrics.csv do evaluate.py ghi).

  python scripts/compare_runs.py runs/unet2d/eval_test runs/transunet2d/eval_test
  python scripts/compare_runs.py runs/transunet2d/eval_test runs/transunet2d_fixed/eval_test runs/transunet2d_lite/eval_test

Lượt đầu tiên là mốc. In ra:
- trung bình / trung vị Dice từng vùng, HD95 trung vị, số ca Dice < 0.5, giây/ca (nếu có cột sec);
- với từng lượt sau: chênh lệch so với mốc tính trên từng ca (điểm Dice x100), khoảng tin cậy 95%
  (bootstrap) và kiểm định Wilcoxon. Khoảng tin cậy chứa 0 nghĩa là chưa đủ bằng chứng để nói lượt nào hơn.
"""
import argparse
import csv
import os

import numpy as np
from scipy.stats import wilcoxon

REG = ["dice_WT", "dice_TC", "dice_ET", "dice_mean"]

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+", help="thư mục eval_<split> (hoặc file metrics.csv); lượt đầu là mốc")
ap.add_argument("--boot", type=int, default=5000)
args = ap.parse_args()


def load(p):
    p = p if p.endswith(".csv") else os.path.join(p, "metrics.csv")
    with open(p, encoding="utf-8") as f:
        return {r["id"]: {k: float(v) for k, v in r.items() if k != "id" and v != ""} for r in csv.DictReader(f)}


runs = [(d.rstrip("/\\"), load(d)) for d in args.dirs]
ids = sorted(set.intersection(*(set(r) for _, r in runs)))
print(f"{len(ids)} ca chung" + "".join(f" | {n}: {len(r)} ca" for n, r in runs))

print(f"\n{'lượt':40s} " + " ".join(f"{k[5:]:>11s}" for k in REG) + f" {'HD95 WT':>8s} {'<0.5':>5s} {'giây/ca':>8s}")
print(f"{'':40s} " + " ".join(f"{'TB / TV':>11s}" for _ in REG) + f" {'(TV)':>8s}")
for name, r in runs:
    cells = []
    for k in REG:
        v = np.array([r[i][k] for i in ids])
        cells.append(f"{v.mean():.3f}/{np.median(v):.3f}")
    hd = np.array([r[i].get("hd95_WT", np.nan) for i in ids])
    sec = np.array([r[i].get("sec", np.nan) for i in ids])
    fail = sum(r[i]["dice_mean"] < 0.5 for i in ids)
    print(f"{name[-40:]:40s} " + " ".join(f"{c:>11s}" for c in cells)
          + f" {np.nanmedian(hd) if not np.all(np.isnan(hd)) else float('nan'):8.2f} {fail:5d} {np.nanmean(sec) if not np.all(np.isnan(sec)) else float('nan'):8.2f}")

base_name, base = runs[0]
rng = np.random.default_rng(0)
for name, r in runs[1:]:
    print(f"\n{name}  so với  {base_name}  (điểm Dice x100, dương = tốt hơn mốc)")
    for k in REG:
        d = np.array([r[i][k] - base[i][k] for i in ids]) * 100
        bs = rng.choice(d, (args.boot, len(d))).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        verdict = "hơn" if lo > 0 else ("kém" if hi < 0 else "chưa phân biệt")
        print(f"  {k:10s} {d.mean():+6.2f}  CI95 [{lo:+.2f}, {hi:+.2f}]  hơn ở {np.sum(d > 0)}/{len(d)} ca  "
              f"Wilcoxon p={p:.3f}  -> {verdict}")
