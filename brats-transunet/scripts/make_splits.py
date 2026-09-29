#!/usr/bin/env python3
"""
Bước 2: chia ca (bệnh nhân) thành train / val / test. Chia theo CA, không theo lát cắt (tránh rò rỉ dữ liệu).

  python scripts/make_splits.py --config configs/base.yaml                 # 70/15/15, toàn bộ ca
  python scripts/make_splits.py --config configs/base.yaml --limit 100 --out data/splits_100.json   # tập nhỏ để thử
"""
import argparse
import json
import os
import random

import _path  # noqa: F401
from brats.io import make_source
from brats.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/base.yaml")
ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N ca (debug / máy yếu)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default=None)
ap.add_argument("--opts", nargs="*", default=[], help="ghi đè config, vd data.source=npz")
args = ap.parse_args()

cfg = load_config(args.config, args.opts)
ids = make_source(cfg["data"]).case_ids
rng = random.Random(args.seed)
rng.shuffle(ids)
if args.limit:
    ids = ids[: args.limit]
n = len(ids)
if n < 3:  # chỉ để chạy thử với 1-2 ca mẫu
    print("CẢNH BÁO: quá ít ca -> dùng cùng các ca cho train/val/test (chỉ để test code)")
    splits = {"seed": args.seed, "train": ids, "val": ids, "test": ids}
    out = args.out or cfg["splits"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1)
    print(f"{n} ca -> {out}")
    raise SystemExit(0)
n_tr = int(round(n * args.ratios[0]))
n_va = int(round(n * args.ratios[1]))
n_tr, n_va = max(1, min(n_tr, n - 2)), max(1, min(n_va, n - n_tr - 1))
splits = {"seed": args.seed, "train": sorted(ids[:n_tr]), "val": sorted(ids[n_tr:n_tr + n_va]),
          "test": sorted(ids[n_tr + n_va:])}
out = args.out or cfg["splits"]
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(splits, f, indent=1)
print(f"{n} ca -> train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])} -> {out}")
