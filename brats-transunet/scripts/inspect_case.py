#!/usr/bin/env python3
"""
Xem 1 ca: kích thước, giá trị cường độ, số voxel mỗi nhãn, và lưu ảnh minh họa.

  python scripts/inspect_case.py --config configs/base.yaml --case BraTS2021_00495
  python scripts/inspect_case.py --config configs/base.yaml --stats 50     # thống kê 50 ca đầu
"""
import argparse
import os

import numpy as np

import _path  # noqa: F401
from brats.io import make_source
from brats.preprocess import preprocess_case
from brats.utils import load_config
from brats.viz import save_overlay

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/base.yaml")
ap.add_argument("--case", default=None)
ap.add_argument("--stats", type=int, default=0)
ap.add_argument("--out", default="figures")
ap.add_argument("--opts", nargs="*", default=[], help="ghi đè config, vd data.source=npz")
args = ap.parse_args()

cfg = load_config(args.config, args.opts)
src = make_source(cfg["data"])
ids = src.case_ids
LAB = {0: "nền", 1: "NCR hoại tử", 2: "ED phù nề", 4: "ET tăng cường"}

if args.stats:
    rows = []
    for cid in ids[: args.stats]:
        raw = src.load(cid)
        c = preprocess_case(raw)
        vol = raw["seg"].size
        cnt = {k: int((raw["seg"] == k).sum()) for k in (1, 2, 4)}
        rows.append((cid, c["image"].shape[1:], *(cnt[k] / vol * 100 for k in (1, 2, 4))))
        print(f"{cid} crop={c['image'].shape[1:]}  NCR={rows[-1][2]:.2f}%  ED={rows[-1][3]:.2f}%  ET={rows[-1][4]:.2f}%")
    arr = np.array([r[2:] for r in rows])
    print("\nTrung bình %% thể tích:  NCR %.2f  ED %.2f  ET %.2f" % tuple(arr.mean(0)))
    print("Số ca KHÔNG có:         NCR %d  ED %d  ET %d" % tuple((arr == 0).sum(0)))
else:
    cid = args.case or ids[0]
    raw = src.load(cid)
    print(f"Ca {cid}")
    for i, m in enumerate(("t1", "t1ce", "t2", "flair")):
        x = raw["image"][i]
        print(f"  {m:6s} shape={x.shape} min={x.min():.0f} max={x.max():.0f} mean(não)={x[x > 0].mean():.1f}")
    u, n = np.unique(raw["seg"], return_counts=True)
    for a, b in zip(u, n):
        print(f"  nhãn {a} ({LAB.get(int(a), '?')}): {b} voxel = {b / raw['seg'].size * 100:.3f}%")
    c = preprocess_case(raw)
    print(f"  sau crop: {c['image'].shape}, bbox={c['bbox'].tolist()}")
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f"{cid}.png")
    save_overlay(c["image"], c["regions"], None, p, title=cid)
    print("  ảnh:", p)
