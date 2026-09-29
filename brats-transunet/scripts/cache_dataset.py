#!/usr/bin/env python3
"""
(Tuỳ chọn) Tạo cache .npz từ tar để train nhanh hơn: crop vùng não, giữ int16, nén.
Ghi từng ca một, có thể dừng/chạy tiếp. Không giải nén file .nii.gz ra đĩa.

  python scripts/cache_dataset.py --config configs/base.yaml --out /home/$USER/brats_cache
  # sau đó trong config: data.source=npz, data.npz_dir=/home/$USER/brats_cache
"""
import argparse
import os
import time

import numpy as np

import _path  # noqa: F401
from brats.io import make_source
from brats.preprocess import brain_bbox
from brats.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/base.yaml")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--opts", nargs="*", default=[], help="ghi đè config, vd data.source=npz")
args = ap.parse_args()

cfg = load_config(args.config, args.opts)
src = make_source(cfg["data"])
ids = src.case_ids[: args.limit] if args.limit else src.case_ids
os.makedirs(args.out, exist_ok=True)
t0, total = time.time(), 0
for k, cid in enumerate(ids):
    dst = os.path.join(args.out, f"{cid}.npz")
    if os.path.exists(dst):
        continue
    c = src.load(cid)
    bb = brain_bbox(c["image"])
    sl = tuple(slice(lo, hi) for lo, hi in zip(bb[0], bb[1]))
    np.savez_compressed(dst + ".tmp.npz", image=c["image"][(slice(None),) + sl].astype(np.int16),
                        seg=c["seg"][sl].astype(np.uint8), bbox=bb, orig_shape=np.array(c["image"].shape[1:]),
                        affine=c["affine"])
    os.replace(dst + ".tmp.npz", dst)
    total += os.path.getsize(dst)
    print(f"[{k + 1}/{len(ids)}] {cid} {os.path.getsize(dst) / 1e6:.1f} MB  ({time.time() - t0:.0f}s)", flush=True)
print(f"Xong. Dung lượng mới ghi: {total / 1e9:.2f} GB")
