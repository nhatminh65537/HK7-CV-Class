#!/usr/bin/env python3
"""
Bước 1: tạo index cho file .tar (vị trí từng file .nii.gz bên trong) và kiểm tra dữ liệu đủ chưa.
Không giải nén gì cả.

  python scripts/build_index.py --config configs/base.yaml
  python scripts/build_index.py --config configs/base.yaml --check 5    # đọc thử 5 ca
"""
import argparse
import time

import _path  # noqa: F401
from brats.io import KINDS, TarSource, make_source
from brats.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/base.yaml")
ap.add_argument("--rebuild", action="store_true")
ap.add_argument("--check", type=int, default=2, help="đọc thử N ca để chắc chắn file không hỏng")
ap.add_argument("--opts", nargs="*", default=[], help="ghi đè config, vd data.source=npz")
args = ap.parse_args()

cfg = load_config(args.config, args.opts)
t0 = time.time()
d = cfg["data"]
src = TarSource(d["tar_paths"], d.get("index_path"), rebuild=args.rebuild) if d["source"] == "tar" else make_source(d)
ids = src.case_ids
print(f"Tìm thấy {len(ids)} ca đủ 5 file ({time.time() - t0:.1f}s)")
if d["source"] == "tar":
    broken = [c for c, v in src.index.items() if not all(k in v for k in KINDS)]
    if broken:
        print("Ca thiếu file:", broken[:10], "..." if len(broken) > 10 else "")
    print("Index lưu tại:", src.index_path)
if len(ids) != 1251:
    print("CẢNH BÁO: BraTS2021 Task 1 có 1251 ca. Có thể file tải chưa xong.")
for cid in ids[: args.check]:
    t1 = time.time()
    c = src.load(cid)
    print(f"  đọc {cid}: image {c['image'].shape} seg labels {sorted(set(c['seg'].ravel().tolist()))} ({time.time() - t1:.2f}s)")
