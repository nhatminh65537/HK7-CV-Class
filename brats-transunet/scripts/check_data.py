#!/usr/bin/env python3
"""
Kiểm tra toàn bộ dữ liệu trước khi train: đủ ca chưa, file có hỏng không, hình học và nhãn thế nào.
Chỉ đọc header của ảnh (rất nhanh) và đọc đầy đủ file nhãn (nhỏ).

  python scripts/check_data.py --config configs/base.yaml
  python scripts/check_data.py --config configs/base.yaml --limit 50     # kiểm tra nhanh 50 ca
"""
import argparse
import collections
import gzip
import struct
import time

import numpy as np

import _path  # noqa: F401
from brats.io import KINDS, make_source
from brats.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/base.yaml")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--opts", nargs="*", default=[])
args = ap.parse_args()

cfg = load_config(args.config, args.opts)
src = make_source(cfg["data"])
ids = src.case_ids
if args.limit:
    ids = ids[: args.limit]
print(f"Nguồn: {cfg['data']['source']} | số ca đủ 5 file: {len(src.case_ids)}"
      + ("" if len(src.case_ids) == 1251 else "  <-- BraTS2021 phải có 1251 ca!"))
if hasattr(src, "index"):
    missing = {c: sorted(v) for c, v in src.index.items() if len(v) != len(KINDS)}
    if missing:
        print("CA THIẾU FILE:", missing)

DT = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 512: np.uint16}
shapes, spac, dt_img, dt_seg, bad = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), []
rows = []
t0 = time.time()
for i, cid in enumerate(ids):
    try:
        if hasattr(src, "read_bytes"):                       # nguồn tar: đọc trực tiếp, nhanh nhất
            for k in KINDS:
                b = gzip.decompress(src.read_bytes(cid, k))
                if b[344:348] not in (b"n+1\0", b"ni1\0"):
                    bad.append((cid, k, "header NIfTI sai")); continue
                dim = struct.unpack_from("<8h", b, 40)[1:4]
                px = tuple(round(x, 3) for x in struct.unpack_from("<8f", b, 76)[1:4])
                dt = struct.unpack_from("<h", b, 70)[0]
                vo = int(struct.unpack_from("<f", b, 108)[0])
                shapes[dim] += 1; spac[px] += 1
                (dt_seg if k == "seg" else dt_img)[DT.get(dt, dt).__name__] += 1
                if k == "seg":
                    seg = np.frombuffer(b, DT[dt], count=int(np.prod(dim)), offset=vo)
        else:
            c = src.load(cid)
            shapes[c["image"].shape[1:]] += 1
            seg = c["seg"]
        u, n = np.unique(seg, return_counts=True)
        if set(u.tolist()) - {0, 1, 2, 4}:
            bad.append((cid, "seg", f"nhãn lạ {sorted(u.tolist())}"))
        cnt = dict(zip(u.tolist(), n.tolist()))
        rows.append((cid, seg.size, cnt.get(1, 0), cnt.get(2, 0), cnt.get(4, 0)))
    except Exception as e:
        bad.append((cid, "?", repr(e)))
    if (i + 1) % 100 == 0:
        print(f"  ...{i + 1}/{len(ids)} ca ({time.time() - t0:.0f}s)", flush=True)

print(f"\nĐã kiểm tra {len(rows)} ca trong {time.time() - t0:.0f}s")
if shapes:
    print("kích thước:", dict(shapes), "| spacing:", dict(spac) or "(không đọc)")
if dt_img:
    print("kiểu dữ liệu ảnh:", dict(dt_img), "| nhãn:", dict(dt_seg), " (khác nhau giữa các ca là bình thường)")
print("LỖI:", bad if bad else "không có")

a = np.array([r[1:] for r in rows], dtype=np.int64)
size, ncr, ed, et = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
print("\n%-8s %9s %9s %9s %9s %9s" % ("vùng", "TB %", "trung vị", "min %", "max %", "ca rỗng"))
for name, v in [("NCR (1)", ncr), ("ED (2)", ed), ("ET (4)", et), ("WT", ncr + ed + et), ("TC", ncr + et)]:
    p = 100 * v / size
    print("%-8s %9.3f %9.3f %9.4f %9.3f %9d" % (name, p.mean(), np.median(p), p.min(), p.max(), (v == 0).sum()))
empty_et = [r[0] for r in rows if r[4] == 0]
if empty_et:
    print(f"\n{len(empty_et)} ca KHÔNG có ET (ảnh hưởng cách tính Dice ET):", ", ".join(empty_et[:8]), "..." if len(empty_et) > 8 else "")
