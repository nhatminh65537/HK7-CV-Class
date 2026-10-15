#!/usr/bin/env python3
"""
Đo chi phí của nhiều cấu hình trên cùng máy, với dữ liệu giả (không cần dataset):
số tham số, thời gian và VRAM đỉnh của 1 bước train, thời gian và VRAM đỉnh của 1 lần suy luận.

  python scripts/benchmark.py --configs configs/transunet2d.yaml configs/transunet2d_lazy.yaml configs/transunet2d_fixed.yaml
  python scripts/benchmark.py --configs configs/transunet3d_small*.yaml --csv figures/bench_3d.csv
  python scripts/benchmark.py --configs configs/transunet3d_small.yaml configs/transunet3d_small_fixed.yaml --opts "train.patch_size=[128,128,128]"

- Bước train giống smoke_test.py: AMP, batch và patch lấy từ config (cộng dồn gradient không tính).
- Suy luận: 1 lần forward không gradient, AMP, trên `eval.batch_size` lát (2D) hoặc 1 khối patch (3D).
  Đây là thời gian của MỘT cửa sổ, không phải cả ca; thời gian cả ca đo bằng evaluate.py (cột sec).
- Bỏ qua vài lần đầu (khởi động cuDNN) rồi lấy trung vị. Chỉ chạy khi GPU rảnh: số đo bị nhiễu nếu GPU đang train.
- Nếu một cấu hình hết VRAM, dòng đó ghi OOM và script chạy tiếp cấu hình sau.
"""
import argparse
import csv
import gc
import glob
import os
import statistics
import time

import torch

import _path  # noqa: F401
from brats.losses import build_loss
from brats.models import build_model
from brats.utils import count_params, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", required=True, help="danh sách config (chấp nhận *)")
ap.add_argument("--opts", nargs="*", default=[], help="ghi đè cho MỌI config, vd train.patch_size=[128,128,128]")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--csv", default="", help="lưu bảng kết quả ra file CSV")
args = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = dev.type == "cuda"
paths = [p for pat in args.configs for p in sorted(glob.glob(pat))]


def sync():
    if cuda:
        torch.cuda.synchronize()


def fake_batch(bs, ps):
    x = torch.randn(bs, 4, *ps, device=dev)
    y = torch.zeros(bs, 3, *ps, device=dev)
    sl = tuple(slice(p // 3, p // 2) for p in ps)
    y[(slice(None), 0) + sl] = 1          # giả lập 1 khối u: WT ⊃ TC ⊃ ET
    y[(slice(None), 1) + sl] = 1
    y[(slice(None), 2) + tuple(slice(p // 3, p // 3 + max(1, p // 12)) for p in ps)] = 1
    return x, y


def timed(fn, n, warmup):
    times = []
    for i in range(warmup + n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1024**3 if cuda else float("nan")


def bench(path):
    cfg = load_config(path, args.opts)
    name = os.path.splitext(os.path.basename(path))[0]
    dims, ps = cfg["model"]["spatial_dims"], cfg["train"]["patch_size"]
    row = {"config": name, "dims": dims, "patch": "x".join(map(str, ps))}
    model = build_model(cfg).to(dev)
    crit = build_loss(cfg).to(dev)
    row["params_M"] = round(count_params(model), 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    # --- 1 bước train ---
    bs = cfg["train"]["batch_size"]
    x, y = fake_batch(bs, ps)
    model.train()

    def step():
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            out = model(x)
        loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    try:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        row["train_s_per_step"] = round(timed(step, args.iters, args.warmup), 4)
        row["train_peak_GB"] = round(peak_gb(), 3)
    except torch.cuda.OutOfMemoryError:
        row["train_s_per_step"], row["train_peak_GB"] = "OOM", "OOM"
    opt.zero_grad(set_to_none=True)
    del x, y
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()

    # --- 1 lần suy luận ---
    ibs = cfg.get("eval", {}).get("batch_size", 16) if dims == 2 else cfg.get("eval", {}).get("sw_batch_size", 1)
    xi = torch.randn(ibs, 4, *ps, device=dev)
    model.eval()

    @torch.no_grad()
    def infer():
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            model(xi)

    try:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        row["infer_batch"] = ibs
        row["infer_s"] = round(timed(infer, args.iters, args.warmup), 4)
        row["infer_peak_GB"] = round(peak_gb(), 3)
    except torch.cuda.OutOfMemoryError:
        row["infer_s"], row["infer_peak_GB"] = "OOM", "OOM"
    del model, crit, opt, xi
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return row


print(f"device={dev}" + (f" | {torch.cuda.get_device_name(0)}" if cuda else "") + f" | opts={args.opts}")
rows = []
for p in paths:
    r = bench(p)
    rows.append(r)
    print("  " + " | ".join(f"{k}={v}" for k, v in r.items()), flush=True)

keys = ["config", "dims", "patch", "params_M", "train_s_per_step", "train_peak_GB", "infer_batch", "infer_s", "infer_peak_GB"]
print("\n" + " | ".join(keys))
for r in rows:
    print(" | ".join(str(r.get(k, "")) for k in keys))
if args.csv:
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print("Đã lưu", args.csv)
