#!/usr/bin/env python3
"""
Kiểm tra model + đo VRAM trước khi train thật. Chạy với dữ liệu ngẫu nhiên, không cần dataset.

  python scripts/smoke_test.py --config configs/transunet2d.yaml
  python scripts/smoke_test.py --config configs/transunet3d_small.yaml --opts train.batch_size=1
"""
import argparse
import time

import torch

import _path  # noqa: F401
from brats.losses import build_loss
from brats.models import build_model, region_probs
from brats.utils import count_params, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--opts", nargs="*", default=[])
args = ap.parse_args()

cfg = load_config(args.config, args.opts)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(dev)
crit = build_loss(cfg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
bs, ps = cfg["train"]["batch_size"], cfg["train"]["patch_size"]
print(f"{cfg['model']['name']} {cfg['model']['spatial_dims']}D | params {count_params(model):.2f}M | batch {bs} | patch {ps} | {dev}")
if dev.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.reset_peak_memory_stats()

x = torch.randn(bs, 4, *ps, device=dev)
y = torch.zeros(bs, 3, *ps, device=dev)
sl = tuple(slice(p // 3, p // 2) for p in ps)
y[(slice(None), 0) + sl] = 1          # giả lập 1 khối u
y[(slice(None), 1) + sl] = 1
scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
for i in range(args.iters):
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        out = model(x)
    loss = crit(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"  iter {i}: loss {loss.item():.4f}  {time.time() - t0:.2f}s")
print("output region_probs:", tuple(region_probs(out).shape))
if dev.type == "cuda":
    print(f"PEAK VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB (reserved {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB)")
