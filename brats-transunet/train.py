#!/usr/bin/env python3
"""
Train U-Net / TransUNet trên BraTS2021.

  python train.py --config configs/transunet2d.yaml
  python train.py --config configs/transunet2d.yaml --resume            # chạy tiếp từ last.pt
  python train.py --config configs/unet2d.yaml --opts train.epochs=2 train.iters_per_epoch=20   # chạy thử

Ctrl+C bất cứ lúc nào: checkpoint được lưu lại, lần sau --resume chạy tiếp.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from brats.datasets import build_train_dataset          # noqa: E402
from brats.inference import binarize, predict_volume   # noqa: E402
from brats.io import make_source                       # noqa: E402
from brats.losses import build_loss                    # noqa: E402
from brats.metrics import case_metrics                 # noqa: E402
from brats.models import build_model                   # noqa: E402
from brats.preprocess import preprocess_case           # noqa: E402
from brats.utils import count_params, load_config, load_splits, seed_everything  # noqa: E402


def lr_at(step, total, warmup, base, min_lr=1e-6):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


def validate(model, cfg, source, case_ids, device):
    rows = []
    for cid in case_ids:
        c = preprocess_case(source.load(cid))
        pred = binarize(predict_volume(model, c["image"], cfg, device), cfg["eval"].get("threshold", 0.5))
        rows.append(case_metrics(pred, c["regions"]))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--opts", nargs="*", default=[], help="ghi đè config, vd train.epochs=5")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    seed_everything(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = cfg["train"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    splits = load_splits(cfg["splits"])
    train_ids, val_ids = splits["train"], splits["val"][: cfg["eval"].get("val_cases", 10)]
    print(f"device={device} | train={len(train_ids)} ca | val(dùng khi train)={len(val_ids)} ca")

    loader = DataLoader(build_train_dataset(cfg, train_ids), batch_size=t["batch_size"],
                        num_workers=t.get("num_workers", 2), pin_memory=device.type == "cuda",
                        persistent_workers=t.get("num_workers", 2) > 0)
    model = build_model(cfg).to(device)
    crit = build_loss(cfg).to(device)
    print(f"model={cfg['model']['name']} {cfg['model']['spatial_dims']}D | params={count_params(model):.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 3e-5))
    use_amp = device.type == "cuda" and t.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ipe, epochs, accum = t["iters_per_epoch"], t["epochs"], t.get("grad_accum", 1)
    total_steps, warmup = ipe * epochs, ipe * t.get("warmup_epochs", 1)

    start_epoch, best = 0, -1.0
    last_ckpt = os.path.join(out_dir, "last.pt")
    if args.resume and os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", -1.0)
        print(f"resume từ epoch {start_epoch}, best dice={best:.4f}")

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "cfg": cfg}, path)

    log_path = os.path.join(out_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(["epoch", "lr", "train_loss", "sec", "dice_WT", "dice_TC", "dice_ET", "dice_mean"])

    source = make_source(cfg["data"])
    it = iter(loader)
    epoch = start_epoch
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            t0, losses = time.time(), []
            for i in range(ipe):
                lr = lr_at(epoch * ipe + i, total_steps, warmup, t["lr"])
                for g in opt.param_groups:
                    g["lr"] = lr
                for _ in range(accum):
                    b = next(it)
                    x, y = b["image"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        out = model(x)
                    loss = crit(out, y) / accum
                    scaler.scale(loss).backward()
                    losses.append(loss.item() * accum)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if (i + 1) % t.get("print_every", 50) == 0:
                    print(f"  ep {epoch} it {i + 1}/{ipe} loss {np.mean(losses[-50:]):.4f} lr {lr:.2e}", flush=True)
            sec = time.time() - t0
            row = [epoch, f"{lr:.3e}", f"{np.mean(losses):.4f}", f"{sec:.0f}"] + [""] * 4
            msg = f"epoch {epoch}: loss {np.mean(losses):.4f} ({sec:.0f}s)"
            if val_ids and ((epoch + 1) % cfg["eval"].get("every", 5) == 0 or epoch == epochs - 1):
                m = validate(model, cfg, source, val_ids, device)
                row[4:] = [f"{m['dice_WT']:.4f}", f"{m['dice_TC']:.4f}", f"{m['dice_ET']:.4f}", f"{m['dice_mean']:.4f}"]
                msg += f" | val Dice WT {m['dice_WT']:.3f} TC {m['dice_TC']:.3f} ET {m['dice_ET']:.3f} mean {m['dice_mean']:.3f}"
                if m["dice_mean"] > best:
                    best = m["dice_mean"]
                    save(os.path.join(out_dir, "best.pt"), epoch)
                    msg += "  *best*"
            print(msg, flush=True)
            writer.writerow(row); logf.flush()
            save(last_ckpt, epoch)
    except KeyboardInterrupt:
        print("\nDừng giữa chừng -> lưu checkpoint (epoch hiện tại chưa xong sẽ chạy lại khi --resume)")
        save(last_ckpt, epoch - 1)
    logf.close()
    print(f"Xong. best val dice_mean = {best:.4f} | output: {out_dir}")


if __name__ == "__main__":
    main()
