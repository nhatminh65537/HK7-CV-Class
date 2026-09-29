#!/usr/bin/env python3
"""
Đánh giá checkpoint trên tập val/test: Dice (+ HD95) theo WT/TC/ET, lưu CSV, NIfTI dự đoán và ảnh minh họa.

  python evaluate.py --ckpt runs/transunet2d/best.pt --split test
  python evaluate.py --ckpt runs/transunet2d/best.pt --split val --limit 20 --save-nifti --hd95
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from brats.inference import binarize, predict_volume          # noqa: E402
from brats.io import make_source                              # noqa: E402
from brats.metrics import case_metrics                        # noqa: E402
from brats.models import build_model                          # noqa: E402
from brats.preprocess import preprocess_case, regions_to_seg, uncrop  # noqa: E402
from brats.utils import load_splits                           # noqa: E402
from brats.viz import save_overlay                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hd95", action="store_true")
    ap.add_argument("--save-nifti", action="store_true")
    ap.add_argument("--figures", type=int, default=3, help="số ca lưu ảnh minh họa")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    source = make_source(cfg["data"])
    ids = load_splits(cfg["splits"])[args.split]
    ids = ids[: args.limit] if args.limit else ids

    out_dir = os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for k, cid in enumerate(ids):
        c = preprocess_case(source.load(cid))
        probs = predict_volume(model, c["image"], cfg, device)
        pred = binarize(probs, cfg["eval"].get("threshold", 0.5))
        m = {"id": cid, **case_metrics(pred, c["regions"], args.hd95)}
        rows.append(m)
        print(f"[{k + 1}/{len(ids)}] {cid}  " + "  ".join(f"{a}={b:.3f}" for a, b in m.items() if a != "id"), flush=True)
        if args.save_nifti:
            seg = uncrop(regions_to_seg(pred), c["bbox"], c["orig_shape"])
            nib.save(nib.Nifti1Image(seg, c["affine"]), os.path.join(out_dir, f"{cid}_pred.nii.gz"))
        if k < args.figures:
            save_overlay(c["image"], c["regions"], pred, os.path.join(out_dir, f"{cid}.png"), title=cid)

    keys = [k for k in rows[0] if k != "id"]
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["id"] + keys)
        w.writeheader(); w.writerows(rows)
    print("\n=== KẾT QUẢ ({} ca) ===".format(len(rows)))
    print(f"{'chỉ số':12s} {'trung bình':>11s} {'trung vị':>10s}")
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        print(f"{k:12s} {np.nanmean(v):11.4f} {np.nanmedian(v):10.4f}")
    worst = sorted(rows, key=lambda r: r["dice_mean"])[:3]
    fail = [r for r in rows if r["dice_mean"] < 0.5]
    print(f"\nSố ca Dice trung bình < 0.5: {len(fail)}/{len(rows)}")
    print("3 ca kém nhất:", ", ".join(f"{r['id']} ({r['dice_mean']:.3f})" for r in worst))
    print("Kết quả lưu ở", out_dir)


if __name__ == "__main__":
    main()
