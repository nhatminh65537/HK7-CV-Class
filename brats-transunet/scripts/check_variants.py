#!/usr/bin/env python3
"""
Kiểm tra đúng/sai cho các phương án cải tiến (docs/08 mục 3), chạy trên CPU với tensor nhỏ (vài giây, ít RAM):

1. lazy_mask cho KẾT QUẢ Y HỆT cách cũ khi cùng trọng số: xác suất vùng lúc suy luận, loss và gradient lúc train
   (Hungarian, cùng các điểm lấy mẫu). Checkpoint cũ nạp được vào model lazy (cùng tên tham số).
2. fixed_queries và shared_layers: forward/backward chạy được, shape đúng, loss hữu hạn, đếm tham số.

  python scripts/check_variants.py            # mặc định CPU
  python scripts/check_variants.py --cuda     # chạy trên GPU (chỉ khi GPU rảnh)
"""
import argparse
import copy

import torch

import _path  # noqa: F401
from brats.losses import build_loss
from brats.models import build_model, region_probs
from brats.utils import count_params, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--cuda", action="store_true")
args = ap.parse_args()
dev = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK ' if cond else 'LỖI'}] {name} {detail}")


def small(cfg_path, dims, **dec):
    """Config thật, thu nhỏ kích thước để chạy nhanh trên CPU."""
    cfg = load_config(cfg_path)
    cfg["model"]["features"] = [8, 16, 32, 64, 64]
    cfg["model"]["decoder"].update(hidden=32, heads=4, **dec)
    cfg["train"]["patch_size"] = [48, 48] if dims == 2 else [32, 32, 32]
    return cfg


def fake(cfg, bs=2):
    ps = cfg["train"]["patch_size"]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(bs, 4, *ps, generator=g)
    y = torch.zeros(bs, 3, *ps)
    sl = tuple(slice(p // 4, p // 2) for p in ps)
    y[(0, 0) + sl] = 1
    y[(0, 1) + sl] = 1
    y[(0, 2) + tuple(slice(p // 4, p // 4 + 3) for p in ps)] = 1   # mẫu 0 có đủ 3 vùng, mẫu 1 không có u
    return x.to(dev), y.to(dev)


def grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


for dims, path in ((2, "configs/transunet2d.yaml"), (3, "configs/transunet3d_small.yaml")):
    print(f"\n=== {dims}D: lazy_mask so với cách cũ (cùng trọng số) ===")
    cfg_a = small(path, dims)
    cfg_b = copy.deepcopy(cfg_a)
    cfg_b["model"]["decoder"]["lazy_mask"] = True
    torch.manual_seed(0)
    a = build_model(cfg_a).to(dev)
    b = build_model(cfg_b).to(dev)
    check("checkpoint cũ nạp được vào model lazy", b.load_state_dict(a.state_dict()) is not None)
    x, y = fake(cfg_a)

    a.eval(); b.eval()
    with torch.no_grad():
        pa, pb = region_probs(a(x)), region_probs(b(x))
    d = (pa - pb).abs().max().item()
    check("xác suất vùng khi suy luận", d < 1e-4, f"max|chênh| = {d:.2e}")

    a.train(); b.train()
    crit = build_loss(cfg_a).to(dev)
    torch.manual_seed(1); la = crit(a(x), y); la.backward()
    torch.manual_seed(1); lb = crit(b(x), y); lb.backward()
    check("loss khi train", abs(la.item() - lb.item()) < 1e-4, f"{la.item():.6f} so với {lb.item():.6f}")
    ga, gb = grads(a), grads(b)
    # so với độ lớn gradient lớn nhất của cả model: vài tham số có gradient thật ~0 (bias conv ngay trước
    # GroupNorm) nên sai khác tương đối theo từng tham số không có nghĩa
    scale = max(g.abs().max().item() for g in ga.values())
    dg = max((ga[k] - gb[k]).abs().max().item() for k in ga) / scale
    check("gradient khi train (sai khác lớn nhất / gradient lớn nhất)", dg < 1e-4 and ga.keys() == gb.keys(), f"{dg:.2e}")
    out_b = b(x)
    check("khi train, lazy không dựng mask đủ độ phân giải", "pred_masks" not in out_b and "mask_pix" in out_b)

    for label, extra in (("fixed_queries", dict(num_queries=3, fixed_queries=True, lazy_mask=True)),
                         ("fixed + FFNx2 + shared", dict(num_queries=3, fixed_queries=True, lazy_mask=True,
                                                         ffn_mult=2, shared_layers=True))):
        print(f"--- {dims}D: {label} ---")
        cfg_c = small(path, dims, **extra)
        c = build_model(cfg_c).to(dev)
        crit_c = build_loss(cfg_c).to(dev)
        c.train()
        out = c(x)
        loss = crit_c(out, y)
        loss.backward()
        check("loss hữu hạn, có gradient", torch.isfinite(loss).item() and len(grads(c)) > 0, f"loss={loss.item():.4f}")
        check("pred_masks có 3 kênh = 3 vùng", tuple(out["pred_masks"].shape[:2]) == (2, 3))
        check("không có class head", "pred_logits" not in out)
        c.eval()
        with torch.no_grad():
            p = region_probs(c(x))
        check("region_probs đúng shape", tuple(p.shape) == (2, 3, *cfg_c["train"]["patch_size"]))
        print(f"      tham số (bản thu nhỏ): baseline {count_params(a):.3f}M, {label} {count_params(c):.3f}M")

print("\nKẾT QUẢ:", "tất cả OK" if ok else "CÓ LỖI")
raise SystemExit(0 if ok else 1)
