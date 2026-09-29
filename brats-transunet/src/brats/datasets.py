"""
Dataset "dạng dòng" (IterableDataset) để train mà KHÔNG phải giải nén / cache toàn bộ dữ liệu.

Ý tưởng: mỗi DataLoader worker giữ một "pool" nhỏ vài ca (volume) trong RAM.
Liên tục: nạp 1 ca mới (đọc thẳng từ tar) -> đẩy ca cũ nhất ra -> cắt nhiều mẫu ngẫu nhiên từ pool.
RAM dùng ≈ num_workers × pool_size × ~30 MB, nên 16 GB RAM vẫn thoải mái.

- SliceStream2D : trả về lát cắt 2D (4, H, W) + nhãn region (3, H, W)
- PatchStream3D : trả về khối 3D  (4, D, H, W) + nhãn region (3, D, H, W)
"""
from __future__ import annotations

import collections
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from monai.data import set_track_meta
from monai import transforms as T

from .io import make_source
from .preprocess import preprocess_case

set_track_meta(False)  # dùng tensor thường, nhanh hơn MetaTensor


def pad_to(x: np.ndarray, size, value=0.0) -> np.ndarray:
    """Pad các trục không gian cuối cho đủ `size` (không crop)."""
    nd = len(size)
    pads = [(0, 0)] * (x.ndim - nd)
    for s, t in zip(x.shape[-nd:], size):
        d = max(t - s, 0)
        pads.append((d // 2, d - d // 2))
    return np.pad(x, pads, constant_values=value) if any(p != (0, 0) for p in pads) else x


def random_crop(arrs, size, center=None, rng=random):
    """Crop cùng 1 vị trí cho nhiều mảng (C, *spatial). center: toạ độ muốn nằm trong patch."""
    shape = arrs[0].shape[1:]
    starts = []
    for i, (s, t) in enumerate(zip(shape, size)):
        if center is None:
            starts.append(rng.randint(0, s - t))
        else:
            lo = max(0, min(center[i] - t // 2, s - t))
            jitter = rng.randint(-t // 4, t // 4)
            starts.append(int(np.clip(lo + jitter, 0, s - t)))
    sl = (slice(None),) + tuple(slice(st, st + t) for st, t in zip(starts, size))
    return [a[sl] for a in arrs]


def build_augment(dims: int, cfg: dict):
    keys = ["image", "label"]
    axes = list(range(dims))
    ts = [T.RandFlipd(keys, prob=0.5, spatial_axis=a) for a in axes]
    if dims == 2 and cfg.get("affine_prob", 0) > 0:
        ts.append(T.RandAffined(keys, prob=cfg["affine_prob"], rotate_range=np.pi / 12,
                                scale_range=0.1, mode=("bilinear", "nearest"), padding_mode="zeros"))
    ts += [
        T.RandScaleIntensityd("image", factors=0.1, prob=0.5),
        T.RandShiftIntensityd("image", offsets=0.1, prob=0.5),
        T.RandGaussianNoised("image", prob=0.15, std=0.1),
    ]
    return T.Compose(ts)


class _VolumeStream(IterableDataset):
    def __init__(self, data_cfg: dict, case_ids: list[str], patch_size, pool_size=4,
                 samples_per_load=16, fg_prob=0.5, augment: dict | None = None, seed=0):
        self.data_cfg, self.case_ids = data_cfg, list(case_ids)
        self.patch_size = tuple(patch_size)
        self.pool_size, self.samples_per_load, self.fg_prob = pool_size, samples_per_load, fg_prob
        self.augment_cfg, self.seed = augment, seed

    # --- phần chung --------------------------------------------------------------------------
    def _setup_worker(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        ids = self.case_ids[wid::nw] or self.case_ids
        rng = random.Random(self.seed * 1000 + wid + (info.seed if info else 0))
        src = make_source(self.data_cfg)  # mỗi worker tự mở file tar
        aug = build_augment(len(self.patch_size), self.augment_cfg) if self.augment_cfg else None
        if aug is not None:
            aug.set_random_state(seed=rng.randint(0, 2**31 - 1))
        return ids, rng, src, aug

    def _load(self, src, cid):
        c = preprocess_case(src.load(cid))
        return {"image": c["image"], "regions": c["regions"], "id": cid}

    def __iter__(self):
        ids, rng, src, aug = self._setup_worker()
        pool = collections.deque(maxlen=self.pool_size)
        order: list[str] = []
        while True:
            if not order:
                order = ids[:]
                rng.shuffle(order)
            pool.append(self._prepare(self._load(src, order.pop())))
            for _ in range(self.samples_per_load):
                vol = rng.choice(pool)
                img, lab = self._sample(vol, rng)
                d = {"image": torch.from_numpy(np.ascontiguousarray(img)),
                     "label": torch.from_numpy(np.ascontiguousarray(lab)).float()}
                if aug is not None:
                    d = aug(d)
                yield {"image": torch.as_tensor(d["image"]).float(), "label": torch.as_tensor(d["label"]).float()}

    def _prepare(self, vol):  # override
        return vol

    def _sample(self, vol, rng):  # override
        raise NotImplementedError


class SliceStream2D(_VolumeStream):
    """Lát cắt axial (trục z của NIfTI). patch_size = (H, W)."""

    def _prepare(self, vol):
        img, reg = vol["image"], vol["regions"]
        brain_z = np.nonzero((img != 0).any((0, 1, 2)))[0]
        tumor_z = np.nonzero(reg[0].any((0, 1)))[0]
        vol.update(brain_z=brain_z, tumor_z=tumor_z)
        return vol

    def _sample(self, vol, rng):
        use_fg = len(vol["tumor_z"]) > 0 and rng.random() < self.fg_prob
        z = int(rng.choice(vol["tumor_z"] if use_fg else vol["brain_z"]))
        img = pad_to(vol["image"][..., z], self.patch_size)
        lab = pad_to(vol["regions"][..., z], self.patch_size)
        center = None
        if use_fg:
            xs, ys = np.nonzero(lab[0])
            k = rng.randrange(len(xs))
            center = (int(xs[k]), int(ys[k]))
        return random_crop([img, lab], self.patch_size, center, rng)


class PatchStream3D(_VolumeStream):
    """Khối 3D ngẫu nhiên. patch_size = (X, Y, Z)."""

    def _prepare(self, vol):
        vol["image"] = pad_to(vol["image"], self.patch_size)
        vol["regions"] = pad_to(vol["regions"], self.patch_size)
        fg = np.argwhere(vol["regions"][0])
        if len(fg) > 5000:  # giữ một mẫu con để tiết kiệm RAM
            fg = fg[np.random.default_rng(0).choice(len(fg), 5000, replace=False)]
        vol["fg"] = fg
        return vol

    def _sample(self, vol, rng):
        center = None
        if len(vol["fg"]) > 0 and rng.random() < self.fg_prob:
            center = tuple(int(v) for v in vol["fg"][rng.randrange(len(vol["fg"]))])
        return random_crop([vol["image"], vol["regions"]], self.patch_size, center, rng)


def build_train_dataset(cfg: dict, case_ids: list[str]):
    t = cfg["train"]
    cls = SliceStream2D if cfg["model"]["spatial_dims"] == 2 else PatchStream3D
    return cls(cfg["data"], case_ids, t["patch_size"], t.get("pool_size", 4),
               t.get("samples_per_load", 16), t.get("fg_prob", 0.5), t.get("augment"), cfg.get("seed", 0))
