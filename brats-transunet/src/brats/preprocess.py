"""
Tiền xử lý 1 ca (làm "on-the-fly", không ghi ra đĩa):

1. Crop về bounding box của não (bỏ nền đen 0)  -> nhỏ hơn ~2 lần, đỡ RAM/VRAM.
2. Chuẩn hóa z-score từng kênh trên vùng não:   x = (x - mean) / std, nền = 0.
3. Đổi nhãn gốc {0,1,2,4} -> 3 kênh region nhị phân (đúng cách BraTS chấm điểm):
       WT (whole tumor) = 1 ∪ 2 ∪ 4
       TC (tumor core)  = 1 ∪ 4
       ET (enhancing)   = 4
"""
from __future__ import annotations

import numpy as np

REGION_NAMES = ("WT", "TC", "ET")


def seg_to_regions(seg: np.ndarray) -> np.ndarray:
    return np.stack([seg > 0, (seg == 1) | (seg == 4), seg == 4], 0).astype(np.uint8)


def regions_to_seg(regions: np.ndarray) -> np.ndarray:
    """Ngược lại: (3, ...) nhị phân -> nhãn BraTS {0,1,2,4} để lưu NIfTI."""
    wt, tc, et = (regions[i].astype(bool) for i in range(3))
    tc, et = tc & wt, et & tc & wt  # đảm bảo lồng nhau
    seg = np.zeros(wt.shape, np.uint8)
    seg[wt] = 2
    seg[tc] = 1
    seg[et] = 4
    return seg


def brain_bbox(image: np.ndarray, margin: int = 0) -> np.ndarray:
    mask = (image != 0).any(0)
    idx = np.nonzero(mask)
    lo = [max(int(i.min()) - margin, 0) for i in idx]
    hi = [min(int(i.max()) + 1 + margin, s) for i, s in zip(idx, mask.shape)]
    return np.array([lo, hi])  # (2, 3)


def normalize(image: np.ndarray) -> np.ndarray:
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        x = image[c]
        m = x != 0
        if m.any():
            v = x[m]
            out[c][m] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def preprocess_case(case: dict) -> dict:
    """case thô (từ io.*Source) -> {"image": (4,x,y,z) float32 chuẩn hóa, "regions": (3,x,y,z) uint8, ...}"""
    img, seg = case["image"], case.get("seg")
    if "bbox" in case:  # NpzSource: đã crop sẵn
        bb, orig_shape = np.asarray(case["bbox"]), case["orig_shape"]
    else:
        bb, orig_shape = brain_bbox(img), img.shape[1:]
        sl = tuple(slice(lo, hi) for lo, hi in zip(bb[0], bb[1]))
        img = img[(slice(None),) + sl]
        seg = seg[sl] if seg is not None else None
    out = {"id": case["id"], "image": normalize(img), "bbox": bb,
           "orig_shape": tuple(orig_shape), "affine": case.get("affine")}
    if seg is not None:
        out["regions"] = seg_to_regions(seg)
    return out


def uncrop(arr: np.ndarray, bbox: np.ndarray, orig_shape) -> np.ndarray:
    """Đặt mảng đã crop (..., x, y, z) về lại kích thước gốc."""
    full = np.zeros(arr.shape[:-3] + tuple(orig_shape), dtype=arr.dtype)
    sl = tuple(slice(lo, hi) for lo, hi in zip(bbox[0], bbox[1]))
    full[(Ellipsis,) + sl] = arr
    return full
