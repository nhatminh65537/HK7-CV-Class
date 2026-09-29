"""Đánh giá theo region WT/TC/ET trên CẢ VOLUME của từng ca (chuẩn BraTS)."""
from __future__ import annotations

import numpy as np

from .preprocess import REGION_NAMES


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Quy ước BraTS: GT rỗng & dự đoán rỗng -> 1.0; GT rỗng nhưng dự đoán có -> 0.0."""
    p, g = pred.astype(bool), gt.astype(bool)
    ps, gs = p.sum(), g.sum()
    if gs == 0:
        return 1.0 if ps == 0 else 0.0
    return float(2 * (p & g).sum() / (ps + gs))


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Khoảng cách Hausdorff 95% (mm, voxel 1mm). Trả nan nếu 1 trong 2 rỗng."""
    from monai.metrics import compute_hausdorff_distance
    import torch
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    p = torch.from_numpy(pred[None, None].astype(np.uint8))
    g = torch.from_numpy(gt[None, None].astype(np.uint8))
    return float(compute_hausdorff_distance(p, g, percentile=95).item())


def case_metrics(pred_regions: np.ndarray, gt_regions: np.ndarray, with_hd95=False) -> dict:
    res = {}
    for i, n in enumerate(REGION_NAMES):
        res[f"dice_{n}"] = dice(pred_regions[i], gt_regions[i])
        if with_hd95:
            res[f"hd95_{n}"] = hd95(pred_regions[i], gt_regions[i])
    res["dice_mean"] = float(np.mean([res[f"dice_{n}"] for n in REGION_NAMES]))
    return res
