"""Suy luận trên cả volume: 2D = từng lát cắt; 3D = sliding window (cửa sổ trượt, chồng lấn 50%)."""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from .models import region_probs


def _amp(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def predict_volume(model, image: np.ndarray, cfg: dict, device) -> np.ndarray:
    """image (4, X, Y, Z) đã chuẩn hóa -> xác suất (3, X, Y, Z) float32."""
    model.eval()
    dims = cfg["model"]["spatial_dims"]
    x = torch.from_numpy(image).to(device)
    if dims == 2:
        div = 2 ** (len(cfg["model"]["features"]) - 1)
        X, Y, Z = image.shape[1:]
        px, py = (-X) % div, (-Y) % div
        probs = torch.zeros(3, X, Y, Z, device=device)
        brain = (x != 0).any(0).any(0).any(0)                       # lát cắt có não
        zs = torch.nonzero(brain).squeeze(1).tolist()
        bs = cfg.get("eval", {}).get("batch_size", 16)
        for i in range(0, len(zs), bs):
            zb = zs[i:i + bs]
            inp = x[..., zb].permute(3, 0, 1, 2)                       # (b, 4, X, Y)
            inp = F.pad(inp, (0, py, 0, px))
            with _amp(device):
                p = region_probs(model(inp))[..., :X, :Y]              # (b, 3, X, Y)
            probs[..., zb] = p.permute(1, 2, 3, 0).float()
        return probs.cpu().numpy()
    from monai.inferers import sliding_window_inference
    e = cfg.get("eval", {})
    with _amp(device):
        p = sliding_window_inference(
            x[None], roi_size=cfg["train"]["patch_size"], sw_batch_size=e.get("sw_batch_size", 1),
            predictor=lambda t: region_probs(model(t)), overlap=e.get("overlap", 0.5), mode="gaussian")
    return p[0].float().cpu().numpy()


def binarize(probs: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Xác suất 3 vùng -> mask nhị phân, đảm bảo ET ⊂ TC ⊂ WT.

    Vì 3 vùng lồng nhau theo định nghĩa, ta lấy max lũy tiến TRƯỚC khi cắt ngưỡng
    (P(WT) ≥ P(TC) ≥ P(ET)). Cách này tốt hơn việc cắt ngưỡng rồi lấy giao: nếu mô hình
    chắc chắn về ET nhưng lưỡng lự về TC thì vùng TC vẫn được giữ, thay vì cắt cụt cả hai.
    """
    p = probs.copy()
    p[1] = np.maximum(p[1], p[2])
    p[0] = np.maximum(p[0], p[1])
    return (p > thr).astype(np.uint8)
