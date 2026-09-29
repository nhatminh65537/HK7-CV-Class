from __future__ import annotations

import torch
import torch.nn.functional as F

from .transunet import TransUNet
from .unet import UNet


def build_model(cfg: dict):
    m = dict(cfg["model"])
    name = m.pop("name")
    dims = m.pop("spatial_dims")
    patch = cfg["train"]["patch_size"]
    if name == "unet":          # baseline, hoặc Encoder-only nếu có vit_bottleneck
        return UNet(dims, 4, 3, m.get("features"), m.get("vit_bottleneck"), patch)
    if name == "transunet":     # Decoder-only (hoặc Encoder+Decoder nếu có vit_bottleneck)
        return TransUNet(dims, 4, 3, m.get("features"), m.get("decoder"), m.get("vit_bottleneck"), patch)
    raise ValueError(name)


def region_probs(out: dict) -> torch.Tensor:
    """Output model -> xác suất 3 region (B, 3, *S) trong [0, 1]."""
    if "logits" in out:
        return out["logits"].float().sigmoid()
    cls = F.softmax(out["pred_logits"].float(), -1)[..., :-1]      # bỏ lớp "no object"
    masks = out["pred_masks"].float().sigmoid()
    return torch.einsum("bqc,bq...->bc...", cls, masks).clamp(0, 1)
