"""
TransUNet (bản tự cài đặt, 2D/3D) = U-Net + Transformer decoder (+ tuỳ chọn ViT ở bottleneck).

  ảnh (B,4,*S) ─> U-Net encoder ─(ViT bottleneck?)─> U-Net decoder ─┬─> feat 1/8, 1/4, 1/2 ─┐
                                                                  └─> feat 1/1 ─> 1x1 ─> F_pixel
                                                  20 query ─> [masked cross-attn, self-attn, FFN] × 3
                                                  ─> pred_logits (B,20,K+1), pred_masks (B,20,*S)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import conv_nd
from .transformer_decoder import MaskedTransformerDecoder
from .unet import UNet


class TransUNet(nn.Module):
    def __init__(self, spatial_dims=2, in_channels=4, num_classes=3, features=(32, 64, 128, 256, 320),
                 decoder: dict | None = None, vit_bottleneck: dict | None = None, patch_size=None):
        super().__init__()
        decoder = dict(decoder or {})
        self.ms_idxs = decoder.pop("ms_idxs", [-4, -3, -2])
        self.unet = UNet(spatial_dims, in_channels, num_classes, features, vit_bottleneck, patch_size)
        ch = self.unet.decoder_channels
        hidden = decoder.get("hidden", 192)
        self.mask_proj = conv_nd(spatial_dims)(ch[-1], hidden, 1)
        self.decoder = MaskedTransformerDecoder(spatial_dims, [ch[i] for i in self.ms_idxs],
                                                num_classes=num_classes, **decoder)

    def forward(self, x):
        feats = self.unet.forward_features(x)
        return self.decoder([feats[i] for i in self.ms_idxs], self.mask_proj(feats[-1]))
