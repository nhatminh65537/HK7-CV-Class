"""
U-Net 2D/3D (encoder–decoder có skip connection).

  input (B,4,*S) ─ enc0 ──────────────────────────────── dec0 ─> head ─> (B,3,*S)
                     └ enc1 ───────────────────── dec1 ┘
                          └ enc2 ─────────── dec2 ┘
                               └ ... bottleneck ┘

forward_features() trả về cả đặc trưng của từng tầng decoder (thấp -> cao độ phân giải),
để TransUNet dùng lại.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock, conv_nd, convT_nd


class UNet(nn.Module):
    def __init__(self, spatial_dims=2, in_channels=4, out_channels=3, features=(32, 64, 128, 256, 320),
                 vit_bottleneck: dict | None = None, patch_size=None):
        super().__init__()
        d = spatial_dims
        self.dims, self.features = d, list(features)
        self.encoders = nn.ModuleList()
        cin = in_channels
        for i, f in enumerate(features):
            self.encoders.append(ConvBlock(d, cin, f, stride=1 if i == 0 else 2))
            cin = f
        # --- (tuỳ chọn) Transformer encoder ở bottleneck = cấu hình "Encoder-only" của paper
        self.vit = None
        if vit_bottleneck:
            from .vit import ViTBottleneck
            down = 2 ** (len(features) - 1)
            grid = [p // down for p in patch_size]
            self.vit = ViTBottleneck(features[-1], grid=grid, **vit_bottleneck)
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(convT_nd(d)(features[i], features[i - 1], 2, 2))
            self.decoders.append(ConvBlock(d, features[i - 1] * 2, features[i - 1]))
        self.head = conv_nd(d)(features[0], out_channels, 1)

    @property
    def decoder_channels(self):  # kênh của các feature trả về từ forward_features (thấp -> cao)
        return [self.features[-1]] + self.features[-2::-1]

    def forward_features(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        if self.vit is not None:
            x = self.vit(x)
        feats = [x]
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), skip], 1))
            feats.append(x)
        return feats  # [bottleneck (1/16), 1/8, 1/4, 1/2, 1/1]

    def forward(self, x):
        return {"logits": self.head(self.forward_features(x)[-1])}
