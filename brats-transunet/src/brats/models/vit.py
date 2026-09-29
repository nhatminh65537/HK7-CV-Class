"""ViT đặt ở bottleneck của U-Net ("CNN-Transformer hybrid encoder", cấu hình Encoder-only)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTBottleneck(nn.Module):
    def __init__(self, channels, grid, depth=1, hidden=384, heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.grid = list(grid)
        self.proj_in = nn.Linear(channels, hidden)  # patch embedding (patch = 1 voxel của feature map)
        self.pos = nn.Parameter(torch.zeros(1, hidden, *self.grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, int(hidden * mlp_ratio), dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, channels)

    def forward(self, x):
        B, C, *S = x.shape
        pos = self.pos
        if list(S) != self.grid:  # ảnh lúc suy luận có thể to hơn patch lúc train
            pos = F.interpolate(pos, size=S, mode="bilinear" if len(S) == 2 else "trilinear", align_corners=False)
        t = self.proj_in(x.flatten(2).transpose(1, 2)) + pos.flatten(2).transpose(1, 2)
        t = self.proj_out(self.norm(self.encoder(t)))
        return x + t.transpose(1, 2).reshape(B, C, *S)  # residual: giữ đặc trưng CNN
