"""Khối cơ bản dùng chung cho 2D và 3D (chọn bằng spatial_dims)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def conv_nd(dims):
    return nn.Conv2d if dims == 2 else nn.Conv3d


def convT_nd(dims):
    return nn.ConvTranspose2d if dims == 2 else nn.ConvTranspose3d


def norm_nd(dims):
    return nn.InstanceNorm2d if dims == 2 else nn.InstanceNorm3d


class ConvBlock(nn.Module):
    """(Conv 3x3 -> InstanceNorm -> LeakyReLU) × 2, giống nnU-Net."""

    def __init__(self, dims, cin, cout, stride=1):
        super().__init__()
        C, N = conv_nd(dims), norm_nd(dims)
        self.net = nn.Sequential(
            C(cin, cout, 3, stride, 1, bias=False), N(cout, affine=True), nn.LeakyReLU(0.01, inplace=True),
            C(cout, cout, 3, 1, 1, bias=False), N(cout, affine=True), nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SinePositionalEncoding(nn.Module):
    """Mã hóa vị trí dạng sin/cos cho lưới 2D hoặc 3D (như DETR / Mask2Former)."""

    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        self.dim, self.temperature = dim, temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, *spatial) -> (B, dim, *spatial)
        spatial = x.shape[2:]
        nd = len(spatial)
        per = self.dim // (2 * nd)
        coords = torch.meshgrid(*[torch.linspace(0, 2 * math.pi, s, device=x.device) for s in spatial], indexing="ij")
        freq = self.temperature ** (torch.arange(per, device=x.device, dtype=torch.float32) / per)
        feats = []
        for c in coords:
            v = c[None] / freq.view(-1, *([1] * nd))
            feats += [v.sin(), v.cos()]
        pe = torch.cat(feats, 0)
        if pe.shape[0] < self.dim:  # bù kênh nếu dim không chia hết
            pe = torch.cat([pe, torch.zeros(self.dim - pe.shape[0], *spatial, device=x.device)], 0)
        return pe.unsqueeze(0).expand(x.shape[0], -1, *spatial).to(x.dtype)
