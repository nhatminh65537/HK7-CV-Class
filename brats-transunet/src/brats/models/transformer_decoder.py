"""
Transformer decoder kiểu Mask2Former — thành phần chính của 3D TransUNet (cấu hình Decoder-only).

Ý tưởng (paper mục III-B):
- Có N "organ/tumor query" (vector học được, N=20 > số lớp K=3).
- Mỗi query sinh ra 1 mask:   mask_q = sigmoid( MLP(query_q) · F_pixel )     (tích vô hướng với feature U-Net)
  và 1 nhãn lớp:              cls_q  = softmax( Linear(query_q) )  trên K lớp + 1 lớp "không phải gì cả".
- Query được tinh chỉnh qua T lớp. Mỗi lớp:
    masked cross-attention  : query "nhìn" vào feature đa tỉ lệ của U-Net, CHỈ trong vùng mask hiện tại
                              (coarse-to-fine: dự đoán vòng trước khoanh vùng cho vòng sau)
    self-attention          : các query trao đổi với nhau
    FFN
- Output mỗi lớp đều được giám sát (deep supervision) bằng Hungarian matching loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import SinePositionalEncoding, conv_nd


class CrossAttentionLayer(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, q, mem, attn_mask, pos, qpos):
        out = self.attn(q + qpos, mem + pos, mem, attn_mask=attn_mask, need_weights=False)[0]
        return self.norm(q + out)


class SelfAttentionLayer(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, q, qpos):
        out = self.attn(q + qpos, q + qpos, q, need_weights=False)[0]
        return self.norm(q + out)


class FFNLayer(nn.Module):
    def __init__(self, d, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, q):
        return self.norm(q + self.net(q))


class MLP(nn.Module):
    def __init__(self, din, dh, dout, n):
        super().__init__()
        dims = [din] + [dh] * (n - 1) + [dout]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x) if i == len(self.layers) - 1 else F.relu(l(x))
        return x


class MaskedTransformerDecoder(nn.Module):
    def __init__(self, spatial_dims, in_channels: list[int], num_classes=3, hidden=192, num_queries=20,
                 layers=3, heads=8, ffn_mult=8, masked_attn=True, dropout=0.0):
        super().__init__()
        self.dims, self.heads, self.masked_attn = spatial_dims, heads, masked_attn
        self.num_levels = len(in_channels)
        C = conv_nd(spatial_dims)
        self.input_proj = nn.ModuleList(nn.Sequential(C(c, hidden, 1), nn.GroupNorm(32, hidden)) for c in in_channels)
        self.level_embed = nn.Embedding(self.num_levels, hidden)
        self.pe = SinePositionalEncoding(hidden)
        self.query_feat = nn.Embedding(num_queries, hidden)
        self.query_embed = nn.Embedding(num_queries, hidden)
        self.cross = nn.ModuleList(CrossAttentionLayer(hidden, heads, dropout) for _ in range(layers))
        self.selfa = nn.ModuleList(SelfAttentionLayer(hidden, heads, dropout) for _ in range(layers))
        self.ffn = nn.ModuleList(FFNLayer(hidden, hidden * ffn_mult, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        self.class_embed = nn.Linear(hidden, num_classes + 1)
        self.mask_embed = MLP(hidden, hidden, hidden, 3)
        self._pe_cache: dict = {}

    def _heads(self, q, mask_features, target_size):
        q = self.norm(q)
        logits = self.class_embed(q)                                        # (B, N, K+1)
        masks = torch.einsum("bqc,bc...->bq...", self.mask_embed(q), mask_features)  # (B, N, *S)
        attn = None
        if self.masked_attn:
            mode = "bilinear" if self.dims == 2 else "trilinear"
            m = F.interpolate(masks.float(), size=target_size, mode=mode, align_corners=False)
            attn = (m.sigmoid().flatten(2) < 0.5)                           # True = KHÔNG được nhìn
            attn = attn.unsqueeze(1).repeat(1, self.heads, 1, 1).flatten(0, 1).detach()
        return logits, masks, attn

    def forward(self, feats: list[torch.Tensor], mask_features: torch.Tensor) -> dict:
        """feats: đặc trưng U-Net đa tỉ lệ (thấp -> cao độ phân giải); mask_features: (B, hidden, *S_full)."""
        B = mask_features.shape[0]
        src, pos, sizes = [], [], []
        cache = self._pe_cache
        for i, f in enumerate(feats):
            sizes.append(f.shape[2:])
            key = (tuple(f.shape[2:]), f.dtype, str(f.device))
            if key not in cache:                     # PE chỉ phụ thuộc kích thước -> tính 1 lần rồi dùng lại
                cache[key] = self.pe(f[:1]).flatten(2).transpose(1, 2)
            pos.append(cache[key].expand(f.shape[0], -1, -1))
            src.append((self.input_proj[i](f).flatten(2) + self.level_embed.weight[i][None, :, None]).transpose(1, 2))
        q = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)
        qpos = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        outs = []
        logits, masks, attn = self._heads(q, mask_features, sizes[0])
        outs.append((logits, masks))
        for i in range(len(self.cross)):
            lvl = i % self.num_levels
            if attn is not None:                                            # query chưa có mask -> cho nhìn cả ảnh
                attn = attn & ~attn.all(-1, keepdim=True)                    # thuần tensor, không đồng bộ GPU
            q = self.cross[i](q, src[lvl], attn, pos[lvl], qpos)
            q = self.selfa[i](q, qpos)
            q = self.ffn[i](q)
            logits, masks, attn = self._heads(q, mask_features, sizes[(i + 1) % self.num_levels])
            outs.append((logits, masks))
        return {
            "pred_logits": outs[-1][0], "pred_masks": outs[-1][1],
            "aux_outputs": [{"pred_logits": l, "pred_masks": m} for l, m in outs[:-1]],
        }
