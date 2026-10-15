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

Ba tùy chọn cải tiến (docs/08 mục 3), đều TẮT mặc định để baseline chạy y như cũ:
- lazy_mask (truyền `pix` + `proj` thay cho `mask_features`): không tạo tensor (B, hidden, *S).
  Vì conv 1x1 `proj` là tuyến tính: MLP(q)·(W f + b) = (MLP(q) W)·f + MLP(q)·b, nên hạ query xuống
  C kênh rồi nhân thẳng với feature f (B, C, *S). Attention mask tính trên f đã thu nhỏ (nội suy cũng
  tuyến tính). Kết quả trùng cách cũ trong sai số số học.
- fixed_queries: N = K = 3 query, query k luôn là vùng k (WT, TC, ET). Không có class head, không Hungarian.
- shared_layers: 3 tầng tinh chỉnh dùng chung một bộ trọng số (cross-attn, self-attn, FFN).
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


def lazy_mask_logits(embed: torch.Tensor, bias: torch.Tensor, pix: torch.Tensor) -> torch.Tensor:
    """Mask logit từ query đã hạ kênh. embed (B, N, C), bias (B, N), pix (B, C, *S) -> (B, N, *S)."""
    m = torch.einsum("bqc,bc...->bq...", embed, pix)
    return m + bias.view(*bias.shape, *[1] * (pix.dim() - 2))


class MaskedTransformerDecoder(nn.Module):
    def __init__(self, spatial_dims, in_channels: list[int], num_classes=3, hidden=192, num_queries=20,
                 layers=3, heads=8, ffn_mult=8, masked_attn=True, dropout=0.0,
                 fixed_queries=False, shared_layers=False):
        super().__init__()
        if fixed_queries and num_queries != num_classes:
            raise ValueError(f"fixed_queries cần num_queries = {num_classes} (mỗi query một vùng), đang là {num_queries}")
        self.dims, self.heads, self.masked_attn = spatial_dims, heads, masked_attn
        self.fixed, self.num_layers, self.shared = fixed_queries, layers, shared_layers
        self.num_levels = len(in_channels)
        C = conv_nd(spatial_dims)
        self.input_proj = nn.ModuleList(nn.Sequential(C(c, hidden, 1), nn.GroupNorm(32, hidden)) for c in in_channels)
        self.level_embed = nn.Embedding(self.num_levels, hidden)
        self.pe = SinePositionalEncoding(hidden)
        self.query_feat = nn.Embedding(num_queries, hidden)
        self.query_embed = nn.Embedding(num_queries, hidden)
        n_blocks = 1 if shared_layers else layers
        self.cross = nn.ModuleList(CrossAttentionLayer(hidden, heads, dropout) for _ in range(n_blocks))
        self.selfa = nn.ModuleList(SelfAttentionLayer(hidden, heads, dropout) for _ in range(n_blocks))
        self.ffn = nn.ModuleList(FFNLayer(hidden, hidden * ffn_mult, dropout) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(hidden)
        self.class_embed = None if fixed_queries else nn.Linear(hidden, num_classes + 1)
        self.mask_embed = MLP(hidden, hidden, hidden, 3)
        self._pe_cache: dict = {}

    def _attn_mask(self, m_low: torch.Tensor) -> torch.Tensor:
        """Mask logit ở độ phân giải của tầng attention -> attn_mask bool (True = KHÔNG được nhìn)."""
        attn = (m_low.sigmoid().flatten(2) < 0.5)
        return attn.unsqueeze(1).repeat(1, self.heads, 1, 1).flatten(0, 1).detach()

    def forward(self, feats: list[torch.Tensor], mask_features: torch.Tensor | None = None,
                pix: torch.Tensor | None = None, proj: nn.Module | None = None) -> dict:
        """feats: đặc trưng U-Net đa tỉ lệ (thấp -> cao độ phân giải).

        Cách cũ: mask_features (B, hidden, *S_full).
        Cách lười: pix (B, C, *S_full) = feature U-Net gốc, proj = conv 1x1 C -> hidden (không gọi nó).
        """
        lazy = mask_features is None
        ref = pix if lazy else mask_features
        B = ref.shape[0]
        mode = "bilinear" if self.dims == 2 else "trilinear"
        src, pos, sizes = [], [], []
        cache = self._pe_cache
        for i, f in enumerate(feats):
            sizes.append(tuple(f.shape[2:]))
            key = (tuple(f.shape[2:]), f.dtype, str(f.device))
            if key not in cache:                     # PE chỉ phụ thuộc kích thước -> tính 1 lần rồi dùng lại
                cache[key] = self.pe(f[:1]).flatten(2).transpose(1, 2)
            pos.append(cache[key].expand(f.shape[0], -1, -1))
            src.append((self.input_proj[i](f).flatten(2) + self.level_embed.weight[i][None, :, None]).transpose(1, 2))
        q = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)
        qpos = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        if lazy:
            W, b = proj.weight.flatten(1), proj.bias          # (hidden, C), (hidden,)
            pix_low: dict = {}                                 # feature đã thu nhỏ theo từng kích thước

        def heads(q, size, full):
            q = self.norm(q)
            out = {} if self.fixed else {"pred_logits": self.class_embed(q)}   # (B, N, K+1)
            e = self.mask_embed(q)                                             # (B, N, hidden)
            attn = None
            if not lazy:
                masks = torch.einsum("bqc,bc...->bq...", e, mask_features)      # (B, N, *S)
                out["pred_masks"] = masks
                if self.masked_attn:
                    attn = self._attn_mask(F.interpolate(masks.float(), size=size, mode=mode, align_corners=False))
                return out, attn
            e2, bias = e @ W, e @ b                                            # (B, N, C), (B, N)
            out["mask_embed"], out["mask_bias"] = e2, bias
            if full:
                out["pred_masks"] = lazy_mask_logits(e2, bias, pix)
            if self.masked_attn:
                if size not in pix_low:
                    pix_low[size] = F.interpolate(pix.float(), size=size, mode=mode, align_corners=False)
                attn = self._attn_mask(lazy_mask_logits(e2.float(), bias.float(), pix_low[size]))
            return out, attn

        def need_full(i):  # chỉ dùng ở cách lười: output nào cần mask đủ độ phân giải
            if self.training:
                return self.fixed            # Hungarian: loss chỉ cần logit tại điểm lấy mẫu
            return i == self.num_layers      # suy luận: chỉ cần output cuối

        outs = []
        out, attn = heads(q, sizes[0], need_full(0))
        outs.append(out)
        for i in range(self.num_layers):
            lvl = i % self.num_levels
            blk = 0 if self.shared else i
            if attn is not None:                                            # query chưa có mask -> cho nhìn cả ảnh
                attn = attn & ~attn.all(-1, keepdim=True)                    # thuần tensor, không đồng bộ GPU
            q = self.cross[blk](q, src[lvl], attn, pos[lvl], qpos)
            q = self.selfa[blk](q, qpos)
            q = self.ffn[blk](q)
            out, attn = heads(q, sizes[(i + 1) % self.num_levels], need_full(i + 1))
            outs.append(out)
        res = dict(outs[-1])
        res["aux_outputs"] = outs[:-1]
        if lazy:
            res["mask_pix"] = pix
        return res
