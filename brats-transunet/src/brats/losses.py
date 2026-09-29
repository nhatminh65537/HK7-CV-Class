"""
Loss functions.

1) RegionDiceBCELoss — cho U-Net thường: mỗi region (WT/TC/ET) là 1 kênh sigmoid.
       loss = BCE + (1 - soft Dice)

2) SetCriterion — Hungarian matching loss cho Transformer decoder (paper Eq. 8):
   - Ground truth của 1 ảnh = danh sách các region KHÔNG rỗng, ví dụ [WT, TC, ET] (tối đa 3 "đối tượng").
   - Model đưa ra N=20 cặp (lớp, mask). Thuật toán Hungarian ghép 1-1 mỗi đối tượng GT với query có
     chi phí nhỏ nhất: cost = 2·(-p_lớp) + 5·BCE(mask) + 5·Dice(mask)   (trọng số giống code gốc).
   - Query được ghép: học đúng lớp + mask. Query không được ghép: học lớp "no object".
   - loss = 0.2·CE_lớp + 0.5·BCE_mask + 0.5·Dice_mask, tính cho output cuối và mọi output trung gian.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class RegionDiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, out: dict, target: torch.Tensor):
        logits = out["logits"].float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = logits.sigmoid()
        axes = (0,) + tuple(range(2, p.ndim))  # batch dice
        inter = (p * target).sum(axes)
        dice = (2 * inter + self.smooth) / (p.sum(axes) + target.sum(axes) + self.smooth)
        return bce + (1 - dice.mean())


def _pair_bce(logits, targets):   # (N, P), (K, P) -> (N, K)
    pos = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
    neg = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
    return (pos @ targets.T + neg @ (1 - targets).T) / logits.shape[1]


def _pair_dice(logits, targets):  # (N, P), (K, P) -> (N, K)
    p = logits.sigmoid()
    num = 2 * p @ targets.T
    den = p.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (num + 1) / (den + 1)


class SetCriterion(nn.Module):
    def __init__(self, num_classes=3, cost_weight=(2.0, 5.0, 5.0), no_object_weight=0.1, num_points=12544,
                 reuse_match=False):
        super().__init__()
        self.K = num_classes
        self.reuse_match = reuse_match        # True: ghép cặp 1 lần rồi dùng lại cho các lớp phụ (nhanh hơn)
        self.num_points = num_points          # lấy mẫu ngẫu nhiên N điểm thay vì dùng toàn bộ voxel
        self.w_cls, self.w_bce, self.w_dice = cost_weight
        empty = torch.ones(num_classes + 1)
        if no_object_weight is not None:
            empty[-1] = no_object_weight
        self.register_buffer("empty_weight", empty)

    @staticmethod
    def make_targets(target: torch.Tensor):
        """target (B, 3, *S) -> list[{labels (k,), masks (k, *S)}] chỉ gồm region không rỗng."""
        present = target.flatten(2).sum(-1) > 0
        return [{"labels": torch.nonzero(present[b]).squeeze(1), "masks": target[b][present[b]]}
                for b in range(target.shape[0])]

    def _points(self, P: int, device):
        """Chỉ số các điểm dùng để tính chi phí ghép cặp và loss mask (None = dùng toàn bộ)."""
        if not self.num_points or self.num_points >= P:
            return None
        return torch.randint(0, P, (self.num_points,), device=device)

    @torch.no_grad()
    def match(self, logits, masks, targets, pts=None):
        """Hungarian matching. Gom mọi ma trận chi phí rồi chuyển về CPU MỘT lần: chuyển từng ca
        một sẽ buộc GPU đồng bộ rất nhiều lần và làm chậm hẳn vòng train."""
        B, N = logits.shape[:2]
        ks = [len(t["labels"]) for t in targets]
        e = torch.empty(0, dtype=torch.long)
        if max(ks, default=0) == 0:
            return [(e, e) for _ in range(B)]
        big = torch.full((B, N, max(ks)), 1e6, device=logits.device)
        prob = logits.softmax(-1)
        for b, t in enumerate(targets):
            if ks[b] == 0:
                continue
            m, tm = masks[b].flatten(1), t["masks"].flatten(1).float()
            if pts is not None:
                m, tm = m[:, pts], tm[:, pts]
            big[b, :, : ks[b]] = (self.w_cls * -prob[b][:, t["labels"]]
                                  + self.w_bce * _pair_bce(m, tm) + self.w_dice * _pair_dice(m, tm))
        cost = big.cpu().numpy()                       # một lần đồng bộ duy nhất cho cả batch
        idx = []
        for b in range(B):
            if ks[b] == 0:
                idx.append((e, e))
                continue
            r, c = linear_sum_assignment(cost[b][:, : ks[b]])
            idx.append((torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)))
        return idx

    def _single(self, logits, masks, targets, pts=None, idx=None):
        logits, masks = logits.float(), masks.float()
        if idx is None:
            idx = self.match(logits, masks, targets, pts)
        B, N = logits.shape[:2]
        tgt_cls = torch.full((B, N), self.K, dtype=torch.long, device=logits.device)
        src_m, tgt_m = [], []
        for b, (r, c) in enumerate(idx):
            if len(r):
                tgt_cls[b, r] = targets[b]["labels"][c]
                sm, tm = masks[b, r].flatten(1), targets[b]["masks"][c].flatten(1).float()
                if pts is not None:
                    sm, tm = sm[:, pts], tm[:, pts]
                src_m.append(sm)
                tgt_m.append(tm)
        loss_cls = F.cross_entropy(logits.transpose(1, 2), tgt_cls, self.empty_weight)
        if not src_m:  # cả batch không có u
            return self.w_cls / 10 * loss_cls, idx
        s, t = torch.cat(src_m), torch.cat(tgt_m)
        n = len(s)
        loss_bce = F.binary_cross_entropy_with_logits(s, t, reduction="none").mean(1).sum() / n
        p = s.sigmoid()
        loss_dice = (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).sum() / n
        return (self.w_cls * loss_cls + self.w_bce * loss_bce + self.w_dice * loss_dice) / 10, idx

    def forward(self, out: dict, target: torch.Tensor):
        targets = self.make_targets(target)
        pts = self._points(int(np.prod(target.shape[2:])), target.device)
        final, idx = self._single(out["pred_logits"], out["pred_masks"], targets, pts)
        reuse = idx if self.reuse_match else None
        aux = [self._single(a["pred_logits"], a["pred_masks"], targets, pts, reuse)[0]
               for a in out.get("aux_outputs", [])]
        return (final + sum(aux) / len(aux)) / 2 if aux else final   # giống max_loss_cal='v1'


def build_loss(cfg: dict):
    if cfg["model"]["name"] == "transunet":
        l = cfg.get("loss", {})
        return SetCriterion(3, tuple(l.get("cost_weight", (2.0, 5.0, 5.0))), l.get("no_object_weight", 0.1),
                            l.get("num_points", 12544), l.get("reuse_match", False))
    return RegionDiceBCELoss()
