"""Vẽ ảnh MRI + nhãn/dự đoán để kiểm tra bằng mắt và đưa vào báo cáo."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

MOD_NAMES = ("T1", "T1ce", "T2", "FLAIR")
# 1 = phù nề (WT\TC), 2 = lõi hoại tử (TC\ET), 3 = tăng cường (ET)
CMAP = ListedColormap(["none", "#2ca02c", "#ffd700", "#d62728"])


def regions_to_display(r: np.ndarray) -> np.ndarray:
    lab = np.zeros(r.shape[1:], np.uint8)
    lab[r[0] > 0] = 1
    lab[r[1] > 0] = 2
    lab[r[2] > 0] = 3
    return lab


def _show(ax, img, lab=None, title=""):
    img = np.rot90(img)
    ax.set_facecolor("black")
    ax.imshow(np.ma.masked_where(img == 0, img), cmap="gray",
              vmin=np.percentile(img[img != 0], 0.5) if (img != 0).any() else None,
              vmax=np.percentile(img[img != 0], 99.5) if (img != 0).any() else None)
    if lab is not None:
        lab = np.rot90(lab)
        ax.imshow(np.ma.masked_where(lab == 0, lab), cmap=CMAP, vmin=0, vmax=3, alpha=0.55, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_overlay(image, gt_regions, pred_regions, path, title=""):
    """image (4,X,Y,Z); regions (3,X,Y,Z). Chọn lát cắt axial có nhiều u nhất."""
    area = gt_regions[0].sum((0, 1))
    z = int(area.argmax()) if area.max() > 0 else image.shape[-1] // 2
    n = 4 + 1 + (pred_regions is not None)
    fig, ax = plt.subplots(1, n, figsize=(3 * n, 3.3))
    for i in range(4):
        _show(ax[i], image[i, :, :, z], title=MOD_NAMES[i])
    _show(ax[4], image[3, :, :, z], regions_to_display(gt_regions[..., z]), "Ground truth")
    if pred_regions is not None:
        _show(ax[5], image[3, :, :, z], regions_to_display(pred_regions[..., z]), "Dự đoán")
    fig.suptitle(f"{title}  (z={z})  xanh=phù nề  vàng=hoại tử  đỏ=tăng cường", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
