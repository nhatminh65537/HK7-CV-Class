from __future__ import annotations

import copy
import json
import os
import random

import numpy as np
import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Đọc YAML; hỗ trợ `base:` (kế thừa file khác) và ghi đè dạng key.sub=value từ dòng lệnh."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        cfg = deep_update(base, cfg)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        d = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(v)
    return cfg


def deep_update(a: dict, b: dict) -> dict:
    a = copy.deepcopy(a)
    for k, v in b.items():
        a[k] = deep_update(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def seed_everything(seed: int):
    import torch  # import trong hàm để các script chỉ xử lý dữ liệu không cần cài torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_splits(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
