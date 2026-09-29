"""
Đọc dữ liệu BraTS2021 từ 3 kiểu nguồn, KHÔNG cần giải nén file .tar lớn:

- TarSource : đọc thẳng từ file .tar (random access nhờ index offset). Không tốn thêm ổ đĩa.
- DirSource : thư mục đã giải nén (BraTS2021_XXXXX/BraTS2021_XXXXX_t1.nii.gz ...).
- NpzSource : cache .npz đã crop + nén (tạo bằng scripts/cache_dataset.py). Đọc nhanh nhất.

Mọi nguồn đều trả về cùng một dict "case" thô:
    {"id": str,
     "image": float32 (4, X, Y, Z)   # thứ tự kênh: t1, t1ce, t2, flair
     "seg":   uint8   (X, Y, Z)      # nhãn gốc BraTS: 0, 1 (NCR), 2 (ED), 4 (ET)
     "affine": (4,4)}
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np

MODALITIES = ("t1", "t1ce", "t2", "flair")
KINDS = MODALITIES + ("seg",)
_PAT = re.compile(r"(BraTS2021_\d{5})_(t1|t1ce|t2|flair|seg)\.nii\.gz$")


def _nifti_from_bytes(raw: bytes) -> nib.Nifti1Image:
    if raw[:2] == b"\x1f\x8b":  # gzip magic
        raw = gzip.decompress(raw)
    return nib.Nifti1Image.from_bytes(raw)


def _assemble(case_id: str, blobs: dict[str, bytes | str]) -> dict:
    """blobs: kind -> bytes (từ tar) hoặc đường dẫn file (từ thư mục)."""
    imgs, affine = [], None
    for m in MODALITIES:
        b = blobs[m]
        im = nib.load(b) if isinstance(b, (str, os.PathLike)) else _nifti_from_bytes(b)
        affine = im.affine
        imgs.append(np.asanyarray(im.dataobj).astype(np.float32))
    out = {"id": case_id, "image": np.stack(imgs, 0), "affine": affine}
    if "seg" in blobs:
        b = blobs["seg"]
        im = nib.load(b) if isinstance(b, (str, os.PathLike)) else _nifti_from_bytes(b)
        out["seg"] = np.asanyarray(im.dataobj).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------- tar
class TarSource:
    """Random access vào 1 hoặc nhiều file .tar (không nén ngoài, bên trong là .nii.gz).

    Lần đầu sẽ quét header của tar (nhanh, chỉ đọc vài KB mỗi file thành viên) và lưu index JSON.
    Sau đó mỗi lần đọc 1 ca chỉ cần seek + đọc ~10 MB.
    """

    def __init__(self, tar_paths: list[str], index_path: str | None = None, rebuild: bool = False):
        self.tar_paths = [str(Path(p).expanduser()) for p in tar_paths]
        self.index_path = index_path or (os.path.splitext(self.tar_paths[0])[0] + ".index.json")
        if rebuild or not os.path.exists(self.index_path):
            self.index = build_tar_index(self.tar_paths)
            os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(self.index, f)
        else:
            with open(self.index_path, encoding="utf-8") as f:
                self.index = json.load(f)
        self._fh: dict[str, io.BufferedReader] = {}  # mỗi process/worker tự mở file riêng
        self._pid = os.getpid()

    @property
    def case_ids(self) -> list[str]:
        return sorted(c for c, v in self.index.items() if all(k in v for k in KINDS))

    def __getstate__(self):  # Windows dùng "spawn": không pickle được file handle đang mở
        d = self.__dict__.copy()
        d["_fh"] = {}
        return d

    def _handle(self, path: str):
        if os.getpid() != self._pid:  # đã fork sang worker khác -> mở lại file
            self._fh, self._pid = {}, os.getpid()
        if path not in self._fh:
            self._fh[path] = open(path, "rb")
        return self._fh[path]

    def read_bytes(self, case_id: str, kind: str, nbytes: int | None = None) -> bytes:
        """nbytes: chỉ đọc mấy byte đầu (đủ để giải nén header NIfTI), dùng khi kiểm tra dữ liệu."""
        tar_path, offset, size = self.index[case_id][kind]
        fh = self._handle(tar_path)
        fh.seek(offset)
        return fh.read(size if nbytes is None else min(size, nbytes))

    def load(self, case_id: str, with_seg: bool = True) -> dict:
        kinds = KINDS if with_seg else MODALITIES
        return _assemble(case_id, {k: self.read_bytes(case_id, k) for k in kinds})


def build_tar_index(tar_paths: list[str]) -> dict:
    """case_id -> kind -> [tar_path, offset_data, size]"""
    index: dict[str, dict] = {}
    for p in tar_paths:
        with tarfile.open(p, "r:") as tf:  # "r:" = tar không nén -> cho phép seek
            for ti in tf:
                name = os.path.basename(ti.name)
                if not ti.isfile() or name.startswith("._"):
                    continue
                m = _PAT.search(name)
                if m:
                    index.setdefault(m.group(1), {})[m.group(2)] = [p, ti.offset_data, ti.size]
    return index


# ----------------------------------------------------------------------------- dir
class DirSource:
    def __init__(self, root: str):
        self.root = Path(root).expanduser()
        self.files: dict[str, dict] = {}
        for f in self.root.rglob("*.nii.gz"):
            m = _PAT.search(f.name)
            if m and not f.name.startswith("._"):
                self.files.setdefault(m.group(1), {})[m.group(2)] = str(f)

    @property
    def case_ids(self) -> list[str]:
        return sorted(c for c, v in self.files.items() if all(k in v for k in KINDS))

    def load(self, case_id: str, with_seg: bool = True) -> dict:
        kinds = KINDS if with_seg else MODALITIES
        return _assemble(case_id, {k: self.files[case_id][k] for k in kinds})


# ----------------------------------------------------------------------------- npz
class NpzSource:
    """Cache .npz (tạo bằng scripts/cache_dataset.py): ảnh int16 đã crop về vùng não + seg uint8.
    Chuẩn hóa vẫn làm lúc load (rẻ), nên cache nhỏ (~nén tốt) và không phụ thuộc cách chuẩn hóa."""

    def __init__(self, root: str):
        self.root = Path(root).expanduser()

    @property
    def case_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("BraTS2021_*.npz"))

    def load(self, case_id: str, with_seg: bool = True) -> dict:
        d = np.load(self.root / f"{case_id}.npz")
        out = {
            "id": case_id,
            "image": d["image"].astype(np.float32),  # đã crop
            "bbox": d["bbox"],
            "orig_shape": tuple(int(x) for x in d["orig_shape"]),
            "affine": d["affine"],
        }
        if with_seg and "seg" in d:
            out["seg"] = d["seg"]
        return out


def make_source(cfg: dict):
    kind = cfg["source"]
    if kind == "tar":
        return TarSource(cfg["tar_paths"], cfg.get("index_path"))
    if kind == "dir":
        return DirSource(cfg["dir"])
    if kind == "npz":
        return NpzSource(cfg["npz_dir"])
    raise ValueError(f"data.source không hợp lệ: {kind}")
