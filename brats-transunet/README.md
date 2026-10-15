# brats-transunet

Tự cài đặt lại **TransUNet** (theo paper *3D TransUNet*, arXiv:2310.07781) để phân vùng u não trên **BraTS2021**,
bằng PyTorch + MONAI. Có bản **2D** (vừa GPU 4 GB) và bản **3D thu nhỏ**.

- Đọc trước: [`docs/01_GIAI_THICH_DE_TAI.md`](docs/01_GIAI_THICH_DE_TAI.md): bài toán, thuật ngữ, dữ liệu, input/output
- Lộ trình + setup: [`docs/02_LO_TRINH_VA_SETUP.md`](docs/02_LO_TRINH_VA_SETUP.md)
- Tra cứu lệnh và tham số: [`docs/04_TRA_CUU_LENH.md`](docs/04_TRA_CUU_LENH.md)
- Tham khảo repo gốc: [`docs/03_THAM_KHAO_REPO_GOC.md`](docs/03_THAM_KHAO_REPO_GOC.md)
- Ý tưởng cải thiện so với paper: [`docs/08_Y_TUONG_CAI_THIEN.md`](docs/08_Y_TUONG_CAI_THIEN.md); hướng dẫn chạy tiếp: [`docs/09_HUONG_DAN_CHAY_TIEP.md`](docs/09_HUONG_DAN_CHAY_TIEP.md); báo cáo đợt 2: [`docs/10_BAO_CAO_DOT_2.md`](docs/10_BAO_CAO_DOT_2.md)

## Chạy nhanh (PowerShell trên Windows, từ thư mục gốc project)

```powershell
uv venv -p 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install torch            # hoặc bản CUDA phù hợp driver, xem docs/02 mục 2.3
uv pip install -r requirements.txt

# chạy thử với 2 ca mẫu
python scripts/build_index.py --config configs/debug_samples.yaml
python scripts/make_splits.py --config configs/debug_samples.yaml
python train.py    --config configs/debug_samples.yaml
python evaluate.py --ckpt runs/debug_samples/best.pt --split test

# dữ liệu đầy đủ
python scripts/build_index.py --config configs/base.yaml
python scripts/check_data.py  --config configs/base.yaml     # kiểm tra 1251 ca, thống kê nhãn
python scripts/make_splits.py --config configs/base.yaml
python scripts/smoke_test.py  --config configs/transunet2d.yaml
python train.py --config configs/unet2d.yaml          # baseline
python train.py --config configs/transunet2d.yaml     # TransUNet decoder-only
python evaluate.py --ckpt runs/transunet2d/best.pt --split test --hd95

# các phương án cải tiến (docs/08)
python scripts/check_variants.py                                   # kiểm tra đúng/sai trên CPU
python scripts/benchmark.py --configs configs/transunet2d*.yaml    # tham số, VRAM, thời gian (GPU phải rảnh)
python scripts/compare_runs.py runs/unet2d/eval_test runs/transunet2d/eval_test   # so từng ca, khoảng tin cậy
```

## Mô hình có sẵn

| Config | Mô hình | Tương ứng paper |
|---|---|---|
| `unet2d.yaml`, `unet3d_small.yaml` | U-Net CNN | baseline (nnU-Net) |
| `transunet2d_encoder.yaml` | U-Net + ViT bottleneck | Encoder-only |
| `transunet2d.yaml`, `transunet3d_small.yaml` | U-Net + Transformer decoder (20 query, masked attention, Hungarian loss) | **Decoder-only** (paper dùng cho BraTS) |
| `*_lazy.yaml` | Decoder-only + tính mask "lười" (cùng trọng số, cùng kết quả) | cải tiến B, docs/08 |
| `*_fixed.yaml` | 3 query cố định cho WT/TC/ET, bỏ Hungarian | cải tiến C, docs/08 |
| `*_lite.yaml` | `*_fixed` + FFN×2 + 3 tầng dùng chung trọng số | cải tiến E, docs/08 |

Dữ liệu được đọc **thẳng từ file `.tar`**, không cần giải nén (xem docs/02 mục 3).
