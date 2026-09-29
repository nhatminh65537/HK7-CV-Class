# 02 — Lộ trình và hướng dẫn setup

**Máy:** Windows 11 (chạy native, không cần WSL), RTX 3050 Ti **4 GB**, RAM 16 GB, ổ D: còn khoảng 47 GB.
**Thời gian:** 8–10 tuần. **Nguyên tắc:** chạy được là chính; làm tới đâu báo cáo tới đó; chấp nhận train lâu.

---

## 1. Lộ trình

| Tuần | Mục tiêu | Việc cụ thể | Đem đi báo cáo |
|---|---|---|---|
| **0** (mai) | Hiểu đề tài, có khung code | Đọc `docs/01`, chạy thử code với 2 ca mẫu | Slide: bài toán, dữ liệu, mô hình, kế hoạch; demo `inspect_case` + chạy thử train |
| **1** | Môi trường + dữ liệu | Setup Python/GPU trên Windows (mục 2); `build_index`; `check_data`; `make_splits`; `inspect_case --stats 100`; `smoke_test` đo VRAM | Thống kê dữ liệu (tỉ lệ vùng u, số ca thiếu NCR/ET), VRAM đo được, thời gian 1 iteration |
| **2** | Baseline U-Net 2D | Train `unet2d` trên 200 ca (tập nhỏ), rồi toàn bộ. Chạy `evaluate.py` | Đường cong loss/Dice, Dice WT/TC/ET của U-Net 2D, hình dự đoán |
| **3** | TransUNet 2D (decoder-only) | Train `transunet2d` cùng số epoch, cùng dữ liệu | Bảng so sánh U-Net và TransUNet 2D |
| **4** | Thí nghiệm nhỏ (ablation) | `masked_attn=false`; `num_queries` 3 / 20; encoder-only (`transunet2d_encoder`) | Bảng ablation giống paper (Table VII) |
| **5–6** | 3D thu nhỏ | `smoke_test` 3D, chỉnh patch/batch cho vừa 4 GB; train `unet3d_small` và `transunet3d_small` (có thể chạy qua đêm nhiều ngày) | So sánh 2D và 3D |
| **7** | Đánh giá cuối | `evaluate.py --split test --hd95` cho mọi model; phân tích ca tốt/xấu; lỗi thường gặp (ET nhỏ, ca không có ET) | Bảng kết quả cuối, hình minh họa |
| **8** | Viết báo cáo | Báo cáo, slide cuối, dọn code, README | Bản nháp báo cáo |
| **9–10** | Dự phòng | Train thêm epoch, fix lỗi, thêm fold / postprocess | Bản cuối |

**Thứ tự ưu tiên nếu thiếu thời gian:** (1) U-Net 2D + TransUNet 2D có kết quả trên test → (2) ablation masked attention → (3) 3D thu nhỏ → (4) HD95, phân tích sâu.

**Gợi ý chia việc** (điều chỉnh theo số người trong nhóm):

- **A:** dữ liệu và đánh giá (index, split, thống kê, `evaluate.py`, hình).
- **B:** model và train (cấu hình, chạy train, theo dõi log).
- **C:** đọc paper, viết báo cáo, slide, ablation.

---

## 2. Setup chi tiết (Windows)

Chạy thẳng trên Windows, **không cần WSL**. (Repo gốc thì bắt buộc Linux vì nó dùng NCCL, nhưng code của nhóm không dùng.)
Mọi lệnh dưới đây gõ trong **PowerShell**, đứng tại thư mục gốc project.

### 2.1 Kiểm tra GPU

```powershell
nvidia-smi          # thấy "RTX 3050 Ti" và dòng "CUDA Version: xx.x"
```

Không thấy thì cập nhật driver NVIDIA cho Windows (GeForce Experience hoặc nvidia.com).

### 2.2 Cài Python và uv

```powershell
winget install -e --id Python.Python.3.11          # nếu chưa có Python 3.11
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"    # cài uv
```

Mở lại PowerShell sau khi cài để `python` và `uv` vào PATH.

### 2.3 Môi trường ảo + thư viện

```powershell
cd "D:\Learning Projects\CV\brats-transunet"
uv venv -p 3.11 .venv
.\.venv\Scripts\Activate.ps1
# nếu PowerShell chặn script: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# PyTorch bản CUDA — xem "CUDA Version" ở nvidia-smi rồi chọn MỘT dòng:
uv pip install torch                                                         # driver mới (CUDA >= 13)
# uv pip install torch --index-url https://download.pytorch.org/whl/cu128    # CUDA 12.8
# uv pip install torch --index-url https://download.pytorch.org/whl/cu126    # driver cũ hơn
# (lệnh đúng cho máy bạn: pytorch.org -> Get Started)

uv pip install -r requirements.txt
python -c "import torch, monai; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Dòng cuối phải in ra `True` và tên GPU. In ra `False` nghĩa là bản torch vừa cài là bản CPU: gỡ (`uv pip uninstall torch`) rồi cài lại đúng index CUDA.

### 2.4 Đường dẫn dữ liệu

`configs/base.yaml` đã trỏ sẵn tới:

```yaml
tar_paths:
  - "D:/Learning Projects/CV/Data/BraTS2021_Training_Data.tar"
```

Trong YAML nên viết dấu `/` (`D:/...`). Nếu muốn dùng `\` thì phải viết gấp đôi: `"D:\\Learning Projects\\..."`.
**Luôn chạy lệnh từ thư mục gốc project**, vì `data/`, `runs/`, `figures/` là đường dẫn tương đối.

### 2.5 Chạy thử với 2 ca mẫu (kiểm tra pipeline)

```powershell
python scripts/build_index.py --config configs/debug_samples.yaml
python scripts/make_splits.py --config configs/debug_samples.yaml
python scripts/inspect_case.py --config configs/debug_samples.yaml --case BraTS2021_00495
python scripts/smoke_test.py  --config configs/transunet2d.yaml      # đo VRAM thật
python train.py --config configs/debug_samples.yaml
python evaluate.py --ckpt runs/debug_samples/best.pt --split test --save-nifti
```

Dice lúc này rất thấp là bình thường (mới train vài chục bước với 2 ca).

### 2.6 Dữ liệu đầy đủ

```powershell
python scripts/build_index.py --config configs/base.yaml --check 3   # kỳ vọng 1251 ca
python scripts/check_data.py  --config configs/base.yaml             # kiểm tra + thống kê nhãn
python scripts/make_splits.py --config configs/base.yaml             # 876 / 188 / 187 ca

python train.py --config configs/unet2d.yaml                         # baseline
python train.py --config configs/transunet2d.yaml
python train.py --config configs/transunet2d.yaml --resume           # chạy tiếp sau khi dừng

python scripts/plot_log.py runs/unet2d/log.csv runs/transunet2d/log.csv
python evaluate.py --ckpt runs/transunet2d/best.pt --split test --hd95 --save-nifti
```

Thử nhanh trên tập nhỏ trước:

```powershell
python scripts/make_splits.py --config configs/base.yaml --limit 200 --out data/splits_200.json
python train.py --config configs/unet2d.yaml --opts splits=data/splits_200.json output_dir=runs/unet2d_200 train.epochs=20
```

### 2.7 Vài điều riêng của Windows

- **Train lâu:** mở một cửa sổ PowerShell riêng cho job train, đừng đóng nó. Ctrl+C an toàn: checkpoint được lưu, chạy lại với `--resume`. Nhớ tắt chế độ ngủ (Settings → Power → Screen and sleep → Never).
- **Đa tiến trình:** Windows tạo tiến trình con kiểu *spawn*, mỗi worker nạp lại module. Code đã có `if __name__ == "__main__"` nên chạy bằng `python train.py` là đúng. Đừng chạy vòng lặp train trong Jupyter Notebook với `num_workers > 0`.
- **VRAM:** Windows và trình duyệt chiếm sẵn 0.5–1 GB VRAM, nên 4 GB thực dùng chỉ còn khoảng 3–3.5 GB. Đóng Chrome khi train, và lấy số từ `smoke_test.py` làm chuẩn.
- **Theo dõi:** `nvidia-smi -l 5` (dùng VRAM và GPU util), file `runs/<tên>/log.csv` (loss và Dice từng epoch).
- **Nếu vẫn muốn WSL:** vẫn chạy được, chỉ cần đổi `tar_paths` sang `/mnt/d/...`. Nhưng đọc dữ liệu từ ổ Windows qua `/mnt/d` chậm hơn đáng kể, nên Windows native là lựa chọn tốt hơn cho project này.

---

## 3. Train mà không giải nén

**Có, code đã làm sẵn.** File `.tar` là dạng "không nén ngoài": các file bên trong nằm liền nhau. `scripts/build_index.py` quét 1 lần để ghi lại **vị trí byte** của từng file `.nii.gz`, lưu ở `data/brats2021_tar_index.json`. Khi train, code **nhảy thẳng tới vị trí đó**, đọc khoảng 10 MB rồi giải nén gzip **trong RAM**. Không có file nào được ghi ra ổ đĩa.

Để RAM không đầy, mỗi worker chỉ giữ **vài ca** trong RAM (`pool_size`): nạp ca mới, bỏ ca cũ, rồi cắt nhiều lát cắt hoặc khối ngẫu nhiên từ các ca đang giữ.

| Cách | Ổ đĩa thêm | Tốc độ nạp 1 ca | Khi nào dùng |
|---|---|---|---|
| `source: tar` (mặc định) | **0 GB** | **0.34 s** đọc + 0.17 s tiền xử lý (đo trên máy bạn) | Bắt đầu, ổ đĩa ít |
| `source: npz` (`scripts/cache_dataset.py`) | ~10 MB/ca → ~12.5 GB | nhanh hơn ~3 lần | GPU phải chờ dữ liệu (GPU util thấp) và ổ đĩa còn chỗ |
| Giải nén một phần rồi xoá | tuỳ | — | Không cần: cách `tar` đã hiệu quả hơn |

**Đã kiểm tra (17/09):** file `BraTS2021_Training_Data.tar` tải xong, 13.396.408.320 byte, chứa **1251 ca đủ 5 file** (6255 file `.nii.gz` + 1 file rác `.DS_Store` bị bỏ qua). Không có file hỏng. Chi tiết thống kê ở `docs/01` mục 4.4; chạy lại bất cứ lúc nào bằng:

```bash
python scripts/check_data.py --config configs/base.yaml     # khoảng 3-5 phút cho cả 1251 ca
python scripts/check_data.py --config configs/base.yaml --limit 100   # bản nhanh
```

Hai file `BraTS2021_00495.tar` và `BraTS2021_00621.tar` là 2 ca **đã có sẵn trong tar lớn**, chỉ dùng để chạy thử; không khai báo chúng cùng lúc với tar lớn trong `tar_paths`.

---

## 4. Tài nguyên và tốc độ (đã đo trên RTX 3050 Ti 4 GB)

| Cấu hình | Params | VRAM đỉnh | Thời gian 1 bước | 1 epoch (250 bước) |
|---|---|---|---|---|
| `unet2d`, batch 8, 160² | 5.7 M | chưa đo (chạy `smoke_test`) | ~0.31 s | **~80 s** |
| `transunet2d`, batch 8, 160² | 8.6 M | **0.84 GB** | 2.25 s → đo lại sau bản tối ưu | cần đo lại |
| `transunet3d_small`, batch 1, 96³ | 6.2 M | chưa đo | chưa đo | chưa đo |

VRAM còn rất dư (0.84 / 4 GB), nên **có thể tăng `train.batch_size` lên 16 hoặc 24** để dùng GPU hiệu quả hơn — cứ chạy `smoke_test.py --opts train.batch_size=16` trước khi đổi.

Quy đổi thời gian: 1 epoch ≈ `250 × thời gian mỗi bước`. Ví dụ U-Net 2D 80 s/epoch → 100 epoch ≈ 2,2 giờ.

**RAM:** mỗi ca sau khi crop chiếm ~31 MB (ảnh giữ ở float16 trong pool). Dữ liệu tốn khoảng `num_workers × pool_size × 31 MB`, cộng ~3 GB cho Python/PyTorch — mặc định (2 × 4) dưới 4 GB.

**Tốc độ đọc dữ liệu:** 1 ca mất ~0.5 s (đọc + tiền xử lý). Với `samples_per_load: 16`, 2 worker cung cấp khoảng 60 mẫu/giây. U-Net 2D batch 8 tiêu thụ khoảng 26 mẫu/giây nên dữ liệu chưa phải nút thắt; nếu tăng batch hoặc dùng model nhanh hơn thì tăng `samples_per_load` / `num_workers`.

---

## 5. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `CUDA out of memory` | Đóng trình duyệt; giảm `train.batch_size`; 3D thì giảm `train.patch_size`; tăng `train.grad_accum` để giữ batch hiệu dụng |
| `torch.cuda.is_available()` trả về False | Đang cài nhầm bản CPU: cài lại torch theo đúng index CUDA (mục 2.3) |
| PowerShell báo không chạy được `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` rồi thử lại |
| GPU util thấp (xem `nvidia-smi -l 1`), train chậm | Dữ liệu là nút thắt: tăng `train.num_workers` (3–4), tăng `samples_per_load`, hoặc dùng cache npz |
| Worker bị kill, `DataLoader worker exited unexpectedly` | Hết RAM: giảm `train.num_workers` hoặc `train.pool_size`. Nếu vẫn lỗi, thử `train.num_workers=0` để xem có phải do đa tiến trình không |
| `build_index` báo < 1251 ca | File tar chưa tải xong hoặc hỏng |
| Loss giảm nhưng Dice ET = 0 | Bình thường ở đầu. Nếu kéo dài: tăng `fg_prob` (0.7), train lâu hơn, kiểm tra `no_object_weight` |
| Đọc dữ liệu chậm, GPU chờ | Tăng `train.samples_per_load` / `pool_size`, hoặc tạo cache npz (mục 3). Loại trừ phần mềm diệt virus quét file tar 13 GB |

---

## 6. Cấu trúc code

```
brats-transunet/
├── configs/                 cấu hình YAML (base.yaml + các biến thể kế thừa)
├── docs/                    tài liệu (file này, giải thích đề tài)
├── scripts/
│   ├── build_index.py       index tar + kiểm tra đủ ca
│   ├── check_data.py        kiểm tra toàn bộ dữ liệu + thống kê nhãn
│   ├── make_splits.py       chia train/val/test theo ca
│   ├── inspect_case.py      xem 1 ca / thống kê dữ liệu + vẽ hình
│   ├── cache_dataset.py     (tuỳ chọn) tar → npz để đọc nhanh
│   ├── smoke_test.py        thử model + đo VRAM
│   └── plot_log.py          vẽ đường cong train
├── src/brats/
│   ├── io.py                đọc tar / thư mục / npz
│   ├── preprocess.py        crop, chuẩn hóa, nhãn → 3 vùng
│   ├── datasets.py          dòng dữ liệu 2D (lát cắt) / 3D (khối) + augmentation MONAI
│   ├── models/unet.py       U-Net 2D/3D
│   ├── models/vit.py        ViT bottleneck (encoder-only)
│   ├── models/transformer_decoder.py   masked-attention decoder (query, coarse-to-fine)
│   ├── models/transunet.py  TransUNet = U-Net + decoder
│   ├── losses.py            Dice+BCE; Hungarian set loss
│   ├── metrics.py           Dice, HD95 theo WT/TC/ET
│   ├── inference.py         dự đoán cả volume (2D từng lát / 3D sliding window)
│   └── viz.py               vẽ ảnh + nhãn
├── train.py                 train (resume, AMP, log CSV, lưu best/last)
└── evaluate.py              đánh giá, lưu CSV / NIfTI / hình
```

---

## 7. Nhật ký tiến độ

| Ngày | Việc | Kết quả |
|---|---|---|
| 17/09 | Đọc paper + code gốc, dựng codebase, chạy thử 2 ca mẫu | Pipeline chạy thông |
| 17/09 | Tải xong tar, `check_data` toàn bộ | 1251 ca, 6255 file, không lỗi; thống kê nhãn ở `docs/01` mục 4.4 |
| 18/09 | Setup Windows native + GPU | `smoke_test` TransUNet 2D: 0.84 GB VRAM, 2.25 s/bước |
| 18/09 | **U-Net 2D, 140 ca train, 20 epoch** (~80 s/epoch) | val Dice mean 0.905; **test (30 ca): WT 0.870, TC 0.857, ET 0.830, mean 0.852**; trung vị 0.914 / 0.920 / 0.892; 1/30 ca Dice < 0.5 |
| 18/09 | Tối ưu tốc độ TransUNet (gom Hungarian về CPU 1 lần, lấy mẫu 12.544 điểm cho loss mask) | 2.25 s → **0.81 s mỗi bước**, epoch từ ~560 s xuống **203 s** |
| 18/09 | **TransUNet 2D, cùng 140 ca, cùng 20 epoch** (~203 s/epoch, 68 phút) | val tốt nhất ở epoch 4: mean 0.868. Về sau WT tăng (0.934) nhưng **TC tụt dần 0.856 → 0.694** |

Nhận xét từ hai lần train đầu:

- Baseline 2D đã đạt mức hợp lý chỉ với 140 ca và 20 epoch. Paper (3D, toàn bộ dữ liệu, 5-fold): WT 93.9 / TC 92.5 / ET 88.9.
- Điểm trung bình bị kéo xuống bởi vài ca hỏng hẳn, không phải do sai lệch đều: ca `BraTS2021_00540` chỉ có 12.486 voxel u (0,14% thể tích, ET vỏn vẹn 157 voxel) và mô hình bỏ sót gần hết. Vì vậy khi báo cáo nên đưa **cả trung vị và số ca Dice < 0.5**, và phân tích riêng nhóm u nhỏ.
- HD95 trung bình cao (16.9 mm) cũng do vài ca hỏng; trung vị thấp hơn nhiều.

**Vì sao TC của TransUNet tụt?** Soi checkpoint cho thấy vấn đề nằm ở phần **phân lớp của query**, không phải ở mask:

- Ca dễ (00495): mỗi vùng có đúng một query "chuyên trách" với xác suất lớp ≈ 1.0 → Dice TC 0.96.
- Ca khó (00621, không có NCR, ET rất nhỏ): tổng xác suất lớp trên mọi query chỉ đạt **TC 0.75 và ET 0.50** — không query nào dám nhận lớp. Vì xác suất vùng = Σ P(lớp | query) × sigmoid(mask), thiếu tự tin ở phần phân lớp làm cả vùng bị nhân với hệ số < 1 và rơi xuống dưới ngưỡng 0.5.
- U-Net không gặp chuyện này vì mỗi vùng là một kênh sigmoid độc lập.

Đã sửa phần hậu xử lý: thay vì cắt ngưỡng rồi lấy giao (làm cụt cả TC lẫn ET), nay lấy **max lũy tiến** P(WT) ≥ P(TC) ≥ P(ET) rồi mới cắt ngưỡng. Thử trên 2 ca mẫu: Dice trung bình 0.683 → 0.707, riêng TC 0.648 → 0.690. Hạ ngưỡng xuống 0.3 hoặc chuẩn hóa lại xác suất lớp (bỏ lớp "không có gì") đều làm **tệ hơn**, nên giữ ngưỡng 0.5.
