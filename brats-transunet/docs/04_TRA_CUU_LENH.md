# 04 — Tra cứu lệnh và tham số

Mọi lệnh chạy trong PowerShell đã `activate` môi trường, **đứng tại thư mục gốc project**:

```powershell
cd "D:\Learning Projects\CV\brats-transunet"
.\.venv\Scripts\Activate.ps1
```

Quy ước chung của tất cả script:

- `--config <file.yaml>` chọn cấu hình. File con kế thừa `base.yaml` qua khóa `base:`.
- `--opts key=value ...` ghi đè bất kỳ khóa nào trong config, dùng dấu chấm cho khóa lồng nhau. Giá trị đọc theo cú pháp YAML: `train.epochs=50`, `model.decoder.num_queries=3`, `train.patch_size=[128,128]`, `train.amp=false`.
- Đường dẫn trong config viết kiểu `D:/...`; trong `--opts` không được có dấu cách (nếu có thì bọc nháy: `--opts "data.tar_paths=[D:/a b/c.tar]"`).

---

## 1. Bảng lệnh nhanh

| Việc | Lệnh |
|---|---|
| Đánh chỉ mục file tar | `python scripts/build_index.py --config configs/base.yaml --check 3` |
| Kiểm tra dữ liệu + thống kê nhãn | `python scripts/check_data.py --config configs/base.yaml` |
| Chia train/val/test | `python scripts/make_splits.py --config configs/base.yaml` |
| Xem 1 ca, xuất ảnh | `python scripts/inspect_case.py --config configs/base.yaml --case BraTS2021_00495` |
| Thống kê nhiều ca | `python scripts/inspect_case.py --config configs/base.yaml --stats 100` |
| Đo VRAM và tốc độ | `python scripts/smoke_test.py --config configs/transunet2d.yaml` |
| Train | `python train.py --config configs/unet2d.yaml` |
| Train tiếp sau khi dừng | `python train.py --config configs/unet2d.yaml --resume` |
| Đánh giá | `python evaluate.py --ckpt runs/unet2d/best.pt --split test --hd95` |
| Vẽ đường cong | `python scripts/plot_log.py runs/unet2d/log.csv runs/transunet2d/log.csv` |
| Tạo cache npz (tuỳ chọn) | `python scripts/cache_dataset.py --config configs/base.yaml --out D:/.../brats_cache` |

---

## 2. Chi tiết từng script

### 2.1 `scripts/build_index.py` — ghi vị trí từng file trong .tar

Chạy **một lần** cho mỗi bộ tar. Kết quả lưu ở `data.index_path`; các lệnh sau đọc lại file này.

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--config` | `configs/base.yaml` | file cấu hình |
| `--rebuild` | tắt | quét lại kể cả khi index đã có (dùng khi đổi/ghi đè file tar) |
| `--check N` | 2 | đọc thử N ca để chắc chắn dữ liệu giải nén được |
| `--opts` | — | ghi đè config |

In ra số ca tìm được. Kỳ vọng **1251**; ít hơn nghĩa là tar thiếu hoặc hỏng.

### 2.2 `scripts/check_data.py` — kiểm tra toàn bộ dữ liệu

Đọc header của 4 ảnh và đọc đầy đủ file nhãn của từng ca. Mất khoảng 3–5 phút cho 1251 ca (dừng giữa chừng bằng Ctrl+C cũng không sao, script chỉ đọc).

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--config` | `configs/base.yaml` | |
| `--limit N` | 0 (tất cả) | chỉ kiểm N ca đầu, để chạy nhanh |
| `--opts` | — | |

In ra: số ca, kích thước, spacing, kiểu dữ liệu, danh sách lỗi, bảng tỉ lệ thể tích NCR/ED/ET/WT/TC và danh sách ca không có ET.

### 2.3 `scripts/make_splits.py` — chia dữ liệu theo bệnh nhân

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--config` | `configs/base.yaml` | |
| `--ratios a b c` | `0.7 0.15 0.15` | tỉ lệ train / val / test |
| `--limit N` | 0 | chỉ lấy N ca (tập nhỏ để thử) |
| `--seed` | 42 | xáo trộn cố định, chạy lại cho kết quả giống hệt |
| `--out` | `splits` trong config | nơi ghi file JSON |
| `--opts` | — | |

Ví dụ tập nhỏ 200 ca: `--limit 200 --out data/splits_200.json` → dùng khi train bằng `--opts splits=data/splits_200.json`.

### 2.4 `scripts/inspect_case.py` — xem dữ liệu

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--case <id>` | ca đầu tiên | in thông tin 1 ca và lưu ảnh vào `figures/<id>.png` |
| `--stats N` | 0 | thay vì 1 ca, thống kê N ca đầu (tỉ lệ NCR/ED/ET, kích thước sau crop) |
| `--out <thư mục>` | `figures` | nơi lưu ảnh |

### 2.5 `scripts/smoke_test.py` — thử model, đo VRAM

Chạy vài bước forward + backward bằng dữ liệu ngẫu nhiên, **không cần dataset**. Dùng để chọn `batch_size` và `patch_size` trước khi train thật.

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--config` | bắt buộc | |
| `--iters N` | 5 | số bước chạy thử |
| `--opts` | — | ví dụ `train.batch_size=16` |

In ra: số tham số, thời gian mỗi bước, **PEAK VRAM**. Nếu báo hết VRAM: giảm `train.batch_size`, rồi tới `train.patch_size`.

### 2.6 `train.py` — huấn luyện

| Tham số | Ý nghĩa |
|---|---|
| `--config` | bắt buộc |
| `--resume` | nạp lại `runs/<tên>/last.pt` và chạy tiếp |
| `--opts` | ghi đè config |

Trong lúc train: mỗi epoch chạy `train.iters_per_epoch` bước; cứ `eval.every` epoch thì đánh giá trên `eval.val_cases` ca val (dự đoán cả volume).

Sinh ra trong `output_dir`:

| File | Nội dung |
|---|---|
| `last.pt` | checkpoint cuối mỗi epoch (dùng cho `--resume`) |
| `best.pt` | checkpoint có `dice_mean` trên val cao nhất |
| `log.csv` | `epoch, lr, train_loss, sec, dice_WT, dice_TC, dice_ET, dice_mean` |

Ctrl+C an toàn: checkpoint đã lưu, chạy lại với `--resume`.

### 2.7 `evaluate.py` — đánh giá và xuất kết quả

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--ckpt` | bắt buộc | ví dụ `runs/unet2d/best.pt` (config được lưu sẵn trong checkpoint) |
| `--split` | `val` | `train` / `val` / `test` |
| `--limit N` | 0 | chỉ đánh giá N ca đầu |
| `--hd95` | tắt | tính thêm khoảng cách biên HD95 (chậm hơn) |
| `--save-nifti` | tắt | xuất `<id>_pred.nii.gz` kích thước 240×240×155, nhãn 0/1/2/4 |
| `--figures N` | 3 | số ca lưu ảnh so sánh nhãn thật và dự đoán |

Kết quả nằm ở `<thư mục checkpoint>/eval_<split>/`: `metrics.csv` (từng ca), ảnh `.png`, và file `.nii.gz` nếu bật.

### 2.8 `scripts/plot_log.py` — vẽ đường cong

```powershell
python scripts/plot_log.py runs/unet2d/log.csv runs/transunet2d/log.csv --out figures/so_sanh.png
```

Truyền nhiều file log để so sánh nhiều mô hình trên cùng một hình.

### 2.9 `scripts/cache_dataset.py` — cache .npz (tuỳ chọn)

Chỉ dùng khi GPU phải chờ dữ liệu. Ghi mỗi ca một file `.npz` đã crop (~10 MB/ca, tổng ~12,5 GB), chạy tiếp được nếu bị ngắt.

```powershell
python scripts/cache_dataset.py --config configs/base.yaml --out "D:/Learning Projects/CV/brats_cache"
python train.py --config configs/unet2d.yaml --opts data.source=npz "data.npz_dir=D:/Learning Projects/CV/brats_cache"
```

---

## 3. Các khóa trong config

### 3.1 `data`

| Khóa | Ý nghĩa |
|---|---|
| `source` | `tar` (đọc thẳng file .tar), `npz` (cache), `dir` (thư mục đã giải nén) |
| `tar_paths` | danh sách file .tar |
| `index_path` | nơi lưu index của tar |
| `npz_dir` / `dir` | dùng với `source: npz` / `dir` |

### 3.2 `model`

| Khóa | Ý nghĩa |
|---|---|
| `name` | `unet` hoặc `transunet` |
| `spatial_dims` | 2 (lát cắt) hoặc 3 (khối) |
| `features` | số kênh mỗi tầng, ví dụ `[32,64,128,256,320]`. Số phần tử = số tầng |
| `decoder.hidden` | chiều vector query (paper: 192) |
| `decoder.num_queries` | số query (paper: 20) |
| `decoder.layers` | số vòng tinh chỉnh (paper: 3) |
| `decoder.heads` | số đầu attention |
| `decoder.ffn_mult` | hệ số nhân chiều ẩn của FFN |
| `decoder.masked_attn` | `true` = coarse-to-fine như paper; `false` = cross-attention thường (dùng cho ablation) |
| `decoder.ms_idxs` | tầng feature dùng cho cross-attention, mặc định `[-4,-3,-2]` (1/8, 1/4, 1/2) |
| `vit_bottleneck` | thêm ViT ở đáy U-Net: `{depth: 4, hidden: 384, heads: 6}`. Bỏ khóa này = không dùng |

### 3.3 `train`

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `patch_size` | `[160,160]` | 2D `(H,W)`; 3D `(X,Y,Z)` |
| `batch_size` | 8 | giảm nếu hết VRAM |
| `grad_accum` | 1 | cộng dồn gradient để giả lập batch lớn |
| `num_workers` | 2 | số tiến trình đọc dữ liệu |
| `pool_size` | 4 | số ca mỗi worker giữ trong RAM (~31 MB/ca) |
| `samples_per_load` | 16 | số mẫu cắt ra mỗi lần nạp một ca mới |
| `fg_prob` | 0.5 | xác suất lấy mẫu rơi vào vùng có u |
| `augment.affine_prob` | 0.2 | xác suất xoay/phóng (chỉ bản 2D); `augment: {}` = chỉ lật + đổi cường độ |
| `epochs` | 100 | |
| `iters_per_epoch` | 250 | 1 epoch = số bước cố định |
| `warmup_epochs` | 2 | tăng dần learning rate lúc đầu |
| `lr` | 3e-4 | |
| `weight_decay` | 3e-5 | |
| `amp` | true | tính 16-bit, tiết kiệm VRAM |
| `print_every` | 50 | in log mỗi bao nhiêu bước |

### 3.4 `eval`

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `every` | 5 | validate mỗi bao nhiêu epoch |
| `val_cases` | 10 | số ca val dùng trong lúc train (đánh giá đầy đủ thì dùng `evaluate.py`) |
| `batch_size` | 16 | số lát cắt mỗi lần suy luận (bản 2D) |
| `sw_batch_size` / `overlap` | 1 / 0.5 | cửa sổ trượt (bản 3D) |
| `threshold` | 0.5 | ngưỡng biến xác suất thành mask. Hạ xuống 0.3–0.4 nếu mô hình dự đoán thiếu (thử trên val, đừng chỉnh theo test) |

### 3.5 `loss` (chỉ dùng cho `transunet`)

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `cost_weight` | `[2.0, 5.0, 5.0]` | trọng số lớp / BCE / Dice khi ghép cặp và tính loss (như paper) |
| `no_object_weight` | 0.1 | trọng số lớp "không có gì", tránh 20 query đều đoán rỗng |
| `num_points` | 12544 | số điểm lấy mẫu ngẫu nhiên khi tính chi phí ghép cặp và loss mask; `0` = dùng toàn bộ voxel (chậm hơn nhiều, nhất là bản 3D) |
| `reuse_match` | false | `true` = ghép cặp Hungarian một lần rồi dùng lại cho các lớp phụ. Nhanh hơn nhưng lệch nhẹ so với paper; chỉ bật nếu cần tăng tốc |

---

## 4. Công thức dùng thường xuyên

**Train thử nhanh (kiểm tra config mới):**

```powershell
python train.py --config configs/transunet2d.yaml --opts train.epochs=1 train.iters_per_epoch=20 eval.every=1 eval.val_cases=2 output_dir=runs/tmp
```

**Train trên tập nhỏ 200 ca:**

```powershell
python scripts/make_splits.py --config configs/base.yaml --limit 200 --out data/splits_200.json
python train.py --config configs/unet2d.yaml --opts splits=data/splits_200.json output_dir=runs/unet2d_200 train.epochs=20
```

**So sánh công bằng U-Net và TransUNet** (cùng split, cùng số epoch, chỉ khác model):

```powershell
python train.py --config configs/unet2d.yaml      --opts splits=data/splits_200.json output_dir=runs/unet2d_200 train.epochs=40
python train.py --config configs/transunet2d.yaml --opts splits=data/splits_200.json output_dir=runs/tu2d_200  train.epochs=40
python scripts/plot_log.py runs/unet2d_200/log.csv runs/tu2d_200/log.csv
```

**Ablation (mục tiêu tuần 4):**

```powershell
# tắt masked attention (coarse-to-fine)
python train.py --config configs/transunet2d.yaml --opts model.decoder.masked_attn=false output_dir=runs/tu2d_nomask
# số query = số lớp
python train.py --config configs/transunet2d.yaml --opts model.decoder.num_queries=3 output_dir=runs/tu2d_q3
# encoder-only (ViT ở bottleneck)
python train.py --config configs/transunet2d_encoder.yaml
```

**Bản 3D thu nhỏ:**

```powershell
python scripts/smoke_test.py --config configs/transunet3d_small.yaml
python scripts/smoke_test.py --config configs/transunet3d_small.yaml --opts train.patch_size=[64,64,64]   # nếu hết VRAM
python train.py --config configs/transunet3d_small.yaml
```

**Train qua đêm rồi sáng xem lại:**

```powershell
python train.py --config configs/transunet2d.yaml --resume
Get-Content runs/transunet2d/log.csv -Tail 5      # xem 5 epoch gần nhất
nvidia-smi -l 5                                    # theo dõi VRAM, GPU util
```

---

## 5. Lỗi hay gặp khi gõ lệnh

| Hiện tượng | Nguyên nhân / cách sửa |
|---|---|
| `FileNotFoundError: data/splits.json` | chưa chạy `make_splits.py`, hoặc quên `--opts splits=...` khi dùng split khác |
| `--opts` đứng cuối mà không có gì theo sau | phải có ít nhất một `key=value`, hoặc bỏ hẳn `--opts` |
| `KeyError` tên khóa khi dùng `--opts` | sai tên khóa: đối chiếu mục 3, nhớ đủ cấp (`model.decoder.num_queries`, không phải `num_queries`) |
| Đường dẫn có dấu cách bị cắt | bọc cả cụm trong nháy kép: `--opts "data.npz_dir=D:/Learning Projects/CV/brats_cache"` |
| `CUDA out of memory` | đóng trình duyệt, giảm `train.batch_size`, rồi `train.patch_size`; tăng `train.grad_accum` để giữ batch hiệu dụng |
| Train chậm, GPU util thấp | tăng `train.samples_per_load`, `train.pool_size`, `train.num_workers`; hoặc dùng cache npz |
| Muốn train lại từ đầu | xoá thư mục `runs/<tên>` (hoặc đổi `output_dir`), vì `--resume` sẽ nạp `last.pt` cũ |
