# 06 — Hỏi đáp (Phần 2/3): Dữ liệu BraTS2021, tiền xử lý, 2D và 3D

> **Bộ hỏi đáp gồm 3 phần:**
>
> 1. Phần 1 — Mô hình: [`05_HOI_DAP_P1_MO_HINH.md`](05_HOI_DAP_P1_MO_HINH.md)
> 2. **Phần 2 — Dữ liệu, tiền xử lý, 2D và 3D** (file này)
> 3. Phần 3 — Train, đánh giá, so với paper, chuẩn bị báo cáo: [`07_HOI_DAP_P3_TRAIN_DANH_GIA.md`](07_HOI_DAP_P3_TRAIN_DANH_GIA.md)
>
> Câu có dấu ⭐ là câu bạn hỏi trực tiếp. Quy ước về shape, đường dẫn và số dòng giống Phần 1.

## Mục lục Phần 2

**Dữ liệu**

- Câu 19 ⭐ BraTS2021 là gì? Một mẫu dữ liệu gồm những gì, định dạng ra sao?
- Câu 20 — NIfTI là gì? Trục, affine, cách mở xem?
- Câu 21 — 4 loại MRI khác nhau thế nào? Tại sao cần cả 4?
- Câu 22 — Nhãn 0/1/2/4 và 3 vùng WT/TC/ET là gì? Tại sao đổi nhãn sang vùng?
- Câu 23 ⭐ Dữ liệu được xử lý thế nào trước khi vào model?
- Câu 24 — Tại sao crop theo vùng não? Tại sao z-score chỉ trên voxel não?
- Câu 25 — Lấy mẫu và "pool" hoạt động thế nào? Tại sao 50% mẫu có u?
- Câu 26 — Augmentation gồm những gì? Khác nnU-Net ra sao?
- Câu 27 — Chia train/val/test thế nào? Tại sao chia theo bệnh nhân?
- Câu 28 — Đọc thẳng từ file .tar mà không giải nén bằng cách nào?
- Câu 29 — Dữ liệu mất cân bằng thế nào? Ca không có ET ảnh hưởng gì?

**2D và 3D**

- Câu 30 ⭐ "2D" là gì, "3D" là gì?
- Câu 31 — Mô hình 2D và 3D trong code khác nhau cụ thể ở đâu?
- Câu 32 — Train và suy luận 2D khác 3D thế nào?
- Câu 33 — Ưu, nhược của 2D và 3D? Vì sao paper dùng 3D mà nhóm bắt đầu bằng 2D?
- Câu 34 — "2.5D" là gì? Có nên thử?
- Tóm tắt Phần 2

---

# A. DỮ LIỆU

## Câu 19 ⭐ — BraTS2021 là gì? Một mẫu dữ liệu gồm những gì, định dạng ra sao?

**BraTS2021** là bộ dữ liệu của cuộc thi phân vùng u não RSNA-ASNR-MICCAI BraTS 2021. Dữ liệu là MRI não **trước phẫu thuật** của bệnh nhân **u thần kinh đệm (glioma)**, thu từ nhiều bệnh viện. Nhóm dùng **1251 ca có nhãn** (tập train của cuộc thi). Tập validation và test của cuộc thi không công bố nhãn nên không dùng được.

**1 ca = 1 bệnh nhân = 5 file NIfTI nén** (`.nii.gz`), nằm trong thư mục `BraTS2021_XXXXX/`:

| File | Nội dung | Shape | Kiểu dữ liệu | Giá trị |
|---|---|---|---|---|
| `..._t1.nii.gz` | MRI T1 | 240×240×155 | int16 (191 ca là float32) | 0 ở nền; trong não từ vài trăm đến vài nghìn; không có đơn vị |
| `..._t1ce.nii.gz` | T1 sau tiêm thuốc cản quang | 240×240×155 | như trên | như trên |
| `..._t2.nii.gz` | T2 | 240×240×155 | như trên | như trên |
| `..._flair.nii.gz` | FLAIR | 240×240×155 | như trên | như trên |
| `..._seg.nii.gz` | **Nhãn** do chuyên gia vẽ | 240×240×155 | uint8 / uint16 / int16 tuỳ ca | 0 = nền, 1 = NCR, 2 = ED, 4 = ET |

- 240×240×155 = 8 928 000 voxel mỗi file. Mỗi voxel là khối 1×1×1 mm.
- Dung lượng nén khoảng 10 MB/ca. Cả bộ là file tar 13.4 GB.
- Ban tổ chức đã làm sẵn: **đăng ký chung khung** (4 ảnh trùng khít đến từng voxel), **đưa về 1 mm³**, **xoá hộp sọ** (skull-stripping, nên nền = 0).

Sau khi đọc, `src/brats/io.py` trả về một dict:

```python
{"id": "BraTS2021_00495",
 "image": float32 (4, 240, 240, 155),   # thứ tự kênh: t1, t1ce, t2, flair
 "seg":   uint8   (240, 240, 155),      # 0, 1, 2, 4
 "affine": (4, 4)}                      # ma trận đổi chỉ số voxel → tọa độ mm
```

**Phân biệt các khái niệm dễ lẫn:**

| Khái niệm | Là gì | Shape trong code |
|---|---|---|
| **Ca** (case) | 1 bệnh nhân, 5 file | image (4, 240, 240, 155) + seg (240, 240, 155) |
| **Volume đã tiền xử lý** | ca sau khi crop, chuẩn hóa, đổi nhãn | image (4, ~140, ~175, ~140) + regions (3, ~140, ~175, ~140) |
| **Mẫu train** (sample) | 1 lát cắt 2D hoặc 1 khối 3D cắt ra từ volume | 2D: image (4, 160, 160), nhãn (3, 160, 160). 3D: (4, 96, 96, 96), (3, 96, 96, 96) |
| **Batch** | nhiều mẫu xếp chồng | 2D: (8, 4, 160, 160) + (8, 3, 160, 160) |

---

## Câu 20 — NIfTI là gì? Trục, affine, cách mở xem?

- NIfTI-1 (`.nii`) gồm **header 348 byte** và **mảng dữ liệu** ngay sau đó. Đuôi `.gz` nghĩa là cả file được nén gzip.
- Header chứa: số chiều và kích thước (`dim`), kích thước voxel theo mm (`pixdim`), kiểu dữ liệu (`datatype`), vị trí bắt đầu mảng dữ liệu (`vox_offset`), và ma trận **affine** 4×4.
- `scripts/check_data.py` đọc thẳng các trường này theo vị trí byte: `dim` ở byte 40, `datatype` ở byte 70, `pixdim` ở byte 76, `vox_offset` ở byte 108, chuỗi nhận dạng `n+1` ở byte 344.
- **Affine** đổi chỉ số voxel (i, j, k) thành tọa độ thật (mm) trong không gian máy chụp. Code giữ affine lại và dùng khi lưu file dự đoán (`evaluate.py --save-nifti`), để file dự đoán chồng khít lên ảnh gốc khi mở bằng phần mềm xem ảnh.
- **Trục của mảng** (theo cách nibabel đọc): `(x, y, z)` ≈ (trái–phải, trước–sau, dưới–trên). Lát **axial** (cắt ngang) là `vol[:, :, z]`, có 155 lát.
- Đọc bằng Python: `nib.load(path)`. Code nhóm đọc từ bytes trong RAM: `nib.Nifti1Image.from_bytes(gzip.decompress(raw))` (`io.py:32-35`).
- Xem bằng mắt: **ITK-SNAP** hoặc **3D Slicer** (mở ảnh FLAIR, rồi mở file seg làm lớp nhãn). Hoặc chạy `python scripts/inspect_case.py --config configs/base.yaml --case BraTS2021_00495` để xuất ảnh PNG.

---

## Câu 21 — 4 loại MRI (modality) khác nhau thế nào? Tại sao cần cả 4?

Cùng một bệnh nhân được chụp 4 kiểu. Mỗi kiểu làm nổi một loại mô. Model nhận 4 ảnh này như **4 kênh màu** của cùng một bức ảnh:

| Modality | Sáng ở đâu | Giúp tìm |
|---|---|---|
| T1 | cấu trúc giải phẫu, chất trắng và chất xám | nền giải phẫu |
| **T1ce** (T1 sau tiêm Gadolinium) | chỗ hàng rào máu não bị phá vỡ → **viền u sáng lên**; lõi hoại tử tối bên trong viền | **ET**, và **NCR** (vùng tối nằm trong viền sáng) → **TC** |
| T2 | dịch, phù nề | phù nề, lõi u |
| **FLAIR** | giống T2 nhưng dịch não tủy bị xoá tối → **phù nề quanh u nổi rõ** | **WT** (toàn bộ vùng bệnh) |

Không modality nào đủ một mình. FLAIR thấy toàn vùng bệnh nhưng không tách được lõi u. T1ce thấy phần tăng cường nhưng không thấy phù nề. Model học cách **kết hợp**, ví dụ: "sáng trên FLAIR nhưng không tăng cường trên T1ce" thì nhiều khả năng là phù nề (ED).

---

## Câu 22 — Nhãn 0/1/2/4 và 3 vùng WT/TC/ET là gì? Tại sao đổi nhãn sang vùng?

Nhãn gốc trong file seg:

- 0 = nền và mô não lành
- 1 = **NCR**: lõi hoại tử (mô chết bên trong u)
- 2 = **ED**: phù nề quanh u
- 4 = **ET**: phần u tăng cường (bắt thuốc cản quang)
- Không có nhãn 3. Nhãn 3 của các năm BraTS cũ đã được gộp vào nhãn 1.

BraTS **chấm điểm theo 3 vùng lồng nhau**:

```
WT (whole tumor) = 1 ∪ 2 ∪ 4    ⊃    TC (tumor core) = 1 ∪ 4    ⊃    ET (enhancing tumor) = 4

┌──────────────── WT ─────────────────┐
│  phù nề (2)                         │
│   ┌──────────── TC ───────────┐     │
│   │  hoại tử (1)              │     │
│   │   ┌────── ET ─────┐       │     │
│   │   │ tăng cường (4)│       │     │
│   │   └───────────────┘       │     │
│   └───────────────────────────┘     │
└─────────────────────────────────────┘
```

Code đổi nhãn (`preprocess.py:18-19`):

```python
np.stack([seg > 0, (seg == 1) | (seg == 4), seg == 4], 0)   # → (3, X, Y, Z), giá trị 0/1: [WT, TC, ET]
```

Chiều ngược lại (`regions_to_seg`, `preprocess.py:22`) dùng khi lưu dự đoán: chỉ thuộc WT → 2, thuộc TC nhưng không thuộc ET → 1, thuộc ET → 4.

**Tại sao model dự đoán vùng thay vì nhãn 1/2/4:**

1. Tối ưu đúng thứ được chấm điểm.
2. 3 vùng **chồng lên nhau**, nên mỗi vùng là một kênh sigmoid độc lập (xem Phần 3, Câu 39).
3. nnU-Net cũng làm vậy cho BraTS ("region-based training").

Repo gốc đánh số khác sau khi chuyển sang định dạng nnU-Net (ED → 1, NCR → 2, ET → 3), và định nghĩa vùng là WT = (1, 2, 3), TC = (2, 3), ET = (3) (`nnUNetTrainerV2_DDP.py:79-83`). **Cùng ý nghĩa**, chỉ khác cách đánh số.

---

## Câu 23 ⭐ — Dữ liệu được xử lý thế nào trước khi vào model?

Mọi bước làm "on-the-fly" trong RAM, không ghi file trung gian nào ra đĩa.

```
(chạy 1 lần) scripts/build_index.py → data/brats2021_tar_index.json
             ghi lại: ca → loại file → [đường dẫn tar, vị trí byte, kích thước]

① ĐỌC        io.TarSource.load        nhảy tới vị trí byte → đọc ~10 MB → giải nén gzip trong RAM → nibabel
             image (4,240,240,155) float32 [t1, t1ce, t2, flair]    seg (240,240,155) uint8
② CROP       preprocess.brain_bbox    tìm hộp bao quanh các voxel ≠ 0 (ở bất kỳ kênh nào) rồi cắt
             image (4,~140,~175,~140)  (lưu lại bbox và shape gốc để sau này trả về 240×240×155)
③ CHUẨN HÓA  preprocess.normalize     từng kênh: (x − mean) / std, chỉ tính trên voxel não; nền giữ 0
④ ĐỔI NHÃN   preprocess.seg_to_regions  seg → regions (3,~140,~175,~140) uint8 [WT, TC, ET]
──────────────── KHI TRAIN (datasets.py) ─────────────────────── KHI SUY LUẬN (inference.py) ─────────
⑤ LẤY MẪU    2D: chọn 1 lát z (50% là lát có u)                  dùng nguyên volume sau bước ④:
             → pad cho đủ 160 → crop 160×160                      2D: chạy lần lượt các lát rồi xếp chồng
               (nếu là mẫu có u: crop quanh một điểm u)            3D: cửa sổ trượt 96³
             3D: crop khối 96³ (50% crop quanh một voxel u)
⑥ AUGMENT    lật, (2D) xoay/phóng nhẹ, đổi cường độ, thêm nhiễu
⑦ BATCH      DataLoader xếp các mẫu: 2D (8,4,160,160) + (8,3,160,160)
                                      3D (1,4,96,96,96) + (1,3,96,96,96)
```

| Bước | Hàm | Input | Output |
|---|---|---|---|
| ① | `TarSource.load` (`io.py:99`) | mã ca | dict: image (4,240,240,155), seg, affine |
| ②③④ | `preprocess_case` (`preprocess.py:52`) | dict thô | image đã chuẩn hóa (4,X,Y,Z), regions (3,X,Y,Z), bbox, orig_shape, affine |
| ⑤ | `SliceStream2D._sample` / `PatchStream3D._sample` (`datasets.py:128`, `:153`) | volume | 1 mẫu (image, label) |
| ⑥ | `build_augment` (`datasets.py:54`) | 1 mẫu | mẫu đã biến đổi |
| ⑦ | `DataLoader` (`train.py:69`) | nhiều mẫu | batch |

Nhãn chỉ đi qua bước ④ và các phép biến đổi hình học (lật, xoay). Các phép hình học áp lên nhãn dùng nội suy **nearest** để nhãn vẫn là 0/1. Nhãn không đi qua các phép đổi cường độ.

---

## Câu 24 — Tại sao crop theo vùng não? Tại sao z-score chỉ trên voxel não?

- **Crop.** Hơn một nửa khối 240×240×155 là nền 0 (ngoài não). Sau crop còn khoảng 140×175×140 ≈ 3.4 triệu voxel (khoảng 38% ban đầu). Lợi ích: đỡ RAM, đỡ tính toán, và mẫu 2D/3D cắt ra chứa nhiều não hơn. nnU-Net cũng có bước này ("crop to non-zero").
- **Z-score** `x' = (x − μ) / σ`. Cường độ MRI **không có đơn vị chuẩn** (khác với CT có đơn vị Hounsfield). Mỗi máy, mỗi lần chụp cho một thang khác nhau. Chuẩn hóa từng ca, từng kênh về trung bình 0 và độ lệch chuẩn 1 để model thấy mọi ca "cùng một thang".
- **Chỉ tính trên voxel ≠ 0.** Nếu tính cả nền, hàng triệu số 0 kéo μ và σ lệch đi, và vì tỉ lệ nền/não khác nhau giữa các ca nên chuẩn hóa sẽ không nhất quán. Sau chuẩn hóa, nền vẫn giữ đúng bằng 0. nnU-Net làm giống vậy (tùy chọn `use_mask_for_norm`).
- **Không cần resample.** Mọi ca đều đã 1 mm³ và cùng kích thước 240×240×155 (đã kiểm tra bằng `check_data.py`).
- **Tại sao không dùng min-max?** Min-max rất nhạy với vài điểm sáng bất thường (outlier).

---

## Câu 25 — Lấy mẫu (sampling) và "pool" hoạt động thế nào? Tại sao 50% mẫu có u?

`src/brats/datasets.py` dùng `IterableDataset`, tức một dòng dữ liệu vô hạn:

1. Mỗi **worker** (tiến trình đọc dữ liệu, mặc định 2) nhận một phần danh sách ca train (`ids[wid::nw]`) rồi xáo trộn.
2. Worker giữ một **pool** tối đa `pool_size = 4` volume trong RAM. Mỗi lần nạp 1 ca mới (đẩy ca cũ nhất ra), worker cắt `samples_per_load = 16` mẫu ngẫu nhiên từ các ca đang có trong pool.
3. Mỗi mẫu có xác suất `fg_prob = 0.5` là **mẫu có u**: chọn lát hoặc khối có chứa u, crop quanh một voxel u có lệch ngẫu nhiên ±¼ kích thước patch. Còn lại là lát hoặc khối **ngẫu nhiên trong não**.

**Tại sao ép 50% mẫu có u?** U chỉ chiếm khoảng 1% thể tích (WT trung bình 1.075%). Nếu lấy ngẫu nhiên, phần lớn mẫu không có u, và một model "luôn đoán nền" cũng có loss thấp. nnU-Net cũng ép như vậy (33% mẫu phải chứa vùng cần tìm). 50% mẫu còn lại giúp model biết nền trông ra sao, tránh đoán u ở khắp nơi.

**RAM.** Code hiện tại giữ ảnh trong pool ở dạng **float32**: khoảng 55 MB ảnh + 10 MB nhãn cho mỗi ca. docs/02 ghi ~31 MB/ca vì giả định float16, nhưng code hiện không đổi sang float16. Với mặc định 2 worker × 4 ca, tổng khoảng 0.5 GB, vẫn nhẹ.

**Một hệ quả nhỏ nên biết.** Mỗi batch được tạo trọn trong một worker, nên các mẫu trong batch chỉ đến từ vài ca đang nằm trong pool của worker đó. Điều này không ảnh hưởng chuẩn hóa (InstanceNorm không dùng thống kê batch), nhưng gradient mỗi bước kém đa dạng hơn so với cách lấy mẫu của nnU-Net.

---

## Câu 26 — Augmentation gồm những gì? Khác nnU-Net ra sao?

Hàm `build_augment` (`datasets.py:54-66`), dùng MONAI:

| Phép biến đổi | Xác suất | Áp dụng cho |
|---|---|---|
| Lật theo từng trục không gian | 0.5 mỗi trục | ảnh + nhãn |
| Xoay ±15°, phóng to/thu nhỏ ±10% (`RandAffined`) | 0.2 | ảnh + nhãn, **chỉ bản 2D** |
| Nhân cường độ với (1 ± 0.1) | 0.5 | ảnh |
| Cộng cường độ ±0.1 | 0.5 | ảnh |
| Thêm nhiễu Gauss, σ = 0.1 | 0.15 | ảnh |

Repo gốc (BraTS, bộ augmentation "insaneDA" của nnU-Net, `nnUNetTrainerV2_DDP.py:134-194`) mạnh hơn nhiều: xoay ±90° theo mỗi trục, phóng 0.65–1.6 (có thể khác nhau theo từng trục), biến dạng đàn hồi (elastic), gamma 0.5–1.6, cộng độ sáng, nhiễu và làm mờ, giả lập độ phân giải thấp, lật gương.

Nhận xét: train ngắn thì augmentation nhẹ không sao. Train dài thì augmentation mạnh giúp đỡ overfit. Nên ghi điểm này vào mục "khác biệt so với paper" khi báo cáo.

**Chi tiết config dễ nhầm.** Các config 3D ghi `augment: {}`, nhưng `load_config` **gộp** (deep merge) với `base.yaml`. Gộp một dict rỗng không xoá được gì, nên kết quả vẫn là `{affine_prob: 0.2}`. Vì phép affine chỉ áp dụng cho 2D, bản 3D thực tế nhận "lật + đổi cường độ". Muốn **tắt hẳn** augmentation: `--opts train.augment=null`.

---

## Câu 27 — Chia train/val/test thế nào? Tại sao chia theo bệnh nhân?

`scripts/make_splits.py` xáo trộn danh sách 1251 ca với seed 42, chia theo tỉ lệ 70/15/15 → **876 / 188 / 187 ca**, lưu vào `data/splits.json`. Chạy lại cho kết quả y hệt.

- **Train:** để học.
- **Val:** để chọn checkpoint (`best.pt`), chọn ngưỡng và siêu tham số.
- **Test:** chỉ dùng một lần ở cuối để báo cáo.

**Chia theo ca, không theo lát cắt.** Các lát kề nhau gần như giống hệt nhau. Nếu lát 70 của một bệnh nhân nằm trong train còn lát 71 nằm trong test, model gần như "đã thấy đáp án", điểm sẽ cao ảo. Đây gọi là rò rỉ dữ liệu (data leakage).

- Paper dùng **5-fold cross-validation** theo cách chia của đội hạng 1 BraTS21 (repo không kèm file chia). Vì vậy điểm của nhóm và của paper **không đo trên cùng tập**; chỉ nên so sánh tương đối.
- `configs/debug_samples.yaml` dùng cùng 2 ca cho cả train, val và test. Cấu hình này **chỉ để thử code**, không phải kết quả.

---

## Câu 28 — Đọc thẳng từ file .tar mà không giải nén bằng cách nào?

- File `.tar` là dạng "đóng gói không nén": các file bên trong nằm **liền nhau**, mỗi file có một header 512 byte đi trước. Vì thế, biết vị trí byte là đọc thẳng được từng file.
- `build_tar_index` (`io.py:104`) duyệt file tar một lần, ghi `offset_data` và `size` của từng file `.nii.gz` vào file JSON.
- Khi cần một ca: `fh.seek(offset); fh.read(size)` → được bytes của file `.nii.gz` → `gzip.decompress` → `Nifti1Image.from_bytes`. Không ghi gì ra đĩa.
- Đo trên máy của nhóm: khoảng 0.34 s để đọc + 0.17 s để tiền xử lý mỗi ca.
- Windows tạo tiến trình con theo kiểu "spawn", không truyền được file đang mở cho worker. Vì vậy `__getstate__` bỏ file handle đi, và mỗi worker tự mở lại file tar (`io.py:80-90`).
- Phương án nhanh hơn nếu GPU phải chờ dữ liệu: tạo cache `.npz` bằng `scripts/cache_dataset.py` (khoảng 12.5 GB).

---

## Câu 29 — Dữ liệu mất cân bằng thế nào? Ca không có ET ảnh hưởng gì?

Thống kê trên toàn bộ 1251 ca (`scripts/check_data.py`, docs/01 mục 4.4):

| Vùng | % thể tích trung bình | Số ca rỗng |
|---|---|---|
| WT | 1.075% | 0 |
| TC | 0.400% | 6 |
| ET | 0.240% | **33** |

Hệ quả:

1. **Accuracy vô nghĩa.** Đoán toàn bộ là nền đã đúng khoảng 99%. Vì vậy dùng **Dice**.
2. Loss phải có thành phần **Dice**, và phải **ép lấy mẫu có u** (Câu 25).
3. **33 ca không có ET.** Theo quy ước BraTS: dự đoán rỗng thì Dice ET = 1; lỡ đoán dù chỉ 1 voxel ET thì Dice ET = 0. Vì thế điểm ET dao động mạnh và thường thấp nhất trong 3 vùng.
4. **U rất nhỏ** (ca nhỏ nhất chỉ có 2808 voxel WT) dễ bị bỏ sót hoàn toàn. Khi báo cáo nên đưa cả **trung vị** và **số ca có Dice < 0.5**, không chỉ trung bình.

---

# B. 2D VÀ 3D

## Câu 30 ⭐ — "2D" là gì, "3D" là gì?

- **Dữ liệu** luôn là 3D: một khối 240×240×155, tức 155 lát cắt 240×240 xếp chồng lên nhau.
- **Mô hình 2D:** mỗi lần chỉ nhận **một lát cắt** (ảnh phẳng H×W, 4 kênh). Muốn có kết quả cho cả khối thì chạy lần lượt từng lát rồi xếp chồng kết quả lại.
- **Mô hình 3D:** mỗi lần nhận **một khối** (X×Y×Z, 4 kênh) và dùng phép tích chập 3D.

| | 2D | 3D |
|---|---|---|
| Tensor | (B, C, H, W) | (B, C, X, Y, Z) |
| Kernel của conv | 3×3, trượt theo 2 hướng | 3×3×3, trượt theo 3 hướng |
| Một voxel "nhìn thấy" | hàng xóm trong **cùng lát** | hàng xóm ở cả **lát trên và lát dưới** |
| Số tham số mỗi conv | 9·C_in·C_out | 27·C_in·C_out |

```
2D: kernel 3×3 trên 1 lát            3D: kernel 3×3×3 xuyên qua 3 lát liền nhau
        lát z                          lát z−1      lát z       lát z+1
     ┌──┬──┬──┐                       ┌──┬──┬──┐  ┌──┬──┬──┐  ┌──┬──┬──┐
     ├──┼──┼──┤   9 ô                 ├──┼──┼──┤  ├──┼──┼──┤  ├──┼──┼──┤   27 ô
     ├──┼──┼──┤                       ├──┼──┼──┤  ├──┼──┼──┤  ├──┼──┼──┤
     └──┴──┴──┘                       └──┴──┴──┘  └──┴──┴──┘  └──┴──┴──┘
```

So sánh dễ nhớ: mô hình 2D giống **đọc từng trang sách riêng lẻ**; mô hình 3D giống **đọc cả đoạn liền nhau** và thấy mạch nối giữa các trang. Khối u là vật thể 3D: nhìn một lát có thể chưa chắc, nhưng nhìn thêm lát trên và lát dưới sẽ rõ hơn nhiều.

---

## Câu 31 — Mô hình 2D và 3D trong code khác nhau cụ thể ở đâu?

Hai bản dùng **cùng một code**, chỉ đổi `model.spatial_dims`. Các hàm ở `blocks.py:10-19` chọn `Conv2d`/`Conv3d`, `InstanceNorm2d`/`InstanceNorm3d`, `ConvTranspose2d`/`ConvTranspose3d` theo số chiều.

| Thành phần | 2D (`transunet2d.yaml`) | 3D (`transunet3d_small.yaml`) |
|---|---|---|
| Conv / Norm / Upsample | Conv2d 3×3 / InstanceNorm2d / ConvTranspose 2×2 | Conv3d 3×3×3 / InstanceNorm3d / ConvTranspose 2×2×2 |
| `features` | [32, 64, 128, 256, 320] | [16, 32, 64, 128, 256] |
| Patch / batch | 160×160 / 8 | 96×96×96 / 1 (+ `grad_accum: 2`) |
| Decoder hidden / FFN | 192 / 1536 | 96 / 384 |
| Mã hóa vị trí | sin/cos theo 2 trục | sin/cos theo 3 trục |
| Thu nhỏ mask cho attention | bilinear | trilinear |
| Dataset | `SliceStream2D` (lát axial) | `PatchStream3D` (khối) |
| Suy luận | từng lát rồi xếp chồng | cửa sổ trượt 96³, chồng 50%, trọng số Gauss |
| Số tham số | 8.6 M | 6.2 M |

**Shape đi qua TransUNet 3D thu nhỏ** (batch 1, patch 96³):

| Mức | Encoder | Feature decoder đưa vào Transformer | Số token |
|---|---|---|---|
| 1/1 | (1, 16, 96, 96, 96) | (1, 16, 96³) → pixel embedding (1, 96, 96³) | — (chỉ dùng để tính mask) |
| 1/2 | (1, 32, 48, 48, 48) | (1, 32, 48³) | 110 592 |
| 1/4 | (1, 64, 24, 24, 24) | (1, 64, 24³) | 13 824 |
| 1/8 | (1, 128, 12, 12, 12) | (1, 128, 12³) | 1 728 |
| 1/16 | (1, 256, 6, 6, 6) = bottleneck | — | — |

Output: `pred_logits` (1, 20, 4) và `pred_masks` (1, 20, 96, 96, 96).

So với bản 2D (400 / 1600 / 6400 token), bản 3D có nhiều token hơn hẳn, nhất là mức 1/2 (110 592 token). Cross-attention vẫn chịu được vì chi phí chỉ tỉ lệ 20 × L (Phần 1, Câu 7).

**Repo gốc 3D** (crop 128³, U-Net 6 mức [32, 64, 128, 256, 320, 320], 5 lần thu nhỏ): bottleneck 4³. Feature đưa vào decoder: 16³ (4 096 token), 32³ (32 768 token), 64³ (262 144 token). Mask output (2, 20, 128, 128, 128) với batch 2.

---

## Câu 32 — Train và suy luận 2D khác 3D thế nào?

**Khi train:**

- **2D:** mỗi mẫu là 1 lát axial (Câu 23). Một ca cho ra khoảng 140 lát khác nhau, nên có rất nhiều mẫu và dùng được batch lớn.
- **3D:** mỗi mẫu là 1 khối 96³, khoảng 1/4 volume đã crop. Batch 1, cộng dồn gradient 2 bước, nên batch hiệu dụng là 2 (bằng batch mỗi GPU trong paper).

**Khi suy luận** (`inference.py:18`):

- **2D:** lấy các lát có não, pad X và Y lên bội số của 16 (vì mạng thu nhỏ 4 lần, mỗi lần ÷2), chạy 16 lát một lượt, rồi **xếp chồng** xác suất lại thành (3, X, Y, Z).
- **3D:** dùng `sliding_window_inference` của MONAI: trượt khối 96³ với độ chồng 50%. Chỗ các cửa sổ chồng nhau được lấy **trung bình có trọng số Gauss** (tâm cửa sổ đáng tin hơn mép). Ví dụ volume 140×175×140 cần 2×3×2 = **12 cửa sổ**.
- Sau đó hai bản giống nhau: cắt ngưỡng → nhãn → tính Dice trên **cả volume 3D**. Kể cả bản 2D cũng được chấm trên khối đã xếp chồng, không chấm theo từng lát.

---

## Câu 33 — Ưu, nhược của 2D và 3D? Vì sao paper dùng 3D mà nhóm bắt đầu bằng 2D?

| | 2D | 3D |
|---|---|---|
| Ngữ cảnh theo trục z | **Không có** | Có |
| Tính liền mạch giữa các lát | Dễ "đứt đoạn", bề mặt u lởm chởm theo trục z | Liền mạch |
| U nhỏ chỉ xuất hiện ở vài lát | Dễ bỏ sót | Tốt hơn |
| Bộ nhớ, tốc độ | Nhẹ, nhanh | Nặng, chậm |
| Số mẫu mỗi batch | Nhiều | Ít (1–2) |
| Dùng được pretrain 2D (ImageNet) | Dễ | Khó |

**Kích thước một mẫu:** 160² = 25 600 pixel. 96³ = 884 736 voxel, **gấp khoảng 35 lần**. 128³ = 2 097 152 voxel, **gấp khoảng 82 lần**. Mọi feature map trong mạng tăng theo đúng tỉ lệ đó.

- VRAM đo được: TransUNet 2D với batch 8 chỉ tốn **0.84 GB**. Bản 3D gốc (128³, batch 2) ước tính cần **khoảng 24 GB** (docs/01). Paper ghi đã train với batch 2 trên **1 GPU NVIDIA RTX 8000** (48 GB).
- BraTS có voxel đẳng hướng 1 mm³, nên thông tin theo trục z tốt ngang theo x và y. Vì vậy 3D thường thắng 2D trên BraTS. Paper và nnU-Net đều dùng 3D.
- Nhóm có GPU 4 GB nên làm 2D trước (chạy được, thử nghiệm nhanh), rồi đến 3D thu nhỏ.
- Khi báo cáo: so sánh **trong cùng số chiều** (U-Net 2D với TransUNet 2D; U-Net 3D với TransUNet 3D). Không đem số của bản 2D so trực tiếp với số của paper.

---

## Câu 34 — "2.5D" là gì? Có nên thử?

2.5D là mô hình 2D nhưng input gồm **vài lát kề nhau**, ví dụ lát z−1, z, z+1 của cả 4 modality, thành 12 kênh. Model 2D nhờ vậy có thêm một chút ngữ cảnh theo trục z, trong khi chi phí gần như bản 2D.

Code hiện **chưa có** phần này. Muốn thử cần sửa `SliceStream2D._sample` (lấy thêm lát kề), `in_channels` của model và vòng lặp suy luận 2D. Đây là hướng mở rộng hợp lý nếu bản 3D quá chậm.

---

## Tóm tắt Phần 2

1. **1 ca** = 5 file NIfTI 240×240×155 (4 MRI + 1 nhãn). Nhãn {0, 1, 2, 4} được đổi thành **3 vùng lồng nhau** WT ⊃ TC ⊃ ET.
2. **Tiền xử lý:** đọc thẳng từ tar → crop vùng não → z-score trên voxel não → đổi nhãn sang vùng → cắt lát 2D (4, 160, 160) hoặc khối 3D (4, 96³), 50% mẫu có u → augmentation → batch.
3. **Dữ liệu rất mất cân bằng** (u ≈ 1% thể tích; 33 ca không có ET): dùng Dice, ép lấy mẫu có u, báo cả trung vị.
4. **2D** xử lý từng lát (nhẹ, nhanh, không có ngữ cảnh z). **3D** xử lý khối (nặng, có ngữ cảnh z). Cùng một code, chỉ đổi `spatial_dims`.
5. Chia dữ liệu **theo bệnh nhân**: 876 / 188 / 187 ca. Khác cách chia 5-fold của paper, nên chỉ so sánh tương đối.

→ Đọc tiếp **Phần 3**: [`07_HOI_DAP_P3_TRAIN_DANH_GIA.md`](07_HOI_DAP_P3_TRAIN_DANH_GIA.md)
