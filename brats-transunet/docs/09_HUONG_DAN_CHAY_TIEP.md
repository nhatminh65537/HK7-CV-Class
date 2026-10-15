# 09 — Hướng dẫn chạy tiếp: baseline 3D và các phương án cải tiến

> Viết ngày 14/10/2026, sau khi xong U-Net 2D, TransUNet 2D và U-Net 3D (baseline đã chốt ở docs/08 mục 6).
> Các phương án B, C, E và cách chúng được cài: docs/08 mục 3 và 8.2.
>
> Mọi lệnh chạy từ thư mục `brats-transunet/`, trong PowerShell. Đặt trước một lần cho mỗi cửa sổ:
>
> ```powershell
> $env:PYTHONUTF8 = "1"
> ```

## 1. Đã kiểm tra trước khi chạy (14/10)

| Kiểm tra | Kết quả |
|---|---|
| Code mới với mọi cờ tắt so với code cũ: output, loss, gradient (2D và 3D) | trùng hoàn toàn: baseline không bị ảnh hưởng |
| `scripts/check_variants.py`: lazy so với cách cũ | xác suất vùng lệch tối đa 2·10⁻⁷, loss trùng, gradient lệch 7·10⁻⁷ so với gradient lớn nhất |
| `scripts/check_variants.py`: fixed và lite | chạy được, đúng shape, loss hữu hạn |

## 2. Kết quả baseline đến giờ

Test 187 ca, `best.pt`, quy ước Dice của BraTS. TB = trung bình, TV = trung vị.

| Lượt | WT (TB / TV) | TC | ET | Trung bình 3 vùng | HD95 WT (TV) | Ca < 0.5 |
|---|---|---|---|---|---|---|
| U-Net 2D | 0.911 / 0.941 | 0.873 / 0.947 | 0.830 / 0.896 | 0.871 / 0.918 | 3.6 mm | 8 |
| TransUNet 2D | 0.912 / 0.937 | 0.873 / 0.936 | 0.842 / 0.902 | 0.876 / 0.916 | 5.7 mm | 5 |

TransUNet − U-Net trên từng ca (`scripts/compare_runs.py`):

| Vùng | Chênh (điểm) | Khoảng tin cậy 95% | Paper (3D, 5-fold) |
|---|---|---|---|
| WT | +0.12 | −0.14 đến +0.39 | +0.11 |
| TC | 0.00 | −1.07 đến +1.07 | +0.56 |
| ET | +1.17 | −0.15 đến +3.01 | +0.80 |
| Trung bình | +0.43 | −0.12 đến +1.08 | +0.49 |

Đọc bảng:

- **Cùng xu hướng với paper:** chênh trung bình gần bằng (+0.43 so với +0.49), lợi nhiều nhất ở ET.
- **Chưa có ý nghĩa thống kê:** mọi khoảng tin cậy đều chứa 0. Paper cũng chỉ báo mức chênh nhỏ như vậy, và các benchmark độc lập cho thấy BraTS2021 đã bão hòa (docs/08 mục 2.1).
- **Hai lượt đầu (140 ca, 20 epoch)** cho TransUNet kém U-Net 5 điểm. Sau khi sửa baseline thì hết, nghĩa là các sửa đổi ở docs/08 mục 6 là cần thiết.
- **U-Net 3D chưa đánh giá trên test.** Val 30 ca: 0.847, gần bằng 2D (U-Net 2D 0.844 trên cùng 30 ca).

## 3. Thứ tự chạy

> **Cập nhật 15/10:** phần 2D (bước 1–5) đã xong. Phần 3D chạy theo **mục 7.3**, thay cho bước 6–7 dưới đây.

Ước lượng thời gian khi máy ở chế độ High performance: U-Net 3D đo được 262 giây/epoch ở chế độ này (675 giây ở chế độ thường). Các lượt 2D vừa rồi chạy ở chế độ thường (190 và 486 giây/epoch); log tháng 9 ở chế độ nhanh là 78 và 203 giây/epoch. Các ước lượng **chưa tính thời gian validation**.

| Bước | Việc | Train? | Ước lượng |
|---|---|---|---|
| 1 | Đánh giá U-Net 3D trên test | không | vài chục phút |
| 2 | Đo chi phí mọi phương án (`benchmark.py`) | không | khoảng 10–20 phút |
| 3 | B-2D: đánh giá lại TransUNet 2D với mask lười | không | như một lần đánh giá |
| 4 | C-2D: 3 query cố định | có | khoảng 2.5 giờ |
| 5 | E-2D: C + decoder gọn | có | khoảng 2.5 giờ |
| 6 | A-3D: TransUNet 3D baseline, rồi B-3D | có | khoảng 9 giờ, nên chạy qua đêm |
| 7 | Phương án tốt nhất ở 3D (C hoặc E) | có | khoảng 7 giờ |

Bước 6 có thể chạy trước bước 4–5 nếu muốn có đủ baseline sớm.

### Bước 1. Đánh giá U-Net 3D

```powershell
python evaluate.py --ckpt runs/unet3d_small/best.pt --split test --hd95
python scripts/compare_runs.py runs/unet2d/eval_test runs/unet3d_small/eval_test
```

### Bước 2. Đo chi phí (GPU phải rảnh, không train song song)

```powershell
python scripts/benchmark.py --configs configs/unet2d.yaml configs/transunet2d.yaml configs/transunet2d_lazy.yaml configs/transunet2d_fixed.yaml configs/transunet2d_lite.yaml --csv figures/bench_2d.csv
python scripts/benchmark.py --configs configs/unet3d_small.yaml configs/transunet3d_small.yaml configs/transunet3d_small_lazy.yaml configs/transunet3d_small_fixed.yaml configs/transunet3d_small_lite.yaml --csv figures/bench_3d.csv
python scripts/benchmark.py --configs configs/unet3d_small.yaml configs/transunet3d_small.yaml configs/transunet3d_small_lazy.yaml configs/transunet3d_small_fixed.yaml --opts "train.patch_size=[128,128,128]" --csv figures/bench_3d_128.csv
```

Dòng thứ ba là kích thước khối của paper. Baseline có thể báo OOM hoặc rất chậm vì tràn VRAM; đó chính là con số cần cho báo cáo.

### Bước 3. Phương án B ở 2D (không train)

```powershell
python evaluate.py --ckpt runs/transunet2d/best.pt --split test --hd95 --opts model.decoder.lazy_mask=true --tag lazy
python scripts/compare_runs.py runs/transunet2d/eval_test runs/transunet2d/eval_test_lazy
```

Dice phải trùng (chênh 0.00). Nếu khác thì dừng lại và báo lỗi. Lần đánh giá baseline cũ chưa có cột thời gian, nên muốn so giây/ca thì đánh giá lại baseline với `--tag timed`.

### Bước 4–5. Phương án C và E ở 2D

```powershell
python train.py --config configs/transunet2d_fixed.yaml
python evaluate.py --ckpt runs/transunet2d_fixed/best.pt --split test --hd95

python train.py --config configs/transunet2d_lite.yaml
python evaluate.py --ckpt runs/transunet2d_lite/best.pt --split test --hd95

python scripts/compare_runs.py runs/transunet2d/eval_test runs/transunet2d_fixed/eval_test runs/transunet2d_lite/eval_test
python scripts/compare_runs.py runs/unet2d/eval_test runs/transunet2d/eval_test runs/transunet2d_fixed/eval_test runs/transunet2d_lite/eval_test
```

### Bước 6. Baseline A-3D, rồi B-3D

```powershell
python train.py --config configs/transunet3d_small.yaml
python evaluate.py --ckpt runs/transunet3d_small/best.pt --split test --hd95
python evaluate.py --ckpt runs/transunet3d_small/best.pt --split test --hd95 --opts model.decoder.lazy_mask=true --tag lazy
python scripts/compare_runs.py runs/unet3d_small/eval_test runs/transunet3d_small/eval_test
```

### Bước 7. Phương án tốt nhất ở 3D

Chọn theo kết quả 2D (mục 4). Ví dụ với C:

```powershell
python train.py --config configs/transunet3d_small_fixed.yaml
python evaluate.py --ckpt runs/transunet3d_small_fixed/best.pt --split test --hd95
python scripts/compare_runs.py runs/transunet3d_small/eval_test runs/transunet3d_small_fixed/eval_test
```

Bị ngắt giữa chừng thì chạy lại đúng lệnh `train.py` kèm `--resume`.

## 4. Đọc kết quả và quyết định

| Kết quả của C so với A (2D) | Làm tiếp |
|---|---|
| Không kém (khoảng tin cậy chứa 0 hoặc dương) | Đó là kết quả cần: chi phí thấp hơn, Dice không kém. Chạy E, rồi đưa phương án tốt hơn sang 3D |
| Kém rõ (khoảng tin cậy nằm hẳn dưới 0) | Báo cáo trung thực là bỏ Hungarian làm giảm Dice. Vẫn còn B (không đổi Dice) để báo cáo hiệu năng. Có thể thử E trên nền A: `--opts model.decoder.ffn_mult=2 model.decoder.shared_layers=true output_dir=runs/transunet2d_slim` |
| Chênh nhỏ nhưng chưa rõ chiều | Thêm một seed cho cả A và C: `--opts seed=43 output_dir=runs/<tên>_s43` |

**Tiêu chí "cải thiện hơn paper"** (docs/08 mục 7): Dice không kém baseline, **và** ít nhất một trong tham số, VRAM, thời gian giảm rõ. Không so Dice tuyệt đối với paper.

## 5. Về tần suất validation

Hiện tại validation 30 ca mỗi 5 epoch (2D) và mỗi 10 epoch (3D).

- **Không nên đổi giữa chừng.** Baseline đã chọn `best.pt` theo đúng tần suất này. Phương án nào chọn checkpoint thường xuyên hơn sẽ được lợi một chút nhờ "chọn trúng", nên so sẽ không công bằng.
- **Lợi ích của eval dày hơn nhỏ.** Lịch học cosine giảm learning rate về gần 0 ở cuối, nên các epoch cuối gần như như nhau. Ở lượt 2D vừa rồi, `best.pt` (epoch 44) và epoch 49 chênh chưa tới 0.001.
- **Chọn trên 30 ca nhiều lần còn gây thiên lệch.** Càng nhiều lần thì càng dễ chọn trúng checkpoint "may mắn" trên 30 ca đó. Con số báo cáo vẫn là test 187 ca, nên không bị ảnh hưởng.

Nếu vẫn muốn eval dày hơn, áp dụng cho **mọi** lượt sẽ so sánh: `--opts eval.every=2`. Các lượt đã xong thì có thể báo thêm kết quả của `last.pt` để kiểm tra.

## 6. Kết quả 2D và nhận xét (15/10)

Test 187 ca, `best.pt`. Chênh lệch tính trên từng ca bằng `scripts/compare_runs.py`; "CI" là khoảng tin cậy 95%.

### 6.1 Độ chính xác

| Lượt | WT | TC | ET | Trung bình (TB / TV) | HD95 WT (TV) | Ca < 0.5 |
|---|---|---|---|---|---|---|
| U-Net 2D | 0.911 | 0.873 | 0.830 | 0.871 / 0.918 | 3.6 mm | 8 |
| A: TransUNet 2D | 0.912 | 0.873 | 0.842 | 0.876 / 0.916 | 5.7 mm | 5 |
| B: A + mask lười | 0.912 | 0.873 | 0.842 | 0.876 / 0.916 | 5.7 mm | 5 |
| C: 3 query cố định | 0.911 | 0.881 | 0.834 | 0.875 / 0.924 | 4.2 mm | 6 |
| E: C + decoder gọn | 0.911 | 0.880 | 0.833 | 0.875 / 0.925 | 4.4 mm | 7 |
| U-Net 3D thu nhỏ | 0.896 | 0.872 | 0.823 | 0.864 / 0.917 | 4.6 mm | 8 |

| So sánh (trung bình 3 vùng, điểm) | Chênh | CI 95% | Kết luận |
|---|---|---|---|
| A − U-Net 2D | +0.43 | −0.12 đến +1.08 | chưa phân biệt |
| B − A | 0.00 | | trùng, đúng như thiết kế |
| C − A | −0.03 | −0.51 đến +0.40 | không kém A (sát ngưỡng ±0.5) |
| E − A | −0.09 | −0.71 đến +0.45 | chưa kết luận được là không kém |
| C − U-Net 2D | +0.40 | +0.05 đến +0.79 | **hơn**, p = 0.004 |
| U-Net 3D − U-Net 2D | −0.76 | −1.37 đến −0.18 | **kém** |

### 6.2 Chi phí (`figures/bench_2d.csv`)

Đo trong cùng một lần chạy; máy có vẻ ở chế độ nguồn thường (U-Net 0.74 s/bước, so với 0.31 ở chế độ nhanh), nên chỉ dùng tỉ lệ.

| Lượt | Tham số | Bước train | VRAM train | Suy luận 16 lát | VRAM suy luận |
|---|---|---|---|---|---|
| U-Net 2D | 5.69M | 0.74 s | 0.37 GB | 0.38 s | 0.30 GB |
| A | 8.57M | 1.98 s | 0.84 GB | 0.99 s | 0.77 GB |
| B | 8.57M | 1.78 s (−10%) | 0.68 GB (−19%) | 0.82 s (−17%) | 0.60 GB (−23%) |
| C | 8.56M | 1.34 s (−33%) | 0.64 GB (−23%) | 0.75 s (−24%) | 0.53 GB (−31%) |
| E | 6.34M (−26%) | 1.31 s (−34%) | 0.61 GB (−27%) | 0.75 s (−24%) | 0.51 GB (−35%) |

Phần trăm so với A. Cột `sec` của `evaluate.py` (giây/ca) **không** dùng để so, vì các lần đánh giá chạy ở chế độ nguồn khác nhau (ví dụ E chậm hơn C ở cột đó, trái với benchmark).

### 6.3 Nhận xét

- **Chênh lệch nhỏ là thật, không phải lỗi.** Trung vị của mọi model đều 0.92–0.95; chênh lệch trung bình bị chi phối bởi một nhóm nhỏ ca khó. 11 ca có Dice < 0.5 ở ít nhất một model; 2 ca hỏng ở cả 5 model.
- **Lợi thế ET của A gần như đến từ 2 ca.** Ở `BraTS2021_01502` và `01509`, A dự đoán ET rỗng và được Dice ET = 1.0, các model khác dự đoán có ET và bị 0.0 (rất có thể nhãn thật không có ET). Mỗi lần đảo 0 ↔ 1 như vậy làm ET trung bình đổi 0.53 điểm. Bỏ 5 ca chênh nhiều nhất thì A − U-Net còn −0.14, trung vị chỉ +0.04. Ngược lại, lợi thế của C so với U-Net trải đều hơn: hơn ở 108/187 ca, bỏ 5 ca chênh nhất vẫn còn +0.29.
- **Hậu xử lý ET (ET quá ít voxel thì xoá) sẽ tác động đúng vào các ca này** và có thể xoá lợi thế ET của A. Nếu làm, phải chọn ngưỡng trên val và áp cùng ngưỡng cho mọi model.
- **Train chưa đủ epoch? Ở 2D thì phần lớn là không.** Lịch cosine đưa learning rate về gần 0; Dice val 10 epoch cuối chỉ tăng 0.001–0.005 ở mọi model. Train dài hơn có thể nâng tất cả lên một chút, nhưng không có bằng chứng là khoảng cách sẽ giãn ra (paper train đầy đủ cũng chỉ chênh 0.49).
- **Ở 3D thì có thể có.** U-Net 3D thu nhỏ kém U-Net 2D, ngược với kỳ vọng. Cấu hình 3D bị bó rất chặt: 16 kênh đầu, batch 1, khối 96³, chỉ 500 khối mỗi epoch. Đây là giới hạn cần nêu rõ trong báo cáo, không phải kết luận rằng 3D kém 2D.
- **Kết quả đáng báo cáo nhất là C:** Dice không kém A, hơn U-Net có ý nghĩa thống kê, train nhanh hơn khoảng 1/3, suy luận nhanh hơn khoảng 1/4, VRAM ít hơn khoảng 1/4. HD95 trung vị cũng tốt hơn A (4.2 so với 5.7 mm).
- **E chỉ thêm giảm tham số** (−26%), nhưng chưa chứng minh được không kém A. Ở 3D, E chỉ giảm khoảng 6% tham số (docs/08 mục 8), nên không đáng ưu tiên.

## 7. Dao động khi train và kế hoạch 3D (15/10)

### 7.1 Dao động của C và E

Dice val (30 ca) qua 10 lần đánh giá (`log.csv`):

| Lượt | Lần tụt lớn nhất | Chênh giữa 3 lần đánh giá cuối |
|---|---|---|
| U-Net 2D | không tụt | 0.003 |
| A | −0.001 | 0.002 |
| C | −0.007 (epoch 24) | 0.005 |
| E | **−0.075** (epoch 14), rồi hồi lại | 0.009 |

- **C dao động trong mức nhiễu.** Với 30 ca val, chỉ một ca không có ET bị đảo 0 ↔ 1 đã làm Dice trung bình đổi khoảng 0.011. Mức tụt 0.007 nhỏ hơn thế.
- **E có một lần tụt thật**, giữa lúc learning rate còn cao. **Giả thuyết:** với `shared_layers`, cùng một bộ trọng số được dùng 3 lần, nên gradient của 3 tầng cộng dồn vào đó; bước cập nhật của decoder vì vậy lớn hơn khoảng 3 lần so với các lượt khác.
- **Lúc kết thúc mọi lượt đều ổn định**, vì lịch cosine đưa learning rate về gần 0. Kết quả test ở mục 6 không bị ảnh hưởng bởi các lần tụt giữa chừng.
- Muốn đo đúng "dao động giữa các lần train" thì phải chạy lại cùng cấu hình với seed khác (mục 7.3), không phải nhìn đường val.

### 7.2 Hướng đi cho 3D

| Phương án | Ở 3D | Lý do |
|---|---|---|
| A (baseline) | **train**, khối 96³ | mốc bắt buộc |
| B (mask lười) | **chỉ đánh giá lại** checkpoint A | không đổi trọng số; ở 3D mới thấy rõ lợi ích VRAM |
| C (3 query cố định) | **train**, khối 96³ | kết quả tốt nhất ở 2D: Dice không kém A, rẻ hơn rõ |
| E (decoder gọn) | **chỉ đo benchmark**, không train | ở 3D chỉ giảm khoảng 6% tham số; ở 2D chưa chứng minh được không kém A và có một lần tụt mạnh; batch 1 ở 3D còn dễ dao động hơn |
| Train ở khối 128³ | **không train, chỉ đo benchmark** | baseline không vừa 4 GB nên không có cặp so sánh công bằng; mỗi lượt sẽ mất khoảng một ngày |

Câu báo cáo cho 128³ dựa trên benchmark: "ở kích thước khối của paper, baseline tràn VRAM (X GB, Y s/bước) còn C vừa (Z GB, W s/bước)". Đây là số đo hiệu năng, không phải số Dice.

C-3D giữ đúng mọi thiết lập của A-3D (50 epoch, val 30 ca mỗi 10 epoch, seed 42); chỉ khác phần decoder.

### 7.3 Lệnh cần chạy, theo thứ tự

Bật High performance, không chạy gì khác trên GPU. Từ `brats-transunet/`:

```powershell
$env:PYTHONUTF8 = "1"

# Bước 1 — benchmark, cùng một phiên (khoảng 20–30 phút; dòng 128³ của baseline có thể chậm hoặc OOM, đó là kết quả cần có)
python scripts/benchmark.py --configs configs/unet2d.yaml configs/transunet2d.yaml configs/transunet2d_lazy.yaml configs/transunet2d_fixed.yaml configs/transunet2d_lite.yaml --csv figures/bench_2d.csv
python scripts/benchmark.py --configs configs/unet3d_small.yaml configs/transunet3d_small.yaml configs/transunet3d_small_lazy.yaml configs/transunet3d_small_fixed.yaml configs/transunet3d_small_lite.yaml --csv figures/bench_3d.csv
python scripts/benchmark.py --configs configs/unet3d_small.yaml configs/transunet3d_small.yaml configs/transunet3d_small_lazy.yaml configs/transunet3d_small_fixed.yaml --opts "train.patch_size=[128,128,128]" --csv figures/bench_3d_128.csv

# Bước 2 — A-3D (khoảng 9 giờ), rồi đánh giá thường và đánh giá lazy (B-3D)
python train.py --config configs/transunet3d_small.yaml
python evaluate.py --ckpt runs/transunet3d_small/best.pt --split test --hd95
python evaluate.py --ckpt runs/transunet3d_small/best.pt --split test --hd95 --opts model.decoder.lazy_mask=true --tag lazy

# Bước 3 — C-3D (khoảng 6–7 giờ)
python train.py --config configs/transunet3d_small_fixed.yaml
python evaluate.py --ckpt runs/transunet3d_small_fixed/best.pt --split test --hd95

# Bước 4 — so sánh
python scripts/compare_runs.py runs/transunet3d_small/eval_test runs/transunet3d_small/eval_test_lazy
python scripts/compare_runs.py runs/unet3d_small/eval_test runs/transunet3d_small/eval_test runs/transunet3d_small_fixed/eval_test
python scripts/compare_runs.py runs/transunet3d_small/eval_test runs/transunet3d_small_fixed/eval_test
```

Bị ngắt giữa chừng thì chạy lại đúng lệnh `train.py` kèm `--resume`. Dòng `B-3D so với A-3D` phải cho chênh 0.00; nếu khác thì dừng lại và báo lỗi.

**Tùy chọn, nếu còn thời gian sau bước 4** — seed thứ hai cho A và C ở 2D (khoảng 5 giờ), để đo dao động giữa các lần train và củng cố kết luận "C không kém A":

```powershell
python train.py --config configs/transunet2d.yaml --opts seed=43 output_dir=runs/transunet2d_s43
python evaluate.py --ckpt runs/transunet2d_s43/best.pt --split test --hd95
python train.py --config configs/transunet2d_fixed.yaml --opts seed=43 output_dir=runs/transunet2d_fixed_s43
python evaluate.py --ckpt runs/transunet2d_fixed_s43/best.pt --split test --hd95
python scripts/compare_runs.py runs/transunet2d/eval_test runs/transunet2d_s43/eval_test runs/transunet2d_fixed/eval_test runs/transunet2d_fixed_s43/eval_test
```

Hậu xử lý ET (chọn ngưỡng trên val, áp cho mọi model) không cần train, làm song song được trong lúc chờ; cần thêm code.

## 8. Hậu xử lý ET (15/10)

**Là gì.** Sau khi cắt ngưỡng, nếu tổng số voxel ET dự đoán của cả ca nhỏ hơn một ngưỡng N thì coi như ca đó không có ET: đổi các voxel ET đó sang NCR, tức là vẫn giữ trong TC và WT. Mẹo này do nnU-Net dùng khi thắng BraTS 2020; N chọn trên val.

**Vì sao có thể giúp.** Theo quy ước BraTS, ca không có ET mà model dự đoán dù chỉ vài voxel ET thì Dice ET = 0; dự đoán rỗng thì Dice ET = 1. Mỗi ca như vậy làm ET trung bình đổi 0.53 điểm (1/187).

**Số liệu trên test của nhóm** (đọc nhãn thật của 187 ca):

- 5 ca không có ET: `01499`, `01502`, `01509`, `01515`, `01666`. A đoán đúng 2 ca (`01502`, `01509`); mọi model khác sai cả 5. Đây chính là nguồn lợi thế ET của A ở mục 6.
- 5 ca có ET rất nhỏ (22–159 voxel). Model nào cũng chỉ đạt Dice ET thấp ở các ca này (phần lớn dưới 0.6), nên xoá ET ở đó thường mất ít.
- **Mức lợi tối đa** nếu sửa đúng mọi ca không có ET: +2.67 điểm ET (+0.89 điểm trung bình 3 vùng) cho U-Net 2D, C, E, U-Net 3D; +1.60 ET (+0.53) cho A. Mức thật thấp hơn: ca nào dự đoán thừa nhiều voxel ET hơn ngưỡng thì không sửa được, và ngưỡng có thể xoá nhầm ET thật ở ca nhỏ.

**Ý nghĩa cho báo cáo.**

- Đây là mẹo đã biết, không phải cải tiến mới. Code gốc của paper không dùng.
- Phải chọn N trên val và áp **cùng N cho mọi model**. Kết quả báo cả có và không có hậu xử lý.
- Nhiều khả năng nó xoá lợi thế ET "may mắn" của A, và làm so sánh C với A công bằng hơn.

**Trạng thái:** chưa cài. Cần `evaluate.py` ghi thêm số voxel ET dự đoán của từng ca, rồi một script chọn N trên val. Không cần train lại.

## 9. Train tiếp từ epoch 50

**Làm được**, bằng `--resume` và đặt số epoch mới. Nhưng cần hiểu ba điều:

1. **Đây là "khởi động lại" learning rate, không phải train 70 epoch từ đầu.** Lịch cosine được tính lại theo tổng số epoch mới. Ở epoch 50/70, learning rate nhảy từ 1·10⁻⁶ lên khoảng 6·10⁻⁵ rồi lại giảm về 0. Dice val có thể tụt nhẹ vài epoch đầu rồi mới lên.
2. **Phải làm giống hệt cho mọi lượt đem so** (U-Net, A, C), nếu không so sánh sẽ không công bằng.
3. **Phải chép thư mục run sang tên mới trước**, vì `--resume` ghi đè `last.pt`, `best.pt` và ghi tiếp `log.csv`. Giữ nguyên kết quả 50 epoch làm mốc.

Mức lợi dự kiến nhỏ: ở 2D, Dice val 10 epoch cuối chỉ tăng 0.001–0.005. Ở 3D có thể lợi hơn vì model chưa bão hòa. Nên làm **sau** A-3D và C-3D, và báo như một thí nghiệm riêng.

Ví dụ thêm 20 epoch cho U-Net 2D (làm tương tự cho mọi lượt cần so):

```powershell
Copy-Item -Recurse runs/unet2d runs/unet2d_e70
Remove-Item -Recurse runs/unet2d_e70/eval_test
python train.py --config configs/unet2d.yaml --resume --opts output_dir=runs/unet2d_e70 train.epochs=70
python evaluate.py --ckpt runs/unet2d_e70/best.pt --split test --hd95
python scripts/compare_runs.py runs/unet2d/eval_test runs/unet2d_e70/eval_test
```
