# 07 — Hỏi đáp (Phần 3/3): Train, đánh giá, so với paper, chuẩn bị báo cáo

> **Bộ hỏi đáp gồm 3 phần:**
>
> 1. Phần 1 — Mô hình: [`05_HOI_DAP_P1_MO_HINH.md`](05_HOI_DAP_P1_MO_HINH.md)
> 2. Phần 2 — Dữ liệu, tiền xử lý, 2D và 3D: [`06_HOI_DAP_P2_DU_LIEU_2D_3D.md`](06_HOI_DAP_P2_DU_LIEU_2D_3D.md)
> 3. **Phần 3 — Train, đánh giá, so với paper, chuẩn bị báo cáo** (file này)
>
> Quy ước về shape, đường dẫn và số dòng giống Phần 1.

## Mục lục Phần 3

**Train và đánh giá**

- Câu 35 — Vòng lặp train chạy thế nào?
- Câu 36 — Validation trong lúc train và chọn checkpoint
- Câu 37 — Dice và HD95 tính thế nào? Quy ước BraTS?
- Câu 38 — Hậu xử lý: từ xác suất ra file nhãn
- Câu 39 — Tại sao sigmoid 3 kênh mà không softmax 4 lớp?

**So với paper và repo gốc**

- Câu 40 — Bản của nhóm khác repo gốc ở những điểm nào?
- Câu 41 — Kết quả paper trên BraTS2021 là gì? So với kết quả của nhóm thế nào cho đúng?
- Câu 42 — Vì sao TransUNet 2D (lần train đầu) bị tụt TC? Nên thử gì?
- Câu 43 — Nên làm ablation nào? Đọc kết quả ra sao?

**Chuẩn bị báo cáo**

- Câu 44 — Các câu hỏi nhanh giảng viên hay hỏi
- Câu 45 — Tóm tắt 1 phút để nói khi báo cáo
- Phụ lục — Tra nhanh: khái niệm ↔ file code

---

# A. TRAIN VÀ ĐÁNH GIÁ

## Câu 35 — Vòng lặp train chạy thế nào?

Code ở `train.py:105-138`. Mỗi **epoch = 250 bước** (`train.iters_per_epoch`), **không** phải "duyệt hết dữ liệu một lượt". Đây là quy ước của nnU-Net: dữ liệu được lấy mẫu ngẫu nhiên liên tục nên khái niệm "hết dữ liệu" không rõ ràng; cố định số bước giúp các epoch so sánh được với nhau.

Mỗi bước:

```
lr ← lr_at(bước hiện tại)                       # warmup rồi cosine
lặp grad_accum lần:
    batch ← next(loader)                        # (8,4,160,160), (8,3,160,160)
    trong autocast fp16: out ← model(x)         # AMP: tính bằng số thực 16-bit
    loss ← crit(out, y) / grad_accum
    scaler.scale(loss).backward()               # GradScaler: nhân loss lên để gradient fp16 không bị tràn về 0
scaler.unscale_(opt); clip_grad_norm_(…, 12)    # cắt gradient quá lớn (giống code gốc)
scaler.step(opt); scaler.update(); opt.zero_grad()
```

Cuối mỗi epoch: ghi `log.csv`, lưu `last.pt`. Cứ `eval.every = 5` epoch thì chạy validation (Câu 36).

| Thiết lập | Nhóm | Paper / repo gốc (BraTS) | nnU-Net gốc |
|---|---|---|---|
| Optimizer | AdamW, lr 3e-4, weight decay 3e-5 | AdamW, lr 3e-4 (paper); weight decay 3e-5 (code) | SGD Nesterov, lr 0.01, momentum 0.99 |
| Lịch learning rate | warmup tuyến tính 2 epoch → cosine xuống 1e-6, cập nhật **mỗi bước** | cosine (paper); code: warmup 10 epoch → cosine, cập nhật mỗi epoch | poly |
| Số epoch | 100 (mặc định; lần chạy thử đầu 20) | paper không ghi; config repo: 125, chú thích "dùng 8 GPU" | 1000 |
| Batch | 8 lát (2D) / 1 khối × 2 bước cộng dồn (3D) | 2 khối 128³ mỗi GPU; paper ghi train trên 1 GPU RTX 8000 | 2 |

Công thức learning rate (`train.py:35-39`):

- Khi bước < số bước warmup (2 epoch = 500 bước): `lr = 3e-4 · (bước + 1) / 500`.
- Sau đó: `lr = 1e-6 + 0.5 · (3e-4 − 1e-6) · (1 + cos(π·t))`, với t đi từ 0 đến 1 trong phần còn lại.

**Vì sao AdamW và warmup?** Transformer nhạy với learning rate lúc đầu: trọng số attention còn ngẫu nhiên, và các ước lượng moment của Adam chưa ổn định. Warmup tăng dần learning rate trong 500 bước đầu để tránh "nổ" loss. AdamW tách weight decay ra khỏi gradient ("decoupled weight decay") và là lựa chọn chuẩn khi train Transformer.

**Các cơ chế phụ:**

- **AMP** (mixed precision): tính phần lớn phép toán bằng fp16, gần như giảm nửa VRAM và nhanh hơn.
- **Grad clipping 12**: nếu độ lớn gradient vượt 12 thì thu nhỏ lại, tránh một bước cập nhật quá mạnh.
- **Gradient accumulation**: bản 3D chỉ vừa batch 1, nên cộng dồn gradient 2 bước rồi mới cập nhật, tương đương batch 2.
- **Resume**: `--resume` nạp `last.pt` (trọng số, optimizer, scaler, epoch). Ctrl+C lưu checkpoint trước khi thoát.

---

## Câu 36 — Validation trong lúc train và chọn checkpoint

- Mỗi 5 epoch, code chạy suy luận **trên cả volume** cho `eval.val_cases = 10` ca đầu của tập val, tính Dice WT/TC/ET và `dice_mean` (`train.py:42-48`).
- `dice_mean` cao nhất được lưu thành `best.pt`. Mọi epoch đều lưu `last.pt` để có thể chạy tiếp.
- Chỉ dùng 10 ca để tiết kiệm thời gian. Vì ít ca nên điểm val "ồn", `best.pt` có thể được chọn nhầm. Đánh giá đầy đủ dùng `evaluate.py` trên toàn bộ val hoặc test.
- Mọi quyết định (checkpoint, ngưỡng, siêu tham số) phải chọn trên **val**, không bao giờ chọn trên **test**. Test chỉ chạy một lần ở cuối để báo cáo.

---

## Câu 37 — Dice và HD95 tính thế nào? Quy ước BraTS?

**Dice** (`metrics.py:9-15`):

```
Dice = 2·∣P ∩ G∣ / (∣P∣ + ∣G∣)
```

Ví dụ: vùng thật 1000 voxel, dự đoán 900 voxel, phần trùng 800 voxel → Dice = 1600 / 1900 ≈ 0.842.

- Tính trên **cả volume** của từng ca, riêng cho WT, TC, ET, rồi lấy trung bình trên các ca.
- **Quy ước khi vùng thật rỗng** (theo BraTS): dự đoán cũng rỗng → Dice = 1.0; dự đoán có voxel → Dice = 0.0.
- "Dice score" dùng để đánh giá, tính trên mask đã cắt ngưỡng (0/1). "Dice loss" dùng khi train, tính trên xác suất (Dice mềm).
- `measure_dice.py` của repo gốc trả **0 khi dự đoán hoặc nhãn thật rỗng**, kể cả khi cả hai đều rỗng (`measure_dice.py:39-48`). Cách này cho điểm thấp hơn ở các ca không có ET. Khi so với số của repo gốc phải nói rõ đang dùng quy ước nào.

**HD95** (`metrics.py:18-26`, bật bằng `evaluate.py --hd95`): đo khoảng cách giữa **biên** vùng dự đoán và **biên** vùng thật, đơn vị mm (voxel 1 mm). Khoảng cách Hausdorff thường lấy điểm xa nhất; HD95 lấy **phân vị 95%** để vài điểm lạc không chi phối kết quả. Càng nhỏ càng tốt. Dice đo mức "trùng thể tích", HD95 đo mức "lệch biên".

- Code nhóm trả `nan` khi dự đoán hoặc nhãn thật rỗng, rồi bỏ qua các giá trị `nan` khi lấy trung bình. Như vậy **lạc quan** hơn bộ chấm chính thức của BraTS, vốn gán một giá trị phạt rất lớn (khoảng 373 mm) khi nhãn thật có u mà dự đoán rỗng. Nên ghi chú điều này khi báo cáo HD95.

---

## Câu 38 — Hậu xử lý: từ xác suất ra file nhãn

Trình tự: `binarize` (`inference.py:48-58`) → `regions_to_seg` → `uncrop` → lưu NIfTI (`evaluate.py:58-60`).

```python
p[1] = max(p[1], p[2])    # P(TC) ≥ P(ET)
p[0] = max(p[0], p[1])    # P(WT) ≥ P(TC)
mask = p > 0.5            # nhờ 2 dòng trên, luôn có ET ⊂ TC ⊂ WT
```

- **Vì sao "max lũy tiến" thay vì cắt ngưỡng rồi lấy giao?** Giả sử model chắc về ET (0.8) nhưng lưỡng lự về TC (0.4). Cắt ngưỡng rồi lấy giao sẽ **xoá cả TC lẫn ET** tại đó. Max lũy tiến giữ lại được cả hai, với TC ⊇ ET. Thử trên 2 ca mẫu: Dice trung bình tăng từ 0.683 lên 0.707 (docs/02). Hạ ngưỡng xuống 0.3 thì kết quả tệ hơn.
- **Đổi vùng về nhãn:** chỉ thuộc WT → 2 (ED); thuộc TC nhưng không thuộc ET → 1 (NCR); thuộc ET → 4.
- **`uncrop`:** đặt kết quả trở lại khối 240×240×155 bằng bbox đã lưu, rồi lưu với affine gốc. File dự đoán mở chồng được lên ảnh MRI trong ITK-SNAP.
- **Repo gốc:** cắt ngưỡng 0.5 cho từng vùng rồi gán nhãn theo thứ tự WT → TC → ET, vùng sau ghi đè vùng trước (`inference.py:478-485` của repo gốc), không có bước lấy max.
- **Hậu xử lý có thể thử thêm** (chưa làm, phải chọn tham số trên val):
  - Xoá các mảnh nhỏ rời rạc (connected components nhỏ).
  - Nếu ET dự đoán quá ít voxel thì đổi phần đó sang NCR. Đây là mẹo phổ biến trong BraTS để giảm lỗi ở các ca không có ET.
  - Với TransUNet: với mỗi lớp, chỉ lấy query có P(lớp) cao nhất và dùng thẳng mask của nó (không nhân với xác suất lớp). Cách này nhắm đúng vào lỗi "xác suất lớp thấp kéo cả vùng xuống" (Câu 42).

---

## Câu 39 — Tại sao sigmoid 3 kênh mà không softmax 4 lớp?

- **Softmax 4 lớp** (nền, NCR, ED, ET): mỗi voxel thuộc đúng một lớp, rồi suy ra WT/TC/ET bằng cách gộp lớp. Loss khi đó tối ưu từng nhãn riêng, không trực tiếp tối ưu 3 vùng được chấm điểm.
- **Sigmoid 3 kênh** (WT, TC, ET): mỗi kênh là một bài toán nhị phân độc lập, nên các vùng được phép **chồng lên nhau**, đúng với bản chất lồng nhau. Dice loss tính thẳng trên vùng được chấm. nnU-Net cho BraTS và repo gốc (chế độ `500Region`) đều làm vậy.
- **Với TransUNet:** lớp của query là {WT, TC, ET, không có gì} (softmax trên **lớp của query**), còn **mask** của mỗi query là sigmoid. Vì vậy 3 vùng vẫn chồng lên nhau một cách tự nhiên.

---

# B. SO VỚI PAPER VÀ REPO GỐC

## Câu 40 — Bản của nhóm khác repo gốc ở những điểm nào?

So với cấu hình BraTS Decoder-only của repo gốc (`configs/Brats/decoder_only.yaml` + trainer nnU-Net):

| Hạng mục | Repo gốc | Bản của nhóm | Ghi chú |
|---|---|---|---|
| Nền tảng | nnU-Net v1, chạy DDP trên Linux (NCCL); paper ghi 1 GPU RTX 8000 48 GB, config ghi "8 GPU" | PyTorch + MONAI, 1 GPU 4 GB, Windows | |
| Số chiều | 3D | 2D (chính), 3D thu nhỏ | **khác biệt lớn nhất** |
| Patch / batch | 128³ / 2 mỗi GPU | 160² / 8; 96³ / 1 × cộng dồn 2 | |
| U-Net | 6 mức, kênh [32, 64, 128, 256, 320, 320], 5 lần ÷2 (đáy 4³) | 5 mức, [32..320] (2D) hoặc [16..256] (3D), 4 lần ÷2 | |
| Khởi tạo trọng số | He (kaiming normal) cho conv, xavier cho Transformer | mặc định của PyTorch | ảnh hưởng nhỏ |
| Transformer decoder | 20 query, 3 lớp, hidden 192, 8 đầu, FFN 1536, feature 1/8–1/2 | giống (2D); 3D: hidden 96, FFN 384 | |
| Độ chính xác của attention | ép fp32 (`is_mhsa_float32: True`) | fp16 theo AMP | |
| Trọng số lớp "không có gì" | không đặt → 1.0 | 0.1 (mặc định Mask2Former) | nên thử cả hai |
| BCE/Dice của mask | trên toàn bộ voxel | trên 12 544 điểm ngẫu nhiên | `loss.num_points=0` để giống gốc |
| Trọng số loss, cách gộp aux | code: 2 : 5 : 5 chia 10, kiểu `'v1'` (paper viết λ₀ = 0.7, λ₁ = 0.3) | giống code gốc | |
| Deep supervision của CNN | tắt (`disable_ds: True`) | không có | giống |
| Optimizer | AdamW 3e-4, wd 3e-5 | giống | |
| Lịch lr | warmup 10 epoch + cosine, 125 epoch | warmup 2 epoch + cosine theo bước, 100 epoch | |
| Augmentation | "insaneDA" của nnU-Net (xoay ±90°, phóng 0.65–1.6, elastic, gamma…) | lật + đổi cường độ (+ affine nhẹ ở 2D) | |
| Tỉ lệ mẫu có u | 33% | 50% | |
| Tiền xử lý | crop vùng khác 0 + z-score trên mask não | cùng ý tưởng | |
| Suy luận | cửa sổ trượt 128³, bước 0.5, trung bình đều, không TTA | 2D: từng lát; 3D: cửa sổ 96³, chồng 0.5, trọng số Gauss | |
| Hậu xử lý | ngưỡng 0.5 từng vùng, gán nhãn ghi đè | max lũy tiến + ngưỡng 0.5 | |
| Chia dữ liệu | 5-fold (nnU-Net tự tạo `splits_final.pkl`) | 70/15/15, seed 42 | |
| Cách tính Dice | 0 nếu dự đoán hoặc nhãn rỗng | quy ước BraTS | |
| Encoder-only | ViT-B 12 lớp, 768 chiều, pretrain, thay hẳn feature | 4 lớp, 384 chiều, không pretrain, cộng residual | |

Khi báo cáo nên chia bảng này làm hai nhóm: (1) khác biệt **bắt buộc** do phần cứng (2D, patch, batch, số epoch); (2) khác biệt **có chủ ý hoặc có thể sửa** (no-object weight, lấy mẫu điểm, augmentation). Nhóm (2) là ứng viên tốt cho thí nghiệm.

---

## Câu 41 — Kết quả paper trên BraTS2021 là gì? So với kết quả của nhóm thế nào cho đúng?

Bảng BraTS2021 trong paper (Dice %, 3D, 5-fold cross-validation với **cùng cách chia** của đội hạng 1 BraTS21):

| Method | ET | TC | WT | Trung bình |
|---|---|---|---|---|
| nnU-Net | 88.05 | 91.92 | 93.79 | 91.25 |
| AxialAttn (của đội hạng 1) | 87.23 | 91.88 | 93.21 | 90.77 |
| nnU-Net Large (hạng 1 BraTS21) | 88.23 | 92.35 | 93.83 | 91.47 |
| **3D TransUNet** | **88.85** | **92.48** | **93.90** | **91.74** |

Paper ghi train với batch 2 trên 1 GPU RTX 8000. docs/01 và docs/03 ghi "8 GPU" là lấy theo chú thích trong config của repo (`max_num_epochs: 125 # used 8 cards as default`), không phải từ lời văn của paper.

- So với nnU-Net: ET +0.80, TC +0.56, WT +0.11, trung bình +0.49.
- Tăng nhiều nhất ở **ET**, vùng nhỏ nhất. Điều này khớp với lập luận "Transformer decoder giúp mục tiêu nhỏ".
- Mức tăng nhỏ (dưới 1 điểm) vì nnU-Net vốn đã rất mạnh trên BraTS.

**Nhóm không thể so trực tiếp với các số này**, vì khác số chiều (2D), khác cách chia dữ liệu, khác số epoch và số GPU, khác augmentation. Cách so đúng:

1. So U-Net với TransUNet **của nhóm**: cùng split, cùng số epoch, cùng pipeline, chỉ khác mô hình.
2. Báo trung bình, **trung vị**, và **số ca có Dice < 0.5**.
3. Nếu được, kiểm định cặp (ví dụ Wilcoxon trên Dice từng ca) để xem chênh lệch có ý nghĩa thống kê không.

**Kết quả hiện có của nhóm** (docs/02, 140 ca train, 20 epoch):

- U-Net 2D, test 30 ca: WT 0.870, TC 0.857, ET 0.830, trung bình 0.852 (trung vị 0.914 / 0.920 / 0.892).
- TransUNet 2D cùng điều kiện: val tốt nhất ở epoch 4 (trung bình 0.868). Về sau WT tăng (0.934) nhưng TC tụt dần từ 0.856 xuống 0.694.

---

## Câu 42 — Vì sao TransUNet 2D (lần train đầu) bị tụt TC? Nên thử gì?

**Chẩn đoán nhóm đã làm** (docs/02): lỗi nằm ở **phần phân lớp của query**, không phải ở mask. Ở ca khó, tổng P(TC) trên mọi query chỉ khoảng 0.75, P(ET) khoảng 0.5. Nhân với xác suất mask làm xác suất vùng rơi xuống dưới 0.5 (xem Phần 1, Câu 12).

**Các giả thuyết** (cần thí nghiệm để kiểm chứng, **chưa phải kết luận**):

1. **Train quá ngắn.** Mô hình kiểu Mask2Former (dự đoán tập hợp + Hungarian matching) thường hội tụ chậm hơn U-Net. 20 epoch × 250 bước có thể chưa đủ để việc phân công query ổn định.
2. **`no_object_weight` khác gốc.** Nhóm dùng 0.1, config BraTS gốc dùng 1.0. Trọng số này quyết định loss phân lớp nghiêng về 3 query thật hay 17 query "rỗng". Thử `--opts loss.no_object_weight=1.0` để xem ảnh hưởng.
3. **Lấy mẫu 12 544 điểm** làm chi phí ghép và loss mask "ồn" hơn. Thử `--opts loss.num_points=0`.
4. **Attention chạy fp16**, trong khi repo gốc ép fp32.
5. **Augmentation và cách lấy mẫu** khác gốc (Phần 2, Câu 25 và 26).

**Cách kiểm tra:**

- In xác suất lớp của 20 query trên vài ca val (như nhóm đã làm với ca 00495 và 00621).
- Vẽ Dice val theo epoch (`scripts/plot_log.py`).
- Mỗi lần **chỉ đổi một yếu tố**, giữ nguyên split, seed và số epoch.
- Thử thêm cách gộp khác ở hậu xử lý (Câu 38, ý cuối), chọn trên val.

---

## Câu 43 — Nên làm ablation nào? Đọc kết quả ra sao?

Ablation là thí nghiệm "bỏ bớt hoặc đổi một thành phần" để xem thành phần đó đóng góp bao nhiêu.

| Thí nghiệm | Cách chạy | Trả lời câu hỏi |
|---|---|---|
| Tắt masked attention | `--opts model.decoder.masked_attn=false` | Coarse-to-fine có giúp không? |
| Số query = số lớp | `--opts model.decoder.num_queries=3` | Nhiều query hơn có giảm bỏ sót không? |
| Encoder-only | `--config configs/transunet2d_encoder.yaml` | Transformer đặt ở đáy hay ở đầu ra tốt hơn? |
| Encoder + Decoder | bật `vit_bottleneck` trong `transunet2d.yaml` | Kết hợp cả hai có hơn không? |
| No-object như gốc | `--opts loss.no_object_weight=1.0` | Khác biệt này ảnh hưởng thế nào? |
| Loss trên đủ điểm | `--opts loss.num_points=0` | Lấy mẫu điểm có làm giảm chất lượng? |

**Số liệu paper để đối chiếu xu hướng** (paper không làm ablation trên BraTS; các bảng dưới là trên MSD Hepatic Vessel, Dice trung bình %):

| Ablation trong paper | Kết quả |
|---|---|
| Số query 5 / 20 / 40 | 67.53 / 67.67 / 67.37 → gần như không đổi |
| Bỏ masked attention (giữ feature đa tỉ lệ) | 67.67 → 67.54 |
| Bỏ cả feature đa tỉ lệ lẫn masked attention | 67.04 |
| Không có Transformer decoder (nnU-Net) | 66.04 |
| Encoder-only ViT 1 lớp / 12 lớp; Encoder+Decoder | 66.30 / 66.35; 67.24 |

Nếu kết quả 2D của nhóm cho xu hướng khác (ví dụ tắt masked attention mà không đổi gì), đó vẫn là một kết quả đáng báo cáo, kèm giải thích về khác biệt 2D/3D và số epoch.

Nguyên tắc:

- Cùng split, cùng seed, cùng số epoch; mỗi lần chỉ đổi một thứ; đặt `output_dir` riêng cho mỗi thí nghiệm.
- So sánh bằng `evaluate.py --split val`. Chỉ chạy test cho cấu hình cuối cùng.
- Chênh lệch dưới khoảng 0.5–1 điểm Dice với chỉ một lần chạy thì **chưa đủ** để kết luận, vì kết quả dao động theo seed. Nếu có thời gian, chạy 2–3 seed.

---

# C. CHUẨN BỊ BÁO CÁO

## Câu 44 — Các câu hỏi nhanh giảng viên hay hỏi

| # | Câu hỏi | Trả lời ngắn |
|---|---|---|
| 1 | Tại sao không dùng accuracy? | Nền chiếm ~99%. Đoán toàn nền đã đạt ~99% accuracy mà không tìm được u nào. Dice chỉ tính trên vùng u. |
| 2 | Transformer trong mô hình của em nằm ở đâu? | Ở **decoder**, gắn sau decoder CNN của U-Net, thay cho conv 1×1 cuối. Nó nhận feature 1/8, 1/4, 1/2 và pixel embedding 1/1; trả ra 20 mask + 20 nhãn lớp. |
| 3 | Tại sao InstanceNorm mà không BatchNorm? | Batch nhỏ (3D chỉ 1–2) làm thống kê của BatchNorm không đáng tin. InstanceNorm chuẩn hóa từng ảnh riêng; đây là lựa chọn chuẩn của nnU-Net. |
| 4 | LayerNorm trong Transformer khác gì? | LayerNorm chuẩn hóa từng token theo chiều đặc trưng (192 số), không phụ thuộc batch hay vị trí. |
| 5 | Hungarian matching có lấy đạo hàm không? | Không. Nó chỉ phân công query nào học vùng nào. Gradient đi qua loss trên các cặp đã ghép. |
| 6 | Masked attention khác attention thường chỗ nào? | Cộng −∞ vào điểm của các vị trí nằm ngoài mask vòng trước, nên query chỉ lấy thông tin trong vùng nó đang khoanh. |
| 7 | Tại sao decoder dùng được feature độ phân giải cao còn ViT thì không? | Cross-attention tốn 20 × L; self-attention của ViT tốn L × L. Ở mức 1/2 bản 2D, chênh nhau 320 lần. |
| 8 | Tại sao 20 query? | Paper đặt N lớn hơn số lớp để có nhiều ứng viên, giảm bỏ sót. Nhưng ablation của paper (MSD Vessel) cho thấy 5 / 20 / 40 query gần như bằng nhau (67.53 / 67.67 / 67.37). Nhóm có thể tự kiểm tra bằng `num_queries=3`. |
| 9 | Có pretrain không? | Không. Decoder-only của repo gốc cũng không pretrain; chỉ Encoder-only dùng ViT pretrain ImageNet. |
| 10 | Có rò rỉ dữ liệu không? | Không: chia theo bệnh nhân, seed cố định, file `data/splits.json` lưu lại. |
| 11 | Làm sao biết không overfit? | Theo dõi train loss và Dice val theo epoch; chọn `best.pt` trên val; đánh giá cuối trên test chưa từng dùng. |
| 12 | Tại sao ET khó nhất? | ET nhỏ nhất, thường là một viền mỏng; 33 ca không có ET nên Dice chỉ là 0 hoặc 1 ở các ca đó. |
| 13 | Sliding window, chồng 50%, trọng số Gauss để làm gì? | Ảnh lớn hơn patch nên phải cắt thành nhiều cửa sổ. Chồng lấn và ưu tiên tâm cửa sổ giúp giảm lỗi ở mép. |
| 14 | Có dùng test-time augmentation (TTA) không? | Chưa. nnU-Net thường lật gương rồi lấy trung bình; có thể thêm như một phần mở rộng. |
| 15 | Kết quả có ý nghĩa thống kê không? | Cần so từng ca giữa hai mô hình (Wilcoxon), và nếu được thì chạy nhiều seed. Một lần chạy chênh < 1 điểm thì chưa kết luận được. |
| 16 | Chạy model trên một bệnh nhân mới thế nào? | 4 file MRI → tiền xử lý giống lúc train → `predict_volume` → `binarize` → `regions_to_seg` → `uncrop` → file `.nii.gz` cùng định dạng file seg. Hiện `evaluate.py` chạy theo danh sách trong split; ca mới cần một script nhỏ gọi các hàm này. |
| 17 | Hạn chế của nhóm? | Bản chính là 2D; ít epoch; một cách chia dữ liệu thay vì 5-fold; augmentation nhẹ; GPU 4 GB. |
| 18 | Hướng phát triển? | 3D đầy đủ hơn, train lâu hơn, 5-fold, TTA, hậu xử lý ET, thử 2.5D, sửa các điểm khác gốc ở Câu 40. |
| 19 | Đóng góp của nhóm là gì? | Tái hiện độc lập bằng PyTorch + MONAI chạy trên GPU 4 GB; đọc dữ liệu thẳng từ tar; thống kê và kiểm tra toàn bộ dữ liệu; so sánh U-Net và TransUNet trong cùng điều kiện; phân tích lỗi phân lớp của query. |
| 20 | Tái lập kết quả (reproducibility) thế nào? | Seed 42 cho split và train, config lưu trong checkpoint, `log.csv` ghi từng epoch. Vẫn còn sai khác nhỏ do một số phép tính trên GPU không tất định. |

---

## Câu 45 — Tóm tắt 1 phút để nói khi báo cáo

> Đề tài của nhóm là phân vùng u não trên MRI của bộ BraTS2021: 1251 bệnh nhân, mỗi người có 4 loại ảnh MRI 3D, và cần tô 3 vùng lồng nhau là toàn bộ khối u (WT), lõi u (TC) và phần u tăng cường (ET).
>
> Paper tham khảo là 3D TransUNet. Paper gắn Transformer vào U-Net và cho thấy: với khối u nhỏ như BraTS, đặt Transformer ở **decoder** là tốt nhất. Cụ thể, có 20 query học được; mỗi query dùng masked cross-attention nhìn vào feature đa tỉ lệ của U-Net, tinh chỉnh qua 3 vòng từ thô tới mịn, rồi xuất ra một mask và một nhãn lớp. Khi train, query được ghép với vùng thật bằng Hungarian matching. Paper đạt Dice 88.85 / 92.48 / 93.90 cho ET / TC / WT, hơn nnU-Net khoảng 0.5 điểm.
>
> Nhóm tự cài đặt lại bằng PyTorch và MONAI, đọc dữ liệu thẳng từ file tar. Vì GPU chỉ có 4 GB, nhóm làm bản 2D trước rồi bản 3D thu nhỏ. Kết quả đầu tiên: U-Net 2D đạt Dice trung bình 0.852 trên 30 ca test sau 20 epoch. TransUNet 2D đang gặp vấn đề xác suất phân lớp của query thấp làm vùng TC bị giảm; nhóm đã tìm ra nguyên nhân và cải thiện bước hậu xử lý.
>
> Bước tiếp theo: train lâu hơn, làm ablation masked attention và số query, và hoàn thiện bản 3D.

---

## Phụ lục — Tra nhanh: khái niệm ↔ file code

| Khái niệm | Code của nhóm (`brats-transunet/`) | Code gốc (`Repo/3D-TransUNet/`) |
|---|---|---|
| Đọc dữ liệu từ tar | `src/brats/io.py:55` `TarSource` | (không có; dùng nnU-Net convert) |
| Crop, z-score, đổi nhãn | `src/brats/preprocess.py:33, 41, 18` | nnU-Net `ImageCropper`, `GenericPreprocessor` |
| Lấy mẫu 2D / 3D | `src/brats/datasets.py:118, 141` | nnU-Net `DataLoader3D` (33% có u) |
| Augmentation | `src/brats/datasets.py:54` | `nnUNetTrainerV2_DDP.py:134` (insaneDA) |
| Khối conv | `src/brats/models/blocks.py:22` | `transunet3d_model.py:32` `ConvDropoutNormNonlin` |
| U-Net | `src/brats/models/unet.py:20` | `transunet3d_model.py:187` (phần CNN) |
| ViT ở bottleneck | `src/brats/models/vit.py:9` | `vit_modeling.py:251` `Transformer` |
| Transformer decoder | `src/brats/models/transformer_decoder.py:68` | `mask2former_transformer_decoder3d.py:244` |
| Ghép U-Net + decoder | `src/brats/models/transunet.py:19` | `transunet3d_model.py:501` `forward` |
| Gộp 20 query → 3 vùng | `src/brats/models/__init__.py:22` | `nnUNetTrainerV2_DDP.py:836-839` |
| Hungarian + loss | `src/brats/losses.py:51` | `transunet3d_model.py:692, 881` |
| Vòng lặp train | `train.py:105` | `nnUNetTrainerV2_DDP.py:506` `run_iteration` |
| Suy luận cả volume | `src/brats/inference.py:18` | `inference.py:286` `predict` |
| Hậu xử lý | `src/brats/inference.py:48` | `inference.py:478-485` |
| Dice, HD95 | `src/brats/metrics.py:9, 18` | `measure_dice.py:39` |
| Config Decoder-only | `configs/transunet2d.yaml` | `configs/Brats/decoder_only.yaml` |

← Quay lại **Phần 1**: [`05_HOI_DAP_P1_MO_HINH.md`](05_HOI_DAP_P1_MO_HINH.md) · **Phần 2**: [`06_HOI_DAP_P2_DU_LIEU_2D_3D.md`](06_HOI_DAP_P2_DU_LIEU_2D_3D.md)
