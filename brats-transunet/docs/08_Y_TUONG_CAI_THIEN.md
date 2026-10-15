# 08 — Ý tưởng cải thiện so với paper

> Viết ngày 13/10/2026. Giảng viên yêu cầu nhóm **cải thiện hơn paper**: tăng độ chính xác, hoặc nếu không được thì tăng hiệu năng (giảm tham số, tăng tốc, giảm bộ nhớ).
>
> File này ghi: hướng nên chọn và lý do, ba thay đổi chính, các ý tưởng phụ, việc phải sửa ở baseline, tiêu chí so sánh, và bảng ước lượng tài nguyên.
>
> **Cách đọc độ tin cậy.** Mỗi con số thuộc một trong ba loại, ghi rõ ở từng chỗ:
>
> - **[nguồn]**: lấy từ paper khác, có link ở mục 9.
> - **[log]**: tính từ `runs/` của nhóm.
> - **[ước lượng]**: tính bằng công thức, chưa đo.
>
> Những câu ghi **giả thuyết** là chưa được thí nghiệm của nhóm kiểm chứng.

## Mục lục

1. Kết luận
2. Vì sao chọn hướng "chi phí của decoder"
3. Ba thay đổi chính
4. Ý tưởng phụ
5. Không nên làm
6. Baseline phải sửa gì trước
7. So với paper theo tiêu chí nào
8. Ước lượng tham số và tài nguyên
9. Nguồn

---

## 1. Kết luận

Không nhắm vào độ chính xác tuyệt đối của paper. Nhắm vào **chi phí của Transformer decoder** và việc nó hội tụ chậm khi train ngắn. Gói đề xuất gồm ba thay đổi, thử ở 2D trước rồi xác nhận ở 3D thu nhỏ:

1. **Tính mask "lười"**: dự đoán giữ nguyên, bớt bộ nhớ và phép tính.
2. **Ba query cố định** cho WT/TC/ET, bỏ Hungarian matching và class head.
3. **Thu gọn decoder** để giảm tham số.

---

## 2. Vì sao chọn hướng "chi phí của decoder"

### 2.1 BraTS2021 đã bão hòa

Benchmark nnU-Net Revisited (MICCAI 2024) train lại nhiều phương pháp, 5-fold trên đủ 1251 ca **[nguồn]**:

| Phương pháp | Dice trung bình (%) | Giờ GPU |
|---|---|---|
| nnU-Net gốc | 91.24 | 9 |
| nnU-Net ResEnc M | 91.26 | 12 |
| MedNeXt L k5 | 91.50 | 233 |
| SwinUNETR | 90.68 | 15 |
| U-Mamba Bot | 91.26 | 24 |
| Cùng mạng đó, bỏ Mamba | 91.26 | 24 |

- Độ lệch chuẩn giữa các fold của cùng một phương pháp: 0.66 điểm. Giữa các phương pháp (bỏ nhóm Auto3DSeg): 0.35 điểm.
- Paper của nhóm báo 91.25 lên 91.74 (+0.49). Mức này nằm trong khoảng nhiễu nói trên. Cách chia fold khác nhau nên hai bảng không so trực tiếp được.

### 2.2 Chính paper cho thấy decoder đắt

Bản tạp chí (Medical Image Analysis 2024) có bảng chi phí mà bản arXiv trong `Doc/` không có **[nguồn, mới đọc qua đoạn trích tìm kiếm, phải mở bản gốc đối chiếu]**:

| | nnU-Net | TransUNet |
|---|---|---|
| Tham số, Decoder-only | 30.8M | 33.6M (+9%) |
| Bộ nhớ GPU, Decoder-only | 5.81 GB | 11.16 GB |
| Suy luận mỗi volume, Encoder+Decoder | 0.016 s | 0.058 s |
| Thời gian mỗi epoch, Encoder+Decoder | 101 s | 188 s |
| FLOPs, Encoder+Decoder | 315.7G | 362.3G (+15%) |

- FLOPs tăng ít, bộ nhớ và thời gian tăng nhiều: nghẽn ở tensor lớn, không ở số phép tính.
- Trong báo cáo BraTS-METS 2023, chính tác giả viết rằng Decoder-only cần train lâu hơn, nên họ nộp bản Encoder-only.

### 2.3 Kết quả của nhóm

Cùng 30 ca test, 140 ca train, 20 epoch **[log]**:

| | U-Net 2D | TransUNet 2D |
|---|---|---|
| Dice trung bình (WT / TC / ET) | 0.870 / 0.857 / 0.830 | 0.829 / 0.782 / 0.789 |
| Dice trung bình 3 vùng | 0.852 | 0.800 |
| HD95 trung vị của WT | 5.9 mm | 49.3 mm |
| Thời gian mỗi epoch | 78 s | 203 s |

U-Net hơn ở 28/30 ca, Wilcoxon p ≈ 7·10⁻⁶.

---

## 3. Ba thay đổi chính

### 3.1 Tính mask "lười"

**Hiện tại.** Mask của mỗi query là tích vô hướng giữa vector query và feature của từng voxel:

- Feature U-Net ở độ phân giải gốc: `(B, 32, *S)` (bản 3D thu nhỏ: 16 kênh).
- `mask_proj` (conv 1×1) nâng lên `(B, hidden, *S)` — `models/transunet.py:28`.
- Nhân với `mask_embed(query)` — `models/transformer_decoder.py:91`.
- Code gốc làm y hệt (`mask2former_transformer_decoder3d.py:487-489`, `transunet3d_model.py:483`).

Tensor `(B, hidden, *S)` là tensor lớn nhất cả mạng. Ở cấu hình paper nó có 403 triệu phần tử cho mỗi mẫu **[ước lượng]**.

**Thay đổi.** Conv 1×1 là phép tuyến tính, nên hạ query xuống số kênh của feature thay vì nâng feature lên `hidden`:

```python
# hiện tại
F_pix = self.mask_proj(feats[-1])                           # (B, 192, *S)
masks = einsum("bqc,bc...->bq...", mask_embed(q), F_pix)    # (B, N, *S)

# tương đương về toán
W, b = self.mask_proj.weight.flatten(1), self.mask_proj.bias    # (192, 32), (192,)
E = mask_embed(q)                                           # (B, N, 192)
masks = einsum("bqk,bk...->bq...", E @ W, feats[-1])        # (B, N, *S)
masks = masks + (E @ b).view(B, N, *[1] * dims)
```

Cùng nguyên lý cho hai chỗ nữa:

- **Attention mask của tầng giữa**: nội suy cũng tuyến tính, nên tính thẳng trên feature đã thu nhỏ. Không cần dựng mask đủ độ phân giải ở mỗi tầng.
- **Loss khi train**: chỉ dùng 12 544 điểm lấy mẫu (`losses.py:91`), nên chỉ cần tính logit tại các điểm đó.

**Đã kiểm chứng tới đâu.** Đẳng thức đã thử bằng số trên tensor ngẫu nhiên (CPU): sai khác tối đa 3·10⁻⁶ ở float32, không bit nào của attention mask đổi. Chưa cài vào model.

**Giới hạn.**

- Tổng phép tính chỉ giảm 8–14% (mục 8). Lợi ích chính là bộ nhớ, và chỉ đáng kể ở 3D.
- Chưa tìm thấy bài nào mô tả đúng thủ thuật này cho 3D TransUNet, nhưng không khẳng định được là mới. Ý gần nhất: Mask2Former tính loss trên điểm lấy mẫu, bộ nhớ train từ 18 GB xuống 6 GB **[nguồn]**.

### 3.2 Ba query cố định, bỏ Hungarian

**Thay đổi.**

- `num_queries = 3`; query thứ k luôn là vùng thứ k (WT, TC, ET).
- Bỏ class head và lớp "no object".
- Loss: Dice + BCE trên 3 mask của mỗi tầng (dùng lại `RegionDiceBCELoss`).
- Suy luận: `sigmoid(pred_masks)`.

**Lý do đáng thử.**

- Xác suất vùng hiện là tổng trên 20 query của P(lớp | query) × sigmoid(mask) (`models/__init__.py:26`; code gốc `inference.py:330-332` cũng vậy). Khi P(lớp) chưa sắc nét, vùng thật bị kéo xuống dưới 0.5. Đây là lỗi nhóm đã chẩn đoán ở docs/02.
- **Giả thuyết:** phần dư xác suất của 17 query thừa cộng dồn thành dương tính giả, và ở vùng không có u thì không mask nào nhận loss (chỉ có loss phân lớp). Đây có thể là nguồn của HD95 49 mm.
- Ablation của paper: 5, 20, 40 query cho kết quả như nhau (67.53 / 67.67 / 67.37 trên MSD Vessel).
- MaskMed (2025): trên ảnh y tế, số query bằng số lớp tốt nhất và hội tụ nhanh nhất; Dice 0.913, tụt dần còn 0.891 khi gấp 4 **[nguồn]**.
- DN-DETR: DETR hội tụ chậm một phần do matching không ổn định **[nguồn]**.

**Bằng chứng ngược.** MaskFormer: bipartite nhỉnh hơn fixed trên ADE20K 150 lớp (44.2 so với 43.7 mIoU) **[nguồn]**. Vì vậy thay đổi này là **giả thuyết**, phải có thí nghiệm.

**Lợi thêm.** U-Net và TransUNet khi đó cùng loss, chỉ khác phần head.

### 3.3 Thu gọn decoder

- Decoder 2D có 2.88M tham số, khớp mức chênh 2.8M trong paper **[ước lượng, khớp kích thước checkpoint]**.
- FFN chiếm 62% (vì `ffn = 8 × hidden`), attention 31%.
- Hạ `ffn_mult` 8 xuống 2 và dùng chung trọng số cho 3 tầng: decoder còn 0.65M (−77%).
- Rủi ro: tụt Dice. Phải đo.

---

## 4. Ý tưởng phụ

Xếp theo mức đáng làm.

| Ý tưởng | Căn cứ [nguồn] | Ghi chú |
|---|---|---|
| Deep supervision cho decoder CNN (head phụ, bỏ khi suy luận) | NVIDIA ở BraTS21: thứ duy nhất có ích trong ablation, 0.9130 lên 0.9149 | Rẻ; config gốc tắt (`disable_ds: True`) |
| Hậu xử lý: ET quá ít voxel thì đổi sang NCR; xoá mảnh nhỏ | nnU-Net BraTS 2020; NVIDIA: 0.9156 lên 0.9163 | Không tốn GPU; chọn ngưỡng trên val |
| Mask annealing: giảm dần masked attention về 0 khi train | EoMT (CVPR 2025): nhanh 2.1 lần, mất 0.2 PQ; tắt thẳng thì sụp 56.2 còn 27.4 | Paper: masked attention chỉ thêm 0.13 Dice |
| 2.5D: ghép 3 lát kề thành 12 kênh vào | GDNet (2026), GPU 8 GB | Chỉ đổi conv đầu |
| Knowledge distillation sang model nhỏ | Uni-Light (2026), DCD (MICCAI 2026) | Gần như đề tài riêng |
| MP-Former: đưa mask thật có nhiễu vào masked attention khi train | CVPR 2023: nửa số bước trên ADE20K | Cài phức tạp |
| TTA lật ảnh | Đội thắng BraTS 2023: tắt TTA tốt hơn trên val, nhanh hơn tới 8 lần | Thử, đừng mặc định có lợi |

---

## 5. Không nên làm

- **Thay Transformer bằng Mamba.** nnU-Net Revisited: lớp Mamba không đóng góp gì.
- **Thêm hoặc phóng to Transformer encoder.** Primus (2025): ở 6/9 kiến trúc lai, thay khối Transformer bằng phép đồng nhất gần như không đổi kết quả.
- **Depthwise conv 3D để giảm FLOPs.** FLOPs giảm nhưng thường không nhanh hơn trên GPU. MedNeXt mất 68–233 giờ train so với 9 giờ của nnU-Net.
- **Dữ liệu tổng hợp bằng GAN, foundation model kiểu SAM.** Quá nặng cho 4 GB.
- **Tin các bài báo Dice 98–99% trên BraTS.** Cách tính khác, không so được.

---

## 6. Baseline phải sửa gì trước

TransUNet 2D hiện kém U-Net 5 điểm Dice và val tốt nhất rơi vào epoch 4 rồi tụt. Lấy nó làm mốc thì mọi cải thiện đều bị hỏi lại.

| Mức | Cần đổi | Lý do |
|---|---|---|
| Bắt buộc | `loss.no_object_weight=1.0` | Config gốc là 1.0, nhóm đang dùng 0.1 |
| Bắt buộc | Train ít nhất 50–100 epoch, bằng nhau cho mọi model | 20 epoch quá ngắn |
| Bắt buộc | Split trên đủ 1251 ca (khoảng 875 / 188 / 188) | 30 ca test không đủ để thấy chênh lệch dưới khoảng 1 điểm |
| Bắt buộc | `eval.val_cases` khoảng 30; ghi rõ đánh giá `best.pt` hay `last.pt` | Chọn checkpoint trên 10 ca rất nhiễu |
| Nên | 2 seed cho cặp so sánh chính | Nhiễu giữa các lần train cỡ 0.66 điểm |
| Chỉ cần ghi chú | Lấy mẫu điểm, attention fp16, 50% mẫu có u, augmentation nhẹ, 5 mức thay vì 6 | |
| Không sửa được | 3D 128³, 5-fold, 8 GPU | Phần cứng |

Dữ liệu train là dòng vô hạn nên thêm ca không làm epoch dài hơn.

**Baseline "ổn" khi:** đường Dice val không sụp giữa chừng, và TransUNet không kém U-Net quá mức nhiễu.

---

## 7. So với paper theo tiêu chí nào

| Tiêu chí | Cách so | Thế nào là ổn |
|---|---|---|
| Dice tuyệt đối | Không so trực tiếp | Không đặt ngưỡng |
| Khoảng cách TransUNet − U-Net | So xu hướng với paper (+0.49 trung bình, +0.80 ở ET) | Cùng dấu, hoặc chênh không có ý nghĩa thống kê |
| Số tham số | So trực tiếp | Cấu hình paper đếm ra 34.07M, paper ghi 33.6M |
| Phép tính ở cấu hình paper | Tính được, không cần train | Báo tỉ lệ so với U-Net |
| VRAM đỉnh, giây/bước, giây/ca | So tỉ lệ với U-Net trên máy nhóm (paper: bộ nhớ gấp 1.9, epoch gấp 1.9) | Bản cải tiến có tỉ lệ thấp hơn baseline |
| HD95, trung vị, số ca hỏng | Paper không báo; nhóm báo thêm | |

Tiêu chí cho bản cải tiến: **Dice không kém baseline** (chênh lệch cặp trong khoảng ±0.5 điểm, Wilcoxon không có ý nghĩa) **và chi phí thấp hơn**. Tính Dice theo cả quy ước BraTS và quy ước của `measure_dice.py`.

**Bảng thí nghiệm** (mỗi lượt đổi một yếu tố, cùng split, seed, số epoch):

| Lượt | Cấu hình | Trả lời câu hỏi |
|---|---|---|
| U | U-Net | mốc |
| A | TransUNet, `no_object_weight=1.0` | baseline đúng gốc |
| B | A + mask lười (không train lại) | bộ nhớ, thời gian; Dice phải y hệt |
| C | 3 query cố định | matching có cần không |
| D | C + mask lười (không train lại) | |
| E | D + decoder gọn | giảm tham số mất bao nhiêu Dice |

---

## 8. Ước lượng tham số và tài nguyên

Tất cả là **[ước lượng]**. "Tensor phụ" là các tensor decoder tạo thêm ở độ phân giải gốc cho mỗi mẫu, tính ở fp16, chưa kể gradient.

**Cấu hình paper, khối 128³**

| Phương án | Tham số | Phép nhân-cộng | Tensor phụ |
|---|---|---|---|
| U | 31.19M | 483 tỉ | 0 |
| A | 34.07M | 556 tỉ | 1088 MB |
| B | 34.07M | 513 tỉ (−8%) | 80 MB |
| C | 34.07M | 527 tỉ (−5%) | 816 MB |
| D | 34.07M | 510 tỉ (−8%) | 12 MB |
| E | 31.85M (−7%) | 510 tỉ (−8%) | 12 MB |

**`transunet3d_small`, khối 96³**

| Phương án | Tham số | Phép nhân-cộng | Tensor phụ |
|---|---|---|---|
| U | 5.65M | 51.5 tỉ | 0 |
| A | 6.15M | 63.0 tỉ | 297 MB |
| B | 6.15M | 55.1 tỉ (−12%) | 34 MB |
| C | 6.15M | 56.8 tỉ (−10%) | 182 MB |
| D | 6.15M | 54.4 tỉ (−14%) | 5 MB |
| E | 5.81M (−6%) | 54.4 tỉ (−14%) | 5 MB |

**Bản 2D, lát 160²**

| Phương án | Tham số | Phép nhân-cộng mỗi lát |
|---|---|---|
| U | 5.69M | 4.52 tỉ |
| A | 8.57M | 5.93 tỉ |
| B | 8.57M | 5.40 tỉ (−9%) |
| D | 8.56M | 5.29 tỉ (−11%) |
| E | 6.34M (−26%) | 5.29 tỉ (−11%) |

Đọc bảng:

- Bộ nhớ là lợi ích chính, chỉ đáng kể ở 3D.
- Giảm tham số chỉ rõ ở 2D. Ở kích thước paper, 91% tham số nằm trong U-Net.
- Phép tính giảm khiêm tốn. Thời gian thực phải đo.

**Thời gian train 2D [log]** (78 s và 203 s mỗi epoch):

| | 20 epoch | 50 epoch | 100 epoch |
|---|---|---|---|
| U-Net 2D | 0.4 giờ | 1.1 giờ | 2.2 giờ |
| TransUNet 2D | 1.1 giờ | 2.8 giờ | 5.6 giờ |

### 8.1 Số đo thực tế (13/10/2026)

Đo bằng `scripts/smoke_test.py`: một bước train với dữ liệu giả, có AMP, trên RTX 3050 Ti 4 GB. Phương án B, D, E (mask lười, query cố định) **chưa cài nên chưa đo**; các dòng "3 query" dưới đây vẫn dùng Hungarian.

**2D, lát 160², batch 8**

| Cấu hình | Tham số | Giây/bước | VRAM đỉnh |
|---|---|---|---|
| U-Net | 5.69M | 0.75 | 0.37 GB |
| TransUNet, 20 query (A) | 8.57M | 2.06 | 0.84 GB |
| TransUNet, 3 query | 8.56M | 1.66 | 0.79 GB |
| TransUNet, 3 query, FFN×2 | 7.23M | 1.64 | 0.77 GB |

**3D thu nhỏ, batch 1**

| Cấu hình | Khối | Giây/bước | VRAM đỉnh |
|---|---|---|---|
| U-Net | 96³ | 1.34 | 0.57 GB |
| TransUNet, 20 query (A) | 96³ | 3.23 | 1.51 GB |
| TransUNet, 3 query | 96³ | 2.42 | 1.27 GB |
| TransUNet, 20 query | 112³ | 5.08 | 2.34 GB |
| U-Net | 128³ | 3.12 | 1.24 GB |
| TransUNet, 20 query | 128³ | 23.87 | 3.44 GB, tràn VRAM |
| TransUNet, 3 query | 128³ | 6.14 | 2.87 GB |

Nhận xét:

- **Decoder làm bộ nhớ tăng 2.6–2.8 lần** so với U-Net ở 3D (0.57 lên 1.51 GB; 1.24 lên 3.44 GB), dù chỉ thêm 9% tham số. Cùng xu hướng với bảng của paper (gấp 1.9).
- **Ở 128³ baseline không vừa 4 GB.** Lần cấp phát thất bại là đúng 402 653 184 byte, bằng kích thước tensor `mask_proj` ở fp16 (96 kênh × 128³ × 2 byte). Đây là bằng chứng trực tiếp cho mục 3.1.
- **Chỉ giảm 20 query xuống 3** đã cho 128³ chạy được (2.87 GB, 6.1 giây/bước) và nhanh hơn 20–25% ở 96³.
- **Số tham số đo khớp ước lượng** ở mọi dòng.
- **Thời gian tuyệt đối không tin được.** Hôm nay máy chạy chậm khoảng 2.5 lần so với log train ngày 18/09 (U-Net 2D: 0.75 so với 0.31 giây/bước). Chỉ nên dùng **tỉ lệ**: TransUNet chậm hơn U-Net 2.7 lần ở 2D và 2.4 lần ở 3D, khớp với log cũ (2.6 lần).

### 8.2 Đã cài đặt (14/10/2026)

Ba phương án là **cờ trong `model.decoder`, mặc định tắt**. Config baseline không đổi; mỗi phương án là một config mới kế thừa baseline bằng `base:`.

| Cờ | Ý nghĩa | Config dùng nó |
|---|---|---|
| `lazy_mask: true` | Tính mask lười (mục 3.1). Không thêm tham số, nạp được checkpoint cũ | `*_lazy.yaml`, `*_fixed.yaml`, `*_lite.yaml` |
| `fixed_queries: true` + `num_queries: 3` | 3 query cố định, loss `FixedQueryLoss` (mục 3.2) | `*_fixed.yaml`, `*_lite.yaml` |
| `ffn_mult: 2`, `shared_layers: true` | Decoder gọn (mục 3.3) | `*_lite.yaml` |

Mỗi phương án có bản 2D (`transunet2d_*.yaml`) và 3D (`transunet3d_small_*.yaml`).

Code đã đổi:

| File | Thay đổi |
|---|---|
| `src/brats/models/transformer_decoder.py` | Ba cờ trên |
| `src/brats/models/transunet.py` | Cờ `lazy_mask` |
| `src/brats/models/__init__.py` | `region_probs` hiểu output mới |
| `src/brats/losses.py` | Loss Hungarian tính mask chỉ tại điểm lấy mẫu; thêm `FixedQueryLoss` |
| `evaluate.py` | Thêm `--opts`, `--tag`; ghi thời gian mỗi ca (cột `sec`) và VRAM đỉnh khi suy luận |

Script mới:

| Script | Việc | Chạy ở đâu |
|---|---|---|
| `scripts/check_variants.py` | Kiểm tra lazy cho đúng kết quả cũ (suy luận, loss, gradient); kiểm tra fixed/lite chạy được | CPU, tensor nhỏ |
| `scripts/benchmark.py` | Tham số, thời gian và VRAM đỉnh của một bước train và một lần suy luận, cho nhiều config cùng lúc | GPU rảnh |

**Trạng thái (14/10):** đã qua các bài kiểm tra trên CPU (docs/09 mục 1). Chưa train, chưa đo trên GPU. Hướng dẫn chạy đầy đủ: docs/09.

Thứ tự chạy khi GPU rảnh, từ `brats-transunet/`:

```powershell
python scripts/check_variants.py
python scripts/benchmark.py --configs configs/unet2d.yaml configs/transunet2d.yaml configs/transunet2d_lazy.yaml configs/transunet2d_fixed.yaml configs/transunet2d_lite.yaml --csv figures/bench_2d.csv
python scripts/benchmark.py --configs configs/unet3d_small.yaml configs/transunet3d_small.yaml configs/transunet3d_small_lazy.yaml configs/transunet3d_small_fixed.yaml configs/transunet3d_small_lite.yaml --csv figures/bench_3d.csv
python scripts/benchmark.py --configs configs/unet3d_small.yaml configs/transunet3d_small.yaml configs/transunet3d_small_lazy.yaml configs/transunet3d_small_fixed.yaml --opts "train.patch_size=[128,128,128]" --csv figures/bench_3d_128.csv

# phương án B: đánh giá lại checkpoint baseline, không train
python evaluate.py --ckpt runs/transunet2d/best.pt --split test --hd95 --opts model.decoder.lazy_mask=true --tag lazy

# phương án C và E: train như baseline
python train.py --config configs/transunet2d_fixed.yaml
python train.py --config configs/transunet2d_lite.yaml
```

---

## 9. Nguồn

- nnU-Net Revisited: <https://arxiv.org/abs/2404.09556>
- TransUNet, Medical Image Analysis 2024: <https://www.sciencedirect.com/science/article/pii/S1361841524002056>
- 3D-TransUNet cho BraTS-METS 2023: <https://arxiv.org/abs/2403.15735>
- MaskMed: <https://arxiv.org/abs/2511.15603>
- MaskFormer: <https://arxiv.org/abs/2107.06278>
- DN-DETR: <https://arxiv.org/abs/2203.01305>
- MP-Former: <https://arxiv.org/abs/2303.07336>
- Mask2Former: <https://arxiv.org/abs/2112.01527>
- EoMT: <https://arxiv.org/abs/2503.19108>
- Primus: <https://arxiv.org/abs/2503.01835>
- Optimized U-Net (NVIDIA, BraTS21): <https://arxiv.org/abs/2110.03352>
- nnU-Net cho BraTS 2020: <https://arxiv.org/abs/2011.00848>
- Đội thắng BraTS 2023: <https://arxiv.org/abs/2402.17317>
- GDNet 2.5D: <https://pmc.ncbi.nlm.nih.gov/articles/PMC13412763/>
- Uni-Light: <https://pith.science/paper/2609.06729>
- DCD (MICCAI 2026): <https://papers.miccai.org/miccai-2026/0265-Paper5877.html>
