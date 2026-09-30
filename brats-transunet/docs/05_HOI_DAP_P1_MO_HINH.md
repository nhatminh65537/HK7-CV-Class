# 05 — Hỏi đáp (Phần 1/3): Mô hình U-Net, TransUNet và Transformer decoder

> **Bộ hỏi đáp gồm 3 phần**, nên đọc theo thứ tự:
>
> 1. **Phần 1 — Mô hình** (file này): U-Net, TransUNet gốc, 3D TransUNet, Transformer decoder, loss.
> 2. **Phần 2 — Dữ liệu, tiền xử lý, 2D và 3D:** [`06_HOI_DAP_P2_DU_LIEU_2D_3D.md`](06_HOI_DAP_P2_DU_LIEU_2D_3D.md)
> 3. **Phần 3 — Train, đánh giá, so với paper, chuẩn bị báo cáo:** [`07_HOI_DAP_P3_TRAIN_DANH_GIA.md`](07_HOI_DAP_P3_TRAIN_DANH_GIA.md)
>
> Câu có dấu ⭐ là câu bạn hỏi trực tiếp. Các câu còn lại là những câu giảng viên hoặc bạn cùng nhóm dễ hỏi.
>
> **Quy ước**
>
> - `B` = batch, `C` = số kênh. Ảnh 2D: `(B, C, H, W)`. Khối 3D: `(B, C, X, Y, Z)`.
> - Đường dẫn `src/...`, `train.py`, `configs/...` tính từ thư mục `brats-transunet/`. Đường dẫn `nn_transunet/...` là code gốc trong `Repo/3D-TransUNet/`.
> - Số sau dấu `:` là số dòng lúc viết tài liệu. Code đổi thì số dòng có thể lệch.
> - Ví dụ về shape dùng `configs/transunet2d.yaml`: 2D, batch 8, patch 160×160.

## Mục lục Phần 1

- 0. Toàn cảnh project trong một hình
- Câu 1 ⭐ U-Net gốc, TransUNet gốc, 3D TransUNet và bản của nhóm: là gì, khác nhau ở đâu?
- Câu 2 ⭐ U-Net gốc (2015) có những lớp gì? Input, output là gì?
- Câu 3 — U-Net trong code nhóm: đi qua từng lớp với shape cụ thể
- Câu 4 — TransUNet gốc (2021) là gì? Transformer đặt ở đâu?
- Câu 5 — 3D TransUNet có 3 cấu hình. Transformer đặt ở đâu trong mỗi cấu hình?
- Câu 6 ⭐ Trong cấu hình Decoder-only, Transformer nằm ở đâu, nhận gì, trả ra gì?
- Câu 7 — Attention hoạt động thế nào? Self-attention khác cross-attention ra sao?
- Câu 8 — "Query" là gì? Tại sao 20 query mà chỉ có 3 vùng?
- Câu 9 — Mask và lớp của mỗi query được tính ra sao?
- Câu 10 — Masked attention ("coarse-to-fine") là gì?
- Câu 11 — Positional encoding, level embedding, query embedding để làm gì?
- Câu 12 — Từ 20 cặp (lớp, mask) làm sao ra 3 bản đồ WT/TC/ET?
- Câu 13 — Hungarian matching là gì? Tại sao cần?
- Câu 14 — Loss của TransUNet tính thế nào?
- Câu 15 — Loss của U-Net baseline? Tại sao BCE + Dice?
- Câu 16 — Cấu hình Encoder-only (ViT ở bottleneck) trong code nhóm chạy thế nào?
- Câu 17 — Có bao nhiêu tham số? Vì sao TransUNet chậm hơn U-Net?
- Câu 18 — Output của U-Net và TransUNet khác nhau thế nào?
- Tóm tắt Phần 1

---

## 0. Toàn cảnh project trong một hình

```
 1 ca bệnh nhân: 4 file MRI + 1 file nhãn (.nii.gz, mỗi file 240×240×155)
        │
        ▼  ─── PHẦN 2: TIỀN XỬ LÝ ───────────────────────────────────────────────
        │  đọc thẳng từ .tar → crop vùng não → chuẩn hóa z-score → nhãn {0,1,2,4} → 3 vùng
        │  → cắt ra lát 2D (4,160,160) hoặc khối 3D (4,96,96,96) → ghép batch
        ▼
   batch ảnh (8, 4, 160, 160)
        │
        ▼  ─── PHẦN 1: MÔ HÌNH (file này) ───────────────────────────────────────
        │  U-Net (CNN encoder + CNN decoder) → feature ở 5 mức phân giải
        │     ├─► [U-Net thường]  conv 1×1 ─────────────────────────► (8, 3, 160, 160)
        │     └─► [TransUNet]     Transformer decoder, 20 query
        │                          → 20 mask + 20 nhãn lớp → gộp ──► (8, 3, 160, 160)
        ▼
   xác suất 3 vùng WT, TC, ET cho từng pixel
        │
        ▼  ─── PHẦN 3: TRAIN / ĐÁNH GIÁ ────────────────────────────────────────
           TRAIN:     so với nhãn → loss → AdamW cập nhật trọng số
           SUY LUẬN:  cả volume → ngưỡng 0.5 → nhãn 0/1/2/4 → Dice, HD95 theo WT/TC/ET
```

---

## Câu 1 ⭐ — "U-Net gốc", "TransUNet gốc", "3D TransUNet" và bản của nhóm: là gì, khác nhau ở đâu?

Có bốn cái tên rất dễ lẫn. Thực ra chúng là các "đời" nối tiếp nhau của cùng một ý tưởng:

| Tên | Công bố | Loại ảnh | Transformer đặt ở đâu | Vai trò trong project |
|---|---|---|---|---|
| **U-Net** (Ronneberger và cộng sự) | MICCAI 2015 | 2D | Không có, thuần CNN | Cái khung hình chữ U mà mọi mô hình sau đều dùng |
| **nnU-Net** (Isensee và cộng sự) | Nature Methods 2021 | 2D và 3D | Không có | U-Net kèm bộ quy tắc tự cấu hình. Là **baseline** trong paper và là **nền code** của repo gốc |
| **TransUNet** (Chen và cộng sự) | arXiv 2021 | 2D | **Encoder**: ViT 12 lớp sau ResNet-50, ở đáy chữ U | "TransUNet gốc" |
| **3D TransUNet** (Chen và cộng sự, arXiv:2310.07781) | 2023 | 3D | Tuỳ cấu hình: **encoder** (ViT ở đáy), **decoder** (Transformer decoder kiểu Mask2Former), hoặc **cả hai** | Paper nhóm tái hiện. Với BraTS, paper dùng **Decoder-only** |
| **Bản của nhóm** (`src/brats/models/`) | — | 2D và 3D thu nhỏ | Decoder-only (chính). Encoder-only và Encoder+Decoder để làm thí nghiệm | Code bạn đang chạy |

Câu để nhớ: **U-Net là cái khung, "Trans" là chỗ gắn thêm Transformer vào khung đó.** TransUNet 2021 gắn Transformer ở **đáy** chữ U (phần encoder). 3D TransUNet cho phép gắn thêm ở **đầu ra** (phần decoder). Cấu hình bạn đang dùng gắn ở đầu ra.

**Điểm hay hiểu nhầm nhất.** Trong cấu hình **Decoder-only**, "Transformer decoder" **không** thay thế nhánh phóng to (decoder CNN) của U-Net. U-Net vẫn giữ nguyên cả encoder lẫn decoder CNN. Transformer decoder là một **khối gắn thêm sau** decoder CNN, thay cho lớp conv 1×1 cuối cùng:

```
U-Net thường :  ảnh → [CNN encoder] → [CNN decoder] → conv 1×1 ─────────────────────────→ 3 vùng
Decoder-only :  ảnh → [CNN encoder] → [CNN decoder] → [Transformer decoder + 20 query] ──→ 3 vùng
                                          (giữ nguyên)      (thay cho conv 1×1)
```

Chữ "decoder" ở đây dùng theo nghĩa của Transformer (DETR, Mask2Former): một khối nhận các **query** rồi "giải mã" chúng thành từng đối tượng (ở đây là từng vùng u) bằng cách nhìn vào feature của ảnh.

---

## Câu 2 ⭐ — U-Net gốc (2015) có những lớp gì? Input, output là gì?

U-Net được thiết kế cho ảnh tế bào dưới kính hiển vi (ảnh xám 2D). Hình chữ U gồm nhánh trái đi xuống, đáy, và nhánh phải đi lên:

```
 INPUT 572×572×1                                                               OUTPUT 388×388×2
   │                                                                                  ▲
 [conv3×3+ReLU]×2 → 64 kênh ───────────── nối tắt (copy + crop) ──────────► ghép → [conv3×3+ReLU]×2 → conv1×1
   │ maxpool 2×2 (÷2)                                                         ▲ up-conv 2×2 (×2)
 [conv3×3+ReLU]×2 → 128 ──────────── nối tắt ────────────► ghép → [conv]×2 → 128
   │ maxpool                                                  ▲ up-conv
 [conv]×2 → 256 ──────────── nối tắt ────► ghép → [conv]×2 → 256
   │ maxpool                                  ▲ up-conv
 [conv]×2 → 512 ─── nối tắt ──► ghép → [conv]×2 → 512
   │ maxpool                      ▲ up-conv
 [conv]×2 → 1024 ── (đáy chữ U = bottleneck) ┘
```

| Phần | Làm gì | Gồm các lớp |
|---|---|---|
| **Encoder** (nhánh co, bên trái) | Thu nhỏ ảnh dần (÷2 mỗi tầng), tăng số kênh (64 → 1024). Học **"cái gì"** có trong ảnh (ngữ nghĩa), nhưng mất dần thông tin **"ở đâu"** | 2 conv 3×3 + ReLU, rồi max-pool 2×2 |
| **Bottleneck** (đáy) | Feature nhỏ nhất, trừu tượng nhất, mỗi điểm "nhìn thấy" vùng ảnh rộng nhất | 2 conv 3×3 |
| **Decoder** (nhánh giãn, bên phải) | Phóng to dần (×2 mỗi tầng) để ra bản đồ nhãn có cỡ bằng ảnh | up-conv 2×2, ghép với skip, 2 conv 3×3 |
| **Skip connection** (đường ngang) | Chuyển feature độ phân giải cao từ encoder sang decoder, giúp khôi phục **biên sắc nét** | ghép (concatenate) theo chiều kênh |
| **Head** (lớp cuối) | Biến feature thành điểm số cho từng lớp tại từng pixel | conv 1×1 |

- **Input:** một ảnh (bản gốc là ảnh xám, 1 kênh).
- **Output:** với **mỗi pixel**, một vector điểm số (logit) cho từng lớp. Qua softmax thành xác suất, lấy lớp lớn nhất. Nói cách khác, output là một **ảnh nhãn**. Bản 2015 cho output nhỏ hơn input (388 so với 572) vì conv không có padding. Các bản hiện đại dùng padding nên output **bằng** input.

Vì sao hình chữ U hiệu quả: encoder cung cấp ngữ cảnh ("vùng này là u"), skip connection cung cấp chi tiết vị trí ("biên u nằm đúng ở pixel này"). Decoder kết hợp hai nguồn đó.

---

## Câu 3 — U-Net trong code nhóm gồm những gì? Đi qua từng lớp với shape cụ thể

Code nằm ở `src/brats/models/unet.py`. Khối cơ bản ở `src/brats/models/blocks.py:22`.

**Khối cơ bản `ConvBlock`** (giống nnU-Net):

```
Conv 3×3 (stride s) → InstanceNorm → LeakyReLU(0.01) → Conv 3×3 → InstanceNorm → LeakyReLU(0.01)
```

- `stride=2` ở conv đầu tiên làm ảnh nhỏ đi một nửa. Cách này thay cho max-pool của U-Net 2015.
- **InstanceNorm** chuẩn hóa riêng từng ảnh, từng kênh, nên không phụ thuộc batch size. Bản 3D thường chỉ có batch 1–2, khi đó BatchNorm ước lượng thống kê rất kém.
- **LeakyReLU** giống ReLU nhưng vẫn để lại một chút gradient cho số âm, đỡ bị "nơ-ron chết".
- `padding=1` giữ nguyên kích thước sau mỗi conv (khác bản 2015).

**Đi qua mạng** với `features: [32, 64, 128, 256, 320]` và input là batch 8 lát cắt 160×160:

| # | Lớp (tên trong code) | Input | Output | Độ phân giải |
|---|---|---|---|---|
| 1 | `encoders[0]`: ConvBlock(4→32, stride 1) | (8, 4, 160, 160) | (8, 32, 160, 160) | 1/1 |
| 2 | `encoders[1]`: ConvBlock(32→64, stride 2) | (8, 32, 160, 160) | (8, 64, 80, 80) | 1/2 |
| 3 | `encoders[2]`: ConvBlock(64→128, stride 2) | (8, 64, 80, 80) | (8, 128, 40, 40) | 1/4 |
| 4 | `encoders[3]`: ConvBlock(128→256, stride 2) | (8, 128, 40, 40) | (8, 256, 20, 20) | 1/8 |
| 5 | `encoders[4]`: ConvBlock(256→320, stride 2) = **bottleneck** | (8, 256, 20, 20) | (8, 320, 10, 10) | 1/16 |
| 6 | `ups[0]`: ConvTranspose 2×2 (320→256) | (8, 320, 10, 10) | (8, 256, 20, 20) | 1/8 |
|   | ghép với skip #4 thành 512 kênh → `decoders[0]`: ConvBlock(512→256) | (8, 512, 20, 20) | (8, 256, 20, 20) | 1/8 |
| 7 | `ups[1]` + ghép skip #3 + `decoders[1]` | (8, 256, 20, 20) | (8, 128, 40, 40) | 1/4 |
| 8 | `ups[2]` + ghép skip #2 + `decoders[2]` | (8, 128, 40, 40) | (8, 64, 80, 80) | 1/2 |
| 9 | `ups[3]` + ghép skip #1 + `decoders[3]` | (8, 64, 80, 80) | (8, 32, 160, 160) | 1/1 |
| 10 | `head`: Conv 1×1 (32→3) | (8, 32, 160, 160) | **(8, 3, 160, 160)** logits | 1/1 |

Sau cùng, `sigmoid` biến 3 kênh logits thành 3 bản đồ xác suất WT, TC, ET.

- `features` có 5 phần tử = 5 tầng = 4 lần thu nhỏ, nên đáy ở 1/16 (160 → 10).
- Ở bước ghép, số kênh tăng gấp đôi (256 từ dưới đi lên + 256 từ skip = 512), rồi ConvBlock đưa về 256.
- Hàm `forward_features()` (`unet.py:48`) trả về **5 feature của nhánh decoder**, thứ tự `[1/16, 1/8, 1/4, 1/2, 1/1]`. TransUNet dùng lại đúng danh sách này (Câu 6).
- Số tham số khoảng **5.7 M** (đo bằng `scripts/smoke_test.py`, ghi trong docs/02).

**So với U-Net 2015:** có padding (output bằng input), InstanceNorm, LeakyReLU, thu nhỏ bằng conv stride 2, input 4 kênh (4 loại MRI), output 3 kênh sigmoid (3 vùng lồng nhau) thay vì softmax.

---

## Câu 4 — TransUNet gốc (2021) là gì? Transformer đặt ở đâu, làm gì?

TransUNet 2021 (*TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, cùng tác giả chính Jieneng Chen) làm trên ảnh 2D, thử nghiệm trên CT đa cơ quan (Synapse) và MRI tim (ACDC).

```
ảnh 224×224 ─► ResNet-50 (CNN) ─► feature 1/16 (14×14) ─► mỗi vị trí = 1 token → 196 token
                 │   │   │                                ─► chiếu lên 768 chiều + position embedding
                 │   │   │                                ─► 12 lớp Transformer (ViT-B/16, pretrain ImageNet)
                 │   │   │                                ─► reshape lại thành (768, 14, 14)
                 │   │   └── skip 1/8 ──┐                                  │
                 │   └────── skip 1/4 ──┼──► decoder CNN "CUP" (phóng ×2 bốn lần, ghép skip) ◄─┘
                 └────────── skip 1/2 ──┘                     └─► nhãn 224×224
```

- **Transformer ở encoder, tại đáy chữ U.** CNN trích chi tiết cục bộ trước. Transformer nhận feature đã nhỏ (14×14) và cho **mọi vị trí nhìn mọi vị trí khác** (self-attention), nên học được quan hệ **toàn cục**. Ví dụ: gan thường nằm cạnh thận phải, lách nằm bên trái.
- **Vì sao đặt ở đáy mà không ở độ phân giải cao?** Chi phí self-attention tỉ lệ với (số token)². Ở 1/16 chỉ có 196 token. Làm ở ảnh gốc 224×224 = 50 176 token thì chi phí lớn gấp khoảng 65 000 lần.
- 3D TransUNet gọi kiểu thiết kế này là **Encoder-only**.

---

## Câu 5 — 3D TransUNet có 3 cấu hình. Transformer đặt ở đâu trong mỗi cấu hình?

3D TransUNet xây trên **nnU-Net 3D**: dùng lại mạng U-Net 3D và toàn bộ quy trình tiền xử lý, train, suy luận của nnU-Net. Sau đó thử gắn Transformer vào hai chỗ:

```
(a) Encoder-only     ảnh ─► CNN enc ─► [ViT ở đáy] ─► CNN dec ─► conv 1×1 ─► nhãn          loss: CE + Dice

(b) Decoder-only ⭐   ảnh ─► CNN enc ─────────────► CNN dec ─┬─ feature 1/8, 1/4, 1/2 ─┐
                                                            └─ feature 1/1 ─────────┐  │
                                                          [Transformer decoder: 20 query, 3 lớp]
                                                            ─► 20 mask + 20 nhãn lớp ─► nhãn   loss: Hungarian

(c) Encoder+Decoder  (a) + (b): có cả ViT ở đáy lẫn Transformer decoder
```

| Cấu hình | Config của repo gốc | Config của nhóm | Params repo gốc (3D, crop 128³, đo trong docs/03) |
|---|---|---|---|
| Encoder-only | `configs/Brats/encoder_only.yaml` (ViT 12 lớp, pretrain) | `configs/transunet2d_encoder.yaml` (ViT 4 lớp, không pretrain) | 116.5 M |
| **Decoder-only** | `configs/Brats/decoder_only.yaml` | `configs/transunet2d.yaml`, `configs/transunet3d_small.yaml` | **33.7 M** |
| Encoder+Decoder | `configs/Brats/encoder_plus_decoder.yaml` (ViT 1 lớp) | bỏ dấu `#` ở dòng `vit_bottleneck` trong `transunet2d.yaml` | 41.4 M |

**Kết luận chính của paper:** mỗi loại bài toán hợp một thiết kế khác nhau. Paper so 3 cấu hình với nnU-Net trên hai bộ dữ liệu (Dice trung bình %, 3D):

| Cấu hình | Synapse (8 cơ quan, CT) | MSD Hepatic Vessel (mạch máu + u gan, CT) |
|---|---|---|
| nnU-Net (không có Transformer) | 87.33 | 66.04 |
| Encoder-only, ViT 1 lớp | 87.65 | 66.30 |
| Encoder-only, ViT 12 lớp pretrain | **88.11** | 66.35 |
| Decoder-only | 87.63 | **67.67** |
| Encoder + Decoder | 88.11 (ViT 12 lớp) | 67.24 (ViT 1 lớp) |

- **Transformer encoder** có lợi cho bài **đa cơ quan**: nhiều cơ quan lớn, quan hệ vị trí giữa chúng quan trọng (+0.8 điểm trên Synapse).
- **Transformer decoder** có lợi cho **mục tiêu nhỏ và khó** như **khối u** (+1.63 điểm trên MSD Vessel). Paper còn thấy xu hướng này ở u tụy (PDAC) và di căn não (BraTS2023-MET).
- Kết hợp cả hai **không** tốt hơn rõ rệt. Paper cho rằng lợi ích của hai phần có thể chồng lên nhau hoặc triệt tiêu nhau.
- Vì vậy paper dùng Encoder-only cho bài đa cơ quan và **Decoder-only cho khối u**. Nhóm cũng lấy Decoder-only làm cấu hình chính cho BraTS.
- Lưu ý: paper **không** có bảng so 3 cấu hình trên BraTS2021. Với BraTS, paper chỉ báo kết quả cuối cùng so với nnU-Net (Phần 3, Câu 41).

---

## Câu 6 ⭐ — Trong cấu hình Decoder-only, Transformer nằm ở đâu, nhận gì, trả ra gì?

Đây là câu quan trọng nhất. Toàn bộ `forward` của TransUNet chỉ có hai dòng (`src/brats/models/transunet.py:32-34`):

```python
feats = self.unet.forward_features(x)                  # 5 feature của decoder CNN: [1/16, 1/8, 1/4, 1/2, 1/1]
return self.decoder([feats[i] for i in self.ms_idxs],  # ms_idxs = [-4, -3, -2] → feature 1/8, 1/4, 1/2
                    self.mask_proj(feats[-1]))         # feature 1/1 → conv 1×1 → 192 kênh ("pixel embedding")
```

Nghĩa là: chạy U-Net như bình thường nhưng **không** dùng lớp conv 1×1 cuối. Thay vào đó lấy các feature trung gian của decoder CNN đưa cho Transformer decoder (`src/brats/models/transformer_decoder.py:100`).

```
 x (8, 4, 160, 160)
   │
   ▼  U-Net: encoder + decoder CNN (Câu 3), bỏ conv 1×1 cuối
   ├─ F_1/8 (8, 256, 20, 20) ──┐
   ├─ F_1/4 (8, 128, 40, 40) ──┼─► input_proj: conv 1×1 → 192 kênh + GroupNorm → trải phẳng thành chuỗi token
   ├─ F_1/2 (8,  64, 80, 80) ──┘   = "bộ nhớ" (memory) cho cross-attention:
   │                                 mức 0: (8, 400, 192)   mức 1: (8, 1600, 192)   mức 2: (8, 6400, 192)
   └─ F_1/1 (8, 32, 160, 160) ─► mask_proj conv 1×1 ─► pixel embedding E (8, 192, 160, 160)

 20 query học được: q (8, 20, 192)
   ├─ vòng 0: đoán thô ngay từ q ──────────────────────────────► lớp (8,20,4), mask (8,20,160,160)   → aux 1
   ├─ lớp Transformer 1: nhìn mức 1/8 (400 token), chỉ trong mask vòng 0  ─► lớp, mask             → aux 2
   ├─ lớp Transformer 2: nhìn mức 1/4 (1600 token), chỉ trong mask vòng 1 ─► lớp, mask             → aux 3
   └─ lớp Transformer 3: nhìn mức 1/2 (6400 token), chỉ trong mask vòng 2 ─► DỰ ĐOÁN CUỐI
                                         pred_logits (8, 20, 4), pred_masks (8, 20, 160, 160)

 Mỗi "lớp Transformer" = masked cross-attention → self-attention → FFN  (xem Câu 7 và Câu 10)
```

**Từng thành phần của Transformer decoder:**

| Thành phần | Là gì | Input → Output | Code |
|---|---|---|---|
| `input_proj[i]` | conv 1×1 + GroupNorm, đưa 3 mức feature về cùng 192 kênh | (8,256,20,20) → (8,400,192), tương tự cho 2 mức kia | `transformer_decoder.py:75, 111` |
| `level_embed` | 3 vector học được, đánh dấu token thuộc mức 1/8, 1/4 hay 1/2 | cộng vào token | `:76` |
| `pe` | mã hóa vị trí sin/cos | cỡ lưới → (8, L, 192) | `blocks.py:37` |
| `mask_proj` | conv 1×1: feature 1/1 (32 kênh) → 192 kênh | (8,32,160,160) → (8,192,160,160) | `transunet.py:28` |
| `query_feat` | **20 query** (phần nội dung) | tham số (20,192) → (8,20,192) | `:78` |
| `query_embed` | phần "vị trí" của 20 query | (8,20,192) | `:79` |
| `cross[i]` | masked cross-attention: query hỏi, token ảnh trả lời | q (8,20,192) + memory (8,L,192) → (8,20,192) | `:24` |
| `selfa[i]` | self-attention giữa 20 query | (8,20,192) → (8,20,192) | `:35` |
| `ffn[i]` | MLP 192 → 1536 → 192, chạy riêng trên từng query | (8,20,192) → (8,20,192) | `:46` |
| `class_embed` | Linear 192 → 4 (WT, TC, ET, "không có gì") | (8,20,192) → (8,20,4) | `:84` |
| `mask_embed` | MLP 3 lớp, 192 → 192 | (8,20,192) → (8,20,192) | `:85` |
| mask | tích vô hướng giữa query và pixel embedding | → (8,20,160,160) | `:91` |

**Nói bằng lời, Transformer decoder làm gì:**

1. Mỗi query là một **"ứng viên vùng"**. Lúc bắt đầu, 20 query giống hệt nhau với mọi ảnh (chỉ là 20 vector học được).
2. Ở mỗi vòng, mỗi query **nhìn vào feature của ảnh** (cross-attention), nhưng chỉ ở chỗ nó đang nghi là vùng của nó (masked), để lấy thông tin về chính ảnh này. Sau đó 20 query **trao đổi với nhau** (self-attention) để phân công, ví dụ "tôi lo ET, bạn lo TC", tránh trùng việc. Cuối cùng mỗi query tự biến đổi thêm qua FFN.
3. Sau mỗi vòng, mỗi query nói ra hai điều: **nó là vùng nào** (lớp) và **vùng đó ở đâu** (mask). Vòng sau sửa kết quả vòng trước, đi từ thô tới mịn.
4. Kết quả cuối là 20 cặp (lớp, mask). Gộp chúng lại thành 3 bản đồ WT/TC/ET (Câu 12).

Hình dung: 20 "thám tử" cùng tìm các vùng u. Vòng 0 mỗi người đoán đại một vùng. Các vòng sau, mỗi người soi kỹ hơn chỉ trong khu mình đã khoanh, bàn bạc với nhau, rồi khoanh lại chính xác hơn. Cuối cùng mỗi người báo cáo "tôi tìm được vùng X, bản đồ của nó đây", hoặc "tôi không tìm được gì".

**Tóm gọn input/output của Transformer decoder:**

- **Input:** 3 feature map đa tỉ lệ của decoder CNN (1/8, 1/4, 1/2) và 1 pixel embedding ở độ phân giải đầy đủ (1/1).
- **Output:** `{"pred_logits": (B, 20, 4), "pred_masks": (B, 20, H, W), "aux_outputs": [3 lần dự đoán trung gian]}`.

**Đối chiếu với code gốc:** `Generic_TransUNet_max_ppbp.forward` (`nn_transunet/networks/transunet3d_model.py:501`) đi đúng luồng này: lấy các feature decoder `ds_feats`, chọn `max_ms_idxs: [-4, -3, -2]`, tạo `mask_features = linear_mask_features(ds_feats[-1])`, rồi gọi `self.predictor(...)`, chính là `MultiScaleMaskedTransformerDecoder3d` (`mask2former_transformer_decoder3d.py:244`).

---

## Câu 7 — Attention hoạt động thế nào? Self-attention khác cross-attention ra sao?

Attention giống một phép **"tra cứu mềm"**:

- **Q** (query) là câu hỏi.
- **K** (key) là "tiêu đề" của từng mục thông tin.
- **V** (value) là nội dung của từng mục.

Mỗi câu hỏi so với tất cả tiêu đề (tích vô hướng Q·K, giống nhau thì điểm cao), chia cho √d cho ổn định, qua softmax thành trọng số có tổng bằng 1. Kết quả là **trung bình có trọng số** của các nội dung V:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
```

**Ví dụ nhỏ:** 1 query, 3 token, điểm Q·K/√d = [2.0, 0.5, −1.0]
→ softmax = [0.786, 0.175, 0.039]
→ output = 0.786·V₁ + 0.175·V₂ + 0.039·V₃. Query "lấy" chủ yếu thông tin từ token 1.

**Multi-head:** 192 chiều được chia thành 8 "đầu" × 24 chiều. Mỗi đầu làm attention riêng (có thể chú ý theo tiêu chí khác nhau), rồi ghép kết quả lại.

| | Self-attention | Cross-attention |
|---|---|---|
| Q lấy từ | chuỗi A | chuỗi A (20 query) |
| K, V lấy từ | **chính chuỗi A** | **chuỗi B** (token của ảnh) |
| Trong decoder của nhóm | 20 query nhìn nhau: ma trận 20 × 20 | 20 query nhìn 400 / 1600 / 6400 token ảnh: ma trận 20 × L |
| Ý nghĩa | các query phối hợp, phân công | query lấy thông tin từ ảnh |

Code cross-attention (`transformer_decoder.py:30-32`):

```python
out = self.attn(q + qpos, mem + pos, mem, attn_mask=attn_mask)[0]  # Q = query + vị trí query, K = token + vị trí, V = token
return self.norm(q + out)                                          # cộng residual rồi LayerNorm
```

**Chi phí tính toán, lý do quan trọng của thiết kế:**

- Self-attention giữa các token ảnh (như ViT) xét L × L cặp. Bản 2D ở mức 1/2 có 6400 token → khoảng 41 triệu cặp cho mỗi đầu.
- Cross-attention của decoder chỉ xét 20 × L cặp. Cũng ở mức 1/2: 20 × 6400 = 128 000 cặp, rẻ hơn 320 lần.

Vì thế ViT (encoder) chỉ đặt được ở đáy chữ U, nơi có ít token. Còn Transformer decoder dùng được cả feature độ phân giải cao (1/2), rất có lợi cho **u nhỏ**.

---

## Câu 8 — "Query" là gì? Tại sao 20 query mà chỉ có 3 vùng? Lớp "không có gì" là gì?

- Trong code, `query_feat = nn.Embedding(20, 192)` là 20 vector 192 chiều. Chúng là **tham số học được**, giống trọng số của conv. Đi kèm là `query_embed`: 20 vector "vị trí" của query.
- Mỗi query cuối cùng trả lời hai câu: **tôi là vùng nào?** (WT/TC/ET/không có gì) và **vùng đó ở đâu?** (mask).
- Lúc đầu 20 query giống nhau với mọi ảnh. Qua cross-attention, chúng "hấp thụ" thông tin của ảnh đang xét và trở thành riêng cho ảnh đó.
- **Tại sao N = 20 lớn hơn số lớp K = 3?** Paper cố ý đặt N lớn hơn nhiều so với K "để giảm nguy cơ bỏ sót" (false negative): mỗi vùng có nhiều ứng viên. Khi train, Hungarian matching chọn ứng viên tốt nhất cho mỗi vùng thật (Câu 13). 17 query còn lại được dạy trả lời "không có gì". Điều kiện bắt buộc duy nhất là N ≥ K.
- **Nhưng ablation của paper cho thấy số query gần như không ảnh hưởng.** Trên MSD Vessel (Dice trung bình %): 5 query → 67.53; 20 query → 67.67; 40 query → 67.37. Nếu giảng viên hỏi "vì sao 20", câu trả lời trung thực là: theo paper và code gốc, và paper tự kiểm tra thấy kết quả không nhạy với con số này.
- **Khởi tạo query.** Paper viết query được khởi tạo bằng 0 rồi cộng positional embedding học được. Code gốc (và code nhóm) theo Mask2Former: cả `query_feat` lẫn `query_embed` đều là tham số học được, khởi tạo ngẫu nhiên. Đây là một khác biệt nhỏ giữa lời văn của paper và code.
- **Lớp "không có gì"** (no object) là lớp thứ 4 (chỉ số 3) của `class_embed`. Khi suy luận, query "không có gì" có xác suất WT/TC/ET gần 0, nên hầu như không đóng góp vào kết quả (Câu 12).
- Trọng số của lớp "không có gì" trong loss: nhóm đặt `no_object_weight: 0.1` (giá trị mặc định của Mask2Former) để 17 query "rỗng" không lấn át 3 query thật. **Lưu ý:** config BraTS của repo gốc không đặt khóa này, nên trọng số là 1.0 (xem Phần 3, Câu 40).
- Thí nghiệm kiểm tra: `--opts model.decoder.num_queries=3` (N = K) để xem N > K có lợi không.

---

## Câu 9 — Mask và lớp của mỗi query được tính ra sao?

Hàm `_heads` (`transformer_decoder.py:88-98`):

```python
q = self.norm(q)                                    # LayerNorm
logits = self.class_embed(q)                        # (B,20,4): điểm số 4 lớp của mỗi query
masks = torch.einsum("bqc,bc...->bq...",            # tích vô hướng tại MỖI pixel:
                     self.mask_embed(q),            #   (B,20,192): "chân dung" của vùng mà query đang tìm
                     mask_features)                 #   (B,192,160,160): "chân dung" của từng pixel
                                                    # → (B,20,160,160): mask logit của mỗi query
```

- **Pixel embedding** E(v) là vector 192 số mô tả "pixel v trông như thế nào". Nó được học qua decoder CNN.
- **Mask embedding** e_q là vector 192 số mô tả "vùng mà query q đang tìm trông như thế nào".
- **Mask logit** m_q(v) = e_q · E(v). Hai vector càng cùng hướng thì logit càng lớn. Qua sigmoid thành xác suất pixel v thuộc vùng của query q.
- Mask luôn có **độ phân giải đầy đủ** (160×160), dù attention chỉ dùng feature 1/8 đến 1/2. Lý do: pixel embedding lấy từ feature 1/1.
- Lớp của query: `softmax(logits)` trên 4 lớp.

Paper viết dự đoán thô ban đầu là `Z⁰ = g(P⁰ × Fᵀ)`, với P⁰ là N query, F là feature khối cuối của U-Net (đã chiếu lên d_dec = 192 chiều), g là sigmoid rồi cắt ngưỡng 0.5. Trong code, P đi qua MLP `mask_embed` trước khi nhân.

**Cách hiểu hay nhất** (chính paper nói): mỗi query giống một **kernel conv 1×1** trong lớp head của U-Net. Khác ở chỗ: kernel của U-Net cố định sau khi train, còn "kernel" của TransUNet được **sinh riêng cho từng ảnh**, vì query đã được cập nhật bằng cross-attention trên chính ảnh đó.

---

## Câu 10 — Masked attention ("coarse-to-fine") là gì? Code làm ở đâu?

Ý tưởng: ở mỗi vòng, một query **chỉ được nhìn vào vùng mà chính nó vừa đoán** ở vòng trước.

Code (`transformer_decoder.py:93-97` và `:120-122`):

```python
m = F.interpolate(masks.float(), size=target_size, ...)  # mask 160×160 thu nhỏ về cỡ của mức sắp nhìn (vd 20×20)
attn = (m.sigmoid().flatten(2) < 0.5)                    # True = điểm nằm ngoài vùng → CẤM nhìn
attn = attn.unsqueeze(1).repeat(1, self.heads, 1, 1).flatten(0, 1).detach()  # nhân bản cho 8 đầu, không lấy gradient
...
attn = attn & ~attn.all(-1, keepdim=True)                # query có mask rỗng → cho nhìn toàn ảnh (tránh NaN)
q = self.cross[i](q, src[lvl], attn, pos[lvl], qpos)
```

- Trong cross-attention, điểm ở vị trí bị cấm được cộng −∞, nên sau softmax có trọng số 0. Công thức trong paper (mượn ý từ Mask2Former):

  ```
  Pᵗ⁺¹ = Pᵗ + Softmax( (Pᵗ·w_q)(𝓕·w_k)ᵀ + h(Zᵗ) ) × 𝓕·w_v

  h(Zᵗ(i, j, s)) = 0    nếu Zᵗ(i, j, s) = 1   (voxel nằm trong mask vòng trước)
                 = −∞   nếu ngược lại
  ```

  Trong đó Pᵗ là các query ở vòng t, 𝓕 là feature U-Net đã chiếu lên 192 chiều, w_q, w_k, w_v là ma trận chiếu. Số hạng Pᵗ cộng ở đầu là đường residual.

- **"Coarse-to-fine"** (thô tới mịn) có hai nghĩa:
  1. **Về vùng:** mask được sửa dần qua các vòng. Vòng 0 đoán thô, các vòng sau khoanh chính xác hơn.
  2. **Về độ phân giải:** vòng 1 nhìn feature 1/8 (20×20), vòng 2 nhìn 1/4 (40×40), vòng 3 nhìn 1/2 (80×80). Code chọn mức bằng `lvl = i % 3` (lần lượt xoay vòng).
- **Vì sao giúp u nhỏ?** U chỉ chiếm khoảng 1% thể tích. Cross-attention thường trải sự chú ý ra cả ảnh, nên thông tin về u bị "pha loãng" bởi 99% nền. Masked attention buộc query tập trung vào vùng ứng viên, nhờ đó tinh chỉnh biên u tốt hơn.
- `.detach()`: mask chỉ đóng vai trò "cổng", không truyền gradient qua phép so sánh `< 0.5` (phép này vốn không lấy đạo hàm được).
- Thí nghiệm: `--opts model.decoder.masked_attn=false` chuyển sang cross-attention thường (nhìn toàn ảnh).
- Code gốc: `mask2former_transformer_decoder3d.py:438` (xử lý mask rỗng) và `:495` (ngưỡng 0.5), cùng logic.

**Paper đo được bao nhiêu?** Ablation trên MSD Vessel (Dice trung bình %):

| Transformer decoder | Feature đa tỉ lệ | Masked attention | Vessel | Tumor | Trung bình |
|---|---|---|---|---|---|
| — | — | — | 63.71 | 68.36 | 66.04 (= nnU-Net) |
| ✓ | — | — | 64.19 | 69.89 | 67.04 |
| ✓ | ✓ | — | 64.37 | 70.71 | 67.54 |
| ✓ | ✓ | ✓ | 64.41 | 70.94 | **67.67** |

Đọc bảng: phần lớn lợi ích đến từ việc **có** Transformer decoder (+1.0) và dùng **feature đa tỉ lệ** (+0.5). Masked attention thêm một chút (+0.13 trung bình, +0.23 ở u). Dòng thứ hai là decoder "đơn giản": mask tính bằng tích vô hướng giữa query và feature cuối, không dùng feature đa tỉ lệ cho cross-attention.

---

## Câu 11 — Positional encoding, level embedding, query embedding để làm gì?

- **Positional encoding (PE).** Attention không biết thứ tự hay vị trí. Khi trải phẳng lưới 20×20 thành chuỗi 400 token, thông tin "token nào nằm cạnh token nào" bị mất. Vì vậy phải cộng thêm **mã hóa vị trí sin/cos** (giống DETR), tính từ tọa độ (x, y), hoặc (x, y, z) với 3D. Code ở `SinePositionalEncoding` (`blocks.py:37`). Bản 2D: 48 tần số × (sin, cos) × 2 trục = 192 kênh. Bản 3D (hidden 96): 16 × 2 × 3 = 96 kênh.
- **Level embedding.** 3 vector học được, cộng vào token để cho biết token đến từ mức 1/8, 1/4 hay 1/2.
- **Query embedding** (`qpos`). Phần "vị trí" học được của từng query, cộng vào Q và K (không cộng vào V) ở cả self-attention lẫn cross-attention.
- PE chỉ phụ thuộc kích thước lưới, nên code tính một lần rồi lưu cache (`transformer_decoder.py:104-110`).

---

## Câu 12 — Từ 20 cặp (lớp, mask) làm sao ra 3 bản đồ WT/TC/ET?

Hàm `region_probs` (`src/brats/models/__init__.py:22-28`):

```python
cls   = softmax(pred_logits)[..., :-1]                        # (B,20,3): P(WT), P(TC), P(ET) của từng query, bỏ lớp "không có gì"
masks = sigmoid(pred_masks)                                   # (B,20,H,W)
probs = einsum("bqc,bq...->bc...", cls, masks).clamp(0, 1)    # (B,3,H,W)
```

Công thức: **P_c(v) = Σ_q P(lớp c ∣ query q) · σ(m_q(v))**, với c ∈ {WT, TC, ET}.

Ví dụ tại một voxel nằm trong vùng ET:

| Query | P(WT) | P(TC) | P(ET) | P(không có gì) | σ(mask) tại voxel |
|---|---|---|---|---|---|
| q7 | 0.95 | 0.02 | 0.01 | 0.02 | 0.90 |
| q12 | 0.03 | 0.90 | 0.04 | 0.03 | 0.85 |
| q3 | 0.01 | 0.05 | 0.80 | 0.14 | 0.90 |
| 17 query còn lại | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 1 | bất kỳ |

- P_WT ≈ 0.95·0.90 + 0.03·0.85 + 0.01·0.90 ≈ 0.89
- P_TC ≈ 0.02·0.90 + 0.90·0.85 + 0.05·0.90 ≈ 0.83
- P_ET ≈ 0.01·0.90 + 0.04·0.85 + 0.80·0.90 ≈ 0.76

Cả ba đều > 0.5, nên voxel thuộc ET (nhãn 4).

**Điểm yếu cần hiểu** (liên quan trực tiếp kết quả của nhóm, docs/02): xác suất vùng = xác suất lớp × xác suất mask. Nếu không query nào "dám nhận" lớp TC, ví dụ query tốt nhất chỉ có P(TC) = 0.45, thì dù mask rất đúng (σ = 0.95), P_TC ≈ 0.45 × 0.95 ≈ 0.43 < 0.5 → mất TC. U-Net không gặp chuyện này vì mỗi vùng là một kênh sigmoid độc lập. Nhóm đã giảm hậu quả bằng "max lũy tiến" trước khi cắt ngưỡng (Phần 3, Câu 38).

Code gốc gộp y hệt: `torch.einsum("bqc,bqdhw->bcdhw", mask_cls, mask_pred)` (`nn_transunet/trainer/nnUNetTrainerV2_DDP.py:836-839`, và `inference.py:338-340` của repo gốc).

---

## Câu 13 — Hungarian matching là gì? Tại sao cần?

**Vấn đề.** Model đưa ra 20 dự đoán không có thứ tự cố định. Nhãn thật của một mẫu có tối đa 3 vùng (WT, TC, ET; vùng nào rỗng trong mẫu đó thì bỏ). Muốn tính loss phải biết **dự đoán nào phụ trách vùng nào**. Không thể gán cứng "query 0 = WT", vì ta muốn model tự chọn ứng viên tốt nhất trong 20.

**Cách làm** (`losses.py:77-104`):

1. Tính **ma trận chi phí** C kích thước 20 × K:
   `C[q, k] = 2·(−P_q(lớp của k)) + 5·BCE(mask_q, GT_k) + 5·DiceLoss(mask_q, GT_k)`.
   Chi phí thấp nghĩa là query q vừa đoán đúng lớp, vừa có mask khớp với vùng k.
2. **Thuật toán Hungarian** (Kuhn–Munkres, hàm `scipy.optimize.linear_sum_assignment`) tìm cách ghép **1–1** (mỗi vùng thật đúng 1 query, mỗi query tối đa 1 vùng) sao cho **tổng chi phí nhỏ nhất**.
3. Query được ghép sẽ học đúng lớp và mask của vùng đó. Query không được ghép sẽ học lớp "không có gì".

Ví dụ rút gọn còn 4 query:

| | WT | TC | ET |
|---|---|---|---|
| q1 | 3.2 | 3.5 | 3.9 |
| q3 | 3.5 | 2.0 | **1.1** |
| q7 | **0.8** | 3.1 | 4.0 |
| q12 | 2.9 | **0.9** | 2.2 |

Hungarian chọn q7 → WT, q12 → TC, q3 → ET, tổng chi phí 0.8 + 0.9 + 1.1 = 2.8, nhỏ nhất trong mọi cách ghép. q1 → "không có gì".

- Bước ghép **không có gradient** (`@torch.no_grad()`). Nó chỉ là bước "phân công". Gradient đi qua loss tính trên các cặp đã ghép.
- Mẫu không có u (K = 0): mọi query học "không có gì".
- Ghép **riêng** cho output cuối và từng output trung gian, trừ khi bật `loss.reuse_match=true`.
- Mẹo tốc độ trong code nhóm: gom ma trận chi phí của cả batch rồi chuyển về CPU **một lần** (docs/02: từ 2.25 s xuống 0.81 s mỗi bước).
- Nguồn gốc ý tưởng: DETR (2020) → MaskFormer, Mask2Former → 3D TransUNet. Code gốc: `HungarianMatcher3D` (`transunet3d_model.py:692`).

---

## Câu 14 — Loss của TransUNet tính thế nào?

Code `SetCriterion` (`losses.py:51-138`), bám theo `compute_loss_hungarian` của repo gốc (`transunet3d_model.py:881-945`).

Với **mỗi** output (output cuối và 3 output trung gian):

```
L_output = (2·L_cls + 5·L_bce + 5·L_dice) / 10 = 0.2·L_cls + 0.5·L_bce + 0.5·L_dice
```

| Thành phần | Tính trên | Ý nghĩa |
|---|---|---|
| L_cls | cả 20 query | cross-entropy 4 lớp. Query đã ghép → lớp đúng. Query còn lại → "không có gì" (trọng số 0.1) |
| L_bce | các query đã ghép | BCE từng pixel giữa mask dự đoán và vùng thật |
| L_dice | các query đã ghép | 1 − Dice mềm giữa mask dự đoán và vùng thật |

Loss tổng (deep supervision, theo kiểu `max_loss_cal: 'v1'` của repo gốc, `losses.py:138`):

```
L = ( L_cuối + trung bình(L_aux1, L_aux2, L_aux3) ) / 2
```

Output cuối chiếm 50% trọng số, ba output trung gian chia nhau 50% còn lại. Giám sát cả output trung gian giúp các vòng đầu cũng học đoán đúng, và vòng sau có mask tốt để làm masked attention.

**Lấy mẫu 12 544 điểm.** Để chạy nhanh, BCE và Dice (cả trong chi phí ghép lẫn trong loss) chỉ tính trên 12 544 pixel chọn ngẫu nhiên đều (`loss.num_points`). Bản 2D: 12 544 / 25 600 ≈ 49% số pixel. Bản 3D 96³: chỉ ≈ 1.4% số voxel. Repo gốc với BraTS tính trên **toàn bộ voxel** (không bật `point_rend`). Muốn giống gốc thì dùng `--opts loss.num_points=0` (chậm hơn).

**Paper viết khác code ở trọng số.** Paper (mục Decoder-only và mục Implementation Details) viết:

```
L = λ₀·(L_ce + L_dice) + λ₁·L_cls,   với λ₀ = 0.7, λ₁ = 0.3
```

(`L_ce` ở đây là BCE của mask, `L_cls` là cross-entropy phân lớp.) Tức là 0.7·BCE + 0.7·Dice + 0.3·CE. Còn code gốc và code nhóm dùng 0.5·BCE + 0.5·Dice + 0.2·CE (tỉ lệ 2 : 5 : 5 chia 10, lấy từ Mask2Former). Tỉ lệ "mask : phân lớp" gần nhau (0.7 : 0.3 ≈ 2.3 so với 0.5 : 0.2 = 2.5) nhưng không giống hệt. Khi báo cáo, nói theo **code** (thứ thực sự chạy) và ghi chú điểm khác này. Paper cũng xác nhận có deep supervision: loss áp cho output ở mọi tầng của Transformer decoder.

---

## Câu 15 — Loss của U-Net baseline? Tại sao BCE + Dice?

Code `RegionDiceBCELoss` (`losses.py:23-35`):

```
L = BCE + (1 − Dice mềm)
```

Tính cho 3 kênh WT, TC, ET rồi lấy trung bình. Dice tính "theo batch": cộng dồn trên cả batch và toàn bộ không gian.

- **BCE** phạt từng pixel riêng lẻ, cho gradient ổn định, nhưng bị nền (99%) chi phối.
- **Dice mềm** = `2·Σ(p·g) / (Σp + Σg)`, với p là xác suất (không cắt ngưỡng nên lấy đạo hàm được). Dice chỉ quan tâm vùng dương, nên không bị nền lấn át. Đây là cách chống mất cân bằng lớp.
- Kết hợp CE (hoặc BCE) với Dice là chuẩn của nnU-Net.
- Khác nnU-Net: baseline của nhóm **không có deep supervision**. nnU-Net tính loss ở nhiều mức phân giải của decoder.

---

## Câu 16 — Cấu hình Encoder-only (ViT ở bottleneck) trong code nhóm chạy thế nào?

Config `configs/transunet2d_encoder.yaml` (`vit_bottleneck: {depth: 4, hidden: 384, heads: 6}`). Code `ViTBottleneck` (`src/brats/models/vit.py`), được gọi ở `unet.py:53-54`:

```
bottleneck (8,320,10,10) ─► trải phẳng: 100 token × 320 ─► Linear 320→384 + position embedding học được (10×10)
   ─► 4 lớp Transformer encoder (self-attention 6 đầu, MLP 1536, pre-norm) ─► LayerNorm ─► Linear 384→320
   ─► reshape về (8,320,10,10) ─► CỘNG vào feature CNN (residual) ─► decoder CNN như U-Net thường
```

- 1 token = 1 vị trí của feature 1/16, tức "patch" kích thước 1×1 trên feature map. Code gốc cũng làm vậy (`nn_transunet/networks/vit_modeling.py:134`, `patch_size = 1`).
- Khi suy luận trên lát to hơn (ví dụ 144×176, lưới 9×11), position embedding được nội suy lại (`vit.py:25-26`).
- Khác repo gốc: gốc dùng ViT-B 12 lớp, 768 chiều, **pretrain ImageNet-21k** (file `R50+ViT-B_16.npz`), và **thay hẳn** feature bằng output của ViT (không cộng residual). Cấu hình Encoder+Decoder của gốc chỉ dùng ViT 1 lớp.

---

## Câu 17 — Mô hình có bao nhiêu tham số? Vì sao TransUNet chậm hơn U-Net?

| Mô hình | Số tham số | Ghi chú |
|---|---|---|
| U-Net 2D `[32..320]` | ~5.7 M | encoder ~2.8 M, decoder ~2.8 M |
| TransUNet 2D | ~8.6 M | thêm ~2.9 M cho Transformer decoder |
| TransUNet 3D thu nhỏ | ~6.2 M | kênh `[16..256]`, hidden 96 |
| Repo gốc 3D, decoder-only (128³) | 33.7 M | đo trong docs/03 |

Phần thêm ~2.9 M của Transformer decoder: mỗi lớp khoảng 0.89 M (2 khối attention × 0.15 M, cộng FFN 192 → 1536 → 192 khoảng 0.59 M), nhân 3 lớp được ~2.66 M. Phần còn lại là `input_proj`, `mask_embed` và các embedding.

**Tốc độ** (docs/02): U-Net 2D khoảng 0.31 s/bước, TransUNet 2D khoảng 0.81 s/bước. Lý do:

1. Có 4 lần dự đoán, mỗi lần sinh 20 mask (8, 20, 160, 160) thay vì 3 kênh như U-Net.
2. Hungarian matching cho 4 output ở mỗi bước, có đồng bộ với CPU.
3. Cross-attention ở mức 1/2 (6400 token).
4. Loss tính trên 4 output.

---

## Câu 18 — Output của U-Net và TransUNet khác nhau thế nào? Lớp conv 1×1 của U-Net có dùng trong TransUNet không?

| | U-Net | TransUNet |
|---|---|---|
| `model(x)` trả về | `{"logits": (B,3,H,W)}` | `{"pred_logits": (B,20,4), "pred_masks": (B,20,H,W), "aux_outputs": [3 dict]}` |
| Đổi sang xác suất 3 vùng | `sigmoid(logits)` | Σ xác suất lớp × sigmoid(mask) (Câu 12) |
| Loss | `RegionDiceBCELoss` | `SetCriterion` (Hungarian) |

Hàm `region_probs()` đưa cả hai về cùng dạng `(B, 3, H, W)`, nên `inference.py` và `evaluate.py` dùng chung cho mọi model.

`self.unet.head` (conv 1×1) vẫn được tạo trong TransUNet vì TransUNet dùng lại class `UNet`, nhưng nó **không bao giờ được gọi**: `TransUNet.forward` chỉ gọi `forward_features()`. Lớp này chỉ có 99 tham số, không nhận gradient, không ảnh hưởng kết quả. Ở repo gốc, với `disable_ds: True`, các đầu ra conv của CNN không được tạo; dự đoán chỉ đến từ Transformer decoder.

---

## Tóm tắt Phần 1

1. **U-Net** = encoder (thu nhỏ, học "cái gì") + decoder (phóng to, khôi phục "ở đâu") + skip connection (giữ chi tiết biên).
2. **TransUNet 2021** đặt ViT ở **đáy** chữ U (encoder). **3D TransUNet** cho thêm lựa chọn đặt **Transformer decoder** ở đầu ra. Với BraTS, paper chọn Decoder-only.
3. **Decoder-only** = U-Net giữ nguyên + 20 query dùng masked cross-attention trên feature 1/8, 1/4, 1/2, qua 3 vòng từ thô tới mịn.
4. Mỗi query cho ra (lớp, mask). Xác suất vùng c = Σ P(lớp c) × mask. Khi train, Hungarian matching ghép query với vùng thật để tính loss.
5. Loss mỗi output trong code = 0.2·CE + 0.5·BCE + 0.5·Dice (paper viết λ₀ = 0.7 cho BCE + Dice, λ₁ = 0.3 cho CE). Loss tổng = (output cuối + trung bình 3 output trung gian) / 2.

→ Đọc tiếp **Phần 2**: [`06_HOI_DAP_P2_DU_LIEU_2D_3D.md`](06_HOI_DAP_P2_DU_LIEU_2D_3D.md)
