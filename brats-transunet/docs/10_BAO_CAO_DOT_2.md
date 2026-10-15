# 10 — Báo cáo đợt 2 (13–15/10/2026): baseline mới, phương án cải tiến, kết quả 2D

> Tóm tắt những gì nhóm đã làm và kết quả đợt này, để báo cáo với giảng viên.
> Chi tiết và lý do: ý tưởng ở `08_Y_TUONG_CAI_THIEN.md`; lệnh, số liệu đầy đủ và nhận xét ở `09_HUONG_DAN_CHAY_TIEP.md`.
>
> Nguồn số liệu: **[log]** = `runs/*/log.csv`, `runs/*/eval_test/metrics.csv` của nhóm; **[đo]** = `figures/bench_2d.csv`; **[nguồn]** = paper khác, link ở docs/08 mục 9.

## 1. Mục tiêu đợt này

Giảng viên yêu cầu nhóm **cải thiện hơn paper**: tăng độ chính xác, hoặc nếu không được thì tăng hiệu năng (ít tham số hơn, nhanh hơn, ít bộ nhớ hơn).

## 2. Hướng đã chọn và lý do

Nhóm chọn cải thiện **chi phí của Transformer decoder**, không nhắm vào độ chính xác tuyệt đối:

- **BraTS2021 đã bão hòa** **[nguồn]**. Benchmark nnU-Net Revisited (MICCAI 2024) cho các phương pháp mạnh cùng nằm quanh 91.1–91.5 Dice; độ lệch giữa các fold của cùng một phương pháp (0.66 điểm) lớn hơn mức chênh paper báo (+0.49).
- **Chính paper cho thấy decoder đắt** **[nguồn]**. Bản tạp chí (Medical Image Analysis 2024): Decoder-only thêm khoảng 9% tham số nhưng gần gấp đôi bộ nhớ GPU so với nnU-Net. Tác giả cũng viết rằng bản Decoder-only cần train lâu hơn.
- **Đo trên máy nhóm** **[đo]**: ở khối 128³ (kích thước paper dùng), TransUNet gốc tràn 4 GB VRAM; lần cấp phát thất bại đúng bằng tensor chiếu feature lên 96 kênh ở độ phân giải gốc.

## 3. Baseline đã sửa

Hai lần train đầu (140 ca, 20 epoch) cho TransUNet 2D kém U-Net 2D 5 điểm Dice. Nhóm chốt lại baseline (docs/08 mục 6):

| Hạng mục | Cũ | Mới |
|---|---|---|
| Dữ liệu | 140 / 30 / 30 ca | 876 / 188 / 187 ca (đủ 1251 ca, seed 42) |
| Số epoch | 20 | 50 cho mọi model |
| `no_object_weight` | 0.1 | 1.0, giống config BraTS gốc |
| Validation khi train | 10 ca (3D: 5 ca) | 30 ca |

Kết quả cũ không còn so được với kết quả mới vì đổi split.

## 4. Ba phương án cải tiến

Cài dưới dạng cờ trong `model.decoder`, mặc định tắt; mỗi phương án là một config mới kế thừa baseline. Code với mọi cờ tắt cho output, loss và gradient **trùng hoàn toàn** code cũ (đã kiểm tra trên CPU).

| Mã | Phương án | Thay đổi | Cần train lại? |
|---|---|---|---|
| A | Baseline TransUNet Decoder-only | 20 query, Hungarian matching | có |
| B | A + tính mask "lười" | Không tạo tensor `(B, hidden, *S)`; chuyển conv 1×1 sang phía query. Cùng trọng số, cùng kết quả | không |
| C | 3 query cố định | Query k luôn là vùng k (WT, TC, ET); bỏ Hungarian và class head; loss Dice + BCE như U-Net. Bật luôn B | có |
| E | C + decoder gọn | FFN ×2 thay vì ×8, 3 tầng tinh chỉnh dùng chung trọng số | có |

## 5. Kết quả 2D

Test 187 ca, checkpoint `best.pt`, quy ước Dice của BraTS **[log]**.

| Lượt | WT | TC | ET | Trung bình (TB / trung vị) | HD95 WT (trung vị) | Ca < 0.5 |
|---|---|---|---|---|---|---|
| U-Net 2D | 0.911 | 0.873 | 0.830 | 0.871 / 0.918 | 3.6 mm | 8 |
| A | 0.912 | 0.873 | 0.842 | 0.876 / 0.916 | 5.7 mm | 5 |
| B | 0.912 | 0.873 | 0.842 | 0.876 / 0.916 | 5.7 mm | 5 |
| C | 0.911 | 0.881 | 0.834 | 0.875 / 0.924 | 4.2 mm | 6 |
| E | 0.911 | 0.880 | 0.833 | 0.875 / 0.925 | 4.4 mm | 7 |
| U-Net 3D thu nhỏ | 0.896 | 0.872 | 0.823 | 0.864 / 0.917 | 4.6 mm | 8 |

So sánh trên từng ca, trung bình 3 vùng (khoảng tin cậy 95% bằng bootstrap) **[log]**:

| So sánh | Chênh (điểm) | Khoảng tin cậy | Kết luận |
|---|---|---|---|
| A − U-Net 2D | +0.43 | −0.12 đến +1.08 | chưa phân biệt; cùng xu hướng paper (+0.49) |
| B − A | 0.00 | | trùng, đúng thiết kế |
| C − A | −0.03 | −0.51 đến +0.40 | không kém A |
| C − U-Net 2D | +0.40 | +0.05 đến +0.79 | hơn, Wilcoxon p = 0.004 |
| E − A | −0.09 | −0.71 đến +0.45 | chưa kết luận được |
| U-Net 3D − U-Net 2D | −0.76 | −1.37 đến −0.18 | kém |

Chi phí so với A **[đo]** (cùng một lần đo; chỉ dùng tỉ lệ vì máy lúc đo có thể không ở chế độ High performance):

| Lượt | Tham số | Bước train | VRAM train | Suy luận | VRAM suy luận |
|---|---|---|---|---|---|
| B | 8.57M | −10% | −19% | −17% | −23% |
| C | 8.56M | −33% | −23% | −24% | −31% |
| E | 6.34M (−26%) | −34% | −27% | −24% | −35% |

## 6. Nhận xét chính

1. **Bản tái hiện đúng xu hướng paper** nhưng chênh lệch nằm trong mức nhiễu, đúng như dự đoán từ việc BraTS2021 đã bão hòa.
2. **C là kết quả cải tiến chính:** Dice không kém A, hơn U-Net có ý nghĩa thống kê, train nhanh hơn khoảng 1/3, suy luận nhanh hơn khoảng 1/4, VRAM ít hơn khoảng 1/4, HD95 trung vị tốt hơn A.
3. **B là cải tiến "miễn phí":** cùng trọng số, cùng dự đoán, ít bộ nhớ và thời gian hơn. Ở 2D lợi ích nhỏ; ở 3D lợi ích lớn hơn nhiều (chưa đo).
4. **E chỉ thêm giảm tham số** (−26% ở 2D, khoảng −6% ở 3D theo ước lượng) nhưng chưa chứng minh được không kém A, và có một lần Dice val tụt mạnh giữa lúc train (giả thuyết: do dùng chung trọng số cho 3 tầng). Không đưa sang 3D.
5. **Lợi thế ET của A gần như đến từ 2 ca.** Test có 5 ca nhãn thật không có ET. A đoán đúng 2 ca, mọi model khác sai cả 5; mỗi ca như vậy làm ET trung bình đổi 0.53 điểm. Hậu xử lý ET sẽ tác động đúng vào các ca này (docs/09 mục 8).
6. **Ở 2D, train thêm epoch không có khả năng đổi kết luận:** Dice val 10 epoch cuối chỉ tăng 0.001–0.005. **Ở 3D thì có thể:** U-Net 3D thu nhỏ kém U-Net 2D, nhiều khả năng do cấu hình 3D bị bó chặt (16 kênh đầu, batch 1, khối 96³). Báo cáo nên nêu đây là giới hạn của cấu hình thu nhỏ.

## 7. Giới hạn cần nói rõ

- Bản 2D và 3D thu nhỏ **không so trực tiếp** với số tuyệt đối của paper (khác số chiều, cách chia dữ liệu, số epoch, augmentation).
- Mỗi cấu hình mới train **một seed**; mức dao động giữa các lần train chưa đo.
- Bảng chi phí của paper mới đọc qua đoạn trích tìm kiếm; cần mở bản gốc để đối chiếu.
- Số đo thời gian bị ảnh hưởng bởi chế độ nguồn của laptop; chỉ dùng tỉ lệ trong cùng một lần đo.

## 8. Việc tiếp theo

| Việc | Trạng thái |
|---|---|
| A-3D (TransUNet 3D baseline) | đang train |
| Benchmark 3D ở 96³ và 128³, chạy lại benchmark 2D ở chế độ High performance | chưa chạy |
| B-3D (đánh giá lại A-3D với mask lười) | sau A-3D |
| C-3D | sau A-3D |
| Hậu xử lý ET (chọn ngưỡng trên val, áp cho mọi model) | chưa cài |
| Seed thứ hai cho A và C ở 2D; train tiếp thêm epoch | tùy chọn |

Lệnh: docs/09 mục 7.3.
