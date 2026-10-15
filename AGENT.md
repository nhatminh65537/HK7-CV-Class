# AGENT.md — Ngữ cảnh project cho các phiên AI

File này ghi những điều **ít thay đổi** về project: mục tiêu, ràng buộc, kiến thức nền, quy ước và các bẫy đã gặp. Những thứ hay đổi (kết quả, tiến độ, siêu tham số, danh sách file) **không** ghi ở đây; mục 4 chỉ chỗ xem.

Nếu file này mâu thuẫn với code hoặc log thì **code và log đúng**. Khi đó hãy sửa file này luôn.

## 1. Project là gì

- Bài tập nhóm môn Thị giác máy tính: **tái hiện paper 3D TransUNet** (arXiv:2310.07781) cho bài toán **phân vùng u não trên bộ BraTS2021**. Yêu cầu của môn: giữ pipeline và dữ liệu của paper, train lại mô hình, báo cáo kết quả cho giảng viên hướng dẫn.
- Mục tiêu thực tế: chạy được, số liệu trung thực, **so sánh công bằng U-Net với TransUNet trong cùng điều kiện**, và giải thích được mọi chỗ làm khác paper. Không đặt mục tiêu đạt số của paper.
- Ràng buộc chi phối mọi quyết định: máy train là laptop **Windows 11, một GPU 4 GB (RTX 3050 Ti), RAM 16 GB**, ổ đĩa hạn chế. Vì vậy nhóm **tự cài đặt lại bằng PyTorch + MONAI** thay vì chạy code gốc (code gốc cần Linux và GPU lớn), và làm **bản 2D trước, bản 3D thu nhỏ sau**.

## 2. Làm việc với người dùng

- Trao đổi, tài liệu và comment trong code đều bằng **tiếng Việt**. Thuật ngữ kỹ thuật giữ tiếng Anh (query, attention, patch, skip connection…).
- Từ dùng thống nhất: **ca** = một bệnh nhân (case); **lát** = lát cắt 2D; **khối** = patch 3D; **vùng** = region WT/TC/ET; **nhãn** = nhãn gốc 0/1/2/4.
- Người dùng là sinh viên đang học môn này: giải thích từ cơ bản, gắn với **shape tensor cụ thể**, chỉ rõ file và hàm trong code.
- Mọi con số đều có thể bị giảng viên hỏi lại. **Không bịa số.** Ghi rõ nguồn (paper, code gốc, hay log của nhóm) và tách điều đã kiểm chứng khỏi giả thuyết.
- Không commit, không push khi chưa được yêu cầu.
- Không tự chạy train dài: chỉ có một GPU và mỗi lần train tính bằng giờ. Hãy kiểm tra thay đổi theo mục 9, rồi đưa lệnh để người dùng tự chạy.
- Không cài lại hay nâng cấp gói trong `.venv` khi chưa hỏi. PyTorch phải là bản CUDA khớp driver (xem tài liệu setup trong `docs/`).
- Tài liệu mới đặt trong `brats-transunet/docs/`, theo cách đánh số và đặt tên của các file đã có.

## 3. Bố cục workspace (theo vai trò)

| Đường dẫn | Vai trò | Git | Quy tắc |
|---|---|---|---|
| `brats-transunet/` | Code do nhóm tự viết lại. Nơi làm việc chính | theo dõi | sửa ở đây |
| `Repo/3D-TransUNet/` | Code gốc của paper (bản clone, commit `9f18182`) | bỏ qua | chỉ đọc để đối chiếu; không sửa, không tìm cách chạy trên Windows |
| `Doc/` | Paper, dạng mã nguồn LaTeX nén `.tar.gz` (không phải PDF) | theo dõi | giải nén ra thư mục tạm ngoài repo; đọc `tmi.tex` và `table_figures/*.tex` |
| `Data/` | BraTS2021: một file `.tar` khoảng 13 GB và 2 file tar mẫu | bỏ qua | **không giải nén**; code đọc thẳng từ tar |

Bên trong `brats-transunet/` (mô tả theo vai trò; hãy liệt kê thư mục để biết file hiện có):

- `configs/`: file YAML. Đây là nguồn sự thật cho mọi siêu tham số.
- `src/brats/`: thư viện, gồm đọc dữ liệu, tiền xử lý, dataset, mô hình, loss, metric, suy luận, vẽ hình.
- `scripts/`: công cụ chạy lẻ (đánh chỉ mục tar, kiểm tra dữ liệu, chia split, smoke test, vẽ log…).
- `train.py`, `evaluate.py`: hai điểm vào chính.
- `data/`: file nhỏ do script sinh ra (chỉ mục tar, danh sách split dạng JSON). Không chứa ảnh.
- `runs/<tên>/`: kết quả mỗi lần train (checkpoint, `log.csv`, thư mục `eval_<split>/`).
- `docs/`: tài liệu tiếng Việt, đánh số theo thứ tự nên đọc.
- `.venv/`: môi trường ảo (uv, Python 3.11). Luôn loại khỏi mọi lệnh tìm kiếm.

## 4. Trạng thái hiện tại xem ở đâu

| Cần biết | Nguồn sự thật |
|---|---|
| Siêu tham số đang dùng | `brats-transunet/configs/*.yaml`. Config thực tế của một lần train được lưu trong checkpoint của lần đó |
| Kết quả, đường cong train | `brats-transunet/runs/<tên>/log.csv` và `runs/<tên>/eval_<split>/metrics.csv` |
| Đã làm tới đâu, nhận xét về kết quả | mục "Nhật ký tiến độ" trong `brats-transunet/docs/` (tìm chuỗi đó), cùng với `git log` |
| Lệnh chạy và tham số | `brats-transunet/README.md`, tài liệu tra cứu lệnh trong `docs/`, hoặc `python <script> --help` |
| Ca nào thuộc train / val / test | `brats-transunet/data/splits*.json` |
| Bản của nhóm khác paper và code gốc ở đâu | bảng so sánh trong bộ hỏi đáp ở `docs/` (tìm "khác repo gốc") |
| Giải thích khái niệm đã viết cho người dùng | bộ hỏi đáp trong `docs/` (tên file có `HOI_DAP`) |

Số liệu trong `docs/` được viết ở nhiều thời điểm nên có thể cũ hơn code. Trước khi trích một con số, hãy kiểm tra lại bằng code hoặc log.

## 5. Kiến thức nền (không đổi theo project)

### 5.1 Dữ liệu BraTS2021

- 1251 ca có nhãn. Mỗi ca gồm 5 file `.nii.gz`: `t1`, `t1ce`, `t2`, `flair`, `seg`. Mỗi file 240×240×155, voxel 1 mm³, đã xoá hộp sọ và căn chỉnh trùng khít; nền đúng bằng 0.
- Nhãn trong `seg`: 0 = nền, 1 = NCR (hoại tử), 2 = ED (phù nề), 4 = ET (tăng cường). Không có nhãn 3.
- BraTS chấm theo 3 vùng lồng nhau: **WT** = 1∪2∪4 ⊃ **TC** = 1∪4 ⊃ **ET** = 4. Model dự đoán thẳng 3 vùng này bằng 3 kênh sigmoid, không dự đoán nhãn 1/2/4.
- Rất mất cân bằng: u chỉ chiếm khoảng 1% thể tích; 33 ca không có ET, 6 ca không có TC. Vì vậy dùng Dice thay cho accuracy, và phải ép lấy mẫu có u khi train.
- Thứ tự cố định trong code: kênh ảnh `[t1, t1ce, t2, flair]`, kênh vùng `[WT, TC, ET]`.

### 5.2 Paper

- Ý tưởng: lấy nnU-Net 3D làm nền rồi gắn Transformer vào hai chỗ. **Encoder**: ViT đặt ở đáy chữ U. **Decoder** (kiểu Mask2Former): N query học được; mỗi tầng dùng masked cross-attention trên feature đa tỉ lệ của decoder CNN để tinh chỉnh từ thô tới mịn; mỗi query cho ra một mask và một nhãn lớp; train bằng Hungarian matching, có deep supervision.
- Ba cấu hình: Encoder-only (hợp bài đa cơ quan), **Decoder-only (hợp khối u, dùng cho BraTS)**, Encoder+Decoder (không tốt hơn).
- Thiết lập BraTS trong paper: crop 128³, batch 2, AdamW lr 3e-4, lịch cosine, 5 lần giảm mẫu mỗi trục, 20 query, 3 tầng tinh chỉnh, chiều query 192, augmentation của nnU-Net, 5-fold theo cách chia của đội hạng 1 BraTS21, train trên 1 GPU RTX 8000.
- Kết quả BraTS2021 trong paper (Dice %, thứ tự ET / TC / WT / trung bình): nnU-Net 88.05 / 91.92 / 93.79 / 91.25; nnU-Net Large 88.23 / 92.35 / 93.83 / 91.47; **3D TransUNet 88.85 / 92.48 / 93.90 / 91.74**.
- Paper **không** có ablation trên BraTS. Ablation của decoder làm trên MSD Hepatic Vessel: số query 5 / 20 / 40 cho kết quả gần như nhau; masked attention chỉ thêm khoảng 0.1 điểm Dice.

### 5.3 Code gốc (`Repo/3D-TransUNet`)

- Là bản fork của nnU-Net v1. Trainer gọi cứng backend NCCL nên chỉ chạy được trên Linux.
- Chỗ cần đọc: mạng, matcher và loss ở `nn_transunet/networks/transunet3d_model.py`; Transformer decoder ở `nn_transunet/networks/mask2former_modeling/transformer_decoder/mask2former_transformer_decoder3d.py`; vòng train và cách gộp loss ở `nn_transunet/trainer/nnUNetTrainerV2_DDP.py`; config ở `configs/Brats/`; suy luận ở `inference.py`; tính Dice ở `measure_dice.py`.
- Chế độ dự đoán theo vùng WT/TC/ET chỉ bật khi **tên file config** chứa chuỗi `500Region`.

### 5.4 Chỗ paper viết khác code gốc (dễ nói sai khi báo cáo)

- **Trọng số loss.** Paper: 0.7·(BCE + Dice) + 0.3·CE. Code: tỉ lệ CE : BCE : Dice = 2 : 5 : 5 rồi chia 10; loss tổng = (output cuối + trung bình các output phụ) / 2.
- **Phần cứng.** Paper ghi 1 GPU, batch 2. Config trong repo ghi 125 epoch kèm chú thích "8 cards".
- **Query.** Paper nói khởi tạo bằng 0 rồi cộng positional embedding. Code dùng embedding học được.
- **Các lựa chọn của code gốc cho BraTS:** không giảm trọng số lớp "no object" (config không đặt `no_object_weight`); loss của mask tính trên toàn bộ voxel; attention ép chạy fp32; augmentation rất mạnh ("insaneDA"); 33% mẫu ép có foreground; suy luận sliding window bước 0.5, không TTA; `measure_dice.py` cho Dice = 0 khi dự đoán hoặc nhãn rỗng.
- Khi trình bày: nói theo **code thực sự chạy**, và ghi chú chỗ paper viết khác.

## 6. Quy ước kiến trúc trong code của nhóm

Phần này mô tả ý đồ thiết kế. Trước khi dựa vào một chi tiết, hãy mở code kiểm tra. Nếu thay đổi một quy ước, hãy cập nhật mục này.

- **Một code cho cả 2D và 3D**, chọn bằng `model.spatial_dims` trong config. Không tách thành hai nhánh code.
- **Luồng dữ liệu:** đọc ca → crop theo vùng não (voxel khác 0) → z-score từng kênh, chỉ tính trên voxel não, nền giữ bằng 0 → đổi nhãn sang 3 vùng. Bbox, shape gốc và affine được giữ lại để trả kết quả về 240×240×155.
- **Đọc thẳng từ `.tar`** qua một file chỉ mục (vị trí byte của từng file). Không ghi dữ liệu trung gian ra đĩa, trừ cache `.npz` tùy chọn.
- **Dữ liệu train là một dòng vô hạn:** mỗi worker giữ vài ca trong RAM và cắt mẫu ngẫu nhiên, có ép một tỉ lệ mẫu chứa u. Một "epoch" là **số bước cố định** (kiểu nnU-Net), không phải một lượt qua dữ liệu.
- **Model nhận** `(B, 4, *không_gian)` **và trả về một dict.** U-Net trả `logits` gồm 3 kênh vùng. TransUNet trả `pred_logits` (mỗi query: 3 lớp vùng + lớp "no object"), `pred_masks` (mỗi query một mask) và `aux_outputs` (dự đoán của các tầng trung gian). Các cờ cải tiến trong `model.decoder` (mặc định tắt, docs/08) đổi dict này: `fixed_queries` bỏ `pred_logits` (query k = vùng k); `lazy_mask` khi train trả `mask_embed`, `mask_bias`, `mask_pix` thay cho `pred_masks`.
- **Phương án cải tiến là config mới** kế thừa config baseline bằng `base:`; không sửa config baseline để đổi phương án.
- **`region_probs()` là nơi duy nhất** đổi output của model thành xác suất 3 vùng `(B, 3, *không_gian)`. Suy luận và đánh giá chỉ đi qua hàm này, nên model mới chỉ cần tuân theo giao diện đó.
- **Hậu xử lý:** ép lồng nhau ET ⊂ TC ⊂ WT → nhãn 0/1/2/4 → đặt lại vào khối gốc → lưu NIfTI với affine gốc.
- **Config:** YAML kế thừa bằng khóa `base:` (gộp sâu); ghi đè từ dòng lệnh bằng `--opts a.b=giá_trị`. Checkpoint lưu kèm config, nên `evaluate.py` dựng lại model từ chính checkpoint.
- **Gói `brats` không cài bằng pip.** Các script tự thêm `src/` vào `sys.path`; script mới làm theo cách đó.

## 7. Nguyên tắc thí nghiệm và báo cáo

- Chia dữ liệu **theo bệnh nhân**, seed cố định, lưu thành file. Các mô hình đem so với nhau phải dùng **cùng một file split**.
- Tập val dùng để chọn checkpoint, ngưỡng và siêu tham số. Tập test chỉ dùng cho con số báo cáo cuối cùng.
- So sánh công bằng: mỗi lần chỉ đổi **một** yếu tố; giữ nguyên split, seed, số epoch; mỗi thí nghiệm một `output_dir` riêng.
- Dice tính trên **cả volume của từng ca**, riêng cho WT, TC, ET, theo quy ước BraTS (nhãn rỗng: dự đoán rỗng thì 1, có dự đoán thì 0). Báo cả trung bình, trung vị và số ca hỏng.
- Không đem số của bản 2D hay 3D thu nhỏ so trực tiếp với số của paper: khác số chiều, cách chia dữ liệu, số epoch và augmentation.
- Mỗi chỗ làm khác paper hoặc code gốc phải được ghi lại kèm lý do: do phần cứng, hay do chủ ý.

## 8. Bẫy đã gặp

- **Tiếng Việt và bảng mã trên Windows.** Các script in thông báo tiếng Việt. Khi stdout không phải console (chạy qua công cụ của agent, pipe, hoặc chuyển hướng ra file), Python dùng cp1252 và dừng với `UnicodeEncodeError`. Đặt `PYTHONUTF8=1` trước khi chạy (PowerShell: `$env:PYTHONUTF8 = "1"`). Trong code, mở file text luôn kèm `encoding="utf-8"`.
- **Đường dẫn có dấu cách** (`Learning Projects`): luôn bọc trong dấu nháy. Trong YAML viết dấu `/`.
- **Chạy lệnh từ thư mục `brats-transunet/`:** các đường dẫn `data/`, `runs/`, `figures/` là tương đối.
- **Windows tạo tiến trình con kiểu spawn.** Script có DataLoader phải nằm trong `if __name__ == "__main__":`. Object gửi sang worker phải pickle được, tức không giữ file handle đang mở.
- **Gộp config:** ghi `khóa: {}` ở file con không xoá được giá trị thừa kế từ `base.yaml`. Muốn tắt một khóa dạng dict, đặt nó bằng `null`.
- **Tar mẫu:** hai file tar mẫu trong `Data/` chứa các ca đã có trong tar lớn. Không khai báo chúng cùng tar lớn; chỉ dùng để thử code.
- **`.venv/` có hàng chục nghìn file.** Tìm kiếm mà không loại nó ra sẽ tràn kết quả.
- **Checkpoint và git.** `runs/` (kể cả file `.pt`) đang được git theo dõi, và đã có checkpoint sát mức 100 MiB mà GitHub chặn. Kiểm tra kích thước trước khi commit checkpoint mới.

## 9. Kiểm tra nhanh một thay đổi

Chạy từ `brats-transunet/`. Agent có thể gọi thẳng `.venv/Scripts/python.exe` mà không cần kích hoạt môi trường.

```powershell
$env:PYTHONUTF8 = "1"
python scripts/smoke_test.py --config configs/<tên>.yaml                           # dựng model, vài bước với dữ liệu giả, đo VRAM
python train.py --config configs/debug_samples.yaml --opts output_dir=runs/_tmp    # 2 ca mẫu, vài chục bước: thử cả pipeline
python evaluate.py --ckpt runs/_tmp/best.pt --split test --limit 2
```

- Luôn đặt `output_dir` khác các thư mục run đã có (vì `runs/` được git theo dõi), và xoá thư mục thử sau khi xong.
- Nếu tên script hoặc tham số đã đổi, `brats-transunet/README.md` là chuẩn.

## 10. Bảo trì file này

- Chỉ ghi điều còn đúng sau nhiều tuần: mục tiêu, ràng buộc, kiến thức nền, quy ước, bẫy. Không ghi kết quả, tiến độ, số dòng, số tham số, thời gian chạy.
- Thứ hay đổi thì ghi **chỗ xem** (mục 4), không chép giá trị vào đây.
- Phát hiện một câu ở đây đã sai, hoặc vừa đổi một quy ước ở mục 6: sửa ngay trong cùng lần thay đổi.
- Giữ file dưới khoảng 200 dòng. Chi tiết dài để trong `brats-transunet/docs/`.
- Rà soát lần cuối: 2026-10-13, đối chiếu với code của nhóm, paper và code gốc.
