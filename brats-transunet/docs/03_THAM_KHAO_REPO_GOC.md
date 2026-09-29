> **Tài liệu tham khảo.** File này phân tích repo gốc `3D-TransUNet` (nnU-Net v1) và cách chạy nó trên GPU lớn. Nhóm **không đi theo hướng này** (xem `02_LO_TRINH_VA_SETUP.md`), nhưng các mục 0, 1.2–1.3 và 5 hữu ích khi đọc code gốc hoặc bị hỏi vì sao không dùng lại repo.

# Lộ trình tái hiện 3D TransUNet trên BraTS2021

> Paper: *3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers* (Chen et al., arXiv:2310.07781)
> Repo gốc: `Beckschen/3D-TransUNet` (commit `9f18182`) · Dataset: BraTS2021 Task 1 (1251 ca MRI có nhãn)
> Mục tiêu: giữ nguyên pipeline và dữ liệu, train lại mô hình, so kết quả với Bảng BraTS2021 trong paper.

---

## 0. Tóm tắt nhanh (dùng để nói lại khi báo cáo)

**Bài toán.** Phân vùng khối u não 3D từ MRI 4 kênh (T1, T1ce, T2, FLAIR), kích thước 240×240×155, voxel 1 mm³.
Nhãn gốc: `1` = NCR (hoại tử), `2` = ED (phù nề), `4` = ET (vùng tăng cường). Đánh giá theo 3 **region** lồng nhau:
WT = 1+2+4, TC = 1+4, ET = 4.

**Ý tưởng paper.** Mở rộng TransUNet 2D thành 3D, xây trên **nnU-Net** (tự cấu hình tiền xử lý, patch, số tầng pooling). Có 3 cấu hình:

| Cấu hình | Thành phần | Loss | Dùng cho |
|---|---|---|---|
| Encoder-only | CNN → ViT (1 hoặc 12 lớp, pretrain ImageNet-21k) ở bottleneck, decoder U-Net thường | CE + Dice | Đa cơ quan (Synapse) |
| **Decoder-only** | Encoder CNN; decoder CNN + **Transformer decoder** kiểu Mask2Former: 20 *organ query*, cross-attention với feature đa tỉ lệ của U-Net, **coarse-to-fine masked attention** (3 vòng) | Hungarian matching: 0.7·(BCE+Dice) + 0.3·CE phân lớp | **Khối u nhỏ (BraTS, MSD Vessel, tụy)** |
| Encoder+Decoder | Cả hai | Hungarian | Không tốt hơn rõ ràng |

Kết luận chính: encoder Transformer có lợi cho bài toán cần ngữ cảnh toàn cục (nhiều cơ quan); decoder Transformer có lợi cho mục tiêu nhỏ và khó (u).

**Thiết lập BraTS trong paper (Bảng implement):** crop 128³, batch 2/GPU, AdamW, lr 3e-4, warmup + cosine, downsample [5,5,5], 20 query, 3 tầng C2F, augmentation mặc định của nnU-Net. 5-fold CV. Config trong repo ghi `max_num_epochs: 125` với 8 GPU.

**Kết quả cần đối chiếu (Dice %, 5-fold CV):**

| Method | ET | TC | WT | Avg |
|---|---|---|---|---|
| nnU-Net | 88.05 | 91.92 | 93.79 | 91.25 |
| nnUNet-Large (hạng 1 BraTS21) | 88.23 | 92.35 | 93.83 | 91.47 |
| **3D TransUNet** | **88.85** | **92.48** | **93.90** | **91.74** |

---

## 1. Những gì đã kiểm tra

### 1.1 Dữ liệu (`Data/`)

| File | Tình trạng |
|---|---|
| `BraTS2021_Training_Data.tar.fdmdownload` | **Đang tải** (FDM, ~13.4 GB, file còn vùng trống). Chưa đọc được hết, **không giải nén lúc này** |
| `BraTS2021_00495.tar`, `BraTS2021_00621.tar` | 2 ca mẫu (~10 MB), đọc được. Có thể trùng với ca trong tar lớn, không ảnh hưởng |

- Phần đầu tar lớn có cấu trúc `./BraTS2021_XXXXX/BraTS2021_XXXXX_{flair,seg,t1,t1ce,t2}.nii.gz` (bản Kaggle của BraTS2021 Task 1).
- Mỗi ca: 5 file NIfTI, shape `(240,240,155)`, spacing 1 mm, ảnh `int16`, nhãn `uint16` ∈ {0,1,2,4}. Ảnh đã skull-strip và co-register.
- Vùng u rất nhỏ: ca 00495 chỉ có ~1.1% voxel là u. Ca 00621 **không có nhãn 1 (NCR)**, tức một số region có thể rỗng (xem mục 5 về cách tính Dice).
- Hình minh họa: `docs/brats_sample.png` (4 modality, nhãn gốc, 3 region).

### 1.2 Code (`Repo/3D-TransUNet/`)

- Là **fork của nnU-Net v1**: cần gói `nnunet==1.7.1` (import `nnunet.*` ở khoảng 20 chỗ), dùng lại pipeline `nnUNet_plan_and_preprocess`.
- `train.py` → `nnUNetTrainerV2_DDP` → mạng `Generic_TransUNet_max_ppbp` (`nn_transunet/networks/transunet3d_model.py`).
- Config BraTS: `configs/Brats/{encoder_only, decoder_only, encoder_plus_decoder}.yaml`, `task: Task500_BraTS2021`.
- `inference.py`: sliding window (step 0.5 cho BraTS) trên fold validation. `measure_dice.py`: tính Dice WT/TC/ET.
- **Repo không có** script chuyển BraTS2021 sang format nnU-Net. `doc/data.md` và `doc/usage.md` rỗng. `scripts/*.sh` còn đường dẫn máy của tác giả.

### 1.3 Những lỗi dễ vướng (đã đọc code và chạy thử để xác nhận)

1. **Region mode được bật theo *tên file config*.** Code dùng `args.config.find('500Region')` để quyết định: train theo region WT/TC/ET, dùng sigmoid, `batch_dice=True`, và `resolution_index=0`.
   Config gốc nằm ở `configs/Brats/decoder_only.yaml`, **không chứa chuỗi `500Region`**. Chạy nguyên như vậy sẽ lỗi `KeyError` khi đọc `plans_per_stage[1]`, vì BraTS chỉ có 1 stage (đã xác nhận trên plans sinh ra).
   **→ Cần copy config sang tên có `500Region`** (xem mục 3.5). Không cần sửa code.
2. **`train.py` gọi cứng `init_process_group(backend='nccl')`.** NCCL không có trên Windows, nên phải chạy trên **Linux hoặc WSL2**, kể cả khi chỉ có 1 GPU.
3. **Thiếu `omegaconf`** trong `install.sh`. Nếu không cài, decoder-only lỗi `ModuleNotFoundError` (đã gặp khi chạy thử). **Không cần detectron2**: code chỉ nhắc tới nó trong comment.
4. `train.py` khai báo `--local-rank` (có gạch ngang). `torch.distributed.launch` của PyTorch ≥ 2.0 truyền đúng tham số này, còn PyTorch 1.x truyền `--local_rank` nên sẽ lỗi. → Dùng **PyTorch 2.x**.
5. File pretrain ViT (`R50+ViT-B_16.npz`) chỉ cần cho `encoder_only` (`is_vit_pretrain: True`), đọc theo đường dẫn tương đối `./vit_checkpoint/imagenet21k/`. Decoder-only (cấu hình paper dùng cho BraTS) không cần.
6. Paper nói dùng split 5-fold của nhóm hạng 1 BraTS21, nhưng repo **không kèm file split**. nnU-Net sẽ tự tạo `splits_final.pkl` (KFold, seed 12345). → Kết quả sẽ không cùng split với paper, cần ghi rõ khi báo cáo.

### 1.4 Đã chạy thử (CPU, 2 ca mẫu)

- `scripts/convert_brats2021_to_nnunet.py` → tạo `Task500_BraTS2021` hợp lệ. Nhãn đổi đúng: ED→1, NCR→2, ET→3.
- `nnUNet_plan_and_preprocess -t 500 --verify_dataset_integrity -pl2d None` → `Dataset OK`, sinh plans 3D 1 stage, chuẩn hóa z-score trên mask não.
- `scripts/smoke_forward.py` dựng đúng mạng như trainer:

| Config (crop 128³) | Params | Output |
|---|---|---|
| encoder_only (ViT 12 lớp) | 116.5 M | 5 mức deep supervision, 3 kênh region |
| **decoder_only** | **33.7 M** | `pred_logits (B,20,4)`, `pred_masks (B,20,128³)` + 3 aux |
| encoder_plus_decoder (ViT 1 lớp) | 41.4 M | cả hai |

---

## 2. Lộ trình tổng thể

Mốc thời gian dưới đây tính theo tuần, cần chỉnh theo lịch môn học.

| Giai đoạn | Việc chính | Đầu ra / mốc báo cáo |
|---|---|---|
| **P0. Hiểu đề tài** (xong) | Đọc paper, đọc code, soi data, tìm các lỗi dễ vướng | Mục 0–1 của file này |
| **P1. Môi trường** (tuần 1) | Chọn máy GPU, cài env, chạy smoke test trên GPU | `smoke_forward.py` chạy được trên GPU, đo VRAM |
| **P2. Dữ liệu** (tuần 1) | Tải xong, kiểm tra 1251 ca, convert, plan & preprocess | `Task500_BraTS2021` + plans + `splits_final.pkl` |
| **P3. Chạy thử train** (tuần 2) | Train 1–2 epoch trên fold 0, thử resume, thử inference và đo Dice trên vài ca | Pipeline chạy thông từ đầu đến cuối |
| **P4. Train chính** (tuần 2–4) | Decoder-only, fold 0, theo quỹ thời gian GPU (mục 4) | Checkpoint, log loss, đường cong Dice |
| **P5. Đánh giá** (tuần 5) | Inference trên fold 0, Dice WT/TC/ET, (tùy chọn) HD95, trực quan hóa | Bảng so với paper, hình minh họa |
| **P6. Mở rộng (nếu còn GPU)** | Baseline nnU-Net cùng fold, hoặc encoder_only / thêm fold | Ablation nhỏ giống paper |
| **P7. Báo cáo cuối** | Viết báo cáo, phân tích chênh lệch với paper | Slide và báo cáo |

**Phạm vi tối thiểu nên cam kết:** *decoder-only, fold 0, số epoch thu gọn*, so với paper kèm giải thích chênh lệch. 5-fold × 8 GPU như paper là không thực tế cho môn học.

---

## 3. Setup chi tiết

### 3.1 Chọn máy

| Phương án | Ưu | Nhược |
|---|---|---|
| **Máy Linux có GPU** (server trường/lab, home server) | Ổn định, train dài ngày, resume dễ | Cần xin quyền truy cập |
| **WSL2 Ubuntu trên Windows** + GPU NVIDIA | Dùng máy cá nhân | Đĩa `D:` qua `/mnt/d` rất chậm, **phải để data trong ext4 của WSL**. RAM WSL cần chỉnh trong `.wslconfig` |
| Kaggle / Colab | Miễn phí | Phiên bị giới hạn giờ, đĩa hạn chế (không đủ chỗ cho bản npy giải nén ~80 GB). Phải dùng `--use_compressed_data` và resume liên tục. Chỉ nên dùng làm phương án dự phòng |

**Cấu hình khuyến nghị** (ước lượng, cần đo lại ở bước 3.6):

- GPU: ≥ 24 GB VRAM để chạy đúng crop 128³ với batch 2. Với 12–16 GB: batch 1 hoặc crop 96³, và phải ghi rõ đây là sai khác so với paper.
- RAM ≥ 32 GB. CPU ≥ 8 luồng (data augmentation của nnU-Net chạy trên CPU).
- Đĩa trống **~150 GB** (xem bảng dung lượng ở mục 4).

### 3.2 Môi trường Python (Linux/WSL2)

```bash
# Python 3.10 + uv (hoặc conda nếu thích)
uv venv -p 3.10 ~/envs/transunet && source ~/envs/transunet/bin/activate

# PyTorch 2.x bản CUDA khớp driver (xem nvidia-smi). Ví dụ CUDA 11.8:
uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# numpy phải < 1.24 cho nnU-Net v1
uv pip install "numpy==1.23.5" nnunet==1.7.1
uv pip install batchgenerators SimpleITK medpy nibabel scikit-image pandas matplotlib tqdm \
               pyyaml einops ml_collections fvcore omegaconf tensorboardX monai

# kiểm tra
python -c "import torch, nnunet, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"
python -c "import torch.distributed as d; print('nccl', d.is_nccl_available())"
```

> Không cần `detectron2`, `gco-wrapper`, `adamp`, `segmentation_models_pytorch` như trong `install.sh`. Nếu sau này có lỗi import thì cài bổ sung.

### 3.3 Biến môi trường và cây thư mục

Nên đặt vào file `env.sh` và chạy `source env.sh` mỗi lần mở terminal:

```bash
export PROJ=$HOME/cv-transunet                 # ổ ext4, KHÔNG dùng /mnt/d nếu là WSL
export nnUNet_raw_data_base=$PROJ/nnUNet_raw_data_base
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export RESULTS_FOLDER=$PROJ/results
export nnUNet_N_proc_DA=8                      # ≈ số luồng CPU - 2 (gốc để 36)
export nnUNet_codebase=$PROJ/3D-TransUNet
mkdir -p $nnUNet_raw_data_base/nnUNet_raw_data $nnUNet_preprocessed $RESULTS_FOLDER
```

```
$PROJ/
├── 3D-TransUNet/                  # repo (copy từ D:\Learning Projects\CV\Repo)
├── scripts/                       # copy từ CV\Plan\scripts (convert + smoke test)
├── nnUNet_raw_data_base/
│   ├── nnUNet_raw_data/Task500_BraTS2021/{imagesTr,labelsTr,dataset.json}
│   └── nnUNet_cropped_data/       # sinh ra khi preprocess, xóa được sau đó
├── nnUNet_preprocessed/Task500_BraTS2021/
│   ├── nnUNetPlansv2.1_plans_3D.pkl, splits_final.pkl, gt_segmentations/
│   └── nnUNetData_plans_v2.1_stage0/*.npz (+ *.npy khi unpack)
└── results/UNet_IN_NANFang/Task500_BraTS2021/nnUNetTrainerV2_DDP__nnUNetPlansv2.1/<hdfs_base>/fold_0/
```

### 3.4 Dữ liệu: kiểm tra → convert → preprocess

**(a) Đợi tải xong rồi kiểm tra tính toàn vẹn**, chưa giải nén:

```bash
TAR=/path/BraTS2021_Training_Data.tar
tar tf $TAR | grep -c 'nii.gz$'                                  # kỳ vọng 6255 (= 1251 × 5)
tar tf $TAR | grep -oE 'BraTS2021_[0-9]{5}' | sort -u | wc -l    # kỳ vọng 1251
```

Nếu `tar` báo lỗi "unexpected EOF", file tải chưa xong hoặc bị hỏng.

**(b) Convert sang nnU-Net** bằng script đọc thẳng từ tar (không cần giải nén trung gian):

```bash
python $PROJ/scripts/convert_brats2021_to_nnunet.py \
  --src $TAR \
  --out $nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
# thử trước:  --limit 10
```

Script đổi `t1,t1ce,t2,flair` thành `_0000.._0003` và đổi nhãn `2→1, 1→2, 4→3` theo đúng quy ước nnU-Net BraTS, khớp với `regions` trong code. Script cũng sinh `dataset.json`.

**(c) Plan & preprocess** (chỉ 3D, bỏ 2D để tiết kiệm thời gian và đĩa):

```bash
nnUNet_plan_and_preprocess -t 500 --verify_dataset_integrity -pl2d None -tl 8 -tf 8
```

Sau đó kiểm tra plans:

```bash
python - <<'EOF'
import pickle, os
p = pickle.load(open(os.environ['nnUNet_preprocessed'] + '/Task500_BraTS2021/nnUNetPlansv2.1_plans_3D.pkl', 'rb'))
s = p['plans_per_stage']; print(s.keys()); print(s[0]['patch_size'], s[0]['pool_op_kernel_sizes'])
EOF
```

Ghi lại `pool_op_kernel_sizes` để so với "downsample [5,5,5]" trong paper. Crop 128³ trong config sẽ ghi đè patch size.

### 3.5 Tạo config region cho BraTS (bắt buộc)

```bash
cd $PROJ/3D-TransUNet
cp configs/Brats/decoder_only.yaml          configs/Brats/GeTU500Region_decoder_only.yaml
cp configs/Brats/encoder_only.yaml          configs/Brats/GeTU500Region_encoder_only.yaml
cp configs/Brats/encoder_plus_decoder.yaml  configs/Brats/GeTU500Region_encoder_plus_decoder.yaml
```

Chỉ đổi tên file, **nội dung giữ nguyên**, để giữ pipeline gốc. Từ đây luôn truyền đúng đường dẫn có `500Region` cho `train.py`, `inference.py` và `measure_dice.py`.

### 3.6 Smoke test trên GPU

```bash
cd $PROJ/3D-TransUNet
PYTHONPATH=. python ../scripts/smoke_forward.py \
  --config configs/Brats/GeTU500Region_decoder_only.yaml \
  --plans $nnUNet_preprocessed/Task500_BraTS2021/nnUNetPlansv2.1_plans_3D.pkl \
  --crop 128 128 128 --bs 2 --device cuda --backward
```

Kết quả `peak VRAM` cho biết có giữ được batch 2 và crop 128³ không. Nếu OOM, thử `--bs 1`, rồi `--crop 96 96 96`.

### 3.7 Train

**Chạy thử 1–2 epoch** (log vào file):

```bash
cd $PROJ/3D-TransUNet && mkdir -p logs
CFG=configs/Brats/GeTU500Region_decoder_only.yaml
nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4322 \
  train.py --fold=0 --config=$CFG --resume='' --max_num_epochs 2 2>&1 | tee logs/dryrun_fold0.log
```

Kiểm tra các điểm sau:

- Lần đầu chạy sẽ tạo `splits_final.pkl` và unpack `.npz` thành `.npy`.
- Log in ra "disable ds", số tham số, và loss giảm dần.
- `fold_0/model_latest.model` được lưu (`save_every=1`).
- Thử dừng rồi chạy lại để kiểm tra resume: sửa `--resume=''` thành `--resume=auto`.

**Train chính:** bỏ `--max_num_epochs 2`, dùng `--resume=auto`, chạy trong `tmux`/`screen`. Theo dõi `progress.png` và `training_log_*.txt` trong thư mục fold.

- Mỗi epoch cố định **250 iteration** (`num_batches_per_epoch`). Paper: 125 epoch × 8 GPU × batch 2, tức ~250k patch. Với 1 GPU batch 2, muốn cùng số mẫu cần ~1000 epoch.
- Nếu RAM hoặc đĩa thiếu: thêm `--use_compressed_data` để không unpack `.npy` (tốn CPU hơn).

### 3.8 Inference & đánh giá (fold 0)

```bash
CFG=configs/Brats/GeTU500Region_decoder_only.yaml
SAVE=$RESULTS_FOLDER/inference/Task500/decoder_only/fold_0
CUDA_VISIBLE_DEVICES=0 python inference.py --config=$CFG --fold=0 \
    --raw_data_folder imagesTr --save_folder=$SAVE
python measure_dice.py --config=$CFG --fold=0 --pred_dir=$SAVE/
```

`inference.py` tự lấy các ca `val` của fold 0 trong `splits_final.pkl` và nạp `model_best` → `model_final_checkpoint` → `model_latest`. `measure_dice.py` so với `nnUNet_preprocessed/Task500_BraTS2021/gt_segmentations/`.

---

## 4. Ước lượng tài nguyên và phương án thu gọn

**Dung lượng đĩa** (ngoại suy từ 2 ca đã preprocess):

| Thành phần | ~Dung lượng |
|---|---|
| Tar gốc | 13 GB (xóa được sau khi convert) |
| `nnUNet_raw_data/Task500` | ~13 GB |
| `nnUNet_cropped_data` | ~15 GB (xóa được sau preprocess) |
| Preprocessed `.npz` | ~17.5 GB |
| Unpacked `.npy` (khi train) | **~80 GB** |
| Checkpoint + inference | 5–10 GB |

**Thời gian:** chưa đo được (cần GPU). Hãy đo `epoch time` sau lần chạy thử ở 3.7, rồi chọn số epoch theo quỹ giờ GPU:

| Phương án | Epoch (1 GPU) | Ghi chú |
|---|---|---|
| A. Tối thiểu | 100–150 | Đủ để hội tụ tương đối, Dice sẽ thấp hơn paper |
| B. Khuyến nghị | 300–500 | Cân bằng thời gian và chất lượng |
| C. Tương đương paper (1 fold) | ~1000 | Cùng số mẫu với 8 GPU × 125 epoch |

Có thể dùng thêm `--total_batch_size` hoặc `--dbs`. Nếu đổi batch hay crop, **phải ghi rõ trong báo cáo**.

---

## 5. Lưu ý khi phân tích kết quả

- **Split khác paper** → chỉ so sánh tương đối.
- **Train fold 0 thay vì 5-fold** → báo cáo số của 1 fold.
- **Ít epoch hơn và ít GPU hơn** → batch toàn cục là 2 thay vì 16. Với AdamW + cosine thì lịch lr theo epoch sẽ khác.
- `measure_dice.py` cho **Dice = 0 khi pred hoặc GT rỗng**, kể cả khi cả hai cùng rỗng. BraTS chính thức tính trường hợp cả hai rỗng là 1. Các ca không có ET/NCR (như 00621) sẽ kéo điểm xuống. Nên báo cáo cả hai cách tính.
- Paper chỉ báo Dice. HD95 đã có sẵn trong code nhưng bị comment, có thể bật lại làm phần mở rộng.

---

## 6. Gợi ý nội dung báo cáo ngày mai

1. **Nhắc lại đề tài** (1 slide): bài toán BraTS, ý tưởng 3D TransUNet, 3 cấu hình, lý do chọn decoder-only cho khối u.
2. **Phân tích dữ liệu** (1–2 slide): 1251 ca, 4 modality, 3 nhãn và 3 region, mất cân bằng lớp (u ~1% thể tích), hình `brats2021_sample.png`.
3. **Phân tích code** (1–2 slide): pipeline nnU-Net v1 → trainer DDP → mạng. Sơ đồ luồng convert → plan/preprocess → train → inference → measure_dice.
4. **Vấn đề kỹ thuật đã phát hiện** (1 slide): 6 lỗi ở mục 1.3 và cách xử lý mà không đổi pipeline.
5. **Đã chạy thử**: convert + preprocess thành công trên ca mẫu, dựng được mạng 3 cấu hình (bảng params).
6. **Kế hoạch tiếp theo**: bảng lộ trình ở mục 2, phương án tài nguyên ở mục 4, phạm vi cam kết (decoder-only, fold 0).
7. **Rủi ro / cần hỗ trợ**: GPU (xin server lab?), dung lượng đĩa, thời gian train.

> Tối nay nên tự chạy lại mục 1.4 trên máy của nhóm (convert 2 ca mẫu + preprocess + smoke test). Như vậy khi báo cáo, nhóm có log và kết quả của chính mình.

---

## 7. Checklist

- [ ] Chốt máy GPU (VRAM, RAM, đĩa) và ghi lại cấu hình
- [ ] Cài env (3.2), `torch.cuda.is_available()` và `is_nccl_available()` đều trả về True
- [ ] Tải xong data, đếm đủ 1251 ca / 6255 file
- [ ] Convert → `Task500_BraTS2021`, `dataset.json` có `numTraining: 1251`
- [ ] `nnUNet_plan_and_preprocess` báo OK, ghi lại plans
- [ ] Tạo config `GeTU500Region_*.yaml`
- [ ] Smoke test GPU, ghi lại peak VRAM
- [ ] Chạy thử train 2 epoch, kiểm tra resume, ghi lại thời gian mỗi epoch
- [ ] Chạy thử inference + measure_dice trên checkpoint chạy thử
- [ ] Chốt số epoch → train chính fold 0
- [ ] Inference + Dice WT/TC/ET, trực quan hóa, so với paper
- [ ] (Tùy chọn) baseline nnU-Net / encoder-only / thêm fold
