# TỔNG QUAN DỰ ÁN: PHÁT HIỆN BỆNH VÕNG MẠC TIỂU ĐƯỜNG BẰNG DEEP LEARNING

## Uncertainty-Aware Attention CNN cho Phân Loại Bệnh Võng Mạc Tiểu Đường

**Tên dự án:** `dr-detect`
**Thời gian thực hiện:** 1 tháng (01/03/2026 – 31/03/2026)
**Loại dự án:** Khóa luận tốt nghiệp — Deep Learning ứng dụng trong Y tế

---

## MỤC LỤC

1. [Bối Cảnh Y Khoa](#1-bối-cảnh-y-khoa)
2. [Cơ Sở Lý Thuyết Deep Learning](#2-cơ-sở-lý-thuyết-deep-learning)
3. [Kiến Trúc Mô Hình Đề Xuất](#3-kiến-trúc-mô-hình-đề-xuất)
4. [Bộ Dữ Liệu](#4-bộ-dữ-liệu)
5. [Tiền Xử Lý Ảnh](#5-tiền-xử-lý-ảnh)
6. [Pipeline Huấn Luyện & Đánh Giá](#6-pipeline-huấn-luyện--đánh-giá)
7. [Cấu Trúc Mã Nguồn](#7-cấu-trúc-mã-nguồn)
8. [Đề Xuất Nghiên Cứu (Proposal)](#8-đề-xuất-nghiên-cứu-proposal)
9. [Kế Hoạch Thực Hiện 1 Tháng](#9-kế-hoạch-thực-hiện-1-tháng)
10. [Kết Quả Kỳ Vọng](#10-kết-quả-kỳ-vọng)
11. [Hướng Dẫn Viết Khóa Luận](#11-hướng-dẫn-viết-khóa-luận)
12. [Tài Liệu Tham Khảo Chính](#12-tài-liệu-tham-khảo-chính)

---

## 1. BỐI CẢNH Y KHOA

### 1.1. Bệnh Võng Mạc Tiểu Đường (Diabetic Retinopathy — DR)

Bệnh võng mạc tiểu đường là **biến chứng mạch máu phổ biến nhất** của bệnh đái tháo đường, gây tổn thương các mạch máu nhỏ ở võng mạc (retina) — lớp mô nhạy sáng ở phía sau mắt.

**Thống kê quan trọng:**

- Trên thế giới có khoảng **537 triệu** người mắc đái tháo đường (IDF 2021).
- Khoảng **1/3** bệnh nhân đái tháo đường phát triển các dấu hiệu DR.
- DR là **nguyên nhân hàng đầu** gây mù lòa ở người trong độ tuổi lao động (20–74 tuổi).
- Nếu được phát hiện sớm, **hơn 90%** trường hợp mất thị lực nghiêm trọng có thể được ngăn chặn.

### 1.2. Hệ Thống Phân Loại DR — Thang 5 Mức

Theo International Clinical DR Severity Scale, DR được phân thành **5 mức độ**:

| Mức | Tên Tiếng Anh    | Tên Tiếng Việt | Mô Tả                                         |
| --- | ---------------- | -------------- | --------------------------------------------- |
| 0   | No DR            | Không DR       | Không có tổn thương vi mạch                   |
| 1   | Mild NPDR        | Nhẹ            | Chỉ có vi phình mạch (microaneurysms)         |
| 2   | Moderate NPDR    | Trung bình     | Vi phình mạch + xuất huyết + exudate          |
| 3   | Severe NPDR      | Nặng           | Xuất huyết lan rộng, thiếu máu võng mạc       |
| 4   | Proliferative DR | Tăng sinh      | Tân mạch (neovascularization), nguy cơ mù cao |

> **NPDR** = Non-Proliferative Diabetic Retinopathy (Bệnh võng mạc tiểu đường không tăng sinh)

**Ngưỡng "Referable DR":** Mức ≥ 2 (Moderate trở lên) → cần chuyển tuyến chuyên khoa.

### 1.3. Tại Sao Cần Ứng Dụng AI?

- **Thiếu bác sĩ nhãn khoa** trầm trọng ở các nước đang phát triển.
- Sàng lọc thủ công tốn thời gian và phụ thuộc vào chuyên gia.
- AI có thể **đạt độ nhạy (sensitivity) ≥ 93%** và độ đặc hiệu (specificity) ≥ 90% — tương đương hoặc vượt bác sĩ (theo meta-analysis 82 nghiên cứu, 887.244 ca, Nature Digital Medicine 2025).
- **25+ thiết bị AI sàng lọc DR** đã được FDA chấp thuận tính đến 2025.

### 1.4. Các Dạng Tổn Thương Trên Ảnh Fundus

Ảnh đáy mắt (fundus photograph) là phương pháp chụp chính cho sàng lọc DR. Các tổn thương bao gồm:

| Tổn Thương     | Tiếng Anh           | Đặc Điểm                               |
| -------------- | ------------------- | -------------------------------------- |
| Vi phình mạch  | Microaneurysms (MA) | Chấm đỏ nhỏ, dấu hiệu sớm nhất         |
| Xuất huyết     | Hemorrhages (HM)    | Vùng đỏ sậm, kích thước lớn hơn MA     |
| Xuất tiết cứng | Hard Exudates (EX)  | Vàng sáng, lắng đọng lipid             |
| Xuất tiết mềm  | Cotton Wool Spots   | Trắng bông, nhồi máu lớp sợi thần kinh |
| Tân mạch       | Neovascularization  | Mạch máu mới bất thường (DR tăng sinh) |

---

## 2. CƠ SỞ LÝ THUYẾT DEEP LEARNING

### 2.1. Convolutional Neural Network (CNN)

CNN là kiến trúc mạng nơ-ron chuyên xử lý dữ liệu dạng lưới (grid-like), đặc biệt hiệu quả với hình ảnh.

**Các thành phần cốt lõi:**

- **Convolutional Layer:** Trích xuất đặc trưng cục bộ (local features) bằng bộ lọc trượt (kernel).
- **Pooling Layer:** Giảm chiều không gian, tăng tính bất biến vị trí.
- **Fully Connected Layer:** Kết hợp đặc trưng để phân loại.

### 2.2. ResNet-50 — Backbone

**ResNet** (Residual Network, He et al. 2015) giải quyết vấn đề **vanishing gradient** trong mạng sâu bằng **skip connections** (kết nối tắt):

```
Residual Block:
    x ──→ [Conv → BN → ReLU → Conv → BN] → (+) → ReLU → output
    └────────────────────────────────────────→ ↗
                    (skip connection / identity)
```

**ResNet-50** có 50 lớp, chia thành 4 stage (layer1–layer4):

| Stage  | Output Size | Channels | Blocks       |
| ------ | ----------- | -------- | ------------ |
| Stem   | 128×128     | 64       | conv + pool  |
| layer1 | 128×128     | 256      | 3 bottleneck |
| layer2 | 64×64       | 512      | 4 bottleneck |
| layer3 | 32×32       | 1024     | 6 bottleneck |
| layer4 | 16×16       | 2048     | 3 bottleneck |

**Tại sao chọn ResNet-50:**

- Cân bằng tốt giữa độ sâu và chi phí tính toán (~23.5M tham số).
- Pre-trained trên ImageNet → transfer learning hiệu quả cho ảnh y tế.
- Đã được chứng minh hiệu quả trong nhiều nghiên cứu DR (survey 2025: 50+ studies).

### 2.3. CBAM — Convolutional Block Attention Module

**CBAM** (Woo et al., ECCV 2018) là cơ chế attention giúp mô hình **tập trung vào các vùng bệnh lý quan trọng** trên ảnh fundus, thay vì xử lý mọi pixel như nhau.

CBAM gồm 2 thành phần, áp dụng tuần tự:

#### a) Channel Attention (Attention theo kênh)

Xác định **kênh đặc trưng nào** quan trọng nhất (ví dụ: kênh phát hiện vi phình mạch vs. kênh phát hiện nền).

```
Input: F ∈ R^(C×H×W)

1. AvgPool(F) → F_avg ∈ R^(C×1×1)
2. MaxPool(F) → F_max ∈ R^(C×1×1)
3. MLP(F_avg) + MLP(F_max) → σ(·) → M_c ∈ R^(C×1×1)
4. Output: M_c ⊗ F

MLP: FC(C→C/r) → ReLU → FC(C/r→C)     [r = 16, reduction ratio]
```

#### b) Spatial Attention (Attention theo không gian)

Xác định **vùng không gian nào** trên ảnh đáng chú ý (ví dụ: vùng có tổn thương vs. vùng nền bình thường).

```
Input: F' (đã qua channel attention)

1. AvgPool_channel(F') → f_avg ∈ R^(1×H×W)
2. MaxPool_channel(F') → f_max ∈ R^(1×H×W)
3. Concat(f_avg, f_max) → Conv7×7 → σ(·) → M_s ∈ R^(1×H×W)
4. Output: M_s ⊗ F'
```

**Trong dự án:** CBAM được **inject sau mỗi stage** của ResNet-50 (layer1–layer4), tổng cộng **4 CBAM blocks**. Điều này giúp mô hình tự học tập trung vào các tổn thương DR ở nhiều mức độ trừu tượng (từ cạnh, kết cấu đến cấu trúc bệnh lý).

### 2.4. MC Dropout — Ước Lượng Bất Định (Uncertainty Estimation)

**Vấn đề:** Mô hình deep learning thông thường chỉ đưa ra **một dự đoán duy nhất** mà không cho biết độ tin cậy. Trong y tế, một dự đoán "Không DR" với xác suất 51% về bản chất rất khác với xác suất 99%.

**Giải pháp — Monte Carlo Dropout (Gal & Ghahramani, 2016):**

Ý tưởng: Sử dụng Dropout **trong cả lúc inference** (không chỉ training) để tạo ra nhiều dự đoán khác nhau cho cùng một ảnh.

```
Quy trình MC Dropout Inference (T = 20 lần forward):

Ảnh x → [Model + Dropout ON] → ŷ₁  (softmax probabilities)
Ảnh x → [Model + Dropout ON] → ŷ₂
...
Ảnh x → [Model + Dropout ON] → ŷ_T

Dự đoán trung bình: ȳ = (1/T) Σ ŷ_t
Bất định (Predictive Entropy): H(ȳ) = -Σ ȳ_c · log(ȳ_c)
Confidence: max(ȳ)
```

**Cách implement trong PyTorch:** Thay `nn.Dropout` bằng custom `MCDropout` class luôn đặt `training=True` trong `F.dropout()`, bất kể model đang ở `eval()` mode.

**Ý nghĩa lâm sàng:**

- **Entropy thấp → Dự đoán đáng tin cậy** → Có thể tự động sàng lọc.
- **Entropy cao → Dự đoán không chắc chắn** → Cần bác sĩ xem lại (human-in-the-loop).

### 2.5. Focal Loss — Xử Lý Mất Cân Bằng Lớp

**Vấn đề:** Dataset APTOS 2019 rất mất cân bằng:

- Mức 0 (Không DR): **~49%** (1805 ảnh)
- Mức 4 (Tăng sinh): **~5%** (193 ảnh)

Cross-entropy thông thường cho phép mô hình đạt accuracy cao chỉ bằng cách dự đoán lớp đa số.

**Focal Loss (Lin et al., 2017):**

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

Trong đó:
  p_t = xác suất dự đoán cho lớp đúng
  γ = 2.0 (focusing parameter) → giảm trọng số mẫu dễ ~100×
  α_t = trọng số lớp (nghịch tần suất) → tăng trọng số lớp thiểu số
```

| p_t (xác suất đúng) | Cross-Entropy | Focal Loss (γ=2) | Giảm bao nhiêu |
| ------------------- | ------------- | ---------------- | -------------- |
| 0.9 (dễ)            | 0.105         | 0.001            | ~100×          |
| 0.5 (trung bình)    | 0.693         | 0.173            | ~4×            |
| 0.1 (khó)           | 2.303         | 1.866            | ~1.2×          |

→ Focal Loss tự động tập trung gradient vào **mẫu khó và lớp thiểu số**.

### 2.6. Mixed-Precision Training (AMP)

Sử dụng `torch.amp` (Automatic Mixed Precision) để:

- Lưu activation bằng **float16** thay vì float32 → **giảm ~50% VRAM**.
- Cho phép `batch_size=16` trên GPU 12GB với ảnh 512×512.
- Duy trì **float32 cho gradient** (qua `GradScaler`) → không mất precision.

### 2.7. Transfer Learning

Sử dụng **pre-trained weights từ ImageNet**, chỉ fine-tune toàn bộ mạng trên dữ liệu DR:

- ImageNet (1.2M ảnh, 1000 lớp) → các đặc trưng tổng quát (cạnh, kết cấu, hình dạng).
- Fine-tuning trên APTOS → chuyên biệt hóa cho tổn thương DR.
- Learning rate thấp (`1e-4`) + `CosineAnnealingLR` → fine-tune ổn định.

---

## 3. KIẾN TRÚC MÔ HÌNH ĐỀ XUẤT

### 3.1. Tên Kiến Trúc

**Uncertainty-Aware Attention CNN** — Mạng CNN Tích Hợp Cơ Chế Chú Ý và Ước Lượng Bất Định.

### 3.2. Sơ Đồ Kiến Trúc

```
Input: Ảnh fundus (B, 3, 512, 512)
         │
         ▼
┌─── ResNet-50 Stem ──────────────────────┐
│  Conv7×7 → BN → ReLU → MaxPool         │
│  Output: (B, 64, 128, 128)             │
└─────────────────────────────────────────┘
         │
    ┌────▼────┐
    │ Layer1  │ → 3 Bottleneck → (B, 256, 128, 128)
    │ + CBAM  │
    └────┬────┘
         │
    ┌────▼────┐
    │ Layer2  │ → 4 Bottleneck → (B, 512, 64, 64)
    │ + CBAM  │
    └────┬────┘
         │
    ┌────▼────┐
    │ Layer3  │ → 6 Bottleneck → (B, 1024, 32, 32)
    │ + CBAM  │
    └────┬────┘
         │
    ┌────▼────┐
    │ Layer4  │ → 3 Bottleneck → (B, 2048, 16, 16)
    │ + CBAM  │
    └────┬────┘
         │
    ┌────▼──────────────────────┐
    │ Adaptive Average Pool     │ → (B, 2048, 1, 1) → flatten → (B, 2048)
    └────┬──────────────────────┘
         │
    ┌────▼──────────────────────┐
    │ MC Dropout (p=0.5)        │ ← Luôn BẬT, kể cả khi eval()
    └────┬──────────────────────┘
         │
    ┌────▼──────────────────────┐
    │ Linear(2048 → 5)          │ → (B, 5) logits → softmax → xác suất
    └───────────────────────────┘
```

### 3.3. Tổng Số Tham Số

| Thành phần         | Tham số ước tính |
| ------------------ | ---------------- |
| ResNet-50 backbone | ~23.5M           |
| 4× CBAM blocks     | ~0.3M            |
| FC head (2048→5)   | ~10K             |
| **Tổng**           | **~23.8M**       |

### 3.4. Điểm Khác Biệt So Với ResNet-50 Baseline

| Đặc điểm    | Baseline ResNet-50          | CBAM-ResNet50 (Đề xuất)           |
| ----------- | --------------------------- | --------------------------------- |
| Attention   | Không                       | CBAM ở 4 stage                    |
| Dropout     | `nn.Dropout` (tắt khi eval) | `MCDropout` (luôn bật)            |
| Uncertainty | Không                       | Predictive Entropy qua MC Dropout |
| Loss        | Cross-Entropy               | Focal Loss + alpha weighting      |
| Precision   | float32                     | Mixed (float16/32) AMP            |

---

## 4. BỘ DỮ LIỆU

### 4.1. APTOS 2019 — Training/Validation

| Thuộc Tính | Giá Trị                          |
| ---------- | -------------------------------- |
| Nguồn      | Kaggle APTOS Blindness Detection |
| Tổng ảnh   | 3,662                            |
| Phân loại  | 5 mức (0–4)                      |
| Format     | PNG                              |
| Phân bổ    | Chia 5-fold Stratified K-Fold    |

**Phân bố lớp (mất cân bằng nghiêm trọng):**

| Mức DR            | Số ảnh | Tỷ lệ |
| ----------------- | ------ | ----- |
| 0 — No DR         | 1,805  | 49.3% |
| 1 — Mild          | 370    | 10.1% |
| 2 — Moderate      | 999    | 27.3% |
| 3 — Severe        | 193    | 5.3%  |
| 4 — Proliferative | 295    | 8.1%  |

### 4.2. Messidor-2 — External Test Set

| Thuộc Tính | Giá Trị                        |
| ---------- | ------------------------------ |
| Nguồn      | Messidor-2 (Decencière et al.) |
| Tổng ảnh   | 690                            |
| Format     | TIFF/JPG/PNG                   |
| Vai trò    | Đánh giá domain generalization |
| Annotation | `adjudicated_dr_grade`         |

**Tại sao dùng Messidor-2:**

- Khác domain hoàn toàn so với APTOS (camera, bệnh viện, dân số khác).
- Đánh giá khả năng **tổng quát hóa** (generalization) của mô hình.
- Có nhãn y khoa đáng tin cậy (adjudicated bởi chuyên gia).

### 4.3. Chiến Lược Chia Dữ Liệu

```
APTOS 2019 (3,662 ảnh)
    │
    ├── 5-Fold Stratified K-Fold
    │   ├── Fold 0: Train 80% | Val 20%
    │   ├── Fold 1: Train 80% | Val 20%
    │   ├── Fold 2: Train 80% | Val 20%
    │   ├── Fold 3: Train 80% | Val 20%
    │   └── Fold 4: Train 80% | Val 20%
    │
    └── Đảm bảo Stratified: phân bố lớp tỷ lệ trong mỗi fold

Messidor-2 (690 ảnh) → External Test Set (không dùng cho training)
```

---

## 5. TIỀN XỬ LÝ ẢNH

### 5.1. Ben Graham's Preprocessing Pipeline

Phương pháp được đề xuất bởi **Ben Graham** (giải nhất Kaggle DR competition 2015), gồm 2 bước:

#### Bước 1: Circular Crop — Cắt viền đen

Ảnh fundus thường có viền đen lớn (do camera). Viền này không chứa thông tin y khoa nhưng có thể gây nhiễu cho mô hình.

```
1. Chuyển ảnh sang grayscale
2. Threshold (> 10) → tách retina khỏi nền đen
3. Morphological closing → lấp lỗ nhỏ
4. Tìm contour lớn nhất → enclosing circle
5. Crop theo bounding box + margin (5%)
6. Áp circular mask → zero-out vùng ngoài đĩa
7. Resize → 512×512
```

#### Bước 2: Local Color Normalization — Chuẩn hóa màu cục bộ

Loại bỏ biến thiên chiếu sáng (illumination variation) nhưng giữ nguyên chi tiết tổn thương:

```
result = 4 × image − 4 × GaussianBlur(image, σ) + 128

Trong đó:
  σ = image_width / 30 (tự thích ứng theo kích thước ảnh)
  ×4: khuếch đại tương phản
  +128: re-center về mid-gray
```

**So sánh với CLAHE:**

- CLAHE xử lý từng kênh riêng → có thể phá vỡ quan hệ màu.
- Ben Graham's method xử lý **3 kênh BGR đồng thời** → giữ nguyên thông tin màu (quan trọng để phân biệt hard exudate vàng vs hemorrhage đỏ).

### 5.2. Data Augmentation (Albumentations)

**Training augmentations:**

| Augmentation             | Tham số           | Mục đích                                |
| ------------------------ | ----------------- | --------------------------------------- |
| HorizontalFlip           | p=0.5             | Bất biến lật ngang                      |
| VerticalFlip             | p=0.5             | Bất biến lật dọc                        |
| Rotate                   | limit=30°         | Bất biến xoay                           |
| RandomBrightnessContrast | limit=0.2         | Thay đổi sáng/tương phản                |
| ShiftScaleRotate         | shift=0.05        | Dịch chuyển/co giãn                     |
| CoarseDropout            | max_holes=8       | Regularization (giống dropout trên ảnh) |
| Normalize                | ImageNet mean/std | Chuẩn hóa giá trị pixel                 |

**Validation/inference:** Chỉ `Resize(512, 512)` + `Normalize(ImageNet mean/std)`.

---

## 6. PIPELINE HUẤN LUYỆN & ĐÁNH GIÁ

### 6.1. Workflow Tổng Thể

```
┌──────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                            │
│                                                                  │
│  APTOS 2019 → Ben Graham Preprocessing → Augmentation            │
│       │                                                          │
│       ▼                                                          │
│  CBAM-ResNet50 + Focal Loss + AdamW + CosineAnnealing            │
│  Mixed-Precision (AMP) + Early Stopping (patience=5)             │
│       │                                                          │
│       ▼                                                          │
│  Best Checkpoint (theo val_kappa) → Lưu .pth                    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      EVALUATION PHASE                            │
│                                                                  │
│  Messidor-2 → Ben Graham Preprocessing                           │
│       │                                                          │
│       ▼                                                          │
│  Load Checkpoint → MC Dropout Inference (T=20 passes)            │
│       │                                                          │
│       ▼                                                          │
│  Metrics: Accuracy, QWK, AUC, F1 (per-class)                    │
│  Uncertainty: Predictive Entropy, Confidence                     │
│  Visualization: Confusion Matrix, ROC, Entropy Histogram         │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2. Hyperparameters

| Hyperparameter | Giá Trị           | Giải Thích                                |
| -------------- | ----------------- | ----------------------------------------- |
| Image Size     | 512×512           | Đủ chi tiết để thấy MA; cân bằng với VRAM |
| Batch Size     | 16                | Tối đa cho GPU 12GB với AMP               |
| Epochs         | 20                | Đủ hội tụ với early stopping              |
| Learning Rate  | 1e-4              | LR thấp cho fine-tuning pre-trained       |
| Optimizer      | AdamW             | Adam + weight decay decoupled             |
| Weight Decay   | 1e-4              | Regularization                            |
| Scheduler      | CosineAnnealingLR | Giảm LR mượt theo cosine                  |
| Focal Loss γ   | 2.0               | Focusing parameter chuẩn                  |
| Focal Loss α   | Auto              | Tính từ nghịch tần suất lớp               |
| MC Dropout p   | 0.5               | Tỷ lệ dropout cho Bayesian head           |
| MC Passes T    | 20                | Số forward pass cho uncertainty           |
| K-Fold         | 5-fold            | Stratified cross-validation               |
| Early Stopping | Patience=5        | Dừng nếu val_kappa không cải thiện        |
| Random Seed    | 42                | Reproducibility                           |

### 6.3. Các Metrics Đánh Giá

| Metric                   | Viết tắt | Ý nghĩa                                                                                    |
| ------------------------ | -------- | ------------------------------------------------------------------------------------------ |
| Quadratic Weighted Kappa | QWK      | Đo agreement giữa dự đoán và nhãn thật, có tính thứ tự (ordinal). **Metric chính cho DR.** |
| Accuracy                 | Acc      | Tỷ lệ dự đoán đúng tổng thể                                                                |
| Macro F1-Score           | F1       | Trung bình F1 các lớp (đánh giá cân bằng)                                                  |
| AUC-ROC (one-vs-rest)    | AUC      | Khả năng phân biệt từng lớp vs. còn lại                                                    |
| Binary Referable AUC     | AUC-Ref  | AUC cho bài toán nhị phân: Referable DR (≥2) vs. Non-referable                             |
| Predictive Entropy       | H        | Mức độ bất định của dự đoán                                                                |
| Confidence               | Conf     | Xác suất cao nhất trong softmax                                                            |
| Sensitivity (Recall)     | Sen      | Tỷ lệ phát hiện đúng bệnh (quan trọng y tế)                                                |
| Specificity              | Spec     | Tỷ lệ loại trừ đúng không bệnh                                                             |

### 6.4. Uncertainty Analysis

Sau khi MC Dropout inference (T=20 passes), mỗi ảnh được gán:

1. **Predictive Entropy H** — Thước đo bất định tổng thể:
   - H ≈ 0 → Rất chắc chắn (tất cả T passes đồng thuận)
   - H > 1.0 → Rất không chắc chắn → Cần bác sĩ xem lại

2. **Confidence** — Xác suất lớp được chọn (từ mean softmax):
   - > 0.9 → Tin cậy cao
   - < 0.5 → Dự đoán yếu

3. **Disagreement across passes** — Nếu T passes cho ra các lớp khác nhau → mô hình không chắc chắn.

**Visualization outputs:**

- **Entropy Histogram:** Phân bố uncertainty, colored by DR grade.
- **Confidence vs Entropy Scatter:** Tương quan giữa confidence và entropy.
- **Per-image CSV:** Từng ảnh với dự đoán, confidence, entropy.

---

## 7. CẤU TRÚC MÃ NGUỒN

```
dr-detect/
├── src/
│   ├── config.py           # Tất cả hyperparameters, paths, constants
│   ├── preprocessing.py    # Ben Graham's preprocessing pipeline
│   ├── dataset.py          # PyTorch Dataset + DataLoader cho APTOS & Messidor-2
│   ├── model.py            # CBAM + MCDropout + CBAMResNet50
│   ├── loss.py             # Focal Loss + compute_class_weights
│   ├── train.py            # Training loop (AMP, early stopping, checkpointing)
│   └── evaluate.py         # MC Dropout inference + metrics + visualization
│
├── base_run/
│   ├── 01_data_preprocessing_eda.ipynb   # EDA notebook
│   └── 02_resnet50_baseline.ipynb        # Baseline ResNet-50 (no CBAM)
│
├── notebooks/
│   └── 01_eda.ipynb                      # Messidor-2 EDA (patient-level)
│
├── docs/
│   ├── TONG_QUAN_DU_AN.md                # Tài liệu này
│   ├── essential-papers-dr-detection.md  # 32 papers tham khảo
│   ├── VASTAI_DEPLOYMENT.md              # Hướng dẫn deploy GPU cloud
│   ├── Project Context_DR.pdf            # Bối cảnh dự án
│   └── From Retinal Pixels to Patients...pdf  # Survey paper chính
│
├── vast/
│   ├── prepare_upload.py      # Script nén data cho upload
│   ├── setup_vastai.sh        # Setup script trên Vast.ai
│   └── run_training.sh        # Training script tự động
│
├── aptos/                     # APTOS 2019 dataset
│   └── aptos2019-blindness-detection/
│       ├── train.csv
│       └── train_images/      # ~3,662 ảnh PNG
│
├── messidor-2/                # Messidor-2 dataset
│   ├── messidor-2.csv
│   └── IMAGES/               # ~690 ảnh
│
├── data/processed/            # Output tiền xử lý (nếu có)
├── outputs/                   # Checkpoints, figures, results, logs
└── requirements.txt           # Python dependencies
```

### 7.1. Vai Trò Từng Module

| Module             | Vai trò                                                                     | Phụ thuộc                            |
| ------------------ | --------------------------------------------------------------------------- | ------------------------------------ |
| `config.py`        | Trung tâm cấu hình: paths, hyperparams, seed                                | Không                                |
| `preprocessing.py` | Ben Graham pipeline: crop + color norm                                      | `config`                             |
| `dataset.py`       | `DRDataset`, `MessidorDataset`, augmentations, dataloaders                  | `config`, `preprocessing`            |
| `model.py`         | `ChannelAttention`, `SpatialAttention`, `CBAM`, `MCDropout`, `CBAMResNet50` | `config`                             |
| `loss.py`          | `FocalLoss`, `compute_class_weights`                                        | `config`                             |
| `train.py`         | `Trainer` class: AMP training loop, checkpointing, early stopping           | `config`, `model`, `dataset`, `loss` |
| `evaluate.py`      | MC Dropout inference, metrics, visualization                                | `config`, `model`, `dataset`         |

---

## 8. ĐỀ XUẤT NGHIÊN CỨU (PROPOSAL)

### 8.1. Tên Đề Tài

**"Uncertainty-Aware Attention CNN cho Phân Loại Bệnh Võng Mạc Tiểu Đường trên Ảnh Đáy Mắt"**

_English: "Uncertainty-Aware Attention CNN for Diabetic Retinopathy Grading from Fundus Photographs"_

### 8.2. Mục Tiêu Nghiên Cứu

**Mục tiêu chính:**

1. Xây dựng mô hình deep learning phân loại 5 mức DR trên ảnh fundus.
2. Tích hợp cơ chế attention (CBAM) để nâng cao khả năng trích xuất đặc trưng bệnh lý.
3. Ước lượng bất định (uncertainty) dự đoán bằng MC Dropout để hỗ trợ quyết định lâm sàng.

**Mục tiêu phụ:** 4. Đánh giá khả năng tổng quát hóa trên external dataset (Messidor-2 — cross-domain). 5. So sánh hiệu quả giữa baseline ResNet-50 và CBAM-ResNet50 đề xuất. 6. Phân tích mối quan hệ giữa uncertainty và độ chính xác dự đoán.

### 8.3. Đóng Góp Dự Kiến

1. **Kiến trúc tích hợp:** Kết hợp attention (CBAM) + uncertainty estimation (MC Dropout) trong một framework thống nhất cho DR grading — ít nghiên cứu kết hợp cả hai.
2. **Ứng dụng lâm sàng:** Cung cấp mức độ tin cậy cho từng dự đoán → hỗ trợ quy trình "human-in-the-loop" trong sàng lọc DR.
3. **Đánh giá cross-domain:** Kiểm chứng trên Messidor-2 (khác hoàn toàn domain so với APTOS), phản ánh thực tế triển khai.
4. **Pipeline tái sản xuất:** Mã nguồn hoàn chỉnh, reproducible (fixed seed, K-Fold CV).

### 8.4. Phạm Vi Giới Hạn

- **Không** thực hiện lesion segmentation (chỉ image-level classification).
- **Không** sử dụng kiến trúc Vision Transformer (ViT) — tập trung vào CNN.
- **Không** deploy thành ứng dụng web/mobile (chỉ pipeline nghiên cứu).
- **Không** sử dụng federated learning hay self-supervised pre-training.

### 8.5. Phương Pháp Luận

```
1. Thu thập & khám phá dữ liệu
   ├── APTOS 2019 (3,662 ảnh, 5 lớp)
   └── Messidor-2 (690 ảnh, external test)

2. Tiền xử lý ảnh
   ├── Ben Graham preprocessing (crop + color norm)
   └── Data augmentation (Albumentations)

3. Xây dựng mô hình
   ├── Baseline: ResNet-50 + FC head
   └── Đề xuất: ResNet-50 + CBAM + MC Dropout Bayesian head

4. Huấn luyện
   ├── 5-Fold Stratified CV trên APTOS
   ├── Focal Loss + AdamW + CosineAnnealing
   └── Mixed-precision training (AMP)

5. Đánh giá
   ├── APTOS validation: Accuracy, QWK, F1, AUC
   ├── Messidor-2 external: MC Dropout inference (T=20)
   └── Uncertainty analysis: Entropy histogram, confidence scatter

6. So sánh & Phân tích
   ├── Baseline vs. CBAM-ResNet50
   ├── Uncertainty vs. Accuracy correlation
   └── Per-class performance analysis
```

---

## 9. KẾ HOẠCH THỰC HIỆN 1 THÁNG

### Tổng Quan Timeline

```
03/01 ─── Tuần 1 ─── 03/07
         EDA + Tiền xử lý + Baseline

03/08 ─── Tuần 2 ─── 03/14
         CBAM-ResNet50 Training + Tuning

03/15 ─── Tuần 3 ─── 03/21
         Messidor-2 Evaluation + Uncertainty Analysis

03/22 ─── Tuần 4 ─── 03/31
         So sánh kết quả + Viết báo cáo + Hoàn thiện
```

---

### TUẦN 1 (01/03 – 07/03): EDA + Tiền Xử Lý + Baseline

#### Ngày 1–2: Chuẩn Bị & EDA

| #   | Nhiệm vụ            | Chi tiết                                                  | Output                     |
| --- | ------------------- | --------------------------------------------------------- | -------------------------- |
| 1.1 | Setup môi trường    | `pip install -r requirements.txt`, verify GPU/CPU         | Môi trường hoạt động       |
| 1.2 | EDA APTOS           | Phân bố lớp, kích thước ảnh, trực quan hóa mẫu            | EDA notebook + figures     |
| 1.3 | EDA Messidor-2      | Phân bố lớp, so sánh domain với APTOS                     | EDA notebook               |
| 1.4 | Kiểm tra tiền xử lý | Test `preprocessing.py` trên vài ảnh, visualize trước/sau | Ảnh minh họa preprocessing |

#### Ngày 3–4: Baseline Training

| #   | Nhiệm vụ                  | Chi tiết                                                       | Output                  |
| --- | ------------------------- | -------------------------------------------------------------- | ----------------------- |
| 1.5 | Chạy baseline ResNet-50   | Notebook `02_resnet50_baseline.ipynb` (CPU: 5 epochs hoặc GPU) | Checkpoint + metrics    |
| 1.6 | Ghi nhận baseline metrics | Accuracy, loss curves, confusion matrix                        | Bảng kết quả baseline   |
| 1.7 | Debug data pipeline       | Verify augmentations, batch shapes, label correctness          | Confirmed data pipeline |

#### Ngày 5–7: GPU Setup + Full Baseline Training

| #    | Nhiệm vụ                    | Chi tiết                            | Output                    |
| ---- | --------------------------- | ----------------------------------- | ------------------------- |
| 1.8  | Setup Vast.ai (nếu cần GPU) | Upload data, run `setup_vastai.sh`  | Instance sẵn sàng         |
| 1.9  | Full baseline training      | 20 epochs, fold 0, GPU              | `resnet50_fold0_best.pth` |
| 1.10 | Ghi chép kết quả baseline   | Val Kappa, Val Acc, training curves | Baseline report           |

**Deliverables Tuần 1:**

- [x] Môi trường cài đặt xong
- [x] EDA hoàn chỉnh (APTOS + Messidor-2)
- [x] Baseline ResNet-50 trained (ít nhất fold 0)
- [x] Baseline metrics ghi nhận

---

### TUẦN 2 (08/03 – 14/03): CBAM-ResNet50 Training + Tuning

#### Ngày 8–10: Training Mô Hình Đề Xuất

| #   | Nhiệm vụ                   | Chi tiết                                       | Output            |
| --- | -------------------------- | ---------------------------------------------- | ----------------- |
| 2.1 | Train CBAM-ResNet50 fold 0 | `python src/train.py --fold 0` trên GPU        | Checkpoint fold 0 |
| 2.2 | So sánh nhanh với baseline | Val Kappa, Val Acc — CBAM vs. baseline         | Bảng so sánh      |
| 2.3 | Train folds 1–4            | Chạy tuần tự hoặc song song                    | 5 checkpoints     |
| 2.4 | Monitor & troubleshoot     | Kiểm tra loss curves, overfitting, LR schedule | Training logs     |

#### Ngày 11–14: Hyperparameter Tuning

| #   | Nhiệm vụ                | Chi tiết                            | Output                 |
| --- | ----------------------- | ----------------------------------- | ---------------------- |
| 2.5 | Thử nghiệm LR           | Test 3e-4, 1e-4, 5e-5               | Best LR                |
| 2.6 | Thử nghiệm augmentation | Thêm/bỏ CutOut, GridDistortion      | Best augmentation set  |
| 2.7 | Thử nghiệm Focal γ      | Test γ = 1.0, 2.0, 3.0              | Best γ                 |
| 2.8 | Tổng hợp best fold      | Chọn best checkpoint theo val_kappa | Final model checkpoint |

**Deliverables Tuần 2:**

- [x] CBAM-ResNet50 trained (5 folds hoặc best fold)
- [x] So sánh metrics với baseline
- [x] Hyperparameter tuning results
- [x] Best model checkpoint

---

### TUẦN 3 (15/03 – 21/03): Evaluation + Uncertainty Analysis

#### Ngày 15–17: Messidor-2 Evaluation

| #   | Nhiệm vụ                             | Chi tiết                                       | Output                    |
| --- | ------------------------------------ | ---------------------------------------------- | ------------------------- |
| 3.1 | MC Dropout inference trên Messidor-2 | `python src/evaluate.py --checkpoint best.pth` | Results CSV + metrics     |
| 3.2 | Đánh giá baseline trên Messidor-2    | Chạy baseline checkpoint trên Messidor-2       | Baseline external metrics |
| 3.3 | So sánh APTOS val vs Messidor-2      | Domain shift analysis                          | Bảng so sánh cross-domain |
| 3.4 | Confusion matrix + ROC curves        | Visualize per-class performance                | Figures                   |

#### Ngày 18–21: Uncertainty Analysis

| #   | Nhiệm vụ                      | Chi tiết                                           | Output                   |
| --- | ----------------------------- | -------------------------------------------------- | ------------------------ |
| 3.5 | Entropy histogram             | Phân bố bất định theo DR grade                     | Figure                   |
| 3.6 | Confidence vs Entropy scatter | Tương quan confidence/entropy                      | Figure                   |
| 3.7 | Phân tích case studies        | Chọn ảnh high/low uncertainty, giải thích lâm sàng | Case study report        |
| 3.8 | Referral threshold analysis   | Đề xuất ngưỡng entropy cho auto-screening          | Threshold recommendation |
| 3.9 | Per-class uncertainty         | So sánh uncertainty giữa các mức DR                | Bảng phân tích           |

**Deliverables Tuần 3:**

- [x] Messidor-2 evaluation hoàn chỉnh
- [x] Uncertainty analysis figures
- [x] Case study (high/low uncertainty examples)
- [x] Cross-domain performance comparison

---

### TUẦN 4 (22/03 – 31/03): Báo Cáo + Hoàn Thiện

#### Ngày 22–25: Tổng Hợp Kết Quả

| #   | Nhiệm vụ                 | Chi tiết                                        | Output                      |
| --- | ------------------------ | ----------------------------------------------- | --------------------------- |
| 4.1 | Tổng hợp tất cả metrics  | Bảng tổng hợp: Baseline vs CBAM vs CBAM+MC      | Final results table         |
| 4.2 | Vẽ figures cho báo cáo   | Training curves, confusion matrix, ROC, entropy | Publication-quality figures |
| 4.3 | Statistical significance | Confidence intervals từ 5-fold CV               | Thống kê                    |

#### Ngày 26–29: Viết Báo Cáo/Khóa Luận

| #   | Nhiệm vụ                         | Chi tiết                                       | Output       |
| --- | -------------------------------- | ---------------------------------------------- | ------------ |
| 4.4 | Viết Introduction & Related Work | Bối cảnh, gap, contribution                    | Chương 1–2   |
| 4.5 | Viết Methodology                 | Kiến trúc, preprocessing, training, evaluation | Chương 3     |
| 4.6 | Viết Results & Discussion        | Bảng kết quả, phân tích, so sánh               | Chương 4–5   |
| 4.7 | Viết Conclusion & Future Work    | Tóm tắt đóng góp, hướng phát triển             | Chương 6     |
| 4.8 | Tài liệu tham khảo               | Sắp xếp 32 papers                              | Bibliography |

#### Ngày 30–31: Hoàn Thiện & Nộp

| #    | Nhiệm vụ               | Chi tiết                          | Output         |
| ---- | ---------------------- | --------------------------------- | -------------- |
| 4.9  | Review & chỉnh sửa     | Đọc lại toàn bộ, fix formatting   | Final draft    |
| 4.10 | Clean code             | Comment, docstrings, PEP8         | Clean codebase |
| 4.11 | README & documentation | Hướng dẫn reproduce, requirements | README.md      |
| 4.12 | Nộp/submit             | Theo yêu cầu trường               | Submitted      |

**Deliverables Tuần 4:**

- [x] Bảng kết quả tổng hợp
- [x] Publication-quality figures
- [x] Khóa luận draft hoàn chỉnh
- [x] Codebase clean + documented

---

## 10. KẾT QUẢ KỲ VỌNG

### 10.1. APTOS Validation

| Metric       | Baseline ResNet-50 | CBAM-ResNet50 (Kỳ vọng) |
| ------------ | ------------------ | ----------------------- |
| Val Accuracy | 75–80%             | 78–85%                  |
| Val QWK      | 0.70–0.80          | 0.75–0.85               |
| Macro F1     | 0.55–0.65          | 0.60–0.72               |

### 10.2. Messidor-2 External

| Metric               | Baseline  | CBAM-ResNet50 (Kỳ vọng) |
| -------------------- | --------- | ----------------------- |
| Accuracy             | 65–75%    | 68–78%                  |
| Binary Referable AUC | 0.85–0.90 | 0.88–0.93               |
| QWK                  | 0.55–0.70 | 0.60–0.75               |

### 10.3. Uncertainty Metrics

| Kỳ vọng                              | Mô tả                                               |
| ------------------------------------ | --------------------------------------------------- |
| Correlation entropy↑ = error↑        | Ảnh dự đoán sai có entropy cao hơn ảnh dự đoán đúng |
| High entropy ở biên lớp              | Ảnh mức 1 (Mild) có entropy cao hơn mức 0 (No DR)   |
| Uncertainty giảm khi confidence tăng | Mối tương quan nghịch rõ ràng                       |

> **Lưu ý:** Hiệu suất trên Messidor-2 thường thấp hơn APTOS 5–15% do domain shift. Đây là kết quả bình thường và là điểm thảo luận quan trọng trong khóa luận.

---

## 11. HƯỚNG DẪN VIẾT KHÓA LUẬN

### 11.1. Cấu Trúc Đề Xuất

| Chương   | Nội Dung                                              | Số trang ước tính |
| -------- | ----------------------------------------------------- | ----------------- |
| 1        | Giới thiệu (bối cảnh, mục tiêu, phạm vi)              | 5–7               |
| 2        | Cơ sở lý thuyết + Các công trình liên quan            | 15–20             |
| 3        | Phương pháp nghiên cứu (kiến trúc, dữ liệu, training) | 12–15             |
| 4        | Kết quả thực nghiệm                                   | 10–15             |
| 5        | Thảo luận (phân tích, so sánh, hạn chế)               | 5–8               |
| 6        | Kết luận & Hướng phát triển                           | 3–5               |
|          | Tài liệu tham khảo                                    | 3–5               |
|          | Phụ lục (code, thêm figures)                          | 5–10              |
| **Tổng** |                                                       | **~60–80 trang**  |

### 11.2. Các Bảng/Hình Bắt Buộc

**Bảng:**

- Bảng phân bố dữ liệu APTOS + Messidor-2
- Bảng hyperparameters
- Bảng so sánh Baseline vs. CBAM-ResNet50 (APTOS val)
- Bảng kết quả Messidor-2 (external test)
- Bảng uncertainty analysis per-class

**Hình:**

- Kiến trúc mô hình (diagram)
- Tiền xử lý trước/sau (ảnh minh họa)
- Training/validation loss curves
- Confusion matrix (APTOS + Messidor-2)
- ROC curves (per-class + binary referable)
- Entropy histogram
- Confidence vs. Entropy scatter
- Case studies (high/low uncertainty images)

### 11.3. Papers Cần Trích Dẫn (Tối Thiểu)

| Chủ đề                   | Paper                              | Năm  |
| ------------------------ | ---------------------------------- | ---- |
| DR tổng quan             | Alyoubi et al. (Review)            | 2020 |
| Survey TOÀN DIỆN         | From Retinal Pixels to Patients    | 2025 |
| ResNet                   | He et al. (Deep Residual Learning) | 2015 |
| CBAM                     | Woo et al. (ECCV)                  | 2018 |
| MC Dropout               | Gal & Ghahramani                   | 2016 |
| Focal Loss               | Lin et al. (RetinaNet)             | 2017 |
| Ben Graham preprocessing | Graham (Kaggle 1st place)          | 2015 |
| APTOS dataset            | Kaggle competition                 | 2019 |
| Messidor-2 dataset       | Decencière et al.                  | 2014 |
| FDA-approved systems     | Nature Digital Medicine            | 2025 |
| ViT trong y tế           | Aburass et al.                     | 2025 |

---

## 12. TÀI LIỆU THAM KHẢO CHÍNH

Danh sách đầy đủ 32 papers với links xem tại:
📄 [essential-papers-dr-detection.md](./essential-papers-dr-detection.md)

**Top 5 papers BẮT BUỘC đọc:**

1. **"From Retinal Pixels to Patients" (2025)** — Survey toàn diện nhất, 50+ studies, benchmarking tables
2. **"Vision Transformers in Medical Imaging" (2025)** — ViT architectures, mathematical formulations
3. **"Systematic Review of FDA-Approved DR Systems" (Nature 2025)** — Real-world validation: 82 studies, 887K exams
4. **RETFound Foundation Model (Nature 2023)** — First foundation model, AUC 0.943 trên APTOS
5. **Alyoubi et al. Review (2020)** — Base review, 33 papers, dataset descriptions

---

## PHỤ LỤC

### A. Lệnh Chạy Nhanh

```bash
# Activate environment
dr-env\Scripts\activate.bat

# View config
python src/config.py

# Train CBAM-ResNet50 (fold 0, 20 epochs)
python src/train.py --fold 0 --epochs 20

# Evaluate trên Messidor-2
python src/evaluate.py --checkpoint outputs/checkpoints/cbam_resnet50_fold0_best.pth

# Train trên Vast.ai GPU
bash vast/run_training.sh resnet50 20 0
```

### B. Dependency Chính

| Package              | Version | Vai trò                        |
| -------------------- | ------- | ------------------------------ |
| PyTorch              | ≥ 2.0   | Framework deep learning        |
| torchvision          | ≥ 0.15  | Pre-trained models, transforms |
| OpenCV               | ≥ 4.8   | Image I/O, preprocessing       |
| Albumentations       | ≥ 1.3   | Data augmentation              |
| scikit-learn         | ≥ 1.3   | Metrics, K-Fold                |
| Pandas               | ≥ 2.0   | CSV/data handling              |
| Matplotlib + Seaborn |         | Visualization                  |
| tqdm                 |         | Progress bars                  |

### C. GPU Cloud Deployment

Xem hướng dẫn chi tiết tại: 📄 [VASTAI_DEPLOYMENT.md](./VASTAI_DEPLOYMENT.md)

- GPU khuyến nghị: RTX 3060/3070/4070 (8–12 GB VRAM)
- Chi phí ước tính: $0.15–$0.30/giờ
- Thời gian training: ~3–5 giờ cho 20 epochs/fold
- Tổng chi phí 5 folds: ~$5–$7

---

_Tài liệu được tạo ngày 01/03/2026 cho dự án dr-detect._
_Cập nhật lần cuối: 01/03/2026_
