# BÁO CÁO TIẾN ĐỘ ĐỀ TÀI

> **Đề tài**: *Uncertainty-Aware Attention CNN for Diabetic Retinopathy Grading from Fundus Photographs*  
> **Người thực hiện**: (bổ sung tên SV)  
> **Mục tiêu báo cáo**: Trình bày ý tưởng, pipeline, cấu trúc mô hình, lý do chọn CBAM, tiến độ hiện tại và các vấn đề đang gặp.  
> **Cập nhật lần cuối**: 2026-04-04 (Hoàn thành Phase 7 - Thử nghiệm cải tiến mô hình)

---

## 1. Ý tưởng nghiên cứu cốt lõi

Đề tài tập trung vào bài toán phân loại mức độ bệnh võng mạc tiểu đường (DR) theo 5 mức (0–4) từ ảnh đáy mắt, với định hướng không chỉ tối ưu độ chính xác mà còn tăng **độ tin cậy lâm sàng**.

Ý tưởng chính là kết hợp 3 thành phần trong cùng một pipeline:

1. **Attention (CBAM)**: giúp mô hình tập trung vào vùng tổn thương (xuất huyết, vi phình mạch, exudate) thay vì xử lý đều toàn ảnh.
2. **Uncertainty (MC Dropout)**: ước lượng độ bất định dự đoán để hỗ trợ cơ chế “chuyển ca không chắc chắn cho bác sĩ”.
3. **Imbalance handling (Focal Loss)**: giảm thiên lệch về lớp đông mẫu (No DR), tăng chú ý tới lớp nặng nhưng hiếm.

Định hướng học thuật của đề tài: xây dựng pipeline có thể tái lập, đánh giá xuyên miền dữ liệu (APTOS → Messidor-2), và báo cáo đầy đủ cả discrimination + calibration.

---

## 2. Pipeline tổng thể của đề tài

Pipeline hiện tại đã hoàn thiện theo luồng:

1. **Dữ liệu**
   - Train/Val: APTOS 2019 (5-fold stratified, đang dùng mạnh fold 0 cho baseline/CBAM).
   - External test: Messidor-2 (lọc ảnh gradable, dùng nhãn adjudicated).

2. **Tiền xử lý ảnh (Ben Graham pipeline)**
   - Circular crop loại vùng đen ngoài võng mạc.
   - Local color normalization: `4*I - 4*GaussianBlur(I) + 128`.
   - Resize 512×512, chuẩn hóa ImageNet.
   - Đã có cơ chế cache preprocess offline (`data/processed`) để tăng tốc.

3. **Huấn luyện**
   - Backbone: ResNet-50 pretrained ImageNet.
   - Loss: Focal Loss (alpha theo tần suất lớp, gamma=2).
   - Optimizer: AdamW + CosineAnnealingLR.
   - AMP + gradient clipping + early stopping.
   - Reproducibility: seed cố định, pipeline thống nhất baseline/CBAM.

4. **Suy luận bất định (MC Dropout)**
   - Chạy T=20 stochastic forward passes.
   - Lấy mean probability, entropy, confidence.
   - Xuất CSV mức ảnh + JSON metrics.

5. **Đánh giá**
   - Đa lớp: Accuracy, QWK, report theo lớp.
   - Referable DR (>=2): AUC, sensitivity, specificity.
   - Calibration: ECE, Brier, reliability diagram.
   - Referral analysis: coverage-performance theo entropy threshold.

---

## 3. Cấu trúc mô hình

## 3.1 Baseline model (mốc so sánh)

- ResNet-50 backbone
- Global average pooling
- MCDropout (p=0.5, bật khi inference MC)
- FC 2048 -> 5 logits

Mục tiêu baseline: tạo mốc công bằng để đo đúng đóng góp của attention.

## 3.2 Mô hình đề xuất CBAM-ResNet50

Kiến trúc:

- ResNet stem
- Layer1 + CBAM
- Layer2 + CBAM
- Layer3 + CBAM
- Layer4 + CBAM
- GAP -> MCDropout -> FC(2048->5)

CBAM gồm:

- **Channel Attention**: học “kênh nào quan trọng”.
- **Spatial Attention**: học “vị trí nào quan trọng”.

Thiết kế đặt CBAM sau mỗi stage nhằm giữ tương thích pretrained weight của ResNet-50.

---

## 4. Vì sao chọn CBAM

Lý do chọn CBAM trong bối cảnh đề tài:

1. **Phù hợp đặc thù ảnh đáy mắt**: tổn thương thường nhỏ, rải rác; attention giúp nhấn mạnh vùng có ý nghĩa bệnh học.
2. **Chi phí tham số thấp**: tăng tham số ít so với backbone, dễ triển khai trong phạm vi luận văn.
3. **Có cơ sở từ tài liệu DR gần đây**: nhiều công trình DR dùng attention dạng channel/spatial.
4. **Phù hợp thiết kế ablation**: có thể so baseline và CBAM trong cùng pipeline, chỉ khác biến attention.

Lưu ý quan trọng theo kết quả hiện tại: chọn CBAM là hợp lý về mặt giả thuyết nghiên cứu, nhưng **hiệu quả thực nghiệm hiện tại chưa vượt baseline**.

---

## 5. Tiến độ thực hiện hiện tại

## 5.1 Các phần đã hoàn thành

### Hạ tầng + huấn luyện (Phase 0–2)

- Sửa các vấn đề kỹ thuật quan trọng:
  - deterministic validation cho MC Dropout
  - gradient clipping
  - đồng bộ pipeline baseline/CBAM
  - wildcard checkpoint resolution
- Baseline fold 0 huấn luyện hoàn chỉnh (GPU).
- CBAM fold 0 huấn luyện hoàn chỉnh.
- Đánh giá external Messidor-2 cho cả baseline và CBAM.
- Tổng hợp artifact vào `phase2-results/`.

### Đánh giá nâng cao (Phase 3A–3D)

- Đã thêm vào `evaluate.py`:
  - ECE
  - Brier score
  - Reliability diagram
  - Referral curve (coverage vs accuracy/sensitivity/specificity)
- Đã rerun full Messidor-2 cho baseline và CBAM với output calibration/referral đầy đủ.
- Tổng hợp artifact vào `phase3-results/`.

### Phân tích nguyên nhân và lập kế hoạch cải tiến (Phase 3E)

- ✅ Hoàn thành phân tích root-cause cho vấn đề recall thấp ở minority classes
- ✅ Lập kế hoạch cải tiến chi tiết với 40+ TODO items (`plans/07-improve-model.md`)
- ✅ Thiết kế 3 phase: Post-hoc fixes → Structural changes → Ordinal loss

### Triển khai và thử nghiệm cải tiến (Phase 7 - Hoàn thành 2026-04-03/04)

**Phase 7A - Post-hoc Fixes** ✅
- Temperature scaling: T=1.0321 (cải thiện ECE: 4.88% → 4.33%, không đáng kể)
- Threshold tuning: Tất cả thresholds = 1.0 (không có cải thiện)
- **Kết luận**: Post-hoc fixes không thể bù đắp cho vấn đề cấu trúc huấn luyện

**Phase 7B - Structural Changes** ✅
- Huấn luyện 3 mô hình mới trên data_split (2,489 train / 623 val):
  1. **Original Baseline** (20260403_120248): Val QWK 0.9153 - làm mốc chuẩn
  2. **Sampler-Only** (20260403_185935): Val QWK 0.9009 - chỉ dùng balanced sampler
  3. **Full Improved** (20260403_192310): Val QWK 0.9169 - tất cả cải tiến
- Đánh giá đầy đủ trên cả APTOS test và Messidor-2

**Cải tiến áp dụng trong Full Improved Model**:
- ✅ Balanced sampler (đảm bảo mỗi class có số mẫu bằng nhau mỗi epoch)
- ✅ Classifier head sâu hơn: 2048 → 512 → 5 (thay vì 2048 → 5)
- ✅ Label smoothing: 0.1
- ✅ LR warmup: 2 epochs (LinearLR → CosineAnnealingLR)
- ✅ Giảm dropout: 0.3 (từ 0.5)
- ✅ Extended training: 25 epochs

## 5.2 Kết quả chính hiện tại

### So sánh trên APTOS Validation (N=623)

| Metric | Original Baseline | Sampler-Only | Full Improved | Model tốt nhất |
|--------|------------------|--------------|---------------|----------------|
| Accuracy | 84.43% | 80.26% | **85.07%** | Full Improved ✅ |
| QWK | 0.9153 | 0.9009 | **0.9169** | Full Improved ✅ |
| AUC | **0.9904** | 0.9839 | 0.9879 | Original |
| Sensitivity | 0.9289 | 0.8972 | **0.9565** | Full Improved ✅ |

### So sánh trên APTOS Test (N=550)

| Metric | Original Baseline | Full Improved | Thay đổi |
|--------|------------------|---------------|----------|
| Accuracy | 81.09% | **82.73%** | **+1.64 pp** ✅ |
| QWK | 0.8959 | **0.8972** | +0.0013 ✅ |
| Sensitivity | 87.44% | **94.17%** | **+6.73 pp** ✅ |
| ECE | 0.0478 | **0.0377** | **-0.0101** ✅ |

**Nhận xét APTOS Test**: Mô hình cải tiến cho kết quả tốt hơn đáng kể - độ chính xác tăng, sensitivity tăng mạnh, calibration tốt hơn.

### So sánh trên Messidor-2 External Test (N=1744) — **THẤT BẠI NGHIÊM TRỌNG**

| Metric | Original Baseline | CBAM | Full Improved | Model tốt nhất |
|--------|------------------|------|---------------|----------------|
| Accuracy | **0.6279** | 0.6095 | 0.6244 | **Original** |
| QWK | **0.6233** | 0.5784 | **0.4582** ❌ | **Original** |
| Referable AUC | **0.8630** | 0.8778 | 0.8602 | CBAM |
| Referable Sensitivity | **0.4398** | 0.3720 | **0.3435** ❌ | **Original** |
| Referable Specificity | 0.9744 | 0.9736 | **0.9845** | Full Improved |
| ECE | **0.1155** | 0.1201 | **0.1601** ❌ | **Original** |
| Brier | **0.5231** | 0.5170 | **0.5645** ❌ | **CBAM** |

**Thất bại nghiêm trọng của Full Improved trên Messidor-2**:
- QWK giảm **26.5%** (0.6233 → 0.4582)
- Mild DR recall: 9.26% → **0.74%** (bỏ sót **92% ca bệnh**)
- Severe DR recall: 56.00% → **28.00%** (giảm **một nửa**)
- Sensitivity giảm 9.63 pp
- Calibration xấu hơn (ECE tăng +0.0446)

### Per-Class Performance (Messidor-2)

| Grade | Original Recall | Full Improved Recall | Thay đổi |
|-------|----------------|---------------------|----------|
| 0 - No DR | 0.9420 | 0.9853 | +0.0433 ✅ |
| 1 - Mild | **0.0926** | **0.0074** | **-0.0852** ❌ THẢM HỌA |
| 2 - Moderate | **0.1614** | 0.1470 | -0.0144 |
| 3 - Severe | **0.5600** | **0.2800** | **-0.2800** ❌ |
| 4 - Proliferative | **0.4000** | 0.3714 | -0.0286 |

---

## 6. Vấn đề đã gặp và bài học rút ra

### 6.1 Vấn đề từ Phase 2-3 (đã được giải quyết)

1. **CBAM chưa vượt baseline**
   - ✅ Đã xác nhận qua thực nghiệm: CBAM không cải thiện performance
   - Đây là kết quả khoa học hợp lệ (negative ablation result)

2. **Sensitivity phát hiện referable DR còn thấp**
   - Original Baseline: ~0.440, CBAM: ~0.372
   - ✅ Đã thử cải tiến trong Phase 7

3. **Độ tin cậy lâm sàng cần hoàn thiện ở tầng policy**
   - ✅ Đã thêm uncertainty và calibration metrics
   - ✅ Đã thử temperature scaling và threshold tuning (không cải thiện)

### 6.2 Phát hiện nghiêm trọng từ Phase 7 ⚠️

**Vấn đề lớn nhất**: **Cải tiến trên internal validation ≠ Cải tiến trên external test**

Mô hình Full Improved:
- ✅ Cải thiện trên APTOS validation (+0.64% accuracy, +2.76 pp sensitivity)
- ✅ Cải thiện trên APTOS test (+1.64% accuracy, +6.73 pp sensitivity)
- ❌ **THẤT BẠI THẢM HẠI** trên Messidor-2 (-26.5% QWK, bỏ sót 92% ca Mild DR)

### 6.3 Nguyên nhân phân tích

1. **Overfitting vào phân phối APTOS**:
   - Balanced sampler + deeper classifier head tăng khả năng ghi nhớ dataset
   - Mô hình học được các pattern đặc thù của APTOS thay vì đặc điểm DR tổng quát

2. **Tăng model capacity phản tác dụng**:
   - Classifier head sâu hơn (2048→512→5) cho phép mô hình memorize artifacts
   - Giảm dropout (0.5→0.3) làm mất tính regularization

3. **Domain shift amplification**:
   - Cải tiến tối ưu cho APTOS distribution
   - Không generalize được sang Messidor-2 (nguồn ảnh khác, phân phối lớp khác)

### 6.4 Cross-fold CBAM (fold 1–4) - Tạm hoãn

- Theo kết quả hiện tại: CBAM không cải thiện performance
- Quyết định: Tập trung vào baseline, không tiếp tục train CBAM folds 1-4

---

## 7. Quyết định cuối cùng và hướng tiếp theo

### 7.1 Quyết định về mô hình production

**✅ CHỌN Original Baseline (20260403_120248) làm mô hình chính**

Lý do:
- External generalization tốt hơn (Messidor-2 QWK: 0.6233 vs 0.4582)
- An toàn hơn cho triển khai lâm sàng
- Ít rủi ro catastrophic failure trên minority classes
- Conservative bias (thiên về Grade 0) giúp generalize tốt hơn

**❌ LOẠI BỎ Full Improved Baseline**

Lý do:
- Thất bại nghiêm trọng trên external data (QWK giảm 26.5%)
- Overfitting vào APTOS distribution
- Không an toàn cho production

**❌ LOẠI BỎ CBAM**

Lý do:
- Không cải thiện performance so với baseline
- Thêm complexity nhưng không có lợi ích

### 7.2 Bài học quan trọng cho luận văn

1. **Validation đa tầng là bắt buộc**:
   - Internal validation không đủ
   - Phải validate trên nhiều external datasets
   - Single-dataset validation có thể gây nhầm lẫn nghiêm trọng

2. **Cải tiến cần được test kỹ trên external data**:
   - Không thể tin vào internal validation improvements
   - Phải prioritize external performance metrics

3. **Model simplicity có thể tốt hơn complexity**:
   - Baseline đơn giản generalize tốt hơn improved model phức tạp
   - Deeper head tăng overfitting capacity

4. **Conservative models có giá trị**:
   - Bias về Grade 0 giúp tránh catastrophic failures
   - Quan trọng cho clinical deployment

### 7.3 Hướng nghiên cứu tiếp theo (nếu có thời gian)

**Ngắn hạn (ưu tiên cao)**:
1. ~~Threshold/referral operating-point selection~~ ✅ Đã test (không cải thiện)
2. ~~Temperature scaling~~ ✅ Đã test (không cải thiện)
3. Viết luận văn với kết quả hiện có
4. Phân tích sâu về domain shift

**Dài hạn (nếu mở rộng)**:
1. Multi-dataset training (APTOS + Messidor-2 mixed)
2. Domain adaptation techniques
3. Simpler architectures với better regularization
4. Focus on domain-agnostic features

### 7.4 Cấu trúc luận văn đề xuất

**Chương 4 - Kết quả thực nghiệm** sẽ bao gồm:

1. **Baseline vs CBAM comparison**:
   - Kết luận: CBAM không cải thiện performance
   - Negative result vẫn là scientific contribution

2. **Improvement attempts analysis** (Phase 7):
   - Thành công trên internal validation
   - Thất bại trên external validation
   - Phân tích nguyên nhân và bài học

3. **Domain shift analysis**:
   - APTOS vs Messidor-2 performance gap
   - Per-class performance comparison
   - Calibration differences

4. **Final model selection**:
   - Original Baseline as best model
   - Justification dựa trên external performance

**Điểm mạnh của luận văn**:
- Quy trình nghiên cứu chặt chẽ, có kiểm soát
- External validation nghiêm túc
- Phân tích thất bại và bài học rõ ràng
- Negative results có giá trị khoa học

---

## 8. Kết luận báo cáo (Cập nhật 2026-04-04)

### Thành tựu đạt được

Đề tài đã hoàn thành đầy đủ quy trình nghiên cứu khoa học nghiêm túc:

✅ **Pipeline đầy đủ và chuyên nghiệp**:
- Preprocessing chuẩn (Ben Graham pipeline)
- Training với reproducibility (fixed seeds, controlled experiments)
- Uncertainty-aware evaluation (MC Dropout với T=20)
- External validation nghiêm túc (Messidor-2)

✅ **Đánh giá toàn diện**:
- Discrimination metrics (Accuracy, QWK, AUC)
- Calibration metrics (ECE, Brier, reliability diagrams)
- Clinical metrics (Sensitivity, Specificity)
- Uncertainty analysis (entropy, confidence)
- Referral curve analysis

✅ **Thực nghiệm có kiểm soát**:
- 3 models trained: Original Baseline, Sampler-Only, Full Improved
- 6+ evaluation runs trên APTOS test và Messidor-2
- Post-hoc fixes tested (temperature scaling, threshold tuning)

✅ **Kết quả khoa học có giá trị**:
- Baseline ResNet-50: QWK 0.9153 (APTOS val), 0.6233 (Messidor-2)
- CBAM không cải thiện performance (negative result có giá trị)
- **Phát hiện quan trọng**: Internal validation improvements không đảm bảo external generalization

### Đóng góp khoa học

1. **Empirical evidence về CBAM cho DR grading**:
   - Chứng minh CBAM không phải luôn cải thiện performance
   - Negative ablation result là đóng góp khoa học hợp lệ

2. **Phân tích domain shift APTOS→Messidor-2**:
   - Documented performance drops
   - Per-class analysis
   - Calibration differences

3. **Lessons về model improvement**:
   - Balanced sampling có thể gây overfitting
   - Deeper architectures không phải luôn tốt hơn
   - External validation là bắt buộc

4. **Uncertainty quantification framework**:
   - MC Dropout implementation
   - Calibration analysis
   - Referral mechanism design

### Kết quả cuối cùng

**Mô hình tốt nhất**: Original Baseline ResNet-50 (20260403_120248)

| Dataset | Accuracy | QWK | Sensitivity | ECE |
|---------|----------|-----|-------------|-----|
| APTOS Val | 84.43% | 0.9153 | 92.89% | N/A |
| APTOS Test | 81.09% | 0.8959 | 87.44% | 0.0478 |
| Messidor-2 | 62.79% | 0.6233 | 43.98% | 0.1155 |

### Trạng thái luận văn

Đề tài đã sẵn sàng cho việc:
- ✅ Viết luận văn hoàn chỉnh
- ✅ Bảo vệ trước hội đồng
- ✅ Publication (nếu cần)

**Điểm mạnh để bảo vệ**:
1. Quy trình nghiên cứu chặt chẽ và có kiểm soát
2. External validation nghiêm túc (không chỉ dựa vào internal validation)
3. Phân tích thất bại minh bạch (improvement attempts that failed)
4. Negative results với phân tích nguyên nhân rõ ràng
5. Clinical-aware evaluation (sensitivity, calibration, uncertainty)

**Câu chuyện luận văn**:
- Bắt đầu với hypothesis: CBAM + MC Dropout sẽ cải thiện DR grading
- Thực hiện nghiêm túc: Baseline → CBAM → Improvements
- Phát hiện: CBAM không cải thiện, improvements overfits
- Kết luận: Simpler baseline model tốt nhất cho external generalization
- Bài học: External validation và domain generalization là quan trọng nhất

Đây là trạng thái **vững chắc** để chuyển sang giai đoạn viết luận văn và chuẩn bị bảo vệ.

