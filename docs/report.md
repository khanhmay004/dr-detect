# BÁO CÁO TIẾN ĐỘ ĐỀ TÀI

> **Đề tài**: *Uncertainty-Aware Attention CNN for Diabetic Retinopathy Grading from Fundus Photographs*  
> **Người thực hiện**: (bổ sung tên SV)  
> **Mục tiêu báo cáo**: Trình bày ý tưởng, pipeline, cấu trúc mô hình, lý do chọn CBAM, tiến độ hiện tại và các vấn đề đang gặp.

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

## 5.2 Kết quả chính hiện tại (Messidor-2, T=20, n=1744)

| Metric | Baseline | CBAM |
|---|---:|---:|
| Accuracy | **0.6399** | 0.6095 |
| QWK | **0.6000** | 0.5784 |
| Referable AUC | **0.8911** | 0.8778 |
| Referable Sensitivity | **0.4464** | 0.3720 |
| Referable Specificity | **0.9767** | 0.9736 |
| ECE | 0.1338 | **0.1201** |
| Brier | **0.5084** | 0.5170 |

Nhận định:

- Baseline đang tốt hơn CBAM ở hầu hết metric cốt lõi.
- CBAM chỉ nhỉnh hơn nhẹ ở ECE, nhưng không chuyển thành lợi ích lâm sàng (đặc biệt sensitivity).
- Cả hai mô hình đều thiên về specificity cao, sensitivity còn thấp cho screening.

---

## 6. Vấn đề đang gặp

Các vấn đề chính tại thời điểm báo cáo:

1. **CBAM chưa vượt baseline**
   - Đây là rủi ro học thuật chính nếu mục tiêu là chứng minh attention luôn cải thiện kết quả.
   - Tuy nhiên vẫn là kết quả khoa học hợp lệ (negative/neutral ablation).

2. **Sensitivity phát hiện referable DR còn thấp**
   - Baseline ~0.446, CBAM ~0.372.
   - Chưa đạt mức mong muốn cho kịch bản sàng lọc an toàn.

3. **Độ tin cậy lâm sàng cần hoàn thiện ở tầng policy**
   - Đã có uncertainty và calibration metrics, nhưng chưa khóa operating point cuối cùng.

4. **Cross-fold CBAM (fold 1–4) đang tạm hoãn**
   - Theo hướng hiện tại: redesign CBAM trước rồi mới mở rộng folds để tiết kiệm tài nguyên.

---

## 7. Hướng xử lý tiếp theo (ngắn hạn)

Theo trạng thái hiện tại, ưu tiên hợp lý:

1. **Threshold/referral operating-point selection** trên baseline hiện có.
2. **Temperature scaling** để cải thiện calibration hậu xử lý.
3. Chốt một **baseline policy** làm mốc chuẩn trước redesign.
4. **Redesign CBAM** có giả thuyết rõ ràng, chạy lại fold 0 + external trước, sau đó mới cân nhắc mở rộng folds 1–4.

---

## 8. Kết luận báo cáo

Đề tài đã đạt tiến độ tốt về mặt kỹ thuật và quy trình nghiên cứu:

- Pipeline đầy đủ từ preprocessing -> train -> uncertainty-aware evaluation.
- Có đánh giá external và bổ sung calibration/referral đúng định hướng luận văn.
- Kết quả hiện tại ủng hộ baseline mạnh, đồng thời chỉ ra rõ bài toán cần cải thiện (clinical sensitivity và thiết kế CBAM).

Đây là trạng thái phù hợp để chuyển sang giai đoạn tối ưu operating policy và redesign mô hình một cách có kiểm soát, minh bạch và có khả năng bảo vệ tốt trước hội đồng/hướng dẫn.

