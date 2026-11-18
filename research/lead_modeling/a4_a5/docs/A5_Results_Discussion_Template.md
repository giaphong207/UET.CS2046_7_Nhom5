# SP2 — Results and Discussion (MẪU, CHƯA CÓ KẾT QUẢ THỰC NGHIỆM)

> **Không điền số liệu bằng suy đoán.** Các đoạn trong dấu `[ ... ]` chỉ được điền sau khi chạy A4/A5 trên file CSV Kaggle gốc và kiểm tra output.

## 3.1. Môi trường và protocol

Protocol dự kiến dùng `diabetes_binary_health_indicators_BRFSS2015.csv`, 253.680 mẫu và 21 thuộc tính. Sau khi thực sự chạy, xác nhận số liệu này từ audit. Chia train/validation/test 70/15/15 với seed 42 và stratification. B1–B3 chọn hyperparameters bằng Stratified 3-Fold CV trên train, metric chính là `average_precision_score`; scaler B1 fit trong fold, mã ordinal giữ dạng số. A1 reuse tham số B3 và fit weighting trên full train; A3 dùng ANOVA `SelectKBest(f_classif)` fit trên train, chọn k trên validation, không chạy weighted/feature-selection CV riêng. Xem `METHODOLOGY_NOTES.md` khi đối chiếu với A2/A3 lịch sử.

**Cần điền sau khi chạy:** A4/A5 run IDs và source identity: `[...]`; SHA256: `[...];` package versions: `[...]`; search yêu cầu/thực chạy từng họ: `[...]`; số mẫu train/val/test và phân bố lớp: `[...]`; vector X trùng giữa split: `[...]`. Exact final AP ties theo thứ tự B1/B2/B3/A2/selected A3; F1 ties ưu tiên threshold cao nhất.

## 3.2. Baseline và mô hình đối chứng

| Model | AP (test) | ROC-AUC | Precision | Recall | Specificity | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 Dummy | [...] | [...] | [...] | [...] | [...] | [...] | 0.5 |
| B1 Logistic Regression | [...] | [...] | [...] | [...] | [...] | [...] | 0.5 |
| B2 Random Forest | [...] | [...] | [...] | [...] | [...] | [...] | 0.5 |
| B3 XGBoost | [...] | [...] | [...] | [...] | [...] | [...] | 0.5 |
| A1 Weighted XGBoost | [...] | [...] | [...] | [...] | [...] | [...] | 0.5 |
| A2 Weighted + val-F1 threshold | [...] | [...] | [...] | [...] | [...] | [...] | [...] |
| A3 Feature reduced | [...] | [...] | [...] | [...] | [...] | [...] | [...] |
| FINAL (locked) | [...] | [...] | [...] | [...] | [...] | [...] | [...] |

**Cần lý giải:** A2 thay đổi threshold của cùng A1, vì vậy AP/ROC-AUC phải giữ nguyên; chỉ các chỉ số theo threshold thay đổi. Nếu AP khác A1/A2, cần kiểm tra lỗi file hoặc code.

## 3.3. Ablation — ảnh hưởng của từng thành phần

- **Class weighting:** So sánh B3 với A1 tại threshold 0.5; không đổi thuật toán/hyperparameters.
- **Threshold:** So sánh A1 và A2 với cùng dự đoán xác suất nhưng ngưỡng 0.5 và ngưỡng tối đa F1 trên validation.
- **Feature reduction:** So sánh A2 với A3_k21, A3_k15, A3_k10, A3_k5 trên validation; chỉ k đã chọn được đánh giá trên test. Xác nhận ranking fit chỉ trên train.
- **Tính toán:** `AP_ref_val - AP_k_val <= 0.01` là tiêu chí đặt trước để chọn mức k nhỏ nhất; không phải kết quả được đảm bảo.

## 3.4. Độ tin cậy của cải thiện

So sánh AP FINAL minus B3 trên **cùng test**. Paired bootstrap dùng 1.000 lần thử, seed 42, cùng chỉ số lấy mẫu cho hai mô hình; loại mẫu single-class và không lấy bù. Điền attempted/valid/excluded: `[...]`; `ΔAP = [...]; CI95% percentile = [..., ...]`. Nếu khoảng chứa 0 thì chưa có bằng chứng rõ rằng chênh lệch AP khác 0 theo phép bootstrap này. Khoảng tin cậy chỉ phản ánh sai số lấy mẫu test có điều kiện theo các mô hình đã fit.

## 3.5. Error analysis

- Tổng số TP, TN, FP, FN: `[...]`.
- False Negative: `[nhận xét trên mẫu thực nghiệm sau khi xem file error_cases.csv]`.
- False Positive: `[...]`.
- Subgroup Sex và Age: báo cáo n, positive_n, status và lý do withheld cho mọi nhóm; metric withheld khi n<50 hoặc single-class. `Age` chỉ là mã nhóm 1–13, không phải số tuổi.
- **Không dùng các kết quả nhóm để tuyên bố công bằng y tế** khi chưa thẩm định dữ liệu và phương pháp.

## 3.6. Interpretability — SHAP/LIME

Nếu thực sự chạy `--shap`, biểu đồ giải thích **xác suất nhãn 1 của toàn mô hình cuối**, dùng model-agnostic permutation SHAP và background từ train. Ghi seed/settings/source rows từ metadata; summary mặc định 15 dòng là summary trên mẫu. Trình bày: `[feature nổi bật]`, `[mức độ biến thiên]`, `[đánh giá tính ổn định]`. Không đánh dấu SHAP/LIME đã verify chỉ từ kiểm tra import hoặc code.

Nếu chạy `--lime`, giải thích TP, FP và FN với LIME. Lưu ý perturbations có thể tạo tổ hợp sức khỏe phi thực tế; SHAP và LIME chỉ mô tả hành vi mô hình, **không phải ảnh hưởng nhân quả lên bệnh lý**.

## 3.7. Giới hạn nghiên cứu

Dữ liệu khảo sát tự báo cáo và cắt ngang; nhãn 1 gộp tiền đái tháo đường/đái tháo đường; không có thời điểm phát bệnh; không phải hệ thống chẩn đoán. Không thể suy rộng trực tiếp sang Việt Nam. Mẫu có vector đầu vào giống nhau không chứng minh cùng một người; việc chia theo vector trùng chỉ là phép phân tích nhạy cảm.

## 3.8. Kết luận theo bằng chứng

Chỉ được viết kết luận sau khi đối chiếu toàn bộ bảng metrics, khoảng tin cậy, ablation và biểu đồ. Nếu cải thiện không rõ, báo cáo trung thực thay vì chọn metric khác để gán kết quả tích cực.
