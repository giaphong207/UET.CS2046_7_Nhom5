# Task tạo trên CoReach (Lead Modeling) — A4 / A5

Mỗi task chỉ đánh dấu **Done** khi có bằng chứng có thể kiểm tra. Nêu rõ người chính và người cùng làm theo đóng góp thật; tránh tạo việc trùng để lấy điểm.

## A4.1 — [SP2] Tích hợp pipeline B1 và kiểm định schema BRFSS
- Blocked by: B1 (pipeline stub), A2 V2.
- Nội dung: xác minh CSV gốc, 21 feature, class prevalence, SHA256, vector trùng, split 70/15/15 theo seed 42.
- Deliverables: `dataset_audit.json`, `split_indices.npz`, `manifest.json` sau khi chạy thật.
- Done khi: dữ liệu đúng tên/schema; train/val/test không giao nhau; không xóa vector trùng tùy tiện.

## A4.2 — [SP2] Huấn luyện baseline B0–B3
- Blocked by: A4.1.
- Nội dung: Dummy, Logistic Regression, Random Forest, XGBoost; CV 3 fold và metric AP.
- Deliverables: `cv_results.csv`, `validation_metrics.csv`, script modeling.
- Done khi: có kết quả chạy trên CSV thật, phân biệt B0 và các baseline.

## A4.3 — [SP2] Tối ưu và xử lý imbalance (class weighting)
- Blocked by: A4.2.
- Nội dung: giữ cấu hình B3 đã tuning để so sánh có/không trọng số lớp; A1 fit toàn bộ train và ratio tính từ nhãn train đó. Wrapper hỗ trợ ratio theo mỗi fit, nhưng weighted CV riêng chưa được thực hiện.
- Deliverables: model A1, phân tích B3/A1.
- Done khi: có cùng protocol và mô tả đánh đổi.

## A4.4 — [SP2] Tối ưu ngưỡng F1 trên validation
- Blocked by: A4.3.
- Nội dung: ngưỡng 0.5 và argmax F1 validation; không chọn ngưỡng từ test.
- Deliverables: `selection.json`, metrics A1/A2.
- Done khi: chứng minh AP/ROC-AUC của A1/A2 không đổi, giải thích Precision/Recall.

## A4.5 — [SP2] Nghiên cứu giảm đặc trưng 21/15/10/5
- Blocked by: A4.3.
- Nội dung: ANOVA SelectKBest(f_classif) fit trên train trong pipeline; k chọn theo AP validation, sai khác tối đa 0.01; không báo cáo A3 CV riêng.
- Deliverables: `feature_sets.json`, validation metrics mọi A3_k, test metrics chỉ k đã chọn.
- Done khi: không có feature-selection leakage, có bảng đánh đổi AP và số biến.

## A4.6 — [SP2] Khóa cấu hình và chạy test cuối
- Blocked by: A4.4 và A4.5.
- Nội dung: lock final model/threshold, đánh giá test một lần theo quyết định đã chọn.
- Deliverables: `test_metrics.csv`, `test_predictions.csv`, `fitted_models.joblib`, README tái lập.
- Done khi: hash và config đầy đủ, không đổi phương pháp dựa trên test.

## A5.1 — [SP2] Bảng metric và PR/ROC curve
- Blocked by: A4.6.
- Deliverables: `precision_recall_test.png`, `roc_test.png`, `confusion_matrix_final.png`.
- Done khi: số liệu phản ánh test đã khóa, có mô tả threshold.

## A5.2 — [SP2] Ablation và paired bootstrap AP
- Blocked by: A4.6.
- Deliverables: bảng ablation B3/A1/A2/A3, `paired_bootstrap_ap.csv`, CI 95% và phân tích.
- Done khi: dùng cùng bootstrap indices; không khẳng định cải thiện khi CI chứa 0.

## A5.3 — [SP2] SHAP global/local (mở rộng)
- Blocked by: A4.6.
- Deliverables: SHAP beeswarm, bar, waterfall cho toàn mô hình cuối.
- Done khi: nêu rõ background, đơn vị xác suất, giới hạn giải thích.

## A5.4 — [SP2] LIME cho TP/FP/FN (mở rộng)
- Blocked by: A4.6.
- Deliverables: `lime_TP.*`, `lime_FP.*`, `lime_FN.*` nếu từng loại xuất hiện.
- Done khi: chú giải đúng local explanations, không suy ra nhân quả.

## A5.5 — [SP2] Results, Discussion, error & subgroup analysis
- Blocked by: A5.1–A5.4 (riêng XAI chỉ khi được triển khai).
- Deliverables: chương Results/Discussion, `error_cases.csv`, `subgroups_test.csv`, limitation.
- Done khi: mọi số liệu có thể truy ngược manifest, CSV, code và log.
