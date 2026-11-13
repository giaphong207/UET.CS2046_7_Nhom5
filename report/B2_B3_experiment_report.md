# B2/B3 – Thiết kế thực nghiệm & Kết quả (nền SP2)

**Phụ trách:** Thành viên B (Data & Pipeline) — đồng sở kết quả với A (mô hình) và C (baseline)
**Môi trường:** Kaggle Notebook (GPU T4), seed = 42, LightGBM + imbalanced-learn, thời gian chạy ~58 phút (đo ở LightGBM CPU; bản GPU build sẵn có trong image Kaggle)
**Cơ sở học thuật:** `report/SP1_survey_report.md` (SP1) + `report/references.bib`
**Mã nguồn:** `code/b2_b3_pipeline.py` · Số liệu gốc: `report/results/B2_B3/` (7 CSV + 3 file `.npy` chỉ số split)
**Biểu đồ:** `report/figures/B2_B3/b2b3_*.png` (sinh tự động bằng `code/make_report_figures.py`)

## 1. Protocol đánh giá (chuẩn hóa, cả 4 thành viên cùng áp dụng)

- **Chỉ số chính: AP (Average Precision)** — phù hợp dữ liệu mất cân bằng 86/14, bất biến theo ngưỡng. Accuracy chỉ báo cáo tham chiếu, không dùng để chọn.
- **Ngưỡng quyết định:** so 2 ngưỡng: 0,5 (mặc định) và **ngưỡng tối đa F1 lớp bệnh chọn trên tập validation rồi khóa trước khi test** — không dùng test để chọn bất kỳ tham số nào.
- **Ablation:** mỗi dòng thí nghiệm chỉ thay đổi đúng 1 yếu tố (dedup / noise / resample), đánh giá bằng 5-fold stratified CV **trên tập train**; resample chỉ chạy bên trong fold (không leakage).
- **Ý nghĩa thống kê:** paired bootstrap 1.000 lần trên cùng tập test → 95% CI cho chênh lệch AP; khác biệt chỉ tuyên bố "đáng kể" khi CI không chứa 0.
- **Ngưỡng F1:** lưới 199 mức trong khoảng 0,01–0,99 (ngoài khoảng này F1 về 0 ở cả hai đầu) — mốc 0,5 có thể nằm ngoài vùng tối ưu với mô hình xác suất không đạt 0,5.

## 2. Tiền xử lý (B3) – ablation yếu tố một

Dữ liệu gốc 253.680 dòng. Ba biến tiền xử lý, mỗi dòng chỉ đổi 1 yếu tố:

| Biến | n_final | CV-AP | ΔAP | Kết luận |
|---|---:|---:|---:|---|
| **dedup + majority-vote** (baseline pipeline) | 227.908 | **0,4094** | 0,0000 | **Chọn** |
| dedup + giữ noise | 229.474 | 0,4056 | −0,0038 | Loại |
| không dedup + majority-vote | 227.908 | 0,4068 | −0,0026 | Loại |

- dedup bỏ 24.206 dòng trùng (9,54%) rồi majority-vote gộp 1.566 nhóm nhãn mâu thuẫn → final **227.908**.
- Mỗi yếu tố góp một lượng nhỏ nhưng **cùng chiều dương** (dedup +0,0026; majority-vote +0,0038) → giữ cả hai, ghi rõ lý do (SP2 §tiền xử lý).

## 3. Ma trận mô hình × chiến lược mất cân bằng (CV trên train, 5 fold)

| Mô hình | none | class_weight | SMOTE | ADASYN | undersample |
|---|---|---|---|---|---|
| Dummy B0 (đề cương §4.4) | *tự động* | — | — | — | — |
| Logistic (C) | **0,4094** | 0,4077 | 0,4073 | 0,4061 | 0,4074 |
| Decision Tree (C) | 0,1957 | 0,1907 | 0,1947 | 0,1950 | 0,2085 |
| Random Forest (C) | 0,4029 | 0,3975 | 0,3886 | 0,3867 | 0,3989 |
| **LightGBM (A, đề xuất)** | 0,4357 | **0,4364** | 0,4346 | 0,4357 | 0,4225 |
| Stacking RF+LGBM (A) | 0,4361 | 0,4366 | 0,3328 | 0,3416 | 0,4253 |

*(AP CV trung bình; ± std đầy đủ trong `grid_model_x_resample.csv`)*

**Dummy B0 (đề cương §4.4):** `DummyClassifier(strategy="prior")` — luôn đoán lớp đa
số → AP = tỷ lệ lớp dương (≈0,139), recall lớp bệnh = 0, accuracy = 86,1%. Đây là
**mốc tham chiếu** chứng minh mô hình thực sự học được chứ không phải hưởng lợi từ
mất cân bằng. Giá trị đầy đủ trong `grid_model_x_resample.csv` (`model = dummy`).

**Quy tắc chọn:** AP cao nhất trong nhóm F1-macro ≥ max − 0,01, tie-break theo recall lớp bệnh → **class_weight (AP = 0,4364, recall lớp bệnh = 0,780)**.

Các đọc chính: (i) class_weight nâng recall lớp bệnh từ 0,158 (none) lên 0,780 gần như không mất AP — đây là hiệu ứng chính; (ii) SMOTE/ADASYN **không vượt** class_weight về AP trên LightGBM dù tốn gấp 3–6 lần thời gian; (iii) undersample nhanh nhất (4,9s vs 61,1s) nhưng tụt 0,014 AP; (iv) RF bị "trội" với resample (SMOTE làm recall tụt về 0,201) — khác LightGBM, ghi vào thảo luận.

## 4. Kết quả cuối trên TEST (fit 1 lần trên train, 2 ngưỡng)

| Mô hình (strategy = class_weight) | AP | Recall bệnh | Precision bệnh | F1-macro | AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM (đề xuất)** — ngưỡng VAL | **0,4368** | 0,595 | 0,394 | **0,677** | **0,826** | **0,372** |
| LightGBM — ngưỡng 0,5 | 0,4368 | 0,776 | 0,322 | 0,636 | 0,826 | 0,366 |
| Stacking | 0,4367 | 0,588 | 0,397 | 0,678 | 0,826 | 0,372 |
| Random Forest (VAL) | 0,3932 | 0,550 | 0,383 | 0,666 | 0,802 | 0,345 |
| Logistic (VAL) | 0,4137 | 0,572 | 0,386 | 0,671 | 0,819 | 0,357 |
| Decision Tree | 0,1897 | 0,288 | 0,295 | 0,585 | 0,585 | 0,171 |

AP bất biến giữa 2 ngưỡng của cùng mô hình (đúng tính chất của chỉ số). Ngưỡng tối ưu F1 trên VAL đã chọn: ~0,4 (giá trị exact trong `final_test_metrics.csv`).

## 5. Ý nghĩa thống kê (paired bootstrap 1.000 lần)

| So sánh | ΔAP | 95% CI | p | Kết luận |
|---|---:|---|---:|---|
| LightGBM − Logistic | **+0,0231** | [+0,0162, +0,0303] | <0,001 | **Đáng kể** |
| LightGBM − Random Forest | **+0,0436** | [+0,0354, +0,0510] | <0,001 | **Đáng kể** |
| LightGBM − Stacking | +0,0000 | [−0,0020, +0,0021] | 0,948 | Không đáng kể |

## 6. Nghiệm mở rộng phụ – bài 3 lớp

| Strategy | AP (lớp "tiểu đường") | F1-macro |
|---|---:|---:|
| none | **0,4389** | 0,4028 |
| class_weight | 0,4119 | **0,4267** |

SMOTE/ADASYN không chạy trên bài 3 lớp (lớp tiền tiểu đường chỉ 1,8% → oversampling không đáng tin). class_weight nâng F1-macro nhưng tụt AP của lớp dương — đánh đổi điển hình; đủ làm nghiệm phụ, **củng cố quyết định lấy 2 lớp làm bài chính**.

## 6b. Biểu đồ

| Hình | Nội dung |
|---|---|
| `report/figures/B2_B3/b2b3_ap_by_strategy.png` | AP theo 5 chiến lược mất cân bằng × 5 mô hình |
| `report/figures/B2_B3/b2b3_recall_vs_ap.png` | Đánh đổi recall lớp bệnh ↔ AP của LightGBM |
| `report/figures/B2_B3/b2b3_ablation_preprocessing.png` | ΔAP của 3 biến tiền xử lý |
| `report/figures/B2_B3/b2b3_bootstrap_ci.png` | ΔAP + 95% CI của paired bootstrap |
| `report/figures/B2_B3/b2b3_final_test_metrics.png` | F1-macro & recall bệnh theo mô hình × quy tắc ngưỡng |
| `report/figures/B2_B3/b2b3_confusion_primary.png` | Confusion matrix trên TEST (LightGBM + class_weight) |
| `report/figures/B2_B3/b2b3_aux_3class.png` | Kết quả nghiệm phụ 3 lớp |

## 6c. Đánh giá bổ sung theo đề cương

> **Lưu ý:** các file được liệt kê trong mục này (`group_split_sensitivity.csv`,
> `subgroup_age_sex.csv`, `test_predictions.npz` và 4 hình `b2b3_*.png` tương ứng)
> **được sinh ra khi chạy `code/b2_b3_pipeline.py` + `code/make_report_figures.py`**
> (chưa có trong repo hiện tại vì pipeline chưa được chạy lại sau khi bổ sung các
> bước này). Chạy lại pipeline là đủ.

**Độ nhạy với trùng lặp (đề cương §4.1).** Đề cương yêu cầu "nếu có nhiều vector giống
nhau, bổ sung đánh giá chia theo nhóm vector trùng để kiểm tra độ nhạy của kết quả"
và cảnh báo không nên xóa mọi dòng trùng vì biến mã hóa theo nhóm có thể khiến hai
người khác nhau có cùng vector đầu vào. Nhóm bổ sung:

- Nhóm vector trùng = hash của 21 đặc trưng (mỗi nhóm gồm các dòng có **cùng toàn
  bộ vector đặc trưng**). Không gọi là "chia theo bệnh nhân" — file không có mã định danh.
- Đánh giá bằng **GroupKFold** (mỗi nhóm chỉ nằm trong một fold) cùng model
  (LightGBM + class_weight) và cùng protocol, rồi so với split stratified thường.
- Kết quả: `report/results/B2_B3/group_split_sensitivity.csv` + biểu đồ
  `report/figures/B2_B3/b2b3_group_split_sensitivity.png`.

**Phân tích theo nhóm tuổi / giới (đề cương §4.6).** Báo cáo AP, recall, precision
theo từng mức `Age` và `Sex` trên test, **luôn kèm số mẫu** từng nhóm để tránh kết
luận trên nhóm quá nhỏ (bỏ nhóm có n < 30). Kết quả:
`report/results/B2_B3/subgroup_age_sex.csv` + `report/figures/B2_B3/b2b3_subgroup_age_sex.png`.

**Đường PR và ROC (đề cương §3).** `report/figures/B2_B3/b2b3_precision_recall_curve.png`
và `b2b3_roc_curve.png`, dựng từ xác suất dự đoán đã lưu
(`report/results/B2_B3/test_predictions.npz`) — có đường tham chiếu base rate trên PR.

**Chuẩn hóa biến (đề cương §4.2.2).** Logistic Regression chạy trong
`Pipeline([StandardScaler, LogisticRegression])`; mô hình cây (RF/LightGBM/DT) không
chuẩn hóa (đã ghi trong README §Protocol).

**Mã băm dữ liệu (đề cương §3.1).** SHA-256 của 1 MB đầu file + kích thước file được
in ra log mỗi lần chạy, ghi trong `run_log.txt`.

## 7. Thảo luận (phần "ưu điểm – hạn chế – hướng phát triển", nội dung 8)

**Ưu điểm:** pipeline tái lập được 100% (seed, chỉ số split, thư viện ghi trong repo); chiến thắng LightGBM+class_weight có bằng chứng thống kê (2/3 bootstrap p<0,001); phát hiện "stacking không thêm giá trị" giúp rút gọn mô hình cuối xuống LightGBM đơn giản — kết luận dựa trên số, không dựa trên thói quen.

**Hạn chế (nêu thẳng):** (1) F1-macro 0,677 < mục tiêu 0,75 và AUC 0,826 < 0,85 đặt trước nguyên nhân: dữ liệu tự báo cáo có noise nội tại, lớp bệnh 13,9% nhỏ; (2) RF×SMOTE và stacking×SMOTE/ADASYN có phản ứng bất thường (AP tụt) — do resample áp ở ngoài rồi CV-inner của stacking tự split lại, thiết kế nên được chỉ ra; (3) ngưỡng tối ưu F1 trên VAL có thể drift khi dữ liệu thay đổi — cần recalibration định kỳ trong bối cảnh ứng dụng.

**Hướng phát triển:** (a) calibration (Platt/isotonic) thay vì grid threshold; (b) ensemble theo hướng Bagging LightGBM (không stacking) — do stacking đã cho thấy không thêm giá trị; (c) nếu được thêm dữ liệu: cross-dataset validation trên tệp 50/50 để kiểm tra tính tổng quát.

## 8. Tái lập

```
pip install -r requirements.txt

# Kaggle: Notebook -> Add Input -> dataset "diabetes-health-indicators-dataset"
#         Settings -> Accelerator: GPU T4
python code/b2_b3_pipeline.py      # ~58 phút (CPU) / nhanh hon khi LightGBM chay GPU
python code/make_report_figures.py # ve bieu do B2/B3 tu report/results/*.csv

Ket qua ghi thang truc tiep vao report/results/B2_B3/ (7 CSV + 3 idx_*.npy)
So ket qua: doi chieu voi report/results/*.csv da commit (cung seed 42, dung so)
```
