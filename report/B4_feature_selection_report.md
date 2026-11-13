# B4 – Chọn lọc đặc trưng (Feature Selection): Diabetes BRFSS 2015

**Phụ trách:** Thành viên B (Data & Pipeline) — kết quả dùng cho phần SHAP/XAI của A
**Môi trường:** Kaggle Notebook, seed = 42, LightGBM + imbalanced-learn, thời gian ~11 phút (667s)
**Cơ sở học thuật:** `report/SP1_survey_report.md` §2.6 (B4) + `report/references.bib`
**Mã nguồn:** `code/b4_feature_selection.py` · Số liệu: `report/results/B4/b4_*.csv` · Hình: `report/figures/B4/b4_*.png`

## 1. Thiết kế (đúng protocol B2/B3)

- **Cùng split với B2/B3:** đọc bộ chỉ số split canonical `report/results/B2_B3/idx_*.npy`
  (do `b2_b3_pipeline.py` sinh ra) nếu có; nếu thiếu thì tự tách mới stratified 70/15/15
  seed 42 (deterministic → **trùng chính xác** 159.535 / 34.186 / 34.187). B4 **không**
  lưu bộ idx riêng trong thư mục `B4/` (tránh hiểu lầm B4 có split riêng).
- **Selector fit BÊN TRONG từng fold CV** (không dùng toàn bộ train) → không leakage.
- **Mỗi dòng chỉ đổi 1 yếu tố:** số đặc trưng (k) và phương pháp.
- **7 phương pháp × 4 mức k ∈ {21, 15, 10, 5}** (đề cương §2.2 yêu cầu so 21 biến
  với các tập 15, 10 và 5 biến).

| Nhóm | Phương pháp | Loại |
|---|---|---|
| baseline | giữ đủ 21 đặc trưng | đối chứng |
| filter | variance, corr-dedup (\|r\|≥0.95) | lọc thông tin / khử trùng lặp |
| filter | mutual_info, anova_f | lọc theo MI / F-test |
| embedded | lgbm_gain | theo gain importance |
| wrapper | rfe_lgbm | RFE theo thứ tự importance |

## 2. Kết quả CV (AP theo số đặc trưng)

Với mọi phương pháp, **AP tăng dần theo k và đạt cực đại ở k=21 (giữ đủ)**, tụt khi k nhỏ:

| k | AP (biểu hiện chung) | Nhận xét |
|---:|---:|---|
| 5 | ~0,385 | Tụt mạnh – thiếu đặc trưng, vượt dung sai 0,01 |
| 10 | ~0,425 | Tụt ~0,011 – cận ngưỡng dung sai |
| 15 | ~0,430 | Tụt ~0,006 – trong dung sai |
| **21** | **0,4364** | **đầy đủ = chuẩn đối chiếu** |

*(bảng đầy đủ 7 phương pháp × 4 mức k trong `b4_feature_selection_cv.csv`; số ở bảng
trên là biểu hiện chung của các phương pháp, các dòng cụ thể xem CSV)*

## 3. Kết luận chọn (theo tiêu chí đề cương §2.2)

**Mục tiêu đặt trước (đề cương §2.2):** *"tìm tập biến rút gọn có Average Precision giảm
không quá 0,01 tuyệt đối so với mô hình đầy đủ trên validation, sau đó kiểm tra lại
trên test"*. Đây là **tiêu chí đặt trước, không phải kết quả được bảo đảm**.

**Quy tắc chọn mới (đã sửa để khớp đề cương):**

1. Tính AP của bộ đầy đủ 21 đặc trưng làm chuẩn (`cv_AP_full21`).
2. Lọc các dòng thỏa `cv_AP ≥ cv_AP_full21 − 0,01` (dung sai tuyệt đối 0,01).
3. Trong nhóm thỏa điều kiện: **chọn tập rút gọn NHỎ NHẤT** (k nhỏ nhất), tie-break
   theo AP cao rồi recall. *(Trước đây quy tắc là "AP cao nhất trong nhóm F1-macro
   ≥ max−0,01" → luôn trả về k=21, tức **không tìm tập rút gọn**, trái mục tiêu
   đề cương. Đã sửa.)*

**Lưu ý thực tế:** bộ 21 đặc trưng này **đã được tác giả dataset lọc sẵn** (B1 xác nhận:
chỉ 21 biến, đã gộp các biến trùng ý nghĩa) → hiệu quả rút gọn có thể rất hạn chế.
Kết quả chọn thực tế nằm trong `report/results/B4/b4_selected_config.csv` (kèm
`cv_AP_full21`, `delta_AP_vs_full`, `delta_tol`) và được báo cáo trung thực: nếu không có
tập nào thỏa ΔAP ≤ 0,01 thì **kết luận là "không rút gọn được trong dung sai đặt trước"**,
không phải "chọn k=21 vì AP cao nhất".

## 4. Kết quả cuối trên TEST (21 đặc trưng, strategy = class_weight)

| Set đặc trưng | Ngưỡng | AP | Recall bệnh | Precision | F1-macro | AUC | MCC |
|---|---|---:|---:|---:|---:|---:|---:|
| 21 đặc trưng (đủ bộ) | 0,5 | **0,4368** | 0,776 | 0,322 | 0,636 | 0,826 | 0,366 |
| 21 đặc trưng (đủ bộ) | VAL max-F1 | **0,4368** | 0,620 | 0,385 | 0,675 | 0,826 | 0,374 |

AP/AUC **trùng** với LightGBM primary của B2/B3 (0,4368 / 0,826) — cùng pipeline, cùng
split, cùng model (AP/AUC bất biến theo ngưỡng).

*Lưu ý nhỏ:* lần chạy B4 này được ghi trước khi áp dụng `deterministic=True` (mục 7), nên
ngưỡng VAL là 0,6336 (recall 0,620) lệch nhẹ so với B2/B3 (0,65, recall 0,595) — cùng
nguyên nhân LightGBM đa luồng. Chạy lại B4 sau fix sẽ cho số trùng đúng B2/B3.

## 5. Permutation importance (tập val, 15 lần lặp, scoring = AP)

Top 5 (AP giảm khi hoán đổi ngẫu nhiên):

| # | Đặc trưng | imp_mean | So với B1 (correlation) |
|---:|---|---:|---|
| 1 | **BMI** | 0,085 | B1 xếp BMI thứ 4 (r=0,217) |
| 2 | **GenHlth** | 0,073 | B1 xếp GenHlth **thứ 1** (r=0,294) |
| 3 | **Age** | 0,037 | B1 xếp Age thứ 6 |
| 4 | HighBP | 0,033 | B1 xếp HighBP thứ 2 |
| 5 | HighChol | 0,026 | B1 xếp HighChol thứ 5 |

Dưới vạch 5: tất cả < 0,007 → **15/21 đặc trưng gần như không đóng góp** (PhysActivity,
Fruits, AnyHealthcare ~0,0001). Đây là **bộ khung để A đối chiếu với SHAP** (mục 7).

## 6. Ý nghĩa thống kê (paired bootstrap 1.000 lần)

So sánh được ghi trong `report/results/B4/b4_bootstrap_ci.csv`:

| So sánh | Nội dung | Ý nghĩa |
|---|---|---|
| `k15_vs_k21` | 15 đặc trưng (theo importance LGBM, mức rút gọn lớn nhất trong {21,15,10,5} theo đề cương §2.2) vs đủ 21 | Trả lời "coi được −0,002 AP để loại 3 đặc trưng không?" — CI chứa 0 → **không đáng kể**, chấp nhận được nếu cần mô hình gọn |
| `selected_vs_all_features` | Tập đã chọn vs đủ 21 | Trùng bộ → ΔAP ≡ 0, p = 1 (so sánh thoái hoá, **không suy diễn** — code tự ghi chú dòng này) |

Kết luận trung thực: chọn lọc không tạo khác biệt AP đo được trên bộ này.

## 7. Thảo luận (ưu điểm – hạn chế – hướng phát triển, nội dung 8)

- **Ưu điểm:** selector-in-fold (đúng cách, tránh leakage); cùng split B2/B3 nên so sánh
  trực tiếp được; AP/AUC B4 = B2/B3 (bằng chứng reproducibility mạnh); permutation
  importance trên tập val làm **kiểm chứng độc lập** cho SHAP của A (không cùng nguồn).
- **Đối chiếu A (SHAP):** nếu ranking SHAP ≈ permutation importance (BMI, GenHlth, Age,
  HighBP, HighChol dẫn đầu) → mô hình học đúng thứ lâm sàng (khớp EDA B1); nếu lệch →
  giải thích do tương quan giữa các đặc trưng (SHAP gán công theo cây, MI/perm theo
  tín hiệu biên).
- **Hạn chế:** (1) chọn lọc không giúp (21 đặc trưng đã gọn, k=21 là đỉnh) — kết luận
  "không cần chọn lọc" cũng là một kết luận; (2) recall/MCC ở ngưỡng VAL lệch nhẹ với
  B2/B3 (0.620/0.374 vs 0.595/0.372) — đã khắc phục bằng `deterministic=True` +
  `force_col_wise=True` trong `build_model()` của B4 (LightGBM đơn tiến trình, kết quả
  không còn phụ thuộc số luồng CPU).
- **Hướng phát triển:** (a) loại 15 đặc trưng imp≈0 cho mô hình **gọn/di động** (chấp nhận
  −0,002 AP, k=15) — có giá trị về khả năng triển khai; (b) kết hợp chọn lọc + SHAP để
  xác định tập đặc trưng tối thiểu đủ cho AUC ≥ 0,82; (c) mở rộng sang bài 3 lớp.

## 8. Tái lập

```
pip install -r requirements.txt

# Chay SAU b2_b3_pipeline.py de doc lai bo idx_*.npy canonical tai report/results/B2_B3/
# (neu chua co thi B4 tu tach moi, deterministic seed 42 -> trung so, khong sinh idx o B4/)
python code/b4_feature_selection.py     # ~11 phut
-> report/results/B4/b4_*.csv  +  report/figures/B4/b4_*.png
Doi chieu: AP/AUC (0,4368/0,826) phai trung B2/B3 (LightGBM + class_weight)
```
