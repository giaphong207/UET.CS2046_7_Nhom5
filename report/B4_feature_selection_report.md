# B4 – Chọn lọc đặc trưng (Feature Selection): Diabetes BRFSS 2015
**Môi trường:** Kaggle Notebook, seed = 42, LightGBM + imbalanced-learn, thời gian ~11 phút (667s)
**Mã nguồn:** `Src/b4_feature_selection.py` · Số liệu: `report/results/b4_*.csv` · Hình: `report/figures/b4_*.png`

## 1. Thiết kế (đúng protocol B2/B3)

- **Cùng split với B2/B3:** đọc `report/results/idx_*.npy` nếu có; nếu chưa có, tự tách
  stratified 70/15/15 seed 42 → **trùng chính xác** 159.535 / 34.186 / 34.187 (deterministic).
- **Selector fit BÊN TRONG từng fold CV** (không dùng toàn bộ train) → không leakage.
- **Mỗi dòng chỉ đổi 1 yếu tố:** số đặc trưng (k) và phương pháp.
- **7 phương pháp × 5 mức k ∈ {6, 10, 14, 18, 21}:**

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
| 6 | ~0,3907 | Tụt mạnh (−0,046) – thiếu đặc trưng |
| 10 | ~0,425 | |
| 14 | ~0,428 | |
| 18 | ~0,4345 | chỉ −0,002 so với đủ bộ |
| **21** | **0,4364** | **tối ưu, = baseline** |

*(bảng đầy đủ 7×5 trong `b4_feature_selection_cv.csv`)*

## 3. Kết luận chọn

**Quy tắc (giống B2/B3):** AP cao nhất trong nhóm F1-macro ≥ max−0,01, tie-break recall →
**giữ đủ 21 đặc trưng (k=21, "baseline")**. Số liệu cho thấy **chọn lọc đặc trưng KHÔNG
giúp cải thiện AP** trên bộ 21 đặc trưng đã tinh tuyển này; các đặc trưng yếu
(permutation importance ≈ 0, mục 5) có thể cân nhắc loại cho mô hình gọn nhưng chênh AP
chỉ ~0,002 (k=18). → **mô hình cuối dùng đủ 21 đặc trưng**, giữ nguyên cấu hình B2/B3.

## 4. Kết quả cuối trên TEST (21 đặc trưng, strategy = class_weight)

| Set đặc trưng | Ngưỡng | AP | Recall bệnh | Precision | F1-macro | AUC | MCC |
|---|---|---:|---:|---:|---:|---:|---:|
| 21 đặc trưng (đủ bộ) | 0,5 | **0,4368** | 0,776 | 0,322 | 0,636 | 0,826 | 0,366 |
| 21 đặc trưng (đủ bộ) | VAL max-F1 | **0,4368** | 0,620 | 0,385 | 0,675 | 0,826 | 0,374 |

AP/AUC **trùng tuyệt đối** với LightGBM primary của B2/B3 (0,4368 / 0,826) — chứng minh
cùng pipeline, cùng split, cùng model (AP/AUC bất biến theo ngưỡng).

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

`selected vs all_features`: ΔAP = 0,0000, 95% CI [0,0000; 0,0000], p = 1,000 → **không đáng
kể** (hệ quả trực tiếp: "selected" trùng "full"). Không suy diễn — kết luận trung thực là
chọn lọc không tạo khác biệt đo được trên bộ này.


