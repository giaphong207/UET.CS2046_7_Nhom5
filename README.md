# Phân loại Nguy cơ Đái tháo đường trên Dữ liệu Khảo sát Sức khỏe BRFSS 2015

**Học phần:** Trí tuệ nhân tạo (`UET.CS2046_7`) – **Nhóm 5** (Research Track)

> **[Tuyên bố Giới hạn Diễn giải]**  
> Nghiên cứu này chỉ thực hiện phân loại trạng thái khảo sát sức khỏe tại thời điểm thu thập dữ liệu (gộp nhãn tiền đái tháo đường và đái tháo đường). Mô hình **tuyệt đối không** dự báo thời điểm mắc bệnh trong tương lai, **không** suy luận quan hệ nhân quả y học và **không** thay thế chẩn đoán lâm sàng.

---

## 1. Cấu trúc Thư mục Dự án

```text
UET.CS2046_7_Nhom5/
├── data/               # Chứa dữ liệu gốc & file lưu chỉ số chia tập (bị chặn bởi .gitignore)
├── notebooks/          # Chứa notebook khám phá dữ liệu (EDA) và kịch bản kiểm thử (QA)
├── src/                # Chứa mã nguồn pipeline chính (chia tập, baseline, mô hình cải tiến, đánh giá)
├── docs/               # Chứa tài liệu khảo sát, bài báo nền và báo cáo tổng hợp (SP1, SP2)
├── .gitignore          # Quy tắc chặn đẩy dữ liệu nặng (*.csv) lên GitHub
├── requirements.txt    # Danh sách thư viện và phiên bản cố định
└── README.md           # Tài liệu hướng dẫn tái lập kết quả
```

---

## 2. Dữ liệu & Kiểm chứng

1. **Tải dữ liệu:** Tải bộ dữ liệu *Diabetes Health Indicators Dataset (BRFSS 2015)* trên Kaggle, sử dụng file `diabetes_binary_health_indicators_BRFSS2015.csv` (253.680 bản ghi, 21 đặc trưng, 1 biến mục tiêu `Diabetes_binary`).
2. **Vị trí lưu:** Đặt file CSV vào thư mục `data/` ở thư mục gốc dự án.
3. **Mã kiểm chứng (Checksum):**
   * **Tên file:** `diabetes_binary_health_indicators_BRFSS2015.csv`
   * **SHA-256:** `[Điền mã SHA-256 sau khi chạy kiểm tra]`

---

## 3. Cài đặt Môi trường (Python 3.10+)

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## 4. Quy tắc Thực nghiệm & Chống Rò rỉ Dữ liệu

* **Tính tái lập:** Cố định `random_state = 42` cho toàn bộ quá trình chia tập, huấn luyện, tối ưu siêu tham số và kiểm định Bootstrap.
* **Toàn vẹn dữ liệu:** Giữ nguyên toàn bộ **253.680 bản ghi**; **tuyệt đối không xóa** 24.206 mẫu trùng lặp bằng `drop_duplicates()`.
* **Chia tập:** Stratified Split theo tỷ lệ **70% Train – 15% Validation – 15% Test** (3-fold Stratified CV trên Train) và lưu file chỉ số chia tập dùng chung.
* **Chống rò rỉ (Data Leakage):** Mọi bước tiền xử lý, chọn lọc đặc trưng hoặc lấy mẫu lại chỉ thực hiện bên trong từng fold của Train; giữ nguyên phân bố gốc của Validation và Test.
* **Tiêu chí đánh giá:**
  * Dùng **Average Precision (AP)** để chọn mô hình tốt nhất trên Validation.
  * Tối ưu ngưỡng quyết định theo **F2-score** trên Validation (đối chiếu với ngưỡng `0.5`, báo cáo kèm Precision, Recall, Specificity, ROC-AUC).
  * Chỉ đánh giá trên tập **Test** một lần duy nhất sau khi đã khóa cấu hình.

---

## 5. Hướng dẫn Tái lập Kết quả (How to Reproduce)

```bash
# Bước 1: Kiểm tra tính toàn vẹn dữ liệu & xuất chỉ số chia tập 70-15-15
python src/split_data.py

# Bước 2: Huấn luyện các mô hình cơ sở (Baseline: E0 – E3)
python src/train_baselines.py

# Bước 3: Huấn luyện mô hình cải tiến & thực nghiệm Ablation (E4 – E10)
python src/train_models.py

# Bước 4: Đánh giá Test, kiểm định Paired Bootstrap (1.000 lần) & giải thích SHAP
python src/evaluate_test.py
```

---

## 6. Phân công Nhóm Thực hiện

| Thành viên | Vai trò | Nhiệm vụ chính |
| :--- | :--- | :--- |
| **Bùi Ngọc Phú Hưng (A)** | Lead Modeling | Cơ sở lý thuyết, huấn luyện mô hình cải tiến (`E4–E10`), SHAP |
| **Trần Tuấn Thành (B)** | Data & Pipeline | Khám phá dữ liệu (EDA), tiền xử lý, chia tập phân tầng `70-15-15` |
| **Đinh Gia Phong (C)** | Baseline & Report | Huấn luyện mô hình cơ sở (`E0–E3`), tổng hợp bảng kết quả, viết báo cáo |
| **Ngô Khánh Ngân (D)** | QA-Admin & AI Log | Quản trị Git, kiểm soát chống rò rỉ dữ liệu, smoke test tái lập, AI Usage Log (`SP3`) |
