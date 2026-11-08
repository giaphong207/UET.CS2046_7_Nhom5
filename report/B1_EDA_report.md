# B1 – Phân tích bộ dữ liệu (EDA): Diabetes Health Indicators (BRFSS 2015)

## 1. Mô tả bộ dữ liệu

Bộ dữ liệu được Alex Teboul dựng từ khảo sát **BRFSS 2015** (CDC — Behavioral Risk Factor
Surveillance System), gồm **21 đặc trưng** về chỉ số sức khỏe và lối sống, phát hiện bởi
21 câu hỏi tự báo cáo. Ba tệp trong cùng một bộ dữ liệu:

| Tệp CSV | Dòng | Cột target | Số lớp | Phân bố lớp |
|---|---:|---|---:|---|
| `diabetes_012_health_indicators_BRFSS2015.csv` | 253.680 | `Diabetes_012` | 3 | 84,2% / **1,8%** / 13,9% |
| `diabetes_binary_health_indicators_BRFSS2015.csv` | 253.680 | `Diabetes_binary` | 2 | 86,1% / 13,9% |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70.692 | `Diabetes_binary` | 2 | 50,0% / 50,0% |

**Quy ước nhãn 3 lớp (`Diabetes_012`):** `0` = không tiểu đường, `1` = tiền tiểu đường,
`2` = tiểu đường. Lớp dương (2) trong tệp 3 lớp có đúng 35.346 ca — trùng khớp với số
ca dương của tệp 2 lớp, và bằng một nửa tệp 50/50 → **ba tệp nhất quán với nhau**, có thể
dùng đồng thời (tệp 50/50 dùng để sanity-check pipeline, tệp 2 lớp làm bài toán chính,
tệp 3 lớp dùng làm nghiệm mở rộng).

**21 đặc trưng** gồm: biến nhị phân (`HighBP`, `HighChol`, `Smoker`, `Stroke`,
`HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`,
`AnyHealthcare`, `NoDocbcCost`, `DiffWalk`), biến thứ tự (`GenHlth`, `MentHlth`,
`PhysHlth`, `Age`, `Education`, `Income`), biến liên tục (`BMI`) và `CholCheck`, `Sex`.

## 2. Chất lượng dữ liệu

| Kiểm tra | Kết quả (tệp 3 lớp / tệp 2 lớp) | Hành động |
|---|---|---|
| Missing value | **0 cột** (không có ô trống nào) | Không cần imputation |
| Mã sentinel BRFSS (777/888/999) | **0 cột** | Không cần quy đổi "không biết" |
| Trùng lặp toàn dòng | 23.899 dòng (**9,42%**) / 24.206 (**9,54%**) | `drop_duplicates(keep='first')` **trước khi split** |
| Nhiễu nhãn (label noise) | 6.120 dòng trong 1.834 nhóm (**2,41%**) / 5.218 dòng trong 1.566 nhóm (**2,06%**) | So 2 phương án: giữ nguyên vs majority-vote theo nhóm |

**Nhận xét quan trọng.** Bộ dữ liệu mà Alex Teboul cung cấp **đã được làm sạch sẵn** — khác
với bản BRFSS thô, nó không còn missing value cũng không còn mã mã hoá 777/888/999. Vì
vậy nhóm **loại bỏ được bước imputation** khỏi kế hoạch tiền xử lý, thay vào đó tập trung
vào hai vấn đề thật sự tồn tại:

- **Trùng lặp ~9,5%.** Đặc trưng của BRFSS: cùng một người có thể được khảo sát ở nhiều
  lần/nhóm, tạo ra các bản ghi giống hệt nhau. Nếu không khử trùng lặp trước khi chia tập,
  các bản ghi trùng có thể nằm ở cả train và test → **đo lệch số liệu (leakage)**, và
  hiệu năng mô hình bị thổi phồng. Nhóm khử trùng lặp trước khi split.
- **Nhiễu nhãn ~2,4%.** Có những nhóm bản ghi **đặc trưng giống hệt nhau nhưng nhãn khác
  nhau** (6.120 dòng thuộc 1.834 nhóm). Đây là nhiễu không tránh khỏi của dữ liệu khảo sát
  tự báo cáo. Nhóm thí nghiệm 2 phương án xử lý — *giữ nguyên* và *majority-vote theo
  nhóm đặc trưng trùng* — rồi chọn theo metric và **ghi rõ quyết định cùng lý do vào SP2**.

## 3. Mất cân bằng lớp – điểm quyết định của bài toán

| Tệp | Tỷ lệ lớp thiểu | Tỷ lệ mất cân bằng (max/min) |
|---|---:|---:|
| 3 lớp (`Diabetes_012`) | **1,8%** (tiền tiểu đường, 4.631 dòng) | **46,1×** |
| 2 lớp unbalanced | 13,9% | **6,2×** |
| 2 lớp 50/50 | 50,0% | 1,0× |

Đây là đặc điểm chi phối toàn bộ thiết kế thực nghiệm: **tệp 3 lớp mất cân bằng 46,1×**,
lớp trung gian (tiền tiểu đường) chỉ có 4.631 mẫu. Các kỹ thuật oversampling (SMOTE,
ADASYN) trên một lớp có 4.631 mẫu rời rạc gần như không cải thiện được gì, và mọi chỉ số
tổng thể (F1-macro, accuracy) bị hai lớp còn lại chi phối.

**Kết luận chọn phạm vi bài toán (đề xuất trình bày trước giảng viên):**

- **Bài toán chính = phân loại 2 lớp trên tệp unbalanced (86,1 / 13,9).** Đây là nơi
  kết luận lâm sàng rõ ràng (tiền đường) và nơi các kỹ thuật xử lý mất cân bằng thực sự
  có tác dụng.
- **Tệp 50/50 chỉ dùng làm sanity-check** để xác nhận pipeline không có lỗi (số liệu
  phải "sạch" và cao hơn hẳn).
- **Tệp 3 lớp giữ làm nghiệm mở rộng phụ** (một thí nghiệm duy nhất với LGBM + class-weight),
  đồng thời là **căn cứ bằng chứng** cho việc nhóm chủ động thu hẹp phạm vi — một lý do
  có số liệu, không phải lựa chọn theo thuận tiện.

## 4. Quan hệ đặc trưng ↔ mục tiêu

**4.1. Correlation (Pearson) — 8 đặc trưng mạnh nhất với `Diabetes_binary` (tệp unbalanced):**

| # | Đặc trưng | |r| | Ý nghĩa lâm sàng |
|---:|---|---:|---|
| 1 | `GenHlth` (tự đánh giá sức khỏe) | 0,2936 | yếu tố dự báo mạnh nhất |
| 2 | `HighBP` (huyết áp cao) | 0,2631 | hội chứng chuyển hóa |
| 3 | `DiffWalk` (giải bộ bị hạn chế) | 0,2183 | dấu hiệu suy giảm chức năng |
| 4 | `BMI` | 0,2168 | thừa cân → insulin kháng |
| 5 | `HighChol` | 0,2003 | rối loạn mỡ máu |
| 6 | `Age` | 0,1774 | tuổi là yếu tố nguy cơ |
| 7 | `HeartDiseaseorAttack` | 0,1773 | tiền sử tim mạch |
| 8 | `PhysHlth` | 0,1713 | sức khỏe thể chất |

**4.2. Lift theo mức đặc trưng** — định nghĩa `lift = P(lớp bệnh | mức) / P(lớp bệnh)`;
lift = 1 nghĩa là mức đó không mang thông tin, lift > 1 nghĩa là tăng nguy cơ.

| Mức đặc trưng | n | P(bệnh \| mức) | Lift |
|---|---:|---:|---:|
| `GenHlth = 5` (sức khỏe tự đánh giá rất kém) | 12.081 | 37,9% | **2,72×** |
| `HeartDiseaseorAttack = 1` (tiền sử tim mạch) | 23.893 | 33,0% | 2,37× |
| `Stroke = 1` (đã từng đột quỵ) | 10.292 | 31,8% | 2,28× |
| `GenHlth = 4` (sức khỏe kém) | 31.570 | 31,0% | 2,23× |
| `DiffWalk = 1` (khó đi lại) | 42.675 | 30,8% | 2,21× |
| `Education = 2` (trình độ thấp) | 4.043 | 29,3% | 2,10× |
| `Income = 2` (thu nhập thấp) | 11.783 | 26,2% | 1,88× |
| `HighBP = 1` | 108.829 | 24,5% | 1,75× |

**4.3. Đối chiếu và ý nghĩa.** Các phát hiện **khớp với tri thức y học**: tự đánh giá sức
khỏe kém, tiền sử tim mạch/đột quỵ, hạn chế vận động và thu nhập thấp đều làm tăng nguy cơ
tiểu đường. Tập phát hiện này tạo **giả thuyết kiểm chứng cho phần giải thích mô hình
(SHAP)**: nếu mô hình học được, các đặc trưng này phải nằm ở vị trí đầu trong biểu đồ
SHAP toàn cục — nếu không, mô hình đang học thứ gì đó không có ý nghĩa lâm sàng.

*Lưu ý phương pháp:* correlation Pearson chỉ bắt quan hệ tuyến tính; lift trên là tín hiệu
phi tuyến bổ sung. Nhóm dùng **cả hai** để tránh kết luận thiên lệch.

## 5. Kết quả B1 → đầu vào cho các task tiếp theo

| Đầu ra của B1 | Task tiếp theo | Hành động cụ thể |
|---|---|---|
| 9,42–9,54% trùng lặp | B3 (tiền xử lý) | `drop_duplicates(keep='first')` **trước** split → 253.680 → ~229.474 dòng |
| 2,06–2,41% nhiễu nhãn | B3 | Thí nghiệm 2 phương án (giữ nguyên / majority-vote), chọn theo metric, ghi lý do trong SP2 |
| Không missing, không sentinel | B3 | Không cần imputation → rút ngắn pipeline |
| Mất cân bằng 6,2× (bài chính) | B3 | So 5 chiến lược: none / class-weight / SMOTE / ADASYN / undersample, chọn theo **recall + F1 của lớp bệnh** trên 5-fold stratified CV |
| Mất cân bằng 46,1× (bài phụ) | A (nghiệm mở rộng) | 1 thí nghiệm LGBM + class-weight; nêu rõ hạn chế của SMOTE 3 lớp |
| Bảng lift + correlation | A (SHAP) | Đặt giả thuyết: SHAP toàn cục phải đứng đầu ở `GenHlth`, `HighBP`, `BMI`, `DiffWalk` |
| Cùng split/seed cho mọi mô hình | C (baseline) | Chạy LogReg / RF / DT trên **cùng pipeline này** để so sánh công bằng |


