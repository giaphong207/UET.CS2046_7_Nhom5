# B1 – Phân tích bộ dữ liệu (EDA): Diabetes Health Indicators, BRFSS 2015

> Phần EDA của SP1. Số liệu lấy trực tiếp từ output script EDA (chạy trên Kaggle,
> dataset `alexteboul/diabetes-health-indicators-dataset`, 3 tệp CSV, seed=42).
> Phụ lục: thư mục `b1_eda_out/` trong notebook (9 CSV + 30 PNG).

## 1. Mô tả bộ dữ liệu

Bộ dữ liệu xây từ khảo sát BRFSS 2015 (CDC), 21 đặc trưng sức khỏe–lối sống, 3 tệp:

| Tệp | Dòng | Lớp target | Phân bố |
|---|---:|---|---|
| `diabetes_012_…` | 253.680 | `Diabetes_012` (3 lớp: 0=không, 1=tiền tiểu đường, 2=tiểu đường) | 84,2% / 1,8% / 13,9% |
| `diabetes_binary_5050split_…` | 70.692 | `Diabetes_binary` (2 lớp) | 50/50 |
| `diabetes_binary_…` | 253.680 | `Diabetes_binary` (2 lớp) | 86,1% / 13,9% |

Lớp 2 (tiểu đường) trong tệp 012 (35.346 ca) trùng khớp đúng tệp unbalanced, và tệp
5050 là bản stratify bằng 1/2 của unbalanced → 3 tệp nhất quán, có thể dùng đồng thời.

## 2. Chất lượng dữ liệu (tìm thấy trong EDA)

- **Không có missing value, không có mã sentinel BRFSS (777/888/999)** → không cần imputation (tiết kiệm một bước của B3).
- **Duplicate ~9,4–9,5%** (23.899/24.206 dòng) trong hai tệp 253.680 dòng: đặc trưng
  BRFSS lặp lại giữa các mẫu được khảo sát hai lần. Xử lý: `drop_duplicates(keep='first')` **trước khi split**.
- **Label noise ~2,1–2,4%** (nhóm feature giống hệt nhau nhưng target khác nhau:
  1.834 nhóm / 6.120 dòng ở tệp 012; 1.566 / 5.218 ở unbalanced). Xử lý: so 2 phương án
  *giữ nguyên* vs *majority-vote theo nhóm feature* — quyết định và lý do ghi vào SP2.

## 3. Mất cân bằng lớp – điểm then chốt của bài toán

- Tệp 2 lớp (chính): **86,1/13,9 → ratio 6,2×**
- Tệp 3 lớp: **46,1×** (lớp tiền tiểu đường chỉ 4.631 dòng, 1,8%) → SMOTE/ADASYN
  3 lớp gần như vô dụng cho lớp trung gian.

**Kết luận chọn bài toán (mang về họp W1):** bài toán chính = phân loại 2 lớp
trên tệp unbalanced (86/14), tiêu chí chọn phương pháp xử lý mất cân bằng =
**recall + F1 của lớp "tiểu đường" trên 5-fold stratified CV** (không nhìn accuracy).
Tệp 5050 chỉ dùng sanity-check pipeline; bài 3 lớp là nghiệm mở rộng phụ.

## 4. Quan hệ đặc trưng ↔ mục tiêu

Correlation cao nhất với `Diabetes_binary` (tệp unbalanced, Pearson):
GenHlth 0,294 > HighBP 0,263 > DiffWalk 0,218 > BMI 0,217 > HighChol 0,200.

Lift cao nhất cho lớp tiểu đường (P(lớp 2|mức) / P(lớp 2)):

| Mức đặc trưng | n | P(lớp 2) | Lift |
|---|---:|---:|---:|
| GenHlth = 5 (sức khỏe tự đánh giá "rất kém") | 12.081 | 37,9% | **2,72×** |
| HeartDiseaseorAttack = 1 | 23.893 | 33,0% | 2,37× |
| Stroke = 1 | 10.292 | 31,8% | 2,28× |
| GenHlth = 4 | 31.570 | 31,0% | 2,23× |
| DiffWalk = 1 (giải bộ bị hạn chế) | 42.675 | 30,8% | 2,21× |
| Education = 2 (thấp) | 4.043 | 29,3% | 2,10× |

Kết quả **khớp tri thức y học** (sức khỏe tự đánh giá thấp, tiền sử tim mạch,
thiếu vận động → nguy cơ cao) → là cơ sở để phần SHAP (XAI) sau này kiểm tra
"mô hình có nói đúng điều bác sĩ biết không". Lưu ý: correlation Pearson chỉ bắt
quan hệ tuyến tính; lift trên là tín hiệu phi tuyến bổ sung.

## 5. Hành động tiếp theo (đầu ra của B1 → đầu vào B2/B3)

1. B3: pipeline chuẩn = dedup → xử lý noise (2 phương án) → split stratified →
   so 5 strategy mất cân bằng (none / class-weight / SMOTE / ADASYN / undersample)
   trên 5-fold stratified CV, chọn theo recall+F1 lớp thiểu.
2. A: dùng bảng lift + correlation trên để đặt giả thuyết cho phần SHAP.
3. C: chạy baseline (LogReg/RF/DT) trên cùng pipeline này để so sánh công bằng.
