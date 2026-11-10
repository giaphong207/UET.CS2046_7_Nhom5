# SP1 – Khảo sát phương pháp & lựa chọn mô hình (Survey & Method Selection)

**Vai trò trong SP:** SP1 cung cấp **cơ sở học thuật** cho mọi lựa chọn trong SP2
(B2/B3/B4): tại sao chọn AP làm chỉ số chính, tại sao so đúng 5 chiến lược mất cân bằng,
tại sao selector phải fit trong fold, và tại sao majority-vote cho label noise.
Mỗi quyết định thực nghiệm trong SP2 đều bám theo một mục của SP1 (bảng mapping §8).

> **Cách dùng:** phần so sánh §3–§7 là "gốc lý thuyết" cho các bảng kết quả trong
> `B2_B3_experiment_report.md` và `B4_feature_selection_report.md`; mục §8 là
> "bản ghi quyết định" (decision log) để trình bày trước giảng viên.

---

## 1. Phạm vi bài toán (tóm tắt từ B1)

Bộ dữ liệu Diabetes Health Indicators (BRFSS 2015, CDC), 253.680 dòng × 21 đặc trưng,
dựng bởi Teboul [14] làm dạng bảng tính sẵn (đã gộp sentinel, không còn missing —
xác nhận trong B1). Ba tệp: 2 lớp không cân bằng (86,1/13,9 — bài chính), 2 lớp 50/50
(sanity-check), 3 lớp `Diabetes_012` (1,8% lớp giữa — nghiệm phụ).

**Ba vấn đề quyết định thiết kế:** (i) mất cân bằng 6,2× (bài chính) / 46× (bài phụ);
(ii) ~9,5% dòng trùng + ~2% label noise; (iii) 21 đặc trưng đã được tác giả lọc sẵn —
liệu có cần chọn lọc thêm (B4).

## 2. Khảo sát văn học (nội dung 3: tìm – đọc – so sánh)

### 2.1 Học trên dữ liệu mất cân bằng

Khảo sát tổng quan chuẩn là [4] (He & Garcia, TKDE 2009): phân loại giải pháp thành
**data-level** (resample) và **algorithm-level** (chi phí/lớp). Bổ sung: [3] (Elkan,
IJCAI 2001) cho cơ sở lý luận của class-weight — học có trọng số lớp là vấn đề
nhiều năm nay, "balanced weight" (n/(c·n_c)) là chuẩn công nghiệp.

Hai hướng resample có bằng chứng học thuật đầy đủ nhất:

- **Oversampling tổng hợp — SMOTE** [1] (JAIR 2002): sinh mẫu bằng nội suy đoạn thẳng
  giữa mẫu thiểu số và k láng giềng; cải thiện recall lớp thiểu số nhưng **sinh vùng
  biên (out-of-distribution)** ở các vùng thưa → có thể tạo điểm giả làm lệch ranh giới
  quyết định (hạn chế được nêu ngay trong [1] và lặp lại trong [4]).
- **ADASYN** [2] (Seo et al., 2011): thay sửa mật độ theo khoảng cách k-NN của từng
  mẫu thiểu số (mẫu "khó", gần biên → sinh nhiều hơn). Imbalanced-learn [8] cài đặt
  cả hai; ADASYN dùng cho các bài mất cân bằng cục bộ (không đều trên toàn không gian).
- **Undersampling lớp đa số**: rẻ, không sinh mẫu rác; nhưng **mất thông tin** — với
  bài có 86% mẫu lớp 0, undersample đến 50/50 chỉ giữ lại ~30k mẫu trong 253k →
  suy giảm đáng kể (được xác nhận thực nghiệm ở B3: AP tụt, recall tăng).

Kết luận học thuật: **không chiến lược nào thắng tuyệt đối** — phải so trực tiếp trên
bài cụ thể bằng cùng một protocol (đúng cách làm ở B3, §3.3).

### 2.2 Chỉ số đánh giá: tại sao phải là AP

Davis & Goadrich [5] (2006) chỉ ra rằng **AP (diện tích dưới đường precision–recall)**
là chỉ số đúng cho bài mất cân bằng vì (i) không phụ thuộc ngưỡng, (ii) không bị
phân bố mẫu số chi phối như F1, (iii) đo đúng câu hỏi "top-k kết quả trả về có bao
nhiêu điểm dương". Trong bài 86/14, accuracy 86,1% được đạt bằng cách đoán "lớp 0"
cho mọi dòng — chỉ số AP chỉ ra được điều này trong khi AUC còn tốt; nhóm dùng
AP làm chỉ số chính, AUC chỉ là tham chiếu, accuracy chỉ báo cáo.

### 2.3 Ngưỡng quyết định & ý nghĩa thống kê

- **Chọn ngưỡng trên validation, không phải test** — nguyên tắc "đánh giá test 1 lần":
  ngưỡng là một tham số, chọn trên test thì test không còn "test" nữa. Puckette &
  Efron [6] (JASA 2012) cung cấp nền tảng cho vấn đề "hiệu năng mô hình với ngưỡng
  tối ưu" (trade-off khi đổi ngưỡng).
- **Paired bootstrap cho so sánh 2 mô hình trên cùng tập test** [7] (Efron &
  Tibshirani, 1993): resample theo **cặp** (cùng chỉ số dòng cho cả 2 mô hình) để
  giữ tương quan giữa các độ đo → CI cho ΔAP chặt hơn so sánh 2 CI độc lập;
  "đáng kể" khi CI 95% không chứa 0. Nhóm áp dụng với 1000 lần lặp, seed cố định.

### 2.4 Mô hình: tại sao LightGBM làm chính

- LightGBM [9] (Ke et al., NeurIPS 2017) — gradient boosting tốc độ cao,
  thường bằng hoặc trội hơn RF/Logistic trên bảng tính ~250k dòng, hỗ trợ
  `class_weight` native. Chọn làm **mô hình chính** của nhóm (A điều khiển phần
  SHAP [11]; SP1 chỉ lập luận lựa chọn, không thay đổi code của A/C).
- Stacking [10] (Wolpert, 1992): kiểm tra như một **nghiệm phụ ensemble** — nếu
  stacking ≈ LightGBM đơn thì không cần giá trị phức tạp (kết luận thực
  B2/B3: ΔAP ≈ 0, p = 0,948 → chọn LightGBM đơn giản).
- LogReg/RF/DecisionTree: giữ làm **baseline phương minh (A/C)** theo protocol —
  không phải để thắng, mà để LightGBM có "đích so" (bootstrap trong B2/B3).

### 2.5 Label noise & duplicate

- Duplicated rows là **leakage nguồn** rõ ràng: cùng một bản ghi nằm trong cả
  train lẫn test → đánh giá bị phồng. Chuẩn công nghiệp: dedup **trước** mọi
  split (B2: split chỉ số được "khóa" bởi seed, dedup nằm phía trước).
- **Majority-vote theo nhóm đặc trưng trùng** là cách sửa label noise phổ biến,
  hợp lý khi label "đúng" có xác suất cao là lớp đa số của các mẫu giống hệt nhau
  (dù [12] (Natarajan, NeurIPS 2013) chứng minh majority-vote chỉ là ước lượng
  xấp xỉ dưới giả định noise không phụ thuộc; với BRFSS, các mẫu cùng người/cùng
  câu trả lời nhiều lần, majority-vote là giả định đơn giản hợp lý nhất).
  B3 ablation 2 phương án (giữ nguyên vs majority-vote) — kết luận: giữ majority-vote,
  vì ΔAP dương +0,0038.

### 2.6 Chọn lọc đặc trưng (B4)

- **Filter** (variance, corr-dedup, F-test [9], MI [13]): nhanh, model-agnostic.
- **Embedded**: importance của LightGBM (theo gain) — chọn theo tín hiệu mà model
  đã dùng; hợp lý vì model chính là LightGBM.
- **Wrapper (RFE)**: loại feature lặp, mỗi lần refit model → chi phí O(k·n_models),
  thường chỉ dùng cho bộ đặc trưng đã nhỏ.
- **Permutation importance** (Ragwitz & Rish [13]): "làm lộn xộn cột X làm mô hình
  tệ đi bao nhiêu" → là phép kiểm chứng **độc lập** về độ quan trọng (không cùng
  nguồn với gain/importance nội tại của cây), dùng làm bộ khung so với SHAP [11].
- **Nguyên tắc chống leakage**: selector phải fit **bên trong fold CV / trên train**
  rồi mới áp dụng vào fold test — nếu fit trên toàn bộ dữ liệu, đặc trưng "rò"
  từ test vào việc chọn (đúng thiết kế của B4: "selector-in-fold").

## 3. So sánh phương án & quyết định (nội dung 4: so sánh – chọn)

Bảng decision log (mỗi dòng = một quyết định được "trình bày trước giảng viên"):

| # | Vấn đề | Phương án cân nhắc | Chọn | Cơ sở học thuật | Bằng chứng thực nghiệm |
|---|---|---|---|---|---|
| D1 | Chỉ số chính | AUC / F1-macro / **AP** | **AP** | [5] – AP đúng cho mất cân bằng, bất biến ngưỡng | B3: 5 strategy × 5 model, AP ổn định theo ngưỡng trong khi F1 thay đổi |
| D2 | Ngưỡng | 0,5 mặc định / tối ưu trên VAL / tối ưu trên TEST | **Tối ưu VAL rồi khóa** | [6] – test dùng 1 lần; [7] | B3: AP của cùng mô hình không đổi giữa 2 ngưỡng (tính chất của AP) |
| D3 | Mất cân bằng | class-weight / SMOTE / ADASYN / undersample / none | **class_weight** | [1][2][3][4] – không chiến lược nào trội a priori | B3: class_weight AP 0,4364 (recall 0,78) ≈ SMOTE/ADASYN (0,4346/0,4357) nhưng **nhanh 3–6× và không sinh mẫu giả** |
| D4 | Mô hình chính | LogReg / DT / RF / **LightGBM** / Stacking | **LightGBM** | [9] – GBDT hiệu năng cao; [10] – ensemble kiểm tra nhưng chỉ đáng kể khi có tín hiệu thực | B2: LightGBM − RF: ΔAP +0,0436, p < 0,001; LightGBM ≈ Stacking (p = 0,948) → chọn đơn giản |
| D5 | Dedup | Trước / Sau split | **Trước** | nguyên tắc chống leakage (mục 2.5) | B3 ablation: +0,0026 AP, n giảm 24.206 dòng |
| D6 | Label noise | Giữ nguyên / Majority-vote / Bỏ nhom | **Majority-vote** | [12] – majority-vote là ước lượng hợp lý khi noise không phụ thuộc | B3 ablation: +0,0038 AP |
| D7 | Chọn lọc đặc trưng | filter / embedded / wrapper, k tối ưu | **Giữ đủ 21 (k=21)** | §2.6; [9] | B4: AP cực đại đúng k=21 = baseline; k=18 chỉ −0,002 → không đáng để chọn lọc trên bộ này |
| D8 | So sánh thống kê | So CI độc lập / **paired bootstrap** | **Paired bootstrap 1000 lần** | [7] | B2: 2/3 cặp so sánh có p < 0,001, CI chặt hơn CI độc lập (bị ràng buộc bởi cùng test) |
| D9 | Nghiệm phụ 3 lớp | class_weight / SMOTE / ADASYN | **class_weight (không resample)** | [4] – oversample 1,8% lớp tạo ~4.600 mẫu giả, làm loãng ranh giới lớp 0–1 | B3 §6: AP lớp 2 của class_weight tụt (0,4119 vs 0,4389) nhưng F1-macro tăng → ghi rõ đánh đổi |

## 4. Ghi chú về hạn chế của SP1

- Các kết luận trong §3 được **kiểm chứng thực nghiệm** (SP2/B3/B4), không chỉ
  trích dẫn: đây là điểm khác biệt với bài báo "survey" thông thường (SP1 không
  dừng ở "giáo trình nói X tốt nên dùng X" mà phải "X nói tốt nhưng tôi đã kiểm
  tra trên đúng bài này, số liệu ở đây").
- **Phạm vi so sánh chưa đầy đủ**: chưa so AdaBoost, XGBoost, và **cá biệt chưa
  so model deep (MLP)** — ghi rõ là "ngoài phạm vi" chứ không giấu (nội dung 8:
  hạn chế).
- [14] (BRFSS/CDC) là nguồn số liệu gốc; [15] (Teboul, Kaggle) là người dựng dạng
  bảng tính — hai nguồn **không cùng cấp bằng chứng**: số liệu y khoa trong [14]
  nhưng pipeline chạy trên bản gộp sẵn của [15] (B1 xác nhận: không missing,
  không sentinel, đã "làm sạch sẵn").

## 5. Mapping task B → quyết định trong SP1

| Task B | Quyết định | Mục trong SP1 |
|---|---|---|
| B2 | AP là chỉ số chính; paired bootstrap 1000 lần | D1, D8 |
| B3 | 5 chiến lược; class_weight thắng; dedup + majority-vote; selector-in-fold | D3, D5, D6, §2.6 |
| B4 | Không chọn lọc (k=21); permutation importance cho A so với SHAP [11] | D7, §2.6 |


