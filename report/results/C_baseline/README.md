Baseline B0–B2 (thành viên C)

Mã: code/c_baseline.py (v3). Dùng pipeline của B (b2_b3_pipeline.py) và idx_*.npy của B, nên dữ liệu và cách chia tập giống hệt B.

Trạng thái: chỉ có kết quả validation. Chưa chạy tập kiểm tra. Số liệu chưa được B hoặc D đối chiếu chéo.

Protocol (theo quyết định của nhóm; chỗ khác đề cương được ghi rõ)
Dữ liệu: toàn bộ 253.680 dòng (không loại dòng trùng, không sửa nhãn). Khác các bản trước (227.908 dòng).
Chia tập: phân tầng 70/15/15, seed 42 → 177.576 / 38.052 / 38.052 dòng (đọc từ idx_*.npy của B; script dừng nếu số dòng không khớp).
Baseline không dùng trọng số lớp (strategy = none): B0 Dummy, B1 Logistic (có chuẩn hóa), B2 Random Forest.
Tìm siêu tham số (đề cương §4.3): CV 3 fold phân tầng trên train, seed 42, tiêu chí AP, 10 cấu hình mỗi họ; tham số khóa trước khi chọn ngưỡng. Chỉ dùng train.
Ngưỡng: chọn trên validation, lưới 0,01–0,99 (199 mức), hòa lấy ngưỡng nhỏ nhất. Mặc định tối ưu F2 (--beta 2). Khác đề cương §4.5 (F1): điều chỉnh có chủ đích, ưu tiên recall. --beta 1 chạy F1 để đối chiếu độ nhạy.
Phụ lục (ngoài đề cương): Decision Tree với --appendix-dt, lưu file riêng.
Cách chạy (Kaggle)
python c_baseline.py --b-code <thư mục có b2_b3_pipeline.py> --idx-dir <thư mục có idx_*.npy> --appendix-dt --out-dir out_F2
python c_baseline.py --b-code ... --idx-dir ... --beta 1 --out-dir out_F1        # đối chiếu F1
python c_baseline.py --b-code ... --idx-dir ... --final                          # thêm TEST: CHỈ chạy MỘT lần, sau khi nhóm khóa
python c_baseline.py --b-code ... --synthetic --quick                            # kiểm thử, không dùng để nộp
File trong thư mục
File	Nội dung
results_C_baseline_none.csv	B0, B1, B2 trên validation, ngưỡng F2
results_C_appendix_decision_tree.csv	Decision Tree (phụ lục)
tune_C_logistic.csv, tune_C_random_forest.csv, tune_C_decision_tree.csv	các cấu hình đã thử và CV-AP
sensitivity_F1/	cùng bảng kết quả khi chọn ngưỡng theo F1 (đối chiếu độ nhạy)
logs/	nhật ký các lần chạy (nếu có)

Đối chứng theo Xie et al. (2019) (X1–X5) nằm ở report/results/C_xie/.

Kết quả validation (ngưỡng F2)
Mô hình	Tham số khóa	CV-AP	AP	ROC-AUC	Recall	Precision	Specificity	F1	F2	Ngưỡng
B0 Dummy	prior	—	0,1393	0,5000	1,000	0,1393	0,000	0,2446	0,4474	0,01
B1 Logistic	C = 0,001	0,4071	0,4042	0,8207	0,841	0,2779	0,646	0,4177	0,5983	0,109
B2 Random Forest	300 cây, độ sâu 12	0,4304	0,4231	0,8238	0,805	0,2946	0,688	0,4314	0,5978	0,139
Phụ lục: Decision Tree	độ sâu 8, lá tối thiểu 50	0,4058	0,3956	0,8135	0,870	0,2551	0,589	0,3945	0,5869	0,094
Đối chiếu độ nhạy: ngưỡng F1 (validation)
Mô hình	Quy tắc	Ngưỡng	Recall	Precision	Specificity	F1	F2	TP	FP
B1 Logistic	F1	0,203	0,628	0,364	0,822	0,4608	0,5482	3.328	5.816
B1 Logistic	F2	0,109	0,841	0,278	0,646	0,4177	0,5983	4.457	11.583
B2 Random Forest	F1	0,233	0,611	0,379	0,838	0,4680	0,5442	3.237	5.295
B2 Random Forest	F2	0,139	0,805	0,295	0,688	0,4314	0,5978	4.268	10.219

AP và ROC-AUC không đổi khi đổi quy tắc ngưỡng. Mốc tham chiếu: Dummy có F2 = 0,4474 (đoán tất cả dương).

Lưu ý
Chênh lệch AP giữa các mô hình trên validation (khoảng 0,003–0,02) cần kiểm định bằng bootstrap ghép cặp trên tập kiểm tra trước khi kết luận.
AP gần như không nhạy với C của Logistic (CV-AP 0,4065–0,4071).
Còn chờ: B3 (XGBoost/LightGBM), lần chạy test chung của nhóm, đối chiếu chéo với B và D.
