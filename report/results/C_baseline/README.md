# Baseline B0-B2 (thành viên C)

Mã: code/c_baseline.py. Dùng pipeline của B (b2_b3_pipeline.py) và idx_*.npy của B.
Chiến lược "none" (không trọng số lớp). Tune Logistic và Random Forest bằng CV 3 fold, chọn theo AP.

Lệnh chạy (cần code của B và dữ liệu Kaggle):
python c_baseline.py --b-code <thư mục code của B> --idx-dir <report/results/B2_B3> --out-dir <thư mục ra>

File kết quả: kết quả trên tập validation (chưa chạy test).
- results_C_baseline_none.csv
- tune_C_logistic.csv
- tune_C_random_forest.csv
