# UET CS2046 – Nhóm 5: Dự đoán tiểu đường (BRFSS 2015)

Pipeline + xử lý mất cân bằng lớp cho bài toán phân loại 2 lớp
(`Diabetes_binary`, 86,1 / 13,9) trên bộ dữ liệu Diabetes Health Indicators
(BRFSS 2015, 253.680 dòng), kèm nghiệm phụ 3 lớp.

## Cấu trúc thư mục

| Thư mục / file | Nội dung |
|---|---|
| `Src/b1_eda.py` | **B1** – EDA: duplicate, label noise, correlation, lift, insight cho B3 |
| `Src/b2_b3_pipeline.py` | **B2 + B3** – pipeline reproducible: tiền xử lý (dedup + majority-vote), split 70/15/15 stratified seed 42, ma trận 5 model × 5 resample (CV trong fold), ngưỡng khoá trên VAL, paired bootstrap, nghiệm phụ 3 lớp |
| `Src/make_report_figures.py` | Vẽ biểu đồ B2/B3 từ CSV trong `report/results/` → `report/figures/` |
| `report/B1_EDA_report.md` | Báo cáo B1 |
| `report/B2_B3_experiment_report.md` | Báo cáo B2/B3 (protocol + kết quả + thống kê) |
| `report/results/` | 7 CSV kết quả + `run_log.txt` + `idx_*.npy` (chỉ số split — sinh khi chạy pipeline) |
| `report/figures/` | Biểu đồ B1 (tag `3-lp_012_*`) + B2/B3 |
| `requirements.txt` | Danh sách thư viện (xem ghi chú GPU dưới đây) |

## Dữ liệu

Dataset Kaggle: **`diabetes-health-indicators-dataset`** (Alex Teboul / CDC BRFSS 2015).

Chạy trên Kaggle: *Notebook → File → Add Input* → chọn dataset đó. Cả 3 file
(`diabetes_012`, `diabetes_binary`, `diabetes_binary_5050split`) nằm cùng folder.
Khi chạy local, đặt file vào `./data/` (script tự tìm `input/`, `data/`, `/kaggle/input`).

## Cài đặt & chạy

```bash
pip install -r requirements.txt
# B1: EDA 3 file CSV
python Src/b1_eda.py
# B2+B3: pipeline chính (trên Kaggle chọn GPU T4; máy khác chạy CPU)
python Src/b2_b3_pipeline.py
# Vẽ lại figures B2/B3 từ CSV kết quả
python Src/make_report_figures.py
```

> **Ghi chú LightGBM GPU:** `lightgbm-gpu` không có trên PyPI — build từ source:
> `CMAKE_ARGS="-DUSE_CUDA=ON" python -m pip install --upgrade --no-binary :all: lightgbm`.
> Script tự động probe GPU (`nvidia-smi`) và fallback CPU nếu không có build GPU —
> toàn bộ kết quả số không thay đổi khi chạy CPU (seed cố định, chỉ khác thời gian).
> Kaggle đã có LightGBM GPU build sẵn trong image, không cần build thêm.

## Protocol (công bằng giữa các thành viên A/B/C)

- Chỉ số chính: **AP (Average Precision)** — bất biến theo ngưỡng, phù hợp 86/14.
- Ngưỡng: so `0,5` (mặc định) và **ngưỡng F1 tối ưu chọn trên VAL rồi khoá** —
  test set không bao giờ tham gia chọn tham số.
- Resample **luôn nằm trong fold CV / trên tập train** (không leakage).
- Seed = 42 ở mọi bước ngẫu nhiên; chỉ số split lưu sẵn trong `report/results/idx_*.npy`.
- A/C thay model: chỉ sửa hàm `build_model()` (PHẦN 5), không sửa tiền xử lý,
  split, ngưỡng hay đánh giá.

## Kết quả chính (từ `report/results/`)

- Chiến lược chọn: **class_weight** (CV-AP 0,4364 — recall lớp bệnh tăng 0,169 → 0,780
  gần như không mất AP).
- Model chính: **LightGBM + class_weight, ngưỡng VAL (≈0,65)** → TEST:
  AP 0,4368 / F1-macro 0,678 / AUC 0,826.
- Paired bootstrap: LightGBM thắng Logistic (+0,0231, p<0,001) và RF (+0,0436, p<0,001);
  ngang Stacking (ΔAP ≈ 0) → chọn LightGBM đơn giản.

## Ghi chú về `report/results/idx_*.npy`

Ba file chỉ số split (`idx_train.npy`, `idx_val.npy`, `idx_test.npy`) do `b2_b3_pipeline.py`
sinh ra khi chạy — lần chạy trước ghi vào `b2_out/` nên **không được commit** (`.gitignore`
loại `*_out/`). Nếu repo không có 3 file này, chạy lại `python Src/b2_b3_pipeline.py`
lần nữa là có (script ghi thẳng vào `report/results/`).

## Tái lập

Chạy lại các script trên cùng môi trường + seed 42 → số trong `report/results/*.csv`
phải trùng tuyệt đối. Bằng chứng lần chạy đã commit nằm trong
`report/results/run_log.txt`.
