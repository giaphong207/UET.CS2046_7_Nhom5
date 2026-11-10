# UET CS2046 – Nhóm 5: Dự đoán tiểu đường (BRFSS 2015)

Pipeline + xử lý mất cân bằng lớp cho bài toán phân loại 2 lớp
(`Diabetes_binary`, 86,1 / 13,9) trên bộ dữ liệu Diabetes Health Indicators
(BRFSS 2015, 253.680 dòng), kèm nghiệm phụ 3 lớp.

## Cấu trúc thư mục

| Thư mục / file | Nội dung |
|---|---|
| `code/b1_eda.py` | **B1** – EDA: duplicate, label noise, correlation, lift, insight cho B3 |
| `code/b2_b3_pipeline.py` | **B2 + B3** – pipeline reproducible: tiền xử lý (dedup + majority-vote), split 70/15/15 stratified seed 42, ma trận 5 model × 5 resample (CV trong fold), ngưỡng khoá trên VAL, paired bootstrap, nghiệm phụ 3 lớp |
| `code/b4_feature_selection.py` | **B4** – chọn lọc đặc trưng (filter / embedded / wrapper) trong fold, cùng split B2/B3, permutation importance để A đối chiếu SHAP |
| `code/make_report_figures.py` | Vẽ biểu đồ B2/B3 từ CSV → `report/figures/B2_B3/` |
| `report/SP1_survey_report.md` | **SP1** – khảo sát phương pháp & quyết định (cơ sở học thuật cho B2/B3/B4), kèm `report/references.bib` (15 mục) |
| `report/B1_EDA_report.md` | Báo cáo B1 |
| `report/B2_B3_experiment_report.md` | Báo cáo B2/B3 (protocol + kết quả + thống kê) |
| `report/B4_feature_selection_report.md` | Báo cáo B4 |
| `report/results/B1/` | CSV correlation + lift của 3 tệp (B1) |
| `report/results/B2_B3/` | 7 CSV + 3 file `.npy` (chỉ số split) + `run_log.txt` |
| `report/results/B4/` | CSV B4 (feature selection + importance) + `.npy` |
| `report/figures/B1/` `B2_B3/` `B4/` | Biểu đồ từng task (B1 + 7 hình B2/B3 + 2 hình B4) |
| `requirements.txt` | Danh sách thư viện (xem ghi chú GPU dưới đây) |

## Kết quả SP1 (decision log)

Mọi lựa chọn trong SP2 đều có cơ sở trong `report/SP1_survey_report.md` (mục 3, bảng D1–D9):
chọn AP làm chỉ số chính [5], chọn class_weight [3][4], LightGBM làm model chính [9],
majority-vote cho label noise [12], selector-in-fold cho B4. SP1 là "báo cáo khảo
sát + trích dẫn" — các report B* là "báo cáo kết quả thực nghiệm".

## Dữ liệu

Dataset Kaggle: **`diabetes-health-indicators-dataset`** (Alex Teboul / CDC BRFSS 2015).

Chạy trên Kaggle: *Notebook → File → Add Input* → chọn dataset đó. Cả 3 file
(`diabetes_012`, `diabetes_binary`, `diabetes_binary_5050split`) nằm cùng folder.
Khi chạy local, đặt file vào `./data/` (script tự tìm `input/`, `data/`, `/kaggle/input`).

## Cài đặt & chạy

```bash
pip install -r requirements.txt
python code/b1_eda.py               # B1: EDA 3 file CSV (~1 phút)
python code/b2_b3_pipeline.py        # B2+B3: pipeline chính (~58 phút CPU, GPU nhanh hơn)
python code/make_report_figures.py   # Vẽ figures B2/B3 từ CSV kết quả
python code/b4_feature_selection.py  # B4: chọn lọc đặc trưng (~11 phút, chạy SAU B2/B3)
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
- Seed = 42 ở mọi bước ngẫu nhiên; chỉ số split lưu trong `report/results/B2_B3/idx_*.npy`
  (B4 đọc lại file này để giữ cùng split; nếu thiếu thì tự tách mới — deterministic seed 42).
- A/C thay model: chỉ sửa hàm `build_model()` (PHẦN 5 trong `code/b2_b3_pipeline.py`
  và `code/b4_feature_selection.py`), không sửa tiền xử lý, split, ngưỡng hay đánh giá.
- **N_JOBS = 2** ở mọi script (Kaggle chỉ có 2 CPU — `n_jobs=-1` gây overhead
  thread vượt trội, không tăng tốc).

## Kết quả chính (từ `report/results/`)

- Chiến lược chọn (B2/B3): **class_weight** (CV-AP 0,4364 — recall lớp bệnh tăng
  0,169 → 0,780 gần như không mất AP).
- Model chính: **LightGBM + class_weight, ngưỡng VAL (≈0,65)** → TEST:
  AP 0,4368 / F1-macro 0,678 / AUC 0,826.
- Paired bootstrap: LightGBM thắng Logistic (+0,0231, p<0,001) và RF (+0,0436, p<0,001);
  ngang Stacking (ΔAP ≈ 0) → chọn LightGBM đơn giản.
- B4: chọn lọc đặc trưng **không cải thiện AP** (AP cực đại ở k=21 = baseline) →
  kết luận trung thực "giữ đủ 21 đặc trưng, không cần chọn lọc trên bộ này";
  k=18 (loại 3 đặc trưng yếu nhất) chỉ tụt −0,002 AP — chấp nhận được nếu cần mô
  hình gọn/di động. Permutation importance (B4) là bộ khung để A đối chiếu với SHAP.

## Ghi chú về `idx_*.npy`

Ba file chỉ số split (`idx_train.npy`, `idx_val.npy`, `idx_test.npy`) do
`b2_b3_pipeline.py` sinh ra khi chạy — được commit trong
`report/results/B2_B3/` (bộ canonical, một bản duy nhất). `b4_feature_selection.py`
đọc từ đây; nếu thiếu thì tự tách mới (deterministic, seed 42 → trùng số).

## Tái lập

Chạy lại các script trên cùng môi trường + seed 42 → số trong `report/results/*/*.csv`
phải trùng tuyệt đối. Bằng chứng lần chạy nằm trong `report/results/B2_B3/run_log.txt`
(hướng dẫn: chạy lại 4 lệnh ở mục "Cài đặt & chạy" và pipe stdout ra file).
