# Diabetes Research Track 1.2 — Task A4 & A5 (V2)

> **Trạng thái bàn giao: CODE + MẪU BÁO CÁO, CHƯA CÓ KẾT QUẢ TRÊN KAGGLE CSV GỐC.**
> Không dùng số liệu của chạy synthetic/quick cho SP2.

Tài liệu nghiên cứu hiện có: `config/A2_protocol_v2.yaml` và A1/A2/A3 trong `docs/lead-modeling/` của repository. Đề bài giảng viên và `De_cuong_Diabetes (1).pdf` được nhắc tới nhưng chưa có trong workspace; chưa thể xác nhận tuân thủ các tài liệu này. [METHODOLOGY_NOTES.md](docs/METHODOLOGY_NOTES.md) đối chiếu phương pháp thực sự chạy với các tài liệu trước đó.

## 1. Nội dung và ranh giới công việc

**A4 — Modeling:** B0 Dummy; B1 Logistic Regression, B2 Random Forest và B3 XGBoost được tuning bằng CV AP trên train. A1 dùng tham số B3 đã chọn, thêm class weighting và fit toàn bộ train; A2 dùng cùng A1 với ngưỡng tối đa F1 trên validation. A3 fit ANOVA `SelectKBest(f_classif)` và weighted XGBoost trên train cho 21/15/10/5 đặc trưng, chọn k trên validation. A1/A3 không có một search CV riêng. Test được đánh giá sau khi khóa mô hình/threshold.

**A5 — Analysis:** biểu đồ PR/ROC, confusion matrix, calibration/Brier, paired bootstrap AP (1.000 lần), lỗi FP/FN, phân tích theo mã nhóm tuổi/giới, SHAP/LIME cho mô hình **cuối** khi được bật. Không thay đổi mô hình theo test. SHAP/LIME không phải bằng chứng nhân quả.

**Stacking + Optuna:** chỉ là mở rộng chưa được triển khai trong gói này; đề cương mới không lấy Stacking làm baseline chính. Mô hình B1–B3 ở đây dùng `RandomizedSearchCV`, không phải Optuna. Không ghi A4 hoàn tất Stacking + Optuna nếu chưa làm phần mở rộng.

## 2. Cài đặt trên macOS / VS Code

```bash
cd ~/Projects/UET.CS2046_7_Nhom5/research/lead_modeling/a4_a5
python3 -m venv .venv-diabetes
source .venv-diabetes/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# nếu muốn chạy SHAP & LIME:
python -m pip install -r requirements-explain.txt
# nếu muốn chạy tests:
python -m pip install -r requirements-dev.txt
```

Hai notebook tự tìm component nested khi mở từ repository root, component hoặc thư mục `notebooks/`. Dùng kernel đã cài core dependencies; đường dẫn `data/`, `results/` và `scripts/` trong notebook luôn thuộc component.

**Không ghi đè `src/`, `requirements.txt` hay README đang có của nhóm một cách máy móc.** Khi merge vào repo hiện có, giữ các file của nhóm và tích hợp dependencies/thư mục theo nhu cầu.

Các lệnh cài đặt, chạy script và pytest dưới đây dùng **thư mục component** này. Từ repository root, đường dẫn tới script là `research/lead_modeling/a4_a5/scripts/run_a4.py`; đường dẫn data/output tương đối luôn tính theo working directory của lệnh.

## 3. Dataset

Nguồn: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

Đặt file đúng tên trong `data/`:

```text
data/diabetes_binary_health_indicators_BRFSS2015.csv
```

Cần đúng 253.680 dòng và 21 feature. Program sẽ từ chối các CSV khác trong chế độ nghiên cứu thông thường. Dữ liệu được kiểm tra schema, miền giá trị, trùng vector, bất đồng nhãn và SHA256. BMI phải >0, không đặt thêm cutoff lâm sàng; các mã ordinal và số ngày phải nguyên trong miền mã hóa. Tên file, số dòng và checksum cục bộ không tự xác thực nguồn Kaggle. Không tự xóa vector trùng.

## 4. Chạy A4 trên file Kaggle thật

```bash
python scripts/run_a4.py \
  --data data/diabetes_binary_health_indicators_BRFSS2015.csv \
  --output-dir results/a4_full_001 \
  --trials 10 --cv 3
```

- 70% train, 15% validation, 15% test; `seed=42`; lưu chỉ số chia tập.
- Hyperparameter search dùng **3-fold Stratified CV trong train**, `scoring=average_precision`.
- B1 dùng `StandardScaler` fit bên trong mỗi CV fold; các mã ordinal được xử lý như số, không có one-hot encoding.
- B1/B2/B3 là baseline không có class weighting. `FoldAwareXGB.fit()` tính trọng số từ y được truyền vào mỗi lần fit; A1/A3 hiện truyền **toàn bộ train**, không chạy weighted CV riêng.
- A3 dùng ANOVA `SelectKBest(f_classif)` fit trên train. Không báo cáo ranking CV/permutation importance hay độ ổn định giữa folds vì chưa thực hiện các phân tích đó.
- Lựa chọn `k` nhỏ nhất đạt AP trên validation giảm **không quá 0,01 tuyệt đối** so với mô hình có weighting đầy đủ; chọn final theo AP validation.
- Khi AP bằng nhau chính xác, thứ tự ưu tiên final là B1, B2, B3, A2, A3 với k đã chọn. Ngưỡng F1 hòa nhau ưu tiên ngưỡng cao nhất. Ngưỡng FINAL luôn chọn từ validation, kể cả khi final là B1/B2/B3; hàng baseline vẫn báo cáo ngưỡng 0.5.
- Audit lưu số mẫu âm/dương và prevalence của từng split, cùng số vector X giống hệt nhau giữa từng cặp split. Vector giống nhau không xác định cùng cá nhân.
- Chỉ sau khi chốt config, test mới được đánh giá. Không refit train+validation sau khi chọn threshold.
- Mọi kết quả nằm dưới `results/` và sẽ không được `git add` mặc định.

> **Ngân sách search:** mặc định 10 cấu hình/họ và 3 folds. B1 thay đổi duy nhất độ mạnh regularization L2 qua `C=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]`, giữ `class_weight=None`. Khi yêu cầu 15, B1 chỉ đánh giá 10 cấu hình khác nhau; không lặp cấu hình để tăng số đếm. Manifest và runtime config lưu ngân sách yêu cầu, ngân sách sau override, số cấu hình thực chạy, số CV/refit fits và trạng thái hết không gian hữu hạn. B2/B3 có không gian đủ cho 10–15 cấu hình. Chạy thật có thể tốn thời gian trên CPU; mọi override phải được công bố.

**Kiểm tra code nhanh với CSV thật (không dùng để viết SP2):**

```bash
python scripts/run_a4.py --data data/diabetes_binary_health_indicators_BRFSS2015.csv \
  --output-dir results/a4_smoke_001 --quick
```

`--quick` lấy tối đa 260 dòng (`min(260, N-2)`), dùng 1 trial/họ và 2 CV folds, đánh dấu `SMOKE_NOT_RESEARCH_RESULT`. `source_row_indices.npy` lưu map subset → vị trí dòng CSV gốc (0-based). Chỉ số split là vị trí trong subset; `test_predictions.csv:row_index` là vị trí trong CSV gốc, không phải ID cá nhân. Không được dùng làm bằng chứng nghiên cứu.

**Kiểm tra code bằng CSV tổng hợp tự tạo:** chỉ dùng `--demo`, đầu ra được đánh dấu `SYNTHETIC_QA_ONLY` và A5 mặc định sẽ từ chối nếu không bật chế độ QA.

## 5. Chạy A5

A4 phải chạy thành công trên cùng CSV thật trước khi chạy A5:

```bash
python scripts/run_a5.py \
  --data data/diabetes_binary_health_indicators_BRFSS2015.csv \
  --a4-dir results/a4_full_001 \
  --output-dir results/a5_full_001 \
  --bootstraps 1000
```

Thêm giải thích cho mô hình cuối (chi phí có thể đáng kể):

```bash
python scripts/run_a5.py \
  --data data/diabetes_binary_health_indicators_BRFSS2015.csv \
  --a4-dir results/a4_full_001 \
  --output-dir results/a5_full_with_xai_001 \
  --bootstraps 1000 --shap --lime
```

A5 kiểm tra hash các artifact quan trọng trước khi load model; xác minh source code, Python major/minor và phiên bản chính xác của các thư viện model. A5 kiểm tra toàn bộ split phải rời nhau, đầy đủ và đúng seed; kiểm tra registry, model IDs, thứ tự feature, feature được selector chọn, threshold và metadata selection. Xác suất test được tính lại từ model đã serialize và so sánh CSV (`atol=1e-8`, `rtol=1e-7`); A5 dùng xác suất tính lại để tránh sai lệch ranking do round-trip CSV. Không refit hoặc thay threshold. Run cũ thiếu manifest v2 phải chạy lại A4; môi trường/source không khớp phải được khôi phục hoặc chạy lại trong môi trường thống nhất.

Quick và full-row demo có thể kiểm thử A5 trên **cùng CSV nguồn** với `--allow-smoke`; A5 mặc định từ chối cả hai. Ví dụ sau vẫn là QA:

```bash
python scripts/run_a5.py --data data/diabetes_binary_health_indicators_BRFSS2015.csv \
  --a4-dir results/a4_smoke_001 --output-dir results/a5_smoke_001 \
  --allow-smoke --bootstraps 1000
```

Bootstrap dùng 1.000 **lần thử**, loại mẫu chỉ có một lớp và không lấy bù; summary lưu attempted/valid/excluded, seed 42, confidence 0.95 và percentile interval. Research A5 yêu cầu 1.000 lần thử; số khác chỉ cho QA và được ghi thành override. Subgroup CSV luôn giữ schema và mọi nhóm Sex/Age quan sát được, gồm n, positive_n, status và lý do withheld khi n<50 hoặc chỉ có một lớp. Quy tắc này không chứng minh đủ power hoặc công bằng y tế.

Mỗi lần chạy dùng **thư mục output mới** để tránh ghi đè hoặc lẫn phiên bản cũ. Các hash phát hiện thay đổi/lẫn artifact, không phải chữ ký xác thực người tạo.

**Cảnh báo an toàn:** `joblib.load()` dùng pickle; chỉ tải `fitted_models.joblib` do chính bạn hoặc nguồn được tin cậy tạo. Không đọc file model không đáng tin từ Internet.

## 6. File đầu ra dự kiến

A4:

- `manifest.json` v2: UUID run ID, UTC start/end, duration, dataset/source/artifact SHA256; `trials_per_family` là map số cấu hình **thực chạy**
- `config.yaml` (snapshot nguyên văn protocol), `runtime_config.json` (CLI yêu cầu/effective, seeds, split/CV, full search spaces/counts, overrides), `source_identity.json` (hash cả code untracked), `versions.json`
- `dataset_sha256.txt`, `dataset_audit.json`, `source_row_indices.npy`, `split_indices.npz`
- `cv_results.csv` (config, AP từng fold/trung bình, fit time, rank)
- `validation_metrics.csv` (chọn mô hình/threshold/feature count)
- `test_metrics.csv`, `test_predictions.csv` (chỉ sau khóa lựa chọn), `ablation.csv` (mọi k trên validation; chỉ k đã chọn trên test)
- `feature_sets.json`, `selection.json`, `fitted_models.joblib`

A5:

- `manifest_a5.json`, `analysis_config.json`, `versions_a5.json`, `source_identity_a5.json`, `artifact_verification.json`: liên kết run ID/hash A4, config/environment A5 và hash đầu ra
- `precision_recall_test.png`, `roc_test.png`, `confusion_matrix_final.png`, `calibration_final.png`
- `paired_bootstrap_ap.csv`, `paired_bootstrap_summary.json` (95% CI hiệu AP final - B3)
- `error_cases.csv`, `case_selection.json`, `subgroups_test.csv`, `additional_metrics.csv`
- (tùy chọn) `shap_beeswarm_final.png`, `shap_bar_final.png`, `shap_waterfall_final.png`, `shap_attributions.npz`, `shap_metadata.json`, `lime_FP/FN/TP.*`, `lime_metadata.json`, `lime_training_source_indices.npy`

SHAP/LIME chưa được runtime-verify trong đợt fix hiện tại vì chưa cài các thư viện optional. Metadata mới ghi seed, model/run ID, sample/background source rows và settings; SHAP summary 15 dòng là summary trên mẫu, không phải toàn quần thể. Xem [CODEX_FIX_REPORT.md](docs/CODEX_FIX_REPORT.md) cho bằng chứng kiểm thử hiện tại và giới hạn; `docs/QA_REPORT.md` là ghi chép QA lịch sử.

Runtime config lưu cả tham số cố định của từng pipeline/learner đã fit, không chỉ tham số được search. A5 đối chiếu các tham số này với registry và CV winner; CV trace phải chứa cấu hình riêng biệt hợp lệ và AP trung bình/std khớp từng fold. Thay đổi các lựa chọn khoa học cố định trong YAML phải được triển khai/đối chiếu trước khi chạy; code từ chối snapshot không phù hợp. CV folds và budget vẫn là CLI overrides công khai.

Giới hạn phân tích hiện tại: CI chỉ cho FINAL minus B3; CI B2/B3 minus B1 theo câu hỏi baseline trong A1/A2 chưa được triển khai. Khi FINAL=B3, đây là self-comparison hợp lệ nhưng không cung cấp thông tin về khác biệt mô hình. Các PDF/Word trước đó là tài liệu lịch sử; phương pháp và lệnh hiện hành được đối chiếu tại `METHODOLOGY_NOTES.md`.

## 7. Quy trình đảm bảo kết quả trung thực

1. Không đưa số liệu từ file `synthetic` hoặc `quick` vào SP2.
2. Không đánh giá model trên test khi vẫn đang tối ưu hyperparameters, chọn threshold hoặc số đặc trưng.
3. Không học scaler/feature selector trên toàn bộ dữ liệu trước split/CV.
4. AP của scikit-learn (`average_precision_score`) không đồng nhất với diện tích nội suy hình thang PR.
5. Khoảng tin cậy bootstrap điều kiện theo mô hình đã huấn luyện; không phải độ bất định của toàn bộ quá trình nghiên cứu.
6. Nhãn gộp tiền đái tháo đường/đái tháo đường trong khảo sát; không suy luận bệnh tương lai hay chỉ định lâm sàng.
7. Nếu pipeline lỗi hoặc thiếu `lime`, sửa và kiểm thử; không tự tạo hình hay bảng kết quả để “cho đẹp”.
8. Ghi **AI Usage Log** đầy đủ theo đề bài: prompt, câu trả lời, kiểm chứng, phần được dùng/loại bỏ.

## 8. Kiểm thử tự động

```bash
PYTHONPATH=src python -m pytest tests -q
```

Bộ kiểm thử xác minh schema/domain, split/diagnostics, threshold/tie, weighting, paired bootstrap, subgroup schema, provenance/search counts, notebook paths, synthetic A4→A5 và fault injection các artifact không nhất quán. Đây chỉ là **code QA**; không thay thế thực nghiệm Kaggle thật. Output synthetic của pytest nằm trong thư mục tạm.

## 9. Đưa lên branch cá nhân

```bash
cd ~/Projects/UET.CS2046_7_Nhom5
git switch feature/lead-modeling-a
git status
git add research/lead_modeling/a4_a5/
# Kiểm tra file sắp commit trước khi thực hiện:
git diff --cached --stat
git commit -m "feat(lead-modeling): add reproducible A4 and A5 pipeline"
git push
```

Nếu repository đã có những file cùng tên, kiểm tra diff kỹ và merge có chọn lọc; **không `git add -f data/*.csv` hoặc commit thư mục `results/`**.

## 10. Tài liệu tham khảo kỹ thuật

- Scikit-learn Average Precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- Scikit-learn permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- XGBoost parameters: https://xgboost.readthedocs.io/en/stable/parameter.html
- SHAP TreeExplainer (tham khảo mở rộng): https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html

**Cần kiểm chứng tương thích phiên bản trên môi trường của nhóm trước khi chạy đầy đủ.**
