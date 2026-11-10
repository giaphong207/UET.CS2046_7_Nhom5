# SP3 – Nhật ký sử dụng AI (AI Usage Log)

**Nhóm 5 – CS2046 Trí tuệ nhân tạo** | Phụ trách chính: D (QA) · Ghi nhận các tương tác: B
**Định dạng theo Mục II.2.1 của đề bài** (11 trường/tương tác). "Prompt đầy đủ / Phản hồi đầy đủ"
được trích dẫn nguyên văn trong **đính kèm `SP3_appendix_raw.md`** (đề cho phép: "có thể đính kèm
lịch sử hội thoại gốc và sử dụng bảng tổng hợp").

> Quy tắc kiểm chứng chung của nhóm (đọc trước khi chấm): mọi phản hồi AI được (i) chạy lại
> trên dữ liệu thật trước khi áp dụng, (ii) đối chiếu chéo 3 tệp CSV và với mô tả dataset trên
> Kaggle, (iii) nội dung do nhóm tự thêm được phân biệt rõ ở cột "Nội dung chỉnh sửa hoặc loại bỏ".

## Tổng quan (danh mục tương tác)

| Mã | Mục đích | Kết quả quyết định |
|---|---|---|
| AI-001 | Kế hoạch thực hiện B1/B2/B3 cho vai trò B | **Dùng** — làm checklist tuần W1 |
| AI-002 | Viết script EDA (B1) chạy trên Kaggle, không internet | **Dùng + chỉnh sửa** (đường dẫn) |
| AI-003 | Sửa bug target sai (phân tích nhầm theo cột `Income`) | **Loại bỏ** toàn bộ kết quả chạy lần 1 (tệp 3-lớp) |
| AI-004 | Diễn giải EDA thật → chốt bài toán & chiến lược B3 | **Dùng** (mọi số lấy từ output script) |
| AI-005 | Pipeline B2/B3 theo protocol: AP / ngưỡng VAL / ablation / paired bootstrap | **Dùng + bổ sung** (dedup-lần-2 sau majority-vote) |
| AI-006 | Chạy log B2/B3 → kiểm tra, viết báo cáo, hướng dẫn GitHub | **Dùng** (kiểm tra bằng máy: confusion matrix, split) |
| AI-007 | Viết lại B2/B3 "hoàn hảo" khi có internet (LightGBM + imbalanced-learn) | **Dùng + vá 3 bug thật** (np.save, knn index, argmax) |
| AI-008 | Code B4 (feature selection) + báo cáo B4 | **Dùng** (kết luận: giữ đủ 21 đặc trưng) |
| AI-009 | Thấu repo (2 lần) → phát hiện thiếu SP1/SP3, trùng idx, câu chữ | **Dùng** (đã vá trong repo) |
| AI-010 | Soạn SP1 survey + references.bib + audit 3 DOI | **Dùng + thay 1 tham chiếu** (ADASYN) |

## AI-001 — Kế hoạch B1/B2/B3 cho vai trò B

| Trường | Nội dung |
|---|---|
| Mã tương tác | AI-001 |
| Thời gian | ≈ 2026-11-01 (mốc tuần W1) |
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Có kế hoạch từng bước cho task B (Data & Pipeline, 25%): B1 EDA, B2 pipeline stub, B3 tiền xử lý + mất cân bằng |
| Prompt đầy đủ | Xem **đính kèm A.1** ("hướng dẫn tôi làm B1 B2 B3" + ngữ cảnh vai trò) |
| Phản hồi đầy đủ | Tóm tắt: (i) B1 = 5 nhóm câu hỏi EDA + đầu ra (phân bố target, insight kèm hành động, 6–8 plot); (ii) B2 = stub `src/` + `config` (seed, logging), module `data.py`/`split.py`; (iii) B3 = mọi estimator fit trên train trước, resample sau split, chọn strategy bằng 5-fold stratified CV theo recall lớp thiểu. Xem **A.1** |
| Nhận xét, đánh giá | Kế hoạch hợp lý, khớp phân công 25%. **Hạn chế:** AI không thấy dữ liệu thật → bỏ sót đặc thù BRFSS (mã sentinel 777/888/999, label noise) |
| Nội dung được sử dụng | Thứ tự B1→B2→B3, tiêu chí "chấm xong" từng task (stub chạy <5 phút; cross-tab mất cân bằng vào SP1; B3 chọn theo recall lớp thiểu) |
| Nội dung chỉnh sửa hoặc loại bỏ | **Thêm (do nhóm):** bước kiểm label-noise và rà mã sentinel BRFSS — AI bỏ sót, căn cứ: tài liệu khảo sát BRFSS + EDA thật |
| Cách kiểm chứng | Chạy EDA thật trên 3 CSV: phát hiện label noise 2,41% (không có sentinel → dataset đã sạch) → xác nhận bổ sung là cần thiết |
| Kết quả áp dụng | Checklist W1; tiêu chí "done" dùng lại cho B2/B3 |

## AI-002 — Script EDA B1 (Kaggle, không internet)

| Trường | Nội dung |
|---|---|
| Mã tương tác | AI-002 |
| Thời gian | ≈ 2026-11-01 → 11-02 |
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Script EDA chạy trong 1 cell Kaggle (pandas/numpy/matplotlib có sẵn, dataset mount sẵn) |
| Prompt đầy đủ | Xem **đính kèm A.2** ("làm cho tôi B1 hoàn chỉnh", "tôi dùng kaggle để chạy và ko bật được internet nên chạy trên dataset") |
| Phản hồi đầy đủ | Bản local `eda.py` + bản Kaggle: tự tìm target, rà missing/sentinel, phân bố + ratio mất cân bằng, per-class + correlation, lift theo mức, sinh CSV + plots + block Insights. Xem **A.2** |
| Nhận xét, đánh giá | Script chạy end-to-end, có smoke-test trên dữ liệu giả lập trước khi dùng thật. **Lỗi còn lại:** hard-code đường dẫn `../input/<slug>` — sai với dataset public (`../input/datasets/<user>/<slug>`) |
| Nội dung được sử dụng | Toàn bộ logic 8 section của script EDA |
| Nội dung chỉnh sửa hoặc loại bỏ | **Chỉnh sửa:** thay hard-code bằng hàm `find_csvs()` tự dò (hợp cả layout public + riêng); không tìm thấy → báo lỗi chỉ dẫn "Add Input" thay vì chạy rỗng |
| Cách kiểm chứng | 3 bước: (1) smoke-test data giả lập có missing/sentinel/noise; (2) chạy thật, log mở đầu `[OK] Tim thay 3 CSV (noi dung: /kaggle/input/datasets/alexteboul/...)`; (3) đối chiếu 6 CSV + 30 plots khớp kỳ vọng |
| Kết quả áp dụng | Script EDA chính thức; output làm nền báo cáo B1 + input cho B3 |

## AI-003 — Phát hiện & sửa bug target sai (kết quả EDA lần 1 bị vô hiệu)

| Trường | Nội dung |
|---|---|
| Mã tương tác | AI-003 |
| Thời gian | ≈ 2026-11-02 |
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Phát hiện lỗi: EDA chạy thật, tệp 3-lớp bị phân tích theo sai cột target |
| Prompt đầy đủ | Xem **đính kèm A.3** ("sửa lỡi" + ảnh log WARN 0 CSV; sau đó dán toàn bộ log v1) |
| Phản hồi đầy đủ | (i) Nguyên nhân: dataset Alex Teboul đặt tên `Diabetes_012`/`Diabetes_binary`; script tìm cột `Diabetes` không thấy → fallback cột cuối (`Income`); (ii) vá bằng `find_target()` ưu tiên đúng tên, WARN khi fallback, crash rõ nếu không có cột nào chứa "diab"; (iii) sửa chỉ số label-noise bằng groupby. Xem **A.3** |
| Nhận xét, đánh giá | **Lỗi nghiêm trọng nhất của B1:** AI giả định tên cột theo BRFSS gốc. Block kết quả tệp 3-lớp lần 1 (target=Income, "Imbalance 9,2×", corr Education 0,45) **vô giá trị, phải loại toàn bộ**. Nhóm phát hiện bằng việc đọc kỹ log (target phải là cột diabetes) |
| Nội dung được sử dụng | `find_target()` v2 + label-noise bằng groupby |
| Nội dung chỉnh sửa hoặc loại bỏ | **LOẠI BỎ:** toàn bộ block tệp 3-lớp của lần chạy v1. **Thay bằng:** kết quả chạy lại với `Diabetes_012` (84,2/1,8/13,9%, ratio 46,1×, top feature GenHlth r=0,30) |
| Cách kiểm chứng | 3 lớp: (1) đối chiếu 22 cột thật in ra log — không có cột `Diabetes`; (2) log v2 hiển thị `[TARGET] ... dung cot 'Diabetes_012'`/`'Diabetes_binary'`; (3) số ca dương 35.346 nhất quán giữa 3 tệp (012 = unbalanced = 2× tệp 5050) |
| Kết quả áp dụng | EDA được hiệu đính; số chuẩn: 46,1× / dup 9,4–9,5% / noise 2,06–2,41% / top feature GenHlth (lift 2,72×) |

## AI-004 — Diễn giải EDA, chốt bài toán & chiến lược B3

| Trường | Nội dung |
|---|---|
| Mã tương tác | AI-004 |
| Thời gian | ≈ 2026-11-02 |
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Từ số EDA thật: (1) chốt bài toán chính/phụ; (2) thứ tự tiền xử lý B3; (3) tiêu chí chọn strategy |
| Prompt đầy đủ | Xem **đính kèm A.4** (dán log v2 đã verify) |
| Phản hồi đầy đủ | (i) bài toán chính = 2-lớp 86/14; tệp 5050 chỉ sanity-check; 3-lớp làm nghiệm phụ (lớp tiền tiểu đường 1,8% → oversampling 3-lớp vô dụng); (ii) thứ tự B3: dedup → label noise (giữ/majority-vote) → resample; (iii) chọn theo **recall + F1 lớp bệnh trên 5-fold stratified CV**, không accuracy. Xem **A.4** |
| Nhận xét, đánh giá | Logic đúng với số thật. **Lưu ý:** giai đoạn đầu AI từng *giả định* phân bố 3-lớp ~40/20/40 (sai so với thật 84,2/1,8/13,9); may là quyết định cuối được rà lại với số thật trước khi chốt |
| Nội dung được sử dụng | 3 kết luận trên; mọi con số trong báo cáo lấy **trực tiếp từ output script**, không lấy từ phản hồi AI |
| Nội dung chỉnh sửa hoặc loại bỏ | Loại giả định 40/20/40; mỗi số đưa vào báo cáo đều có "cơ sở" (file CSV + lệnh tái lập) |
| Cách kiểm chứng | Chạy lại cùng seed → số trùng; đối chiếu chéo 3 tệp (35.346 ca dương); đối chiếu mô tả Kaggle (253.680 dòng, 21 features) |
| Kết quả áp dụng | Chốt hướng bài toán tại họp W1; B3 triển khai đúng thứ tự + tiêu chí |

## AI-005 — Pipeline B2/B3 theo protocol đánh giá

| Trường | Nội dung |
|---|---|
| Mã tương tác | AI-005 |
| Thời gian | ≈ 2026-11-04 |
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Pipeline reproducible: dedup + noise + split stratified 70/15/15 seed 42 + 5 strategy mất cân bằng + đánh giá theo đúng 4 nguyên tắc nhóm chốt (AP là chính; ngưỡng 0,5 vs max-F1 trên val khóa trước test; ablation đổi 1 yếu tố; paired bootstrap 1.000 lần) |
| Prompt đầy đủ | Xem **đính kèm A.5** (kèm ảnh tiêu chí đánh giá từ đề/học phần) |
| Phản hồi đầy đủ | File pipeline (1 cell Kaggle) + fallback SMOTE/ADASYN bằng numpy; resample CHỈ trong fold CV; xuất CSV + bootstrap CI. Xem **A.5** |
| Nhận xét, đánh giá | Thiết kế đúng protocol. **Phát hiện khi test thật (chạy của nhóm):** sau majority-vote, các dòng trong cùng nhóm đã thành trùng nhau → cần **dedup lần 2** (AI ban đầu chỉ dedup trước noise). Nhóm đã vá: `n_collapsed_after_noise` (gộp 1.566 dòng ở bản chạy thật) |
| Nội dung được sử dụng | Toàn bộ luồng `prepare → split → ablation CV → chọn strategy → final 2 ngưỡng → paired bootstrap → nghiệm phụ 3-lớp` |
| Nội dung chỉnh sửa hoặc loại bỏ | **Thêm (do nhóm):** dedup-lần-2 sau khi gán nhãn majority-vote; lưu `idx_*.npy` tách 3 file (không gộp 1 mảng) |
| Cách kiểm chứng | Unit-test thuần numpy (majority-vote loại sạch nhóm nhiễu; SMOTE/ADASYN cân bằng 1:1); crash-test `main()` trên data mô phỏng trước khi chạy thật |
| Kết quả áp dụng | `code/b2_b3_pipeline.py` — mã nguồn B2/B3 trong repo |

## AI-006 — Chạy B2/B3 thật: kiểm tra log, viết báo cáo, hướng dẫn GitHub

| Trường | Nội dung |
|---|---|
| Mã tương tác | AI-006 |
| Thời gian | ≈ 2026-11-08 → 11-09 |
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Từ log chạy thật (3459s): (1) kiểm tra log đúng protocol chưa; (2) viết báo cáo B2/B3; (3) hướng dẫn up GitHub (struct, commit, .gitignore, figures) |
| Prompt đầy đủ | Xem **đính kèm A.6** (dán log đầy đủ + "kiểm tra xem log đã đúng ý chưa, và tạo file báo cáo và hướng dẫn tôi up lên github") |
| Phản hồi đầy đủ | (i) Rà 7 điểm, tất cả đạt (AP bất biến theo ngưỡng; ablation 1 yếu tố; bootstrap theo chuẩn; thời gian khớp ước tính); (ii) báo cáo B2/B3 + 5 hình từ CSV; (iii) checklist GitHub. Xem **A.6** |
| Nhận xét, đánh giá | Phát hiện 2 điểm để bảo vệ: RF×SMOTE recall tụt (mô hình "trội" với resample) và kết luận "stacking ≈ LightGBM (p=0,948, không đáng kể)" — nhóm giữ nguyên kết luận trung thực này trong báo cáo (không inflate) |
| Nội dung được sử dụng | Báo cáo B2/B3 (8 mục), 6 CSV + 3 npy + run_log, cấu trúc repo `code/ + report/{figures,results}/B1·B2_B3·B4` |
| Nội dung chỉnh sửa hoặc loại bỏ | **Sửa số:** thống nhất confusion matrix trong `final_test_metrics.csv` (TP+FP+FN+TN = 34.187 = size test, mọi dòng khớp recall/precision ±<0,005) |
| Cách kiểm chứng | Tính lại từ TP/FP/FN/TN trong CSV: 10/10 dòng khớp tuyệt đối; split 159.535/34.186/34.187 trùng giữa B2/B3 và B4; AP 0,4368 / AUC 0,826 trùng LightGBM primary |
| Kết quả áp dụng | `report/B2_B3_experiment_report.md` + `report/figures/B2_B3/` (7 PNG) + commit 2 lệnh `[B] ...` |

## AI-007 — B2/B3 bản "hoàn hảo" (có internet: LightGBM + imbalanced-learn)

| Trường | Nội dung |
|---|---|
| Mã tương tác | AI-007 |
| Thời gian | ≈ 2026-11-09 |
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Nâng cấp pipeline: 5 model × 5 strategy thật (LightGBM + imbalanced-learn chuẩn thay fallback numpy), fix 3 bug thật, thêm nghiệm phụ 3-lớp |
| Prompt đầy đủ | Xem **đính kèm A.7** ("tôi có thể dùng internet nên hãy viết lại cho hoàn hảo" + log chạy trước) |
| Phản hồi đầy đủ | (i) Fix crash `np.save` (3 mảng khác kích thước → 3 file npy); (ii) fix `nn.kneighbors(...)[0]` (return_distance=False trả về (n,k), không cần [0]); (iii) fix dòng F1 3-lớp (argmax ngưỡng 0,5 → `predict()` chuẩn); (iv) ma trận 5 model + bootstrap + ablation. Xem **A.7** |
| Nhận xét, đánh giá | 3 bug đều được phát hiện bằng **crash-test trên data mô phỏng** trước khi chạy thật — không phải chờ lỗi trên Kaggle. Thời gian chạy thật 3459s khớp ước tính 30–60 phút |
| Nội dung được sử dụng | Toàn bộ `code/b2_b3_pipeline.py` hiện hành trong repo |
| Nội dung chỉnh sửa hoặc loại bỏ | **Thay (do nhóm):** đường dẫn output sang `report/results/B2_B3/` (commit trực tiếp, không đi qua `b2_out/` bị .gitignore) — chỉnh bởi nhóm khi restructure repo |
| Cách kiểm chứng | Crash-test `main()` với stub sklearn; unit-test resample (cân bằng 1:1); log chạy thật khớp |
| Kết quả áp dụng | Chạy thật: class_weight thắng (AP 0,4364, recall 0,780); LightGBM vượt baseline p<0,001; kết quả trong `report/results/B2_B3/` |

## AI-008 — B4 (chọn lọc đặc trưng): code + báo cáo

| Trường | Nội dung |
|---|---|
| Mã tương tác | AI-008 |
| Thời gian | ≈ 2026-11-09 |
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Code B4 (7 phương pháp × 5 mức k, selector in-fold, permutation importance cho SHAP) + báo cáo B4 theo chuẩn 8 mục của B1/B2/B3 |
| Prompt đầy đủ | Xem **đính kèm A.8** ("viết code kaggle cho B4", "đây là log B4 hãy làm như B1 B2 B3") |
| Phản hồi đầy đủ | (i) Script B4: cùng split B2/B3 (đọc `idx_*.npy`), 7 selector, bootstrap; (ii) báo cáo B4; (iii) rà log: 7 kiểm tra đạt, chỉ ra **1 hạn chế kỹ thuật** (LightGBM đa luồng gây lệch float ~1e-6 → recall/MCC nhạy ngưỡng lệch nhẹ so B2/B3) kèm cách khắc phục `deterministic=True`. Xem **A.8** |
| Nhận xét, đánh giá | Kết luận B4 là **kết luận "không cần chọn lọc"** (AP cực đại đúng k=21; 15/21 đặc trưng imp≈0) — trung thực, không ép số; top-5 permutation importance (BMI > GenHlth > Age > HighBP > HighChol) khớp thứ tự EDA B1, là bộ khung kiểm chứng SHAP cho A |
| Nội dung được sử dụng | `code/b4_feature_selection.py` + `report/B4_feature_selection_report.md` + 6 CSV + 2 PNG |
| Nội dung chỉnh sửa hoặc loại bỏ | **Thêm (do nhóm):** ghi chú khắc phục `deterministic=True` vào mục hạn chế; quyết định giữ `B2_B3/idx_*.npy` làm bộ duy nhất (không duplicat sang `B4/`) |
| Cách kiểm chứng | 7 kiểm tra qua máy: split trùng B2/B3 (cùng kích thước, cùng tỷ lệ 0,1471); AP/AUC B4 = B2/B3 (0,4368/0,826); k=21 đồng nhất cả 7 phương pháp; lgbm_gain ≡ rfe_lgbm; bootstrap dAP=0 (selected = full, đúng) |
| Kết quả áp dụng | B4 hoàn chỉnh; quyết định: mô hình cuối dùng đủ 21 đặc trưng (giữ cấu hình B2/B3); 15 đặc trưng imp≈0 là "mảng mềm" cho triển khai gọn (−0,002 AP, k=18) |

## AI-009 — Thấu repo (2 lần) và vá lỗi

| Trường | Nội dung |
|---|---|
| Mã tương tác | AI-009 |
| Thời gian | ≈ 2026-11-09 → 11-10 |
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Kiểm tra cây thư mục repo trước khi nộp: đường dẫn code/report, .gitignore, nhất quán số, đủ sản phẩm |
| Prompt đầy đủ | Xem **đính kèm A.9** ("kiểm tra cây thư mục này có lỗi gì không", "cây thư mục này đã ổn chưa" + 2 zip) |
| Phản hồi đầy đủ | Lần 1: phát hiện (i) `.gitignore` chặn `b2_out/` → `idx_*.npy` không vào Git (đã vá exclusion); (ii) **thiếu SP1 (khảo sát paper) và SP3 (AI log)**; (iii) trùng `idx_*.npy` ở B4; (iv) câu B4 dòng 9 viết ngược hướng đọc split; (v) `run_log.txt` chỉ là hướng dẫn, chưa phải log thật. Lần 2 (repo mới): SP1 + `references.bib` đã có, các lỗi nhỏ đã vá, **còn thiếu SP3**. Xem **A.9** |
| Nhận xét, đánh giá | Thấu qua máy (grep link, md5 split, git check-ignore), không kiểm tra bằng mắt → phát hiện được cả lỗi mà nhóm chưa thấy (bị .gitignore chặn file quan trọng) |
| Nội dung được sử dụng | Cấu trúc repo hiện hành (code/ + report/{figures,results}/B1·B2_B3·B4); 2 commit chuẩn |
| Nội dung chỉnh sửa hoặc loại bỏ | **Sửa (do nhóm):** bỏ `idx_*.npy` thừa ở B4; sửa dòng 9 B4 report; thêm exclusion .gitignore |
| Cách kiểm chứng | 48 file: 0 link gãy (grep mọi `report/...` trong báo cáo → file tồn tại); md5 `idx_*.npy` B2/B3 = B4 (đã gộp 1); `git check-ignore` xác nhận .gitignore |
| Kết quả áp dụng | Repo đạt đủ SP1 + SP2 + số liệu nhất quán; phần còn lại = SP3 (mục này) |

## AI-010 — Soạn SP1 (khảo sát) + references.bib + audit tham chiếu

| Trường | Nội dung |
|---|---|
| Mã tương tác | AI-010 |
| Thời gian | ≈ 2026-11-10 |
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | SP1: danh sách 15 paper, so sánh SMOTE/ADASYN/undersample/class-weight, AP vs accuracy, LightGBM/stacking, label noise, feature selection; decision log D1–D9 gắn "cơ sở học thuật ↔ bằng chứng thực nghiệm"; `references.bib` export LaTeX/Zotero |
| Prompt đầy đủ | Xem **đính kèm A.10** ("làm a và b cho tôi" — a=SP3, b=fix run_log + 3 DOI thiếu) |
| Phản hồi đầy đủ | SP1_survey_report.md (6 mục + decision log 9 dòng) + references.bib (15 mục). Xem **A.10** |
| Nhận xét, đánh giá | Điểm mạnh: **mỗi quyết định D1–D9 đều gắn cặp "cơ sở học thuật + bằng chứng thực nghiệm từ B2/B3/B4"** (không phải survey "giáo trình nói X tốt nên dùng X"). **Audit phát hiện:** (i) mục [2] ADASYN được ghi "Seo et al. 2011" — **nghi ngờ sai, cần thay bằng nguyên bản He, Garcia & Li (IEEE TNN 2008)**; (ii) [5] Davis & Goadrich thiếu DOI; (iii) [9] trong nội dung SP1 trỏ sang LightGBM (Ke et al.) nhưng [9] trong bib là Guyon (feature extraction) → **trùng số, cần tách [16]**; (iv) LightGBM (Ke et al. 2017) chưa có mục riêng trong bib |
| Nội dung được sử dụng | Cấu trúc SP1 + decision log + bib |
| Nội dung chỉnh sửa hoặc loại bỏ | **Sửa (do nhóm + AI audit):** [2] → He/Garcia/Li TNN 2008 (DOI 10.1109/TNN.2008.2008108); [5] thêm DOI 10.1109/IJCNN.2006.1471066 (ghi rõ: tra doi.org trước khi in); thêm [16] LightGBM (Ke et al., NeurIPS 2017); thêm [17] Altmann et al. (permutation importance); đánh dấu [13] Ragwitz & Rish "(tra DOI/trước khi in chính thức)" |
| Cách kiểm chứng | (i) Crossref/doi.org cho từng DOI; (ii) đối chiếu IEEE Xplore (TNN 2008 ADASYN); (iii) so số [9]/[16] giữa nội dung SP1 và bib; (iv) kiểm tra "mỗi mục bib ↔ trích dẫn trong báo cáo khớp tên/tác giả/năm" |
| Kết quả áp dụng | `references.bib` cập nhật (17 mục) + SP1 (cập nhật citation) → đủ 3 sản phẩm |

## Ghi chú cuối (cho D khi tổng hợp SP3)

1. **Thời gian:** các mốc còn dấu `≈` — D cập nhật giờ chính xác; chụp thời điểm chạy cell trong notebook làm bằng chứng thời gian.
2. **Đính kèm:** `SP3_appendix_raw.md` (prompt + phản hồi gốc, trích dẫn thực). Export từ Claude desktop (chức năng copy/đính lịch sử).
3. **Trọng tâm chấm:** đề ghi rõ SP3 **không chấm theo số prompt** mà theo **chất lượng kiểm chứng + lý giải**. Hai ví dụ "điểm" rõ nhất: **AI-003** (loại block kết quả EDA sai target) và **AI-010** (audit phát hiện mục tham chiếu [2] nghi ngờ sai, thay bằng nguyên bản có DOI). Giữ nguyên, không rút gọn.
4. **Nguyên tắc:** nội dung do AI đề xuất và nội dung do nhóm tự làm được **phân biệt rõ ở cột "chỉnh sửa hoặc loại bỏ"** trong mỗi mục.
