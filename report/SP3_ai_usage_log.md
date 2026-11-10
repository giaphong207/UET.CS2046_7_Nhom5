# SP3 – Nhật ký sử dụng AI (AI Usage Log)


 — Kế hoạch B1/B2/B3 cho vai trò B

| Trường | Nội dung |
|---|---|
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Có kế hoạch từng bước cho task B (Data & Pipeline, 25%): B1 EDA, B2 pipeline stub, B3 tiền xử lý + mất cân bằng |
| Nhận xét, đánh giá | Kế hoạch hợp lý, khớp phân công 25%. **Hạn chế:** AI không thấy dữ liệu thật → bỏ sót đặc thù BRFSS (mã sentinel 777/888/999, label noise) |
| Nội dung được sử dụng | Thứ tự B1→B2→B3, tiêu chí "chấm xong" từng task (stub chạy <5 phút; cross-tab mất cân bằng vào SP1; B3 chọn theo recall lớp thiểu) |
| Nội dung chỉnh sửa hoặc loại bỏ | **Thêm (do nhóm):** bước kiểm label-noise và rà mã sentinel BRFSS — AI bỏ sót, căn cứ: tài liệu khảo sát BRFSS + EDA thật |
| Cách kiểm chứng | Chạy EDA thật trên 3 CSV: phát hiện label noise 2,41% (không có sentinel → dataset đã sạch) → xác nhận bổ sung là cần thiết |
| Kết quả áp dụng | Checklist W1; tiêu chí "done" dùng lại cho B2/B3 |


 Phát hiện & sửa bug target sai (kết quả EDA lần 1 bị vô hiệu)

| Trường | Nội dung |
|---|---|
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Phát hiện lỗi: EDA chạy thật, tệp 3-lớp bị phân tích theo sai cột target |
| Nhận xét, đánh giá | **Lỗi nghiêm trọng nhất của B1:** AI giả định tên cột theo BRFSS gốc. Block kết quả tệp 3-lớp lần 1 (target=Income, "Imbalance 9,2×", corr Education 0,45) **vô giá trị, phải loại toàn bộ**. Nhóm phát hiện bằng việc đọc kỹ log (target phải là cột diabetes) |
| Nội dung được sử dụng | `find_target()` v2 + label-noise bằng groupby |
| Nội dung chỉnh sửa hoặc loại bỏ | **LOẠI BỎ:** toàn bộ block tệp 3-lớp của lần chạy v1. **Thay bằng:** kết quả chạy lại với `Diabetes_012` (84,2/1,8/13,9%, ratio 46,1×, top feature GenHlth r=0,30) |
| Cách kiểm chứng | 3 lớp: (1) đối chiếu 22 cột thật in ra log — không có cột `Diabetes`; (2) log v2 hiển thị `[TARGET] ... dung cot 'Diabetes_012'`/`'Diabetes_binary'`; (3) số ca dương 35.346 nhất quán giữa 3 tệp (012 = unbalanced = 2× tệp 5050) |
| Kết quả áp dụng | EDA được hiệu đính; số chuẩn: 46,1× / dup 9,4–9,5% / noise 2,06–2,41% / top feature GenHlth (lift 2,72×) |
 Diễn giải EDA, chốt bài toán & chiến lược B3

| Trường | Nội dung |
|---|---|
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Từ số EDA thật: (1) chốt bài toán chính/phụ; (2) thứ tự tiền xử lý B3; (3) tiêu chí chọn strategy |
| Nhận xét, đánh giá | Logic đúng với số thật. **Lưu ý:** giai đoạn đầu AI từng *giả định* phân bố 3-lớp ~40/20/40 (sai so với thật 84,2/1,8/13,9); may là quyết định cuối được rà lại với số thật trước khi chốt |
| Nội dung được sử dụng | 3 kết luận trên; mọi con số trong báo cáo lấy **trực tiếp từ output script**, không lấy từ phản hồi AI |
| Nội dung chỉnh sửa hoặc loại bỏ | Loại giả định 40/20/40; mỗi số đưa vào báo cáo đều có "cơ sở" (file CSV + lệnh tái lập) |
| Cách kiểm chứng | Chạy lại cùng seed → số trùng; đối chiếu chéo 3 tệp (35.346 ca dương); đối chiếu mô tả Kaggle (253.680 dòng, 21 features) |
| Kết quả áp dụng | Chốt hướng bài toán tại họp W1; B3 triển khai đúng thứ tự + tiêu chí |

 Pipeline B2/B3 theo protocol đánh giá

| Trường | Nội dung |
|---|---|
| Công cụ AI | Claude (Claude desktop) |
| Mục đích | Pipeline reproducible: dedup + noise + split stratified 70/15/15 seed 42 + 5 strategy mất cân bằng + đánh giá theo đúng 4 nguyên tắc nhóm chốt (AP là chính; ngưỡng 0,5 vs max-F1 trên val khóa trước test; ablation đổi 1 yếu tố; paired bootstrap 1.000 lần) |
| Nhận xét, đánh giá | Thiết kế đúng protocol. **Phát hiện khi test thật (chạy của nhóm):** sau majority-vote, các dòng trong cùng nhóm đã thành trùng nhau → cần **dedup lần 2** (AI ban đầu chỉ dedup trước noise). Nhóm đã vá: `n_collapsed_after_noise` (gộp 1.566 dòng ở bản chạy thật) |
| Nội dung được sử dụng | Toàn bộ luồng `prepare → split → ablation CV → chọn strategy → final 2 ngưỡng → paired bootstrap → nghiệm phụ 3-lớp` |
| Nội dung chỉnh sửa hoặc loại bỏ | **Thêm (do nhóm):** dedup-lần-2 sau khi gán nhãn majority-vote; lưu `idx_*.npy` tách 3 file (không gộp 1 mảng) |
| Cách kiểm chứng | Unit-test thuần numpy (majority-vote loại sạch nhóm nhiễu; SMOTE/ADASYN cân bằng 1:1); crash-test `main()` trên data mô phỏng trước khi chạy thật |
| Kết quả áp dụng | `code/b2_b3_pipeline.py` — mã nguồn B2/B3 trong repo |


