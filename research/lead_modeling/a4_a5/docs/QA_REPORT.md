# Quality Assurance — A4/A5 V2 (17/11/2026)

> Ghi chép QA lịch sử, không phải bằng chứng runtime của phiên bản hiện tại. Đợt audit/fix 18/11/2026 dùng môi trường khác và không có SHAP/LIME; chưa xác minh lại claim SHAP phía dưới. Xem [CODEX_FIX_REPORT.md](CODEX_FIX_REPORT.md) cho kiểm thử hiện tại và [METHODOLOGY_NOTES.md](METHODOLOGY_NOTES.md) cho phương pháp thực sự chạy.

**Phạm vi QA:** tính đúng luồng code, không đánh giá hiệu năng mô hình trên BRFSS 2015.

- Môi trường kiểm tra: Python 3.11, scikit-learn 1.8, XGBoost 3.1.3, SHAP 0.50 (kiểm tra bằng `versions.json` tại từng run thực tế).
- Bộ `tests/test_protocol.py`: **8/8 pass** (schema, split 70/15/15, threshold F1, metric, paired bootstrap, fold weight, chặn file không đúng, SHA).
- Chạy A4 bằng **CSV tổng hợp**, có cờ `--demo`: thành công, manifest gắn `SYNTHETIC_QA_ONLY`.
- Chạy A5 trên kết quả demo với `--allow-smoke`: thành công, biểu đồ PR/ROC, confusion matrix, bootstrap và phân tích nhóm xuất được.
- A5 bật `--shap`: thành công trên dữ liệu tổng hợp; xuất SHAP beeswarm/bar/waterfall.
- **LIME chưa được kiểm thử end-to-end:** môi trường hiện không cài được gói `lime` do không truy cập index package. Có mã tích hợp theo API LIME; khi chạy trên máy/Kaggle cần cài `requirements-explain.txt` và kiểm tra đầu ra.
- Hai file Word A4/A5 đã render sang PDF, kiểm tra từng trang, không thấy phần chữ bị cắt hay tràn.

**Chưa làm:** tải CSV Kaggle thật, chạy 10 trial x 3-fold trên dữ liệu thật, kiểm tra độ tin cậy, phản biện kết quả, review/merge repository.

**Không chuyển số liệu synthetic vào bài báo cáo SP2.**
