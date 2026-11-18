# Đưa A4/A5 lên branch cá nhân của Nhóm 5

Repository: https://github.com/giaphong207/UET.CS2046_7_Nhom5

```bash
cd ~/Projects/UET.CS2046_7_Nhom5
git fetch origin
git switch feature/lead-modeling-a
# Component đã nằm trong research/lead_modeling/a4_a5/; giữ nguyên cấu trúc nested.
git status
git add research/lead_modeling/a4_a5/
git diff --cached --stat
git commit -m "feat(a4-a5): add reproducible diabetes experiments and analysis"
git push
```

Lệnh Git trên dùng repository root; cài requirements và chạy script dùng working directory `research/lead_modeling/a4_a5/` như README. Kiểm tra nội dung staged và ignore trước khi commit. Đừng push thư mục `results/`, file `data/*.csv`, `.env`, token hoặc các model artifacts không rõ nguồn. Tài liệu này là hướng dẫn thao tác thủ công, không ghi nhận rằng commit/push đã được thực hiện.

Sau khi kết quả thật có, hãy đưa **bảng tổng hợp đã kiểm tra, biểu đồ và báo cáo cuối** lên PR (trừ khi nhóm có nơi lưu khác), đính kèm commit link vào CoReach. Không biến dataset hoặc kết quả synthetic thành "thành tích" của nhóm.
