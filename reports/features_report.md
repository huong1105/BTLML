# Bàn giao features lịch sử TV1

Nguồn: bảng nền Parquet; CSV và nhãn gốc không thay đổi.

Mỗi dòng học/dự báo có khóa (origin, id). Features lịch sử cố định tại origin; horizon là số ngày dự báo trước.
Độ lệch chuẩn chia N (ddof=0); cửa sổ gồm origin. Ngày thiếu giữ NaN; trung bình/tỷ lệ số 0 chỉ tính trên ngày đã quan sát.
Tập features kiểm tra/test không chứa sales, is_test hay sales_observed; đáp án kiểm tra lưu riêng.

| Bộ | Origin | Dòng học | Dòng dự báo | Số mốc học bỏ qua do thiếu ngày đích |
|---|---|---:|---:|---:|
| tuning | 2017-07-14 | 2,765,664 | 28,512 | 4 |
| validation | 2017-07-30 | 2,794,176 | 28,512 | 4 |
| test | 2017-08-15 | 2,822,688 | 28,512 | 4 |

Mốc học cách nhau 16 ngày, chỉ dùng các khoảng có đủ ngày/chuỗi và toàn bộ đáp án không vượt cutoff của từng bộ.
Các cột đầu vào được liệt kê trong features_manifest.json; id/date/origin là metadata, sales là đáp án của train.parquet.
TV2 tự chọn/mã hóa đầu vào; TV3 dùng bảng nền để tạo chuỗi; TV4 quản lý phép chấm và XAI.
