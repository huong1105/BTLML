# Phần việc TV1: dữ liệu, EDA và features dùng chung

TV1 đã hoàn thiện bảng nền, EDA và bộ features lịch sử theo kế hoạch nhóm. Không có XGBoost, LightGBM hoặc GRU được huấn luyện trong phần việc này.

## Phạm vi và đầu ra

| Phần việc | Đầu ra |
|---|---|
| Kiểm tra 7 CSV, giữ ID/nhãn, phân biệt số 0/ngày thiếu | `reports/data_quality.md`, `.json` và `reports/data_manifest.json` |
| Ghép cửa hàng, lịch và ngày lễ theo đúng phạm vi | `data/processed/base_panel.parquet`: 3.029.400 dòng, 29 cột với bộ nguồn hiện tại |
| Phân tích khám phá dữ liệu | `notebooks/01_eda.ipynb`, 6 ảnh trong `outputs/figures/`, `reports/eda_report.md` và `eda_summary.json` |
| Tạo features lịch sử riêng từng cửa hàng–nhóm hàng | `src/store_sales/features.py`, `scripts/prepare_features.py` |
| Bàn giao dữ liệu học, điều chỉnh, kiểm tra cuối và test | Ba bộ trong `data/processed/features/`: `tuning`, `validation`, `test` |
| Kiểm tra tính đúng và không dùng nhãn tương lai | `tests/test_features.py`; bộ test baseline/metric giữ trong `tests/test_baseline.py` |
| Mô tả cột, mốc và nguồn dữ liệu | [Từ điển dữ liệu](data_dictionary.md), [hợp đồng features](features.md), `reports/features_manifest.json`, `reports/features_report.md` |

## Features đã bổ sung

- Sales tại mốc dự báo và trước mốc 7/14/28 ngày lịch.
- Trung bình và độ lệch chuẩn sales 7/14/28 ngày, gồm cả ngày mốc; std dùng mẫu số N.
- Tỷ lệ số 0 trong các ngày có quan sát ở cửa sổ 28 ngày.
- Số ngày quan sát ở từng cửa sổ và cờ thiếu ở từng ngày lag.
- `origin` xác định giới hạn lịch sử, `horizon` xác định số ngày dự báo trước. Đầu vào ngày đích giữ thông tin cửa hàng/lịch/ngày lễ/khuyến mãi được cung cấp.

Cửa sổ không tự điền số 0 vào ngày thiếu. Lịch sử cố định trong cả khoảng dự báo 16 ngày, không cập nhật bằng sales thật của khoảng này. Những nhãn giữ lại để chấm không nằm trong bảng features kiểm tra/test.

## Bàn giao cho từng người

- **TV2 — XGBoost và LightGBM:** dùng train và features của cùng một bộ; nhận danh sách `feature_columns` trong manifest. TV2 tự chọn/mã hóa đầu vào, xử lý NaN và học model. Đáp án train giữ ở thang sales gốc.
- **TV3 — GRU:** nhận bảng nền và quy tắc nhãn/ngày thiếu. TV3 tự tạo chuỗi 56 ngày, mã hóa/scaling và kiến trúc GRU.
- **TV4 — đánh giá/XAI:** nhận các mốc, ID và labels của hai khoảng kiểm tra, baseline/metric đã có; TV4 tiếp quản chấm điểm, baseline bổ sung và giải thích model.

Baseline trung bình cùng thứ 28 ngày đã có điểm RMSLE 0,531130 ở khoảng 31/07–15/08/2017. Đây là kết quả tham khảo của TV4, không thay cho bộ features lịch sử hoặc kết quả model.

## Tạo lại sau khi clone

CSV/Parquet không đưa lên Git. Mỗi người đặt cùng 7 CSV vào `data/raw/`, cài môi trường theo [README](../README.md), rồi từ gốc repo chạy:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/prepare_features.py
```

Chạy lại EDA/baseline là tùy chọn khi cần tái tạo các kết quả tương ứng. Bảng nền vẫn không chứa lag; các features mới nằm riêng trong thư mục features, vì cùng một ngày đích có thể được dự báo từ nhiều mốc.

## Giới hạn trách nhiệm TV1

TV1 chịu trách nhiệm dữ liệu/features chung và kiểm tra tính đúng của chúng. Việc chọn bộ features cuối cùng, imputation/encoding/scaling của model, huấn luyện, điều chỉnh tham số thuộc TV2/TV3. Chấm điểm tổng hợp và XAI thuộc TV4. Oil và transactions được kiểm tra chất lượng nhưng chưa dùng trong bản so sánh đầu tiên.
