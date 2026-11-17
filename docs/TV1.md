# Phần việc TV1: dữ liệu và EDA

**Đã hoàn thành trên 7 CSV thật.** Notebook đã chạy, bảng nền đã tạo và các báo cáo/biểu đồ đã có trong dự án. Đây là phần dữ liệu của TV1, không chốt mô hình hoặc cách kiểm định cho các thành viên khác.

**Bổ sung baseline để bàn giao TV4:** một mốc tham khảo giữ 16 ngày cuối train, trung bình cộng sales cùng thứ trong 28 ngày trước và gồm origin. Nhóm có thể thay mốc bằng tham số; các phương pháp cần đối chiếu trên cùng khoảng/ID và hàm RMSLE.

**1. Kiểm tra dữ liệu.** Đọc 7 CSV; ghi kích thước, kiểu cột, khoảng ngày, thiếu/trùng và số cửa hàng/nhóm hàng. Phân biệt số 0 thật với ngày không có quan sát.

**2. Chuẩn hóa dữ liệu.** Ghép thông tin cửa hàng, lịch và ngày lễ. Kiểm tra phạm vi địa phương, ngày lễ chuyển và trường hợp nhiều sự kiện trong một ngày; số dòng bán hàng không được tăng sau merge. Không sửa CSV gốc hay điền nhãn sales thiếu thành 0.

**3. Làm EDA.** Xem xu hướng theo ngày, mùa vụ theo thứ, phân bố lượng bán/số 0, khác biệt giữa nhóm hàng/cửa hàng và khuyến mãi. Mỗi biểu đồ có nhận xét và đơn vị rõ ràng.

**4. Bàn giao.** Gửi notebook, hình cần dùng, báo cáo chất lượng, mô tả cột và cách tạo bảng dữ liệu. Các thành viên khác tự xây đặc trưng lịch sử, chọn mô hình và thống nhất kiểm định.

## Các tệp đã tạo và công dụng

| Đầu ra | Vị trí |
|---|---|
| Bảng dữ liệu nền, mỗi thành viên tự tạo sau khi clone | `data/processed/base_panel.parquet` (sinh cục bộ, không commit) |
| Kiểm tra thiếu/trùng, ID, lưới chuỗi/ngày và độ phủ bảng phụ | `reports/data_quality.json`, `reports/data_quality.md` |
| Checksum 7 nguồn và bảng nền, schema, môi trường, xác nhận nguồn không đổi | `reports/data_manifest.json` |
| 6 ảnh để thảo luận và đưa vào phần slide dữ liệu | `outputs/figures/` |
| Thống kê, biểu đồ và nhận xét có kết quả sẵn | `notebooks/01_eda.ipynb` |
| Nhận xét tiếng Việt và thống kê máy đọc được | `reports/eda_report.md`, `reports/eda_summary.json` |
| Mốc so sánh dự báo có điểm RMSLE thực tế | `reports/baseline_report.md`, `reports/baseline_metrics.json` |
| Dự báo từng dòng validation để TV4 đối chiếu | `outputs/baseline/validation_predictions.csv` (sinh cục bộ, không commit) |
| Notebook baseline đã có kết quả | `notebooks/02_baseline.ipynb` |

## Mã nguồn và tài liệu dùng để làm gì

| Tệp | Công dụng |
|---|---|
| `scripts/prepare_data.py` | Chạy toàn bộ kiểm tra, ghép bảng nền, lưu Parquet và manifest |
| `scripts/run_eda.py` | Tính thống kê, tạo 6 hình và viết báo cáo EDA |
| `src/store_sales/data.py` | Đọc 7 CSV, kiểm tra chất lượng, giữ ID/nhãn và ghép dữ liệu |
| `src/store_sales/calendar.py` | Đặc trưng lịch và ngày lễ theo phạm vi từng cửa hàng |
| `src/store_sales/eda.py` | Các phép tổng hợp, vẽ hình và nhận xét dùng chung giữa script/notebook |
| `scripts/run_baseline.py`, `src/store_sales/baseline.py` | Tạo baseline cùng thứ và lưu dự báo/điểm, không dùng sales thật trong tương lai |
| `src/store_sales/metrics.py` | Hàm RMSLE và đối chiếu đúng ID dùng chung cho các thành viên |
| `tests/test_baseline.py` | Kiểm tra đáp án tính tay, số 0/fallback, không rò rỉ nhãn và ghép ID |
| `docs/data_dictionary.md` | Tra ý nghĩa cột nguồn và cột của bảng nền |
| `README.md`, `requirements.txt` | Cách cài thư viện, chạy lại và tự đẩy repo |

Các tệp `.gitkeep` giữ những thư mục dữ liệu trống khi đưa mã lên Git.

## Kết quả chính

- Đã kiểm tra 3.000.888 dòng train và 28.512 dòng test: không trùng khóa/id, ID test khớp sample_submission.
- Bảng nền 3.029.400 dòng giữ nguyên số dòng và nhãn train; test không bị điền nhãn giả.
- Giữ 939.130 số 0 thật và 462.113 nhãn có phần thập phân; không tự xóa giá trị lớn.
- Phát hiện thiếu ngày 25/12/2013-2016; oil thiếu giá và transactions không có cho test. Quy tắc bàn giao nêu rõ trong báo cáo.
- Ghép stores/lịch/lễ mà không nhân bản dòng; giữ riêng phạm vi và trạng thái chuyển lễ.
- EDA có xu hướng ngày, thứ trong tuần, phân bố, khác biệt nhóm hàng, khuyến mãi và ví dụ chuỗi. Liên hệ với khuyến mãi không được diễn giải thành tác động nhân quả.

Các số liệu trên mô tả đúng dữ liệu hiện tại; báo cáo/script tính lại các số khi dữ liệu thay đổi.

## Chạy lại

Từ thư mục repo và môi trường đã cài `requirements.txt`, các lệnh đọc trực tiếp 7 CSV có sẵn trong `data/raw/`:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/run_eda.py
.\.venv\Scripts\python.exe scripts/run_baseline.py
```

Khi bàn giao qua Git, không commit CSV gốc hoặc Parquet. Thành viên khác clone mã, cài môi trường, đặt cùng bộ 7 CSV vào `data/raw/`, rồi chạy `scripts/prepare_data.py` để tự tạo bảng nền. Hướng dẫn đầy đủ, danh sách tệp và cách kiểm tra đầu ra nằm trong [README](../README.md).

Chỉ bước chuẩn bị dữ liệu là cần thiết để có Parquet; EDA và baseline là các bước tùy chọn khi cần tái tạo kết quả tương ứng. Notebook cũng chạy lại được sau khi chuẩn bị dữ liệu. Mã và notebook dùng đường dẫn tương đối theo repo.

## Cách dùng bảng dữ liệu

`load_raw_data(raw_dir)` trả về các bảng CSV theo tên không có đuôi. `quality_report(tables)` đo chất lượng dữ liệu. `build_base_panel(tables)` ghép bảng nền; mã nằm trong `src/store_sales/data.py`.

Bảng nền giữ khóa `(date, store_nbr, family)` và `id` gốc, thêm thông tin cửa hàng, lịch và cờ ngày lễ. `sales_observed` cho biết có nhãn thật; số 0 ở train vẫn là quan sát. Nhãn test giữ thiếu, `is_test = True`. Chưa tạo lag hay cửa sổ chuỗi và chưa chọn mô hình.

Dữ liệu nền chưa tự thêm ngày bán hàng không có trong nguồn. Khi thành viên khác cần chuỗi liên tục, cần xử lý lịch ngày và cờ thiếu phù hợp. Thống kê từ sales/giao dịch lịch sử chỉ được dùng khi đã biết tại mốc dự báo.

Scaling, mã hóa đầu vào, biến đổi target và đặc trưng lịch sử cho mô hình vẫn do người xây mô hình lựa chọn; không có một bản scaled bắt buộc dùng chung. Baseline tham khảo dùng sales gốc và không cần scaler. Điểm baseline chỉ mô tả một holdout, chưa thay việc đánh giá các mô hình ở nhiều giai đoạn và chưa phải điểm Kaggle.
