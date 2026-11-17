# Store Sales - Phần việc TV1

Phần **dữ liệu và EDA của TV1 đã chạy trên 7 CSV thật**. Có bảng dữ liệu nền, báo cáo chất lượng, 6 biểu đồ và notebook đã chạy có kết quả. Những thành viên khác tự chọn mô hình và bổ sung phần việc của mình sau.

Đã bổ sung **baseline tham khảo**: trung bình số học sales cùng thứ trong 28 ngày lịch sử, dự báo cố định 16 ngày. Kết quả thật ở [báo cáo baseline](reports/baseline_report.md); đây là phép kiểm định nội bộ, không phải điểm Kaggle. Nhóm có thể thống nhất lại mốc trước khi đối chiếu các phương pháp.

```text
store-sales-forecasting/
├── data/
│   ├── raw/                  # 7 CSV gốc, không đưa lên Git
│   └── processed/            # Dữ liệu sau xử lý
├── notebooks/
│   ├── 01_eda.ipynb          # Phân tích, biểu đồ và nhận xét của TV1
│   └── 02_baseline.ipynb     # Baseline và điểm đối sánh
├── src/store_sales/          # Dữ liệu, lịch, baseline và metric
├── scripts/
│   ├── prepare_data.py       # Tạo bảng chuẩn và báo cáo chất lượng
│   ├── run_eda.py            # Tạo biểu đồ và nhận xét từ dữ liệu thật
│   └── run_baseline.py       # Chạy dự báo tham khảo và tính RMSLE
├── docs/
│   ├── data_dictionary.md    # Ý nghĩa các cột
│   └── TV1.md                # Công việc và nội dung bàn giao
├── outputs/figures/          # Biểu đồ EDA
├── outputs/baseline/         # Dự báo validation, không đưa lên Git
├── reports/                  # Chất lượng, EDA, baseline và manifest
├── requirements.txt
└── .gitignore
```

## Thành viên khác: tạo Parquet sau khi clone

Repo không chứa CSV gốc, bảng Parquet hoặc dự báo validation từng dòng. Sau khi clone, các thư mục dữ liệu chỉ có tệp `.gitkeep`. Mỗi thành viên lấy cùng bộ 7 CSV nguồn rồi chạy bước chuẩn bị để tự tạo `data/processed/base_panel.parquet` trên máy mình. Script không tự tải dữ liệu.

### 1. Clone mã và cài môi trường

Dùng Python 3.12. Các lệnh dưới đây dành cho PowerShell trên Windows; thay URL mẫu bằng địa chỉ repo thật của nhóm:

```powershell
git clone https://github.com/USERNAME/TEN-REPO.git store-sales-forecasting
cd store-sales-forecasting
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu đã clone và đã cài môi trường, bỏ qua bước này. Không cần chạy `Activate.ps1` vì các lệnh gọi trực tiếp Python trong `.venv`.

### 2. Đặt đủ 7 CSV nguồn

Lấy cùng bộ CSV mà nhóm sử dụng; nguồn ban đầu là [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). Đặt trực tiếp các tệp vào `data/raw/`, giữ nguyên tên và nội dung:

```text
data/raw/
├── train.csv
├── test.csv
├── stores.csv
├── oil.csv
├── transactions.csv
├── holidays_events.csv
└── sample_submission.csv
```

Không đặt thêm một thư mục trung gian bên dưới `data/raw/`. Code không thể tạo lại dữ liệu nguồn nếu chưa có các CSV này.

### 3. Tạo bảng Parquet

Mở terminal tại thư mục gốc repo, nơi có `requirements.txt` và `scripts/`, rồi chạy:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
```

Chỉ lệnh này là đủ để tạo bảng nền. Chương trình đọc và kiểm tra 7 CSV, ghép cửa hàng/lịch/ngày lễ, rồi sinh hoặc cập nhật:

- `data/processed/base_panel.parquet` — bảng nền dùng cho các mô hình.
- `reports/data_quality.md` và `reports/data_quality.json` — kết quả kiểm tra nguồn.
- `reports/data_manifest.json` — checksum nguồn/đầu ra, kiểu cột và môi trường chạy.

Khi thành công, terminal in `Prepared ... observed rows` và vị trí báo cáo. Với đúng bộ CSV hiện được nhóm kiểm tra, bảng có **3.029.400 dòng và 29 cột**. Script giữ nguyên ID, sales train và số 0; sales test vẫn thiếu. Không có bước scaling hoặc tạo lag trong bảng nền.

### 4. Kiểm tra đầu ra và dùng tiếp

Kiểm tra số dòng/cột bằng metadata Parquet, không cần nạp toàn bộ bảng:

```powershell
.\.venv\Scripts\python.exe -c "import pyarrow.parquet as pq; p = pq.ParquetFile('data/processed/base_panel.parquet'); print(p.metadata.num_rows, len(p.schema_arrow))"
```

Với bộ CSV hiện tại, kết quả là `3029400 29`. Khi viết mã mô hình, đọc bảng này bằng `pandas.read_parquet(...)`; không cần tự viết lại phần ghép các CSV. Chỉ các dòng train có nhãn thật, được nhận diện bằng `is_test=False` và `sales_observed=True`. Xem ý nghĩa cột trong [từ điển dữ liệu](docs/data_dictionary.md).

CSV và Parquet sinh tại máy mỗi người tiếp tục được `.gitignore` loại khỏi Git. Việc clone mã không cung cấp sẵn những tệp dữ liệu này.

### Các bước tùy chọn

Chỉ chạy các lệnh sau nếu cần tạo lại biểu đồ/nhận xét EDA hoặc kết quả baseline; chúng không phải điều kiện để có bảng Parquet:

```powershell
.\.venv\Scripts\python.exe scripts/run_eda.py
.\.venv\Scripts\python.exe scripts/run_baseline.py
```

### Khi chạy gặp lỗi

- Không tìm thấy `py` hoặc Python 3.12: cài/chọn trình thông dịch Python 3.12 trước khi tạo `.venv`.
- `FileNotFoundError` trỏ đến CSV: kiểm tra đủ 7 tên tệp và vị trí trực tiếp trong `data/raw/`.
- `ModuleNotFoundError`: cài `requirements.txt` bằng đúng Python trong `.venv` như bước 1.
- Không tìm thấy script hoặc `requirements.txt`: chuyển terminal về thư mục gốc repo.
- Báo trùng khóa, ID, metadata hoặc sales không hợp lệ: kiểm tra lại bộ CSV nguồn; không bỏ qua kiểm tra hoặc tự điền nhãn để chạy tiếp.

Mở `notebooks/01_eda.ipynb` bằng VS Code và chọn Python trong `.venv` để chạy lại. Notebook đã có kết quả lưu sẵn để xem; khi chạy lại sẽ đọc CSV cục bộ. Nếu muốn giao diện JupyterLab, có thể cài thêm `jupyterlab`; các lệnh script trên không cần giao diện notebook.

`notebooks/02_baseline.ipynb` trình bày và chạy baseline. Mặc định giữ 16 ngày cuối train; có thể đổi bằng `.\.venv\Scripts\python.exe scripts/run_baseline.py --validation-start 2017-07-31`. Cửa sổ lịch sử kết thúc trước ngày validation đầu tiên, không cập nhật sales thật trong khoảng dự báo. Baseline không cần scaling.

Để dùng cùng hàm đánh giá, import `evaluate_by_id` từ `store_sales.metrics`: nhãn có cột `id,sales`, dự báo có `id,sales_pred`. Hàm kiểm tra đúng tập ID, đối chiếu theo ID và tính RMSLE; từ chối thiếu/trùng ID, giá trị âm hoặc không hữu hạn. Khi chạy Python từ gốc repo, thêm `src/` vào `sys.path` như các notebook/script đã làm, hoặc đặt `PYTHONPATH=src` phù hợp môi trường.

Kiểm tra quy tắc dự báo và metric bằng `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`; không cần dữ liệu Kaggle cho các kiểm thử này.

Kết quả hiện có:

- [Notebook EDA đã chạy](notebooks/01_eda.ipynb): bảng thống kê, 6 hình và nhận xét.
- [Báo cáo chất lượng](reports/data_quality.md): thiếu, trùng, ID, lịch ngày và độ phủ bảng phụ.
- [Báo cáo EDA](reports/eda_report.md): các phát hiện và giới hạn diễn giải.
- `data/processed/base_panel.parquet`: 3.029.400 dòng train/test, giữ nguyên nhãn train và ID.
- `reports/data_manifest.json`: checksum nguồn/đầu ra, schema và môi trường; nguồn gốc đã được kiểm tra không đổi.

## Nội dung TV1 bàn giao

- Dữ liệu đã kiểm tra và chuẩn hóa, giữ nguyên khóa/id.
- Notebook EDA có biểu đồ và nhận xét dựa trên toàn bộ train.
- Mô tả cột, quy tắc giữ nhãn thiếu/số 0 và ghép ngày lễ đúng phạm vi.
- Cách đọc dữ liệu để các thành viên khác dùng tiếp. Không áp đặt mô hình hoặc cách kiểm định của họ.

Xem chi tiết ngắn gọn trong [docs/TV1.md](docs/TV1.md).

Giữ CSV gốc và Parquet ngoài Git theo `.gitignore`, kể cả các bản CSV sao chép sang vị trí khác trong dự án. Có thể commit notebook, mã, báo cáo tổng hợp và hình EDA. Người nhận repo thực hiện hướng dẫn ở trên để tự tạo bảng nền từ cùng bộ 7 CSV; không cần chạy EDA hoặc baseline nếu chỉ cần Parquet. Bảng nền chưa có lag/scaler hay mô hình; oil và transactions được kiểm tra nhưng chưa tự ghép vào đầu vào dự báo.

Baseline dùng cùng raw train đã kiểm tra, không sửa bảng nền. Các mô hình khác vẫn tự quyết định scaling/encoding và đặc trưng lịch sử, chỉ fit bước học thống kê trên train của từng lần kiểm định. So sánh với baseline trên cùng origin, ngày/ID validation và RMSLE. Không kết luận mô hình nào tốt hơn nếu chưa có điểm thật của mô hình đó.

## Đẩy lên GitHub

Tự tạo repo trống trên GitHub, rồi chạy trong thư mục này:

```powershell
git init -b main
git add .
git commit -m "Add TV1 data and EDA starter"
git remote add origin https://github.com/USERNAME/TEN-REPO.git
git push -u origin main
```

Thay địa chỉ remote bằng repo của bạn. `.gitignore` đã loại dữ liệu gốc, dữ liệu processed và môi trường Python.
#   B T L M L  
 