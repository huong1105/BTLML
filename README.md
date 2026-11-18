# Store Sales

Phần **dữ liệu, EDA và features lịch sử của TV1 đã chạy trên dữ liệu thật**. Có bảng nền, báo cáo chất lượng, 6 biểu đồ, notebook đã chạy và các bộ dữ liệu học/dự báo theo mốc cho TV2. TV3 dùng bảng nền để tạo chuỗi GRU; TV4 tiếp quản đánh giá và XAI.

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
│   ├── prepare_features.py   # Tạo dữ liệu học/validation/test và features lịch sử
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

Repo không chứa CSV gốc, bảng Parquet hoặc dự báo validation từng dòng. Sau khi clone, các thư mục dữ liệu chỉ có tệp `.gitkeep`. Mỗi thành viên lấy cùng bộ 7 CSV nguồn rồi chạy bước chuẩn bị để tạo `data/processed/base_panel.parquet` trên máy cục bộ. Script không tự tải dữ liệu.

### 1. Clone mã và cài môi trường

Dùng Python 3.12. Các lệnh dưới đây dành cho PowerShell trên Windows:

```powershell
git clone https://github.com/huong1105/BTLML.git store-sales-forecasting
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

### 5. Tạo bộ features lịch sử cho TV2

Sau khi có bảng nền, chạy:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_features.py
```

Lệnh tạo ba bộ trong `data/processed/features/`:

| Bộ | Dữ liệu được biết tới | Khoảng cần dự báo |
|---|---|---|
| `tuning/` | 14/07/2017 | 15/07–30/07/2017 |
| `validation/` | 30/07/2017 | 31/07–15/08/2017 |
| `test/` | 15/08/2017 | 16/08–31/08/2017 |

Mỗi bộ có `train.parquet` chứa mẫu học và đáp án `sales` ở thang gốc, cùng `features.parquet` chứa 28.512 dòng cần dự báo. Hai bộ kiểm tra có thêm `labels.parquet`; bộ test không có nhãn. **Dùng train của chính bộ đang đánh giá** để không học nhầm đáp án của giai đoạn sau.

Mỗi dòng features gồm metadata `id,date,origin` và danh sách đầu vào trong `reports/features_manifest.json`. Bộ đầu vào giữ các đặc trưng nền, thêm `horizon`, 11 features lịch sử sales và 7 cột độ phủ/cờ thiếu. Lịch sử được chốt tại origin, không thay đổi theo sales thật trong 16 ngày dự báo. Báo cáo số dòng thực tế nằm ở `reports/features_report.md`.

Mẫu học mặc định lấy mốc cách nhau 16 ngày và có ít nhất 28 ngày lịch sử; toàn bộ khoảng đáp án của mỗi mẫu không vượt cutoff của từng bộ. Các khoảng đích thiếu ngày bị bỏ, không điền nhãn giả. Xem đầy đủ tên cột, quy tắc ngày thiếu và cách bàn giao trong [tài liệu features](docs/features.md).

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

Kiểm tra features lịch sử, giới hạn thời gian, tệp bàn giao, baseline và metric bằng `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`; không cần dữ liệu Kaggle cho các kiểm thử này.

Kết quả hiện có:

- [Notebook EDA đã chạy](notebooks/01_eda.ipynb): bảng thống kê, 6 hình và nhận xét.
- [Báo cáo chất lượng](reports/data_quality.md): thiếu, trùng, ID, lịch ngày và độ phủ bảng phụ.
- [Báo cáo EDA](reports/eda_report.md): các phát hiện và giới hạn diễn giải.
- `data/processed/base_panel.parquet`: 3.029.400 dòng train/test, giữ nguyên nhãn train và ID.
- `reports/data_manifest.json`: checksum nguồn/đầu ra, schema và môi trường; nguồn gốc đã được kiểm tra không đổi.
- `data/processed/features/`: ba bộ dữ liệu học/dự báo có features lịch sử; sinh cục bộ, không commit.
- [Báo cáo features](reports/features_report.md) và `reports/features_manifest.json`: số dòng, mốc, schema, độ phủ và checksum đầu ra.

## Nội dung TV1 bàn giao

- Dữ liệu đã kiểm tra và chuẩn hóa, giữ nguyên khóa/id.
- Notebook EDA có biểu đồ và nhận xét dựa trên toàn bộ train.
- Features lịch sử đúng từng cửa hàng–nhóm hàng, dữ liệu học/kiểm tra/test theo mốc và kiểm thử không dùng nhãn tương lai.
- Mô tả cột, quy tắc giữ nhãn thiếu/số 0 và ghép ngày lễ đúng phạm vi.
- Cách đọc dữ liệu để các thành viên khác dùng tiếp. Không áp đặt mô hình hoặc cách kiểm định của họ.

Xem chi tiết ngắn gọn trong [docs/TV1.md](docs/TV1.md).

Giữ CSV gốc và Parquet ngoài Git theo `.gitignore`, kể cả các bản CSV sao chép sang vị trí khác trong dự án. Có thể commit notebook, mã, báo cáo tổng hợp và hình EDA. Người nhận tạo bảng nền bằng `prepare_data.py`, rồi tạo features lịch sử bằng `prepare_features.py`; không cần chạy EDA hoặc baseline để có các bảng này. Bảng nền vẫn giữ 29 cột; features lịch sử nằm riêng. Oil/transactions chưa dùng trong bản đầu; scaling/encoding/huấn luyện do chủ model phụ trách.

Baseline dùng cùng raw train đã kiểm tra, không sửa bảng nền. Các mô hình khác vẫn tự quyết định scaling/encoding và đặc trưng lịch sử, chỉ fit bước học thống kê trên train của từng lần kiểm định. So sánh với baseline trên cùng origin, ngày/ID validation và RMSLE. Không kết luận mô hình nào tốt hơn nếu chưa có điểm thật của mô hình đó.
