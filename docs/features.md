# Bàn giao features lịch sử của TV1

TV1 giữ `base_panel.parquet` làm bảng nền cho cả nhóm và sinh thêm các bộ dữ liệu dạng bảng trong `data/processed/features/`. TV2 dùng các bộ này cho XGBoost/LightGBM. TV3 tiếp tục lấy bảng nền để tạo chuỗi GRU; TV4 dùng nhãn kiểm tra và ID để chấm. CSV/Parquet được tạo cục bộ, không commit.

## Tái tạo dữ liệu

Sau khi đã đặt đủ 7 CSV và tạo bảng nền theo README, chạy từ gốc repo:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_features.py
```

Mã chính: `src/store_sales/features.py`. Có thể dùng `HistoryFeatureBuilder(panel).snapshot(origin)` để lấy một dòng lịch sử cho mỗi cửa hàng–nhóm hàng, hoặc `.build(origin)` để ghép với các dòng đích trong 16 ngày tiếp theo. `.build` không trả về sales/nhãn của ngày đích. `.labels` lấy đáp án riêng khi khoảng đích có nhãn thật.

## Các bộ dữ liệu được tạo

| Thư mục | Sales được biết đến | `features.parquet` dự báo | Các tệp |
|---|---|---|---|
| `tuning/` | 14/07/2017 | 15/07–30/07 | `train.parquet`, `features.parquet`, `labels.parquet` |
| `validation/` | 30/07/2017 | 31/07–15/08 | `train.parquet`, `features.parquet`, `labels.parquet` |
| `test/` | 15/08/2017 | 16/08–31/08 | `train.parquet`, `features.parquet`; không có nhãn test |

- `train.parquet`: mẫu học từ các mốc lịch sử trước đó, có cột đáp án `sales` ở thang gốc. Toàn bộ 16 ngày đáp án của mỗi mẫu đều không vượt ngày cutoff của bộ tương ứng.
- `features.parquet`: 28.512 dòng cần dự báo ở mốc hiện tại; không có `sales`, `is_test`, `sales_observed`. `labels.parquet` của hai bộ kiểm tra có `id,sales` để đối chiếu riêng.
- Mỗi bộ có lịch sử học riêng. Dùng **train của chính bộ đó**; không dùng train của `test/` để huấn luyện model đang chấm ở `tuning/` hoặc `validation/`.
- Model được chọn/điều chỉnh trên `tuning`; sau khi cố định cấu hình mới dùng `validation`. `test` phục vụ huấn luyện lại và xuất dự báo Kaggle; không có đáp án test để chấm nội bộ.

## Ý nghĩa và giới hạn thời gian

`origin` là ngày cuối được biết sales. `date` là ngày đích, `horizon` là số ngày từ origin đến date (1–16). Mọi thống kê lịch sử được giữ nguyên trong cả 16 ngày của cùng chuỗi/mốc. Lịch, ngày lễ và khuyến mãi được lấy theo ngày đích vì là thông tin được cung cấp cho bài toán.

Ví dụ origin 30/07: `sales_origin_lag_7` lấy 23/07, không phải lấy 7 ngày trước từng ngày đích. `sales_mean_28` dùng 03/07–30/07, không sử dụng nhãn tháng 8.

| Cột | Quy tắc |
|---|---|
| `sales_at_origin` | Sales đúng ngày origin |
| `sales_origin_lag_7/14/28` | Sales đúng ngày origin trừ 7/14/28 ngày lịch |
| `sales_mean_7/14/28` | Trung bình trong 7/14/28 ngày lịch, bao gồm origin |
| `sales_std_7/14/28` | Độ lệch chuẩn chia số ngày quan sát, `ddof=0`; một quan sát có std=0 |
| `sales_zero_rate_28` | Số ngày quan sát có sales=0 chia số ngày có quan sát trong cửa sổ 28 ngày |
| `sales_observed_days_7/14/28` | Số ngày có nhãn thật trong từng cửa sổ, không phải số ngày sales dương |
| `sales_at_origin_missing` | Có thiếu nhãn ở đúng ngày origin không |
| `sales_origin_lag_7/14/28_missing` | Có thiếu nhãn ở đúng ngày lag tương ứng không |

Có **11 features lịch sử về sales** và **7 cột về độ phủ/thiếu lịch sử**, tổng cộng 18 cột mới. Tất cả tính riêng từng cặp `store_nbr,family`.

Ngày không có trong nguồn được giữ là thiếu: lag chính xác nhận NaN; thống kê chỉ tính trên ngày đã quan sát. Không điền số 0 cho ngày thiếu. Khi không có lịch sử trong cửa sổ, mean/std/zero rate nhận NaN và count=0. Bước imputation/encoding/scaling do chủ model lựa chọn và chỉ fit trên train của từng bộ.

## Các mẫu học lịch sử

Mặc định lấy các mốc học cách nhau **16 ngày**, để các khoảng đáp án không chồng nhau và giữ kích thước dữ liệu vừa phải. Mốc sớm nhất có ít nhất 28 ngày lịch sử kể từ đầu nguồn. Mốc cuối của từng bộ bằng cutoff trừ 16 ngày. Mỗi mốc tạo một khoảng dự báo đủ 1.782 chuỗi × 16 ngày.

Khoảng đích nào thiếu một ngày/chuỗi hoặc thiếu nhãn thì không được tự điền để học. Các khoảng thiếu ngày lịch trong nguồn được bỏ toàn bộ và ghi vào manifest. Ngày thiếu trong **lịch sử** vẫn được phép tồn tại, thể hiện qua NaN/count/mask. Có thể thay mật độ mốc bằng `--train-step-days`; bước nhỏ hơn tạo nhiều mẫu hơn và tăng dung lượng.

Khóa của một mẫu là **`(origin, id)`**, vì cùng một ID có thể được dự báo từ nhiều mốc nếu đổi bước lấy mẫu. Không tự loại trùng chỉ bằng ID khi dùng nhiều mốc.

## Cột đầu vào, nhãn và kiểm tra bàn giao

- `reports/features_manifest.json` liệt kê chính xác `feature_columns`. Nhóm này gồm thông tin cửa hàng/lịch/ngày lễ/khuyến mãi, `horizon` và 18 cột lịch sử/độ phủ; không gồm `id,date,origin` hoặc đáp án `sales`.
- TV2 chọn đầu vào từ danh sách đó. Đáp án học lấy từ `train.parquet["sales"]`; nếu học log1p thì TV2 thực hiện biến đổi riêng. Dự báo bàn giao TV4 phải ở thang sales gốc.
- TV3 nhận dữ liệu chuỗi từ bảng nền, tự phụ trách cửa sổ 56 ngày, cờ thiếu và các bước biến đổi riêng. Không có model nào được huấn luyện trong bước tạo features của TV1.
- `reports/features_report.md` ghi số dòng thực tế và số khoảng bỏ qua. Manifest lưu các mốc, schema đầu vào, số dòng, kiểm kê thiếu, checksum nguồn/đầu ra và phiên bản môi trường. Chỉ dùng các đầu ra của lần chạy thành công có `complete=true`; không sử dụng tệp `.partial`.
- `tests/test_features.py` kiểm tra đáp án tính tay, cửa sổ ngày lịch, tách chuỗi, số 0, lịch sử thiếu, giữ ID, đổi nhãn tương lai không đổi features, giới hạn nhãn học theo cutoff và các tệp bàn giao.

Chạy kiểm tra:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
