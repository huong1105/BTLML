# Từ điển dữ liệu

Nguồn: 7 CSV của [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), đặt cục bộ tại `data/raw/`. Các CSV gốc được `.gitignore` loại khỏi Git; không chỉnh sửa chúng trong quá trình chuẩn hóa.

| File | Một dòng biểu diễn | Cột và ý nghĩa |
|---|---|---|
| `train.csv` | Ngày × cửa hàng × nhóm hàng đã quan sát | `id`: mã dòng; `date`: ngày; `store_nbr`: cửa hàng; `family`: nhóm hàng; `sales`: lượng bán; `onpromotion`: số mặt hàng trong nhóm đang khuyến mãi |
| `test.csv` | Ngày × cửa hàng × nhóm hàng cần dự báo | Như train nhưng không có `sales`; phải giữ đúng `id` |
| `sample_submission.csv` | Một dự báo mẫu cho mỗi dòng test | `id`, `sales`; chỉ dùng làm chuẩn định dạng/đối chiếu id |
| `stores.csv` | Một cửa hàng | `store_nbr`, `city`, `state`, `type`, `cluster` |
| `holidays_events.csv` | Một ngày lễ hoặc sự kiện | `date`, `type`, `locale`, `locale_name`, `description`, `transferred` |
| `oil.csv` | Giá dầu được cung cấp cho một ngày | `date`, `dcoilwtico` |
| `transactions.csv` | Số giao dịch của một cửa hàng trong ngày | `date`, `store_nbr`, `transactions` |

`sales` là lượng bán, có thể là số thập phân do hàng bán theo cân. `onpromotion` là số đếm, không phải biến có/không. Khóa bán hàng là `(date, store_nbr, family)`; `id` liên kết đến bản nộp.

**Kết quả kiểm tra.** Số dòng, kiểu dữ liệu, null, trùng, khoảng ngày, mức phủ và các ngày thiếu của đúng phiên bản CSV được tính trong [báo cáo chất lượng](../reports/data_quality.md). Chi tiết máy đọc được ở `reports/data_quality.json`; checksum và môi trường chạy ở `reports/data_manifest.json`. Không dùng các con số dự kiến thay cho kết quả kiểm tra nguồn thật.

- `stores` phải ghép nhiều-một theo `store_nbr`.
- Lễ có thể có nhiều sự kiện trong một ngày. Tổng hợp theo ngày và cửa hàng, xét phạm vi National/Regional/Local trước khi ghép.
- Ngày gốc có `transferred = True` không được tự coi là ngày nghỉ đang áp dụng; tìm sự kiện `Transfer`. `Bridge`, `Additional`, `Work Day` có ý nghĩa riêng.
- Giao dịch thực cùng ngày trong test không có sẵn. Bản đầu chưa dùng `transactions` làm đầu vào trực tiếp.
- Giá dầu có giá trị và ngày thiếu. Nếu bổ sung, phải ghi cách điền và giới hạn thông tin tại origin.

**Bảng nền bàn giao:** `data/processed/base_panel.parquet`, được sinh cục bộ và không đưa lên Git. Sau khi clone, mỗi thành viên đặt đủ 7 CSV nguồn vào `data/raw/` và chạy `scripts/prepare_data.py` theo [hướng dẫn trong README](../README.md). Một dòng vẫn là một ngày-cửa hàng-nhóm hàng từ train/test gốc, sắp theo `id`; không tự thêm ngày thiếu hoặc điền nhãn.

| Nhóm cột | Ý nghĩa và quy tắc |
|---|---|
| `id`, `date`, `store_nbr`, `family`, `onpromotion` | Giữ giá trị nguồn. `family` được lưu dạng category để tiết kiệm bộ nhớ |
| `sales` | Float64; giữ nguyên nhãn train, kể cả số 0 và phần thập phân; NaN cho test |
| `is_test`, `sales_observed` | Cờ nguồn và cờ có nhãn. Test có `is_test=True`, `sales_observed=False` |
| `city`, `state`, `type`, `cluster` | Metadata cửa hàng, ghép nhiều-một bằng `store_nbr` |
| `year`, `month`, `day`, `dayofweek` | Lịch ngày; `dayofweek=0` là thứ Hai |
| `is_weekend`, `is_payday` | Cuối tuần; ngày 15 hoặc cuối tháng là cờ ngày trả lương theo ghi chú dữ liệu |
| `is_holiday`, `is_national_holiday`, `is_regional_holiday`, `is_local_holiday` | Ngày lễ đang áp dụng theo phạm vi quốc gia/tỉnh/thành phố |
| `is_transfer_day`, `is_bridge_day`, `is_additional_day` | Ngày nghỉ chuyển, ngày cầu nối, ngày nghỉ bổ sung; không coi ngày gốc đã chuyển là ngày nghỉ |
| `is_work_day`, `is_event`, `is_transferred_original` | Ngày làm bù, sự kiện, ngày gốc của lễ đã chuyển |
| `is_workday` | Cờ lịch quy ước: `(không cuối tuần hoặc ngày làm bù) và không là ngày lễ`. Không chứng minh siêu thị thực sự mở cửa |

`oil` và `transactions` vẫn được kiểm tra nhưng chưa ghép vào bảng nền. Không nội suy/chuẩn hóa theo toàn bộ tương lai. Features lịch sử của TV1 được sinh riêng trong `data/processed/features/` và chỉ lấy thông tin đã biết tại mốc dự báo; xem [quy tắc và từ điển features](features.md). Bảng nền vẫn giữ nguyên 29 cột. Nhiều nhóm hàng có đơn vị khác nhau; tổng `sales` giữa các nhóm là tổng kỹ thuật của biến đích, không phải doanh thu hay một đơn vị vật lý thống nhất.

Khi cần chuỗi liên tục, reindex riêng từng cửa hàng-nhóm hàng và giữ cờ thiếu. Ngày không có quan sát không đồng nghĩa lượng bán bằng 0. Hướng dẫn chạy và nội dung bàn giao ở [TV1.md](TV1.md).
