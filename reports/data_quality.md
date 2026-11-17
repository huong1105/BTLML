# Báo cáo chất lượng dữ liệu TV1

Tính trực tiếp từ 7 CSV trong data/raw; không sửa tệp gốc.

| Tệp | Số dòng | Số cột | Giá trị thiếu | Dòng trùng hoàn toàn |
|---|---:|---:|---:|---:|
| train.csv | 3,000,888 | 6 | 0 | 0 |
| test.csv | 28,512 | 5 | 0 | 0 |
| stores.csv | 54 | 5 | 0 | 0 |
| oil.csv | 1,218 | 2 | 43 | 0 |
| transactions.csv | 83,488 | 3 | 0 | 0 |
| holidays_events.csv | 350 | 6 | 0 | 0 |
| sample_submission.csv | 28,512 | 2 | 0 | 0 |

## Phạm vi bán hàng

| Phần dữ liệu | Số dòng | Ngày đầu | Ngày cuối | Cửa hàng | Nhóm hàng | Chuỗi |
|---|---:|---|---|---:|---:|---:|
| train | 3,000,888 | 2013-01-01 | 2017-08-15 | 54 | 33 | 1782 |
| test | 28,512 | 2017-08-16 | 2017-08-31 | 54 | 33 | 1782 |

- Ngày lịch không có dòng trong train: 2013-12-25, 2014-12-25, 2015-12-25, 2016-12-25.
- train: 0 khóa trùng, 0 id trùng; 1,782-1,782 dòng mỗi ngày quan sát; 0 ngày thiếu lưới cửa hàng × nhóm hàng.
- test: 0 khóa trùng, 0 id trùng; 1,782-1,782 dòng mỗi ngày quan sát; 0 ngày thiếu lưới cửa hàng × nhóm hàng.

Train có 939,130 dòng sales = 0 (31.30%) và 462,113 dòng có phần thập phân. Giữ nguyên các giá trị này.

## Kiểm tra đối chiếu

- train_test_id_disjoint: đạt.
- test_after_train: đạt.
- test_starts_next_day: đạt.
- sample_submission_same_ids: đạt.
- sample_submission_same_id_order: đạt.
- test_pairs_seen_in_train: đạt.
- test_has_16_observed_days: đạt.
- test_calendar_contiguous: đạt.

## Bảng phụ và quyết định xử lý

- Ngày lễ: 312 ngày có 350 sự kiện, tối đa 4 sự kiện/ngày. Có 0 dòng địa phương không khớp cửa hàng hiện có.
- Lễ gốc đã chuyển ngày không được đánh dấu là ngày nghỉ; giữ cờ riêng và dùng dòng Transfer ở ngày thực tế. National áp dụng toàn quốc, Regional theo state, Local theo city. Không nhân bản dòng bán hàng.
- Oil có 43 ô giá thiếu và 0 ngày trùng. Chưa điền hay ghép giá dầu vào bảng nền.
- Oil đối chiếu train: 481 ngày bán hàng không có dòng oil; tổng cộng 521 ngày không có giá dùng được (bao gồm dòng thiếu giá).
- Oil đối chiếu test: 4 ngày bán hàng không có dòng oil; tổng cộng 4 ngày không có giá dùng được (bao gồm dòng thiếu giá).
- Transactions kết thúc 2017-08-15; phủ 83,488/90,936 cặp cửa hàng-ngày train (91.81%); 0 khóa trùng. Chưa ghép giao dịch thực tương lai vào bảng nền.

## Kiểu dữ liệu sau khi đọc

- train.csv: id=int64, date=datetime64[us], store_nbr=int16, family=category, sales=float64, onpromotion=int32.
- test.csv: id=int64, date=datetime64[us], store_nbr=int16, family=category, onpromotion=int32.
- stores.csv: store_nbr=int16, city=str, state=str, type=str, cluster=int64.
- oil.csv: date=datetime64[us], dcoilwtico=float64.
- transactions.csv: date=datetime64[us], store_nbr=int16, transactions=int32.
- holidays_events.csv: date=datetime64[us], type=str, locale=str, locale_name=str, description=str, transferred=bool.
- sample_submission.csv: id=int64, sales=float64.

## Nội dung bàn giao

- sales là lượng bán, có thể có phần thập phân; onpromotion là số mặt hàng khuyến mãi.
- Ngày thiếu khác số 0 thật. Không điền sales test hoặc ngày không quan sát thành 0.
- Ghép stores nhiều-một; tổng hợp lễ theo ngày và phạm vi áp dụng của từng cửa hàng trước khi merge.
- Giữ nguyên giá trị nguồn; bảng nền chưa tạo lag, scaler hay cửa sổ chuỗi. Các thành viên tự xử lý theo mốc dự báo.
- Bảng nền chưa ghép oil/transactions. Giá dầu thiếu ngày/giá trị; giao dịch thực của ngày tương lai chưa biết.
