# Báo cáo khám phá features và quan hệ với target sales

## 1. Phạm vi phân tích

Báo cáo tổng hợp notebook `notebooks/03_features_analysis.ipynb` và dữ liệu ở `data/processed/features/`. Các cell được chạy trên ba bộ features dự báo; thống kê tương quan target chỉ dùng nhãn tuning và validation. Không huấn luyện mô hình trong phân tích này.

Mỗi dòng là một tổ hợp cửa hàng–nhóm hàng–ngày cần dự báo. `origin` là ngày cuối cùng có thể dùng sales lịch sử. Target là sales thật ở ngày đích. Mỗi chuỗi có 16 dòng dự báo liên tiếp; features lịch sử được cố định tại origin cho cả horizon.

## 2. Schema và định dạng

Mỗi `features.parquet` của kỳ dự báo có 46 cột: 43 đầu vào theo manifest và 3 metadata. Mỗi `train.parquet` có thêm cột target `sales`, tổng cộng 47 cột. Target/labels tách riêng khỏi features dự báo.

| Nhóm | Số cột | Cột | Kiểu dữ liệu và ý nghĩa |
|---|---:|---|---|
| Định danh chuỗi | 2 | `store_nbr`, `family` | `store_nbr`: số nguyên ID cửa hàng (`int16`); `family`: chuỗi phân loại dạng dictionary/categorical. |
| Thuộc tính cửa hàng | 4 | `city`, `state`, `type`, `cluster` | `city`, `state`, `type` là categorical; `cluster` lưu số nguyên nhưng là mã nhóm, không nên mặc định xem như đại lượng liên tục. |
| Khuyến mãi | 1 | `onpromotion` | Số nguyên (`int32`), số mặt hàng được khuyến mãi cho chuỗi ở ngày đích. |
| Lịch | 6 | `year`, `month`, `day`, `dayofweek`, `is_weekend`, `is_payday` | Năm/tháng/ngày/thứ là integer; cuối tuần và ngày 15/cuối tháng là Boolean. |
| Ngày lễ và ngày làm việc | 11 | `is_holiday`, `is_national_holiday`, `is_regional_holiday`, `is_local_holiday`, `is_transfer_day`, `is_bridge_day`, `is_additional_day`, `is_work_day`, `is_event`, `is_transferred_original`, `is_workday` | Các cờ Boolean. Phạm vi quốc gia/vùng/thành phố đã được gắn theo cửa hàng. Ngày lễ bị chuyển có cờ riêng và không mặc định là ngày nghỉ thực tế. |
| Vị trí horizon | 1 | `horizon` | Integer (`int16`), khoảng cách 1–16 ngày từ origin tới ngày đích. |
| Mức sales lịch sử | 11 | `sales_at_origin`, `sales_origin_lag_7/14/28`, `sales_mean_7/14/28`, `sales_std_7/14/28`, `sales_zero_rate_28` | Số thực (`float64`). Lag tính theo ngày lịch; cửa sổ mean/std bao gồm origin; độ lệch chuẩn dùng `ddof=0`. |
| Độ phủ lịch sử | 7 | `sales_observed_days_7/14/28`, `sales_at_origin_missing`, `sales_origin_lag_7/14/28_missing` | Số ngày quan sát là integer (`int16`); các cờ thiếu là Boolean. Ngày thiếu không bị biến thành sales 0. |
| Metadata | 3 | `id`, `date`, `origin` | `id`: integer (`int64`); `date`, `origin`: timestamp. Giữ để nối, sắp xếp và truy vết, không phải predictor mặc định. |
| Target | 1 | `sales` | Số thực (`float64`) ở thang sales gốc. Có trong train/labels, không có trong forecast features của test. |

Các cửa sổ lịch sử dùng 7, 14 và 28 ngày kết thúc tại origin. Các cột `sales_origin_lag_7/14/28` lấy sales đúng tại origin trừ số ngày tương ứng. Các cột `sales_observed_days_*` đếm ngày có nhãn thật, không đếm ngày sales dương.

## 3. Target và cách hiểu quan hệ feature–target

Target huấn luyện là `sales` của chính ngày đích, không phải tổng sales của cả 16 ngày. Một chuỗi có thể xuất hiện nhiều lần trong train với các origin khác nhau. Mỗi mẫu lịch sử được giữ khi toàn bộ horizon 16 ngày có đủ chuỗi/ngày và nhãn không vượt cutoff của stage.

`build()` tạo features dự báo nhưng không trả về sales hoặc cờ quan sát/kiểm tra. `labels()` xuất riêng `id,sales`; khi tạo train lịch sử, code mới ghép labels vào cột `sales`. Vì vậy input thời gian dự báo dùng covariates biết tại ngày đích và lịch sử đến origin, không dùng sales thật trong tương lai.

Tương quan dưới đây đo liên hệ đơn biến trên các dòng forecast có target thật, không phải độ quan trọng đã được kiểm chứng trong mô hình. Pearson nhạy với độ lớn và giá trị cực đoan; Spearman đo xu hướng theo thứ hạng, thường phù hợp hơn khi sales lệch phải. Tương quan không chứng minh nhân quả.

## 4. Kết quả khám phá chính

### 4.1 Quy mô và độ đầy đủ

| Stage | Origin | Khoảng dự báo | Dòng features | Chuỗi | Dòng train | Origin train | Labels forecast |
|---|---|---|---:|---:|---:|---:|---|
| tuning | 2017-07-14 | 2017-07-15 – 2017-07-30 | 28.512 | 1.782 | 2.765.664 | 97 | Có |
| validation | 2017-07-30 | 2017-07-31 – 2017-08-15 | 28.512 | 1.782 | 2.794.176 | 98 | Có |
| test | 2017-08-15 | 2017-08-16 – 2017-08-31 | 28.512 | 1.782 | 2.822.688 | 99 | Không |

Mỗi ngày horizon có 1.782 dòng, tương ứng 1.782 ID/chuỗi, tổng cộng 28.512 dòng cho 16 ngày. Cả ba forecast features đều không có giá trị thiếu trong 43 cột đầu vào. Trong các forecast window này, `sales_observed_days_7/14/28` lần lượt cố định ở 7/14/28 và các cờ lag thiếu đều false. Điều này cho biết lịch sử gần origin đầy đủ ở các dòng dự báo hiện tại; không nên suy rộng rằng mọi mẫu train lịch sử cũng có độ phủ như vậy.

Manifest ghi nhận bốn origin lịch sử bị bỏ ở cả ba stage vì không có đủ lưới ngày/chuỗi đích: `2013-12-10`, `2014-12-13`, `2015-12-16`, `2016-12-18`.

### 4.2 Ví dụ một chuỗi trong validation

Notebook hiển thị cửa hàng 1, nhóm `AUTOMOTIVE`, từ 31/07 đến 15/08/2017. Tại origin 30/07, `sales_mean_7` là 4,71, `sales_mean_28` là 4,57, `sales_zero_rate_28` là 0,071. Các giá trị này lặp lại trong cả 16 ngày vì chúng được tính tại origin; `horizon`, lịch, khuyến mãi và target thật thay đổi theo ngày. Sales thật trong ví dụ dao động từ 1 đến 9.

Ví dụ này cho thấy features lịch sử là mức tham chiếu của chuỗi, không phải lag cập nhật hằng ngày trong horizon. Dự báo recursive có cập nhật lịch sử sau mỗi ngày sẽ cần thiết kế khác.

### 4.3 Phân bố lịch sử sales

Trên forecast validation, `sales_at_origin` có trung bình 630,61 và trung vị 36; `sales_mean_28` có trung bình 473,42 và trung vị 30,80. `sales_origin_lag_28` có trung bình 727,49, trung vị 43,64. Trung bình lớn hơn rất nhiều trung vị, phản ánh phân bố lệch phải và sự khác nhau lớn về quy mô giữa các chuỗi. Do đó notebook dùng `log1p` để trực quan hóa, nhưng file features vẫn giữ thang gốc.

Thống kê cửa sổ và lag có tương quan cao với nhau, phù hợp vì chúng đều mô tả mức/quy mô sales lịch sử. Chúng có thể bổ sung tín hiệu về mùa vụ ngắn hạn và biến động, nhưng cũng có khả năng dư thừa tuyến tính; cần đánh giá bằng validation theo thời gian hoặc ablation.

### 4.4 Khác biệt giữa nhóm hàng

Mỗi nhóm hàng có 864 dòng trong validation, tương ứng 54 cửa hàng × 16 ngày. `GROCERY I` có sales target trung bình khoảng 4.606,96 và `sales_mean_28` trung bình 4.517,14. `BEVERAGES` và `PRODUCE` cũng có quy mô lịch sử lớn. Ngược lại, `BOOKS` có target trung bình khoảng 0,01 và 99% dòng sales bằng 0; `BABY CARE` và `HOME APPLIANCES` cũng có tỷ lệ 0 cao, lần lượt khoảng 88% và 85%.

`family` vì thế chứa tín hiệu phân tầng mạnh giữa các nhóm có quy mô sales khác nhau. Cần mã hóa như biến phân loại; không gán ý nghĩa số học cho nhãn family. Các thống kê theo family bị gộp qua cửa hàng, nên chưa phản ánh toàn bộ khác biệt giữa từng cửa hàng.

### 4.5 Tương quan số với target

Các hệ số được tính riêng trên tuning và validation forecast labels. Bảng nêu Spearman và Pearson; thứ tự theo trị tuyệt đối Spearman.

| Feature | Tuning Spearman | Tuning Pearson | Validation Spearman | Validation Pearson |
|---|---:|---:|---:|---:|
| `sales_mean_7` | 0,981 | 0,958 | 0,976 | 0,967 |
| `sales_mean_14` | 0,981 | 0,959 | 0,976 | 0,967 |
| `sales_mean_28` | 0,981 | 0,959 | 0,976 | 0,967 |
| `sales_at_origin` | 0,971 | 0,940 | 0,967 | 0,947 |
| `sales_origin_lag_7` | 0,971 | 0,944 | 0,964 | 0,947 |
| `sales_origin_lag_14` | 0,969 | 0,948 | 0,963 | 0,947 |
| `sales_origin_lag_28` | 0,967 | 0,943 | 0,962 | 0,950 |
| `sales_std_7` | 0,965 | 0,875 | 0,960 | 0,891 |
| `sales_std_14` | 0,969 | 0,907 | 0,964 | 0,899 |
| `sales_std_28` | 0,969 | 0,909 | 0,966 | 0,898 |
| `sales_zero_rate_28` | -0,769 | -0,172 | -0,757 | -0,178 |
| `onpromotion` | 0,727 | 0,631 | 0,758 | 0,567 |
| `horizon` | 0,001 | 0,003 | -0,005 | -0,024 |

**Diễn giải:**

- Mean lịch sử có liên hệ đơn điệu dương rất mạnh với target ở cả hai giai đoạn. Lag gần và sales tại origin cũng mạnh, cho thấy persistence của doanh số theo chuỗi.
- Độ lệch chuẩn cũng dương mạnh theo Spearman. Các chuỗi có quy mô sales lớn thường có dao động tuyệt đối lớn; std không đồng nghĩa với hiệu ứng làm tăng sales.
- `sales_zero_rate_28` có tương quan Spearman âm mạnh nhưng Pearson âm yếu. Điều này cho thấy quan hệ có cấu trúc phi tuyến/ảnh hưởng thứ hạng và phân bố lệch, nên không nên diễn giải Pearson một mình.
- `onpromotion` dương với target trong thống kê gộp. Tuy nhiên số lượng khuyến mãi thường lớn hơn ở nhóm hàng/cửa hàng có quy mô bán cao; kết quả này không chứng minh khuyến mãi gây tăng sales. Nên so sánh trong nhóm hàng/cửa hàng và bằng validation hoặc mô hình có kiểm soát đặc trưng.
- `horizon` gần như không có tương quan đơn biến với target tổng thể. Vẫn nên giữ ban đầu vì có thể tương tác với thứ trong tuần, ngày lễ hoặc lịch sử, và tương quan gộp có thể triệt tiêu khác biệt theo từng ngày.
- Các cột coverage và cờ missing không xuất hiện trong bảng tương quan vì chúng không đổi trong các forecast window này. Chúng vẫn có ích cho các mẫu lịch sử nếu lịch sử thiếu.

### 4.6 Khuyến mãi và target

Trên validation, nhóm dòng có `onpromotion=0` có sales target trung bình 55,10, trung vị 6 và 25,9% sales bằng 0. Nhóm có 1–5 mặt hàng khuyến mãi có mean 155,20, median 61, không có sales bằng 0 trong quan sát này. Nhóm có từ 6 mặt hàng có mean 1.568,04, median 683,42 và cũng không có sales bằng 0.

Độ chênh lệch rất lớn có khả năng phản ánh cả cơ cấu nhóm hàng và quy mô bán, không chỉ khuyến mãi. Dùng các thống kê này như dấu hiệu cần phân tầng/kiểm soát, không như kết luận nhân quả.

### 4.7 Lịch, lễ và chênh lệch stage

Trong validation, horizon 12 rơi vào ngày lễ toàn quốc nên cả 1.782 dòng cửa hàng–nhóm hàng đều có `is_holiday=true`. Horizon 6 và 16 mỗi ngày có 33 dòng ngày lễ theo phạm vi địa phương/vùng. Điều này minh họa vì sao cờ holiday được gắn theo cửa hàng thay vì chỉ theo ngày. Horizon 6, 7, 13 và 14 là cuối tuần trong khoảng này.

Tỷ lệ dòng có khuyến mãi trong forecast lần lượt là 40,80% ở tuning, 43,77% ở validation và 44,08% ở test. `sales_mean_28` trung bình giảm nhẹ từ 477,49 qua 473,42 đến 469,71; tỷ lệ zero-rate lịch sử trung bình tăng nhẹ từ 14,29% lên 14,74%. Đây là thay đổi giữa ba cutoff, không đủ để kết luận xu hướng dài hạn hoặc drift có ý nghĩa thống kê.

### 4.8 Phân bố nhãn forecast

| Chỉ số | Tuning | Validation |
|---|---:|---:|
| Sales trung bình | 481,76 | 467,14 |
| Sales trung vị | 29 | 30 |
| Tỷ lệ sales bằng 0 | 14,74% | 14,58% |
| Sales lớn nhất | 18.340 | 15.190 |
| Tổng sales kỳ | 13.736.027,85 | 13.319.179,78 |

Target lệch phải mạnh và có nhiều sales bằng 0. Nên dùng metric đúng yêu cầu bài toán trên thang gốc, chú ý ảnh hưởng của giá trị rất lớn; nếu thử biến đổi `log1p` cho target thì phải đảo biến đổi khi tạo dự báo và tính metric theo quy ước của dự án.

## 5. Gợi ý dùng kết quả làm cơ sở huấn luyện

1. Bắt đầu với toàn bộ cột trong `manifest["feature_columns"]`, nhưng mã hóa các categorical rõ ràng: `family`, `city`, `state`, `type`; cân nhắc `store_nbr` và `cluster` là mã nhóm thay vì số liên tục.
2. Giữ các thống kê lịch sử làm baseline predictors. Mean và lag có tương quan rất cao; thử regularization hoặc ablation để xem nhóm nào bổ sung hiệu năng thay vì loại chỉ vì tương quan giữa các feature.
3. Đánh giá riêng theo `family`, quy mô sales và horizon để thấy sai số có bị chi phối bởi nhóm hàng lớn hoặc horizon xa hay không.
4. Kiểm tra quan hệ `onpromotion` có còn hữu ích sau khi kiểm soát `family`, cửa hàng, lịch sử và ngày hay không. Không suy ra tác động khuyến mãi từ tương quan gộp.
5. Fit imputation, encoding, scaling hoặc biến đổi target chỉ trên train của stage. Không dùng labels validation để tạo input hoặc fit preprocessing.
6. Dùng tuning để lựa chọn; dùng validation để xác nhận; test chỉ cho dự báo cuối vì không có nhãn nội bộ.

Notebook cung cấp bảng, biểu đồ và các hệ số mô tả dữ liệu. Các kết quả này là phân tích khám phá, không phải feature importance của mô hình và chưa thay thế đánh giá dự báo RMSLE.