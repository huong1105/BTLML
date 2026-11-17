# Baseline tham khảo - trung bình cùng thứ

Baseline là quy tắc dự báo đơn giản để đối chiếu, không lựa chọn mô hình ML cho các thành viên.

- Mốc cuối được biết sales: **2017-07-30**.
- Cửa sổ lịch sử: **2017-07-03 đến 2017-07-30**, 28 ngày lịch.
- Validation: **2017-07-31 đến 2017-08-15**, 16 ngày.
- Chấm **28,512 dòng**, 1,782 chuỗi; giữ nguyên số 0 và lượng bán thập phân.

## Cách dự báo

Với mỗi cửa hàng-nhóm hàng-ngày đích, lấy trung bình cộng sales của các ngày cùng thứ trong cửa sổ lịch sử. Toàn bộ khoảng dự báo dùng cùng một mốc lịch sử; không cập nhật bằng sales thật của validation.
Nếu không có ngày cùng thứ, dùng trung bình cửa hàng-nhóm hàng trong cửa sổ; nếu không có lịch sử của chuỗi thì dùng 0. Cả ba trường hợp đều được đếm trong JSON.

## Kết quả thực tế

| Phương pháp | RMSLE validation |
|---|---:|
| Trung bình cùng thứ, 28 ngày lịch sử | **0.531130** |

RMSLE = sqrt(mean((log1p(dự báo) - log1p(thực tế))²)); càng thấp càng tốt. Tính trên toàn bộ dòng sau khi đối chiếu ID, không lấy trung bình điểm từng nhóm hàng để thay thế.

Nguồn dự báo: same_weekday: 28,512.

## Bàn giao và giới hạn

- Dự báo đối chiếu và nhãn validation: `outputs/baseline/validation_predictions.csv`. Đây là tệp nội bộ, không phải submission Kaggle.
- `baseline_metrics.json` ghi mốc, cửa sổ, điểm, số dòng, checksum nguồn/dự báo và môi trường.
- Các phương pháp khác cần dùng cùng origin, các ID/ngày validation và hàm RMSLE để so sánh công bằng. TV4 có thể thống nhất thêm các mốc khác với nhóm.
- Đây là một holdout tham khảo; chưa kết luận chất lượng trên các giai đoạn khác hoặc điểm Kaggle. Baseline chưa dùng khuyến mãi, ngày lễ hay xu hướng.
- Sales được giữ ở thang gốc; log1p chỉ xuất hiện trong phép chấm RMSLE. Không có scaler được fit hoặc mô hình ML được huấn luyện.
- CSV gốc được đối chiếu checksum trước/sau và không thay đổi.
