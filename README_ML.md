# Hướng dẫn huấn luyện XGBoost và LightGBM

Tài liệu này mô tả pipeline hiện có trong `scripts/run_models.py`, cấu hình ở `configs/models.json` và notebook Kaggle `notebooks/04_xgboost_lightgbm_kaggle.ipynb`.

## 1. Pipeline và cách chia dữ liệu

Pipeline dùng các Parquet do `scripts/prepare_features.py` tạo. Mỗi mẫu là một cặp cửa hàng–nhóm hàng–ngày dự báo; features lịch sử được chốt tại origin và `horizon` cho biết ngày cần dự báo cách origin bao xa. `id`, `date`, `origin` là metadata, không được đưa vào mô hình. Target là `sales`.

Ba stage có vai trò:

| Stage | Train | Dữ liệu dự báo | Mục đích |
|---|---|---|---|
| `tuning` | `tuning/train.parquet` | `tuning/features.parquet` + `tuning/labels.parquet` | Chọn cấu hình và kiểm tra trên giai đoạn tuning |
| `validation` | `validation/train.parquet` | `validation/features.parquet` + `validation/labels.parquet` | Đánh giá nội bộ sau khi đã chọn cấu hình |
| `test` | `test/train.parquet` | `test/features.parquet` | Fit cuối và tạo submission; không có nhãn để tính metric |

Trong mỗi stage, tùy chọn CV mặc định lấy ba origin cuối từ `train.parquet`. Mỗi fold dự báo trọn block 16 ngày. Một mẫu chỉ được dùng để train fold nếu toàn bộ horizon target của nó kết thúc trước hoặc đúng ngày origin validation của fold. Không chia ngẫu nhiên theo dòng.

CV dùng early stopping với `early_stopping_rounds` trong config. Số vòng cho lần fit cuối được chọn bằng trung vị `best_iteration` của các fold. Nhãn của forecast thuộc stage chỉ dùng để chấm sau khi fit, không dùng để early stop. Nếu bỏ CV, lần fit cuối dùng `n_estimators` trong config.

## 2. Cài đặt và tạo dữ liệu

Dùng Python 3.12. Đặt đủ bảy CSV của cuộc thi vào `data/raw/` theo tên trong hướng dẫn chính của repo. Sau đó chạy ở thư mục gốc dự án trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/prepare_features.py
```

Lần lượt, các lệnh này cài dependency, tạo `data/processed/base_panel.parquet`, rồi tạo các file trong `data/processed/features/{tuning,validation,test}/`. CSV và Parquet dữ liệu không được commit lên Git.

## 3. Chạy huấn luyện

Chạy tuning trước:

```powershell
.\.venv\Scripts\python.exe scripts/run_models.py --stage tuning --cv-folds 3
```

Sau khi điều chỉnh cấu hình, đánh giá validation:

```powershell
.\.venv\Scripts\python.exe scripts/run_models.py --stage validation --cv-folds 3
```

Khi đã chốt mô hình, tạo dự đoán test và submission:

```powershell
.\.venv\Scripts\python.exe scripts/run_models.py --stage test --cv-folds 3
```

Mặc định mỗi lệnh chạy cả XGBoost lẫn LightGBM. Để chạy riêng một model:

```powershell
.\.venv\Scripts\python.exe scripts/run_models.py --stage tuning --models lightgbm --cv-folds 3
```

### Tham số dòng lệnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--stage` | `tuning` | Chọn `tuning`, `validation` hoặc `test` |
| `--models` | `xgboost lightgbm` | Chọn một hoặc cả hai model |
| `--cv-folds` | `3` | Số origin cuối dùng làm fold; `0` bỏ CV và dùng nguyên `n_estimators` trong config |
| `--log-period` | `100` | In tiến trình mỗi N vòng boosting; nên dùng 50 hoặc 100 để theo dõi mà không tạo quá nhiều log |
| `--features-root` | `data/processed/features` | Thư mục chứa ba stage Parquet |
| `--config` | `configs/models.json` | File cấu hình JSON |
| `--output-dir` | `outputs/models` | Thư mục ghi prediction, metric, submission và model |
| `--sample-submission` | `data/raw/sample_submission.csv` | File mẫu dùng khi chạy stage test |

Ví dụ theo dõi thường xuyên hơn:

```powershell
.\.venv\Scripts\python.exe scripts/run_models.py --stage tuning --cv-folds 3 --log-period 25
```

Trong lúc CV, XGBoost và LightGBM in metric trên fold validation theo chu kỳ log; mỗi fold cũng in RMSLE, `best_iteration` và tổng thời gian fit. Lần fit cuối in số vòng và thời gian đã chạy. `--log-period 1` in mỗi vòng và thường tạo log quá dày.

## 4. Cấu hình model và tiền xử lý

Chỉnh `configs/models.json` để đổi tham số. Các mục hiện có:

- `target_transform`: `log1p` hoặc `none`. Mặc định dùng `log1p(sales)`; dự đoán được đổi về thang sales gốc bằng `expm1` và chặn ở mức 0.
- `categorical_features`: mặc định gồm `store_nbr`, `family`, `city`, `state`, `type`, `cluster`. Các mức categorical được lấy từ train và dùng nhất quán cho forecast.
- `models.xgboost`: `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, sampling, regularization, `max_bin`, số vòng early stopping và số luồng.
- `models.lightgbm`: `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, sampling, regularization, `max_bin`, số vòng early stopping và số luồng.

Features số không được scale. Giá trị thiếu lịch sử giữ dạng NaN để model xử lý; không điền missing thành 0. Model dùng objective RMSE trên target đã biến đổi log. Các metric forecast tính trên sales gốc: RMSLE là metric chính, RMSE và MAE là metric phụ.

## 5. Đầu ra

Với thư mục mặc định `outputs/models/`, mỗi lần chạy tạo:

- `{stage}_{model}_predictions.parquet`: hai cột `id,sales_pred`, theo thứ tự dòng của features forecast.
- `{stage}_{model}_metrics.json`: chỉ có ở tuning/validation; RMSLE, RMSE, MAE tách theo `family` và `horizon`.
- `{stage}_summary.json`: số dòng, điểm CV từng fold, RMSLE CV trung bình, số vòng fit cuối, metric forecast hoặc đường dẫn submission.
- `xgboost/model.json` và `xgboost/metadata.json`, hoặc `lightgbm/model.txt` và `lightgbm/metadata.json`: model cuối và thông tin target/categorical.
- `submission_xgboost.csv` và `submission_lightgbm.csv`: chỉ tạo ở stage test.

Submission được ghép theo `id` với `data/raw/sample_submission.csv`, kiểm tra ID duy nhất và đủ số lượng, giữ thứ tự cùng schema `id,sales`.

**Các stage dùng chung thư mục model mặc định.** Chạy stage khác sẽ ghi đè model artifact trong `outputs/models/xgboost/` và `outputs/models/lightgbm/`; các prediction và summary có tên stage riêng. Để giữ model/artifact riêng cho từng stage, đặt `--output-dir` khác nhau:

```powershell
.\.venv\Scripts\python.exe scripts/run_models.py --stage tuning --cv-folds 3 --output-dir outputs/models/tuning
.\.venv\Scripts\python.exe scripts/run_models.py --stage validation --cv-folds 3 --output-dir outputs/models/validation
.\.venv\Scripts\python.exe scripts/run_models.py --stage test --cv-folds 3 --output-dir outputs/models/test
```

Khi dùng thư mục riêng, file prediction và summary vẫn mang tiền tố stage, ví dụ `outputs/models/validation/validation_summary.json`.

## 6. Chạy notebook trên Kaggle

Notebook: [notebooks/04_xgboost_lightgbm_kaggle.ipynb](notebooks/04_xgboost_lightgbm_kaggle.ipynb).

Notebook gọi cùng `scripts/run_models.py`; nó không tự sinh Parquet features. Cần attach ba nguồn vào Kaggle Notebook:

1. Dataset code của repo, có `scripts/`, `src/`, `configs/models.json` và `requirements.txt`.
2. Dataset chứa thư mục `features/` đã sinh, với `tuning/`, `validation/`, `test/` và các Parquet bên trong.
3. Competition dataset Store Sales để notebook tìm `sample_submission.csv` khi chạy stage test.

Notebook tự dò các đường dẫn trong `/kaggle/input`. Nếu dò thấy nhiều ứng viên hoặc không tìm được, điền `PROJECT_ROOT`, `FEATURES_ROOT` và `SAMPLE_SUBMISSION` trong cell cấu hình. Các biến `STAGE`, `MODELS`, `CV_FOLDS` và `LOG_PERIOD` cũng chỉnh được tại đó. Mặc định notebook chạy validation, cả hai model, ba fold và log mỗi 100 vòng. Kết quả được ghi dưới `/kaggle/working/outputs/`; đổi `STAGE` thành `test` để tạo submission.

CPU là đủ để bắt đầu; không bắt buộc bật GPU. Nếu package thiếu, notebook thử cài scikit-learn, XGBoost và LightGBM; việc cài đặt cần môi trường Kaggle có Internet hoặc các package đã được cài sẵn.

## 7. Dependency và giới hạn hiện tại

`requirements.txt` khai báo scikit-learn, XGBoost và LightGBM cùng các dependency xử lý dữ liệu của dự án. Sau khi thay đổi requirements, cài lại bằng lệnh ở mục 2.

Runner hiện dùng bộ tham số cố định trong `configs/models.json`; CV chọn số vòng boosting cuối qua early stopping nhưng không tự tìm kiếm các tổ hợp hyperparameter. Hãy chỉnh config, chạy tuning để so sánh, rồi chỉ dùng validation làm đánh giá nội bộ sau khi chọn cấu hình. Test không có nhãn nên chỉ dùng để tạo dự đoán cuối.
