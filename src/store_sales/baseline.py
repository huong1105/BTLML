"""Fixed-origin same-weekday mean baseline for a 16-day forecast.

The complete forecast uses only observations in the trailing window ending at
origin. Actual sales inside the forecast horizon are never used for prediction.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .data import sha256_file


SERIES_KEYS = ["store_nbr", "family"]
TARGET_COLUMNS = ["id", "date", *SERIES_KEYS]


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _day(value, name: str) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
        raise ValueError(f"{name} must be a timezone-naive calendar date")
    return value


def _dates(values: pd.Series, name: str) -> pd.Series:
    values = pd.to_datetime(values, errors="raise")
    if values.isna().any() or values.dt.tz is not None or values.ne(values.dt.normalize()).any():
        raise ValueError(f"{name} must contain calendar dates without timezone/time")
    return values


def predict_same_weekday(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    origin: pd.Timestamp,
    window_days: int = 28,
    horizon_days: int = 16,
) -> pd.DataFrame:
    """Return forecasts in targets' original order, ignoring any target labels.

    Fallbacks: same weekday mean -> recent series mean -> zero when the series
    has no observed sales in the permitted window. Real zeros remain in means.
    History may contain later dates, but these rows are removed before any
    target-derived computation or sales validation.
    """
    window_days = _positive_integer(window_days, "window_days")
    horizon_days = _positive_integer(horizon_days, "horizon_days")
    origin = _day(origin, "origin")
    required_history = {"date", *SERIES_KEYS, "sales"}
    if not required_history.issubset(history.columns):
        raise ValueError(f"History missing columns: {sorted(required_history - set(history.columns))}")
    if not set(TARGET_COLUMNS).issubset(targets.columns):
        raise ValueError("Targets require id, date, store_nbr and family")
    if targets.empty:
        raise ValueError("Targets must not be empty")

    requested = targets[TARGET_COLUMNS].copy()
    requested["date"] = _dates(requested["date"], "target dates")
    if requested[TARGET_COLUMNS].isna().any().any():
        raise ValueError("Target keys and IDs must not be null")
    if requested["id"].duplicated().any() or requested.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError("Target IDs and series-date keys must be unique")
    if not requested["date"].between(origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=horizon_days)).all():
        raise ValueError("All target dates must lie within the forecast horizon after origin")
    requested["_target_order"] = np.arange(len(requested))
    requested["_weekday"] = requested["date"].dt.dayofweek

    permitted = history[["date", *SERIES_KEYS, "sales"]].copy()
    permitted["date"] = _dates(permitted["date"], "history dates")
    start = origin - pd.Timedelta(days=window_days - 1)
    permitted = permitted.loc[permitted["date"].between(start, origin)].copy()
    if permitted[["date", *SERIES_KEYS, "sales"]].isna().any().any():
        raise ValueError("Observed history in the window must not have missing keys/sales")
    permitted["sales"] = pd.to_numeric(permitted["sales"], errors="raise").astype("float64")
    if not np.isfinite(permitted["sales"]).all() or permitted["sales"].lt(0).any():
        raise ValueError("Observed history sales must be finite and nonnegative")
    if permitted.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError("History has duplicate series-date keys inside the window")

    if permitted.empty:
        requested["sales_pred"] = 0.0
        requested["prediction_source"] = "zero_no_history"
    else:
        permitted["_weekday"] = permitted["date"].dt.dayofweek
        weekday_means = permitted.groupby([*SERIES_KEYS, "_weekday"], observed=True, as_index=False)["sales"].mean().rename(columns={"sales": "_weekday_mean"})
        series_means = permitted.groupby(SERIES_KEYS, observed=True, as_index=False)["sales"].mean().rename(columns={"sales": "_series_mean"})
        requested = requested.merge(weekday_means, on=[*SERIES_KEYS, "_weekday"], how="left", validate="many_to_one", sort=False)
        requested = requested.merge(series_means, on=SERIES_KEYS, how="left", validate="many_to_one", sort=False)
        requested["prediction_source"] = np.select(
            [requested["_weekday_mean"].notna(), requested["_series_mean"].notna()],
            ["same_weekday", "series_mean"], default="zero_no_history",
        )
        requested["sales_pred"] = requested["_weekday_mean"].fillna(requested["_series_mean"]).fillna(0.0)
    result = requested.sort_values("_target_order")[TARGET_COLUMNS + ["sales_pred", "prediction_source"]].reset_index(drop=True)
    if len(result) != len(targets) or not np.array_equal(result["id"].to_numpy(), targets["id"].to_numpy()):
        raise AssertionError("Prediction IDs/order changed")
    if not np.isfinite(result["sales_pred"]).all() or result["sales_pred"].lt(0).any():
        raise AssertionError("Predictions must be finite and nonnegative")
    return result


def run_baseline(
    raw_dir: Path,
    output_dir: Path,
    report_dir: Path,
    validation_start=None,
    window_days: int = 28,
    horizon_days: int = 16,
) -> dict:
    """Run one chronological holdout and save its predictions/actual score."""
    from .metrics import evaluate_by_id

    window_days = _positive_integer(window_days, "window_days")
    horizon_days = _positive_integer(horizon_days, "horizon_days")
    raw_dir, output_dir, report_dir = map(Path, (raw_dir, output_dir, report_dir))
    source = raw_dir / "train.csv"
    source_hash = sha256_file(source)
    train = pd.read_csv(source, usecols=[*TARGET_COLUMNS, "sales"],
                        dtype={"id": "int64", "store_nbr": "int16", "family": "category", "sales": "float64"})
    train["date"] = _dates(train["date"], "train dates")
    if train.empty or train[TARGET_COLUMNS + ["sales"]].isna().any().any():
        raise ValueError("Train must contain observed rows with complete keys/sales")
    if train["id"].duplicated().any() or train.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError("Train IDs and series-date keys must be unique")
    if not np.isfinite(train["sales"]).all() or train["sales"].lt(0).any():
        raise ValueError("Train sales must be finite and nonnegative")
    start = _day(validation_start, "validation_start") if validation_start is not None else train["date"].max() - pd.Timedelta(days=horizon_days - 1)
    origin = start - pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=horizon_days - 1)
    actual = train.loc[train["date"].between(start, end)].copy()
    expected_dates = pd.date_range(start, end)
    if not pd.DatetimeIndex(actual["date"].unique()).sort_values().equals(expected_dates):
        raise ValueError("Validation must have labels for all requested calendar days")
    pairs = train[SERIES_KEYS].drop_duplicates()
    if len(actual) != len(pairs) * horizon_days:
        raise ValueError("Validation must cover every series on every requested day")
    if not actual.groupby("date", observed=True).size().eq(len(pairs)).all():
        raise ValueError("Validation series grid differs across dates")
    permitted = train.loc[train["date"].between(origin - pd.Timedelta(days=window_days - 1), origin)]
    if permitted.empty:
        raise ValueError("No observations available in the history window")
    predictions = predict_same_weekday(train, actual[TARGET_COLUMNS], origin, window_days, horizon_days)
    scores = evaluate_by_id(actual[["id", "sales"]], predictions[["id", "sales_pred"]])
    detailed = actual.rename(columns={"sales": "sales_actual"}).merge(predictions[["id", "sales_pred", "prediction_source"]], on="id", how="left", validate="one_to_one", sort=False)
    detailed["horizon"] = (detailed["date"] - origin).dt.days
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "validation_predictions.csv"
    detailed.to_csv(prediction_path, index=False, float_format="%.12g")
    if sha256_file(source) != source_hash:
        raise RuntimeError("Raw train.csv changed during baseline execution")
    project_root = Path(__file__).resolve().parents[2]
    try:
        displayed_path = prediction_path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        displayed_path = prediction_path.name
    metrics = {
        "method": "same_weekday_mean_28d" if window_days == 28 else f"same_weekday_mean_{window_days}d",
        "aggregation": "arithmetic_mean", "window_days": window_days, "horizon_days": horizon_days,
        "origin": str(origin.date()), "history_start": str((origin - pd.Timedelta(days=window_days - 1)).date()),
        "history_end": str(origin.date()), "last_observed_history_date": str(permitted["date"].max().date()),
        "validation_start": str(start.date()), "validation_end": str(end.date()),
        "n_rows": int(scores["n_rows"]), "n_series": len(pairs), "rmsle": float(scores["rmsle"]),
        "fallback_counts": {str(key): int(value) for key, value in predictions["prediction_source"].value_counts().items()},
        "raw_train_sha256": source_hash, "raw_train_unchanged": True,
        "predictions_path": displayed_path, "predictions_sha256": sha256_file(prediction_path),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (report_dir / "baseline_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Baseline tham khảo - trung bình cùng thứ", "",
        "Baseline là quy tắc dự báo đơn giản để đối chiếu, không lựa chọn mô hình ML cho các thành viên.", "",
        f"- Mốc cuối được biết sales: **{metrics['origin']}**.",
        f"- Cửa sổ lịch sử: **{metrics['history_start']} đến {metrics['history_end']}**, {window_days} ngày lịch.",
        f"- Validation: **{metrics['validation_start']} đến {metrics['validation_end']}**, {horizon_days} ngày.",
        f"- Chấm **{metrics['n_rows']:,} dòng**, {metrics['n_series']:,} chuỗi; giữ nguyên số 0 và lượng bán thập phân.", "",
        "## Cách dự báo", "",
        "Với mỗi cửa hàng-nhóm hàng-ngày đích, lấy trung bình cộng sales của các ngày cùng thứ trong cửa sổ lịch sử. Toàn bộ khoảng dự báo dùng cùng một mốc lịch sử; không cập nhật bằng sales thật của validation.",
        "Nếu không có ngày cùng thứ, dùng trung bình cửa hàng-nhóm hàng trong cửa sổ; nếu không có lịch sử của chuỗi thì dùng 0. Cả ba trường hợp đều được đếm trong JSON.", "",
        "## Kết quả thực tế", "", "| Phương pháp | RMSLE validation |", "|---|---:|",
        f"| Trung bình cùng thứ, {window_days} ngày lịch sử | **{metrics['rmsle']:.6f}** |", "",
        "RMSLE = sqrt(mean((log1p(dự báo) - log1p(thực tế))²)); càng thấp càng tốt. Tính trên toàn bộ dòng sau khi đối chiếu ID, không lấy trung bình điểm từng nhóm hàng để thay thế.", "",
        "Nguồn dự báo: " + ", ".join(f"{key}: {value:,}" for key, value in metrics["fallback_counts"].items()) + ".", "",
        "## Bàn giao và giới hạn", "",
        f"- Dự báo đối chiếu và nhãn validation: `{metrics['predictions_path']}`. Đây là tệp nội bộ, không phải submission Kaggle.",
        "- `baseline_metrics.json` ghi mốc, cửa sổ, điểm, số dòng, checksum nguồn/dự báo và môi trường.",
        "- Các phương pháp khác cần dùng cùng origin, các ID/ngày validation và hàm RMSLE để so sánh công bằng. TV4 có thể thống nhất thêm các mốc khác với nhóm.",
        "- Đây là một holdout tham khảo; chưa kết luận chất lượng trên các giai đoạn khác hoặc điểm Kaggle. Baseline chưa dùng khuyến mãi, ngày lễ hay xu hướng.",
        "- Sales được giữ ở thang gốc; log1p chỉ xuất hiện trong phép chấm RMSLE. Không có scaler được fit hoặc mô hình ML được huấn luyện.",
        "- CSV gốc được đối chiếu checksum trước/sau và không thay đổi.",
    ]
    (report_dir / "baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
