"""Reusable TV1 starter: read, inspect and prepare the competition data.

Original targets are never imputed. Historical features are generated separately
by store_sales.features; model-specific learned transforms fit on train only.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .calendar import HOLIDAY_COLUMNS, calendar_features, holiday_features


RAW_FILES = (
    "train.csv", "test.csv", "stores.csv", "oil.csv", "transactions.csv",
    "holidays_events.csv", "sample_submission.csv",
)
KEY = ["date", "store_nbr", "family"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load seven unchanged CSVs from raw_dir, parsing date fields explicitly."""
    raw_dir = Path(raw_dir)
    missing = [name for name in RAW_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CSVs in {raw_dir}: {', '.join(missing)}")
    tables: dict[str, pd.DataFrame] = {}
    for name in RAW_FILES:
        dtypes = {}
        if name in {"train.csv", "test.csv"}:
            dtypes = {"id": "int64", "store_nbr": "int16", "family": "category", "onpromotion": "int32"}
        elif name == "stores.csv":
            dtypes = {"store_nbr": "int16"}
        elif name == "transactions.csv":
            dtypes = {"store_nbr": "int16", "transactions": "int32"}
        frame = pd.read_csv(raw_dir / name, dtype=dtypes)
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        tables[Path(name).stem] = frame
    labels = sorted(set(tables["train"]["family"].dropna()) | set(tables["test"]["family"].dropna()))
    family_dtype = pd.CategoricalDtype(categories=labels)
    for name in ("train", "test"):
        tables[name]["family"] = tables[name]["family"].astype(family_dtype)
    return tables


def _validate_keys(tables: dict[str, pd.DataFrame]) -> None:
    for name in ("train", "test"):
        frame = tables[name]
        required = ["id", *KEY, "onpromotion", *(["sales"] if name == "train" else [])]
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if frame[required].isna().any().any():
            raise ValueError(f"{name}: null values in required raw columns")
        if frame.empty:
            raise ValueError(f"{name}: empty sales partition")
        if frame.duplicated(KEY).any() or frame["id"].duplicated().any():
            raise ValueError(f"{name}: duplicate id or (date, store_nbr, family)")
        promotion = frame["onpromotion"]
        if (promotion.lt(0) | ~np.isfinite(promotion) | promotion.mod(1).ne(0)).any():
            raise ValueError(f"{name}: onpromotion must be a finite nonnegative count")
        if frame["date"].ne(frame["date"].dt.normalize()).any():
            raise ValueError(f"{name}: dates must not contain time-of-day values")
    if tables["train"]["id"].isin(tables["test"]["id"]).any():
        raise ValueError("train/test ids overlap")
    if tables["train"]["date"].max() >= tables["test"]["date"].min():
        raise ValueError("test must follow the complete training date range")
    target = tables["train"]["sales"]
    if (target.lt(0) | ~np.isfinite(target)).any():
        raise ValueError("train: sales must be finite and nonnegative")
    if tables["stores"]["store_nbr"].duplicated().any():
        raise ValueError("stores: duplicate store_nbr")
    stores_required = ["store_nbr", "city", "state", "type", "cluster"]
    if not set(stores_required).issubset(tables["stores"].columns):
        raise ValueError("stores: missing metadata columns")
    if tables["stores"][stores_required].isna().any().any():
        raise ValueError("stores: null metadata values")
    sample = tables["sample_submission"]
    if set(sample.columns) != {"id", "sales"} or sample["id"].isna().any() or sample["id"].duplicated().any():
        raise ValueError("sample_submission must contain unique id and sales columns")
    if len(sample) != len(tables["test"]) or set(sample["id"]) != set(tables["test"]["id"]):
        raise ValueError("sample_submission ids must match test ids exactly")


def build_base_panel(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Observed train + test rows, known calendar/promo/store covariates.

    Keep zeros as observations and test sales as NaN. Missing Christmas dates
    are not inserted here; model owners reindex within each series later,
    preserving an observed-target mask. Oil/transactions are excluded in v1.
    """
    _validate_keys(tables)
    train = tables["train"].copy()
    test = tables["test"].copy()
    train["is_test"], test["is_test"] = False, True
    train["sales_observed"], test["sales_observed"] = True, False
    test["sales"] = np.nan
    panel = pd.concat([train, test], ignore_index=True)
    if panel.duplicated(KEY).any():
        raise ValueError("train/test series-date keys overlap")
    panel = panel.merge(tables["stores"], on="store_nbr", how="left", validate="many_to_one", indicator=True)
    if panel["_merge"].ne("both").any():
        raise ValueError("Some store_nbr values have no stores metadata")
    panel = panel.drop(columns="_merge")
    panel = panel.merge(calendar_features(panel["date"]), on="date", how="left", validate="many_to_one")
    holidays = holiday_features(tables["holidays_events"], tables["stores"])
    panel = panel.merge(holidays, on=["date", "store_nbr"], how="left", validate="many_to_one")
    for name in HOLIDAY_COLUMNS:
        panel[name] = panel[name].eq(True)
    panel["is_workday"] = (~panel["is_weekend"] | panel["is_work_day"]) & ~panel["is_holiday"]
    panel["sales"] = panel["sales"].astype("float64")
    for name in ("family", "city", "state", "type"):
        if name in panel:
            panel[name] = panel[name].astype("category")
    panel = panel.sort_values("id").reset_index(drop=True)
    if len(panel) != len(train) + len(test) or panel["id"].duplicated().any():
        raise AssertionError("Base panel must preserve every source row exactly once")
    expected_ids = np.sort(np.concatenate([train["id"].to_numpy(), test["id"].to_numpy()]))
    if not np.array_equal(panel["id"].to_numpy(), expected_ids):
        raise AssertionError("Base panel ids differ from original source ids")
    if not panel.loc[panel["is_test"], "sales"].isna().all():
        raise AssertionError("Test rows must not acquire a sales label")
    expected_train = train[["id", "sales"]].sort_values("id").reset_index(drop=True)
    actual_train = panel.loc[~panel["is_test"], ["id", "sales"]].reset_index(drop=True)
    if (not np.array_equal(actual_train["id"].to_numpy(), expected_train["id"].to_numpy())
            or not np.array_equal(actual_train["sales"].to_numpy(), expected_train["sales"].to_numpy())):
        raise AssertionError("Training ids/targets changed during preparation")
    if not panel["sales_observed"].eq(~panel["is_test"]).all():
        raise AssertionError("Observed-target mask does not match source partition")
    return panel


def quality_report(tables: dict[str, pd.DataFrame]) -> dict:
    """Return measurements, including missing calendar dates and join risks."""
    report: dict = {"files": {}, "partitions": {}}
    for name, frame in tables.items():
        report["files"][name] = {
            "rows": len(frame), "columns": list(frame.columns),
            "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
            "nulls": {key: int(value) for key, value in frame.isna().sum().items()},
            "duplicate_rows": int(frame.duplicated().sum()),
        }
    for name in ("train", "test"):
        frame = tables[name]
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"].unique())).sort_values()
        missing = pd.date_range(dates.min(), dates.max()).difference(dates)
        partition = {
            "rows": len(frame), "date_min": dates.min().strftime("%Y-%m-%d"),
            "date_max": dates.max().strftime("%Y-%m-%d"), "observed_days": len(dates),
            "missing_dates": missing.strftime("%Y-%m-%d").tolist(),
            "stores": int(frame["store_nbr"].nunique()), "families": int(frame["family"].nunique()),
            "series": int(frame[["store_nbr", "family"]].drop_duplicates().shape[0]),
            "duplicate_keys": int(frame.duplicated(KEY).sum()), "duplicate_ids": int(frame["id"].duplicated().sum()),
        }
        daily_rows = frame.groupby("date", observed=True).size()
        expected_series = partition["stores"] * partition["families"]
        partition.update(
            expected_store_family_pairs=expected_series,
            pair_coverage_fraction=partition["series"] / expected_series,
            daily_rows_min=int(daily_rows.min()),
            daily_rows_max=int(daily_rows.max()),
            days_with_incomplete_pair_grid=int(daily_rows.ne(expected_series).sum()),
            null_required_values=int(frame[["id", *KEY, "onpromotion"]].isna().sum().sum()),
        )
        if name == "train":
            sales = frame["sales"]
            partition.update(zero_sales_rows=int(sales.eq(0).sum()), zero_sales_fraction=float(sales.eq(0).mean()),
                             fractional_sales_rows=int((sales.notna() & sales.mod(1).ne(0)).sum()))
        report["partitions"][name] = partition
    holidays, transactions = tables["holidays_events"], tables["transactions"]
    oil = tables["oil"]
    oil_unique = oil.drop_duplicates("date").set_index("date")["dcoilwtico"]
    oil_coverage = {}
    for name in ("train", "test"):
        dates = pd.DatetimeIndex(tables[name]["date"].unique()).sort_values()
        prices = oil_unique.reindex(dates)
        absent = ~dates.isin(oil_unique.index)
        oil_coverage[name] = {
            "sales_dates": len(dates),
            "dates_without_oil_row": int(absent.sum()),
            "dates_with_missing_oil_value": int(prices.isna().sum()),
        }
    sales_store_dates = tables["train"][["date", "store_nbr"]].drop_duplicates()
    transaction_keys = transactions[["date", "store_nbr"]].drop_duplicates()
    transaction_matches = sales_store_dates.merge(transaction_keys.assign(_matched=True), on=["date", "store_nbr"], how="left", validate="one_to_one")
    matched = int(transaction_matches["_matched"].eq(True).sum())
    unmatched_holidays = []
    stores = tables["stores"]
    for row in holidays.itertuples(index=False):
        if row.locale == "Regional" and not stores["state"].eq(row.locale_name).any():
            unmatched_holidays.append({"date": pd.Timestamp(row.date).strftime("%Y-%m-%d"), "locale": row.locale, "locale_name": str(row.locale_name)})
        elif row.locale == "Local" and not stores["city"].eq(row.locale_name).any():
            unmatched_holidays.append({"date": pd.Timestamp(row.date).strftime("%Y-%m-%d"), "locale": row.locale, "locale_name": str(row.locale_name)})
    report["auxiliary"] = {
        "holiday_unique_dates": int(holidays["date"].nunique()),
        "holiday_max_events_per_date": int(holidays.groupby("date").size().max()) if len(holidays) else 0,
        "holiday_duplicate_date_store_warning": "Aggregate events by date and applicable store before merge.",
        "holiday_unmatched_locality_rows": unmatched_holidays,
        "oil_missing_values": int(oil["dcoilwtico"].isna().sum()),
        "oil_duplicate_dates": int(oil["date"].duplicated().sum()),
        "oil_coverage": oil_coverage,
        "transactions_date_max": pd.Timestamp(transactions["date"].max()).strftime("%Y-%m-%d"),
        "transactions_available_during_test": bool(transactions["date"].ge(tables["test"]["date"].min()).any()),
        "transactions_duplicate_keys": int(transactions.duplicated(["date", "store_nbr"]).sum()),
        "transactions_matched_train_store_days": matched,
        "train_store_days": len(sales_store_dates),
        "transactions_train_coverage_fraction": matched / len(sales_store_dates),
    }
    test_pairs = pd.MultiIndex.from_frame(tables["test"][["store_nbr", "family"]].drop_duplicates())
    train_pairs = pd.MultiIndex.from_frame(tables["train"][["store_nbr", "family"]].drop_duplicates())
    sample = tables["sample_submission"]
    report["checks"] = {
        "train_test_id_disjoint": not tables["train"]["id"].isin(tables["test"]["id"]).any(),
        "test_after_train": bool(tables["test"]["date"].min() > tables["train"]["date"].max()),
        "test_starts_next_day": bool(tables["test"]["date"].min() == tables["train"]["date"].max() + pd.Timedelta(days=1)),
        "sample_submission_same_ids": bool(len(sample) == len(tables["test"]) and not sample["id"].duplicated().any() and set(sample["id"]) == set(tables["test"]["id"])),
        "sample_submission_same_id_order": bool(sample["id"].reset_index(drop=True).equals(tables["test"]["id"].reset_index(drop=True))),
        "test_pairs_seen_in_train": bool(test_pairs.isin(train_pairs).all()),
        "test_has_16_observed_days": report["partitions"]["test"]["observed_days"] == 16,
        "test_calendar_contiguous": not report["partitions"]["test"]["missing_dates"],
    }
    report["notes"] = [
        "sales là lượng bán, có thể có phần thập phân; onpromotion là số mặt hàng khuyến mãi.",
        "Ngày thiếu khác số 0 thật. Không điền sales test hoặc ngày không quan sát thành 0.",
        "Ghép stores nhiều-một; tổng hợp lễ theo ngày và phạm vi áp dụng của từng cửa hàng trước khi merge.",
        "Giữ nguyên giá trị nguồn; bảng nền chưa có lag/scaler/cửa sổ chuỗi. TV1 tạo features lịch sử riêng bằng scripts/prepare_features.py; chủ model xử lý encoding/scaling/chuỗi theo mốc.",
        "Bảng nền chưa ghép oil/transactions. Giá dầu thiếu ngày/giá trị; giao dịch thực của ngày tương lai chưa biết.",
    ]
    return report


def write_quality_report(report: dict, report_dir: Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Báo cáo chất lượng dữ liệu TV1", "", "Tính trực tiếp từ 7 CSV trong data/raw; không sửa tệp gốc.", "",
             "| Tệp | Số dòng | Số cột | Giá trị thiếu | Dòng trùng hoàn toàn |",
             "|---|---:|---:|---:|---:|"]
    for name, item in report["files"].items():
        lines.append(f"| {name}.csv | {item['rows']:,} | {len(item['columns'])} | {sum(item['nulls'].values()):,} | {item['duplicate_rows']:,} |")
    lines.extend(["", "## Phạm vi bán hàng", "", "| Phần dữ liệu | Số dòng | Ngày đầu | Ngày cuối | Cửa hàng | Nhóm hàng | Chuỗi |",
             "|---|---:|---|---|---:|---:|---:|"]
    )
    missing_notes = []
    for name, part in report["partitions"].items():
        lines.append(f"| {name} | {part['rows']:,} | {part['date_min']} | {part['date_max']} | {part['stores']} | {part['families']} | {part['series']} |")
        if part["missing_dates"]:
            missing_notes.append(f"- Ngày lịch không có dòng trong {name}: {', '.join(part['missing_dates'])}.")
        missing_notes.append(f"- {name}: {part['duplicate_keys']} khóa trùng, {part['duplicate_ids']} id trùng; {part['daily_rows_min']:,}-{part['daily_rows_max']:,} dòng mỗi ngày quan sát; {part['days_with_incomplete_pair_grid']} ngày thiếu lưới cửa hàng × nhóm hàng.")
    lines.extend(["", *missing_notes])
    train = report["partitions"]["train"]
    lines.extend(["", f"Train có {train['zero_sales_rows']:,} dòng sales = 0 ({train['zero_sales_fraction']:.2%}) và {train['fractional_sales_rows']:,} dòng có phần thập phân. Giữ nguyên các giá trị này.", "", "## Kiểm tra đối chiếu", ""])
    for name, passed in report["checks"].items():
        lines.append(f"- {name}: {'đạt' if passed else 'cần kiểm tra'}.")
    aux = report["auxiliary"]
    lines.extend(["", "## Bảng phụ và quyết định xử lý", "",
        f"- Ngày lễ: {aux['holiday_unique_dates']} ngày có {report['files']['holidays_events']['rows']} sự kiện, tối đa {aux['holiday_max_events_per_date']} sự kiện/ngày. Có {len(aux['holiday_unmatched_locality_rows'])} dòng địa phương không khớp cửa hàng hiện có.",
        "- Lễ gốc đã chuyển ngày không được đánh dấu là ngày nghỉ; giữ cờ riêng và dùng dòng Transfer ở ngày thực tế. National áp dụng toàn quốc, Regional theo state, Local theo city. Không nhân bản dòng bán hàng.",
        f"- Oil có {aux['oil_missing_values']} ô giá thiếu và {aux['oil_duplicate_dates']} ngày trùng. Chưa điền hay ghép giá dầu vào bảng nền.",
    ])
    for name, coverage in aux["oil_coverage"].items():
        lines.append(f"- Oil đối chiếu {name}: {coverage['dates_without_oil_row']} ngày bán hàng không có dòng oil; tổng cộng {coverage['dates_with_missing_oil_value']} ngày không có giá dùng được (bao gồm dòng thiếu giá).")
    lines.extend([
        f"- Transactions kết thúc {aux['transactions_date_max']}; phủ {aux['transactions_matched_train_store_days']:,}/{aux['train_store_days']:,} cặp cửa hàng-ngày train ({aux['transactions_train_coverage_fraction']:.2%}); {aux['transactions_duplicate_keys']} khóa trùng. Chưa ghép giao dịch thực tương lai vào bảng nền.",
        "", "## Kiểu dữ liệu sau khi đọc", "",
    ])
    for name, item in report["files"].items():
        lines.append(f"- {name}.csv: " + ", ".join(f"{key}={value}" for key, value in item["dtypes"].items()) + ".")
    lines.extend(["", "## Nội dung bàn giao", "", *[f"- {note}" for note in report["notes"]]])
    (report_dir / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
