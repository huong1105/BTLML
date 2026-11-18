"""Fixed-origin historical features and reproducible TV1 handoff datasets.

History is indexed by calendar date and store/family, never by adjacent rows.
Targets retain only known covariates; validation labels are stored separately.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import json
import platform
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .calendar import HOLIDAY_COLUMNS
from .data import sha256_file

SERIES_KEYS = ["store_nbr", "family"]
WINDOWS = (7, 14, 28)
LAGS = (0, 7, 14, 28)
LAG_NAMES = {0: "sales_at_origin", **{n: f"sales_origin_lag_{n}" for n in LAGS if n}}
HISTORY_COLUMNS = [*LAG_NAMES.values(),
                   *[f"sales_mean_{n}" for n in WINDOWS],
                   *[f"sales_std_{n}" for n in WINDOWS], "sales_zero_rate_28",
                   *[f"sales_observed_days_{n}" for n in WINDOWS],
                   *[f"{name}_missing" for name in LAG_NAMES.values()]]
KNOWN_COLUMNS = [*SERIES_KEYS, "onpromotion", "city", "state", "type", "cluster",
                 "year", "month", "day", "dayofweek", "is_weekend", "is_payday",
                 *HOLIDAY_COLUMNS, "is_workday"]
DEFAULT_STAGES = {"tuning": "2017-07-14", "validation": "2017-07-30", "test": "2017-08-15"}


def calendar_day(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
        raise ValueError("Origin/cutoff must be a calendar date without time or timezone")
    return value


class HistoryFeatureBuilder:
    """Reuse a date/series sales matrix for multiple forecast origins."""

    def __init__(self, panel: pd.DataFrame):
        required = {"id", "date", *SERIES_KEYS, "sales", "is_test", "sales_observed"}
        if not required.issubset(panel.columns):
            raise ValueError(f"Panel missing columns: {sorted(required - set(panel.columns))}")
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"], errors="raise")
        if panel.empty or panel[["id", "date", *SERIES_KEYS]].isna().any().any():
            raise ValueError("Panel must have nonempty, complete keys")
        if panel["date"].dt.tz is not None or panel["date"].ne(panel["date"].dt.normalize()).any():
            raise ValueError("Panel dates must be calendar dates without timezone")
        if panel["id"].duplicated().any() or panel.duplicated(["date", *SERIES_KEYS]).any():
            raise ValueError("Panel IDs and series/date keys must be unique")
        for column in ("is_test", "sales_observed"):
            if not pd.api.types.is_bool_dtype(panel[column]) or panel[column].isna().any():
                raise ValueError(f"{column} must contain nonmissing booleans")
        if not panel["sales_observed"].eq(~panel["is_test"]).all():
            raise ValueError("Source/observed masks disagree")
        actual = panel.loc[panel["sales_observed"], "sales"]
        if actual.empty or actual.isna().any() or not np.isfinite(actual).all() or actual.lt(0).any():
            raise ValueError("Observed sales must be finite and nonnegative")
        if panel.loc[panel["is_test"], "sales"].notna().any():
            raise ValueError("Test must not have sales labels")
        observed = panel.loc[panel["sales_observed"]]
        if panel["is_test"].any() and observed["date"].max() >= panel.loc[panel["is_test"], "date"].min():
            raise ValueError("Test must follow training dates")
        self.pairs = panel[SERIES_KEYS].drop_duplicates().reset_index(drop=True)
        self.pair_index = pd.MultiIndex.from_frame(self.pairs)
        self.sales = observed.pivot(index="date", columns=SERIES_KEYS, values="sales").reindex(columns=self.pair_index).sort_index()
        self.panel = panel.sort_values(["date", "id"]).set_index("date", drop=False)
        self.panel.index.name = "calendar_index"
        self.covariates = [c for c in KNOWN_COLUMNS if c in panel.columns]
        self.feature_columns = [*self.covariates, "horizon", *HISTORY_COLUMNS]

    def snapshot(self, origin) -> pd.DataFrame:
        """One row per series; windows include origin, std uses ddof=0.

        A missing date stays NaN. Means/std/zero fractions use observed days;
        counts and exact-lag masks describe the missingness explicitly.
        """
        origin = calendar_day(origin)
        dates = pd.date_range(origin - pd.Timedelta(days=28), origin)
        values = self.sales.reindex(dates).to_numpy(dtype="float64")
        result = self.pairs.copy()
        for lag, name in LAG_NAMES.items():
            result[name] = values[-lag - 1]
        means, deviations, counts = {}, {}, {}
        for window in WINDOWS:
            a = values[-window:]
            count = np.isfinite(a).sum(axis=0)
            mean = np.divide(np.nansum(a, axis=0), count,
                             out=np.full(len(count), np.nan), where=count > 0)
            variance = np.divide(np.nansum((a - mean) ** 2, axis=0), count,
                                 out=np.full(len(count), np.nan), where=count > 0)
            means[window], deviations[window], counts[window] = mean, np.sqrt(variance), count
        for window in WINDOWS:
            result[f"sales_mean_{window}"] = means[window]
        for window in WINDOWS:
            result[f"sales_std_{window}"] = deviations[window]
        result["sales_zero_rate_28"] = np.divide((values[-28:] == 0).sum(axis=0), counts[28],
                                                  out=np.full(len(counts[28]), np.nan), where=counts[28] > 0)
        for window in WINDOWS:
            result[f"sales_observed_days_{window}"] = counts[window].astype("int16")
        for name in LAG_NAMES.values():
            result[f"{name}_missing"] = result[name].isna()
        return result

    def targets(self, origin, horizon_days: int = 16) -> pd.DataFrame:
        origin = calendar_day(origin)
        if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days < 1:
            raise ValueError("horizon_days must be a positive integer")
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon_days)
        targets = self.panel.loc[dates[0]:dates[-1]].copy()
        counts = targets.groupby("date", observed=True).size().reindex(dates, fill_value=0)
        if not counts.eq(len(self.pairs)).all():
            raise ValueError("Forecast horizon must contain every series on every calendar day")
        return targets.sort_values("id").reset_index(drop=True)

    def build(self, origin, horizon_days: int = 16) -> pd.DataFrame:
        """Return features in original ID order; never include target sales/masks."""
        origin = calendar_day(origin)
        targets = self.targets(origin, horizon_days)
        frame = targets[["id", "date", *self.covariates]].copy()
        frame["origin"] = origin
        frame["horizon"] = (frame["date"] - origin).dt.days.astype("int16")
        frame = frame.merge(self.snapshot(origin), on=SERIES_KEYS, how="left", validate="many_to_one", sort=False)
        if len(frame) != len(targets) or not frame["id"].equals(targets["id"]):
            raise AssertionError("Feature join changed target IDs/order")
        return frame[["id", "date", "origin", *self.feature_columns]]

    def labels(self, origin, horizon_days: int = 16) -> pd.DataFrame:
        targets = self.targets(origin, horizon_days)
        if not targets["sales_observed"].all():
            raise ValueError("Labels unavailable for this horizon")
        return targets[["id", "sales"]].copy()


def training_origins(first_date, cutoff, horizon_days: int = 16, step_days: int = 16) -> list[pd.Timestamp]:
    """Calendar origins whose ENTIRE label horizon is known at cutoff."""
    for name, value in (("horizon_days", horizon_days), ("step_days", step_days)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    first = calendar_day(first_date) + pd.Timedelta(days=28)
    last = calendar_day(cutoff) - pd.Timedelta(days=horizon_days)
    if first > last:
        return []
    return list(reversed(pd.date_range(start=last, periods=(last - first).days // step_days + 1,
                                      freq=f"-{step_days}D")))


def prepare_feature_datasets(panel_path: Path, output_dir: Path, report_dir: Path,
                             stages: dict | None = None, horizon_days: int = 16,
                             step_days: int = 16, progress=None) -> dict:
    """Stream historical training blocks to separate, cutoff-safe stage files."""
    panel_path, output_dir, report_dir = map(Path, (panel_path, output_dir, report_dir))
    stage_days = {name: calendar_day(day) for name, day in (DEFAULT_STAGES if stages is None else stages).items()}
    if set(stage_days) != {"tuning", "validation", "test"}:
        raise ValueError("Stages must be tuning, validation and test")
    if not stage_days["tuning"] < stage_days["validation"] < stage_days["test"]:
        raise ValueError("Stage origins must increase")
    before = sha256_file(panel_path)
    builder = HistoryFeatureBuilder(pd.read_parquet(panel_path))
    report = {"complete": False, "source_file": panel_path.name, "source_sha256": before, "horizon_days": horizon_days,
              "training_step_days": step_days, "std_ddof": 0, "windows_include_origin": True,
              "missing_policy": "NaN retained; statistics/zero rates use observed days only; no label imputation",
              "feature_columns": builder.feature_columns, "metadata_columns": ["id", "date", "origin"],
              "history_columns": HISTORY_COLUMNS, "stages": {},
              "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pa.__version__}}
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    partials: list[tuple[Path, Path]] = []
    job_id = uuid.uuid4().hex

    def destination(stage, name):
        folder = output_dir / stage
        folder.mkdir(parents=True, exist_ok=True)
        final = folder / name
        partial = folder / f".{name}.{job_id}.partial"
        partials.append((partial, final))
        return partial

    def write_frame(stage, name, frame):
        frame.to_parquet(destination(stage, name), index=False, engine="pyarrow")

    try:
        with ExitStack() as stack:
            writers = {}
            for name, origin in stage_days.items():
                frame = builder.build(origin, horizon_days)
                if "feature_schema" not in report:
                    report["feature_schema"] = [{"name": field.name, "type": str(field.type)}
                                                for field in pa.Table.from_pandas(frame, preserve_index=False).schema]
                targets = builder.targets(origin, horizon_days)
                if name == "test":
                    if not targets["is_test"].all():
                        raise ValueError("Test stage must refer only to original test rows")
                elif not targets["sales_observed"].all():
                    raise ValueError("Validation stages require observed labels")
                write_frame(name, "features.parquet", frame)
                if name != "test":
                    write_frame(name, "labels.parquet", builder.labels(origin, horizon_days))
                report["stages"][name] = {
                    "origin": str(origin.date()), "forecast_start": str((origin + pd.Timedelta(days=1)).date()),
                    "forecast_end": str((origin + pd.Timedelta(days=horizon_days)).date()),
                    "forecast_rows": len(frame), "train_rows": 0, "train_origins": [], "skipped_origins": [],
                    "history_missing_cells": {c: int(frame[c].isna().sum()) for c in HISTORY_COLUMNS},
                }
                # Build backwards from each cutoff, including its latest eligible origin.
                origins = training_origins(builder.sales.index.min(), origin, horizon_days, step_days)
                report["stages"][name]["candidate_origins"] = [str(d.date()) for d in origins]
                writers[name] = (destination(name, "train.parquet"), None)
            all_origins = sorted({pd.Timestamp(day) for stage in report["stages"].values() for day in stage["candidate_origins"]})
            for number, origin in enumerate(all_origins, 1):
                eligible = [name for name in stage_days if str(origin.date()) in report["stages"][name]["candidate_origins"]]
                try:
                    labels = builder.labels(origin, horizon_days)
                    frame = builder.build(origin, horizon_days)
                except ValueError as error:
                    if "every series on every calendar day" not in str(error):
                        raise
                    for name in eligible:
                        report["stages"][name]["skipped_origins"].append(str(origin.date()))
                    continue
                frame["sales"] = labels["sales"].to_numpy(dtype="float64")
                table = pa.Table.from_pandas(frame, preserve_index=False)
                for name in eligible:
                    if origin + pd.Timedelta(days=horizon_days) > stage_days[name]:
                        raise AssertionError("Training labels exceed stage cutoff")
                    path, writer = writers[name]
                    if writer is None:
                        writer = stack.enter_context(pq.ParquetWriter(path, table.schema, compression="snappy"))
                        writers[name] = (path, writer)
                    writer.write_table(table)
                    report["stages"][name]["train_rows"] += len(frame)
                    report["stages"][name]["train_origins"].append(str(origin.date()))
                if progress is not None and (number % 10 == 0 or number == len(all_origins)):
                    progress(f"Historical origins: {number}/{len(all_origins)}; latest={origin.date()}")
            if any(stage["train_rows"] == 0 for stage in report["stages"].values()):
                raise ValueError("Not enough observed history to create training samples for every stage")
        if sha256_file(panel_path) != before:
            raise RuntimeError("Base panel changed during feature generation")
        report["files"] = {}
        for partial, final in partials:
            meta = pq.read_metadata(partial)
            report["files"][final.relative_to(output_dir).as_posix()] = {
                "rows": meta.num_rows, "columns": meta.num_columns, "sha256": sha256_file(partial)}
        for partial, final in partials:
            partial.replace(final)
        report["complete"] = True
        (report_dir / "features_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        lines = ["# Bàn giao features lịch sử TV1", "", "Nguồn: bảng nền Parquet; CSV và nhãn gốc không thay đổi.", "",
                 "Mỗi dòng học/dự báo có khóa (origin, id). Features lịch sử cố định tại origin; horizon là số ngày dự báo trước.",
                 "Độ lệch chuẩn chia N (ddof=0); cửa sổ gồm origin. Ngày thiếu giữ NaN; trung bình/tỷ lệ số 0 chỉ tính trên ngày đã quan sát.",
                 "Tập features kiểm tra/test không chứa sales, is_test hay sales_observed; đáp án kiểm tra lưu riêng.", "",
                 "| Bộ | Origin | Dòng học | Dòng dự báo | Số mốc học bỏ qua do thiếu ngày đích |", "|---|---|---:|---:|---:|"]
        for name, info in report["stages"].items():
            lines.append(f"| {name} | {info['origin']} | {info['train_rows']:,} | {info['forecast_rows']:,} | {len(info['skipped_origins'])} |")
        lines += ["", f"Mốc học cách nhau {step_days} ngày, chỉ dùng các khoảng có đủ ngày/chuỗi và toàn bộ đáp án không vượt cutoff của từng bộ.",
                  "Các cột đầu vào được liệt kê trong features_manifest.json; id/date/origin là metadata, sales là đáp án của train.parquet.",
                  "TV2 tự chọn/mã hóa đầu vào; TV3 dùng bảng nền để tạo chuỗi; TV4 quản lý phép chấm và XAI."]
        (report_dir / "features_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report
    finally:
        for partial, _ in partials:
            if partial.exists():
                partial.unlink()
