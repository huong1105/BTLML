"""Calendar and store-scoped holiday features; no sales-derived features."""
from __future__ import annotations

import pandas as pd


HOLIDAY_COLUMNS = [
    "is_holiday", "is_national_holiday", "is_regional_holiday",
    "is_local_holiday", "is_transfer_day", "is_bridge_day",
    "is_additional_day", "is_work_day", "is_event",
    "is_transferred_original",
]


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """One row per observed date. No implicit insertion of missing sales dates."""
    frame = pd.DataFrame({"date": pd.to_datetime(dates).drop_duplicates()})
    frame = frame.sort_values("date").reset_index(drop=True)
    date = frame["date"].dt
    frame["year"] = date.year.astype("int16")
    frame["month"] = date.month.astype("int8")
    frame["day"] = date.day.astype("int8")
    frame["dayofweek"] = date.dayofweek.astype("int8")
    frame["is_weekend"] = date.dayofweek.ge(5)
    frame["is_payday"] = date.day.eq(15) | date.is_month_end
    return frame


def holiday_features(holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate holiday flags to (date, store_nbr) before joining sales.

    A transferred original date is recorded but is not treated as a holiday.
    Regional/local events apply only to matching state/city, respectively.
    """
    records: list[dict] = []
    for row in holidays.itertuples(index=False):
        if pd.isna(row.date) or pd.isna(row.transferred):
            raise ValueError("Holiday date and transferred must be present")
        locale = str(row.locale)
        if locale == "National":
            applicable = stores
        elif locale == "Regional":
            applicable = stores.loc[stores["state"].eq(row.locale_name)]
        elif locale == "Local":
            applicable = stores.loc[stores["city"].eq(row.locale_name)]
        else:
            raise ValueError(f"Unknown holiday locale: {locale!r}")
        raw_transferred = row.transferred
        if isinstance(raw_transferred, str) and raw_transferred.lower() not in {"true", "false"}:
            raise ValueError(f"Invalid transferred value: {raw_transferred!r}")
        transferred = (
            str(raw_transferred).lower() == "true"
            if isinstance(raw_transferred, str) else bool(raw_transferred)
        )
        kind = str(row.type)
        if kind not in {"Holiday", "Transfer", "Bridge", "Additional", "Work Day", "Event"}:
            raise ValueError(f"Unknown holiday type: {kind!r}")
        actual_holiday = kind in {"Holiday", "Transfer", "Bridge", "Additional"} and not transferred
        for store in applicable["store_nbr"]:
            flags = dict.fromkeys(HOLIDAY_COLUMNS, False)
            flags["is_holiday"] = actual_holiday
            flags[f"is_{locale.lower()}_holiday"] = actual_holiday
            flags["is_transfer_day"] = kind == "Transfer" and not transferred
            flags["is_bridge_day"] = kind == "Bridge" and not transferred
            flags["is_additional_day"] = kind == "Additional" and not transferred
            flags["is_work_day"] = kind == "Work Day" and not transferred
            flags["is_event"] = kind == "Event" and not transferred
            flags["is_transferred_original"] = transferred
            records.append({"date": pd.Timestamp(row.date), "store_nbr": store, **flags})
    columns = ["date", "store_nbr", *HOLIDAY_COLUMNS]
    if not records:
        return pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            "store_nbr": pd.Series(dtype=stores["store_nbr"].dtype),
            **{name: pd.Series(dtype=bool) for name in HOLIDAY_COLUMNS},
        })[columns]
    return pd.DataFrame(records).groupby(["date", "store_nbr"], as_index=False)[HOLIDAY_COLUMNS].max()
