"""Shared RMSLE scoring with explicit ID alignment for forecast comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmsle(actual, predicted) -> float:
    """Compute RMSLE for equally sized, finite, nonnegative 1-D vectors.

    Negative predictions are rejected rather than silently changed. A model
    that needs clipping or inverse transformation must do so before scoring.
    """
    try:
        actual_array = np.asarray(actual, dtype=np.float64)
        predicted_array = np.asarray(predicted, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("RMSLE inputs must be numeric 1-D vectors") from error
    if actual_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError("RMSLE inputs must be 1-D vectors")
    if actual_array.size == 0 or actual_array.shape != predicted_array.shape:
        raise ValueError("RMSLE inputs must be nonempty and have the same length")
    if not np.isfinite(actual_array).all() or not np.isfinite(predicted_array).all():
        raise ValueError("RMSLE inputs must contain only finite values")
    if (actual_array < 0).any() or (predicted_array < 0).any():
        raise ValueError("RMSLE inputs must be nonnegative")
    difference = np.log1p(predicted_array) - np.log1p(actual_array)
    return float(np.sqrt(np.mean(difference * difference)))


def evaluate_by_id(actual: pd.DataFrame, predicted: pd.DataFrame) -> dict:
    """Score exact matching forecast IDs, independently of row order.

    actual requires ``id,sales``; predicted requires ``id,sales_pred``.
    Missing, extra, duplicate or null IDs are errors, not rows to discard.
    """
    for name, frame, required in (
        ("actual", actual, {"id", "sales"}),
        ("predicted", predicted, {"id", "sales_pred"}),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{name} must be a DataFrame")
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} requires columns {sorted(required)}")
        if frame.empty:
            raise ValueError(f"{name} must not be empty")
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError(f"{name} IDs must be present and unique")
    if len(actual) != len(predicted) or set(actual["id"]) != set(predicted["id"]):
        raise ValueError("Predicted IDs must match actual IDs exactly")
    aligned = actual[["id", "sales"]].merge(
        predicted[["id", "sales_pred"]],
        on="id", how="left", sort=False, validate="one_to_one",
    )
    return {"rmsle": rmsle(aligned["sales"], aligned["sales_pred"]), "n_rows": len(aligned)}
