"""Small correctness tests: temporal isolation, hand-derived means and ID scoring."""

from pathlib import Path
import math
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from store_sales.baseline import predict_same_weekday
from store_sales.metrics import evaluate_by_id, rmsle


class MetricTests(unittest.TestCase):
    def test_known_rmsle_and_zero_error(self):
        self.assertAlmostEqual(rmsle([0, 3], [3, 0]), math.log(4))
        self.assertEqual(rmsle([0, 0.5, 9], [0, 0.5, 9]), 0.0)

    def test_invalid_vectors_rejected(self):
        cases = [([], []), ([1], [1, 2]), ([[1]], [[1]]),
                 ([-1], [0]), ([0], [-1]), ([np.nan], [0]),
                 ([0], [np.inf]), (["not numeric"], [0])]
        for actual, predicted in cases:
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    rmsle(actual, predicted)

    def test_id_alignment_does_not_score_by_row_position(self):
        actual = pd.DataFrame({"id": [10, 20], "sales": [0.0, 3.0]})
        predicted = pd.DataFrame({"id": [20, 10], "sales_pred": [3.0, 0.0]})
        self.assertEqual(evaluate_by_id(actual, predicted), {"rmsle": 0.0, "n_rows": 2})

    def test_missing_extra_duplicate_or_null_ids_rejected(self):
        actual = pd.DataFrame({"id": [10, 20], "sales": [0.0, 3.0]})
        bad_ids = [[10], [10, 20, 30], [10, 10], [10, 30], [10, None]]
        for ids in bad_ids:
            with self.subTest(ids=ids):
                predicted = pd.DataFrame({"id": ids, "sales_pred": [0.0] * len(ids)})
                with self.assertRaises(ValueError):
                    evaluate_by_id(actual, predicted)
        duplicate_actual = pd.DataFrame({"id": [10, 10], "sales": [0.0, 3.0]})
        with self.assertRaises(ValueError):
            evaluate_by_id(duplicate_actual, pd.DataFrame({"id": [10, 20], "sales_pred": [0.0, 3.0]}))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.origin = pd.Timestamp("2024-01-28")
        # Four Mondays in the 28-day window have values 0,4,8,12 -> mean 6.
        # Sunday at origin contributes 16, giving all-series mean (0+4+8+12+16)/5=8.
        # The older Monday (999) and future truth must not influence forecasts.
        self.history = pd.DataFrame({
            "id": range(1, 8),
            "date": pd.to_datetime(["2023-12-25", "2024-01-01", "2024-01-08",
                                    "2024-01-15", "2024-01-22", "2024-01-28", "2024-01-29"]),
            "store_nbr": [1] * 7, "family": ["A"] * 7,
            "sales": [999.0, 0.0, 4.0, 8.0, 12.0, 16.0, 9000.0],
        })

    def targets(self, dates, ids=None, stores=None):
        count = len(dates)
        return pd.DataFrame({"id": ids if ids is not None else range(100, 100 + count),
                             "date": pd.to_datetime(dates),
                             "store_nbr": stores if stores is not None else [1] * count,
                             "family": ["A"] * count})

    def test_mean_includes_zero_and_excludes_old_or_future_rows(self):
        result = predict_same_weekday(self.history, self.targets(["2024-01-29"]), self.origin)
        self.assertEqual(result["sales_pred"].tolist(), [6.0])
        self.assertEqual(result["prediction_source"].tolist(), ["same_weekday"])

    def test_entire_horizon_uses_fixed_origin_and_ignores_target_sales(self):
        target = self.targets(pd.date_range("2024-01-29", periods=16))
        target["sales"] = np.arange(16) * 10000.0
        first = predict_same_weekday(self.history, target, self.origin)
        changed = self.history.copy(deep=True)
        changed.loc[changed["date"] > self.origin, "sales"] = 1e12
        changed_target = target.assign(sales=1e15)
        second = predict_same_weekday(changed, changed_target, self.origin)
        pd.testing.assert_frame_equal(first, second)
        # Week 2 and week 3 Mondays still use the four historical Mondays.
        monday = first["date"].isin(pd.to_datetime(["2024-01-29", "2024-02-05", "2024-02-12"]))
        self.assertEqual(first.loc[monday, "sales_pred"].tolist(), [6.0, 6.0, 6.0])

    def test_target_order_and_inputs_preserved(self):
        target = self.targets(["2024-02-12", "2024-01-29", "2024-02-05"], ids=[30, 10, 20])
        before_history, before_target = self.history.copy(deep=True), target.copy(deep=True)
        result = predict_same_weekday(self.history, target, self.origin)
        self.assertEqual(result["id"].tolist(), [30, 10, 20])
        pd.testing.assert_frame_equal(self.history, before_history)
        pd.testing.assert_frame_equal(target, before_target)

    def test_fallbacks_for_unseen_weekday_and_unseen_series(self):
        target = self.targets(["2024-01-31", "2024-01-31"], stores=[1, 99])
        result = predict_same_weekday(self.history, target, self.origin)
        self.assertEqual(result["sales_pred"].tolist(), [8.0, 0.0])
        self.assertEqual(result["prediction_source"].tolist(), ["series_mean", "zero_no_history"])

    def test_invalid_forecast_dates_and_parameters_rejected(self):
        for date in ("2024-01-28", "2024-01-27", "2024-02-14"):
            with self.subTest(date=date):
                with self.assertRaises(ValueError):
                    predict_same_weekday(self.history, self.targets([date]), self.origin)
        for name in ("window_days", "horizon_days"):
            with self.subTest(parameter=name):
                with self.assertRaises(ValueError):
                    predict_same_weekday(self.history, self.targets(["2024-01-29"]), self.origin, **{name: 0})


if __name__ == "__main__":
    unittest.main()
