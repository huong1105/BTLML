"""Hand-calculated calendar features, temporal isolation and handoff contracts."""
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from store_sales.features import HistoryFeatureBuilder, prepare_feature_datasets, training_origins


def sample_panel():
    # 29 history dates; store 2 makes cross-series mistakes easy to detect.
    values = [5] + [10] * 7 + [20] * 7 + [30] * 7 + [0, 20, 20, 20, 20, 20, 40]
    rows = []
    for i, day in enumerate(pd.date_range("2017-07-02", "2017-08-31")):
        for store in (1, 2):
            is_test = day > pd.Timestamp("2017-08-15")
            sales = (values[i] if i < len(values) else 5000) * store
            rows.append({"id": len(rows), "date": day, "store_nbr": store, "family": "A",
                         "sales": np.nan if is_test else sales, "is_test": is_test,
                         "sales_observed": not is_test, "onpromotion": i % 3,
                         "dayofweek": day.dayofweek})
    return pd.DataFrame(rows)


class HistoryFeatureTests(unittest.TestCase):
    def setUp(self):
        self.panel = sample_panel()
        self.origin = pd.Timestamp("2017-07-30")

    def test_hand_calculated_lags_means_std_and_zeros(self):
        snapshot = HistoryFeatureBuilder(self.panel).snapshot(self.origin).set_index("store_nbr")
        row = snapshot.loc[1]
        self.assertEqual([row["sales_at_origin"], row["sales_origin_lag_7"], row["sales_origin_lag_14"], row["sales_origin_lag_28"]], [40, 30, 20, 5])
        self.assertEqual([row[f"sales_mean_{n}"] for n in (7, 14, 28)], [20, 25, 20])
        self.assertAlmostEqual(row["sales_std_7"], math.sqrt(800 / 7))
        self.assertAlmostEqual(row["sales_std_14"], math.sqrt(1150 / 14))
        self.assertAlmostEqual(row["sales_std_28"], math.sqrt(2200 / 28))
        self.assertAlmostEqual(row["sales_zero_rate_28"], 1 / 28)
        self.assertEqual(row["sales_observed_days_28"], 28)
        self.assertEqual(snapshot.loc[2, "sales_at_origin"], 80)
        self.assertAlmostEqual(snapshot.loc[2, "sales_std_7"], 2 * row["sales_std_7"])

    def test_missing_calendar_date_is_not_zero_or_previous_row(self):
        missing = self.panel.loc[~((self.panel.date == "2017-07-23") & (self.panel.store_nbr == 1))]
        row = HistoryFeatureBuilder(missing).snapshot(self.origin).set_index("store_nbr").loc[1]
        self.assertTrue(pd.isna(row["sales_origin_lag_7"]))
        self.assertTrue(row["sales_origin_lag_7_missing"])
        self.assertEqual(row["sales_observed_days_28"], 27)
        self.assertAlmostEqual(row["sales_mean_28"], 530 / 27)
        self.assertAlmostEqual(row["sales_zero_rate_28"], 1 / 27)

    def test_future_labels_do_not_change_any_feature_for_whole_horizon(self):
        before = self.panel.copy(deep=True)
        first = HistoryFeatureBuilder(self.panel).build(self.origin)
        changed = self.panel.copy()
        changed.loc[(changed.date > self.origin) & changed.sales_observed, "sales"] = 1e12
        second = HistoryFeatureBuilder(changed).build(self.origin)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(before, self.panel)
        self.assertFalse({"sales", "log_sales", "is_test", "sales_observed"} & set(first.columns))
        for _, group in first.groupby("store_nbr"):
            self.assertEqual(group["sales_at_origin"].nunique(), 1)
            self.assertEqual(group["horizon"].tolist(), list(range(1, 17)))

    def test_unordered_input_preserves_source_ids(self):
        builder = HistoryFeatureBuilder(self.panel.sample(frac=1, random_state=42))
        frame = builder.build(self.origin)
        expected = self.panel.loc[self.panel.date.between("2017-07-31", "2017-08-15"), "id"].tolist()
        self.assertEqual(frame.id.tolist(), expected)
        self.assertEqual(len(frame), 32)

    def test_no_history_and_short_history_are_explicit(self):
        builder = HistoryFeatureBuilder(self.panel)
        row = builder.snapshot("2017-06-30").iloc[0]
        self.assertTrue(pd.isna(row.sales_mean_28))
        self.assertTrue(pd.isna(row.sales_zero_rate_28))
        self.assertEqual(row.sales_observed_days_28, 0)
        self.assertTrue(row.sales_at_origin_missing)
        row = builder.snapshot("2017-07-02").iloc[0]
        self.assertEqual(row.sales_observed_days_7, 1)
        self.assertEqual(row.sales_mean_7, 5)
        self.assertEqual(row.sales_std_7, 0)

    def test_test_features_have_no_labels_and_missing_targets_raise(self):
        builder = HistoryFeatureBuilder(self.panel)
        frame = builder.build("2017-08-15")
        self.assertEqual(len(frame), 32)
        self.assertNotIn("sales", frame)
        with self.assertRaises(ValueError):
            builder.labels("2017-08-15")
        incomplete = self.panel.loc[self.panel.id != 80]
        with self.assertRaises(ValueError):
            HistoryFeatureBuilder(incomplete).build(self.origin)

    def test_bad_keys_labels_masks_and_origin_raise(self):
        bad = pd.concat([self.panel, self.panel.iloc[:1]])
        with self.assertRaises(ValueError):
            HistoryFeatureBuilder(bad)
        for column, value in (("sales", -1), ("sales_observed", False)):
            bad = self.panel.copy()
            bad.loc[0, column] = value
            with self.assertRaises(ValueError):
                HistoryFeatureBuilder(bad)
        with self.assertRaises(ValueError):
            HistoryFeatureBuilder(self.panel).snapshot("2017-07-30 12:00")


class HandoffTests(unittest.TestCase):
    def test_training_origins_end_before_cutoff_and_have_warmup(self):
        origins = training_origins("2013-01-01", "2017-07-14")
        self.assertEqual(origins[-1], pd.Timestamp("2017-06-28"))
        self.assertGreaterEqual(origins[0], pd.Timestamp("2013-01-29"))
        self.assertTrue(all(o + pd.Timedelta(days=16) <= pd.Timestamp("2017-07-14") for o in origins))
        self.assertTrue(all((b - a).days == 16 for a, b in zip(origins, origins[1:])))
        for step in (0, -1, True):
            with self.assertRaises(ValueError):
                training_origins("2013-01-01", "2017-07-14", step_days=step)

    def test_generated_files_have_stage_safe_labels_and_manifest(self):
        dates = pd.date_range("2017-01-01", "2017-08-31")
        panel = pd.DataFrame({"id": range(len(dates)), "date": dates, "store_nbr": 1, "family": "A",
                              "sales": np.arange(len(dates), dtype=float), "onpromotion": 0})
        panel["is_test"] = panel.date.gt("2017-08-15")
        panel["sales_observed"] = ~panel.is_test
        panel.loc[panel.is_test, "sales"] = np.nan
        # Missing target days exclude the entire affected training block, not fill labels.
        panel = panel.loc[panel.date != "2017-03-01"].copy()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "base.parquet"
            panel.to_parquet(source, index=False)
            original = source.read_bytes()
            result = prepare_feature_datasets(source, root / "features", root / "reports")
            self.assertTrue(result["complete"])
            for stage, info in result["stages"].items():
                train = pd.read_parquet(root / "features" / stage / "train.parquet")
                forecast = pd.read_parquet(root / "features" / stage / "features.parquet")
                self.assertLessEqual(train.date.max(), pd.Timestamp(info["origin"]))
                self.assertTrue((train.origin + pd.Timedelta(days=16) <= pd.Timestamp(info["origin"])).all())
                self.assertFalse(train.duplicated(["origin", "id"]).any())
                self.assertNotIn("sales", forecast)
                self.assertEqual(len(forecast), 16)
                self.assertGreater(len(info["skipped_origins"]), 0)
                if stage != "test":
                    labels = pd.read_parquet(root / "features" / stage / "labels.parquet")
                    self.assertEqual(labels.id.tolist(), forecast.id.tolist())
                    actual = panel.set_index("id").loc[labels.id, "sales"].to_numpy()
                    np.testing.assert_array_equal(labels.sales, actual)
                else:
                    self.assertFalse((root / "features" / stage / "labels.parquet").exists())
            manifest = json.loads((root / "reports/features_manifest.json").read_text(encoding="utf-8"))
            for relative, metadata in manifest["files"].items():
                self.assertEqual(pq.read_metadata(root / "features" / relative).num_rows, metadata["rows"])
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(list((root / "features").rglob("*.partial")))


if __name__ == "__main__":
    unittest.main()
