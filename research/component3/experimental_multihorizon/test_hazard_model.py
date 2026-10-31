"""Unit checks plus opt-in raw-data integration checks. No files/models written."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import json
import os
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from hazard_model import (FEATURES, FEATURE_NUMERIC, HORIZONS, PooledHazardModel,
                          cumulative_probabilities, landmark_outcomes, person_period)
from train_evaluate import evaluate, feasibility, ordering, cluster_deltas
from build_and_evaluate import load_csv
from model_inference import machine_features


T = pd.Timestamp("2015-09-02 06:00:00")


def one_case(delay, followup=31):
    snapshots = pd.DataFrame({"machineID": [1], "prediction_time": [T]})
    times = [] if delay is None else [T + delay]
    failures = pd.DataFrame({"machineID": [1] * len(times), "datetime": pd.to_datetime(times)})
    out = landmark_outcomes(snapshots, failures, T + pd.Timedelta(days=followup))
    return out, person_period(out)


def fixture(count=32, seed=42):
    """Contrived unit-test data only; never used as benchmark results."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.uniform(1, 30, count) for name in FEATURE_NUMERIC})
    frame["machine_model"] = ["model" + str(i % 4 + 1) for i in range(count)]
    frame.loc[:3, "days_since_last_failure"] = np.nan
    outcomes = pd.DataFrame({"landmark_position": np.arange(count),
                             "first_event_days": np.resize([1., 8., 20., np.nan], count),
                             "followup_days": np.full(count, 31.)})
    return frame, person_period(outcomes)


class EventAndCensoringTests(unittest.TestCase):
    def test_exact_event_boundaries_preserve_original_cumulative_labels(self):
        cases = [(pd.Timedelta(0), [0, 0, 0], [0, 0, 0]),
                 (pd.Timedelta(1, "ns"), [1, 1, 1], [1]),
                 (pd.Timedelta(days=7) - pd.Timedelta(1, "ns"), [1, 1, 1], [1]),
                 (pd.Timedelta(days=7), [0, 1, 1], [0, 1]),
                 (pd.Timedelta(days=14), [0, 0, 1], [0, 0, 1]),
                 (pd.Timedelta(days=30), [0, 0, 0], [0, 0, 0])]
        for delay, labels, events in cases:
            with self.subTest(delay=delay):
                out, pp = one_case(delay)
                self.assertEqual(out[[f"label_failure_{h}d" for h in HORIZONS]].iloc[0].tolist(), labels)
                self.assertEqual(pp.event.tolist(), events)

    def test_only_first_future_event_and_no_periods_after_failure(self):
        snapshots = pd.DataFrame({"machineID": [1], "prediction_time": [T]})
        failures = pd.DataFrame({"machineID": [1] * 5,
                                 "datetime": [T, T + pd.Timedelta(days=8), T + pd.Timedelta(days=8),
                                              T + pd.Timedelta(days=15), T + pd.Timedelta(days=25)]})
        out = landmark_outcomes(snapshots, failures, T + pd.Timedelta(days=31))
        self.assertEqual(out.first_event_days.iloc[0], 8)
        self.assertEqual(person_period(out).event.tolist(), [0, 1])

    def test_censored_partial_intervals_are_not_false_negatives(self):
        for followup, events in [(3, []), (7, [0]), (10, [0]), (14, [0, 0]), (29, [0, 0]), (30, [0, 0, 0])]:
            with self.subTest(followup=followup):
                out, pp = one_case(None, followup)
                self.assertEqual(pp.event.tolist(), events)
                self.assertEqual(bool(out.complete_30d.iloc[0]), followup >= 30)
                for h in HORIZONS:
                    if h > followup:
                        self.assertTrue(pd.isna(out[f"label_failure_{h}d"].iloc[0]))

    def test_event_in_partially_observed_interval_is_included(self):
        out, pp = one_case(pd.Timedelta(days=8), followup=10)
        self.assertEqual(pp.event.tolist(), [0, 1])
        self.assertEqual(out.label_failure_30d.iloc[0], 1)
        self.assertFalse(out.complete_30d.iloc[0])

    def test_event_exactly_at_exclusive_censoring_end_is_not_observed(self):
        out, pp = one_case(pd.Timedelta(days=10), followup=10)
        self.assertTrue(pd.isna(out.first_event_days.iloc[0]))
        self.assertTrue(pd.isna(out.label_failure_14d.iloc[0]))
        self.assertEqual(pp.event.tolist(), [0])

    def test_followup_alignment_and_invalid_dates_fail(self):
        snapshots = pd.DataFrame({"machineID": [1], "prediction_time": [T]})
        failures = pd.DataFrame({"machineID": [], "datetime": pd.to_datetime([])})
        for end in [T, T - pd.Timedelta(days=1), pd.NaT]:
            with self.assertRaises(ValueError):
                landmark_outcomes(snapshots, failures, end)
        with self.assertRaisesRegex(ValueError, "align"):
            landmark_outcomes(snapshots, failures, pd.Series([T + pd.Timedelta(days=30)], index=[9]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            landmark_outcomes(pd.concat([snapshots, snapshots], ignore_index=True), failures, T + pd.Timedelta(days=31))


class HazardModelTests(unittest.TestCase):
    def setUp(self):
        self.train, self.periods = fixture()
        self.val, self.val_periods = fixture(16, seed=57)
        self.model = PooledHazardModel().fit(self.train, self.periods)

    def test_probability_composition_including_zero_one_and_extreme_hazards(self):
        hazards = np.vstack([np.random.default_rng(1).uniform(0, 1, (1000, 3)),
                             [0, 0, 0], [1, 1, 1], [0, .5, 1], [1e-15, 1e-15, 1e-15]])
        p = cumulative_probabilities(hazards)
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertTrue((np.diff(p, axis=1) >= 0).all())
        np.testing.assert_allclose(p[:, 0], hazards[:, 0], atol=1e-16)
        np.testing.assert_allclose(p[:, 1], 1 - (1 - hazards[:, 0]) * (1 - hazards[:, 1]))
        for bad in [np.array([[np.nan, 0, 0]]), np.array([[-.01, 0, 0]]), np.ones((2, 4))]:
            with self.assertRaises(ValueError): cumulative_probabilities(bad)

    def test_preprocessing_uses_unique_training_landmarks_and_stays_frozen(self):
        numeric = self.model.preprocessor_.named_transformers_["num"]
        self.assertEqual(numeric.named_steps["scale"].n_samples_seen_, len(self.train))
        np.testing.assert_allclose(numeric.named_steps["impute"].statistics_, np.nanmedian(self.train[FEATURE_NUMERIC], axis=0))
        self.assertGreater(self.model.training_periods_, self.model.training_landmarks_)
        state = joblib.hash((self.model.preprocessor_, self.model.classifier_))
        shifted = self.val.copy(); shifted[FEATURE_NUMERIC] += 100000
        self.model.calibrate(shifted, self.val_periods)
        self.assertEqual(state, joblib.hash((self.model.preprocessor_, self.model.classifier_)))
        self.assertEqual(self.model.calibration_landmarks_, len(self.val))

    def test_training_inference_columns_and_single_batch_predictions_agree(self):
        self.model.calibrate(self.val, self.val_periods)
        padded = self.val.copy()
        padded["label_failure_7d"] = 999999
        padded["first_event_days"] = -100
        for calibrated in [False, True]:
            p = self.model.predict(self.val, calibrated=calibrated)
            np.testing.assert_array_equal(p, self.model.predict(padded, calibrated=calibrated))
            for index in range(len(self.val)):
                np.testing.assert_allclose(p[index], self.model.predict(self.val.iloc[[index]], calibrated=calibrated)[0], atol=1e-12)
            self.assertEqual(ordering(p)["either"], 0)

    def test_reproducibility(self):
        other = PooledHazardModel().fit(self.train, self.periods).calibrate(self.val, self.val_periods)
        self.model.calibrate(self.val, self.val_periods)
        for calibrated in [False, True]:
            np.testing.assert_array_equal(self.model.predict(self.val, calibrated=calibrated), other.predict(self.val, calibrated=calibrated))

    def test_invalid_risk_sets_and_missing_features_rejected(self):
        for bad in [pd.DataFrame({"landmark_position": [0, 0], "interval": [1, 2], "event": [1, 0]}),
                    pd.DataFrame({"landmark_position": [0], "interval": [2], "event": [0]}),
                    pd.DataFrame({"landmark_position": [999], "interval": [1], "event": [1]})]:
            with self.assertRaises(ValueError): PooledHazardModel().fit(self.train, bad)
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.model.predict(self.val.drop(columns=[FEATURES[0]]))

    def test_diagnostics_and_cluster_bootstrap(self):
        y = np.array([0, 1, 0, 1]); p = np.array([0., .1, .9, 1.])
        score, bins = evaluate(y, p, "fixture", 7)
        self.assertEqual(sum(row["count"] for row in bins), 4)
        self.assertEqual(bins[-1]["count"], 2)
        self.assertEqual(score["threshold"], .5)
        same = np.tile(p[:, None], (1, 3))
        predictions = {name: same for name in ["independent_sigmoid", "pooled_hazard", "pooled_hazard_sigmoid"]}
        values = cluster_deltas(np.array([1, 1, 2, 2]), np.tile(y[:, None], (1, 3)), predictions, n_bootstrap=20)
        self.assertTrue(all(row["brier_delta_vs_independent_sigmoid"] == row["cluster_bootstrap_95_low"] == row["cluster_bootstrap_95_high"] == 0 for row in values))


DATA = os.environ.get("C3_EXPERIMENT_DATA_DIR")


@unittest.skipUnless(DATA, "Set C3_EXPERIMENT_DATA_DIR to run raw-benchmark integration tests")
class BenchmarkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = tuple(load_csv(Path(DATA), name, name != "machines") for name in ["machines", "maint", "failures", "errors"])
        cls.splits, cls.outcomes, cls.periods, cls.counts = feasibility(cls.sources, "2016-01-01 00:00:00")
        cls.model = PooledHazardModel().fit(cls.splits["train"], cls.periods["train"]).calibrate(cls.splits["validation"], cls.periods["validation"])

    def test_actual_at_risk_counts_and_full_followup(self):
        self.assertEqual([self.counts[name]["person_periods"] for name in self.counts], [4740, 1320, 3450])
        self.assertEqual([g["events"] for g in self.counts["train"]["intervals"]], [213, 234, 409])
        for name in self.counts:
            self.assertTrue(self.outcomes[name].complete_30d.all())

    def test_existing_feature_inference_parity_and_future_event_invariance(self):
        test = self.splits["test"]
        original = test[test.machineID.eq(1) & test.prediction_time.eq(T)]
        one = machine_features(*self.sources, 1, T)
        for name in FEATURES:
            if name == "machine_model": self.assertEqual(original[name].iloc[0], one[name].iloc[0])
            else: np.testing.assert_allclose(original[name], one[name], equal_nan=True)
        altered = [frame.copy() for frame in self.sources]
        for i in [1, 2, 3]:
            added = pd.concat([altered[i].iloc[[0]], altered[i].iloc[[0]]], ignore_index=True)
            added["machineID"] = 1; added["datetime"] = [T, T + pd.Timedelta(days=3)]
            altered[i] = pd.concat([altered[i], added], ignore_index=True)
        future = machine_features(*altered, 1, T)
        pd.testing.assert_frame_equal(one, future)
        for calibrated in [False, True]:
            np.testing.assert_array_equal(self.model.predict(original, calibrated=calibrated), self.model.predict(one, calibrated=calibrated))
            self.assertEqual(ordering(self.model.predict(test, calibrated=calibrated))["either"], 0)

    def test_incomplete_or_unjustified_followup_stops_evaluation(self):
        for end in ["2015-12-01", "2027-01-01"]:
            with self.assertRaises(ValueError): feasibility(self.sources, end)

    def test_generated_results_match_fitted_model_and_original_reference(self):
        result = Path(__file__).parent / "results/summary.json"
        if not result.exists(): self.skipTest("Run train_evaluate.py to produce numerical summaries")
        summary = json.loads(result.read_text(encoding="utf-8"))
        test = self.splits["test"]
        for method, calibrated in [("pooled_hazard", False), ("pooled_hazard_sigmoid", True)]:
            p = self.model.predict(test, calibrated=calibrated)
            for index, horizon in enumerate(HORIZONS):
                actual, _ = evaluate(test[f"label_failure_{horizon}d"].to_numpy(), p[:, index], method, horizon)
                stored = next(row for row in summary["metrics"] if row["method"] == method and row["horizon_days"] == horizon)
                self.assertAlmostEqual(actual["brier"], stored["brier"], places=12)
        self.assertEqual(summary["ordering"]["independent_sigmoid"]["either"], 272)
        self.assertTrue(all(round(row["brier"], 6) == row["saved_brier"] for row in summary["reference_fidelity"]))


if __name__ == "__main__":
    unittest.main()
