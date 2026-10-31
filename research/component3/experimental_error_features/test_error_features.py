"""Strict-time, missing-history, ablation, calibration and real-data checks."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import os
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from error_features import (GENERIC_ERROR_FEATURES, build_error_features, category_feature_names,
                            training_categories, validate_errors)
from run_experiment import (FEATURE_CAT, FEATURE_NUMERIC, METHODS, TARGET, calibration_diagnostics,
                            evaluate, fit_calibrated, numeric_features, prepare)

T = pd.Timestamp("2015-09-02 06:00:00")
VOCAB = ("error1", "error2", "error3", "error4", "error5")


def landmarks():
    return pd.DataFrame({"machineID": [1, 2], "prediction_time": [T, T]})


def errors_at(times, categories):
    return pd.DataFrame({"datetime": pd.to_datetime(times), "machineID": [1] * len(times), "errorID": categories})


class HistoricalErrorTests(unittest.TestCase):
    def test_exact_7_30_90_day_and_prediction_time_boundaries(self):
        events = errors_at([T - pd.Timedelta(days=91), T - pd.Timedelta(days=90),
                            T - pd.Timedelta(days=30), T - pd.Timedelta(days=7),
                            T - pd.Timedelta(1, "ns"), T, T + pd.Timedelta(1, "ns")],
                           ["error5", "error1", "error2", "error3", "error4", "error5", "error1"])
        result = build_error_features(landmarks(), validate_errors(events, [1, 2]), VOCAB)
        row = result.iloc[0]
        self.assertEqual([row.error_events_prior, row.error_events_7d, row.error_events_30d, row.error_events_90d], [5, 2, 3, 4])
        self.assertAlmostEqual(row.days_since_last_error, 1 / pd.Timedelta(days=1).value, places=25)
        self.assertEqual(row[category_feature_names(VOCAB)].tolist(), [1, 1, 1, 0])

    def test_no_prior_errors_has_zero_counts_and_missing_recency(self):
        events = errors_at([T, T + pd.Timedelta(days=1)], ["error1", "error2"])
        result = build_error_features(landmarks(), events, VOCAB)
        self.assertTrue(result.days_since_last_error.isna().all())
        counts = [c for c in result if c != "days_since_last_error"]
        self.assertTrue(result[counts].eq(0).all().all())

    def test_future_errors_and_future_category_cannot_change_features_or_vocabulary(self):
        old = errors_at([T - pd.Timedelta(days=1)], ["error1"])
        future = errors_at([T, T + pd.Timedelta(days=100)], ["error2", "novel_future_category"])
        augmented = pd.concat([future, old], ignore_index=True)
        pd.testing.assert_frame_equal(build_error_features(landmarks(), old, VOCAB), build_error_features(landmarks(), augmented, VOCAB))
        self.assertEqual(training_categories(augmented, T), ("error1",))

    def test_distinct_simultaneous_categories_are_retained(self):
        events = errors_at([T - pd.Timedelta(days=1)] * 2, ["error1", "error2"])
        events = validate_errors(events, [1, 2])
        row = build_error_features(landmarks(), events, VOCAB).iloc[0]
        self.assertEqual(row.error_events_prior, 2)
        self.assertEqual(row.error_category_error2_30d, 1)

    def test_sorting_and_other_machines_do_not_contaminate_features(self):
        events = errors_at([T - pd.Timedelta(days=10), T - pd.Timedelta(days=1)], ["error2", "error1"])
        reverse = events.iloc[::-1]
        pd.testing.assert_frame_equal(build_error_features(landmarks(), events, VOCAB), build_error_features(landmarks(), reverse, VOCAB))
        self.assertEqual(build_error_features(landmarks(), events, VOCAB).error_events_prior.tolist(), [2, 0])

    def test_validation_rejects_bad_schema_missing_unknown_and_duplicate_records(self):
        valid = errors_at([T], ["error1"])
        cases = [valid.drop(columns=["errorID"]), valid.assign(datetime=pd.NaT),
                 valid.assign(machineID=999), valid.assign(machineID=1.5), valid.assign(errorID=" "),
                 pd.concat([valid, valid], ignore_index=True), valid.assign(datetime="invalid")]
        for bad in cases:
            with self.assertRaises((ValueError, TypeError)):
                validate_errors(bad, [1, 2])

    def test_unseen_historical_category_does_not_silently_become_reference_category(self):
        events = errors_at([T - pd.Timedelta(days=1)], ["unknown_error"])
        with self.assertRaisesRegex(ValueError, "Unseen prior"):
            build_error_features(landmarks(), events, VOCAB)

    def test_category_reference_coding_is_deterministic(self):
        self.assertEqual(category_feature_names(VOCAB), [f"error_category_error{i}_30d" for i in range(2, 6)])
        with self.assertRaisesRegex(ValueError, "sorted"):
            category_feature_names(tuple(reversed(VOCAB)))


def fixture(count=40, seed=42):
    rng = np.random.default_rng(seed)
    columns = numeric_features("category_augmented", VOCAB)
    frame = pd.DataFrame({name: rng.uniform(1, 30, count) for name in columns})
    frame["machine_model"] = [f"model{i % 4 + 1}" for i in range(count)]
    frame[TARGET] = np.arange(count) % 3 == 0
    frame.loc[:2, "days_since_last_error"] = np.nan
    return frame


class ModelDisciplineTests(unittest.TestCase):
    def test_three_feature_sets_are_true_ablations_and_reference_matches_original(self):
        history = numeric_features("history_only", VOCAB)
        reference = numeric_features("existing_reference", VOCAB)
        augmented = numeric_features("category_augmented", VOCAB)
        self.assertEqual(reference, FEATURE_NUMERIC)
        self.assertFalse(set(history) & set(GENERIC_ERROR_FEATURES))
        self.assertEqual(set(reference) - set(history), set(GENERIC_ERROR_FEATURES))
        self.assertEqual(set(augmented) - set(reference), set(category_feature_names(VOCAB)))
        self.assertEqual([len(history) + 1, len(reference) + 1, len(augmented) + 1], [13, 18, 22])

    def test_imputation_scaling_use_training_only_and_calibration_base_is_frozen(self):
        train, validation = fixture(), fixture(20, 57)
        columns = numeric_features("category_augmented", VOCAB)
        validation[columns] += 10000
        base, calibrated = fit_calibrated(train, validation, columns)
        numeric = base.named_steps["transform"].named_transformers_["num"]
        self.assertEqual(numeric.named_steps["scale"].n_samples_seen_, len(train))
        np.testing.assert_allclose(numeric.named_steps["impute"].statistics_, np.nanmedian(train[columns], axis=0))
        self.assertIs(calibrated.estimator.estimator, base)
        self.assertEqual(base.named_steps["logistic"].max_iter, 2000)
        self.assertIsNone(base.named_steps["logistic"].class_weight)

    def test_labels_and_extra_future_fields_are_not_prediction_features(self):
        train, validation = fixture(), fixture(20, 57)
        columns = numeric_features("existing_reference", VOCAB)
        _, model = fit_calibrated(train, validation, columns)
        original = model.predict_proba(validation[columns + FEATURE_CAT])
        padded = validation.copy()
        padded[TARGET] = -999
        padded["future_error_count"] = 1000000
        np.testing.assert_array_equal(original, model.predict_proba(padded))

    def test_fit_and_inference_are_reproducible(self):
        train, validation = fixture(), fixture(20, 57)
        columns = numeric_features("category_augmented", VOCAB)
        _, first = fit_calibrated(train, validation, columns)
        _, second = fit_calibrated(train, validation, columns)
        x = validation[columns + FEATURE_CAT]
        np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))

    def test_calibration_bin_counts_and_brier_match_direct_metrics(self):
        y, p = np.array([0, 1, 0, 1]), np.array([0, .1, .9, 1])
        metrics, bins = calibration_diagnostics(y, p, "unit")
        self.assertEqual(sum(row["cases"] for row in bins), 4)
        self.assertEqual(bins[-1]["cases"], 2)
        self.assertEqual(metrics["brier"], brier_score_loss(y, p))
        self.assertEqual(metrics["average_precision"], average_precision_score(y, p))


DATA = os.environ.get("C3_EXPERIMENT_DATA_DIR")


@unittest.skipUnless(DATA, "Set C3_EXPERIMENT_DATA_DIR for actual benchmark integration checks")
class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.splits, cls.vocabulary, cls.quality, cls.chronology = prepare(Path(DATA))
        cls.tables, cls.reference, cls.detail, cls.models = evaluate(cls.panel, cls.splits, cls.vocabulary)

    def test_reference_probability_parity_and_original_chronology(self):
        self.assertLessEqual(self.reference["max_probability_difference_from_artifact"], 1e-10)
        self.assertAlmostEqual(self.reference["brier"], .09148540592410336, places=12)
        self.assertAlmostEqual(self.reference["average_precision"], .2521463037932662, places=12)
        self.assertEqual([len(f) for f in self.splits.values()], [1800, 500, 1300])
        self.assertEqual(self.chronology["independent_threshold_selection_cases"], 0)

    def test_error_data_quality_vocabulary_and_availability(self):
        self.assertEqual(self.quality["records"], 3919)
        self.assertEqual(self.quality["duplicate_complete_rows"], 0)
        self.assertEqual(self.vocabulary, VOCAB)
        available = self.tables["error_availability"].set_index("split")
        self.assertEqual(available.cases_with_prior_errors.tolist(), [1797, 500, 1300])
        self.assertEqual(available.cases_without_prior_errors.sum(), 3)

    def test_fixed_budgets_and_top10_across_all_dates(self):
        metrics = self.tables["model_metrics"].set_index("method")
        self.assertEqual(metrics.captured_at_10.tolist(), [32, 37, 36])
        budgets = self.tables["matched_alert_budgets"]
        self.assertTrue(budgets.dates.eq(13).all())
        self.assertTrue(budgets.selected_machine_time_cases.eq(13 * budgets.k).all())
        self.assertTrue((budgets.captured_positive_windows + budgets.false_positive_windows).eq(13 * budgets.k).all())

    def test_actual_metrics_and_threshold_candidates_do_not_change_model_states(self):
        for method in METHODS:
            base, model, columns = self.models[method]
            state = joblib.hash(base)
            p = model.predict_proba(self.splits["test"][columns])[:, 1]
            np.testing.assert_array_equal(p, self.detail[method])
            self.assertEqual(state, joblib.hash(base))
        candidates = self.tables["exploratory_recall_thresholds"]
        self.assertTrue(candidates.selection.eq("exploratory_reuse_of_sigmoid_calibration_cases").all())
        self.assertTrue((candidates.test_TP + candidates.test_FP + candidates.test_TN + candidates.test_FN).eq(1300).all())


if __name__ == "__main__":
    unittest.main()
