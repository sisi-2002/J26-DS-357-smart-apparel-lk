"""Time-aware selection, exact feature/preprocessing controls and real-model checks."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from models import (ALGORITHMS, FEATURES, GRIDS, TARGET, development_folds,
                    make_model, select_parameters, xgboost_availability)
from run_comparison import evaluate, fit_final, paired_diagnostics, prepare


def training_keys():
    return pd.DataFrame({"machineID": np.tile(np.arange(1, 101), 18),
                         "prediction_time": np.repeat(pd.date_range("2015-02-01 06:00", periods=18, freq="7D"), 100),
                         "split": "train", TARGET: np.tile((np.arange(100) % 7 == 0).astype(int), 18)})


def fixture(count=40, seed=42):
    """Contrived test matrix only, never benchmark predictions."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.uniform(1, 30, count) for name in FEATURES if name != "machine_model"})
    frame["machine_model"] = [f"model{i % 4 + 1}" for i in range(count)]
    frame[TARGET] = (np.arange(count) % 3 == 0).astype(int)
    frame.loc[:2, "days_since_last_error"] = np.nan
    return frame


class FoldAndSelectionTests(unittest.TestCase):
    def test_expanding_folds_keep_dates_whole_and_purge_target_windows(self):
        train = training_keys()
        folds = development_folds(train)
        self.assertEqual([len(fit) for fit, _, _ in folds], [500, 800, 1100])
        self.assertEqual([len(score) for _, score, _ in folds], [300, 300, 300])
        seen = []
        for fit, score, _ in folds:
            self.assertLess(train.iloc[fit].prediction_time.max() + pd.Timedelta(days=30), train.iloc[score].prediction_time.min())
            self.assertFalse(set(fit) & set(score))
            self.assertEqual(train.iloc[score].groupby("prediction_time").size().tolist(), [100] * 3)
            self.assertEqual(train.iloc[fit].machineID.nunique(), 100)
            seen += score.tolist()
        self.assertEqual(len(set(seen)), 900)

    def test_insufficient_dates_missing_machines_or_nontraining_cases_fail(self):
        train = training_keys()
        for bad in [train.iloc[:-100], train.assign(split="validation"), train.assign(machineID=1),
                    pd.concat([train.iloc[:-1], train.iloc[[0]]], ignore_index=True)]:
            with self.assertRaises(ValueError): development_folds(bad)

    def test_selection_only_fits_training_prefixes_and_never_extra_feature_columns(self):
        train = training_keys()
        for name in FEATURES:
            train[name] = "model1" if name == "machine_model" else 1.
        train["future_test_label"] = 999
        fitting_indices = []
        scoring_indices = []
        class Probe:
            def fit(self, x, y):
                self_outer.assertEqual(list(x.columns), FEATURES)
                self_outer.assertTrue(set(x.index).issubset(set(train.index)))
                fitting_indices.append(x.index.tolist())
                return self
            def predict_proba(self, x):
                scoring_indices.append(x.index.tolist())
                return np.tile([.8, .2], (len(x), 1))
        self_outer = self
        with patch("models.make_model", side_effect=lambda *args: Probe()):
            selected, scores, _ = select_parameters(train, {"available": False})
        self.assertNotIn("xgboost", selected)
        self.assertEqual(len(scores), 21)
        self.assertEqual(selected["random_forest"], GRIDS["random_forest"][0])
        for fitting, scoring in zip(fitting_indices, scoring_indices):
            self.assertLess(train.loc[fitting].prediction_time.max() + pd.Timedelta(days=30), train.loc[scoring].prediction_time.min())

    def test_xgboost_absence_is_explicit_without_install_or_import(self):
        with patch("models.importlib.util.find_spec", return_value=None), patch("models.importlib.import_module") as importing:
            status = xgboost_availability()
            self.assertFalse(status["available"])
            importing.assert_not_called()
        with patch("models.xgboost_availability", return_value={"available": False}):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                make_model("xgboost")


class ModelDisciplineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.calibration = fixture(), fixture(20, 57)
        cls.available = list(ALGORITHMS[:3]) + (["xgboost"] if xgboost_availability()["available"] else [])
        cls.models = {name: fit_final(name, {} if name == "logistic_regression" else GRIDS[name][0], cls.train, cls.calibration)
                      for name in cls.available}

    def test_all_algorithms_receive_exact_same_original_18_columns(self):
        self.assertEqual(len(FEATURES), 18)
        self.assertEqual(len([name for name in FEATURES if name.startswith("error_") or name == "days_since_last_error"]), 5)
        reference = self.models["logistic_regression"][0].named_steps["transform"].transform(self.train[FEATURES])
        for base, _ in self.models.values():
            transform = base.named_steps["transform"]
            columns = [column for _, _, values in transform.transformers for column in values]
            self.assertEqual(columns, FEATURES)
            np.testing.assert_array_equal(reference, transform.transform(self.train[FEATURES]))

    def test_preprocessing_fits_training_only_and_calibration_keeps_base_frozen(self):
        numeric_names = [name for name in FEATURES if name != "machine_model"]
        for base, calibrated in self.models.values():
            numeric = base.named_steps["transform"].named_transformers_["num"]
            self.assertEqual(numeric.named_steps["scale"].n_samples_seen_, len(self.train))
            np.testing.assert_allclose(numeric.named_steps["impute"].statistics_, np.nanmedian(self.train[numeric_names], axis=0))
            self.assertIs(calibrated.estimator.estimator, base)
            state = joblib.hash(base)
            shifted = self.calibration[FEATURES].copy()
            shifted[numeric_names] += 100000
            calibrated.predict_proba(shifted)
            self.assertEqual(state, joblib.hash(base))

    def test_no_implicit_random_validation_or_early_stopping(self):
        hgb = self.models["hist_gradient_boosting"][0].steps[-1][1]
        self.assertFalse(hgb.early_stopping)
        if "xgboost" in self.models:
            self.assertIsNone(self.models["xgboost"][0].steps[-1][1].get_params().get("early_stopping_rounds"))
        with self.assertRaises(ValueError): make_model("logistic_regression", {"C": 5})

    def test_future_label_columns_are_ignored_and_probabilities_valid(self):
        for _, calibrated in self.models.values():
            probability = calibrated.predict_proba(self.calibration[FEATURES])[:, 1]
            padded = self.calibration.copy()
            padded[TARGET] = -100000
            padded["future_telemetry"] = 999
            padded["error_category_error9_30d"] = 999
            np.testing.assert_array_equal(probability, calibrated.predict_proba(padded)[:, 1])
            self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_seeded_random_forest_reproduction(self):
        _, other = fit_final("random_forest", GRIDS["random_forest"][0], self.train, self.calibration)
        np.testing.assert_array_equal(other.predict_proba(self.calibration[FEATURES]),
                                      self.models["random_forest"][1].predict_proba(self.calibration[FEATURES]))

    def test_machine_cluster_differences_are_zero_for_identical_scores(self):
        frame = pd.DataFrame({"machineID": [1, 2, 1, 2], TARGET: [0, 0, 1, 1],
                              "logistic_regression": [.1, .2, .7, .8], "random_forest": [.1, .2, .7, .8]})
        result = paired_diagnostics(frame).iloc[0]
        self.assertEqual(result.brier_delta_vs_lr, 0)
        self.assertEqual(result.brier_conditional_95_low, 0)
        self.assertEqual(result.brier_conditional_95_high, 0)
        self.assertEqual(result.ap_delta_vs_lr, 0)


DATA = os.environ.get("C3_EXPERIMENT_DATA_DIR")


@unittest.skipUnless(DATA, "Set C3_EXPERIMENT_DATA_DIR to run benchmark integration checks")
class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits, cls.chronology = prepare(Path(DATA))
        directory = Path(__file__).parent / "results"
        cls.summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        cls.development = pd.read_csv(directory / "development_scores.csv", float_precision="round_trip")
        cls.tables, cls.predictions, cls.reference, cls.models = evaluate(cls.splits, cls.summary["selected_parameters"])

    def test_reference_parity_exact_cohort_and_all_identical_prediction_keys(self):
        self.assertEqual([len(frame) for frame in self.splits.values()], [1800, 500, 1300])
        self.assertAlmostEqual(self.reference["brier"], .09148540592410336, places=12)
        self.assertAlmostEqual(self.reference["average_precision"], .2521463037932662, places=12)
        self.assertLessEqual(self.reference["max_probability_difference_from_artifact"], 1e-10)
        self.assertEqual(len(self.predictions), 1300)
        self.assertEqual(self.predictions.machineID.nunique(), 100)
        self.assertEqual(self.predictions.prediction_time.nunique(), 13)
        self.assertFalse(self.chronology["validation_reused_for_threshold_candidates"])
        self.assertFalse(self.chronology["threshold_selection_performed"])

    def test_parameter_choices_follow_development_brier_only(self):
        for algorithm, parameters in self.summary["selected_parameters"].items():
            rows = self.development[self.development.algorithm.eq(algorithm)]
            self.assertEqual(len(rows), 3 if algorithm == "logistic_regression" else 9)
            means = rows.groupby("candidate").agg(brier=("raw_development_brier", "mean"), ap=("raw_development_ap", "mean")).reset_index()
            selected = int(means.sort_values(["brier", "ap", "candidate"], ascending=[True, False, True]).iloc[0].candidate)
            self.assertEqual(parameters, {} if algorithm == "logistic_regression" else GRIDS[algorithm][selected])
            self.assertTrue(pd.to_datetime(rows.scoring_last).le(self.splits["train"].prediction_time.max()).all())
            self.assertTrue(rows.scoring_cases.eq(300).all())

    def test_full_saved_probabilities_and_metrics_reproduce(self):
        path = Path(__file__).parent / "outputs/heldout_predictions.csv"
        stored = pd.read_csv(path, parse_dates=["prediction_time"], float_precision="round_trip")
        pd.testing.assert_frame_equal(stored, self.predictions.reset_index(drop=True), check_exact=True, check_dtype=False)
        previous = pd.DataFrame(self.summary["metrics"]).set_index("algorithm")
        current = self.tables["model_metrics"].set_index("algorithm")
        for name in ["brier", "average_precision", "log_loss", "precision_at_10", "recall_at_10", "ece_10_equal_width"]:
            np.testing.assert_array_equal(previous[name], current[name])

    def test_top10_budgets_and_calibration_bins_cover_identical_cases(self):
        budgets = self.tables["matched_alert_budgets"]
        self.assertTrue(budgets.dates.eq(13).all())
        self.assertTrue(budgets.selected_machine_time_cases.eq(13 * budgets.k).all())
        self.assertTrue((budgets.captured_positive_windows + budgets.false_positive_windows).eq(13 * budgets.k).all())
        self.assertTrue(self.tables["calibration_bins"].groupby("method").cases.sum().eq(1300).all())
        self.assertEqual(self.tables["model_metrics"].set_index("algorithm").loc["logistic_regression", "captured_at_10"], 37)


if __name__ == "__main__":
    unittest.main()
