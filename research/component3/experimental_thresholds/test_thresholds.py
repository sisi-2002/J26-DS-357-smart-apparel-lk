"""Numerical, chronology, selection-separation and real-data regression checks."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import json
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score

from threshold_metrics import (choose_validation_candidates, chronology_guard,
                               threshold_sweep, top_k_by_date)
from evaluate_thresholds import METHODS, compare, json_safe, reproduce


def toy_split(timestamp):
    ids = np.arange(1, 101)
    y7 = ids % 7 == 0
    y14 = y7 | (ids % 3 == 0)
    return pd.DataFrame({"machineID": ids, "prediction_time": pd.Timestamp(timestamp),
                         "label_failure_7d": y7.astype(int), "label_failure_14d": y14.astype(int),
                         "label_failure_30d": (y14 | (ids % 2 == 0)).astype(int),
                         "failure_events_90d": ids % 5})


class MetricTests(unittest.TestCase):
    def test_hand_calculation_confusion_f1_f2_and_burden(self):
        rows = threshold_sweep([1, 0, 1, 0], [.8, .7, .4, .1], [.5, .4]).set_index("threshold")
        r = rows.loc[.5]
        self.assertEqual([r.TP, r.FP, r.TN, r.FN], [1, 1, 1, 1])
        self.assertEqual([r.precision, r.recall, r.f1, r.f2], [.5] * 4)
        self.assertEqual([r.flagged_cases, r.flagged_pct, r.false_positives_per_100_cases], [2, 50, 25])
        r = rows.loc[.4]
        self.assertAlmostEqual(r.precision, 2 / 3)
        self.assertEqual(r.recall, 1)
        self.assertAlmostEqual(r.f1, .8)
        self.assertAlmostEqual(r.f2, 10 / 11)

    def test_all_exact_ties_match_sklearn(self):
        y, p = np.array([0, 1, 1, 0, 1, 0]), np.array([0, .2, .2, .5, 1, 1])
        for row in threshold_sweep(y, p).itertuples(index=False):
            prediction = p >= row.threshold
            tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
            self.assertEqual([row.TP, row.FP, row.TN, row.FN], [tp, fp, tn, fn])
            self.assertAlmostEqual(row.precision, precision_score(y, prediction))
            self.assertAlmostEqual(row.recall, recall_score(y, prediction))
            self.assertAlmostEqual(row.f1, fbeta_score(y, prediction, beta=1))
            self.assertAlmostEqual(row.f2, fbeta_score(y, prediction, beta=2))

    def test_no_alert_and_no_positive_cases_have_explicit_missing_metrics(self):
        r = threshold_sweep([0, 1], [.1, .2], [.5]).iloc[0]
        self.assertTrue(pd.isna(r.precision)); self.assertEqual(r.recall, 0)
        self.assertEqual(r.flagged_cases, 0)
        r = threshold_sweep([0, 0], [.1, .2], [.1]).iloc[0]
        self.assertTrue(pd.isna(r.recall)); self.assertEqual(r.precision, 0)
        self.assertEqual(json_safe({"precision": float("nan")}), {"precision": None})

    def test_invalid_inputs_rejected(self):
        for y, p in [([], []), ([0], [np.nan]), ([1], [1.1]), ([2], [.5]), ([0, 1], [.1])]:
            with self.assertRaises(ValueError): threshold_sweep(y, p)
        for threshold in [-.1, 1.1, np.inf]:
            with self.assertRaises(ValueError): threshold_sweep([1], [.5], [threshold])

    def test_lower_threshold_cannot_reduce_capture_or_alerts(self):
        sweep = threshold_sweep([0, 1, 1, 0], [.1, .3, .8, .8])
        self.assertTrue((np.diff(sweep.TP) <= 0).all())
        self.assertTrue((np.diff(sweep.flagged_cases) <= 0).all())
        self.assertTrue((np.diff(sweep.FP) <= 0).all())


class SelectionAndChronologyTests(unittest.TestCase):
    def test_recall_candidates_choose_highest_threshold_and_preserve_ties(self):
        y, p = np.array([1, 1, 1, 1, 0, 0]), np.array([.9, .7, .7, .2, .8, .1])
        choices = {r["candidate"]: r for r in choose_validation_candidates(y, p)}
        self.assertEqual(choices["recall_50"]["threshold"], .7)
        self.assertEqual(choices["recall_70"]["threshold"], .7)
        self.assertEqual(choices["recall_80"]["threshold"], .2)
        self.assertTrue(all(r["threshold"] is None for r in choose_validation_candidates([0, 0], [.1, .2])))

    def test_max_f1_f2_candidates_match_exhaustive_validation_sweep(self):
        y, p = np.array([1, 0, 1, 0, 1]), np.array([.9, .8, .5, .3, .1])
        sweep = threshold_sweep(y, p)
        for row in choose_validation_candidates(y, p)[:2]:
            metric = row["candidate"][4:]
            selected = threshold_sweep(y, p, [row["threshold"]]).iloc[0]
            self.assertEqual(selected[metric], sweep[metric].max())

    def test_test_labels_change_evaluation_but_not_selected_thresholds(self):
        splits = {"train": toy_split("2015-01-01"), "validation": toy_split("2015-03-01"), "test": toy_split("2015-05-01")}
        rng = np.random.default_rng(42)
        probabilities = {split: {method: np.sort(rng.uniform(0, 1, (100, 3)), axis=1) for method in METHODS}
                         for split in ["validation", "test"]}
        original_scores = {split: {method: p.copy() for method, p in scores.items()} for split, scores in probabilities.items()}
        before = compare(splits, probabilities)
        for h in [7, 14, 30]:
            splits["test"][f"label_failure_{h}d"] = 1 - splits["test"][f"label_failure_{h}d"]
        after = compare(splits, probabilities)
        pd.testing.assert_frame_equal(before["validation_candidates"][["candidate", "threshold"]],
                                      after["validation_candidates"][["candidate", "threshold"]])
        self.assertFalse(before["model_metrics"].brier.equals(after["model_metrics"].brier))
        for split, scores in probabilities.items():
            for method, p in scores.items():
                np.testing.assert_array_equal(p, original_scores[split][method])

    def test_chronology_rejects_target_overlap_and_reports_calibration_reuse(self):
        train, val, test = (toy_split(t) for t in ["2015-01-01", "2015-03-01", "2015-05-01"])
        info = chronology_guard(train, val, test)
        self.assertEqual(info["independent_threshold_selection_cases"], 0)
        for timestamp in ["2015-03-15", "2015-03-31"]:
            with self.assertRaises(ValueError): chronology_guard(train, val, toy_split(timestamp))
        with self.assertRaises(ValueError): chronology_guard(pd.concat([train, train]), val, test)


class RankingTests(unittest.TestCase):
    def test_per_date_grouping_tie_break_and_micro_macro_denominators(self):
        frame = pd.DataFrame({"machineID": [1, 2, 3, 4] * 2,
                              "prediction_time": [pd.Timestamp("2015-09-02")] * 4 + [pd.Timestamp("2015-09-09")] * 4,
                              "label": [1, 0, 0, 0, 1, 1, 1, 0], "probability": [.5] * 8})
        dates, summary = top_k_by_date(frame, k=2, expected_machines=4)
        self.assertEqual(dates.selected_machine_ids.tolist(), ["1,2", "1,2"])
        self.assertEqual(summary["captured_positive_windows"], 3)
        self.assertEqual(summary["micro_precision_at_k"], .75)
        self.assertEqual(summary["micro_recall_at_k"], .75)
        self.assertAlmostEqual(summary["macro_recall_at_k"], (1 + 2 / 3) / 2)
        self.assertEqual(summary["random_expected_captured"], 2)
        self.assertEqual(summary["random_expected_micro_recall"], .5)

    def test_missing_duplicate_or_unknown_machines_rejected(self):
        frame = toy_split("2015-09-02").assign(label=0, probability=.2)
        for bad in [frame.iloc[:-1], pd.concat([frame, frame.iloc[[0]]]), frame.assign(machineID=np.arange(2, 102))]:
            with self.assertRaises(ValueError): top_k_by_date(bad)

    def test_no_positives_date_has_no_invented_recall(self):
        frame = toy_split("2015-09-02").assign(label=0, probability=.2)
        dates, summary = top_k_by_date(frame)
        self.assertTrue(pd.isna(dates.recall_at_k.iloc[0]))
        self.assertIsNone(summary["micro_recall_at_k"])


DATA = os.environ.get("C3_EXPERIMENT_DATA_DIR")


@unittest.skipUnless(DATA, "Set C3_EXPERIMENT_DATA_DIR for real-model reproduction checks")
class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits, cls.probabilities, cls.counts, cls.chronology, cls.fidelity = reproduce(Path(DATA))
        cls.tables = compare(cls.splits, cls.probabilities)

    def test_saved_scores_artifact_parity_and_chronological_cases(self):
        self.assertTrue(all(r["max_absolute_test_probability_difference"] <= 1e-10 for r in self.fidelity))
        self.assertEqual([self.counts[s]["landmarks"] for s in ["train", "validation", "test"]], [1800, 500, 1300])
        self.assertEqual(self.chronology["independent_threshold_selection_cases"], 0)

    def test_actual_candidates_are_validation_only_and_test_counts_correct(self):
        candidates = self.tables["validation_candidates"]
        self.assertEqual(len(candidates), 30)
        self.assertTrue((~candidates.independent_threshold_validation).all())
        self.assertTrue((candidates.test_TP + candidates.test_FP + candidates.test_TN + candidates.test_FN).eq(1300).all())
        for row in candidates[candidates.candidate.str.startswith("recall_")].itertuples():
            target = int(row.candidate.split("_")[1]) / 100
            self.assertGreaterEqual(row.validation_recall, target)

    def test_actual_top10_capture_and_threshold_invariant_scores(self):
        ranks = self.tables["top10_summary"]
        independent = ranks[ranks.method.eq("independent_sigmoid")]
        hazard = ranks[ranks.method.eq("pooled_hazard_sigmoid")]
        self.assertEqual(independent.captured_positive_windows.tolist(), [37, 57, 95])
        self.assertEqual(hazard.captured_positive_windows.tolist(), [25, 54, 96])
        self.assertTrue(ranks.dates.eq(13).all()); self.assertTrue(ranks.selected_machine_time_cases.eq(130).all())
        summary = json.loads((Path(__file__).parent / "results/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["ordering"]["independent_sigmoid"]["either"], 272)
        self.assertEqual(summary["ordering"]["pooled_hazard_sigmoid"]["either"], 0)
        original = pd.DataFrame(summary["model_metrics"])
        np.testing.assert_array_equal(original.brier, self.tables["model_metrics"].brier)
        np.testing.assert_array_equal(original.average_precision, self.tables["model_metrics"].average_precision)


if __name__ == "__main__":
    unittest.main()
