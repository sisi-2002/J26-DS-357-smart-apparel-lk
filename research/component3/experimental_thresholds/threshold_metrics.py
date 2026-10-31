"""Pure threshold/ranking diagnostics; no training, calibration or model loading."""
from __future__ import annotations

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_scores(y, probability):
    y, p = np.asarray(y), np.asarray(probability, dtype=float)
    require(y.ndim == p.ndim == 1 and len(y) == len(p) and len(y) > 0, "Expected aligned nonempty one-dimensional labels/scores")
    require(np.isin(y, [0, 1]).all(), "Labels must be binary and observed")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Probabilities must be finite in [0,1]")
    return y.astype(int), p


def threshold_sweep(y, probability, thresholds=None):
    """Exact >= decisions, including score ties, plus diagnostic grid points.

    Missing precision means no flagged cases; missing recall means no positives.
    F scores use count formulas and zero when their denominator is zero.
    """
    y, p = validate_scores(y, probability)
    if thresholds is None:
        thresholds = np.unique(np.r_[p, np.linspace(0, 1, 101), .025, .075, .125, .175])
    thresholds = np.asarray(thresholds, dtype=float)
    require(thresholds.ndim == 1 and len(thresholds) > 0 and np.isfinite(thresholds).all()
            and ((thresholds >= 0) & (thresholds <= 1)).all(), "Thresholds must be finite in [0,1]")
    thresholds = np.unique(thresholds)
    order = np.argsort(p, kind="stable")
    cumulative = np.r_[0, np.cumsum(y[order])]
    start = np.searchsorted(p[order], thresholds, side="left")
    n, positive = len(y), int(y.sum())
    flagged = n - start
    tp = positive - cumulative[start]
    fp, fn = flagged - tp, positive - tp
    tn = n - positive - fp
    precision = np.divide(tp, flagged, out=np.full(len(tp), np.nan), where=flagged > 0)
    recall = tp / positive if positive else np.full(len(tp), np.nan)
    d1, d2 = 2 * tp + fp + fn, 5 * tp + fp + 4 * fn
    return pd.DataFrame({"threshold": thresholds, "cases": n, "positive_cases": positive,
                         "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                         "precision": precision, "recall": recall,
                         "f1": np.divide(2 * tp, d1, out=np.zeros(len(tp)), where=d1 > 0),
                         "f2": np.divide(5 * tp, d2, out=np.zeros(len(tp)), where=d2 > 0),
                         "flagged_cases": flagged, "flagged_pct": 100 * flagged / n,
                         "false_positives_per_100_cases": 100 * fp / n})


def choose_validation_candidates(y_validation, p_validation):
    """Exploratory only: these labels also fitted the sigmoid calibrator.

    No test input is accepted. Highest threshold breaks ties, minimizing alerts
    among equivalent choices. Recall targets minimize alert count by choosing
    the highest observed threshold attaining the target on calibration data.
    """
    sweep = threshold_sweep(y_validation, p_validation)
    rows = []
    for name in ["max_f1", "max_f2", "recall_50", "recall_70", "recall_80"]:
        if int(np.sum(y_validation)) == 0:
            rows.append({"candidate": name, "threshold": None, "status": "unattainable_no_positive_validation_labels"})
            continue
        if name.startswith("max_"):
            score = name[4:]
            eligible = sweep[sweep[score].eq(sweep[score].max())]
        else:
            target = int(name.split("_")[1]) / 100
            eligible = sweep[sweep.recall.ge(target)]
        if eligible.empty:
            rows.append({"candidate": name, "threshold": None, "status": "unattainable"})
        else:
            best = eligible.sort_values("threshold", ascending=False).iloc[0]
            rows.append({"candidate": name, "threshold": float(best.threshold),
                         "status": "attainable_on_reused_calibration_data_only"})
    return rows


def chronology_guard(train, validation, test, horizon_days=30):
    """Assert embargo and no shared machine-time keys, not machine independence."""
    groups = [train, validation, test]
    for frame in groups:
        require(not frame.empty and not frame.duplicated(["machineID", "prediction_time"]).any(), "Empty or duplicated split")
        require(frame.prediction_time.notna().all(), "Missing prediction timestamps")
    for earlier, later in zip(groups, groups[1:]):
        require(earlier.prediction_time.max() + pd.Timedelta(days=horizon_days) < later.prediction_time.min(),
                "Chronological separation must exceed the full outcome horizon")
    all_keys = pd.concat([f[["machineID", "prediction_time"]] for f in groups])
    require(not all_keys.duplicated().any(), "Machine-time keys overlap across splits")
    return {"train_last": str(train.prediction_time.max()),
            "calibration_first": str(validation.prediction_time.min()),
            "calibration_last": str(validation.prediction_time.max()),
            "calibration_last_outcome_end": str(validation.prediction_time.max() + pd.Timedelta(days=horizon_days)),
            "test_first": str(test.prediction_time.min()), "separation_days": horizon_days,
            "independent_threshold_selection_cases": 0,
            "validation_reused_for_threshold_candidates": True}


def top_k_by_date(frame, k=10, expected_machines=100):
    """Count positive machine-time windows, not distinct physical failure events.

    Requires columns machineID, prediction_time, label, probability. Deterministic
    machine-ID tie breaking matches the dashboard. Random baseline is an exact
    expectation for sampling k machines uniformly without replacement per date.
    """
    require(0 < k <= expected_machines, "Invalid top-k size")
    require(not frame.duplicated(["machineID", "prediction_time"]).any(), "Duplicate ranking keys")
    validate_scores(frame.label, frame.probability)
    rows = []
    for timestamp, group in frame.groupby("prediction_time", sort=True):
        require(len(group) == expected_machines and set(group.machineID) == set(range(1, expected_machines + 1)),
                "Each ranking date must include every expected machine exactly once")
        ranked = group.sort_values(["probability", "machineID"], ascending=[False, True], kind="stable").head(k)
        positive, captured = int(group.label.sum()), int(ranked.label.sum())
        rows.append({"prediction_time": str(timestamp), "cases": len(group), "positive_cases": positive,
                     "selected_cases": k, "captured_positive_windows": captured, "false_positive_windows": k - captured,
                     "precision_at_k": captured / k, "recall_at_k": captured / positive if positive else np.nan,
                     "random_expected_captured": k * positive / len(group),
                     "random_expected_precision": positive / len(group),
                     "random_expected_recall": k / len(group) if positive else np.nan,
                     "selected_machine_ids": ",".join(map(str, ranked.machineID)),
                     "captured_machine_ids": ",".join(map(str, ranked.loc[ranked.label.eq(1), "machineID"]))})
    details = pd.DataFrame(rows)
    positive, captured, selected = (int(details[c].sum()) for c in ["positive_cases", "captured_positive_windows", "selected_cases"])
    summary = {"k": k, "dates": len(details), "selected_machine_time_cases": selected,
               "positive_machine_time_cases": positive, "captured_positive_windows": captured,
               "false_positive_windows": selected - captured, "micro_precision_at_k": captured / selected,
               "micro_recall_at_k": captured / positive if positive else None,
               "macro_precision_at_k": float(details.precision_at_k.mean()),
               "macro_recall_at_k": float(details.recall_at_k.mean()) if details.recall_at_k.notna().any() else None,
               "dates_with_no_positives": int(details.positive_cases.eq(0).sum()),
               "random_expected_captured": float(details.random_expected_captured.sum()),
               "random_expected_micro_precision": positive / int(details.cases.sum()),
               "random_expected_micro_recall": k / expected_machines if positive else None,
               "precision_lift_vs_random": (captured / selected) / (positive / details.cases.sum()) if positive else None}
    return details, summary
