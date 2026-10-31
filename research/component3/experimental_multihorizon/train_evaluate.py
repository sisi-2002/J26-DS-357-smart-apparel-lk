#!/usr/bin/env python3
"""Run a fixed, isolated pooled-hazard experiment; never serialize models."""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_score, recall_score

from hazard_model import FEATURES, HORIZONS, PooledHazardModel, landmark_outcomes, person_period, require
from build_and_evaluate import build_panel, load_csv, make_baseline

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
BENCHMARK_END = pd.Timestamp("2016-01-01 00:00:00")
THRESHOLD = 0.5  # Fixed before evaluation; not an operational maintenance rule.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordering(probabilities):
    p = np.asarray(probabilities)
    require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all(), "Invalid probability matrix")
    require(((p >= 0) & (p <= 1)).all(), "Probability outside [0,1]")
    a, b = p[:, 0] > p[:, 1], p[:, 1] > p[:, 2]
    return {"cases": len(p), "P7_gt_P14": int(a.sum()), "P14_gt_P30": int(b.sum()),
            "either": int((a | b).sum()), "both": int((a & b).sum()),
            "either_pct": float((a | b).mean() * 100)}


def evaluate(y, p, method, horizon):
    bins = []
    assigned = np.minimum((np.asarray(p) * 10).astype(int), 9)
    for index in range(10):
        selected = assigned == index
        bins.append({"method": method, "horizon_days": horizon, "bin": index + 1,
                     "lower": index / 10, "upper": (index + 1) / 10, "count": int(selected.sum()),
                     "mean_probability": float(np.mean(p[selected])) if selected.any() else np.nan,
                     "observed_rate": float(np.mean(y[selected])) if selected.any() else np.nan})
    ece = sum(row["count"] * abs(row["mean_probability"] - row["observed_rate"]) for row in bins if row["count"]) / len(y)
    return {"method": method, "horizon_days": horizon, "cases": len(y), "positive_cases": int(y.sum()),
            "brier": float(brier_score_loss(y, p)), "average_precision": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "threshold": THRESHOLD,
            "precision": float(precision_score(y, p >= THRESHOLD, zero_division=0)),
            "recall": float(recall_score(y, p >= THRESHOLD, zero_division=0)),
            "predicted_positive_cases": int((p >= THRESHOLD).sum()),
            "true_positives": int(((p >= THRESHOLD) & (y == 1)).sum()),
            "mean_probability": float(np.mean(p)), "observed_prevalence": float(np.mean(y)),
            "mean_probability_minus_prevalence": float(np.mean(p) - np.mean(y)), "ece_10_equal_width": float(ece)}, bins


def feasibility(sources, observation_end):
    """Abort before fitting if the fixed cohort, labels or follow-up cannot be supported."""
    end = pd.Timestamp(observation_end)
    require(end.tzinfo is None and not pd.isna(end) and end <= BENCHMARK_END,
            "Declare a timezone-naive administrative end no later than 2016-01-01; do not infer coverage from last failures")
    panel = build_panel(*sources)
    stored = pd.read_csv(RESEARCH / "outputs_corrected/machine_prediction_examples.csv", parse_dates=["prediction_time"])
    keys = ["machineID", "prediction_time"]
    require(not panel.duplicated(keys).any(), "Duplicate machine-time cases")
    pd.testing.assert_frame_equal(panel.set_index(keys).sort_index(), stored.set_index(keys).sort_index(),
                                  check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    splits = {name: panel[panel.split.eq(name)].copy() for name in ["train", "validation", "test"]}
    require([len(splits[name]) for name in splits] == [1800, 500, 1300], "Original chronological cohort changed")
    require(splits["train"].prediction_time.max() + pd.Timedelta(days=30) < splits["validation"].prediction_time.min(), "Train/validation target overlap")
    require(splits["validation"].prediction_time.max() + pd.Timedelta(days=30) < splits["test"].prediction_time.min(), "Validation/test target overlap")
    outcomes, periods, counts = {}, {}, {}
    for name, frame in splits.items():
        out = landmark_outcomes(frame, sources[2], end)
        require(out.complete_30d.all(), "All original cases need complete 30-day administrative follow-up; do not silently drop cases")
        for horizon in HORIZONS:
            require(np.array_equal(out[f"label_failure_{horizon}d"], frame[f"label_failure_{horizon}d"]), "First-event labels differ from original strict-future labels")
        require((out.label_failure_7d <= out.label_failure_14d).all() and (out.label_failure_14d <= out.label_failure_30d).all(), "Non-nested outcomes")
        pp = person_period(out)
        require(set(pp.interval) == {1, 2, 3}, "An interval has no eligible records")
        for _, group in pp.groupby("interval"):
            require(group.event.nunique() == 2, "An interval lacks events/non-events")
        outcomes[name], periods[name] = out, pp
        counts[name] = {"landmarks": len(frame), "machines": int(frame.machineID.nunique()),
                        "snapshots": int(frame.prediction_time.nunique()), "person_periods": len(pp),
                        "first_snapshot": str(frame.prediction_time.min()), "last_snapshot": str(frame.prediction_time.max()),
                        "latest_30day_end": str(frame.prediction_time.max() + pd.Timedelta(days=30)),
                        "complete_30d": int(out.complete_30d.sum()),
                        "intervals": [{"interval": int(i), "at_risk": len(g), "events": int(g.event.sum())} for i, g in pp.groupby("interval")],
                        "first_events_exactly_at_boundaries": {str(h): int(out.first_event_days.eq(h).sum()) for h in HORIZONS}}
    return splits, outcomes, periods, counts


def independent_references(train, validation, test):
    """Reproduce reference models in memory, verifying existing saved outputs."""
    summary = pd.read_csv(RESEARCH / "outputs_calibration/calibration_brier_comparison.csv")
    saved = pd.read_csv(RESEARCH / "outputs_corrected/heldout_predictions_with_evidence.csv", parse_dates=["prediction_time"])
    require(not saved.duplicated(["machineID", "prediction_time", "horizon_days"]).any(), "Duplicate reference predictions")
    raw, calibrated, fidelity = [], [], []
    keys = pd.MultiIndex.from_frame(test[["machineID", "prediction_time"]])
    for horizon in HORIZONS:
        target = f"label_failure_{horizon}d"
        base = make_baseline()
        base.fit(train[FEATURES], train[target].to_numpy())
        p = base.predict_proba(test[FEATURES])[:, 1]
        model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        model.fit(validation[FEATURES], validation[target])
        pc = model.predict_proba(test[FEATURES])[:, 1]
        old = saved[saved.horizon_days.eq(horizon)].set_index(["machineID", "prediction_time"])
        require(set(old.index) == set(keys), "Reference test-key mismatch")
        old = old.reindex(keys)
        require(np.array_equal(old[target], test[target]), "Reference labels differ")
        max_delta = float(np.max(np.abs(p - old.predicted_failure_probability)))
        require(max_delta <= 1e-10, "Cannot reproduce original uncalibrated per-instance probabilities")
        for method, probability in [("uncalibrated_original", p), ("sigmoid_calibrated_on_later_validation", pc)]:
            ref = summary[summary.horizon_days.eq(horizon) & summary.method.eq(method)]
            require(len(ref) == 1, "Missing reference metrics")
            ref = ref.iloc[0]
            brier = float(brier_score_loss(test[target], probability))
            ap = float(average_precision_score(test[target], probability))
            require(round(brier, 6) == ref.brier_score and round(ap, 6) == ref.pr_auc_average_precision,
                    "Cannot reproduce original calibration evaluation; comparisons aborted")
            fidelity.append({"horizon_days": horizon, "method": method, "brier": brier,
                             "saved_brier": float(ref.brier_score), "max_raw_probability_difference": max_delta})
        raw.append(p); calibrated.append(pc)
    return np.column_stack(raw), np.column_stack(calibrated), fidelity


def cluster_deltas(machine_ids, y, predictions, n_bootstrap=500):
    """Paired machine-cluster resampling of fixed losses, not model refitting."""
    ids = np.unique(machine_ids)
    draws = np.random.default_rng(42).integers(0, len(ids), size=(n_bootstrap, len(ids)))
    counts = np.array([np.sum(machine_ids == mid) for mid in ids])
    reference_loss = (y - predictions["independent_sigmoid"]) ** 2
    results = []
    for method in ["pooled_hazard", "pooled_hazard_sigmoid"]:
        difference = (y - predictions[method]) ** 2 - reference_loss
        sums = np.stack([difference[machine_ids == mid].sum(axis=0) for mid in ids])
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
        for index, horizon in enumerate(HORIZONS):
            results.append({"method": method, "horizon_days": horizon,
                            "brier_delta_vs_independent_sigmoid": float(difference[:, index].mean()),
                            "cluster_bootstrap_95_low": float(np.quantile(boot[:, index], .025)),
                            "cluster_bootstrap_95_high": float(np.quantile(boot[:, index], .975)),
                            "machines": len(ids), "resamples": n_bootstrap})
    return results


def run(data_dir, observation_end, output_dir=HERE / "results", prediction_output=HERE / "outputs/heldout_predictions.csv"):
    for output in [Path(output_dir).resolve(), Path(prediction_output).resolve()]:
        require(output != HERE and output.is_relative_to(HERE) and output.relative_to(HERE).parts[0] in ["results", "outputs"],
                "Outputs must remain inside this experiment's results/ or outputs/ directories")
    sources = tuple(load_csv(Path(data_dir), name, name != "machines") for name in ["machines", "maint", "failures", "errors"])
    splits, outcomes, periods, counts = feasibility(sources, observation_end)
    train, validation, test = (splits[name] for name in ["train", "validation", "test"])
    print("FEASIBLE: first-future-recorded-failure landmarks under explicit administrative follow-up assumption", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = PooledHazardModel().fit(train, periods["train"])
        raw_hazard = model.predict(test)
        model.calibrate(validation, periods["validation"])
        calibrated_hazard = model.predict(test, calibrated=True)
        independent_raw, independent_calibrated, fidelity = independent_references(train, validation, test)
    targets = test[[f"label_failure_{h}d" for h in HORIZONS]].to_numpy()
    prior = np.tile(train[[f"label_failure_{h}d" for h in HORIZONS]].mean().to_numpy(), (len(test), 1))
    predictions = {"train_prevalence": prior, "independent_raw": independent_raw, "independent_sigmoid": independent_calibrated,
                   "pooled_hazard": raw_hazard, "pooled_hazard_sigmoid": calibrated_hazard}
    metrics, bins = [], []
    coherence = {method: ordering(p) for method, p in predictions.items()}
    require(coherence["independent_raw"]["either"] == 660 and coherence["independent_sigmoid"]["either"] == 272,
            "Original cross-horizon audit did not reproduce")
    require(coherence["pooled_hazard"]["either"] == coherence["pooled_hazard_sigmoid"]["either"] == 0,
            "Hazard composition failed probability coherence")
    details = test[["machineID", "prediction_time", *[f"label_failure_{h}d" for h in HORIZONS]]].copy()
    for method, p in predictions.items():
        for index, horizon in enumerate(HORIZONS):
            metric, calibration_bins = evaluate(targets[:, index], p[:, index], method, horizon)
            metrics.append(metric); bins.extend(calibration_bins)
            details[f"{method}_P{horizon}"] = p[:, index]
    delta = cluster_deltas(test.machineID.to_numpy(), targets, predictions)
    provenance = {"raw/" + f"PdM_{name}.csv": sha256(Path(data_dir) / f"PdM_{name}.csv") for name in ["machines", "maint", "failures", "errors"]}
    for path in [RESEARCH / "scripts/build_and_evaluate.py", RESEARCH / "scripts/model_inference.py",
                 RESEARCH / "C3_calibration_validation_experiment.py",
                 RESEARCH / "outputs_corrected/machine_prediction_examples.csv",
                 RESEARCH / "outputs_corrected/heldout_predictions_with_evidence.csv",
                 RESEARCH / "outputs_calibration/calibration_brier_comparison.csv",
                 HERE / "hazard_model.py", Path(__file__)]:
        provenance[str(path.relative_to(RESEARCH))] = sha256(path)
    summary = {"schema_version": "experimental-pooled-hazard-v1", "dataset": "Synthetic Azure industrial-machine benchmark; not validated apparel-factory data",
               "formulation": "Next recorded machine failure after repeated landmarks; pooled discrete-time logistic hazard with shared slopes and interval intercepts",
               "intervals_days": ["(0,7)", "[7,14)", "[14,30)"], "administrative_followup_end_exclusive": str(pd.Timestamp(observation_end)),
               "followup_assumption": "All benchmark machines observed throughout 2015. Individual entry/exit and downtime coverage are not verified; last event is not censoring time.",
               "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
               "parameters": {"C": 1.0, "max_iter": 2000, "random_state": 42, "class_weight": None,
                              "preprocessing_fit": "unique training landmarks only", "interval_encoding": "categorical, first dropped",
                              "primary": "pooled_hazard", "prespecified_secondary": "pooled_hazard_sigmoid",
                              "calibration": "one validation-person-period sigmoid on frozen conditional-hazard classifier; compose afterward",
                              "precision_recall_threshold": THRESHOLD, "threshold_selection": "fixed before evaluation; no maintenance recommendation"},
               "split_counts": counts, "reference_fidelity": fidelity, "ordering": coherence, "metrics": metrics,
               "paired_machine_cluster_brier_deltas": delta, "source_sha256": provenance,
               "fitted_models": {"experimental_hazard": 1, "validation_hazard_sigmoid": 1,
                                 "reproduced_independent_logistic": 3, "reproduced_independent_sigmoid": 3, "serialized": 0},
               "limitations": ["Repeated landmarks, overlapping windows and recurrent failures create dependent observations; likelihood is a working composite likelihood, no iid coefficient inference.",
                               "No known uptime/repair-duration risk sets: predicts next recorded failure, not physical lifetime or time while operating.",
                               "Interval 3 spans 16 days; hazards are interval probabilities, not daily rates or continuous-time hazard ratios.",
                               "Shared slopes may underfit interval-specific relationships; no interactions or hyperparameter search were added after inspecting test results.",
                               "Test period was previously examined; this is exploratory comparison, not untouched external validation.",
                               "Cluster bootstrap resamples machines with all snapshots using fixed predictions; excludes model-fitting uncertainty and has no multiplicity correction.",
                               "ECE depends on the chosen bins; Brier measures overall probability error and is not proof of calibration alone.",
                               "Monotonicity is structural and does not demonstrate accuracy, reliability or apparel-domain validity."]}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    Path(prediction_output).parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False, float_format="%.17g")
    pd.DataFrame(bins).to_csv(output_dir / "calibration_bins.csv", index=False, float_format="%.17g")
    pd.DataFrame(delta).to_csv(output_dir / "paired_brier_deltas.csv", index=False, float_format="%.17g")
    details.to_csv(prediction_output, index=False, float_format="%.17g")
    print(pd.DataFrame(metrics)[["method", "horizon_days", "brier", "average_precision", "precision", "recall", "ece_10_equal_width"]].to_string(index=False), flush=True)
    print("ORDERING", json.dumps(coherence), flush=True)
    return summary, details, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--observation-end", required=True, help="Explicit exclusive administrative end; standard benchmark: 2016-01-01 00:00:00")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--prediction-output", type=Path, default=HERE / "outputs/heldout_predictions.csv")
    args = parser.parse_args()
    run(args.data_dir, args.observation_end, args.output_dir, args.prediction_output)
