#!/usr/bin/env python3
"""Exploratory calibrated-model threshold and Top 10 research; no deployment."""
from __future__ import annotations

import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from threshold_metrics import (choose_validation_candidates, chronology_guard,
                               require, threshold_sweep, top_k_by_date)

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
HAZARD = RESEARCH / "experimental_multihorizon"
sys.path.insert(0, str(HAZARD))
from hazard_model import FEATURES, HORIZONS, PooledHazardModel  # noqa: E402
from train_evaluate import feasibility, ordering, sha256  # noqa: E402
from build_and_evaluate import load_csv, make_baseline  # noqa: E402

METHODS = ("independent_sigmoid", "pooled_hazard_sigmoid")
DIAGNOSTIC_THRESHOLDS = [0, .025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .6, .7, .8, .9, 1]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def reproduce(data_dir):
    """Refit evaluation copies only; fail unless all saved test scores reproduce.

    New fits are needed for the missing validation probabilities. The deployed
    artifact is loaded only for read-only prediction parity, never passed to fit.
    """
    sources = tuple(load_csv(Path(data_dir), name, name != "machines") for name in ["machines", "maint", "failures", "errors"])
    splits, outcomes, periods, counts = feasibility(sources, "2016-01-01 00:00:00")
    train, validation, test = (splits[s] for s in ["train", "validation", "test"])
    chronology = chronology_guard(train, validation, test)
    saved = pd.read_csv(HAZARD / "outputs/heldout_predictions.csv", parse_dates=["prediction_time"], float_precision="round_trip")
    keys = ["machineID", "prediction_time"]
    require(not saved.duplicated(keys).any() and len(saved) == len(test), "Saved test keys are not unique/matched")
    saved = saved.set_index(keys)
    test_keys = pd.MultiIndex.from_frame(test[keys])
    require(set(saved.index) == set(test_keys), "Test key set changed")
    saved = saved.reindex(test_keys)
    for h in HORIZONS:
        require(np.array_equal(saved[f"label_failure_{h}d"], test[f"label_failure_{h}d"]), "Saved test labels changed")
    probabilities = {split: {method: [] for method in METHODS} for split in ["validation", "test"]}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for horizon in HORIZONS:
            target = f"label_failure_{horizon}d"
            base = make_baseline().fit(train[FEATURES], train[target].to_numpy())
            calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
            calibrated.fit(validation[FEATURES], validation[target].to_numpy())
            for split in probabilities:
                probabilities[split]["independent_sigmoid"].append(calibrated.predict_proba(splits[split][FEATURES])[:, 1])
        hazard = PooledHazardModel().fit(train, periods["train"]).calibrate(validation, periods["validation"])
        for split in probabilities:
            probabilities[split]["pooled_hazard_sigmoid"] = hazard.predict(splits[split], calibrated=True)
    for split in probabilities:
        probabilities[split]["independent_sigmoid"] = np.column_stack(probabilities[split]["independent_sigmoid"])
    fidelity = []
    previous = json.loads((HAZARD / "results/summary.json").read_text(encoding="utf-8"))
    for method in METHODS:
        old = saved[[f"{method}_P{h}" for h in HORIZONS]].to_numpy()
        delta = float(np.max(np.abs(probabilities["test"][method] - old)))
        require(delta <= 1e-10, "Evaluation copies do not reproduce saved model probabilities")
        for index, horizon in enumerate(HORIZONS):
            prior = next(m for m in previous["metrics"] if m["method"] == method and m["horizon_days"] == horizon)
            y, p = test[f"label_failure_{horizon}d"], probabilities["test"][method][:, index]
            require(abs(brier_score_loss(y, p) - prior["brier"]) <= 1e-12, "Original Brier mismatch")
            require(abs(average_precision_score(y, p) - prior["average_precision"]) <= 1e-12, "Original AP mismatch")
        fidelity.append({"method": method, "max_absolute_test_probability_difference": delta})
    artifact = RESEARCH / "models/c3_7day_sigmoid.joblib"
    saved_model = joblib.load(artifact)
    require(saved_model["horizon_days"] == 7 and saved_model["feature_columns"] == FEATURES,
            "Deployed artifact feature/horizon mismatch")
    deployed_p = saved_model["model"].predict_proba(test[FEATURES])[:, 1]
    delta = float(np.max(np.abs(deployed_p - probabilities["test"]["independent_sigmoid"][:, 0])))
    require(delta <= 1e-10, "Deployed 7-day scores differ from evaluation model; do not conflate them")
    fidelity.append({"method": "deployed_7day_read_only", "max_absolute_test_probability_difference": delta})
    return splits, probabilities, counts, chronology, fidelity


def compare(splits, probabilities):
    """Select on reused calibration data, then evaluate unchanged on test."""
    metrics, diagnostic, candidates, full_sweeps, curves, per_date, rank_summary = [], [], [], [], [], [], []
    for method in METHODS:
        for index, horizon in enumerate(HORIZONS):
            key = {"method": method, "horizon_days": horizon}
            yv = splits["validation"][f"label_failure_{horizon}d"].to_numpy()
            yt = splits["test"][f"label_failure_{horizon}d"].to_numpy()
            pv, pt = (probabilities[s][method][:, index] for s in ["validation", "test"])
            chosen = choose_validation_candidates(yv, pv)  # Never receives test labels/scores.
            for candidate in chosen:
                row = {**key, **candidate, "selection_split": "validation_reused_for_sigmoid_calibration",
                       "independent_threshold_validation": False}
                if candidate["threshold"] is not None:
                    for prefix, y, p in [("validation", yv, pv), ("test", yt, pt)]:
                        measured = threshold_sweep(y, p, [candidate["threshold"]]).to_dict("records")[0]
                        row.update({f"{prefix}_{k}": value for k, value in measured.items() if k != "threshold"})
                candidates.append(row)
            metrics.append({**key, "cases": len(yt), "positive_cases": int(yt.sum()),
                            "prevalence": float(yt.mean()), "brier": float(brier_score_loss(yt, pt)),
                            "average_precision": float(average_precision_score(yt, pt))})
            for split, y, p in [("validation", yv, pv), ("test", yt, pt)]:
                sweep = threshold_sweep(y, p).assign(**key, split=split)
                full_sweeps.append(sweep)
                diagnostic.append(threshold_sweep(y, p, DIAGNOSTIC_THRESHOLDS).assign(**key, split=split))
            precision, recall, thresholds = precision_recall_curve(yt, pt, drop_intermediate=False)
            curves.append(pd.DataFrame({"precision": precision, "recall": recall,
                                        "threshold": np.r_[thresholds, np.nan], **key}))
            ranked = splits["test"][["machineID", "prediction_time"]].assign(label=yt, probability=pt)
            dates, aggregate = top_k_by_date(ranked)
            per_date.append(dates.assign(**key)); rank_summary.append({**key, **aggregate})
    # A historical-count heuristic uses only the same past-only feature rows.
    for horizon in HORIZONS:
        test = splits["test"]
        counts = test.failure_events_90d.to_numpy(dtype=float)
        # A monotone bounded transform preserves the count ranking; not an ML probability.
        heuristic = test[["machineID", "prediction_time"]].assign(
            label=test[f"label_failure_{horizon}d"].to_numpy(), probability=counts / (1 + counts))
        dates, aggregate = top_k_by_date(heuristic)
        key = {"method": "past_90day_failure_count_heuristic", "horizon_days": horizon}
        per_date.append(dates.assign(**key)); rank_summary.append({**key, **aggregate})
    return {"model_metrics": pd.DataFrame(metrics), "diagnostic_thresholds": pd.concat(diagnostic, ignore_index=True),
            "validation_candidates": pd.DataFrame(candidates), "full_threshold_sweeps": pd.concat(full_sweeps, ignore_index=True),
            "pr_curve_points": pd.concat(curves, ignore_index=True), "top10_by_date": pd.concat(per_date, ignore_index=True),
            "top10_summary": pd.DataFrame(rank_summary)}


def plot_curves(tables, directory):
    plt.rcParams.update({"font.size": 11, "svg.hashsalt": "component3-threshold-study-v1"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    styles = [("independent_sigmoid", "Independent calibrated LR", "#165f85", "-"),
              ("pooled_hazard_sigmoid", "Pooled hazard + sigmoid", "#146b63", "--")]
    for ax, horizon in zip(axes, HORIZONS):
        for method, label, color, style in styles:
            points = tables["pr_curve_points"].query("method == @method and horizon_days == @horizon")
            ap = tables["model_metrics"].query("method == @method and horizon_days == @horizon").iloc[0].average_precision
            ax.step(points.recall, points.precision, where="post", color=color, linestyle=style,
                    linewidth=1.7, label=f"{label}: AP {ap:.3f}")
        prevalence = tables["model_metrics"].query("horizon_days == @horizon").iloc[0].prevalence
        ax.axhline(prevalence, color="#5b6370", linestyle=":", label=f"Prevalence: {prevalence:.3f}")
        ax.set(xlabel="Recall (positive machine-time windows)", ylabel="Precision", title=f"{horizon}-day horizon", xlim=(0, 1), ylim=(0, 1.03))
        ax.grid(alpha=.2); ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Exploratory precision–recall curves | synthetic Azure benchmark | 1,300 previously inspected test cases")
    fig.savefig(directory / "precision_recall.png", dpi=160)
    fig.savefig(directory / "precision_recall.svg", metadata={"Date": None})
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, horizon in zip(axes, HORIZONS):
        for method, label, color, style in styles:
            sweep = tables["full_threshold_sweeps"].query("method == @method and horizon_days == @horizon and split == 'test'")
            ax.step(sweep.flagged_pct, sweep.recall, where="post", color=color,
                    linestyle=style, linewidth=1.7, label=label)
        ax.plot([0, 100], [0, 1], color="#5b6370", linestyle=":", label="Uniform random selection: expectation")
        ax.set(xlabel="Flagged machine-time cases (%)", ylabel="Recall", title=f"{horizon}-day horizon", xlim=(0, 100), ylim=(0, 1.03))
        ax.grid(alpha=.2); ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Exploratory recall versus alert burden | synthetic Azure benchmark | no validated inspection capacity")
    fig.savefig(directory / "recall_alert_burden.png", dpi=160)
    fig.savefig(directory / "recall_alert_burden.svg", metadata={"Date": None})
    plt.close(fig)


def write_report(tables, summary, directory):
    """Render verified tables directly; no manually copied research values."""
    names = {"independent_sigmoid": "Calibrated LR", "pooled_hazard_sigmoid": "Hazard + sigmoid",
             "past_90day_failure_count_heuristic": "Past 90d failure count"}
    def table(frame, columns):
        lines = ["| " + " | ".join(label for _, label in columns) + " |",
                 "| " + " | ".join("---" for _ in columns) + " |"]
        for record in frame.to_dict("records"):
            values = []
            for key, _ in columns:
                value = record[key]
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    value = "undefined"
                elif key == "method":
                    value = names[value]
                elif key == "prediction_time":
                    value = str(value).split()[0]
                elif isinstance(value, float):
                    value = f"{value:.6f}"
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)
    lines = ["# Exploratory threshold and Top 10 evaluation", "",
             "**Synthetic Azure industrial-machine data, not validated apparel-factory data.**",
             "All 500 July validation cases fitted sigmoid calibration. Thresholds selected on those same cases are exploratory, not independently validated.",
             "Candidate thresholds never use test labels; the full test curves and sweeps are retrospective descriptions. The 1,300 test cases have been inspected repeatedly.",
             "", "## Threshold-independent scores", "",
             table(tables["model_metrics"], [("method", "Model"), ("horizon_days", "Days"), ("prevalence", "Prevalence"),
                                             ("brier", "Brier"), ("average_precision", "AP")]),
             "", "AP is average precision, not trapezoidal PR area. Merely changing thresholds leaves probabilities, AP, Brier and score ordering unchanged.",
             "", "## Diagnostic threshold 0.5", ""]
    reference = tables["diagnostic_thresholds"].query("split == 'test' and threshold == 0.5")
    lines.append(table(reference, [("method", "Model"), ("horizon_days", "Days"), ("precision", "Precision"), ("recall", "Recall"),
                                   ("TP", "TP"), ("FP", "FP"), ("TN", "TN"), ("FN", "FN"), ("flagged_cases", "Flagged"), ("flagged_pct", "Flagged %")]))
    lines += ["", "## Validation-derived candidates, evaluated unchanged on test", "",
              "Recall targets apply to reused calibration data; they are not guaranteed on test. Highest thresholds break objective ties and minimize alerts at a recall target. All three targets are attainable on calibration data, sometimes with substantial alert burden.",
              "Each table reports test metrics unless marked validation. Full validation/test confusion counts, F1, F2 and burden are in validation_candidates.csv.", ""]
    for method in METHODS:
        candidates = tables["validation_candidates"].query("method == @method")
        lines += [f"### {names[method]}", "", table(candidates, [("horizon_days", "Days"), ("candidate", "Criterion"),
                  ("threshold", "Threshold"), ("validation_recall", "Validation recall"),
                  ("test_precision", "Precision"), ("test_recall", "Recall"), ("test_f1", "F1"), ("test_f2", "F2"),
                  ("test_flagged_cases", "Flagged"), ("test_flagged_pct", "Flagged %"), ("test_FP", "FP"),
                  ("test_false_positives_per_100_cases", "FP / 100")]), ""]
    lines += ["## Candidate ranges for fresh validation", "",
              "These ranges span the validation-derived 50–80% recall-target thresholds. They are research candidates, not validated operational choices. F1/F2 candidates are separate objectives, not verified maintenance costs.", ""]
    ranges = tables["validation_candidates"].query("candidate.str.startswith('recall_')", engine="python").groupby(["method", "horizon_days"]).threshold.agg(["min", "max"]).reset_index()
    lines += [table(ranges, [("method", "Model"), ("horizon_days", "Days"), ("min", "Minimum threshold"), ("max", "Maximum threshold")]),
              "", "## Top 10 across 13 test timestamps", "",
              "Every model selects 10 of 100 machines per timestamp: 130 selected machine-time cases. Micro precision is total captured positive windows / 130; micro recall is captured / all positive windows. Macro recall averages per-date recall. These are positive machine-time windows, not unique physical failures, repairs or prevented breakdowns.", "",
              table(tables["top10_summary"], [("method", "Ranking"), ("horizon_days", "Days"), ("captured_positive_windows", "Captured"),
                    ("positive_machine_time_cases", "Total positives"), ("micro_precision_at_k", "Micro P@10"),
                    ("micro_recall_at_k", "Micro R@10"), ("macro_recall_at_k", "Macro R@10"),
                    ("random_expected_captured", "Random expected captured")]),
              "", "The random baseline is the exact expectation for choosing 10 machines uniformly without replacement each date. The historical heuristic ranks past 90-day failure counts with machine-ID tie breaking; it fits nothing.", ""]
    for horizon in HORIZONS:
        lines += [f"### {horizon}-day capture by date", ""]
        frames = []
        for method in METHODS:
            dates = tables["top10_by_date"].query("method == @method and horizon_days == @horizon")
            frames.append(dates.set_index("prediction_time")[["captured_positive_windows"]].rename(columns={"captured_positive_windows": method}))
        dates = tables["top10_by_date"].query("method == 'independent_sigmoid' and horizon_days == @horizon").set_index("prediction_time")
        counts = dates[["positive_cases", "random_expected_captured"]].join(frames).reset_index()
        lines += [table(counts, [("prediction_time", "Date"), ("positive_cases", "Positive windows"),
                               ("independent_sigmoid", "LR captured"), ("pooled_hazard_sigmoid", "Hazard captured"),
                               ("random_expected_captured", "Random expected captured")]), ""]
    lines += ["## Interpretation and limits", "",
              "Lower thresholds recover recall but can flag most of the population. All-positive classification has precision equal to prevalence, 100% recall and 100% alert burden; it does not establish usefulness.",
              "At seven days the hazard model has lower AP and Top 10 capture than calibrated LR. Threshold reduction remedies zero recall at 0.5, but does not establish operational usefulness or repair poorer ranking.",
              "There is no validated acceptable false-alarm burden, inspection capacity or missed-failure cost. Repeated alerts, unique-event deduplication and maintenance benefit are not modeled.",
              "Different horizon-specific thresholds need not produce nested binary alerts, even when cumulative probabilities are ordered.",
              f"Cross-horizon probability violations remain {summary['ordering']['independent_sigmoid']['either']}/1,300 for independent calibrated LR and {summary['ordering']['pooled_hazard_sigmoid']['either']}/1,300 for hazard + sigmoid.",
              "A genuinely fresh chronological threshold-selection period separate from calibration, followed by a fresh holdout and real-factory validation, is required before operational decisions. The benchmark follow-up assumptions and repeated-landmark limitations from the hazard experiment still apply.", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(data_dir):
    splits, probabilities, counts, chronology, fidelity = reproduce(data_dir)
    tables = compare(splits, probabilities)
    results, outputs = HERE / "results", HERE / "outputs"
    results.mkdir(exist_ok=True); outputs.mkdir(exist_ok=True)
    for name, frame in tables.items():
        destination = outputs if name in ["full_threshold_sweeps", "pr_curve_points"] else results
        frame.to_csv(destination / f"{name}.csv", index=False, float_format="%.17g")
    plot_curves(tables, results)
    scores = splits["validation"][["machineID", "prediction_time", *[f"label_failure_{h}d" for h in HORIZONS]]].copy()
    for method in METHODS:
        for index, h in enumerate(HORIZONS):
            scores[f"{method}_P{h}"] = probabilities["validation"][method][:, index]
    scores.to_csv(outputs / "validation_predictions.csv", index=False, float_format="%.17g")
    paths = [Path(__file__), HERE / "threshold_metrics.py", HAZARD / "hazard_model.py", HAZARD / "train_evaluate.py",
             HAZARD / "results/summary.json", HAZARD / "outputs/heldout_predictions.csv",
             RESEARCH / "C3_calibration_validation_experiment.py", RESEARCH / "scripts/build_and_evaluate.py",
             RESEARCH / "scripts/model_inference.py", RESEARCH / "scripts/export_7day_model.py",
             RESEARCH / "models/c3_7day_sigmoid.joblib", RESEARCH / "outputs_calibration/calibration_brier_comparison.csv",
             RESEARCH / "outputs_corrected/machine_prediction_examples.csv"]
    hashes = {str(p.relative_to(RESEARCH)): sha256(p) for p in paths}
    hashes.update({"raw/" + name: sha256(Path(data_dir) / name) for name in
                   ["PdM_machines.csv", "PdM_maint.csv", "PdM_failures.csv", "PdM_errors.csv"]})
    summary = {"schema_version": "c3-exploratory-threshold-study-v1",
               "dataset": "Synthetic Azure industrial-machine benchmark; not validated apparel-factory data",
               "status": "Exploratory; no operationally validated threshold or factory inspection capacity",
               "split_counts": counts, "chronology": chronology, "prediction_fidelity": fidelity,
               "models_fitted_per_run": {"independent_base_copies": 3, "independent_validation_sigmoids": 3,
                                         "pooled_hazard_copy": 1, "pooled_validation_sigmoid": 1,
                                         "deployed_artifact_refits": 0, "serialized_artifacts": 0},
               "model_metrics": tables["model_metrics"].to_dict("records"),
               "ordering": {method: ordering(probabilities["test"][method]) for method in METHODS},
               "validation_candidates": tables["validation_candidates"].to_dict("records"),
               "top10_summary": tables["top10_summary"].to_dict("records"),
               "definitions": {"decision": "probability >= threshold, including exact ties",
                               "f1": "2*TP/(2*TP+FP+FN)", "f2": "5*TP/(5*TP+4*FN+FP)",
                               "undefined_precision": "null/blank when no cases flagged; F scores zero at zero denominator",
                               "AP": "average_precision_score, not trapezoidal PR area",
                               "FP_per_100": "100*FP/all evaluated cases, not false-positive rate among negatives",
                               "top10_micro_precision": "sum captured positive windows / 130 selected machine-time windows",
                               "top10_micro_recall": "sum captured positive windows / all positive machine-time windows",
                               "top10_macro_recall": "mean per-date recall excluding dates with no positive windows",
                               "random_baseline": "Exact expected capture when choosing 10 of 100 uniformly without replacement per date",
                               "historical_heuristic": "Rank past 90-day failure counts, machine-ID ascending tie break; no fitted parameters"},
               "limitations": ["All 500 July validation cases fitted sigmoid calibration; candidates reuse those labels and are not independently validated.",
                               "The validation snapshots span only 28 days: no later threshold subset with a full 30-day gap exists inside that original split.",
                               "Test labels do not select exported candidate thresholds; whole-test curves/sweeps are retrospective exploratory descriptions.",
                               "The same test period has been examined repeatedly; candidate test results are not untouched final performance.",
                               "Repeated machines and overlapping windows are dependent; captured positive windows are not unique failures or prevented breakdowns.",
                               "No unique-event deduplication, repeated-alert suppression, inspection cost or operational workload study is modeled.",
                               "Threshold changes do not change probabilities, ranking, AP, Brier or cross-horizon probability ordering.",
                               "No tested factory capacity/cost establishes an acceptable number of false positives or alerts.",
                               "Fresh chronological threshold-selection data separate from calibration, followed by a fresh holdout and factory validation, are required."],
               "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                            "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
               "source_sha256": hashes}
    (results / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(tables, summary, results)
    print("EXPLORATORY: all 500 threshold-selection cases also fitted calibration; no independent threshold holdout.")
    print(tables["model_metrics"].to_string(index=False))
    print(tables["validation_candidates"][["method", "horizon_days", "candidate", "threshold", "test_precision", "test_recall", "test_flagged_pct", "test_FP"]].to_string(index=False))
    print(tables["top10_summary"][["method", "horizon_days", "captured_positive_windows", "micro_precision_at_k", "micro_recall_at_k", "random_expected_captured"]].to_string(index=False))
    return summary, tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    run(parser.parse_args().data_dir)
