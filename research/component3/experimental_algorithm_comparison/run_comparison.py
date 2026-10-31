#!/usr/bin/env python3
"""Training-only algorithm selection and frozen chronological calibration; research only."""
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

from models import (ALGORITHMS, FEATURES, GRIDS, RESEARCH, TARGET, development_folds,
                    make_model, require, select_parameters, xgboost_availability)
from build_and_evaluate import build_panel, load_csv
sys.path.insert(0, str(RESEARCH / "experimental_thresholds"))
sys.path.insert(0, str(RESEARCH / "experimental_error_features"))
from threshold_metrics import chronology_guard, top_k_by_date  # noqa: E402
from run_experiment import calibration_diagnostics, clean, sha256, verify_reference  # noqa: E402

HERE = Path(__file__).resolve().parent
NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
         "hist_gradient_boosting": "HistGradientBoosting", "xgboost": "XGBoost"}


def prepare(data_dir):
    sources = tuple(load_csv(Path(data_dir), name, name != "machines") for name in ["machines", "maint", "failures", "errors"])
    panel = build_panel(*sources)
    saved = pd.read_csv(RESEARCH / "outputs_corrected/machine_prediction_examples.csv", parse_dates=["prediction_time"])
    keys = ["machineID", "prediction_time"]
    require(not panel.duplicated(keys).any(), "Duplicate original machine-time cases")
    pd.testing.assert_frame_equal(panel.set_index(keys).sort_index(), saved.set_index(keys).sort_index(),
                                  check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    splits = {s: panel[panel.split.eq(s)].copy() for s in ["train", "validation", "test"]}
    require([len(frame) for frame in splits.values()] == [1800, 500, 1300], "Original chronological cohort changed")
    require(len(FEATURES) == 18 and len(set(FEATURES)) == 18, "Use exactly the original 18 feature columns")
    require(panel.prediction_time.max() + pd.Timedelta(days=30) < pd.Timestamp("2016-01-01"), "Outcome windows exceed assumed benchmark observation period")
    chronology = chronology_guard(*splits.values())
    chronology["validation_reused_for_threshold_candidates"] = False
    chronology["threshold_selection_performed"] = False
    # Independent strict-future lookup, rather than the legacy >= metadata text.
    failures = {int(mid): np.sort(group.datetime.astype("int64").to_numpy()) for mid, group in sources[2].groupby("machineID")}
    reconstructed = []
    for mid, timestamp in zip(panel.machineID, panel.prediction_time):
        events = failures.get(int(mid), np.empty(0, dtype="int64"))
        reconstructed.append(int(np.searchsorted(events, (timestamp + pd.Timedelta(days=7)).value, side="left") >
                                 np.searchsorted(events, timestamp.value, side="right")))
    require(np.array_equal(reconstructed, panel[TARGET]), "Strict-future labels disagree with saved cases")
    return splits, chronology


def fit_final(algorithm, parameters, train, calibration):
    """Only training and calibration frames accepted; no test data in fitting."""
    model = make_model(algorithm, parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train[FEATURES], train[TARGET].to_numpy())
        state = joblib.hash(model)
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        calibrated.fit(calibration[FEATURES], calibration[TARGET].to_numpy())
    require(state == joblib.hash(model), "Calibration modified training-fitted state")
    return model, calibrated


def evaluate(splits, selected):
    train, calibration, test = (splits[name] for name in ["train", "validation", "test"])
    metrics, bins, budgets, dates, diagnostics, models = [], [], [], [], [], {}
    predictions = test[["machineID", "prediction_time", TARGET]].copy()
    reference = None
    for algorithm in ALGORITHMS:
        if algorithm not in selected: continue
        base, calibrated = fit_final(algorithm, selected[algorithm], train, calibration)
        probability = calibrated.predict_proba(test[FEATURES])[:, 1]
        require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), "Invalid final probabilities")
        if algorithm == "logistic_regression": reference = verify_reference(test, probability)
        models[algorithm] = (base, calibrated)
        predictions[algorithm] = probability
        score, calibration_bins = calibration_diagnostics(test[TARGET].to_numpy(), probability, algorithm)
        bins += calibration_bins
        ranking = test[["machineID", "prediction_time"]].assign(label=test[TARGET].to_numpy(), probability=probability)
        for k in [10, 20, 30]:
            daily, aggregate = top_k_by_date(ranking, k=k)
            budgets.append({"algorithm": algorithm, **aggregate, "flagged_pct": 100 * aggregate["selected_machine_time_cases"] / len(test),
                            "false_positives_per_100_cases": 100 * aggregate["false_positive_windows"] / len(test)})
            if k == 10:
                dates.append(daily.assign(algorithm=algorithm))
                score.update({"precision_at_10": aggregate["micro_precision_at_k"], "recall_at_10": aggregate["micro_recall_at_k"],
                              "captured_at_10": aggregate["captured_positive_windows"], "false_positives_at_10": aggregate["false_positive_windows"],
                              "macro_recall_at_10": aggregate["macro_recall_at_k"]})
        metrics.append({"algorithm": algorithm, "cases": len(test), **score})
        for split, frame in splits.items():
            for stage, estimator in [("raw", base), ("sigmoid", calibrated)]:
                p = estimator.predict_proba(frame[FEATURES])[:, 1]
                diagnostic, _ = calibration_diagnostics(frame[TARGET].to_numpy(), p, algorithm)
                diagnostics.append({"algorithm": algorithm, "split": split, "stage": stage, "cases": len(frame),
                                    "interpretation": {"train": "base_training_resubstitution", "validation": "sigmoid_calibration_resubstitution",
                                                       "test": "previously_inspected_exploratory_holdout"}[split], **diagnostic})
    return {"model_metrics": pd.DataFrame(metrics), "calibration_bins": pd.DataFrame(bins),
            "matched_alert_budgets": pd.DataFrame(budgets), "top10_by_date": pd.concat(dates, ignore_index=True),
            "split_fit_diagnostics": pd.DataFrame(diagnostics)}, predictions, reference, models


def paired_diagnostics(predictions):
    """Conditional fixed-prediction uncertainty; no model refits or winner selection."""
    y = predictions[TARGET].to_numpy()
    ids = predictions.machineID.to_numpy()
    unique = np.unique(ids)
    groups = [np.flatnonzero(ids == mid) for mid in unique]
    require(len(set(map(len, groups))) == 1, "Expected equal per-machine snapshot counts")
    groups = np.stack(groups)
    draws = np.random.default_rng(42).integers(0, len(unique), size=(500, len(unique)))
    indices = groups[draws].reshape(500, -1)
    reference = predictions.logistic_regression.to_numpy()
    ref_ap = np.array([average_precision_score(y[i], reference[i]) for i in indices])
    ref_loss = (y - reference) ** 2
    rows = []
    for algorithm in ALGORITHMS[1:]:
        if algorithm not in predictions: continue
        p = predictions[algorithm].to_numpy()
        difference = (y - p) ** 2 - ref_loss
        brier_draws = difference[indices].mean(axis=1)
        ap_draws = np.array([average_precision_score(y[i], p[i]) for i in indices]) - ref_ap
        rows.append({"algorithm": algorithm, "brier_delta_vs_lr": float(difference.mean()),
                     "brier_conditional_95_low": float(np.quantile(brier_draws, .025)),
                     "brier_conditional_95_high": float(np.quantile(brier_draws, .975)),
                     "ap_delta_vs_lr": float(average_precision_score(y, p) - average_precision_score(y, reference)),
                     "ap_conditional_95_low": float(np.quantile(ap_draws, .025)),
                     "ap_conditional_95_high": float(np.quantile(ap_draws, .975)), "machines": len(unique), "resamples": 500})
    return pd.DataFrame(rows)


def figures(tables, predictions, directory):
    plt.rcParams.update({"font.size": 11, "svg.hashsalt": "c3-algorithm-comparison-v1"})
    colors = {"logistic_regression": "#146b63", "random_forest": "#1f5f99", "hist_gradient_boosting": "#774882", "xgboost": "#82551b"}
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for algorithm, style in zip(ALGORITHMS, line_styles):
        if algorithm not in predictions: continue
        p, r, _ = precision_recall_curve(predictions[TARGET], predictions[algorithm], drop_intermediate=False)
        ap = tables["model_metrics"].set_index("algorithm").loc[algorithm, "average_precision"]
        ax.step(r, p, where="post", color=colors[algorithm], linestyle=style, linewidth=1.7, label=f"{NAMES[algorithm]}: AP {ap:.3f}")
    ax.axhline(predictions[TARGET].mean(), color="#555555", linestyle=":", label="Prevalence")
    ax.set(xlabel="Recall (positive machine-time windows)", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.03),
           title="Calibrated 7-day algorithm comparison\nSynthetic Azure benchmark; previously inspected test period")
    ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.savefig(directory / "precision_recall.png", dpi=160)
    fig.savefig(directory / "precision_recall.svg", metadata={"Date": None}); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    ax.plot([0, 1], [0, 1], color="#555555", linestyle=":", label="Ideal calibration reference")
    for algorithm, marker in zip(ALGORITHMS, markers):
        points = tables["calibration_bins"].query("method == @algorithm and cases > 0")
        if points.empty: continue
        ax.scatter(points.mean_probability, points.observed_prevalence, color=colors[algorithm], marker=marker,
                   s=65, alpha=.85, label=NAMES[algorithm])
    ax.set(xlabel="Mean predicted probability per bin", ylabel="Observed positive-window fraction", xlim=(0, 1), ylim=(0, 1),
           title="Calibration: ten fixed equal-width bins\nEmpirical points; counts in calibration_bins.csv")
    ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.savefig(directory / "calibration.png", dpi=160)
    fig.savefig(directory / "calibration.svg", metadata={"Date": None}); plt.close(fig)


def report(summary, tables, directory):
    def table(frame, columns):
        lines = ["| " + " | ".join(title for _, title in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.to_dict("records"):
            values = []
            for key, _ in columns:
                v = row[key]
                if key == "algorithm": v = NAMES[v]
                elif isinstance(v, float): v = f"{v:.6f}"
                values.append(str(v))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)
    raw = tables["split_fit_diagnostics"].query("split == 'test' and stage == 'raw'").set_index("algorithm")
    calibrated = tables["split_fit_diagnostics"].query("split == 'test' and stage == 'sigmoid'").set_index("algorithm")
    calibration_changes = raw[["brier", "ece_10_equal_width"]].join(calibrated[["brier", "ece_10_equal_width"]], lsuffix="_raw", rsuffix="_sigmoid").reset_index()
    lines = ["# Exploratory 7-day ML algorithm comparison", "",
             "**Synthetic Azure benchmark; not validated apparel-factory predictions. No model is deployed or automatically selected for deployment.**", "",
             "All algorithms use exactly the original 18 features, 1,800 training, 500 later calibration and 1,300 identical test cases. There are no category extensions or telemetry inputs.",
             "The original 30-day outer separation is retained. Three training-only expanding development folds also use a full 30-day purge; dates keep all 100 machines together.",
             "Tree settings are selected by mean uncalibrated development Brier; AP breaks exact ties, then candidate index. The 500 later cases are reserved for sigmoid calibration, with the full-training base pipeline frozen.", "",
             "## Availability and reference parity", "", f"XGBoost: {summary['xgboost_availability']}",
             f"LR reference Brier {summary['reference_reproduction']['brier']:.12f}; AP {summary['reference_reproduction']['average_precision']:.12f}; max deployed-score difference {summary['reference_reproduction']['max_probability_difference_from_artifact']:.3g}.", "",
             "## Calibrated held-out metrics", "",
             table(tables["model_metrics"], [("algorithm", "Algorithm"), ("brier", "Brier"), ("average_precision", "AP"),
                   ("precision_at_10", "Precision@10"), ("recall_at_10", "Recall@10"), ("captured_at_10", "Captured windows"),
                   ("log_loss", "Log loss"), ("ece_10_equal_width", "ECE (10 bins)"),
                   ("mean_probability_minus_prevalence", "Probability minus prevalence")]), "",
             f"Test prevalence {summary['prevalence_baseline']['test_prevalence']:.6f}; constant training-prevalence Brier {summary['prevalence_baseline']['brier']:.6f}.",
             "AP means average precision, not trapezoidal PR area. Brier/log loss reflect total probability error; ECE is bin-dependent. No calibration regression is fitted on test labels.", "",
             "## Effect of the required sigmoid calibration", "",
             table(calibration_changes, [("algorithm", "Algorithm"), ("brier_raw", "Raw Brier"), ("brier_sigmoid", "Sigmoid Brier"),
                   ("ece_10_equal_width_raw", "Raw ECE"), ("ece_10_equal_width_sigmoid", "Sigmoid ECE")]), "",
             "These raw values are diagnostics, not mixed into the calibrated main comparison. Calibration is not automatically beneficial. Sklearn uses decision_function when available (LR/HGB) and positive-class predict_proba otherwise (RF/XGBoost); the standard frozen sigmoid protocol is retained without post-test calibration searches.", "",
             "## Selected parameters", "", "```json", json.dumps(summary["selected_parameters"], indent=2), "```", "",
             "Complete fixed settings are in summary.json; all attempted candidates and fold scores are in development_scores.csv/development_summary.csv.", "",
             "## Matched alert budgets", "",
             table(tables["matched_alert_budgets"], [("algorithm", "Algorithm"), ("k", "Per-date budget"), ("captured_positive_windows", "Captured windows"),
                   ("false_positive_windows", "False positives"), ("micro_precision_at_k", "Precision"), ("micro_recall_at_k", "Recall"),
                   ("false_positives_per_100_cases", "FP per 100 evaluated")]), "",
             "Budgets 10/20/30 of 100 per date are fixed comparisons, not validated factory capacity. Top 10 micro precision is captured / 130 selections; recall is captured / 146 positive windows. These are machine-time windows, not unique physical failures or prevented breakdowns.", "",
             "## Conditional uncertainty and date consistency", "",
             table(tables["paired_differences"], [("algorithm", "Algorithm"), ("brier_delta_vs_lr", "Brier delta vs LR"),
                   ("brier_conditional_95_low", "Brier 95% low"), ("brier_conditional_95_high", "Brier 95% high"),
                   ("ap_delta_vs_lr", "AP delta vs LR"), ("ap_conditional_95_low", "AP 95% low"), ("ap_conditional_95_high", "AP 95% high")]), "",
             "These are 500 machine-cluster resamples of fixed predictions, retaining every snapshot for sampled machines. They exclude model-fitting/shared-date uncertainty and multiplicity adjustments; they do not establish independent significance or a validated winner.", "",
             table(tables["date_consistency"], [("algorithm", "Algorithm"), ("better_dates", "Top 10 better dates"),
                   ("tied_dates", "Tied dates"), ("worse_dates", "Worse dates")]), "",
             "## Calibrated training/calibration/test diagnostics", "",
             table(tables["split_fit_diagnostics"].query("stage == 'sigmoid'"), [("algorithm", "Algorithm"), ("split", "Split"),
                   ("brier", "Brier"), ("average_precision", "AP"), ("interpretation", "Interpretation")]), "",
             "## Overfitting and limitations", "",
             "split_fit_diagnostics.csv separates raw/calibrated training, calibration resubstitution and test metrics. Training/test gaps are descriptive; differences in time distribution and prevalence prevent treating them as proof of overfitting alone.",
             "Development folds are small (300 cases and 32/33/36 positives each), and fit prefixes contain 500/800/1,100 cases. Hyperparameter comparisons share machines and overlapping historical features. Same-machine overlap is intentional for this forecasting task, not a claim about unseen assets.",
             "Selection optimizes raw development Brier, not independently calibrated development performance. It deliberately reserves all 500 designated cases for final calibration. Internal random validation/early stopping is disabled; no test labels fit or select anything.",
             "The repeatedly inspected test period makes all comparisons exploratory. A small fixed search cannot establish a universal algorithm ranking. Fresh chronological evaluation, future calibration monitoring, real-factory data and validated operational costs are required before replacement.",
             "Keep the existing LR deployed during research review. Full per-date capture/selected IDs are in top10_by_date.csv. No deployed artifact, old experiment, API or UI is changed.", ""]
    daily = tables["top10_by_date"].pivot(index="prediction_time", columns="algorithm", values="captured_positive_windows")
    positive = tables["top10_by_date"].query("algorithm == 'logistic_regression'").set_index("prediction_time").positive_cases
    daily = daily.join(positive).reset_index()
    lines += ["## Top 10 capture on every test date", "",
              table(daily, [("prediction_time", "Timestamp"), ("positive_cases", "Positive windows")] +
                    [(name, NAMES[name]) for name in ALGORITHMS if name in daily]), ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(data_dir):
    splits, chronology = prepare(data_dir)
    availability = xgboost_availability()
    results, outputs = HERE / "results", HERE / "outputs"
    results.mkdir(exist_ok=True); outputs.mkdir(exist_ok=True)
    # This fixed plan is saved before tuning, calibration or test scoring.
    plan = {"features": FEATURES, "candidate_grids": GRIDS, "development_start_indices": [9, 12, 15],
            "development_score_dates_per_fold": 3, "purge_days": 30, "selection": "mean raw development Brier; mean AP then candidate index tie break",
            "calibration": "500 designated later cases only, frozen base", "alert_budgets": [10, 20, 30], "xgboost": availability,
            "model_factory_sha256": sha256(HERE / "models.py"), "evaluation_script_sha256": sha256(Path(__file__))}
    (results / "selection_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    selected, development, folds = select_parameters(splits["train"], availability)
    tables, predictions, reference, fitted = evaluate(splits, selected)
    tables["development_scores"] = development
    tables["development_summary"] = development.groupby(["algorithm", "candidate", "parameters"], sort=False).agg(
        mean_brier=("raw_development_brier", "mean"), std_brier=("raw_development_brier", "std"),
        mean_ap=("raw_development_ap", "mean"), std_ap=("raw_development_ap", "std")).reset_index()
    tables["paired_differences"] = paired_diagnostics(predictions)
    counts = tables["top10_by_date"].pivot(index="prediction_time", columns="algorithm", values="captured_positive_windows")
    consistency = []
    for algorithm in selected:
        if algorithm == "logistic_regression": continue
        delta = counts[algorithm] - counts.logistic_regression
        consistency.append({"algorithm": algorithm, "better_dates": int(delta.gt(0).sum()), "tied_dates": int(delta.eq(0).sum()),
                            "worse_dates": int(delta.lt(0).sum()), "captured_delta": int(delta.sum())})
    tables["date_consistency"] = pd.DataFrame(consistency)
    for name, frame in tables.items(): frame.to_csv(results / f"{name}.csv", index=False, float_format="%.17g")
    predictions.to_csv(outputs / "heldout_predictions.csv", index=False, float_format="%.17g")
    figures(tables, predictions, results)
    y = splits["test"][TARGET].to_numpy()
    prior = float(splits["train"][TARGET].mean())
    baseline, _ = calibration_diagnostics(y, np.full(len(y), prior), "train_prevalence")
    paths = [HERE / "models.py", Path(__file__), RESEARCH / "scripts/build_and_evaluate.py", RESEARCH / "C3_calibration_validation_experiment.py",
             RESEARCH / "scripts/model_inference.py", RESEARCH / "models/c3_7day_sigmoid.joblib",
             RESEARCH / "outputs_corrected/machine_prediction_examples.csv", RESEARCH / "outputs_calibration/calibration_brier_comparison.csv",
             RESEARCH / "experimental_error_features/run_experiment.py", RESEARCH / "experimental_thresholds/threshold_metrics.py"]
    provenance = {str(path.relative_to(RESEARCH)): sha256(path) for path in paths}
    provenance.update({f"raw/PdM_{name}.csv": sha256(Path(data_dir) / f"PdM_{name}.csv") for name in ["machines", "maint", "failures", "errors"]})
    summary = {"schema_version": "c3-algorithm-comparison-v1", "dataset": "Synthetic Azure benchmark; not validated apparel-factory data",
               "features": FEATURES, "feature_count": len(FEATURES), "chronology": chronology, "development_folds": folds,
               "selection_plan_sha256": sha256(results / "selection_plan.json"), "selected_parameters": selected,
               "full_final_classifier_parameters": {name: {key: "NaN (library default)" if isinstance(value, float) and np.isnan(value) else value
                                                             for key, value in pipeline.steps[-1][1].get_params().items()}
                                                    for name, (pipeline, _) in fitted.items()},
               "sigmoid_input_response": {name: "decision_function" if hasattr(pipeline, "decision_function") else "predict_proba[:,1]"
                                          for name, (pipeline, _) in fitted.items()},
               "xgboost_availability": availability,
               "status": {name: "ran_successfully" if name in selected else "unavailable_not_installed_or_importable" for name in ALGORITHMS},
               "reference_reproduction": reference, "metrics": tables["model_metrics"].to_dict("records"),
               "prevalence_baseline": {"train_prevalence": prior, "test_prevalence": float(y.mean()), "test_positive_cases": int(y.sum()), **baseline},
               "matched_alert_budgets": tables["matched_alert_budgets"].to_dict("records"),
               "paired_differences": tables["paired_differences"].to_dict("records"), "date_consistency": consistency,
               "fitted_per_run": {"development_models": len(development), "final_base_models": len(fitted),
                                  "validation_sigmoids": len(fitted), "deployed_artifact_refits": 0, "serialized_models": 0},
               "limitations": ["Repeatedly inspected test period; no independently validated winner.",
                               "Same-machine repeated snapshots and historical-window overlap are dependent; not evaluation on unseen machines.",
                               "Small development folds and conservative fixed grids; raw development Brier may rank candidates differently from calibrated loss.",
                               "Calibration-case diagnostic metrics are resubstitution; no independent threshold-selection set is claimed.",
                               "Only original 18 features; no telemetry, category extensions, machine-ID feature or future aggregates.",
                               "Full 2015 administrative observation is assumed; individual operating coverage/repair downtime is not verified.",
                               "Bootstrap intervals condition on fitted scores, omit training/shared-date uncertainty and lack multiplicity correction.",
                               "No operational capacity, factory fault semantics, maintenance recommendations or reliability thresholds are validated."],
               "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                            "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__, "xgboost": availability["version"]}, "source_sha256": provenance}
    (results / "summary.json").write_text(json.dumps(clean(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report(summary, tables, results)
    print("FINAL CALIBRATED METRICS", flush=True)
    print(tables["model_metrics"].to_string(index=False), flush=True)
    print(tables["paired_differences"].to_string(index=False), flush=True)
    print(tables["date_consistency"].to_string(index=False), flush=True)
    return summary, tables, predictions, fitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    run(parser.parse_args().data_dir)
