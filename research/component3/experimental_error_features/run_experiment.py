#!/usr/bin/env python3
"""Isolated 7-day error-history ablation; preserves every existing artifact."""
from __future__ import annotations

import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve

from error_features import (GENERIC_ERROR_FEATURES, build_error_features, category_feature_names,
                            feature_distribution, require, training_categories, validate_errors)

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH / "scripts"))
sys.path.insert(0, str(RESEARCH / "experimental_thresholds"))
from build_and_evaluate import FEATURE_CAT, FEATURE_NUMERIC, build_panel, load_csv, make_baseline  # noqa: E402
from threshold_metrics import choose_validation_candidates, chronology_guard, threshold_sweep, top_k_by_date  # noqa: E402

TARGET = "label_failure_7d"
METHODS = ("history_only", "existing_reference", "category_augmented")
NAMES = {"history_only": "History-only control (no errors)",
         "existing_reference": "A: existing reference (already includes generic errors)",
         "category_augmented": "B: reference + category counts"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def prepare(data_dir):
    sources = tuple(load_csv(Path(data_dir), name, name != "machines") for name in ["machines", "maint", "failures", "errors"])
    machines, maintenance, failures, errors = sources
    errors = validate_errors(errors, machines.machineID)
    panel = build_panel(machines, maintenance, failures, errors)
    saved = pd.read_csv(RESEARCH / "outputs_corrected/machine_prediction_examples.csv", parse_dates=["prediction_time"])
    keys = ["machineID", "prediction_time"]
    require(not panel.duplicated(keys).any(), "Duplicate machine-time cases")
    pd.testing.assert_frame_equal(panel.set_index(keys).sort_index(), saved.set_index(keys).sort_index(),
                                  check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    splits = {split: panel[panel.split.eq(split)].copy() for split in ["train", "validation", "test"]}
    require([len(frame) for frame in splits.values()] == [1800, 500, 1300], "Original cohort counts changed")
    chronology = chronology_guard(*splits.values())
    require(panel.prediction_time.max() + pd.Timedelta(days=30) < pd.Timestamp("2016-01-01"),
            "Incomplete outcomes under the original benchmark administrative-coverage assumption")
    # Independent event lookup checks the strict label, not the metadata's erroneous >= text.
    events = {int(mid): np.sort(g.datetime.astype("int64").to_numpy()) for mid, g in failures.groupby("machineID")}
    labels = []
    for mid, t in zip(panel.machineID, panel.prediction_time):
        values = events.get(int(mid), np.empty(0, dtype="int64"))
        labels.append(int(np.searchsorted(values, (t + pd.Timedelta(days=7)).value, side="left") >
                          np.searchsorted(values, t.value, side="right")))
    require(np.array_equal(labels, panel[TARGET]), "Strict-future labels changed")
    vocabulary = training_categories(errors, splits["train"].prediction_time.max())
    extra_names = category_feature_names(vocabulary)
    require(len(extra_names) > 0, "Training history lacks supported category detail")
    generated = build_error_features(panel, errors, vocabulary)
    pd.testing.assert_frame_equal(generated[GENERIC_ERROR_FEATURES], panel[GENERIC_ERROR_FEATURES],
                                  check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    panel[extra_names] = generated[extra_names]
    splits = {split: panel[panel.split.eq(split)].copy() for split in splits}
    training_records = errors[errors.datetime < splits["train"].prediction_time.max()]
    quality = {"schema": errors.dtypes.astype(str).to_dict(), "records": len(errors),
               "missing_fields": errors.isna().sum().to_dict(), "duplicate_complete_rows": int(errors.duplicated().sum()),
               "additional_records_at_shared_machine_times": int(errors.duplicated(["machineID", "datetime"]).sum()),
               "shared_timestamp_handling": "Distinct error categories retained as distinct records; not confirmed failures",
               "first_error": str(errors.datetime.min()), "last_error": str(errors.datetime.max()),
               "machine_count": int(errors.machineID.nunique()), "unknown_machine_ids": sorted(set(errors.machineID) - set(machines.machineID)),
               "raw_rows_already_time_sorted": bool(errors.datetime.is_monotonic_increasing),
               "category_counts_full_file_descriptive_only": errors.errorID.value_counts().to_dict(),
               "category_machine_counts_full_file": errors.groupby("errorID").machineID.nunique().to_dict(),
               "vocabulary_cutoff_exclusive": str(splits["train"].prediction_time.max()),
               "category_counts_before_training_cutoff": training_records.errorID.value_counts().to_dict(),
               "frozen_training_vocabulary": vocabulary, "omitted_reference_category": vocabulary[0],
               "interpretation": "Generic Azure categories have no validated sewing-machine fault semantics"}
    return panel, splits, vocabulary, quality, chronology


def numeric_features(method, vocabulary):
    if method == "history_only": return [name for name in FEATURE_NUMERIC if name not in GENERIC_ERROR_FEATURES]
    if method == "existing_reference": return list(FEATURE_NUMERIC)
    require(method == "category_augmented", "Unknown ablation model")
    return list(FEATURE_NUMERIC) + category_feature_names(vocabulary)


def fit_calibrated(train, validation, numeric):
    """Original LR/preprocessing configuration with explicit feature columns."""
    base = make_baseline()
    transformers = base.named_steps["transform"].transformers
    base.set_params(transform__transformers=[(name, transformer, numeric if name == "num" else columns)
                                             for name, transformer, columns in transformers])
    columns = numeric + FEATURE_CAT
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        base.fit(train[columns], train[TARGET].to_numpy())
        frozen_hash = joblib.hash(base)
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        calibrated.fit(validation[columns], validation[TARGET].to_numpy())
    require(joblib.hash(base) == frozen_hash, "Calibration modified the training-fitted estimator/preprocessing")
    return base, calibrated


def calibration_diagnostics(y, p, method):
    bins = []
    assigned = np.minimum((p * 10).astype(int), 9)
    for index in range(10):
        selected = assigned == index
        bins.append({"method": method, "bin": index + 1, "lower": index / 10, "upper": (index + 1) / 10,
                     "cases": int(selected.sum()), "mean_probability": float(p[selected].mean()) if selected.any() else np.nan,
                     "observed_prevalence": float(y[selected].mean()) if selected.any() else np.nan})
    ece = sum(b["cases"] * abs(b["mean_probability"] - b["observed_prevalence"]) for b in bins if b["cases"]) / len(y)
    return {"brier": float(brier_score_loss(y, p)), "average_precision": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "mean_probability": float(p.mean()),
            "prevalence": float(y.mean()), "mean_probability_minus_prevalence": float(p.mean() - y.mean()),
            "ece_10_equal_width": float(ece)}, bins


def verify_reference(test, p):
    previous = pd.read_csv(RESEARCH / "outputs_calibration/calibration_brier_comparison.csv")
    row = previous[previous.horizon_days.eq(7) & previous.method.eq("sigmoid_calibrated_on_later_validation")]
    require(len(row) == 1, "Missing original chronological reference metrics")
    brier, ap = brier_score_loss(test[TARGET], p), average_precision_score(test[TARGET], p)
    require(round(brier, 6) == row.iloc[0].brier_score and round(ap, 6) == row.iloc[0].pr_auc_average_precision,
            "Reference parity failed: stop before interpreting the ablation")
    artifact = joblib.load(RESEARCH / "models/c3_7day_sigmoid.joblib")
    require(artifact["horizon_days"] == 7 and artifact["feature_columns"] == FEATURE_NUMERIC + FEATURE_CAT,
            "Deployed artifact feature definition changed")
    deployed = artifact["model"].predict_proba(test[artifact["feature_columns"]])[:, 1]
    delta = float(np.abs(deployed - p).max())
    require(delta <= 1e-10, "Reference probabilities disagree with the deployed 7-day artifact")
    return {"brier": float(brier), "average_precision": float(ap), "saved_brier": float(row.iloc[0].brier_score),
            "saved_average_precision": float(row.iloc[0].pr_auc_average_precision), "max_probability_difference_from_artifact": delta}


def paired_brier(machine_ids, labels, probabilities):
    ids = np.unique(machine_ids)
    draws = np.random.default_rng(42).integers(0, len(ids), size=(500, len(ids)))
    sizes = np.array([np.sum(machine_ids == mid) for mid in ids])
    rows = []
    for previous, augmented in [("history_only", "existing_reference"), ("existing_reference", "category_augmented")]:
        differences = (labels - probabilities[augmented]) ** 2 - (labels - probabilities[previous]) ** 2
        sums = np.array([differences[machine_ids == mid].sum() for mid in ids])
        bootstrap = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        rows.append({"reference": previous, "augmented": augmented, "brier_delta": float(differences.mean()),
                     "machine_cluster_95_low": float(np.quantile(bootstrap, .025)),
                     "machine_cluster_95_high": float(np.quantile(bootstrap, .975)), "resamples": 500, "machines": len(ids)})
    return pd.DataFrame(rows)


def evaluate(panel, splits, vocabulary):
    train, validation, test = (splits[name] for name in ["train", "validation", "test"])
    models, predictions, metrics, bins, top10, budgets, thresholds, coefficients, split_metrics = {}, {}, [], [], [], [], [], [], []
    reference_check = None
    # Reference reproduction is a gate before either ablation model is fitted.
    for method in ["existing_reference", "history_only", "category_augmented"]:
        numeric = numeric_features(method, vocabulary)
        base, calibrated = fit_calibrated(train, validation, numeric)
        columns = numeric + FEATURE_CAT
        pt = calibrated.predict_proba(test[columns])[:, 1]
        pv = calibrated.predict_proba(validation[columns])[:, 1]
        require(np.isfinite(pt).all() and ((pt >= 0) & (pt <= 1)).all(), "Invalid calibrated probabilities")
        if method == "existing_reference": reference_check = verify_reference(test, pt)
        predictions[method] = pt
        models[method] = (base, calibrated, columns)
        for split, frame in splits.items():
            p = calibrated.predict_proba(frame[columns])[:, 1]
            score, _ = calibration_diagnostics(frame[TARGET].to_numpy(), p, method)
            split_metrics.append({"method": method, "split": split, "cases": len(frame),
                                  "interpretation": {"train": "in_sample_base_training_cases", "validation": "sigmoid_calibration_resubstitution",
                                                     "test": "previously_inspected_exploratory_holdout"}[split], **score})
        diagnostics, calibration_bins = calibration_diagnostics(test[TARGET].to_numpy(), pt, method)
        bins += calibration_bins
        ranked = test[["machineID", "prediction_time"]].assign(label=test[TARGET].to_numpy(), probability=pt)
        for k in [10, 20, 30]:
            dates, aggregate = top_k_by_date(ranked, k=k)
            budgets.append({"method": method, **aggregate, "flagged_pct": 100 * aggregate["selected_machine_time_cases"] / len(test),
                            "false_positives_per_100_cases": 100 * aggregate["false_positive_windows"] / len(test)})
            if k == 10:
                top10.append(dates.assign(method=method))
                diagnostics.update({"precision_at_10": aggregate["micro_precision_at_k"],
                                    "recall_at_10": aggregate["micro_recall_at_k"],
                                    "captured_at_10": aggregate["captured_positive_windows"]})
        metrics.append({"method": method, "raw_feature_count": len(columns), "test_cases": len(test), **diagnostics})
        for choice in choose_validation_candidates(validation[TARGET].to_numpy(), pv):
            if not choice["candidate"].startswith("recall_"): continue
            row = {"method": method, **choice, "selection": "exploratory_reuse_of_sigmoid_calibration_cases"}
            for split, labels, p in [("validation", validation[TARGET].to_numpy(), pv), ("test", test[TARGET].to_numpy(), pt)]:
                measured = threshold_sweep(labels, p, [choice["threshold"]]).to_dict("records")[0]
                row.update({f"{split}_{key}": value for key, value in measured.items() if key != "threshold"})
            thresholds.append(row)
        names = base.named_steps["transform"].get_feature_names_out()
        for name, coefficient in zip(names, base.named_steps["logistic"].coef_[0]):
            coefficients.append({"method": method, "transformed_feature": name, "training_base_coefficient": float(coefficient),
                                 "interpretation": "Training-fitted correlated feature coefficient; not causal or validated feature importance"})
    metric_frame = pd.DataFrame(metrics).set_index("method").loc[list(METHODS)].reset_index()
    comparisons = []
    for before, after in [("history_only", "existing_reference"), ("existing_reference", "category_augmented")]:
        values = metric_frame.set_index("method")
        for name in ["brier", "average_precision", "precision_at_10", "recall_at_10", "ece_10_equal_width", "log_loss"]:
            comparisons.append({"reference": before, "augmented": after, "metric": name,
                                "reference_value": float(values.loc[before, name]), "augmented_value": float(values.loc[after, name]),
                                "difference_augmented_minus_reference": float(values.loc[after, name] - values.loc[before, name])})
    availability = panel.groupby("split", sort=False).agg(cases=("machineID", "size"), timestamps=("prediction_time", "nunique"),
                                                         cases_with_prior_errors=("error_events_prior", lambda s: int(s.gt(0).sum()))).reset_index()
    availability["cases_without_prior_errors"] = availability.cases - availability.cases_with_prior_errors
    availability["prior_error_available_pct"] = 100 * availability.cases_with_prior_errors / availability.cases
    tables = {"model_metrics": metric_frame, "split_fit_diagnostics": pd.DataFrame(split_metrics),
              "comparisons": pd.DataFrame(comparisons), "calibration_bins": pd.DataFrame(bins),
              "top10_by_date": pd.concat(top10, ignore_index=True), "matched_alert_budgets": pd.DataFrame(budgets),
              "exploratory_recall_thresholds": pd.DataFrame(thresholds), "training_coefficients": pd.DataFrame(coefficients),
              "error_feature_distribution": feature_distribution(panel, GENERIC_ERROR_FEATURES + category_feature_names(vocabulary)),
              "error_availability": availability,
              "paired_brier_differences": paired_brier(test.machineID.to_numpy(), test[TARGET].to_numpy(), predictions)}
    detail = test[["machineID", "prediction_time", TARGET]].copy()
    for method in METHODS: detail[method] = predictions[method]
    return tables, reference_check, detail, models


def plot_figures(tables, detail, directory):
    plt.rcParams.update({"font.size": 11, "svg.hashsalt": "c3-error-history-ablation-v1"})
    styles = [("history_only", "#626977", "^"), ("existing_reference", "#126b64", "o"), ("category_augmented", "#1d5c91", "s")]
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for method, color, _ in styles:
        precision, recall, _ = precision_recall_curve(detail[TARGET], detail[method], drop_intermediate=False)
        ap = tables["model_metrics"].set_index("method").loc[method, "average_precision"]
        ax.step(recall, precision, where="post", color=color, label=f"{NAMES[method]}: AP {ap:.3f}", linewidth=1.5)
    ax.axhline(detail[TARGET].mean(), color="black", linestyle=":", label="Test prevalence")
    ax.set(xlabel="Recall (positive machine-time windows)", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.03),
           title="7-day exploratory feature ablation\nSynthetic Azure benchmark; 1,300 previously inspected test cases")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(directory / "precision_recall.png", dpi=160)
    fig.savefig(directory / "precision_recall.svg", metadata={"Date": None}); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", label="Ideal calibration reference")
    for method, color, marker in styles:
        points = tables["calibration_bins"].query("method == @method and cases > 0")
        ax.scatter(points.mean_probability, points.observed_prevalence, color=color, marker=marker,
                   s=65, alpha=.85, label=NAMES[method])
    ax.set(xlabel="Mean predicted probability per bin", ylabel="Observed failure-window fraction", xlim=(0, 1), ylim=(0, 1),
           title="Calibration: ten fixed equal-width bins\nEmpirical points; empty bins omitted; counts in numerical table")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.2)
    fig.savefig(directory / "calibration.png", dpi=160)
    fig.savefig(directory / "calibration.svg", metadata={"Date": None}); plt.close(fig)


def write_report(summary, tables, directory):
    """All displayed numerical values come directly from this experiment."""
    display = {"history_only": "History-only control", "existing_reference": "A: existing reference", "category_augmented": "B: category extension"}
    measured = tables["model_metrics"].set_index("method")
    positive_cases = summary["test_positive_cases"]
    test_cases = int(measured.loc["existing_reference", "test_cases"])
    selected_cases = int(tables["matched_alert_budgets"].query("method == 'existing_reference' and k == 10").iloc[0].selected_machine_time_cases)
    contrasts = []
    for previous, augmented in [("history_only", "existing_reference"), ("existing_reference", "category_augmented")]:
        contrasts.append(f"{display[augmented]} versus {display[previous]}: Brier change {measured.loc[augmented, 'brier'] - measured.loc[previous, 'brier']:+.6f}, AP change {measured.loc[augmented, 'average_precision'] - measured.loc[previous, 'average_precision']:+.6f}, Top 10 captured-window change {int(measured.loc[augmented, 'captured_at_10'] - measured.loc[previous, 'captured_at_10']):+d}.")
    def table(frame, columns):
        lines = ["| " + " | ".join(title for _, title in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.to_dict("records"):
            cells = []
            for name, _ in columns:
                value = row[name]
                if name in ["method", "reference", "augmented"]: value = display[value]
                elif value is None or isinstance(value, float) and not np.isfinite(value): value = "undefined"
                elif isinstance(value, float): value = f"{value:.6f}"
                cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
    lines = ["# Historical-error ablation: exploratory results", "",
             "**Synthetic Azure industrial-machine benchmark, not validated apparel-factory data. No telemetry is used.**", "",
             "## Baseline-definition correction", "",
             "The existing deployed-feature reference already includes five generic error-history features. Its published Brier/AP cannot be attributed to a maintenance/failure-history-only model.",
             "This experiment therefore includes a history-only control, A (the exact existing reference), and B (A plus category counts). The first contrast estimates generic-error value; the second estimates additional category detail.", "",
             "## Reference parity and cohort", "",
             f"Reference Brier: {summary['reference_reproduction']['brier']:.12f}; AP: {summary['reference_reproduction']['average_precision']:.12f}; maximum score difference from deployed artifact: {summary['reference_reproduction']['max_probability_difference_from_artifact']:.3g}.",
             "Training/calibration/test: 1,800 / 500 / 1,300 original machine-time cases, with the original full 30-day separation. Only events strictly before prediction time are features. Targets use t < failure < t+7 days.",
             f"Test prevalence: {summary['test_positive_cases']}/1,300 = {100 * summary['test_prevalence']:.4f}%. Training-prevalence constant baseline Brier: {summary['train_prevalence_baseline_brier']:.9f}.", "",
             "## Held-out model comparison", "",
             table(tables["model_metrics"], [("method", "Model"), ("brier", "Brier"), ("average_precision", "AP"),
                   ("precision_at_10", "Precision@10"), ("recall_at_10", "Recall@10"), ("captured_at_10", "Captured windows"),
                   ("ece_10_equal_width", "ECE (10 bins)"), ("log_loss", "Log loss")]), "",
             "Lower Brier/log loss/ECE is better; higher AP is better. AP is average precision, not trapezoidal PR area. ECE is bin-dependent; Brier alone does not prove calibration improvement.", "",
             table(tables["comparisons"], [("reference", "Reference"), ("augmented", "Augmented"), ("metric", "Metric"),
                   ("difference_augmented_minus_reference", "Augmented minus reference")]), "",
             "## Interpretation", "",
             *contrasts,
             "These are preliminary group-level predictive associations, not causal effects or a validated benefit of every error feature. The measured category extension improves Brier/log loss slightly but worsens AP, Top 10 capture and ten-bin ECE: it is not an across-metric improvement.",
             "Conditional machine-cluster bootstrap diagnostics (500 resamples of fixed predictions) are below. They exclude model-fitting and shared-date uncertainty, have no multiplicity correction, and are not a claim of externally validated statistical significance.", "",
             table(tables["paired_brier_differences"], [("reference", "Reference"), ("augmented", "Augmented"), ("brier_delta", "Brier delta"),
                   ("machine_cluster_95_low", "Conditional 95% low"), ("machine_cluster_95_high", "Conditional 95% high")]), "",
             "## False positives at matched alert budgets", "",
             "Select the highest-scoring 10, 20 or 30 of 100 machines independently per date. These budgets are fixed research comparisons, not validated factory inspection capacity. No test labels select these budgets or the ranked machines.", "",
             table(tables["matched_alert_budgets"], [("method", "Model"), ("k", "Per-date budget"), ("selected_machine_time_cases", "Selected windows"),
                   ("captured_positive_windows", "Captured"), ("false_positive_windows", "False positives"),
                   ("micro_precision_at_k", "Precision"), ("micro_recall_at_k", "Recall"), ("false_positives_per_100_cases", "FP / 100 evaluated")]), "",
             "Matched-budget false positives show no consistent reduction from category detail. Threshold-independent rankings do not change merely by selecting another classification threshold.", "",
             "## Exploratory validation-derived recall thresholds", "",
             "All 500 validation cases fitted sigmoid calibration. Reusing them for 50/70/80% recall candidates is not independent threshold validation. These thresholds never use test labels and are evaluated unchanged on test. Different observed test recalls prevent interpreting their FP counts as an exactly matched-recall comparison; fixed alert-budget comparisons above avoid that ambiguity.", "",
             table(tables["exploratory_recall_thresholds"], [("method", "Model"), ("candidate", "Validation target"), ("threshold", "Threshold"),
                   ("validation_recall", "Calibration-case recall"), ("test_precision", "Test precision"), ("test_recall", "Test recall"),
                   ("test_flagged_cases", "Test flagged"), ("test_FP", "Test FP")]), "",
             "## Prior-error availability and sparsity", "",
             table(tables["error_availability"], [("split", "Split"), ("cases", "Cases"), ("cases_with_prior_errors", "Prior errors available"),
                   ("cases_without_prior_errors", "No prior errors"), ("prior_error_available_pct", "Available %")]), "",
             "No-history error recency is NaN; training-only median imputation and its missingness indicator handle it. Only three training cases have no prior errors, so behavior for genuinely missing error logs is not validated by this benchmark.", "",
             table(tables["error_feature_distribution"].query("split == 'test'"), [("feature", "Feature"), ("zero_pct", "Zero %"),
                   ("mean", "Mean"), ("median", "Median"), ("min", "Minimum"), ("max", "Maximum")]), "",
             "Category error5 counts are especially sparse. Correlated base-model coefficients are available in training_coefficients.csv. Among category terms, error2 has the largest absolute fitted coefficient, followed by error3; this is not validated independent feature importance, significance or a sewing-machine fault interpretation.", "",
             "## Top 10 capture across every test timestamp", "",
             f"Micro precision is captured / {selected_cases} selected windows; micro recall is captured / {positive_cases} positive windows. Repeated landmarks are dependent. Captured windows are not unique physical failures, repairs or prevented breakdowns. Ties use ascending machine ID after descending full-precision probability.", ""]
    dates = tables["top10_by_date"].pivot(index="prediction_time", columns="method", values="captured_positive_windows")
    positives = tables["top10_by_date"].query("method == 'existing_reference'").set_index("prediction_time").positive_cases
    dates = dates.join(positives).reset_index()
    lines += [table(dates, [("prediction_time", "Timestamp"), ("positive_cases", "Positive windows"),
                            ("history_only", "History-only captured"), ("existing_reference", "A captured"), ("category_augmented", "B captured")]), "",
              f"Uniform random Top 10 selection has expected precision equal to test prevalence and expected recall 10%; expected capture is {selected_cases * positive_cases / test_cases:.1f} positive windows over these dates. This is a descriptive expectation, not a significance test.", "",
              "## Recommendation and limitations", "",
              "Keep the existing calibrated reference for the current demo. Treat additional category features as optional research inputs. Keep a history-only fallback as a separately validated model rather than filling missing error logs with invented zero histories.",
              "Continue investigating only with predeclared evaluation on genuinely fresh chronological data. The benchmark is synthetic, repeated machine snapshots are dependent, and the test period has already been inspected. Factory error semantics, actual logging coverage, inspection capacity and missed-failure/false-alarm costs remain unvalidated.",
              "No temporal or fitting leakage was detected by the checks, but prior test reuse limits confirmatory interpretation. Training and calibration-case diagnostic metrics are in split_fit_diagnostics.csv; calibration resubstitution is not independent validation. No test-driven search, feature selection or production integration was performed.", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(data_dir):
    panel, splits, vocabulary, quality, chronology = prepare(data_dir)
    tables, reference, detail, models = evaluate(panel, splits, vocabulary)
    results, outputs = HERE / "results", HERE / "outputs"
    results.mkdir(exist_ok=True); outputs.mkdir(exist_ok=True)
    for name, table in tables.items(): table.to_csv(results / f"{name}.csv", index=False, float_format="%.17g")
    detail.to_csv(outputs / "heldout_predictions.csv", index=False, float_format="%.17g")
    panel[["machineID", "prediction_time", "split", *GENERIC_ERROR_FEATURES, *category_feature_names(vocabulary)]].to_csv(
        outputs / "historical_error_features.csv", index=False, float_format="%.17g")
    plot_figures(tables, detail, results)
    train_prevalence = float(splits["train"][TARGET].mean())
    paths = [Path(__file__), HERE / "error_features.py", RESEARCH / "scripts/build_and_evaluate.py",
             RESEARCH / "C3_calibration_validation_experiment.py", RESEARCH / "scripts/model_inference.py",
             RESEARCH / "models/c3_7day_sigmoid.joblib", RESEARCH / "outputs_calibration/calibration_brier_comparison.csv",
             RESEARCH / "outputs_corrected/machine_prediction_examples.csv", RESEARCH / "experimental_thresholds/threshold_metrics.py"]
    provenance = {str(path.relative_to(RESEARCH)): sha256(path) for path in paths}
    provenance.update({f"raw/PdM_{name}.csv": sha256(Path(data_dir) / f"PdM_{name}.csv") for name in ["machines", "maint", "failures", "errors"]})
    summary = {"schema_version": "c3-error-history-ablation-v1", "dataset": "Synthetic Azure industrial-machine data; not validated apparel-factory data",
               "baseline_correction": "The existing baseline already contains five generic error features; its published metrics cannot be attributed to a history-only model.",
               "comparison": {"history_only": "Control: existing numeric features minus all five error features; unchanged machine-model category",
                              "existing_reference": "A: exact original 17 numeric features plus machine-model category, including generic errors",
                              "category_augmented": "B: exact existing reference plus four training-vocabulary category counts over prior 30 days"},
               "error_data_quality": quality, "chronology": chronology, "reference_reproduction": reference,
               "model_features": {method: numeric_features(method, vocabulary) + FEATURE_CAT for method in METHODS},
               "model_configuration": {"logistic": clean(models["existing_reference"][0].named_steps["logistic"].get_params()),
                                       "preprocessing": "Original median imputation with missing indicators, scaling and machine-model one-hot encoding; fit training only",
                                       "calibration": "Original frozen base pipeline; sigmoid fitted on all 500 later validation cases"},
               "split_counts": {split: len(frame) for split, frame in splits.items()}, "test_positive_cases": int(detail[TARGET].sum()),
               "train_prevalence": train_prevalence, "test_prevalence": float(detail[TARGET].mean()),
               "train_prevalence_baseline_brier": float(brier_score_loss(detail[TARGET], np.full(len(detail), train_prevalence))),
               "metrics": tables["model_metrics"].to_dict("records"), "comparisons": tables["comparisons"].to_dict("records"),
               "matched_alert_budgets": tables["matched_alert_budgets"].to_dict("records"),
               "paired_brier_differences": tables["paired_brier_differences"].to_dict("records"),
               "fitted_per_run": {"new_research_logistic_models": 3, "validation_sigmoids": 3, "deployed_model_refits": 0, "serialized_models": 0},
               "limitations": ["All test findings are exploratory; the same benchmark period has been examined repeatedly.",
                               "Errors are generic recorded events, not confirmed failures or validated apparel fault categories.",
                               "No telemetry, temperatures, voltages, vibration, pressure or future-year aggregates are model features.",
                               "No prior errors means no recorded events, not verified error-free operation or proven complete logging.",
                               "Recency is time since the latest recorded error, not database freshness; no-history recency is NaN, imputed using training only.",
                               "Repeated machines and overlapping windows are dependent; Top 10 counts positive machine-time windows, not unique failures.",
                               "Recall thresholds reuse sigmoid calibration cases and are not independently validated; test labels never select them.",
                               "Top 10/20/30 are fixed illustrative alert budgets, not validated inspection capacity or maintenance recommendations.",
                               "Coefficients are correlated training-model diagnostics, not causal effects or validated feature importance.",
                               "Cluster bootstrap retains all snapshots per sampled machine but conditions on fitted models, omits shared-date/fitting uncertainty and has no multiplicity correction.",
                               "Complete target follow-up assumes the benchmark machines remained observed through 2015; individual downtime/coverage is unverified.",
                               "Feature sets were fixed before the comparison; no test-driven hyperparameter search or feature selection was performed."],
               "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                            "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__}, "source_sha256": provenance}
    (results / "summary.json").write_text(json.dumps(clean(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary, tables, results)
    print("BASELINE CORRECTION: existing reference already includes five generic error-history features.")
    print(tables["model_metrics"].to_string(index=False))
    print(tables["comparisons"].to_string(index=False))
    print(tables["matched_alert_budgets"][["method", "k", "captured_positive_windows", "false_positive_windows", "micro_precision_at_k", "micro_recall_at_k"]].to_string(index=False))
    return summary, tables, detail, models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    run(parser.parse_args().data_dir)
