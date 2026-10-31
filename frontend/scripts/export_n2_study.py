#!/usr/bin/env python3
"""Export/verify aggregate N2 results using saved files only (Python stdlib).

No research modules, ML libraries, raw datasets or model artifacts are imported.
Run from any directory. --check reads without writing. The normal frontend uses
the checked-in aggregate snapshot and never requires these local source files.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import sys

FRONTEND = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = FRONTEND.parent / "research/component3"
DEFAULT_OUTPUT = FRONTEND / "src/components/component3/data/n2Study.snapshot.json"
CONDITIONS = [("full", "Full observed history", "Full", None),
              ("180d", "180-day history", "180 days", 180),
              ("90d", "90-day history", "90 days", 90)]
VARIABLES = {"H": "observed_maintenance_span_days", "M": "maintenance_events_prior",
             "F": "failure_events_prior", "R": "days_since_last_maintenance"}
ALIASES = {"H": "H_observed_span_days", "M": "M_maintenance_count",
           "F": "F_prior_failure_count", "R": "R_days_since_maintenance"}
MODELS = ["constant_mean_loss", "risk_only", "risk_plus_recency",
          "risk_plus_HMF_machine", "risk_plus_HMFR_machine"]
FILES = ["outputs_corrected/n2_initial_window_comparison.csv",
         "outputs_corrected/n2_paired_test_prediction_losses.csv",
         "outputs_n2_continuous/n2_prediction_evidence_and_losses.csv",
         "outputs_n2_continuous/n2_descriptive_spearman_correlations.csv",
         "outputs_n2_continuous/n2_evidence_loss_quantile_summary.csv",
         "outputs_n2_continuous/n2_grouped_cv_relationship_metrics.csv",
         "outputs_n2_continuous/n2_grouped_cv_loss_predictions.csv",
         "outputs_n2_continuous/n2_methodology_and_limitations.json",
         "outputs_n2_incremental/n2_risk_vs_evidence_cv_metrics.csv",
         "outputs_n2_incremental/n2_risk_vs_evidence_cv_predictions.csv",
         "outputs_n2_incremental/n2_risk_vs_evidence_incremental_mae.csv",
         "outputs_n2_incremental/n2_risk_vs_evidence_methodology.json"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    result = float(value)
    require(math.isfinite(result), f"Non-finite number: {value!r}")
    return result


def close(actual, expected, context, tolerance=1e-10):
    require(abs(number(actual) - number(expected)) <= tolerance, f"Inconsistent {context}: {actual} != {expected}")


def key(row):
    return row["history_condition"], int(row["machineID"]), row["prediction_time"]


def index_rows(rows):
    indexed = {key(row): row for row in rows}
    require(len(indexed) == len(rows), "Duplicate machine/time/condition rows")
    return indexed


def ranks(values):
    ordered = sorted(values)
    positions = defaultdict(list)
    for i, value in enumerate(ordered, 1):
        positions[value].append(i)
    return [fmean(positions[value]) for value in values]


def spearman(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    a, b = ranks(x), ranks(y)
    ma, mb = fmean(a), fmean(b)
    return sum((u-ma)*(v-mb) for u, v in zip(a, b)) / math.sqrt(
        sum((u-ma)**2 for u in a) * sum((v-mb)**2 for v in b))


def check_oof(rows, source, loss_column, estimate_columns):
    indexed = index_rows(rows)
    require(indexed.keys() == source.keys(), "OOF cohort does not match paired losses")
    machine_folds = {}
    for k, row in indexed.items():
        original = source[k]
        close(row[loss_column], original["sq_error"], "OOF observed loss")
        fold = int(row["machine_group_fold"])
        require(1 <= fold <= 5, "Invalid machine-group fold")
        group = k[:2]
        require(machine_folds.setdefault(group, fold) == fold, "Machine crosses OOF folds")
        for column in estimate_columns:
            require(0 <= number(row[column]) <= 1, "OOF loss estimate outside [0, 1]")
    return indexed


def metrics(rows, target, estimate):
    y = [number(row[target]) for row in rows]
    p = [number(row[estimate]) for row in rows]
    mse = fmean([(a-b)**2 for a, b in zip(y, p)])
    mean = fmean(y)
    variance = fmean([(a-mean)**2 for a in y])
    return {"mae": fmean([abs(a-b) for a, b in zip(y, p)]), "rmse": math.sqrt(mse), "r2": 1-mse/variance}


def build_snapshot(source_dir: Path):
    tables = {}
    provenance = []
    for name in FILES:
        path = source_dir / name
        raw = path.read_bytes()  # Missing sources fail explicitly; no fallback.
        if path.suffix == ".csv":
            tables[path.name] = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
            require(bool(tables[path.name]), f"Empty source: {name}")
        else:
            tables[path.name] = json.loads(raw)
        provenance.append({"path": "research/component3/" + name,
                           "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                           **({"rows": len(tables[path.name])} if path.suffix == ".csv" else {})})
    source = index_rows(tables["n2_paired_test_prediction_losses.csv"])
    linked = index_rows(tables["n2_prediction_evidence_and_losses.csv"])
    require(source.keys() == linked.keys(), "Condition-specific evidence cohort differs")
    for k, row in source.items():
        p, y = number(row["predicted_probability"]), number(row["label_failure_7d"])
        require(0 <= p <= 1 and y in (0, 1), "Invalid probability or binary outcome")
        close(row["sq_error"], (y-p)**2, "squared prediction loss")
        for field in ["predicted_probability", "label_failure_7d", "sq_error"]:
            close(row[field], linked[k][field], f"linked {field}")
        for symbol, variable in VARIABLES.items():
            close(linked[k][variable], linked[k][ALIASES[symbol]], f"{symbol} evidence alias")
        close(linked[k]["C_field_completeness_pct"], 100, "constant C")
        close(linked[k]["published_field_completeness_pct"], 100, "published completeness")
    expected_conditions = {item[1] for item in CONDITIONS}
    require({k[0] for k in source} == expected_conditions, "Missing/unknown history conditions")
    method = tables["n2_methodology_and_limitations.json"]
    incremental_method = tables["n2_risk_vs_evidence_methodology.json"]
    require(method["prediction_horizon_days"] == 7 and set(method["history_conditions"]) == expected_conditions,
            "Unexpected experimental methodology")
    require("uncalibrated" in method["loss_definition"] and "uncalibrated" in incremental_method["loss"],
            "Expected original uncalibrated N2 models")
    require(incremental_method["comparison"] == MODELS, "Unexpected incremental alternatives")
    continuous_oof = check_oof(tables["n2_grouped_cv_loss_predictions.csv"], source,
                               "observed_individual_sq_error", ["crossfit_estimated_conditional_mean_loss", "crossfit_constant_baseline_mean_loss"])
    incremental_oof = check_oof(tables["n2_risk_vs_evidence_cv_predictions.csv"], source,
                                "sq_error", ["estimated_loss__"+model for model in MODELS])
    for filename, expected in [("n2_initial_window_comparison.csv", 3),
                               ("n2_descriptive_spearman_correlations.csv", 15),
                               ("n2_grouped_cv_relationship_metrics.csv", 3),
                               ("n2_risk_vs_evidence_cv_metrics.csv", 15),
                               ("n2_risk_vs_evidence_incremental_mae.csv", 3)]:
        require(len(tables[filename]) == expected, f"Unexpected aggregate row count: {filename}")
    for oof in [continuous_oof, incremental_oof]:
        for condition in expected_conditions:
            require({int(row["machine_group_fold"]) for k, row in oof.items() if k[0] == condition} == set(range(1, 6)),
                    "Expected five nonempty machine-group folds per condition")

    def one(filename, **filters):
        found = [row for row in tables[filename] if all(row[k] == value for k, value in filters.items())]
        require(len(found) == 1, f"Expected one {filename} row for {filters}")
        return found[0]

    conditions, reference = [], None
    for identifier, label, short, lookback in CONDITIONS:
        rows = [row for k, row in source.items() if k[0] == label]
        cohort = {(k[1], k[2]): number(row["label_failure_7d"]) for k, row in source.items() if k[0] == label}
        if reference is None:
            reference = cohort
        require(cohort == reference, "Paired conditions have different cases or outcomes")
        machines, dates = {k[0] for k in cohort}, {k[1] for k in cohort}
        require(len(rows) == method["n_paired_machine_snapshots_per_condition"] and
                len(machines) == method["n_machines_per_condition"], "Metadata sample counts disagree")
        brier = fmean([number(row["sq_error"]) for row in rows])
        positives = int(sum(cohort.values()))
        window = one("n2_initial_window_comparison.csv", history_condition=label)
        close(brier, window["brier_score"], "rounded Brier summary", tolerance=0.0000051)
        close(window["positive_cases"], positives, "positive cases")
        close(window["sample_count"], len(rows), "window sample count")
        require(window["trained_separately"] == "True", "History models must be separately refitted")
        evidence = {}
        detail = [row for k, row in linked.items() if k[0] == label]
        for symbol, variable in VARIABLES.items():
            values = [number(row[variable]) for row in detail]
            rho = spearman(values, [number(row["sq_error"]) for row in detail])
            corr = one("n2_descriptive_spearman_correlations.csv", history_condition=label, evidence_variable=variable)
            close(rho, corr["spearman_rho_descriptive"], "descriptive Spearman rho")
            close(corr["observations_used"], len(values), "correlation sample count")
            ordered = sorted(values)
            def quantile(q):
                position = (len(ordered)-1)*q
                low = math.floor(position); high = math.ceil(position)
                return ordered[low] + (ordered[high]-ordered[low])*(position-low)
            edges = sorted(set(quantile(i/5) for i in range(6)))
            groups = defaultdict(list)
            for row, value in zip(detail, values):
                groups[max(0, bisect_left(edges, value)-1)+1].append(row)
            saved_bins = [row for row in tables["n2_evidence_loss_quantile_summary.csv"]
                          if row["history_condition"] == label and row["evidence_variable"] == variable]
            require(len(saved_bins) == len(groups), "Quantile group count disagrees (ties must remain together)")
            bins = []
            for saved in sorted(saved_bins, key=lambda row: int(row["bin_index"])):
                group = groups[int(saved["bin_index"])]
                observed = [number(row[variable]) for row in group]
                expected = {"n_snapshots": len(group), "n_machines": len({row["machineID"] for row in group}),
                            "evidence_mean": fmean(observed), "evidence_min": min(observed), "evidence_max": max(observed),
                            "mean_squared_prediction_error_brier": fmean([number(row["sq_error"]) for row in group]),
                            "observed_failure_prevalence": fmean([number(row["label_failure_7d"]) for row in group])}
                for field, value in expected.items():
                    close(value, saved[field], f"quantile {symbol}/{field}")
                bins.append({"index": int(saved["bin_index"]), **expected})
            evidence[symbol] = {"rho": rho, "observations": len(values), "groups": bins}
        c_corr = one("n2_descriptive_spearman_correlations.csv", history_condition=label, evidence_variable="published_field_completeness_pct")
        require(c_corr["spearman_rho_descriptive"] == "" and int(c_corr["unique_evidence_values"]) == 1,
                "Constant C cannot have an estimated correlation")
        cv = one("n2_grouped_cv_relationship_metrics.csv", history_condition=label)
        cv_rows = [row for k, row in continuous_oof.items() if k[0] == label]
        close(cv["mean_observed_squared_error"], brier, "CV mean observed loss")
        for name, column in [("evidence_ridge", "crossfit_estimated_conditional_mean_loss"),
                             ("constant_mean_baseline", "crossfit_constant_baseline_mean_loss")]:
            actual = metrics(cv_rows, "observed_individual_sq_error", column)
            for metric in ["mae", "rmse"]:
                close(actual[metric], cv[f"{metric}_{name}"], "evidence CV " + metric)
            if name == "evidence_ridge":
                close(actual["r2"], cv["r2_evidence_ridge"], "evidence CV R2")
        alternatives = []
        inc_rows = [row for k, row in incremental_oof.items() if k[0] == label]
        for model in MODELS:
            score = one("n2_risk_vs_evidence_cv_metrics.csv", history_condition=label, model=model)
            actual = metrics(inc_rows, "sq_error", "estimated_loss__"+model)
            for metric, value in actual.items():
                close(value, score[metric], "incremental " + metric)
            for field, value in [("test_machine_snapshots", len(rows)), ("unique_machines", len(machines)), ("machine_group_folds", 5)]:
                close(score[field], value, "incremental sample/fold count")
            alternatives.append({"model": model, **actual})
        delta = one("n2_risk_vs_evidence_incremental_mae.csv", history_condition=label)
        delta_mae = alternatives[1]["mae"]-alternatives[-1]["mae"]
        close(delta_mae, delta["delta_mae_risk_minus_full_evidence"], "incremental MAE delta")
        close(alternatives[1]["mae"], delta["mae_risk_only"], "risk-only MAE")
        close(alternatives[-1]["mae"], delta["mae_risk_plus_HMFR"], "risk-plus-evidence MAE")
        conditions.append({"id": identifier, "label": label, "short_label": short, "lookback_days": lookback,
                           "snapshots": len(rows), "machines": len(machines), "dates": len(dates),
                           "positive_cases": positives, "brier": brier, "evidence": evidence,
                           "evidence_cv": {k: number(cv[k]) for k in cv if k.startswith(("mae_", "rmse_", "r2_"))},
                           "incremental": alternatives, "delta_mae_risk_minus_evidence": delta_mae})
    require(len(source) == method["n_observation_rows_total"], "Metadata total rows disagree")
    return {"schema_version": "n2-study-v1", "model_family": "original_uncalibrated_n2_pilot",
            "horizon_days": 7, "C_constant_pct": 100,
            "cohort": {"first_snapshot": min(k[1] for k in reference), "last_snapshot": max(k[1] for k in reference)},
            "conditions": conditions, "methodology": method, "incremental_methodology": incremental_method,
            "provenance": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Validate sources and require identical snapshot; write nothing")
    args = parser.parse_args()
    try:
        snapshot = build_snapshot(args.source_dir)
        encoded = json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.check:
            require(args.output.read_text(encoding="utf-8") == encoded, "Snapshot differs; regenerate and review changes")
            print("PASS: aggregate snapshot matches all saved experimental sources; no fitting or writes")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
            print(f"Exported validated aggregate snapshot: {args.output} ({len(encoded.encode('utf-8'))} bytes); no fitting")
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"N2 export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
