#!/usr/bin/env python3
"""N2 exploratory incremental-value study: can historical evidence estimate loss
beyond the original model's predicted failure probability?

Run with --input outputs_n2_continuous/n2_prediction_evidence_and_losses.csv
The input must be the ORIGINAL uncalibrated, held-out 7-day N2 pilot predictions.
This is secondary, exploratory model development on an already inspected test period.
It is not an independent evaluation of reliability on new apparel machines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED = [
    "machineID", "prediction_time", "history_condition", "machine_model",
    "predicted_probability", "label_failure_7d", "sq_error",
    "H_observed_span_days", "M_maintenance_count", "F_prior_failure_count",
    "C_field_completeness_pct", "R_days_since_maintenance",
]
NUMERIC_EVIDENCE = [
    "H_observed_span_days", "M_maintenance_count", "F_prior_failure_count"
]
RECENCY = ["R_days_since_maintenance"]
RISK = ["risk_p", "risk_p_squared"]
# Comparisons are defined in advance, not selected based on held-out results.
SPECS = {
    "constant_mean_loss": ([], []),
    "risk_only": (RISK, []),
    "risk_plus_recency": (RISK + RECENCY, []),
    "risk_plus_HMF_machine": (RISK + NUMERIC_EVIDENCE, ["machine_model"]),
    "risk_plus_HMFR_machine": (RISK + NUMERIC_EVIDENCE + RECENCY, ["machine_model"]),
}


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    absent = [c for c in REQUIRED if c not in frame.columns]
    if absent:
        raise ValueError(f"Input lacks required columns: {absent}")
    if frame.empty:
        raise ValueError("No data rows found")
    f = frame.copy()
    f["machineID"] = pd.to_numeric(f["machineID"], errors="raise").astype(int)
    f["prediction_time"] = pd.to_datetime(f["prediction_time"], errors="raise")
    numeric = ["predicted_probability", "label_failure_7d", "sq_error",
               "H_observed_span_days", "M_maintenance_count", "F_prior_failure_count",
               "C_field_completeness_pct", "R_days_since_maintenance"]
    for col in numeric:
        f[col] = pd.to_numeric(f[col], errors="raise")
    if f[["predicted_probability", "label_failure_7d", "sq_error"]].isna().any().any():
        raise ValueError("Missing probabilities, labels or observed losses")
    if not f["predicted_probability"].between(0, 1).all():
        raise ValueError("Predicted probabilities must be between 0 and 1")
    if not f["label_failure_7d"].isin([0, 1]).all():
        raise ValueError("Future failure labels must be binary")
    expected = (f["label_failure_7d"] - f["predicted_probability"]) ** 2
    if not np.allclose(expected, f["sq_error"], atol=1e-9, rtol=1e-7):
        raise ValueError("Loss values do not match (label - probability)^2")
    if f.duplicated(["history_condition", "machineID", "prediction_time"]).any():
        raise ValueError("Repeated machine/timestamp within a history condition")
    if f["machine_model"].isna().any():
        raise ValueError("Missing machine model")
    if f["history_condition"].isna().any():
        raise ValueError("Missing history-condition identifier")
    f["risk_p"] = f["predicted_probability"]
    f["risk_p_squared"] = f["predicted_probability"] ** 2
    return f


def make_estimator(numerics: list[str], cats: list[str]):
    if not numerics and not cats:
        return DummyRegressor(strategy="mean")
    parts = []
    if numerics:
        steps = [("impute", SimpleImputer(strategy="median")),
                 ("standardize", StandardScaler())]
        parts.append(("num", Pipeline(steps), numerics))
    if cats:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats))
    prep = ColumnTransformer(parts, remainder="drop")
    return Pipeline([("preprocess", prep), ("ridge", Ridge(alpha=5.0))])


def run_experiment(frame: pd.DataFrame, folds: int = 5):
    f = validate_input(frame)
    summary = []
    joined = []
    for condition, data in f.groupby("history_condition", sort=True):
        data = data.sort_values(["machineID", "prediction_time"]).reset_index(drop=True)
        n_machines = data["machineID"].nunique()
        if n_machines < folds:
            raise ValueError(f"{condition}: need at least {folds} distinct machines")
        y = data["sq_error"].to_numpy(dtype=float)
        groups = data["machineID"].to_numpy()
        out = data[["history_condition", "machineID", "prediction_time",
                    "label_failure_7d", "predicted_probability", "sq_error",
                    "H_observed_span_days", "M_maintenance_count", "F_prior_failure_count",
                    "C_field_completeness_pct", "R_days_since_maintenance", "machine_model"]].copy()
        for spec in SPECS:
            out[f"estimated_loss__{spec}"] = np.nan
        out["machine_group_fold"] = -1
        splitter = GroupKFold(n_splits=folds)
        for fold, (train_idx, eval_idx) in enumerate(splitter.split(data, y, groups), start=1):
            if not set(groups[train_idx]).isdisjoint(set(groups[eval_idx])):
                raise AssertionError("Machine group leakage in cross-validation")
            out.loc[eval_idx, "machine_group_fold"] = fold
            for spec, (nums, cats) in SPECS.items():
                reg = make_estimator(nums, cats)
                # Loss is a target ONLY. Neither label nor observed loss enters X.
                if nums or cats:
                    cols = nums + cats
                    reg.fit(data.loc[train_idx, cols], y[train_idx])
                    pred = reg.predict(data.loc[eval_idx, cols])
                else:
                    reg.fit(np.zeros((len(train_idx), 1)), y[train_idx])
                    pred = reg.predict(np.zeros((len(eval_idx), 1)))
                # Bound loss expectations in their mathematically valid [0, 1] range.
                out.loc[eval_idx, f"estimated_loss__{spec}"] = np.clip(pred, 0.0, 1.0)
        if out["machine_group_fold"].lt(1).any():
            raise AssertionError("Some observations received no out-of-fold predictions")
        for spec in SPECS:
            estimate = out[f"estimated_loss__{spec}"].to_numpy()
            if np.isnan(estimate).any():
                raise AssertionError(f"Missing out-of-fold estimate for {spec}")
            summary.append({
                "history_condition": condition,
                "model": spec,
                "test_machine_snapshots": len(data),
                "unique_machines": n_machines,
                "machine_group_folds": folds,
                "mae": float(mean_absolute_error(y, estimate)),
                "rmse": float(np.sqrt(mean_squared_error(y, estimate))),
                "r2": float(r2_score(y, estimate)),
            })
        joined.append(out)
    scores = pd.DataFrame(summary)
    predictions = pd.concat(joined, ignore_index=True)
    # The principal incremental comparison is defined before inspecting its result.
    pivot = scores.pivot(index="history_condition", columns="model", values="mae")
    deltas = pd.DataFrame({
        "history_condition": pivot.index,
        "mae_risk_only": pivot["risk_only"].to_numpy(),
        "mae_risk_plus_HMFR": pivot["risk_plus_HMFR_machine"].to_numpy(),
        "delta_mae_risk_minus_full_evidence": (pivot["risk_only"]-pivot["risk_plus_HMFR_machine"]).to_numpy(),
        "delta_mae_recency_minus_full_evidence": (pivot["risk_plus_recency"]-pivot["risk_plus_HMFR_machine"]).to_numpy(),
    })
    return scores, predictions, deltas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="N2 continuous prediction/evidence CSV")
    p.add_argument("--out-dir", type=Path, default=Path("outputs_n2_incremental"))
    args = p.parse_args()
    scores, predictions, deltas = run_experiment(pd.read_csv(args.input))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out_dir / "n2_risk_vs_evidence_cv_metrics.csv", index=False)
    predictions.to_csv(args.out_dir / "n2_risk_vs_evidence_cv_predictions.csv", index=False)
    deltas.to_csv(args.out_dir / "n2_risk_vs_evidence_incremental_mae.csv", index=False)
    metadata = {
        "research_question": "Does H/M/F/R (+ machine model) add out-of-fold loss-prediction information beyond original risk p?",
        "dataset": "Synthetic Microsoft Azure PdM; not real apparel-machine data",
        "loss": "Observed squared error from the ORIGINAL, uncalibrated 7-day N2 model",
        "comparison": list(SPECS),
        "risk_only_features": RISK,
        "risk_plus_evidence_features": RISK+NUMERIC_EVIDENCE+RECENCY+["machine_model"],
        "C_field_completeness": "Excluded: fixed at 100% in this published dataset",
        "recency_warning": "R measures days since a maintenance event; this is operational machine state, not verified log-data freshness",
        "validity_warning": "Original risk model previously trained on earlier records for these machines. This is second-stage grouped CV on an already examined original test period; not independent validation of new machines or apparel factories.",
        "dependence_warning": "Weekly observations and failure outcomes for a machine are correlated. GroupKFold separates machines only in the second-stage relationship analysis.",
        "interpretation": "A positive delta_mae_risk_minus_full_evidence means lower OOF MAE for the risk+evidence model, not a demonstrated causal sufficiency effect or a validated machine reliability percentage.",
        "model_selection_warning": "Do not select one method solely from these reused test-period results and report its score as fresh external validation.",
    }
    (args.out_dir / "n2_risk_vs_evidence_methodology.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nOUT-OF-FOLD MEAN ABSOLUTE ERROR (lower is better)")
    print(scores.pivot(index="history_condition", columns="model", values="mae").round(5).to_string())
    print("\nPREDECLARED INCREMENTAL COMPARISONS (positive delta favors added evidence)")
    print(deltas.round(5).to_string(index=False))
    print(f"\nSaved results to: {args.out_dir}")
    print("NOTE: Secondary exploratory validation on a previously inspected synthetic test period.")


if __name__ == "__main__":
    main()
