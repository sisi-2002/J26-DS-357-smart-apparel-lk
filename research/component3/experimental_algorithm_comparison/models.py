"""Fixed 18-feature algorithms and training-only chronological development folds."""
from __future__ import annotations

from pathlib import Path
import sys
sys.dont_write_bytecode = True
import importlib
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH / "scripts"))
from build_and_evaluate import FEATURE_NUMERIC, FEATURE_CAT, make_baseline  # noqa: E402

FEATURES = FEATURE_NUMERIC + FEATURE_CAT
TARGET = "label_failure_7d"
ALGORITHMS = ("logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost")
GRIDS = {
    "random_forest": [{"max_depth": 4, "min_samples_leaf": 20},
                      {"max_depth": 6, "min_samples_leaf": 20},
                      {"max_depth": 8, "min_samples_leaf": 10}],
    "hist_gradient_boosting": [{"max_leaf_nodes": 7, "min_samples_leaf": 30, "l2_regularization": 1.},
                               {"max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 1.},
                               {"max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 3.}],
    "xgboost": [{"max_depth": 2, "min_child_weight": 10, "reg_lambda": 5.},
                {"max_depth": 3, "min_child_weight": 10, "reg_lambda": 5.},
                {"max_depth": 3, "min_child_weight": 20, "reg_lambda": 10.}],
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def xgboost_availability():
    if importlib.util.find_spec("xgboost") is None:
        return {"available": False, "version": None, "reason": "Not installed; no installation attempted"}
    try:
        module = importlib.import_module("xgboost")
    except (ImportError, OSError) as error:
        return {"available": False, "version": None, "reason": f"Installed package could not import: {error}"}
    return {"available": True, "version": module.__version__, "reason": "Existing local installation; no packages installed"}


def make_model(algorithm, parameters=None):
    parameters = parameters or {}
    if algorithm == "logistic_regression":
        require(not parameters, "Reference Logistic Regression is fixed, not retuned")
        return make_baseline()
    if algorithm == "random_forest":
        classifier = RandomForestClassifier(n_estimators=300, max_features=.7, bootstrap=True,
                                            class_weight=None, random_state=42, n_jobs=1, **parameters)
    elif algorithm == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05,
                                                    early_stopping=False, random_state=42, **parameters)
    elif algorithm == "xgboost":
        require(xgboost_availability()["available"], "XGBoost unavailable; do not install automatically")
        from xgboost import XGBClassifier
        classifier = XGBClassifier(n_estimators=150, learning_rate=.05, subsample=.8, colsample_bytree=.8,
                                   objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                   random_state=42, n_jobs=1, **parameters)
    else:
        raise ValueError("Unknown algorithm")
    processor = clone(make_baseline().named_steps["transform"])
    # Identical transformations/inputs; dense output is required by histogram boosting.
    processor.set_params(sparse_threshold=0.)
    return Pipeline([("transform", processor), ("classifier", classifier)])


def development_folds(train):
    """Three expanding date-block folds, entirely within original training cases.

    Each validation block has three whole dates. A full 30-day purge is more
    conservative than required by the seven-day labels. Same-machine overlap
    is intentional: the task is later predictions for the existing machines.
    """
    require(train.split.eq("train").all(), "Development may use only original training cases")
    require(not train.duplicated(["machineID", "prediction_time"]).any(), "Duplicate development keys")
    times = sorted(train.prediction_time.unique())
    require(len(times) == 18 and len(train) == 1800, "Need original 18 training dates and 1,800 cases")
    for _, group in train.groupby("prediction_time"):
        require(set(group.machineID) == set(range(1, 101)) and len(group) == 100, "Keep all 100 machines together within each date")
    folds = []
    for number, start in enumerate([9, 12, 15], 1):
        score_times = times[start:start + 3]
        first = pd.Timestamp(score_times[0])
        fitting = np.flatnonzero((train.prediction_time + pd.Timedelta(days=30) < first).to_numpy())
        scoring = np.flatnonzero(train.prediction_time.isin(score_times).to_numpy())
        fit_frame, score_frame = train.iloc[fitting], train.iloc[scoring]
        require(fit_frame[TARGET].nunique() == score_frame[TARGET].nunique() == 2, "Development fold lacks both outcomes")
        require(fit_frame.prediction_time.max() + pd.Timedelta(days=30) < score_frame.prediction_time.min(), "Development outcome overlap")
        folds.append((fitting, scoring, {"fold": number, "fitting_cases": len(fitting), "scoring_cases": len(scoring),
                     "fitting_first": str(fit_frame.prediction_time.min()), "fitting_last": str(fit_frame.prediction_time.max()),
                     "scoring_first": str(score_frame.prediction_time.min()), "scoring_last": str(score_frame.prediction_time.max()),
                     "fitting_positives": int(fit_frame[TARGET].sum()), "scoring_positives": int(score_frame[TARGET].sum()),
                     "fitting_machines": int(fit_frame.machineID.nunique()), "scoring_machines": int(score_frame.machineID.nunique()),
                     "purge_days": 30}))
    return folds


def select_parameters(train, availability=None):
    """No calibration/test arguments: selection is by mean RAW development Brier.

    Mean AP breaks exact Brier ties, then candidate index. Preprocessing is fitted
    afresh in each fitting prefix. LR is recorded for context but never tuned.
    """
    availability = availability or xgboost_availability()
    folds = development_folds(train)
    rows, choices = [], {"logistic_regression": {}}
    for algorithm in ALGORITHMS:
        if algorithm == "xgboost" and not availability["available"]:
            continue
        candidates = [{}] if algorithm == "logistic_regression" else GRIDS[algorithm]
        for candidate, parameters in enumerate(candidates):
            for fit_positions, score_positions, info in folds:
                fitting, scoring = train.iloc[fit_positions], train.iloc[score_positions]
                model = make_model(algorithm, parameters)
                model.fit(fitting[FEATURES], fitting[TARGET].to_numpy())
                p = model.predict_proba(scoring[FEATURES])[:, 1]
                rows.append({"algorithm": algorithm, "candidate": candidate, "parameters": json.dumps(parameters, sort_keys=True),
                             **info, "raw_development_brier": float(brier_score_loss(scoring[TARGET], p)),
                             "raw_development_ap": float(average_precision_score(scoring[TARGET], p)),
                             "scoring_prevalence": float(scoring[TARGET].mean()), "mean_probability": float(p.mean())})
        scores = pd.DataFrame(rows)
        means = scores[scores.algorithm.eq(algorithm)].groupby("candidate").agg(
            mean_brier=("raw_development_brier", "mean"), mean_ap=("raw_development_ap", "mean")).reset_index()
        winner = int(means.sort_values(["mean_brier", "mean_ap", "candidate"], ascending=[True, False, True]).iloc[0].candidate)
        choices[algorithm] = dict(candidates[winner])
        print(f"Selected {algorithm} using training-only development: {choices[algorithm]}", flush=True)
    return choices, pd.DataFrame(rows), [info for _, _, info in folds]
