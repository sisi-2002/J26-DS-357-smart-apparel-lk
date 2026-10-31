"""Experimental landmark next-recorded-failure model; never used by the API.

Intervals preserve the existing open-upper-bound cumulative targets:
(0, 7), [7, 14), [14, 30). One feature vector is frozen at the landmark.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.dont_write_bytecode = True  # Reusing research imports must not rewrite their caches.

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_and_evaluate import FEATURE_NUMERIC, FEATURE_CAT, make_baseline  # noqa: E402

FEATURES = FEATURE_NUMERIC + FEATURE_CAT
HORIZONS = (7, 14, 30)
INTERVALS = ((0, 7), (7, 14), (14, 30))
DAY = pd.Timedelta(days=1).value


def require(ok, message):
    if not ok:
        raise ValueError(message)


def landmark_outcomes(snapshots, failures, observation_end):
    """Find first strictly future event before explicit exclusive follow-up end.

    observation_end is a shared administrative timestamp or an aligned Series of
    per-landmark end timestamps. Never infer follow-up from the last failure.
    Unknown negatives remain NaN; an observed failure determines positive labels.
    """
    require(not snapshots.duplicated(["machineID", "prediction_time"]).any(), "Duplicate machine-time landmarks")
    require(snapshots.index.is_unique, "Landmark index must be unique")
    times = pd.to_datetime(snapshots.prediction_time, errors="raise")
    require(times.notna().all() and times.dt.tz is None, "Landmark times must be nonmissing and timezone-naive")
    if isinstance(observation_end, pd.Series):
        require(observation_end.index.equals(snapshots.index), "Follow-up Series must align exactly with landmarks")
        ends = pd.to_datetime(observation_end, errors="raise")
    else:
        ends = pd.Series(pd.Timestamp(observation_end), index=snapshots.index)
    require(ends.notna().all() and ends.dt.tz is None and (ends > times).all(), "Follow-up must end strictly after each landmark")
    event_times = pd.to_datetime(failures.datetime, errors="raise")
    require(event_times.notna().all() and event_times.dt.tz is None, "Invalid failure event times")
    events = failures.assign(datetime=event_times)
    grouped = {int(mid): np.sort(g.datetime.astype("int64").unique()) for mid, g in events.groupby("machineID")}
    records = []
    for position, (mid, timestamp, end) in enumerate(zip(snapshots.machineID, times, ends)):
        values = grouped.get(int(mid), np.empty(0, dtype="int64"))
        offset = np.searchsorted(values, timestamp.value, side="right")
        first = values[offset] if offset < len(values) and values[offset] < end.value else None
        delay = np.nan if first is None else (first - timestamp.value) / DAY
        followup = (end - timestamp) / pd.Timedelta(days=1)
        record = {"landmark_position": position, "machineID": int(mid), "prediction_time": timestamp,
                  "followup_end": end, "followup_days": float(followup), "first_event_days": delay,
                  "complete_30d": followup >= 30}
        for horizon in HORIZONS:
            record[f"label_failure_{horizon}d"] = (1.0 if pd.notna(delay) and delay < horizon
                                                  else 0.0 if followup >= horizon else np.nan)
        records.append(record)
    return pd.DataFrame(records)


def person_period(outcomes):
    """One row per eligible interval, ending at first event or censoring.

    A partially observed, event-free interval is omitted, never labelled zero.
    An observed event contributes its interval even if follow-up ends inside it.
    """
    require(outcomes.landmark_position.is_unique, "Duplicate landmark positions")
    require((outcomes.followup_days > 0).all(), "Follow-up must be positive")
    observed = outcomes.first_event_days.notna()
    require((np.isfinite(outcomes.loc[observed, "first_event_days"]) &
             (outcomes.loc[observed, "first_event_days"] > 0) &
             (outcomes.loc[observed, "first_event_days"] < outcomes.loc[observed, "followup_days"])).all(),
            "Observed first events must be strictly after landmark and before follow-up end")
    rows = []
    for row in outcomes.itertuples(index=False):
        for interval, (lower, upper) in enumerate(INTERVALS, 1):
            event = pd.notna(row.first_event_days) and lower <= row.first_event_days < upper
            if pd.notna(row.first_event_days) and row.first_event_days < lower:
                break
            if event or row.followup_days >= upper:
                rows.append({"landmark_position": row.landmark_position, "interval": interval, "event": int(event)})
            else:
                break
            if event:
                break
    return pd.DataFrame(rows, columns=["landmark_position", "interval", "event"])


def cumulative_probabilities(hazards):
    hazards = np.asarray(hazards, dtype=float)
    require(hazards.ndim == 2 and hazards.shape[1] == 3, "Expected three conditional hazards per landmark")
    require(np.isfinite(hazards).all() and ((hazards >= 0) & (hazards <= 1)).all(), "Hazards must be in [0, 1]")
    return 1.0 - np.cumprod(1.0 - hazards, axis=1)


class PooledHazardModel(BaseEstimator):
    """Shared linear slopes + categorical interval intercepts; fixed L2 C=1.

    Preprocessing is fitted on unique TRAIN landmarks, not risk-set-expanded
    records. Optional pooled sigmoid calibration uses validation risk sets only.
    Both versions compose conditional hazards into ordered cumulative outputs.
    """

    @staticmethod
    def _features(frame):
        require(set(FEATURES).issubset(frame), "Missing original temporal feature columns")
        require(not frame[FEATURE_CAT].isna().any().any(), "Missing machine model/type")
        values = frame[FEATURE_NUMERIC].to_numpy(dtype=float)
        require(not np.isinf(values).any(), "Infinite numeric features")
        return frame[FEATURES].reset_index(drop=True)

    @staticmethod
    def _validate_periods(periods, count):
        require(not periods.empty, "No observable person-period rows")
        require(not periods.duplicated(["landmark_position", "interval"]).any(), "Duplicate interval rows")
        require(periods.interval.isin([1, 2, 3]).all() and periods.event.isin([0, 1]).all(), "Invalid interval/event indicator")
        require(periods.landmark_position.between(0, count - 1).all()
                and (periods.landmark_position % 1 == 0).all(), "Invalid landmark positions")
        for _, group in periods.groupby("landmark_position"):
            group = group.sort_values("interval")
            require(group.interval.tolist() == list(range(1, len(group) + 1)), "Risk set must start at interval 1 without gaps")
            require(group.event.sum() <= 1 and not group.event.iloc[:-1].any(), "Rows after first event are ineligible")

    def _design(self, features, periods):
        transformed = self.preprocessor_.transform(features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.column_stack([transformed[periods.landmark_position.to_numpy(dtype=int)],
                                self.interval_encoder_.transform(periods[["interval"]])])

    def fit(self, train_features, train_periods):
        features = self._features(train_features)
        self._validate_periods(train_periods, len(features))
        require(train_periods.event.nunique() == 2, "Training needs event and event-free interval records")
        self.preprocessor_ = clone(make_baseline().named_steps["transform"])
        self.preprocessor_.fit(features)
        self.interval_encoder_ = OneHotEncoder(categories=[[1, 2, 3]], drop="first", sparse_output=False, handle_unknown="error")
        self.interval_encoder_.fit(train_periods[["interval"]])
        self.classifier_ = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        self.classifier_.fit(self._design(features, train_periods), train_periods.event.to_numpy())
        self.training_landmarks_ = len(features)
        self.training_periods_ = len(train_periods)
        if hasattr(self, "calibrator_"):
            del self.calibrator_
        return self

    def calibrate(self, validation_features, validation_periods):
        check_is_fitted(self, "classifier_")
        features = self._features(validation_features)
        self._validate_periods(validation_periods, len(features))
        require(validation_periods.event.nunique() == 2, "Calibration needs both interval outcomes")
        before = joblib.hash((self.preprocessor_, self.interval_encoder_, self.classifier_))
        calibrator = CalibratedClassifierCV(FrozenEstimator(self.classifier_), method="sigmoid")
        calibrator.fit(self._design(features, validation_periods), validation_periods.event.to_numpy())
        require(before == joblib.hash((self.preprocessor_, self.interval_encoder_, self.classifier_)), "Calibration changed fitted training state")
        self.calibrator_ = calibrator
        self.calibration_landmarks_ = len(features)
        self.calibration_periods_ = len(validation_periods)
        return self

    def predict_hazards(self, snapshot_features, calibrated=False):
        check_is_fitted(self, "classifier_")
        features = self._features(snapshot_features)
        if not len(features):
            return np.empty((0, 3))
        periods = pd.DataFrame({"landmark_position": np.repeat(np.arange(len(features)), 3),
                                "interval": np.tile([1, 2, 3], len(features))})
        estimator = self.classifier_
        if calibrated:
            check_is_fitted(self, "calibrator_")
            estimator = self.calibrator_
        return estimator.predict_proba(self._design(features, periods))[:, 1].reshape(-1, 3)

    def predict(self, snapshot_features, calibrated=False):
        return cumulative_probabilities(self.predict_hazards(snapshot_features, calibrated=calibrated))
