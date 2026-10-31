"""Past-only generic Azure error features; no targets, training or telemetry."""
from __future__ import annotations

import numpy as np
import pandas as pd

GENERIC_ERROR_FEATURES = ["error_events_prior", "error_events_7d", "error_events_30d",
                          "error_events_90d", "days_since_last_error"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_errors(errors, machine_ids):
    require(set(errors.columns) == {"datetime", "machineID", "errorID"}, "Expected datetime, machineID, errorID error schema")
    require(not errors.isna().any().any(), "Missing error timestamps, machine IDs or categories")
    times = pd.to_datetime(errors.datetime, errors="raise")
    require(times.dt.tz is None, "Error timestamps must be timezone-naive")
    ids = pd.to_numeric(errors.machineID, errors="raise")
    require(np.isfinite(ids).all() and ids.mod(1).eq(0).all(), "Machine IDs must be finite integers")
    require(set(ids).issubset(set(machine_ids)), "Unknown error machine IDs")
    require(errors.errorID.map(lambda x: isinstance(x, str) and bool(x.strip()) and x == x.strip()).all(), "Invalid error categories")
    normalized = errors.assign(datetime=times, machineID=ids.astype(int))
    require(not normalized.duplicated().any(), "Duplicate complete error records; resolve explicitly instead of silently deduplicating")
    # Distinct categories at one machine-time are retained as distinct records.
    return normalized


def training_categories(errors, training_end):
    """Vocabulary from records strictly before the final training landmark only."""
    end = pd.Timestamp(training_end)
    require(pd.notna(end) and end.tzinfo is None, "Invalid training vocabulary cutoff")
    return tuple(sorted(errors.loc[errors.datetime < end, "errorID"].unique()))


def category_feature_names(categories):
    """Drop the alphabetically first category: total 30d count supplies its count.

    This avoids exact linear redundancy while retaining all category information.
    Vocabulary must already be frozen from training, never inferred from test.
    """
    require(tuple(categories) == tuple(sorted(set(categories))), "Categories must be sorted and unique")
    return [f"error_category_{category}_30d" for category in categories[1:]]


def build_error_features(landmarks, errors, categories=()):
    """Windows include t-window and exclude t; recency is NaN without prior errors.

    Total prior errors refers to the available historical file, not proven logging
    coverage. Error recency is time since an event, not database freshness.
    """
    require(not landmarks.duplicated(["machineID", "prediction_time"]).any() and landmarks.index.is_unique,
            "Duplicate landmark keys/index")
    times = pd.to_datetime(landmarks.prediction_time, errors="raise")
    require(times.notna().all() and times.dt.tz is None, "Invalid landmark timestamps")
    category_feature_names(categories)
    grouped = {int(mid): group.sort_values("datetime") for mid, group in errors.groupby("machineID")}
    result = []
    for mid, timestamp in zip(landmarks.machineID, times):
        group = grouped.get(int(mid))
        values = np.empty(0, dtype="int64") if group is None else group.datetime.astype("int64").to_numpy()
        stop = int(np.searchsorted(values, timestamp.value, side="left"))
        if categories and group is not None:
            require(set(group.errorID.to_numpy()[:stop]).issubset(set(categories)),
                    "Unseen prior error category: frozen training vocabulary cannot represent its detail")
        row = {"error_events_prior": stop,
               "days_since_last_error": (timestamp.value - values[stop - 1]) / pd.Timedelta(days=1).value if stop else np.nan}
        starts = {}
        for days in [7, 30, 90]:
            start = int(np.searchsorted(values, (timestamp - pd.Timedelta(days=days)).value, side="left"))
            starts[days] = start
            row[f"error_events_{days}d"] = stop - start
        labels = np.empty(0, dtype=object) if group is None else group.errorID.to_numpy()[starts[30]:stop]
        for category, name in zip(categories[1:], category_feature_names(categories)):
            row[name] = int(np.sum(labels == category))
        result.append(row)
    return pd.DataFrame(result, index=landmarks.index, columns=GENERIC_ERROR_FEATURES + category_feature_names(categories))


def feature_distribution(panel, feature_names):
    rows = []
    for split, frame in panel.groupby("split", sort=False):
        for name in feature_names:
            values = frame[name]
            rows.append({"split": split, "feature": name, "cases": len(frame), "available": int(values.notna().sum()),
                         "missing": int(values.isna().sum()), "zeros": int(values.eq(0).sum()),
                         "zero_pct": float(100 * values.eq(0).mean()), "mean": float(values.mean()),
                         "std": float(values.std()), "min": float(values.min()), "p25": float(values.quantile(.25)),
                         "median": float(values.median()), "p75": float(values.quantile(.75)), "max": float(values.max())})
    return pd.DataFrame(rows)
