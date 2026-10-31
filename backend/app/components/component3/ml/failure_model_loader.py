"""Load configured, locally trusted artifacts once; never fit an estimator."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import joblib
import pandas as pd

from app.core.config import Settings
from .failure_inference import get_research_inference


@dataclass(frozen=True)
class InferenceResources:
    saved: dict
    sources: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    machine_ids: frozenset[int]
    model_version: str


def _validate_sources(sources: tuple) -> None:
    machines, *events = sources
    if not {"machineID", "model", "age"}.issubset(machines.columns) or machines.empty:
        raise ValueError("Benchmark machine metadata is missing or empty")
    if machines[["machineID", "model", "age"]].isna().any().any():
        raise ValueError("Benchmark machine metadata contains missing fields")
    if machines.machineID.duplicated().any():
        raise ValueError("Benchmark machine IDs must be unique")
    for values in (machines.machineID, machines.age):
        numeric = pd.to_numeric(values, errors="raise")
        if not ((numeric % 1 == 0) & (numeric >= 0)).all():
            raise ValueError("Benchmark IDs and ages must be nonnegative integers")
    if not (machines.machineID > 0).all():
        raise ValueError("Benchmark machine IDs must be positive")
    for frame, event_field in zip(events, ("comp", "failure", "errorID")):
        required = ["machineID", "datetime", event_field]
        if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
            raise ValueError("Benchmark event source contains missing fields")
        times = pd.to_datetime(frame.datetime, errors="raise")
        if times.dt.tz is not None:
            raise ValueError("Benchmark source timestamps must be timezone-naive")
        if not frame.machineID.isin(machines.machineID).all():
            raise ValueError("Benchmark event source contains unknown machine IDs")


def load_resources(settings: Settings) -> InferenceResources:
    if settings.c3_model_path is None or settings.c3_data_dir is None:
        raise ValueError("Set C3_MODEL_PATH and C3_DATA_DIR before starting the API")
    inference = get_research_inference()
    # Joblib can execute code: this path must identify the user's trusted model.
    payload = settings.c3_model_path.read_bytes()
    saved = joblib.load(BytesIO(payload))
    if not isinstance(saved, dict):
        raise ValueError("Expected a saved Component 3 artifact dictionary")
    if saved.get("schema_version") != inference.SCHEMA_VERSION:
        raise ValueError("Saved model schema version does not match research inference")
    if saved.get("feature_columns") != inference.FEATURE_COLUMNS:
        raise ValueError("Saved model feature specification does not match research inference")
    if saved.get("horizon_days") != 7:
        raise ValueError("Only the saved 7-day model is supported")
    if not callable(getattr(saved.get("model"), "predict_proba", None)):
        raise ValueError("Saved artifact has no probability estimator")
    sources = inference.read_demo_sources(settings.c3_data_dir)
    _validate_sources(sources)
    return InferenceResources(
        saved=saved,
        sources=sources,
        machine_ids=frozenset(int(mid) for mid in sources[0].machineID),
        model_version="sha256:" + sha256(payload).hexdigest(),
    )
