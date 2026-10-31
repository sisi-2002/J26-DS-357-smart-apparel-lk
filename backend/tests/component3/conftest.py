"""Temporary benchmark-shaped fixtures; no model training or repository writes."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.config import Settings
from app.main import create_app
from app.components.component3.ml.failure_inference import get_research_inference


class FixtureProbabilityEstimator:
    """Test double only; real artifact parity is checked in integration tests."""
    def predict_proba(self, features):
        return np.tile([0.75, 0.25], (len(features), 1))


@pytest.fixture(autouse=True)
def prohibit_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Training/calibration fitting is prohibited in API tests")

    for estimator in (LogisticRegression, CalibratedClassifierCV, FrozenEstimator,
                      Pipeline, ColumnTransformer, DummyClassifier):
        monkeypatch.setattr(estimator, "fit", forbidden)
    for estimator in (Pipeline, ColumnTransformer):
        monkeypatch.setattr(estimator, "fit_transform", forbidden)


@pytest.fixture
def fixture_settings(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    pd.DataFrame({"machineID": [1, 2], "model": ["model1", "model2"], "age": [5, 2]}).to_csv(
        data / "PdM_machines.csv", index=False,
    )
    for name, field, times in (
        ("maint", "comp", ["2015-08-23 06:00:00", "2015-08-31 06:00:00"]),
        ("failures", "failure", ["2015-08-20 06:00:00"]),
        ("errors", "errorID", ["2015-08-30 06:00:00"]),
    ):
        pd.DataFrame({"machineID": [1] * len(times), "datetime": times,
                      field: ["comp1" if field != "errorID" else "error1"] * len(times)}).to_csv(
            data / f"PdM_{name}.csv", index=False,
        )
    inference = get_research_inference()
    model = tmp_path / "fixture.joblib"
    joblib.dump({"schema_version": inference.SCHEMA_VERSION,
                 "feature_columns": inference.FEATURE_COLUMNS,
                 "horizon_days": 7, "model": FixtureProbabilityEstimator()}, model)
    return Settings(c3_model_path=model, c3_data_dir=data)


@pytest.fixture
def client(fixture_settings):
    with TestClient(create_app(fixture_settings)) as test_client:
        yield test_client


@pytest.fixture
def local_settings():
    settings = Settings.from_env()
    if settings.c3_model_path is None or settings.c3_data_dir is None:
        pytest.skip("Set C3_MODEL_PATH and C3_DATA_DIR to run actual-artifact integration tests")
    # When configured, a bad path is a test failure, never a silently skipped check.
    assert settings.c3_model_path.is_file(), "Configured integration artifact is missing"
    assert settings.c3_data_dir.is_dir(), "Configured integration dataset directory is missing"
    return settings
