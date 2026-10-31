from dataclasses import replace
from hashlib import sha256
import json
import sys

from fastapi.testclient import TestClient
import joblib
import pandas as pd
import pytest

from app.core.config import REPOSITORY_ROOT, Settings
from app.main import create_app
from app.components.component3.ml import failure_inference
from app.components.component3.ml.failure_inference import get_research_inference


URL = "/api/v1/component3/predict-risk"
BODY = {"machine_id": 1, "prediction_time": "2015-09-02T06:00:00"}


def test_prediction_contract_and_evidence(client):
    response = client.post(URL, json=BODY)
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_failure_probability"] == 0.25  # Explicit test double.
    assert result["horizon_days"] == 7
    assert result["prediction_time"] == BODY["prediction_time"]
    assert result["evidence_profile"] == {
        "H_observed_maintenance_span_days": 10.0,
        "M_prior_maintenance_events": 2,
        "F_prior_failure_events": 1,
        "R_days_since_latest_maintenance": 2.0,
        "C_published_field_completeness_pct": 100.0,
    }
    assert result["research_only"] is True
    assert result["data_status"] == "synthetic_azure_historical_demo"
    assert "Synthetic Azure benchmark" in result["note"]
    assert result["model_schema_version"] == "c3-pdm-7day-v1"
    assert result["model_version"].startswith("sha256:")
    assert not any("reliability" in key or "recommendation" in key for key in result)
    assert result["request_id"]


def test_repeatability(client):
    first = client.post(URL, json=BODY).json()
    second = client.post(URL, json=BODY).json()
    assert first.pop("request_id") != second.pop("request_id")
    assert first == second


def test_machine_without_history_returns_json_nulls(client):
    response = client.post(URL, json={**BODY, "machine_id": 2})
    assert response.status_code == 200
    profile = response.json()["evidence_profile"]
    assert profile["H_observed_maintenance_span_days"] is None
    assert profile["R_days_since_latest_maintenance"] is None
    assert profile["M_prior_maintenance_events"] == 0
    assert profile["F_prior_failure_events"] == 0
    assert "NaN" not in response.text


@pytest.mark.parametrize("body", [
    {}, {"machine_id": 1}, {"prediction_time": BODY["prediction_time"]},
    {**BODY, "machine_id": "1"}, {**BODY, "machine_id": True},
    {**BODY, "machine_id": 1.5}, {**BODY, "machine_id": 0},
    {**BODY, "machine_id": -1}, {**BODY, "machine_id": None},
    {**BODY, "prediction_time": "invalid"}, {**BODY, "prediction_time": "NaT"},
    {**BODY, "prediction_time": "2026-10-18T06:00:00"},
    {**BODY, "prediction_time": "2015-09-03T06:00:00"},
    {**BODY, "prediction_time": "2015-09-02T06:00:01"},
    {**BODY, "prediction_time": "2015-09-02T06:00:00.000001"},
    {**BODY, "prediction_time": "2015-09-02T06:00:00Z"},
    {**BODY, "prediction_time": "2015-09-02T06:00:00+05:30"},
    {**BODY, "prediction_time": 1441173600}, {**BODY, "prediction_time": None},
    {**BODY, "horizon_days": 14}, {**BODY, "model_path": "untrusted.joblib"},
])
def test_invalid_inputs(client, body):
    assert client.post(URL, json=body).status_code == 422


def test_malformed_json(client):
    assert client.post(URL, content="{", headers={"Content-Type": "application/json"}).status_code == 422


def test_unknown_machine(client):
    response = client.post(URL, json={**BODY, "machine_id": 999999})
    assert response.status_code == 404
    assert "synthetic Azure benchmark" in response.json()["detail"]


@pytest.mark.parametrize("time", get_research_inference().valid_demo_timestamps())
def test_all_thirteen_historical_times(client, time):
    assert client.post(URL, json={**BODY, "prediction_time": time.isoformat()}).status_code == 200


def test_health_and_openapi(client):
    assert client.get("/api/v1/health").json()["component3_ready"] is True
    schema = client.get("/openapi.json").json()
    assert URL in schema["paths"]
    assert "post" in schema["paths"][URL]
    assert client.get("/docs").status_code == 200


@pytest.mark.parametrize("missing", ["model", "data", "both_unconfigured"])
def test_missing_resources(fixture_settings, tmp_path, missing):
    settings = fixture_settings
    if missing == "model":
        settings = replace(settings, c3_model_path=tmp_path / "missing.joblib")
    elif missing == "data":
        settings = replace(settings, c3_data_dir=tmp_path / "missing_raw")
    else:
        settings = Settings()
    with TestClient(create_app(settings)) as client:
        response = client.post(URL, json=BODY)
        assert response.status_code == 503
        assert str(tmp_path) not in response.text
        assert client.get("/api/v1/health").status_code == 503
        assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("source", ["machines", "maint", "failures", "errors"])
def test_each_missing_source(fixture_settings, source):
    (fixture_settings.c3_data_dir / f"PdM_{source}.csv").unlink()
    with TestClient(create_app(fixture_settings)) as client:
        assert client.post(URL, json=BODY).status_code == 503


@pytest.mark.parametrize("change", [
    {"schema_version": "wrong"}, {"feature_columns": []}, {"horizon_days": 14}, {"model": None},
])
def test_incompatible_artifact(fixture_settings, change):
    saved = joblib.load(fixture_settings.c3_model_path)
    joblib.dump({**saved, **change}, fixture_settings.c3_model_path)
    with TestClient(create_app(fixture_settings)) as client:
        assert client.post(URL, json=BODY).status_code == 503


def test_corrupt_artifact(fixture_settings):
    fixture_settings.c3_model_path.write_bytes(b"invalid test artifact")
    with TestClient(create_app(fixture_settings)) as client:
        assert client.post(URL, json=BODY).status_code == 503


@pytest.mark.parametrize("problem", ["missing_column", "missing_value", "unknown_machine", "bad_date"])
def test_malformed_data(fixture_settings, problem):
    path = fixture_settings.c3_data_dir / "PdM_maint.csv"
    frame = pd.read_csv(path)
    if problem == "missing_column":
        frame = frame.drop(columns="comp")
    elif problem == "missing_value":
        frame.loc[0, "comp"] = None
    elif problem == "unknown_machine":
        frame.loc[0, "machineID"] = 99999
    else:
        frame.loc[0, "datetime"] = "not-a-date"
    frame.to_csv(path, index=False)
    with TestClient(create_app(fixture_settings)) as client:
        assert client.post(URL, json=BODY).status_code == 503


def test_resources_loaded_once(fixture_settings, monkeypatch):
    inference = get_research_inference()
    calls = {"model": 0, "data": 0}
    original_load = joblib.load
    original_read = inference.read_demo_sources

    def counted_load(*args, **kwargs):
        calls["model"] += 1
        return original_load(*args, **kwargs)

    def counted_read(*args, **kwargs):
        calls["data"] += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counted_load)
    monkeypatch.setattr(inference, "read_demo_sources", counted_read)
    with TestClient(create_app(fixture_settings)) as client:
        assert client.post(URL, json=BODY).status_code == 200
        assert client.post(URL, json=BODY).status_code == 200
    assert calls == {"model": 1, "data": 1}


def test_unexpected_inference_failure_is_sanitized(client, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("private internal path and traceback")

    monkeypatch.setattr(get_research_inference(), "predict_from_loaded", broken)
    response = client.post(URL, json=BODY)
    assert response.status_code == 500
    assert response.json() == {"detail": "Component 3 prediction failed"}


def test_environment_paths_resolve_from_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("C3_MODEL_PATH", "research/component3/models/c3_7day_sigmoid.joblib")
    monkeypatch.setenv("C3_DATA_DIR", "local/raw")
    settings = Settings.from_env()
    assert settings.c3_model_path == REPOSITORY_ROOT / "research/component3/models/c3_7day_sigmoid.joblib"
    assert settings.c3_data_dir == REPOSITORY_ROOT / "local/raw"


def test_import_bridge_restores_legacy_name_and_search_path(monkeypatch):
    sentinel = object()
    original_path = sys.path.copy()
    monkeypatch.setitem(sys.modules, "build_and_evaluate", sentinel)
    monkeypatch.setattr(failure_inference, "_inference", None)
    assert get_research_inference().SCHEMA_VERSION == "c3-pdm-7day-v1"
    assert sys.modules["build_and_evaluate"] is sentinel
    assert sys.path == original_path


@pytest.mark.integration
def test_actual_artifact_parity_repeatability_and_no_retraining(local_settings):
    # The autouse fixture blocks fitting throughout startup and all requests.
    before = sha256(local_settings.c3_model_path.read_bytes()).hexdigest()
    inference = get_research_inference()
    expected = inference.predict_saved(
        local_settings.c3_model_path, local_settings.c3_data_dir, 1, BODY["prediction_time"],
    )
    with TestClient(create_app(local_settings)) as client:
        first_response = client.post(URL, json=BODY)
        assert first_response.status_code == 200
        first = first_response.json()
        second = client.post(URL, json=BODY).json()
        for key in expected:
            if key != "prediction_time":
                assert first[key] == expected[key]
        assert pd.Timestamp(first["prediction_time"]) == pd.Timestamp(expected["prediction_time"])
        assert 0 <= first["predicted_failure_probability"] <= 1
        assert first["model_version"] == "sha256:" + before
        assert first.pop("request_id") != second.pop("request_id")
        assert first == second
    assert sha256(local_settings.c3_model_path.read_bytes()).hexdigest() == before


@pytest.mark.integration
def test_actual_api_strict_past_integrity(local_settings):
    with TestClient(create_app(local_settings)) as client:
        original = client.post(URL, json=BODY).json()
        service = client.app.state.component3_service
        # Change cached frames in this test process only; never write source CSVs.
        sources = service.resources.sources
        for frame in sources[1:]:
            for time in [BODY["prediction_time"], "2015-09-05T06:00:00"]:
                added = frame.iloc[0].copy()
                added["machineID"] = 1
                added["datetime"] = pd.Timestamp(time)
                frame.loc[len(frame)] = added
        altered = client.post(URL, json=BODY).json()
        original.pop("request_id")
        altered.pop("request_id")
        assert original == altered
        json.dumps(altered, allow_nan=False)
