"""Batch ranking checks; inherited autouse fixture prohibits all fitting."""
from dataclasses import replace
from hashlib import sha256

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from app.main import create_app
from app.components.component3.ml.failure_inference import get_research_inference
from app.components.component3.ml.failure_model_loader import load_resources
from app.components.component3.ml.failure_ranking import score_machines


URL = "/api/v1/component3/top-risk-machines"
DETAIL_URL = "/api/v1/component3/predict-risk"
BODY = {"prediction_time": "2015-09-02T06:00:00"}


@pytest.fixture
def ranking_settings(fixture_settings):
    # Deliberately reversed source order: row position must not become identity.
    ids = list(range(100, 0, -1))
    pd.DataFrame({"machineID": ids, "model": ["model1"] * 100, "age": ids}).to_csv(
        fixture_settings.c3_data_dir / "PdM_machines.csv", index=False,
    )
    return fixture_settings


@pytest.fixture
def ranking_client(ranking_settings):
    with TestClient(create_app(ranking_settings)) as client:
        yield client


def test_one_batch_call_all_machines_order_and_detail_parity(ranking_client, monkeypatch):
    calls = []

    def probabilities(features):
        calls.append(features.copy())
        p = features.machine_age_years.to_numpy() / 101
        return np.column_stack((1 - p, p))

    model = ranking_client.app.state.component3_service.resources.saved["model"]
    monkeypatch.setattr(model, "predict_proba", probabilities)
    response = ranking_client.post(URL, json=BODY)
    assert response.status_code == 200
    result = response.json()
    assert len(calls) == 1
    assert len(calls[0]) == 100
    assert list(calls[0].columns) == get_research_inference().FEATURE_COLUMNS
    assert calls[0].machine_age_years.tolist() == list(range(1, 101))
    assert result["scored_machine_count"] == 100
    assert result["returned_machine_count"] == 10
    assert result["horizon_days"] == 7
    assert result["research_only"] is True
    assert result["data_status"] == "synthetic_azure_historical_demo"
    assert "Relative ranking of 100 synthetic Azure" in result["note"]
    assert [row["rank"] for row in result["machines"]] == list(range(1, 11))
    assert [row["machine_id"] for row in result["machines"]] == list(range(100, 90, -1))
    for row in result["machines"]:
        detail = ranking_client.post(DETAIL_URL, json={**BODY, "machine_id": row["machine_id"]})
        assert detail.status_code == 200
        assert detail.json()["predicted_failure_probability"] == row["predicted_failure_probability"]
        assert detail.json()["model_version"] == result["model_version"]
        assert "evidence_profile" in detail.json()


def test_exact_ties_are_resolved_by_machine_id(ranking_client):
    result = ranking_client.post(URL, json=BODY).json()
    assert [row["machine_id"] for row in result["machines"]] == list(range(1, 11))
    assert all(row["predicted_failure_probability"] == 0.25 for row in result["machines"])


def test_ranking_uses_unrounded_probabilities(ranking_client, monkeypatch):
    def probabilities(features):
        p = np.full(len(features), 0.1)
        p[0], p[1] = 0.9000001, 0.9000002
        return np.column_stack((1 - p, p))

    monkeypatch.setattr(ranking_client.app.state.component3_service.resources.saved["model"], "predict_proba", probabilities)
    rows = ranking_client.post(URL, json=BODY).json()["machines"]
    assert [row["machine_id"] for row in rows[:2]] == [2, 1]
    assert [row["predicted_failure_probability"] for row in rows[:2]] == [0.9, 0.9]


def test_repeatability_and_cached_resources(ranking_client, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Ranking must not reload model or CSV sources")

    monkeypatch.setattr(get_research_inference(), "read_demo_sources", forbidden)
    import joblib
    monkeypatch.setattr(joblib, "load", forbidden)
    first = ranking_client.post(URL, json=BODY).json()
    second = ranking_client.post(URL, json=BODY).json()
    assert first.pop("request_id") != second.pop("request_id")
    assert first == second


@pytest.mark.parametrize("body", [
    {}, {"prediction_time": None}, {"prediction_time": 1441173600},
    {"prediction_time": "invalid"}, {"prediction_time": "2026-10-18T06:00:00"},
    {"prediction_time": "2015-09-03T06:00:00"}, {"prediction_time": "2015-09-02T06:00:01"},
    {"prediction_time": "2015-09-02T06:00:00Z"}, {"prediction_time": "2015-09-02T06:00:00+05:30"},
    {**BODY, "machine_id": 1}, {**BODY, "limit": 10},
])
def test_invalid_ranking_inputs(ranking_client, body):
    assert ranking_client.post(URL, json=body).status_code == 422


@pytest.mark.parametrize("time", ["2015-09-02T06:00:00", "2015-11-25T06:00:00"])
def test_supported_historical_boundaries(ranking_client, time):
    assert ranking_client.post(URL, json={"prediction_time": time}).status_code == 200


def test_incomplete_benchmark_is_not_silently_ranked(client):
    response = client.post(URL, json=BODY)  # Existing fixture contains only two IDs.
    assert response.status_code == 503
    assert "all 100" in response.json()["detail"]


@pytest.mark.parametrize("problem", ["missing_id", "extra_id"])
def test_wrong_benchmark_machine_set(ranking_settings, problem):
    path = ranking_settings.c3_data_dir / "PdM_machines.csv"
    frame = pd.read_csv(path)
    if problem == "missing_id":
        frame = frame[frame.machineID != 100]
    else:
        frame.loc[len(frame)] = [101, "model1", 2]
    frame.to_csv(path, index=False)
    with TestClient(create_app(ranking_settings)) as client:
        assert client.post(URL, json=BODY).status_code == 503


@pytest.mark.parametrize("missing", ["model", "data"])
def test_unavailable_resources(ranking_settings, tmp_path, missing):
    settings = replace(ranking_settings, **{
        "c3_model_path" if missing == "model" else "c3_data_dir": tmp_path / "missing",
    })
    with TestClient(create_app(settings)) as client:
        assert client.post(URL, json=BODY).status_code == 503


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1, "wrong_shape"])
def test_bad_model_outputs_fail_without_partial_ranking(ranking_client, monkeypatch, bad):
    def probabilities(features):
        if isinstance(bad, str):
            return np.zeros((99, 2))
        p = np.full(100, 0.25)
        p[99] = bad
        return np.column_stack((1 - p, p))

    monkeypatch.setattr(ranking_client.app.state.component3_service.resources.saved["model"], "predict_proba", probabilities)
    response = ranking_client.post(URL, json=BODY)
    assert response.status_code == 500
    assert response.json() == {"detail": "Component 3 ranking failed"}


def test_openapi_registers_new_endpoint(ranking_client):
    assert URL in ranking_client.get("/openapi.json").json()["paths"]


@pytest.mark.integration
def test_actual_batch_agrees_with_all_individual_results_and_true_top_ten(local_settings):
    before = sha256(local_settings.c3_model_path.read_bytes()).hexdigest()
    resources = load_resources(local_settings)
    inference = get_research_inference()
    scored = score_machines(resources, BODY["prediction_time"])
    assert len(scored) == 100
    individual_raw = []
    for machine_id, batch_probability in scored:
        detail = inference.predict_from_loaded(resources.saved, resources.sources, machine_id, BODY["prediction_time"])
        assert abs(batch_probability - detail["predicted_failure_probability"]) <= 0.000000500001
        features = inference.machine_features(*resources.sources, machine_id, BODY["prediction_time"])
        raw = float(resources.saved["model"].predict_proba(features[inference.FEATURE_COLUMNS])[0, 1])
        assert np.isclose(batch_probability, raw, rtol=0, atol=1e-12)
        individual_raw.append((machine_id, raw))
    expected = sorted(individual_raw, key=lambda item: (-item[1], item[0]))[:10]
    with TestClient(create_app(local_settings)) as client:
        result = client.post(URL, json=BODY)
        assert result.status_code == 200
        rows = result.json()["machines"]
        assert [row["machine_id"] for row in rows] == [mid for mid, _ in expected]
        assert [row["predicted_failure_probability"] for row in rows] == [round(p, 6) for _, p in expected]
        for row in rows:
            detail = client.post(DETAIL_URL, json={**BODY, "machine_id": row["machine_id"]}).json()
            assert detail["predicted_failure_probability"] == row["predicted_failure_probability"]
            assert detail["model_version"] == "sha256:" + before
    assert sha256(local_settings.c3_model_path.read_bytes()).hexdigest() == before


@pytest.mark.integration
def test_actual_ranking_strict_past_integrity(local_settings):
    with TestClient(create_app(local_settings)) as client:
        before = client.post(URL, json=BODY).json()
        for frame in client.app.state.component3_service.resources.sources[1:]:
            for time in [BODY["prediction_time"], "2015-09-05T06:00:00"]:
                added = frame.iloc[0].copy()
                added["machineID"] = 1
                added["datetime"] = pd.Timestamp(time)
                frame.loc[len(frame)] = added
        after = client.post(URL, json=BODY).json()
        before.pop("request_id")
        after.pop("request_id")
        assert before == after
