"""Tests for Component 1 health and readiness endpoints."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

BEST_PT = os.path.normpath(os.path.join(
    BACKEND_DIR, "..", "runs", "component1_training",
    "E02_optimized", "weights", "best.pt",
))
HAS_WEIGHTS = os.path.isfile(BEST_PT)


@pytest.fixture()
def client_with_model(monkeypatch):
    """TestClient with real model loaded."""
    monkeypatch.setenv("C1_MODEL_WEIGHTS", BEST_PT)
    monkeypatch.setenv("C1_MODEL_DEVICE", "cpu")
    monkeypatch.setenv("C1_AUTH_DISABLED", "true")

    from app.components.component1.services import DetectionService
    svc = DetectionService()
    svc.load_model(BEST_PT, device="cpu")

    import app.components.component1.endpoints as ep
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c
    ep.detection_service = original


@pytest.fixture()
def client_no_model(monkeypatch):
    """TestClient with missing model weights."""
    monkeypatch.setenv("C1_MODEL_WEIGHTS", "/nonexistent/model.pt")
    monkeypatch.setenv("C1_AUTH_DISABLED", "true")

    from app.components.component1.services import DetectionService
    svc = DetectionService()
    svc.load_model("/nonexistent/model.pt")

    import app.components.component1.endpoints as ep
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c
    ep.detection_service = original


@pytest.mark.skipif(not HAS_WEIGHTS, reason="best.pt not found")
class TestHealthWithModel:

    def test_health_returns_200(self, client_with_model):
        resp = client_with_model.get("/api/v1/component1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_has_model_version(self, client_with_model):
        data = client_with_model.get("/api/v1/component1/health").json()
        assert "ultralytics" in data["model_version"]

    def test_health_has_ultralytics_version(self, client_with_model):
        data = client_with_model.get("/api/v1/component1/health").json()
        assert data["ultralytics_version"] == "8.4.175"

    def test_health_has_seven_classes(self, client_with_model):
        data = client_with_model.get("/api/v1/component1/health").json()
        assert len(data["classes"]) == 7
        assert "baekra" in data["classes"]
        assert "stain" in data["classes"]

    def test_health_has_device(self, client_with_model):
        data = client_with_model.get("/api/v1/component1/health").json()
        assert data["device"] == "cpu"

    def test_health_has_loaded_at(self, client_with_model):
        data = client_with_model.get("/api/v1/component1/health").json()
        assert data["loaded_at"] is not None

    def test_readiness_true(self, client_with_model):
        data = client_with_model.get("/api/v1/component1/readiness").json()
        assert data["ready"] is True
        assert data["model_loaded"] is True


class TestHealthWithoutModel:

    def test_health_returns_503(self, client_no_model):
        resp = client_no_model.get("/api/v1/component1/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unavailable"

    def test_readiness_false(self, client_no_model):
        data = client_no_model.get("/api/v1/component1/readiness").json()
        assert data["ready"] is False
        assert data["model_loaded"] is False


class TestConfig:

    def test_settings_load(self):
        from app.core.config import Settings
        s = Settings(C1_MODEL_WEIGHTS="/tmp/test.pt", C1_MODEL_DEVICE="cpu")
        assert s.C1_MODEL_WEIGHTS == "/tmp/test.pt"
        assert s.C1_AUTH_DISABLED is True

    def test_ml_import_path_resolves(self):
        from app.components.component1.ml import CLASS_NAMES, VERIFY_SCORE_ATOL
        assert len(CLASS_NAMES) == 7
        assert VERIFY_SCORE_ATOL == 1e-4
