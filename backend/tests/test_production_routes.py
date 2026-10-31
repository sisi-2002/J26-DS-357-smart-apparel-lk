"""Production Function 1 remains independent of the removed evaluation workspace."""
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient


def test_only_function1_routes_registered():
    from app.main import app
    paths = set(app.openapi()["paths"])
    assert not any("evaluation" in path for path in paths)
    assert {
        "/api/v1/component1/inspect",
        "/api/v1/component1/inspections",
        "/api/v1/component1/inspections/{inspection_id}",
        "/api/v1/component1/inspections/{inspection_id}/explanations",
        "/api/v1/component1/health",
    } <= paths
    with TestClient(app) as client:
        assert client.get("/api/v1/component1/evaluations").status_code == 404


def test_frozen_e02_startup_and_synthetic_inspection(monkeypatch, tmp_path):
    from app.main import app, detection_service
    from app.core.config import settings
    from app.components.component1 import endpoints
    from app.components.component1.storage import ImageStorage
    root = Path(os.environ.get("C1_TEST_REPO_ROOT", Path(__file__).resolve().parents[2]))
    monkeypatch.setattr(settings, "C1_MODEL_WEIGHTS", str(root / "runs/component1_training/E02_optimized/weights/best.pt"))
    monkeypatch.setattr(settings, "C1_MODEL_FREEZE_MANIFEST", str(root / "runs/component1_training/freeze/model_freeze_E02_optimized_v1.json"))
    monkeypatch.setattr(endpoints, "detection_service", detection_service)
    monkeypatch.setattr(endpoints, "_storage", ImageStorage(str(tmp_path / "images")))
    # Existing autouse fixture provides and guards disposable SQLite, never PostgreSQL.
    assert settings.DATABASE_URL.startswith("sqlite+aiosqlite:///")
    with TestClient(app) as client:
        health = client.get("/api/v1/component1/health")
        assert health.status_code == 200, health.text
        assert health.json()["checkpoint_verified"] is True
        assert health.json()["experiment_id"] == "E02_optimized"
        _, image = cv2.imencode(".png", np.full((128, 160, 3), 128, dtype=np.uint8))
        response = client.post("/api/v1/component1/inspect", files={"file": ("synthetic.png", image.tobytes(), "image/png")})
        assert response.status_code == 201, response.text
        assert response.json()["model"]["experiment_id"] == "E02_optimized"
        assert client.get("/api/v1/component1/inspections/" + response.json()["id"]).status_code == 200
