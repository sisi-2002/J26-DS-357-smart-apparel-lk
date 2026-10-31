"""Regression tests using the existing fail-closed disposable SQLite fixture."""
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.components.component1 import endpoints as ep
from app.components.component1.model_identity import resolve_inspection_experiment_id
from app.components.component1.models import DefectInspection
from app.components.component1.storage import ImageStorage
from app.core.config import settings
import app.db.session as sessions

REPO_ROOT = Path(os.environ.get("C1_TEST_REPO_ROOT", Path(__file__).resolve().parents[2]))
MANIFEST = REPO_ROOT / "runs/component1_training/freeze/model_freeze_E02_optimized_v1.json"
E02_HASH = json.loads(MANIFEST.read_text(encoding="utf-8"))["selected_experiment"]["checkpoint_sha256"]
BASE = "/api/v1/component1"

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "C1_MODEL_FREEZE_MANIFEST", str(MANIFEST))
    monkeypatch.setattr(ep, "_storage", ImageStorage(str(tmp_path / "images")))
    monkeypatch.setattr(ep, "detection_service", SimpleNamespace(
        is_loaded=True, model_version="ultralytics-same-for-all-models", device="cpu",
        weights_hash="sha256:" + E02_HASH, experiment_id="E02_optimized",
        run_detection=AsyncMock(return_value=(SimpleNamespace(error_message="", detections=[]), 1)),
    ))
    sessions.init_engine()
    app = FastAPI()
    app.include_router(ep.router)
    with TestClient(app) as client:
        yield client


def test_fresh_post_get_and_history(client):
    _, image = cv2.imencode(".png", np.zeros((16, 16, 3), dtype=np.uint8))
    response = client.post(BASE + "/inspect", files={"file": ("fixture.png", image.tobytes(), "image/png")})
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["model"]["experiment_id"] == "E02_optimized"
    assert created["model"]["model_weights_hash"] == "sha256:" + E02_HASH
    ep.detection_service.experiment_id = "different-live-model"
    ep.detection_service.weights_hash = "sha256:" + "f" * 64
    for _ in range(2):
        assert client.get(BASE + "/inspections/" + created["id"]).json()["model"] == created["model"]
    history = client.get(BASE + "/inspections").json()
    assert history["items"][0]["id"] == created["id"]
    assert client.get(BASE + "/inspections/" + history["items"][0]["id"]).json()["model"]["experiment_id"] == "E02_optimized"


@pytest.mark.parametrize("digest, expected", [("sha256:" + E02_HASH, "E02_optimized"), (E02_HASH.upper(), "E02_optimized"), (None, None), ("sha256:" + "f" * 64, None), (E02_HASH[:12], None)])
def test_saved_records_use_only_their_own_hash(client, digest, expected):
    inspection_id = uuid.uuid4()
    async def seed():
        async with sessions.async_session_factory() as session:
            session.add(DefectInspection(id=inspection_id, image_path="fixture.png", image_width=16, image_height=16, image_hash="fixture", detected_at=datetime.now(timezone.utc), model_version="ultralytics-same-for-all-models", model_weights_hash=digest, total_defects=0, confidence_threshold=.25, iou_threshold=.5, imgsz=640))
            await session.commit()
    asyncio.run(seed())
    response = client.get(f"{BASE}/inspections/{inspection_id}")
    assert response.status_code == 200, response.text
    assert response.json()["model"]["experiment_id"] == expected
    assert response.json()["model"]["model_weights_hash"] == digest


@pytest.mark.parametrize("contents", [None, "not json", "[]", '{}', '{"selected_experiment": {"checkpoint_sha256": "abc", "experiment_id": "E02_optimized"}}'])
def test_unusable_manifest_fails_closed(tmp_path, contents):
    path = tmp_path / "manifest.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    assert resolve_inspection_experiment_id(E02_HASH, str(path)) is None


def test_no_manifest_cannot_attribute_identity():
    assert resolve_inspection_experiment_id(E02_HASH, None) is None
