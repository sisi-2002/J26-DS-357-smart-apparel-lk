"""Tests for POST /inspect and GET /inspections/{id} endpoints."""
import io
import os
import shutil
import sys
import tempfile
import uuid

import cv2
import numpy as np
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

VALID_DIR = os.path.normpath(os.path.join(
    BACKEND_DIR, "..", "data-preprocessing", "master_fabric_dataset",
    "processed", "chenab_detection_training_v1", "valid", "images",
))
HAS_VALID = os.path.isdir(VALID_DIR)


def _make_jpeg_bytes(width=100, height=100, color=(128, 128, 128)):
    img = np.full((height, width, 3), color, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


def _make_png_bytes(width=100, height=100):
    img = np.full((height, width, 3), (64, 64, 64), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return bytes(buf)


class _FakeDetection:
    def __init__(self, idx=0, cls_id=0, conf=0.9, bbox=None):
        self.detection_id = idx
        self.class_id = cls_id
        self.class_name = "stain"
        self.confidence = conf
        self.bbox_xyxy = bbox or [10.5, 20.3, 50.7, 80.1]
        self.pre_nms_index = 100 + idx


class _FakeInferenceResult:
    def __init__(self, detections=None, shape=(100, 100), error=""):
        self.detections = detections if detections is not None else []
        self.image_shape = shape
        self.model_version = "ultralytics-8.4.175"
        self.error_message = error


@pytest.fixture()
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("C1_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("C1_AUTH_DISABLED", "true")
    monkeypatch.setenv("C1_MODEL_WEIGHTS", BEST_PT if HAS_WEIGHTS else "/fake/model.pt")
    monkeypatch.setenv("C1_MODEL_DEVICE", "cpu")
    import app.components.component1.endpoints as ep
    ep._storage = None
    yield tmp_path
    ep._storage = None


@pytest.fixture()
def mock_service(monkeypatch, tmp_storage):
    """Client with mocked detection service — deterministic tests."""
    from app.components.component1.services import DetectionService

    svc = DetectionService()
    svc._xai = True
    svc._model_version = "ultralytics-8.4.175"
    svc._device = "cpu"
    svc._weights_hash = "sha256:fakehash"
    svc._loaded_at = None

    async def fake_run(image_bgr, conf_threshold=0.25, iou_threshold=0.5):
        return _FakeInferenceResult(
            detections=[_FakeDetection(0)],
            shape=image_bgr.shape[:2],
        ), 42

    svc.run_detection = fake_run

    import app.components.component1.endpoints as ep
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c
    ep.detection_service = original


@pytest.fixture()
def mock_service_empty(monkeypatch, tmp_storage):
    """Client with mocked service returning zero detections."""
    from app.components.component1.services import DetectionService

    svc = DetectionService()
    svc._xai = True
    svc._model_version = "ultralytics-8.4.175"
    svc._device = "cpu"
    svc._weights_hash = "sha256:fakehash"
    svc._loaded_at = None

    async def fake_run(image_bgr, conf_threshold=0.25, iou_threshold=0.5):
        return _FakeInferenceResult(detections=[], shape=image_bgr.shape[:2]), 10

    svc.run_detection = fake_run

    import app.components.component1.endpoints as ep
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c
    ep.detection_service = original


@pytest.fixture()
def mock_service_multi(monkeypatch, tmp_storage):
    """Client returning multiple detections."""
    from app.components.component1.services import DetectionService

    svc = DetectionService()
    svc._xai = True
    svc._model_version = "ultralytics-8.4.175"
    svc._device = "cpu"
    svc._weights_hash = "sha256:fakehash"
    svc._loaded_at = None

    async def fake_run(image_bgr, conf_threshold=0.25, iou_threshold=0.5):
        dets = [
            _FakeDetection(0, cls_id=6, conf=0.95, bbox=[10.1, 20.2, 30.3, 40.4]),
            _FakeDetection(1, cls_id=3, conf=0.80, bbox=[50.5, 60.6, 70.7, 80.8]),
            _FakeDetection(2, cls_id=0, conf=0.55, bbox=[90.0, 10.0, 99.9, 50.5]),
        ]
        return _FakeInferenceResult(detections=dets, shape=image_bgr.shape[:2]), 55

    svc.run_detection = fake_run

    import app.components.component1.endpoints as ep
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c
    ep.detection_service = original


@pytest.fixture()
def mock_service_fail(monkeypatch, tmp_storage):
    """Client with service that raises on inference."""
    from app.components.component1.services import DetectionService

    svc = DetectionService()
    svc._xai = True
    svc._model_version = "ultralytics-8.4.175"
    svc._device = "cpu"
    svc._weights_hash = "sha256:fakehash"
    svc._loaded_at = None

    async def fail_run(image_bgr, conf_threshold=0.25, iou_threshold=0.5):
        raise RuntimeError("GPU error")

    svc.run_detection = fail_run

    import app.components.component1.endpoints as ep
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c
    ep.detection_service = original


@pytest.fixture()
def unloaded_service(monkeypatch, tmp_storage):
    """Client with model not loaded."""
    from app.components.component1.services import DetectionService
    svc = DetectionService()

    import app.components.component1.endpoints as ep
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c
    ep.detection_service = original


# ---------- Upload validation ----------

class TestUploadValidation:

    def test_valid_jpeg(self, mock_service):
        data = _make_jpeg_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["image_width"] == 100
        assert body["image_height"] == 100
        assert body["summary"]["total_defects"] == 1

    def test_valid_png(self, mock_service):
        data = _make_png_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.png", io.BytesIO(data), "image/png")},
        )
        assert resp.status_code == 201

    def test_corrupt_image(self, mock_service):
        data = b"\xff\xd8\xff" + b"\x00" * 100
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("bad.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 422
        assert "decoded" in resp.json()["detail"].lower()

    def test_unsupported_format(self, mock_service):
        data = b"GIF89a" + b"\x00" * 100
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.gif", io.BytesIO(data), "image/gif")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_empty_file(self, mock_service):
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_oversized_image(self, mock_service, monkeypatch):
        import app.components.component1.storage as st
        monkeypatch.setattr(st, "MAX_IMAGE_BYTES", 100)
        data = _make_jpeg_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("big.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "limit" in resp.json()["detail"].lower()

    def test_invalid_threshold_too_high(self, mock_service):
        data = _make_jpeg_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect?confidence_threshold=1.5",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_invalid_threshold_negative(self, mock_service):
        data = _make_jpeg_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect?confidence_threshold=-0.1",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 422


# ---------- Detection results ----------

class TestDetectionResults:

    def test_empty_detections_is_success(self, mock_service_empty):
        data = _make_jpeg_bytes()
        resp = mock_service_empty.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["summary"]["total_defects"] == 0
        assert body["detections"] == []

    def test_multiple_detections(self, mock_service_multi):
        data = _make_jpeg_bytes()
        resp = mock_service_multi.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["summary"]["total_defects"] == 3
        assert len(body["detections"]) == 3
        orders = [d["detection_order"] for d in body["detections"]]
        assert orders == [0, 1, 2]
        assert all("id" in d for d in body["detections"])
        assert all(d["pre_nms_index"] >= 0 for d in body["detections"])

    def test_bbox_is_integer_in_response(self, mock_service):
        data = _make_jpeg_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        body = resp.json()
        bbox = body["detections"][0]["bbox"]
        for key in ("x_min", "y_min", "x_max", "y_max"):
            assert isinstance(bbox[key], int)

    def test_response_has_model_identity(self, mock_service):
        data = _make_jpeg_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        body = resp.json()
        assert body["model"]["model_version"] == "ultralytics-8.4.175"
        assert body["model"]["confidence_threshold"] == 0.25
        assert body["model"]["iou_threshold"] == 0.5

    def test_response_has_artifact_urls(self, mock_service):
        data = _make_jpeg_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        body = resp.json()
        assert body["image_url"].startswith("/api/v1/component1/artifacts/")
        assert body["thumbnail_url"].startswith("/api/v1/component1/artifacts/")

    def test_complexity_fields_null(self, mock_service):
        data = _make_jpeg_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        body = resp.json()
        assert body["complexity_tier"] is None
        assert body["complexity_score"] is None


# ---------- Persistence and retrieval ----------

class TestPersistence:

    def test_get_inspection_after_create(self, mock_service):
        data = _make_jpeg_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 201
        insp_id = resp.json()["id"]

        get_resp = mock_service.get(f"/api/v1/component1/inspections/{insp_id}")
        assert get_resp.status_code == 200
        body = get_resp.json()
        assert body["id"] == insp_id
        assert body["summary"]["total_defects"] == 1
        assert len(body["detections"]) == 1

    def test_get_missing_inspection(self, mock_service):
        fake_id = str(uuid.uuid4())
        resp = mock_service.get(f"/api/v1/component1/inspections/{fake_id}")
        assert resp.status_code == 404

    def test_artifact_serving(self, mock_service):
        data = _make_jpeg_bytes()
        resp = mock_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        image_url = resp.json()["image_url"]
        art_resp = mock_service.get(image_url)
        assert art_resp.status_code == 200
        assert "image" in art_resp.headers.get("content-type", "")

    def test_artifact_path_traversal_blocked(self, mock_service):
        resp = mock_service.get("/api/v1/component1/artifacts/../../etc/passwd")
        assert resp.status_code == 404


# ---------- Error handling ----------

class TestErrorHandling:

    def test_model_unavailable(self, unloaded_service):
        data = _make_jpeg_bytes()
        resp = unloaded_service.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 503

    def test_inference_failure_cleans_up(self, mock_service_fail, tmp_storage):
        data = _make_jpeg_bytes()
        resp = mock_service_fail.post(
            "/api/v1/component1/inspect",
            files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 503
        inspections_dir = tmp_storage / "inspections"
        if inspections_dir.exists():
            remaining = list(inspections_dir.iterdir())
            assert len(remaining) == 0


# ---------- Real model tests ----------

@pytest.mark.skipif(not HAS_WEIGHTS or not HAS_VALID, reason="model or validation images unavailable")
class TestRealModel:

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, tmp_path):
        monkeypatch.setenv("C1_STORAGE_DIR", str(tmp_path))
        monkeypatch.setenv("C1_AUTH_DISABLED", "true")
        monkeypatch.setenv("C1_MODEL_WEIGHTS", BEST_PT)
        monkeypatch.setenv("C1_MODEL_DEVICE", "cpu")

        from app.components.component1.services import DetectionService
        svc = DetectionService()
        svc.load_model(BEST_PT, device="cpu")

        import app.components.component1.endpoints as ep
        ep._storage = None
        original = ep.detection_service
        ep.detection_service = svc
        from app.main import app
        with TestClient(app) as c:
            self.client = c
            yield
        ep.detection_service = original
        ep._storage = None

    def _get_val_image(self, index=0):
        images = sorted(os.listdir(VALID_DIR))
        path = os.path.join(VALID_DIR, images[index])
        with open(path, "rb") as f:
            return f.read()

    def test_real_detection_with_val_image(self):
        data = self._get_val_image(0)
        resp = self.client.post(
            "/api/v1/component1/inspect",
            files={"file": ("val_image.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "id" in body
        assert body["image_width"] > 0
        assert body["image_height"] > 0
        assert body["model"]["model_version"].startswith("ultralytics-")
        assert body["inference_time_ms"] is not None
        assert isinstance(body["detections"], list)

    def test_real_get_after_inspect(self):
        data = self._get_val_image(0)
        post_resp = self.client.post(
            "/api/v1/component1/inspect",
            files={"file": ("val_image.jpg", io.BytesIO(data), "image/jpeg")},
        )
        insp_id = post_resp.json()["id"]

        get_resp = self.client.get(f"/api/v1/component1/inspections/{insp_id}")
        assert get_resp.status_code == 200
        get_body = get_resp.json()
        post_body = post_resp.json()
        assert get_body["summary"]["total_defects"] == post_body["summary"]["total_defects"]
        assert len(get_body["detections"]) == len(post_body["detections"])
