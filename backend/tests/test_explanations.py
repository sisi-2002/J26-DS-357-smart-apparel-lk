"""Tests for POST/GET explanation endpoints."""
import io
import os
import sys
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


def _make_jpeg_bytes(width=100, height=100):
    img = np.full((height, width, 3), (128, 128, 128), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
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


class _FakeExplanationResult:
    def __init__(self, status="success", method="gradcam", overlay=None, heatmap=None, error_msg=""):
        self.status = status
        self.method = method
        self.detection_id = 0
        self.class_id = 0
        self.class_name = "stain"
        self.target_layer = "model.model[18]"
        self.target_scale = 1
        self.target_cell_gy_gx = (5, 5)
        self.target_logit = 2.5
        self.target_sigmoid = 0.92
        self.heatmap_raw = heatmap if heatmap is not None else np.zeros((20, 20), dtype=np.float32)
        self.heatmap_overlay = overlay if overlay is not None else np.zeros((100, 100, 3), dtype=np.uint8)
        self.is_finite = True
        self.is_nonzero = status == "success"
        self.heatmap_max = 1.0 if status == "success" else 0.0
        self.generation_time_s = 0.5
        self.error_message = error_msg
        self.model_version = "ultralytics-8.4.175"
        self.settings = {}


def _make_service(detect_result=None, explain_result=None):
    from app.components.component1.services import DetectionService
    svc = DetectionService()
    svc._xai = True
    svc._model_version = "ultralytics-8.4.175"
    svc._device = "cpu"
    svc._weights_hash = "sha256:fakehash"
    svc._loaded_at = None

    det = detect_result or _FakeInferenceResult(
        detections=[_FakeDetection(0)], shape=(100, 100)
    )

    async def fake_detect(image_bgr, conf_threshold=0.25, iou_threshold=0.5):
        return det, 42

    svc.run_detection = fake_detect

    expl = explain_result or _FakeExplanationResult()

    async def fake_explain(image_bgr, detection, method="gradcam",
                           conf_threshold=0.25, iou_threshold=0.5):
        return expl, 500

    svc.run_explanation = fake_explain
    return svc


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("C1_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("C1_AUTH_DISABLED", "true")
    monkeypatch.setenv("C1_MODEL_WEIGHTS", BEST_PT if HAS_WEIGHTS else "/fake/model.pt")
    monkeypatch.setenv("C1_MODEL_DEVICE", "cpu")

    import app.components.component1.endpoints as ep
    ep._storage = None
    svc = _make_service()
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c, svc
    ep.detection_service = original
    ep._storage = None


def _create_inspection(client):
    data = _make_jpeg_bytes()
    resp = client.post(
        "/api/v1/component1/inspect",
        files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
    )
    assert resp.status_code == 201
    return resp.json()


class TestExplanationGeneration:

    def test_gradcam_success(self, client):
        c, _ = client
        insp = _create_inspection(c)
        det_id = insp["detections"][0]["id"]
        resp = c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "completed"
        assert body["outcome"] == "success"
        assert body["xai_method"] == "grad-cam"
        assert body["overlay_url"] is not None
        assert body["detection_id"] == det_id

    def test_gradcampp_success(self, client):
        c, _ = client
        insp = _create_inspection(c)
        det_id = insp["detections"][0]["id"]
        resp = c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam++"},
        )
        assert resp.status_code == 201
        assert resp.json()["xai_method"] == "grad-cam++"

    def test_invalid_method_rejected(self, client):
        c, _ = client
        insp = _create_inspection(c)
        det_id = insp["detections"][0]["id"]
        resp = c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "invalid"},
        )
        assert resp.status_code == 422

    def test_wrong_inspection_detection(self, client):
        c, _ = client
        insp = _create_inspection(c)
        fake_det_id = str(uuid.uuid4())
        resp = c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": fake_det_id, "method": "grad-cam"},
        )
        assert resp.status_code == 404

    def test_missing_inspection(self, client):
        c, _ = client
        fake_insp_id = str(uuid.uuid4())
        fake_det_id = str(uuid.uuid4())
        resp = c.post(
            f"/api/v1/component1/inspections/{fake_insp_id}/explanations",
            json={"detection_id": fake_det_id, "method": "grad-cam"},
        )
        assert resp.status_code == 404

    def test_unique_artifact_paths(self, client):
        c, _ = client
        insp = _create_inspection(c)
        det_id = insp["detections"][0]["id"]
        r1 = c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        r2 = c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam++"},
        )
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["overlay_url"] != r2.json()["overlay_url"]
        assert r1.json()["id"] != r2.json()["id"]


class TestExplanationOutcomes:

    @pytest.fixture()
    def degenerate_client(self, monkeypatch, tmp_path):
        monkeypatch.setenv("C1_STORAGE_DIR", str(tmp_path))
        monkeypatch.setenv("C1_AUTH_DISABLED", "true")
        monkeypatch.setenv("C1_MODEL_WEIGHTS", "/fake/model.pt")
        monkeypatch.setenv("C1_MODEL_DEVICE", "cpu")

        import app.components.component1.endpoints as ep
        ep._storage = None
        svc = _make_service(
            explain_result=_FakeExplanationResult(status="degenerate")
        )
        original = ep.detection_service
        ep.detection_service = svc
        from app.main import app
        with TestClient(app) as c:
            yield c
        ep.detection_service = original
        ep._storage = None

    @pytest.fixture()
    def unavailable_client(self, monkeypatch, tmp_path):
        monkeypatch.setenv("C1_STORAGE_DIR", str(tmp_path))
        monkeypatch.setenv("C1_AUTH_DISABLED", "true")
        monkeypatch.setenv("C1_MODEL_WEIGHTS", "/fake/model.pt")
        monkeypatch.setenv("C1_MODEL_DEVICE", "cpu")

        import app.components.component1.endpoints as ep
        ep._storage = None
        svc = _make_service(
            explain_result=_FakeExplanationResult(status="unavailable")
        )
        original = ep.detection_service
        ep.detection_service = svc
        from app.main import app
        with TestClient(app) as c:
            yield c
        ep.detection_service = original
        ep._storage = None

    @pytest.fixture()
    def error_client(self, monkeypatch, tmp_path):
        monkeypatch.setenv("C1_STORAGE_DIR", str(tmp_path))
        monkeypatch.setenv("C1_AUTH_DISABLED", "true")
        monkeypatch.setenv("C1_MODEL_WEIGHTS", "/fake/model.pt")
        monkeypatch.setenv("C1_MODEL_DEVICE", "cpu")

        import app.components.component1.endpoints as ep
        ep._storage = None
        svc = _make_service(
            explain_result=_FakeExplanationResult(status="error", error_msg="Test error")
        )
        original = ep.detection_service
        ep.detection_service = svc
        from app.main import app
        with TestClient(app) as c:
            yield c
        ep.detection_service = original
        ep._storage = None

    def test_degenerate_outcome(self, degenerate_client):
        insp = _create_inspection(degenerate_client)
        det_id = insp["detections"][0]["id"]
        resp = degenerate_client.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        assert resp.status_code == 201
        assert resp.json()["outcome"] == "degenerate"

    def test_unavailable_outcome(self, unavailable_client):
        insp = _create_inspection(unavailable_client)
        det_id = insp["detections"][0]["id"]
        resp = unavailable_client.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        assert resp.status_code == 201
        assert resp.json()["outcome"] == "unavailable"

    def test_error_outcome(self, error_client):
        insp = _create_inspection(error_client)
        det_id = insp["detections"][0]["id"]
        resp = error_client.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["outcome"] == "error"
        assert body["error_message"] == "Test error"


class TestModelMismatch:

    def test_model_version_mismatch(self, monkeypatch, tmp_path):
        monkeypatch.setenv("C1_STORAGE_DIR", str(tmp_path))
        monkeypatch.setenv("C1_AUTH_DISABLED", "true")
        monkeypatch.setenv("C1_MODEL_WEIGHTS", "/fake/model.pt")
        monkeypatch.setenv("C1_MODEL_DEVICE", "cpu")

        import app.components.component1.endpoints as ep
        ep._storage = None
        svc = _make_service()
        svc._model_version = "ultralytics-9.0.0"
        original = ep.detection_service
        ep.detection_service = svc
        from app.main import app
        with TestClient(app) as c:
            svc._model_version = "ultralytics-8.4.175"
            insp = _create_inspection(c)
            det_id = insp["detections"][0]["id"]

            svc._model_version = "ultralytics-9.0.0"
            resp = c.post(
                f"/api/v1/component1/inspections/{insp['id']}/explanations",
                json={"detection_id": det_id, "method": "grad-cam"},
            )
            assert resp.status_code == 201
            body = resp.json()
            assert body["outcome"] == "unavailable"
            assert "mismatch" in body["error_message"].lower()

        ep.detection_service = original
        ep._storage = None


class TestExplanationRetrieval:

    def test_get_explanation(self, client):
        c, _ = client
        insp = _create_inspection(c)
        det_id = insp["detections"][0]["id"]
        create_resp = c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        expl_id = create_resp.json()["id"]
        get_resp = c.get(
            f"/api/v1/component1/inspections/{insp['id']}/explanations/{expl_id}"
        )
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == expl_id

    def test_get_missing_explanation(self, client):
        c, _ = client
        insp = _create_inspection(c)
        fake_id = str(uuid.uuid4())
        resp = c.get(
            f"/api/v1/component1/inspections/{insp['id']}/explanations/{fake_id}"
        )
        assert resp.status_code == 404

    def test_list_explanations_for_detection(self, client):
        c, _ = client
        insp = _create_inspection(c)
        det_id = insp["detections"][0]["id"]
        c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam++"},
        )
        resp = c.get(
            f"/api/v1/component1/inspections/{insp['id']}/detections/{det_id}/explanations"
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_detection_survives_xai_failure(self, client):
        c, svc = client

        async def fail_explain(*args, **kwargs):
            raise RuntimeError("GPU OOM")

        svc.run_explanation = fail_explain
        insp = _create_inspection(c)
        det_id = insp["detections"][0]["id"]
        resp = c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "failed"
        assert resp.json()["outcome"] == "error"

        get_insp = c.get(f"/api/v1/component1/inspections/{insp['id']}")
        assert get_insp.status_code == 200
        assert get_insp.json()["summary"]["total_defects"] == 1
        assert len(get_insp.json()["detections"]) == 1

    def test_overlay_serving(self, client):
        c, _ = client
        insp = _create_inspection(c)
        det_id = insp["detections"][0]["id"]
        resp = c.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        overlay_url = resp.json()["overlay_url"]
        assert overlay_url is not None
        art_resp = c.get(overlay_url)
        assert art_resp.status_code == 200
        assert "image" in art_resp.headers.get("content-type", "")


@pytest.mark.skipif(not HAS_WEIGHTS or not HAS_VALID, reason="model or validation images unavailable")
class TestRealModelXAI:

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

    def _upload_val_image(self, index=0):
        images = sorted(os.listdir(VALID_DIR))
        path = os.path.join(VALID_DIR, images[index])
        with open(path, "rb") as f:
            data = f.read()
        resp = self.client.post(
            "/api/v1/component1/inspect",
            files={"file": ("val.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 201
        return resp.json()

    def test_real_gradcam(self):
        insp = self._upload_val_image(0)
        if not insp["detections"]:
            pytest.skip("No detections on this validation image")
        det_id = insp["detections"][0]["id"]
        resp = self.client.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "completed"
        assert body["outcome"] in ("success", "degenerate", "unavailable", "ambiguous")
        assert body["inference_time_ms"] is not None

    def test_real_gradcampp(self):
        insp = self._upload_val_image(0)
        if not insp["detections"]:
            pytest.skip("No detections on this validation image")
        det_id = insp["detections"][0]["id"]
        resp = self.client.post(
            f"/api/v1/component1/inspections/{insp['id']}/explanations",
            json={"detection_id": det_id, "method": "grad-cam++"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "completed"
        assert body["xai_method"] == "grad-cam++"
