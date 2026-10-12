"""End-to-end integration tests for the full Component 1 workflow.

Chains: upload → detect → persist → retrieve → XAI (both methods) →
retrieve saved explanations → history/filter/paginate → reopen result.

Uses mock DetectionService for deterministic results. Real-model integration
is covered by test_inspect::TestRealModel and test_explanations::TestRealModelXAI.
"""
import io
import os
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

BEST_PT = os.path.normpath(os.path.join(
    BACKEND_DIR, "..", "data-preprocessing", "training_runs",
    "baseline_v1", "train", "weights", "best.pt",
))
HAS_WEIGHTS = os.path.isfile(BEST_PT)

BASE = "/api/v1/component1"


def _jpeg(width=200, height=150):
    img = np.full((height, width, 3), (100, 120, 140), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


def _png(width=80, height=80):
    img = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return bytes(buf)


class _FakeDet:
    def __init__(self, idx, cls_id, cls_name, conf=0.85, bbox=None):
        self.detection_id = idx
        self.class_id = cls_id
        self.class_name = cls_name
        self.confidence = conf
        self.bbox_xyxy = bbox or [10.1, 20.2, 90.9, 130.8]
        self.pre_nms_index = 200 + idx


class _FakeInf:
    def __init__(self, dets, shape):
        self.detections = dets
        self.image_shape = shape
        self.model_version = "ultralytics-8.4.175"
        self.error_message = ""


class _FakeExpl:
    def __init__(self, status="success", method="gradcam", error_msg=""):
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
        self.heatmap_raw = np.zeros((20, 20), dtype=np.float32)
        self.heatmap_overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        self.is_finite = True
        self.is_nonzero = status == "success"
        self.heatmap_max = 1.0 if status == "success" else 0.0
        self.generation_time_s = 0.42
        self.error_message = error_msg
        self.model_version = "ultralytics-8.4.175"
        self.settings = {}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("C1_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("C1_AUTH_DISABLED", "true")
    monkeypatch.setenv("C1_MODEL_WEIGHTS", BEST_PT if HAS_WEIGHTS else "/fake/model.pt")
    monkeypatch.setenv("C1_MODEL_DEVICE", "cpu")

    from app.components.component1.services import DetectionService

    svc = DetectionService()
    svc._xai = True
    svc._model_version = "ultralytics-8.4.175"
    svc._device = "cpu"
    svc._weights_hash = "sha256:fakehash"
    svc._loaded_at = None

    async def fake_detect(image_bgr, conf_threshold=0.25, iou_threshold=0.5):
        h, w = image_bgr.shape[:2]
        return _FakeInf(
            dets=[
                _FakeDet(0, 6, "stain", 0.92),
                _FakeDet(1, 3, "cut", 0.78),
            ],
            shape=(h, w),
        ), 55

    async def fake_explain(image_bgr, detection, method="gradcam",
                           conf_threshold=0.25, iou_threshold=0.5):
        return _FakeExpl(method=method), 500

    svc.run_detection = fake_detect
    svc.run_explanation = fake_explain

    import app.components.component1.endpoints as ep
    ep._storage = None
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c
    ep.detection_service = original
    ep._storage = None


def _upload(c, data=None):
    resp = c.post(
        f"{BASE}/inspect",
        files={"file": ("test.jpg", io.BytesIO(data or _jpeg()), "image/jpeg")},
    )
    assert resp.status_code == 201
    return resp.json()


class TestFullWorkflow:
    """Upload → detect → retrieve → XAI → history → reopen."""

    def test_upload_detect_retrieve_xai_history(self, client):
        # 1) Upload and detect
        insp = _upload(client)
        insp_id = insp["id"]
        assert insp["summary"]["total_defects"] == 2
        assert sorted(insp["summary"]["classes_found"]) == ["cut", "stain"]
        assert len(insp["detections"]) == 2

        # 2) Retrieve persisted inspection
        get_resp = client.get(f"{BASE}/inspections/{insp_id}")
        assert get_resp.status_code == 200
        got = get_resp.json()
        assert got["id"] == insp_id
        assert got["summary"]["total_defects"] == 2
        assert len(got["detections"]) == 2

        # 3) Image artifact accessible
        img_url = got["image_url"]
        art_resp = client.get(img_url)
        assert art_resp.status_code == 200
        assert art_resp.headers["content-type"] in ("image/jpeg", "image/png")

        # 4) Generate Grad-CAM for first detection
        det0_id = got["detections"][0]["id"]
        xai1 = client.post(
            f"{BASE}/inspections/{insp_id}/explanations",
            json={"detection_id": det0_id, "method": "grad-cam"},
        )
        assert xai1.status_code == 201
        xai1_body = xai1.json()
        assert xai1_body["outcome"] == "success"
        assert xai1_body["overlay_url"] is not None

        # 5) Generate Grad-CAM++ for same detection
        xai2 = client.post(
            f"{BASE}/inspections/{insp_id}/explanations",
            json={"detection_id": det0_id, "method": "grad-cam++"},
        )
        assert xai2.status_code == 201
        assert xai2.json()["xai_method"] == "grad-cam++"

        # 6) Retrieve saved explanations (no ML rerun)
        saved = client.get(
            f"{BASE}/inspections/{insp_id}/detections/{det0_id}/explanations"
        )
        assert saved.status_code == 200
        methods = {e["xai_method"] for e in saved.json()}
        assert methods == {"grad-cam", "grad-cam++"}

        # 7) Get single explanation by ID
        expl_id = xai1_body["id"]
        single = client.get(f"{BASE}/inspections/{insp_id}/explanations/{expl_id}")
        assert single.status_code == 200
        assert single.json()["id"] == expl_id

        # 8) Overlay artifact accessible
        overlay_url = xai1_body["overlay_url"]
        overlay_resp = client.get(overlay_url)
        assert overlay_resp.status_code == 200

        # 9) History lists the inspection
        hist = client.get(f"{BASE}/inspections?limit=1")
        assert hist.status_code == 200
        items = hist.json()["items"]
        newest_id = items[0]["id"]
        assert newest_id == insp_id

        # 10) History class filter
        hist_stain = client.get(f"{BASE}/inspections?class_name=stain&limit=1")
        assert hist_stain.json()["items"][0]["id"] == insp_id

        # 11) Reopen from history (same as step 2)
        reopen = client.get(f"{BASE}/inspections/{newest_id}")
        assert reopen.status_code == 200
        assert len(reopen.json()["detections"]) == 2


class TestCrossEndpointErrors:
    """Error paths that span multiple endpoints."""

    def test_xai_for_wrong_inspection(self, client):
        insp = _upload(client)
        det_id = insp["detections"][0]["id"]
        import uuid
        fake_insp = str(uuid.uuid4())
        resp = client.post(
            f"{BASE}/inspections/{fake_insp}/explanations",
            json={"detection_id": det_id, "method": "grad-cam"},
        )
        assert resp.status_code == 404

    def test_get_missing_inspection(self, client):
        import uuid
        resp = client.get(f"{BASE}/inspections/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_artifact_path_traversal(self, client):
        resp = client.get(f"/api/v1/component1/artifacts/../../etc/passwd")
        assert resp.status_code in (400, 404)

    def test_upload_then_filter_by_wrong_class(self, client):
        _upload(client)
        resp = client.get(f"{BASE}/inspections?class_name=nonexistent")
        assert resp.status_code == 400


class TestPngUploadWorkflow:
    """Verify PNG upload through the same pipeline."""

    def test_png_upload_and_retrieve(self, client):
        data = _png()
        resp = client.post(
            f"{BASE}/inspect",
            files={"file": ("test.png", io.BytesIO(data), "image/png")},
        )
        assert resp.status_code == 201
        insp = resp.json()
        assert insp["summary"]["total_defects"] == 2

        got = client.get(f"{BASE}/inspections/{insp['id']}")
        assert got.status_code == 200


class TestDateFilteredHistory:
    """Date filters with upload timestamps."""

    def test_recent_upload_in_range(self, client):
        past = urllib.parse.quote(
            (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        )
        future = urllib.parse.quote(
            (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        )
        url = f"{BASE}/inspections?from_date={past}&to_date={future}"
        before = client.get(url).json()["total"]
        _upload(client)
        after = client.get(url).json()["total"]
        assert after == before + 1

    def test_far_future_excludes_all(self, client):
        _upload(client)
        far = urllib.parse.quote(
            (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        )
        resp = client.get(f"{BASE}/inspections?from_date={far}")
        assert resp.json()["total"] == 0
