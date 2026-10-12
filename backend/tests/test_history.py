"""Tests for GET /inspections (history list) endpoint."""
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


def _make_jpeg_bytes(width=100, height=100):
    img = np.full((height, width, 3), (128, 128, 128), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


class _FakeDetection:
    def __init__(self, idx=0, cls_id=0, cls_name="stain", conf=0.9, bbox=None):
        self.detection_id = idx
        self.class_id = cls_id
        self.class_name = cls_name
        self.confidence = conf
        self.bbox_xyxy = bbox or [10.5, 20.3, 50.7, 80.1]
        self.pre_nms_index = 100 + idx


class _FakeInferenceResult:
    def __init__(self, detections=None, shape=(100, 100)):
        self.detections = detections if detections is not None else []
        self.image_shape = shape
        self.model_version = "ultralytics-8.4.175"
        self.error_message = ""


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


def _make_mock_service(detections_fn=None):
    from app.components.component1.services import DetectionService

    svc = DetectionService()
    svc._xai = True
    svc._model_version = "ultralytics-8.4.175"
    svc._device = "cpu"
    svc._weights_hash = "sha256:fakehash"
    svc._loaded_at = None

    async def fake_run(image_bgr, conf_threshold=0.25, iou_threshold=0.5):
        dets = detections_fn() if detections_fn else [_FakeDetection(0)]
        return _FakeInferenceResult(
            detections=dets,
            shape=image_bgr.shape[:2],
        ), 42

    svc.run_detection = fake_run
    return svc


@pytest.fixture()
def mock_client(monkeypatch, tmp_storage):
    svc = _make_mock_service()
    import app.components.component1.endpoints as ep
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c
    ep.detection_service = original


@pytest.fixture()
def multi_class_client(monkeypatch, tmp_storage):
    """Client whose model returns two detections of different classes."""
    svc = _make_mock_service(
        detections_fn=lambda: [
            _FakeDetection(0, cls_id=0, cls_name="stain"),
            _FakeDetection(1, cls_id=3, cls_name="cut"),
        ]
    )
    import app.components.component1.endpoints as ep
    original = ep.detection_service
    ep.detection_service = svc
    from app.main import app
    with TestClient(app) as c:
        yield c
    ep.detection_service = original


def _upload(client, jpeg_bytes=None):
    data = jpeg_bytes or _make_jpeg_bytes()
    resp = client.post(
        f"{BASE}/inspect",
        files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
    )
    assert resp.status_code == 201
    return resp.json()


def _get_baseline(client):
    resp = client.get(f"{BASE}/inspections?limit=1")
    assert resp.status_code == 200
    return resp.json()["total"]


def _encode_dt(dt):
    return urllib.parse.quote(dt.isoformat())


# ── Response shape ─────────────────────────────────────────────

class TestResponseShape:
    def test_shape_and_defaults(self, mock_client):
        resp = mock_client.get(f"{BASE}/inspections")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "items" in data
        assert data["skip"] == 0
        assert data["limit"] == 20

    def test_item_fields(self, mock_client):
        _upload(mock_client)
        resp = mock_client.get(f"{BASE}/inspections")
        item = resp.json()["items"][0]
        for field in ("id", "detected_at", "total_defects", "defect_classes",
                      "model_version", "thumbnail_url"):
            assert field in item


# ── Pagination ─────────────────────────────────────────────────

class TestPagination:
    def test_uploads_increase_total(self, mock_client):
        baseline = _get_baseline(mock_client)
        for _ in range(3):
            _upload(mock_client)
        resp = mock_client.get(f"{BASE}/inspections")
        assert resp.json()["total"] == baseline + 3

    def test_limit(self, mock_client):
        baseline = _get_baseline(mock_client)
        for _ in range(3):
            _upload(mock_client)
        resp = mock_client.get(f"{BASE}/inspections?limit=2")
        data = resp.json()
        assert data["total"] == baseline + 3
        assert len(data["items"]) == 2

    def test_skip_past_new(self, mock_client):
        baseline = _get_baseline(mock_client)
        _upload(mock_client)
        resp = mock_client.get(f"{BASE}/inspections?skip={baseline + 100}")
        data = resp.json()
        assert data["total"] == baseline + 1
        assert len(data["items"]) == 0


# ── Ordering ───────────────────────────────────────────────────

class TestOrdering:
    def test_newest_first(self, mock_client):
        ids = []
        for _ in range(3):
            ids.append(_upload(mock_client)["id"])
        resp = mock_client.get(f"{BASE}/inspections?limit=3")
        data = resp.json()
        returned_ids = [item["id"] for item in data["items"]]
        assert returned_ids[:3] == list(reversed(ids))


# ── Class filter ───────────────────────────────────────────────

class TestClassFilter:
    def test_filter_by_class(self, multi_class_client):
        before_stain = multi_class_client.get(
            f"{BASE}/inspections?class_name=stain"
        ).json()["total"]
        _upload(multi_class_client)
        after_stain = multi_class_client.get(
            f"{BASE}/inspections?class_name=stain"
        ).json()["total"]
        assert after_stain == before_stain + 1

    def test_no_duplicates_from_multi_class(self, multi_class_client):
        """An inspection with stain+cut must appear once when filtering by stain."""
        before = multi_class_client.get(
            f"{BASE}/inspections?class_name=stain"
        ).json()["total"]
        _upload(multi_class_client)
        after = multi_class_client.get(
            f"{BASE}/inspections?class_name=stain"
        ).json()["total"]
        assert after - before == 1

    def test_filter_nonexistent_class_in_upload(self, mock_client):
        """Mock returns stain only; filtering by baekra should not find it."""
        before = mock_client.get(
            f"{BASE}/inspections?class_name=baekra"
        ).json()["total"]
        _upload(mock_client)
        after = mock_client.get(
            f"{BASE}/inspections?class_name=baekra"
        ).json()["total"]
        assert after == before

    def test_invalid_class_name(self, mock_client):
        resp = mock_client.get(f"{BASE}/inspections?class_name=invalid_class")
        assert resp.status_code == 400

    def test_defect_classes_sorted(self, multi_class_client):
        insp = _upload(multi_class_client)
        detail = multi_class_client.get(f"{BASE}/inspections?limit=1")
        item = detail.json()["items"][0]
        assert item["id"] == insp["id"]
        assert sorted(item["defect_classes"]) == ["cut", "stain"]


# ── Date filter ────────────────────────────────────────────────

class TestDateFilter:
    def test_future_from_date_excludes_past_uploads(self, mock_client):
        _upload(mock_client)
        future = _encode_dt(datetime.now(timezone.utc) + timedelta(days=1))
        resp = mock_client.get(f"{BASE}/inspections?from_date={future}")
        assert resp.json()["total"] == 0

    def test_past_to_date_excludes_recent(self, mock_client):
        _upload(mock_client)
        past = _encode_dt(datetime.now(timezone.utc) - timedelta(days=365))
        resp = mock_client.get(f"{BASE}/inspections?to_date={past}")
        assert resp.json()["total"] == 0

    def test_wide_range_includes_upload(self, mock_client):
        past = _encode_dt(datetime.now(timezone.utc) - timedelta(hours=1))
        future = _encode_dt(datetime.now(timezone.utc) + timedelta(hours=1))
        url = f"{BASE}/inspections?from_date={past}&to_date={future}"
        before = mock_client.get(url).json()["total"]
        _upload(mock_client)
        after = mock_client.get(url).json()["total"]
        assert after == before + 1

    def test_from_after_to_rejected(self, mock_client):
        future = _encode_dt(datetime.now(timezone.utc) + timedelta(days=1))
        past = _encode_dt(datetime.now(timezone.utc) - timedelta(days=1))
        resp = mock_client.get(
            f"{BASE}/inspections?from_date={future}&to_date={past}"
        )
        assert resp.status_code == 400


# ── Validation ─────────────────────────────────────────────────

class TestValidation:
    def test_negative_skip(self, mock_client):
        resp = mock_client.get(f"{BASE}/inspections?skip=-1")
        assert resp.status_code == 422

    def test_limit_zero(self, mock_client):
        resp = mock_client.get(f"{BASE}/inspections?limit=0")
        assert resp.status_code == 422

    def test_limit_above_max(self, mock_client):
        resp = mock_client.get(f"{BASE}/inspections?limit=200")
        assert resp.status_code == 422


# ── Detail navigation ─────────────────────────────────────────

class TestDetailNavigation:
    def test_list_item_links_to_detail(self, mock_client):
        insp = _upload(mock_client)
        resp = mock_client.get(f"{BASE}/inspections?limit=1")
        item_id = resp.json()["items"][0]["id"]
        detail = mock_client.get(f"{BASE}/inspections/{item_id}")
        assert detail.status_code == 200
        assert detail.json()["id"] == insp["id"]
