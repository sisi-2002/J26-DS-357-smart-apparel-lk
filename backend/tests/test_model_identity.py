"""Tests for checkpoint identity verification (F1.2).

Covers model_identity.py pure-logic behaviour and its integration into
DetectionService.load_model(): expected-hash/freeze-manifest resolution,
mismatch rejection, and legacy-fallback prevention.

No real YOLO model is loaded — DetectionXAI is mocked for the positive
("loads successfully") path, and a fake non-model file is used for the
negative (mismatch/missing) paths, which never reach DetectionXAI.
"""
import hashlib
import json
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.components.component1.model_identity import (
    FreezeManifestError,
    load_freeze_manifest,
    resolve_expected_checkpoint,
    verify_checkpoint,
    _normalize_hash,
)
from app.components.component1.services import DetectionService


def _write_fake_checkpoint(path, content=b"FAKE_CHECKPOINT_BYTES_12345"):
    with open(path, "wb") as f:
        f.write(content)
    h = hashlib.sha256(content).hexdigest()
    return h


def _write_freeze_manifest(path, checkpoint_sha256, experiment_id="E02_optimized"):
    manifest = {
        "selected_experiment": {
            "experiment_id": experiment_id,
            "checkpoint_sha256": checkpoint_sha256,
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)


# ── _normalize_hash ──────────────────────────────────────────────

class TestNormalizeHash:
    def test_strips_sha256_prefix(self):
        assert _normalize_hash("sha256:ABCDEF") == "abcdef"

    def test_no_prefix_unchanged_lowercased(self):
        assert _normalize_hash("ABCDEF") == "abcdef"

    def test_whitespace_stripped(self):
        assert _normalize_hash("  abcdef  ") == "abcdef"


# ── resolve_expected_checkpoint ──────────────────────────────────

class TestResolveExpectedCheckpoint:
    def test_conflicting_explicit_hash_and_manifest_raises(self, tmp_path):
        """Both configured but disagreeing must be a hard error, never a
        silent 'pick one' — this is exactly the scenario that could mask
        a stale manifest or a wrong override."""
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, "deadbeef", "E01_from_manifest")

        with pytest.raises(FreezeManifestError, match="Conflicting"):
            resolve_expected_checkpoint(
                expected_sha256="cafebabe",
                freeze_manifest_path=manifest_path,
            )

    def test_agreeing_explicit_hash_and_manifest_succeeds(self, tmp_path):
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, "deadbeef", "E02_optimized")

        result = resolve_expected_checkpoint(
            expected_sha256="DEADBEEF",  # same value, different case
            freeze_manifest_path=manifest_path,
        )
        assert result.expected_sha256 == "deadbeef"
        assert result.experiment_id == "E02_optimized"
        assert result.source == "explicit_hash_and_freeze_manifest_agree"

    def test_freeze_manifest_used_when_no_explicit_hash(self, tmp_path):
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, "deadbeef", "E02_optimized")

        result = resolve_expected_checkpoint(freeze_manifest_path=manifest_path)
        assert result.expected_sha256 == "deadbeef"
        assert result.source == "freeze_manifest"
        assert result.experiment_id == "E02_optimized"

    def test_none_when_nothing_configured(self):
        assert resolve_expected_checkpoint() is None

    def test_malformed_manifest_raises(self, tmp_path):
        manifest_path = str(tmp_path / "bad.json")
        with open(manifest_path, "w") as f:
            f.write("not json{{{")
        with pytest.raises(FreezeManifestError):
            resolve_expected_checkpoint(freeze_manifest_path=manifest_path)

    def test_manifest_missing_checkpoint_hash_raises(self, tmp_path):
        manifest_path = str(tmp_path / "incomplete.json")
        with open(manifest_path, "w") as f:
            json.dump({"selected_experiment": {"experiment_id": "E02"}}, f)
        with pytest.raises(FreezeManifestError):
            resolve_expected_checkpoint(freeze_manifest_path=manifest_path)

    def test_missing_manifest_file_raises(self, tmp_path):
        with pytest.raises(FreezeManifestError):
            resolve_expected_checkpoint(
                freeze_manifest_path=str(tmp_path / "does_not_exist.json")
            )


# ── verify_checkpoint ────────────────────────────────────────────

class TestVerifyCheckpoint:
    def test_no_expectation_configured_passes(self, tmp_path):
        ckpt = str(tmp_path / "model.pt")
        _write_fake_checkpoint(ckpt)
        result = verify_checkpoint(ckpt)
        assert result.verified is True
        assert result.expected is None

    def test_matching_explicit_hash_passes(self, tmp_path):
        ckpt = str(tmp_path / "model.pt")
        actual_hash = _write_fake_checkpoint(ckpt)
        result = verify_checkpoint(ckpt, expected_sha256=actual_hash)
        assert result.verified is True
        assert result.actual_sha256 == actual_hash

    def test_matching_explicit_hash_with_sha256_prefix(self, tmp_path):
        ckpt = str(tmp_path / "model.pt")
        actual_hash = _write_fake_checkpoint(ckpt)
        result = verify_checkpoint(ckpt, expected_sha256=f"sha256:{actual_hash}")
        assert result.verified is True

    def test_mismatched_explicit_hash_fails(self, tmp_path):
        ckpt = str(tmp_path / "model.pt")
        _write_fake_checkpoint(ckpt)
        result = verify_checkpoint(ckpt, expected_sha256="0" * 64)
        assert result.verified is False
        assert "mismatch" in result.detail.lower()

    def test_matching_freeze_manifest_passes(self, tmp_path):
        ckpt = str(tmp_path / "model.pt")
        actual_hash = _write_fake_checkpoint(ckpt)
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, actual_hash, "E02_optimized")

        result = verify_checkpoint(ckpt, freeze_manifest_path=manifest_path)
        assert result.verified is True
        assert result.expected.experiment_id == "E02_optimized"

    def test_mismatched_freeze_manifest_fails_with_experiment_id_in_detail(self, tmp_path):
        ckpt = str(tmp_path / "model.pt")
        _write_fake_checkpoint(ckpt)
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, "f" * 64, "E02_optimized")

        result = verify_checkpoint(ckpt, freeze_manifest_path=manifest_path)
        assert result.verified is False
        assert "E02_optimized" in result.detail

    def test_detail_does_not_leak_full_hash(self, tmp_path):
        """Mismatch detail should show only a short prefix, not the full hash."""
        ckpt = str(tmp_path / "model.pt")
        _write_fake_checkpoint(ckpt)
        full_wrong_hash = "a" * 64
        result = verify_checkpoint(ckpt, expected_sha256=full_wrong_hash)
        assert full_wrong_hash not in result.detail


# ── DetectionService.load_model integration ──────────────────────

class TestLoadModelLegacyBehavior:
    """No expectation configured — must behave exactly as before (backward compat)."""

    def test_loads_when_no_verification_configured(self, tmp_path, monkeypatch):
        ckpt = str(tmp_path / "model.pt")
        _write_fake_checkpoint(ckpt)

        class _FakeXAI:
            _model_version = "ultralytics-fake"

        import app.components.component1.ml as ml_module
        monkeypatch.setattr(ml_module, "DetectionXAI", lambda path, device="cpu": _FakeXAI())

        svc = DetectionService()
        svc.load_model(ckpt, device="cpu")
        assert svc.is_loaded is True
        assert svc.load_error is None
        assert svc.checkpoint_verified is False  # no expectation was configured
        assert svc.experiment_id is None

    def test_missing_file_does_not_load(self):
        svc = DetectionService()
        svc.load_model("/nonexistent/path/model.pt")
        assert svc.is_loaded is False
        assert svc.load_error is not None


class TestLoadModelVerification:
    def test_matching_freeze_manifest_loads_and_records_experiment_id(self, tmp_path, monkeypatch):
        ckpt = str(tmp_path / "model.pt")
        actual_hash = _write_fake_checkpoint(ckpt)
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, actual_hash, "E02_optimized")

        class _FakeXAI:
            _model_version = "ultralytics-fake"

        import app.components.component1.ml as ml_module
        monkeypatch.setattr(ml_module, "DetectionXAI", lambda path, device="cpu": _FakeXAI())

        svc = DetectionService()
        svc.load_model(ckpt, device="cpu", freeze_manifest_path=manifest_path)

        assert svc.is_loaded is True
        assert svc.checkpoint_verified is True
        assert svc.experiment_id == "E02_optimized"

    def test_checkpoint_mismatch_blocks_loading(self, tmp_path, monkeypatch):
        """Core requirement: mismatched checkpoint must NOT load, and must
        NOT silently load the configured file anyway."""
        ckpt = str(tmp_path / "model.pt")
        _write_fake_checkpoint(ckpt)
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, "0" * 64, "E02_optimized")

        class _FakeXAI:
            _model_version = "ultralytics-fake"

        load_attempted = {"called": False}

        def _tracking_fake_xai(path, device="cpu"):
            load_attempted["called"] = True
            return _FakeXAI()

        import app.components.component1.ml as ml_module
        monkeypatch.setattr(ml_module, "DetectionXAI", _tracking_fake_xai)

        svc = DetectionService()
        svc.load_model(ckpt, device="cpu", freeze_manifest_path=manifest_path)

        assert svc.is_loaded is False
        assert load_attempted["called"] is False, "DetectionXAI must never be constructed on mismatch"
        assert "mismatch" in svc.load_error.lower()

    def test_no_legacy_fallback_on_mismatch(self, tmp_path, monkeypatch):
        """Simulates the real scenario: .env points at the frozen E02
        checkpoint, but the file on disk is actually the old legacy model.
        The service must refuse to load rather than silently using it."""
        legacy_ckpt = str(tmp_path / "legacy_baseline_v1.pt")
        legacy_hash = _write_fake_checkpoint(legacy_ckpt, b"OLD_LEGACY_BASELINE_V1_WEIGHTS")

        expected_e02_hash = "c578003ca158a892ac90de3db7b7552aefabd5292e311c6dd0c76faf717628d3"
        assert legacy_hash != expected_e02_hash

        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, expected_e02_hash, "E02_optimized")

        import app.components.component1.ml as ml_module
        monkeypatch.setattr(
            ml_module, "DetectionXAI",
            lambda path, device="cpu": pytest.fail("Must not load mismatched legacy checkpoint"),
        )

        svc = DetectionService()
        svc.load_model(legacy_ckpt, device="cpu", freeze_manifest_path=manifest_path)

        assert svc.is_loaded is False
        assert svc.model_version is None

    def test_explicit_and_manifest_agreeing_both_load(self, tmp_path, monkeypatch):
        ckpt = str(tmp_path / "model.pt")
        actual_hash = _write_fake_checkpoint(ckpt)
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, actual_hash, "E02_optimized")

        class _FakeXAI:
            _model_version = "ultralytics-fake"

        import app.components.component1.ml as ml_module
        monkeypatch.setattr(ml_module, "DetectionXAI", lambda path, device="cpu": _FakeXAI())

        svc = DetectionService()
        svc.load_model(
            ckpt, device="cpu",
            expected_sha256=actual_hash,
            freeze_manifest_path=manifest_path,
        )
        assert svc.is_loaded is True
        assert svc.experiment_id == "E02_optimized"

    def test_conflicting_explicit_and_manifest_blocks_loading(self, tmp_path, monkeypatch):
        """Core requirement: a conflicting explicit-hash vs freeze-manifest
        configuration must never silently 'claim a valid freeze' — it must
        block loading entirely, regardless of whether the actual checkpoint
        happens to match one of the two conflicting expectations."""
        ckpt = str(tmp_path / "model.pt")
        actual_hash = _write_fake_checkpoint(ckpt)
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, "0" * 64, "E02_optimized")  # disagrees with explicit

        load_attempted = {"called": False}

        import app.components.component1.ml as ml_module
        monkeypatch.setattr(
            ml_module, "DetectionXAI",
            lambda path, device="cpu": load_attempted.update(called=True),
        )

        svc = DetectionService()
        svc.load_model(
            ckpt, device="cpu",
            expected_sha256=actual_hash,  # matches the real file...
            freeze_manifest_path=manifest_path,  # ...but manifest disagrees
        )
        assert svc.is_loaded is False
        assert load_attempted["called"] is False
        assert "conflicting" in svc.load_error.lower()

    def test_malformed_manifest_blocks_loading_safely(self, tmp_path):
        ckpt = str(tmp_path / "model.pt")
        _write_fake_checkpoint(ckpt)
        manifest_path = str(tmp_path / "bad.json")
        with open(manifest_path, "w") as f:
            f.write("{not valid json")

        svc = DetectionService()
        svc.load_model(ckpt, device="cpu", freeze_manifest_path=manifest_path)
        assert svc.is_loaded is False
        assert svc.load_error is not None


# ── Health endpoint exposes identity fields ──────────────────────

class TestHealthExposesIdentity:
    def test_get_health_includes_experiment_id_and_verified_flag(self, tmp_path, monkeypatch):
        ckpt = str(tmp_path / "model.pt")
        actual_hash = _write_fake_checkpoint(ckpt)
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, actual_hash, "E02_optimized")

        class _FakeXAI:
            _model_version = "ultralytics-fake"

        import app.components.component1.ml as ml_module
        monkeypatch.setattr(ml_module, "DetectionXAI", lambda path, device="cpu": _FakeXAI())

        svc = DetectionService()
        svc.load_model(ckpt, device="cpu", freeze_manifest_path=manifest_path)

        health = svc.get_health()
        assert health["status"] == "healthy"
        assert health["experiment_id"] == "E02_optimized"
        assert health["checkpoint_verified"] is True

    def test_unavailable_health_does_not_leak_full_path(self, tmp_path):
        """The health 'detail' field on failure must not embed the raw
        absolute checkpoint path (avoid leaking local filesystem info)."""
        ckpt = str(tmp_path / "model.pt")
        _write_fake_checkpoint(ckpt)
        manifest_path = str(tmp_path / "freeze.json")
        _write_freeze_manifest(manifest_path, "0" * 64, "E02_optimized")

        svc = DetectionService()
        svc.load_model(ckpt, device="cpu", freeze_manifest_path=manifest_path)
        health = svc.get_health()
        assert health["status"] == "unavailable"
        assert str(tmp_path) not in health["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
