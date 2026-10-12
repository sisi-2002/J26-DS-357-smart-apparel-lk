"""Component 1 detection service — wraps DetectionXAI with thread safety."""
import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DetectionService:
    """Singleton service managing the DetectionXAI model instance.

    All model calls are serialized via a threading lock because the model
    is not thread-safe. Callers in async context should use
    asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(self) -> None:
        self._xai = None
        self._lock = threading.Lock()
        self._loaded_at: Optional[datetime] = None
        self._model_version: Optional[str] = None
        self._device: Optional[str] = None
        self._weights_path: Optional[str] = None
        self._weights_hash: Optional[str] = None
        self._experiment_id: Optional[str] = None
        self._checkpoint_verified: bool = False
        self._load_error: Optional[str] = None

    def load_model(
        self,
        weights_path: str,
        device: str = "cpu",
        expected_sha256: Optional[str] = None,
        freeze_manifest_path: Optional[str] = None,
    ) -> None:
        """Load the detector, optionally verifying checkpoint identity first.

        If expected_sha256 or freeze_manifest_path is given and the
        checkpoint's hash does not match, the model is NOT loaded — this
        service never silently falls back to a different checkpoint.
        """
        self._load_error = None

        if not os.path.isfile(weights_path):
            self._load_error = "Model weights not found at configured path"
            logger.error("Model weights not found: %s", weights_path)
            return

        from app.components.component1.model_identity import (
            FreezeManifestError,
            verify_checkpoint,
        )

        try:
            result = verify_checkpoint(
                weights_path,
                expected_sha256=expected_sha256,
                freeze_manifest_path=freeze_manifest_path,
            )
        except FreezeManifestError as e:
            self._load_error = str(e)
            logger.exception("Freeze manifest error")
            return

        if not result.verified:
            self._load_error = result.detail
            logger.error("Checkpoint identity verification FAILED: %s", result.detail)
            return

        from app.components.component1.ml import DetectionXAI

        with self._lock:
            self._xai = DetectionXAI(weights_path, device=device)
            self._loaded_at = datetime.now(timezone.utc)
            self._model_version = self._xai._model_version
            self._device = device
            self._weights_path = weights_path
            self._weights_hash = f"sha256:{result.actual_sha256}"
            self._checkpoint_verified = result.expected is not None
            self._experiment_id = result.expected.experiment_id if result.expected else None
            logger.info(
                "Model loaded: %s on %s (checkpoint_verified=%s, experiment_id=%s)",
                self._model_version, device, self._checkpoint_verified, self._experiment_id,
            )

    @property
    def is_loaded(self) -> bool:
        return self._xai is not None

    @property
    def model_version(self) -> Optional[str]:
        return self._model_version

    @property
    def weights_hash(self) -> Optional[str]:
        return self._weights_hash

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def experiment_id(self) -> Optional[str]:
        return self._experiment_id

    @property
    def checkpoint_verified(self) -> bool:
        return self._checkpoint_verified

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def get_health(self) -> dict:
        import torch
        import ultralytics
        from app.components.component1.ml import CLASS_NAMES

        if not self.is_loaded:
            return {
                "status": "unavailable",
                "detail": self._load_error or "Model not loaded",
            }
        return {
            "status": "healthy",
            "model_version": self._model_version,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "device": self._device,
            "classes": list(CLASS_NAMES),
            "loaded_at": self._loaded_at,
            "experiment_id": self._experiment_id,
            "checkpoint_verified": self._checkpoint_verified,
        }

    def _detect_sync(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ):
        from app.components.component1.ml import InferenceResult
        with self._lock:
            t0 = time.perf_counter()
            result = self._xai.detect(
                image_bgr,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return result, elapsed_ms

    async def run_detection(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ):
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return await asyncio.to_thread(
            self._detect_sync, image_bgr, conf_threshold, iou_threshold
        )

    def _explain_sync(
        self,
        image_bgr: np.ndarray,
        detection,
        method: str,
        conf_threshold: float,
        iou_threshold: float,
    ):
        with self._lock:
            t0 = time.perf_counter()
            result = self._xai.explain(
                image_bgr,
                detection,
                method=method,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return result, elapsed_ms

    async def run_explanation(
        self,
        image_bgr: np.ndarray,
        detection,
        method: str = "gradcam",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ):
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return await asyncio.to_thread(
            self._explain_sync, image_bgr, detection, method,
            conf_threshold, iou_threshold,
        )


detection_service = DetectionService()
