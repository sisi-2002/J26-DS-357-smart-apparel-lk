"""Image storage: validation, save, thumbnail generation, artifact serving."""
import hashlib
import io
import logging
import os
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
}

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PIXELS = 50_000_000  # 50 megapixels
THUMBNAIL_SIZE = (256, 256)


class ImageValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def detect_content_type(data: bytes) -> str | None:
    for magic, ctype in ALLOWED_CONTENT_TYPES.items():
        if data[:len(magic)] == magic:
            return ctype
    return None


def validate_image_bytes(data: bytes) -> tuple[str, np.ndarray]:
    if not data:
        raise ImageValidationError("Empty file", status_code=422)

    if len(data) > MAX_IMAGE_BYTES:
        raise ImageValidationError(
            f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit",
            status_code=400,
        )

    content_type = detect_content_type(data)
    if content_type is None:
        raise ImageValidationError(
            "Unsupported format. Use JPEG or PNG",
            status_code=400,
        )

    arr = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ImageValidationError("Image could not be decoded", status_code=422)

    h, w = image_bgr.shape[:2]
    if h * w > MAX_PIXELS:
        raise ImageValidationError(
            f"Image exceeds {MAX_PIXELS // 1_000_000} megapixel limit",
            status_code=400,
        )

    return content_type, image_bgr


def compute_hash(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


class ImageStorage:
    def __init__(self, base_dir: str):
        self._base = Path(base_dir)

    def _inspection_dir(self, inspection_id: uuid.UUID) -> Path:
        return self._base / "inspections" / str(inspection_id)

    def save_original(self, inspection_id: uuid.UUID, data: bytes, content_type: str) -> str:
        ext = ".jpg" if "jpeg" in content_type else ".png"
        d = self._inspection_dir(inspection_id)
        d.mkdir(parents=True, exist_ok=True)
        filename = f"original{ext}"
        path = d / filename
        path.write_bytes(data)
        # as_posix(): these paths become URL segments — must use forward
        # slashes regardless of host OS (Path.__str__() uses the native
        # separator, which is backslash on Windows and breaks on a Linux
        # deployment even though it happens to work in most browsers).
        return path.relative_to(self._base).as_posix()

    def save_thumbnail(self, inspection_id: uuid.UUID, image_bgr: np.ndarray) -> str:
        d = self._inspection_dir(inspection_id)
        d.mkdir(parents=True, exist_ok=True)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        pil.thumbnail(THUMBNAIL_SIZE, PILImage.Resampling.LANCZOS)
        path = d / "thumbnail.jpg"
        pil.save(path, "JPEG", quality=85)
        return path.relative_to(self._base).as_posix()

    def cleanup_inspection(self, inspection_id: uuid.UUID) -> None:
        d = self._inspection_dir(inspection_id)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    def load_original_bytes(self, inspection_id: uuid.UUID) -> bytes | None:
        d = self._inspection_dir(inspection_id)
        for ext in (".jpg", ".png"):
            p = d / f"original{ext}"
            if p.is_file():
                return p.read_bytes()
        return None

    def save_overlay(self, inspection_id: uuid.UUID, explanation_id: uuid.UUID, overlay_bgr: np.ndarray) -> str:
        d = self._inspection_dir(inspection_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"xai_{explanation_id}.png"
        cv2.imwrite(str(path), overlay_bgr)
        return path.relative_to(self._base).as_posix()

    def save_raw_heatmap(self, inspection_id: uuid.UUID, explanation_id: uuid.UUID, heatmap: np.ndarray) -> str:
        d = self._inspection_dir(inspection_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"xai_{explanation_id}_raw.npy"
        np.save(str(path), heatmap)
        return path.relative_to(self._base).as_posix()

    def remove_explanation_artifacts(self, inspection_id: uuid.UUID, explanation_id: uuid.UUID) -> None:
        d = self._inspection_dir(inspection_id)
        for pattern in (f"xai_{explanation_id}.*",):
            for f in d.glob(pattern):
                f.unlink(missing_ok=True)

    def resolve_artifact_path(self, relative_path: str) -> Path | None:
        full = (self._base / relative_path).resolve()
        if not str(full).startswith(str(self._base.resolve())):
            return None
        if not full.is_file():
            return None
        return full
