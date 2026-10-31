"""
Fabric visual-complexity feature extraction.

Reusable module — designed for import by both research scripts and the
FastAPI backend (backend/app/components/component1/service.py).

Three baseline features measured on the fabric image:
  1. Luminance variation  (σ of L channel in CIE L*a*b*)
  2. GLCM entropy         (Shannon entropy of GLCM, base-2)
  3. Colour variation     (mean σ across a* and b* channels in CIE L*a*b*)

See measurement_protocol.md for mathematical definitions, limitations
and provisional parameter choices.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Protocol configuration (provisional — freeze via calibration.json later)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ProtocolConfig:
    resize_long_edge: int = 512
    interpolation: str = "INTER_AREA"
    glcm_levels: int = 64
    glcm_distances: tuple[int, ...] = (1,)
    glcm_angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    glcm_symmetric: bool = True
    glcm_normed: bool = True
    entropy_log_base: int = 2
    protocol_version: str = "proto_v0.1_provisional"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["glcm_distances"] = list(d["glcm_distances"])
        d["glcm_angles"] = [round(a, 6) for a in d["glcm_angles"]]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProtocolConfig":
        d = dict(d)
        d["glcm_distances"] = tuple(d["glcm_distances"])
        d["glcm_angles"] = tuple(d["glcm_angles"])
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> "ProtocolConfig":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data.get("protocol", data))


DEFAULT_CONFIG = ProtocolConfig()


# ---------------------------------------------------------------------------
# Feature result
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ComplexityResult:
    luminance_variation: Optional[float] = None
    glcm_entropy: Optional[float] = None
    colour_variation: Optional[float] = None
    status: str = "ok"
    quality_flags: list[str] = dataclasses.field(default_factory=list)
    protocol_version: str = ""
    image_height: int = 0
    image_width: int = 0
    resized_height: int = 0
    resized_width: int = 0
    valid_pixel_fraction: float = 1.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resize_preserve_aspect(img: np.ndarray, long_edge: int,
                            interp: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    scale = long_edge / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


_INTERP_MAP = {
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_NEAREST": cv2.INTER_NEAREST,
}


def _compute_glcm(gray: np.ndarray, cfg: ProtocolConfig) -> np.ndarray:
    """Compute symmetric, normalised GLCM.

    Uses a direct NumPy implementation to avoid the skimage dependency
    in production.  For each (distance, angle) pair the co-occurrence
    matrix is accumulated, symmetrised, then normalised to a joint
    probability.
    """
    levels = cfg.glcm_levels
    # Quantise to [0, levels)
    g = (gray.astype(np.float64) / 256.0 * levels).astype(np.int32)
    g = np.clip(g, 0, levels - 1)

    glcm_sum = np.zeros((levels, levels), dtype=np.float64)

    for d in cfg.glcm_distances:
        for angle in cfg.glcm_angles:
            dx = round(d * math.cos(angle))
            dy = -round(d * math.sin(angle))

            h, w = g.shape
            r_lo = max(0, -dy)
            r_hi = min(h, h - dy)
            c_lo = max(0, -dx)
            c_hi = min(w, w - dx)

            if r_hi <= r_lo or c_hi <= c_lo:
                continue

            ref = g[r_lo:r_hi, c_lo:c_hi]
            neigh = g[r_lo + dy:r_hi + dy, c_lo + dx:c_hi + dx]

            mat = np.zeros((levels, levels), dtype=np.float64)
            np.add.at(mat, (ref.ravel(), neigh.ravel()), 1)

            if cfg.glcm_symmetric:
                mat = mat + mat.T

            glcm_sum += mat

    total = glcm_sum.sum()
    if total > 0:
        glcm_sum /= total

    return glcm_sum


def _shannon_entropy(p: np.ndarray, log_base: int) -> float:
    """Shannon entropy H = -Σ p·log(p), ignoring zero entries."""
    mask = p > 0
    if not mask.any():
        return 0.0
    vals = p[mask]
    log_vals = np.log(vals) / math.log(log_base)
    return float(-np.sum(vals * log_vals))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_complexity(
    image_path: str | Path,
    config: ProtocolConfig = DEFAULT_CONFIG,
    mask: Optional[np.ndarray] = None,
) -> ComplexityResult:
    """Compute fabric visual-complexity features for a single image.

    Parameters
    ----------
    image_path : path to the image file (JPEG/PNG).
    config : measurement protocol settings.
    mask : optional boolean mask (True = valid fabric pixel).  If None,
           all pixels are treated as fabric.  Applied after resize.

    Returns
    -------
    ComplexityResult with raw features and quality flags.
    """
    result = ComplexityResult(protocol_version=config.protocol_version)
    flags = result.quality_flags

    # --- Load ---
    path = Path(image_path)
    if not path.is_file():
        result.status = "error_file_not_found"
        flags.append("file_not_found")
        return result

    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        result.status = "error_unreadable"
        flags.append("unreadable_image")
        return result

    h, w = img_bgr.shape[:2]
    result.image_height, result.image_width = h, w

    if h < 8 or w < 8:
        result.status = "error_too_small"
        flags.append("image_too_small")
        return result

    # --- Resize (preserve aspect ratio) ---
    interp = _INTERP_MAP.get(config.interpolation, cv2.INTER_AREA)
    img_bgr = _resize_preserve_aspect(img_bgr, config.resize_long_edge, interp)
    rh, rw = img_bgr.shape[:2]
    result.resized_height, result.resized_width = rh, rw

    # --- Mask handling ---
    if mask is not None:
        if mask.shape[:2] != (rh, rw):
            mask = cv2.resize(mask.astype(np.uint8), (rw, rh),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask = np.ones((rh, rw), dtype=bool)

    valid_frac = mask.sum() / mask.size
    result.valid_pixel_fraction = float(valid_frac)
    if valid_frac < 0.05:
        result.status = "error_insufficient_fabric"
        flags.append("insufficient_valid_pixels")
        return result

    if valid_frac < 0.50:
        flags.append("low_valid_pixel_fraction")

    # --- Convert to CIE L*a*b* ---
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    # OpenCV LAB ranges: L [0,255] mapped from [0,100], a/b [0,255] centred at 128
    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]

    L_valid = L[mask]
    a_valid = a[mask]
    b_valid = b[mask]

    # --- 1. Luminance variation: σ(L) ---
    lum_std = float(np.std(L_valid))
    result.luminance_variation = lum_std

    if lum_std < 0.5:
        flags.append("near_uniform_luminance")

    # --- 2. GLCM entropy ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None and valid_frac < 1.0:
        # For masked images, zero out non-fabric to avoid boundary artefacts.
        # Flag this limitation.
        gray_masked = gray.copy()
        gray_masked[~mask] = 0
        flags.append("glcm_includes_masked_zeros")
    else:
        gray_masked = gray

    glcm = _compute_glcm(gray_masked, config)
    ent = _shannon_entropy(glcm, config.entropy_log_base)
    result.glcm_entropy = ent

    # --- 3. Colour variation: mean(σ(a*), σ(b*)) ---
    std_a = float(np.std(a_valid))
    std_b = float(np.std(b_valid))
    result.colour_variation = (std_a + std_b) / 2.0

    # --- Quality checks ---
    for val in [result.luminance_variation, result.glcm_entropy,
                result.colour_variation]:
        if val is not None and not math.isfinite(val):
            result.status = "error_non_finite"
            flags.append("non_finite_value")
            break

    return result


def compute_complexity_from_array(
    img_bgr: np.ndarray,
    config: ProtocolConfig = DEFAULT_CONFIG,
    mask: Optional[np.ndarray] = None,
) -> ComplexityResult:
    """Same as compute_complexity but accepts an in-memory BGR array."""
    result = ComplexityResult(protocol_version=config.protocol_version)
    flags = result.quality_flags

    if img_bgr is None or img_bgr.size == 0:
        result.status = "error_empty_array"
        flags.append("empty_array")
        return result

    h, w = img_bgr.shape[:2]
    result.image_height, result.image_width = h, w

    if h < 8 or w < 8:
        result.status = "error_too_small"
        flags.append("image_too_small")
        return result

    interp = _INTERP_MAP.get(config.interpolation, cv2.INTER_AREA)
    img_bgr = _resize_preserve_aspect(img_bgr, config.resize_long_edge, interp)
    rh, rw = img_bgr.shape[:2]
    result.resized_height, result.resized_width = rh, rw

    if mask is not None:
        if mask.shape[:2] != (rh, rw):
            mask = cv2.resize(mask.astype(np.uint8), (rw, rh),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask = np.ones((rh, rw), dtype=bool)

    valid_frac = mask.sum() / mask.size
    result.valid_pixel_fraction = float(valid_frac)
    if valid_frac < 0.05:
        result.status = "error_insufficient_fabric"
        flags.append("insufficient_valid_pixels")
        return result

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]

    L_valid = L[mask]
    a_valid = a[mask]
    b_valid = b[mask]

    result.luminance_variation = float(np.std(L_valid))
    if result.luminance_variation < 0.5:
        flags.append("near_uniform_luminance")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    glcm = _compute_glcm(gray, config)
    result.glcm_entropy = _shannon_entropy(glcm, config.entropy_log_base)

    std_a = float(np.std(a_valid))
    std_b = float(np.std(b_valid))
    result.colour_variation = (std_a + std_b) / 2.0

    for val in [result.luminance_variation, result.glcm_entropy,
                result.colour_variation]:
        if val is not None and not math.isfinite(val):
            result.status = "error_non_finite"
            flags.append("non_finite_value")
            break

    return result
