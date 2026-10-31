"""F2.1B1 Task 2/3 — tests for the consistent 3-feature background adapter
(complexity/background_features.py).

These tests check that the adapter:
  - uses one consistent mask across all three features,
  - correctly propagates background_region's state machine
    (unresolved / clean_negative / boxes_excluded / insufficient_background),
  - applies and reports the boundary-erosion mitigation only when there is
    an actual exclusion boundary to protect, and
  - produces a mask-aware glcm_entropy that genuinely differs from a naive
    unmasked computation when a high-texture region is excluded.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from research.component1.complexity_calibration.src.complexity_features import (
    DEFAULT_CONFIG,
    compute_complexity_from_array,
)
from research.component1.function2.complexity.background_features import (
    compute_background_features,
)


def _solid_image(h, w, value=120):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_unresolved_annotation_never_computes_features():
    img = _solid_image(200, 200)
    r = compute_background_features(img, yolo_boxes=None, is_negative=None)
    assert r.status == "error_unresolved_missing_annotation"
    assert r.luminance_variation is None
    assert r.glcm_entropy is None
    assert r.colour_variation is None


def test_clean_negative_uses_full_image_no_erosion():
    img = _solid_image(200, 200)
    r = compute_background_features(img, yolo_boxes=[], is_negative=True)
    assert r.background_status == "clean_negative"
    assert r.status == "ok"
    assert r.boundary_erosion_px == 0  # no exclusion boundary -> erosion skipped
    assert r.glcm_entropy == pytest.approx(0.0, abs=1e-9)


def test_small_valid_region_flagged_through_to_overall_status():
    rng = np.random.default_rng(10)
    img = rng.integers(0, 256, size=(300, 300, 3), dtype=np.uint8)
    # A box covering ~95% of the image -> insufficient_background per the
    # background_region 70% rule, and almost certainly < 5% valid pixels
    # for the shared feature module's own guard too.
    boxes = [(0.5, 0.5, 0.97, 0.97)]
    r = compute_background_features(img, yolo_boxes=boxes, is_negative=False)
    assert r.background_status == "insufficient_background"
    # Either the shared module's 5% guard or our own min-pairs guard (or
    # both) should surface as a non-"ok" status -- never silently "ok".
    assert r.status != "ok"


def test_excluding_high_texture_region_changes_glcm_entropy_vs_unmasked():
    """Demonstrates the actual bug fix end-to-end: the OLD (shared module,
    unmasked) glcm_entropy ignores the exclusion and is computed over the
    whole image; the NEW adapter's glcm_entropy genuinely differs because
    it excludes the high-texture region."""
    rng = np.random.default_rng(11)
    img = _solid_image(256, 256, value=100)
    # Make the right half of the image (which we'll fully exclude via a
    # GT box) high-texture noise; the left half stays a flat background.
    img[:, 128:] = rng.integers(0, 256, size=(256, 128, 3), dtype=np.uint8)

    # GT box covering the noisy right half (cx=0.75, cy=0.5, w=0.5, h=1.0).
    boxes = [(0.75, 0.5, 0.5, 1.0)]

    old_unmasked = compute_complexity_from_array(img, config=DEFAULT_CONFIG, mask=None)
    new_masked = compute_background_features(img, yolo_boxes=boxes, is_negative=False)

    assert new_masked.background_status == "boxes_excluded"
    assert new_masked.glcm_entropy is not None
    # The old, buggy, whole-image entropy must be clearly higher than the
    # new background-only entropy, since the whole image includes the
    # excluded noisy half and the new value does not.
    assert old_unmasked.glcm_entropy > new_masked.glcm_entropy
    # The corrected background-only entropy of an almost-flat region
    # should be low.
    assert new_masked.glcm_entropy < 2.0


def test_determinism_same_input_twice():
    rng = np.random.default_rng(12)
    img = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
    boxes = [(0.3, 0.3, 0.2, 0.2), (0.7, 0.6, 0.15, 0.25)]
    r1 = compute_background_features(img, yolo_boxes=boxes, is_negative=False)
    r2 = compute_background_features(img, yolo_boxes=boxes, is_negative=False)
    assert r1.to_dict() == r2.to_dict()


def test_non_square_image_handled():
    rng = np.random.default_rng(13)
    img = rng.integers(0, 256, size=(300, 700, 3), dtype=np.uint8)  # wide image
    boxes = [(0.2, 0.5, 0.1, 0.3)]
    r = compute_background_features(img, yolo_boxes=boxes, is_negative=False)
    assert r.status in ("ok", "ok_glcm_insufficient_pairs")
    assert r.resized_width == 512  # long edge (700) resized to 512
    assert r.resized_height == pytest.approx(300 * 512 / 700, abs=1)
