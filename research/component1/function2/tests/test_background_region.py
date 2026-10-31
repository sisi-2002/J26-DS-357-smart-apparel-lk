"""F2.1A Task 4/7 — tests for the new background-region selection adapter.

research/component1/function2/complexity/background_region.py is new code
(not a shared/frozen module), so these tests verify its own logic directly,
including coordinate alignment against hand-computed expected pixel ranges.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.component1.function2.complexity.background_region import (
    DEFAULT_MAX_DEFECT_COVERAGE,
    compute_background_mask,
    yolo_box_to_pixel_xyxy,
)


def test_missing_annotation_is_unresolved_not_clean():
    """Per the Task 4 design decision: an image with yolo_boxes=None (label
    file missing / unavailable) must NEVER be treated as a clean background."""
    result = compute_background_mask(image_height=100, image_width=100, yolo_boxes=None)
    assert result.status == "unresolved_missing_annotation"
    assert result.mask is None
    assert result.background_coverage_fraction == 0.0


def test_confirmed_negative_image_is_full_background():
    result = compute_background_mask(
        image_height=50, image_width=50, yolo_boxes=[], is_negative=True
    )
    assert result.status == "clean_negative"
    assert result.mask is not None
    assert result.mask.all()
    assert result.background_coverage_fraction == pytest.approx(1.0)


def test_empty_boxes_without_negative_flag_is_flagged():
    result = compute_background_mask(
        image_height=50, image_width=50, yolo_boxes=[], is_negative=False
    )
    assert result.status == "clean_no_boxes_provided"
    assert any("double-checked" in w for w in result.warnings)


def test_coordinate_alignment_single_centered_box():
    """A single box at cx=cy=0.5, w=h=0.2 on a 100x100 image should exclude
    pixel rows/cols [40, 60) exactly (hand-computed)."""
    result = compute_background_mask(
        image_height=100, image_width=100,
        yolo_boxes=[(0.5, 0.5, 0.2, 0.2)],
    )
    assert result.status == "boxes_excluded"
    mask = result.mask
    assert mask is not None

    # Inside the box: must be excluded (False)
    assert not mask[50, 50]
    assert not mask[41, 41]
    assert not mask[59, 59]
    # Just outside the box: must remain background (True)
    assert mask[39, 50]
    assert mask[60, 50]
    assert mask[50, 39]
    assert mask[50, 60]

    expected_coverage = 1.0 - (20 * 20) / (100 * 100)
    assert result.background_coverage_fraction == pytest.approx(expected_coverage, abs=1e-3)


def test_yolo_box_to_pixel_xyxy_matches_hand_computation():
    x1, y1, x2, y2 = yolo_box_to_pixel_xyxy((0.5, 0.5, 0.2, 0.2), img_w=100, img_h=100)
    assert (x1, y1, x2, y2) == pytest.approx((40.0, 40.0, 60.0, 60.0))


def test_box_clipped_to_image_bounds():
    """A box that nominally extends past the image edge must be clipped,
    not produce negative indices or an exception."""
    x1, y1, x2, y2 = yolo_box_to_pixel_xyxy((0.0, 0.0, 0.5, 0.5), img_w=100, img_h=100)
    assert x1 == 0.0 and y1 == 0.0


def test_large_box_triggers_insufficient_background():
    """A box covering ~90% of the image must flag insufficient_background
    under the default 70% max-defect-coverage threshold."""
    result = compute_background_mask(
        image_height=100, image_width=100,
        yolo_boxes=[(0.5, 0.5, 0.95, 0.95)],
    )
    assert result.status == "insufficient_background"
    assert result.background_coverage_fraction < (1.0 - DEFAULT_MAX_DEFECT_COVERAGE)


def test_multiple_boxes_are_unioned_not_double_counted():
    """Two overlapping boxes should not reduce coverage below what the
    unioned excluded area implies (raster mask naturally unions)."""
    result = compute_background_mask(
        image_height=100, image_width=100,
        yolo_boxes=[(0.5, 0.5, 0.4, 0.4), (0.5, 0.5, 0.4, 0.4)],  # identical, fully overlapping
    )
    single = compute_background_mask(
        image_height=100, image_width=100,
        yolo_boxes=[(0.5, 0.5, 0.4, 0.4)],
    )
    assert result.background_coverage_fraction == pytest.approx(
        single.background_coverage_fraction
    )
    assert result.num_boxes_excluded == 2  # both boxes counted, even though they overlap


def test_invalid_dimensions_rejected():
    result = compute_background_mask(image_height=0, image_width=100, yolo_boxes=[])
    assert result.status == "error_invalid_input"
    assert result.mask is None


def test_determinism_same_input_same_mask():
    boxes = [(0.3, 0.3, 0.1, 0.1), (0.7, 0.6, 0.2, 0.15)]
    r1 = compute_background_mask(image_height=120, image_width=200, yolo_boxes=boxes)
    r2 = compute_background_mask(image_height=120, image_width=200, yolo_boxes=boxes)
    assert np.array_equal(r1.mask, r2.mask)
    assert r1.background_coverage_fraction == r2.background_coverage_fraction
    assert r1.status == r2.status


def test_to_dict_excludes_mask_array():
    result = compute_background_mask(
        image_height=50, image_width=50, yolo_boxes=[], is_negative=True
    )
    d = result.to_dict()
    assert "mask" not in d
    assert d["status"] == "clean_negative"
