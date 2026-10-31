"""F2.1A Task 3/7 — verification tests for the shared complexity_features
module, using synthetic inputs with sanity-checkable expected behaviour.

These tests import and call the REAL shared module
(research/component1/complexity_calibration/src/complexity_features.py).
They do not reimplement its math, and they never modify that file.

Empirical reference values below were measured once against this exact
module (see the F2.1A verification report,
research/component1/function2/reports/task3_complexity_feature_verification.md)
and are asserted with generous tolerance — the point of these tests is to
catch regressions / unexpected behaviour changes in the shared module, not
to pin exact floating-point output.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from research.component1.complexity_calibration.src.complexity_features import (
    DEFAULT_CONFIG,
    compute_complexity,
    compute_complexity_from_array,
)


# ---------------------------------------------------------------------------
# Flat / constant image edge case
# ---------------------------------------------------------------------------

def test_flat_image_is_defined_not_nan(flat_gray_image):
    """A perfectly flat image must not raise, divide by zero, or produce NaN."""
    r = compute_complexity_from_array(flat_gray_image)
    assert r.status == "ok"
    assert r.luminance_variation is not None
    assert math.isfinite(r.luminance_variation)
    assert math.isfinite(r.glcm_entropy)
    assert math.isfinite(r.colour_variation)


def test_flat_image_has_zero_variation_and_zero_entropy(flat_gray_image):
    r = compute_complexity_from_array(flat_gray_image)
    assert r.luminance_variation == pytest.approx(0.0, abs=1e-9)
    assert r.colour_variation == pytest.approx(0.0, abs=1e-9)
    # Single GLCM bin with probability 1 -> Shannon entropy is exactly 0
    # (module returns -0.0 due to floating point sign; 0.0 == -0.0).
    assert r.glcm_entropy == pytest.approx(0.0, abs=1e-9)


def test_flat_image_sets_near_uniform_flag(flat_gray_image):
    r = compute_complexity_from_array(flat_gray_image)
    assert "near_uniform_luminance" in r.quality_flags


# ---------------------------------------------------------------------------
# Texture ordering sanity checks (flat < gradient/checkerboard < noise)
# ---------------------------------------------------------------------------

def test_entropy_ordering_flat_lt_checkerboard_lt_noise(
    flat_gray_image, checkerboard_image, random_noise_image
):
    r_flat = compute_complexity_from_array(flat_gray_image)
    r_cb = compute_complexity_from_array(checkerboard_image)
    r_noise = compute_complexity_from_array(random_noise_image)
    assert r_flat.glcm_entropy < r_cb.glcm_entropy < r_noise.glcm_entropy


def test_noise_image_has_high_colour_and_luminance_variation(random_noise_image):
    r = compute_complexity_from_array(random_noise_image)
    assert r.status == "ok"
    assert r.luminance_variation > 20.0
    assert r.colour_variation > 20.0


def test_grayscale_equal_channel_image_has_zero_colour_variation(gradient_image):
    """A BGR image with R==G==B everywhere (true grayscale content) must
    have ~zero a*/b* variation: luminance can vary, chroma should not."""
    r = compute_complexity_from_array(gradient_image)
    assert r.status == "ok"
    assert r.luminance_variation > 10.0  # it IS a gradient, so L varies
    assert r.colour_variation == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# BGR vs RGB channel-order handling
# ---------------------------------------------------------------------------

def test_uses_bgr_to_lab_not_rgb_to_lab():
    """The module's docstring and code call cv2.COLOR_BGR2LAB, assuming the
    caller passes BGR (OpenCV's native channel order). We verify this by
    constructing an image that is NOT symmetric under channel swap (pure
    blue in BGR order) and checking the module's luminance output matches
    a manual BGR2LAB conversion, not a manual RGB2LAB conversion."""
    img_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    img_bgr[:, :, 0] = 255  # pure blue in BGR order: (B=255, G=0, R=0)

    expected_lab_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    expected_lab_rgb_mistake = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2LAB).astype(np.float64)
    # Sanity: the two conversions must actually differ for this image,
    # otherwise the test would not discriminate anything.
    assert not np.allclose(expected_lab_bgr[:, :, 0], expected_lab_rgb_mistake[:, :, 0])

    r = compute_complexity_from_array(img_bgr)
    # Pure color, flat -> std should be 0 either way, so instead check L mean
    # indirectly is consistent with BGR2LAB by comparing a non-flat variant.
    gradient_channel = np.linspace(0, 255, 64).astype(np.uint8)
    img_bgr2 = np.zeros((64, 64, 3), dtype=np.uint8)
    img_bgr2[:, :, 0] = np.tile(gradient_channel, (64, 1))  # varying B channel only

    expected = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2LAB).astype(np.float64)
    expected_l_std = float(np.std(expected[:, :, 0]))

    result = compute_complexity_from_array(img_bgr2)
    assert result.luminance_variation == pytest.approx(expected_l_std, rel=1e-6)


# ---------------------------------------------------------------------------
# Resizing behaviour
# ---------------------------------------------------------------------------

def test_large_image_is_resized_preserving_aspect_ratio():
    rng = np.random.default_rng(7)
    big = rng.integers(0, 256, size=(1200, 900, 3), dtype=np.uint8)
    r = compute_complexity_from_array(big, config=DEFAULT_CONFIG)
    assert max(r.resized_height, r.resized_width) == DEFAULT_CONFIG.resize_long_edge
    # Aspect ratio preserved (900/1200 == resized_w/resized_h)
    assert r.resized_width / r.resized_height == pytest.approx(900 / 1200, rel=1e-2)


def test_small_image_below_long_edge_is_not_resized():
    rng = np.random.default_rng(7)
    small = rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
    r = compute_complexity_from_array(small, config=DEFAULT_CONFIG)
    assert (r.resized_height, r.resized_width) == (200, 300)


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------

def test_same_input_twice_gives_identical_output(random_noise_image):
    r1 = compute_complexity_from_array(random_noise_image)
    r2 = compute_complexity_from_array(random_noise_image)
    assert r1.to_dict() == r2.to_dict()


def test_output_is_native_python_float_dtype(flat_gray_image):
    r = compute_complexity_from_array(flat_gray_image)
    assert isinstance(r.luminance_variation, float)
    assert isinstance(r.glcm_entropy, float)
    assert isinstance(r.colour_variation, float)


# ---------------------------------------------------------------------------
# Invalid-input handling
# ---------------------------------------------------------------------------

def test_too_small_image_array_rejected():
    tiny = np.zeros((4, 4, 3), dtype=np.uint8)
    r = compute_complexity_from_array(tiny)
    assert r.status == "error_too_small"
    assert r.luminance_variation is None


def test_empty_array_rejected():
    r = compute_complexity_from_array(np.zeros((0, 0, 3), dtype=np.uint8))
    assert r.status == "error_empty_array"


def test_none_array_rejected():
    r = compute_complexity_from_array(None)
    assert r.status == "error_empty_array"


def test_nonexistent_file_path_rejected(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    r = compute_complexity(missing)
    assert r.status == "error_file_not_found"


def test_unreadable_file_rejected(tmp_path):
    bogus = tmp_path / "not_an_image.jpg"
    bogus.write_bytes(b"this is not image data")
    r = compute_complexity(bogus)
    assert r.status == "error_unreadable"


# ---------------------------------------------------------------------------
# Masked-region computation (used by the Task 4 background adapter)
# ---------------------------------------------------------------------------

def test_mask_restricts_statistics_to_valid_pixels():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=bool)
    mask[:, :50] = False  # exclude left half

    r = compute_complexity_from_array(img, mask=mask)
    assert r.status == "ok"
    assert r.valid_pixel_fraction == pytest.approx(0.5, abs=1e-6)


def test_mask_below_five_percent_is_rejected_as_insufficient():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[:3, :3] = True  # 9 / 10000 = 0.09% valid pixels

    r = compute_complexity_from_array(img, mask=mask)
    assert r.status == "error_insufficient_fabric"
    assert "insufficient_valid_pixels" in r.quality_flags
