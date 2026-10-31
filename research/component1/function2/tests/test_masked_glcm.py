"""F2.1B1 Task 3 — scientific validation tests for the new mask-aware GLCM
entropy implementation (complexity/masked_glcm.py).

Numerical tolerances used in this file (documented once, here):
  - Determinism / identical-input-twice checks: exact equality (the
    algorithm is pure NumPy integer/float arithmetic with no RNG, so two
    runs on the same array must be bit-identical).
  - "Matches shared module when mask is fully valid" comparison: tight
    tolerance of 1e-9 absolute. Both implementations use the identical
    quantisation formula, identical (distance, angle) offsets, identical
    symmetrisation, and identical Shannon-entropy formula; when every pixel
    pair is valid (mask=all True) the two algorithms build IDENTICAL
    co-occurrence matrices, so any difference must be floating-point
    summation-order noise only, several orders of magnitude below 1e-9.
  - Degenerate flat-image entropy: pytest.approx(0.0, abs=1e-9) (same
    convention the existing F2.1A synthetic tests use for the shared
    module's flat-image case).
"""

from __future__ import annotations

import numpy as np
import pytest

from research.component1.complexity_calibration.src.complexity_features import (
    DEFAULT_CONFIG,
    compute_complexity_from_array,
)
from research.component1.function2.complexity.masked_glcm import (
    DEFAULT_MIN_VALID_PAIRS,
    compute_masked_glcm_entropy,
)


# ---------------------------------------------------------------------------
# Degenerate / flat cases
# ---------------------------------------------------------------------------

def test_uniform_background_has_zero_entropy():
    gray = np.full((64, 64), 128, dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)
    r = compute_masked_glcm_entropy(gray, mask)
    assert r.status == "valid"
    assert r.entropy == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Patterned background -> nonzero, sane entropy
# ---------------------------------------------------------------------------

def test_checkerboard_background_has_nonzero_sane_entropy():
    gray = np.zeros((64, 64), dtype=np.uint8)
    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            if (i // 8 + j // 8) % 2 == 0:
                gray[i:i + 8, j:j + 8] = 255
    mask = np.ones((64, 64), dtype=bool)
    r = compute_masked_glcm_entropy(gray, mask)
    assert r.status == "valid"
    assert r.entropy is not None
    assert 0.0 < r.entropy < 7.0  # log2(64 levels^2) = 12 is the theoretical max


# ---------------------------------------------------------------------------
# Core bug-fix regression test: excluded-region pixel values must have
# ZERO effect on the measured background entropy.
# ---------------------------------------------------------------------------

def test_excluded_high_texture_region_does_not_affect_background_entropy():
    rng = np.random.default_rng(0)
    gray = np.full((80, 80), 100, dtype=np.uint8)  # flat background
    mask = np.ones((80, 80), dtype=bool)
    mask[20:60, 20:60] = False  # exclude a 40x40 "defect" region

    # Fill the excluded region with high-texture noise (simulating a
    # genuine high-complexity defect pattern).
    gray_with_defect = gray.copy()
    gray_with_defect[20:60, 20:60] = rng.integers(0, 256, size=(40, 40), dtype=np.uint8)

    r_plain = compute_masked_glcm_entropy(gray, mask)
    r_with_defect = compute_masked_glcm_entropy(gray_with_defect, mask)

    assert r_plain.status == "valid"
    assert r_with_defect.status == "valid"
    # The flat background alone -> entropy 0 (no spatial variation at all
    # among valid pairs).
    assert r_plain.entropy == pytest.approx(0.0, abs=1e-9)
    # CRITICAL: changing pixel values ONLY inside the excluded region must
    # not change the computed background entropy AT ALL.
    assert r_with_defect.entropy == pytest.approx(r_plain.entropy, abs=1e-9)
    assert r_with_defect.valid_pair_count == r_plain.valid_pair_count


def test_changing_excluded_region_values_never_changes_entropy_param_sweep():
    """Same hard assertion as above, but repeated across several different
    excluded-region fill patterns, to rule out a coincidental match."""
    rng = np.random.default_rng(1)
    base_gray = rng.integers(40, 80, size=(80, 80), dtype=np.uint8)  # textured-but-moderate background
    mask = np.ones((80, 80), dtype=bool)
    mask[10:70, 15:55] = False

    baseline = compute_masked_glcm_entropy(base_gray, mask)
    assert baseline.status == "valid"

    for fill_value in (0, 255, 128):
        variant = base_gray.copy()
        variant[10:70, 15:55] = fill_value
        r = compute_masked_glcm_entropy(variant, mask)
        assert r.entropy == pytest.approx(baseline.entropy, abs=1e-9)
        assert r.valid_pair_count == baseline.valid_pair_count

    # Also try replacing the excluded region with high-frequency noise.
    variant_noise = base_gray.copy()
    variant_noise[10:70, 15:55] = rng.integers(0, 256, size=(60, 40), dtype=np.uint8)
    r_noise = compute_masked_glcm_entropy(variant_noise, mask)
    assert r_noise.entropy == pytest.approx(baseline.entropy, abs=1e-9)
    assert r_noise.valid_pair_count == baseline.valid_pair_count


# ---------------------------------------------------------------------------
# Non-square image + non-square / off-axis mask alignment
# ---------------------------------------------------------------------------

def test_non_square_image_and_mask_alignment():
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, size=(60, 140), dtype=np.uint8)  # H=60, W=140
    mask = np.ones((60, 140), dtype=bool)
    # Off-axis rectangular exclusion, not centred, not symmetric.
    mask[5:25, 90:135] = False

    r = compute_masked_glcm_entropy(gray, mask)
    assert r.status == "valid"
    assert r.entropy is not None and r.entropy > 0.0

    # Sanity: a shape mismatch between gray and mask must be rejected, not
    # silently broadcast or truncated.
    bad_mask = np.ones((60, 139), dtype=bool)
    r_bad = compute_masked_glcm_entropy(gray, bad_mask)
    assert r_bad.status == "error_invalid_input"
    assert r_bad.entropy is None


# ---------------------------------------------------------------------------
# Multiple excluded boxes -> union, no double counting
# ---------------------------------------------------------------------------

def test_multiple_excluded_regions_union_not_double_counted():
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 256, size=(100, 100), dtype=np.uint8)

    mask_two_boxes = np.ones((100, 100), dtype=bool)
    mask_two_boxes[10:30, 10:30] = False
    mask_two_boxes[60:80, 60:80] = False

    # Overlapping duplicate of one of the boxes added on top should not
    # change the result (this is what "union, no double counting" means at
    # the mask level -- the boolean mask itself has no concept of "twice
    # excluded", so applying the same exclusion again is idempotent).
    mask_two_boxes_dup = mask_two_boxes.copy()
    mask_two_boxes_dup[10:30, 10:30] = False  # already False; idempotent re-application

    r1 = compute_masked_glcm_entropy(gray, mask_two_boxes)
    r2 = compute_masked_glcm_entropy(gray, mask_two_boxes_dup)

    assert r1.status == "valid" and r2.status == "valid"
    assert r1.entropy == pytest.approx(r2.entropy, abs=1e-9)
    assert r1.valid_pair_count == r2.valid_pair_count

    # And the two-box result must differ from a single-box result (proving
    # the second box's exclusion actually had an effect, i.e. this test
    # would catch a union bug that silently dropped one of the boxes).
    mask_one_box = np.ones((100, 100), dtype=bool)
    mask_one_box[10:30, 10:30] = False
    r_one = compute_masked_glcm_entropy(gray, mask_one_box)
    assert r_one.valid_pair_count != r1.valid_pair_count


# ---------------------------------------------------------------------------
# Small / zero valid-region flagging
# ---------------------------------------------------------------------------

def test_very_small_valid_region_is_flagged_insufficient():
    rng = np.random.default_rng(4)
    gray = rng.integers(0, 256, size=(100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[:3, :3] = True  # tiny 3x3 valid patch -- well under DEFAULT_MIN_VALID_PAIRS

    r = compute_masked_glcm_entropy(gray, mask)
    assert r.status == "insufficient_pairs"
    assert r.valid_pair_count < DEFAULT_MIN_VALID_PAIRS
    assert r.entropy is not None  # value still reported, just flagged as unreliable
    assert any("insufficient" in f.lower() or "min_valid_pairs" in f for f in r.quality_flags)


def test_zero_valid_pairs_fully_excluded_mask_is_invalid_not_fabricated():
    rng = np.random.default_rng(5)
    gray = rng.integers(0, 256, size=(50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)  # fully excluded

    r = compute_masked_glcm_entropy(gray, mask)
    assert r.status == "all_excluded"
    assert r.entropy is None  # never a fabricated 0.0
    assert r.valid_pair_count == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_same_input_twice():
    rng = np.random.default_rng(6)
    gray = rng.integers(0, 256, size=(90, 90), dtype=np.uint8)
    mask = np.ones((90, 90), dtype=bool)
    mask[30:50, 30:50] = False

    r1 = compute_masked_glcm_entropy(gray, mask)
    r2 = compute_masked_glcm_entropy(gray, mask)
    assert r1.entropy == r2.entropy
    assert r1.valid_pair_count == r2.valid_pair_count
    assert r1.status == r2.status


# ---------------------------------------------------------------------------
# Comparison test: fully valid mask must match the shared module's
# (unmasked, equivalent-to-fully-valid) GLCM entropy computation.
# ---------------------------------------------------------------------------

def test_fully_valid_mask_matches_shared_module_entropy():
    rng = np.random.default_rng(7)
    img_bgr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    gray = __import__("cv2").cvtColor(img_bgr, __import__("cv2").COLOR_BGR2GRAY)
    mask_all_true = np.ones((96, 96), dtype=bool)

    shared_result = compute_complexity_from_array(img_bgr, config=DEFAULT_CONFIG, mask=None)
    masked_result = compute_masked_glcm_entropy(gray, mask_all_true)

    assert masked_result.status == "valid"
    assert masked_result.entropy == pytest.approx(shared_result.glcm_entropy, abs=1e-9)


def test_protocol_dict_records_reproducibility_metadata():
    gray = np.full((32, 32), 50, dtype=np.uint8)
    mask = np.ones((32, 32), dtype=bool)
    r = compute_masked_glcm_entropy(gray, mask)
    assert r.protocol["glcm_levels"] == 64
    assert r.protocol["glcm_distances"] == [1]
    assert len(r.protocol["glcm_angles"]) == 4
    assert "code_version" in r.protocol
    assert "min_valid_pairs" in r.protocol
