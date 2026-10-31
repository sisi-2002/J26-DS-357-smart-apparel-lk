"""F2.4-B Task 3 / Task 14 regression tests for ``evaluation/stability_v2.py``
(the degenerate-low-information-heatmap audit fix)."""
from __future__ import annotations

import numpy as np
import pytest

from research.component1.function2.evaluation import stability_v2 as sv2


def test_informative_fraction_all_zero_is_zero():
    hm = np.zeros((10, 10), dtype=np.float32)
    assert sv2.informative_pixel_fraction(hm) == 0.0
    assert sv2.is_low_information_heatmap(hm)


def test_informative_fraction_uniform_nonzero_is_one():
    hm = np.full((10, 10), 5.0, dtype=np.float32)
    assert sv2.informative_pixel_fraction(hm) == pytest.approx(1.0)
    assert not sv2.is_low_information_heatmap(hm)


def test_tiny_blob_on_flat_field_is_flagged_low_information():
    """Reproduces the exact failure mode the audit found: a heatmap that's
    >99% near-zero with a tiny nonzero blob."""
    hm = np.zeros((100, 100), dtype=np.float32)
    hm[50, 50] = 1.0  # 1 pixel out of 10000 = 0.01% informative
    assert sv2.informative_pixel_fraction(hm) < 0.01
    assert sv2.is_low_information_heatmap(hm)


def test_moderately_sparse_heatmap_not_flagged():
    """A heatmap with a reasonably sized (not pathological) activation
    region must NOT be flagged -- the threshold should not be so aggressive
    it swallows ordinary sparse-but-meaningful Grad-CAM output."""
    hm = np.zeros((100, 100), dtype=np.float32)
    hm[30:50, 30:50] = 1.0  # 400/10000 = 4% informative
    assert not sv2.is_low_information_heatmap(hm)


def test_degenerate_guard_reported_as_distinct_status_not_silently_valid():
    """A low-information original heatmap must short-circuit BEFORE any
    perturbation/re-detection work and be reported with its own status,
    never silently folded into 'valid'."""
    from unittest.mock import MagicMock
    hm = np.zeros((50, 50), dtype=np.float32)
    hm[0, 0] = 1.0
    result = sv2.evaluate_stability_one_perturbation_v2(
        xai=MagicMock(), explain_fn=MagicMock(), image_bgr=np.zeros((50, 50, 3), dtype=np.uint8),
        original_heatmap=hm, class_id=0, original_matched_bbox_xyxy=(0, 0, 10, 10),
        perturbation_name="brightness_plus5pct", gt_instance_id="synthetic__inst000", method="gradcam",
    )
    assert result.status == "degenerate_low_information"
    assert result.similarity is None
    assert result.original_informative_fraction is not None
    assert result.original_informative_fraction < 0.01


def test_v1_v2_identical_on_non_degenerate_case_shape_only():
    """Sanity: v2's dataclass carries every v1 field plus two new ones --
    confirm no v1 field was dropped (so v2 can be reported alongside v1
    without losing information)."""
    import dataclasses
    v1_fields = {"gt_instance_id", "method", "perturbation", "status",
                 "matched_iou_vs_original", "similarity", "reason"}
    v2_fields = {f.name for f in dataclasses.fields(sv2.StabilityResultV2)}
    assert v1_fields.issubset(v2_fields)
