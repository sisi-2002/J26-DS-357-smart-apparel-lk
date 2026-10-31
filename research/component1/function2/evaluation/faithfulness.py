"""F2.4-A Task 4 -- faithfulness metrics (insertion / deletion curves).

IMPORTANT SCOPE NOTE (restated at every call site that matters): insertion
and deletion AUC are NOT proof of causal explanation quality. They measure
whether masking pixels the heatmap calls "important" changes the SAME
frozen detection score in the expected direction, under one particular
perturbation/baseline scheme. A method can score well here while still
misattributing importance to spatially-correlated-but-not-causal pixels
(e.g. texture), and a low score does not certify unfaithfulness -- it is one
diagnostic signal, not a ground truth label.

Mechanism (critical -- see ``common.frozen_target_score``): every step
re-evaluates the model on a perturbed image but reads the detection score at
the SAME frozen ``(class_id, pre_nms_index)`` cell that F2.3-B's
``explain()`` call originally resolved and verified for this instance. NMS
and GT-rematching are never re-run mid-curve -- doing so would silently let
the "target" drift to a different detection as pixels are perturbed, which
would make the resulting curve meaningless (it would no longer describe one
instance's explanation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from research.component1.function2.evaluation.common import (
    frozen_target_score, gaussian_blur_baseline,
)

N_STEPS = 10  # fixed perturbation budget, identical for insertion, deletion,
              # the random control curve, and both methods (Task 4: "using
              # the same perturbation budget/step count for both methods")
RANDOM_SEED = 357  # reuses F2.2B's sampling seed per the frozen protocol


@dataclass
class CurveResult:
    status: str  # "valid", "target_invalid_shape", "nan_encountered"
    order: str  # "most_important_first"
    scores: list = field(default_factory=list)       # len N_STEPS+1 (step 0 = unperturbed start state)
    fractions_revealed_or_removed: list = field(default_factory=list)
    auc: Optional[float] = None
    first_invalid_step: Optional[int] = None


def _ranked_pixel_order(heatmap: np.ndarray) -> np.ndarray:
    """Flat pixel indices (row-major) sorted by DESCENDING heatmap value --
    "most important first". Ties broken by lowest flat index (stable sort),
    matching the same deterministic row-major tie convention used in
    ``localization.pointing_accuracy``."""
    flat = heatmap.ravel()
    # stable sort ascending, then reverse -> ties keep original (row-major) order among equal values
    order_asc = np.argsort(flat, kind="mergesort")
    return order_asc[::-1]


def _random_pixel_order(n_pixels: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    order = np.arange(n_pixels)
    rng.shuffle(order)
    return order


def _step_masks(order: np.ndarray, n_steps: int):
    """n_steps cumulative pixel-count boundaries, evenly spaced (each step
    reveals/removes an additional 1/n_steps of all pixels, in `order`)."""
    n = len(order)
    boundaries = [round(n * k / n_steps) for k in range(n_steps + 1)]
    return boundaries


def _run_curve(xai, image_bgr, baseline_bgr, class_id, pre_nms_index,
                pixel_order: np.ndarray, mode: str) -> CurveResult:
    """mode: 'insertion' (start at baseline, reveal original pixels in
    `pixel_order`) or 'deletion' (start at original, replace with baseline
    pixels in `pixel_order`)."""
    h, w = image_bgr.shape[:2]
    boundaries = _step_masks(pixel_order, N_STEPS)
    scores, fractions = [], []
    status = "valid"
    first_invalid = None

    flat_orig = image_bgr.reshape(-1, image_bgr.shape[2]).copy()
    flat_base = baseline_bgr.reshape(-1, image_bgr.shape[2]).copy()

    for step, n_applied in enumerate(boundaries):
        if mode == "insertion":
            canvas = flat_base.copy()
            idx = pixel_order[:n_applied]
            canvas[idx] = flat_orig[idx]
        else:
            canvas = flat_orig.copy()
            idx = pixel_order[:n_applied]
            canvas[idx] = flat_base[idx]
        img_step = canvas.reshape(h, w, image_bgr.shape[2])

        result = frozen_target_score(xai, img_step, class_id, pre_nms_index)
        if result["status"] != "valid":
            status = result["status"]
            first_invalid = step
            break
        scores.append(result["score"])
        fractions.append(n_applied / len(pixel_order))

    auc = float(np.trapezoid(scores, fractions)) if len(scores) >= 2 else None
    return CurveResult(status=status, order="most_important_first",
                        scores=scores, fractions_revealed_or_removed=fractions,
                        auc=auc, first_invalid_step=first_invalid)


def _run_random_curve(xai, image_bgr, baseline_bgr, class_id, pre_nms_index,
                       n_pixels: int, seed: int) -> CurveResult:
    order = _random_pixel_order(n_pixels, seed)
    return _run_curve(xai, image_bgr, baseline_bgr, class_id, pre_nms_index,
                       order, mode="insertion")


@dataclass
class FaithfulnessResult:
    gt_instance_id: str
    method: str
    insertion: CurveResult
    deletion: CurveResult
    random_control_insertion: CurveResult
    baseline: str = "gaussian_blur_ksize31"
    n_steps: int = N_STEPS


def evaluate_faithfulness(xai, image_bgr: np.ndarray, heatmap: np.ndarray,
                           class_id: int, pre_nms_index: int,
                           gt_instance_id: str, method: str,
                           random_seed: int = RANDOM_SEED) -> FaithfulnessResult:
    """Insertion + deletion + random-control curves for one (instance,
    method) pair, on a FIXED, shared Gaussian-blur baseline and a FIXED
    pixel order per curve. Non-finite / degenerate heatmaps still run (the
    pixel order degenerates to row-major arbitrary order; `CurveResult`
    records no special status for this case since it is a property of the
    heatmap, already reported by Task 3's localization metrics -- a reader
    cross-references the two)."""
    baseline_bgr = gaussian_blur_baseline(image_bgr)
    pixel_order = _ranked_pixel_order(heatmap)

    insertion = _run_curve(xai, image_bgr, baseline_bgr, class_id, pre_nms_index,
                            pixel_order, mode="insertion")
    deletion = _run_curve(xai, image_bgr, baseline_bgr, class_id, pre_nms_index,
                           pixel_order, mode="deletion")
    random_curve = _run_random_curve(xai, image_bgr, baseline_bgr, class_id,
                                      pre_nms_index, heatmap.size, random_seed)

    return FaithfulnessResult(
        gt_instance_id=gt_instance_id, method=method,
        insertion=insertion, deletion=deletion,
        random_control_insertion=random_curve,
    )
