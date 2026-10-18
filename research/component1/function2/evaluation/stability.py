"""F2.4-A Task 5 -- stability metrics (robustness to small nuisance
perturbation).

IMPORTANT SCOPE NOTE: a high similarity score here measures robustness to
small, defect-preserving nuisance perturbation ONLY. It is not evidence of
faithfulness (Task 4 covers that, separately and insufficiently on its own
too) and not evidence that the explanation correctly localizes the true
causal region -- a method could be perfectly stable and still be wrong.

Perturbations (all small, pre-specified, applied to the ORIGINAL image
before any re-detection):
  - brightness +5%  : pixel * 1.05, clipped to [0, 255]
  - brightness -5%  : pixel * 0.95, clipped to [0, 255]
  - contrast  +5%   : (pixel - 127.5) * 1.05 + 127.5, clipped
  - contrast  -5%   : (pixel - 127.5) * 0.95 + 127.5, clipped
  - gaussian_noise  : + N(0, sigma=5) i.i.d. per pixel/channel, fixed sigma,
                      fixed seed (RANDOM_SEED), clipped
  - pixel_shift     : translate by (dx, dy) = (2, 2) pixels (border
                      replicated), heatmap compared after shifting the
                      PERTURBED heatmap back by (-dx, -dy) ("inverse
                      alignment") so both heatmaps are compared in the
                      SAME (original) coordinate frame

For each perturbation, the policy is: re-run the SAME targeting mechanism
(``DetectionXAI.detect()`` + ``gt_matching`` same-class/highest-IoU rule) on
the perturbed image -- NOT a cached/fixed target, because a real detector
genuinely can shift its output slightly under nuisance perturbation. The
perturbed detection is accepted as "still the same GT instance" iff its IoU
against the ORIGINAL matched box is >= 0.5 -- the identical threshold
``gt_matching.IOU_MATCH_THRESHOLD`` already uses, so no new unrelated
constant is introduced. If no same-class detection reaches this threshold,
the instance is recorded INVALID for this perturbation (never force-compared
against an unrelated detection's heatmap).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from research.component1.function2.evaluation.common import (
    best_same_class_iou, spearman_corr,
)
from research.component1.function2.xai.gt_matching import IOU_MATCH_THRESHOLD

RANDOM_SEED = 357
PIXEL_SHIFT_DXDY = (2, 2)
NOISE_SIGMA = 5.0


def perturb_brightness(image_bgr: np.ndarray, pct: float) -> np.ndarray:
    out = image_bgr.astype(np.float32) * (1.0 + pct)
    return np.clip(out, 0, 255).astype(np.uint8)


def perturb_contrast(image_bgr: np.ndarray, pct: float) -> np.ndarray:
    out = (image_bgr.astype(np.float32) - 127.5) * (1.0 + pct) + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def perturb_gaussian_noise(image_bgr: np.ndarray, sigma: float = NOISE_SIGMA,
                            seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, sigma, size=image_bgr.shape)
    out = image_bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def perturb_pixel_shift(image_bgr: np.ndarray, dxdy=PIXEL_SHIFT_DXDY) -> np.ndarray:
    dx, dy = dxdy
    h, w = image_bgr.shape[:2]
    mat = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(image_bgr, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)


def inverse_align_shifted_heatmap(heatmap: np.ndarray, dxdy=PIXEL_SHIFT_DXDY) -> np.ndarray:
    """Shift a heatmap computed on a pixel-shifted image back by (-dx, -dy)
    so it lines up with the ORIGINAL (unshifted) image's coordinate frame
    before comparison."""
    dx, dy = dxdy
    h, w = heatmap.shape[:2]
    mat = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
    return cv2.warpAffine(heatmap.astype(np.float32), mat, (w, h),
                           borderMode=cv2.BORDER_REPLICATE)


PERTURBATIONS = {
    "brightness_plus5pct": lambda img: perturb_brightness(img, 0.05),
    "brightness_minus5pct": lambda img: perturb_brightness(img, -0.05),
    "contrast_plus5pct": lambda img: perturb_contrast(img, 0.05),
    "contrast_minus5pct": lambda img: perturb_contrast(img, -0.05),
    "gaussian_noise_sigma5": lambda img: perturb_gaussian_noise(img),
    "pixel_shift_2px": lambda img: perturb_pixel_shift(img),
}


@dataclass
class StabilityResult:
    gt_instance_id: str
    method: str
    perturbation: str
    status: str  # "valid", "target_lost"
    matched_iou_vs_original: Optional[float]
    similarity: Optional[float]  # Spearman rank correlation, original vs perturbed heatmap
    reason: str = ""


def evaluate_stability_one_perturbation(
    xai, explain_fn, image_bgr: np.ndarray, original_heatmap: np.ndarray,
    class_id: int, original_matched_bbox_xyxy, perturbation_name: str,
    gt_instance_id: str, method: str,
) -> StabilityResult:
    """``explain_fn(image_bgr, detection) -> ExplanationResult`` is a thin
    partial over ``DetectionXAI.explain(method=...)`` supplied by the
    caller (keeps this function method-agnostic)."""
    perturb_fn = PERTURBATIONS[perturbation_name]
    perturbed_img = perturb_fn(image_bgr)

    best_iou, best_det = best_same_class_iou(
        perturbed_img, xai, class_id, original_matched_bbox_xyxy,
    )
    if best_det is None or best_iou < IOU_MATCH_THRESHOLD:
        return StabilityResult(
            gt_instance_id, method, perturbation_name, "target_lost",
            round(max(best_iou, 0.0), 4), None,
            reason=(f"Best same-class candidate IoU={max(best_iou, 0.0):.4f} against the "
                    f"ORIGINAL matched box is below {IOU_MATCH_THRESHOLD} after "
                    f"'{perturbation_name}' -- no longer treated as the same instance."),
        )

    expl = explain_fn(perturbed_img, best_det)
    if expl.status != "success" or expl.heatmap_raw is None:
        return StabilityResult(
            gt_instance_id, method, perturbation_name, "target_lost",
            round(best_iou, 4), None,
            reason=f"Re-detected same instance (IoU={best_iou:.4f}) but explanation "
                   f"generation failed: {expl.status} / {expl.error_message}",
        )

    perturbed_heatmap = expl.heatmap_raw
    if perturbation_name == "pixel_shift_2px":
        perturbed_heatmap = inverse_align_shifted_heatmap(perturbed_heatmap)

    if perturbed_heatmap.shape != original_heatmap.shape:
        return StabilityResult(
            gt_instance_id, method, perturbation_name, "target_lost",
            round(best_iou, 4), None,
            reason=f"Heatmap shape mismatch after perturbation: "
                   f"{original_heatmap.shape} vs {perturbed_heatmap.shape}",
        )

    sim = spearman_corr(original_heatmap, perturbed_heatmap)
    return StabilityResult(gt_instance_id, method, perturbation_name, "valid",
                            round(best_iou, 4),
                            None if np.isnan(sim) else round(sim, 6))
