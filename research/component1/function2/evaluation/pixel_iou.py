"""F2.4-A Task 8 -- Spatial IoU against verified pixel-level reference masks.

Scope note (restated): this is a SEPARATE, much smaller pilot than the main
6-9 instance F2.4-A pilot (Task 9). It runs ONLY on whichever of F2.2C's 9
verified reference masks happen to intersect F2.3-B's 65 paired-success
explanations -- an intersection that must be computed, not assumed, because
the SAM mask pilot (F2.2C) and the 60-image XAI primary sample (F2.2B) were
built independently. See ``reports/f24_pilot_report.md`` Task 8 section for
the actual intersection count found for this run.

Binarization (fixed, PREDECLARED, never tuned per instance):
  "top 20% of positive activation BY ENERGY" -- sort all strictly-positive
  heatmap pixels descending by value, take the smallest prefix of that
  sorted list whose cumulative SUM reaches >= 20% of the total positive
  energy, and set exactly those pixels to 1 (all others, including all
  zero/negative pixels, to 0). This is an ENERGY-based threshold, not a
  pixel-COUNT-based one (e.g. "top 20% of pixels by count" would be a
  materially different, also-defensible rule -- this module picks the
  energy definition and documents it explicitly so the choice is
  auditable, not ambiguous).

No threshold search of any kind is performed against the reference mask --
the 20% figure is fixed before any mask is ever loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

ENERGY_THRESHOLD_FRACTION = 0.20


def binarize_top_energy_fraction(heatmap: np.ndarray,
                                  fraction: float = ENERGY_THRESHOLD_FRACTION) -> np.ndarray:
    """Binary mask (uint8, same shape as ``heatmap``) of the smallest set of
    strictly-positive pixels whose cumulative energy reaches ``fraction`` of
    the heatmap's total positive energy."""
    positive = np.where(heatmap > 0, heatmap, 0.0)
    total = positive.sum()
    out = np.zeros(heatmap.shape, dtype=np.uint8)
    if total <= 1e-8:
        return out  # degenerate heatmap -> empty mask, not an error (caller checks separately)
    flat = positive.ravel()
    order = np.argsort(flat)[::-1]  # descending by value
    cumsum = np.cumsum(flat[order])
    target = fraction * total
    n_needed = int(np.searchsorted(cumsum, target) + 1)
    n_needed = min(n_needed, flat.size)
    chosen = order[:n_needed]
    out_flat = out.reshape(-1)
    out_flat[chosen] = 1
    return out_flat.reshape(heatmap.shape)


def spatial_iou(binary_heatmap_mask: np.ndarray, reference_mask: np.ndarray) -> float:
    """Standard IoU between two same-shape binary masks."""
    a = binary_heatmap_mask.astype(bool)
    b = reference_mask.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")  # both masks empty -- undefined, not 0 or 1
    return float(inter) / float(union)


@dataclass
class PixelIoUResult:
    gt_instance_id: str
    method: str
    status: str  # "valid", "degenerate_heatmap", "shape_mismatch"
    iou: Optional[float]
    threshold_fraction: float = ENERGY_THRESHOLD_FRACTION
    reason: str = ""


def evaluate_pixel_iou(heatmap: np.ndarray, reference_mask: np.ndarray,
                        gt_instance_id: str, method: str) -> PixelIoUResult:
    if heatmap.shape != reference_mask.shape[:2]:
        return PixelIoUResult(gt_instance_id, method, "shape_mismatch", None,
                               reason=f"heatmap shape {heatmap.shape} != reference mask shape "
                                      f"{reference_mask.shape[:2]} -- caller must resample the "
                                      f"mask or heatmap before calling, this function refuses to "
                                      f"silently broadcast/guess alignment.")
    positive_total = float(np.where(heatmap > 0, heatmap, 0.0).sum())
    if positive_total <= 1e-8:
        return PixelIoUResult(gt_instance_id, method, "degenerate_heatmap", None,
                               reason="Heatmap has ~zero total positive energy; no meaningful "
                                      "top-20%-energy mask can be formed.")
    binary = binarize_top_energy_fraction(heatmap)
    ref_binary = (reference_mask > 0).astype(np.uint8)
    if ref_binary.ndim == 3:
        ref_binary = ref_binary[..., 0]
    iou = spatial_iou(binary, ref_binary)
    return PixelIoUResult(gt_instance_id, method, "valid",
                           None if np.isnan(iou) else round(iou, 6))
