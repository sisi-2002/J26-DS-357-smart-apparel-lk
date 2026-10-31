"""F2.4-A Task 3 -- localization metrics.

Two metrics, both computed from a single raw (un-normalized) Grad-CAM /
Grad-CAM++ heatmap array (as written by F2.3-B to
``heatmaps/<method>/*.npy``) and a single GT box (pixel xyxy, as recorded in
``execution_manifest.csv`` / ``target_linkage.csv``):

  (A) Pointing accuracy -- binary hit/miss of the heatmap's argmax pixel
      against the GT box.
  (B) GT-box energy concentration -- fraction of the heatmap's total
      positive mass that falls inside the GT box. This is explicitly NOT
      pixel-level segmentation IoU (there is no reference mask here, only a
      box) -- see ``box_energy_concentration``'s docstring.

Coordinate convention: pixel xyxy, (x1, y1, x2, y2), in the ORIGINAL image's
pixel coordinate frame, matching ``gt_matching.GTInstance.bbox_xyxy`` and
``execution_manifest.csv``'s ``gt_box`` column exactly (no normalization to
[0,1], no xywh).

Heatmap-vs-image resolution: F2.3-B's heatmaps are already resampled back to
the SOURCE image's exact (H, W) by ``DetectionXAI._map_to_original`` (this
was independently re-verified in F2.4-A Task 1 against 10 spot-checked
instances: heatmap.shape == source image.shape[:2] in every case). The
functions below do not assume this, however: if a heatmap's resolution ever
differs from the image resolution a caller supplies, the GT box is rescaled
by the (heatmap_w/image_w, heatmap_h/image_h) ratio before use, and this
rescaling is recorded in the result so silent misalignment cannot happen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ZERO_ENERGY_EPS = 1e-8


@dataclass
class PointingResult:
    status: str  # "valid", "invalid_nonfinite", "invalid_all_zero"
    hit: Optional[bool]
    argmax_xy: Optional[tuple]  # (x, y) in image pixel coords, after any rescale
    argmax_row_major_flat_index: Optional[int]
    heatmap_shape: tuple
    image_shape: tuple
    box_rescaled: bool


@dataclass
class EnergyResult:
    status: str  # "valid", "invalid_nonfinite", "degenerate_zero_energy"
    energy_fraction: Optional[float]
    box_area_fraction_of_image: float
    total_positive_energy: float
    box_positive_energy: float
    heatmap_shape: tuple
    image_shape: tuple
    box_rescaled: bool


def _rescale_box(gt_box_xyxy, heatmap_hw, image_hw):
    """Scale a GT box from the image's pixel frame into the heatmap's pixel
    frame. A no-op (returns the box unchanged, rescaled=False) when the two
    resolutions already match, which F2.3-B's own alignment guarantee makes
    the common case."""
    hh, hw = heatmap_hw
    ih, iw = image_hw
    if (hh, hw) == (ih, iw):
        return tuple(gt_box_xyxy), False
    sx = hw / iw
    sy = hh / ih
    x1, y1, x2, y2 = gt_box_xyxy
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy), True


def pointing_accuracy(heatmap: np.ndarray, gt_box_xyxy, image_shape_hw) -> PointingResult:
    """Argmax-pixel pointing-game hit/miss.

    Tie-break: ``np.argmax`` on a flattened (row-major / C-order) array
    always returns the FIRST occurrence of the maximum value in row-major
    (top-to-bottom, then left-to-right) order -- this is documented NumPy
    behaviour, not an incidental property, and is the tie-break used here.

    Boundary convention: INCLUSIVE on all four edges -- a pixel exactly on
    the GT box edge counts as a hit (``x1 <= x <= x2`` and ``y1 <= y <= y2``
    in the possibly-rescaled box's own coordinate frame).

    An all-zero heatmap (a degenerate Grad-CAM output, e.g. F2.3-B's one
    recorded one-sided degenerate-Grad-CAM failure) is explicitly reported
    as status "invalid_all_zero" with ``hit=None`` -- never silently scored
    as a miss (which would unfairly penalize the method for a failure this
    metric did not cause) and never scored as a hit.
    """
    heatmap_hw = heatmap.shape[:2]
    if not np.all(np.isfinite(heatmap)):
        return PointingResult("invalid_nonfinite", None, None, None,
                               heatmap_hw, tuple(image_shape_hw), False)
    if not np.any(heatmap != 0):
        return PointingResult("invalid_all_zero", None, None, None,
                               heatmap_hw, tuple(image_shape_hw), False)

    box_rescaled_xyxy, was_rescaled = _rescale_box(gt_box_xyxy, heatmap_hw, image_shape_hw)
    flat_idx = int(np.argmax(heatmap.ravel()))  # first max, row-major
    gy, gx = np.unravel_index(flat_idx, heatmap.shape)
    x1, y1, x2, y2 = box_rescaled_xyxy
    hit = (x1 <= gx <= x2) and (y1 <= gy <= y2)
    return PointingResult(
        "valid", bool(hit), (float(gx), float(gy)), flat_idx,
        heatmap_hw, tuple(image_shape_hw), was_rescaled,
    )


def box_energy_concentration(heatmap: np.ndarray, gt_box_xyxy, image_shape_hw) -> EnergyResult:
    """GT-box energy concentration (NOT pixel-level segmentation IoU -- there
    is no reference mask here, only a bounding box, so this measures how
    much of the heatmap's positive mass a box captures, not shape overlap).

    energy_fraction = sum(heatmap[inside box AND heatmap > 0])
                      / sum(heatmap[heatmap > 0])

    "Positive" uses strict ``> 0`` (Grad-CAM/Grad-CAM++ outputs are already
    ReLU'd so are never negative; ``> 0`` excludes exact zeros from the
    denominator, which matters for degenerate-energy detection).
    """
    heatmap_hw = heatmap.shape[:2]
    ih, iw = image_shape_hw
    box_area_fraction = 0.0
    x1, y1, x2, y2 = gt_box_xyxy
    image_area = float(ih * iw)
    if image_area > 0:
        box_area_fraction = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / image_area

    if not np.all(np.isfinite(heatmap)):
        return EnergyResult("invalid_nonfinite", None, box_area_fraction, 0.0, 0.0,
                             heatmap_hw, tuple(image_shape_hw), False)

    box_rescaled_xyxy, was_rescaled = _rescale_box(gt_box_xyxy, heatmap_hw, image_shape_hw)
    bx1, by1, bx2, by2 = box_rescaled_xyxy
    h, w = heatmap_hw
    cx1 = int(np.clip(np.floor(bx1), 0, w))
    cx2 = int(np.clip(np.ceil(bx2), 0, w))
    cy1 = int(np.clip(np.floor(by1), 0, h))
    cy2 = int(np.clip(np.ceil(by2), 0, h))

    positive = np.where(heatmap > 0, heatmap, 0.0)
    total_positive = float(positive.sum())

    if total_positive <= ZERO_ENERGY_EPS:
        return EnergyResult("degenerate_zero_energy", None, box_area_fraction,
                             0.0, 0.0, heatmap_hw, tuple(image_shape_hw), was_rescaled)

    box_region = positive[cy1:cy2, cx1:cx2]
    box_positive = float(box_region.sum())
    fraction = box_positive / total_positive
    return EnergyResult("valid", fraction, box_area_fraction, total_positive,
                         box_positive, heatmap_hw, tuple(image_shape_hw), was_rescaled)


@dataclass
class LocalizationSummary:
    n_total: int = 0
    n_valid: int = 0
    n_invalid_nonfinite: int = 0
    n_invalid_all_zero_or_degenerate: int = 0
    n_hits: int = 0
    mean_energy_fraction: Optional[float] = None
    per_instance: list = field(default_factory=list)


def summarize_method(pointing_results: list[PointingResult],
                      energy_results: list[EnergyResult]) -> LocalizationSummary:
    s = LocalizationSummary(n_total=len(pointing_results))
    energies = []
    for p, e in zip(pointing_results, energy_results):
        if p.status == "valid":
            s.n_valid += 1
            if p.hit:
                s.n_hits += 1
        elif p.status == "invalid_nonfinite":
            s.n_invalid_nonfinite += 1
        else:
            s.n_invalid_all_zero_or_degenerate += 1
        if e.status == "valid":
            energies.append(e.energy_fraction)
    if energies:
        s.mean_energy_fraction = float(np.mean(energies))
    return s
