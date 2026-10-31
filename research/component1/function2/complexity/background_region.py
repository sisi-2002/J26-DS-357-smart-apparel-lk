"""Background-region selection adapter (F2.1A, Task 4).

Goal
----
Function 2 wants to measure fabric *background* visual complexity, not
defect-area complexity. The shared ``complexity_features.compute_complexity``
function already accepts an optional boolean ``mask`` (True = pixel included
in the measurement). This module's only job is to build that mask from the
dataset's authoritative YOLO ground-truth bounding boxes, by marking
GT-box pixels as excluded (not background) and everything else as
background.

This is explicitly a MINIMAL, provisional adapter for F2.1A smoke-testing,
not the final calibration pipeline.

Key design decisions and their limitations (see also
research/component1/function2/reports/task4_background_region_strategy.md):

1. Boxes are an approximation, not a segmentation.
   YOLO GT boxes are axis-aligned rectangles. They are NOT pixel-accurate
   defect masks. For elongated or diagonal defects (e.g. "cut", which in
   this dataset frequently has YOLO width ~= 1.0, i.e. spans the full image
   width — see reports/task4_background_region_strategy.md for examples
   pulled from the real train_manifest), the rectangular box over-excludes
   true background pixels that lie inside the box but outside the actual
   defect. We do not attempt to shrink/erode the box to compensate — doing
   so would require a segmentation model or manual re-annotation, which is
   out of scope for F2.1A.

2. Ground truth only, never model predictions.
   This module takes ``yolo_boxes`` as an explicit input. Callers MUST pass
   GT boxes parsed from the dataset's label .txt files
   (see dataset/manifest_loader.read_yolo_label_file). Nothing in this
   module runs the frozen E02 model or accepts/filters by confidence score.

3. Missing/ambiguous annotation != clean background.
   If the caller cannot determine the GT annotation state for an image
   (e.g. the label file could not be parsed), ``yolo_boxes=None`` must be
   passed, and this function returns status
   ``"unresolved_missing_annotation"`` with mask=None. It NEVER falls back
   to treating an unresolved image as if it had zero defects.
   A confirmed-negative image (label file exists and is empty) is a
   different, explicit case: pass ``yolo_boxes=[]`` with
   ``is_negative=True``.

4. Insufficient background.
   If GT boxes (unioned, accounting for overlap via the raster mask itself)
   cover more than ``max_defect_coverage`` of the image, status becomes
   "insufficient_background" — the image is flagged as not useable for a
   meaningful background-only complexity measurement, rather than silently
   measuring complexity over a tiny or empty background region.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

# If GT boxes cover more than this fraction of the image, remaining
# background is judged too small/fragmented to measure meaningfully.
# Chosen conservatively: 70% defect coverage leaves at most 30% background,
# often as thin unconnected strips for full-width "cut" defects in this
# dataset. This threshold is provisional (not calibrated) — see the F2.1A
# report for the real coverage distribution found in the train-split sample.
DEFAULT_MAX_DEFECT_COVERAGE = 0.70

YoloBox = tuple[float, float, float, float]  # (cx, cy, w, h), normalised 0-1


@dataclasses.dataclass
class BackgroundRegionResult:
    mask: Optional[np.ndarray]           # bool (H, W); True = usable background pixel
    background_coverage_fraction: float  # background pixels / total pixels; 0.0 if unresolved
    status: str
    num_boxes_excluded: int
    warnings: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("mask")  # mask is not JSON-serialisable / not meant for CSV rows
        return d


def yolo_box_to_pixel_xyxy(box: YoloBox, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert a normalised YOLO (cx, cy, w, h) box to pixel (x1, y1, x2, y2),
    clipped to image bounds."""
    cx, cy, w, h = box
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    x1 = max(0.0, min(x1, img_w))
    x2 = max(0.0, min(x2, img_w))
    y1 = max(0.0, min(y1, img_h))
    y2 = max(0.0, min(y2, img_h))
    return x1, y1, x2, y2


def compute_background_mask(
    image_height: int,
    image_width: int,
    yolo_boxes: Optional[Sequence[YoloBox]],
    is_negative: Optional[bool] = None,
    max_defect_coverage: float = DEFAULT_MAX_DEFECT_COVERAGE,
) -> BackgroundRegionResult:
    """Build a background-only pixel mask from GT YOLO boxes.

    Parameters
    ----------
    image_height, image_width : pixel dimensions of the (unresized) image
        the mask will be applied to. Must match what will be passed to
        complexity_features (which itself resizes internally and nearest-
        neighbour resizes the mask to match — see that module's docstring).
    yolo_boxes : list of normalised (cx, cy, w, h) GT boxes, or:
        - None            -> annotation status unknown/unavailable (unresolved)
        - []  (empty list) with is_negative=True -> confirmed clean image
    is_negative : whether the dataset manifest marks this image as a
        confirmed negative (no defects). Only meaningful combined with
        yolo_boxes=[].
    max_defect_coverage : see DEFAULT_MAX_DEFECT_COVERAGE docstring above.

    Returns
    -------
    BackgroundRegionResult
    """
    warnings: list[str] = []

    if image_height <= 0 or image_width <= 0:
        return BackgroundRegionResult(
            mask=None, background_coverage_fraction=0.0,
            status="error_invalid_input", num_boxes_excluded=0,
            warnings=["non_positive_image_dimensions"],
        )

    if yolo_boxes is None:
        return BackgroundRegionResult(
            mask=None, background_coverage_fraction=0.0,
            status="unresolved_missing_annotation", num_boxes_excluded=0,
            warnings=["annotation_missing_or_unavailable: treated as unresolved, NOT assumed clean"],
        )

    mask = np.ones((image_height, image_width), dtype=bool)
    excluded = 0
    for box in yolo_boxes:
        x1, y1, x2, y2 = yolo_box_to_pixel_xyxy(box, image_width, image_height)
        xi1, yi1 = int(np.floor(x1)), int(np.floor(y1))
        xi2, yi2 = int(np.ceil(x2)), int(np.ceil(y2))
        xi1, yi1 = max(0, xi1), max(0, yi1)
        xi2, yi2 = min(image_width, xi2), min(image_height, yi2)
        if xi2 > xi1 and yi2 > yi1:
            mask[yi1:yi2, xi1:xi2] = False
            excluded += 1
        else:
            warnings.append("degenerate_box_skipped")

    coverage = float(mask.sum()) / float(mask.size)

    if len(yolo_boxes) == 0:
        if is_negative:
            status = "clean_negative"
        else:
            status = "clean_no_boxes_provided"
            warnings.append(
                "empty yolo_boxes but is_negative is not True: treating as full-image "
                "background, but this should be double-checked upstream against the "
                "manifest's is_negative flag"
            )
    else:
        status = "boxes_excluded"

    if coverage < (1.0 - max_defect_coverage):
        status = "insufficient_background"
        warnings.append(
            f"GT box coverage exceeds max_defect_coverage={max_defect_coverage:.0%}; "
            f"remaining background fraction={coverage:.3f}"
        )

    return BackgroundRegionResult(
        mask=mask,
        background_coverage_fraction=coverage,
        status=status,
        num_boxes_excluded=excluded,
        warnings=warnings,
    )
