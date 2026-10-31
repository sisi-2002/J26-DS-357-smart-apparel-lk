"""Ground-truth <-> predicted-detection matching policy (Task 3).

This is new logic: detection_xai.py has no notion of ground truth at all
(it operates purely on a model + an already-identified ``Detection``).
Everything here is specified BEFORE any heatmap is generated and is applied
identically to every instance -- no cherry-picking based on explanation
quality is possible because this module never sees a heatmap.

Policy (documented once, used everywhere):
  1. Ground truth comes from the authoritative Chenab YOLO-format label file
     for the image (class_id, normalized cx, cy, w, h), converted to
     absolute xyxy pixel coordinates in the ORIGINAL image's coordinate
     frame.
  2. Predicted detections come from ``DetectionXAI.detect()`` (post-NMS,
     original-image coordinates -- see detection_xai.py's own preprocessing
     docstring for the shared letterbox convention).
  3. For a given GT instance, a candidate predicted detection must have the
     SAME class_id. Among same-class candidates, compute IoU(GT, prediction)
     in original-image pixel coordinates.
  4. The match is the same-class candidate with the HIGHEST IoU, accepted
     only if that IoU >= IOU_MATCH_THRESHOLD (0.5, the conventional
     detection-matching threshold; chosen because it is the same threshold
     used throughout object-detection evaluation, e.g. PASCAL VOC / COCO
     AP50, and is also the NMS IoU threshold already used by this model's
     own inference defaults, so no new unrelated constant is introduced).
  5. If no same-class candidate reaches the threshold (including the case
     of zero predictions of that class), the instance is recorded as
     UNMATCHED with an explicit technical reason. No forced/fallback match
     is ever produced.
  6. Ties (two candidates with exactly equal IoU) are broken by the lowest
     ``detection_id`` (the order ``detect()`` already returns detections
     in, which is itself deterministic for a fixed input and fixed model
     weights -- see test_matching_is_deterministic).

This makes the whole procedure a pure function of (GT boxes, detections):
same input always produces the same match, independent of any human
judgement about explanation quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IOU_MATCH_THRESHOLD = 0.5


@dataclass(frozen=True)
class GTInstance:
    instance_index: int  # 0-based position within the label file, stable order
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]  # original-image pixel coords


@dataclass(frozen=True)
class MatchResult:
    gt: GTInstance
    status: str  # "matched" or "unmatched"
    matched_detection_id: Optional[int] = None
    matched_class_id: Optional[int] = None
    matched_confidence: Optional[float] = None
    matched_bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    iou: float = 0.0
    reason: str = ""


def iou_xyxy(box_a: tuple[float, float, float, float],
             box_b: tuple[float, float, float, float]) -> float:
    """IoU of two axis-aligned boxes in (x1, y1, x2, y2) format.

    Pure-Python/no-dependency reference implementation, independently
    written (not imported from detection_xai.py) because GT-vs-prediction
    matching is new logic this stage must add, not existing logic it is
    reusing. Hand-verifiable: e.g. two identical unit boxes -> IoU 1.0;
    two boxes touching at an edge only -> IoU 0.0 (see tests).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def load_yolo_gt_instances(label_path: Path, img_w: int, img_h: int,
                            class_names: list[str]) -> list[GTInstance]:
    """Parse a YOLO-format label file into absolute-pixel GT instances.

    Lines are ``class_id cx cy w h`` (space-separated, normalized 0..1,
    cx/cy/w/h relative to image width/height). Order in the file is
    preserved as ``instance_index`` so multi-defect images keep stable,
    reproducible per-instance identity.
    """
    instances: list[GTInstance] = []
    text = Path(label_path).read_text(encoding="utf-8")
    idx = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        cx_abs, cy_abs = cx * img_w, cy * img_h
        w_abs, h_abs = w * img_w, h * img_h
        x1, y1 = cx_abs - w_abs / 2.0, cy_abs - h_abs / 2.0
        x2, y2 = cx_abs + w_abs / 2.0, cy_abs + h_abs / 2.0
        class_name = (class_names[class_id] if 0 <= class_id < len(class_names)
                      else f"class_{class_id}")
        instances.append(GTInstance(
            instance_index=idx, class_id=class_id, class_name=class_name,
            bbox_xyxy=(x1, y1, x2, y2),
        ))
        idx += 1
    return instances


def match_gt_to_detections(gt: GTInstance, detections: list,
                            iou_threshold: float = IOU_MATCH_THRESHOLD) -> MatchResult:
    """Apply the documented deterministic matching policy for one GT instance.

    ``detections`` is the list of ``Detection`` objects returned by
    ``DetectionXAI.detect()`` (already in original-image coordinates).
    """
    same_class = [d for d in detections if d.class_id == gt.class_id]
    if not same_class:
        return MatchResult(
            gt=gt, status="unmatched", reason=(
                f"No predicted detection of class '{gt.class_name}' "
                f"(class_id={gt.class_id}) at all -- the detector produced "
                f"zero same-class predictions for this image at the "
                f"configured confidence/NMS thresholds."
            ),
        )

    best = None
    best_iou = -1.0
    for d in sorted(same_class, key=lambda d: d.detection_id):
        iou = iou_xyxy(gt.bbox_xyxy, tuple(d.bbox_xyxy))
        if iou > best_iou:
            best_iou = iou
            best = d

    if best is None or best_iou < iou_threshold:
        return MatchResult(
            gt=gt, status="unmatched", iou=round(max(best_iou, 0.0), 4),
            reason=(
                f"Best same-class candidate IoU={max(best_iou, 0.0):.4f} is "
                f"below the required threshold {iou_threshold} -- the "
                f"detector's closest same-class prediction does not "
                f"sufficiently overlap this GT instance (undetected / "
                f"mislocalized defect at this confidence threshold)."
            ),
        )

    return MatchResult(
        gt=gt, status="matched",
        matched_detection_id=best.detection_id,
        matched_class_id=best.class_id,
        matched_confidence=best.confidence,
        matched_bbox_xyxy=tuple(best.bbox_xyxy),
        iou=round(best_iou, 4),
        reason="",
    )
