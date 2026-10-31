"""Visualization helpers for the F2.3-A dry run (Task 5).

All drawing happens in ORIGINAL image pixel coordinates. The heatmap arrays
consumed here (``ExplanationResult.heatmap_raw``) are already mapped back to
original coordinates by ``DetectionXAI._map_to_original`` inside
``detection_xai.explain()`` -- this module does not re-derive that mapping,
it only verifies the shape is consistent with the source image (see
``check_alignment``) and renders it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

GT_COLOR_BGR = (0, 255, 0)       # green
PRED_COLOR_BGR = (0, 140, 255)   # orange


def check_alignment(heatmap_raw: np.ndarray, image_bgr: np.ndarray) -> bool:
    """True iff the heatmap's (H, W) exactly matches the source image's."""
    return heatmap_raw.shape[:2] == image_bgr.shape[:2]


def colorize_heatmap(heatmap_raw: np.ndarray) -> np.ndarray:
    """Min-max normalize (documented convention, matching
    ``DetectionXAI._normalize``) then apply a JET colormap. Pure colorized
    heatmap -- NOT blended with the original image (that is the separate
    "overlay" output, which detection_xai.explain() already returns)."""
    vmin, vmax = float(heatmap_raw.min()), float(heatmap_raw.max())
    if vmax - vmin < 1e-8:
        norm = np.zeros_like(heatmap_raw, dtype=np.float32)
    else:
        norm = (heatmap_raw - vmin) / (vmax - vmin)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def draw_boxes(image_bgr: np.ndarray, gt_bbox_xyxy, pred_bbox_xyxy=None,
               gt_label: str = "", pred_label: str = "") -> np.ndarray:
    img = image_bgr.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in gt_bbox_xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR_BGR, 2)
    if gt_label:
        cv2.putText(img, gt_label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR_BGR, 1, cv2.LINE_AA)
    if pred_bbox_xyxy is not None:
        px1, py1, px2, py2 = (int(round(v)) for v in pred_bbox_xyxy)
        cv2.rectangle(img, (px1, py1), (px2, py2), PRED_COLOR_BGR, 2)
        if pred_label:
            cv2.putText(img, pred_label, (px1, min(img.shape[0] - 2, py2 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR_BGR, 1, cv2.LINE_AA)
    return img


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    labeled = img.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(labeled, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def build_comparison_panel(boxed_image: np.ndarray,
                            gradcam_overlay: np.ndarray,
                            gradcampp_overlay: np.ndarray) -> np.ndarray:
    """Side-by-side panel: [boxed original | Grad-CAM overlay | Grad-CAM++ overlay].

    All three inputs share the original image's (H, W); only horizontal
    concatenation is used so no cropping/stretching can break alignment.
    """
    h = boxed_image.shape[0]
    panels = [boxed_image, gradcam_overlay, gradcampp_overlay]
    labels = ["GT (green) / Matched pred (orange)", "Grad-CAM", "Grad-CAM++"]
    resized = []
    for p, label in zip(panels, labels):
        if p.shape[0] != h:
            scale = h / p.shape[0]
            p = cv2.resize(p, (int(round(p.shape[1] * scale)), h))
        resized.append(_label_panel(p, label))
    return np.concatenate(resized, axis=1)


def save_png(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise OSError(f"cv2.imwrite failed for {path}")


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)
