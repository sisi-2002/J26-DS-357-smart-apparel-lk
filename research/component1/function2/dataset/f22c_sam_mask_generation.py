"""F2.2C-Pilot Task B4-B8 -- SAM box-prompted mask generation + QC package.

Stage: F2.2C-Pilot. Reads the Task B2 selection manifests (``manifests/
image_selection.csv`` / ``instance_manifest.csv``), opens each selected
VALID-split image's pixels (explicitly permitted for this pilot stage, per
the task brief -- unlike the metadata-only F2.2A audit), converts each
instance's normalised YOLO box to pixel coordinates, and runs Meta's
Segment Anything Model (SAM, ViT-B checkpoint) with the box as a prompt to
produce an automatic, UN-REVIEWED binary mask proposal per instance.

Nothing in this module is a human expert mask, an approved mask, or a
ground-truth segmentation. Every mask produced here is explicitly an
algorithmic SAM proposal pending human QC review (see
``quality_review/mask_review_template.csv``, whose Accept/Needs
Correction/Reject/Unreviewed status column is always written as
"Unreviewed" by this script -- never auto-approved).

Split-membership safety
------------------------
This module ONLY ever resolves image paths via
``config.paths.CHENAB_VALID_IMAGES_DIR`` joined with a filename taken from
``split_manifest.csv`` (via Task B2's already-verified
``manifests/instance_manifest.csv``, itself built from the authoritative
manifest, never from a filename-prefix guess). It never touches
``CHENAB_TEST_IMAGES_DIR`` or ``CHENAB_TRAIN_IMAGES_DIR``.

Environment
-----------
This script requires the ``segment_anything`` package + a SAM checkpoint,
which live ONLY in the separate
``data-preprocessing/master_fabric_dataset/research/.sam_venv`` virtualenv
(not in ``backend/.venv``, which has no segment_anything install). Run with:

    data-preprocessing/master_fabric_dataset/research/.sam_venv/Scripts/python.exe \
        -m research.component1.function2.dataset.f22c_sam_mask_generation

No checkpoint was downloaded and no package was installed to produce this
pilot -- both were already present locally (see
``reports/f22c_task_b3_environment_check.md`` for the full environment
audit this stage performed before writing any code).
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.config.paths import (  # noqa: E402
    CHENAB_VALID_IMAGES_DIR,
    CHENAB_TEST_IMAGES_DIR,
    CHENAB_TRAIN_IMAGES_DIR,
)

PILOT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "reference_masks" / "pilot_v1"
MANIFESTS_DIR = PILOT_DIR / "manifests"
SAM_PROPOSALS_DIR = PILOT_DIR / "sam_proposals"
OVERLAYS_DIR = PILOT_DIR / "overlays"
QUALITY_REVIEW_DIR = PILOT_DIR / "quality_review"
REPORTS_DIR = PILOT_DIR / "reports"

SAM_CHECKPOINT = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "research" / ".sam_weights" / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

FORBIDDEN_IMAGE_DIRS = (CHENAB_TEST_IMAGES_DIR.resolve(), CHENAB_TRAIN_IMAGES_DIR.resolve())


def yolo_box_to_pixels(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert a normalised YOLO box (cx, cy, w, h in [0,1]) to pixel-space
    (x1, y1, x2, y2), clipped to the image bounds. Pure arithmetic, no I/O."""
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    x1 = max(0, min(img_w - 1, round(x1)))
    y1 = max(0, min(img_h - 1, round(y1)))
    x2 = max(0, min(img_w, round(x2)))
    y2 = max(0, min(img_h, round(y2)))
    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)
    return x1, y1, x2, y2


def _resolve_valid_image_path(filename: str) -> Path:
    path = (CHENAB_VALID_IMAGES_DIR / filename).resolve()
    for forbidden in FORBIDDEN_IMAGE_DIRS:
        if forbidden in path.parents or path == forbidden:
            raise RuntimeError(f"REFUSING to open non-VALID path: {path}")
    if CHENAB_VALID_IMAGES_DIR.resolve() not in path.parents and path.parent != CHENAB_VALID_IMAGES_DIR.resolve():
        raise RuntimeError(f"REFUSING to open path outside CHENAB_VALID_IMAGES_DIR: {path}")
    return path


def _mask_quality_flags(mask: "np.ndarray", box_xyxy: tuple[int, int, int, int], img_shape: tuple[int, int],
                         is_elongated: bool, is_small: bool) -> dict:
    import numpy as np
    import cv2

    img_h, img_w = img_shape
    total_px = img_h * img_w
    mask_px = int(mask.sum())
    area_fraction_of_image = mask_px / total_px if total_px else 0.0

    x1, y1, x2, y2 = box_xyxy
    box_mask = np.zeros_like(mask, dtype=bool)
    box_mask[y1:y2, x1:x2] = True
    inside_box_px = int((mask & box_mask).sum())
    outside_box_px = mask_px - inside_box_px
    leakage_fraction = (outside_box_px / mask_px) if mask_px > 0 else 0.0

    n_components, labels_im = cv2.connectedComponents(mask.astype("uint8"))
    n_fragments = max(0, n_components - 1)  # exclude background label 0

    empty_mask = mask_px == 0
    near_full_image = area_fraction_of_image >= 0.95
    leakage_outside_box = (not empty_mask) and leakage_fraction > 0.30
    fragmented = n_fragments > 1
    thin_defect_mishandled = (is_elongated or is_small) and (empty_mask or area_fraction_of_image < 0.0005)

    warnings = []
    if empty_mask:
        warnings.append("empty_mask")
    if near_full_image:
        warnings.append("near_full_image_mask")
    if leakage_outside_box:
        warnings.append("leakage_outside_gt_box")
    if fragmented:
        warnings.append("fragmented_mask")
    if thin_defect_mishandled:
        warnings.append("possible_thin_defect_mishandling")

    return {
        "mask_pixel_count": mask_px,
        "area_fraction_of_image": round(area_fraction_of_image, 6),
        "inside_box_pixel_count": inside_box_px,
        "outside_box_pixel_count": outside_box_px,
        "leakage_fraction": round(leakage_fraction, 6),
        "n_connected_components": n_fragments,
        "quality_warnings": warnings,
    }


def run() -> dict:
    import numpy as np
    import cv2
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    SAM_PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SAM_CHECKPOINT.is_file():
        raise RuntimeError(f"SAM checkpoint not found at {SAM_CHECKPOINT}; refusing to fabricate masks.")

    with open(MANIFESTS_DIR / "instance_manifest.csv", newline="", encoding="utf-8") as f:
        instances = list(csv.DictReader(f))

    instances_by_image: dict[str, list[dict]] = defaultdict(list)
    for inst in instances:
        instances_by_image[inst["image_id"]].append(inst)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    results = []
    n_success = 0
    n_failed = 0

    for image_id, inst_list in sorted(instances_by_image.items()):
        filename = inst_list[0]["filename"]
        image_path = _resolve_valid_image_path(filename)
        if not image_path.is_file():
            for inst in inst_list:
                n_failed += 1
                results.append({
                    "instance_id": inst["instance_id"], "image_id": image_id,
                    "status": "failed", "failure_reason": f"image file not found: {image_path}",
                })
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            for inst in inst_list:
                n_failed += 1
                results.append({
                    "instance_id": inst["instance_id"], "image_id": image_id,
                    "status": "failed", "failure_reason": f"cv2.imread returned None for {image_path}",
                })
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]

        try:
            predictor.set_image(image_rgb)
        except Exception as exc:  # noqa: BLE001
            for inst in inst_list:
                n_failed += 1
                results.append({
                    "instance_id": inst["instance_id"], "image_id": image_id,
                    "status": "failed", "failure_reason": f"predictor.set_image failed: {exc}",
                })
            continue

        for inst in inst_list:
            instance_id = inst["instance_id"]
            try:
                cx, cy, w, h = (float(inst[k]) for k in ("cx", "cy", "w", "h"))
                box_xyxy = yolo_box_to_pixels(cx, cy, w, h, img_w, img_h)
                box_np = np.array(box_xyxy, dtype=np.float32)

                masks, scores, logits = predictor.predict(
                    box=box_np, multimask_output=True,
                )
                best_idx = int(np.argmax(scores))
                mask = masks[best_idx].astype(bool)
                sam_score = float(scores[best_idx])

                is_elongated = inst["elongated"] == "True"
                is_small = inst["small"] == "True"
                quality = _mask_quality_flags(mask, box_xyxy, (img_h, img_w), is_elongated, is_small)

                mask_png = (mask.astype("uint8") * 255)
                mask_out_path = SAM_PROPOSALS_DIR / f"{instance_id}.png"
                cv2.imwrite(str(mask_out_path), mask_png)

                # --- QC overlay: 2x2 grid (orig+box | orig+translucent mask |
                # mask-only binary | orig+mask boundary contour). ---
                panel_tl = image_bgr.copy()
                x1, y1, x2, y2 = box_xyxy
                cv2.rectangle(panel_tl, (x1, y1), (x2, y2), (0, 0, 255), 2)

                panel_tr = image_bgr.copy()
                overlay_color = np.zeros_like(panel_tr)
                overlay_color[mask] = (0, 255, 0)
                panel_tr = cv2.addWeighted(panel_tr, 1.0, overlay_color, 0.5, 0)

                panel_bl = cv2.cvtColor(mask_png, cv2.COLOR_GRAY2BGR)

                panel_br = image_bgr.copy()
                contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel_br, contours, -1, (255, 0, 255), 2)
                cv2.rectangle(panel_br, (x1, y1), (x2, y2), (0, 0, 255), 1)

                top = np.hstack([panel_tl, panel_tr])
                bottom = np.hstack([panel_bl, panel_br])
                grid = np.vstack([top, bottom])
                overlay_out_path = OVERLAYS_DIR / f"{instance_id}_overlay.png"
                cv2.imwrite(str(overlay_out_path), grid)

                n_success += 1
                results.append({
                    "instance_id": instance_id, "image_id": image_id, "status": "success",
                    "sam_confidence_score": round(sam_score, 6),
                    "box_xyxy_pixels": list(box_xyxy),
                    "image_width": img_w, "image_height": img_h,
                    "mask_path": str(mask_out_path.relative_to(REPO_ROOT).as_posix()),
                    "overlay_path": str(overlay_out_path.relative_to(REPO_ROOT).as_posix()),
                    **quality,
                })
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                results.append({
                    "instance_id": instance_id, "image_id": image_id,
                    "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}",
                })

    # --- Quality review template: genuinely blank human-review fields. -----
    review_rows = []
    for r in results:
        review_rows.append({
            "instance_id": r["instance_id"],
            "image_id": r["image_id"],
            "status": r.get("status", "failed"),
            "automated_quality_warnings": ";".join(r.get("quality_warnings", [])) if r.get("status") == "success" else (r.get("failure_reason", "")),
            "sam_confidence_score": r.get("sam_confidence_score", ""),
            "human_review_status": "Unreviewed",
            "reviewer_id": "",
            "review_date": "",
            "reviewer_comments": "",
        })
    review_fieldnames = [
        "instance_id", "image_id", "status", "automated_quality_warnings",
        "sam_confidence_score", "human_review_status", "reviewer_id",
        "review_date", "reviewer_comments",
    ]
    with open(QUALITY_REVIEW_DIR / "mask_review_template.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=review_fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    with open(QUALITY_REVIEW_DIR / "cvat_export_instructions.md", "w", encoding="utf-8") as f:
        f.write(_cvat_instructions_markdown())

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "F2.2C-Pilot Task B4-B8",
        "device_used": device,
        "sam_model_type": SAM_MODEL_TYPE,
        "sam_checkpoint": str(SAM_CHECKPOINT.relative_to(REPO_ROOT).as_posix()),
        "n_instances_attempted": len(instances),
        "n_success": n_success,
        "n_failed": n_failed,
        "per_instance_results": results,
        "status_label": "Algorithmic SAM proposals -- NOT human-verified masks, NOT ground truth",
    }
    with open(REPORTS_DIR / "f22c_mask_generation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _cvat_instructions_markdown() -> str:
    return """# CVAT-compatible manual-correction export instructions

No local CVAT instance/install was found or set up for this pilot (CVAT was
not checked/installed as part of this stage -- no new package was installed
without stopping to report it first, per this stage's constraints). Masks
were instead produced as plain binary PNGs
(`results/reference_masks/pilot_v1/sam_proposals/<instance_id>.png`) at the
original image resolution, one file per instance.

To bring these into a CVAT project for manual correction:

1. Create a CVAT task over the 18 selected VALID images (file list in
   `manifests/image_selection.csv`).
2. Import each instance's binary mask PNG as a pre-annotation segmentation
   mask (CVAT supports a "Segmentation mask" import format where each
   class/instance is one PNG; our masks are already binary 0/255 with one
   instance per file, named `<image_id>__inst_NNN.png`, which maps 1:1 onto
   CVAT's expected pre-annotation file naming once the image stem matches).
3. Reviewers correct/accept/reject each instance mask inside CVAT as normal.
4. Export corrected masks back out of CVAT in the same binary-PNG-per-instance
   format, and record the outcome in
   `quality_review/mask_review_template.csv`'s `human_review_status` column
   (`Accept` / `Needs Correction` / `Reject`) plus `reviewer_id` /
   `review_date` / `reviewer_comments` -- all of which are currently blank
   and must be filled in by an actual human reviewer, never auto-filled.
"""


if __name__ == "__main__":
    summary = run()
    print(json.dumps({k: v for k, v in summary.items() if k != "per_instance_results"}, indent=2))
