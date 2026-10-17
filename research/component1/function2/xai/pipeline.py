"""F2.3-A orchestrator: 9-image dry-run execution (Tasks 3, 4, 5, 6).

Wires together:
  - manifest_verification (Task 2): re-verify the frozen 9-image sample,
  - gt_matching (Task 3): deterministic GT-instance <-> prediction matching,
  - the PROTECTED DetectionXAI (detect() + explain()) for inference and the
    actual Grad-CAM / Grad-CAM++ math (Task 4) and coordinate alignment
    (Task 5),
  - visualization (drawing + saving outputs).

This module performs ONLY the 9-image dry run. It does not sample, does not
touch TEST-split files (manifest_verification raises before this module ever
opens a file if the authoritative split is not 'valid'), and computes no
IoU/faithfulness/stability/sanity statistics across the result set.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_XAI_SRC = _REPO_ROOT / "research" / "component1" / "xai_experiments" / "src"
if str(_XAI_SRC) not in sys.path:
    sys.path.insert(0, str(_XAI_SRC))
from detection_xai import DetectionXAI, CLASS_NAMES  # noqa: E402  (protected module, read-only import)

from research.component1.function2.xai.gt_matching import (  # noqa: E402
    IOU_MATCH_THRESHOLD,
    load_yolo_gt_instances,
    match_gt_to_detections,
)
from research.component1.function2.xai.manifest_verification import (  # noqa: E402
    verify_dry_run_sample,
)
from research.component1.function2.xai import visualization as viz  # noqa: E402

CONF_THRESHOLD = 0.25
IOU_NMS_THRESHOLD = 0.5
DEVICE = "cpu"
METHODS = ("gradcam", "gradcampp")


@dataclass
class InstanceRecord:
    image_id: str
    complexity_group: str
    source_family_id: str
    instance_index: int
    gt_class_id: int
    gt_class_name: str
    gt_bbox_xyxy: tuple
    match_status: str
    match_reason: str
    matched_detection_id: object = None
    matched_confidence: object = None
    matched_iou: float = 0.0
    matched_bbox_xyxy: object = None
    target_layer: str = ""
    target_scale: object = None
    target_cell_gy_gx: object = None
    target_logit: object = None
    gradcam_status: str = "not_attempted"
    gradcam_is_finite: object = None
    gradcam_is_nonzero: object = None
    gradcam_heatmap_shape: object = None
    gradcam_alignment_ok: object = None
    gradcam_time_s: object = None
    gradcampp_status: str = "not_attempted"
    gradcampp_is_finite: object = None
    gradcampp_is_nonzero: object = None
    gradcampp_heatmap_shape: object = None
    gradcampp_alignment_ok: object = None
    gradcampp_time_s: object = None
    same_target_for_both_methods: object = None
    error_message: str = ""


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_e02_checkpoint(freeze_manifest_path: Path, checkpoint_path: Path) -> dict:
    manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    expected = manifest["selected_experiment"]["checkpoint_sha256"]
    actual = _sha256(checkpoint_path)
    return {
        "expected_sha256": expected,
        "actual_sha256": actual,
        "verified": actual == expected,
        "class_names_from_manifest": manifest["dataset_identity"]["class_names_and_order"],
        "class_names_from_detection_xai": list(CLASS_NAMES),
        "class_names_match": manifest["dataset_identity"]["class_names_and_order"] == list(CLASS_NAMES),
    }


def run_dry_run(
    dry_run_sample_csv: Path,
    split_manifest_csv: Path,
    valid_complexity_manifest_csv: Path,
    dataset_root: Path,
    freeze_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict:
    t_start = time.time()
    output_dir = Path(output_dir)
    heatmaps_dir = output_dir / "heatmaps"
    overlays_dir = output_dir / "overlays"
    comparisons_dir = output_dir / "comparisons"
    manifests_dir = output_dir / "manifests"
    for d in (heatmaps_dir, overlays_dir, comparisons_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    checkpoint_check = verify_e02_checkpoint(freeze_manifest_path, checkpoint_path)
    if not checkpoint_check["verified"]:
        raise RuntimeError(
            f"E02 checkpoint SHA256 mismatch -- refusing to run. "
            f"expected={checkpoint_check['expected_sha256']} "
            f"actual={checkpoint_check['actual_sha256']}"
        )

    verified_rows = verify_dry_run_sample(
        dry_run_sample_csv, split_manifest_csv, valid_complexity_manifest_csv, dataset_root,
    )

    xai = DetectionXAI(str(checkpoint_path), device=DEVICE)

    instance_records: list[InstanceRecord] = []
    image_records: list[dict] = []

    for row in verified_rows:
        img_t0 = time.time()
        image_bgr = cv2.imread(str(row.image_path))
        if image_bgr is None:
            image_records.append({
                "image_id": row.image_id, "complexity_group": row.complexity_group,
                "status": "image_unreadable", "n_gt_instances": 0,
                "n_matched": 0, "n_unmatched": 0, "execution_time_s": 0.0,
            })
            continue

        gt_instances = load_yolo_gt_instances(
            row.label_path, image_bgr.shape[1], image_bgr.shape[0], CLASS_NAMES,
        )

        det_result = xai.detect(image_bgr, conf_threshold=CONF_THRESHOLD,
                                 iou_threshold=IOU_NMS_THRESHOLD)

        n_matched = 0
        n_unmatched = 0

        for gt in gt_instances:
            match = match_gt_to_detections(gt, det_result.detections, IOU_MATCH_THRESHOLD)
            rec = InstanceRecord(
                image_id=row.image_id, complexity_group=row.complexity_group,
                source_family_id=row.source_family_id,
                instance_index=gt.instance_index, gt_class_id=gt.class_id,
                gt_class_name=gt.class_name, gt_bbox_xyxy=gt.bbox_xyxy,
                match_status=match.status, match_reason=match.reason,
                matched_detection_id=match.matched_detection_id,
                matched_confidence=match.matched_confidence,
                matched_iou=match.iou, matched_bbox_xyxy=match.matched_bbox_xyxy,
            )

            if match.status != "matched":
                n_unmatched += 1
                instance_records.append(rec)
                continue

            n_matched += 1
            matched_detection = next(
                d for d in det_result.detections if d.detection_id == match.matched_detection_id
            )

            explanations = {}
            for method in METHODS:
                expl = xai.explain(
                    image_bgr, matched_detection, method=method,
                    conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_NMS_THRESHOLD,
                )
                explanations[method] = expl

            gc, gcpp = explanations["gradcam"], explanations["gradcampp"]
            rec.target_layer = gc.target_layer
            rec.target_scale = gc.target_scale
            rec.target_cell_gy_gx = gc.target_cell_gy_gx
            rec.target_logit = gc.target_logit
            rec.same_target_for_both_methods = (
                gc.target_layer == gcpp.target_layer
                and gc.target_scale == gcpp.target_scale
                and gc.target_cell_gy_gx == gcpp.target_cell_gy_gx
            )

            rec.gradcam_status = gc.status
            rec.gradcam_is_finite = gc.is_finite
            rec.gradcam_is_nonzero = gc.is_nonzero
            rec.gradcam_time_s = gc.generation_time_s
            rec.gradcampp_status = gcpp.status
            rec.gradcampp_is_finite = gcpp.is_finite
            rec.gradcampp_is_nonzero = gcpp.is_nonzero
            rec.gradcampp_time_s = gcpp.generation_time_s

            inst_id = f"{row.image_id}__inst{gt.instance_index:03d}"

            for method, expl in explanations.items():
                if expl.status != "success" or expl.heatmap_raw is None:
                    if method == "gradcam":
                        rec.error_message = expl.error_message
                    continue
                aligned = viz.check_alignment(expl.heatmap_raw, image_bgr)
                if method == "gradcam":
                    rec.gradcam_alignment_ok = aligned
                else:
                    rec.gradcampp_alignment_ok = aligned

                viz.save_npy(heatmaps_dir / f"{inst_id}__{method}.npy", expl.heatmap_raw)
                viz.save_png(heatmaps_dir / f"{inst_id}__{method}_colorized.png",
                             viz.colorize_heatmap(expl.heatmap_raw))
                if expl.heatmap_overlay is not None:
                    viz.save_png(overlays_dir / f"{inst_id}__{method}_overlay.png",
                                 expl.heatmap_overlay)

            boxed = viz.draw_boxes(
                image_bgr, gt.bbox_xyxy, match.matched_bbox_xyxy,
                gt_label=f"GT:{gt.class_name}",
                pred_label=f"pred:{matched_detection.class_name}:{matched_detection.confidence:.2f}",
            )
            viz.save_png(comparisons_dir / f"{inst_id}__boxed.png", boxed)

            if (gc.status == "success" and gcpp.status == "success"
                    and gc.heatmap_overlay is not None and gcpp.heatmap_overlay is not None):
                panel = viz.build_comparison_panel(boxed, gc.heatmap_overlay, gcpp.heatmap_overlay)
                viz.save_png(comparisons_dir / f"{inst_id}__comparison.png", panel)

            instance_records.append(rec)

        image_records.append({
            "image_id": row.image_id, "complexity_group": row.complexity_group,
            "source_family_id": row.source_family_id,
            "status": "processed", "n_gt_instances": len(gt_instances),
            "n_matched": n_matched, "n_unmatched": n_unmatched,
            "execution_time_s": round(time.time() - img_t0, 3),
        })

    manifests_dir.mkdir(parents=True, exist_ok=True)
    _write_execution_manifest(manifests_dir / "execution_manifest.csv", image_records)
    _write_target_linkage(manifests_dir / "target_linkage.csv", instance_records)

    total_time = time.time() - t_start
    return {
        "checkpoint_check": checkpoint_check,
        "device": DEVICE,
        "python_version": platform.python_version(),
        "n_images_attempted": len(verified_rows),
        "n_images_processed": sum(1 for r in image_records if r["status"] == "processed"),
        "n_instances_total": len(instance_records),
        "n_instances_matched": sum(1 for r in instance_records if r.match_status == "matched"),
        "n_instances_unmatched": sum(1 for r in instance_records if r.match_status == "unmatched"),
        "n_gradcam_success": sum(1 for r in instance_records if r.gradcam_status == "success"),
        "n_gradcampp_success": sum(1 for r in instance_records if r.gradcampp_status == "success"),
        "total_execution_time_s": round(total_time, 3),
        "image_records": image_records,
        "instance_records": instance_records,
    }


def _write_execution_manifest(path: Path, image_records: list[dict]) -> None:
    fieldnames = ["image_id", "complexity_group", "source_family_id", "status",
                  "n_gt_instances", "n_matched", "n_unmatched", "execution_time_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in image_records:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_target_linkage(path: Path, records: list[InstanceRecord]) -> None:
    fieldnames = [
        "image_id", "complexity_group", "source_family_id", "instance_index",
        "gt_class_id", "gt_class_name", "gt_bbox_xyxy", "match_status", "match_reason",
        "matched_detection_id", "matched_confidence", "matched_iou", "matched_bbox_xyxy",
        "target_layer", "target_scale", "target_cell_gy_gx", "target_logit",
        "same_target_for_both_methods",
        "gradcam_status", "gradcam_is_finite", "gradcam_is_nonzero",
        "gradcam_alignment_ok", "gradcam_time_s",
        "gradcampp_status", "gradcampp_is_finite", "gradcampp_is_nonzero",
        "gradcampp_alignment_ok", "gradcampp_time_s", "error_message",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            d = {k: getattr(r, k) for k in fieldnames}
            w.writerow(d)
