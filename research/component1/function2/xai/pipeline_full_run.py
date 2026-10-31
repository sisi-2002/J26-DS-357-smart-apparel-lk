"""F2.3-B orchestrator: 60-image PRIMARY-sample full run (Tasks 3-9).

Runs the SAME, already-verified F2.3-A pipeline (DetectionXAI.detect()/
.explain() from the protected ``detection_xai.py``, the same GT-matching
policy from ``gt_matching.py``, the same visualization helpers from
``visualization.py``) across the frozen 60-image F2.2B primary sample
instead of the 9-image dry-run sample.

Nothing about the matching threshold, target-layer policy, preprocessing,
confidence/NMS thresholds, or heatmap normalization is changed here -- this
module only widens the INPUT sample and extends the OUTPUT manifest schema
per the F2.3-B task brief (Task 7 columns) with honest per-group
bookkeeping and safe checkpoint/resume.

This module computes NO final XAI metric (no localization/faithfulness/
stability/pointing-accuracy/Spatial-IoU) and performs NO complexity-group
statistical comparison -- it only executes detection + matching + paired
Grad-CAM/Grad-CAM++ generation and records structured, traceable output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_XAI_SRC = _REPO_ROOT / "research" / "component1" / "xai_experiments" / "src"
if str(_XAI_SRC) not in sys.path:
    sys.path.insert(0, str(_XAI_SRC))
from detection_xai import DetectionXAI, CLASS_NAMES  # noqa: E402

from research.component1.function2.xai.gt_matching import (  # noqa: E402
    IOU_MATCH_THRESHOLD,
    load_yolo_gt_instances,
    match_gt_to_detections,
)
from research.component1.function2.xai.primary_sample_verification import (  # noqa: E402
    verify_primary_sample,
)
from research.component1.function2.xai import visualization as viz  # noqa: E402
from research.component1.function2.xai.pipeline import verify_e02_checkpoint  # noqa: E402

CONF_THRESHOLD = 0.25
IOU_NMS_THRESHOLD = 0.5
DEVICE = "cpu"
METHODS = ("gradcam", "gradcampp")
SCHEMA_VERSION = "f23b_execution_manifest_v1"

EXECUTION_MANIFEST_FIELDS = [
    "image_id", "source_family_id", "complexity_group", "gt_instance_id",
    "gt_defect_class", "gt_box", "prediction_status", "matched_prediction_id",
    "predicted_class", "prediction_confidence", "matching_iou", "target_layer",
    "target_identifier", "gradcam_status", "gradcampp_status",
    "gradcam_heatmap_path", "gradcampp_heatmap_path", "visualization_paths",
    "failure_reason", "device", "execution_time",
]

TARGET_LINKAGE_FIELDS = [
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

PAIRED_MANIFEST_FIELDS = [
    "gt_instance_id", "image_id", "complexity_group", "source_family_id",
    "gt_defect_class", "target_layer", "target_identifier",
    "gradcam_npy_path", "gradcampp_npy_path",
    "gradcam_shape", "gradcampp_shape", "gradcam_dtype", "gradcampp_dtype",
    "gradcam_finite", "gradcampp_finite", "gradcam_alignment_ok",
    "gradcampp_alignment_ok",
]

FAILURES_FIELDS = [
    "image_id", "complexity_group", "source_family_id", "gt_instance_id",
    "failure_category", "failure_reason", "stage",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _target_identifier(scale, cell) -> str:
    if scale is None or cell is None:
        return ""
    gy, gx = cell
    return f"scale{scale}_gy{gy}_gx{gx}"


@dataclass
class RunResult:
    n_images_attempted: int = 0
    n_images_with_success: int = 0
    n_images_zero_success: int = 0
    n_images_failed_entirely: int = 0
    n_gt_instances: int = 0
    n_matched: int = 0
    n_unmatched: int = 0
    n_gradcam_success: int = 0
    n_gradcampp_success: int = 0
    n_paired_success: int = 0
    per_group: dict = field(default_factory=dict)
    target_layer_usage: dict = field(default_factory=dict)
    target_layer_usage_by_group: dict = field(default_factory=dict)
    total_execution_time_s: float = 0.0
    device: str = DEVICE
    checkpoint_check: dict = field(default_factory=dict)


def _cached_record_files_intact(cached: dict, output_dir: Path) -> bool:
    """True iff every artifact path a cached per-image record claims to have
    written is still present on disk. Used to decide whether a resumed run
    may trust a prior partial run's cached bookkeeping for one image."""
    candidate_paths: list[str] = []
    for r in cached.get("exec_rows", []):
        for key in ("gradcam_heatmap_path", "gradcampp_heatmap_path"):
            if r.get(key):
                candidate_paths.append(r[key])
        vis = r.get("visualization_paths") or ""
        candidate_paths.extend(p for p in vis.split(";") if p)
    for r in cached.get("paired_rows", []):
        for key in ("gradcam_npy_path", "gradcampp_npy_path"):
            if r.get(key):
                candidate_paths.append(r[key])
    for rel in candidate_paths:
        if not (output_dir / rel).is_file():
            return False
    return True


def _load_progress_ledger(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_progress_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def run_full_sample(
    primary_sample_csv: Path,
    split_manifest_csv: Path,
    valid_complexity_manifest_csv: Path,
    dataset_root: Path,
    freeze_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    config_hash: str,
    resume: bool = True,
) -> dict:
    t_start = time.time()
    output_dir = Path(output_dir)
    heatmaps_gradcam_dir = output_dir / "heatmaps" / "gradcam"
    heatmaps_gradcampp_dir = output_dir / "heatmaps" / "gradcampp"
    overlays_dir = output_dir / "overlays"
    comparisons_dir = output_dir / "comparisons"
    manifests_dir = output_dir / "manifests"
    checksums_dir = output_dir / "checksums"
    for d in (heatmaps_gradcam_dir, heatmaps_gradcampp_dir, overlays_dir,
              comparisons_dir, manifests_dir, checksums_dir):
        d.mkdir(parents=True, exist_ok=True)

    checkpoint_check = verify_e02_checkpoint(freeze_manifest_path, checkpoint_path)
    if not checkpoint_check["verified"]:
        raise RuntimeError(
            f"E02 checkpoint SHA256 mismatch -- refusing to run. "
            f"expected={checkpoint_check['expected_sha256']} "
            f"actual={checkpoint_check['actual_sha256']}"
        )

    verified_rows = verify_primary_sample(
        primary_sample_csv, split_manifest_csv, valid_complexity_manifest_csv, dataset_root,
    )

    ledger_path = checksums_dir / "progress_ledger.json"
    ledger = _load_progress_ledger(ledger_path) if resume else {}
    if ledger.get("config_hash") not in (None, config_hash):
        # A prior partial run used a DIFFERENT frozen config -- its outputs
        # cannot be trusted as a resume base. Start clean rather than mixing
        # outputs produced under different settings.
        ledger = {}
    ledger["config_hash"] = config_hash
    ledger.setdefault("completed_image_ids", [])

    xai = DetectionXAI(str(checkpoint_path), device=DEVICE)

    exec_rows: list[dict] = []
    linkage_rows: list[dict] = []
    paired_rows: list[dict] = []
    failure_rows: list[dict] = []

    result = RunResult(device=DEVICE, checkpoint_check=checkpoint_check)
    result.per_group = {g: {"images": 0, "gt_instances": 0, "matched": 0,
                             "unmatched": 0, "gradcam_success": 0,
                             "gradcampp_success": 0, "paired_success": 0,
                             "images_with_success": 0, "images_zero_success": 0}
                         for g in ("Simple", "Moderate", "High")}
    result.target_layer_usage = {}
    result.target_layer_usage_by_group = {g: {} for g in ("Simple", "Moderate", "High")}

    for row in verified_rows:
        result.n_images_attempted += 1
        grp = row.complexity_group
        result.per_group[grp]["images"] += 1

        already_done = row.image_id in ledger["completed_image_ids"]
        if already_done:
            # Trust prior output ONLY if (a) the config hash matches (checked
            # above, whole-ledger granularity) AND (b) every artifact file
            # this cached record claims to have written still exists on disk
            # right now -- a partial/interrupted prior run could have updated
            # the ledger before a file write completed, or a file could have
            # been externally deleted/moved since. If any referenced file is
            # missing, the cached record is NOT trusted and this image is
            # fully reprocessed instead (never silently accepted as valid).
            cached = ledger.get("image_records", {}).get(row.image_id)
            if cached is not None and _cached_record_files_intact(cached, output_dir):
                exec_rows.extend(cached["exec_rows"])
                linkage_rows.extend(cached["linkage_rows"])
                paired_rows.extend(cached["paired_rows"])
                failure_rows.extend(cached["failure_rows"])
                _accumulate_group_stats(result, grp, cached)
                continue
            # No cached detail available, or a referenced file is missing --
            # fall through and recompute this image from scratch.

        img_t0 = time.time()
        image_bgr = cv2.imread(str(row.image_path))
        if image_bgr is None:
            failure_rows.append({
                "image_id": row.image_id, "complexity_group": grp,
                "source_family_id": row.source_family_id, "gt_instance_id": "",
                "failure_category": "image_unreadable",
                "failure_reason": f"cv2.imread returned None for {row.image_path}",
                "stage": "image_load",
            })
            result.n_images_failed_entirely += 1
            result.per_group[grp]["images_zero_success"] += 1
            ledger["completed_image_ids"].append(row.image_id)
            continue

        gt_instances = load_yolo_gt_instances(
            row.label_path, image_bgr.shape[1], image_bgr.shape[0], CLASS_NAMES,
        )

        det_result = xai.detect(image_bgr, conf_threshold=CONF_THRESHOLD,
                                 iou_threshold=IOU_NMS_THRESHOLD)

        img_exec_rows, img_linkage_rows, img_paired_rows, img_failure_rows = [], [], [], []
        n_matched_img = 0
        n_unmatched_img = 0
        n_success_img = 0

        for gt in gt_instances:
            inst_id = f"{row.image_id}__inst{gt.instance_index:03d}"
            match = match_gt_to_detections(gt, det_result.detections, IOU_MATCH_THRESHOLD)
            result.n_gt_instances += 1
            result.per_group[grp]["gt_instances"] += 1

            linkage = {
                "image_id": row.image_id, "complexity_group": grp,
                "source_family_id": row.source_family_id,
                "instance_index": gt.instance_index, "gt_class_id": gt.class_id,
                "gt_class_name": gt.class_name, "gt_bbox_xyxy": gt.bbox_xyxy,
                "match_status": match.status, "match_reason": match.reason,
                "matched_detection_id": match.matched_detection_id,
                "matched_confidence": match.matched_confidence,
                "matched_iou": match.iou, "matched_bbox_xyxy": match.matched_bbox_xyxy,
                "target_layer": "", "target_scale": None, "target_cell_gy_gx": None,
                "target_logit": None, "same_target_for_both_methods": None,
                "gradcam_status": "not_attempted", "gradcam_is_finite": None,
                "gradcam_is_nonzero": None, "gradcam_alignment_ok": None,
                "gradcam_time_s": None, "gradcampp_status": "not_attempted",
                "gradcampp_is_finite": None, "gradcampp_is_nonzero": None,
                "gradcampp_alignment_ok": None, "gradcampp_time_s": None,
                "error_message": "",
            }

            if match.status != "matched":
                n_unmatched_img += 1
                result.n_unmatched += 1
                result.per_group[grp]["unmatched"] += 1
                img_linkage_rows.append(linkage)
                img_exec_rows.append({
                    "image_id": row.image_id, "source_family_id": row.source_family_id,
                    "complexity_group": grp, "gt_instance_id": inst_id,
                    "gt_defect_class": gt.class_name, "gt_box": gt.bbox_xyxy,
                    "prediction_status": "unmatched", "matched_prediction_id": "",
                    "predicted_class": "", "prediction_confidence": "",
                    "matching_iou": match.iou, "target_layer": "",
                    "target_identifier": "", "gradcam_status": "not_attempted",
                    "gradcampp_status": "not_attempted", "gradcam_heatmap_path": "",
                    "gradcampp_heatmap_path": "", "visualization_paths": "",
                    "failure_reason": match.reason, "device": DEVICE,
                    "execution_time": 0.0,
                })
                img_failure_rows.append({
                    "image_id": row.image_id, "complexity_group": grp,
                    "source_family_id": row.source_family_id, "gt_instance_id": inst_id,
                    "failure_category": "unmatched_gt_instance",
                    "failure_reason": match.reason, "stage": "gt_matching",
                })
                continue

            n_matched_img += 1
            result.n_matched += 1
            result.per_group[grp]["matched"] += 1
            matched_detection = next(
                d for d in det_result.detections if d.detection_id == match.matched_detection_id
            )

            inst_t0 = time.time()
            explanations = {}
            for method in METHODS:
                expl = xai.explain(
                    image_bgr, matched_detection, method=method,
                    conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_NMS_THRESHOLD,
                )
                explanations[method] = expl
            inst_elapsed = round(time.time() - inst_t0, 4)

            gc, gcpp = explanations["gradcam"], explanations["gradcampp"]
            linkage["target_layer"] = gc.target_layer
            linkage["target_scale"] = gc.target_scale
            linkage["target_cell_gy_gx"] = gc.target_cell_gy_gx
            linkage["target_logit"] = gc.target_logit
            linkage["same_target_for_both_methods"] = (
                gc.target_layer == gcpp.target_layer
                and gc.target_scale == gcpp.target_scale
                and gc.target_cell_gy_gx == gcpp.target_cell_gy_gx
            )
            linkage["gradcam_status"] = gc.status
            linkage["gradcam_is_finite"] = gc.is_finite
            linkage["gradcam_is_nonzero"] = gc.is_nonzero
            linkage["gradcam_time_s"] = gc.generation_time_s
            linkage["gradcampp_status"] = gcpp.status
            linkage["gradcampp_is_finite"] = gcpp.is_finite
            linkage["gradcampp_is_nonzero"] = gcpp.is_nonzero
            linkage["gradcampp_time_s"] = gcpp.generation_time_s
            if gc.status != "success":
                linkage["error_message"] = gc.error_message
            elif gcpp.status != "success":
                linkage["error_message"] = gcpp.error_message

            target_layer = gc.target_layer
            target_id = _target_identifier(gc.target_scale, gc.target_cell_gy_gx)
            if target_layer:
                result.target_layer_usage[target_layer] = result.target_layer_usage.get(target_layer, 0) + 1
                result.target_layer_usage_by_group[grp][target_layer] = (
                    result.target_layer_usage_by_group[grp].get(target_layer, 0) + 1
                )

            heatmap_paths = {"gradcam": "", "gradcampp": ""}
            vis_paths = []
            method_shapes = {}
            for method, expl in explanations.items():
                method_dir = heatmaps_gradcam_dir if method == "gradcam" else heatmaps_gradcampp_dir
                if expl.status != "success" or expl.heatmap_raw is None:
                    img_failure_rows.append({
                        "image_id": row.image_id, "complexity_group": grp,
                        "source_family_id": row.source_family_id, "gt_instance_id": inst_id,
                        "failure_category": f"{method}_generation_failed_{expl.status}",
                        "failure_reason": expl.error_message, "stage": "explanation",
                    })
                    continue
                aligned = viz.check_alignment(expl.heatmap_raw, image_bgr)
                if method == "gradcam":
                    linkage["gradcam_alignment_ok"] = aligned
                else:
                    linkage["gradcampp_alignment_ok"] = aligned

                npy_path = method_dir / f"{inst_id}__{method}.npy"
                viz.save_npy(npy_path, expl.heatmap_raw)
                png_path = method_dir / f"{inst_id}__{method}_colorized.png"
                viz.save_png(png_path, viz.colorize_heatmap(expl.heatmap_raw))
                heatmap_paths[method] = str(npy_path.relative_to(output_dir))
                method_shapes[method] = (expl.heatmap_raw.shape, str(expl.heatmap_raw.dtype),
                                          bool(np.all(np.isfinite(expl.heatmap_raw))), aligned)
                if expl.heatmap_overlay is not None:
                    overlay_path = overlays_dir / f"{inst_id}__{method}_overlay.png"
                    viz.save_png(overlay_path, expl.heatmap_overlay)
                    vis_paths.append(str(overlay_path.relative_to(output_dir)))
                if method == "gradcam" and expl.status == "success":
                    result.n_gradcam_success += 1
                    result.per_group[grp]["gradcam_success"] += 1
                if method == "gradcampp" and expl.status == "success":
                    result.n_gradcampp_success += 1
                    result.per_group[grp]["gradcampp_success"] += 1

            boxed = viz.draw_boxes(
                image_bgr, gt.bbox_xyxy, match.matched_bbox_xyxy,
                gt_label=f"GT:{gt.class_name}",
                pred_label=f"pred:{matched_detection.class_name}:{matched_detection.confidence:.2f}",
            )
            boxed_path = comparisons_dir / f"{inst_id}__boxed.png"
            viz.save_png(boxed_path, boxed)
            vis_paths.append(str(boxed_path.relative_to(output_dir)))

            paired_ok = (gc.status == "success" and gcpp.status == "success"
                         and gc.heatmap_overlay is not None and gcpp.heatmap_overlay is not None)
            if paired_ok:
                panel = viz.build_comparison_panel(boxed, gc.heatmap_overlay, gcpp.heatmap_overlay)
                panel_path = comparisons_dir / f"{inst_id}__comparison.png"
                viz.save_png(panel_path, panel)
                vis_paths.append(str(panel_path.relative_to(output_dir)))
                result.n_paired_success += 1
                result.per_group[grp]["paired_success"] += 1
                n_success_img += 1
                gshape, gdtype, gfinite, galign = method_shapes["gradcam"]
                ppshape, ppdtype, ppfinite, ppalign = method_shapes["gradcampp"]
                img_paired_rows.append({
                    "gt_instance_id": inst_id, "image_id": row.image_id,
                    "complexity_group": grp, "source_family_id": row.source_family_id,
                    "gt_defect_class": gt.class_name, "target_layer": target_layer,
                    "target_identifier": target_id,
                    "gradcam_npy_path": heatmap_paths["gradcam"],
                    "gradcampp_npy_path": heatmap_paths["gradcampp"],
                    "gradcam_shape": gshape, "gradcampp_shape": ppshape,
                    "gradcam_dtype": gdtype, "gradcampp_dtype": ppdtype,
                    "gradcam_finite": gfinite, "gradcampp_finite": ppfinite,
                    "gradcam_alignment_ok": galign, "gradcampp_alignment_ok": ppalign,
                })

            failure_reason = "" if paired_ok else (
                linkage["error_message"] or "one or both methods did not succeed"
            )
            img_exec_rows.append({
                "image_id": row.image_id, "source_family_id": row.source_family_id,
                "complexity_group": grp, "gt_instance_id": inst_id,
                "gt_defect_class": gt.class_name, "gt_box": gt.bbox_xyxy,
                "prediction_status": "matched",
                "matched_prediction_id": match.matched_detection_id,
                "predicted_class": matched_detection.class_name,
                "prediction_confidence": matched_detection.confidence,
                "matching_iou": match.iou, "target_layer": target_layer,
                "target_identifier": target_id,
                "gradcam_status": gc.status, "gradcampp_status": gcpp.status,
                "gradcam_heatmap_path": heatmap_paths["gradcam"],
                "gradcampp_heatmap_path": heatmap_paths["gradcampp"],
                "visualization_paths": ";".join(vis_paths),
                "failure_reason": failure_reason, "device": DEVICE,
                "execution_time": inst_elapsed,
            })
            img_linkage_rows.append(linkage)

        exec_rows.extend(img_exec_rows)
        linkage_rows.extend(img_linkage_rows)
        paired_rows.extend(img_paired_rows)
        failure_rows.extend(img_failure_rows)

        if n_success_img > 0:
            result.n_images_with_success += 1
            result.per_group[grp]["images_with_success"] += 1
        else:
            result.n_images_zero_success += 1
            result.per_group[grp]["images_zero_success"] += 1

        img_elapsed = round(time.time() - img_t0, 4)
        cached_record = {
            "exec_rows": img_exec_rows, "linkage_rows": img_linkage_rows,
            "paired_rows": img_paired_rows, "failure_rows": img_failure_rows,
            "n_gt_instances": len(gt_instances), "n_matched": n_matched_img,
            "n_unmatched": n_unmatched_img, "n_success_img": n_success_img,
            "complexity_group": grp, "execution_time_s": img_elapsed,
            "target_layer_counts": _count_target_layers(img_linkage_rows),
        }
        ledger.setdefault("image_records", {})[row.image_id] = cached_record
        ledger["completed_image_ids"].append(row.image_id)
        _save_progress_ledger(ledger_path, ledger)

    manifests_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(manifests_dir / "execution_manifest.csv", EXECUTION_MANIFEST_FIELDS, exec_rows)
    _write_csv(manifests_dir / "target_linkage.csv", TARGET_LINKAGE_FIELDS, linkage_rows)
    _write_csv(manifests_dir / "paired_explanations_manifest.csv", PAIRED_MANIFEST_FIELDS, paired_rows)
    _write_csv(manifests_dir / "failures.csv", FAILURES_FIELDS, failure_rows)

    result.total_execution_time_s = round(time.time() - t_start, 3)
    return {
        "result": asdict(result),
        "device": DEVICE,
        "python_version": platform.python_version(),
        "schema_version": SCHEMA_VERSION,
        "exec_rows": exec_rows,
        "linkage_rows": linkage_rows,
        "paired_rows": paired_rows,
        "failure_rows": failure_rows,
    }


def _count_target_layers(linkage_rows: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in linkage_rows:
        if r.get("target_layer"):
            out[r["target_layer"]] = out.get(r["target_layer"], 0) + 1
    return out


def _accumulate_group_stats(result: RunResult, grp: str, cached: dict) -> None:
    result.n_gt_instances += cached["n_gt_instances"]
    result.n_matched += cached["n_matched"]
    result.n_unmatched += cached["n_unmatched"]
    result.per_group[grp]["gt_instances"] += cached["n_gt_instances"]
    result.per_group[grp]["matched"] += cached["n_matched"]
    result.per_group[grp]["unmatched"] += cached["n_unmatched"]
    n_paired = sum(1 for _ in cached["paired_rows"])
    result.n_paired_success += n_paired
    result.per_group[grp]["paired_success"] += n_paired
    n_gc = sum(1 for r in cached["linkage_rows"] if r.get("gradcam_status") == "success")
    n_gcpp = sum(1 for r in cached["linkage_rows"] if r.get("gradcampp_status") == "success")
    result.n_gradcam_success += n_gc
    result.n_gradcampp_success += n_gcpp
    result.per_group[grp]["gradcam_success"] += n_gc
    result.per_group[grp]["gradcampp_success"] += n_gcpp
    for layer, count in cached.get("target_layer_counts", {}).items():
        result.target_layer_usage[layer] = result.target_layer_usage.get(layer, 0) + count
        result.target_layer_usage_by_group[grp][layer] = (
            result.target_layer_usage_by_group[grp].get(layer, 0) + count
        )
    if cached["n_success_img"] > 0:
        result.n_images_with_success += 1
        result.per_group[grp]["images_with_success"] += 1
    else:
        result.n_images_zero_success += 1
        result.per_group[grp]["images_zero_success"] += 1


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
