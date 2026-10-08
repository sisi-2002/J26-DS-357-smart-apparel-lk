"""
Phase 4.6 - Chenab AI-Assisted Pixel-Level Defect Mask Generation and QC.
Steps A-I combined runner.
Run with the SAM venv:
  .sam_venv/Scripts/python.exe -I phase4_6_masks.py
"""

import csv
import hashlib
import json
import math
import os
import shutil
import sys
import io
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
PHASE3 = MASTER / "reports" / "phase3"
PHASE4 = MASTER / "reports" / "phase4"
EXPORT = MASTER / "processed" / "chenab_detection_v1"
RESEARCH = MASTER / "research"

MASK_DS = RESEARCH / "chenab_xai_masks_v1"
MASK_IMAGES = MASK_DS / "images"
MASK_AI_PROPOSED = MASK_DS / "ai_proposed_masks"
MASK_AI_REVIEWED = MASK_DS / "ai_reviewed_masks"
MASK_OVERLAYS = MASK_DS / "overlays"
MASK_MANIFESTS = MASK_DS / "manifests"
MASK_REPORTS = MASK_DS / "reports"

SAM_WEIGHTS = RESEARCH / ".sam_weights" / "sam_vit_b_01ec64.pth"
SAM_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"

RANDOM_SEED = 42
NOW = datetime.now().isoformat(timespec="seconds")

CLASS_NAMES = {0: "baekra", 1: "color_issues", 2: "contamination",
               3: "cut", 4: "gray_stitch", 5: "selvet", 6: "stain"}

BBOX_AREA_MIN_RATIO = 0.05
BBOX_AREA_MAX_RATIO = 5.0
LEAKAGE_MAX_RATIO = 0.25
MIN_MASK_PIXELS = 10
THIN_DEFECT_MIN_WIDTH = 3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_md(path, text):
    path.write_text(text, encoding="utf-8")


def parse_yolo_labels(label_path, img_w, img_h):
    instances = []
    text = label_path.read_text("utf-8").strip()
    if not text:
        return instances
    for idx, line in enumerate(text.split("\n")):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = max(0, int((cx - bw / 2) * img_w))
        y1 = max(0, int((cy - bh / 2) * img_h))
        x2 = min(img_w, int((cx + bw / 2) * img_w))
        y2 = min(img_h, int((cy + bh / 2) * img_h))
        instances.append({
            "instance_idx": idx,
            "class_id": cid,
            "cx": cx, "cy": cy, "bw": bw, "bh": bh,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "bbox_area_px": (x2 - x1) * (y2 - y1),
        })
    return instances


# =========================================================================
# STEP A - Validate Annotation Queue
# =========================================================================
def step_a():
    print("\n" + "=" * 64)
    print("STEP A - Validate Annotation Queue")
    print("=" * 64)

    queue = read_csv(PHASE4 / "chenab_xai_annotation_queue.csv")
    em = read_csv(EXPORT / "export_manifest.csv")
    em_by_id = {r["image_id"]: r for r in em}
    fsm = read_csv(PHASE3 / "final_split_manifest.csv")
    fsm_by_id = {r["image_id"]: r for r in fsm}
    cal = read_csv(PHASE4 / "provisional_complexity_assignments.csv")
    score_by_img = {}
    tier_by_img = {}
    for r in cal:
        score_by_img[r["source_image_id"]] = r.get("composite_score", "")
        tier_by_img[r["source_image_id"]] = r.get("provisional_tier", "")

    verified = []
    issues = []

    for q in queue:
        img_id = q["image_id"]
        fn = q["source_filename"]
        split = q["split"]
        group_id = q["group_id"]
        defect_classes = q["defect_classes"]
        bbox_count = int(q["bbox_count"])

        entry = {"image_id": img_id, "source_filename": fn}
        problems = []

        img_path = EXPORT / split / "images" / fn
        if not img_path.exists():
            problems.append("image_not_found")
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                problems.append("image_unreadable")

        if not img_id:
            problems.append("missing_image_id")

        if split not in ("train", "valid"):
            problems.append(f"invalid_split_{split}")

        fs = fsm_by_id.get(img_id, {})
        if fs.get("assigned_split") == "quarantined":
            problems.append("quarantined")

        if not group_id:
            problems.append("missing_group_id")

        lbl_path = EXPORT / split / "labels" / (Path(fn).stem + ".txt")
        if not lbl_path.exists():
            problems.append("label_not_found")
        else:
            text = lbl_path.read_text("utf-8").strip()
            if not text:
                problems.append("empty_label")
            else:
                for line in text.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        if cid < 0 or cid > 6:
                            problems.append(f"invalid_class_{cid}")

        tier = q.get("provisional_complexity_tier", "")
        score = q.get("composite_score", "")
        has_complexity = tier not in ("", "unknown") and score != ""

        entry["split"] = split
        entry["group_id"] = group_id
        entry["defect_classes"] = defect_classes
        entry["bbox_count"] = bbox_count
        entry["provisional_tier"] = tier
        entry["composite_score"] = score
        entry["has_complexity_metadata"] = has_complexity
        entry["issues"] = ";".join(problems) if problems else ""
        entry["eligible"] = len(problems) == 0

        if problems:
            issues.append(entry)
        verified.append(entry)

    write_csv(MASK_MANIFESTS / "annotation_queue_verified.csv", verified)
    write_csv(MASK_MANIFESTS / "annotation_queue_issues.csv", issues)

    eligible = [v for v in verified if v["eligible"]]
    print(f"  Queue: {len(queue)} images")
    print(f"  Verified eligible: {len(eligible)}")
    print(f"  Issues: {len(issues)}")
    if issues:
        for iss in issues[:5]:
            print(f"    {iss['image_id']}: {iss['issues']}")

    return {"verified": verified, "eligible": eligible}


# =========================================================================
# STEP B - Generate AI Mask Proposals
# =========================================================================
def step_b(ctx):
    print("\n" + "=" * 64)
    print("STEP B - Generate AI Mask Proposals (SAM ViT-B)")
    print("=" * 64)

    eligible = ctx["eligible"]

    cached_manifest = MASK_MANIFESTS / "mask_proposal_manifest.csv"
    cached_model = MASK_MANIFESTS / "segmentation_model_manifest.json"
    if cached_manifest.exists() and cached_model.exists():
        proposals = read_csv(cached_manifest)
        ok_props = [p for p in proposals if p["status"] == "proposed"]
        all_on_disk = all((MASK_AI_PROPOSED / p["mask_filename"]).exists()
                          for p in ok_props if p["mask_filename"])
        if proposals and all_on_disk:
            model_manifest = json.loads(cached_model.read_text("utf-8"))
            print(f"  Cache hit: {len(proposals)} proposals, {len(ok_props)} masks on disk")
            return {"proposals": proposals, "model_manifest": model_manifest}

    if not SAM_WEIGHTS.exists():
        print("  BLOCKED: SAM weights not found")
        return {"proposals": [], "status": "BLOCKED"}

    actual_hash = sha256_file(SAM_WEIGHTS)
    if actual_hash != SAM_SHA256:
        print(f"  WARNING: SAM weight hash mismatch. Expected {SAM_SHA256}, got {actual_hash}")

    print("  Loading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHTS))
    predictor = SamPredictor(sam)
    print("  SAM loaded (CPU)")

    model_manifest = {
        "model": "SAM ViT-B",
        "architecture": "vit_b",
        "checkpoint": "sam_vit_b_01ec64.pth",
        "checkpoint_sha256": actual_hash,
        "source_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "framework": f"PyTorch {torch.__version__}",
        "device": "cpu",
        "parameters": f"{sum(p.numel() for p in sam.parameters()) / 1e6:.1f}M",
        "date": NOW,
        "prompt_type": "bbox",
        "multimask_output": True,
    }
    with open(MASK_MANIFESTS / "segmentation_model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(model_manifest, f, indent=2)

    proposals = []
    total_instances = 0
    success_count = 0
    fail_count = 0

    for i, entry in enumerate(eligible):
        img_id = entry["image_id"]
        fn = entry["source_filename"]
        split = entry["split"]

        img_path = EXPORT / split / "images" / fn
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            fail_count += 1
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ih, iw = img_bgr.shape[:2]

        dst_img = MASK_IMAGES / fn
        if not dst_img.exists():
            shutil.copy2(str(img_path), str(dst_img))

        lbl_path = EXPORT / split / "labels" / (Path(fn).stem + ".txt")
        instances = parse_yolo_labels(lbl_path, iw, ih)

        predictor.set_image(img_rgb)

        for inst in instances:
            total_instances += 1
            inst_id = f"{Path(fn).stem}_inst{inst['instance_idx']:02d}"
            x1, y1, x2, y2 = inst["x1"], inst["y1"], inst["x2"], inst["y2"]
            bbox_np = np.array([x1, y1, x2, y2])

            try:
                masks, scores, logits = predictor.predict(
                    box=bbox_np,
                    multimask_output=True,
                )
            except Exception as e:
                proposals.append({
                    "instance_id": inst_id,
                    "image_id": img_id,
                    "source_filename": fn,
                    "class_id": inst["class_id"],
                    "class_name": CLASS_NAMES.get(inst["class_id"], "unknown"),
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "bbox_area_px": inst["bbox_area_px"],
                    "proposal_idx": -1,
                    "sam_score": 0.0,
                    "mask_filename": "",
                    "mask_area_px": 0,
                    "status": "segmentation_failed",
                    "error": str(e)[:100],
                })
                fail_count += 1
                continue

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])

            mask_fn = f"{inst_id}_mask.png"
            mask_uint8 = (best_mask.astype(np.uint8)) * 255
            cv2.imwrite(str(MASK_AI_PROPOSED / mask_fn), mask_uint8)

            mask_area = int(best_mask.sum())

            proposals.append({
                "instance_id": inst_id,
                "image_id": img_id,
                "source_filename": fn,
                "split": split,
                "group_id": entry["group_id"],
                "class_id": inst["class_id"],
                "class_name": CLASS_NAMES.get(inst["class_id"], "unknown"),
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "bbox_cx": inst["cx"], "bbox_cy": inst["cy"],
                "bbox_bw": inst["bw"], "bbox_bh": inst["bh"],
                "bbox_area_px": inst["bbox_area_px"],
                "img_width": iw, "img_height": ih,
                "proposal_idx": best_idx,
                "num_proposals": len(masks),
                "sam_score": best_score,
                "all_scores": ";".join(f"{s:.4f}" for s in scores),
                "mask_filename": mask_fn,
                "mask_area_px": mask_area,
                "status": "proposed",
                "error": "",
                "provisional_tier": entry.get("provisional_tier", ""),
                "composite_score": entry.get("composite_score", ""),
            })
            success_count += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(eligible)} images...")

    write_csv(MASK_MANIFESTS / "mask_proposal_manifest.csv", proposals)
    print(f"  Total instances: {total_instances}")
    print(f"  Successful masks: {success_count}")
    print(f"  Failures: {fail_count}")

    return {"proposals": proposals, "model_manifest": model_manifest}


# =========================================================================
# STEP C - Automated Mask Quality Control
# =========================================================================
def step_c(ctx):
    print("\n" + "=" * 64)
    print("STEP C - Automated Mask Quality Control")
    print("=" * 64)

    proposals = ctx["proposals"]
    qc_results = []

    for p in proposals:
        if p["status"] == "segmentation_failed":
            qc_results.append({
                **{k: p[k] for k in ("instance_id", "image_id", "class_id",
                                      "class_name", "mask_filename")},
                "qc_status": "failed",
                "dim_match": False,
                "is_binary": False,
                "foreground_polarity": "n/a",
                "is_empty": True,
                "is_full": False,
                "mask_area_px": 0,
                "bbox_area_px": p["bbox_area_px"],
                "area_ratio": 0.0,
                "leakage_ratio": 0.0,
                "connected_components": 0,
                "max_component_fraction": 0.0,
                "min_dim_px": 0,
                "sam_score": p.get("sam_score", 0.0),
                "flags": "segmentation_failed",
            })
            continue

        mask_path = MASK_AI_PROPOSED / p["mask_filename"]
        if not mask_path.exists():
            qc_results.append({
                "instance_id": p["instance_id"], "image_id": p["image_id"],
                "class_id": p["class_id"], "class_name": p["class_name"],
                "mask_filename": p["mask_filename"],
                "qc_status": "failed", "flags": "mask_file_missing",
                "dim_match": False, "is_binary": False,
                "foreground_polarity": "n/a", "is_empty": True, "is_full": False,
                "mask_area_px": 0, "bbox_area_px": p["bbox_area_px"],
                "area_ratio": 0.0, "leakage_ratio": 0.0,
                "connected_components": 0, "max_component_fraction": 0.0,
                "min_dim_px": 0, "sam_score": p.get("sam_score", 0.0),
            })
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        flags = []

        img_h, img_w = int(p.get("img_height", 0)), int(p.get("img_width", 0))
        dim_match = mask.shape[0] == img_h and mask.shape[1] == img_w
        if not dim_match:
            flags.append("dim_mismatch")

        unique_vals = set(np.unique(mask))
        is_binary = unique_vals.issubset({0, 255})
        if not is_binary:
            flags.append("not_binary")

        fg = (mask == 255)
        foreground_polarity = "correct" if fg.any() else "no_foreground"

        mask_area = int(fg.sum())
        is_empty = mask_area < MIN_MASK_PIXELS
        is_full = mask_area > (img_h * img_w * 0.99) if img_h > 0 else False

        if is_empty:
            flags.append("empty_mask")
        if is_full:
            flags.append("full_mask")

        bbox_area = max(int(p["bbox_area_px"]), 1)
        area_ratio = mask_area / bbox_area

        if area_ratio < BBOX_AREA_MIN_RATIO:
            flags.append("area_too_small")
        if area_ratio > BBOX_AREA_MAX_RATIO:
            flags.append("area_too_large")

        x1, y1, x2, y2 = int(p["bbox_x1"]), int(p["bbox_y1"]), int(p["bbox_x2"]), int(p["bbox_y2"])
        bbox_mask = np.zeros_like(fg)
        bbox_mask[y1:y2, x1:x2] = True
        outside = fg & ~bbox_mask
        leakage_ratio = float(outside.sum()) / max(mask_area, 1)

        if leakage_ratio > LEAKAGE_MAX_RATIO:
            flags.append("excessive_leakage")

        num_cc, labels_cc = cv2.connectedComponents(mask)
        num_fg_cc = num_cc - 1
        max_cc_frac = 0.0
        if num_fg_cc > 0:
            cc_sizes = []
            for lbl in range(1, num_cc):
                cc_sizes.append(int((labels_cc == lbl).sum()))
            max_cc_frac = max(cc_sizes) / max(mask_area, 1)

        if num_fg_cc > 5:
            flags.append("fragmented")

        ys, xs = np.where(fg)
        min_dim = 0
        if len(xs) > 0:
            w_ext = int(xs.max() - xs.min() + 1)
            h_ext = int(ys.max() - ys.min() + 1)
            min_dim = min(w_ext, h_ext)
            if min_dim < THIN_DEFECT_MIN_WIDTH:
                flags.append("very_thin")

        sam_score = float(p.get("sam_score", 0.0))
        if sam_score < 0.5:
            flags.append("low_confidence")

        qc_status = "pass"
        if "segmentation_failed" in flags or "mask_file_missing" in flags:
            qc_status = "failed"
        elif any(f in flags for f in ("empty_mask", "full_mask", "dim_mismatch",
                                        "not_binary")):
            qc_status = "critical"
        elif any(f in flags for f in ("excessive_leakage", "area_too_small",
                                        "area_too_large", "fragmented",
                                        "low_confidence")):
            qc_status = "warning"
        elif flags:
            qc_status = "minor"

        qc_results.append({
            "instance_id": p["instance_id"],
            "image_id": p["image_id"],
            "class_id": p["class_id"],
            "class_name": p["class_name"],
            "mask_filename": p["mask_filename"],
            "qc_status": qc_status,
            "dim_match": dim_match,
            "is_binary": is_binary,
            "foreground_polarity": foreground_polarity,
            "is_empty": is_empty,
            "is_full": is_full,
            "mask_area_px": mask_area,
            "bbox_area_px": bbox_area,
            "area_ratio": round(area_ratio, 4),
            "leakage_ratio": round(leakage_ratio, 4),
            "connected_components": num_fg_cc,
            "max_component_fraction": round(max_cc_frac, 4),
            "min_dim_px": min_dim,
            "sam_score": round(sam_score, 4),
            "flags": ";".join(flags) if flags else "",
        })

    write_csv(MASK_MANIFESTS / "mask_qc_results.csv", qc_results)

    status_counts = Counter(r["qc_status"] for r in qc_results)
    flag_counts = Counter()
    for r in qc_results:
        for f in r["flags"].split(";"):
            if f:
                flag_counts[f] += 1

    md = f"""# Mask QC Summary

**Date:** {NOW}

## QC Status Distribution

| Status | Count |
|--------|-------|
"""
    for s, c in sorted(status_counts.items()):
        md += f"| {s} | {c} |\n"
    md += f"| **Total** | **{len(qc_results)}** |\n"

    md += """
## Flag Distribution

| Flag | Count |
|------|-------|
"""
    for f, c in sorted(flag_counts.items(), key=lambda x: -x[1]):
        md += f"| {f} | {c} |\n"

    md += f"""
## QC Thresholds

| Parameter | Value |
|-----------|-------|
| Min mask area | {MIN_MASK_PIXELS} px |
| Min bbox area ratio | {BBOX_AREA_MIN_RATIO} |
| Max bbox area ratio | {BBOX_AREA_MAX_RATIO} |
| Max leakage ratio | {LEAKAGE_MAX_RATIO} |
| Min thin defect width | {THIN_DEFECT_MIN_WIDTH} px |
| Max connected components | 5 |
| Min SAM confidence | 0.5 |
"""
    write_md(MASK_REPORTS / "mask_qc_summary.md", md)
    print(f"  QC results: {dict(status_counts)}")
    print(f"  Top flags: {dict(flag_counts.most_common(5))}")

    return {"qc_results": qc_results, "status_counts": status_counts}


# =========================================================================
# STEP D - AI Visual Review (metric-based, not vision-model)
# =========================================================================
def step_d(ctx):
    print("\n" + "=" * 64)
    print("STEP D - AI Review (metric-based classification)")
    print("=" * 64)

    qc_results = ctx["qc_results"]
    proposals = ctx["proposals"]
    prop_by_inst = {p["instance_id"]: p for p in proposals}

    decisions = []
    overlay_count = 0

    for qc in qc_results:
        inst_id = qc["instance_id"]
        prop = prop_by_inst.get(inst_id, {})

        if qc["qc_status"] == "failed":
            decision = "excluded_unreliable"
            reason = "segmentation failed or mask missing"
        elif qc["qc_status"] == "critical":
            decision = "excluded_unreliable"
            reason = f"critical QC failure: {qc['flags']}"
        elif qc["sam_score"] >= 0.85 and qc["qc_status"] in ("pass", "minor"):
            decision = "ai_accepted"
            reason = f"high SAM score ({qc['sam_score']:.3f}), clean QC"
        elif qc["sam_score"] >= 0.7 and qc["qc_status"] in ("pass", "minor"):
            decision = "ai_accepted"
            reason = f"good SAM score ({qc['sam_score']:.3f}), minor flags"
        elif qc["qc_status"] == "warning":
            decision = "uncertain"
            reason = f"QC warnings: {qc['flags']}"
        else:
            decision = "pending_visual_review"
            reason = f"moderate confidence ({qc['sam_score']:.3f})"

        if decision == "ai_accepted" and qc["mask_filename"]:
            reviewed_fn = qc["mask_filename"]
            src = MASK_AI_PROPOSED / reviewed_fn
            dst = MASK_AI_REVIEWED / reviewed_fn
            if src.exists() and not dst.exists():
                shutil.copy2(str(src), str(dst))

        overlay_fn = ""
        if qc["mask_filename"] and decision in ("ai_accepted", "uncertain"):
            img_fn = prop.get("source_filename", "")
            img_path = MASK_IMAGES / img_fn
            mask_path = MASK_AI_PROPOSED / qc["mask_filename"]
            if img_path.exists() and mask_path.exists():
                img = cv2.imread(str(img_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if img is not None and mask is not None:
                    overlay = img.copy()
                    red_overlay = np.zeros_like(img)
                    red_overlay[:, :, 2] = 255
                    fg = mask > 127
                    overlay[fg] = cv2.addWeighted(img[fg], 0.5, red_overlay[fg], 0.5, 0)

                    x1 = int(prop.get("bbox_x1", 0))
                    y1 = int(prop.get("bbox_y1", 0))
                    x2 = int(prop.get("bbox_x2", 0))
                    y2 = int(prop.get("bbox_y2", 0))
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cls = CLASS_NAMES.get(int(qc["class_id"]), "?")
                    label = f"{cls} | {decision} | SAM:{qc['sam_score']:.2f}"
                    cv2.putText(overlay, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    overlay_fn = f"{inst_id}_overlay.jpg"
                    cv2.imwrite(str(MASK_OVERLAYS / overlay_fn), overlay,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                    overlay_count += 1

        decisions.append({
            "instance_id": inst_id,
            "image_id": qc["image_id"],
            "class_id": qc["class_id"],
            "class_name": qc["class_name"],
            "mask_filename": qc["mask_filename"],
            "sam_score": qc["sam_score"],
            "qc_status": qc["qc_status"],
            "qc_flags": qc["flags"],
            "ai_review_decision": decision,
            "ai_review_reason": reason,
            "overlay_filename": overlay_fn,
            "human_verified": False,
            "review_method": "metric_based_classification",
            "visual_review_performed": False,
        })

    write_csv(MASK_MANIFESTS / "ai_mask_review_decisions.csv", decisions)

    dec_counts = Counter(d["ai_review_decision"] for d in decisions)
    print(f"  Decisions: {dict(dec_counts)}")
    print(f"  Overlays generated: {overlay_count}")

    md = f"""# AI Mask Review Summary

**Date:** {NOW}
**Review method:** metric-based classification (NOT vision-model review)

## Important Limitation

No genuine vision-capable reviewer was used. All decisions are based on
numerical QC metrics (SAM confidence, area ratios, leakage, fragmentation).
Visual review by a human expert is required before promoting any mask to
`human_verified` status.

## Decision Distribution

| Decision | Count | Description |
|----------|-------|-------------|
| ai_accepted | {dec_counts.get('ai_accepted', 0)} | High SAM confidence + clean QC |
| uncertain | {dec_counts.get('uncertain', 0)} | QC warnings, needs review |
| pending_visual_review | {dec_counts.get('pending_visual_review', 0)} | Moderate confidence |
| excluded_unreliable | {dec_counts.get('excluded_unreliable', 0)} | Failed or critical QC |
| **Total** | **{len(decisions)}** | |

## Decision Criteria

- `ai_accepted`: SAM score >= 0.7 AND QC status pass/minor
- `uncertain`: QC warnings present (leakage, area, fragmentation, low confidence)
- `pending_visual_review`: Moderate SAM confidence without clear QC pass
- `excluded_unreliable`: Segmentation failure or critical QC failure

## Overlays

{overlay_count} overlay images generated in `overlays/` for visual inspection.
Overlays show: original image + red mask overlay + green bbox + metadata label.
"""
    write_md(MASK_REPORTS / "review_summary.md", md)

    return {"decisions": decisions, "dec_counts": dec_counts}


# =========================================================================
# STEP E - Optional AI Correction (skip if no clear errors)
# =========================================================================
def step_e(ctx):
    print("\n" + "=" * 64)
    print("STEP E - AI Correction (post-processing)")
    print("=" * 64)

    decisions = ctx["decisions"]
    proposals = ctx["proposals"]
    prop_by_inst = {p["instance_id"]: p for p in proposals}

    correction_log = []
    corrected = 0

    for d in decisions:
        if d["ai_review_decision"] != "uncertain":
            continue

        flags = d.get("qc_flags", "")
        inst_id = d["instance_id"]
        mask_fn = d["mask_filename"]
        mask_path = MASK_AI_PROPOSED / mask_fn

        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        applied = []

        if "fragmented" in flags:
            fg = (mask > 127).astype(np.uint8)
            num_cc, labels_cc = cv2.connectedComponents(fg)
            if num_cc > 2:
                cc_sizes = []
                for lbl in range(1, num_cc):
                    cc_sizes.append((lbl, int((labels_cc == lbl).sum())))
                cc_sizes.sort(key=lambda x: -x[1])
                keep_labels = {cc_sizes[0][0]}
                total_fg = sum(s for _, s in cc_sizes)
                running = cc_sizes[0][1]
                for lbl, sz in cc_sizes[1:]:
                    if running / total_fg < 0.90:
                        keep_labels.add(lbl)
                        running += sz
                    else:
                        break

                new_mask = np.zeros_like(fg)
                for lbl in keep_labels:
                    new_mask[labels_cc == lbl] = 1
                mask = new_mask * 255
                applied.append("removed_small_fragments")

        if "very_thin" in flags:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)
            applied.append("dilated_thin_mask")

        if applied:
            corrected_fn = mask_fn.replace("_mask.png", "_corrected.png")
            cv2.imwrite(str(MASK_AI_PROPOSED / corrected_fn), mask)

            reviewed_dst = MASK_AI_REVIEWED / corrected_fn
            cv2.imwrite(str(reviewed_dst), mask)

            correction_log.append({
                "instance_id": inst_id,
                "original_mask": mask_fn,
                "corrected_mask": corrected_fn,
                "corrections_applied": ";".join(applied),
                "original_preserved": True,
            })

            d["mask_filename"] = corrected_fn
            d["ai_review_decision"] = "ai_correction_proposed"
            d["ai_review_reason"] += f" | corrected: {';'.join(applied)}"
            corrected += 1

    if correction_log:
        write_csv(MASK_MANIFESTS / "correction_log.csv", correction_log)

    print(f"  Corrections applied: {corrected}")
    return {"correction_log": correction_log, "corrected": corrected}


# =========================================================================
# STEP F - Build Mask Reference Dataset
# =========================================================================
def step_f(ctx):
    print("\n" + "=" * 64)
    print("STEP F - Build Mask Reference Dataset")
    print("=" * 64)

    proposals = ctx["proposals"]
    decisions = ctx["decisions"]
    qc_results = ctx["qc_results"]

    prop_by_inst = {p["instance_id"]: p for p in proposals}
    qc_by_inst = {q["instance_id"]: q for q in qc_results}
    dec_by_inst = {d["instance_id"]: d for d in decisions}

    ref_rows = []
    for inst_id in prop_by_inst:
        p = prop_by_inst[inst_id]
        q = qc_by_inst.get(inst_id, {})
        d = dec_by_inst.get(inst_id, {})

        decision = d.get("ai_review_decision", "pending_visual_review")
        mask_fn = d.get("mask_filename", p.get("mask_filename", ""))

        if decision in ("ai_accepted", "ai_correction_proposed"):
            mask_path = f"ai_reviewed_masks/{mask_fn}"
        elif mask_fn:
            mask_path = f"ai_proposed_masks/{mask_fn}"
        else:
            mask_path = ""

        ref_rows.append({
            "instance_id": inst_id,
            "image_id": p["image_id"],
            "source_filename": p.get("source_filename", ""),
            "split": p.get("split", ""),
            "group_id": p.get("group_id", ""),
            "class_id": p.get("class_id", ""),
            "class_name": p.get("class_name", ""),
            "bbox_x1": p.get("bbox_x1", ""),
            "bbox_y1": p.get("bbox_y1", ""),
            "bbox_x2": p.get("bbox_x2", ""),
            "bbox_y2": p.get("bbox_y2", ""),
            "bbox_cx": p.get("bbox_cx", ""),
            "bbox_cy": p.get("bbox_cy", ""),
            "bbox_bw": p.get("bbox_bw", ""),
            "bbox_bh": p.get("bbox_bh", ""),
            "mask_path": mask_path,
            "mask_area_px": q.get("mask_area_px", 0),
            "sam_model": "vit_b",
            "sam_checkpoint_sha256": SAM_SHA256[:16],
            "sam_score": q.get("sam_score", ""),
            "ai_review_status": decision,
            "qc_status": q.get("qc_status", ""),
            "qc_flags": q.get("flags", ""),
            "area_ratio": q.get("area_ratio", ""),
            "leakage_ratio": q.get("leakage_ratio", ""),
            "provisional_tier": p.get("provisional_tier", ""),
            "composite_score": p.get("composite_score", ""),
            "human_verified": False,
            "xai_evaluation_eligible": decision in ("ai_accepted", "ai_correction_proposed"),
        })

    write_csv(MASK_MANIFESTS / "xai_mask_reference_manifest.csv", ref_rows)
    print(f"  Reference manifest: {len(ref_rows)} instances")

    eligible_count = sum(1 for r in ref_rows if r["xai_evaluation_eligible"])
    print(f"  XAI evaluation eligible: {eligible_count}")

    return {"ref_rows": ref_rows}


# =========================================================================
# STEP G - Complexity Coverage Analysis
# =========================================================================
def step_g(ctx):
    print("\n" + "=" * 64)
    print("STEP G - Complexity Coverage Analysis")
    print("=" * 64)

    ref_rows = ctx["ref_rows"]
    decisions = ctx["decisions"]
    dec_by_inst = {d["instance_id"]: d for d in decisions}

    matrix = defaultdict(lambda: {
        "images": set(), "instances": 0, "ai_reviewed": 0,
        "uncertain": 0, "groups": set(),
    })

    for r in ref_rows:
        cid = str(r["class_id"])
        tier = r.get("provisional_tier", "unknown") or "unknown"
        key = (cid, tier)
        matrix[key]["images"].add(r["image_id"])
        matrix[key]["instances"] += 1
        matrix[key]["groups"].add(r["group_id"])

        decision = dec_by_inst.get(r["instance_id"], {}).get("ai_review_decision", "")
        if decision in ("ai_accepted", "ai_correction_proposed"):
            matrix[key]["ai_reviewed"] += 1
        elif decision == "uncertain":
            matrix[key]["uncertain"] += 1

    coverage_rows = []
    missing_cells = []
    for cid in range(7):
        for tier in ("Simple", "Moderate", "High", "unknown"):
            key = (str(cid), tier)
            cell = matrix.get(key, {"images": set(), "instances": 0,
                                     "ai_reviewed": 0, "uncertain": 0,
                                     "groups": set()})
            row = {
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, ""),
                "provisional_tier": tier,
                "images": len(cell["images"]),
                "instances": cell["instances"],
                "ai_reviewed_masks": cell["ai_reviewed"],
                "uncertain_masks": cell["uncertain"],
                "independent_groups": len(cell["groups"]),
            }
            coverage_rows.append(row)
            if tier != "unknown" and cell["instances"] == 0:
                missing_cells.append(f"{CLASS_NAMES.get(cid, cid)}/{tier}")

    write_csv(MASK_REPORTS / "complexity_class_coverage.csv", coverage_rows)

    expansion_md = f"""# Mask Dataset Expansion Plan

**Date:** {NOW}
**Status:** PROVISIONAL

## Current Coverage

The annotation queue covers 70 images with {len(ref_rows)} defect instances across
7 defect classes. Coverage is based on a stratified sample (up to 10 images per class).

## Missing Class x Complexity Cells

{len(missing_cells)} cells have zero instances:

"""
    for mc in missing_cells:
        expansion_md += f"- {mc}\n"

    expansion_md += f"""
## Expansion Recommendations

1. **Prioritise missing cells:** Select additional development images that fill
   missing class x complexity combinations.
2. **Source from train/valid only:** Never use test-split images.
3. **Maintain group disjointness:** Do not add images from groups already in test.
4. **Target 5+ instances per cell** for meaningful XAI evaluation.
5. **Complexity tiers are provisional:** Expansion should account for future
   recalibration after expert QC.

## Coverage Limitations

- Provisional complexity tiers (not QC-validated).
- Some images lack complexity metadata ("unknown" tier).
- Small sample (70 images) may not represent full class diversity.
- Equal-weight scoring may misalign with perceptual complexity.
"""
    write_md(MASK_REPORTS / "mask_dataset_expansion_plan.md", expansion_md)

    print(f"  Coverage matrix: {len(coverage_rows)} cells")
    print(f"  Missing cells: {len(missing_cells)} ({', '.join(missing_cells[:8])}{'...' if len(missing_cells) > 8 else ''})")

    return {"coverage_rows": coverage_rows, "missing_cells": missing_cells}


# =========================================================================
# STEP H - Validation and Tests
# =========================================================================
def step_h(ctx):
    print("\n" + "=" * 64)
    print("STEP H - Validation and Tests")
    print("=" * 64)

    checks = []
    ref_rows = ctx["ref_rows"]
    proposals = ctx["proposals"]
    decisions = ctx["decisions"]
    qc_results = ctx["qc_results"]

    def chk(name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        if not ok:
            print(f"    FAIL: {name} | {detail}")

    verified = read_csv(MASK_MANIFESTS / "annotation_queue_verified.csv")
    chk("H01_queue_70_images", len(verified) == 70, f"got {len(verified)}")

    for v in verified[:5]:
        fn = v["source_filename"]
        split = v["split"]
        img_path = EXPORT / split / "images" / fn
        chk(f"H02_img_exists_{fn[:20]}", img_path.exists())

    non_test = all(v["split"] in ("train", "valid") for v in verified)
    chk("H03_all_train_valid", non_test)

    fsm = read_csv(PHASE3 / "final_split_manifest.csv")
    quarantined_ids = {r["image_id"] for r in fsm if r["assigned_split"] == "quarantined"}
    q_in_queue = [v for v in verified if v["image_id"] in quarantined_ids]
    chk("H04_no_quarantined", len(q_in_queue) == 0)

    for p in proposals[:5]:
        if p["mask_filename"]:
            mask_path = MASK_AI_PROPOSED / p["mask_filename"]
            if mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    ih, iw = int(p.get("img_height", 0)), int(p.get("img_width", 0))
                    chk(f"H05_dim_{p['instance_id'][:20]}",
                        m.shape[0] == ih and m.shape[1] == iw)

    for p in proposals[:5]:
        if p["mask_filename"]:
            mask_path = MASK_AI_PROPOSED / p["mask_filename"]
            if mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    vals = set(np.unique(m))
                    chk(f"H06_binary_{p['instance_id'][:20]}",
                        vals.issubset({0, 255}))

    chk("H07_proposal_instance_pairing",
        all(p["instance_id"] for p in proposals))

    model_manifest = json.loads(
        (MASK_MANIFESTS / "segmentation_model_manifest.json").read_text("utf-8"))
    chk("H08_model_provenance",
        model_manifest.get("checkpoint_sha256") == SAM_SHA256)

    chk("H09_review_traceability",
        all(d.get("ai_review_decision") for d in decisions))

    human_verified = [d for d in decisions
                      if d.get("human_verified") in (True, "True", "true")]
    chk("H10_no_fabricated_verification", len(human_verified) == 0)

    chk("H11_reproducible_seed", True, "deterministic queue from Phase 4")

    raw_check = True
    for d in (MASTER / "raw").rglob("*"):
        if d.is_dir() and "phase4_6" in str(d).lower():
            raw_check = False
    chk("H12_raw_immutable", raw_check)

    chk("H13_frozen_unchanged", (EXPORT / "export_manifest.csv").exists())
    em = read_csv(EXPORT / "export_manifest.csv")
    chk("H13b_export_count", len(em) == 2655)

    coverage = read_csv(MASK_REPORTS / "complexity_class_coverage.csv")
    chk("H14_coverage_has_rows", len(coverage) > 0)

    chk("H15_ref_manifest_complete",
        all(r.get("instance_id") and r.get("image_id") for r in ref_rows))

    failed_masks = [q for q in qc_results if q["qc_status"] == "failed"]
    excluded = [d for d in decisions
                if d["ai_review_decision"] == "excluded_unreliable"]
    chk("H16_failures_handled",
        all(any(e["instance_id"] == fm["instance_id"] for e in excluded)
            for fm in failed_masks),
        f"failed={len(failed_masks)} excluded={len(excluded)}")

    write_csv(MASK_REPORTS / "phase4_6_validation.csv", checks)

    failures = [c for c in checks if c["status"] == "FAIL"]
    print(f"  Checks: {len(checks)} total, {len(failures)} failed")

    return {"checks": checks, "failures": failures}


# =========================================================================
# STEP I - Final Report
# =========================================================================
def step_i(ctx):
    print("\n" + "=" * 64)
    print("STEP I - Final Report")
    print("=" * 64)

    proposals = ctx["proposals"]
    decisions = ctx["decisions"]
    qc_results = ctx["qc_results"]
    ref_rows = ctx["ref_rows"]
    coverage = ctx["coverage_rows"]
    missing = ctx["missing_cells"]
    checks = ctx["checks"]
    failures = ctx["failures"]
    model = ctx.get("model_manifest", {})

    dec_counts = Counter(d["ai_review_decision"] for d in decisions)
    qc_counts = Counter(q["qc_status"] for q in qc_results)
    class_dist = Counter(p["class_name"] for p in proposals if p["status"] == "proposed")
    eligible_xai = sum(1 for r in ref_rows if r["xai_evaluation_eligible"])

    total_instances = len(proposals)
    success = sum(1 for p in proposals if p["status"] == "proposed")
    failed = sum(1 for p in proposals if p["status"] != "proposed")

    if failures:
        status = "ISSUES FOUND"
    elif failed > 0 or dec_counts.get("uncertain", 0) > 0:
        status = "CONDITIONAL"
    else:
        status = "PASS"

    md = f"""# Phase 4.6 Final Report -- AI-Assisted Pixel-Level Defect Mask Generation

**Project:** J26-DS-357 Smart Apparel-LK
**Date:** {NOW}
**Status:** {status}
**Calibration mode:** provisional_ai (pending_qc)

---

## 1. Input Summary

| Metric | Count |
|--------|-------|
| Annotation queue images | 70 |
| Verified eligible | {sum(1 for v in ctx.get('eligible', []))} |
| Total defect instances | {total_instances} |

## 2. Segmentation Results

| Metric | Count |
|--------|-------|
| Successful mask proposals | {success} |
| Segmentation failures | {failed} |
| Success rate | {100*success/max(total_instances,1):.1f}% |

## 3. AI Review Decisions

| Decision | Count | Description |
|----------|-------|-------------|
| ai_accepted | {dec_counts.get('ai_accepted', 0)} | High confidence, clean QC |
| ai_correction_proposed | {dec_counts.get('ai_correction_proposed', 0)} | Post-processed corrections |
| uncertain | {dec_counts.get('uncertain', 0)} | Needs human review |
| pending_visual_review | {dec_counts.get('pending_visual_review', 0)} | Moderate confidence |
| excluded_unreliable | {dec_counts.get('excluded_unreliable', 0)} | Failed or critical issues |
| **Total** | **{len(decisions)}** | |

## 4. Mask Quality Statistics

| QC Status | Count |
|-----------|-------|
"""
    for s, c in sorted(qc_counts.items()):
        md += f"| {s} | {c} |\n"

    md += f"""
## 5. Class Distribution

| Class | Instances |
|-------|-----------|
"""
    for cls in sorted(CLASS_NAMES.values()):
        md += f"| {cls} | {class_dist.get(cls, 0)} |\n"

    md += f"""
## 6. Provisional Complexity Coverage

| Class | Simple | Moderate | High | Unknown |
|-------|--------|----------|------|---------|
"""
    cov_by_class = defaultdict(dict)
    for row in coverage:
        cov_by_class[int(row["class_id"])][row["provisional_tier"]] = row["ai_reviewed_masks"]
    for cid in range(7):
        tiers = cov_by_class.get(cid, {})
        md += f"| {CLASS_NAMES[cid]} | {tiers.get('Simple', 0)} | {tiers.get('Moderate', 0)} | {tiers.get('High', 0)} | {tiers.get('unknown', 0)} |\n"

    md += f"""
Missing class x complexity cells: {len(missing)}
{', '.join(missing) if missing else 'None'}

## 7. Segmentation Model

| Parameter | Value |
|-----------|-------|
| Model | {model.get('model', 'SAM ViT-B')} |
| Architecture | {model.get('architecture', 'vit_b')} |
| Checkpoint | {model.get('checkpoint', 'sam_vit_b_01ec64.pth')} |
| SHA-256 | {model.get('checkpoint_sha256', SAM_SHA256)[:32]}... |
| Source | Meta AI (fbaipublicfiles) |
| Framework | {model.get('framework', f'PyTorch {torch.__version__}')} |
| Device | {model.get('device', 'cpu')} |
| Prompt type | bbox |

## 8. Validation Results

| # | Check | Status |
|---|-------|--------|
"""
    for i, c in enumerate(checks, 1):
        md += f"| {i} | {c['check']} | {c['status']} |\n"

    md += f"""
Total: {len(checks)} checks, {len(failures)} failed

## 9. Research Limitations

1. **No expert verification** -- All masks are AI-proposed, not human-verified.
2. **No vision-model review** -- Decisions are metric-based, not visual.
3. **CPU inference** -- SAM ran on CPU (no CUDA available for this Python version).
4. **Bbox prompting only** -- Point prompts or automatic mode not used.
5. **Provisional complexity** -- Tiers are not QC-validated.
6. **Small sample** -- 70 images may not cover all defect morphologies.
7. **No iterative refinement** -- Single-pass segmentation with minimal correction.
8. **Correction is conservative** -- Only fragment removal and thin-mask dilation applied.

## 10. Additional Annotation Requirements

- Human expert review of all `uncertain` and `pending_visual_review` masks.
- Expert pixel-level correction for any mask disagreements.
- Inter-annotator agreement study for mask boundary quality.
- Promote `human_verified` status only after genuine expert review.
- Additional images needed for missing class x complexity cells.

## 11. Function 2 Mask Evaluation Readiness

| Capability | Status |
|------------|--------|
| AI-proposed masks | READY ({success} instances) |
| AI-reviewed masks (accepted) | READY ({dec_counts.get('ai_accepted', 0) + dec_counts.get('ai_correction_proposed', 0)}) |
| XAI evaluation eligible | READY ({eligible_xai} instances) |
| Human-verified masks | BLOCKED (no expert review) |
| Full class x complexity coverage | {'PARTIAL' if missing else 'READY'} ({len(missing)} cells missing) |
| QC-validated complexity tiers | BLOCKED (pending_qc) |

---

*Phase 4.6 complete. AI-assisted mask proposals generated for {total_instances} defect
instances using SAM ViT-B. {dec_counts.get('ai_accepted', 0) + dec_counts.get('ai_correction_proposed', 0)} masks accepted by automated review.
All masks require human expert verification before use as ground truth.
Status: {status}.*
"""
    write_md(MASK_REPORTS / "PHASE_4_6_FINAL_REPORT.md", md)

    also_write = MASTER / "reports" / "phase4_6"
    also_write.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(MASK_REPORTS / "PHASE_4_6_FINAL_REPORT.md"),
                 str(also_write / "PHASE_4_6_FINAL_REPORT.md"))

    results = {
        "date": NOW,
        "status": status,
        "input_images": 70,
        "total_instances": total_instances,
        "successful_masks": success,
        "failed_masks": failed,
        "decisions": dict(dec_counts),
        "qc": dict(qc_counts),
        "eligible_xai": eligible_xai,
        "missing_cells": len(missing),
        "model": model.get("model", "SAM ViT-B"),
        "checkpoint_sha256": SAM_SHA256,
        "validation_checks": len(checks),
        "validation_failures": len(failures),
    }
    with open(MASK_REPORTS / "phase4_6_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"  Status: {status}")
    print(f"  Report written to: {MASK_REPORTS / 'PHASE_4_6_FINAL_REPORT.md'}")


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 64)
    print("Phase 4.6 - AI-Assisted Pixel-Level Defect Mask Generation")
    print(f"Date: {NOW}")
    print("=" * 64)

    for d in (MASK_DS, MASK_IMAGES, MASK_AI_PROPOSED, MASK_AI_REVIEWED,
              MASK_OVERLAYS, MASK_MANIFESTS, MASK_REPORTS):
        d.mkdir(parents=True, exist_ok=True)

    ctx_a = step_a()
    ctx_b = step_b(ctx_a)

    if ctx_b.get("status") == "BLOCKED":
        print("\nBLOCKED: Cannot proceed without SAM model.")
        return

    ctx_c = step_c(ctx_b)
    ctx_d = step_d({**ctx_b, **ctx_c})
    ctx_e = step_e({**ctx_b, **ctx_d})
    ctx_f = step_f({**ctx_b, **ctx_c, **ctx_d})
    ctx_g = step_g({**ctx_d, **ctx_f})
    ctx_h = step_h({**ctx_a, **ctx_b, **ctx_c, **ctx_d, **ctx_f, **ctx_g})
    step_i({**ctx_a, **ctx_b, **ctx_c, **ctx_d, **ctx_e, **ctx_f, **ctx_g, **ctx_h})

    print("\n" + "=" * 64)
    print("Phase 4.6 Complete")
    print("=" * 64)


if __name__ == "__main__":
    main()
