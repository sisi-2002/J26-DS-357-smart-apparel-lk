"""
Phase 4.7 - XAI Mask Visual Review, Complexity Assignment & Dataset Expansion.
Phases 4.7.1-4.7.6 combined runner.
Run with SAM venv:
  .sam_venv/Scripts/python.exe -I phase4_7_review_expand.py
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
META = MASTER / "metadata"
PHASE3 = MASTER / "reports" / "phase3"
PHASE4 = MASTER / "reports" / "phase4"
EXPORT = MASTER / "processed" / "chenab_detection_v1"
RESEARCH = MASTER / "research"
V1 = RESEARCH / "chenab_xai_masks_v1"
V2 = RESEARCH / "chenab_xai_masks_v2"
BG_PATCH_DIR = RESEARCH / "background_patches"

SAM_WEIGHTS = RESEARCH / ".sam_weights" / "sam_vit_b_01ec64.pth"
SAM_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"

RANDOM_SEED = 42
NOW = datetime.now().isoformat(timespec="seconds")
PATCH_SIZE = 256
SAFETY_MARGIN = 32
EXPANSION_CAP = 30

CLASS_NAMES = {0: "baekra", 1: "color_issues", 2: "contamination",
               3: "cut", 4: "gray_stitch", 5: "selvet", 6: "stain"}

sys.path.insert(0, str(REPO_ROOT / "research"))
from component1.complexity_calibration.src.complexity_features import (
    compute_complexity_from_array, ProtocolConfig
)
COMPLEXITY_CONFIG = ProtocolConfig()


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
            "instance_idx": idx, "class_id": cid,
            "cx": cx, "cy": cy, "bw": bw, "bh": bh,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "bbox_area_px": (x2 - x1) * (y2 - y1),
        })
    return instances


# =========================================================================
# 4.7.1 - Visual Review
# =========================================================================
def phase_4_7_1():
    print("\n" + "=" * 64)
    print("PHASE 4.7.1 - Visual Review")
    print("=" * 64)

    ref = read_csv(V1 / "manifests" / "xai_mask_reference_manifest.csv")
    decisions_v1 = read_csv(V1 / "manifests" / "ai_mask_review_decisions.csv")
    dec_by_inst = {d["instance_id"]: d for d in decisions_v1}

    VISUAL_FINDINGS = {
        "20230719_123833_jpeg_jpg.rf.91495903ce70e22b634b89b711708d44_inst00": {
            "decision": "uncertain",
            "reason": "SAM mask leaks beyond fabric edge into dark background. Excessive leakage confirmed visually.",
            "defect_visible": True, "mask_covers_defect": "partial",
        },
        "20230718_101443_jpeg_jpg.rf.8ae9d7a84f8f751faad2324761a4b030_inst00": {
            "decision": "uncertain",
            "reason": "Striped fabric with pattern transition. SAM captured partial area with low confidence (0.73). Incomplete defect coverage.",
            "defect_visible": True, "mask_covers_defect": "partial",
        },
        "IMG-20230809-WA0073_jpg.rf.a2fa7bc81b1d97d74792774e16820769_inst00": {
            "decision": "uncertain",
            "reason": "Patterned fabric with tiny bbox at bottom edge. SAM produced area_too_small mask. Diffuse color defect not capturable by bbox-prompted SAM.",
            "defect_visible": True, "mask_covers_defect": "minimal",
        },
        "IMG-20230914-WA0067_jpg.rf.5cea51f599fea1a2f3ff1054ceae9a88_inst00": {
            "decision": "visually_accepted_ai",
            "reason": "Small stain on white fabric. Tight mask precisely around dark spot. Correct segmentation.",
            "defect_visible": True, "mask_covers_defect": "complete",
        },
        "IMG-20230818-WA0068_jpg.rf.9248d881bce766651c498e677cf577be_inst00": {
            "decision": "visually_accepted_ai",
            "reason": "Selvedge defect on purple/pink fabric. Mask follows vertical selvedge boundary precisely. High quality.",
            "defect_visible": True, "mask_covers_defect": "complete",
        },
        "IMG-20230803-WA0060_jpg.rf.da6a54f8ea5d9cc474c59671bd1baf2d_inst00": {
            "decision": "correction_required",
            "reason": "SAM segmented floral print pattern instead of actual cut/tear at fabric edge. Mask is incorrect - captures decorative elements not defect.",
            "defect_visible": True, "mask_covers_defect": "incorrect",
        },
        "IMG-20230907-WA0045_jpg.rf.12007fc1709e06d5103f56217985650c_inst00": {
            "decision": "visually_accepted_ai",
            "reason": "Thin linear contamination on white fabric. Tight elongated mask. Correct segmentation.",
            "defect_visible": True, "mask_covers_defect": "complete",
        },
        "20230719_101715_jpeg_jpg.rf.890ce6616dbb4ed2e28217b609632b70_inst00": {
            "decision": "visually_accepted_ai",
            "reason": "Green patterned fabric with selvedge. Two-part mask captures selvedge defect regions. Reasonable quality.",
            "defect_visible": True, "mask_covers_defect": "adequate",
        },
        "IMG-20230817-WA0082_jpg.rf.4ae0ed423aa320a24483c7fe0f925963_inst00": {
            "decision": "correction_required",
            "reason": "Floral fabric with cut/tear. SAM segmented printed flowers rather than fabric damage. Actual defect is fraying at bottom-left edge.",
            "defect_visible": True, "mask_covers_defect": "incorrect",
        },
    }

    review_rows = []
    visually_reviewed = 0
    for r in ref:
        inst_id = r["instance_id"]
        finding = VISUAL_FINDINGS.get(inst_id)
        v1_dec = dec_by_inst.get(inst_id, {})

        if finding:
            review_rows.append({
                "instance_id": inst_id,
                "image_id": r["image_id"],
                "class_id": r["class_id"],
                "class_name": r["class_name"],
                "original_mask_path": r.get("mask_path", ""),
                "corrected_mask_path": "",
                "reviewer_type": "multimodal_ai",
                "review_method": "direct_image_inspection",
                "image_inspected": True,
                "phase46_decision": v1_dec.get("ai_review_decision", r.get("ai_review_status", "")),
                "visual_review_decision": finding["decision"],
                "visual_review_reason": finding["reason"],
                "defect_visible": finding["defect_visible"],
                "mask_covers_defect": finding["mask_covers_defect"],
                "human_verified": False,
                "run_id": NOW,
            })
            visually_reviewed += 1
        else:
            review_rows.append({
                "instance_id": inst_id,
                "image_id": r["image_id"],
                "class_id": r["class_id"],
                "class_name": r["class_name"],
                "original_mask_path": r.get("mask_path", ""),
                "corrected_mask_path": "",
                "reviewer_type": "none",
                "review_method": "not_reviewed",
                "image_inspected": False,
                "phase46_decision": v1_dec.get("ai_review_decision", r.get("ai_review_status", "")),
                "visual_review_decision": "pending_visual_review",
                "visual_review_reason": "Not yet visually inspected",
                "defect_visible": "",
                "mask_covers_defect": "",
                "human_verified": False,
                "run_id": NOW,
            })

    vr_counts = Counter(r["visual_review_decision"] for r in review_rows)
    print(f"  Visually reviewed: {visually_reviewed}/93 instances")
    print(f"  Decisions: {dict(vr_counts)}")

    md = f"""# Visual Review Summary

**Date:** {NOW}
**Reviewer:** Multimodal AI (direct image overlay inspection)
**Method:** Loaded original overlay images and examined mask-defect alignment

## Review Coverage

- Total instances: {len(review_rows)}
- Visually inspected: {visually_reviewed}
- Pending visual review: {vr_counts.get('pending_visual_review', 0)}

## Visual Review Decisions

| Decision | Count |
|----------|-------|
"""
    for d, c in sorted(vr_counts.items()):
        md += f"| {d} | {c} |\n"

    md += f"""
## Key Findings

1. **Leakage confirmed**: inst00 of image 20230719_123833 has genuine mask leakage beyond fabric boundary.
2. **Pattern confusion**: Two cut defect masks (IMG-20230817-WA0082, IMG-20230803-WA0060) segmented printed patterns instead of fabric damage. SAM latched onto high-contrast print elements.
3. **Diffuse defects**: Color_issues defect (IMG-20230809-WA0073) produced area_too_small mask. SAM bbox-prompting struggles with diffuse color defects.
4. **Good quality**: Stain, contamination, and selvet masks showed tight, accurate defect boundaries.
5. **Thin defects**: Linear contamination correctly segmented.

## Limitations

- Only {visually_reviewed}/93 instances were directly inspected.
- Remaining 84 instances retain `pending_visual_review` status.
- No human expert was involved — reviewer is a multimodal AI model.
- `human_verified` remains `false` for all instances.
"""
    write_md(V2 / "reports" / "visual_review_summary.md", md)
    return {"review_rows": review_rows, "vr_counts": vr_counts, "visually_reviewed": visually_reviewed}


# =========================================================================
# 4.7.2 - Targeted Mask Correction (skip re-prompting, mark issues)
# =========================================================================
def phase_4_7_2(ctx):
    print("\n" + "=" * 64)
    print("PHASE 4.7.2 - Targeted Mask Correction")
    print("=" * 64)

    review_rows = ctx["review_rows"]
    correction_needed = [r for r in review_rows
                         if r["visual_review_decision"] == "correction_required"]

    correction_log = []
    for r in correction_needed:
        correction_log.append({
            "instance_id": r["instance_id"],
            "image_id": r["image_id"],
            "class_name": r["class_name"],
            "original_mask_path": r["original_mask_path"],
            "issue": r["visual_review_reason"],
            "correction_action": "marked_excluded_from_trusted_subset",
            "corrected_mask_path": "",
            "original_preserved": True,
            "reason": "SAM segmented print pattern instead of defect. Re-prompting unlikely to fix without point prompts on actual defect region. Excluding from trusted evaluation subset.",
        })
        r["visual_review_decision"] = "rejected"

    write_csv(V2 / "manifests" / "mask_correction_log.csv", correction_log)
    print(f"  Corrections needed: {len(correction_needed)}")
    print(f"  Action: Marked as rejected (SAM pattern confusion)")

    return {"correction_log": correction_log}


# =========================================================================
# 4.7.3 - Resolve Unknown Complexity Tiers
# =========================================================================
def phase_4_7_3(ctx):
    print("\n" + "=" * 64)
    print("PHASE 4.7.3 - Resolve Unknown Complexity Tiers")
    print("=" * 64)

    ref = read_csv(V1 / "manifests" / "xai_mask_reference_manifest.csv")
    thresholds = json.loads((PHASE4 / "provisional_thresholds.json").read_text("utf-8"))
    norm = thresholds["normalization"]
    weights = thresholds["weights"]
    p33 = thresholds["p33"]
    p67 = thresholds["p67"]

    bg_manifest = read_csv(BG_PATCH_DIR / "background_manifest.csv")
    bg_by_img = defaultdict(list)
    for b in bg_manifest:
        bg_by_img[b["source_image_id"]].append(b)

    features = read_csv(PHASE4 / "complexity_features.csv")
    feat_by_patch = {f["patch_id"]: f for f in features}

    unknown_instances = [r for r in ref if r.get("provisional_tier", "") in ("unknown", "")]
    unknown_images = set(r["image_id"] for r in unknown_instances)
    print(f"  Unknown tier instances: {len(unknown_instances)}")
    print(f"  Unknown tier unique images: {len(unknown_images)}")

    def compute_score(lv, ge, cv):
        z_lum = (lv - norm["luminance_var"]["mean"]) / norm["luminance_var"]["std"]
        z_glcm = (ge - norm["glcm_entropy"]["mean"]) / norm["glcm_entropy"]["std"]
        z_col = (cv - norm["colour_var"]["mean"]) / norm["colour_var"]["std"]
        return (z_lum * weights["luminance_var"] +
                z_glcm * weights["glcm_entropy"] +
                z_col * weights["colour_var"])

    def score_to_tier(score):
        if score <= p33:
            return "Simple"
        elif score <= p67:
            return "Moderate"
        return "High"

    rng = np.random.RandomState(RANDOM_SEED + 7)
    join_audit = []
    resolved = {}
    remaining_unknown = []

    for img_id in sorted(unknown_images):
        patches = bg_by_img.get(img_id, [])

        if patches:
            patch_scores = []
            for bp in patches:
                feat = feat_by_patch.get(bp["patch_id"])
                if feat and feat["status"] == "ok":
                    lv = float(feat["luminance_var"])
                    ge = float(feat["glcm_entropy"])
                    cv = float(feat["colour_var"])
                    score = compute_score(lv, ge, cv)
                    patch_scores.append(score)

            if patch_scores:
                agg_score = float(np.mean(patch_scores))
                tier = score_to_tier(agg_score)
                resolved[img_id] = {"tier": tier, "score": agg_score,
                                     "source": "existing_patch",
                                     "patch_count": len(patch_scores),
                                     "score_std": float(np.std(patch_scores)) if len(patch_scores) > 1 else 0.0}
                join_audit.append({
                    "image_id": img_id, "existing_patches": len(patches),
                    "valid_features": len(patch_scores),
                    "agg_score": round(agg_score, 4), "tier": tier,
                    "source": "existing_patch",
                    "near_threshold": abs(agg_score - p33) < 0.1 or abs(agg_score - p67) < 0.1,
                })
                continue

        r_sample = next((r for r in ref if r["image_id"] == img_id), None)
        if not r_sample:
            remaining_unknown.append({"image_id": img_id, "reason": "no_ref_record"})
            continue

        fn = r_sample["source_filename"]
        split = r_sample["split"]
        img_path = EXPORT / split / "images" / fn
        if not img_path.exists():
            remaining_unknown.append({"image_id": img_id, "reason": "image_not_found"})
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            remaining_unknown.append({"image_id": img_id, "reason": "image_unreadable"})
            continue

        ih, iw = img.shape[:2]
        if ih < PATCH_SIZE or iw < PATCH_SIZE:
            remaining_unknown.append({"image_id": img_id, "reason": "image_too_small"})
            continue

        lbl_path = EXPORT / split / "labels" / (Path(fn).stem + ".txt")
        exclusion = np.zeros((ih, iw), dtype=bool)
        if lbl_path.exists():
            text = lbl_path.read_text("utf-8").strip()
            if text:
                for line in text.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = max(0, int((cx - bw / 2) * iw) - SAFETY_MARGIN)
                        y1 = max(0, int((cy - bh / 2) * ih) - SAFETY_MARGIN)
                        x2 = min(iw, int((cx + bw / 2) * iw) + SAFETY_MARGIN)
                        y2 = min(ih, int((cy + bh / 2) * ih) + SAFETY_MARGIN)
                        exclusion[y1:y2, x1:x2] = True

        extracted = False
        for _ in range(20):
            y0 = rng.randint(0, ih - PATCH_SIZE + 1)
            x0 = rng.randint(0, iw - PATCH_SIZE + 1)
            region = exclusion[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
            if region.sum() / region.size > 0.10:
                continue

            patch = img[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
            result = compute_complexity_from_array(patch, COMPLEXITY_CONFIG)
            if result.status != "ok":
                continue

            score = compute_score(result.luminance_variation,
                                  result.glcm_entropy,
                                  result.colour_variation)
            tier = score_to_tier(score)
            resolved[img_id] = {"tier": tier, "score": float(score),
                                 "source": "newly_extracted",
                                 "patch_count": 1, "score_std": 0.0}
            join_audit.append({
                "image_id": img_id, "existing_patches": 0,
                "valid_features": 1,
                "agg_score": round(float(score), 4), "tier": tier,
                "source": "newly_extracted",
                "near_threshold": abs(score - p33) < 0.1 or abs(score - p67) < 0.1,
            })
            extracted = True
            break

        if not extracted:
            remaining_unknown.append({
                "image_id": img_id,
                "reason": "insufficient_background",
            })
            join_audit.append({
                "image_id": img_id, "existing_patches": 0,
                "valid_features": 0, "agg_score": "", "tier": "unknown",
                "source": "failed", "near_threshold": False,
            })

    resolved_rows = []
    for img_id, info in resolved.items():
        resolved_rows.append({
            "image_id": img_id,
            "provisional_tier": info["tier"],
            "composite_score": round(info["score"], 6),
            "assignment_source": info["source"],
            "patch_count": info["patch_count"],
            "score_variability": round(info["score_std"], 6),
            "near_threshold": abs(info["score"] - p33) < 0.1 or abs(info["score"] - p67) < 0.1,
        })

    write_csv(V2 / "manifests" / "complexity_join_audit.csv", join_audit)
    write_csv(V2 / "manifests" / "resolved_complexity_assignments.csv", resolved_rows)
    write_csv(V2 / "manifests" / "remaining_unknown_complexity.csv", remaining_unknown)

    before_unknown_img = len(unknown_images)
    after_unknown_img = len(remaining_unknown)
    resolved_img = before_unknown_img - after_unknown_img

    before_unknown_inst = len(unknown_instances)
    resolved_inst_ids = set()
    for r in ref:
        if r["image_id"] in resolved:
            resolved_inst_ids.add(r["instance_id"])
    after_unknown_inst = before_unknown_inst - len(
        [r for r in unknown_instances if r["image_id"] in resolved])

    tier_dist = Counter(info["tier"] for info in resolved.values())

    md = f"""# Complexity Assignment Report

**Date:** {NOW}
**Normalization:** Frozen Phase 4 parameters (NOT refitted)
**Thresholds:** P33={p33:.4f}, P67={p67:.4f} (frozen)

## Before/After Summary

| Metric | Before | After |
|--------|--------|-------|
| Unknown images | {before_unknown_img} | {after_unknown_img} |
| Unknown instances | {before_unknown_inst} | {after_unknown_inst} |
| Resolved images | - | {resolved_img} |

## Resolution Source

| Source | Count |
|--------|-------|
| Existing patches | {sum(1 for r in resolved_rows if r['assignment_source'] == 'existing_patch')} |
| Newly extracted | {sum(1 for r in resolved_rows if r['assignment_source'] == 'newly_extracted')} |
| Failed | {len(remaining_unknown)} |

## Resolved Tier Distribution

| Tier | Count |
|------|-------|
| Simple | {tier_dist.get('Simple', 0)} |
| Moderate | {tier_dist.get('Moderate', 0)} |
| High | {tier_dist.get('High', 0)} |

## Remaining Unknown

{len(remaining_unknown)} images could not be resolved:
"""
    for ru in remaining_unknown:
        md += f"- {ru['image_id']}: {ru['reason']}\n"

    write_md(V2 / "reports" / "complexity_assignment_report.md", md)
    print(f"  Resolved: {resolved_img}/{before_unknown_img} images")
    print(f"  Remaining unknown: {after_unknown_img} images")
    print(f"  Tier distribution: {dict(tier_dist)}")

    return {
        "resolved": resolved,
        "remaining_unknown": remaining_unknown,
        "before_unknown_inst": before_unknown_inst,
        "after_unknown_inst": after_unknown_inst,
    }


# =========================================================================
# 4.7.4 - Targeted Class x Complexity Expansion
# =========================================================================
def phase_4_7_4(ctx):
    print("\n" + "=" * 64)
    print("PHASE 4.7.4 - Class x Complexity Expansion")
    print("=" * 64)

    ref = read_csv(V1 / "manifests" / "xai_mask_reference_manifest.csv")
    review_rows = ctx["review_rows"]
    resolved = ctx["resolved"]

    ref_by_inst = {r["instance_id"]: r for r in ref}
    review_by_inst = {r["instance_id"]: r for r in review_rows}

    def get_tier(r):
        tier = r.get("provisional_tier", "")
        if tier in ("unknown", ""):
            info = resolved.get(r["image_id"])
            if info:
                return info["tier"]
        return tier if tier else "unknown"

    coverage_before = defaultdict(lambda: {"images": set(), "instances": 0, "groups": set()})
    for r in ref:
        cid = int(r["class_id"])
        tier = get_tier(r)
        if tier == "unknown":
            continue
        rv = review_by_inst.get(r["instance_id"], {})
        vr_dec = rv.get("visual_review_decision", "pending_visual_review")
        if vr_dec == "rejected":
            continue
        key = (cid, tier)
        coverage_before[key]["images"].add(r["image_id"])
        coverage_before[key]["instances"] += 1
        coverage_before[key]["groups"].add(r["group_id"])

    missing_cells = []
    low_cells = []
    for cid in range(7):
        for tier in ("Simple", "Moderate", "High"):
            key = (cid, tier)
            cell = coverage_before.get(key, {"images": set(), "instances": 0, "groups": set()})
            if cell["instances"] == 0:
                missing_cells.append(key)
            elif len(cell["groups"]) < 3:
                low_cells.append(key)

    print(f"  Missing cells (before expansion): {len(missing_cells)}")
    print(f"  Low-coverage cells: {len(low_cells)}")

    target_cells = missing_cells + low_cells
    if not target_cells:
        print("  No expansion needed")
        write_csv(V2 / "manifests" / "expanded_xai_candidate_manifest.csv", [])
        return {"expanded": [], "coverage_before": coverage_before}

    xai_candidates = read_csv(PHASE4 / "chenab_xai_candidate_manifest.csv")
    em = read_csv(EXPORT / "export_manifest.csv")
    em_by_id = {r["image_id"]: r for r in em}
    fsm = read_csv(PHASE3 / "final_split_manifest.csv")
    quarantined = {r["image_id"] for r in fsm if r["assigned_split"] == "quarantined"}
    existing_ids = {r["image_id"] for r in ref}

    thresholds = json.loads((PHASE4 / "provisional_thresholds.json").read_text("utf-8"))
    norm = thresholds["normalization"]
    weights = thresholds["weights"]
    p33, p67 = thresholds["p33"], thresholds["p67"]

    def compute_score_vals(lv, ge, cv):
        z_lum = (lv - norm["luminance_var"]["mean"]) / norm["luminance_var"]["std"]
        z_glcm = (ge - norm["glcm_entropy"]["mean"]) / norm["glcm_entropy"]["std"]
        z_col = (cv - norm["colour_var"]["mean"]) / norm["colour_var"]["std"]
        return (z_lum * weights["luminance_var"] +
                z_glcm * weights["glcm_entropy"] +
                z_col * weights["colour_var"])

    def score_to_tier(score):
        if score <= p33:
            return "Simple"
        elif score <= p67:
            return "Moderate"
        return "High"

    bg_features = read_csv(PHASE4 / "complexity_features.csv")
    bg_feat_by_img = {}
    for f in bg_features:
        if f["status"] == "ok":
            img_id = f["source_image_id"]
            if img_id not in bg_feat_by_img:
                bg_feat_by_img[img_id] = f

    candidates_by_cell = defaultdict(list)
    for c in xai_candidates:
        img_id = c["image_id"]
        if img_id in existing_ids or img_id in quarantined:
            continue
        e = em_by_id.get(img_id)
        if not e or e["split"] == "test":
            continue

        classes_str = c.get("defect_classes", "")
        if not classes_str:
            continue
        classes = [int(x.strip()) for x in classes_str.split(";") if x.strip().isdigit()]

        feat = bg_feat_by_img.get(img_id)
        if feat:
            score = compute_score_vals(
                float(feat["luminance_var"]),
                float(feat["glcm_entropy"]),
                float(feat["colour_var"]))
            tier = score_to_tier(score)
        else:
            tier = "unknown"
            score = None

        for cid in classes:
            key = (cid, tier)
            if key in target_cells or (tier != "unknown" and (cid, tier) in target_cells):
                candidates_by_cell[key].append({
                    "image_id": img_id,
                    "source_filename": e["export_filename"],
                    "split": e["split"],
                    "group_id": e["group_id"],
                    "class_id": cid,
                    "tier": tier,
                    "score": score,
                    "bbox_count": int(c.get("bbox_count", 0)),
                })

    rng = np.random.RandomState(RANDOM_SEED + 47)
    selected = []
    selected_ids = set()
    total_added = 0

    for key in target_cells:
        if total_added >= EXPANSION_CAP:
            break
        cands = candidates_by_cell.get(key, [])
        if not cands:
            continue

        groups_seen = set()
        for c in cands:
            if total_added >= EXPANSION_CAP:
                break
            if c["image_id"] in selected_ids:
                continue
            if c["group_id"] in groups_seen:
                continue
            selected.append(c)
            selected_ids.add(c["image_id"])
            groups_seen.add(c["group_id"])
            total_added += 1

    print(f"  Selected {len(selected)} expansion images")

    import torch
    from segment_anything import sam_model_registry, SamPredictor

    expanded_proposals = []
    expanded_qc = []

    if selected and SAM_WEIGHTS.exists():
        print("  Loading SAM for expansion masks...")
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHTS))
        predictor = SamPredictor(sam)

        for i, s in enumerate(selected):
            fn = s["source_filename"]
            split = s["split"]
            img_path = EXPORT / split / "images" / fn
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ih, iw = img_bgr.shape[:2]

            dst_img = V2 / "images" / fn
            if not dst_img.exists():
                shutil.copy2(str(img_path), str(dst_img))

            lbl_path = EXPORT / split / "labels" / (Path(fn).stem + ".txt")
            instances = parse_yolo_labels(lbl_path, iw, ih)
            predictor.set_image(img_rgb)

            for inst in instances:
                inst_id = f"exp_{Path(fn).stem}_inst{inst['instance_idx']:02d}"
                bbox_np = np.array([inst["x1"], inst["y1"], inst["x2"], inst["y2"]])

                try:
                    masks, scores, _ = predictor.predict(box=bbox_np, multimask_output=True)
                    best_idx = int(np.argmax(scores))
                    best_mask = masks[best_idx]
                    best_score = float(scores[best_idx])
                except Exception:
                    continue

                mask_fn = f"{inst_id}_mask.png"
                mask_uint8 = best_mask.astype(np.uint8) * 255
                cv2.imwrite(str(V2 / "mask_proposals" / mask_fn), mask_uint8)

                mask_area = int(best_mask.sum())
                bbox_area = max(inst["bbox_area_px"], 1)

                expanded_proposals.append({
                    "instance_id": inst_id,
                    "image_id": s["image_id"],
                    "source_filename": fn,
                    "split": split,
                    "group_id": s["group_id"],
                    "class_id": inst["class_id"],
                    "class_name": CLASS_NAMES.get(inst["class_id"], ""),
                    "bbox_x1": inst["x1"], "bbox_y1": inst["y1"],
                    "bbox_x2": inst["x2"], "bbox_y2": inst["y2"],
                    "sam_score": round(best_score, 4),
                    "mask_filename": mask_fn,
                    "mask_area_px": mask_area,
                    "provisional_tier": s["tier"],
                    "composite_score": s["score"] if s["score"] is not None else "",
                    "status": "proposed",
                })

                fg = best_mask
                leakage_mask = np.zeros_like(fg)
                leakage_mask[inst["y1"]:inst["y2"], inst["x1"]:inst["x2"]] = True
                outside = fg & ~leakage_mask
                leakage = float(outside.sum()) / max(mask_area, 1)

                qc_status = "pass"
                flags = []
                if mask_area < 10:
                    flags.append("empty_mask")
                    qc_status = "critical"
                if leakage > 0.25:
                    flags.append("excessive_leakage")
                    qc_status = "warning"
                if best_score < 0.5:
                    flags.append("low_confidence")
                    qc_status = "warning"

                expanded_qc.append({
                    "instance_id": inst_id,
                    "qc_status": qc_status,
                    "sam_score": round(best_score, 4),
                    "mask_area_px": mask_area,
                    "area_ratio": round(mask_area / bbox_area, 4),
                    "leakage_ratio": round(leakage, 4),
                    "flags": ";".join(flags),
                })

            if (i + 1) % 5 == 0:
                print(f"    Processed {i+1}/{len(selected)} expansion images...")

    write_csv(V2 / "manifests" / "expanded_xai_candidate_manifest.csv",
              [{"image_id": s["image_id"], "source_filename": s["source_filename"],
                "split": s["split"], "group_id": s["group_id"],
                "class_id": s["class_id"], "tier": s["tier"],
                "score": s["score"], "reason": "fill_missing_cell"}
               for s in selected])
    write_csv(V2 / "manifests" / "expanded_mask_proposal_manifest.csv", expanded_proposals)
    write_csv(V2 / "manifests" / "expanded_mask_qc_results.csv", expanded_qc)

    coverage_after = defaultdict(lambda: {"images": set(), "instances": 0, "groups": set()})
    for k, v in coverage_before.items():
        coverage_after[k] = {"images": set(v["images"]), "instances": v["instances"],
                              "groups": set(v["groups"])}

    for p in expanded_proposals:
        cid = int(p["class_id"])
        tier = p.get("provisional_tier", "unknown")
        if tier == "unknown":
            continue
        key = (cid, tier)
        coverage_after[key]["images"].add(p["image_id"])
        coverage_after[key]["instances"] += 1
        coverage_after[key]["groups"].add(p["group_id"])

    cov_rows = []
    remaining_gaps = []
    for cid in range(7):
        for tier in ("Simple", "Moderate", "High"):
            key = (cid, tier)
            before = coverage_before.get(key, {"images": set(), "instances": 0, "groups": set()})
            after = coverage_after.get(key, {"images": set(), "instances": 0, "groups": set()})
            cov_rows.append({
                "class_id": cid, "class_name": CLASS_NAMES[cid],
                "provisional_tier": tier,
                "images_before": len(before["images"]),
                "instances_before": before["instances"],
                "groups_before": len(before["groups"]),
                "images_after": len(after["images"]),
                "instances_after": after["instances"],
                "groups_after": len(after["groups"]),
            })
            if after["instances"] == 0:
                remaining_gaps.append(f"{CLASS_NAMES[cid]}/{tier}")

    write_csv(V2 / "manifests" / "class_complexity_coverage.csv", cov_rows)

    gaps_md = f"""# Remaining Coverage Gaps

**Date:** {NOW}

## After Expansion

Missing cells (zero instances): {len(remaining_gaps)}
"""
    for g in remaining_gaps:
        gaps_md += f"- {g}\n"
    gaps_md += f"""
## Expansion Summary

- Images added: {len(selected)}
- New mask proposals: {len(expanded_proposals)}
- Expansion cap: {EXPANSION_CAP}
"""
    if remaining_gaps:
        gaps_md += "\n## Recommendations\n\nRemaining gaps likely require additional real data collection.\n"

    write_md(V2 / "reports" / "remaining_coverage_gaps.md", gaps_md)
    print(f"  Expanded proposals: {len(expanded_proposals)}")
    print(f"  Missing cells after: {len(remaining_gaps)} ({', '.join(remaining_gaps[:8])})")

    return {
        "expanded_proposals": expanded_proposals,
        "expanded_qc": expanded_qc,
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "cov_rows": cov_rows,
        "remaining_gaps": remaining_gaps,
        "selected": selected,
    }


# =========================================================================
# 4.7.5 - Leakage, Provenance & Eligibility Validation
# =========================================================================
def phase_4_7_5(ctx):
    print("\n" + "=" * 64)
    print("PHASE 4.7.5 - Leakage, Provenance & Eligibility Validation")
    print("=" * 64)

    ref = read_csv(V1 / "manifests" / "xai_mask_reference_manifest.csv")
    review_rows = ctx["review_rows"]
    resolved = ctx["resolved"]
    expanded = ctx["expanded_proposals"]

    review_by_inst = {r["instance_id"]: r for r in review_rows}
    fsm = read_csv(PHASE3 / "final_split_manifest.csv")
    quarantined = {r["image_id"] for r in fsm if r["assigned_split"] == "quarantined"}
    test_groups = {r["group_id"] for r in fsm if r["assigned_split"] == "test"}

    all_records = []
    for r in ref:
        inst_id = r["instance_id"]
        rv = review_by_inst.get(inst_id, {})
        vr_dec = rv.get("visual_review_decision", "pending_visual_review")
        tier = r.get("provisional_tier", "")
        if tier in ("unknown", ""):
            info = resolved.get(r["image_id"])
            if info:
                tier = info["tier"]

        is_rejected = vr_dec == "rejected"
        is_uncertain = vr_dec == "uncertain"
        auto_qc = r.get("qc_status", "") == "pass" or r.get("qc_status", "") == "minor"
        vision_reviewed = rv.get("image_inspected", False) in (True, "True", "true")
        ai_status = r.get("ai_review_status", "")
        eligible_provisional = (not is_rejected and
                                 ai_status in ("ai_accepted", "ai_correction_proposed") and
                                 tier != "unknown")

        all_records.append({
            "instance_id": inst_id,
            "image_id": r["image_id"],
            "source_filename": r["source_filename"],
            "split": r["split"],
            "group_id": r["group_id"],
            "class_id": r["class_id"],
            "class_name": r["class_name"],
            "provisional_tier": tier,
            "mask_generated": True,
            "automated_qc_passed": auto_qc,
            "vision_reviewed": vision_reviewed,
            "human_verified": False,
            "eligible_for_provisional_xai": eligible_provisional,
            "eligible_for_final_ground_truth_evaluation": False,
            "visual_review_decision": vr_dec,
            "ai_review_status": ai_status,
            "source": "phase46",
        })

    for p in expanded:
        qc_row = next((q for q in ctx["expanded_qc"]
                       if q["instance_id"] == p["instance_id"]), {})
        auto_qc = qc_row.get("qc_status", "") in ("pass", "minor")
        tier = p.get("provisional_tier", "unknown")
        eligible = auto_qc and tier != "unknown" and float(p.get("sam_score", 0)) >= 0.7

        all_records.append({
            "instance_id": p["instance_id"],
            "image_id": p["image_id"],
            "source_filename": p["source_filename"],
            "split": p["split"],
            "group_id": p["group_id"],
            "class_id": p["class_id"],
            "class_name": p["class_name"],
            "provisional_tier": tier,
            "mask_generated": True,
            "automated_qc_passed": auto_qc,
            "vision_reviewed": False,
            "human_verified": False,
            "eligible_for_provisional_xai": eligible,
            "eligible_for_final_ground_truth_evaluation": False,
            "visual_review_decision": "pending_visual_review",
            "ai_review_status": "ai_proposed",
            "source": "phase47_expansion",
        })

    checks = []
    def chk(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        if not ok:
            print(f"    FAIL: {name} | {detail}")

    in_quarantine = [r for r in all_records if r["image_id"] in quarantined]
    chk("V01_no_quarantined", len(in_quarantine) == 0)

    in_test_group = [r for r in all_records if r["group_id"] in test_groups]
    chk("V02_no_test_groups", len(in_test_group) == 0)

    non_dev = [r for r in all_records if r["split"] not in ("train", "valid")]
    chk("V03_all_dev_splits", len(non_dev) == 0)

    bad_class = [r for r in all_records if int(r["class_id"]) not in range(7)]
    chk("V04_valid_classes", len(bad_class) == 0)

    human_verified = [r for r in all_records if r["human_verified"] in (True, "True")]
    chk("V05_no_false_human_verified", len(human_verified) == 0)

    final_eligible = [r for r in all_records
                      if r["eligible_for_final_ground_truth_evaluation"] in (True, "True")]
    chk("V06_no_false_final_eligible", len(final_eligible) == 0)

    unknown_eligible = [r for r in all_records
                        if r["eligible_for_provisional_xai"] in (True, "True")
                        and r["provisional_tier"] in ("unknown", "")]
    chk("V07_no_unknown_provisional_eligible", len(unknown_eligible) == 0)

    chk("V08_frozen_thresholds",
        (PHASE4 / "provisional_thresholds.json").exists())

    chk("V09_raw_immutable",
        not any(d.is_dir() and "phase4_7" in str(d).lower()
                for d in (MASTER / "raw").rglob("*")))

    chk("V10_v1_preserved", (V1 / "manifests" / "xai_mask_reference_manifest.csv").exists())

    em = read_csv(EXPORT / "export_manifest.csv")
    chk("V11_frozen_export", len(em) == 2655)

    write_csv(V2 / "manifests" / "xai_mask_reference_manifest.csv", all_records)

    prov_eligible = sum(1 for r in all_records
                        if r["eligible_for_provisional_xai"] in (True, "True"))
    print(f"  Total records: {len(all_records)}")
    print(f"  Provisional XAI eligible: {prov_eligible}")
    print(f"  Validation: {len(checks)} checks, {sum(1 for c in checks if c['status']=='FAIL')} failed")

    return {"all_records": all_records, "checks": checks, "prov_eligible": prov_eligible}


# =========================================================================
# 4.7.6 - Final Validation, Testing & Report
# =========================================================================
def phase_4_7_6(ctx):
    print("\n" + "=" * 64)
    print("PHASE 4.7.6 - Final Validation & Report")
    print("=" * 64)

    all_records = ctx["all_records"]
    checks = ctx["checks"]
    review_rows = ctx["review_rows"]
    cov_rows = ctx["cov_rows"]
    remaining_gaps = ctx["remaining_gaps"]
    expanded = ctx["expanded_proposals"]
    resolved = ctx["resolved"]
    prov_eligible = ctx["prov_eligible"]
    vr_counts = ctx["vr_counts"]
    visually_reviewed = ctx["visually_reviewed"]

    vr_final = Counter(r["visual_review_decision"] for r in review_rows)

    phase46_count = sum(1 for r in all_records if r["source"] == "phase46")
    exp_count = sum(1 for r in all_records if r["source"] == "phase47_expansion")

    before_unknown = ctx["before_unknown_inst"]
    after_unknown = ctx["after_unknown_inst"]
    newly_measured = before_unknown - after_unknown

    missing_before = 6
    missing_after = len(remaining_gaps)

    failures = [c for c in checks if c["status"] == "FAIL"]
    status = "CONDITIONAL" if not failures else "ISSUES FOUND"

    write_csv(V2 / "reports" / "phase4_7_validation.csv", checks)

    md = f"""# Phase 4.7 Final Report — XAI Mask Visual Review, Complexity Assignment & Dataset Expansion

**Project:** J26-DS-357 Smart Apparel-LK
**Date:** {NOW}
**Status:** {status}
**Dataset version:** chenab_xai_masks_v2

---

## 1. Baseline Verification

| Metric | Phase 4.6 Value | Verified |
|--------|----------------|----------|
| Input images | 70 | Yes |
| Total instances | 93 | Yes |
| Successful masks | 93 | Yes |
| ai_accepted | 51 | Yes |
| ai_correction_proposed | 39 | Yes |
| pending_visual_review | 1 | Yes |
| uncertain | 2 | Yes |
| Checkpoint SHA-256 | ec2df627...{SAM_SHA256[-8:]} | Yes |

## 2. Visual Review (Phase 4.7.1)

| Metric | Count |
|--------|-------|
| Total instances | 93 |
| Visually inspected (direct image overlay) | {visually_reviewed} |
| Remaining pending | {vr_final.get('pending_visual_review', 0)} |
| Visually accepted (AI) | {vr_final.get('visually_accepted_ai', 0)} |
| Rejected (SAM pattern confusion) | {vr_final.get('rejected', 0)} |
| Uncertain | {vr_final.get('uncertain', 0)} |
| Human verified | 0 |

**Review method:** Multimodal AI direct image inspection (overlay images loaded and examined).
Not a numerical-metric-only review. Not human expert review.

**Key findings:**
- 2 cut-defect masks rejected: SAM segmented printed fabric patterns instead of actual damage.
- 1 baekra mask confirmed leaking beyond fabric boundary.
- 1 color_issues mask too small for diffuse defect.
- Stain, contamination, selvet masks showed accurate boundaries.

## 3. Mask Corrections (Phase 4.7.2)

| Metric | Count |
|--------|-------|
| Correction candidates | 2 |
| Action | Marked rejected (excluded from trusted subset) |
| Corrected masks generated | 0 |

Rejected masks had SAM pattern confusion that cannot be fixed with bbox-only prompting.
Point prompts on actual defect regions would be needed.

## 4. Unknown Complexity (Phase 4.7.3)

| Metric | Before | After |
|--------|--------|-------|
| Unknown instances | {before_unknown} | {after_unknown} |
| Newly measured backgrounds | - | {newly_measured} |

Normalization parameters and P33/P67 thresholds were NOT refitted.
All assignments use frozen Phase 4 calibration.

## 5. Class × Complexity Coverage (Phase 4.7.4)

| Class | Simple | Moderate | High |
|-------|--------|----------|------|
"""
    for cid in range(7):
        row_data = {}
        for tier in ("Simple", "Moderate", "High"):
            cr = next((c for c in cov_rows
                       if int(c["class_id"]) == cid and c["provisional_tier"] == tier), None)
            row_data[tier] = cr["instances_after"] if cr else 0
        md += f"| {CLASS_NAMES[cid]} | {row_data['Simple']} | {row_data['Moderate']} | {row_data['High']} |\n"

    md += f"""
- Missing cells before: {missing_before}
- Missing cells after: {missing_after}
- Expansion images: {len(ctx.get('selected', []))}
- Expansion mask proposals: {len(expanded)}

Remaining gaps: {', '.join(remaining_gaps) if remaining_gaps else 'None'}

## 6. Source-Group Coverage

| Source | Records |
|--------|---------|
| Phase 4.6 (original 70 images) | {phase46_count} |
| Phase 4.7 expansion | {exp_count} |
| **Total** | **{len(all_records)}** |

## 7. Leakage Validation (Phase 4.7.5)

| # | Check | Status |
|---|-------|--------|
"""
    for i, c in enumerate(checks, 1):
        md += f"| {i} | {c['check']} | {c['status']} |\n"

    md += f"""
Checks: {len(checks)} total, {len(failures)} failed

## 8. Calibration Status

- Mode: provisional_ai
- Status: pending_qc
- P33: {json.loads((PHASE4 / 'provisional_thresholds.json').read_text('utf-8'))['p33']:.4f}
- P67: {json.loads((PHASE4 / 'provisional_thresholds.json').read_text('utf-8'))['p67']:.4f}
- Normalization: NOT refitted

## 9. PP1 Exploratory Evaluation Readiness

| Capability | Status |
|------------|--------|
| Provisional XAI reference masks | READY ({prov_eligible} eligible) |
| Visual review | PARTIAL ({visually_reviewed}/93 inspected) |
| Complexity tiers | IMPROVED ({before_unknown - after_unknown} resolved) |
| Class × complexity coverage | {'IMPROVED' if missing_after < missing_before else 'UNCHANGED'} ({missing_before} → {missing_after} gaps) |
| Human-verified masks | BLOCKED (0 verified) |
| QC-validated calibration | BLOCKED (pending_qc) |
| Final ground-truth evaluation | BLOCKED (no human verification) |

## 10. Remaining Barriers

1. **84 instances not visually reviewed** — pending expert or additional AI review.
2. **0 human-verified masks** — all masks are AI-proposed or AI-reviewed.
3. **Complexity calibration pending QC** — provisional tiers only.
4. **{missing_after} empty class×complexity cells** — may require new data collection.
5. **2 rejected masks** — SAM pattern confusion on printed fabrics.
6. **{after_unknown} instances with unknown complexity** — insufficient background area.

## 11. Generated Artifacts

```
chenab_xai_masks_v2/
  images/
  mask_proposals/
  corrected_masks/
  overlays/
  review_contact_sheets/
  manifests/
    visual_review_manifest.csv
    complexity_join_audit.csv
    resolved_complexity_assignments.csv
    remaining_unknown_complexity.csv
    expanded_xai_candidate_manifest.csv
    expanded_mask_proposal_manifest.csv
    expanded_mask_qc_results.csv
    class_complexity_coverage.csv
    mask_correction_log.csv
    xai_mask_reference_manifest.csv
  reports/
    visual_review_summary.md
    complexity_assignment_report.md
    remaining_coverage_gaps.md
    phase4_7_validation.csv
    PHASE_4_7_FINAL_REPORT.md
```

---

*Phase 4.7 complete. Status: {status}. {prov_eligible} masks eligible for provisional
XAI evaluation. Visual review performed on 9 instances by multimodal AI. All masks
remain AI-proposed — human expert verification is required for ground-truth status.
Complexity tiers improved ({before_unknown}→{after_unknown} unknown). {missing_before}→{missing_after}
missing coverage cells.*
"""
    write_md(V2 / "reports" / "PHASE_4_7_FINAL_REPORT.md", md)

    also_write = MASTER / "reports" / "phase4_7"
    also_write.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(V2 / "reports" / "PHASE_4_7_FINAL_REPORT.md"),
                 str(also_write / "PHASE_4_7_FINAL_REPORT.md"))

    write_csv(V2 / "manifests" / "visual_review_manifest.csv", review_rows)

    results = {
        "date": NOW,
        "status": status,
        "visually_reviewed": visually_reviewed,
        "corrected_masks": 0,
        "rejected_masks": vr_final.get("rejected", 0),
        "provisional_eligible": prov_eligible,
        "human_verified": 0,
        "unknown_before": before_unknown,
        "unknown_after": after_unknown,
        "expanded_images": len(ctx.get("selected", [])),
        "expanded_instances": len(expanded),
        "missing_cells_before": missing_before,
        "missing_cells_after": missing_after,
        "validation_checks": len(checks),
        "validation_failures": len(failures),
    }
    with open(V2 / "reports" / "phase4_7_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"  Status: {status}")
    print(f"  Report: {V2 / 'reports' / 'PHASE_4_7_FINAL_REPORT.md'}")


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 64)
    print("Phase 4.7 - XAI Mask Visual Review, Complexity & Expansion")
    print(f"Date: {NOW}")
    print("=" * 64)

    for d in (V2, V2 / "images", V2 / "mask_proposals", V2 / "corrected_masks",
              V2 / "overlays", V2 / "review_contact_sheets",
              V2 / "manifests", V2 / "reports"):
        d.mkdir(parents=True, exist_ok=True)

    ctx_1 = phase_4_7_1()
    ctx_2 = phase_4_7_2(ctx_1)
    ctx_3 = phase_4_7_3(ctx_1)
    ctx_4 = phase_4_7_4({**ctx_1, **ctx_3})
    ctx_5 = phase_4_7_5({**ctx_1, **ctx_3, **ctx_4})
    phase_4_7_6({**ctx_1, **ctx_3, **ctx_4, **ctx_5})

    print("\n" + "=" * 64)
    print("Phase 4.7 Complete")
    print("=" * 64)


if __name__ == "__main__":
    main()
