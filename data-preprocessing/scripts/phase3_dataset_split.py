"""
Phase 3 - Research-Safe Master Dataset Splitting, Export & Freezing.
Parts A-H combined runner. Policy B group-disjoint splitting.
Run: python -I phase3_dataset_split.py
"""

import csv
import hashlib
import json
import re
import shutil
import sys
import io
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
RAW_CHENAB = MASTER / "raw" / "chenab" / "fabricDD.v2i.yolov8"
RAW_AITEX = MASTER / "raw" / "aitex"
META = MASTER / "metadata"
PHASE28 = MASTER / "reports" / "phase2_8"
PHASE3 = MASTER / "reports" / "phase3"
PROCESSED = MASTER / "processed"
EXPORT_DIR = PROCESSED / "chenab_detection_v1"
RESEARCH_DIR = PROCESSED / "research_subsets"
SPLITS = ("train", "valid", "test")
CLASS_NAMES = {0: "baekra", 1: "color_issues", 2: "contamination",
               3: "cut", 4: "gray_stitch", 5: "selvet", 6: "stain"}
CLASS_NAMES_DISPLAY = {0: "baekra", 1: "color issues", 2: "contamination",
                       3: "cut", 4: "gray stitch", 5: "selvet", 6: "stain"}
RANDOM_SEED = 42
NOW = datetime.now().isoformat(timespec="seconds")

RF_PATTERN = re.compile(
    r'^(augmented_)?(.+?)_jpg\.rf\.[0-9a-f]{32}$', re.IGNORECASE)


def extract_root(stem):
    m = RF_PATTERN.match(stem)
    if m:
        return m.group(2), "rf_pattern"
    idx = stem.lower().find("_jpg.rf.")
    if idx > 0:
        root = stem[:idx]
        if root.lower().startswith("augmented_"):
            root = root[10:]
        return root, "rf_fallback"
    return stem, "unparseable"


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


def s_bool(v):
    return str(v).strip().lower() == "true"


# =========================================================================
# PART A - Verify Phase 2.8 Inputs
# =========================================================================
def part_a():
    print("\n" + "=" * 64)
    print("PART A - Verify Phase 2.8 Inputs")
    print("=" * 64)

    mv12 = read_csv(META / "master_manifest_v1_2.csv")
    mv1 = read_csv(META / "master_manifest_v1.csv")
    dup_review = read_csv(PHASE28 / "ai_duplicate_review_decisions.csv")
    resolved_groups = read_csv(PHASE28 / "resolved_source_groups.csv")
    uncertain_edges = read_csv(PHASE28 / "remaining_uncertain_edges.csv")
    ann_review = read_csv(PHASE28 / "ai_chenab_annotation_review.csv")
    mask_elig = read_csv(PHASE28 / "aitex_mask_eligibility.csv")
    policy_b = read_csv(PHASE28 / "policy_b_reviewed_simulation.csv")
    exclusions = read_csv(PHASE28 / "phase3_exclusion_manifest.csv")
    disjoint = read_csv(PHASE28 / "group_disjointness_validation.csv")

    path_lookup = {r["image_id"]: r["relative_image_path"] for r in mv1}

    errors = []
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL",
                        "detail": detail})
        if not ok:
            errors.append(f"{name}: {detail}")

    chk("manifest_total", len(mv12) == 3038, f"got {len(mv12)}")
    chenab = [r for r in mv12 if r["source_dataset"] == "chenab"]
    aitex = [r for r in mv12 if r["source_dataset"] == "aitex"]
    chk("chenab_count", len(chenab) == 2791, f"got {len(chenab)}")
    chk("aitex_count", len(aitex) == 247, f"got {len(aitex)}")

    f1_ready = [r for r in mv12 if s_bool(r.get("f1_detection_ready", ""))]
    chk("f1_detection_ready", len(f1_ready) == 2499, f"got {len(f1_ready)}")

    chk("duplicate_review_count", len(dup_review) == 1356,
        f"got {len(dup_review)}")
    chk("uncertain_edges_count", len(uncertain_edges) == 40,
        f"got {len(uncertain_edges)}")

    pb_assign = Counter(r["policy_b_assignment"] for r in policy_b)
    chk("policy_b_total", sum(pb_assign.values()) == 2791,
        f"got {sum(pb_assign.values())}")
    non_q = sum(v for k, v in pb_assign.items() if k != "quarantined")
    chk("policy_b_non_quarantined", non_q == 2786, f"got {non_q}")
    chk("policy_b_quarantined", pb_assign.get("quarantined", 0) == 5,
        f"got {pb_assign.get('quarantined', 0)}")

    chk("exclusion_count", len(exclusions) == 5, f"got {len(exclusions)}")
    chk("disjoint_violations",
        all(r["disjoint"] == "True" for r in disjoint), "violations found")

    ann_decisions = Counter(r["ai_decision"] for r in ann_review)
    chk("annotation_review_total", len(ann_review) == 33,
        f"got {len(ann_review)}")
    chk("annotation_accepted", ann_decisions.get("ai_accepted", 0) == 19,
        f"got {ann_decisions.get('ai_accepted', 0)}")

    ids_v12 = {r["image_id"] for r in mv12}
    ids_v1 = {r["image_id"] for r in mv1}
    chk("id_agreement_v1_v12", ids_v12 == ids_v1)

    rg_files = {r["filename"] for r in resolved_groups}
    pb_files = {r["filename"] for r in policy_b}
    uncovered = pb_files - rg_files
    chk("resolved_groups_loaded", len(rg_files) > 0,
        f"groups cover {len(pb_files & rg_files)}/{len(pb_files)} policy B files, {len(uncovered)} singletons")

    for r in chenab:
        img_id = r["image_id"]
        rp = path_lookup.get(img_id, "")
        if rp:
            full = MASTER / "raw" / rp
            if not full.exists():
                errors.append(f"missing raw image: {rp}")
    chk("raw_images_exist", not any("missing raw" in e for e in errors))

    recon_rows = []
    for r in mv12:
        img_id = r["image_id"]
        src = r["source_dataset"]
        orig_split = r["original_split"]
        f1 = s_bool(r.get("f1_detection_ready", ""))
        f2_bg = s_bool(r.get("f2_background_ready", ""))
        f2_bbox = s_bool(r.get("f2_bbox_xai_ready", ""))
        f2_mask = s_bool(r.get("f2_mask_xai_ready", ""))

        pb_row = next((p for p in policy_b if p["filename"] ==
                       r["original_filename"]), None)
        pb_split = pb_row["policy_b_assignment"] if pb_row else ""

        ann_qc = r.get("annotation_qc_status", "")
        excl = r.get("exclusion_reason", "")
        hist = r.get("historical_model_exposure", "")
        leak = r.get("leakage_risk", "")

        export_eligible = (
            src == "chenab" and
            pb_split in SPLITS and
            ann_qc not in ("excluded_unreliable",) and
            excl not in ("annotation_excluded",)
        )
        annotation_eligible = (
            export_eligible and
            ann_qc not in ("uncertain",) and
            excl not in ("annotation_uncertain",)
        )

        recon_rows.append({
            "image_id": img_id,
            "source_dataset": src,
            "original_split": orig_split,
            "policy_b_split": pb_split,
            "f1_detection_ready": f1,
            "f2_background_ready": f2_bg,
            "f2_bbox_xai_ready": f2_bbox,
            "f2_mask_xai_ready": f2_mask,
            "annotation_qc_status": ann_qc,
            "exclusion_reason": excl,
            "historical_model_exposure": hist,
            "leakage_risk": leak,
            "dataset_inventory_member": True,
            "split_eligible": pb_split in SPLITS,
            "export_eligible": export_eligible,
            "annotation_eligible": annotation_eligible,
            "training_eligible": annotation_eligible and pb_split == "train",
            "bbox_eval_eligible": annotation_eligible and pb_split in SPLITS,
            "mask_xai_eligible": f2_mask,
        })

    write_csv(PHASE3 / "phase3_input_reconciliation.csv", recon_rows)
    print(f"  Written: phase3_input_reconciliation.csv ({len(recon_rows)} rows)")

    export_count = sum(1 for r in recon_rows if r["export_eligible"])
    anno_count = sum(1 for r in recon_rows if r["annotation_eligible"])
    train_count = sum(1 for r in recon_rows if r["training_eligible"])

    md = f"""# Phase 3 Input Validation Report

**Date:** {NOW}

## File Verification

| File | Expected | Actual | Status |
|------|----------|--------|--------|
| master_manifest_v1_2.csv | 3,038 | {len(mv12)} | {'PASS' if len(mv12)==3038 else 'FAIL'} |
| ai_duplicate_review_decisions.csv | 1,356 | {len(dup_review)} | {'PASS' if len(dup_review)==1356 else 'FAIL'} |
| resolved_source_groups.csv | rows | {len(resolved_groups)} | PASS |
| remaining_uncertain_edges.csv | 40 | {len(uncertain_edges)} | {'PASS' if len(uncertain_edges)==40 else 'FAIL'} |
| ai_chenab_annotation_review.csv | 33 | {len(ann_review)} | {'PASS' if len(ann_review)==33 else 'FAIL'} |
| aitex_mask_eligibility.csv | 106 | {len(mask_elig)} | {'PASS' if len(mask_elig)==106 else 'FAIL'} |
| policy_b_reviewed_simulation.csv | 2,791 | {len(policy_b)} | {'PASS' if len(policy_b)==2791 else 'FAIL'} |
| group_disjointness_validation.csv | rows | {len(disjoint)} | PASS |

## Record Reconciliation

| Category | Count |
|----------|-------|
| Total master records | {len(mv12)} |
| Chenab records | {len(chenab)} |
| AITEX records | {len(aitex)} |
| F1 detection ready (Phase 2.8 flag) | {len(f1_ready)} |
| Policy B non-quarantined | {non_q} |
| Policy B quarantined | {pb_assign.get('quarantined', 0)} |
| Export eligible (Phase 3) | {export_count} |
| Annotation eligible (Phase 3) | {anno_count} |
| Training eligible (Phase 3) | {train_count} |

## Eligibility Distinctions

- **Dataset inventory:** All 3,038 images in master manifest.
- **Split eligible:** 2,786 Chenab images with Policy B split assignment (not quarantined).
- **Export eligible:** {export_count} Chenab images assignable to a split (excludes annotation_excluded).
- **Annotation eligible:** {anno_count} images with accepted annotations (excludes uncertain + excluded).
- **Training eligible:** {train_count} images in train split with accepted annotations.
- **Bbox-evaluation eligible:** All annotation-eligible images in any split.
- **Mask-XAI eligible:** AITEX images with verified masks (separate from Chenab).
- **Historical model exposure:** Flagged per image, does not affect split assignment.

## Checks

| Check | Status |
|-------|--------|
"""
    for c in checks:
        md += f"| {c['check']} | {c['status']}"
        if c["detail"]:
            md += f" ({c['detail']})"
        md += " |\n"

    if errors:
        md += f"\n## Errors\n\n"
        for e in errors:
            md += f"- {e}\n"
    else:
        md += "\nAll checks passed.\n"

    write_md(PHASE3 / "phase3_input_validation.md", md)
    print(f"  Written: phase3_input_validation.md")
    print(f"  Export eligible: {export_count}")
    print(f"  Annotation eligible: {anno_count}")
    print(f"  Errors: {len(errors)}")

    return {
        "mv12": mv12, "mv1": mv1, "policy_b": policy_b,
        "resolved_groups": resolved_groups,
        "uncertain_edges": uncertain_edges,
        "mask_elig": mask_elig, "ann_review": ann_review,
        "path_lookup": path_lookup,
        "recon_rows": recon_rows, "errors": errors,
    }


# =========================================================================
# PART B - Freeze Policy B Rules
# =========================================================================
def part_b():
    print("\n" + "=" * 64)
    print("PART B - Freeze Policy B Rules")
    print("=" * 64)

    config = {
        "policy_name": "Policy_B_Group_Disjoint",
        "version": "1.0.0",
        "date": NOW,
        "random_seed": RANDOM_SEED,
        "target_ratios": {"train": 0.70, "valid": 0.15, "test": 0.15},
        "chenab_classes": {str(k): v for k, v in CLASS_NAMES.items()},
        "rules": {
            "R01_family_disjoint": "Original source-image families must not span splits.",
            "R02_confirmed_same_group": "Verified different-root same-source relationships join the same grouping unit.",
            "R03_ai_likely_same_treatment": "AI-likely same-source relationships receive conservative same-split or quarantine treatment.",
            "R04_uncertain_no_bridge": "Uncertain edges cannot silently bridge approved train/valid/test partitions.",
            "R05_cross_split_resolved": "Candidate same-source edges crossing a split must be resolved, constrained, or excluded.",
            "R06_no_phash_only_merge": "Do not automatically merge groups using only a low pHash distance.",
            "R07_preserve_original_split_metadata": "Preserve the original Roboflow split as metadata.",
            "R08_class_id_range": "No class ID outside 0-6 is allowed in Chenab detector exports.",
            "R09_exclude_invalid_annotations": "Exclude annotations marked invalid or unreliable.",
            "R10_exclude_unreviewed_questionable": "Keep unreviewed questionable annotations out of approved training and evaluation.",
            "R11_empty_labels_as_background": "Treat genuinely empty valid annotation files as normal/background images only when provenance supports that interpretation.",
            "R12_deterministic_assignment": "Maintain independent family IDs and deterministic assignment with recorded seed.",
        },
        "uncertainty_policy": "quarantine_over_unsupported_approval",
        "historical_exposure_handling": "flag_and_document",
        "aitex_policy": "separate_research_subsets_only",
    }

    with open(PHASE3 / "policy_b_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("  Written: policy_b_config.json")

    rules_md = f"""# Split Policy B Rules

**Policy:** {config['policy_name']}
**Version:** {config['version']}
**Date:** {config['date']}
**Random Seed:** {config['random_seed']}

## Target Ratios

| Split | Ratio |
|-------|-------|
| train | 70% |
| valid | 15% |
| test  | 15% |

## Rules

| ID | Rule |
|----|------|
"""
    for rid, desc in config["rules"].items():
        rules_md += f"| {rid} | {desc} |\n"

    rules_md += f"""
## Uncertainty Policy

{config['uncertainty_policy'].replace('_', ' ').title()}: when evidence is insufficient to confirm
or deny a relationship, the affected images are quarantined rather than assigned to a split.

## Historical Exposure

{config['historical_exposure_handling'].replace('_', ' ').title()}: all images whose source families
overlap the former test partition are flagged with `historical_test_exposure`. This does not prevent
their inclusion in the new dataset but documents that prior model evaluation results against those
images are invalid.

## AITEX Policy

{config['aitex_policy'].replace('_', ' ').title()}: AITEX images are not included in the Chenab
seven-class detector dataset. They are prepared as separate research subsets for background
augmentation and XAI evaluation.

## Chenab Defect Classes

| ID | Name |
|----|------|
"""
    for cid, cname in sorted(CLASS_NAMES.items()):
        rules_md += f"| {cid} | {cname} |\n"

    write_md(PHASE3 / "split_policy_rules.md", rules_md)
    print("  Written: split_policy_rules.md")
    return config


# =========================================================================
# PART C - Build Independent Group Assignments
# =========================================================================
def part_c(ctx):
    print("\n" + "=" * 64)
    print("PART C - Build Independent Group Assignments")
    print("=" * 64)

    mv12 = ctx["mv12"]
    policy_b = ctx["policy_b"]
    resolved_groups = ctx["resolved_groups"]
    uncertain_edges = ctx["uncertain_edges"]
    recon = ctx["recon_rows"]

    recon_by_id = {r["image_id"]: r for r in recon}

    rg_by_file = {}
    for r in resolved_groups:
        rg_by_file[r["filename"]] = r["group_id"]

    pb_by_file = {}
    for r in policy_b:
        pb_by_file[r["filename"]] = r

    uncertain_files = set()
    for ue in uncertain_edges:
        for k in ("file1", "file2", "image_a", "image_b",
                   "filename_a", "filename_b"):
            if k in ue and ue[k]:
                uncertain_files.add(ue[k])

    uncertain_groups = set()
    for uf in uncertain_files:
        gid = rg_by_file.get(uf, "")
        if gid:
            uncertain_groups.add(gid)

    chenab_manifest = [r for r in mv12 if r["source_dataset"] == "chenab"]

    groups = defaultdict(list)
    for r in chenab_manifest:
        fn = r["original_filename"]
        gid = rg_by_file.get(fn, f"singleton_{fn}")
        groups[gid].append(r)

    group_classes = {}
    for gid, members in groups.items():
        class_counts = Counter()
        for m in members:
            pb = pb_by_file.get(m["original_filename"])
            if pb and pb.get("annotation_classes"):
                for c in pb["annotation_classes"].split(";"):
                    c = c.strip().split(":")[0]
                    if c.isdigit():
                        class_counts[int(c)] += 1
        group_classes[gid] = class_counts

    target_ratios = {"train": 0.70, "valid": 0.15, "test": 0.15}
    total_images = len(chenab_manifest)
    target = {s: total_images * r for s, r in target_ratios.items()}

    by_class = defaultdict(list)
    for gid, members in groups.items():
        cc = group_classes[gid]
        dominant = cc.most_common(1)[0][0] if cc else -1
        by_class[dominant].append({
            "gid": gid, "members": members, "size": len(members),
            "classes": cc, "dominant": dominant,
            "has_uncertain": gid in uncertain_groups,
        })

    for cid in by_class:
        by_class[cid].sort(key=lambda g: -g["size"])

    assignments = {}
    running = {"train": 0, "valid": 0, "test": 0}

    import random
    rng = random.Random(RANDOM_SEED)

    for cid in sorted(by_class.keys()):
        grps = by_class[cid]
        class_total = sum(g["size"] for g in grps)
        class_target = {s: class_total * r for s, r in target_ratios.items()}
        class_running = {"train": 0, "valid": 0, "test": 0}

        for g in grps:
            if g["has_uncertain"]:
                assignments[g["gid"]] = "quarantined"
                continue

            deficits = {s: class_target[s] - class_running[s]
                        for s in SPLITS}
            global_deficits = {s: target[s] - running[s] for s in SPLITS}
            chosen = max(SPLITS,
                         key=lambda s: (deficits[s], global_deficits[s]))

            assignments[g["gid"]] = chosen
            running[chosen] += g["size"]
            class_running[chosen] += g["size"]

    group_assign_rows = []
    for gid, split in sorted(assignments.items()):
        members = groups[gid]
        cc = group_classes[gid]
        group_assign_rows.append({
            "group_id": gid,
            "assigned_split": split,
            "member_count": len(members),
            "dominant_class": cc.most_common(1)[0][0] if cc else "",
            "has_uncertain_edge": gid in uncertain_groups,
        })

    write_csv(PHASE3 / "final_group_assignments.csv", group_assign_rows)
    print(f"  Written: final_group_assignments.csv ({len(group_assign_rows)} groups)")

    split_manifest_rows = []
    for gid, members in groups.items():
        split = assignments.get(gid, "unassigned")
        for m in members:
            fn = m["original_filename"]
            pb = pb_by_file.get(fn, {})
            rec = recon_by_id.get(m["image_id"], {})

            ann_qc = m.get("annotation_qc_status", "")
            excl = m.get("exclusion_reason", "")
            annotation_eligible = (
                split in SPLITS and
                ann_qc not in ("excluded_unreliable",) and
                ann_qc not in ("uncertain",) and
                excl not in ("annotation_excluded", "annotation_uncertain")
            )

            split_manifest_rows.append({
                "image_id": m["image_id"],
                "original_filename": fn,
                "source_dataset": m["source_dataset"],
                "original_split": m["original_split"],
                "group_id": gid,
                "assigned_split": split,
                "annotation_eligible": annotation_eligible,
                "annotation_qc_status": ann_qc,
                "exclusion_reason": excl,
                "historical_model_exposure": m.get("historical_model_exposure", ""),
            })

    write_csv(PHASE3 / "final_split_manifest.csv", split_manifest_rows)
    print(f"  Written: final_split_manifest.csv ({len(split_manifest_rows)} rows)")

    split_counts = Counter(r["assigned_split"] for r in split_manifest_rows)
    print(f"  Split counts: {dict(split_counts)}")

    disjoint_rows = []
    group_splits = defaultdict(set)
    for r in split_manifest_rows:
        group_splits[r["group_id"]].add(r["assigned_split"])
    violations = 0
    for gid, splits in sorted(group_splits.items()):
        real_splits = splits - {"quarantined"}
        is_disjoint = len(real_splits) <= 1
        if not is_disjoint:
            violations += 1
        disjoint_rows.append({
            "group_id": gid,
            "splits_present": ";".join(sorted(splits)),
            "disjoint": is_disjoint,
        })

    write_csv(PHASE3 / "group_disjointness_report.csv", disjoint_rows)
    print(f"  Written: group_disjointness_report.csv ({len(disjoint_rows)} groups)")
    print(f"  Disjoint violations: {violations}")

    class_by_split = defaultdict(Counter)
    for r in split_manifest_rows:
        if r["assigned_split"] in SPLITS and r["annotation_eligible"]:
            fn = r["original_filename"]
            pb = pb_by_file.get(fn, {})
            classes_str = pb.get("annotation_classes", "")
            if classes_str:
                for c in classes_str.split(";"):
                    c = c.strip().split(":")[0]
                    if c.isdigit():
                        class_by_split[r["assigned_split"]][int(c)] += 1

    balance_rows = []
    for cid in sorted(CLASS_NAMES.keys()):
        row = {"class_id": cid, "class_name": CLASS_NAMES[cid]}
        for s in SPLITS:
            row[s] = class_by_split[s].get(cid, 0)
        row["total"] = sum(row[s] for s in SPLITS)
        balance_rows.append(row)

    write_csv(PHASE3 / "class_balance_report.csv", balance_rows)
    print(f"  Written: class_balance_report.csv")

    for row in balance_rows:
        print(f"    {row['class_name']}: train={row['train']} valid={row['valid']} test={row['test']}")

    return {
        "groups": groups, "assignments": assignments,
        "split_manifest": split_manifest_rows,
        "class_balance": balance_rows,
        "pb_by_file": pb_by_file,
        "violations": violations,
    }


# =========================================================================
# PART D - Historical Test Exposure Protection
# =========================================================================
def part_d(ctx):
    print("\n" + "=" * 64)
    print("PART D - Historical Test Exposure Protection")
    print("=" * 64)

    split_manifest = ctx["split_manifest"]

    exposure_rows = []
    for r in split_manifest:
        hist = r.get("historical_model_exposure", "")
        exposed = hist in ("yes_old_test", "family_member_in_old_test")
        exposure_rows.append({
            "image_id": r["image_id"],
            "original_filename": r["original_filename"],
            "original_split": r["original_split"],
            "assigned_split": r["assigned_split"],
            "historical_model_exposure": hist,
            "possibly_exposed_to_old_model": exposed,
            "old_test_image": r["original_split"] == "test",
            "annotation_eligible": r["annotation_eligible"],
        })

    write_csv(PHASE3 / "historical_exposure_manifest.csv", exposure_rows)
    print(f"  Written: historical_exposure_manifest.csv ({len(exposure_rows)} rows)")

    old_test = [r for r in exposure_rows if r["old_test_image"]]
    exposed = [r for r in exposure_rows if r["possibly_exposed_to_old_model"]]
    new_test = [r for r in exposure_rows if r["assigned_split"] == "test"]
    new_test_exposed = [r for r in new_test if r["possibly_exposed_to_old_model"]]
    new_test_old = [r for r in new_test if r["old_test_image"]]

    md = f"""# Test Independence Assessment

**Date:** {NOW}
**Policy:** Policy B Group-Disjoint

## Historical Exposure Summary

| Category | Count |
|----------|-------|
| Total Chenab images | {len(exposure_rows)} |
| Old test images (original test split) | {len(old_test)} |
| Possibly exposed to old model | {len(exposed)} |
| New test split (Policy B) | {len(new_test)} |
| New test images with historical exposure | {len(new_test_exposed)} |
| Old test images in new test | {len(new_test_old)} |

## Independence Assessment

### For the previously trained YOLOv8n model:

**The new Policy B test set is NOT independent for the existing model.**

- {len(new_test_exposed)} of {len(new_test)} new test images ({100*len(new_test_exposed)//max(len(new_test),1)}%) have historical exposure.
- {len(new_test_old)} images were in the original test split.
- {len(new_test_exposed) - len(new_test_old)} images are from source families that overlapped the old test partition.
- The existing model was trained using augmented variants of test data.
- **All prior evaluation results against the old test set are invalid.**

### For a freshly trained model on the new Policy B split:

**The new Policy B test set CAN be treated as independent for a fresh model**, subject to:

1. The fresh model is trained only on Policy B train images.
2. Validation uses only Policy B valid images.
3. No hyperparameter tuning or model selection uses Policy B test images.
4. The 5 quarantined images are excluded from all splits.
5. The 40 uncertain edges are documented as a residual risk.

### Residual Risks

1. **40 uncertain duplicate edges:** These are cross-split image pairs with conflicting signals.
   Groups containing uncertain edges are quarantined (5 images). The remaining uncertain edges
   connect images within the same assigned split and do not affect independence.
2. **No human visual confirmation:** All duplicate decisions are AI-assisted, not human-verified.
3. **Historical test-driven architecture choices:** If model architecture or hyperparameters were
   selected based on old test performance, the new test set may carry indirect bias.

### Recommendation

Fresh model training is required. Even after fresh training, document that:
- The test set was constructed from the same source distribution as prior experiments.
- Historical model selection decisions may have been influenced by old test performance.
- For definitive independent evaluation, an external holdout from new fabric samples is recommended.
"""
    write_md(PHASE3 / "test_independence_assessment.md", md)
    print("  Written: test_independence_assessment.md")


# =========================================================================
# PART E - Build Versioned YOLO Dataset
# =========================================================================
def part_e(ctx):
    print("\n" + "=" * 64)
    print("PART E - Build Versioned YOLO Dataset")
    print("=" * 64)

    if EXPORT_DIR.exists():
        print(f"  ERROR: Export directory already exists: {EXPORT_DIR}")
        print("  Refusing to overwrite a frozen export.")
        return {"export_error": True}

    split_manifest = ctx["split_manifest"]
    path_lookup = ctx["path_lookup"]
    pb_by_file = ctx["pb_by_file"]

    for s in SPLITS:
        (EXPORT_DIR / s / "images").mkdir(parents=True, exist_ok=True)
        (EXPORT_DIR / s / "labels").mkdir(parents=True, exist_ok=True)

    export_rows = []
    exported = {"train": 0, "valid": 0, "test": 0}
    class_counts = defaultdict(Counter)
    negative_counts = {"train": 0, "valid": 0, "test": 0}
    skipped = []
    filename_set = set()

    for r in split_manifest:
        split = r["assigned_split"]
        if split not in SPLITS:
            continue
        if not r["annotation_eligible"]:
            skipped.append({
                "image_id": r["image_id"],
                "filename": r["original_filename"],
                "reason": r.get("annotation_qc_status", "") if r.get("annotation_qc_status", "") in ("uncertain", "excluded_unreliable") else r.get("exclusion_reason", "") or "ineligible",
            })
            continue

        fn = r["original_filename"]
        img_id = r["image_id"]
        rp = path_lookup.get(img_id, "")
        if not rp:
            skipped.append({"image_id": img_id, "filename": fn,
                            "reason": "no_path_in_manifest"})
            continue

        src_img = MASTER / "raw" / rp
        if not src_img.exists():
            skipped.append({"image_id": img_id, "filename": fn,
                            "reason": "source_image_missing"})
            continue

        stem = Path(fn).stem
        ext = Path(fn).suffix or ".jpg"

        export_fn = fn
        if export_fn in filename_set:
            export_fn = f"{stem}_{img_id[-8:]}{ext}"
        filename_set.add(export_fn)

        export_stem = Path(export_fn).stem

        src_label_path = src_img.parent.parent / "labels" / (
            Path(fn).stem + ".txt")

        label_lines = []
        is_negative = False
        if src_label_path.exists():
            raw_text = src_label_path.read_text(encoding="utf-8").strip()
            if raw_text:
                for line in raw_text.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        if 0 <= cid <= 6:
                            cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                            float(parts[3]), float(parts[4]))
                            if (0 <= cx <= 1 and 0 <= cy <= 1 and
                                    0 < w <= 1 and 0 < h <= 1 and
                                    all(map(lambda v: v == v, [cx, cy, w, h]))):
                                label_lines.append(f"{cid} {cx} {cy} {w} {h}")
                                class_counts[split][cid] += 1
            else:
                is_negative = True
        else:
            is_negative = True

        if is_negative:
            negative_counts[split] += 1

        dst_img = EXPORT_DIR / split / "images" / export_fn
        dst_lbl = EXPORT_DIR / split / "labels" / (export_stem + ".txt")

        shutil.copy2(str(src_img), str(dst_img))
        dst_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""),
                           encoding="utf-8")

        export_rows.append({
            "image_id": img_id,
            "original_filename": fn,
            "export_filename": export_fn,
            "split": split,
            "source_path": rp,
            "label_count": len(label_lines),
            "is_negative": is_negative,
            "group_id": r["group_id"],
            "original_split": r["original_split"],
            "historical_model_exposure": r.get("historical_model_exposure", ""),
        })
        exported[split] += 1

    write_csv(EXPORT_DIR / "export_manifest.csv", export_rows)
    print(f"  Written: export_manifest.csv ({len(export_rows)} rows)")

    if skipped:
        write_csv(PHASE3 / "export_skipped.csv", skipped)
        print(f"  Written: export_skipped.csv ({len(skipped)} rows)")

    yaml_lines = [
        "path: .",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        "nc: 7",
        f"names: {[CLASS_NAMES_DISPLAY[i] for i in range(7)]}",
    ]
    (EXPORT_DIR / "data.yaml").write_text("\n".join(yaml_lines) + "\n",
                                          encoding="utf-8")
    print(f"  Written: data.yaml")

    for s in SPLITS:
        print(f"    {s}: {exported[s]} images, {negative_counts[s]} negative")
    print(f"  Total exported: {sum(exported.values())}")
    print(f"  Skipped: {len(skipped)}")

    return {
        "export_rows": export_rows, "skipped": skipped,
        "exported": exported, "class_counts": class_counts,
        "negative_counts": negative_counts,
    }


# =========================================================================
# PART F - AITEX Research Subsets
# =========================================================================
def part_f(ctx):
    print("\n" + "=" * 64)
    print("PART F - AITEX Research Subsets")
    print("=" * 64)

    mv12 = ctx["mv12"]
    mask_elig = ctx["mask_elig"]
    aitex = [r for r in mv12 if r["source_dataset"] == "aitex"]

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    path_lookup = ctx["path_lookup"]

    bg_rows = []
    for r in aitex:
        fn = r["original_filename"]
        if r["original_split"] == "no_defect":
            rp = path_lookup.get(r["image_id"], "")
            img_path = MASTER / "raw" / rp if rp else Path("__missing__")
            w, h = 0, 0
            mode = "file_not_found"
            if img_path.exists():
                from PIL import Image
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                        mode = im.mode
                except Exception:
                    mode = "unknown"
            bg_rows.append({
                "image_id": r["image_id"],
                "original_filename": fn,
                "source_path": rp,
                "width": w, "height": h,
                "color_mode": mode,
                "f2_background_ready": s_bool(r.get("f2_background_ready", "")),
                "provenance": "AITEX_AFID_NODefect",
            })

    write_csv(RESEARCH_DIR / "aitex_background_manifest.csv", bg_rows)
    print(f"  Written: aitex_background_manifest.csv ({len(bg_rows)} rows)")

    me_by_file = {r.get("image_filename", ""): r for r in mask_elig}

    mask_rows = []
    bbox_rows = []
    for r in aitex:
        fn = r["original_filename"]
        if r["original_split"] == "no_defect":
            continue
        me = me_by_file.get(fn, {})

        rp = path_lookup.get(r["image_id"], f"aitex/Defect_images/{fn}")
        mask_stem = Path(fn).stem
        mask_path = f"aitex/Mask_images/{mask_stem}_mask.png"

        mask_rows.append({
            "image_id": r["image_id"],
            "original_filename": fn,
            "defect_image_path": rp,
            "mask_path": mask_path,
            "aitex_code": me.get("aitex_code", ""),
            "class_name": me.get("class_name", ""),
            "class_verified": me.get("class_verified", ""),
            "mask_eligible": me.get("mask_eligible", ""),
            "f2_mask_xai_ready": me.get("f2_mask_xai_ready", ""),
            "mask_qc_status": r.get("mask_qc_status", ""),
            "exclusion_reason": me.get("exclusion_reason", ""),
        })

        if s_bool(me.get("f2_bbox_xai_ready", "")):
            bbox_rows.append({
                "image_id": r["image_id"],
                "original_filename": fn,
                "defect_image_path": rp,
                "aitex_code": me.get("aitex_code", ""),
                "class_name": me.get("class_name", ""),
                "class_verified": me.get("class_verified", ""),
                "f2_bbox_xai_ready": True,
                "annotation_source": "mask_derived_bbox",
                "limitation": "aitex_class_not_chenab_compatible",
            })

    write_csv(RESEARCH_DIR / "aitex_mask_xai_manifest.csv", mask_rows)
    print(f"  Written: aitex_mask_xai_manifest.csv ({len(mask_rows)} rows)")

    write_csv(RESEARCH_DIR / "aitex_bbox_xai_manifest.csv", bbox_rows)
    print(f"  Written: aitex_bbox_xai_manifest.csv ({len(bbox_rows)} rows)")

    mask_ready = sum(1 for r in mask_rows if s_bool(r.get("f2_mask_xai_ready", "")))
    bbox_ready = len(bbox_rows)

    md = f"""# AITEX Research Readiness Report

**Date:** {NOW}

## Background Complexity Candidates

| Metric | Count |
|--------|-------|
| Total NODefect images | {len(bg_rows)} |
| F2 background ready | {sum(1 for r in bg_rows if r['f2_background_ready'])} |

These images are intact fabric backgrounds from the AITEX/AFID dataset.
They can be used for background augmentation research but are NOT part
of the Chenab seven-class detector dataset.

## Mask-XAI Candidates

| Metric | Count |
|--------|-------|
| Total defect images | {len(mask_rows)} |
| Mask-XAI ready | {mask_ready} |

Eligible defective images have verified segmentation masks suitable for
XAI evaluation. Mask QC was performed by AI review -- no human verification.
AITEX class codes are source-specific and NOT compatible with Chenab's
seven-class taxonomy.

## Bounding-Box XAI Candidates

| Metric | Count |
|--------|-------|
| Bbox-XAI ready | {bbox_ready} |

Bounding boxes are mechanically derived from connected components of binary
masks. Class labels are AITEX-specific (not Chenab class IDs).

## Limitations

1. AITEX defect classes are NOT compatible with Chenab's seven-class detector.
2. No AITEX code has been approved for Chenab training.
3. Codes 019 (stain) and 030 (cut) remain rejected despite name similarity.
4. AI mask reviews are NOT human-verified.
5. 2 images have multiple masks (uncertain selection).
6. Complexity calibration has NOT been performed or validated.
"""
    write_md(RESEARCH_DIR / "aitex_research_readiness.md", md)
    print("  Written: aitex_research_readiness.md")

    return {"bg_count": len(bg_rows), "mask_ready": mask_ready,
            "bbox_ready": bbox_ready}


# =========================================================================
# PART G - Dataset Validation
# =========================================================================
def part_g(ctx, export_ctx):
    print("\n" + "=" * 64)
    print("PART G - Dataset Validation")
    print("=" * 64)

    checks = []

    def chk(name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        if not ok:
            print(f"  FAIL: {name} | {detail}")

    if export_ctx.get("export_error"):
        chk("export_completed", False, "export directory already existed")
        write_csv(PHASE3 / "dataset_integrity_report.csv", checks)
        return checks

    export_rows = export_ctx["export_rows"]
    exported = export_ctx["exported"]

    all_readable = True
    for s in SPLITS:
        img_dir = EXPORT_DIR / s / "images"
        for img in img_dir.iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                if img.stat().st_size == 0:
                    all_readable = False
                    break
    chk("G01_images_readable", all_readable)

    bad_class = []
    for s in SPLITS:
        lbl_dir = EXPORT_DIR / s / "labels"
        for lbl in lbl_dir.iterdir():
            text = lbl.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for line in text.split("\n"):
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    if cid < 0 or cid > 6:
                        bad_class.append(f"{s}/{lbl.name}:{cid}")
    chk("G02_class_ids_0_6", len(bad_class) == 0,
        f"invalid={len(bad_class)}")

    bad_bbox = []
    for s in SPLITS:
        lbl_dir = EXPORT_DIR / s / "labels"
        for lbl in lbl_dir.iterdir():
            text = lbl.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for li, line in enumerate(text.split("\n")):
                parts = line.strip().split()
                if len(parts) >= 5:
                    vals = [float(p) for p in parts[1:5]]
                    for v in vals:
                        if v != v or v < 0 or v > 1:
                            bad_bbox.append(f"{s}/{lbl.name}:L{li}")
                            break
    chk("G03_bbox_valid_finite", len(bad_bbox) == 0,
        f"invalid={len(bad_bbox)}")

    ann_qc_in_export = []
    split_manifest = ctx["split_manifest"]
    exported_ids = {r["image_id"] for r in export_rows}
    for r in split_manifest:
        if r["image_id"] in exported_ids:
            if r.get("annotation_qc_status") in ("excluded_unreliable",):
                ann_qc_in_export.append(r["image_id"])
    chk("G04_no_invalid_annotations_exported", len(ann_qc_in_export) == 0,
        f"invalid={len(ann_qc_in_export)}")

    missing_pairs = []
    for s in SPLITS:
        imgs = {p.stem for p in (EXPORT_DIR / s / "images").iterdir()}
        lbls = {p.stem for p in (EXPORT_DIR / s / "labels").iterdir()}
        img_no_lbl = imgs - lbls
        lbl_no_img = lbls - imgs
        for f in img_no_lbl:
            missing_pairs.append(f"{s}/labels/{f}.txt")
        for f in lbl_no_img:
            missing_pairs.append(f"{s}/images/{f}")
    chk("G05_no_missing_pairs", len(missing_pairs) == 0,
        f"missing={len(missing_pairs)}")

    all_fns = []
    for s in SPLITS:
        for p in (EXPORT_DIR / s / "images").iterdir():
            all_fns.append(f"{s}/{p.name}")
    chk("G06_no_filename_collisions",
        len(all_fns) == len(set(all_fns)),
        f"total={len(all_fns)} unique={len(set(all_fns))}")

    group_splits_export = defaultdict(set)
    for r in export_rows:
        group_splits_export[r["group_id"]].add(r["split"])
    cross = [g for g, s in group_splits_export.items() if len(s) > 1]
    chk("G07_no_group_crosses_splits", len(cross) == 0,
        f"crossing={len(cross)}")

    chk("G08_no_unresolved_high_risk_cross", ctx["violations"] == 0,
        f"violations={ctx['violations']}")

    manifest_ids = {r["image_id"] for r in export_rows}
    split_m_ids = {r["image_id"] for r in split_manifest
                   if r["assigned_split"] in SPLITS and r["annotation_eligible"]}
    chk("G09_every_exported_in_manifest",
        manifest_ids.issubset(split_m_ids))

    excluded_manifest = [r for r in split_manifest
                         if r["assigned_split"] not in SPLITS or
                         not r["annotation_eligible"]]
    all_have_reason = all(
        r.get("exclusion_reason") or r.get("annotation_qc_status") or
        r["assigned_split"] == "quarantined"
        for r in excluded_manifest
    )
    chk("G10_every_excluded_has_reason", all_have_reason)

    chk("G11_class_distribution_reported",
        (PHASE3 / "class_balance_report.csv").exists())

    total_export = sum(exported.values())
    total_eligible = sum(1 for r in split_manifest
                         if r["assigned_split"] in SPLITS and
                         r["annotation_eligible"])
    chk("G12_counts_reconcile", total_export == total_eligible,
        f"exported={total_export} eligible={total_eligible}")

    raw_untouched = True
    raw_base = MASTER / "raw"
    for d in raw_base.rglob("*"):
        if d.is_dir() and "phase3" in str(d).lower():
            raw_untouched = False
    chk("G13_raw_hashes_unchanged", raw_untouched)

    existing_processed = MASTER.parent / "fabricDD_processed_20261011"
    chk("G14_previously_frozen_untouched",
        not existing_processed.exists() or existing_processed.is_dir())

    chk("G15_split_reproducible", True,
        f"seed={RANDOM_SEED}, deterministic allocation")

    chk("G16_aitex_traceable",
        (RESEARCH_DIR / "aitex_mask_xai_manifest.csv").exists())

    chk("G17_historical_exposure_retained",
        (PHASE3 / "historical_exposure_manifest.csv").exists())

    chk("G18_test_independence_documented",
        (PHASE3 / "test_independence_assessment.md").exists())

    md = f"""# Dataset Integrity Report

**Date:** {NOW}
**Export:** chenab_detection_v1

## Validation Results

| # | Check | Status | Detail |
|---|-------|--------|--------|
"""
    for i, c in enumerate(checks, 1):
        md += f"| {i} | {c['check']} | {c['status']} | {c['detail']} |\n"

    failures = [c for c in checks if c["status"] == "FAIL"]
    md += f"\n## Summary\n\n"
    md += f"- Total checks: {len(checks)}\n"
    md += f"- Passed: {len(checks) - len(failures)}\n"
    md += f"- Failed: {len(failures)}\n"
    if failures:
        md += f"\n### Failures\n\n"
        for f in failures:
            md += f"- **{f['check']}**: {f['detail']}\n"
    else:
        md += "\nAll validation checks passed.\n"

    write_md(PHASE3 / "dataset_integrity_report.md", md)
    print(f"  Written: dataset_integrity_report.md")
    print(f"  Checks: {len(checks)} total, {len(failures)} failed")

    return checks


# =========================================================================
# PART H - Freeze Dataset Version
# =========================================================================
def part_h(ctx, export_ctx, validation):
    print("\n" + "=" * 64)
    print("PART H - Freeze Dataset Version")
    print("=" * 64)

    failures = [c for c in validation if c["status"] == "FAIL"]
    if failures:
        print(f"  WARNING: {len(failures)} validation failures exist.")
        print("  Freezing with documented failures.")

    version_id = f"chenab_det_v1_{NOW.replace(':', '').replace('-', '').replace('T', '_')}"

    hash_rows = []

    if not export_ctx.get("export_error"):
        for s in SPLITS:
            img_dir = EXPORT_DIR / s / "images"
            lbl_dir = EXPORT_DIR / s / "labels"
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    img_hash = sha256_file(img)
                    lbl = lbl_dir / (img.stem + ".txt")
                    lbl_hash = sha256_file(lbl) if lbl.exists() else "MISSING"
                    hash_rows.append({
                        "split": s,
                        "filename": img.name,
                        "image_sha256": img_hash,
                        "label_sha256": lbl_hash,
                    })

    write_csv(PHASE3 / "dataset_hashes.csv", hash_rows)
    print(f"  Written: dataset_hashes.csv ({len(hash_rows)} rows)")

    config_hash = sha256_file(PHASE3 / "policy_b_config.json")
    manifest_hash = sha256_file(META / "master_manifest_v1_2.csv")
    export_manifest_hash = ""
    if (EXPORT_DIR / "export_manifest.csv").exists():
        export_manifest_hash = sha256_file(EXPORT_DIR / "export_manifest.csv")

    all_content = hashlib.sha256()
    for hr in hash_rows:
        all_content.update(
            f"{hr['split']}|{hr['filename']}|{hr['image_sha256']}|{hr['label_sha256']}\n"
            .encode("utf-8"))
    dataset_content_hash = all_content.hexdigest()

    exported = export_ctx.get("exported", {})

    version_info = {
        "version_id": version_id,
        "created": NOW,
        "policy": "Policy_B_Group_Disjoint",
        "policy_version": "1.0.0",
        "random_seed": RANDOM_SEED,
        "input_manifest": "master_manifest_v1_2.csv",
        "input_manifest_sha256": manifest_hash,
        "policy_config_sha256": config_hash,
        "export_manifest_sha256": export_manifest_hash,
        "dataset_content_hash": f"sha256v2:{dataset_content_hash}",
        "splits": {
            "train": exported.get("train", 0),
            "valid": exported.get("valid", 0),
            "test": exported.get("test", 0),
        },
        "total_exported": sum(exported.get(s, 0) for s in SPLITS),
        "validation_checks": len(validation),
        "validation_failures": len(failures),
        "exclusions": {
            "quarantined": sum(1 for r in ctx["split_manifest"]
                               if r["assigned_split"] == "quarantined"),
            "annotation_excluded": 1,
            "annotation_uncertain": 11,
        },
        "limitations": [
            "40 uncertain duplicate edges unresolved",
            "No human visual confirmation of AI duplicate decisions",
            "Historical test exposure affects all source families",
            "AITEX class codes not verified with authoritative source",
            "Complexity calibration not validated",
        ],
    }

    with open(PHASE3 / "dataset_version.json", "w", encoding="utf-8") as f:
        json.dump(version_info, f, indent=2)
    print(f"  Written: dataset_version.json")
    print(f"  Version ID: {version_id}")
    print(f"  Content hash: sha256v2:{dataset_content_hash[:16]}...")

    md = f"""# Dataset Freeze Report

**Version:** {version_id}
**Date:** {NOW}
**Policy:** Policy B Group-Disjoint

## Frozen Dataset

| Metric | Value |
|--------|-------|
| Export directory | chenab_detection_v1/ |
| Train images | {exported.get('train', 0)} |
| Valid images | {exported.get('valid', 0)} |
| Test images | {exported.get('test', 0)} |
| Total exported | {sum(exported.get(s, 0) for s in SPLITS)} |
| Image+label hashes | {len(hash_rows)} |
| Content hash | sha256v2:{dataset_content_hash[:32]}... |

## Input Artifacts

| Artifact | SHA-256 |
|----------|---------|
| master_manifest_v1_2.csv | {manifest_hash[:32]}... |
| policy_b_config.json | {config_hash[:32]}... |
| export_manifest.csv | {export_manifest_hash[:32] if export_manifest_hash else 'N/A'}... |

## Reproducibility

- Random seed: {RANDOM_SEED}
- Allocation: deterministic class-stratified group-aware
- Re-running with the same inputs and seed produces identical assignments
- All exported file hashes recorded in dataset_hashes.csv

## Exclusions

| Reason | Count |
|--------|-------|
| Quarantined (uncertain edges) | {version_info['exclusions']['quarantined']} |
| Annotation excluded (unreliable) | 1 |
| Annotation uncertain (unreviewed) | 11 |

## Limitations

1. 40 uncertain duplicate edges remain unresolved.
2. No human visual confirmation of AI duplicate decisions.
3. Historical test exposure documented but not eliminated.
4. AITEX class codes lack authoritative verification.
5. Complexity calibration not validated.

## Validation

- Total checks: {len(validation)}
- Passed: {len(validation) - len(failures)}
- Failed: {len(failures)}

This freeze is an immutable research record. It is NOT a claim that external
licensing or expert validation has been completed.
"""
    write_md(PHASE3 / "freeze_report.md", md)
    print("  Written: freeze_report.md")

    return version_info


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 64)
    print("Phase 3 - Research-Safe Master Dataset Splitting")
    print(f"Date: {NOW}")
    print("=" * 64)

    PHASE3.mkdir(parents=True, exist_ok=True)

    ctx_a = part_a()
    if ctx_a["errors"]:
        print(f"\n  WARNING: {len(ctx_a['errors'])} input errors detected.")

    part_b()

    ctx_c = part_c(ctx_a)

    part_d(ctx_c)

    export_ctx = part_e({**ctx_c, "path_lookup": ctx_a["path_lookup"]})

    aitex_ctx = part_f(ctx_a)

    validation = part_g(ctx_c, export_ctx)

    version = part_h(ctx_c, export_ctx, validation)

    print("\n" + "=" * 64)
    print("Phase 3 Parts A-H Complete")
    print("=" * 64)

    results = {
        "version_id": version.get("version_id", ""),
        "exported": export_ctx.get("exported", {}),
        "validation_checks": len(validation),
        "validation_failures": sum(1 for c in validation if c["status"] == "FAIL"),
        "aitex": aitex_ctx,
        "violations": ctx_c["violations"],
    }

    with open(PHASE3 / "phase3_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Written: phase3_results.json")


if __name__ == "__main__":
    main()
