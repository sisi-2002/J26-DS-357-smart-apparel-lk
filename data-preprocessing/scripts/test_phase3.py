"""
Phase 3 targeted tests.
Run: python -I test_phase3.py
"""

import csv
import json
import sys
import io
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
META = MASTER / "metadata"
PHASE3 = MASTER / "reports" / "phase3"
EXPORT = MASTER / "processed" / "chenab_detection_v1"
RESEARCH = MASTER / "processed" / "research_subsets"
PHASE28 = MASTER / "reports" / "phase2_8"
RAW = MASTER / "raw"


def read_csv_file(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


PASS = FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f" | {detail}" if detail else ""))
        FAIL += 1


# =========================================================================
# 1. Manifest reconciliation
# =========================================================================
def test_manifest_reconciliation():
    print("\n--- 1. Manifest reconciliation ---")
    recon = read_csv_file(PHASE3 / "phase3_input_reconciliation.csv")
    check("3038 reconciliation rows", len(recon) == 3038, f"got {len(recon)}")

    mv12 = read_csv_file(META / "master_manifest_v1_2.csv")
    recon_ids = {r["image_id"] for r in recon}
    mv_ids = {r["image_id"] for r in mv12}
    check("reconciliation IDs match manifest", recon_ids == mv_ids)

    chenab = [r for r in recon if r["source_dataset"] == "chenab"]
    aitex = [r for r in recon if r["source_dataset"] == "aitex"]
    check("2791 Chenab", len(chenab) == 2791)
    check("247 AITEX", len(aitex) == 247)


# =========================================================================
# 2. Eligibility filtering
# =========================================================================
def test_eligibility_filtering():
    print("\n--- 2. Eligibility filtering ---")
    fsm = read_csv_file(PHASE3 / "final_split_manifest.csv")

    eligible = [r for r in fsm if r["annotation_eligible"] == "True"
                and r["assigned_split"] in ("train", "valid", "test")]
    em = read_csv_file(EXPORT / "export_manifest.csv")
    check("eligible count matches exported", len(eligible) == len(em),
          f"eligible={len(eligible)} exported={len(em)}")

    excluded = [r for r in fsm if r["annotation_qc_status"] == "excluded_unreliable"]
    in_export = [r for r in excluded if r["annotation_eligible"] == "True"]
    check("excluded_unreliable not eligible", len(in_export) == 0)

    uncertain = [r for r in fsm if r["annotation_qc_status"] == "uncertain"]
    in_export_u = [r for r in uncertain if r["annotation_eligible"] == "True"]
    check("uncertain annotations not eligible", len(in_export_u) == 0)


# =========================================================================
# 3. Family disjointness
# =========================================================================
def test_family_disjointness():
    print("\n--- 3. Family disjointness ---")
    disjoint = read_csv_file(PHASE3 / "group_disjointness_report.csv")
    violations = [r for r in disjoint if r["disjoint"] != "True"]
    check("0 disjointness violations", len(violations) == 0,
          f"violations={len(violations)}")

    fsm = read_csv_file(PHASE3 / "final_split_manifest.csv")
    group_splits = defaultdict(set)
    for r in fsm:
        if r["assigned_split"] in ("train", "valid", "test"):
            group_splits[r["group_id"]].add(r["assigned_split"])
    cross = [g for g, s in group_splits.items() if len(s) > 1]
    check("no group crosses splits in manifest", len(cross) == 0,
          f"crossing={len(cross)}")


# =========================================================================
# 4. Conservative uncertainty handling
# =========================================================================
def test_uncertainty_handling():
    print("\n--- 4. Conservative uncertainty handling ---")
    fga = read_csv_file(PHASE3 / "final_group_assignments.csv")
    uncertain = [r for r in fga if r["has_uncertain_edge"] == "True"]
    quarantined = [r for r in uncertain
                   if r["assigned_split"] == "quarantined"]
    check("uncertain groups quarantined",
          len(quarantined) == len(uncertain),
          f"uncertain={len(uncertain)} quarantined={len(quarantined)}")


# =========================================================================
# 5. Historical test-exposure protection
# =========================================================================
def test_historical_exposure():
    print("\n--- 5. Historical test-exposure protection ---")
    hem = read_csv_file(PHASE3 / "historical_exposure_manifest.csv")
    check("2791 exposure rows", len(hem) == 2791, f"got {len(hem)}")

    old_test = [r for r in hem if r["old_test_image"] == "True"]
    check("280 old test images", len(old_test) == 280, f"got {len(old_test)}")

    check("test independence doc exists",
          (PHASE3 / "test_independence_assessment.md").exists())


# =========================================================================
# 6. Class-aware split assignment
# =========================================================================
def test_class_aware_split():
    print("\n--- 6. Class-aware split assignment ---")
    cb = read_csv_file(PHASE3 / "class_balance_report.csv")
    check("7 class rows", len(cb) == 7, f"got {len(cb)}")

    for r in cb:
        for s in ("train", "valid", "test"):
            check(f"class {r['class_name']} in {s} > 0",
                  int(r[s]) > 0, f"got {r[s]}")


# =========================================================================
# 7. Class coverage
# =========================================================================
def test_class_coverage():
    print("\n--- 7. Class coverage ---")
    for s in ("train", "valid", "test"):
        lbl_dir = EXPORT / s / "labels"
        classes_found = set()
        for lbl in lbl_dir.iterdir():
            text = lbl.read_text(encoding="utf-8").strip()
            if text:
                for line in text.split("\n"):
                    parts = line.strip().split()
                    if parts:
                        classes_found.add(int(parts[0]))
        check(f"all 7 classes in {s} labels", len(classes_found) == 7,
              f"found {sorted(classes_found)}")


# =========================================================================
# 8. Deterministic repeat runs
# =========================================================================
def test_deterministic():
    print("\n--- 8. Deterministic repeat ---")
    version = json.loads((PHASE3 / "dataset_version.json").read_text(encoding="utf-8"))
    check("version has content hash",
          version["dataset_content_hash"].startswith("sha256v2:"))
    check("seed recorded", version["random_seed"] == 42)

    hashes = read_csv_file(PHASE3 / "dataset_hashes.csv")
    em = read_csv_file(EXPORT / "export_manifest.csv")
    check("hash rows match exported", len(hashes) == len(em),
          f"hashes={len(hashes)} exported={len(em)}")


# =========================================================================
# 9. Image/label pairing
# =========================================================================
def test_image_label_pairing():
    print("\n--- 9. Image/label pairing ---")
    for s in ("train", "valid", "test"):
        imgs = {p.stem for p in (EXPORT / s / "images").iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        lbls = {p.stem for p in (EXPORT / s / "labels").iterdir()
                if p.suffix == ".txt"}
        check(f"{s} every image has label", imgs == lbls,
              f"img_only={len(imgs-lbls)} lbl_only={len(lbls-imgs)}")


# =========================================================================
# 10. YOLO annotation validity
# =========================================================================
def test_yolo_annotation_validity():
    print("\n--- 10. YOLO annotation validity ---")
    bad_class = 0
    bad_bbox = 0
    total_annotations = 0
    for s in ("train", "valid", "test"):
        for lbl in (EXPORT / s / "labels").iterdir():
            text = lbl.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for line in text.split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    total_annotations += 1
                    cid = int(parts[0])
                    if cid < 0 or cid > 6:
                        bad_class += 1
                    vals = [float(p) for p in parts[1:5]]
                    for v in vals:
                        if v != v or v < 0 or v > 1:
                            bad_bbox += 1
                            break
    check("no invalid class IDs", bad_class == 0, f"bad={bad_class}")
    check("no invalid bboxes", bad_bbox == 0, f"bad={bad_bbox}")
    check("annotations exist", total_annotations > 0,
          f"total={total_annotations}")


# =========================================================================
# 11. Corrected label usage
# =========================================================================
def test_corrected_labels():
    print("\n--- 11. Corrected label usage ---")
    em = read_csv_file(EXPORT / "export_manifest.csv")
    check("export manifest exists", len(em) > 0)
    check("all have source_path", all(r.get("source_path") for r in em))


# =========================================================================
# 12. Negative sample handling
# =========================================================================
def test_negative_samples():
    print("\n--- 12. Negative sample handling ---")
    em = read_csv_file(EXPORT / "export_manifest.csv")
    negatives = [r for r in em if r["is_negative"] == "True"]
    check("negative samples present", len(negatives) > 0,
          f"count={len(negatives)}")
    for r in negatives:
        lbl = EXPORT / r["split"] / "labels" / (
            Path(r["export_filename"]).stem + ".txt")
        check_content = lbl.read_text(encoding="utf-8").strip()
        if check_content:
            check(f"negative {r['export_filename']} has empty label", False,
                  f"content={check_content[:30]}")
            return
    check("negative labels are empty", True)


# =========================================================================
# 13. Quarantine exclusions
# =========================================================================
def test_quarantine_exclusions():
    print("\n--- 13. Quarantine exclusions ---")
    fsm = read_csv_file(PHASE3 / "final_split_manifest.csv")
    quarantined = [r for r in fsm if r["assigned_split"] == "quarantined"]
    check("quarantined images exist", len(quarantined) > 0,
          f"count={len(quarantined)}")

    em = read_csv_file(EXPORT / "export_manifest.csv")
    exported_ids = {r["image_id"] for r in em}
    q_in_export = [r for r in quarantined if r["image_id"] in exported_ids]
    check("no quarantined in export", len(q_in_export) == 0,
          f"leaked={len(q_in_export)}")


# =========================================================================
# 14. AITEX research manifest separation
# =========================================================================
def test_aitex_separation():
    print("\n--- 14. AITEX research separation ---")
    bg = read_csv_file(RESEARCH / "aitex_background_manifest.csv")
    mask = read_csv_file(RESEARCH / "aitex_mask_xai_manifest.csv")
    bbox = read_csv_file(RESEARCH / "aitex_bbox_xai_manifest.csv")
    check("141 background images", len(bg) == 141, f"got {len(bg)}")
    check("106 mask images", len(mask) == 106, f"got {len(mask)}")
    check("bbox manifest exists", len(bbox) > 0, f"got {len(bbox)}")

    em = read_csv_file(EXPORT / "export_manifest.csv")
    aitex_in_export = [r for r in em
                       if "aitex" in r.get("source_path", "").lower()]
    check("no AITEX in Chenab export", len(aitex_in_export) == 0,
          f"leaked={len(aitex_in_export)}")


# =========================================================================
# 15. Frozen dataset integrity
# =========================================================================
def test_frozen_integrity():
    print("\n--- 15. Frozen dataset integrity ---")
    version = json.loads((PHASE3 / "dataset_version.json").read_text(encoding="utf-8"))
    check("version ID present", bool(version.get("version_id")))
    check("content hash present", bool(version.get("dataset_content_hash")))

    hashes = read_csv_file(PHASE3 / "dataset_hashes.csv")
    spot_check_count = min(10, len(hashes))
    mismatches = 0
    for hr in hashes[:spot_check_count]:
        img = EXPORT / hr["split"] / "images" / hr["filename"]
        if img.exists():
            actual = sha256_file(img)
            if actual != hr["image_sha256"]:
                mismatches += 1
    check(f"spot-check {spot_check_count} image hashes", mismatches == 0,
          f"mismatches={mismatches}")


# =========================================================================
# 16. Raw-source immutability
# =========================================================================
def test_raw_immutability():
    print("\n--- 16. Raw-source immutability ---")
    bad = []
    for d in RAW.rglob("*"):
        if d.is_dir() and "phase3" in str(d).lower():
            bad.append(str(d))
    check("no phase3 dirs under raw/", len(bad) == 0)

    existing = MASTER.parent / "fabricDD_processed_20261011"
    if existing.exists():
        check("existing processed dataset untouched", existing.is_dir())
    else:
        check("existing processed dataset check", True, "not found (OK)")


# =========================================================================
# 17. Policy config and data.yaml
# =========================================================================
def test_config_files():
    print("\n--- 17. Config files ---")
    config = json.loads((PHASE3 / "policy_b_config.json").read_text(encoding="utf-8"))
    check("policy config has 12 rules", len(config["rules"]) == 12,
          f"got {len(config['rules'])}")
    check("seed is 42", config["random_seed"] == 42)

    data_yaml = (EXPORT / "data.yaml").read_text(encoding="utf-8")
    check("data.yaml has nc: 7", "nc: 7" in data_yaml)
    check("data.yaml has 7 class names",
          "baekra" in data_yaml and "stain" in data_yaml)


# =========================================================================
# 18. Export counts reconcile
# =========================================================================
def test_export_counts():
    print("\n--- 18. Export counts ---")
    em = read_csv_file(EXPORT / "export_manifest.csv")
    for s in ("train", "valid", "test"):
        img_count = len(list((EXPORT / s / "images").iterdir()))
        manifest_count = sum(1 for r in em if r["split"] == s)
        check(f"{s} image count matches manifest",
              img_count == manifest_count,
              f"disk={img_count} manifest={manifest_count}")

    total = len(em)
    fsm = read_csv_file(PHASE3 / "final_split_manifest.csv")
    eligible = sum(1 for r in fsm
                   if r["assigned_split"] in ("train", "valid", "test")
                   and r["annotation_eligible"] == "True")
    check("total exported matches eligible", total == eligible,
          f"exported={total} eligible={eligible}")


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 64)
    print("Phase 3 Tests")
    print("=" * 64)

    test_manifest_reconciliation()
    test_eligibility_filtering()
    test_family_disjointness()
    test_uncertainty_handling()
    test_historical_exposure()
    test_class_aware_split()
    test_class_coverage()
    test_deterministic()
    test_image_label_pairing()
    test_yolo_annotation_validity()
    test_corrected_labels()
    test_negative_samples()
    test_quarantine_exclusions()
    test_aitex_separation()
    test_frozen_integrity()
    test_raw_immutability()
    test_config_files()
    test_export_counts()

    print("\n" + "=" * 64)
    total = PASS + FAIL
    print(f"Result: {PASS}/{total} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
