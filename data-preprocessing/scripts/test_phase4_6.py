"""
Phase 4.6 targeted tests.
Run: python -I test_phase4_6.py
"""

import csv
import json
import sys
import io
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
PHASE3 = MASTER / "reports" / "phase3"
PHASE4 = MASTER / "reports" / "phase4"
EXPORT = MASTER / "processed" / "chenab_detection_v1"
MASK_DS = MASTER / "research" / "chenab_xai_masks_v1"
RAW = MASTER / "raw"

PASS = FAIL = 0


def read_csv_file(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f" | {detail}" if detail else ""))
        FAIL += 1


# =========================================================================
# 1. Annotation queue integrity
# =========================================================================
def test_queue_integrity():
    print("\n--- 1. Annotation queue integrity ---")
    verified = read_csv_file(MASK_DS / "manifests" / "annotation_queue_verified.csv")
    check("70 queue entries", len(verified) == 70, f"got {len(verified)}")

    eligible = [v for v in verified if v["eligible"] in ("True", "true")]
    check("eligible entries > 0", len(eligible) > 0, f"eligible={len(eligible)}")


# =========================================================================
# 2. Original image and bbox references
# =========================================================================
def test_image_bbox_refs():
    print("\n--- 2. Image and bbox references ---")
    proposals = read_csv_file(MASK_DS / "manifests" / "mask_proposal_manifest.csv")
    check("proposals exist", len(proposals) > 0)

    ok_props = [p for p in proposals if p["status"] == "proposed"]
    for p in ok_props[:5]:
        fn = p["source_filename"]
        split = p["split"]
        img = EXPORT / split / "images" / fn
        check(f"img_exists_{fn[:25]}", img.exists())

        lbl = EXPORT / split / "labels" / (Path(fn).stem + ".txt")
        check(f"lbl_exists_{fn[:25]}", lbl.exists())


# =========================================================================
# 3. Split eligibility
# =========================================================================
def test_split_eligibility():
    print("\n--- 3. Split eligibility ---")
    verified = read_csv_file(MASK_DS / "manifests" / "annotation_queue_verified.csv")
    non_test = all(v["split"] in ("train", "valid") for v in verified)
    check("all train/valid", non_test)


# =========================================================================
# 4. Test/quarantine protection
# =========================================================================
def test_protection():
    print("\n--- 4. Test/quarantine protection ---")
    fsm = read_csv_file(PHASE3 / "final_split_manifest.csv")
    quarantined_ids = {r["image_id"] for r in fsm
                       if r["assigned_split"] == "quarantined"}

    proposals = read_csv_file(MASK_DS / "manifests" / "mask_proposal_manifest.csv")
    q_in_proposals = [p for p in proposals if p["image_id"] in quarantined_ids]
    check("no quarantined in proposals", len(q_in_proposals) == 0)

    test_in_proposals = [p for p in proposals if p.get("split") == "test"]
    check("no test in proposals", len(test_in_proposals) == 0)


# =========================================================================
# 5. Mask dimensions
# =========================================================================
def test_mask_dimensions():
    print("\n--- 5. Mask dimensions ---")
    proposals = read_csv_file(MASK_DS / "manifests" / "mask_proposal_manifest.csv")
    ok_props = [p for p in proposals if p["status"] == "proposed"]

    checked = 0
    for p in ok_props[:10]:
        mask_path = MASK_DS / "ai_proposed_masks" / p["mask_filename"]
        if mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                ih = int(p["img_height"])
                iw = int(p["img_width"])
                check(f"dim_{p['instance_id'][:25]}",
                      m.shape[0] == ih and m.shape[1] == iw,
                      f"mask={m.shape} img=({ih},{iw})")
                checked += 1
    check("checked_at_least_5_dims", checked >= 5, f"checked={checked}")


# =========================================================================
# 6. Binary mask validity
# =========================================================================
def test_binary_validity():
    print("\n--- 6. Binary mask validity ---")
    proposals = read_csv_file(MASK_DS / "manifests" / "mask_proposal_manifest.csv")
    ok_props = [p for p in proposals if p["status"] == "proposed"]

    checked = 0
    for p in ok_props[:10]:
        mask_path = MASK_DS / "ai_proposed_masks" / p["mask_filename"]
        if mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                vals = set(np.unique(m))
                check(f"binary_{p['instance_id'][:25]}",
                      vals.issubset({0, 255}), f"vals={vals}")
                checked += 1
    check("checked_at_least_5_binary", checked >= 5, f"checked={checked}")


# =========================================================================
# 7. Mask-to-instance pairing
# =========================================================================
def test_instance_pairing():
    print("\n--- 7. Mask-to-instance pairing ---")
    proposals = read_csv_file(MASK_DS / "manifests" / "mask_proposal_manifest.csv")
    ref = read_csv_file(MASK_DS / "manifests" / "xai_mask_reference_manifest.csv")

    prop_ids = {p["instance_id"] for p in proposals}
    ref_ids = {r["instance_id"] for r in ref}
    check("ref covers all proposals", prop_ids == ref_ids,
          f"prop_only={len(prop_ids-ref_ids)} ref_only={len(ref_ids-prop_ids)}")


# =========================================================================
# 8. Segmentation model provenance
# =========================================================================
def test_model_provenance():
    print("\n--- 8. Model provenance ---")
    manifest = json.loads(
        (MASK_DS / "manifests" / "segmentation_model_manifest.json").read_text("utf-8"))
    check("model is SAM", "SAM" in manifest.get("model", ""))
    check("architecture vit_b", manifest.get("architecture") == "vit_b")
    check("checkpoint hash present", len(manifest.get("checkpoint_sha256", "")) == 64)
    check("source URL present", "fbaipublicfiles" in manifest.get("source_url", ""))


# =========================================================================
# 9. AI review traceability
# =========================================================================
def test_review_traceability():
    print("\n--- 9. AI review traceability ---")
    decisions = read_csv_file(MASK_DS / "manifests" / "ai_mask_review_decisions.csv")
    check("decisions exist", len(decisions) > 0)

    valid_decisions = {"ai_accepted", "ai_correction_proposed", "uncertain",
                       "pending_visual_review", "excluded_unreliable"}
    all_valid = all(d["ai_review_decision"] in valid_decisions for d in decisions)
    check("all decisions valid", all_valid)

    check("all have reason",
          all(d.get("ai_review_reason") for d in decisions))


# =========================================================================
# 10. No fabricated human verification
# =========================================================================
def test_no_fabricated_verification():
    print("\n--- 10. No fabricated verification ---")
    decisions = read_csv_file(MASK_DS / "manifests" / "ai_mask_review_decisions.csv")
    human = [d for d in decisions
             if d.get("human_verified") in ("True", "true")]
    check("no human_verified=True", len(human) == 0, f"got {len(human)}")

    check("visual_review_not_claimed",
          all(d.get("visual_review_performed") in ("False", "false", "")
              for d in decisions))

    ref = read_csv_file(MASK_DS / "manifests" / "xai_mask_reference_manifest.csv")
    human_ref = [r for r in ref
                 if r.get("human_verified") in ("True", "true")]
    check("no human_verified in ref", len(human_ref) == 0)


# =========================================================================
# 11. Reproducibility
# =========================================================================
def test_reproducibility():
    print("\n--- 11. Reproducibility ---")
    model = json.loads(
        (MASK_DS / "manifests" / "segmentation_model_manifest.json").read_text("utf-8"))
    check("model version recorded", bool(model.get("framework")))
    check("checkpoint recorded", bool(model.get("checkpoint")))


# =========================================================================
# 12. Source immutability
# =========================================================================
def test_source_immutability():
    print("\n--- 12. Source immutability ---")
    bad = []
    for d in RAW.rglob("*"):
        if d.is_dir() and "phase4_6" in str(d).lower():
            bad.append(str(d))
    check("no phase4_6 dirs under raw/", len(bad) == 0)


# =========================================================================
# 13. Frozen dataset unchanged
# =========================================================================
def test_frozen_dataset():
    print("\n--- 13. Frozen dataset ---")
    em = read_csv_file(EXPORT / "export_manifest.csv")
    check("export 2655", len(em) == 2655, f"got {len(em)}")
    check("data.yaml exists", (EXPORT / "data.yaml").exists())


# =========================================================================
# 14. Coverage matrix accuracy
# =========================================================================
def test_coverage_matrix():
    print("\n--- 14. Coverage matrix ---")
    coverage = read_csv_file(MASK_DS / "reports" / "complexity_class_coverage.csv")
    check("coverage rows exist", len(coverage) > 0)

    classes_covered = {int(r["class_id"]) for r in coverage}
    check("all 7 classes in coverage", len(classes_covered) == 7,
          f"classes={sorted(classes_covered)}")


# =========================================================================
# 15. Manifest correctness
# =========================================================================
def test_manifest_correctness():
    print("\n--- 15. Manifest correctness ---")
    ref = read_csv_file(MASK_DS / "manifests" / "xai_mask_reference_manifest.csv")
    check("ref has instance_id", all(r.get("instance_id") for r in ref))
    check("ref has image_id", all(r.get("image_id") for r in ref))
    check("ref has class_id", all(r.get("class_id") for r in ref))
    check("ref has mask_path", all(r.get("mask_path") for r in ref
                                   if r.get("ai_review_status") != "excluded_unreliable"))

    class_ids = {int(r["class_id"]) for r in ref}
    check("class IDs 0-6", class_ids.issubset(set(range(7))),
          f"classes={sorted(class_ids)}")


# =========================================================================
# 16. QC failure handling
# =========================================================================
def test_qc_failure_handling():
    print("\n--- 16. QC failure handling ---")
    qc = read_csv_file(MASK_DS / "manifests" / "mask_qc_results.csv")
    decisions = read_csv_file(MASK_DS / "manifests" / "ai_mask_review_decisions.csv")

    failed_qc = {q["instance_id"] for q in qc if q["qc_status"] == "failed"}
    excluded = {d["instance_id"] for d in decisions
                if d["ai_review_decision"] == "excluded_unreliable"}

    if failed_qc:
        check("failed QC excluded", failed_qc.issubset(excluded),
              f"not_excluded={failed_qc - excluded}")
    else:
        check("no failed QC to handle", True)

    check("qc summary exists",
          (MASK_DS / "reports" / "mask_qc_summary.md").exists())


# =========================================================================
# 17. Final report exists
# =========================================================================
def test_final_report():
    print("\n--- 17. Final report ---")
    check("final report exists",
          (MASK_DS / "reports" / "PHASE_4_6_FINAL_REPORT.md").exists())
    check("results json exists",
          (MASK_DS / "reports" / "phase4_6_results.json").exists())

    results = json.loads(
        (MASK_DS / "reports" / "phase4_6_results.json").read_text("utf-8"))
    check("results has status", bool(results.get("status")))
    check("results has model", bool(results.get("model")))
    check("results has checkpoint", len(results.get("checkpoint_sha256", "")) == 64)


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 64)
    print("Phase 4.6 Tests")
    print("=" * 64)

    test_queue_integrity()
    test_image_bbox_refs()
    test_split_eligibility()
    test_protection()
    test_mask_dimensions()
    test_binary_validity()
    test_instance_pairing()
    test_model_provenance()
    test_review_traceability()
    test_no_fabricated_verification()
    test_reproducibility()
    test_source_immutability()
    test_frozen_dataset()
    test_coverage_matrix()
    test_manifest_correctness()
    test_qc_failure_handling()
    test_final_report()

    print("\n" + "=" * 64)
    total = PASS + FAIL
    print(f"Result: {PASS}/{total} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
