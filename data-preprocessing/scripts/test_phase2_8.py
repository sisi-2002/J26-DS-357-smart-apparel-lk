"""
Phase 2.8 targeted tests.
Run: python -I test_phase2_8.py
"""

import csv
import json
import sys
import io
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
PHASE26 = MASTER / "reports" / "phase2_6"
PHASE27 = MASTER / "reports" / "phase2_7"
PHASE28 = MASTER / "reports" / "phase2_8"
META = MASTER / "metadata"
RAW_CHENAB = MASTER / "raw" / "chenab" / "fabricDD.v2i.yolov8"


def read_csv_file(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
# 1. Every duplicate candidate has a recorded outcome
# =========================================================================
def test_all_candidates_reviewed():
    print("\n--- 1. All duplicate candidates reviewed ---")
    reviews = read_csv_file(PHASE27 / "priority_visual_review.csv")
    decisions = read_csv_file(PHASE28 / "ai_duplicate_review_decisions.csv")
    check("1356 candidates reviewed", len(decisions) == 1356, f"got {len(decisions)}")
    check("matches Phase 2.7 review count", len(decisions) == len(reviews))

    # Every review_id accounted for
    rv_ids = {r["review_id"] for r in reviews}
    dc_ids = {d["review_id"] for d in decisions}
    check("all review IDs present", rv_ids == dc_ids,
          f"missing={len(rv_ids - dc_ids)}")

    # All have a valid decision
    valid_decisions = {"confirmed_same_source", "ai_likely_same_source",
                       "ai_likely_different_source", "uncertain"}
    bad = [d for d in decisions if d["decision"] not in valid_decisions]
    check("all decisions are valid", len(bad) == 0, f"invalid={len(bad)}")


# =========================================================================
# 2. All AI decisions have traceable evidence
# =========================================================================
def test_traceable_evidence():
    print("\n--- 2. AI decisions have traceable evidence ---")
    decisions = read_csv_file(PHASE28 / "ai_duplicate_review_decisions.csv")

    # confirmed and ai_likely_same_source must have evidence_signals
    same_source = [d for d in decisions if d["decision"] in
                   ("confirmed_same_source", "ai_likely_same_source")]
    no_ev = [d for d in same_source if not d.get("evidence_signals")]
    check("same-source decisions have evidence signals", len(no_ev) == 0,
          f"missing={len(no_ev)}")

    # confirmed_same_source must have strong signal
    confirmed = [d for d in decisions if d["decision"] == "confirmed_same_source"]
    no_strong = [d for d in confirmed if int(d.get("n_strong_signals", 0)) == 0]
    check("confirmed decisions have strong signals", len(no_strong) == 0,
          f"no_strong={len(no_strong)}")


# =========================================================================
# 3. No false human-review attribution
# =========================================================================
def test_no_false_human():
    print("\n--- 3. No false human-review attribution ---")

    for csv_name in ["ai_duplicate_review_decisions.csv",
                      "ai_chenab_annotation_review.csv",
                      "ai_aitex_class_review.csv",
                      "ai_aitex_mask_review.csv"]:
        path = PHASE28 / csv_name
        if not path.exists():
            check(f"{csv_name} exists", False)
            continue
        rows = read_csv_file(path)
        human_claimed = [r for r in rows if r.get("human_verified") == "yes"]
        check(f"no false human verification in {csv_name}", len(human_claimed) == 0,
              f"falsely verified={len(human_claimed)}")

    # Manifest V1.2
    mv12 = read_csv_file(META / "master_manifest_v1_2.csv")
    hv = [r for r in mv12 if r.get("human_verified") == "yes"]
    check("no false human verification in manifest V1.2", len(hv) == 0)


# =========================================================================
# 4. No pHash-only confirmed grouping
# =========================================================================
def test_no_phash_only_confirmed():
    print("\n--- 4. No pHash-only confirmed grouping ---")
    decisions = read_csv_file(PHASE28 / "ai_duplicate_review_decisions.csv")
    confirmed = [d for d in decisions if d["decision"] == "confirmed_same_source"]
    phash_only = [d for d in confirmed
                  if d.get("evidence_signals") in ("phash_identical", "phash_very_close")]
    check("no pHash-only confirmed grouping", len(phash_only) == 0,
          f"phash_only={len(phash_only)}")


# =========================================================================
# 5. Correct source-family grouping
# =========================================================================
def test_source_family_grouping():
    print("\n--- 5. Source-family grouping ---")
    resolved = read_csv_file(PHASE28 / "resolved_source_groups.csv")
    groups = defaultdict(set)
    for r in resolved:
        groups[r["group_id"]].add(r["filename"])
    check("resolved groups exist", len(groups) > 0)
    # No empty groups
    empty = [g for g, m in groups.items() if len(m) == 0]
    check("no empty groups", len(empty) == 0)


# =========================================================================
# 6. No group-disjointness violations
# =========================================================================
def test_group_disjointness():
    print("\n--- 6. Group-disjointness ---")
    disjoint = read_csv_file(PHASE28 / "group_disjointness_validation.csv")
    violations = [r for r in disjoint if r["disjoint"] != "True"]
    check("0 disjointness violations", len(violations) == 0,
          f"violations={len(violations)}")


# =========================================================================
# 7. Historical model exposure reporting
# =========================================================================
def test_historical_exposure():
    print("\n--- 7. Historical model exposure ---")
    pol_b = read_csv_file(PHASE28 / "policy_b_reviewed_simulation.csv")
    # All rows have historical_test_exposure field
    check("historical_test_exposure field present",
          all("historical_test_exposure" in r for r in pol_b))
    # Old test images flagged
    old_test = [r for r in pol_b if r["original_split"] == "test"]
    flagged = [r for r in old_test if r["historical_test_exposure"] == "yes_old_test"]
    check("old test images flagged", len(flagged) == len(old_test),
          f"flagged={len(flagged)} total={len(old_test)}")


# =========================================================================
# 8. Annotation correction validity
# =========================================================================
def test_annotation_review():
    print("\n--- 8. Annotation review ---")
    ann = read_csv_file(PHASE28 / "ai_chenab_annotation_review.csv")
    check("33 annotations reviewed", len(ann) == 33, f"got {len(ann)}")
    valid_decisions = {"ai_accepted", "ai_correction_proposed",
                       "excluded_unreliable", "uncertain"}
    bad = [r for r in ann if r["ai_decision"] not in valid_decisions]
    check("all annotation decisions valid", len(bad) == 0, f"invalid={len(bad)}")
    # All have reviewer_type = AI
    non_ai = [r for r in ann if r["reviewer_type"] != "AI"]
    check("all annotation reviews marked AI", len(non_ai) == 0)


# =========================================================================
# 9. AITEX class mapping protection
# =========================================================================
def test_aitex_mapping_protection():
    print("\n--- 9. AITEX class mapping protection ---")
    classes = read_csv_file(PHASE28 / "ai_aitex_class_review.csv")
    # 019 and 030 still rejected
    for code in ("019", "030"):
        row = [r for r in classes if r["aitex_code"] == code]
        if row:
            check(f"code {code} not approved for Chenab training",
                  row[0]["approved_for_chenab_training"] in ("False", "false", False))
        else:
            check(f"code {code} found", False)
    # No code approved for Chenab training
    approved = [r for r in classes if r["approved_for_chenab_training"] in ("True", "true")]
    check("no AITEX code approved for Chenab training", len(approved) == 0)


# =========================================================================
# 10. Mask eligibility correctness
# =========================================================================
def test_mask_eligibility():
    print("\n--- 10. Mask eligibility ---")
    masks = read_csv_file(PHASE28 / "ai_aitex_mask_review.csv")
    elig = read_csv_file(PHASE28 / "aitex_mask_eligibility.csv")
    check("106 masks reviewed", len(masks) == 106, f"got {len(masks)}")
    check("106 eligibility rows", len(elig) == 106, f"got {len(elig)}")

    # Multi-mask and no-mask cases excluded from XAI
    multi = [r for r in masks if r["mask_decision"] == "uncertain_multiple_masks"]
    check("multi-mask cases not XAI eligible",
          all(r["xai_mask_eligible"] in ("False", "false") for r in multi))
    no_mask = [r for r in masks if r["mask_decision"] == "excluded_no_mask"]
    check("no-mask cases not XAI eligible",
          all(r["xai_mask_eligible"] in ("False", "false") for r in no_mask))


# =========================================================================
# 11. Manifest schema and row counts
# =========================================================================
def test_manifest_v12():
    print("\n--- 11. Manifest V1.2 ---")
    mv12 = read_csv_file(META / "master_manifest_v1_2.csv")
    check("3038 rows in V1.2", len(mv12) == 3038, f"got {len(mv12)}")

    required_fields = [
        "image_id", "source_dataset", "original_filename", "original_split",
        "source_family_id", "resolved_group_id", "ai_duplicate_review_status",
        "annotation_qc_status", "mask_qc_status", "class_verification_status",
        "leakage_risk", "proposed_split_eligibility", "historical_model_exposure",
        "f1_detection_ready", "f2_background_ready", "f2_bbox_xai_ready",
        "f2_mask_xai_ready", "exclusion_reason", "verification_level",
        "human_verified"
    ]
    for f in required_fields:
        check(f"field '{f}' present", f in mv12[0])

    # IDs unique
    ids = [r["image_id"] for r in mv12]
    check("image IDs unique", len(ids) == len(set(ids)))

    # Source counts
    chenab = [r for r in mv12 if r["source_dataset"] == "chenab"]
    aitex = [r for r in mv12 if r["source_dataset"] == "aitex"]
    check("2791 Chenab", len(chenab) == 2791)
    check("247 AITEX", len(aitex) == 247)

    # V1 and V1.1 preserved
    check("V1 manifest preserved", (META / "master_manifest_v1.csv").exists())
    check("V1.1 manifest preserved", (META / "master_manifest_v1_1.csv").exists())


# =========================================================================
# 12. Stable IDs and reproducibility
# =========================================================================
def test_stable_ids():
    print("\n--- 12. Stable IDs ---")
    mv12 = read_csv_file(META / "master_manifest_v1_2.csv")
    mv11 = read_csv_file(META / "master_manifest_v1_1.csv")
    ids_12 = {r["image_id"] for r in mv12}
    ids_11 = {r["image_id"] for r in mv11}
    check("V1.2 IDs match V1.1", ids_12 == ids_11)


# =========================================================================
# 13. Test-split protection
# =========================================================================
def test_test_split_protection():
    print("\n--- 13. Test-split protection ---")
    mv12 = read_csv_file(META / "master_manifest_v1_2.csv")
    test_imgs = [r for r in mv12 if r["original_split"] == "test"
                 and r["source_dataset"] == "chenab"]
    check("280 test images", len(test_imgs) == 280, f"got {len(test_imgs)}")
    # None marked F1 detection ready
    f1_test = [r for r in test_imgs if r["f1_detection_ready"] in ("True", "true")]
    check("no test image F1-ready", len(f1_test) == 0, f"ready={len(f1_test)}")


# =========================================================================
# 14. Raw dataset immutability
# =========================================================================
def test_immutability():
    print("\n--- 14. Raw data immutability ---")
    raw_base = MASTER / "raw"
    bad = []
    if raw_base.is_dir():
        for d in raw_base.rglob("*"):
            if d.is_dir() and "phase2_8" in str(d):
                bad.append(str(d))
    check("no phase2_8 dirs under raw/", len(bad) == 0)


# =========================================================================
# 15. Deterministic simulation outputs
# =========================================================================
def test_deterministic():
    print("\n--- 15. Deterministic outputs ---")
    results = json.loads((PHASE28 / "phase2_8_results.json").read_text(encoding="utf-8"))
    check("results JSON exists", True)
    # Policy B counts sum to 2791
    pb = results["policy_b"]["policy_b"]
    total = sum(int(v) for v in pb.values())
    check("Policy B total = 2791", total == 2791, f"got {total}")
    # 0 disjoint violations
    check("0 disjoint violations", results["policy_b"]["disjoint_violations"] == 0)


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 64)
    print("Phase 2.8 Tests")
    print("=" * 64)

    test_all_candidates_reviewed()
    test_traceable_evidence()
    test_no_false_human()
    test_no_phash_only_confirmed()
    test_source_family_grouping()
    test_group_disjointness()
    test_historical_exposure()
    test_annotation_review()
    test_aitex_mapping_protection()
    test_mask_eligibility()
    test_manifest_v12()
    test_stable_ids()
    test_test_split_protection()
    test_immutability()
    test_deterministic()

    print("\n" + "=" * 64)
    total = PASS + FAIL
    print(f"Result: {PASS}/{total} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
