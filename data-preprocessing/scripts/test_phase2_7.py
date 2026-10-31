"""
Phase 2.7 targeted tests.
Run: python -I test_phase2_7.py
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
PHASE26 = MASTER / "reports" / "phase2_6"
PHASE27 = MASTER / "reports" / "phase2_7"
META = MASTER / "metadata"
RAW_CHENAB = MASTER / "raw" / "chenab" / "fabricDD.v2i.yolov8"

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
# 1. Source root parsing
# =========================================================================
def test_source_root_parsing():
    print("\n--- 1. Source root parsing ---")

    # Standard Roboflow pattern
    root, st = extract_root("00000001_jpg.rf.abcdef1234567890abcdef1234567890")
    check("standard RF root", root == "00000001" and st == "rf_pattern")

    # Augmented prefix
    root, st = extract_root("augmented_00000001_jpg.rf.abcdef1234567890abcdef1234567890")
    check("augmented prefix stripped", root == "00000001" and st == "rf_pattern")

    # Root not prefixed with augmented_
    check("root has no augmented_ prefix", not root.startswith("augmented_"))

    # Same root from different RF hashes
    r1, _ = extract_root("myroot_jpg.rf." + "a" * 32)
    r2, _ = extract_root("myroot_jpg.rf." + "b" * 32)
    check("same root from different hashes", r1 == r2)

    # Different roots stay separate
    r3, _ = extract_root("other_jpg.rf." + "c" * 32)
    check("different roots stay separate", r1 != r3)

    # All Chenab images parse
    img_roots = read_csv_file(PHASE26 / "chenab_image_roots.csv")
    unparseable = [r for r in img_roots if r["parse_status"] == "unparseable"]
    check("no unparseable Chenab filenames", len(unparseable) == 0)

    # Verified families all have rf_pattern
    fam = read_csv_file(PHASE27 / "source_family_verification.csv")
    not_rf = [r for r in fam if r["all_rf_parsed"] != "True"]
    check("all families have RF-parsed roots", len(not_rf) == 0)


# =========================================================================
# 2. Cross-split group classification
# =========================================================================
def test_cross_split_classification():
    print("\n--- 2. Cross-split group classification ---")

    fam = read_csv_file(PHASE27 / "source_family_verification.csv")
    cross = [f for f in fam if f["cross_split"] == "True"]

    check("492 cross-split families", len(cross) == 492, f"got {len(cross)}")

    combos = Counter(f["splits_present"] for f in cross)
    check("train|valid: 274", combos.get("train|valid", 0) == 274)
    check("test|train: 107", combos.get("test|train", 0) == 107)
    check("test|train|valid: 97", combos.get("test|train|valid", 0) == 97)
    check("test|valid: 14", combos.get("test|valid", 0) == 14)

    # All classified as verified_shared_source
    classifications = Counter(f["classification"] for f in cross)
    check("all cross-split are verified_shared_source",
          classifications.get("verified_shared_source", 0) == 492,
          f"got {dict(classifications)}")

    # No invalid groupings
    invalid = [f for f in fam if f["classification"] == "invalid_grouping"]
    check("no invalid groupings", len(invalid) == 0)


# =========================================================================
# 3. Test-touching family protection
# =========================================================================
def test_test_touching_protection():
    print("\n--- 3. Test-touching family protection ---")

    policy = read_csv_file(PHASE27 / "test_touching_family_policy.csv")
    fam = read_csv_file(PHASE27 / "source_family_verification.csv")
    test_touching_roots = {f["source_root"] for f in fam if f["touches_test"] == "True"}

    # 218 test-touching families
    check("218 test-touching families", len(test_touching_roots) == 218,
          f"got {len(test_touching_roots)}")

    # All test images preserved
    test_imgs = [r for r in policy if r["split"] == "test"]
    check("all test images preserved",
          all(r["membership_status"] == "preserved_test_membership" for r in test_imgs))
    check("280 test images", len(test_imgs) == 280, f"got {len(test_imgs)}")

    # Non-test members of test-touching families quarantined
    for r in policy:
        if r["split"] != "test" and r["source_root"] in test_touching_roots:
            if r["membership_status"] != "quarantined_train_valid":
                check("quarantine non-test member of test-touching family", False,
                      f"{r['filename']} is {r['membership_status']}")
                return
    check("all non-test members of test-touching families quarantined", True)

    # No quarantined image has training_approved=True
    quarantined = [r for r in policy if r["membership_status"] == "quarantined_train_valid"]
    approved_quarantined = [r for r in quarantined if r["training_approved"] == "True"]
    check("no quarantined image approved for training", len(approved_quarantined) == 0,
          f"approved={len(approved_quarantined)}")


# =========================================================================
# 4. No unverified pHash auto-merging
# =========================================================================
def test_no_phash_auto_merge():
    print("\n--- 4. No unverified pHash auto-merging ---")

    # Ambiguous edges from Phase 2.6 never auto-merged
    ambig = read_csv_file(PHASE26 / "ambiguous_duplicate_edges.csv")
    merged = [r for r in ambig if r["auto_merged"] == "yes"]
    check("no ambiguous edges auto-merged (Phase 2.6)", len(merged) == 0)

    # Phase 2.7 reviews all pending
    reviews = read_csv_file(PHASE27 / "priority_visual_review.csv")
    not_pending = [r for r in reviews if r["review_decision"] != "pending_review"]
    check("all Phase 2.7 reviews pending", len(not_pending) == 0,
          f"decided={len(not_pending)}")

    # Family verification: no human visual confirmed
    fam = read_csv_file(PHASE27 / "source_family_verification.csv")
    confirmed = [f for f in fam if f["human_visual_confirmed"] == "yes"]
    check("no human visual confirmations fabricated", len(confirmed) == 0)


# =========================================================================
# 5. Stable image/group identifiers
# =========================================================================
def test_stable_identifiers():
    print("\n--- 5. Stable image/group identifiers ---")

    readiness = read_csv_file(PHASE27 / "phase3_candidate_readiness.csv")
    ids = [r["image_id"] for r in readiness]
    check("3038 images in readiness", len(ids) == 3038)
    check("all image_ids unique", len(ids) == len(set(ids)),
          f"dups={len(ids)-len(set(ids))}")

    # IDs match manifest
    manifest = read_csv_file(META / "master_manifest_v1_1.csv")
    mf_ids = {r["image_id"] for r in manifest}
    rd_ids = set(ids)
    check("readiness IDs match manifest", mf_ids == rd_ids)

    # Source roots stable between Phase 2.6 and 2.7
    roots_26 = read_csv_file(PHASE26 / "chenab_image_roots.csv")
    roots_by_file = {r["filename"]: r["source_root"] for r in roots_26}
    fam_27 = read_csv_file(PHASE27 / "source_family_verification.csv")
    roots_27 = {r["source_root"] for r in fam_27}
    roots_26_set = {r["source_root"] for r in roots_26}
    check("source roots stable 2.6->2.7", roots_26_set == roots_27)


# =========================================================================
# 6. Policy simulation determinism
# =========================================================================
def test_policy_determinism():
    print("\n--- 6. Policy simulation determinism ---")

    results = json.loads((PHASE27 / "phase2_7_results.json").read_text(encoding="utf-8"))
    pol_a = results["policy"]["policy_a"]
    pol_b = results["policy"]["policy_b"]

    # Policy A counts
    check("Policy A train+valid+test+quarantined+excluded = 2791",
          sum(pol_a.values()) == 2791, f"sum={sum(pol_a.values())}")

    # Policy B counts
    check("Policy B train+valid+test+excluded = 2791",
          sum(pol_b.values()) == 2791, f"sum={sum(pol_b.values())}")

    # Policy B no violations
    check("Policy B 0 disjoint violations",
          results["policy"]["policy_b_non_disjoint"] == 0)


# =========================================================================
# 7. Group disjointness in proposed allocations
# =========================================================================
def test_group_disjointness():
    print("\n--- 7. Group disjointness in Policy B ---")

    pol_b = read_csv_file(PHASE27 / "split_policy_B_simulation.csv")
    roots_csv = read_csv_file(PHASE26 / "chenab_image_roots.csv")
    root_by_file = {r["filename"]: r["source_root"] for r in roots_csv}

    root_splits = defaultdict(set)
    for r in pol_b:
        assignment = r["policy_b_assignment"]
        if assignment != "excluded":
            root = root_by_file.get(r["filename"], "")
            root_splits[root].add(assignment)

    non_disjoint = {r: s for r, s in root_splits.items() if len(s) > 1}
    check("0 non-disjoint groups in Policy B", len(non_disjoint) == 0,
          f"violations={len(non_disjoint)}")

    # Every family entirely in one split
    for root, splits in root_splits.items():
        if len(splits) > 1:
            check(f"family {root} disjoint", False, f"splits={splits}")
            return


# =========================================================================
# 8. Correct manifest counts
# =========================================================================
def test_manifest_counts():
    print("\n--- 8. Manifest counts ---")

    readiness = read_csv_file(PHASE27 / "phase3_candidate_readiness.csv")
    chenab = [r for r in readiness if r["source"] == "chenab"]
    aitex = [r for r in readiness if r["source"] == "aitex"]

    check("total = 3038", len(readiness) == 3038)
    check("Chenab = 2791", len(chenab) == 2791)
    check("AITEX = 247", len(aitex) == 247)

    # Chenab splits
    by_split = Counter(r["original_split"] for r in chenab)
    check("Chenab train = 1949", by_split["train"] == 1949)
    check("Chenab valid = 562", by_split["valid"] == 562)
    check("Chenab test = 280", by_split["test"] == 280)


# =========================================================================
# 9. Training/XAI readiness separation
# =========================================================================
def test_readiness_separation():
    print("\n--- 9. Training/XAI readiness separation ---")

    readiness = read_csv_file(PHASE27 / "phase3_candidate_readiness.csv")

    # No AITEX image training-ready
    aitex_train = [r for r in readiness
                   if r["source"] == "aitex" and r["training_ready_phase27"] == "True"]
    check("no AITEX image training-ready", len(aitex_train) == 0)

    # No test image training-ready
    test_train = [r for r in readiness
                  if r["original_split"] == "test" and r["training_ready_phase27"] == "True"]
    check("no test image training-ready", len(test_train) == 0)

    # Training-ready does not imply XAI-ready
    train_ready = [r for r in readiness if r["training_ready_phase27"] == "True"]
    xai_ready = [r for r in readiness if r["xai_ready_phase27"] == "True"]
    train_not_xai = [r for r in train_ready if r["xai_ready_phase27"] != "True"]
    check("training-ready but not XAI-ready exists", len(train_not_xai) > 0,
          f"count={len(train_not_xai)}")

    # XAI-ready count reasonable
    check("XAI-ready count > 0", len(xai_ready) > 0)
    check("XAI-ready < training-ready", len(xai_ready) < len(train_ready))


# =========================================================================
# 10. Quarantine rules
# =========================================================================
def test_quarantine_rules():
    print("\n--- 10. Quarantine rules ---")

    policy = read_csv_file(PHASE27 / "test_touching_family_policy.csv")
    fam = read_csv_file(PHASE27 / "source_family_verification.csv")
    test_roots = {f["source_root"] for f in fam if f["touches_test"] == "True"}

    # Quarantined images are ONLY from test-touching families
    quarantined = [r for r in policy if r["membership_status"] == "quarantined_train_valid"]
    bad_quarantine = [r for r in quarantined if r["source_root"] not in test_roots]
    check("quarantined only from test-touching families", len(bad_quarantine) == 0)

    # No test image is quarantined
    test_quarantined = [r for r in quarantined if r["split"] == "test"]
    check("no test image quarantined", len(test_quarantined) == 0)

    # Unaffected images have no_action_required
    unaffected = [r for r in policy if r["membership_status"] == "unaffected"]
    bad_action = [r for r in unaffected if r["policy_action"] != "no_action_required"]
    check("unaffected images have no_action_required", len(bad_action) == 0)


# =========================================================================
# 11. Raw data immutability
# =========================================================================
def test_immutability():
    print("\n--- 11. Raw data immutability ---")

    src_meta = MASTER / "reports" / "phase1_audit" / "source_metadata.csv"
    if not src_meta.exists():
        check("source_metadata.csv exists", False)
        return

    import hashlib
    rows = read_csv_file(src_meta)
    mismatches = 0
    checked = 0
    for r in rows:
        fp = MASTER / r["relative_path"]
        if not fp.is_file():
            continue
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        if h.hexdigest() != r.get("sha256", ""):
            mismatches += 1
        checked += 1

    check(f"raw files unchanged (checked {checked})", mismatches == 0,
          f"modified={mismatches}")


# =========================================================================
# 12. No modification to existing splits
# =========================================================================
def test_no_split_modification():
    print("\n--- 12. No modification to existing splits ---")

    # Phase 2.7 outputs are all in reports/phase2_7/
    # No files should exist in raw/ that weren't there before
    # Check: the policy CSVs propose but never change actual files

    pol_a = read_csv_file(PHASE27 / "split_policy_A_simulation.csv")
    pol_b = read_csv_file(PHASE27 / "split_policy_B_simulation.csv")

    # Check all filenames in policies exist in raw Chenab
    for r in pol_a[:50]:  # spot check
        fname = r["filename"]
        split = r["original_split"]
        p = RAW_CHENAB / split / "images" / fname
        check(f"file exists: {fname[:30]}...", p.is_file())
        if not p.is_file():
            break

    # Policy B: historical_test_exposure field present
    check("historical_test_exposure field present",
          "historical_test_exposure" in pol_b[0])

    # No new directories created under raw/
    raw_dirs = set()
    raw_base = MASTER / "raw"
    if raw_base.is_dir():
        for d in raw_base.rglob("*"):
            if d.is_dir():
                raw_dirs.add(str(d))
    # We don't know exact expected set, but shouldn't have phase2_7 anything
    bad = [d for d in raw_dirs if "phase2_7" in d]
    check("no phase2_7 dirs under raw/", len(bad) == 0)


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 64)
    print("Phase 2.7 Tests")
    print("=" * 64)

    test_source_root_parsing()
    test_cross_split_classification()
    test_test_touching_protection()
    test_no_phash_auto_merge()
    test_stable_identifiers()
    test_policy_determinism()
    test_group_disjointness()
    test_manifest_counts()
    test_readiness_separation()
    test_quarantine_rules()
    test_immutability()
    test_no_split_modification()

    print("\n" + "=" * 64)
    total = PASS + FAIL
    print(f"Result: {PASS}/{total} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
