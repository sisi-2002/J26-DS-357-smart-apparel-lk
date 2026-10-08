"""
Phase 2.6 targeted tests.
Run: python -I test_phase2_6.py
"""

import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
PHASE25 = MASTER / "reports" / "phase2_5"
PHASE26 = MASTER / "reports" / "phase2_6"
META = MASTER / "metadata"
RAW_CHENAB = MASTER / "raw" / "chenab" / "fabricDD.v2i.yolov8"

REQUIRED_P26_FILES = [
    PHASE26 / "corrected_duplicate_pairs.csv",
    PHASE26 / "source_family_candidates.csv",
    PHASE26 / "chenab_image_roots.csv",
    PHASE26 / "confirmed_duplicate_groups.csv",
    PHASE26 / "cross_split_group_conflicts.csv",
    PHASE26 / "ambiguous_duplicate_edges.csv",
    PHASE26 / "visual_review_records.csv",
    PHASE26 / "phase3_grouping_proposal.csv",
    PHASE26 / "phase3_excluded_pending_review.csv",
    PHASE26 / "phase3_leakage_readiness.md",
    PHASE26 / "phase2_6_results.json",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------
PASS = FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f" | {detail}" if detail else ""))
        FAIL += 1


# ===========================================================================
# Test Group 1 – Output files exist
# ===========================================================================
def test_output_files_exist():
    print("\n--- 1. Output files exist ---")
    for path in REQUIRED_P26_FILES:
        check(f"exists: {path.name}", path.exists())
        if path.exists() and path.suffix == ".csv":
            rows = read_csv(path)
            check(f"non-empty: {path.name}", len(rows) > 0,
                  f"got {len(rows)} rows")


# ===========================================================================
# Test Group 2 – Filename root extraction
# ===========================================================================
RF_PATTERN = re.compile(
    r'^(augmented_)?(.+?)_jpg\.rf\.[0-9a-f]{32}$',
    re.IGNORECASE
)


def extract_root(stem: str) -> tuple:
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


def test_filename_parsing():
    print("\n--- 2. Filename root extraction ---")

    # Standard Roboflow pattern
    stem1 = "00000001_jpg.rf.abcdef1234567890abcdef1234567890"
    root, status = extract_root(stem1)
    check("8-digit numeric root", root == "00000001" and status == "rf_pattern")

    # Augmented prefix
    stem2 = "augmented_00000001_jpg.rf.abcdef1234567890abcdef1234567890"
    root, status = extract_root(stem2)
    check("augmented_ prefix stripped", root == "00000001" and status == "rf_pattern")

    # Augmented prefix does NOT appear in root
    check("augmented prefix not in root", not root.lower().startswith("augmented_"))

    # Descriptive name
    stem3 = "fabric_sample_001_jpg.rf.deadbeefdeadbeefdeadbeefdeadbeef"
    root, status = extract_root(stem3)
    check("descriptive root", root == "fabric_sample_001")

    # Collision: two stems with same root -> same family
    stem_a = "mystem_jpg.rf." + "a" * 32
    stem_b = "mystem_jpg.rf." + "b" * 32
    root_a, _ = extract_root(stem_a)
    root_b, _ = extract_root(stem_b)
    check("same root -> same family", root_a == root_b)

    # Different roots stay separate
    stem_c = "other_jpg.rf." + "c" * 32
    root_c, _ = extract_root(stem_c)
    check("different roots stay separate", root_a != root_c)

    # All actual Chenab images parse correctly
    rows = read_csv(PHASE26 / "chenab_image_roots.csv")
    unparseable = [r for r in rows if r["parse_status"] == "unparseable"]
    check("no unparseable Chenab filenames", len(unparseable) == 0,
          f"unparseable={len(unparseable)}")


# ===========================================================================
# Test Group 3 – SHA-256 vs pHash terminology
# ===========================================================================
def test_sha256_vs_phash_terminology():
    print("\n--- 3. SHA-256 vs pHash terminology ---")

    corrected = read_csv(PHASE26 / "corrected_duplicate_pairs.csv")

    # No row should use the old "exact_duplicate" classification_v26
    bad_v26 = [r for r in corrected if r["classification_v26"] == "exact_duplicate"]
    check("no 'exact_duplicate' in classification_v26", len(bad_v26) == 0,
          f"found {len(bad_v26)} rows")

    # pHash-0 pairs classified as phash_identical
    phash0 = [r for r in corrected if int(r["hamming_distance"]) == 0]
    not_phash_identical = [r for r in phash0 if r["classification_v26"] != "phash_identical"]
    check("all dist=0 rows classified phash_identical", len(not_phash_identical) == 0,
          f"miscategorised={len(not_phash_identical)}")

    # sha256_verified column present and never claims verified
    all_sha_field = all("sha256_verified" in r for r in corrected)
    check("sha256_verified column present", all_sha_field)
    claimed_verified = [r for r in corrected if r.get("sha256_verified") == "yes"]
    check("sha256_verified never set to yes", len(claimed_verified) == 0,
          f"falsely verified={len(claimed_verified)}")

    # evidence_type uses phash_ prefix, not sha256_
    bad_evidence = [r for r in corrected if r.get("evidence_type", "").startswith("sha256")]
    check("evidence_type uses phash_ not sha256_", len(bad_evidence) == 0,
          f"bad={len(bad_evidence)}")

    # SHA-256 exact duplicates remain 0 (from Phase 2.5 baseline)
    old_pairs = read_csv(PHASE25 / "near_duplicate_candidates.csv")
    old_phash0 = [r for r in old_pairs if int(r["hamming_distance"]) == 0]
    check("Phase 2.5 had phash-0 pairs (not SHA-256 exact)", len(old_phash0) > 0,
          f"old phash0={len(old_phash0)}")
    # Phase 2.6 resets sha256_verified to no for all of these
    p26_phash0 = [r for r in corrected if int(r["hamming_distance"]) == 0]
    check("Phase 2.6 phash0 row count matches Phase 2.5",
          len(p26_phash0) == len(old_phash0),
          f"v26={len(p26_phash0)} v25={len(old_phash0)}")


# ===========================================================================
# Test Group 4 – Same vs cross-split classification
# ===========================================================================
def test_same_vs_cross_split():
    print("\n--- 4. Same vs cross-split classification ---")

    corrected = read_csv(PHASE26 / "corrected_duplicate_pairs.csv")
    cross = [r for r in corrected if r["cross_split"] == "True"]
    same = [r for r in corrected if r["cross_split"] == "False"]

    check("cross-split pairs exist", len(cross) > 0, f"cross={len(cross)}")
    check("same-split pairs exist", len(same) > 0, f"same={len(same)}")
    check("total = cross + same", len(cross) + len(same) == len(corrected),
          f"cross={len(cross)} same={len(same)} total={len(corrected)}")

    # Cross-split: split1 != split2
    bad_cross = [r for r in cross if r["split1"] == r["split2"]]
    check("cross-split: split1 != split2", len(bad_cross) == 0,
          f"bad={len(bad_cross)}")

    # Same-split: split1 == split2
    bad_same = [r for r in same if r["split1"] != r["split2"]]
    check("same-split: split1 == split2", len(bad_same) == 0,
          f"bad={len(bad_same)}")

    # Expected counts from investigation
    cs_phash0 = [r for r in cross if int(r["hamming_distance"]) == 0]
    check("65 cross-split phash-0 pairs", len(cs_phash0) == 65,
          f"got {len(cs_phash0)}")
    cs_near = [r for r in cross if 0 < int(r["hamming_distance"]) <= 5]
    check("70 cross-split near-dup pairs (d=1-5)", len(cs_near) == 70,
          f"got {len(cs_near)}")


# ===========================================================================
# Test Group 5 – Source family consistency
# ===========================================================================
def test_source_family_consistency():
    print("\n--- 5. Source family consistency ---")

    families = read_csv(PHASE26 / "source_family_candidates.csv")
    img_roots = read_csv(PHASE26 / "chenab_image_roots.csv")

    # Every root in img_roots maps to a family
    img_root_set = {r["source_root"] for r in img_roots}
    fam_root_set = {r["source_root"] for r in families}
    check("all image roots have a family entry", img_root_set == fam_root_set,
          f"img_only={len(img_root_set - fam_root_set)} fam_only={len(fam_root_set - img_root_set)}")

    # Unique root count = 891
    check("891 unique source roots", len(families) == 891, f"got {len(families)}")

    # 492 cross-split families
    cross = [f for f in families if f["cross_split"] == "True"]
    check("492 cross-split families", len(cross) == 492, f"got {len(cross)}")

    # 218 touching test split
    test_touch = [f for f in families if f["touches_test_split"] == "True"]
    check("218 families touching test split", len(test_touch) == 218,
          f"got {len(test_touch)}")

    # Augmented count <= image count for each family
    bad = [f for f in families if int(f["augmented_count"]) > int(f["image_count"])]
    check("augmented_count <= image_count for all families", len(bad) == 0,
          f"bad={len(bad)}")

    # n_splits = number of unique splits in splits_present
    inconsistent = []
    for f in families:
        parts = [p for p in f["splits_present"].split("|") if p]
        if str(len(parts)) != str(f["n_splits"]):
            inconsistent.append(f)
    check("n_splits matches splits_present length", len(inconsistent) == 0,
          f"inconsistent={len(inconsistent)}")


# ===========================================================================
# Test Group 6 – Confirmed groups consistency
# ===========================================================================
def test_confirmed_groups():
    print("\n--- 6. Confirmed duplicate groups ---")

    groups = read_csv(PHASE26 / "confirmed_duplicate_groups.csv")
    conflicts = read_csv(PHASE26 / "cross_split_group_conflicts.csv")

    # 492 groups (one per cross-split family)
    unique_groups = {r["group_id"] for r in groups}
    check("492 unique confirmed groups", len(unique_groups) == 492,
          f"got {len(unique_groups)}")

    # 492 conflict rows
    check("492 conflict rows", len(conflicts) == 492, f"got {len(conflicts)}")

    # Every group in confirmed_groups has a conflict row
    conflict_group_ids = {r["group_id"] for r in conflicts}
    check("every group has conflict row", unique_groups == conflict_group_ids,
          f"missing={len(unique_groups - conflict_group_ids)}")

    # All images in groups have evidence_type = filename_root_ancestry
    bad = [r for r in groups if r["evidence_type"] != "filename_root_ancestry"]
    check("all groups use filename_root_ancestry evidence", len(bad) == 0,
          f"bad={len(bad)}")

    # Test split conflicts = 218
    test_conflicts = [r for r in conflicts
                      if r["conflict_type"] == "source_family_spans_test_split"]
    check("218 test-split conflicts", len(test_conflicts) == 218,
          f"got {len(test_conflicts)}")


# ===========================================================================
# Test Group 7 – Ambiguous edges: no auto-merge
# ===========================================================================
def test_no_auto_merge():
    print("\n--- 7. Ambiguous edges: no auto-merge ---")

    ambig = read_csv(PHASE26 / "ambiguous_duplicate_edges.csv")

    # No auto_merged = yes
    merged = [r for r in ambig if r["auto_merged"] == "yes"]
    check("no ambiguous edges auto-merged", len(merged) == 0,
          f"auto-merged={len(merged)}")

    # All have human_decision = pending_visual_review
    not_pending = [r for r in ambig if r["human_decision"] != "pending_visual_review"]
    check("all ambiguous edges pending visual review", len(not_pending) == 0,
          f"not_pending={len(not_pending)}")

    # All edges have different source roots (otherwise they'd be in confirmed groups)
    same_root = [r for r in ambig if r["root1"] == r["root2"]]
    check("ambiguous edges have different roots", len(same_root) == 0,
          f"same_root={len(same_root)}")

    # All are cross-split
    not_cross = [r for r in ambig if r["split1"] == r["split2"]]
    check("all ambiguous edges are cross-split", len(not_cross) == 0,
          f"same_split={len(not_cross)}")


# ===========================================================================
# Test Group 8 – Locked test split protection
# ===========================================================================
def test_locked_test_split():
    print("\n--- 8. Locked test split protection ---")

    proposal = read_csv(PHASE26 / "phase3_grouping_proposal.csv")

    # Test split images always have training_block = True
    test_imgs = [r for r in proposal if r["original_split"] == "test" and r["source"] == "chenab"]
    check("test split images exist in proposal", len(test_imgs) > 0,
          f"test={len(test_imgs)}")
    test_not_blocked = [r for r in test_imgs if r["training_block"] != "True"]
    check("all test images have training_block=True", len(test_not_blocked) == 0,
          f"unblocked={len(test_not_blocked)}")

    # Excluded CSV never contains test-split rows
    excluded = read_csv(PHASE26 / "phase3_excluded_pending_review.csv")
    test_in_excluded = [r for r in excluded if r["original_split"] == "test"]
    check("test split absent from excluded CSV", len(test_in_excluded) == 0,
          f"test_in_excluded={len(test_in_excluded)}")

    # Locked test split count = 280
    locked = [r for r in proposal
              if r["original_split"] == "test" and r["source"] == "chenab"
              and "locked" in r.get("block_reason", "")]
    check("280 locked test images", len(locked) == 280, f"got {len(locked)}")


# ===========================================================================
# Test Group 9 – Phase 3 proposal integrity
# ===========================================================================
def test_phase3_proposal():
    print("\n--- 9. Phase 3 proposal integrity ---")

    proposal = read_csv(PHASE26 / "phase3_grouping_proposal.csv")
    manifest = read_csv(META / "master_manifest_v1_1.csv")

    # All manifest rows present in proposal
    check("proposal row count = manifest row count", len(proposal) == len(manifest),
          f"proposal={len(proposal)} manifest={len(manifest)}")

    # No image_id appears twice
    ids = [r["image_id"] for r in proposal]
    check("no duplicate image_ids", len(ids) == len(set(ids)),
          f"dups={len(ids)-len(set(ids))}")

    # Chenab total = 2791
    chenab = [r for r in proposal if r["source"] == "chenab"]
    check("2791 Chenab rows", len(chenab) == 2791, f"got {len(chenab)}")

    # Approved + blocked + locked = total Chenab
    approved = [r for r in chenab if r["training_block"] == "False"]
    blocked = [r for r in chenab if r["training_block"] == "True"
               and r["original_split"] != "test"]
    locked = [r for r in chenab if r["original_split"] == "test"]
    check("approved+blocked+locked = 2791",
          len(approved) + len(blocked) + len(locked) == 2791,
          f"sum={len(approved)+len(blocked)+len(locked)}")

    # AITEX rows all have empty source_root
    aitex = [r for r in proposal if r["source"] == "aitex"]
    bad_aitex = [r for r in aitex if r["source_root"] != ""]
    check("AITEX rows have empty source_root", len(bad_aitex) == 0,
          f"bad={len(bad_aitex)}")


# ===========================================================================
# Test Group 10 – Deterministic output
# ===========================================================================
def test_deterministic():
    print("\n--- 10. Deterministic output ---")

    # Read key counts from results JSON
    results_path = PHASE26 / "phase2_6_results.json"
    if not results_path.exists():
        check("phase2_6_results.json exists", False)
        return

    results = json.loads(results_path.read_text(encoding="utf-8"))

    # Task A counts stable
    check("Task A total pairs = 3342", results["dup"]["total"] == 3342,
          f"got {results['dup']['total']}")
    check("Task A cs_phash0 = 65", results["dup"]["cs_phash0"] == 65)
    check("Task A cs_near = 70", results["dup"]["cs_near"] == 70)

    # Task B counts stable
    fam = results.get("family", {})
    check("Task B unique_roots = 891", fam.get("unique_roots") == 891,
          f"got {fam.get('unique_roots')}")
    check("Task B cross_split_roots = 492", fam.get("cross_split_roots") == 492,
          f"got {fam.get('cross_split_roots')}")
    check("Task B test_touching_roots = 218", fam.get("test_touching_roots") == 218,
          f"got {fam.get('test_touching_roots')}")


# ===========================================================================
# Test Group 11 – Raw data immutability
# ===========================================================================
def test_immutability():
    print("\n--- 11. Raw data immutability ---")

    src_meta = MASTER / "reports" / "phase1_audit" / "source_metadata.csv"
    if not src_meta.exists():
        check("source_metadata.csv exists", False)
        return

    import hashlib
    rows = read_csv(src_meta)
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


# ===========================================================================
# Test Group 12 – Visual review records schema
# ===========================================================================
def test_visual_review_schema():
    print("\n--- 12. Visual review records schema ---")

    records = read_csv(PHASE26 / "visual_review_records.csv")
    required_fields = ["review_id", "priority", "pair_type", "file1", "split1",
                       "file2", "split2", "hamming_distance", "evidence",
                       "proposed_action", "human_decision"]

    if not records:
        check("visual_review_records non-empty", False)
        return

    first = records[0]
    for field in required_fields:
        check(f"field '{field}' present", field in first)

    # 135 review records (65 + 70)
    check("135 review records", len(records) == 135, f"got {len(records)}")

    # All have human_decision = pending_visual_review
    not_pending = [r for r in records if r["human_decision"] != "pending_visual_review"]
    check("all reviews pending_visual_review", len(not_pending) == 0,
          f"not_pending={len(not_pending)}")

    # Priority 1 = 65 (phash-0)
    p1 = [r for r in records if r["priority"] == "1"]
    check("65 priority-1 reviews", len(p1) == 65, f"got {len(p1)}")

    # Priority 2 = 70 (near-dup)
    p2 = [r for r in records if r["priority"] == "2"]
    check("70 priority-2 reviews", len(p2) == 70, f"got {len(p2)}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 64)
    print("Phase 2.6 Tests")
    print("=" * 64)

    test_output_files_exist()
    test_filename_parsing()
    test_sha256_vs_phash_terminology()
    test_same_vs_cross_split()
    test_source_family_consistency()
    test_confirmed_groups()
    test_no_auto_merge()
    test_locked_test_split()
    test_phase3_proposal()
    test_deterministic()
    test_immutability()
    test_visual_review_schema()

    print("\n" + "=" * 64)
    total = PASS + FAIL
    print(f"Result: {PASS}/{total} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
