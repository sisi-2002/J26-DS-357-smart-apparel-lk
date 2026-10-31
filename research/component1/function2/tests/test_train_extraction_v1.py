"""F2.1B2 Task 9 -- tests for dataset/train_extraction_v1.py.

Covers:
  - split-safety regression AGAINST THE REAL PRODUCED ARTIFACT (not just a
    mock): every image_id in the actual
    results/calibration/train_features_v1_full.csv is re-verified to be
    split=='train' in the authoritative manifest, and every relative_path
    is confirmed to point under train/images/ -- never valid/ or test/.
  - a smaller, isolated (tmp_path) run of the same pipeline with --limit,
    verifying split-safety end-to-end on a small controlled sample and
    reproducibility (two runs, same limit, byte-identical CSV).
  - family_size column correctness against the authoritative train manifest.
  - the eligibility columns stored in the full CSV are internally
    consistent with complexity.eligibility.determine_eligibility given the
    row's own stored background/feature fields (integration check between
    Task 1's policy and Task 2's extraction script).
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pytest

from research.component1.function2.complexity.eligibility import determine_eligibility
from research.component1.function2.config.paths import REPO_ROOT
from research.component1.function2.dataset import train_extraction_v1
from research.component1.function2.dataset.manifest_loader import (
    build_split_lookup,
    load_train_manifest,
)

FULL_CSV_PATH = (
    REPO_ROOT / "research" / "component1" / "function2" / "results"
    / "calibration" / "train_features_v1_full.csv"
)
FULL_METADATA_PATH = (
    REPO_ROOT / "research" / "component1" / "function2" / "results"
    / "calibration" / "train_features_v1_full_metadata.json"
)


def _to_float(v):
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    import math
    return f if math.isfinite(f) else None


@pytest.fixture(scope="module")
def split_lookup():
    return build_split_lookup()


# ---------------------------------------------------------------------------
# Regression test against the REAL produced Task 2 artifact (not a mock).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FULL_CSV_PATH.is_file(), reason="Task 2 full extraction has not been run yet")
def test_real_full_extraction_touched_only_train_split(split_lookup):
    with open(FULL_CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 1000, "expected near-full TRAIN coverage, got far fewer rows than expected"

    for row in rows:
        record = split_lookup.get(row["image_id"])
        assert record is not None, f"image_id {row['image_id']} not found in authoritative split manifest"
        assert record.split == "train"
        assert record.split != "valid"
        assert record.split != "test"
        rel_path_norm = row["relative_path"].replace("\\", "/")
        assert "/train/images/" in rel_path_norm
        assert "/valid/images/" not in rel_path_norm
        assert "/test/images/" not in rel_path_norm


@pytest.mark.skipif(not FULL_METADATA_PATH.is_file(), reason="Task 2 full extraction has not been run yet")
def test_real_full_extraction_metadata_confirms_train_only():
    with open(FULL_METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["split_touched"] == ["train"]
    assert meta["all_rows_confirmed_train_split"] is True


@pytest.mark.skipif(not FULL_CSV_PATH.is_file(), reason="Task 2 full extraction has not been run yet")
def test_real_full_extraction_family_size_matches_manifest():
    train_by_id = {r.image_id: r for r in load_train_manifest()}
    family_counts: dict[str, int] = {}
    for r in train_by_id.values():
        family_counts[r.group_id] = family_counts.get(r.group_id, 0) + 1

    with open(FULL_CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    checked = 0
    for row in rows[:200]:
        manifest_rec = train_by_id[row["image_id"]]
        assert row["source_family_group_id"] == manifest_rec.group_id
        assert int(row["family_size"]) == family_counts[manifest_rec.group_id]
        checked += 1
    assert checked == 200


@pytest.mark.skipif(not FULL_CSV_PATH.is_file(), reason="Task 2 full extraction has not been run yet")
def test_real_full_extraction_eligibility_matches_policy_recomputation():
    """Cross-check Task 1 (eligibility.py) against Task 2's stored decision:
    recompute eligibility from each row's own stored fields and confirm it
    matches what was written, for a deterministic sample of rows covering
    both eligible and ineligible outcomes."""
    with open(FULL_CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    eligible_sample = [r for r in rows if r["eligible"] == "True"][:100]
    ineligible_sample = [r for r in rows if r["eligible"] == "False"][:100]

    for row in eligible_sample + ineligible_sample:
        recomputed = determine_eligibility(
            split_confirmed_train=True,
            background_status=row["background_status"],
            luminance_variation=_to_float(row["luminance_variation"]),
            glcm_entropy=_to_float(row["glcm_entropy"]),
            glcm_status=row["glcm_status"],
            glcm_valid_pair_count=int(row["glcm_valid_pair_count"]) if row["glcm_valid_pair_count"] not in ("", None) else 0,
            colour_variation=_to_float(row["colour_variation"]),
        )
        assert recomputed.eligible == (row["eligible"] == "True"), row["image_id"]
        if not recomputed.eligible:
            assert recomputed.rejection_reason == row["rejection_reason"], row["image_id"]


@pytest.mark.skipif(not FULL_CSV_PATH.is_file(), reason="Task 2 full extraction has not been run yet")
def test_real_full_extraction_confirmed_negative_and_box_excluded_kept_separate():
    with open(FULL_CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sample_types = {r["sample_type"] for r in rows}
    assert "confirmed_negative" in sample_types
    assert "box_excluded_defective" in sample_types
    # No row should be ambiguously double-classified: a confirmed_negative
    # row must have is_negative_manifest == True.
    for r in rows:
        if r["sample_type"] == "confirmed_negative":
            assert r["is_negative_manifest"] == "True"


# ---------------------------------------------------------------------------
# Isolated small-scale run (tmp_path) -- fast, controlled, deterministic.
# ---------------------------------------------------------------------------

def test_small_limited_run_touches_only_train_split(tmp_path, split_lookup):
    meta = train_extraction_v1.run(limit=10, output_dir=tmp_path)
    assert meta["split_touched"] == ["train"]
    assert meta["all_rows_confirmed_train_split"] is True
    assert meta["sample_size_produced"] == 10

    with open(tmp_path / "train_features_v1_full.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    for row in rows:
        record = split_lookup[row["image_id"]]
        assert record.split == "train"


def test_small_limited_run_reproducible(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    train_extraction_v1.run(limit=8, output_dir=out1)
    train_extraction_v1.run(limit=8, output_dir=out2)
    csv1 = (out1 / "train_features_v1_full.csv").read_text(encoding="utf-8")
    csv2 = (out2 / "train_features_v1_full.csv").read_text(encoding="utf-8")
    assert csv1 == csv2


def test_small_limited_run_has_eligible_and_ineligible_rows(tmp_path):
    """Sanity check against degeneracy: a 200-image deterministic sample
    should contain both eligible and ineligible rows (the real dataset is
    known, from the full run, to have ~8% ineligible overall)."""
    meta = train_extraction_v1.run(limit=200, output_dir=tmp_path)
    assert meta["eligibility_counts"]["eligible"] > 0
    assert meta["eligibility_counts"]["ineligible"] > 0


def test_small_limited_run_sample_type_schema_present(tmp_path):
    meta = train_extraction_v1.run(limit=50, output_dir=tmp_path)
    with open(tmp_path / "train_features_v1_full.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert row["sample_type"] in (
            "confirmed_negative", "box_excluded_defective",
            "insufficient_background_defective", "ambiguous_clean_no_boxes",
            "unresolved_annotation", "unreadable_image",
        ) or row["sample_type"].startswith("other_")
        # defect_class_ids/names must be valid JSON (never a bare repr string).
        ast.literal_eval(row["defect_class_ids"])  # JSON-compatible list literal
        json.loads(row["defect_class_ids"])
        json.loads(row["defect_class_names"])
        json.loads(row["failed_checks"])
