"""F2.1A Task 5/7 — provenance, split-safety, and reproducibility tests for
the smoke-test extraction pipeline.

These tests run the real pipeline (manifest_loader + background_region +
the shared complexity_features module) end to end, but always against an
isolated tmp_path output directory — never the shared
results/calibration/ directory that holds the real Task 5 deliverable.
"""

from __future__ import annotations

import csv
import json

import pytest

from research.component1.function2.dataset import smoke_test_extraction
from research.component1.function2.dataset.manifest_loader import (
    SplitMembershipError,
    assert_train_split,
    build_split_lookup,
    load_split_manifest,
)


@pytest.fixture(scope="module")
def split_lookup():
    return build_split_lookup()


def test_locked_splits_are_never_touched_by_smoke_test(tmp_path, split_lookup):
    """Hard requirement from the task brief: assert none of the sampled
    images have split == 'test' (or 'valid') — checked against the
    AUTHORITATIVE manifest, not folder/filename heuristics."""
    meta = smoke_test_extraction.run(sample_size=15, output_dir=tmp_path)
    assert meta["split_touched"] == ["train"]
    assert meta["all_rows_confirmed_train_split"] is True

    with open(tmp_path / "smoke_test_features.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 15
    for row in rows:
        assert row["split"] == "train"
        record = split_lookup[row["image_id"]]
        assert record.split == "train"
        assert record.split != "test"
        assert record.split != "valid"


def test_image_id_naming_is_not_a_reliable_split_signal(split_lookup):
    """Documented F2.1A finding: some train-split image_ids contain the
    substring 'chenab_test_' (an artefact of the historical Roboflow
    export name), even though the authoritative split_manifest.csv marks
    them split='train'. This is exactly why assert_train_split() checks the
    manifest instead of string-matching the image_id."""
    misleading = [
        r for r in load_split_manifest()
        if r.split == "train" and "test" in r.image_id.lower()
    ]
    assert len(misleading) > 0, (
        "expected at least one train-split image_id containing 'test' in this "
        "dataset (observed during F2.1A verification) -- if this no longer "
        "holds, the finding in the F2.1A report should be revisited"
    )
    for rec in misleading[:5]:
        confirmed = assert_train_split(rec.image_id, split_lookup)
        assert confirmed.split == "train"


def test_assert_train_split_rejects_valid_and_test_ids(split_lookup):
    valid_id = next(r.image_id for r in load_split_manifest() if r.split == "valid")
    test_id = next(r.image_id for r in load_split_manifest() if r.split == "test")
    with pytest.raises(SplitMembershipError):
        assert_train_split(valid_id, split_lookup)
    with pytest.raises(SplitMembershipError):
        assert_train_split(test_id, split_lookup)


def test_assert_train_split_rejects_unknown_id(split_lookup):
    with pytest.raises(SplitMembershipError):
        assert_train_split("not_a_real_image_id", split_lookup)


def test_source_file_provenance_matches_manifest(tmp_path):
    """Every row's relative_path must point at a real file under
    train/images/ whose filename matches the manifest for that image_id."""
    from research.component1.function2.dataset.manifest_loader import load_train_manifest

    meta = smoke_test_extraction.run(sample_size=10, output_dir=tmp_path)
    assert meta["sample_size_produced"] == 10

    train_by_id = {r.image_id: r for r in load_train_manifest()}
    with open(tmp_path / "smoke_test_features.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        manifest_rec = train_by_id[row["image_id"]]
        assert row["relative_path"].endswith(manifest_rec.filename)
        assert "train/images/" in row["relative_path"].replace("\\", "/")
        assert row["source_family_group_id"] == manifest_rec.group_id


def test_output_reproducible_same_sample_size_twice(tmp_path):
    """Running the pipeline twice with the same sample_size (and therefore
    the same deterministic sample + the same deterministic feature module)
    must produce byte-identical CSV content."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    smoke_test_extraction.run(sample_size=8, output_dir=out1)
    smoke_test_extraction.run(sample_size=8, output_dir=out2)

    csv1 = (out1 / "smoke_test_features.csv").read_text(encoding="utf-8")
    csv2 = (out2 / "smoke_test_features.csv").read_text(encoding="utf-8")
    assert csv1 == csv2


def test_metadata_json_has_no_fabricated_complexity_labels(tmp_path):
    """F2.1A explicitly forbids Simple/Moderate/High labels or thresholds
    at this stage; only raw measurements/config should be emitted."""
    meta = smoke_test_extraction.run(sample_size=5, output_dir=tmp_path)
    serialized = json.dumps(meta).lower()
    for forbidden in ("simple", "moderate complexity", "high complexity", "complexity_label", "complexity_class"):
        assert forbidden not in serialized


def test_sample_includes_at_least_one_negative_and_one_positive_image(tmp_path):
    """Sanity check that the deterministic sample is not degenerate (e.g.
    all-negative or all-positive), which would make the background-adapter
    test coverage misleading."""
    meta = smoke_test_extraction.run(sample_size=20, output_dir=tmp_path)
    with open(tmp_path / "smoke_test_features.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    negatives = [r for r in rows if r["is_negative_manifest"] == "True"]
    positives = [r for r in rows if r["is_negative_manifest"] == "False"]
    assert len(negatives) > 0
    assert len(positives) > 0
    assert meta["sample_size_produced"] == 20
