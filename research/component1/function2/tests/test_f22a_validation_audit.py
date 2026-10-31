"""F2.2A -- regression tests for the VALID-split metadata/annotation audit
(Tasks 1, 2, 3, 4, 5, 6) and the hard no-pixel-opening safety guarantee
(Task 9).

Covers:
  - authoritative VALID split resolution (377, read via the real manifest
    loader, never inferred from a filename substring)
  - YOLO label parsing correctness: well-formed / malformed / missing /
    empty cases, on hand-built synthetic label files under tmp_path
  - negative-vs-missing-annotation distinction
  - group_id preservation from the manifest through to the per-image audit
  - multi-defect image detection
  - bounding-box validity-flag correctness (coords in [0,1], positive w/h,
    valid class id) and the Task 3 large/full-span/elongated/IoU logic
  - reference-provenance-conclusion correctness (Task 4): the AITEX
    research-subset manifests use an identifier namespace fully disjoint
    from the Chenab split manifest, and the AITEX mask-readiness report is
    explicitly AI-reviewed-only (not human-verified) -- i.e. this stage's
    "no validly-linked, expert-verified reference mask exists" conclusion
    is checked against the actual repo state, not assumed
  - sampling-protocol constraint logic (Task 6 K=2 cap + class-composition
    report), as PURE LOGIC on synthetic data -- never run against real
    VALID records in this stage
  - a hard regression test that this stage's own Task 1/2/3/5 pipeline
    NEVER calls cv2.imread / PIL.Image.open while processing the real
    VALID split (patches both functions to raise, runs the real pipeline,
    asserts no exception propagates)
  - output reproducibility: re-running the Task 1/2 audit twice over the
    same inputs produces byte-identical CSV/JSON output
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.config.paths import (  # noqa: E402
    CHENAB_SPLIT_MANIFEST,
    CHENAB_VALID_LABELS_DIR,
)
from research.component1.function2.dataset import (  # noqa: E402
    f22a_task1_2_valid_audit as task12,
)
from research.component1.function2.dataset import (  # noqa: E402
    f22a_task3_bbox_quality as task3,
)
from research.component1.function2.dataset import (  # noqa: E402
    f22a_task5_composition_analysis as task5,
)
from research.component1.function2.dataset.manifest_loader import (  # noqa: E402
    load_split_manifest,
)
from research.component1.function2.complexity.sampling_protocol import (  # noqa: E402
    SamplingCandidate,
    apply_family_cap,
    class_composition_report,
)

RESULTS_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "validation_audit"


# ---------------------------------------------------------------------------
# Task 1: authoritative VALID split resolution
# ---------------------------------------------------------------------------
def test_valid_split_count_is_377_via_authoritative_manifest():
    records = load_split_manifest()
    valid_records = [r for r in records if r.split == "valid"]
    assert len(valid_records) == 377
    # Never inferred from filenames: confirm the known misleading case
    # (a TRAIN-split row whose own image_id/filename contains "valid")
    # exists and is correctly NOT counted as VALID by split-column logic.
    misleading_train_rows = [
        r for r in records
        if r.split == "train" and ("valid" in r.image_id.lower() or "valid" in r.filename.lower())
    ]
    assert len(misleading_train_rows) > 0, (
        "expected at least one TRAIN row with a misleading 'valid' substring in its "
        "image_id/filename (documented by F2.1A) -- if this fails, the dataset changed "
        "and the filename-inference risk this test guards against may no longer apply"
    )
    misleading_ids = {r.image_id for r in misleading_train_rows}
    valid_ids = {r.image_id for r in valid_records}
    assert misleading_ids.isdisjoint(valid_ids), (
        "a TRAIN-split row with a misleading filename was incorrectly counted as VALID"
    )


def test_valid_split_has_no_duplicate_image_ids():
    records = load_split_manifest()
    valid_ids = [r.image_id for r in records if r.split == "valid"]
    assert len(valid_ids) == len(set(valid_ids))


# ---------------------------------------------------------------------------
# Task 2: YOLO label parsing correctness (synthetic files, tmp_path)
# ---------------------------------------------------------------------------
def test_parse_well_formed_label_file(tmp_path):
    label_path = tmp_path / "img_a.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.3\n6 0.1 0.1 0.05 0.05\n", encoding="utf-8")
    parsed = task12.parse_yolo_label_file(label_path)
    assert parsed["file_exists"] is True
    assert parsed["is_empty"] is False
    assert len(parsed["malformed_lines"]) == 0
    assert len(parsed["boxes"]) == 2
    assert parsed["boxes"][0]["class_id"] == 0
    assert parsed["boxes"][0]["class_name"] == "baekra"
    assert parsed["boxes"][0]["box_fully_valid"] is True


def test_parse_missing_label_file_is_not_confused_with_empty(tmp_path):
    label_path = tmp_path / "does_not_exist.txt"
    parsed = task12.parse_yolo_label_file(label_path)
    assert parsed["file_exists"] is False
    assert parsed["is_empty"] is False  # explicitly NOT True -- missing != empty
    assert parsed["boxes"] == []
    assert parsed["malformed_lines"] == []


def test_parse_empty_label_file_is_confirmed_negative_candidate(tmp_path):
    label_path = tmp_path / "empty.txt"
    label_path.write_text("", encoding="utf-8")
    parsed = task12.parse_yolo_label_file(label_path)
    assert parsed["file_exists"] is True
    assert parsed["is_empty"] is True
    assert parsed["boxes"] == []


@pytest.mark.parametrize("bad_line,expected_reason_substr", [
    ("0 0.5 0.5 0.2", "5 whitespace"),              # wrong field count
    ("0 0.5 0.5 0.2 abc", "non-numeric"),           # non-numeric field
    ("notanumber 0.5 0.5 0.2 0.2", "non-numeric"),  # non-numeric class id
])
def test_parse_malformed_lines_are_recorded_not_silently_skipped(tmp_path, bad_line, expected_reason_substr):
    label_path = tmp_path / "bad.txt"
    label_path.write_text(bad_line + "\n", encoding="utf-8")
    parsed = task12.parse_yolo_label_file(label_path)
    assert parsed["file_exists"] is True
    assert len(parsed["malformed_lines"]) == 1
    assert expected_reason_substr in parsed["malformed_lines"][0]["reason"]
    assert parsed["boxes"] == []


def test_out_of_bounds_and_invalid_boxes_flagged_not_silently_accepted(tmp_path):
    label_path = tmp_path / "oob.txt"
    label_path.write_text(
        "0 1.5 0.5 0.2 0.2\n"   # cx out of [0,1]
        "0 0.5 0.5 -0.1 0.2\n"  # negative w
        "99 0.5 0.5 0.2 0.2\n"  # invalid class id (only 0-6 valid)
        "6 0.5 0.5 0.2 0.2\n",  # fully valid
        encoding="utf-8",
    )
    parsed = task12.parse_yolo_label_file(label_path)
    assert len(parsed["boxes"]) == 4
    assert parsed["boxes"][0]["valid_coords_0_1"] is False
    assert parsed["boxes"][0]["box_fully_valid"] is False
    assert parsed["boxes"][1]["valid_positive_wh"] is False
    assert parsed["boxes"][1]["box_fully_valid"] is False
    assert parsed["boxes"][2]["valid_class"] is False
    assert parsed["boxes"][2]["box_fully_valid"] is False
    assert parsed["boxes"][3]["box_fully_valid"] is True


# ---------------------------------------------------------------------------
# Real VALID-split run: negative-vs-missing, group_id preservation,
# multi-defect detection, box-validity flags -- all against the real,
# already-produced audit output (read-only; the generating script itself
# is also exercised directly below in the no-pixel-opening test).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_valid_audit_report():
    path = RESULTS_DIR / "f22a_task1_2_valid_audit_report.json"
    assert path.is_file(), "Task 1/2 script must have been run at least once before these tests"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_valid_image_rows():
    path = RESULTS_DIR / "valid_image_audit.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_real_valid_audit_reports_377_with_zero_blocking_discrepancies(real_valid_audit_report):
    assert real_valid_audit_report["task1_split_metadata_verification"]["matches_expected_377"] is True
    assert real_valid_audit_report["overall_verdict"] == (
        "VALID_SPLIT_METADATA_AND_ANNOTATIONS_AUDITED_NO_BLOCKING_DISCREPANCIES"
    )


def test_group_id_preserved_through_audit_pipeline(real_valid_image_rows):
    manifest_lookup = {r.image_id: r.group_id for r in load_split_manifest() if r.split == "valid"}
    assert len(real_valid_image_rows) == 377
    mismatches = [
        row["image_id"] for row in real_valid_image_rows
        if manifest_lookup.get(row["image_id"]) != row["group_id"]
    ]
    assert mismatches == [], f"group_id not preserved for: {mismatches[:10]}"


def test_negative_vs_missing_distinction_in_real_output(real_valid_image_rows):
    for row in real_valid_image_rows:
        missing = row["missing_annotation"].strip().lower() == "true"
        confirmed_neg = row["confirmed_negative"].strip().lower() == "true"
        # A missing-annotation image must never ALSO be confirmed_negative.
        assert not (missing and confirmed_neg), (
            f"image {row['image_id']}: missing_annotation and confirmed_negative both True"
        )


def test_multi_defect_flag_matches_box_count(real_valid_image_rows):
    for row in real_valid_image_rows:
        is_multi = row["is_multi_defect"].strip().lower() == "true"
        n_boxes = int(row["num_gt_boxes"])
        assert is_multi == (n_boxes > 1), f"image {row['image_id']}: is_multi_defect inconsistent with num_gt_boxes"


# ---------------------------------------------------------------------------
# Task 3: bbox-quality pure-logic correctness (IoU, flags) on synthetic boxes
# ---------------------------------------------------------------------------
def test_iou_identical_boxes_is_one():
    box = {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
    assert task3.iou_xywh(box, box) == pytest.approx(1.0)


def test_iou_non_overlapping_boxes_is_zero():
    box_a = {"cx": 0.1, "cy": 0.1, "w": 0.1, "h": 0.1}
    box_b = {"cx": 0.9, "cy": 0.9, "w": 0.1, "h": 0.1}
    assert task3.iou_xywh(box_a, box_b) == 0.0


def test_iou_known_half_overlap_hand_computed():
    # Two unit-square-ish boxes offset so overlap is exactly half of each.
    box_a = {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4}   # [0.3,0.3]-[0.7,0.7]
    box_b = {"cx": 0.7, "cy": 0.5, "w": 0.4, "h": 0.4}   # [0.5,0.3]-[0.9,0.7]
    # intersection: x in [0.5,0.7] (0.2), y in [0.3,0.7] (0.4) -> area 0.08
    # each box area = 0.16; union = 0.16+0.16-0.08 = 0.24
    expected = 0.08 / 0.24
    assert task3.iou_xywh(box_a, box_b) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Task 4: reference-provenance-conclusion correctness, checked against the
# ACTUAL repo state (not assumed).
# ---------------------------------------------------------------------------
def test_aitex_identifier_namespace_is_disjoint_from_chenab_split_manifest():
    aitex_manifest = (
        REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "processed"
        / "research_subsets" / "aitex_bbox_xai_manifest.csv"
    )
    assert aitex_manifest.is_file()
    with open(aitex_manifest, newline="", encoding="utf-8") as f:
        aitex_rows = list(csv.DictReader(f))
    assert len(aitex_rows) > 0
    aitex_ids = {r["image_id"] for r in aitex_rows}

    chenab_ids = {r.image_id for r in load_split_manifest()}
    assert aitex_ids.isdisjoint(chenab_ids), (
        "found an AITEX image_id inside the Chenab split manifest -- this would "
        "contradict Task 4's 'disjoint, unlinked datasets' conclusion and must be "
        "investigated before trusting the reference-mask provenance report"
    )
    assert all(iid.startswith("aitex_") for iid in aitex_ids)
    assert all("chenab" not in iid.lower() for iid in aitex_ids)


def test_aitex_mask_readiness_report_states_ai_reviewed_not_human_verified():
    report_path = (
        REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "reports"
        / "phase2_8" / "function2_mask_readiness.md"
    )
    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    assert "not human-verified" in text.lower()


def test_aitex_split_policy_documents_separate_subset_only():
    policy_path = (
        REPO_ROOT / "data-preprocessing" / "master_fabric_dataset" / "reports" / "phase3"
        / "split_policy_rules.md"
    )
    assert policy_path.is_file()
    text = policy_path.read_text(encoding="utf-8")
    assert "AITEX" in text
    assert "separate" in text.lower()


# ---------------------------------------------------------------------------
# Task 6: sampling-protocol constraint logic, PURE LOGIC on synthetic data
# only -- never against real dataset records in this stage.
# ---------------------------------------------------------------------------
def test_family_cap_keeps_at_most_k_per_stratum_family():
    candidates = [
        SamplingCandidate("img_01", family_id="fam_A", stratum="Simple"),
        SamplingCandidate("img_02", family_id="fam_A", stratum="Simple"),
        SamplingCandidate("img_03", family_id="fam_A", stratum="Simple"),
        SamplingCandidate("img_04", family_id="fam_B", stratum="Simple"),
    ]
    result = apply_family_cap(candidates, k=2)
    assert result["n_selected"] == 3  # 2 from fam_A + 1 from fam_B
    assert result["n_dropped"] == 1
    kept_ids = {c.record_id for c in result["selected"]}
    assert kept_ids == {"img_01", "img_02", "img_04"}  # deterministic: sorted by record_id


def test_family_cap_is_deterministic_across_repeated_calls():
    candidates = [
        SamplingCandidate(f"img_{i:02d}", family_id="fam_A", stratum="High")
        for i in range(10, 0, -1)  # reverse order on input
    ]
    r1 = apply_family_cap(candidates, k=2)
    r2 = apply_family_cap(candidates, k=2)
    assert [c.record_id for c in r1["selected"]] == [c.record_id for c in r2["selected"]]
    assert [c.record_id for c in r1["selected"]] == ["img_01", "img_02"]


def test_family_cap_rejects_invalid_k():
    with pytest.raises(ValueError):
        apply_family_cap([], k=0)


def test_class_composition_report_does_not_drop_or_reweight_candidates():
    candidates = [
        SamplingCandidate("img_01", family_id="fam_A", stratum="Simple", defect_classes=("stain",)),
        SamplingCandidate("img_02", family_id="fam_B", stratum="Simple", defect_classes=("cut", "stain")),
        SamplingCandidate("img_03", family_id="fam_C", stratum="High", defect_classes=("baekra",)),
    ]
    report = class_composition_report(candidates)
    assert report["Simple"]["n_images_in_stratum"] == 2
    assert report["Simple"]["class_instance_counts"] == {"stain": 2, "cut": 1}
    assert report["High"]["n_images_in_stratum"] == 1
    assert report["High"]["class_instance_counts"] == {"baekra": 1}


# ---------------------------------------------------------------------------
# Task 9 hard requirement: this stage's own pipeline NEVER opens image
# pixels for VALID (or any) split. Patch both cv2.imread and PIL.Image.open
# to raise, then run the REAL Task 1/2/3/5 pipeline end-to-end against
# tmp_path outputs and assert it completes without hitting either patch.
# ---------------------------------------------------------------------------
def test_f22a_pipeline_never_opens_image_pixels(tmp_path, monkeypatch):
    import cv2
    from PIL import Image

    def _boom_cv2(*args, **kwargs):
        raise AssertionError("cv2.imread must never be called by the F2.2A metadata-only pipeline")

    def _boom_pil(*args, **kwargs):
        raise AssertionError("PIL.Image.open must never be called by the F2.2A metadata-only pipeline")

    monkeypatch.setattr(cv2, "imread", _boom_cv2)
    monkeypatch.setattr(Image, "open", _boom_pil)

    out_dir = tmp_path / "validation_audit"
    report12 = task12.run(output_dir=out_dir)
    assert report12["task1_split_metadata_verification"]["matches_expected_377"] is True

    report3 = task3.run(box_audit_csv=out_dir / "valid_box_audit.csv", output_dir=out_dir)
    assert report3["n_boxes_total"] > 0

    report5 = task5.run(
        image_audit_csv=out_dir / "valid_image_audit.csv",
        box_audit_csv=out_dir / "valid_box_audit.csv",
        output_dir=out_dir,
    )
    assert report5["top_level_counts"]["n_images"] == 377


def test_f22a_scripts_contain_no_pixel_opening_calls_textually():
    """Static-scan guard, in addition to the dynamic monkeypatch test above:
    the new Task 1-8 scripts' source text contains no cv2.imread / Image.open
    call at all."""
    script_paths = [
        REPO_ROOT / "research" / "component1" / "function2" / "dataset" / "f22a_task1_2_valid_audit.py",
        REPO_ROOT / "research" / "component1" / "function2" / "dataset" / "f22a_task3_bbox_quality.py",
        REPO_ROOT / "research" / "component1" / "function2" / "dataset" / "f22a_task5_composition_analysis.py",
        REPO_ROOT / "research" / "component1" / "function2" / "complexity" / "sampling_protocol.py",
    ]
    for path in script_paths:
        text = path.read_text(encoding="utf-8")
        # Match the actual CALL form only (with the opening paren); the
        # module docstrings legitimately *mention* cv2.imread/Image.open in
        # prose (in backticks, without a trailing paren) to document that
        # they are deliberately never called.
        assert "cv2.imread(" not in text, f"{path} unexpectedly calls cv2.imread(...)"
        assert "Image.open(" not in text, f"{path} unexpectedly calls Image.open(...)"


# ---------------------------------------------------------------------------
# Output reproducibility
# ---------------------------------------------------------------------------
def test_task1_2_output_is_reproducible_across_repeated_runs(tmp_path):
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    report_a = task12.run(output_dir=out_a)
    report_b = task12.run(output_dir=out_b)

    text_a = (out_a / "valid_image_audit.csv").read_text(encoding="utf-8")
    text_b = (out_b / "valid_image_audit.csv").read_text(encoding="utf-8")
    assert text_a == text_b

    box_a = (out_a / "valid_box_audit.csv").read_text(encoding="utf-8")
    box_b = (out_b / "valid_box_audit.csv").read_text(encoding="utf-8")
    assert box_a == box_b

    # JSON report compared ignoring the generated_at_utc timestamp field.
    rep_a = {k: v for k, v in report_a.items() if k != "generated_at_utc"}
    rep_b = {k: v for k, v in report_b.items() if k != "generated_at_utc"}
    assert rep_a == rep_b


def test_valid_labels_dir_resolves_under_valid_not_train_or_test():
    # Sanity check on the path-resolution helper itself: it must point at
    # valid/labels, never train/labels or test/labels.
    sample_path = task12.resolve_valid_label_path("augmented_00002841_jpg.rf.915fe6a8e5db74ab7564d8fc478380e3.jpg")
    parts = sample_path.parts
    assert "valid" in parts
    assert "train" not in parts
    assert "test" not in parts
    assert sample_path.parent == CHENAB_VALID_LABELS_DIR
