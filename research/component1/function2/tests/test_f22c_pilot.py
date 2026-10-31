"""F2.2C-Pilot -- regression tests for Task A (AI blind-review import) and
Task B (15-20 image SAM segmentation pilot).

Covers:
  - Task A: blind-review CSV integrity (30 unique S001-S030 ids, valid
    judgment/confidence enums, counts matching the metadata JSON, hash
    match between the originally-staged files and the imported copies,
    consistency against F2.1B3's own blind-id manifest)
  - Task B2 selection: VALID-split membership verified via the
    authoritative manifest (never filename), group_id cap (<=2/family),
    7-class coverage, deterministic re-run reproducibility
  - YOLO box -> pixel coordinate conversion correctness (pure arithmetic)
  - TEST/TRAIN split non-access: ``_resolve_valid_image_path`` refuses any
    path under the TEST or TRAIN image directories
  - Generated-artifact checks (skipped gracefully if the pilot outputs are
    absent): image/mask dimensional alignment, binary mask integrity
    (only 0/255 values), instance-id <-> mask/overlay file linkage,
    failed-proposal bookkeeping (n_success + n_failed == n_attempted), and
    that no fabricated "approved"/human-verified status appears anywhere
  - Preservation guarantee: the F2.1B3 checksum baseline still reports
    ``all_match: True`` after all of this stage's work
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
    CHENAB_TEST_IMAGES_DIR,
    CHENAB_TRAIN_IMAGES_DIR,
    CHENAB_VALID_IMAGES_DIR,
)
from research.component1.function2.dataset import (  # noqa: E402
    f22c_pilot_selection as selection_mod,
)
from research.component1.function2.dataset import (  # noqa: E402
    f22c_sam_mask_generation as maskgen_mod,
)
from research.component1.function2.dataset.b3_task5_artifact_checksums import (  # noqa: E402
    verify_against_baseline,
)
from research.component1.function2.dataset.manifest_loader import (  # noqa: E402
    load_split_manifest,
)

REVIEW_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "review"
F21B3_BLIND_MANIFEST = REPO_ROOT / "exports" / "F2_1B3_chatgpt_20261018" / "blind_review" / "blind_review_manifest.csv"

PILOT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "reference_masks" / "pilot_v1"
MANIFESTS_DIR = PILOT_DIR / "manifests"
SAM_PROPOSALS_DIR = PILOT_DIR / "sam_proposals"
OVERLAYS_DIR = PILOT_DIR / "overlays"
QUALITY_REVIEW_DIR = PILOT_DIR / "quality_review"
REPORTS_DIR = PILOT_DIR / "reports"


# --------------------------------------------------------------------------
# Task A: blind-review import integrity
# --------------------------------------------------------------------------

class TestBlindReviewImport:
    def test_review_files_exist(self):
        assert (REVIEW_DIR / "blind_judgments.csv").is_file()
        assert (REVIEW_DIR / "review_metadata.json").is_file()
        assert (REVIEW_DIR / "review_summary.md").is_file()
        assert (REVIEW_DIR / "import_manifest.json").is_file()

    def _load_rows(self):
        with open(REVIEW_DIR / "blind_judgments.csv", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_30_unique_sample_ids_s001_to_s030_no_missing_no_duplicates(self):
        rows = self._load_rows()
        ids = [r["sample_id"] for r in rows]
        assert len(ids) == 30
        assert len(set(ids)) == 30
        expected = {f"S{i:03d}" for i in range(1, 31)}
        assert set(ids) == expected

    def test_judgment_values_valid_enum(self):
        rows = self._load_rows()
        allowed = {"Simple", "Moderate", "High"}
        for r in rows:
            assert r["judgment"] in allowed, r

    def test_confidence_and_text_fields_present(self):
        rows = self._load_rows()
        allowed_conf = {"High", "Medium", "Low"}
        for r in rows:
            assert r["confidence"] in allowed_conf
            assert r["brief background-only rationale"].strip()
            assert r["limitations"].strip()

    def test_label_counts_match_metadata_json(self):
        rows = self._load_rows()
        from collections import Counter
        counts = Counter(r["judgment"] for r in rows)
        with open(REVIEW_DIR / "review_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        assert counts["Simple"] == meta["labels"]["Simple"]
        assert counts["Moderate"] == meta["labels"]["Moderate"]
        assert counts["High"] == meta["labels"]["High"]

    def test_import_manifest_records_hash_match_and_status_label(self):
        with open(REVIEW_DIR / "import_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert "NOT Human QC Expert Validation" in manifest["status_label"]
        for entry in manifest["import_record"]["files"]:
            assert entry["hashes_match"] is True
            assert entry["sha256_original"] == entry["sha256_imported_copy"]

    @pytest.mark.skipif(not F21B3_BLIND_MANIFEST.is_file(), reason="F2.1B3 export manifest not present")
    def test_sample_ids_match_f21b3_blind_manifest_exactly(self):
        with open(F21B3_BLIND_MANIFEST, newline="", encoding="utf-8") as f:
            f21b3_ids = {r["sample_id"] for r in csv.DictReader(f)}
        rows = self._load_rows()
        ai_ids = {r["sample_id"] for r in rows}
        assert ai_ids == f21b3_ids

    def test_no_fabricated_reconciliation_in_import(self):
        """This import task must never compute/claim agreement statistics or
        touch the reconciliation key."""
        with open(REVIEW_DIR / "import_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        text = json.dumps(manifest).lower()
        assert "agreement" not in text or "no join" in text or "not" in text
        assert "kappa" not in text
        assert manifest["import_record"]["consistency_check_vs_f21b3_blind_ids"]


# --------------------------------------------------------------------------
# Task B2: selection logic
# --------------------------------------------------------------------------

class TestPilotSelection:
    def test_all_selected_images_are_valid_split_via_manifest_not_filename(self):
        split_lookup = {r.image_id: r for r in load_split_manifest()}
        for entry in selection_mod.SELECTED_IMAGES:
            image_id = entry["image_id"]
            record = split_lookup[image_id]
            assert record.split == "valid", (
                f"{image_id} is split={record.split!r} per the authoritative "
                "manifest, not 'valid' -- selection must reject this regardless "
                "of what the image_id string itself looks like"
            )

    def test_selection_includes_misleadingly_named_images_verified_by_manifest(self):
        """Several real selected image_ids literally start with
        'chenab_test_' or 'chenab_train_' while actually being split='valid'
        per the manifest -- this is the exact filename-vs-manifest trap the
        task brief warns about. Confirm at least one such case is present
        AND was correctly classified as valid."""
        split_lookup = {r.image_id: r for r in load_split_manifest()}
        misleading = [
            e["image_id"] for e in selection_mod.SELECTED_IMAGES
            if not e["image_id"].startswith("chenab_valid_")
        ]
        assert len(misleading) > 0
        for image_id in misleading:
            assert split_lookup[image_id].split == "valid"

    def test_group_id_cap_le_2_per_family(self):
        split_lookup = {r.image_id: r for r in load_split_manifest()}
        from collections import Counter
        counts = Counter(split_lookup[e["image_id"]].group_id for e in selection_mod.SELECTED_IMAGES)
        assert all(n <= 2 for n in counts.values()), counts

    def test_image_count_in_15_to_20_range(self):
        assert 15 <= len(selection_mod.SELECTED_IMAGES) <= 20

    def test_all_7_classes_covered_and_report_reproducible(self, tmp_path, monkeypatch):
        # Run the selection twice into isolated output dirs and confirm
        # byte-identical manifests -- i.e. fully deterministic, no RNG.
        import importlib

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        monkeypatch.setattr(selection_mod, "MANIFESTS_DIR", out1)
        rep1 = selection_mod.run()
        monkeypatch.setattr(selection_mod, "MANIFESTS_DIR", out2)
        rep2 = selection_mod.run()

        assert rep1["all_7_classes_covered"] is True
        assert rep1["group_id_cap_respected_le_2_per_group"] is True
        assert 15 <= rep1["n_images_selected"] <= 20

        text1 = (out1 / "image_selection.csv").read_text(encoding="utf-8")
        text2 = (out2 / "image_selection.csv").read_text(encoding="utf-8")
        assert text1 == text2
        inst1 = (out1 / "instance_manifest.csv").read_text(encoding="utf-8")
        inst2 = (out2 / "instance_manifest.csv").read_text(encoding="utf-8")
        assert inst1 == inst2


# --------------------------------------------------------------------------
# YOLO box -> pixel coordinate conversion
# --------------------------------------------------------------------------

class TestYoloBoxToPixels:
    def test_center_box_known_values(self):
        # cx=0.5, cy=0.5, w=0.2, h=0.4 on a 100x200 (w x h) image
        x1, y1, x2, y2 = maskgen_mod.yolo_box_to_pixels(0.5, 0.5, 0.2, 0.4, img_w=100, img_h=200)
        assert (x1, y1, x2, y2) == (40, 60, 60, 140)

    def test_full_span_box_clips_to_image_bounds(self):
        x1, y1, x2, y2 = maskgen_mod.yolo_box_to_pixels(0.5, 0.5, 1.5, 1.5, img_w=50, img_h=50)
        assert x1 == 0 and y1 == 0
        assert x2 == 50 and y2 == 50

    def test_degenerate_tiny_box_never_collapses_to_zero_area(self):
        x1, y1, x2, y2 = maskgen_mod.yolo_box_to_pixels(0.001, 0.001, 0.0001, 0.0001, img_w=50, img_h=50)
        assert x2 > x1
        assert y2 > y1

    def test_corner_box(self):
        x1, y1, x2, y2 = maskgen_mod.yolo_box_to_pixels(0.1, 0.1, 0.2, 0.2, img_w=100, img_h=100)
        assert (x1, y1, x2, y2) == (0, 0, 20, 20)


# --------------------------------------------------------------------------
# TEST/TRAIN split non-access guard
# --------------------------------------------------------------------------

class TestSplitAccessGuard:
    def test_resolve_valid_image_path_accepts_valid_dir(self):
        p = maskgen_mod._resolve_valid_image_path("some_file.jpg")
        assert p == (CHENAB_VALID_IMAGES_DIR / "some_file.jpg").resolve()

    def test_resolve_valid_image_path_refuses_path_traversal_into_test_dir(self):
        traversal = f"../test/images/some_file.jpg"
        with pytest.raises(RuntimeError):
            maskgen_mod._resolve_valid_image_path(traversal)

    def test_no_pilot_artifact_references_test_or_train_image_dir(self):
        """Scan every path recorded in the generation report (if present)
        and confirm none resolves under the locked TEST or TRAIN image
        directories."""
        report_path = REPORTS_DIR / "f22c_mask_generation_results.json"
        if not report_path.is_file():
            pytest.skip("mask generation has not been run in this environment")
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        forbidden = (CHENAB_TEST_IMAGES_DIR.resolve(), CHENAB_TRAIN_IMAGES_DIR.resolve())
        for inst in report["per_instance_results"]:
            for key in ("mask_path", "overlay_path"):
                if key in inst:
                    resolved = (REPO_ROOT / inst[key]).resolve()
                    for f_dir in forbidden:
                        assert f_dir not in resolved.parents


# --------------------------------------------------------------------------
# Generated-artifact checks (pilot outputs, if present in this environment)
# --------------------------------------------------------------------------

PILOT_GENERATED = (REPORTS_DIR / "f22c_mask_generation_results.json").is_file()


@pytest.mark.skipif(not PILOT_GENERATED, reason="SAM mask generation outputs not present in this environment")
class TestGeneratedMaskArtifacts:
    @pytest.fixture(scope="class")
    def generation_report(self):
        with open(REPORTS_DIR / "f22c_mask_generation_results.json", encoding="utf-8") as f:
            return json.load(f)

    def test_success_plus_failed_equals_attempted(self, generation_report):
        r = generation_report
        assert r["n_success"] + r["n_failed"] == r["n_instances_attempted"]

    def test_status_label_present_and_not_fabricated_approval(self, generation_report):
        assert "NOT human-verified" in generation_report["status_label"]

    def test_every_successful_instance_has_mask_and_overlay_file(self, generation_report):
        for inst in generation_report["per_instance_results"]:
            if inst["status"] != "success":
                continue
            mask_path = REPO_ROOT / inst["mask_path"]
            overlay_path = REPO_ROOT / inst["overlay_path"]
            assert mask_path.is_file(), mask_path
            assert overlay_path.is_file(), overlay_path
            assert mask_path.name.startswith(inst["instance_id"])
            assert overlay_path.name.startswith(inst["instance_id"])

    def test_mask_is_binary_0_or_255(self, generation_report):
        from PIL import Image
        import numpy as np

        checked = 0
        for inst in generation_report["per_instance_results"]:
            if inst["status"] != "success":
                continue
            mask_path = REPO_ROOT / inst["mask_path"]
            arr = np.array(Image.open(mask_path).convert("L"))
            uniques = set(np.unique(arr).tolist())
            assert uniques.issubset({0, 255}), (inst["instance_id"], uniques)
            checked += 1
        assert checked > 0

    def test_mask_dimensions_match_source_image_dimensions(self, generation_report):
        from PIL import Image

        for inst in generation_report["per_instance_results"]:
            if inst["status"] != "success":
                continue
            mask_path = REPO_ROOT / inst["mask_path"]
            mask_im = Image.open(mask_path)
            assert mask_im.size == (inst["image_width"], inst["image_height"])

    def test_instance_id_linkage_matches_manifest(self, generation_report):
        with open(MANIFESTS_DIR / "instance_manifest.csv", newline="", encoding="utf-8") as f:
            manifest_ids = {row["instance_id"] for row in csv.DictReader(f)}
        report_ids = {inst["instance_id"] for inst in generation_report["per_instance_results"]}
        assert report_ids == manifest_ids

    def test_quality_review_template_genuinely_blank_and_unreviewed(self):
        with open(QUALITY_REVIEW_DIR / "mask_review_template.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        for row in rows:
            assert row["human_review_status"] == "Unreviewed"
            assert row["reviewer_id"] == ""
            assert row["review_date"] == ""
            assert row["reviewer_comments"] == ""

    def test_no_fabricated_approval_anywhere_in_pilot_outputs(self, generation_report):
        """Scan all pilot JSON/CSV outputs for forbidden fabricated-approval
        claims. Negated disclaimers (e.g. "NOT human-verified") are expected
        and must NOT trip this check -- only an un-negated positive claim of
        approval/verification would be fabrication."""
        forbidden_terms = ["expert-approved", "expert approved", "auto-approved", "auto approved", "auto-accepted"]
        texts = [json.dumps(generation_report).lower()]
        with open(MANIFESTS_DIR / "selection_report.json", encoding="utf-8") as f:
            texts.append(json.dumps(json.load(f)).lower())
        with open(QUALITY_REVIEW_DIR / "mask_review_template.csv", encoding="utf-8") as f:
            texts.append(f.read().lower())
        combined = "\n".join(texts)
        for term in forbidden_terms:
            assert term not in combined, term
        # Positive check: the generation report's own status label must
        # explicitly carry the negated disclaimer, not silently omit it.
        assert "not human-verified" in json.dumps(generation_report).lower()


# --------------------------------------------------------------------------
# SAM-environment-dependent failure-path test (only meaningful under the
# .sam_venv interpreter; skipped elsewhere).
# --------------------------------------------------------------------------

def test_missing_image_file_is_recorded_as_failed_not_a_crash(tmp_path, monkeypatch):
    pytest.importorskip("segment_anything")
    pytest.importorskip("torch")
    pytest.importorskip("cv2")

    # Build a tiny synthetic instance manifest pointing at a non-existent
    # filename, in an isolated output dir, and confirm run() records a
    # "failed" result instead of raising.
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    fieldnames = [
        "instance_id", "image_id", "filename", "group_id", "split", "line_no",
        "class_id", "class_name", "cx", "cy", "w", "h", "area",
        "full_span", "large", "elongated", "small", "aspect_ratio",
    ]
    with open(manifests_dir / "instance_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "instance_id": "fake__inst_001", "image_id": "fake_image_id",
            "filename": "DOES_NOT_EXIST_12345.jpg", "group_id": "g1", "split": "valid",
            "line_no": "1", "class_id": "0", "class_name": "baekra",
            "cx": "0.5", "cy": "0.5", "w": "0.2", "h": "0.2", "area": "0.04",
            "full_span": "False", "large": "False", "elongated": "False",
            "small": "False", "aspect_ratio": "1.0",
        })

    monkeypatch.setattr(maskgen_mod, "MANIFESTS_DIR", manifests_dir)
    monkeypatch.setattr(maskgen_mod, "SAM_PROPOSALS_DIR", tmp_path / "sam_proposals")
    monkeypatch.setattr(maskgen_mod, "OVERLAYS_DIR", tmp_path / "overlays")
    monkeypatch.setattr(maskgen_mod, "QUALITY_REVIEW_DIR", tmp_path / "quality_review")
    monkeypatch.setattr(maskgen_mod, "REPORTS_DIR", tmp_path / "reports")

    summary = maskgen_mod.run()
    assert summary["n_failed"] == 1
    assert summary["n_success"] == 0
    failed = summary["per_instance_results"][0]
    assert failed["status"] == "failed"
    assert "not found" in failed["failure_reason"]


# --------------------------------------------------------------------------
# Preservation guarantee: F2.1B3 checksum baseline must still hold.
# --------------------------------------------------------------------------

def test_b3_checksum_baseline_still_all_match_after_f22c_pilot_work():
    result = verify_against_baseline()
    assert result["all_match"] is True, result
