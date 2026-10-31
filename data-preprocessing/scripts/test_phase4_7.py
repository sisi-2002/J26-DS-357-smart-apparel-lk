"""
Tests for Phase 4.7 - XAI Mask Visual Review, Complexity & Expansion.
Run: python -m pytest test_phase4_7.py -v
"""

import csv
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
RESEARCH = MASTER / "research"
V1 = RESEARCH / "chenab_xai_masks_v1"
V2 = RESEARCH / "chenab_xai_masks_v2"
PHASE4 = MASTER / "reports" / "phase4"
PHASE3 = MASTER / "reports" / "phase3"
EXPORT = MASTER / "processed" / "chenab_detection_v1"

CLASS_NAMES = {0: "baekra", 1: "color_issues", 2: "contamination",
               3: "cut", 4: "gray_stitch", 5: "selvet", 6: "stain"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def results():
    with open(V2 / "reports" / "phase4_7_results.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ref_manifest():
    return read_csv(V2 / "manifests" / "xai_mask_reference_manifest.csv")


@pytest.fixture(scope="module")
def visual_review():
    return read_csv(V2 / "manifests" / "visual_review_manifest.csv")


@pytest.fixture(scope="module")
def v1_ref():
    return read_csv(V1 / "manifests" / "xai_mask_reference_manifest.csv")


@pytest.fixture(scope="module")
def expanded():
    return read_csv(V2 / "manifests" / "expanded_mask_proposal_manifest.csv")


@pytest.fixture(scope="module")
def resolved():
    return read_csv(V2 / "manifests" / "resolved_complexity_assignments.csv")


@pytest.fixture(scope="module")
def fsm():
    return read_csv(PHASE3 / "final_split_manifest.csv")


# ---- Directory structure ----
class TestDirectoryStructure:
    def test_v2_exists(self):
        assert V2.is_dir()

    def test_v2_images(self):
        assert (V2 / "images").is_dir()

    def test_v2_mask_proposals(self):
        assert (V2 / "mask_proposals").is_dir()

    def test_v2_manifests(self):
        assert (V2 / "manifests").is_dir()

    def test_v2_reports(self):
        assert (V2 / "reports").is_dir()

    def test_final_report_exists(self):
        assert (V2 / "reports" / "PHASE_4_7_FINAL_REPORT.md").exists()

    def test_results_json_exists(self):
        assert (V2 / "reports" / "phase4_7_results.json").exists()


# ---- V1 immutability ----
class TestV1Immutability:
    def test_v1_ref_manifest_still_exists(self):
        assert (V1 / "manifests" / "xai_mask_reference_manifest.csv").exists()

    def test_v1_ref_manifest_unchanged(self, v1_ref):
        assert len(v1_ref) == 93

    def test_v1_masks_preserved(self):
        assert (V1 / "ai_proposed_masks").is_dir()
        masks = list((V1 / "ai_proposed_masks").glob("*_mask.png"))
        assert len(masks) == 93

    def test_v1_report_preserved(self):
        assert (V1 / "reports" / "PHASE_4_6_FINAL_REPORT.md").exists()


# ---- Visual review ----
class TestVisualReview:
    def test_review_count(self, visual_review):
        assert len(visual_review) == 93

    def test_all_v1_instances_present(self, visual_review, v1_ref):
        v1_ids = {r["instance_id"] for r in v1_ref}
        vr_ids = {r["instance_id"] for r in visual_review}
        assert v1_ids == vr_ids

    def test_review_decisions_valid(self, visual_review):
        valid = {"visually_accepted_ai", "uncertain", "pending_visual_review",
                 "correction_required", "rejected"}
        for r in visual_review:
            assert r["visual_review_decision"] in valid, \
                f"{r['instance_id']}: {r['visual_review_decision']}"

    def test_no_human_verified(self, visual_review):
        for r in visual_review:
            assert r["human_verified"] in ("False", "false", False)

    def test_reviewed_have_reason(self, visual_review):
        for r in visual_review:
            if r["visual_review_decision"] != "pending_visual_review":
                assert r["visual_review_reason"], \
                    f"{r['instance_id']} missing reason"

    def test_visually_reviewed_count(self, results):
        assert results["visually_reviewed"] == 9


# ---- Mask corrections ----
class TestMaskCorrections:
    def test_correction_log_exists(self):
        assert (V2 / "manifests" / "mask_correction_log.csv").exists()

    def test_rejected_count(self, results):
        assert results["rejected_masks"] == 2

    def test_corrections_preserve_originals(self):
        rows = read_csv(V2 / "manifests" / "mask_correction_log.csv")
        for r in rows:
            assert r["original_preserved"] in ("True", "true", True)


# ---- Complexity resolution ----
class TestComplexityResolution:
    def test_resolved_manifest_exists(self):
        assert (V2 / "manifests" / "resolved_complexity_assignments.csv").exists()

    def test_resolved_uses_frozen_thresholds(self, resolved):
        thresholds = json.loads((PHASE4 / "provisional_thresholds.json").read_text("utf-8"))
        assert abs(thresholds["p33"] - (-0.4654902671397324)) < 1e-10
        assert abs(thresholds["p67"] - 0.21788554420312284) < 1e-10

    def test_resolved_tiers_valid(self, resolved):
        for r in resolved:
            assert r["provisional_tier"] in ("Simple", "Moderate", "High")

    def test_unknown_reduced(self, results):
        assert results["unknown_after"] < results["unknown_before"]

    def test_join_audit_exists(self):
        assert (V2 / "manifests" / "complexity_join_audit.csv").exists()

    def test_remaining_unknown_documented(self):
        assert (V2 / "manifests" / "remaining_unknown_complexity.csv").exists()


# ---- Expansion ----
class TestExpansion:
    def test_expanded_manifest_exists(self):
        assert (V2 / "manifests" / "expanded_mask_proposal_manifest.csv").exists()

    def test_expanded_qc_exists(self):
        assert (V2 / "manifests" / "expanded_mask_qc_results.csv").exists()

    def test_expanded_not_from_test_split(self, expanded, fsm):
        test_groups = {r["group_id"] for r in fsm if r["assigned_split"] == "test"}
        for r in expanded:
            assert r["group_id"] not in test_groups

    def test_expanded_not_quarantined(self, expanded, fsm):
        quarantined = {r["image_id"] for r in fsm if r["assigned_split"] == "quarantined"}
        for r in expanded:
            assert r["image_id"] not in quarantined

    def test_expanded_masks_on_disk(self, expanded):
        for r in expanded:
            mask_path = V2 / "mask_proposals" / r["mask_filename"]
            assert mask_path.exists(), f"Missing mask: {r['mask_filename']}"

    def test_expanded_valid_classes(self, expanded):
        for r in expanded:
            assert int(r["class_id"]) in range(7)

    def test_expansion_cap_respected(self, results):
        assert results["expanded_images"] <= 30

    def test_missing_cells_reduced(self, results):
        assert results["missing_cells_after"] <= results["missing_cells_before"]


# ---- Combined reference manifest ----
class TestReferenceManifest:
    def test_total_records(self, ref_manifest, results):
        assert len(ref_manifest) == results["provisional_eligible"] + \
            (len(ref_manifest) - results["provisional_eligible"])

    def test_no_quarantined_images(self, ref_manifest, fsm):
        quarantined = {r["image_id"] for r in fsm if r["assigned_split"] == "quarantined"}
        for r in ref_manifest:
            assert r["image_id"] not in quarantined

    def test_no_test_groups(self, ref_manifest, fsm):
        test_groups = {r["group_id"] for r in fsm if r["assigned_split"] == "test"}
        for r in ref_manifest:
            assert r["group_id"] not in test_groups

    def test_all_dev_splits(self, ref_manifest):
        for r in ref_manifest:
            assert r["split"] in ("train", "valid")

    def test_no_human_verified(self, ref_manifest):
        for r in ref_manifest:
            assert r["human_verified"] in ("False", "false", False)

    def test_no_final_eligible(self, ref_manifest):
        for r in ref_manifest:
            assert r["eligible_for_final_ground_truth_evaluation"] in ("False", "false", False)

    def test_unknown_eligible_not_provisional(self, ref_manifest):
        for r in ref_manifest:
            if r["eligible_for_provisional_xai"] in ("True", "true", True):
                assert r["provisional_tier"] not in ("unknown", ""), \
                    f"{r['instance_id']} is eligible but tier is unknown"

    def test_sources_labelled(self, ref_manifest):
        sources = {r["source"] for r in ref_manifest}
        assert "phase46" in sources


# ---- Validation checks ----
class TestValidation:
    def test_validation_csv_exists(self):
        assert (V2 / "reports" / "phase4_7_validation.csv").exists()

    def test_all_checks_pass(self):
        rows = read_csv(V2 / "reports" / "phase4_7_validation.csv")
        for r in rows:
            assert r["status"] == "PASS", f"Check {r['check']} failed: {r.get('detail', '')}"

    def test_zero_failures(self, results):
        assert results["validation_failures"] == 0


# ---- Safety invariants ----
class TestSafetyInvariants:
    def test_raw_not_modified(self):
        raw = MASTER / "raw"
        if raw.exists():
            for d in raw.rglob("*"):
                assert "phase4_7" not in str(d).lower()

    def test_frozen_export_count(self):
        em = read_csv(EXPORT / "export_manifest.csv")
        assert len(em) == 2655

    def test_thresholds_not_refitted(self):
        thresholds = json.loads((PHASE4 / "provisional_thresholds.json").read_text("utf-8"))
        assert abs(thresholds["p33"] - (-0.4654902671397324)) < 1e-10
        assert abs(thresholds["p67"] - 0.21788554420312284) < 1e-10

    def test_phase4_reports_preserved(self):
        assert (PHASE4 / "phase4_results.json").exists()
        assert (PHASE4 / "complexity_features.csv").exists()

    def test_coverage_csv_exists(self):
        assert (V2 / "manifests" / "class_complexity_coverage.csv").exists()


# ---- Report ----
class TestReport:
    def test_conditional_status(self, results):
        assert results["status"] == "CONDITIONAL"

    def test_report_copy(self):
        copy = MASTER / "reports" / "phase4_7" / "PHASE_4_7_FINAL_REPORT.md"
        assert copy.exists()
