"""Tests for Phase 4.8 — Final AI Mask Review & Phase 4 Completion."""

import csv, json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
RESEARCH = MASTER / "research"
V1 = RESEARCH / "chenab_xai_masks_v1"
V2 = RESEARCH / "chenab_xai_masks_v2"
V3 = RESEARCH / "chenab_xai_masks_v3"
PHASE3 = MASTER / "reports" / "phase3"
PHASE4 = MASTER / "reports" / "phase4"
EXPORT = MASTER / "processed" / "chenab_detection_v1"

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

@pytest.fixture(scope="module")
def results():
    with open(V3 / "reports" / "phase4_8_results.json", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def manifest():
    return read_csv(V3 / "manifests" / "final_mask_manifest.csv")

@pytest.fixture(scope="module")
def fsm():
    return read_csv(PHASE3 / "final_split_manifest.csv")

@pytest.fixture(scope="module")
def review():
    return read_csv(V2 / "manifests" / "visual_review_final.csv")

# --- Directory structure ---
class TestV3Structure:
    def test_v3_exists(self):
        assert V3.is_dir()
    def test_images_dir(self):
        assert (V3 / "images").is_dir()
    def test_reference_masks_dir(self):
        assert (V3 / "reference_masks").is_dir()
    def test_manifests_dir(self):
        assert (V3 / "manifests").is_dir()
    def test_configs_dir(self):
        assert (V3 / "configs").is_dir()
    def test_reports_dir(self):
        assert (V3 / "reports").is_dir()
    def test_final_report(self):
        assert (V3 / "reports" / "PHASE_4_8_FINAL_REPORT.md").exists()
    def test_complete_report(self):
        assert (PHASE4 / "PHASE_4_COMPLETE_REPORT.md").exists()

# --- V1/V2 immutability ---
class TestImmutability:
    def test_v1_manifest(self):
        rows = read_csv(V1 / "manifests" / "xai_mask_reference_manifest.csv")
        assert len(rows) == 93
    def test_v1_masks(self):
        assert len(list((V1 / "ai_proposed_masks").glob("*_mask.png"))) == 93
    def test_v2_manifest(self):
        rows = read_csv(V2 / "manifests" / "xai_mask_reference_manifest.csv")
        assert len(rows) == 133
    def test_frozen_export(self):
        assert len(read_csv(EXPORT / "export_manifest.csv")) == 2655
    def test_frozen_thresholds(self):
        t = json.loads((PHASE4 / "provisional_thresholds.json").read_text("utf-8"))
        assert abs(t["p33"] - (-0.4654902671397324)) < 1e-10
        assert abs(t["p67"] - 0.21788554420312284) < 1e-10

# --- Visual review ---
class TestVisualReview:
    def test_all_inspected(self, review):
        for r in review:
            assert r["image_inspected"] in ("True", "true", True)
    def test_133_records(self, review):
        assert len(review) == 133
    def test_valid_decisions(self, review):
        valid = {"visually_accepted_ai", "uncertain", "rejected"}
        for r in review:
            assert r["visual_review_decision"] in valid
    def test_no_human_verified(self, review):
        for r in review:
            assert r["human_verified"] in ("False", "false", False)
    def test_reasons_present(self, review):
        for r in review:
            assert r["visual_review_reason"]

# --- Final manifest ---
class TestFinalManifest:
    def test_total_records(self, manifest):
        assert len(manifest) == 133
    def test_no_test_leakage(self, manifest, fsm):
        test_groups = {r["group_id"] for r in fsm if r["assigned_split"] == "test"}
        for r in manifest:
            assert r["group_id"] not in test_groups
    def test_no_quarantined(self, manifest, fsm):
        quarantined = {r["image_id"] for r in fsm if r["assigned_split"] == "quarantined"}
        for r in manifest:
            assert r["image_id"] not in quarantined
    def test_valid_classes(self, manifest):
        for r in manifest:
            assert int(r["class_id"]) in range(7)
    def test_no_human_verified(self, manifest):
        for r in manifest:
            assert r["human_verified"] in ("False", "false", False)
    def test_no_final_gt(self, manifest):
        for r in manifest:
            assert r["final_ground_truth_eligible"] in ("False", "false", False)
    def test_masks_on_disk(self, manifest):
        missing = []
        for r in manifest:
            if r["mask_filename"]:
                p = V3 / "reference_masks" / r["mask_filename"]
                if not p.exists():
                    missing.append(r["instance_id"])
        assert len(missing) == 0, f"Missing: {missing[:5]}"
    def test_eligible_have_complexity(self, manifest):
        for r in manifest:
            if r["ai_reference_eligible"] in ("True", "true", True):
                assert r["provisional_tier"] not in ("unknown", "")

# --- Eligibility ---
class TestEligibility:
    def test_eligible_count(self, results):
        assert results["ai_reference_eligible"] >= 100
    def test_total_images(self, results):
        assert results["total_images"] == 100
    def test_total_instances(self, results):
        assert results["total_instances"] == 133

# --- Configs ---
class TestConfigs:
    def test_calibration_config(self):
        c = json.loads((V3 / "configs" / "provisional_complexity_calibration.json").read_text("utf-8"))
        assert c["mode"] == "provisional_ai"
        assert c["expert_validation"] == "pending_qc"
    def test_qc_protocol(self):
        c = json.loads((V3 / "configs" / "mask_qc_protocol.json").read_text("utf-8"))
        assert c["sam_model"] == "vit_b"
        assert c["human_verification"] == "not_performed"

# --- Validation ---
class TestValidation:
    def test_all_checks_pass(self):
        rows = read_csv(V3 / "reports" / "phase4_8_validation.csv")
        for r in rows:
            assert r["status"] == "PASS", f"{r['check']}: {r.get('detail','')}"
    def test_zero_failures(self, results):
        assert results["validation_failures"] == 0

# --- Status ---
class TestStatus:
    def test_technically_complete(self, results):
        assert results["phase48_technical_status"] == "TECHNICALLY COMPLETE"
    def test_phase4_complete(self, results):
        assert results["phase4_technical_status"] == "TECHNICALLY COMPLETE"
    def test_conditional_science(self, results):
        assert results["scientific_readiness"] == "CONDITIONAL"

# --- Coverage ---
class TestCoverage:
    def test_coverage_csv(self):
        assert (V3 / "manifests" / "final_class_complexity_coverage.csv").exists()
    def test_missing_cells(self, results):
        assert results["missing_coverage_cells"] <= 1
    def test_hashes(self):
        assert (V3 / "manifests" / "file_hashes.csv").exists()
        rows = read_csv(V3 / "manifests" / "file_hashes.csv")
        assert len(rows) > 0

# --- Safety ---
class TestSafety:
    def test_raw_not_modified(self):
        raw = MASTER / "raw"
        if raw.exists():
            for d in raw.rglob("*"):
                assert "phase4_8" not in str(d).lower()
                assert "v3" not in d.name.lower()
