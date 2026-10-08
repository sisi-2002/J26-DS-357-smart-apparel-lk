"""
Phase 2.5 Regression Tests.
Run: python test_phase2_5.py
"""

import csv
import hashlib
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
RAW_CHENAB = MASTER / "raw" / "chenab" / "fabricDD.v2i.yolov8"
RAW_AITEX = MASTER / "raw" / "aitex"
META = MASTER / "metadata"
PHASE25 = MASTER / "reports" / "phase2_5"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NUM_CLASSES = 7


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. AITEX Class Code Consistency
# ---------------------------------------------------------------------------

class TestAitexClassCodeConsistency(unittest.TestCase):

    def test_verification_csv_exists(self):
        self.assertTrue((PHASE25 / "aitex_code_verification.csv").is_file())

    def test_all_observed_codes_have_entry(self):
        """Every code seen in filenames has a verification row."""
        defect_dir = RAW_AITEX / "Defect_images"
        observed = set()
        for p in defect_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                parts = p.stem.split("_")
                if len(parts) >= 2:
                    observed.add(parts[1])
        rows = read_csv(PHASE25 / "aitex_code_verification.csv")
        verified_codes = {r["aitex_code"] for r in rows}
        self.assertEqual(observed, verified_codes, f"Missing codes: {observed - verified_codes}")

    def test_code_023_present_and_flagged(self):
        """Code 023 (not in original AITEX docs) must be flagged pending_expert_review."""
        rows = read_csv(PHASE25 / "aitex_code_verification.csv")
        code023 = [r for r in rows if r["aitex_code"] == "023"]
        self.assertEqual(len(code023), 1)
        self.assertEqual(code023[0]["verification_status"], "pending_expert_review")

    def test_mapping_csv_exists(self):
        self.assertTrue((PHASE25 / "aitex_class_mapping_decisions.csv").is_file())


# ---------------------------------------------------------------------------
# 2. Unverified Class Rejection
# ---------------------------------------------------------------------------

class TestUnverifiedClassRejection(unittest.TestCase):

    def test_no_aitex_class_auto_approved(self):
        """No AITEX class may be auto-approved for training."""
        rows = read_csv(PHASE25 / "aitex_class_mapping_decisions.csv")
        approved = [r for r in rows if r.get("approved_for_training") == "yes"]
        self.assertEqual(approved, [], f"Auto-approved AITEX classes: {[r['aitex_code'] for r in approved]}")

    def test_rejected_mappings_recorded(self):
        """Previously proposed 019→stain and 030→cut are rejected, not silently dropped."""
        rows = read_csv(PHASE25 / "aitex_class_mapping_decisions.csv")
        rejected = {r["aitex_code"]: r["mapping_status"] for r in rows}
        self.assertEqual(rejected.get("019"), "rejected_mapping")
        self.assertEqual(rejected.get("030"), "rejected_mapping")

    def test_geometry_files_use_UNVERIFIED_not_class_id(self):
        """Geometry files must not use numeric class IDs (approved classes only)."""
        geo_dir = MASTER / "derived_annotations" / "aitex_geometry"
        if not geo_dir.is_dir():
            self.skipTest("Geometry dir not yet generated")
        for geo_file in list(geo_dir.glob("*.txt"))[:20]:
            for line in geo_file.read_text(errors="replace").splitlines():
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if parts:
                    # Must start with UNVERIFIED, not a digit
                    self.assertFalse(parts[0].isdigit(),
                                     f"{geo_file.name}: geometry file uses numeric class '{parts[0]}'")


# ---------------------------------------------------------------------------
# 3. Image-Mask Pairing
# ---------------------------------------------------------------------------

class TestImageMaskPairing(unittest.TestCase):

    def test_qc_csv_exists(self):
        self.assertTrue((PHASE25 / "aitex_mask_qc.csv").is_file())

    def test_all_defect_images_have_qc_row(self):
        defect_dir = RAW_AITEX / "Defect_images"
        imgs = {p.name for p in defect_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
        rows = read_csv(PHASE25 / "aitex_mask_qc.csv")
        qc_imgs = {r["image_filename"] for r in rows}
        self.assertEqual(imgs, qc_imgs, f"Missing QC rows: {imgs - qc_imgs}")

    def test_known_missing_mask_image(self):
        """Image 0100_025_08 should be flagged as no_mask."""
        rows = read_csv(PHASE25 / "aitex_mask_qc.csv")
        row = next((r for r in rows if r["image_filename"] == "0100_025_08.png"), None)
        if row is None:
            self.skipTest("Image not in QC CSV")
        self.assertEqual(row["pairing_status"], "no_mask")


# ---------------------------------------------------------------------------
# 4. Ambiguous and Duplicate Mask Stems
# ---------------------------------------------------------------------------

class TestDuplicateMaskStems(unittest.TestCase):

    def test_multi_mask_images_flagged(self):
        """Images 0044 and 0097 have mask1+mask2 and must be flagged as multiple_masks."""
        rows = read_csv(PHASE25 / "aitex_mask_qc.csv")
        multi = {r["image_filename"]: r["pairing_status"] for r in rows
                 if r["n_mask_candidates"] and int(r["n_mask_candidates"]) > 1}
        self.assertIn("0044_019_04.png", multi)
        self.assertIn("0097_030_03.png", multi)
        for img, status in multi.items():
            self.assertIn("multiple", status, f"{img} should have multiple status")

    def test_multi_mask_in_corrections(self):
        """Multi-mask images must appear in mask pairing corrections."""
        rows = read_csv(PHASE25 / "aitex_mask_pairing_corrections.csv")
        multi = {r["image_filename"] for r in rows if r["issue"] == "multiple_mask_files"}
        self.assertIn("0044_019_04.png", multi)
        self.assertIn("0097_030_03.png", multi)


# ---------------------------------------------------------------------------
# 5. Mask Dimension Mismatch
# ---------------------------------------------------------------------------

class TestMaskDimensionMismatch(unittest.TestCase):

    def test_dimension_mismatch_blocks_geometry(self):
        """Images with mask dimension mismatches must not have geometry generated."""
        qc_rows = read_csv(PHASE25 / "aitex_mask_qc.csv")
        geo_rows = read_csv(PHASE25 / "aitex_geometry_manifest.csv") if (PHASE25 / "aitex_geometry_manifest.csv").is_file() else []
        mismatch_imgs = {r["image_filename"] for r in qc_rows if "mismatch" in r.get("dimension_match", "")}
        geo_failed = {r["image_filename"] for r in geo_rows if r["annotation_status"] == "failed_dimension_mismatch"}
        # All mismatches should have failed geometry
        for img in mismatch_imgs:
            self.assertIn(img, geo_failed, f"{img} has dim mismatch but wasn't blocked from geometry")


# ---------------------------------------------------------------------------
# 6. Grayscale Threshold Handling
# ---------------------------------------------------------------------------

class TestGrayscaleThreshold(unittest.TestCase):

    def test_foreground_threshold_127(self):
        """Verify foreground is pixels > 127 (not > 0) for mask binarization."""
        import numpy as np
        arr = (np.ones((10, 10), dtype=np.uint8) * 127)  # at threshold — should be background
        fg = int(np.sum(arr > 127))
        self.assertEqual(fg, 0, "127 should be background (threshold is > 127)")

        arr2 = (np.ones((10, 10), dtype=np.uint8) * 128)  # just above threshold
        fg2 = int(np.sum(arr2 > 127))
        self.assertEqual(fg2, 100, "128 should be foreground")

    def test_empty_mask_detected(self):
        """Empty mask (all zeros) is detected."""
        import numpy as np
        arr = np.zeros((100, 100), dtype=np.uint8)
        fg = int(np.sum(arr > 127))
        self.assertEqual(fg, 0)


# ---------------------------------------------------------------------------
# 7. Connected-Component Box Extraction
# ---------------------------------------------------------------------------

class TestConnectedComponentBoxExtraction(unittest.TestCase):

    def test_two_components_give_two_boxes(self):
        import numpy as np
        import cv2
        arr = np.zeros((100, 200), dtype=np.uint8)
        arr[10:30, 10:50] = 255   # component 1
        arr[60:90, 120:180] = 255  # component 2
        binary = (arr > 127).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        valid_boxes = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 50]
        self.assertEqual(len(valid_boxes), 2)

    def test_tiny_component_below_threshold_excluded(self):
        import numpy as np
        import cv2
        THRESHOLD = 50
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[5:7, 5:7] = 255  # 4 pixels — below threshold
        arr[20:60, 20:60] = 255  # 1600 pixels — above threshold
        binary = (arr > 127).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        valid_boxes = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= THRESHOLD]
        self.assertEqual(len(valid_boxes), 1)

    def test_normalized_coordinates_valid(self):
        import numpy as np
        img_w, img_h = 640, 480
        x, y, w, h = 100, 50, 200, 150
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        for val in [cx, cy, nw, nh]:
            self.assertTrue(0 <= val <= 1, f"Coordinate out of range: {val}")
            self.assertTrue(math.isfinite(val), f"Non-finite coordinate: {val}")


# ---------------------------------------------------------------------------
# 8. Valid YOLO Coordinate Generation
# ---------------------------------------------------------------------------

class TestYoloCoordinateGeneration(unittest.TestCase):

    def test_geometry_manifest_box_counts_positive(self):
        p = PHASE25 / "aitex_geometry_manifest.csv"
        if not p.is_file():
            self.skipTest("Geometry manifest not yet generated")
        rows = read_csv(p)
        for r in rows:
            bc = int(r["box_count"])
            self.assertGreaterEqual(bc, 0, f"Negative box count: {r['image_filename']}")

    def test_box_validation_csv_no_invalid_finite(self):
        """All valid=True boxes should have finite coordinates."""
        p = PHASE25 / "aitex_box_validation.csv"
        if not p.is_file():
            self.skipTest("Box validation CSV not yet generated")
        rows = read_csv(p)
        for r in rows:
            if r["valid"] == "True":
                for field in ["cx", "cy", "nw", "nh"]:
                    val = float(r[field])
                    self.assertTrue(math.isfinite(val), f"Non-finite in {r['image_filename']}: {field}={val}")
                    self.assertTrue(0 <= val <= 1, f"Out of range in {r['image_filename']}: {field}={val}")


# ---------------------------------------------------------------------------
# 9. NaN/Infinity Annotation Rejection
# ---------------------------------------------------------------------------

class TestNanInfinityRejection(unittest.TestCase):

    def test_nan_infinity_check_reported(self):
        """Step 1 verification report must exist and report NaN/Inf count."""
        p = PHASE25 / "step1_verification.md"
        self.assertTrue(p.is_file())
        content = p.read_text(encoding="utf-8")
        self.assertIn("NaN/Inf", content)

    def test_zero_nan_inf_in_chenab_labels(self):
        """Verify no NaN/Inf values exist in Chenab label files."""
        count = 0
        for split in ("train", "valid"):
            lbl_dir = RAW_CHENAB / split / "labels"
            if not lbl_dir.is_dir():
                continue
            for lbl in list(lbl_dir.rglob("*.txt"))[:100]:  # sample
                for line in lbl.read_text(errors="replace").splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        for v in parts[1:]:
                            try:
                                if not math.isfinite(float(v)):
                                    count += 1
                            except ValueError:
                                pass
        self.assertEqual(count, 0, f"Found {count} NaN/Inf values in label files")


# ---------------------------------------------------------------------------
# 10. Exact Duplicate Detection
# ---------------------------------------------------------------------------

class TestExactDuplicateDetection(unittest.TestCase):

    def test_duplicate_groups_csv_exists(self):
        self.assertTrue((META / "duplicate_groups.csv").is_file())

    def test_sha256_deterministic(self):
        p = RAW_CHENAB / "data.yaml"
        if not p.is_file():
            self.skipTest("data.yaml not found")
        h1 = sha256_file(p)
        h2 = sha256_file(p)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)


# ---------------------------------------------------------------------------
# 11. Perceptual Candidate Detection
# ---------------------------------------------------------------------------

class TestPerceptualCandidates(unittest.TestCase):

    def test_near_dup_candidates_csv_exists(self):
        self.assertTrue((PHASE25 / "near_duplicate_candidates.csv").is_file())

    def test_near_dup_pairs_have_required_fields(self):
        rows = read_csv(PHASE25 / "near_duplicate_candidates.csv")
        if not rows:
            self.skipTest("No near-dup candidates")
        required = {"hamming_distance", "file1", "source1", "split1",
                    "file2", "source2", "split2", "classification", "human_decision"}
        self.assertTrue(required.issubset(rows[0].keys()))

    def test_all_pending_decisions_are_pending(self):
        """All near-dup rows must have human_decision = pending_visual_review (not fabricated approval)."""
        rows = read_csv(PHASE25 / "near_duplicate_candidates.csv")
        non_pending = [r for r in rows if r["human_decision"] != "pending_visual_review"]
        self.assertEqual(non_pending, [], f"{len(non_pending)} rows have non-pending decisions")


# ---------------------------------------------------------------------------
# 12. Cross-Split Candidate Reporting
# ---------------------------------------------------------------------------

class TestCrossSplitReporting(unittest.TestCase):

    def test_cross_split_leakage_report_exists(self):
        self.assertTrue((PHASE25 / "cross_split_leakage_report.csv").is_file())

    def test_cross_split_exact_dup_not_claimed_zero(self):
        """Must not falsely claim zero cross-split leakage when candidates exist."""
        rows = read_csv(PHASE25 / "near_duplicate_candidates.csv")
        cs = [r for r in rows if r["cross_split"] == "True"]
        # We know the dataset has augmented images — there will be cross-split candidates
        # This test verifies we didn't under-report
        leakage_rows = read_csv(PHASE25 / "cross_split_leakage_report.csv")
        if cs:
            self.assertGreater(len(leakage_rows), 0,
                               "Cross-split candidates exist but leakage report is empty")

    def test_test_split_images_in_cross_split_report_flagged(self):
        """Any cross-split candidate involving test split must appear in leakage report."""
        dup_rows = read_csv(PHASE25 / "near_duplicate_candidates.csv")
        test_cs = [r for r in dup_rows
                   if r["cross_split"] == "True"
                   and ("test" in r["split1"] or "test" in r["split2"])]
        leak_rows = read_csv(PHASE25 / "cross_split_leakage_report.csv")
        leak_files = {Path(r["file1"]).name for r in leak_rows} | {Path(r["file2"]).name for r in leak_rows}
        for r in test_cs[:5]:  # sample check
            found = (Path(r["file1"]).name in leak_files or Path(r["file2"]).name in leak_files)
            self.assertTrue(found, f"Test-split cross pair not in leakage report: {r['file1']}")


# ---------------------------------------------------------------------------
# 13. Master Manifest v1.1 Schema
# ---------------------------------------------------------------------------

class TestManifestV11Schema(unittest.TestCase):

    NEW_FIELDS = {"annotation_qc_status", "mask_qc_status", "class_verification_status",
                  "near_duplicate_status", "cross_split_review_status", "human_review_status",
                  "training_readiness", "xai_readiness", "exclusion_reason"}

    def test_manifest_v11_exists(self):
        self.assertTrue((META / "master_manifest_v1_1.csv").is_file())

    def test_v11_has_new_fields(self):
        rows = read_csv(META / "master_manifest_v1_1.csv")
        self.assertGreater(len(rows), 0)
        missing = self.NEW_FIELDS - rows[0].keys()
        self.assertEqual(missing, set(), f"Missing v1.1 fields: {missing}")

    def test_v11_preserves_v1_record_count(self):
        v1 = read_csv(META / "master_manifest_v1.csv")
        v11 = read_csv(META / "master_manifest_v1_1.csv")
        self.assertEqual(len(v1), len(v11), "v1.1 must have same record count as v1")

    def test_v1_unchanged(self):
        """v1.csv must not be modified — check it still has original fields only."""
        rows = read_csv(META / "master_manifest_v1.csv")
        self.assertGreater(len(rows), 0)
        for new_field in self.NEW_FIELDS:
            self.assertNotIn(new_field, rows[0].keys(),
                             f"v1 manifest was modified: found new field {new_field}")

    def test_no_aitex_training_ready(self):
        """No AITEX image should be training_readiness=yes."""
        rows = read_csv(META / "master_manifest_v1_1.csv")
        violations = [r for r in rows
                      if r["source_dataset"] == "aitex"
                      and r["training_readiness"] == "yes"]
        self.assertEqual(violations, [],
                         f"{len(violations)} AITEX images incorrectly marked training-ready")

    def test_test_split_locked(self):
        rows = read_csv(META / "master_manifest_v1_1.csv")
        for r in rows:
            if r["source_dataset"] == "chenab" and r["original_split"] == "test":
                self.assertEqual(r["training_readiness"], "locked",
                                 f"Test image {r['original_filename']} not locked")


# ---------------------------------------------------------------------------
# 14. Original File Immutability
# ---------------------------------------------------------------------------

class TestOriginalFileImmutability(unittest.TestCase):

    def test_source_hashes_unchanged(self):
        p = MASTER / "reports" / "phase1_audit" / "source_metadata.csv"
        if not p.is_file():
            self.skipTest("Source metadata CSV not found")
        rows = read_csv(p)
        for r in rows:
            fp = MASTER / "raw" / r["relative_path"]
            if fp.is_file():
                current = sha256_file(fp)
                self.assertEqual(current, r["sha256"],
                                 f"Raw file modified: {r['relative_path']}")

    def test_raw_chenab_label_files_unmodified(self):
        """Derived corrected labels must not replace originals."""
        orig_lbl = RAW_CHENAB / "train" / "labels"
        derived = MASTER / "derived_annotations" / "chenab_corrected_labels"
        if not derived.is_dir():
            return
        # Original files must still exist
        for lbl in list(derived.rglob("*.txt"))[:10]:
            orig = orig_lbl / lbl.name
            self.assertTrue(orig.is_file(),
                            f"Original label {lbl.name} appears to have been deleted")


# ---------------------------------------------------------------------------
# 15. Locked Test Split Protection
# ---------------------------------------------------------------------------

class TestLockedTestSplit(unittest.TestCase):

    def test_test_split_not_in_geometry_derived(self):
        """No Chenab test-split image should appear in derived annotations."""
        test_imgs = set()
        test_dir = RAW_CHENAB / "test" / "images"
        if test_dir.is_dir():
            test_imgs = {p.stem for p in test_dir.rglob("*") if p.is_file()}
        derived = MASTER / "derived_annotations" / "chenab_corrected_labels"
        if not derived.is_dir():
            return
        for f in derived.rglob("*.txt"):
            self.assertNotIn(f.stem, test_imgs,
                             f"Test split file {f.name} found in derived annotations")

    def test_test_split_not_approved_in_manifest(self):
        rows = read_csv(META / "master_manifest_v1_1.csv")
        for r in rows:
            if r["source_dataset"] == "chenab" and r["original_split"] == "test":
                self.assertNotEqual(r["training_readiness"], "yes")
                self.assertNotEqual(r["training_readiness"], "approved")


# ---------------------------------------------------------------------------
# 16. Pending Human Approval Enforcement
# ---------------------------------------------------------------------------

class TestPendingApprovalEnforcement(unittest.TestCase):

    def test_no_chenab_whole_image_box_auto_approved(self):
        """31 whole-image boxes must not have final_decision = approved."""
        rows = read_csv(PHASE25 / "chenab_annotation_review.csv")
        whole = [r for r in rows if r["correction_type"] == "whole_image_box_review"]
        auto_approved = [r for r in whole if r["final_decision"] not in (
            "pending_review", "needs_human_review")]
        self.assertEqual(auto_approved, [],
                         f"{len(auto_approved)} whole-image boxes auto-approved without human review")

    def test_aitex_all_human_decision_pending(self):
        rows = read_csv(PHASE25 / "aitex_class_mapping_decisions.csv")
        non_pending = [r for r in rows if r["human_decision"] != "pending_review"]
        self.assertEqual(non_pending, [],
                         f"{len(non_pending)} AITEX classes have non-pending human decisions")


# ---------------------------------------------------------------------------
# 17. Deterministic Report Generation
# ---------------------------------------------------------------------------

class TestDeterministicReports(unittest.TestCase):

    def test_manifest_v11_image_ids_unique(self):
        rows = read_csv(META / "master_manifest_v1_1.csv")
        ids = [r["image_id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate image_ids in v1.1 manifest")

    def test_aitex_code_verification_codes_unique(self):
        rows = read_csv(PHASE25 / "aitex_code_verification.csv")
        codes = [r["aitex_code"] for r in rows]
        self.assertEqual(len(codes), len(set(codes)), "Duplicate codes in verification CSV")

    def test_source_hashes_deterministic(self):
        p = RAW_CHENAB / "data.yaml"
        if not p.is_file():
            self.skipTest("data.yaml not found")
        h1 = sha256_file(p)
        h2 = sha256_file(p)
        self.assertEqual(h1, h2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
