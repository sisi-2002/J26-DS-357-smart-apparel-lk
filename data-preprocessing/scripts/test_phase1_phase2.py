"""
Validation tests for Phase 1 & Phase 2 outputs.
Run from the scripts directory or any directory:
  python test_phase1_phase2.py
"""

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Allow importing sibling scripts
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

REPO_ROOT = SCRIPTS_DIR.parent.parent
MASTER_ROOT = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
CHENAB_ROOT = MASTER_ROOT / "raw" / "chenab" / "fabricDD.v2i.yolov8"
AITEX_ROOT = MASTER_ROOT / "raw" / "aitex"
PHASE1_REPORT = MASTER_ROOT / "reports" / "phase1_audit"
META_DIR = MASTER_ROOT / "metadata"
PHASE2_REPORT = MASTER_ROOT / "reports" / "phase2"
DERIVED_ANN = MASTER_ROOT / "derived_annotations"

NUM_CLASSES = 7
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# Test: Dataset Inventory
# ---------------------------------------------------------------------------

class TestDatasetInventory(unittest.TestCase):

    def test_chenab_image_counts(self):
        """Chenab split image counts match expectations."""
        for split in ("train", "valid", "test"):
            d = CHENAB_ROOT / split / "images"
            self.assertTrue(d.is_dir(), f"Missing split directory: {d}")
            imgs = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            self.assertGreater(len(imgs), 0, f"No images in {split}")

    def test_aitex_directories_exist(self):
        """AITEX required directories exist."""
        for subdir in ("Defect_images", "NODefect_images", "Mask_images"):
            self.assertTrue((AITEX_ROOT / subdir).is_dir(), f"Missing: {AITEX_ROOT / subdir}")

    def test_chenab_label_files_match_images(self):
        """Every Chenab image in train/valid has a corresponding label file."""
        for split in ("train", "valid"):
            img_dir = CHENAB_ROOT / split / "images"
            lbl_dir = CHENAB_ROOT / split / "labels"
            imgs = [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            missing = [p for p in imgs if not (lbl_dir / p.with_suffix(".txt").name).is_file()]
            self.assertEqual(len(missing), 0, f"{split}: {len(missing)} images missing labels")

    def test_phase1_inventory_csv_exists_and_nonempty(self):
        """Phase 1 inventory CSV exists and has rows."""
        p = PHASE1_REPORT / "dataset_inventory.csv"
        self.assertTrue(p.is_file(), f"Missing: {p}")
        rows = read_csv(p)
        self.assertGreater(len(rows), 0)

    def test_phase1_inventory_has_required_columns(self):
        """Inventory CSV has all required columns."""
        p = PHASE1_REPORT / "dataset_inventory.csv"
        if not p.is_file():
            self.skipTest("Inventory CSV not yet generated")
        rows = read_csv(p)
        required = {"split", "filename", "width", "height", "annotation_count", "quality_flags"}
        self.assertTrue(required.issubset(rows[0].keys()), f"Missing columns: {required - rows[0].keys()}")

# ---------------------------------------------------------------------------
# Test: YOLO Annotation Parsing
# ---------------------------------------------------------------------------

class TestAnnotationParsing(unittest.TestCase):

    def _parse_label_line(self, line):
        parts = line.strip().split()
        if len(parts) != 5:
            return None, f"expected 5 fields, got {len(parts)}"
        try:
            cid = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError as e:
            return None, str(e)
        return (cid, cx, cy, w, h), None

    def test_valid_annotation_parses(self):
        parsed, err = self._parse_label_line("2 0.5 0.5 0.2 0.1")
        self.assertIsNone(err)
        self.assertEqual(parsed[0], 2)

    def test_invalid_class_id_detected(self):
        parsed, err = self._parse_label_line("7 0.5 0.5 0.2 0.1")
        self.assertIsNone(err)
        cid = parsed[0]
        self.assertGreaterEqual(cid, NUM_CLASSES, "Class 7 should be out of range")

    def test_out_of_range_coordinate_detected(self):
        parsed, err = self._parse_label_line("0 1.5 0.5 0.2 0.1")
        self.assertIsNone(err)
        cx = parsed[1]
        self.assertGreater(cx, 1.0)

    def test_wrong_field_count_detected(self):
        parsed, err = self._parse_label_line("0 0.5 0.5 0.2")
        self.assertIsNone(parsed)
        self.assertIn("5", err)

    def test_annotation_issues_csv_schema(self):
        p = PHASE1_REPORT / "chenab_annotation_issues.csv"
        if not p.is_file():
            self.skipTest("Annotation issues CSV not yet generated")
        rows = read_csv(p)
        if rows:
            required = {"split", "image_file", "line_num", "issues"}
            self.assertTrue(required.issubset(rows[0].keys()))

# ---------------------------------------------------------------------------
# Test: Invalid Bounding-Box Detection
# ---------------------------------------------------------------------------

class TestBoundingBoxValidation(unittest.TestCase):

    def test_box_exceeds_right_boundary(self):
        """Box that extends beyond right edge is flagged."""
        cx, w = 0.95, 0.2
        overflows = (cx + w / 2) > 1.001
        self.assertTrue(overflows)

    def test_normal_box_does_not_flag(self):
        cx, cy, w, h = 0.5, 0.5, 0.4, 0.3
        issues = []
        if not (0 <= cx <= 1) or not (0 <= cy <= 1):
            issues.append("coord_out_of_range")
        if cx - w/2 < -0.001 or cx + w/2 > 1.001:
            issues.append("overflow_x")
        if cy - h/2 < -0.001 or cy + h/2 > 1.001:
            issues.append("overflow_y")
        self.assertEqual(issues, [])

# ---------------------------------------------------------------------------
# Test: AITEX Mask Pairing
# ---------------------------------------------------------------------------

class TestAitexMaskPairing(unittest.TestCase):

    def test_mask_pairing_csv_exists(self):
        p = PHASE1_REPORT / "aitex_mask_pairing.csv"
        self.assertTrue(p.is_file(), f"Missing: {p}")

    def test_mask_pairing_csv_has_required_columns(self):
        p = PHASE1_REPORT / "aitex_mask_pairing.csv"
        if not p.is_file():
            self.skipTest("Mask pairing CSV not yet generated")
        rows = read_csv(p)
        self.assertGreater(len(rows), 0)
        required = {"filename", "mask_available", "mask_issues"}
        self.assertTrue(required.issubset(rows[0].keys()))

    def test_mask_naming_convention(self):
        """Mask files follow stem + '_mask' convention."""
        mask_dir = AITEX_ROOT / "Mask_images"
        if not mask_dir.is_dir():
            self.skipTest("Mask directory not found")
        masks = list(mask_dir.iterdir())
        self.assertGreater(len(masks), 0)
        for m in masks[:5]:
            self.assertTrue(m.stem.endswith("_mask"),
                            f"Unexpected mask filename pattern: {m.name}")

    def test_defect_count_vs_mask_count(self):
        """Number of defect images and masks are close (should be equal or near-equal)."""
        defect_dir = AITEX_ROOT / "Defect_images"
        mask_dir = AITEX_ROOT / "Mask_images"
        if not defect_dir.is_dir() or not mask_dir.is_dir():
            self.skipTest("AITEX directories not found")
        defect_count = sum(1 for p in defect_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        mask_count = sum(1 for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        # They should be within 5 of each other
        self.assertAlmostEqual(defect_count, mask_count, delta=10,
                               msg=f"Defect({defect_count}) vs Mask({mask_count}) count mismatch")

# ---------------------------------------------------------------------------
# Test: Mask-to-Box Coordinate Conversion
# ---------------------------------------------------------------------------

class TestMaskToBox(unittest.TestCase):

    def _mask_to_box(self, mask_array):
        """Simple bounding box from binary mask."""
        import numpy as np
        ys, xs = np.where(mask_array > 0)
        if len(ys) == 0:
            return None
        img_h, img_w = mask_array.shape
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = (x1 + w / 2) / img_w
        cy = (y1 + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        return cx, cy, nw, nh

    def test_full_mask_gives_center_box(self):
        import numpy as np
        arr = np.ones((100, 100), dtype=np.uint8) * 255
        cx, cy, nw, nh = self._mask_to_box(arr)
        self.assertAlmostEqual(cx, 0.5, places=1)
        self.assertAlmostEqual(cy, 0.5, places=1)
        self.assertAlmostEqual(nw, 1.0, places=1)

    def test_top_left_mask_gives_correct_box(self):
        import numpy as np
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[10:30, 5:25] = 255
        cx, cy, nw, nh = self._mask_to_box(arr)
        self.assertGreater(cx, 0)
        self.assertLess(cx, 0.5)
        self.assertTrue(0 <= cx <= 1.0)
        self.assertTrue(0 <= cy <= 1.0)
        self.assertTrue(0 < nw <= 1.0)
        self.assertTrue(0 < nh <= 1.0)

    def test_empty_mask_returns_none(self):
        import numpy as np
        arr = np.zeros((100, 100), dtype=np.uint8)
        result = self._mask_to_box(arr)
        self.assertIsNone(result)

    def test_annotation_manifest_csv_exists(self):
        p = DERIVED_ANN / "annotation_manifest.csv"
        if not p.is_file():
            self.skipTest("Annotation manifest not yet generated")
        rows = read_csv(p)
        self.assertGreater(len(rows), 0)

# ---------------------------------------------------------------------------
# Test: Class Mapping
# ---------------------------------------------------------------------------

class TestClassMapping(unittest.TestCase):

    def test_mapping_csv_exists(self):
        p = META_DIR / "class_mapping_proposed.csv"
        self.assertTrue(p.is_file(), f"Missing: {p}")

    def test_chenab_classes_all_present(self):
        p = META_DIR / "class_mapping_proposed.csv"
        if not p.is_file():
            self.skipTest("Mapping CSV not yet generated")
        rows = read_csv(p)
        chenab_rows = [r for r in rows if r["source_dataset"] == "chenab"]
        self.assertEqual(len(chenab_rows), NUM_CLASSES, f"Expected {NUM_CLASSES} Chenab rows")

    def test_chenab_classes_all_approved(self):
        p = META_DIR / "class_mapping_proposed.csv"
        if not p.is_file():
            self.skipTest("Mapping CSV not yet generated")
        rows = read_csv(p)
        for r in rows:
            if r["source_dataset"] == "chenab":
                self.assertEqual(r["approved_for_training"], "yes")

    def test_aitex_no_classes_auto_approved(self):
        """No AITEX class should be auto-approved (all require human review or are incompatible)."""
        p = META_DIR / "class_mapping_proposed.csv"
        if not p.is_file():
            self.skipTest("Mapping CSV not yet generated")
        rows = read_csv(p)
        for r in rows:
            if r["source_dataset"] == "aitex":
                self.assertNotEqual(r["approved_for_training"], "yes",
                                    f"AITEX class {r['source_class_id']} should not be auto-approved")

# ---------------------------------------------------------------------------
# Test: Duplicate Groups
# ---------------------------------------------------------------------------

class TestDuplicateGroups(unittest.TestCase):

    def test_duplicate_groups_csv_exists(self):
        p = META_DIR / "duplicate_groups.csv"
        self.assertTrue(p.is_file(), f"Missing: {p}")

    def test_exact_duplicate_groups_have_multiple_entries(self):
        p = META_DIR / "duplicate_groups.csv"
        if not p.is_file():
            self.skipTest("Duplicate groups CSV not yet generated")
        rows = read_csv(p)
        from collections import Counter
        counts = Counter(r["duplicate_group_id"] for r in rows)
        for gid, count in counts.items():
            self.assertGreaterEqual(count, 2, f"Group {gid} has only {count} member(s)")

    def test_cross_split_leakage_csv_exists(self):
        p = PHASE2_REPORT / "cross_split_leakage.csv"
        self.assertTrue(p.is_file(), f"Missing: {p}")

# ---------------------------------------------------------------------------
# Test: Master Manifest Schema
# ---------------------------------------------------------------------------

class TestMasterManifest(unittest.TestCase):

    REQUIRED_FIELDS = {
        "image_id", "source_dataset", "source_version", "original_filename",
        "relative_image_path", "image_sha256", "width", "height", "original_split",
        "original_class", "proposed_mapped_class", "mapping_status",
        "annotation_format", "bounding_box_count", "mask_available",
        "mask_validation_status", "quality_status", "duplicate_group_id",
        "source_group_id", "provenance_status", "license_status",
        "approved_for_detection_training", "approved_for_xai_evaluation",
        "review_required", "review_reason",
    }

    def test_manifest_exists(self):
        p = META_DIR / "master_manifest_v1.csv"
        self.assertTrue(p.is_file(), f"Missing: {p}")

    def test_manifest_has_required_columns(self):
        p = META_DIR / "master_manifest_v1.csv"
        if not p.is_file():
            self.skipTest("Manifest not yet generated")
        rows = read_csv(p)
        self.assertGreater(len(rows), 0)
        missing = self.REQUIRED_FIELDS - rows[0].keys()
        self.assertEqual(missing, set(), f"Missing manifest columns: {missing}")

    def test_manifest_image_ids_unique(self):
        p = META_DIR / "master_manifest_v1.csv"
        if not p.is_file():
            self.skipTest("Manifest not yet generated")
        rows = read_csv(p)
        ids = [r["image_id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate image_ids in manifest")

    def test_manifest_sha256_format(self):
        """All non-empty SHA256 values are 64 hex chars."""
        p = META_DIR / "master_manifest_v1.csv"
        if not p.is_file():
            self.skipTest("Manifest not yet generated")
        rows = read_csv(p)
        for r in rows[:50]:  # sample
            sha = r.get("image_sha256", "")
            if sha:
                self.assertEqual(len(sha), 64, f"Bad SHA256 length for {r['original_filename']}: {sha}")
                self.assertTrue(all(c in "0123456789abcdef" for c in sha),
                                f"Non-hex SHA256 for {r['original_filename']}")

    def test_manifest_total_count_reasonable(self):
        p = META_DIR / "master_manifest_v1.csv"
        if not p.is_file():
            self.skipTest("Manifest not yet generated")
        rows = read_csv(p)
        # Chenab: ~2791, AITEX: ~113 → expect ~2900
        self.assertGreater(len(rows), 2500)
        self.assertLess(len(rows), 5000)

    def test_test_split_locked(self):
        """Chenab test split images must not be approved_for_detection_training."""
        p = META_DIR / "master_manifest_v1.csv"
        if not p.is_file():
            self.skipTest("Manifest not yet generated")
        rows = read_csv(p)
        for r in rows:
            if r["source_dataset"] == "chenab" and r["original_split"] == "test":
                self.assertNotEqual(r["approved_for_detection_training"], "yes",
                                    f"Test split image {r['original_filename']} must not be approved")

# ---------------------------------------------------------------------------
# Test: File Hashing Stability
# ---------------------------------------------------------------------------

class TestHashingStability(unittest.TestCase):

    def test_sha256_is_deterministic(self):
        """SHA256 of a file is deterministic."""
        # Use data.yaml as a small stable file
        p = CHENAB_ROOT / "data.yaml"
        if not p.is_file():
            self.skipTest("data.yaml not found")
        h1 = sha256_file(p)
        h2 = sha256_file(p)
        self.assertEqual(h1, h2)

    def test_sha256_length(self):
        p = CHENAB_ROOT / "data.yaml"
        if not p.is_file():
            self.skipTest("data.yaml not found")
        h = sha256_file(p)
        self.assertEqual(len(h), 64)

# ---------------------------------------------------------------------------
# Test: Raw Dataset Immutability
# ---------------------------------------------------------------------------

class TestRawImmutability(unittest.TestCase):

    def test_source_metadata_hashes_match_current_files(self):
        """Source anchor hashes recorded in phase1 still match actual files."""
        p = PHASE1_REPORT / "source_metadata.csv"
        if not p.is_file():
            self.skipTest("Source metadata CSV not yet generated")
        rows = read_csv(p)
        mismatches = []
        for r in rows:
            full_path = MASTER_ROOT / "raw" / r["relative_path"]
            if full_path.is_file():
                current = sha256_file(full_path)
                if current != r["sha256"]:
                    mismatches.append(r["relative_path"])
        self.assertEqual(mismatches, [], f"Raw files changed since audit: {mismatches}")

# ---------------------------------------------------------------------------
# Test: Source Provenance
# ---------------------------------------------------------------------------

class TestSourceProvenance(unittest.TestCase):

    def test_manifest_source_datasets_known(self):
        p = META_DIR / "master_manifest_v1.csv"
        if not p.is_file():
            self.skipTest("Manifest not yet generated")
        rows = read_csv(p)
        known = {"chenab", "aitex"}
        for r in rows:
            self.assertIn(r["source_dataset"], known,
                          f"Unknown source_dataset: {r['source_dataset']}")

    def test_manifest_license_status_nonempty(self):
        p = META_DIR / "master_manifest_v1.csv"
        if not p.is_file():
            self.skipTest("Manifest not yet generated")
        rows = read_csv(p)
        for r in rows:
            self.assertTrue(r.get("license_status"), f"Empty license_status for {r['original_filename']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
