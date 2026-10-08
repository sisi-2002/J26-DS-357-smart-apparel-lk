"""Tests for Phase 5 — Training Augmentation, Dataset Validation & Version Freezing."""

import csv
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
RESEARCH = MASTER / "research"
V1_EXPORT = MASTER / "processed" / "chenab_detection_v1"
V1_TRAINING = MASTER / "processed" / "chenab_detection_training_v1"
V3 = RESEARCH / "chenab_xai_masks_v3"
PHASE3 = MASTER / "reports" / "phase3"
PHASE4 = MASTER / "reports" / "phase4"
PHASE5 = MASTER / "reports" / "phase5"
PREVIEW_DIR = RESEARCH / "augmentation_previews"

CLASS_NAMES = ["baekra", "color issues", "contamination", "cut",
               "gray stitch", "selvet", "stain"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def results():
    with open(PHASE5 / "phase5_results.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def version_manifest():
    with open(V1_TRAINING / "version_manifest.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fsm():
    return read_csv(PHASE3 / "final_split_manifest.csv")


# --- Phase 5.1: Integrity ---
class TestIntegrity:
    def test_integrity_csv_exists(self):
        assert (PHASE5 / "phase5_integrity_validation.csv").exists()

    def test_integrity_all_pass(self):
        rows = read_csv(PHASE5 / "phase5_integrity_validation.csv")
        for r in rows:
            assert r["status"] == "PASS", f"{r['check']}: {r.get('detail', '')}"

    def test_integrity_summary_exists(self):
        assert (PHASE5 / "phase5_integrity_summary.md").exists()


# --- Phase 5.2: Augmentation Policy ---
class TestAugmentationPolicy:
    def test_policy_json_exists(self):
        assert (V1_TRAINING / "augmentation" / "augmentation_policy.json").exists()

    def test_yaml_config_exists(self):
        assert (V1_TRAINING / "augmentation" / "yolo_training_augmentation_config.yaml").exists()

    def test_rationale_exists(self):
        assert (V1_TRAINING / "augmentation" / "augmentation_rationale.md").exists()

    def test_policy_seed(self):
        p = json.loads((V1_TRAINING / "augmentation" / "augmentation_policy.json").read_text("utf-8"))
        assert p["seed"] == 42

    def test_policy_mode(self):
        p = json.loads((V1_TRAINING / "augmentation" / "augmentation_policy.json").read_text("utf-8"))
        assert p["mode"] == "on_the_fly"

    def test_disabled_transforms(self):
        p = json.loads((V1_TRAINING / "augmentation" / "augmentation_policy.json").read_text("utf-8"))
        assert not p["transforms"]["shear"]["enabled"]
        assert not p["transforms"]["perspective"]["enabled"]
        assert not p["transforms"]["mixup"]["enabled"]
        assert not p["transforms"]["copy_paste"]["enabled"]

    def test_conservative_hsv(self):
        p = json.loads((V1_TRAINING / "augmentation" / "augmentation_policy.json").read_text("utf-8"))
        assert p["transforms"]["hsv_h"]["value"] <= 0.015
        assert p["transforms"]["hsv_s"]["value"] <= 0.3
        assert p["transforms"]["hsv_v"]["value"] <= 0.3

    def test_conservative_rotation(self):
        p = json.loads((V1_TRAINING / "augmentation" / "augmentation_policy.json").read_text("utf-8"))
        assert p["transforms"]["degrees"]["value"] <= 10.0

    def test_yaml_no_forbidden(self):
        text = (V1_TRAINING / "augmentation" / "yolo_training_augmentation_config.yaml").read_text("utf-8")
        assert "shear: 0.0" in text
        assert "perspective: 0.0" in text
        assert "mixup: 0.0" in text
        assert "copy_paste: 0.0" in text


# --- Phase 5.3: Augmentation Previews ---
class TestAugmentationPreviews:
    def test_preview_dir_exists(self):
        assert PREVIEW_DIR.is_dir()

    def test_preview_manifest_exists(self):
        assert (PREVIEW_DIR / "augmentation_preview_manifest.csv").exists()

    def test_preview_images_generated(self):
        previews = [f for f in os.listdir(PREVIEW_DIR) if f.endswith(".jpg")]
        assert len(previews) >= 10

    def test_preview_manifest_has_rows(self):
        rows = read_csv(PREVIEW_DIR / "augmentation_preview_manifest.csv")
        assert len(rows) >= 10

    def test_no_defects_dropped_all(self):
        rows = read_csv(PREVIEW_DIR / "augmentation_preview_manifest.csv")
        dropped = [r for r in rows if r["defect_dropped_all"] == "True"]
        assert len(dropped) == 0

    def test_preview_sources_from_train(self):
        rows = read_csv(PREVIEW_DIR / "augmentation_preview_manifest.csv")
        em = read_csv(V1_EXPORT / "export_manifest.csv")
        train_files = {r["export_filename"] for r in em if r["split"] == "train"}
        for r in rows:
            assert r["source_image"] in train_files

    def test_validation_summary_exists(self):
        assert (PREVIEW_DIR / "augmentation_validation_summary.md").exists()


# --- Phase 5.4: Augmentation QC ---
class TestAugmentationQC:
    def test_qc_csv_exists(self):
        assert (PHASE5 / "augmentation_qc_results.csv").exists()

    def test_qc_all_pass(self):
        rows = read_csv(PHASE5 / "augmentation_qc_results.csv")
        for r in rows:
            assert r["status"] == "PASS", f"{r['check']}: {r.get('detail', '')}"

    def test_leakage_report_exists(self):
        assert (PHASE5 / "augmentation_leakage_report.md").exists()


# --- Phase 5.5: Training Export ---
class TestTrainingExport:
    def test_training_dir_exists(self):
        assert V1_TRAINING.is_dir()

    def test_train_image_count(self):
        imgs = [f for f in os.listdir(V1_TRAINING / "train" / "images")
                if f.endswith((".jpg", ".png", ".jpeg"))]
        assert len(imgs) == 1900

    def test_valid_image_count(self):
        imgs = [f for f in os.listdir(V1_TRAINING / "valid" / "images")
                if f.endswith((".jpg", ".png", ".jpeg"))]
        assert len(imgs) == 377

    def test_test_image_count(self):
        imgs = [f for f in os.listdir(V1_TRAINING / "test" / "images")
                if f.endswith((".jpg", ".png", ".jpeg"))]
        assert len(imgs) == 378

    def test_train_label_count(self):
        lbls = [f for f in os.listdir(V1_TRAINING / "train" / "labels") if f.endswith(".txt")]
        assert len(lbls) == 1900

    def test_valid_label_count(self):
        lbls = [f for f in os.listdir(V1_TRAINING / "valid" / "labels") if f.endswith(".txt")]
        assert len(lbls) == 377

    def test_test_label_count(self):
        lbls = [f for f in os.listdir(V1_TRAINING / "test" / "labels") if f.endswith(".txt")]
        assert len(lbls) == 378

    def test_data_yaml_exists(self):
        assert (V1_TRAINING / "data.yaml").exists()

    def test_data_yaml_classes(self):
        text = (V1_TRAINING / "data.yaml").read_text("utf-8")
        assert "nc: 7" in text
        for name in CLASS_NAMES:
            assert name in text

    def test_data_yaml_paths(self):
        text = (V1_TRAINING / "data.yaml").read_text("utf-8")
        assert "train: train/images" in text
        assert "val: valid/images" in text
        assert "test: test/images" in text

    def test_train_manifest_exists(self):
        assert (V1_TRAINING / "train_manifest.csv").exists()

    def test_train_manifest_count(self):
        rows = read_csv(V1_TRAINING / "train_manifest.csv")
        assert len(rows) == 1900

    def test_split_manifest_exists(self):
        assert (V1_TRAINING / "split_manifest.csv").exists()

    def test_split_manifest_count(self):
        rows = read_csv(V1_TRAINING / "split_manifest.csv")
        assert len(rows) == 2655

    def test_version_manifest_exists(self):
        assert (V1_TRAINING / "version_manifest.json").exists()

    def test_version_id(self, version_manifest):
        assert version_manifest["version_id"] == "chenab_detection_training_v1"

    def test_file_hashes_exist(self):
        assert (V1_TRAINING / "file_hashes.csv").exists()
        rows = read_csv(V1_TRAINING / "file_hashes.csv")
        assert len(rows) == 5310

    def test_image_label_pairing(self):
        for split in ["train", "valid", "test"]:
            img_stems = {os.path.splitext(f)[0]
                         for f in os.listdir(V1_TRAINING / split / "images")
                         if f.endswith((".jpg", ".png", ".jpeg"))}
            lbl_stems = {os.path.splitext(f)[0]
                         for f in os.listdir(V1_TRAINING / split / "labels")
                         if f.endswith(".txt")}
            assert img_stems == lbl_stems, f"{split}: mismatch"


# --- Phase 5.5: Leakage & Safety ---
class TestLeakageSafety:
    def test_group_disjoint(self, fsm):
        from collections import defaultdict
        groups = defaultdict(set)
        for r in fsm:
            if r["assigned_split"] in ("train", "valid", "test"):
                groups[r["assigned_split"]].add(r["group_id"])
        for s1 in ("train", "valid", "test"):
            for s2 in ("train", "valid", "test"):
                if s1 < s2:
                    assert not (groups[s1] & groups[s2]), f"overlap {s1}/{s2}"

    def test_no_quarantined_in_export(self, fsm):
        q_ids = {r["image_id"] for r in fsm if r["assigned_split"] == "quarantined"}
        sm = read_csv(V1_TRAINING / "split_manifest.csv")
        export_ids = {r["image_id"] for r in sm}
        assert not (q_ids & export_ids)

    def test_v1_export_unchanged(self):
        em = read_csv(V1_EXPORT / "export_manifest.csv")
        assert len(em) == 2655

    def test_frozen_thresholds(self):
        t = json.loads((PHASE4 / "provisional_thresholds.json").read_text("utf-8"))
        assert abs(t["p33"] - (-0.4654902671397324)) < 1e-10
        assert abs(t["p67"] - 0.21788554420312284) < 1e-10


# --- Phase 4 Immutability ---
class TestPhase4Immutability:
    def test_v3_exists(self):
        assert V3.is_dir()

    def test_v3_manifest_count(self):
        rows = read_csv(V3 / "manifests" / "final_mask_manifest.csv")
        assert len(rows) == 133

    def test_v3_images_count(self):
        assert len(os.listdir(V3 / "images")) == 100

    def test_raw_not_modified(self):
        raw = MASTER / "raw"
        if raw.exists():
            for d in raw.rglob("*"):
                assert "phase5" not in str(d).lower()
                assert "training_v1" not in d.name.lower()


# --- Phase 5.6: Final Report ---
class TestFinalReport:
    def test_report_exists(self):
        assert (PHASE5 / "PHASE_5_FINAL_REPORT.md").exists()

    def test_results_json_exists(self):
        assert (PHASE5 / "phase5_results.json").exists()

    def test_status_pass(self, results):
        assert results["status"] == "PASS"

    def test_total_images(self, results):
        assert results["total_images"] == 2655

    def test_validation_zero_failures(self, results):
        assert results["validation_failures"] == 0

    def test_phase6_ready(self, results):
        assert results["phase6_ready"] is True

    def test_phase4_preserved(self, results):
        assert results["phase4_preserved"] is True

    def test_validation_csv_exists(self):
        assert (PHASE5 / "phase5_validation.csv").exists()

    def test_all_validation_pass(self):
        rows = read_csv(PHASE5 / "phase5_validation.csv")
        for r in rows:
            assert r["status"] == "PASS", f"{r['check']}: {r.get('detail', '')}"
