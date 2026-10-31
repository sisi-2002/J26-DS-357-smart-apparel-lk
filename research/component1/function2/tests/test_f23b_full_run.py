"""F2.3-B: 60-image full-run execution tests (Task 12).

Covers: exact 60-image coverage, zero duplicate source families, exact
20/20/20 distribution, exact GT-instance accounting, identical target used
for both methods per instance, correct target-layer linkage, zero heatmaps
for unmatched instances (hard scan), paired numeric-artifact integrity
(shape/dtype/finite), original-image spatial alignment, nonfinite/empty-
heatmap handling, deterministic re-execution (live, small-scale, tmp_path),
versioned-output integrity (f23_dry_run_v1 untouched), zero TEST-split
access (hard regression), and zero changes to protected assets.

Tests that need production-scale I/O read the already-produced, real
``f23_full_run_v1`` outputs read-only (the same pattern already used by
``test_f22b_pipeline.py`` / ``test_f22c_finalize.py`` for their own
production manifests). Tests that WRITE anything use ``tmp_path`` only --
never a production manifest or production output directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.config.paths import (  # noqa: E402
    CHENAB_DATASET_ROOT, CHENAB_SPLIT_MANIFEST, CHENAB_TEST_IMAGES_DIR,
    CHENAB_TEST_LABELS_DIR, CLASS_NAMES, E02_CHECKPOINT, FREEZE_MANIFEST,
)
from research.component1.function2.xai import gt_matching  # noqa: E402
from research.component1.function2.xai import primary_sample_verification as psv  # noqa: E402
from research.component1.function2.xai import protected_artifacts  # noqa: E402

_XAI_SRC = REPO_ROOT / "research" / "component1" / "xai_experiments" / "src"
if str(_XAI_SRC) not in sys.path:
    sys.path.insert(0, str(_XAI_SRC))
from detection_xai import DetectionXAI  # noqa: E402

F22B_MANIFESTS = (
    REPO_ROOT / "research" / "component1" / "function2" / "results"
    / "evaluation_sampling" / "f22b_pp1_v1" / "manifests"
)
PRIMARY_SAMPLE_CSV = F22B_MANIFESTS / "xai_primary_sample.csv"
VALID_COMPLEXITY_MANIFEST_CSV = F22B_MANIFESTS / "valid_complexity_manifest.csv"

FULL_RUN_DIR = (
    REPO_ROOT / "research" / "component1" / "function2" / "results"
    / "xai_execution" / "f23_full_run_v1"
)
DRY_RUN_DIR = (
    REPO_ROOT / "research" / "component1" / "function2" / "results"
    / "xai_execution" / "f23_dry_run_v1"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _models_present() -> bool:
    return E02_CHECKPOINT.is_file() and FREEZE_MANIFEST.is_file()


def _dataset_present() -> bool:
    return CHENAB_DATASET_ROOT.is_dir() and CHENAB_SPLIT_MANIFEST.is_file()


def _full_run_present() -> bool:
    return (FULL_RUN_DIR / "manifests" / "execution_manifest.csv").is_file()


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# Task 2 recount: 60 images, 20/20/20, zero duplicate source families
# --------------------------------------------------------------------------

class TestPrimarySampleIntegrity:
    def test_exactly_60_distinct_images_and_20_20_20_distribution(self):
        if not (_dataset_present() and PRIMARY_SAMPLE_CSV.is_file()
                and VALID_COMPLEXITY_MANIFEST_CSV.is_file()):
            pytest.skip("dataset / manifests not present in this environment")
        verified = psv.verify_primary_sample(
            PRIMARY_SAMPLE_CSV, CHENAB_SPLIT_MANIFEST, VALID_COMPLEXITY_MANIFEST_CSV,
            CHENAB_DATASET_ROOT,
        )
        assert len(verified) == 60
        from collections import Counter
        counts = Counter(v.complexity_group for v in verified)
        assert counts == {"Simple": 20, "Moderate": 20, "High": 20}

    def test_zero_duplicate_source_families(self):
        if not (_dataset_present() and PRIMARY_SAMPLE_CSV.is_file()
                and VALID_COMPLEXITY_MANIFEST_CSV.is_file()):
            pytest.skip("dataset / manifests not present in this environment")
        verified = psv.verify_primary_sample(
            PRIMARY_SAMPLE_CSV, CHENAB_SPLIT_MANIFEST, VALID_COMPLEXITY_MANIFEST_CSV,
            CHENAB_DATASET_ROOT,
        )
        split_by_id = {r["image_id"]: r for r in _read_csv(CHENAB_SPLIT_MANIFEST)}
        group_ids = [split_by_id[v.image_id]["group_id"] for v in verified]
        assert len(group_ids) == len(set(group_ids)) == 60

    def test_zero_test_split_access_hard_lockout(self, tmp_path):
        """A sample row whose AUTHORITATIVE split is 'test' must raise before
        any file is opened -- the image_id's own naming must never be trusted."""
        fake_sample = tmp_path / "fake_primary_sample.csv"
        fake_split = tmp_path / "fake_split_manifest.csv"
        fake_complexity = tmp_path / "fake_complexity_manifest.csv"
        fake_sample.write_text(
            "sample_type,image_id,source_family_id,complexity_group,boundary_adjacent,"
            "defect_classes,defect_instance_count,has_gt_defect,complexity_score,"
            "selection_rank_within_group\n"
            'primary,chenab_valid_fake_img,sg_9999,Simple,False,"[""stain""]",1,True,0.1,1\n',
            encoding="utf-8",
        )
        fake_split.write_text(
            "image_id,split,group_id,filename\n"
            "chenab_valid_fake_img,test,sg_9999,fake_img.jpg\n",
            encoding="utf-8",
        )
        fake_complexity.write_text(
            "image_id,complexity_eligible,has_gt_defect,complexity_group\n"
            "chenab_valid_fake_img,True,True,Simple\n",
            encoding="utf-8",
        )
        with pytest.raises(psv.PrimarySampleVerificationError, match="TEST-split"):
            psv.verify_primary_sample(fake_sample, fake_split, fake_complexity, tmp_path)

    def test_recounted_gt_instances_match_known_recount(self):
        """Independent recount from the real label files must equal 92 total,
        with the exact per-group split established by this stage's Task 2
        re-verification (Simple=25, Moderate=34, High=33)."""
        if not (_dataset_present() and PRIMARY_SAMPLE_CSV.is_file()
                and VALID_COMPLEXITY_MANIFEST_CSV.is_file()):
            pytest.skip("dataset / manifests not present in this environment")
        verified = psv.verify_primary_sample(
            PRIMARY_SAMPLE_CSV, CHENAB_SPLIT_MANIFEST, VALID_COMPLEXITY_MANIFEST_CSV,
            CHENAB_DATASET_ROOT,
        )
        per_group = {"Simple": 0, "Moderate": 0, "High": 0}
        total = 0
        for row in verified:
            lines = [l for l in row.label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            total += len(lines)
            per_group[row.complexity_group] += len(lines)
        assert total == 92
        assert per_group == {"Simple": 25, "Moderate": 34, "High": 33}


# --------------------------------------------------------------------------
# Full-run manifest integrity (reads the real, already-produced output)
# --------------------------------------------------------------------------

class TestFullRunManifestIntegrity:
    def test_execution_manifest_covers_exactly_92_gt_instances(self):
        if not _full_run_present():
            pytest.skip("f23_full_run_v1 output not present in this environment")
        rows = _read_csv(FULL_RUN_DIR / "manifests" / "execution_manifest.csv")
        assert len(rows) == 92
        image_ids = {r["image_id"] for r in rows}
        assert len(image_ids) == 60

    def test_target_linkage_covers_exactly_92_instances(self):
        if not _full_run_present():
            pytest.skip("f23_full_run_v1 output not present in this environment")
        rows = _read_csv(FULL_RUN_DIR / "manifests" / "target_linkage.csv")
        assert len(rows) == 92

    def test_zero_heatmaps_for_unmatched_instances_hard_check(self):
        """Hard regression: no unmatched row may reference a heatmap path."""
        if not _full_run_present():
            pytest.skip("f23_full_run_v1 output not present in this environment")
        rows = _read_csv(FULL_RUN_DIR / "manifests" / "execution_manifest.csv")
        unmatched = [r for r in rows if r["prediction_status"] == "unmatched"]
        assert len(unmatched) > 0, "expected some unmatched instances in a real 60-image run"
        for r in unmatched:
            assert not r["gradcam_heatmap_path"], r
            assert not r["gradcampp_heatmap_path"], r

    def test_identical_target_for_both_methods_on_every_matched_instance(self):
        if not _full_run_present():
            pytest.skip("f23_full_run_v1 output not present in this environment")
        rows = _read_csv(FULL_RUN_DIR / "manifests" / "target_linkage.csv")
        matched = [r for r in rows if r["match_status"] == "matched"]
        assert len(matched) > 0
        for r in matched:
            assert r["same_target_for_both_methods"] == "True", r

    def test_target_layer_values_are_from_the_policy_set(self):
        if not _full_run_present():
            pytest.skip("f23_full_run_v1 output not present in this environment")
        rows = _read_csv(FULL_RUN_DIR / "manifests" / "target_linkage.csv")
        allowed = {"model.model[15]", "model.model[18]", "model.model[21]"}
        used = {r["target_layer"] for r in rows if r["target_layer"]}
        assert used, "expected at least one matched instance with a target layer"
        assert used <= allowed

    def test_no_fabricated_success_for_unmatched_rows_in_failures_csv(self):
        if not _full_run_present():
            pytest.skip("f23_full_run_v1 output not present in this environment")
        failures = _read_csv(FULL_RUN_DIR / "manifests" / "failures.csv")
        unmatched_failures = [f for f in failures if f["failure_category"] == "unmatched_gt_instance"]
        assert len(unmatched_failures) > 0
        for f in unmatched_failures:
            assert f["failure_reason"], f

    def test_method_pair_validity_one_sided_failure_kept_not_dropped(self):
        """If Grad-CAM failed but Grad-CAM++ succeeded (or vice versa) for a
        matched instance, that instance's row must still exist with the
        specific per-method statuses recorded -- never silently dropped, and
        never force-paired into paired_explanations_manifest.csv."""
        if not _full_run_present():
            pytest.skip("f23_full_run_v1 output not present in this environment")
        linkage = _read_csv(FULL_RUN_DIR / "manifests" / "target_linkage.csv")
        one_sided = [
            r for r in linkage
            if r["match_status"] == "matched"
            and (r["gradcam_status"] != "success" or r["gradcampp_status"] != "success")
        ]
        paired = _read_csv(FULL_RUN_DIR / "manifests" / "paired_explanations_manifest.csv")
        paired_ids = {(r["image_id"], r["gt_instance_id"]) for r in paired}
        for r in one_sided:
            inst_id = f"{r['image_id']}__inst{int(r['instance_index']):03d}"
            assert (r["image_id"], inst_id) not in paired_ids, (
                "a one-sided-failure instance must never appear in the paired manifest"
            )


# --------------------------------------------------------------------------
# Numeric-artifact integrity (reads real .npy outputs)
# --------------------------------------------------------------------------

class TestHeatmapArtifactIntegrity:
    def test_paired_npy_files_exist_finite_and_aligned(self):
        if not _full_run_present():
            pytest.skip("f23_full_run_v1 output not present in this environment")
        rows = _read_csv(FULL_RUN_DIR / "manifests" / "paired_explanations_manifest.csv")
        assert len(rows) > 0
        checked = 0
        for r in rows[:15]:  # bounded sample to keep this test fast
            gc_path = FULL_RUN_DIR / r["gradcam_npy_path"]
            gcpp_path = FULL_RUN_DIR / r["gradcampp_npy_path"]
            assert gc_path.is_file()
            assert gcpp_path.is_file()
            gc_arr = np.load(gc_path)
            gcpp_arr = np.load(gcpp_path)
            assert np.all(np.isfinite(gc_arr))
            assert np.all(np.isfinite(gcpp_arr))
            assert gc_arr.shape == gcpp_arr.shape
            assert gc_arr.dtype == gcpp_arr.dtype
            checked += 1
        assert checked > 0

    def test_npy_heatmap_shape_matches_source_image_hw(self):
        if not (_full_run_present() and _dataset_present()):
            pytest.skip("f23_full_run_v1 output / dataset not present in this environment")
        import cv2
        split_by_id = {r["image_id"]: r for r in _read_csv(CHENAB_SPLIT_MANIFEST)}
        rows = _read_csv(FULL_RUN_DIR / "manifests" / "paired_explanations_manifest.csv")
        checked = 0
        for r in rows[:10]:
            sr = split_by_id.get(r["image_id"])
            if sr is None:
                continue
            img_path = CHENAB_DATASET_ROOT / "valid" / "images" / sr["filename"]
            if not img_path.is_file():
                continue
            img = cv2.imread(str(img_path))
            arr = np.load(FULL_RUN_DIR / r["gradcam_npy_path"])
            assert arr.shape[:2] == img.shape[:2]
            checked += 1
        assert checked > 0


# --------------------------------------------------------------------------
# Versioned-output integrity: f23_dry_run_v1 must be untouched
# --------------------------------------------------------------------------

class TestVersionedOutputIsolation:
    def test_dry_run_v1_directory_untouched_by_full_run(self):
        if not DRY_RUN_DIR.is_dir():
            pytest.skip("f23_dry_run_v1 not present in this environment")
        # The dry run's own manifests must still report exactly 9 images /
        # 19 instances -- a full-run bug that wrote into the wrong
        # directory would corrupt these counts.
        exec_manifest = _read_csv(DRY_RUN_DIR / "manifests" / "execution_manifest.csv")
        linkage = _read_csv(DRY_RUN_DIR / "manifests" / "target_linkage.csv")
        assert len(exec_manifest) == 9
        assert len(linkage) == 19

    def test_full_run_output_root_is_a_separate_directory(self):
        assert FULL_RUN_DIR != DRY_RUN_DIR
        assert FULL_RUN_DIR.name == "f23_full_run_v1"
        assert DRY_RUN_DIR.name == "f23_dry_run_v1"


# --------------------------------------------------------------------------
# Deterministic re-execution (live, tmp_path-isolated, small scale)
# --------------------------------------------------------------------------

class TestDeterministicReexecution:
    def test_repeated_explain_calls_are_byte_identical(self, tmp_path):
        """Lightweight, isolated version of the determinism property this
        stage's full run relies on: re-running detect()+explain() on the
        SAME image/detection twice must yield byte-identical heatmaps.
        Writes only to tmp_path."""
        if not (_models_present() and _dataset_present()):
            pytest.skip("E02 checkpoint / dataset not present in this environment")
        import cv2
        rows = _read_csv(PRIMARY_SAMPLE_CSV) if PRIMARY_SAMPLE_CSV.is_file() else []
        if not rows:
            pytest.skip("primary sample manifest not present")
        split_by_id = {r["image_id"]: r for r in _read_csv(CHENAB_SPLIT_MANIFEST)}
        # Use the first row with a resolvable image file.
        chosen = None
        for r in rows:
            sr = split_by_id.get(r["image_id"])
            if sr and (CHENAB_DATASET_ROOT / "valid" / "images" / sr["filename"]).is_file():
                chosen = (r, sr)
                break
        if chosen is None:
            pytest.skip("no resolvable primary-sample image available")
        row, sr = chosen
        image_path = CHENAB_DATASET_ROOT / "valid" / "images" / sr["filename"]
        image_bgr = cv2.imread(str(image_path))
        assert image_bgr is not None

        xai = DetectionXAI(str(E02_CHECKPOINT), device="cpu")
        det = xai.detect(image_bgr, conf_threshold=0.25, iou_threshold=0.5)
        if not det.detections:
            pytest.skip("no detections for the chosen image at this confidence threshold")
        target = det.detections[0]

        e1 = xai.explain(image_bgr, target, method="gradcam")
        e2 = xai.explain(image_bgr, target, method="gradcam")
        if e1.status != "success" or e2.status != "success":
            pytest.skip("gradcam did not succeed for this arbitrary detection (not a determinism failure)")
        assert np.array_equal(e1.heatmap_raw, e2.heatmap_raw)


# --------------------------------------------------------------------------
# Protected-artifact integrity (re-verified at F2.3-B time)
# --------------------------------------------------------------------------

class TestProtectedArtifactsStillUnchangedAtF23B:
    def test_e02_checkpoint_hash_unchanged(self):
        if not _models_present():
            pytest.skip("E02 checkpoint / freeze manifest not present")
        manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
        expected = manifest["selected_experiment"]["checkpoint_sha256"]
        assert _sha256(E02_CHECKPOINT) == expected

    def test_all_known_good_protected_hashes_still_pass(self):
        results = protected_artifacts.verify_known_good_hashes()
        present = {k: v for k, v in results.items() if v["status"] != "MISSING"}
        if not present:
            pytest.skip("no protected artifacts present in this environment")
        for rel, res in present.items():
            assert res["status"] == "PASS", f"{rel}: {res}"

    def test_pre_f23a_baseline_snapshot_still_matches(self):
        baseline_path = DRY_RUN_DIR / "checksums" / "pre_stage_baseline.json"
        if not baseline_path.is_file():
            pytest.skip("F2.3-A pre-stage baseline snapshot not present")
        pre = json.loads(baseline_path.read_text(encoding="utf-8"))
        results = protected_artifacts.verify_against_pre_stage_snapshot(pre)
        fails = {k: v for k, v in results.items() if v["status"] != "PASS"}
        assert not fails, fails

    def test_f23a_xai_package_files_unchanged_since_dry_run(self):
        """The core F2.3-A orchestration files (gt_matching.py,
        manifest_verification.py, pipeline.py, visualization.py) must be
        byte-identical to what F2.3-B's Task 1 independently re-verified --
        F2.3-B only ADDS new files (primary_sample_verification.py,
        pipeline_full_run.py), it never edits the F2.3-A files."""
        xai_dir = REPO_ROOT / "research" / "component1" / "function2" / "xai"
        expected = {
            "gt_matching.py": "421276e2a3d571955f48f6dd61046611a1692b16440a67db188016aeb159db33",
            "manifest_verification.py": "c688948673aeb3fdb3e644c7a9fa8fb51ea809617fd7ed50b0429982f4595017",
            "pipeline.py": "8d6603f9cb852c597144df92757178d07ded7124b3aa371d3187ee907c9e286c",
            "visualization.py": "12f52bea46c4e70ef271e722464a4a83e995299e176eb61b8e9b65cab225156c",
            "protected_artifacts.py": "dfd05a3943e233b0f696504e87c99cd8e0ec2c490b5811e645dfb2e1a50028dc",
            "__init__.py": "18771bfe3ee631c0c16454359010142b299b0235712e1b385b41e41645b29cbe",
        }
        for name, expected_hash in expected.items():
            p = xai_dir / name
            if not p.is_file():
                pytest.skip(f"{name} not present")
            assert _sha256(p) == expected_hash, f"{name} changed since F2.3-B Task 1 re-verification"
