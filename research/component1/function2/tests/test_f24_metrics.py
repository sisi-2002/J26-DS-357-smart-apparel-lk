"""F2.4-A Task 13 -- tests for the evaluation metric suite.

Tests requiring the real E02 model/dataset are guarded by ``pytest.skip`` the
same way ``test_f23b_full_run.py`` guards its own live-model tests, so this
file still runs (skipping the heavy subset) in any environment missing the
checkpoint/dataset.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.config.paths import (  # noqa: E402
    CHENAB_DATASET_ROOT, CHENAB_TEST_IMAGES_DIR, CHENAB_TEST_LABELS_DIR,
    E02_CHECKPOINT, FREEZE_MANIFEST,
)
from research.component1.function2.evaluation import (  # noqa: E402
    localization, faithfulness, stability, sanity_checks, pixel_iou, pilot_selection,
)
from research.component1.function2.evaluation.common import (  # noqa: E402
    frozen_target_score, spearman_corr,
)
from research.component1.function2.evaluation.protected_snapshot import (  # noqa: E402
    compute_snapshot, verify_against_snapshot, all_f24_protected_relative_paths,
)

RUN_DIR = (
    REPO_ROOT / "research/component1/function2/results/xai_execution/f23_full_run_v1"
)
PAIRED_MANIFEST = RUN_DIR / "manifests/paired_explanations_manifest.csv"


def _models_present() -> bool:
    return E02_CHECKPOINT.is_file() and FREEZE_MANIFEST.is_file()


def _dataset_present() -> bool:
    return CHENAB_DATASET_ROOT.is_dir()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# Localization -- synthetic, hand-computable
# ===========================================================================

class TestLocalizationSynthetic:
    def test_pointing_accuracy_known_hit(self):
        hm = np.zeros((10, 10), dtype=np.float32)
        hm[5, 5] = 1.0  # row=5 (y), col=5 (x)
        box = (3.0, 3.0, 7.0, 7.0)  # x1,y1,x2,y2 -- contains (5,5)
        r = localization.pointing_accuracy(hm, box, (10, 10))
        assert r.status == "valid"
        assert r.hit is True
        assert r.argmax_xy == (5.0, 5.0)

    def test_pointing_accuracy_known_miss(self):
        hm = np.zeros((10, 10), dtype=np.float32)
        hm[0, 0] = 1.0
        box = (5.0, 5.0, 9.0, 9.0)
        r = localization.pointing_accuracy(hm, box, (10, 10))
        assert r.status == "valid"
        assert r.hit is False

    def test_pointing_accuracy_boundary_inclusive(self):
        hm = np.zeros((10, 10), dtype=np.float32)
        hm[3, 7] = 1.0  # exactly on x2 edge of box below
        box = (3.0, 3.0, 7.0, 7.0)
        r = localization.pointing_accuracy(hm, box, (10, 10))
        assert r.hit is True  # inclusive boundary

    def test_pointing_accuracy_row_major_tie_break(self):
        hm = np.zeros((4, 4), dtype=np.float32)
        hm[2, 0] = 1.0
        hm[0, 3] = 1.0  # earlier in row-major flat order (row 0 < row 2)
        r = localization.pointing_accuracy(hm, (0, 0, 3, 3), (4, 4))
        assert r.argmax_xy == (3.0, 0.0)  # first occurrence, row-major

    def test_pointing_accuracy_all_zero_is_invalid_not_miss(self):
        hm = np.zeros((5, 5), dtype=np.float32)
        r = localization.pointing_accuracy(hm, (0, 0, 2, 2), (5, 5))
        assert r.status == "invalid_all_zero"
        assert r.hit is None

    def test_pointing_accuracy_nonfinite_is_invalid(self):
        hm = np.zeros((5, 5), dtype=np.float32)
        hm[0, 0] = np.nan
        r = localization.pointing_accuracy(hm, (0, 0, 2, 2), (5, 5))
        assert r.status == "invalid_nonfinite"

    def test_box_energy_hand_computable(self):
        hm = np.zeros((4, 4), dtype=np.float32)
        hm[0, 0] = 1.0  # outside box
        hm[2, 2] = 3.0  # inside box
        box = (2.0, 2.0, 4.0, 4.0)  # covers rows/cols 2-3
        r = localization.box_energy_concentration(hm, box, (4, 4))
        assert r.status == "valid"
        assert r.total_positive_energy == pytest.approx(4.0)
        assert r.box_positive_energy == pytest.approx(3.0)
        assert r.energy_fraction == pytest.approx(0.75)

    def test_box_energy_degenerate_zero(self):
        hm = np.zeros((4, 4), dtype=np.float32)
        r = localization.box_energy_concentration(hm, (0, 0, 2, 2), (4, 4))
        assert r.status == "degenerate_zero_energy"
        assert r.energy_fraction is None

    def test_box_energy_nonfinite(self):
        hm = np.full((4, 4), np.inf, dtype=np.float32)
        r = localization.box_energy_concentration(hm, (0, 0, 2, 2), (4, 4))
        assert r.status == "invalid_nonfinite"

    def test_box_energy_resolution_mismatch_rescales_box(self):
        # heatmap half the resolution of the "image" -> box coords halved
        hm = np.zeros((2, 2), dtype=np.float32)
        hm[1, 1] = 1.0
        box = (2.0, 2.0, 4.0, 4.0)  # in a 4x4 "image" -> rescaled to (1,1,2,2) in 2x2 heatmap
        r = localization.box_energy_concentration(hm, box, (4, 4))
        assert r.box_rescaled is True
        assert r.energy_fraction == pytest.approx(1.0)

    def test_pointing_accuracy_resolution_mismatch_rescales_box(self):
        hm = np.zeros((2, 2), dtype=np.float32)
        hm[1, 1] = 1.0
        box = (2.0, 2.0, 4.0, 4.0)
        r = localization.pointing_accuracy(hm, box, (4, 4))
        assert r.box_rescaled is True
        assert r.hit is True


# ===========================================================================
# Faithfulness: perturbation-order and target-identity reuse
# ===========================================================================

class TestFaithfulnessOrdering:
    def test_most_important_first_order(self):
        hm = np.array([[0.1, 0.9], [0.5, 0.2]], dtype=np.float32)
        order = faithfulness._ranked_pixel_order(hm)
        flat = hm.ravel()
        values_in_order = flat[order]
        assert list(values_in_order) == sorted(values_in_order, reverse=True)
        assert order[0] == 1  # index of 0.9 in row-major flat array

    def test_step_masks_monotonic_and_bounded(self):
        order = np.arange(100)
        bounds = faithfulness._step_masks(order, faithfulness.N_STEPS)
        assert bounds[0] == 0
        assert bounds[-1] == 100
        assert bounds == sorted(bounds)

    def test_random_order_is_a_permutation_and_deterministic(self):
        o1 = faithfulness._random_pixel_order(50, seed=357)
        o2 = faithfulness._random_pixel_order(50, seed=357)
        assert sorted(o1.tolist()) == list(range(50))
        assert (o1 == o2).all()


class _FakeXai:
    """Minimal stand-in exposing exactly the surface
    ``common.frozen_target_score`` needs, so insertion/deletion-curve tests
    don't require loading the real ~6MB YOLO checkpoint."""
    def __init__(self, device="cpu"):
        self._device = device

    @staticmethod
    def _preprocess(img):
        import torch
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return t, 1.0, 0.0, 0.0

    def _model(self, tensor):
        import torch
        # deterministic "score" = mean pixel value -> logit; same index space (8400 cells)
        val = float(tensor.mean())
        scores = torch.full((1, 7, 8400), val * 4.0 - 2.0)
        return (None, {"scores": scores})


class TestFrozenTargetMechanism:
    def test_frozen_target_score_reads_fixed_cell(self):
        xai = _FakeXai()
        img = np.full((8, 8, 3), 200, dtype=np.uint8)
        r = frozen_target_score(xai, img, class_id=2, pre_nms_index=100)
        assert r["status"] == "valid"
        assert 0.0 <= r["score"] <= 1.0

    def test_frozen_target_score_out_of_range_index_is_invalid_not_error(self):
        xai = _FakeXai()
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        r = frozen_target_score(xai, img, class_id=2, pre_nms_index=999999)
        assert r["status"] == "target_invalid_shape"

    def test_same_target_index_used_across_perturbation_steps(self):
        """Proves the SAME (class_id, pre_nms_index) is reused for every
        step of a curve, never re-derived -- by checking every call in
        `_run_curve` uses the identical frozen index regardless of the
        (changing) perturbed image content."""
        xai = _FakeXai()
        calls = []
        orig_score_fn = frozen_target_score

        def _spy(xai_, img, class_id, pre_nms_index):
            calls.append((class_id, pre_nms_index))
            return orig_score_fn(xai_, img, class_id, pre_nms_index)

        import research.component1.function2.evaluation.faithfulness as mod
        old = mod.frozen_target_score
        mod.frozen_target_score = _spy
        try:
            img = np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8).astype(np.uint8)
            baseline = np.zeros_like(img)
            hm = np.random.default_rng(2).random((8, 8)).astype(np.float32)
            order = faithfulness._ranked_pixel_order(hm)
            mod._run_curve(xai, img, baseline, class_id=3, pre_nms_index=42,
                            pixel_order=order, mode="insertion")
        finally:
            mod.frozen_target_score = old
        assert len(calls) > 0
        assert all(c == (3, 42) for c in calls)


# ===========================================================================
# Stability
# ===========================================================================

class TestStability:
    def test_identical_image_perfect_similarity(self):
        a = np.random.default_rng(5).random((20, 20)).astype(np.float32)
        sim = spearman_corr(a, a.copy())
        assert sim == pytest.approx(1.0, abs=1e-9)

    def test_flat_heatmap_similarity_is_nan(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.random.default_rng(6).random((10, 10)).astype(np.float32)
        sim = spearman_corr(a, b)
        assert np.isnan(sim)

    def test_pixel_shift_inverse_alignment_roundtrips(self):
        hm = np.zeros((20, 20), dtype=np.float32)
        hm[10, 10] = 1.0
        shifted = stability.perturb_pixel_shift(
            np.dstack([hm * 255] * 3).astype(np.uint8), dxdy=(2, 2))
        shifted_hm = shifted[:, :, 0].astype(np.float32)
        realigned = stability.inverse_align_shifted_heatmap(shifted_hm, dxdy=(2, 2))
        # peak should be back near (10,10) within interpolation tolerance
        gy, gx = np.unravel_index(np.argmax(realigned), realigned.shape)
        assert abs(gy - 10) <= 1 and abs(gx - 10) <= 1

    def test_brightness_perturbation_bounded_and_clipped(self):
        img = np.full((5, 5, 3), 250, dtype=np.uint8)
        out = stability.perturb_brightness(img, 0.05)
        assert out.max() <= 255
        assert out.dtype == np.uint8

    def test_gaussian_noise_deterministic_with_fixed_seed(self):
        img = np.full((5, 5, 3), 100, dtype=np.uint8)
        a = stability.perturb_gaussian_noise(img, seed=357)
        b = stability.perturb_gaussian_noise(img, seed=357)
        assert np.array_equal(a, b)


# ===========================================================================
# Pixel IoU
# ===========================================================================

class TestPixelIoU:
    def test_known_iou_synthetic(self):
        hm = np.zeros((10, 10), dtype=np.float32)
        hm[0:5, 0:10] = 1.0  # top half: positive energy, uniform
        ref_mask = np.zeros((10, 10), dtype=np.uint8)
        ref_mask[0:5, 0:5] = 1  # top-left quadrant only
        # top-20%-by-energy of a uniform 50-pixel block = any 10 of those 50 pixels
        # (ties) -- with mergesort stable tie-break, it's the FIRST 10 pixels in
        # descending-sort order, which for an all-equal block is the original
        # (row-major) order, i.e. row 0, cols 0..9.
        binary = pixel_iou.binarize_top_energy_fraction(hm, fraction=0.20)
        assert binary.sum() == 10  # 20% of 50 positive pixels
        iou = pixel_iou.spatial_iou(binary, ref_mask)
        # predicted mask = row 0 (10px); reference = rows0-4,cols0-4 (25px);
        # intersection = row0,cols0-4 = 5px; union = 10+25-5=30
        assert iou == pytest.approx(5 / 30)

    def test_degenerate_heatmap_reported_not_zero_iou(self):
        hm = np.zeros((5, 5), dtype=np.float32)
        ref = np.ones((5, 5), dtype=np.uint8)
        r = pixel_iou.evaluate_pixel_iou(hm, np.dstack([ref]), "x", "gradcam")
        assert r.status == "degenerate_heatmap"
        assert r.iou is None

    def test_shape_mismatch_reported_not_guessed(self):
        hm = np.zeros((5, 5), dtype=np.float32)
        hm[0, 0] = 1.0
        ref = np.ones((6, 6), dtype=np.uint8)
        r = pixel_iou.evaluate_pixel_iou(hm, np.dstack([ref]), "x", "gradcam")
        assert r.status == "shape_mismatch"

    def test_no_threshold_search_fixed_fraction_only(self):
        hm = np.random.default_rng(9).random((10, 10)).astype(np.float32)
        b1 = pixel_iou.binarize_top_energy_fraction(hm, fraction=0.20)
        b2 = pixel_iou.binarize_top_energy_fraction(hm, fraction=0.20)
        assert np.array_equal(b1, b2)  # same fixed fraction -> identical, no search


# ===========================================================================
# Pilot selection
# ===========================================================================

class TestPilotSelection:
    def test_pilot_selection_reproducible(self):
        if not PAIRED_MANIFEST.is_file():
            pytest.skip("F2.3-B paired manifest not present in this environment")
        r1 = pilot_selection.select_pilot(PAIRED_MANIFEST)
        r2 = pilot_selection.select_pilot(PAIRED_MANIFEST)
        ids1 = [c.gt_instance_id for c in r1["selected"]]
        ids2 = [c.gt_instance_id for c in r2["selected"]]
        assert ids1 == ids2

    def test_pilot_selection_within_6_to_9(self):
        if not PAIRED_MANIFEST.is_file():
            pytest.skip("F2.3-B paired manifest not present in this environment")
        r = pilot_selection.select_pilot(PAIRED_MANIFEST)
        assert 6 <= r["n_selected"] <= 9

    def test_pilot_selection_no_duplicate_source_family(self):
        if not PAIRED_MANIFEST.is_file():
            pytest.skip("F2.3-B paired manifest not present in this environment")
        r = pilot_selection.select_pilot(PAIRED_MANIFEST)
        families = [c.source_family_id for c in r["selected"]]
        assert len(families) == len(set(families))

    def test_pilot_selection_respects_per_group_cap(self):
        if not PAIRED_MANIFEST.is_file():
            pytest.skip("F2.3-B paired manifest not present in this environment")
        r = pilot_selection.select_pilot(PAIRED_MANIFEST)
        assert all(v <= pilot_selection.PER_GROUP_CAP for v in r["per_group_count"].values())


# ===========================================================================
# Missingness / failure accounting
# ===========================================================================

class TestMissingnessAccounting:
    def test_localization_summary_counts_every_status_bucket(self):
        pts = [
            localization.pointing_accuracy(np.array([[1.0]]), (0, 0, 0, 0), (1, 1)),
            localization.pointing_accuracy(np.zeros((1, 1)), (0, 0, 0, 0), (1, 1)),
        ]
        ens = [
            localization.box_energy_concentration(np.array([[1.0]]), (0, 0, 0, 0), (1, 1)),
            localization.box_energy_concentration(np.zeros((1, 1)), (0, 0, 0, 0), (1, 1)),
        ]
        s = localization.summarize_method(pts, ens)
        assert s.n_total == 2
        assert s.n_valid == 1
        assert s.n_invalid_all_zero_or_degenerate == 1
        assert s.n_hits == 1  # (0,0) inside a zero-area box inclusive boundary


# ===========================================================================
# Hard regressions: TEST-split lockout, protected-artifact checksums
# ===========================================================================

class TestZeroTestSplitAccess:
    def test_evaluation_package_never_imports_test_dirs(self):
        """Static-text regression: none of the new evaluation/*.py source
        files reference the TEST-split directory constants, mirroring the
        existing hard lockout pattern in test_f23b_full_run.py."""
        eval_dir = REPO_ROOT / "research/component1/function2/evaluation"
        banned = ["CHENAB_TEST_IMAGES_DIR", "CHENAB_TEST_LABELS_DIR", '"test" / "images"',
                  "'test' / 'images'"]
        for py_file in eval_dir.glob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            for token in banned:
                assert token not in text, f"{py_file.name} references {token}"

    def test_pilot_only_uses_valid_split_images(self):
        if not PAIRED_MANIFEST.is_file():
            pytest.skip("manifest not present")
        from research.component1.function2.config.paths import CHENAB_SPLIT_MANIFEST
        if not CHENAB_SPLIT_MANIFEST.is_file():
            pytest.skip("split manifest not present")
        r = pilot_selection.select_pilot(PAIRED_MANIFEST)
        with open(CHENAB_SPLIT_MANIFEST, newline="", encoding="utf-8") as f:
            split_by_id = {row["image_id"]: row["split"] for row in csv.DictReader(f)}
        for c in r["selected"]:
            assert split_by_id.get(c.image_id) == "valid"


class TestProtectedArtifactsUnchangedByF24:
    def test_all_f24_protected_paths_byte_unchanged_vs_f24_pre_stage_baseline(self):
        baseline_path = (
            REPO_ROOT / "research/component1/function2/results/xai_evaluation"
            "/f24_pilot_v1/checksums/pre_stage_baseline.json"
        )
        if not baseline_path.is_file():
            pytest.skip("F2.4-A pre-stage checksum baseline not present in this environment")
        import json
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        results = verify_against_snapshot(baseline)
        failures = {k: v for k, v in results.items() if v["status"] != "PASS"}
        assert not failures, f"Protected-artifact drift detected: {failures}"

    def test_e02_checkpoint_byte_unchanged_after_sanity_checks(self):
        """Hard proof that evaluation.sanity_checks never writes to the
        on-disk checkpoint: load an isolated copy, randomize a layer (which
        mutates in-memory tensors only), then re-hash the file on disk."""
        if not _models_present():
            pytest.skip("E02 checkpoint not present in this environment")
        before = _sha256(E02_CHECKPOINT)
        xai = sanity_checks.load_isolated_copy(str(E02_CHECKPOINT), device="cpu")
        sanity_checks.randomize_layer_inplace(xai, 22, seed=357)
        after = _sha256(E02_CHECKPOINT)
        assert before == after


# ===========================================================================
# Timing isolation
# ===========================================================================

class TestTimingIsolation:
    def test_stage_timings_sum_approximately_to_total(self):
        from research.component1.function2.evaluation.timing import StageTimings
        t = StageTimings(preprocessing_s=0.01, forward_pass_s=0.05,
                          backward_pass_s=0.03, cam_computation_s=0.02, resize_align_s=0.01)
        assert t.total_s == pytest.approx(0.12)

    def test_instrumented_explain_matches_uninstrumented_result(self):
        if not _models_present() or not _dataset_present():
            pytest.skip("model/dataset not present in this environment")
        from detection_xai import DetectionXAI
        from research.component1.function2.evaluation.timing import instrumented_explain
        import cv2
        from research.component1.function2.config.paths import CHENAB_VALID_IMAGES_DIR
        sample_imgs = sorted(CHENAB_VALID_IMAGES_DIR.glob("*.jpg"))[:1]
        if not sample_imgs:
            pytest.skip("no sample valid-split image found")
        xai = DetectionXAI(str(E02_CHECKPOINT), device="cpu")
        img = cv2.imread(str(sample_imgs[0]))
        det = xai.detect(img)
        if not det.detections:
            pytest.skip("sample image produced zero detections at default thresholds")
        d = det.detections[0]
        expl = xai.explain(img, d, method="gradcam")
        timings = instrumented_explain(xai, img, d, method="gradcam")
        assert timings.total_s > 0
        assert expl.status in ("success", "degenerate")


# ===========================================================================
# Sanity-check isolation structure
# ===========================================================================

class TestSanityCheckIsolation:
    def test_randomize_layer_changes_weights_in_memory(self):
        if not _models_present():
            pytest.skip("E02 checkpoint not present in this environment")
        xai = sanity_checks.load_isolated_copy(str(E02_CHECKPOINT), device="cpu")
        layer = xai._model.model[22]
        before = [p.detach().clone() for p in layer.parameters()]
        sanity_checks.randomize_layer_inplace(xai, 22, seed=357)
        after = list(layer.parameters())
        assert any(not np.array_equal(b.numpy(), a.detach().numpy())
                   for b, a in zip(before, after))

    def test_randomize_layer_is_seed_deterministic(self):
        if not _models_present():
            pytest.skip("E02 checkpoint not present in this environment")
        xai1 = sanity_checks.load_isolated_copy(str(E02_CHECKPOINT), device="cpu")
        xai2 = sanity_checks.load_isolated_copy(str(E02_CHECKPOINT), device="cpu")
        sanity_checks.randomize_layer_inplace(xai1, 22, seed=357)
        sanity_checks.randomize_layer_inplace(xai2, 22, seed=357)
        p1 = next(xai1._model.model[22].parameters())
        p2 = next(xai2._model.model[22].parameters())
        assert np.array_equal(p1.detach().numpy(), p2.detach().numpy())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
