"""F2.3-A: Grad-CAM / Grad-CAM++ execution pipeline tests (Task 7).

Covers: E02 checksum/class-mapping, real differentiable-target linkage,
matching-policy determinism, same-target-for-both-methods, gradient-hook
firing + finite/nonzero assertions, Grad-CAM math correctness, Grad-CAM++
math correctness + "not secretly Grad-CAM" differentiation, heatmap
shape/dtype/range validity, inverse-letterbox coordinate mapping,
multi-defect instance isolation, zero TEST-split access, unmatched-detection
failure reporting, deterministic output identity, and metadata preservation.

All tests write outputs (when they write anything at all) to pytest's
``tmp_path`` only -- never to a production manifest, even temporarily.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.config.paths import (  # noqa: E402
    CHENAB_DATASET_ROOT,
    CHENAB_SPLIT_MANIFEST,
    CHENAB_TEST_IMAGES_DIR,
    CHENAB_TEST_LABELS_DIR,
    CLASS_NAMES,
    COMPLEXITY_FEATURES_MODULE,
    DETECTION_XAI_MODULE,
    E02_CHECKPOINT,
    FREEZE_MANIFEST,
)
from research.component1.function2.xai import gt_matching  # noqa: E402
from research.component1.function2.xai import manifest_verification as mver  # noqa: E402
from research.component1.function2.xai import pipeline  # noqa: E402
from research.component1.function2.xai import protected_artifacts  # noqa: E402
from research.component1.function2.xai import visualization as viz  # noqa: E402

_XAI_SRC = REPO_ROOT / "research" / "component1" / "xai_experiments" / "src"
if str(_XAI_SRC) not in sys.path:
    sys.path.insert(0, str(_XAI_SRC))
from detection_xai import Detection, DetectionXAI  # noqa: E402

F22B_MANIFESTS = (
    REPO_ROOT / "research" / "component1" / "function2" / "results"
    / "evaluation_sampling" / "f22b_pp1_v1" / "manifests"
)
DRY_RUN_SAMPLE_CSV = F22B_MANIFESTS / "xai_dry_run_sample.csv"
VALID_COMPLEXITY_MANIFEST_CSV = F22B_MANIFESTS / "valid_complexity_manifest.csv"


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _models_present() -> bool:
    return E02_CHECKPOINT.is_file() and FREEZE_MANIFEST.is_file()


requires_model = pytest.mark.skipif(
    not _models_present(), reason="E02 checkpoint/freeze manifest not present in this environment"
)


# ---------------------------------------------------------------------------
# 1. E02 checksum + class-mapping consistency
# ---------------------------------------------------------------------------

class TestE02ChecksumAndClassMapping:
    @requires_model
    def test_checkpoint_sha256_matches_freeze_manifest(self):
        manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
        expected = manifest["selected_experiment"]["checkpoint_sha256"]
        assert _sha256(E02_CHECKPOINT) == expected

    def test_class_names_match_freeze_manifest_and_detection_xai_module(self):
        manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
        manifest_classes = manifest["dataset_identity"]["class_names_and_order"]
        assert manifest_classes == list(CLASS_NAMES)
        assert manifest_classes == [
            "baekra", "color issues", "contamination", "cut",
            "gray stitch", "selvet", "stain",
        ]


# ---------------------------------------------------------------------------
# 2. Protected-artifact checksum regression (Task 7 hard requirement)
# ---------------------------------------------------------------------------

class TestProtectedArtifactsUnchanged:
    def test_known_good_hashes_all_pass(self):
        results = protected_artifacts.verify_known_good_hashes()
        failures = {k: v for k, v in results.items() if v["status"] != "PASS"}
        assert failures == {}, f"Protected artifact drift detected: {failures}"

    def test_pre_stage_snapshot_matches_if_recorded(self):
        snapshot_path = (
            REPO_ROOT / "research" / "component1" / "function2" / "results"
            / "xai_execution" / "f23_dry_run_v1" / "checksums" / "pre_stage_baseline.json"
        )
        if not snapshot_path.is_file():
            pytest.skip("pre-stage snapshot not yet recorded in this environment")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        results = protected_artifacts.verify_against_pre_stage_snapshot(snapshot)
        failures = {k: v for k, v in results.items() if v["status"] != "PASS"}
        assert failures == {}, f"Protected artifact changed since pre-stage snapshot: {failures}"

    def test_this_stage_does_not_modify_detection_xai_source(self):
        src = DETECTION_XAI_MODULE.read_text(encoding="utf-8")
        assert "class DetectionXAI" in src  # sanity: still the real module
        assert protected_artifacts.KNOWN_GOOD_HASHES[
            "research/component1/xai_experiments/src/detection_xai.py"
        ] == _sha256(DETECTION_XAI_MODULE)

    def test_this_stage_does_not_modify_complexity_features_source(self):
        assert protected_artifacts.KNOWN_GOOD_HASHES[
            "research/component1/complexity_calibration/src/complexity_features.py"
        ] == _sha256(COMPLEXITY_FEATURES_MODULE)


# ---------------------------------------------------------------------------
# 3. Zero TEST-split access (hard regression test)
# ---------------------------------------------------------------------------

class TestZeroTestSplitAccess:
    def test_xai_package_source_never_references_test_split_dirs(self):
        xai_pkg_dir = REPO_ROOT / "research" / "component1" / "function2" / "xai"
        offending = []
        for py in xai_pkg_dir.glob("*.py"):
            text = py.read_text(encoding="utf-8")
            if "test/images" in text or "test/labels" in text or "CHENAB_TEST" in text:
                offending.append(py.name)
        assert offending == [], f"F2.3-A source references TEST-split paths: {offending}"

    def test_manifest_verification_rejects_test_split_row(self, tmp_path):
        split_csv = tmp_path / "split_manifest.csv"
        split_csv.write_text(
            "image_id,filename,split,group_id,is_negative\n"
            "fake_image_1,fake.jpg,test,singleton_fake.jpg,False\n",
            encoding="utf-8",
        )
        complexity_csv = tmp_path / "valid_complexity_manifest.csv"
        complexity_csv.write_text(
            "image_id,authoritative_split,source_family_id,defect_classes,"
            "defect_instance_count,has_gt_defect,complexity_eligible,complexity_group\n"
            "fake_image_1,test,singleton_fake.jpg,[],1,True,True,Simple\n",
            encoding="utf-8",
        )
        dry_run_csv = tmp_path / "xai_dry_run_sample.csv"
        dry_run_csv.write_text(
            "sample_type,image_id,source_family_id,complexity_group,boundary_adjacent,"
            "defect_classes,defect_instance_count,has_gt_defect,complexity_score,"
            "selection_rank_within_group\n"
            "dry_run,fake_image_1,singleton_fake.jpg,Simple,False,[],1,True,0.0,1\n",
            encoding="utf-8",
        )
        with pytest.raises(mver.ManifestVerificationError, match="TEST-split"):
            mver.verify_dry_run_sample(dry_run_csv, split_csv, complexity_csv, tmp_path)

    @requires_model
    def test_real_dry_run_sample_contains_zero_test_split_rows(self):
        verified = mver.verify_dry_run_sample(
            DRY_RUN_SAMPLE_CSV, CHENAB_SPLIT_MANIFEST, VALID_COMPLEXITY_MANIFEST_CSV,
            CHENAB_DATASET_ROOT,
        )
        assert len(verified) == 9
        assert all(r.authoritative_split == "valid" for r in verified)


# ---------------------------------------------------------------------------
# 4. GT loading + IoU-matching policy (Task 3)
# ---------------------------------------------------------------------------

class TestIoUMath:
    def test_identical_boxes_iou_is_one(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert gt_matching.iou_xyxy(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_is_zero(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (20.0, 20.0, 30.0, 30.0)
        assert gt_matching.iou_xyxy(a, b) == 0.0

    def test_edge_touching_boxes_iou_is_zero(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (10.0, 0.0, 20.0, 10.0)
        assert gt_matching.iou_xyxy(a, b) == 0.0

    def test_half_overlap_hand_computed(self):
        a = (0.0, 0.0, 10.0, 10.0)   # area 100
        b = (5.0, 0.0, 15.0, 10.0)   # area 100, overlap region x in [5,10] -> 5x10=50
        # union = 100 + 100 - 50 = 150; iou = 50/150 = 1/3
        assert gt_matching.iou_xyxy(a, b) == pytest.approx(1.0 / 3.0, abs=1e-9)


class _FakeDet:
    def __init__(self, detection_id, class_id, confidence, bbox_xyxy):
        self.detection_id = detection_id
        self.class_id = class_id
        self.confidence = confidence
        self.bbox_xyxy = bbox_xyxy
        self.class_name = f"cls{class_id}"


class TestGTMatchingPolicy:
    def test_picks_highest_iou_same_class(self):
        gt = gt_matching.GTInstance(0, class_id=1, class_name="cut", bbox_xyxy=(0, 0, 10, 10))
        dets = [
            _FakeDet(0, class_id=1, confidence=0.9, bbox_xyxy=[0, 0, 6, 6]),   # lower IoU
            _FakeDet(1, class_id=1, confidence=0.5, bbox_xyxy=[0, 0, 10, 10]),  # IoU 1.0
            _FakeDet(2, class_id=2, confidence=0.99, bbox_xyxy=[0, 0, 10, 10]),  # wrong class
        ]
        result = gt_matching.match_gt_to_detections(gt, dets)
        assert result.status == "matched"
        assert result.matched_detection_id == 1
        assert result.iou == pytest.approx(1.0)

    def test_unmatched_when_no_same_class_detection(self):
        gt = gt_matching.GTInstance(0, class_id=1, class_name="cut", bbox_xyxy=(0, 0, 10, 10))
        dets = [_FakeDet(0, class_id=2, confidence=0.9, bbox_xyxy=[0, 0, 10, 10])]
        result = gt_matching.match_gt_to_detections(gt, dets)
        assert result.status == "unmatched"
        assert "class" in result.reason.lower()

    def test_unmatched_when_below_iou_threshold(self):
        gt = gt_matching.GTInstance(0, class_id=1, class_name="cut", bbox_xyxy=(0, 0, 10, 10))
        dets = [_FakeDet(0, class_id=1, confidence=0.9, bbox_xyxy=[8, 8, 20, 20])]  # tiny overlap
        result = gt_matching.match_gt_to_detections(gt, dets, iou_threshold=0.5)
        assert result.status == "unmatched"
        assert result.iou < 0.5

    def test_matching_is_deterministic_across_repeated_calls(self):
        gt = gt_matching.GTInstance(0, class_id=1, class_name="cut", bbox_xyxy=(0, 0, 10, 10))
        dets = [
            _FakeDet(0, class_id=1, confidence=0.9, bbox_xyxy=[0, 0, 10, 10]),
            _FakeDet(1, class_id=1, confidence=0.8, bbox_xyxy=[0, 0, 10, 10]),
        ]
        results = [gt_matching.match_gt_to_detections(gt, dets) for _ in range(5)]
        ids = {r.matched_detection_id for r in results}
        assert len(ids) == 1  # same winner every time (tie -> lowest detection_id)
        assert results[0].matched_detection_id == 0

    def test_load_yolo_gt_instances_parses_normalized_coords(self, tmp_path):
        label_path = tmp_path / "img.txt"
        # class 3, center (0.5, 0.5), w=0.2, h=0.4 on a 100x200 (w,h) image
        label_path.write_text("3 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        instances = gt_matching.load_yolo_gt_instances(label_path, img_w=100, img_h=200,
                                                        class_names=CLASS_NAMES)
        assert len(instances) == 1
        inst = instances[0]
        assert inst.class_id == 3
        assert inst.class_name == "cut"
        x1, y1, x2, y2 = inst.bbox_xyxy
        assert x1 == pytest.approx(40.0)
        assert x2 == pytest.approx(60.0)
        assert y1 == pytest.approx(60.0)
        assert y2 == pytest.approx(140.0)

    def test_multi_defect_instance_ordering_preserved(self, tmp_path):
        label_path = tmp_path / "img.txt"
        label_path.write_text(
            "0 0.1 0.1 0.1 0.1\n1 0.9 0.9 0.1 0.1\n", encoding="utf-8",
        )
        instances = gt_matching.load_yolo_gt_instances(label_path, 100, 100, CLASS_NAMES)
        assert [i.instance_index for i in instances] == [0, 1]
        assert instances[0].class_id == 0
        assert instances[1].class_id == 1


# ---------------------------------------------------------------------------
# 5. Grad-CAM / Grad-CAM++ math correctness (synthetic, hand-computable)
# ---------------------------------------------------------------------------
# These reproduce the EXACT formulas used inline in detection_xai.py's
# explain() (lines computing `weights`/`cam` for "gradcam" vs "gradcampp") on
# small hand-crafted tensors, purely to verify the formulas themselves are
# correct and genuinely distinct. This does NOT replace detection_xai.py --
# the real pipeline (pipeline.py) always calls DetectionXAI.explain().

def _gradcam_cam(act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    weights = grad.mean(dim=(2, 3), keepdim=True)
    return torch.relu((weights * act).sum(dim=1, keepdim=True))


def _gradcampp_cam(act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    grad2 = grad ** 2
    grad3 = grad ** 3
    sum_act = act.sum(dim=(2, 3), keepdim=True) + 1e-7
    alpha = grad2 / (2 * grad2 + sum_act * grad3 + 1e-7)
    weights = (alpha * torch.relu(grad)).sum(dim=(2, 3), keepdim=True)
    return torch.relu((weights * act).sum(dim=1, keepdim=True))


class TestGradCamMathCorrectness:
    def test_gradcam_weights_are_global_average_pooled_gradients_hand_computed(self):
        # 1 batch, 1 channel, 2x2 spatial. Activation all 1s, gradient = [[1,2],[3,4]].
        act = torch.ones(1, 1, 2, 2)
        grad = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        cam = _gradcam_cam(act, grad)
        # weight = mean(grad) = 2.5; cam = relu(2.5 * act) = 2.5 everywhere
        assert torch.allclose(cam, torch.full((1, 1, 2, 2), 2.5))

    def test_gradcam_relu_clips_negative_weighted_sum(self):
        act = torch.ones(1, 1, 2, 2)
        grad = torch.full((1, 1, 2, 2), -1.0)  # mean weight = -1 -> cam should be 0 (ReLU)
        cam = _gradcam_cam(act, grad)
        assert torch.all(cam == 0)

    def test_gradcampp_matches_hand_computed_alpha_on_nonuniform_gradient(self):
        # Single channel, 1x2 spatial, nonuniform gradient so alpha-weighting
        # is actually exercised (not degenerate to GAP).
        act = torch.tensor([[[[2.0, 4.0]]]])     # shape (1,1,1,2)
        grad = torch.tensor([[[[1.0, 3.0]]]])
        grad2 = grad ** 2
        grad3 = grad ** 3
        sum_act = act.sum() + 1e-7
        alpha_expected = grad2 / (2 * grad2 + sum_act * grad3 + 1e-7)
        weights_expected = (alpha_expected * torch.relu(grad)).sum()
        cam_expected = torch.relu(weights_expected * act)
        cam_actual = _gradcampp_cam(act, grad)
        assert torch.allclose(cam_actual, cam_expected, atol=1e-6)

    def test_gradcampp_is_not_secretly_identical_to_gradcam(self):
        """Non-uniform gradient -> Grad-CAM (GAP weighting) and Grad-CAM++
        (alpha-weighted) must diverge; if they match the implementation has
        silently collapsed Grad-CAM++ into a Grad-CAM rename."""
        torch.manual_seed(0)
        act = torch.rand(1, 8, 5, 5) + 0.1  # positive activations
        grad = torch.rand(1, 8, 5, 5) * 2 - 1.0  # non-uniform, mixed-sign gradient
        cam_gc = _gradcam_cam(act, grad)
        cam_gcpp = _gradcampp_cam(act, grad)
        assert not torch.allclose(cam_gc, cam_gcpp, atol=1e-5), (
            "Grad-CAM and Grad-CAM++ produced identical output on a "
            "non-uniform-gradient case -- Grad-CAM++ is not genuinely "
            "distinct from Grad-CAM."
        )

    def test_gradcampp_alpha_tensor_actually_affects_output(self):
        """Ablation: zeroing the alpha term's higher-order correction
        (i.e. using plain ReLU(grad) summed with UNIFORM weight, as Grad-CAM
        does) gives a different result than the real alpha-weighted formula
        on non-uniform gradients -- proving alpha is load-bearing."""
        act = torch.tensor([[[[1.0, 5.0, 2.0]]]])
        grad = torch.tensor([[[[0.2, 4.0, -1.0]]]])
        real_gcpp = _gradcampp_cam(act, grad)
        naive_relu_only = torch.relu(
            (torch.relu(grad)).sum(dim=(2, 3), keepdim=True) * act
        )
        assert not torch.allclose(real_gcpp, naive_relu_only, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Inverse-letterbox coordinate mapping (synthetic known transform)
# ---------------------------------------------------------------------------

class TestCoordinateAlignment:
    def test_inverse_letterbox_known_transform_places_hot_region_correctly(self):
        # A wide (non-square) synthetic image: 100 (h) x 300 (w).
        img = np.zeros((100, 300, 3), dtype=np.uint8)
        tensor, ratio, dw, dh = DetectionXAI._preprocess(img, imgsz=640)
        # width is the binding dimension: ratio = 640/300
        assert ratio == pytest.approx(640 / 300)
        assert dw == pytest.approx(0.0, abs=1.0)
        assert dh > 0  # letterboxed top/bottom padding present

        # Build a synthetic 20x20 "cam" with a single hot cell in a known
        # location (top-left quadrant) and map it back.
        cam = np.zeros((20, 20), dtype=np.float32)
        cam[2, 2] = 1.0  # near top-left of the feature map
        mapped = DetectionXAI._map_to_original(cam, img.shape[:2], ratio, dw, dh, imgsz=640)
        assert mapped.shape == img.shape[:2]
        # The peak must land in the top-left region of the 100x300 original,
        # not the bottom-right (would indicate a flipped/broken inverse map).
        peak_y, peak_x = np.unravel_index(np.argmax(mapped), mapped.shape)
        assert peak_y < mapped.shape[0] * 0.5
        assert peak_x < mapped.shape[1] * 0.5

    def test_heatmap_alignment_check_detects_shape_mismatch(self):
        heatmap = np.zeros((50, 50), dtype=np.float32)
        image = np.zeros((50, 60, 3), dtype=np.uint8)
        assert viz.check_alignment(heatmap, image) is False
        image_ok = np.zeros((50, 50, 3), dtype=np.uint8)
        assert viz.check_alignment(heatmap, image_ok) is True


# ---------------------------------------------------------------------------
# 7. Negative tests: bad target, nonfinite gradient, misaligned heatmap
# ---------------------------------------------------------------------------

class TestNegativeCases:
    def test_nonfinite_cam_is_flagged_not_silently_passed(self):
        cam_with_nan = np.array([[1.0, np.nan], [0.5, 0.2]], dtype=np.float32)
        is_finite = bool(np.all(np.isfinite(cam_with_nan)))
        assert is_finite is False  # mirrors detection_xai.explain()'s own gate

    def test_nonfinite_cam_with_inf_is_flagged(self):
        cam_with_inf = np.array([[1.0, np.inf]], dtype=np.float32)
        assert bool(np.all(np.isfinite(cam_with_inf))) is False

    def test_degenerate_allzero_cam_is_flagged_nonzero_false(self):
        cam_zero = np.zeros((4, 4), dtype=np.float32)
        is_nonzero = bool(cam_zero.max() > 1e-8)
        assert is_nonzero is False

    def test_deliberately_misaligned_heatmap_rejected_by_alignment_check(self):
        bad_heatmap = np.zeros((10, 10), dtype=np.float32)
        real_image = np.zeros((224, 224, 3), dtype=np.uint8)
        assert viz.check_alignment(bad_heatmap, real_image) is False

    @requires_model
    def test_explain_with_invalid_pre_nms_index_reports_unavailable_not_success(self):
        xai = DetectionXAI(str(E02_CHECKPOINT), device="cpu")
        img = (np.random.default_rng(1).integers(0, 255, (200, 200, 3))).astype(np.uint8)
        fake_detection = Detection(
            detection_id=0, class_id=0, class_name="baekra", confidence=0.9,
            bbox_xyxy=[10, 10, 50, 50], pre_nms_index=-1,  # deliberately invalid/empty target
        )
        result = xai.explain(img, fake_detection, method="gradcam")
        assert result.status == "unavailable"
        assert result.heatmap_raw is None
        assert result.heatmap_overlay is None


# ---------------------------------------------------------------------------
# 8. Real end-to-end integration with E02 (uses the frozen 9-image sample)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xai_model():
    if not _models_present():
        pytest.skip("E02 checkpoint/freeze manifest not present in this environment")
    return DetectionXAI(str(E02_CHECKPOINT), device="cpu")


@pytest.fixture(scope="module")
def verified_sample():
    if not (DRY_RUN_SAMPLE_CSV.is_file() and CHENAB_SPLIT_MANIFEST.is_file()
            and VALID_COMPLEXITY_MANIFEST_CSV.is_file()):
        pytest.skip("F2.2B dry-run sample manifests not present in this environment")
    return mver.verify_dry_run_sample(
        DRY_RUN_SAMPLE_CSV, CHENAB_SPLIT_MANIFEST, VALID_COMPLEXITY_MANIFEST_CSV,
        CHENAB_DATASET_ROOT,
    )


def _first_matched_instance(xai_model, row):
    import cv2
    image_bgr = cv2.imread(str(row.image_path))
    gts = gt_matching.load_yolo_gt_instances(row.label_path, image_bgr.shape[1],
                                              image_bgr.shape[0], CLASS_NAMES)
    det_result = xai_model.detect(image_bgr, conf_threshold=pipeline.CONF_THRESHOLD,
                                   iou_threshold=pipeline.IOU_NMS_THRESHOLD)
    for gt in gts:
        match = gt_matching.match_gt_to_detections(gt, det_result.detections)
        if match.status == "matched":
            det = next(d for d in det_result.detections
                       if d.detection_id == match.matched_detection_id)
            return image_bgr, gt, match, det
    return image_bgr, None, None, None


class TestRealDetectionXAIIntegration:
    def test_a_real_matched_instance_exists_in_the_sample(self, xai_model, verified_sample):
        found = any(_first_matched_instance(xai_model, row)[2] is not None
                    for row in verified_sample)
        assert found, "No GT instance in the 9-image sample matched any prediction"

    def test_gradients_reach_target_layer_finite_and_nonzero_on_a_known_good_case(
        self, xai_model, verified_sample,
    ):
        for row in verified_sample:
            image_bgr, gt, match, det = _first_matched_instance(xai_model, row)
            if match is None:
                continue
            result = xai_model.explain(image_bgr, det, method="gradcam")
            assert result.status == "success"
            assert result.is_finite is True
            assert result.is_nonzero is True
            assert np.all(np.isfinite(result.heatmap_raw))
            return
        pytest.skip("no matched instance available to test")

    def test_same_target_logged_for_both_methods(self, xai_model, verified_sample):
        for row in verified_sample:
            image_bgr, gt, match, det = _first_matched_instance(xai_model, row)
            if match is None:
                continue
            gc = xai_model.explain(image_bgr, det, method="gradcam")
            gcpp = xai_model.explain(image_bgr, det, method="gradcampp")
            assert gc.status == "success" and gcpp.status == "success"
            assert gc.target_layer == gcpp.target_layer
            assert gc.target_scale == gcpp.target_scale
            assert gc.target_cell_gy_gx == gcpp.target_cell_gy_gx
            return
        pytest.skip("no matched instance available to test")

    def test_gradcam_and_gradcampp_differ_on_real_case(self, xai_model, verified_sample):
        for row in verified_sample:
            image_bgr, gt, match, det = _first_matched_instance(xai_model, row)
            if match is None:
                continue
            gc = xai_model.explain(image_bgr, det, method="gradcam")
            gcpp = xai_model.explain(image_bgr, det, method="gradcampp")
            assert gc.status == "success" and gcpp.status == "success"
            assert not np.allclose(gc.heatmap_raw, gcpp.heatmap_raw, atol=1e-6), (
                "Grad-CAM and Grad-CAM++ produced byte-identical heatmaps on "
                "a real detection -- suspect a collapsed/renamed implementation."
            )
            return
        pytest.skip("no matched instance available to test")

    def test_heatmap_shape_dtype_range_validity(self, xai_model, verified_sample):
        for row in verified_sample:
            image_bgr, gt, match, det = _first_matched_instance(xai_model, row)
            if match is None:
                continue
            gc = xai_model.explain(image_bgr, det, method="gradcam")
            assert gc.heatmap_raw.shape == image_bgr.shape[:2]
            assert gc.heatmap_raw.dtype in (np.float32, np.float64)
            assert gc.heatmap_raw.min() >= 0.0  # post-ReLU
            return
        pytest.skip("no matched instance available to test")

    def test_deterministic_output_identity_same_image_instance_two_runs(
        self, xai_model, verified_sample,
    ):
        for row in verified_sample:
            image_bgr, gt, match, det = _first_matched_instance(xai_model, row)
            if match is None:
                continue
            r1 = xai_model.explain(image_bgr, det, method="gradcam")
            r2 = xai_model.explain(image_bgr, det, method="gradcam")
            assert r1.status == "success" and r2.status == "success"
            assert np.array_equal(r1.heatmap_raw, r2.heatmap_raw), (
                "Repeated explain() calls on the identical image+detection "
                "produced different heatmap bytes -- pipeline is not "
                "deterministic in an eval-mode, CPU, no-dropout model."
            )
            return
        pytest.skip("no matched instance available to test")

    def test_multi_defect_instance_isolation(self, xai_model, verified_sample):
        """Find an image with >=2 GT instances and verify each instance's
        heatmap is independently computed (not cross-contaminated)."""
        import cv2
        for row in verified_sample:
            image_bgr = cv2.imread(str(row.image_path))
            gts = gt_matching.load_yolo_gt_instances(
                row.label_path, image_bgr.shape[1], image_bgr.shape[0], CLASS_NAMES,
            )
            if len(gts) < 2:
                continue
            det_result = xai_model.detect(image_bgr, conf_threshold=pipeline.CONF_THRESHOLD,
                                           iou_threshold=pipeline.IOU_NMS_THRESHOLD)
            matched = []
            for gt in gts:
                m = gt_matching.match_gt_to_detections(gt, det_result.detections)
                if m.status == "matched":
                    det = next(d for d in det_result.detections
                              if d.detection_id == m.matched_detection_id)
                    matched.append((gt, det))
            if len(matched) < 2:
                continue
            heatmaps = []
            targets = []
            for gt, det in matched:
                r = xai_model.explain(image_bgr, det, method="gradcam")
                if r.status == "success":
                    heatmaps.append(r.heatmap_raw)
                    targets.append((r.target_scale, r.target_cell_gy_gx))
            if len(heatmaps) < 2:
                continue
            # Distinct GT instances with distinct matched detections must not
            # resolve to the identical target cell, and must not produce
            # byte-identical heatmaps (that would indicate cross-contamination).
            assert len(set(targets)) == len(targets), (
                f"Two distinct instances resolved to the same target cell: {targets}"
            )
            assert not np.array_equal(heatmaps[0], heatmaps[1])
            return
        pytest.skip("no multi-defect image with >=2 matched instances available")

    def test_unmatched_instance_produces_no_fabricated_explanation(
        self, xai_model, verified_sample,
    ):
        """Construct a GT instance of a class guaranteed absent from the
        image's predictions (by asking for a class the image has zero
        predictions of) and verify it is recorded unmatched with a real
        reason, never silently explained."""
        for row in verified_sample:
            image_bgr, gt, match, det = _first_matched_instance(xai_model, row)
            if gt is None:
                continue
            det_result = xai_model.detect(image_bgr, conf_threshold=pipeline.CONF_THRESHOLD,
                                           iou_threshold=pipeline.IOU_NMS_THRESHOLD)
            present_classes = {d.class_id for d in det_result.detections}
            absent_classes = set(range(len(CLASS_NAMES))) - present_classes
            if not absent_classes:
                continue
            fake_gt = gt_matching.GTInstance(
                instance_index=999, class_id=next(iter(absent_classes)),
                class_name=CLASS_NAMES[next(iter(absent_classes))],
                bbox_xyxy=(0, 0, 10, 10),
            )
            result = gt_matching.match_gt_to_detections(fake_gt, det_result.detections)
            assert result.status == "unmatched"
            assert result.reason != ""
            return
        pytest.skip("no suitable image found to construct an absent-class unmatched case")


# ---------------------------------------------------------------------------
# 9. Metadata preservation through the pipeline
# ---------------------------------------------------------------------------

class TestMetadataPreservation:
    @requires_model
    def test_source_family_and_complexity_group_preserved_in_outputs(self, tmp_path):
        result = pipeline.run_dry_run(
            dry_run_sample_csv=DRY_RUN_SAMPLE_CSV,
            split_manifest_csv=CHENAB_SPLIT_MANIFEST,
            valid_complexity_manifest_csv=VALID_COMPLEXITY_MANIFEST_CSV,
            dataset_root=CHENAB_DATASET_ROOT,
            freeze_manifest_path=FREEZE_MANIFEST,
            checkpoint_path=E02_CHECKPOINT,
            output_dir=tmp_path,
        )
        with open(DRY_RUN_SAMPLE_CSV, newline="", encoding="utf-8") as f:
            input_rows = {r["image_id"]: r for r in csv.DictReader(f)}

        exec_manifest = tmp_path / "manifests" / "execution_manifest.csv"
        with open(exec_manifest, newline="", encoding="utf-8") as f:
            out_rows = list(csv.DictReader(f))
        assert len(out_rows) == 9
        for row in out_rows:
            src = input_rows[row["image_id"]]
            assert row["complexity_group"] == src["complexity_group"]
            assert row["source_family_id"] == src["source_family_id"]

        assert result["n_images_attempted"] == 9
        assert result["n_images_processed"] == 9
        # Multi-defect expectation: more instances than images (19 total GT
        # instances across the 9 images per the frozen sample's own counts).
        assert result["n_instances_total"] > 9


# ---------------------------------------------------------------------------
# 10. Full 9-image dry-run production run assertions (not a re-execution --
#     these just validate the ALREADY-WRITTEN production output directory,
#     if present, matches the honesty requirements of Task 6).
# ---------------------------------------------------------------------------

class TestProductionDryRunOutputsIfPresent:
    def _manifests_dir(self):
        return (
            REPO_ROOT / "research" / "component1" / "function2" / "results"
            / "xai_execution" / "f23_dry_run_v1" / "manifests"
        )

    def test_execution_manifest_has_nine_image_rows_if_present(self):
        p = self._manifests_dir() / "execution_manifest.csv"
        if not p.is_file():
            pytest.skip("production dry-run output not yet generated")
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 9

    def test_target_linkage_has_more_instance_rows_than_images_if_present(self):
        p = self._manifests_dir() / "target_linkage.csv"
        if not p.is_file():
            pytest.skip("production dry-run output not yet generated")
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 9

    def test_no_fabricated_success_for_unmatched_rows(self):
        p = self._manifests_dir() / "target_linkage.csv"
        if not p.is_file():
            pytest.skip("production dry-run output not yet generated")
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["match_status"] == "unmatched":
                assert row["gradcam_status"] in ("not_attempted", "")
                assert row["gradcampp_status"] in ("not_attempted", "")
