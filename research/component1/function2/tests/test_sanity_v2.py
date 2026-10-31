"""F2.4-B Task 2 / Task 14 regression tests for
``evaluation/sanity_checks_v2.py`` (classification-branch-only
randomization) and its subset runner.

Covers: cv2/cv3 parameter disjointness (the architectural claim the whole
module's design rests on), isolation (on-disk checkpoint byte-unchanged),
narrow-scope randomization (only the targeted scale's cv3 changes; cv2/dfl/
other-scale cv3 untouched), target-validity not force-invalidated (unlike
the whole-head check), genuine-gradient checks, and reproducibility.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_XAI_SRC = _REPO_ROOT / "research" / "component1" / "xai_experiments" / "src"
if str(_XAI_SRC) not in sys.path:
    sys.path.insert(0, str(_XAI_SRC))

from research.component1.function2.config import paths as cfg_paths
from research.component1.function2.evaluation import sanity_checks_v2 as scv2
from research.component1.function2.evaluation.common import frozen_target_score

CHECKPOINT = cfg_paths.E02_CHECKPOINT


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def trained_xai():
    from detection_xai import DetectionXAI
    return DetectionXAI(str(CHECKPOINT), device="cpu")


def test_cv2_cv3_share_no_parameters(trained_xai):
    """Architectural precondition this whole module relies on: box-regression
    (cv2) and classification (cv3) branches must not share any parameter
    tensor, at any scale."""
    head = trained_xai._model.model[22]
    cv2_ids = {id(p) for branch in head.cv2 for p in branch.parameters()}
    cv3_ids = {id(p) for branch in head.cv3 for p in branch.parameters()}
    assert cv2_ids.isdisjoint(cv3_ids)
    assert len(cv2_ids) > 0 and len(cv3_ids) > 0


def test_cv3_modulelist_has_three_independent_scales(trained_xai):
    head = trained_xai._model.model[22]
    assert len(head.cv3) == 3
    param_sets = []
    for branch in head.cv3:
        param_sets.append({id(p) for p in branch.parameters()})
    for i in range(3):
        for j in range(3):
            if i != j:
                assert param_sets[i].isdisjoint(param_sets[j])


def test_e02_checkpoint_byte_unchanged_after_v2_sanity():
    before = _sha256(CHECKPOINT)
    iso = scv2.load_isolated_copy(str(CHECKPOINT), device="cpu")
    scv2.randomize_classification_branch_inplace(iso, scale_idx=0, seed=357)
    scv2.randomize_classification_branch_inplace(iso, scale_idx=1, seed=357)
    scv2.randomize_classification_branch_inplace(iso, scale_idx=2, seed=357)
    after = _sha256(CHECKPOINT)
    assert before == after


def test_randomization_only_touches_targeted_scale(trained_xai):
    """Randomizing cv3[0] must leave cv3[1], cv3[2], ALL of cv2, and dfl
    bit-for-bit identical to a freshly (unrandomized) loaded model."""
    baseline = scv2.load_isolated_copy(str(CHECKPOINT), device="cpu")
    iso = scv2.load_isolated_copy(str(CHECKPOINT), device="cpu")
    scv2.randomize_classification_branch_inplace(iso, scale_idx=0, seed=357)

    head_base, head_iso = baseline._model.model[22], iso._model.model[22]

    # cv3[0] MUST have changed
    changed = any(not torch.equal(pb, pi) for pb, pi in
                  zip(head_base.cv3[0].parameters(), head_iso.cv3[0].parameters()))
    assert changed, "cv3[0] should have been randomized"

    # cv3[1], cv3[2] MUST be unchanged
    for scale in (1, 2):
        for pb, pi in zip(head_base.cv3[scale].parameters(), head_iso.cv3[scale].parameters()):
            assert torch.equal(pb, pi), f"cv3[{scale}] must be untouched"

    # cv2 (all scales) MUST be unchanged
    for scale in range(3):
        for pb, pi in zip(head_base.cv2[scale].parameters(), head_iso.cv2[scale].parameters()):
            assert torch.equal(pb, pi), f"cv2[{scale}] (box regression) must be untouched"

    # dfl MUST be unchanged
    for pb, pi in zip(head_base.dfl.parameters(), head_iso.dfl.parameters()):
        assert torch.equal(pb, pi), "dfl must be untouched"

    # The three Grad-CAM target layers (backbone/neck, upstream of model[22])
    # must also be untouched.
    from detection_xai import TARGET_LAYER_INDICES
    for li in TARGET_LAYER_INDICES.values():
        lb, lic = baseline._model.model[li], iso._model.model[li]
        for pb, pi in zip(lb.parameters(), lic.parameters()):
            assert torch.equal(pb, pi), f"model.model[{li}] (Grad-CAM target layer) must be untouched"


def test_randomization_is_deterministic_given_seed():
    iso1 = scv2.load_isolated_copy(str(CHECKPOINT), device="cpu")
    iso2 = scv2.load_isolated_copy(str(CHECKPOINT), device="cpu")
    scv2.randomize_classification_branch_inplace(iso1, scale_idx=1, seed=357)
    scv2.randomize_classification_branch_inplace(iso2, scale_idx=1, seed=357)
    for p1, p2 in zip(iso1._model.model[22].cv3[1].parameters(),
                       iso2._model.model[22].cv3[1].parameters()):
        assert torch.equal(p1, p2)


def test_frozen_index_remains_structurally_valid_after_randomization(trained_xai):
    """Unlike the whole-head check (which force-invalidates ~100% of
    targets because box geometry moves), the frozen-index mechanism must
    remain shape-valid after classification-only randomization for any
    legitimate (class_id, pre_nms_index)."""
    import cv2
    sample_img = np.zeros((640, 640, 3), dtype=np.uint8)
    iso = scv2.load_isolated_copy(str(CHECKPOINT), device="cpu")
    scv2.randomize_classification_branch_inplace(iso, scale_idx=0, seed=357)
    r = frozen_target_score(iso, sample_img, class_id=0, pre_nms_index=100)
    assert r["status"] == "valid"


def test_classification_branch_gradient_is_genuine(trained_xai):
    """The CAM must actually be computed from REAL randomized-model
    gradients, not a cached/stale trained-model heatmap -- verify the
    randomized-model heatmap differs from the trained-model heatmap for a
    cell inside the randomized scale."""
    import cv2
    rng = np.random.RandomState(0)
    sample_img = (rng.rand(320, 320, 3) * 255).astype(np.uint8)

    det = trained_xai.detect(sample_img, conf_threshold=0.01, iou_threshold=0.5)
    if not det.detections:
        pytest.skip("no detections on synthetic random image at this threshold")
    d = det.detections[0]

    expl_trained = trained_xai.explain(sample_img, d, method="gradcam")
    if expl_trained.status != "success":
        pytest.skip("trained-model explanation not successful on synthetic image")

    scale_idx, _ = trained_xai._flat_idx_to_scale_cell(d.pre_nms_index)
    iso = scv2.load_isolated_copy(str(CHECKPOINT), device="cpu")
    scv2.randomize_classification_branch_inplace(iso, scale_idx=scale_idx, seed=357)

    orig_score = scv2.original_score_for_instance(trained_xai, sample_img, d.class_id, d.pre_nms_index)
    assert orig_score is not None

    layer_name = f"model.model[{__import__('detection_xai').TARGET_LAYER_INDICES[scale_idx]}]"
    result = scv2.evaluate_classification_branch_sanity_check(
        iso, sample_img, expl_trained.heatmap_raw, orig_score,
        d.class_id, d.pre_nms_index, layer_name, "gradcam", "synthetic__inst000",
    )
    assert result.status == "valid"
    assert result.randomized_score is not None
    # genuineness: randomized-model CAM must come from a differentiable
    # graph through RANDOMIZED weights -- heatmap should differ from the
    # trained-model one (not byte-identical, i.e. not a stale/cached reuse)
    assert result.similarity_vs_trained is None or result.similarity_vs_trained < 0.999999


def test_unrecognized_target_layer_reported_not_crashed():
    iso = scv2.load_isolated_copy(str(CHECKPOINT), device="cpu")
    scv2.randomize_classification_branch_inplace(iso, scale_idx=0, seed=357)
    r = scv2.evaluate_classification_branch_sanity_check(
        iso, np.zeros((64, 64, 3), dtype=np.uint8),
        np.zeros((64, 64), dtype=np.float32), 0.5, 0, 0,
        "model.model[999]", "gradcam", "bad_layer__inst000",
    )
    assert r.status == "unrecognized_target_layer"


def test_invalid_scale_idx_raises():
    iso = scv2.load_isolated_copy(str(CHECKPOINT), device="cpu")
    with pytest.raises(ValueError):
        scv2.randomize_classification_branch_inplace(iso, scale_idx=3, seed=357)
