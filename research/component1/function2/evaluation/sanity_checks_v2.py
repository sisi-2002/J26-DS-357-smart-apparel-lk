"""F2.4-B Task 2 -- classification-branch-only sanity check.

RESOLVES the sanity-check design question F2.4-A's pilot report explicitly
raised (readiness note #1 in ``f24_pilot_v1/reports/f24_pilot_report.md``,
section 7/14): the original ``sanity_checks.py`` whole-Detect-head
randomization (``model.model[22]``, BOTH the box-regression branch ``cv2``
AND the classification branch ``cv3``, plus the shared ``dfl`` decode)
invalidated 100% (18/18) of its pilot targets, because scrambling the box
branch destroys every box proposal's geometry, so no post-randomization
detection can ever re-match the original box at IoU>=0.5. That result is
architecturally expected and is KEPT, UNCHANGED, as a legitimate
"target-invalidation diagnostic" (see ``sanity_checks.py``, never edited by
this module) -- it correctly shows that scrambling box-regression destroys
localizability, which is itself informative. This module adds the SEPARATE
check the researcher asked for: randomize ONLY the classification branch,
so a similarity comparison becomes possible.

ARCHITECTURE -- independently reverified THIS STAGE, not assumed from
general YOLO knowledge, by instantiating the actual E02 checkpoint and
printing ``xai._model.model[22]``'s real module tree (ultralytics 8.4.175,
torch 2.14.1+cpu):

    model.model[22]  ->  ultralytics.nn.modules.head.Detect
        .cv2  ->  ModuleList of 3 Sequential BOX-REGRESSION branches
                  (one per detection scale P3/P4/P5), each
                  Conv(k3)->Conv(k3)->Conv2d(1x1, out_channels=64
                  = 16 DFL bins x 4 box sides). Input channels 64/128/256
                  for scale 0/1/2 respectively.
        .cv3  ->  ModuleList of 3 Sequential CLASSIFICATION branches
                  (one per scale), each Conv(k3)->Conv(k3)->Conv2d(1x1,
                  out_channels=7 = num_classes). Input channels 64/128/256
                  for scale 0/1/2 -- i.e. cv3[s] reads the SAME per-scale
                  feature map cv2[s] does, but is a completely disjoint set
                  of weights (verified: no parameter tensor is shared
                  between any cv2[*] and cv3[*] -- see
                  ``test_sanity_v2.py::test_cv2_cv3_share_no_parameters``).
        .dfl  ->  one shared DFL (distribution-focal-loss decode) module,
                  downstream of cv2 ONLY. Never reads cv3's output.

    ``detection_xai.TARGET_LAYER_INDICES = {0: 15, 1: 18, 2: 21}`` names the
    BACKBONE/NECK feature-producing layer Grad-CAM/Grad-CAM++ differentiate
    for each scale -- these are upstream of, and architecturally distinct
    from, model.model[22]'s cv2/cv3 split. Randomizing cv3[s] never touches
    model.model[15]/[18]/[21], so the Grad-CAM activation/gradient hooks
    this stage (and F2.3-B/F2.4-A before it) already use are unaffected by
    construction, other than through the one gradient path this check
    exists to perturb (the backward pass FROM the now-randomized class
    logit back through the shared backbone to the target layer).

    ``scores_all`` (the ``(1, nc, 8400)`` raw-logit tensor every F2.4
    faithfulness/sanity metric reads via ``common.frozen_target_score``) is
    exactly the per-scale concatenation of cv3[s]'s own output -- so
    randomizing cv3[scale_idx] ONLY, for the scale_idx matching an
    instance's actual matched target scale, changes EXACTLY the class
    logits that scale's grid cells produce, while the box-decode path
    (cv2 + dfl, hence every predicted box's geometry at every scale) is
    mathematically bit-for-bit untouched. This is a clean, empirically
    verified separation (see ``test_sanity_v2.py`` isolation tests), not an
    approximation or a hopeful assumption.

WHY THIS CHECK USES A FROZEN TARGET INDEX (NOT re-detect-and-IoU-rematch
like the whole-head check does):

Because cv2/dfl are untouched, the predicted BOX at any grid cell is
identical before/after this randomization -- the index space itself never
moves. The ``(class_id, pre_nms_index)`` frozen target (the SAME mechanism
``common.frozen_target_score`` and ``faithfulness.py`` already use) is
therefore a well-defined, shape-valid tensor location in effectively 100%
of cases -- not because the semantic detection is preserved, but because
geometry is architecturally guaranteed not to move. Re-running full
``detect()``+NMS+IoU-rematch here (the whole-head check's mechanism) would
reintroduce exactly the same confound the whole-head check has: NMS
survival depends on CONFIDENCE too, and confidence is exactly what this
check randomizes, so a naive full re-match would still invalidate a large
and architecturally-uninteresting fraction of instances for reasons that
have nothing to do with whether the EXPLANATION changed. Using the frozen
index is therefore the correct, not merely convenient, choice for this
specific randomization -- it isolates "did the explanation at this
architecturally-unmoved location change" from "did this cell additionally
survive NMS at the pipeline's default confidence threshold", and reports
the latter separately as a secondary diagnostic (see
``redetect_same_class_iou`` / ``redetect_survives_conf_threshold`` below),
never as a gate on whether a row is reported.

WHAT THIS DOES AND DOES NOT CLAIM (read before interpreting any number this
module produces -- restated in every report that cites it):

  - This is a TENSOR-INDEX match, not a re-verified semantic detection. It
    does NOT confirm that, after randomization, this class is still the
    argmax class at this cell, nor that this cell's post-randomization
    confidence would survive NMS at the pipeline's own 0.25 threshold, nor
    that no OTHER cell would now out-compete it at NMS time. The additional
    ``redetect_*`` fields answer exactly that question, separately, with
    their own (typically much lower) validity rate -- they are a weaker-N
    supplementary diagnostic, not part of the primary gate.
  - It IS a stronger spatial claim than a bare "the array index still
    exists" tautology: because box-regression is untouched, index
    ``pre_nms_index`` denotes the literal SAME physical image region
    (identical decoded box) before and after randomization -- "the
    explanation changed at this same real location" is a meaningful
    sentence here, in a way it is NOT for the whole-head check (where the
    box itself can move or vanish, making "same location" undefined for
    ~100% of instances, which is exactly why that check reports a 100%
    invalidation rate instead of a similarity distribution).
  - It does NOT establish that Grad-CAM/Grad-CAM++ are being probed "in
    total causal isolation from the box branch" in some idealized sense --
    the shared backbone/neck still has gradients flowing from the
    randomized classification head back through it, and the box-regression
    branch's own (unrandomized) existence is not causally inert during
    training-time feature learning (it already shaped the backbone weeks
    before this check ever ran). What this check isolates is the ONE
    thing it can cleanly isolate POST HOC on a frozen checkpoint: which
    DOWNSTREAM HEAD BRANCH's weights are scrambled at evaluation time.

Isolation guarantee (same regression test pattern as ``sanity_checks.py``):
every call here operates on a freshly-loaded, independent ``DetectionXAI``
instance; nothing in this module ever calls ``torch.save`` or otherwise
touches the on-disk checkpoint -- a hard regression test
(``test_sanity_v2.py::test_e02_checkpoint_byte_unchanged_after_v2_sanity``)
asserts the on-disk file's SHA256 is identical before/after.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_XAI_SRC = _REPO_ROOT / "research" / "component1" / "xai_experiments" / "src"
if str(_XAI_SRC) not in sys.path:
    sys.path.insert(0, str(_XAI_SRC))
from detection_xai import DetectionXAI, TARGET_LAYER_INDICES  # noqa: E402

from research.component1.function2.evaluation.common import (  # noqa: E402
    best_same_class_iou, frozen_target_score, spearman_corr, TOTAL_CELLS,
)
from research.component1.function2.evaluation.sanity_checks import (  # noqa: E402
    RANDOMIZATION_SEED, load_isolated_copy,
)
from research.component1.function2.xai.gt_matching import IOU_MATCH_THRESHOLD  # noqa: E402

CLASSIFICATION_BRANCH_SEED = RANDOMIZATION_SEED  # 357, same shared protocol seed
_LAYER_TO_SCALE = {f"model.model[{v}]": k for k, v in TARGET_LAYER_INDICES.items()}


def randomize_classification_branch_inplace(xai: DetectionXAI, scale_idx: int,
                                             seed: int = CLASSIFICATION_BRANCH_SEED) -> dict:
    """Reinitialize ONLY ``model.model[22].cv3[scale_idx]``'s parameters (the
    classification-branch conv stack for ONE detection scale), in place, on
    an already-isolated in-memory model. Leaves ``cv2`` (box regression,
    every scale), ``dfl``, and the OTHER two scales' ``cv3`` entries
    completely untouched. Per-tensor Gaussian reinit matched to each
    tensor's own (mean, std) -- identical scheme to
    ``sanity_checks.randomize_layer_inplace``, applied to a narrower module
    scope.
    """
    if scale_idx not in (0, 1, 2):
        raise ValueError(f"scale_idx must be 0, 1, or 2 (P3/P4/P5 respectively); got {scale_idx}")
    detect_head = xai._model.model[22]
    branch = detect_head.cv3[scale_idx]
    gen = torch.Generator().manual_seed(seed + scale_idx)  # distinct-but-deterministic per-scale stream
    n_params = 0
    for p in branch.parameters():
        mean = float(p.data.mean())
        std = float(p.data.std()) if p.data.numel() > 1 else 1.0
        std = std if std > 1e-8 else 1.0
        noise = torch.randn(p.data.shape, generator=gen)
        p.data = (noise * std + mean).to(p.data.dtype)
        n_params += p.data.numel()
    return {
        "randomized_branch": f"model.model[22].cv3[{scale_idx}]",
        "scale_idx": scale_idx,
        "n_parameters_randomized": n_params,
        "seed": seed + scale_idx,
        "box_branch_untouched": "model.model[22].cv2 (all 3 scales) + model.model[22].dfl",
        "other_scales_cv3_untouched": [s for s in (0, 1, 2) if s != scale_idx],
    }


def _frozen_index_randomized_explain(xai: DetectionXAI, image_bgr: np.ndarray,
                                      class_id: int, pre_nms_index: int, method: str) -> dict:
    """Grad-CAM/Grad-CAM++ at a FROZEN ``(class_id, pre_nms_index)`` cell on
    a (classification-branch-randomized) model -- same preprocessing, hook
    mechanism, and CAM formulas as ``detection_xai.DetectionXAI.explain()``
    and ``sanity_checks.py``, retargeted to not require NMS re-verification
    (see module docstring for why that is the correct mechanism for THIS
    randomization specifically)."""
    orig_shape = image_bgr.shape[:2]
    input_tensor, ratio, dw, dh = xai._preprocess(image_bgr)
    input_tensor = input_tensor.to(xai._device)
    input_tensor.requires_grad_(True)

    all_activations, all_gradients, hooks = {}, {}, []
    for s in range(3):
        li = TARGET_LAYER_INDICES[s]
        layer = xai._model.model[li]

        def _make(scale):
            def fwd(m, i, o):
                all_activations[scale] = o

            def bwd(m, gi, go):
                all_gradients[scale] = go[0]
            return fwd, bwd

        f, b = _make(s)
        hooks.append(layer.register_forward_hook(f))
        hooks.append(layer.register_full_backward_hook(b))

    try:
        xai._model.zero_grad()
        output = xai._model(input_tensor)
        if not isinstance(output, tuple):
            raise RuntimeError("Expected tuple output from model in eval mode")
        _decoded, raw_preds = output
        preds = raw_preds["one2many"] if isinstance(raw_preds, dict) and "one2many" in raw_preds else raw_preds
        scores_all = preds["scores"]
        nc, n_cells = scores_all.shape[1], scores_all.shape[2]
        if not (0 <= class_id < nc) or not (0 <= pre_nms_index < n_cells) or n_cells != TOTAL_CELLS:
            return {"status": "target_invalid_shape"}

        target_logit = scores_all[0, class_id, pre_nms_index]
        if not torch.isfinite(target_logit):
            return {"status": "nan_or_inf"}
        if not target_logit.requires_grad or target_logit.grad_fn is None:
            return {"status": "grad_missing",
                    "reason": "target_logit is not part of a differentiable autograd graph"}

        randomized_score = float(torch.sigmoid(target_logit.detach()).item())
        target_logit.backward(retain_graph=False)

        scale_idx, _cell = xai._flat_idx_to_scale_cell(pre_nms_index)
        if all_gradients.get(scale_idx) is None:
            return {"status": "grad_missing",
                    "reason": "no gradient reached the target Grad-CAM layer's activation "
                              "(backward hook never fired)"}

        act = all_activations[scale_idx].detach()
        grad = all_gradients[scale_idx].detach()
        if not torch.all(torch.isfinite(grad)):
            return {"status": "nan_or_inf", "reason": "non-finite gradient at target layer"}

        if method == "gradcam":
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        else:
            grad2, grad3 = grad ** 2, grad ** 3
            sum_act = act.sum(dim=(2, 3), keepdim=True) + 1e-7
            alpha = grad2 / (2 * grad2 + sum_act * grad3 + 1e-7)
            weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))

        cam_np = cam.squeeze().cpu().numpy()
        cam_orig = xai._map_to_original(cam_np, orig_shape, ratio, dw, dh)
        return {"status": "valid", "heatmap": cam_orig, "score": randomized_score,
                "scale_idx": scale_idx}
    finally:
        for h in hooks:
            h.remove()
        if input_tensor.grad is not None:
            input_tensor.grad = None


@dataclass
class ClassificationBranchSanityResult:
    gt_instance_id: str
    method: str
    randomized_branch: str
    scale_idx: int
    status: str  # "valid", "target_invalid_shape", "nan_or_inf", "grad_missing", "shape_mismatch", "unrecognized_target_layer"
    original_score: Optional[float]
    randomized_score: Optional[float]
    score_delta: Optional[float]
    similarity_vs_trained: Optional[float]
    redetect_same_class_iou: Optional[float]
    redetect_survives_conf_threshold: Optional[bool]
    reason: str = ""


def evaluate_classification_branch_sanity_check(
    isolated_xai: DetectionXAI, image_bgr: np.ndarray, original_heatmap: np.ndarray,
    original_score: float, class_id: int, pre_nms_index: int, target_layer: str,
    method: str, gt_instance_id: str, original_matched_bbox_xyxy=None,
    conf_threshold: float = 0.25, iou_threshold: float = 0.5,
) -> ClassificationBranchSanityResult:
    """``isolated_xai`` must already have had EXACTLY ONE scale's ``cv3``
    randomized by ``randomize_classification_branch_inplace`` before this is
    called (the scale matching ``target_layer``, enforced below by refusing
    to proceed if ``target_layer`` doesn't resolve to a known scale).
    ``original_heatmap`` and ``original_score`` (the TRAINED model's
    ``sigmoid(target_logit)`` at this exact frozen cell -- computed via
    ``common.frozen_target_score`` / ``original_score_for_instance`` on the
    UNRANDOMIZED model, BEFORE calling this) are both supplied explicitly so
    no call here can accidentally read a randomized-model score as if it
    were the trained baseline. This function never re-resolves the target
    itself -- it reuses the SAME frozen ``(class_id, pre_nms_index)``
    F2.3-B/F2.4-A already settled on for this instance.
    """
    scale_idx = _LAYER_TO_SCALE.get(target_layer)
    if scale_idx is None:
        return ClassificationBranchSanityResult(
            gt_instance_id, method, "unknown", -1, "unrecognized_target_layer",
            round(original_score, 6) if original_score is not None else None,
            None, None, None, None, None,
            reason=f"Unrecognized target_layer '{target_layer}'")

    branch_name = f"model.model[22].cv3[{scale_idx}]"
    result = _frozen_index_randomized_explain(isolated_xai, image_bgr, class_id,
                                               pre_nms_index, method)
    if result["status"] != "valid":
        return ClassificationBranchSanityResult(
            gt_instance_id, method, branch_name, scale_idx, result["status"],
            round(original_score, 6) if original_score is not None else None,
            None, None, None, None, None,
            reason=result.get("reason", result["status"]))

    randomized_score = result["score"]
    randomized_heatmap = result["heatmap"]
    if randomized_heatmap.shape != original_heatmap.shape:
        return ClassificationBranchSanityResult(
            gt_instance_id, method, branch_name, scale_idx, "shape_mismatch",
            round(original_score, 6), round(randomized_score, 6), None, None, None, None,
            reason=f"shape mismatch: original {original_heatmap.shape} vs "
                   f"randomized {randomized_heatmap.shape}")

    sim = spearman_corr(original_heatmap, randomized_heatmap)

    redetect_iou, redetect_survives = None, None
    if original_matched_bbox_xyxy is not None:
        best_iou, best_det = best_same_class_iou(
            image_bgr, isolated_xai, class_id, original_matched_bbox_xyxy,
            conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        )
        redetect_iou = round(max(best_iou, 0.0), 4)
        redetect_survives = bool(best_det is not None and best_iou >= IOU_MATCH_THRESHOLD)

    return ClassificationBranchSanityResult(
        gt_instance_id=gt_instance_id, method=method, randomized_branch=branch_name,
        scale_idx=scale_idx, status="valid",
        original_score=round(original_score, 6), randomized_score=round(randomized_score, 6),
        score_delta=round(randomized_score - original_score, 6),
        similarity_vs_trained=None if np.isnan(sim) else round(sim, 6),
        redetect_same_class_iou=redetect_iou,
        redetect_survives_conf_threshold=redetect_survives,
    )


def original_score_for_instance(trained_xai: DetectionXAI, image_bgr: np.ndarray,
                                 class_id: int, pre_nms_index: int) -> Optional[float]:
    """Convenience wrapper: trained-model sigmoid score at the frozen cell,
    via the SAME shared helper every other F2.4 metric uses (no parallel
    reimplementation)."""
    r = frozen_target_score(trained_xai, image_bgr, class_id, pre_nms_index)
    return r["score"] if r["status"] == "valid" else None
