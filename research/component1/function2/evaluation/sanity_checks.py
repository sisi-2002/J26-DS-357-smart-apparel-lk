"""F2.4-A Task 6 -- sanity checks (parameter-randomization test).

Implements a BOUNDED version of the cascading parameter-randomization
sanity check from Adebayo et al. ("Sanity Checks for Saliency Maps", 2018):
reinitialize one or more layers' weights, starting from the layer closest to
the model's output and working backward, and check whether the explanation
changes. An explanation that stays nearly identical after the WEIGHTS that
produced it have been scrambled is a red flag (it suggests the method is
not actually sensitive to what the model learned).

Bounded-scope justification (explicit, per Task 6's own fallback clause):
this stage's pilot covers only ~9 instances x 2 methods. The full Adebayo
cascade (randomize every layer, one at a time, output->input, for every
instance) multiplies the already-nontrivial per-instance explain() cost by
the number of layers in a YOLOv8n backbone+head (dozens) -- appropriate for
F2.4-B's full run, not justified for a PILOT whose job (Task 9) is to prove
the MECHANISM is correct (isolation, target-validity handling, similarity
computation, no silent NaN), not to produce a publishable randomization
curve. This module therefore randomizes ONLY the single closest-to-output
block, ``model.model[22]`` (the YOLOv8 ``Detect`` head -- the literal final
layer whose output IS the class logit Grad-CAM/Grad-CAM++ differentiate),
with a second, slightly-further-back layer (``model.model[21]``, the P5 C2f
block that is also one of the three Grad-CAM target layers) available via
the same function for instances whose target layer IS 21, so that at least
one data point exists where the RANDOMIZED layer and the EXPLAINED layer
are the same block. The two layers are randomized INDEPENDENTLY (two
separate isolated model copies, each with exactly one layer scrambled) --
this is a bounded two-point probe, not a cumulative/cascading
randomization (true Adebayo-style cascading, where layer 21 would be
randomized ON TOP of an already-randomized layer 22, is explicitly deferred
to F2.4-B, see ``reports/f24_pilot_report.md``).

Isolation guarantee: every call here operates on a FRESH ``DetectionXAI``
instance constructed by this module (a new ``YOLO(weights_path)`` load),
never the instance used elsewhere for the trained-model explanations, and
mutates only in-memory ``torch.nn.Parameter`` tensors. Nothing in this
module ever calls ``torch.save`` or otherwise writes to the checkpoint path
-- a hard regression test
(``tests/test_f24_metrics.py::test_e02_checkpoint_byte_unchanged_after_sanity``)
asserts the on-disk file's SHA256 is identical before and after.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_XAI_SRC = _REPO_ROOT / "research" / "component1" / "xai_experiments" / "src"
if str(_XAI_SRC) not in sys.path:
    sys.path.insert(0, str(_XAI_SRC))
from detection_xai import DetectionXAI  # noqa: E402

from research.component1.function2.evaluation.common import (
    best_same_class_iou, spearman_corr,
)
from research.component1.function2.xai.gt_matching import IOU_MATCH_THRESHOLD

RANDOMIZATION_SEED = 357
CASCADE_LAYER_INDICES = [22, 21]  # output-first order: Detect head, then P5 C2f


def load_isolated_copy(checkpoint_path: str, device: str = "cpu") -> DetectionXAI:
    """A brand-new, independent in-memory model instance. Never reuses or
    shares tensors with any other loaded ``DetectionXAI`` -- a fresh
    ``YOLO(weights_path)`` load every call."""
    return DetectionXAI(str(checkpoint_path), device=device)


def randomize_layer_inplace(xai: DetectionXAI, layer_index: int,
                             seed: int = RANDOMIZATION_SEED) -> dict:
    """Reinitialize every parameter tensor of ``xai._model.model[layer_index]``
    in place, IN MEMORY ONLY, with fixed-seed Gaussian noise matched to each
    original parameter's own (mean, std) -- this randomizes the learned
    values while keeping each tensor's overall scale plausible (avoiding the
    degenerate case of e.g. BatchNorm weights becoming implausibly huge/zero
    and producing NaNs purely from scale mismatch, which would confound the
    sanity check with an unrelated numerical-stability failure).
    """
    layer = xai._model.model[layer_index]
    gen = torch.Generator().manual_seed(seed)
    n_params = 0
    for p in layer.parameters():
        mean = float(p.data.mean())
        std = float(p.data.std()) if p.data.numel() > 1 else 1.0
        std = std if std > 1e-8 else 1.0
        noise = torch.randn(p.data.shape, generator=gen)
        p.data = (noise * std + mean).to(p.data.dtype)
        n_params += p.data.numel()
    return {"layer_index": layer_index, "layer_repr": f"model.model[{layer_index}]",
            "n_parameters_randomized": n_params, "seed": seed}


@dataclass
class SanityCheckResult:
    gt_instance_id: str
    method: str
    randomized_layer: str
    status: str  # "valid", "target_invalid_after_randomization"
    matched_iou_vs_original: Optional[float]
    similarity_vs_trained: Optional[float]
    reason: str = ""


def evaluate_sanity_check(
    isolated_xai: DetectionXAI, image_bgr: np.ndarray, original_heatmap: np.ndarray,
    class_id: int, original_matched_bbox_xyxy, method: str,
    gt_instance_id: str, randomized_layer_index: int,
) -> SanityCheckResult:
    """Re-detect on the SAME (unperturbed) image through the randomized
    in-memory model (the forward pass itself changes once weights change,
    so re-detection -- not a frozen target -- is the only coherent choice:
    the frozen-target mechanism assumes the SAME model produced the index,
    which is no longer true post-randomization). If the randomized model no
    longer produces a same-class detection overlapping the original matched
    box at IoU>=0.5, this is recorded as ``target_invalid_after_randomization``
    rather than forced onto an unrelated detection -- expected to happen for
    at least some instances, since randomizing the output layer can and
    often does destroy the model's ability to detect anything coherent at
    that location, which is itself part of what this check is diagnosing.
    """
    layer_name = f"model.model[{randomized_layer_index}]"
    best_iou, best_det = best_same_class_iou(
        image_bgr, isolated_xai, class_id, original_matched_bbox_xyxy,
    )
    if best_det is None or best_iou < IOU_MATCH_THRESHOLD:
        return SanityCheckResult(
            gt_instance_id, method, layer_name, "target_invalid_after_randomization",
            round(max(best_iou, 0.0), 4), None,
            reason=(f"After randomizing {layer_name}, best same-class IoU against the "
                    f"original matched box is {max(best_iou, 0.0):.4f} (< {IOU_MATCH_THRESHOLD}) "
                    f"-- target no longer meaningful, not force-compared."),
        )

    expl = isolated_xai.explain(image_bgr, best_det, method=method)
    if expl.status != "success" or expl.heatmap_raw is None:
        return SanityCheckResult(
            gt_instance_id, method, layer_name, "target_invalid_after_randomization",
            round(best_iou, 4), None,
            reason=f"Re-detected (IoU={best_iou:.4f}) but explanation generation "
                   f"failed post-randomization: {expl.status}/{expl.error_message}",
        )

    randomized_heatmap = expl.heatmap_raw
    if randomized_heatmap.shape != original_heatmap.shape:
        return SanityCheckResult(
            gt_instance_id, method, layer_name, "target_invalid_after_randomization",
            round(best_iou, 4), None,
            reason=f"Heatmap shape mismatch post-randomization: "
                   f"{original_heatmap.shape} vs {randomized_heatmap.shape}",
        )

    sim = spearman_corr(original_heatmap, randomized_heatmap)
    return SanityCheckResult(gt_instance_id, method, layer_name, "valid",
                              round(best_iou, 4), None if np.isnan(sim) else round(sim, 6))
