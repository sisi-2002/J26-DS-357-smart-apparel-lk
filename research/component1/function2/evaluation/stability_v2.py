"""F2.4-B Task 3 -- corrected stability metric (audits and fixes a specific
gap found in F2.4-A's ``stability.py``/``common.spearman_corr``).

DOES NOT MODIFY ``stability.py`` or ``common.py`` (both remain F2.4-A
outputs, read-only per Hard Restrictions). This module REUSES every
perturbation function, the retargeting/IoU policy, and the shared
``spearman_corr`` formula from those files UNCHANGED -- it adds exactly one
new guard, described below, and nothing else differs from v1.

WHAT WAS AUDITED (see ``reports/stability_discrepancy_audit.md`` for the
full before/after numbers and narrative): F2.4-A's pilot reported Grad-CAM
mean stability similarity 0.901 vs Grad-CAM++ 0.287 across the 9-instance
pilot. This stage re-derived that number from the raw
``f24_pilot_v1/metrics/stability_pilot.csv`` (confirmed: 0.9005 / 0.2873,
matches) and inspected per-instance heatmap sparsity directly
(``heatmaps/gradcam/*.npy`` for the 9 pilot instances):

    2 of the 9 instances' Grad-CAM heatmaps have >99% of pixels within 1% of
    the heatmap's own max value (99.44% and 99.89% respectively) -- i.e.
    the heatmap is a tiny nonzero blob on an otherwise near-flat background.
    ``common.spearman_corr``'s existing degenerate guard
    (``ra.std() < 1e-12``) only catches an EXACTLY constant array; it does
    NOT catch this near-constant/tie-dominated case, because the tiny
    nonzero blob gives the rank array nonzero (if small) variance.

WHY THIS MATTERS: Spearman correlation over a domain where >99% of pixels
are tied at (approximately) the same rank, in BOTH the original and
perturbed heatmap, is mechanically dominated by that enormous tied block --
the correlation can come out "high" almost independent of what happens to
the tiny informative region, which is exactly the "trivially high
similarity via a degenerate correlation" failure mode the F2.4-B brief
asked this audit to check for. ``stability.py`` has NO status value for
this (unlike ``localization.py``'s ``invalid_all_zero`` or
``pixel_iou.py``'s ``degenerate_heatmap``) -- every one of those 2
instances' Grad-CAM rows was silently counted as ordinary "valid" data in
F2.4-A's reported mean. This IS a genuine gap (missing degenerate-case
handling, inconsistent with this codebase's own established convention
elsewhere), not a hypothetical concern.

THE FIX (narrowly scoped, pre-specified BEFORE recomputing the mean -- see
audit doc for the exact threshold-selection reasoning): flag a heatmap as
``degenerate_low_information`` if fewer than 1% of its pixels reach >=1% of
its own max value (i.e. the "informative fraction" < 0.01). This threshold
was chosen by inspecting the ACTUAL distribution of informative-fraction
across all 18 (instance, method) pilot heatmaps once (not iteratively
tuned against the resulting mean): it cleanly separates the 2 genuinely
pathological cases (informative fraction 0.56% and 0.11%) from every other
heatmap in the pilot (next-lowest is 11.9%) -- a full order of magnitude
of separation, not an arbitrary cutoff sitting in a contested region.
Flagged rows get ``status='degenerate_low_information'``,
``similarity=None``, and are excluded from the headline mean but RETAINED
in accounting (never silently dropped).

RESULT OF THE AUDIT (see ``stability_discrepancy_audit.md`` for full
detail): after this fix, re-run on the SAME 9-instance pilot, Grad-CAM's
mean similarity moves from 0.9005 (n=52, v1) to 0.9160 (n=42, v2,
10 rows now excluded as degenerate_low_information) -- i.e. the gap does
NOT shrink after removing the 2 pathological instances; it is essentially
unchanged (marginally larger). **Conclusion: no code BUG was found that
inflates the gap artificially in the direction of making it look bigger
than it is -- a real degenerate-case-handling GAP was found and is fixed
here, but fixing it does not explain away the Grad-CAM-vs-Grad-CAM++
stability difference. The original F2.4-A ``stability.py`` is therefore
RETAINED UNCHANGED for the full 65-pair run's headline numbers (Task 7),
and this v2 module is run ALONGSIDE it as a sensitivity/robustness check,
not a replacement** -- both are reported in Task 7's output
(``metrics/stability.csv`` carries v1 columns; a separate
``metrics/stability_v2_degenerate_flagged.csv`` carries the v2
degenerate-aware re-annotation), per the brief's explicit instruction to
retain v1 for the full run when no result-altering bug is found, while
still fixing and reporting the narrower gap that genuinely exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from research.component1.function2.evaluation.common import (
    best_same_class_iou, spearman_corr,
)
from research.component1.function2.evaluation.stability import PERTURBATIONS
from research.component1.function2.xai.gt_matching import IOU_MATCH_THRESHOLD

LOW_INFORMATION_FRACTION_THRESHOLD = 0.01  # see module docstring for selection reasoning
NEAR_ZERO_RELATIVE_TO_MAX = 0.01


def informative_pixel_fraction(heatmap: np.ndarray) -> float:
    """Fraction of pixels at or above 1% of the heatmap's own max value.
    A heatmap with (nearly) all its mass in a handful of pixels on an
    otherwise flat near-zero field has a tiny informative fraction."""
    mx = float(heatmap.max())
    if mx <= 0:
        return 0.0
    return float((heatmap >= NEAR_ZERO_RELATIVE_TO_MAX * mx).mean())


def is_low_information_heatmap(heatmap: np.ndarray,
                                threshold: float = LOW_INFORMATION_FRACTION_THRESHOLD) -> bool:
    return informative_pixel_fraction(heatmap) < threshold


@dataclass
class StabilityResultV2:
    gt_instance_id: str
    method: str
    perturbation: str
    status: str  # "valid", "target_lost", "degenerate_low_information"
    matched_iou_vs_original: Optional[float]
    similarity: Optional[float]
    original_informative_fraction: Optional[float]
    perturbed_informative_fraction: Optional[float]
    reason: str = ""


def evaluate_stability_one_perturbation_v2(
    xai, explain_fn, image_bgr: np.ndarray, original_heatmap: np.ndarray,
    class_id: int, original_matched_bbox_xyxy, perturbation_name: str,
    gt_instance_id: str, method: str,
) -> StabilityResultV2:
    """Identical mechanism to ``stability.evaluate_stability_one_perturbation``
    (same perturbation functions, same re-detect + IoU>=0.5 retargeting
    policy, same inverse-alignment for pixel_shift, same Spearman formula)
    PLUS the one new guard described in the module docstring."""
    orig_frac = informative_pixel_fraction(original_heatmap)
    if orig_frac < LOW_INFORMATION_FRACTION_THRESHOLD:
        return StabilityResultV2(
            gt_instance_id, method, perturbation_name, "degenerate_low_information",
            None, None, round(orig_frac, 6), None,
            reason=f"Original heatmap's informative pixel fraction ({orig_frac:.4%}) is below "
                   f"the {LOW_INFORMATION_FRACTION_THRESHOLD:.0%} threshold -- heatmap is a tiny "
                   f"nonzero blob on an otherwise near-flat field; a rank correlation computed "
                   f"over this heatmap would be dominated by tied near-zero background pixels, "
                   f"not a meaningful comparison of the informative region.",
        )

    perturb_fn = PERTURBATIONS[perturbation_name]
    perturbed_img = perturb_fn(image_bgr)

    best_iou, best_det = best_same_class_iou(
        perturbed_img, xai, class_id, original_matched_bbox_xyxy,
    )
    if best_det is None or best_iou < IOU_MATCH_THRESHOLD:
        return StabilityResultV2(
            gt_instance_id, method, perturbation_name, "target_lost",
            round(max(best_iou, 0.0), 4), None, round(orig_frac, 6), None,
            reason=(f"Best same-class candidate IoU={max(best_iou, 0.0):.4f} against the "
                    f"ORIGINAL matched box is below {IOU_MATCH_THRESHOLD} after "
                    f"'{perturbation_name}' -- no longer treated as the same instance."),
        )

    expl = explain_fn(perturbed_img, best_det)
    if expl.status != "success" or expl.heatmap_raw is None:
        return StabilityResultV2(
            gt_instance_id, method, perturbation_name, "target_lost",
            round(best_iou, 4), None, round(orig_frac, 6), None,
            reason=f"Re-detected same instance (IoU={best_iou:.4f}) but explanation "
                   f"generation failed: {expl.status} / {expl.error_message}",
        )

    perturbed_heatmap = expl.heatmap_raw
    if perturbation_name == "pixel_shift_2px":
        from research.component1.function2.evaluation.stability import inverse_align_shifted_heatmap
        perturbed_heatmap = inverse_align_shifted_heatmap(perturbed_heatmap)

    if perturbed_heatmap.shape != original_heatmap.shape:
        return StabilityResultV2(
            gt_instance_id, method, perturbation_name, "target_lost",
            round(best_iou, 4), None, round(orig_frac, 6), None,
            reason=f"Heatmap shape mismatch after perturbation: "
                   f"{original_heatmap.shape} vs {perturbed_heatmap.shape}",
        )

    pert_frac = informative_pixel_fraction(perturbed_heatmap)
    if pert_frac < LOW_INFORMATION_FRACTION_THRESHOLD:
        return StabilityResultV2(
            gt_instance_id, method, perturbation_name, "degenerate_low_information",
            round(best_iou, 4), None, round(orig_frac, 6), round(pert_frac, 6),
            reason=f"Perturbed heatmap's informative pixel fraction ({pert_frac:.4%}) collapsed "
                   f"below the {LOW_INFORMATION_FRACTION_THRESHOLD:.0%} threshold post-perturbation.",
        )

    sim = spearman_corr(original_heatmap, perturbed_heatmap)
    return StabilityResultV2(
        gt_instance_id, method, perturbation_name, "valid",
        round(best_iou, 4), None if np.isnan(sim) else round(sim, 6),
        round(orig_frac, 6), round(pert_frac, 6),
    )
