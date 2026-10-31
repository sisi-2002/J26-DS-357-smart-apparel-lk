"""F2.2A Task 6 -- pure sampling-protocol logic (DESIGN ONLY).

This module implements the *constraint logic* proposed in
``reports/f22a_task6_sampling_protocol_proposal.md`` as plain, dependency-
free functions, so the logic can be unit-tested against synthetic data
(F2.2A Task 9). It is deliberately NEVER called against any real dataset
record in this stage -- F2.2A does not select, output, or freeze any
actual complexity-stratified image list. Calling this against real data is
explicitly reserved for a LATER stage (F2.2B or beyond), once complexity
calibration has actually been decided (per F2.1B3's panel-defense
recommendation #4).

Two independent pieces of logic are provided, matching the task brief's
explicit two-part design:
  1. ``apply_family_cap`` -- a deterministic "at most K images per
     source-family (``group_id``) per stratum" selection rule.
  2. ``class_composition_report`` -- a SEPARATE reporting step (not a
     selection rule) that summarises defect-class composition of whatever
     set ``apply_family_cap`` (or any other selector) produced, so a
     reviewer can distinguish a complexity effect from a defect-class
     effect. Per F2.1B3 Task 4's explicit finding, the family cap ALONE
     does not fix class-composition confounding (classes are not evenly
     distributed across families either) -- this function exists
     specifically so that fact is always visible, never silently dropped.
"""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from typing import Hashable, Optional, Sequence


@dataclasses.dataclass(frozen=True)
class SamplingCandidate:
    """One candidate record for stratified sampling.

    Deliberately generic / dataset-agnostic: this module does not import
    anything from ``manifest_loader`` or any real-data module, so it has
    zero risk of being accidentally pointed at real image paths.
    """
    record_id: str
    family_id: Hashable
    stratum: Hashable
    defect_classes: tuple[str, ...] = ()


def apply_family_cap(
    candidates: Sequence[SamplingCandidate],
    k: int,
    *,
    tie_break: str = "record_id",
) -> dict:
    """Deterministically select at most ``k`` candidates per
    (stratum, family_id) pair.

    Within a (stratum, family_id) group, candidates are sorted by
    ``record_id`` (lexicographic, matching ``manifest_loader``'s existing
    "sort by image_id, no RNG" deterministic-sampling convention) and the
    first ``k`` are kept. This function does not care what a "stratum"
    actually represents (complexity group, defect class, anything) -- it
    is pure selection logic over whatever grouping the caller provides.

    Returns a dict with:
      selected: list[SamplingCandidate]   kept candidates, deterministic order
      dropped: list[SamplingCandidate]    candidates cut by the cap
      per_group_counts: {(stratum, family_id): {"available": n, "kept": n}}
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if tie_break != "record_id":
        raise ValueError("only 'record_id' tie-break is currently implemented")

    groups: dict[tuple, list[SamplingCandidate]] = defaultdict(list)
    for c in candidates:
        groups[(c.stratum, c.family_id)].append(c)

    selected: list[SamplingCandidate] = []
    dropped: list[SamplingCandidate] = []
    per_group_counts: dict = {}

    for key, group_candidates in groups.items():
        ordered = sorted(group_candidates, key=lambda c: c.record_id)
        kept = ordered[:k]
        cut = ordered[k:]
        selected.extend(kept)
        dropped.extend(cut)
        per_group_counts[key] = {"available": len(ordered), "kept": len(kept)}

    # Deterministic overall ordering: by stratum, then family_id, then record_id.
    selected.sort(key=lambda c: (str(c.stratum), str(c.family_id), c.record_id))
    dropped.sort(key=lambda c: (str(c.stratum), str(c.family_id), c.record_id))

    return {
        "selected": selected,
        "dropped": dropped,
        "per_group_counts": per_group_counts,
        "k_applied": k,
        "n_candidates_in": len(candidates),
        "n_selected": len(selected),
        "n_dropped": len(dropped),
    }


def class_composition_report(candidates: Sequence[SamplingCandidate]) -> dict:
    """Report defect-class composition of a candidate set, broken down by
    stratum. This is a REPORTING step, not a selection rule -- it never
    drops or reweights any candidate. It exists specifically because
    F2.1B3 Task 4 found that complexity group and defect class are
    confounded in this dataset (e.g. 'stain' dominates the Simple TRAIN
    group, 'cut' dominates High) -- a family cap alone does not fix this,
    since classes are not evenly distributed across families either.
    Running this report alongside any stratified selection lets a reviewer
    see and account for this confound rather than have it silently hidden.
    """
    by_stratum: dict[Hashable, Counter] = defaultdict(Counter)
    images_per_stratum: Counter = Counter()
    for c in candidates:
        images_per_stratum[c.stratum] += 1
        for cls in c.defect_classes:
            by_stratum[c.stratum][cls] += 1

    report = {}
    for stratum, class_counts in by_stratum.items():
        n_images = images_per_stratum[stratum]
        report[stratum] = {
            "n_images_in_stratum": n_images,
            "class_instance_counts": dict(class_counts),
            "class_image_share_pct": {
                cls: round(100.0 * n / n_images, 1) if n_images else 0.0
                for cls, n in class_counts.items()
            },
        }
    # Strata with zero defect classes recorded (e.g. all-negative strata)
    for stratum, n_images in images_per_stratum.items():
        report.setdefault(stratum, {
            "n_images_in_stratum": n_images,
            "class_instance_counts": {},
            "class_image_share_pct": {},
        })
    return report
