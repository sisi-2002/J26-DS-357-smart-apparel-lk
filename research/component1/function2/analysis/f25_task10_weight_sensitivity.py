"""F2.5 Task 10 -- complexity-scoring weight-sensitivity check.

Checks whether the 60 sampled images' complexity-group labels (and hence
the broad XAI-vs-complexity-group patterns in Tasks 4-9) would look
qualitatively different under an alternative weighting scheme.

Honest limitation, checked here rather than assumed: `weight_sensitivity_v2.csv`
(the F2.1B3 authoritative alternative-weighting file) is a TRAIN-split
artifact (1747 TRAIN-eligible images; matches complexity_protocol_snapshot's
recorded TRAIN provenance exactly). The 60 XAI-sampled images are drawn from
the VALID split. This script explicitly checks for any image_id overlap
between the two (expected near-zero, confirmed below) before concluding
that NO per-image alternative-weighting label is reproducibly available for
the 60 sampled images from this file. The best available VALID-side
evidence is F2.2B's own aggregate VALID-side diagnostic
(f22b_task6_distribution_audit.json): 37.3% of 354 VALID-eligible images
are stable across all 7 schemes (vs TRAIN's 30.6%) -- an aggregate
population statistic, NOT a per-image label for the 60 sampled images. No
alternative label is fabricated for any of the 60 images.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT, load_all, write_json


def run():
    d = load_all()
    primary = d["xai_primary_sample"]
    train_weights = d["weight_sensitivity_train"]
    valid_audit = d["f22b_task6_distribution_audit"]

    primary_ids = {r["image_id"] for r in primary}
    train_ids = {r["image_id"] for r in train_weights}
    overlap = primary_ids & train_ids

    out = {
        "n_primary_sample_images": len(primary_ids),
        "n_train_weight_sensitivity_images": len(train_ids),
        "n_overlap_image_ids": len(overlap),
        "overlap_sample": sorted(overlap)[:10],
        "conclusion_on_per_image_alternative_labels": (
            "ZERO overlap between the 60 XAI-sampled (VALID-split) images and the 1747-image "
            "TRAIN-split weight_sensitivity_v2.csv file."
            if len(overlap) == 0 else
            f"{len(overlap)} of the 60 sampled images DO have a reproducible alternative-weighting "
            "label in weight_sensitivity_v2.csv -- used below."
        ),
    }

    if overlap:
        per_image = {}
        tw_by_id = {r["image_id"]: r for r in train_weights}
        n_moved = 0
        for gid in overlap:
            r = tw_by_id[gid]
            baseline = r["group_baseline_equal"]
            moved = r["stable_across_all_weightings"] != "True"
            if moved:
                n_moved += 1
            per_image[gid] = {"baseline_group": baseline, "stable_across_all_7": not moved}
        out["per_image_alternative_labels"] = per_image
        out["n_of_overlap_that_move_under_at_least_one_scheme"] = n_moved
    else:
        out["per_image_alternative_labels"] = None

    # best available VALID-side aggregate evidence (not per-image, documented honestly as such)
    wsd = valid_audit.get("weighting_sensitivity_diagnostic_valid_descriptive_only", {}) if valid_audit else {}
    out["valid_side_aggregate_diagnostic_population_level_not_per_image"] = {
        "n_eligible_valid_images_compared": wsd.get("n_eligible_valid_images_compared"),
        "fraction_stable_across_all_7_schemes": wsd.get("fraction_stable"),
        "per_alternative_scheme_vs_baseline": wsd.get("per_alternative_scheme_vs_baseline"),
        "train_side_comparison_fraction_stable": 0.3057,
        "note": "This is a population-level statistic over 354 VALID-eligible images, NOT a "
                "per-image label for the specific 60 images XAI-sampled. It is the best honestly "
                "available evidence of how sensitive the broader VALID-side complexity labeling "
                "is to the weighting choice; it cannot say which of the 60 sampled images "
                "specifically would move group, or whether THIS STUDY's headline Simple/"
                "Moderate/High findings would change in direction under an alternative scheme.",
    }

    out["honest_limitation_statement"] = (
        "Full, reproducible alternative-weighting group labels are NOT available for all 60 "
        "selected images from any file this stage located (weight_sensitivity_v2.csv covers only "
        "TRAIN-split images; no equivalent per-image VALID-split alternative-label file exists in "
        "the repository at this stage). This is disclosed honestly rather than fabricating "
        "per-image alternative labels for the 60-image sample. The qualitative conclusion carried "
        "forward is indirect: BOTH splits (TRAIN 30.6% stable, VALID 37.3% stable) show roughly "
        "two-thirds of eligible images changing complexity group under at least one of the 7 "
        "tested weighting schemes, so it is PLAUSIBLE (not verified for this specific 60-image "
        "sample) that a meaningful fraction of the 60 sampled images would be relabeled under an "
        "alternative weighting -- this is reported as a standing, unresolved sensitivity concern "
        "(see scientific_limitations.md), not as a quantified per-image robustness check of "
        "Tasks 4-9's specific findings."
    )

    write_json(F25_OUT / "sensitivity" / "weight_sensitivity_check.json", out)
    return out


if __name__ == "__main__":
    out = run()
    print(out["conclusion_on_per_image_alternative_labels"])
    print(out["valid_side_aggregate_diagnostic_population_level_not_per_image"]["fraction_stable_across_all_7_schemes"])
