"""F2.5 Task 3 -- freeze the analysis plan honestly, BEFORE Tasks 4-11 compute
anything new.

This stage is EXPLORATORY, not confirmatory/preregistered in the classical
sense: F2.4-B already computed and published (in a prior, separately-gated
stage) every method/group summary number this analysis re-examines --
pointing-accuracy hit rates, mean energy fractions, faithfulness AUCs,
stability similarities, the full complexity-group breakdown. Those numbers
were already known before this stage's detailed family-level and
confound-stratified comparisons were run. Nothing in this stage should be
presented as a prospective/blind hypothesis confirmation -- it is a
secondary, more careful (family-aware, confound-stratified) look at
already-seen numbers.

Per the brief: any inferential statistic must be pre-registered in THIS file
(written before being computed), must disclose multiplicity, and must never
be read as causal evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT, SEED, write_json
from research.component1.function2.analysis.f25_bootstrap import N_RESAMPLES, MIN_FAMILIES_FOR_CI

PLAN = {
    "stage": "F2.5",
    "classification": "EXPLORATORY -- NOT confirmatory, NOT preregistered-before-first-look. "
                       "Every headline method/group number re-examined here (pointing-accuracy "
                       "hit rates, energy fractions, faithfulness AUCs, stability similarities, "
                       "per-group breakdowns) was already computed and reported by F2.4-B before "
                       "this stage began. F2.5 re-expresses those same numbers at the "
                       "source-family level with uncertainty intervals and confound stratification "
                       "-- it is a secondary descriptive/exploratory pass, not a blind prospective test.",

    "independence_unit": {
        "primary_reporting_and_inference_unit": "source_family_id",
        "justification": "Per statistical_analysis_plan.md Sec.2 and the research brief: GT "
                          "instances from the same image/family share image-level confounds "
                          "(lighting, fabric batch, acquisition session). Family-first aggregation "
                          "(mean within family, then mean/bootstrap across families) is used for "
                          "every headline comparison in Tasks 4/5/9. Per-instance numbers are "
                          "reported alongside as a secondary, clearly-labeled lens (matching "
                          "F2.4-B's own convention), never as the headline N.",
        "n_distinct_families_in_65_paired_pool": "computed in task2_unit_definitions_summary.json, "
                                                  "NOT assumed equal to 60 or 65 a priori",
    },

    "inferential_approach": {
        "method": "Family-level block bootstrap percentile confidence intervals on (a) each "
                  "method's family-first mean for each metric, and (b) the family-level paired "
                  "(Grad-CAM minus Grad-CAM++) difference, where the pairing is well-defined "
                  "(same family, same instances, both methods).",
        "no_null_hypothesis_significance_testing": "This stage does NOT compute a p-value or a "
            "formal null-hypothesis test (e.g. a paired t-test or permutation p-value) anywhere. "
            "Rationale, decided BEFORE any Task 4-11 number was computed: (1) several strata "
            "(per-layer, per-defect-class, High-group families) have very small effective family "
            "counts (see the sparsity cutoff below), where a p-value would be fragile and "
            "liable to be over-read as a binary decision; (2) the brief explicitly asks this "
            "stage to 'prefer effect sizes + uncertainty intervals + missingness reporting over "
            "bare significance claims' and to avoid 'opportunistic significance testing to get a "
            "favorable result' -- committing to effect-size-and-CI-only, decided here before "
            "computation, removes the opportunity to pick a test post hoc because a p-value "
            "looked favorable. Effect sizes (family-first mean differences) with percentile "
            "bootstrap CIs are reported instead, everywhere inference is reported at all.",
        "multiplicity_disclosure": "This stage computes family-level bootstrap CIs for multiple "
            "metrics (pointing-accuracy hit rate, energy fraction, insertion AUC, deletion AUC, "
            "stability v1 similarity, stability v2 similarity) across multiple strata (overall, "
            "x3 complexity groups, x3 target layers where non-sparse, x defect classes where "
            "non-sparse). The EXACT count of CIs actually computed and reported is written to "
            "sensitivity/multiplicity_disclosure.json by the tasks that compute them -- every one "
            "is disclosed, none is cherry-picked or suppressed for being 'uninteresting', and no "
            "single CI is interpreted as a stand-alone confirmed finding independent of the "
            "others.",
        "causal_language_prohibition": "No finding in any F2.5 output describes a complexity "
            "group, a target layer, or a defect class as CAUSING a metric difference. Phrasing "
            "throughout is restricted to descriptive/associational forms (e.g. 'Grad-CAM++ "
            "achieved a higher family-first mean energy fraction in this pool, under this "
            "checkpoint and this sample') per statistical_analysis_plan.md Sec.4.",
    },

    "bootstrap_protocol": {
        "resampling_unit": "source_family_id (whole family resampled together, preserving all "
                            "instances x both methods within a resampled family)",
        "seed": SEED,
        "n_resamples": N_RESAMPLES,
        "n_resamples_justification": "See f25_bootstrap.py module docstring: Monte-Carlo error on "
                                      "a percentile bound at this resample count is small relative "
                                      "to the effective family counts in this pool (14-48 per "
                                      "group, fewer per stratified cell), and two independent runs "
                                      "at the same seed are required (and tested) to reproduce an "
                                      "identical CI.",
        "ci_construction": "percentile (2.5th / 97.5th percentile of the resampled family-first "
                            "means)",
        "sparsity_cutoff_for_ci": {
            "min_families_required": MIN_FAMILIES_FOR_CI,
            "rule": "A stratum (e.g. a single target layer within a single complexity group, or a "
                    "single defect class within a group) with FEWER than this many distinct "
                    "families contributing a valid value for a given metric is reported with "
                    "descriptive statistics ONLY (family-first mean/median/min/max) -- no "
                    "bootstrap CI is generated or shown for it.",
            "justification": "A percentile interval built from under 5 independent resampling "
                              "units does not estimate a meaningful interval (the bootstrap "
                              "distribution itself becomes a near-discrete relabeling of 2-4 "
                              "points) -- reporting a numeric-looking CI in that situation would "
                              "overstate precision that the data does not support.",
        },
    },

    "metric_directionality_conventions": {
        "pointing_accuracy_hit_rate": "higher = better (heatmap argmax falls inside the GT box more often)",
        "gt_box_energy_fraction": "no universally agreed better/worse direction in isolation -- it "
            "measures CONCENTRATION of heatmap energy inside the GT box, not correctness; reported "
            "descriptively, not framed as higher-is-better without qualification",
        "box_area_fraction_of_image": "a property of the GT box / image, not of the explanation "
            "method -- reported as a baseline/context column only, never as a method score",
        "insertion_auc": "higher = better (confidence recovers faster as the most-important pixels, "
                          "per the heatmap, are inserted)",
        "deletion_auc": "LOWER = better (confidence should drop faster as the most-important pixels "
                         "are deleted first) -- stated explicitly per the brief's requirement, since "
                         "this is the opposite direction convention from insertion AUC and from "
                         "pointing accuracy",
        "random_control_auc": "a shared-per-instance baseline curve (same random order for both "
            "methods on a given instance), not a method score -- used only as a sanity reference, "
            "never compared as if it were a competing method",
        "stability_similarity_v1_and_v2": "higher = better (heatmap more ROBUST to the six small "
            "nuisance perturbations) -- this is a robustness measure only, NOT a correctness or "
            "faithfulness measure; a method can be perfectly stable while being spatially wrong",
        "sanity_classification_branch_validity_rate": "higher = better evidence the mechanism "
            "works as designed (NOT evidence either method is more correct)",
    },

    "pixel_iou_separation": "Per statistical_analysis_plan.md Sec.6, reaffirmed here: the pixel-IoU "
        "pool (9 reference masks, 2-instance intersection with the 65-pair pool, 1 instance "
        "evaluated, IoU=0.0) is reported entirely separately (Task 13) and is NEVER pooled with, "
        "extrapolated from, or used to adjust any number in Tasks 4-11.",

    "what_this_plan_forbids_for_this_stage": [
        "Choosing a different independence unit, bootstrap design, or sparsity cutoff AFTER seeing "
        "which direction a given comparison comes out.",
        "Computing a p-value or NHST test anywhere in this stage.",
        "Reporting a bootstrap CI for any stratum below the declared family-count sparsity cutoff.",
        "Describing any complexity-group, target-layer, or defect-class association as causal.",
        "Pooling the pixel-IoU pilot with the primary 65-pair localization/faithfulness/stability "
        "results in either direction.",
        "Re-deriving or adjusting the complexity group labels, thresholds, or weights based on "
        "any XAI metric result (the complexity protocol remains frozen and provisional).",
    ],
}


if __name__ == "__main__":
    out = F25_OUT / "config" / "f25_preregistered_analysis_plan.json"
    write_json(out, PLAN)
    print("Written:", out)
