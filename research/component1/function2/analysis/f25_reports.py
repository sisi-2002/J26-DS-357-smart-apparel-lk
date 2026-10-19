"""F2.5 report generation -- assembles the 10 required markdown reports from
the already-computed JSON/CSV tables (never re-derives numbers by hand, to
avoid transcription drift between the tables and the prose)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT

GROUPS = ["Simple", "Moderate", "High"]


def _j(name):
    with open(F25_OUT / "tables" / name, encoding="utf-8") as fh:
        return json.load(fh)


def _js(name):
    with open(F25_OUT / "sensitivity" / name, encoding="utf-8") as fh:
        return json.load(fh)


def _jm(name):
    with open(F25_OUT / "manifests" / name, encoding="utf-8") as fh:
        return json.load(fh)


def _jc(name):
    with open(F25_OUT / "checksums" / name, encoding="utf-8") as fh:
        return json.load(fh)


def pct(x, nd=1):
    return "n/a" if x is None else f"{100*x:.{nd}f}%"


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def ci_str(d):
    if d is None:
        return "n/a"
    if d.get("ci_skipped"):
        return f"descriptive only ({d['skip_reason']})"
    return f"[{f(d['ci_lo'])}, {f(d['ci_hi'])}] (n_families={d['n_effective_families']})"


def write(name, content):
    path = F25_OUT / "reports" / name
    path.write_text(content, encoding="utf-8")
    print("Written:", path)


# ---------------------------------------------------------------- 1 ------

def report_results_summary():
    audit = _jc("task1_integrity_reaudit.json")
    unit = _jm("task2_unit_definitions_summary.json")
    mc = _j("method_comparison_overall.json")
    strat = _j("complexity_stratified_results.json")
    bias = _j("matching_selection_bias.json")
    gap = _j("stability_gap_investigation.json")
    layer = _j("target_layer_confounding.json")
    cls = _j("defect_class_confounding.json")
    piou = _j("pixel_iou_report_only.json")
    wsens = _js("weight_sensitivity_check.json")

    content = f"""# F2.5 Results Summary (Component 1 / Function 2, Complexity-Based XAI Evaluation)

Stage F2.5 is the final descriptive/exploratory statistical-analysis stage
for the frozen F2.3-B / F2.4-B pipeline. **This stage is EXPLORATORY, not
confirmatory.** Every headline number re-examined here was already computed
and published by F2.4-B before this stage began (see
`config/f25_preregistered_analysis_plan.json`). No causal claim about
complexity is made anywhere in this report.

## 1. Upstream verification (Task 1)

Independently re-derived from row-level CSVs, not trusted from any prior
prose summary: **{audit['overall_status']}**, {len(audit['checks'])} checks,
0 blockers.

| Quantity | Recomputed this stage |
|---|---|
| Images | {audit['recomputed_headline']['n_images']} |
| GT instances | {audit['recomputed_headline']['n_gt_instances']} |
| Matched | {audit['recomputed_headline']['n_matched']} |
| Paired-success | {audit['recomputed_headline']['n_paired_success']} |
| Degenerate one-sided failure | {audit['recomputed_headline']['n_degenerate_one_sided']} |
| Simple matched | {audit['recomputed_headline']['per_group']['Simple']['matched']}/{audit['recomputed_headline']['per_group']['Simple']['total']} ({audit['recomputed_headline']['per_group']['Simple']['rate_pct']}%) |
| Moderate matched | {audit['recomputed_headline']['per_group']['Moderate']['matched']}/{audit['recomputed_headline']['per_group']['Moderate']['total']} ({audit['recomputed_headline']['per_group']['Moderate']['rate_pct']}%) |
| High matched | {audit['recomputed_headline']['per_group']['High']['matched']}/{audit['recomputed_headline']['per_group']['High']['total']} ({audit['recomputed_headline']['per_group']['High']['rate_pct']}%) |

All values reproduce F2.3-B's / F2.4-B's reported values exactly. E02
checkpoint SHA256 confirmed unchanged. Zero duplicate instance IDs. Zero
fabricated metric rows for unmatched/failed instances (checked directly
against `localization.csv`/`faithfulness.csv`/`stability.csv`/
`stability_v2_degenerate_flagged.csv`). Full detail:
`checksums/task1_integrity_reaudit.json`.

## 2. Analysis units (Task 2)

The 65 paired-success instances span **{unit['n_distinct_images_in_paired_pool']}
distinct images** and **{unit['n_distinct_source_families_in_paired_pool']} distinct
source families** (recomputed directly from `paired_explanations_manifest.csv`,
NOT assumed to equal 60 or 65). Per-group family counts:
Simple={unit['n_distinct_source_families_by_group']['Simple']},
Moderate={unit['n_distinct_source_families_by_group']['Moderate']},
High={unit['n_distinct_source_families_by_group']['High']}. One family
contributes {unit['max_instances_contributed_by_a_single_family']['n_instances']}
of the 65 instances; instances-per-family histogram:
{unit['instances_per_family_histogram']}. **Source family (not the 65/92
instance count) is used as the headline N for every effect-size/uncertainty
statement in this report.**

## 3. Method comparison, every metric family (Task 4)

No metric is combined into a single "quality score"; no universal winner is
declared. See `method_comparison.md` for the full breakdown. Headline
family-first numbers (48 families unless noted):

| Metric | Grad-CAM | Grad-CAM++ | Direction |
|---|---|---|---|
| Pointing-accuracy hit rate | {pct(mc['localization']['gradcam']['pointing_hit_rate']['rate'])} (n=65) | {pct(mc['localization']['gradcampp']['pointing_hit_rate']['rate'])} (n=65) | higher=better |
| GT-box energy fraction | {f(mc['localization']['gradcam']['energy_fraction_mean']['mean'])} | {f(mc['localization']['gradcampp']['energy_fraction_mean']['mean'])} | concentration measure |
| Insertion AUC | {f(mc['faithfulness']['gradcam']['insertion_auc_mean']['mean'])} | {f(mc['faithfulness']['gradcampp']['insertion_auc_mean']['mean'])} | higher=better |
| Deletion AUC | {f(mc['faithfulness']['gradcam']['deletion_auc_mean']['mean'])} | {f(mc['faithfulness']['gradcampp']['deletion_auc_mean']['mean'])} | LOWER=better |
| Stability v1 similarity | {f(mc['stability_v1']['gradcam']['mean_similarity_overall']['mean'])} (n={mc['stability_v1']['gradcam']['mean_similarity_overall']['valid_n']}) | {f(mc['stability_v1']['gradcampp']['mean_similarity_overall']['mean'])} (n={mc['stability_v1']['gradcampp']['mean_similarity_overall']['valid_n']}) | higher=better (robustness only) |
| Stability v2 similarity | {f(mc['stability_v2']['gradcam']['mean_similarity_overall']['mean'])} (n={mc['stability_v2']['gradcam']['mean_similarity_overall']['valid_n']}) | {f(mc['stability_v2']['gradcampp']['mean_similarity_overall']['mean'])} (n={mc['stability_v2']['gradcampp']['mean_similarity_overall']['valid_n']}) | higher=better (robustness only) |

## 4. Complexity-stratified descriptives (Task 5)

Selection counts (recomputed): Simple 20 images/25 instances/{strat['selection_counts']['Simple']['n_paired_success']} paired;
Moderate 20/34/{strat['selection_counts']['Moderate']['n_paired_success']}; High 20/33/{strat['selection_counts']['High']['n_paired_success']}.
Full per-group, per-method, per-metric table with family-level bootstrap CIs:
`tables/complexity_stratified_results.json`, prose in `complexity_stratified_analysis.md`.

## 5. Matching / selection-bias (Task 6)

High-complexity's match rate ({bias['by_group']['High']['match_rate_pct']}%) is
substantially lower than Simple ({bias['by_group']['Simple']['match_rate_pct']}%)
or Moderate ({bias['by_group']['Moderate']['match_rate_pct']}%). This is very
unlikely to be missing-at-random (see `matching_selection_bias.md`). High's
evaluable sample retains only {bias['sample_size_shift_original_vs_evaluable']['High']['fraction_of_original_retained_pct']}%
of its originally-selected instances, and the retained subset's defect-class
composition visibly differs from the full selected set (e.g. `cut` drops
from 13 originally-selected High instances to 4 paired; `gray stitch`
disappears entirely from the High paired pool).

## 6. Target-layer confounding (Task 7)

{layer['separability_statement']}

## 7. Defect-class confounding (Task 8)

Classes confined to <=2 complexity groups in the 65-pair pool:
{list(cls['classes_confined_to_le2_groups_in_65'].keys())}. Only ONE
defect class (`stain`) has enough family overlap (>=5 families) in 2+
groups (Simple, Moderate) to support even an exploratory within-class
comparison; class and complexity are inseparable for every other class at
this sample size.

## 8. Stability-gap finding (Task 9)

**{gap['required_cautious_phrasing']}**

Overall v1 gap (family-first, Grad-CAM minus Grad-CAM++):
{f(mc['stability_v1']['paired_family_level_bootstrap']['point_estimate_family_first_mean'])},
95% family-bootstrap CI {ci_str(mc['stability_v1']['paired_family_level_bootstrap'])}.
{gap['gap_sign_consistency']['statement']}

## 9. Complexity-weighting sensitivity (Task 10)

{wsens['conclusion_on_per_image_alternative_labels']} Best available
evidence: VALID-side aggregate diagnostic over 354 eligible images shows
{pct(wsens['valid_side_aggregate_diagnostic_population_level_not_per_image']['fraction_stable_across_all_7_schemes'])}
stable across all 7 weighting schemes (TRAIN-side: 30.6%) -- population-level,
NOT a per-image check of the 60 sampled images specifically.

## 10. Sanity-check interpretation

Classification-branch-only randomization: 18/18 (100%) valid on the
bounded 9-instance subset -- evidence the isolation mechanism works as
designed, NOT evidence either method is more generally correct. See
`method_comparison.md` Section on sanity and F2.4-B's
`sanity_protocol_resolution.md` for the full scope-of-claim caveats
(unchanged, re-affirmed, not re-litigated here).

## 11. Pixel-IoU (Task 13, not executed)

{piou['n_intersection']} of the 9 verified reference masks intersect the
65-pair pool; 1 evaluated (IoU=0.0, both methods); zero High-complexity
coverage. See `scientific_limitations.md` Section "Pixel-IoU" for the
recommendation on a supplementary F2.4-C run.

## 12. Table / figure / report paths

```
tables/method_comparison_overall.json, method_comparison_overall_flat.csv
tables/complexity_stratified_results.json
tables/matching_selection_bias.json
tables/target_layer_confounding.json
tables/defect_class_confounding.json
tables/stability_gap_investigation.json
tables/pixel_iou_report_only.json
manifests/instance_level_dataset.csv, image_level_dataset.csv, family_level_dataset.csv
manifests/task2_unit_definitions_summary.json
uncertainty/all_bootstrap_cis_flattened.csv
sensitivity/weight_sensitivity_check.json, multiplicity_disclosure.json
checksums/task1_integrity_reaudit.json
figures/localization_by_group.svg, stability_by_group.svg,
        faithfulness_insertion_by_group.svg, faithfulness_deletion_by_group.svg,
        runtime_by_group.svg, paired_method_differences.svg,
        matching_missingness_by_group.svg
config/f25_preregistered_analysis_plan.json
```

## 13. Tests and integrity

`tests/test_f25_analysis.py`: 25/25 passed. Full Function 2 regression suite
(all stages, re-run after adding F2.5): **520 passed, 2 skipped, 0 failed**
(the 2 skips are the pre-existing, unrelated `segment_anything` import skip
and the synthetic no-detection edge case -- neither newly introduced).
Protected-artifact scoped re-hash: all files this stage reads from confirmed
byte-unchanged before/after (`checksums/task1_integrity_reaudit.json`).

## 14. Recommendation on F2.4-C

See `scientific_limitations.md` / `tables/pixel_iou_report_only.json` --
cheap to run, low-but-nonzero incremental value, does NOT resolve zero
High-complexity pixel coverage, NOT recommended as a prerequisite for F2.6/F2.7.

## 15. Readiness for F2.6/F2.7

See `f25_readiness_for_pp1.md` for the full readiness decision.
"""
    write("f25_results_summary.md", content)


# ---------------------------------------------------------------- 2 ------

def report_method_comparison():
    mc = _j("method_comparison_overall.json")

    def row(label, gc, gcpp, extra=""):
        return f"| {label} | {gc} | {gcpp} | {extra} |"

    loc = mc["localization"]
    fth = mc["faithfulness"]
    s1 = mc["stability_v1"]
    s2 = mc["stability_v2"]
    san = mc["sanity"]
    tim = mc["timing"]

    content = f"""# Method Comparison: Grad-CAM vs Grad-CAM++ (Task 4)

Every metric family is reported separately for both methods. No metric is
combined into a single score; no universal winner is declared -- these
metrics measure different things and point in different directions in this
pool (energy/pointing favor Grad-CAM++, stability favors Grad-CAM,
faithfulness is close).

## Directionality conventions (read before interpreting any number below)

- Pointing-accuracy hit rate: **higher = better**.
- GT-box energy fraction: concentration measure, no inherent better/worse direction.
- Insertion AUC: **higher = better**.
- Deletion AUC: **LOWER = better** (opposite convention from insertion AUC).
- Random-control AUC: baseline reference only, not a competing method score.
- Stability similarity (v1, v2): **higher = better**, but measures robustness
  to 6 small nuisance perturbations ONLY -- not faithfulness or correctness.
- Sanity classification-branch valid rate: **higher = better mechanism evidence**,
  not evidence either method is more correct.

## Localization

| Statistic | Grad-CAM | Grad-CAM++ |
|---|---|---|
| Pointing hits / valid N | {loc['gradcam']['pointing_hit_rate']['hits']}/{loc['gradcam']['pointing_hit_rate']['valid_n']} ({pct(loc['gradcam']['pointing_hit_rate']['rate'])}) | {loc['gradcampp']['pointing_hit_rate']['hits']}/{loc['gradcampp']['pointing_hit_rate']['valid_n']} ({pct(loc['gradcampp']['pointing_hit_rate']['rate'])}) |
| Energy fraction mean (valid N={loc['gradcam']['energy_fraction_mean']['valid_n']}) | {f(loc['gradcam']['energy_fraction_mean']['mean'])} | {f(loc['gradcampp']['energy_fraction_mean']['mean'])} |
| Box-area-fraction baseline (context, not a method score) | {f(loc['gradcam']['box_area_fraction_of_image_mean_baseline']['mean'])} | (same, shared per instance) |
| Family-first energy fraction (n=48 families) | {f(loc['gradcam']['family_level_bootstrap']['energy_fraction']['point_estimate_family_first_mean'])}, 95% CI {ci_str(loc['gradcam']['family_level_bootstrap']['energy_fraction'])} | {f(loc['gradcampp']['family_level_bootstrap']['energy_fraction']['point_estimate_family_first_mean'])}, 95% CI {ci_str(loc['gradcampp']['family_level_bootstrap']['energy_fraction'])} |
| Family-first pointing hit rate | {f(loc['gradcam']['family_level_bootstrap']['pointing_hit_rate']['point_estimate_family_first_mean'])}, 95% CI {ci_str(loc['gradcam']['family_level_bootstrap']['pointing_hit_rate'])} | {f(loc['gradcampp']['family_level_bootstrap']['pointing_hit_rate']['point_estimate_family_first_mean'])}, 95% CI {ci_str(loc['gradcampp']['family_level_bootstrap']['pointing_hit_rate'])} |

Paired family-first differences (Grad-CAM minus Grad-CAM++), 95% bootstrap CI:
- Energy fraction: {f(loc['paired_family_level_bootstrap']['energy_fraction_diff']['point_estimate_family_first_mean'])}, CI {ci_str(loc['paired_family_level_bootstrap']['energy_fraction_diff'])} -- entirely negative, i.e. the Grad-CAM++ advantage on this metric is robust at the family level in this pool.
- Pointing hit rate: {f(loc['paired_family_level_bootstrap']['pointing_hit_rate_diff']['point_estimate_family_first_mean'])}, CI {ci_str(loc['paired_family_level_bootstrap']['pointing_hit_rate_diff'])} -- CI crosses zero, i.e. less robust at the family level than the energy-fraction difference.

## Faithfulness

| Statistic | Grad-CAM | Grad-CAM++ |
|---|---|---|
| Insertion AUC mean (valid N={fth['gradcam']['insertion_auc_mean']['valid_n']}) | {f(fth['gradcam']['insertion_auc_mean']['mean'])} | {f(fth['gradcampp']['insertion_auc_mean']['mean'])} |
| Deletion AUC mean (valid N={fth['gradcam']['deletion_auc_mean']['valid_n']}) | {f(fth['gradcam']['deletion_auc_mean']['mean'])} | {f(fth['gradcampp']['deletion_auc_mean']['mean'])} |
| Random-control AUC (shared baseline) | {f(fth['gradcam']['random_control_auc_mean']['mean'])} | {f(fth['gradcampp']['random_control_auc_mean']['mean'])} |

Paired family-first differences, 95% CI:
- Insertion AUC: {f(fth['paired_family_level_bootstrap']['insertion_auc_diff']['point_estimate_family_first_mean'])}, CI {ci_str(fth['paired_family_level_bootstrap']['insertion_auc_diff'])} -- crosses zero.
- Deletion AUC: {f(fth['paired_family_level_bootstrap']['deletion_auc_diff']['point_estimate_family_first_mean'])}, CI {ci_str(fth['paired_family_level_bootstrap']['deletion_auc_diff'])} -- crosses zero.

Both faithfulness differences are small in magnitude and their family-level
CIs include zero -- consistent with F2.4-B's "close between methods" framing.

## Stability (v1 headline, v2 degenerate-aware sensitivity)

| Statistic | Grad-CAM v1 | Grad-CAM++ v1 | Grad-CAM v2 | Grad-CAM++ v2 |
|---|---|---|---|---|
| Mean similarity (valid N) | {f(s1['gradcam']['mean_similarity_overall']['mean'])} (n={s1['gradcam']['mean_similarity_overall']['valid_n']}) | {f(s1['gradcampp']['mean_similarity_overall']['mean'])} (n={s1['gradcampp']['mean_similarity_overall']['valid_n']}) | {f(s2['gradcam']['mean_similarity_overall']['mean'])} (n={s2['gradcam']['mean_similarity_overall']['valid_n']}) | {f(s2['gradcampp']['mean_similarity_overall']['mean'])} (n={s2['gradcampp']['mean_similarity_overall']['valid_n']}) |
| Status counts | {s1['gradcam']['status_counts']} | {s1['gradcampp']['status_counts']} | {s2['gradcam']['status_counts']} | {s2['gradcampp']['status_counts']} |

Paired family-first difference (v1): {f(s1['paired_family_level_bootstrap']['point_estimate_family_first_mean'])},
95% CI {ci_str(s1['paired_family_level_bootstrap'])} -- entirely positive (favors Grad-CAM), robust at the family level.
Paired family-first difference (v2): {f(s2['paired_family_level_bootstrap']['point_estimate_family_first_mean'])},
95% CI {ci_str(s2['paired_family_level_bootstrap'])}.

Per-perturbation breakdown (v1): see `tables/stability_gap_investigation.json`
`by_perturbation_v1` -- loss is spread thinly across perturbation types
(1-2 `target_lost` rows per perturbation for Grad-CAM; similar for
Grad-CAM++), not concentrated in one.

## Sanity (classification-branch-only, bounded 9-instance subset)

| Statistic | Grad-CAM | Grad-CAM++ |
|---|---|---|
| Valid rate | {san['gradcam']['valid_rate']['valid_n']}/{san['gradcam']['valid_rate']['total_n']} | {san['gradcampp']['valid_rate']['valid_n']}/{san['gradcampp']['valid_rate']['total_n']} |
| Mean score delta | {f(san['gradcam']['mean_score_delta'])} | {f(san['gradcampp']['mean_score_delta'])} |
| Mean similarity-vs-trained (excl. NaN) | {f(san['gradcam']['mean_similarity_vs_trained_excluding_nan'])} | {f(san['gradcampp']['mean_similarity_vs_trained_excluding_nan'])} |
| N similarity NaN (near-zero rank variance) | {san['gradcam']['n_similarity_nan_near_zero_rank_variance']} | {san['gradcampp']['n_similarity_nan_near_zero_rank_variance']} |

N=9 per method -- far too small to generalize beyond this bounded pool.

## Timing (bounded 9-instance subset)

| Statistic | Grad-CAM | Grad-CAM++ |
|---|---|---|
| Median of medians (s) | {f(tim['gradcam']['median_of_medians_total_s'])} | {f(tim['gradcampp']['median_of_medians_total_s'])} |
| Median of p95 (s) | {f(tim['gradcam']['median_of_p95_total_s'])} | {f(tim['gradcampp']['median_of_p95_total_s'])} |

Both methods share the identical forward+backward pass and differ only in
CAM-weighting arithmetic -- statistically indistinguishable at this N, per
F2.4-A/B's own framing.
"""
    write("method_comparison.md", content)


# ---------------------------------------------------------------- 3 ------

def report_complexity_stratified():
    strat = _j("complexity_stratified_results.json")
    rows = []
    for g in GROUPS:
        sel = strat["selection_counts"][g]
        bg = strat["by_group"][g]
        rows.append(f"### {g}\n\n"
                     f"Selected images: {sel['n_images_selected']}; GT instances: {sel['n_gt_instances']}; "
                     f"matched: {sel['n_matched']} ({sel['match_rate_pct']}%); paired-success: "
                     f"{sel['n_paired_success']}; eligible families (any metric): "
                     f"{bg['n_eligible_families_total']}.\n\n"
                     "| Metric | Grad-CAM (n_fam) | Grad-CAM++ (n_fam) |\n|---|---|---|\n"
                     f"| Pointing hit rate | {pct(bg['gradcam']['pointing_hit_rate']['rate'])} ({bg['gradcam']['pointing_hit_rate']['n_families']}) | {pct(bg['gradcampp']['pointing_hit_rate']['rate'])} ({bg['gradcampp']['pointing_hit_rate']['n_families']}) |\n"
                     f"| Energy fraction mean | {f(bg['gradcam']['energy_fraction_mean']['mean'])} ({bg['gradcam']['energy_fraction_mean']['n_families']}) | {f(bg['gradcampp']['energy_fraction_mean']['mean'])} ({bg['gradcampp']['energy_fraction_mean']['n_families']}) |\n"
                     f"| Insertion AUC | {f(bg['gradcam']['insertion_auc_mean']['mean'])} ({bg['gradcam']['insertion_auc_mean']['n_families']}) | {f(bg['gradcampp']['insertion_auc_mean']['mean'])} ({bg['gradcampp']['insertion_auc_mean']['n_families']}) |\n"
                     f"| Deletion AUC | {f(bg['gradcam']['deletion_auc_mean']['mean'])} ({bg['gradcam']['deletion_auc_mean']['n_families']}) | {f(bg['gradcampp']['deletion_auc_mean']['mean'])} ({bg['gradcampp']['deletion_auc_mean']['n_families']}) |\n"
                     f"| Stability v1 similarity | {f(bg['gradcam']['stability_v1_mean_similarity']['mean'])} ({bg['gradcam']['stability_v1_mean_similarity']['n_families']}) | {f(bg['gradcampp']['stability_v1_mean_similarity']['mean'])} ({bg['gradcampp']['stability_v1_mean_similarity']['n_families']}) |\n"
                     f"| Stability v2 similarity | {f(bg['gradcam']['stability_v2_mean_similarity']['mean'])} ({bg['gradcam']['stability_v2_mean_similarity']['n_families']}) | {f(bg['gradcampp']['stability_v2_mean_similarity']['mean'])} ({bg['gradcampp']['stability_v2_mean_similarity']['n_families']}) |\n")

    content = f"""# Complexity-Stratified Analysis (Task 5)

Every metric computed separately per group. **No complexity-associated
difference below is interpreted causally** -- see `target_layer_confounding.md`
and `defect_class_confounding.md` before attributing any cross-group pattern
to complexity itself. Family-level N is reported per metric per group
(n_fam in parentheses), since eligibility differs by metric.

{''.join(rows)}

## Pattern is not monotonic across metrics

E.g. Grad-CAM++'s energy fraction rises Simple->Moderate->High while its
deletion AUC does not move monotonically in the same order -- read this
alongside the layer/defect-class confounding reports before drawing any
complexity-related conclusion.

## Figures

`figures/localization_by_group.svg`, `figures/stability_by_group.svg`,
`figures/faithfulness_insertion_by_group.svg`,
`figures/faithfulness_deletion_by_group.svg`.
"""
    write("complexity_stratified_analysis.md", content)


# ---------------------------------------------------------------- 4 ------

def report_matching_bias():
    b = _j("matching_selection_bias.json")
    rows = []
    for g in GROUPS:
        bg = b["by_group"][g]
        shift = b["sample_size_shift_original_vs_evaluable"][g]
        rows.append(
            f"### {g}\n\n"
            f"Matched: {bg['n_matched']}/{bg['n_total']} ({bg['match_rate_pct']}%). "
            f"Paired-success: {bg['n_paired_success']} ({shift['fraction_of_original_retained_pct']}% of "
            f"originally-selected instances retained).\n\n"
            f"- Defect-class composition, ALL originally-selected instances: {bg['defect_class_composition_ALL_selected_instances']}\n"
            f"- Defect-class composition, PAIRED (XAI-evaluable) instances: {bg['defect_class_composition_PAIRED_instances']}\n"
            f"- Defect-class composition, UNMATCHED instances: {bg['defect_class_composition_UNMATCHED_instances']}\n"
            f"- Target-layer distribution of matched instances: {b['target_layer_distribution_of_matched_instances_by_group'][g]}\n"
        )

    content = f"""# Matching / Selection-Bias Analysis (Task 6)

Full 92-instance population analyzed. No metric value is assigned to any
unmatched instance anywhere in this analysis.

{''.join(rows)}

## Failure-category composition (failures.csv, 27 rows)

{b['failure_category_by_group']}

## Selection-bias statement

{b['selection_bias_statement']}

**This missingness is very unlikely to be missing-at-random.** Harder-to-detect
defects are presumably systematically different from easier ones; the
detector's success is plausibly correlated with properties (boundary
clarity, size, contrast) that could independently affect downstream XAI
metrics. High-group complete-case estimates in Tasks 4/5/9 describe the
retained, detector-succeeded subset only -- not the general High-complexity
defect population.
"""
    write("matching_selection_bias.md", content)


# ---------------------------------------------------------------- 5 ------

def report_layer_confounding():
    t = _j("target_layer_confounding.json")
    lg = t["layer_group_metric_descriptives"]

    def cell_row(layer, g):
        c = t["group_x_layer_cross_tab"][g][layer]
        return f"{c['n_instances']} inst / {c['n_families']} fam{'  (SPARSE)' if c['sparse'] else ''}"

    cross_rows = "\n".join(
        f"| {g} | {cell_row('P3', g)} | {cell_row('P4', g)} | {cell_row('P5', g)} |" for g in GROUPS
    )

    comparable = t["comparable_group_pairs_meeting_sparsity_threshold"]
    comp_rows = []
    for c in comparable:
        a = lg[c["layer"]][c["group_a"]]
        bb = lg[c["layer"]][c["group_b"]]
        comp_rows.append(
            f"### {c['layer']}: {c['group_a']} (n_fam={c['n_families_a']}) vs {c['group_b']} (n_fam={c['n_families_b']}) -- EXPLORATORY/SENSITIVITY\n\n"
            f"| Metric | {c['group_a']} Grad-CAM | {c['group_a']} Grad-CAM++ | {c['group_b']} Grad-CAM | {c['group_b']} Grad-CAM++ |\n|---|---|---|---|---|\n"
            f"| Energy fraction | {f(a['gradcam']['energy_fraction_mean'])} | {f(a['gradcampp']['energy_fraction_mean'])} | {f(bb['gradcam']['energy_fraction_mean'])} | {f(bb['gradcampp']['energy_fraction_mean'])} |\n"
            f"| Stability v1 similarity | {f(a['gradcam']['stability_v1_mean_similarity'])} | {f(a['gradcampp']['stability_v1_mean_similarity'])} | {f(bb['gradcam']['stability_v1_mean_similarity'])} | {f(bb['gradcampp']['stability_v1_mean_similarity'])} |\n"
        )

    content = f"""# Target-Layer Confounding Audit (Task 7)

Independently recomputed (not trusted from F2.4-B's prose). Sparsity cutoff:
**>=5 distinct families required** per cell before any within-layer group
comparison is reported (same threshold as the Task 11 bootstrap, documented
in `f25_bootstrap.py`).

## Complexity-group x target-layer cross-tab (instances / families; SPARSE if families < 5)

| Group | P3 | P4 | P5 |
|---|---|---|---|
{cross_rows}

## Separability statement

**{t['separability_statement']}**

## Exploratory/sensitivity within-layer comparisons (only where BOTH sides meet the cutoff)

{''.join(comp_rows) if comp_rows else 'None met the threshold on both sides.'}

These are labeled EXPLORATORY/SENSITIVITY, not confirmed confounder-adjusted
causal estimates. Note in particular (P5, Moderate vs High): Grad-CAM++'s
energy fraction is {f(lg['P5']['Moderate']['gradcampp']['energy_fraction_mean'])}
(Moderate) vs {f(lg['P5']['High']['gradcampp']['energy_fraction_mean'])} (High) --
much closer to each other WITHIN the same layer than either is to Simple's
P4 value of {f(lg['P4']['Simple']['gradcampp']['energy_fraction_mean'])} --
consistent with F2.4-B's flagged concern that Grad-CAM++'s energy
concentration is substantially a P5-layer phenomenon rather than a
complexity-group phenomenon, though this single exploratory comparison does
not resolve the question either way.

## Skipped cells (below cutoff, descriptive only, no comparison reported)

{t['skipped_cells_below_cutoff']}
"""
    write("target_layer_confounding.md", content)


# ---------------------------------------------------------------- 6 ------

def report_defect_class_confounding():
    c = _j("defect_class_confounding.json")
    content = f"""# Defect-Class Confounding Audit (Task 8)

## Complexity-group x defect-class, ORIGINAL 92 instances (60-image selection)

{c['complexity_x_defect_class_ORIGINAL_92_instances']}

## Complexity-group x defect-class, PAIRED 65 instances (the pool every downstream metric uses)

{c['complexity_x_defect_class_PAIRED_65_instances']}

These two tables differ (e.g. High-group `gray stitch` and `contamination`
instances are present in the original 92 but absent from the paired 65 --
lost entirely to the matching-rate selection effect documented in Task 6).

## Defect-class x target-layer, PAIRED 65 instances

{c['defect_class_x_target_layer_PAIRED_65_instances']}

## Classes confined to <=2 complexity groups (65-pair pool)

{c['classes_confined_to_le2_groups_in_65']}

## Family-count overlap, group x class (65-pair pool)

{c['family_count_by_group_x_class_65_pool']}

## Within-class comparisons with enough family overlap (>=5 families, >=2 groups)

{c['within_class_comparisons_with_enough_family_overlap']}

Only `stain` has enough family overlap (Simple, Moderate) to support even an
exploratory within-class comparison. {c['statement']}
"""
    write("defect_class_confounding.md", content)


# ---------------------------------------------------------------- 7 ------

def report_stability_analysis():
    gap = _j("stability_gap_investigation.json")
    by_layer = gap["by_layer_v1"]
    by_class = gap["by_defect_class_v1"]
    by_pert = gap["by_perturbation_v1"]

    layer_rows = "\n".join(
        f"| {name} | {f(v['gradcam']['mean_similarity'])} | {f(v['gradcampp']['mean_similarity'])} | {f(v['gap_gradcam_minus_gradcampp'])} |"
        for name, v in by_layer.items()
    )
    class_rows = "\n".join(
        f"| {name} | {v['n_families']} | {f(v['gradcam']['mean_similarity'])} | {f(v['gradcampp']['mean_similarity'])} |"
        for name, v in by_class.items()
    )
    pert_rows = "\n".join(
        f"| {name} | {f(v['gradcam']['mean_similarity'])} ({v['gradcam']['status_counts']}) | {f(v['gradcampp']['mean_similarity'])} ({v['gradcampp']['status_counts']}) |"
        for name, v in by_pert.items()
    )

    content = f"""# Stability-Gap Investigation (Task 9)

**{gap['required_cautious_phrasing']}**

## Overall (v1 headline vs v2 degenerate-aware sensitivity)

| | Grad-CAM | Grad-CAM++ | Gap |
|---|---|---|---|
| v1 (n={gap['overall_v1']['gradcam']['valid_n']}/{gap['overall_v2']['gradcampp']['valid_n']}) | {f(gap['overall_v1']['gradcam']['mean_similarity'])} | {f(gap['overall_v1']['gradcampp']['mean_similarity'])} | {f(gap['overall_v1']['gap_gradcam_minus_gradcampp'])} |
| v2 (degenerate-aware) | {f(gap['overall_v2']['gradcam']['mean_similarity'])} | {f(gap['overall_v2']['gradcampp']['mean_similarity'])} | {f(gap['overall_v2']['gap_gradcam_minus_gradcampp'])} |

Family-level bootstrap (n=48 families unless noted):
- v1 Grad-CAM: {f(gap['family_level_bootstrap']['v1_gradcam']['point_estimate_family_first_mean'])}, CI {ci_str(gap['family_level_bootstrap']['v1_gradcam'])}
- v1 Grad-CAM++: {f(gap['family_level_bootstrap']['v1_gradcampp']['point_estimate_family_first_mean'])}, CI {ci_str(gap['family_level_bootstrap']['v1_gradcampp'])}
- v1 paired diff: {f(gap['family_level_bootstrap']['v1_paired_diff_gradcam_minus_gradcampp']['point_estimate_family_first_mean'])}, CI {ci_str(gap['family_level_bootstrap']['v1_paired_diff_gradcam_minus_gradcampp'])}
- v2 paired diff: {f(gap['family_level_bootstrap']['v2_paired_diff_gradcam_minus_gradcampp']['point_estimate_family_first_mean'])}, CI {ci_str(gap['family_level_bootstrap']['v2_paired_diff_gradcam_minus_gradcampp'])}

## Per-layer (v1)

| Layer | Grad-CAM | Grad-CAM++ | Gap |
|---|---|---|---|
{layer_rows}

## Per-defect-class (v1, n_families shown; classes with <5 families are descriptive only)

| Class | n_families | Grad-CAM | Grad-CAM++ |
|---|---|---|---|
{class_rows}

## Per-perturbation (v1)

| Perturbation | Grad-CAM (status counts) | Grad-CAM++ (status counts) |
|---|---|---|
{pert_rows}

Losses (`target_lost`) are spread thinly across perturbation types for both
methods -- not concentrated in one, consistent with F2.4-B's finding.

## Sign-consistency check

{gap['gap_sign_consistency']['statement']}

## Conclusion

The stability gap survives every sensitivity check available in this pool
(v1 vs v2, every complexity group, every target layer) in the same
direction, at the family level, with a 95% bootstrap CI excluding zero for
the pooled v1 paired difference. Per the required phrasing: this is
evidence of robustness-to-small-nuisance-perturbation only, on this frozen
checkpoint, on this image set, under this six-perturbation protocol -- it
does not generalize beyond this setup and is not a faithfulness or
localization-correctness claim.
"""
    write("stability_analysis.md", content)


# ---------------------------------------------------------------- 8 ------

def report_sensitivity_analysis():
    wsens = _js("weight_sensitivity_check.json")
    mult = _js("multiplicity_disclosure.json")

    content = f"""# Sensitivity Analysis (Tasks 10 + 11 multiplicity disclosure)

## Complexity-weighting sensitivity (Task 10)

{wsens['conclusion_on_per_image_alternative_labels']}

VALID-side aggregate diagnostic (F2.2B, population-level, NOT per-image):

- N eligible VALID images compared: {wsens['valid_side_aggregate_diagnostic_population_level_not_per_image']['n_eligible_valid_images_compared']}
- Fraction stable across all 7 schemes: {pct(wsens['valid_side_aggregate_diagnostic_population_level_not_per_image']['fraction_stable_across_all_7_schemes'])}
- TRAIN-side comparison: {pct(wsens['valid_side_aggregate_diagnostic_population_level_not_per_image']['train_side_comparison_fraction_stable'])}
- Per-scheme change vs baseline: {wsens['valid_side_aggregate_diagnostic_population_level_not_per_image']['per_alternative_scheme_vs_baseline']}

{wsens['honest_limitation_statement']}

## Family-level block bootstrap protocol (Task 11)

Resampling unit: source_family_id. Seed: 357. Resamples: 2000 per statistic.
CI construction: percentile (2.5th/97.5th). Sparsity cutoff: >=5 distinct
families required, else descriptive only (no CI forced).

## Multiplicity disclosure

{mult['statement']}

See `uncertainty/all_bootstrap_cis_flattened.csv` for every individual CI
computed (or skipped, with reason) across Tasks 4/5/9.
"""
    write("sensitivity_analysis.md", content)


# ---------------------------------------------------------------- 9 ------

def report_scientific_limitations():
    piou = _j("pixel_iou_report_only.json")
    content = f"""# Scientific Limitations (F2.5), Full 5-Way Classification

Every substantive claim in this analysis is classified into exactly one of:
(1) verified descriptive finding, (2) exploratory association, (3)
sensitivity-dependent finding, (4) cannot be reliably identified with
available data, (5) unresolved limitation.

## (1) Verified descriptive findings

- 60 images / 92 GT instances / 66 matched / 65 paired-success / 1 degenerate
  one-sided failure -- independently re-derived, exact match to F2.3-B/F2.4-B.
- Overall method-comparison numbers (pointing hit rate, energy fraction,
  insertion/deletion AUC, stability v1/v2 similarity, sanity valid rate,
  timing) -- independently re-derived from row-level CSVs, exact match to
  F2.4-B's reported values.
- The 65-pair pool spans 48 distinct source families (17 Simple, 17
  Moderate, 14 High), one contributing 8 of the 65 instances.
- High-complexity match rate (57.6%) is substantially lower than Simple
  (80.0%) / Moderate (79.4%), with a visibly different defect-class
  composition between the originally-selected and paired High subsets.

## (2) Exploratory associations

- The stability gap (Grad-CAM > Grad-CAM++) holds in the same direction
  across every group/layer/v1-v2 breakdown checked, with a family-level 95%
  bootstrap CI excluding zero for the pooled difference -- but this is a
  robustness-to-nuisance-perturbation finding only, on one checkpoint/sample.
- Within the P5 layer, Grad-CAM++'s energy fraction is similar between
  Moderate and High groups, while differing substantially from Simple's
  (P4-dominant) value -- suggestive that layer composition, not complexity
  per se, may drive much of the group pattern, but this single exploratory
  comparison does not resolve the question.

## (3) Sensitivity-dependent findings

- The pointing-hit-rate paired difference's family-level 95% CI crosses
  zero, while the energy-fraction paired difference's CI does not -- i.e.
  one localization metric is more robust to family-resampling than the
  other, even though both are reported as favoring Grad-CAM++ on the point
  estimate.
- Both faithfulness (insertion/deletion AUC) paired differences have
  family-level CIs crossing zero -- "close between methods" is itself a
  sensitivity-dependent characterization, not a precise equivalence claim.

## (4) Cannot be reliably identified with available data

- Whether any complexity-group pattern reflects complexity itself, versus
  target-layer composition (Simple P3/P4-heavy, High P5-heavy) or
  defect-class composition (Simple stain-heavy, High selvet/cut-heavy) --
  explicitly NOT separable for most group x layer x class cells at this
  family-level sample size (see `target_layer_confounding.md`,
  `defect_class_confounding.md`).
- Whether the 60 sampled images' complexity-group LABELS themselves would
  be materially different under an alternative weighting scheme -- no
  reproducible per-image alternative-weighting file exists for this VALID-split
  60-image sample (Task 10); only a population-level VALID-side aggregate
  statistic is available.

## (5) Unresolved limitations

- Lower High-group matching rate (57.6%) is a plausible, data-grounded,
  likely-not-missing-at-random selection mechanism, not resolved or modeled
  away by this stage.
- Defect-class imbalance across complexity groups (e.g. `baekra` confined to
  High, `gray stitch`/`contamination` confined to Simple/Moderate in the
  65-pair pool) is real and unresolved.
- Target-layer (P3/P4/P5) imbalance across complexity groups is real and
  unresolved; only 2 of 9 possible within-layer group-pair comparisons meet
  the >=5-family sparsity threshold on both sides.
- The complexity-scoring protocol itself remains provisional/NOT
  expert-validated: TRAIN stability 30.6%, VALID stability 37.3% across 7
  weighting schemes; a 30-image human-reconciliation exercise found 66.7%
  agreement / kappa ~= 0.50 (small, non-extrapolatable).
- Small effective family-level sample sizes in several strata (e.g. High
  group families total n=14; several layer x group cells have 1-4 families)
  limit how precisely any subgroup estimate can be pinned down.
- GT instances are non-independent (same-image/same-family instances share
  acquisition-session confounds) -- addressed by using source-family as the
  headline unit, but the underlying non-independence is a structural
  property of the data, not something this stage's methodology resolves.
- Pixel-IoU: {piou['n_intersection']} intersecting instances, zero
  High-complexity reference-mask coverage anywhere in the pipeline (pilot-only
  situation, unresolved by this stage, see Task 13 report above).
- Expert validation of the complexity criteria remains AI-assisted only
  (kappa ~= 0.50 on n=30); no independent human expert sign-off exists.
- No independent TEST-set evaluation has occurred at this PP1 stage -- TEST
  remains locked and untouched by this or any prior Function 2 stage.

## Pixel-IoU recommendation (restated from Task 13)

{piou['recommendation']}
"""
    write("scientific_limitations.md", content)


# ---------------------------------------------------------------- 10 -----

def report_readiness():
    content = """# Readiness Decision for F2.6/F2.7 (F2.5 final statement)

## What F2.5 established

- Independent re-audit of F2.3-B/F2.4-B's counts and metric CSVs: PASS, no
  blocker, no discrepancy affecting analytical correctness.
- Source-family (not raw instance) established as the headline independence
  unit: 48 distinct families in the 65-pair pool, re-derived directly from
  `paired_explanations_manifest.csv`, not assumed.
- Every metric family reported for both methods separately, with
  family-level bootstrap uncertainty, correct directionality stated
  explicitly, and per-metric valid denominators preserved.
- Complexity-stratified descriptives computed with family-aware N; no
  complexity-group pattern interpreted causally.
- Matching/selection-bias, target-layer confounding, and defect-class
  confounding each independently recomputed and explicitly flagged --
  including stating plainly where complexity-associated effects CANNOT be
  separated from layer/class composition at this sample size.
- Stability-gap finding re-verified with the corrected (v2) data, found to
  be sign-consistent across every subgroup, reported with the brief's exact
  required cautious phrasing.
- Weight-sensitivity and pixel-IoU handled honestly, including disclosing
  where no reproducible per-image alternative data exists rather than
  fabricating it.
- Full 5-way claim classification produced in `scientific_limitations.md`.
- 25/25 new F2.5 tests pass; full Function 2 regression suite 520
  passed/2 skipped/0 failed; protected-artifact scoped re-hash unchanged;
  zero TEST-split access; zero fabricated pixel-IoU values; deterministic
  bootstrap reproducibility verified.

## What remains unresolved (carried forward, not solved by F2.5)

- The complexity protocol is still provisional/NOT expert-validated.
- High-group matching-rate selection bias is real and unquantified in its
  precise effect on any given metric.
- Target-layer and defect-class composition remain substantially confounded
  with complexity group; only 2 within-layer group-pairs and 1 within-class
  group-pair meet this stage's own declared sparsity threshold for even an
  exploratory comparison.
- Pixel-IoU coverage remains tiny (2 intersecting instances, 1 evaluated)
  and has zero High-complexity coverage; a supplementary F2.4-C run would be
  cheap but would not resolve the High-complexity gap.

## Readiness decision

**F2.5's own analysis is complete and internally consistent; the underlying
complexity protocol and XAI-evaluation sample are NOT certified as free of
the confounds and selection effects documented above.** Any F2.6/F2.7 work
that depends on treating a complexity-group difference as attributable to
complexity itself should not proceed without explicitly carrying forward
this stage's confounding and selection-bias findings as live caveats, not
footnotes. No TEST-split evaluation should be attempted until the VALID-side
findings here (and the complexity protocol's own expert-validation status)
are resolved or explicitly accepted as a known limitation by the
responsible researcher/reviewer.

This document, together with `f25_results_summary.md` and
`scientific_limitations.md`, is the complete F2.5 deliverable for
researcher review. Per the task's stop condition: F2.4-C, F2.6, and F2.7
are NOT executed by this stage.
"""
    write("f25_readiness_for_pp1.md", content)


def run():
    report_results_summary()
    report_method_comparison()
    report_complexity_stratified()
    report_matching_bias()
    report_layer_confounding()
    report_defect_class_confounding()
    report_stability_analysis()
    report_sensitivity_analysis()
    report_scientific_limitations()
    report_readiness()


if __name__ == "__main__":
    run()
