"""F2.5 Task 6 -- matching / selection-bias analysis over the full
92-instance population.

Characterizes how High-complexity's lower match rate (57.6% vs ~80%)
changes both the SIZE and the COMPOSITION of the High-group XAI-evaluable
sample vs. the originally-selected 33-instance High population. States
plainly that this missingness is very unlikely to be missing-at-random.
No metric value is ever assigned to an unmatched instance anywhere in this
module.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT, load_all, to_bool, write_json

GROUPS = ["Simple", "Moderate", "High"]


def class_composition(rows, class_field="gt_defect_class"):
    out = {}
    for r in rows:
        c = r.get(class_field) or r.get("defect_class") or r.get("failure_category")
        out[c] = out.get(c, 0) + 1
    return out


def run():
    d = load_all()
    acc = d["accounting"]
    linkage = d["target_linkage"]
    failures = d["failures"]

    out = {"population_n": len(acc)}

    # 1. matched/unmatched + pair-success rates by group
    by_group = {}
    for g in GROUPS:
        g_rows = [r for r in acc if r["complexity_group"] == g]
        matched = [r for r in g_rows if to_bool(r["detection_matched"])]
        unmatched = [r for r in g_rows if not to_bool(r["detection_matched"])]
        paired = [r for r in g_rows if to_bool(r["paired_explanation_success"])]
        degenerate = [r for r in g_rows if to_bool(r["degenerate_one_sided_failure"])]
        by_group[g] = {
            "n_total": len(g_rows),
            "n_matched": len(matched),
            "n_unmatched": len(unmatched),
            "match_rate_pct": round(100.0 * len(matched) / len(g_rows), 1) if g_rows else None,
            "n_paired_success": len(paired),
            "n_degenerate_one_sided": len(degenerate),
            "pair_success_rate_given_matched_pct": (round(100.0 * len(paired) / len(matched), 1)
                                                      if matched else None),
            "defect_class_composition_ALL_selected_instances": class_composition(g_rows),
            "defect_class_composition_MATCHED_instances": class_composition(matched),
            "defect_class_composition_PAIRED_instances": class_composition(paired),
            "defect_class_composition_UNMATCHED_instances": class_composition(unmatched),
        }
    out["by_group"] = by_group

    # 2. defect-class composition of failures (failures.csv), by group and by failure_category
    fail_by_group_category = {}
    for r in failures:
        g = r["complexity_group"]
        cat = r["failure_category"]
        fail_by_group_category.setdefault(g, {}).setdefault(cat, 0)
        fail_by_group_category[g][cat] += 1
    out["failure_category_by_group"] = fail_by_group_category

    fail_class_by_group = {}
    # failures.csv doesn't carry gt_defect_class directly -- join back to accounting
    acc_by_id = {r["gt_instance_id"]: r for r in acc}
    for r in failures:
        g = r["complexity_group"]
        inst = acc_by_id.get(r["gt_instance_id"])
        cls = inst["gt_defect_class"] if inst else "UNKNOWN"
        fail_class_by_group.setdefault(g, {}).setdefault(cls, 0)
        fail_class_by_group[g][cls] += 1
    out["failure_defect_class_by_group"] = fail_class_by_group

    # 3. target-layer distribution of successful (matched) instances, by group
    layer_by_group = {}
    for g in GROUPS:
        g_matched_linkage = [r for r in linkage if r["complexity_group"] == g and r["match_status"] == "matched"]
        counts = {}
        for r in g_matched_linkage:
            counts[r["target_layer"]] = counts.get(r["target_layer"], 0) + 1
        layer_by_group[g] = counts
    out["target_layer_distribution_of_matched_instances_by_group"] = layer_by_group

    # 4. selection-bias characterization: size + composition shift, High vs others
    size_shift = {}
    for g in GROUPS:
        bg = by_group[g]
        size_shift[g] = {
            "n_in_original_60_image_sample_instances": bg["n_total"],
            "n_xai_evaluable_paired": bg["n_paired_success"],
            "fraction_of_original_retained_pct": round(100.0 * bg["n_paired_success"] / bg["n_total"], 1) if bg["n_total"] else None,
        }
    out["sample_size_shift_original_vs_evaluable"] = size_shift

    out["selection_bias_statement"] = (
        "High-complexity's substantially lower match rate (57.6% vs Simple 80.0%/Moderate 79.4%) "
        "means the High-group XAI-evaluable (paired-success) sample is both SMALLER (19 of 33 "
        "originally-selected High instances, vs 20/25 Simple and 27/34 Moderate) and almost "
        "certainly COMPOSITIONALLY DIFFERENT from the full originally-selected High-complexity "
        "population: compare defect_class_composition_ALL_selected_instances vs "
        "defect_class_composition_PAIRED_instances for the High group above -- classes/instances "
        "that are harder for the detector to localize are systematically more likely to be dropped, "
        "not missing at random. This selection mechanism means High-group complete-case metric "
        "values in Tasks 4/5/9 describe the (necessarily easier-than-typical) subset of "
        "High-complexity instances the detector succeeded on, not the general High-complexity "
        "defect population. This is a plausible, data-grounded mechanism (not a hypothetical "
        "caveat) and is NOT evidence that complexity itself causes any downstream metric "
        "difference."
    )

    write_json(F25_OUT / "tables" / "matching_selection_bias.json", out)
    return out


if __name__ == "__main__":
    out = run()
    for g in GROUPS:
        print(g, out["by_group"][g]["defect_class_composition_ALL_selected_instances"],
              "->", out["by_group"][g]["defect_class_composition_PAIRED_instances"])
