"""F2.5 Task 5 -- complexity-stratified results (Simple/Moderate/High).

Every metric computed separately per group: images selected, GT instances,
matched, paired, eligible families with valid metric results (per metric),
method-specific failures, group-level descriptives, family-aware
uncertainty where estimable. No complexity-associated difference is
interpreted causally anywhere in this module.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT, load_all, to_bool, write_json
from research.component1.function2.analysis.f25_task2_datasets import build_instance_level, build_family_level
from research.component1.function2.analysis.f25_bootstrap import bootstrap_family_mean, MIN_FAMILIES_FOR_CI

GROUPS = ["Simple", "Moderate", "High"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def group_selection_counts(d):
    acc = d["accounting"]
    out = {}
    for g in GROUPS:
        g_rows = [r for r in acc if r["complexity_group"] == g]
        images = {r["image_id"] for r in g_rows}
        matched = [r for r in g_rows if to_bool(r["detection_matched"])]
        paired = [r for r in g_rows if to_bool(r["paired_explanation_success"])]
        out[g] = {
            "n_images_selected": len(images),
            "n_gt_instances": len(g_rows),
            "n_matched": len(matched),
            "n_paired_success": len(paired),
            "match_rate_pct": round(100.0 * len(matched) / len(g_rows), 1) if g_rows else None,
        }
    return out


def family_metric_by_group(family_rows, group, method, field):
    return {r["source_family_id"]: r[field] for r in family_rows
            if r["method"] == method and r["complexity_group"] == group}


def run():
    d = load_all()
    instance_rows = build_instance_level(d)
    family_rows = build_family_level(instance_rows)

    selection = group_selection_counts(d)

    out = {"selection_counts": selection, "by_group": {}}
    for g in GROUPS:
        g_instance = [r for r in instance_rows if r["complexity_group"] == g]
        g_family = [r for r in family_rows if r["complexity_group"] == g]
        n_fam_total = len({r["source_family_id"] for r in g_family})

        group_out = {"n_eligible_families_total": n_fam_total}
        for method in ["gradcam", "gradcampp"]:
            m_inst = [r for r in g_instance if r["method"] == method]
            valid_pt = [r for r in m_inst if r["pointing_status"] == "valid"]
            hits = sum(1 for r in valid_pt if r["pointing_hit"] in ("True", True))
            valid_en = [r for r in m_inst if r["energy_status"] == "valid"]
            valid_ins = [r for r in m_inst if r["insertion_status"] == "valid"]
            valid_del = [r for r in m_inst if r["deletion_status"] == "valid"]

            fam_energy = family_metric_by_group(family_rows, g, method, "energy_fraction_mean")
            fam_hit = family_metric_by_group(family_rows, g, method, "pointing_hit_rate")
            fam_ins = family_metric_by_group(family_rows, g, method, "insertion_auc_mean")
            fam_del = family_metric_by_group(family_rows, g, method, "deletion_auc_mean")
            fam_stab1 = family_metric_by_group(family_rows, g, method, "stability_v1_mean_similarity")
            fam_stab2 = family_metric_by_group(family_rows, g, method, "stability_v2_mean_similarity")

            group_out[method] = {
                "n_instances": len(m_inst),
                "n_eligible_families_with_any_metric": len({r["source_family_id"] for r in m_inst}),
                "pointing_hit_rate": {"hits": hits, "valid_n": len(valid_pt),
                                      "rate": (hits / len(valid_pt)) if valid_pt else None,
                                      "n_families": len(fam_hit),
                                      "bootstrap": bootstrap_family_mean(fam_hit)},
                "energy_fraction_mean": {"mean": _mean([r["energy_fraction"] for r in valid_en]),
                                         "valid_n": len(valid_en), "n_families": len(fam_energy),
                                         "bootstrap": bootstrap_family_mean(fam_energy)},
                "insertion_auc_mean": {"mean": _mean([r["insertion_auc"] for r in valid_ins]),
                                       "valid_n": len(valid_ins), "n_families": len(fam_ins),
                                       "bootstrap": bootstrap_family_mean(fam_ins)},
                "deletion_auc_mean": {"mean": _mean([r["deletion_auc"] for r in valid_del]),
                                      "valid_n": len(valid_del), "n_families": len(fam_del),
                                      "bootstrap": bootstrap_family_mean(fam_del)},
                "stability_v1_mean_similarity": {"mean": _mean([r["stability_v1_mean_similarity"] for r in m_inst]),
                                                 "n_families": len(fam_stab1),
                                                 "bootstrap": bootstrap_family_mean(fam_stab1)},
                "stability_v2_mean_similarity": {"mean": _mean([r["stability_v2_mean_similarity"] for r in m_inst]),
                                                 "n_families": len(fam_stab2),
                                                 "bootstrap": bootstrap_family_mean(fam_stab2)},
            }
        out["by_group"][g] = group_out

    out["sparsity_cutoff_for_ci"] = MIN_FAMILIES_FOR_CI
    out["causal_language_note"] = ("Group differences below are descriptive only. See "
                                    "target_layer_confounding.md and defect_class_confounding.md "
                                    "before attributing any cross-group pattern to complexity itself.")

    write_json(F25_OUT / "tables" / "complexity_stratified_results.json", out)
    return out


if __name__ == "__main__":
    out = run()
    print("Selection counts:", out["selection_counts"])
