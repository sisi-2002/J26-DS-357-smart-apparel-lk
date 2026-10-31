"""F2.5 Task 7 -- target-layer confounding audit (independently recomputed).

Builds the complexity-group x target-layer cross-tab with ELIGIBLE FAMILY
counts per cell (not raw instance counts, though both are reported),
localization/stability scores by layer x group, defect-class composition
within each stratum, and flags empty/sparse cells explicitly.

Sparsity cutoff: a within-layer complexity-group comparison is reported
ONLY for cells with >= MIN_FAMILIES_FOR_CI (5) distinct families -- same
threshold as the Task 11 bootstrap, documented in f25_bootstrap.py, reused
here for consistency rather than inventing a second ad hoc rule.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT, load_all, write_json
from research.component1.function2.analysis.f25_task2_datasets import build_instance_level, build_family_level
from research.component1.function2.analysis.f25_bootstrap import MIN_FAMILIES_FOR_CI

GROUPS = ["Simple", "Moderate", "High"]
LAYERS = ["model.model[15]", "model.model[18]", "model.model[21]"]
LAYER_NAMES = {"model.model[15]": "P3", "model.model[18]": "P4", "model.model[21]": "P5"}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def run():
    d = load_all()
    instance_rows = build_instance_level(d)
    family_rows = build_family_level(instance_rows)
    paired = d["paired_explanations"]

    # cross-tab: instance counts and family counts, per group x layer
    cross_tab = {}
    for g in GROUPS:
        cross_tab[g] = {}
        for layer in LAYERS:
            cell_instances = [r for r in paired if r["complexity_group"] == g and r["target_layer"] == layer]
            cell_families = {r["source_family_id"] for r in cell_instances}
            cross_tab[g][LAYER_NAMES[layer]] = {
                "n_instances": len(cell_instances),
                "n_families": len(cell_families),
                "sparse": len(cell_families) < MIN_FAMILIES_FOR_CI,
            }

    # per layer x group: localization energy / stability v1 similarity descriptives
    layer_group_metrics = {}
    reportable_comparisons = []
    skipped_comparisons = []
    for layer in LAYERS:
        layer_group_metrics[LAYER_NAMES[layer]] = {}
        for g in GROUPS:
            cell_fam_ids = {r["source_family_id"] for r in paired
                            if r["complexity_group"] == g and r["target_layer"] == layer}
            n_fam = len(cell_fam_ids)
            cell_instances = [r for r in instance_rows
                              if r["complexity_group"] == g and r["target_layer"] == layer]
            per_method = {}
            for method in ["gradcam", "gradcampp"]:
                m_rows = [r for r in cell_instances if r["method"] == method]
                valid_en = [r for r in m_rows if r["energy_status"] == "valid"]
                per_method[method] = {
                    "n_instances": len(m_rows),
                    "energy_fraction_mean": _mean([r["energy_fraction"] for r in valid_en]),
                    "stability_v1_mean_similarity": _mean([r["stability_v1_mean_similarity"] for r in m_rows]),
                }
            layer_group_metrics[LAYER_NAMES[layer]][g] = {"n_families": n_fam, **per_method}

            entry = {"layer": LAYER_NAMES[layer], "group": g, "n_families": n_fam}
            if n_fam >= MIN_FAMILIES_FOR_CI:
                reportable_comparisons.append(entry)
            else:
                skipped_comparisons.append({**entry, "reason": f"n_families={n_fam} < cutoff {MIN_FAMILIES_FOR_CI}"})

    # which within-layer group PAIRS are actually comparable (both sides >= cutoff)?
    comparable_pairs = []
    for layer in LAYERS:
        ln = LAYER_NAMES[layer]
        fam_counts = {g: layer_group_metrics[ln][g]["n_families"] for g in GROUPS}
        for i, g1 in enumerate(GROUPS):
            for g2 in GROUPS[i + 1:]:
                if fam_counts[g1] >= MIN_FAMILIES_FOR_CI and fam_counts[g2] >= MIN_FAMILIES_FOR_CI:
                    comparable_pairs.append({"layer": ln, "group_a": g1, "group_b": g2,
                                              "n_families_a": fam_counts[g1], "n_families_b": fam_counts[g2]})

    # defect-class composition within each group x layer stratum
    class_by_stratum = {}
    for g in GROUPS:
        class_by_stratum[g] = {}
        for layer in LAYERS:
            cell = [r for r in paired if r["complexity_group"] == g and r["target_layer"] == layer]
            counts = {}
            for r in cell:
                counts[r["gt_defect_class"]] = counts.get(r["gt_defect_class"], 0) + 1
            class_by_stratum[g][LAYER_NAMES[layer]] = counts

    overall_separability_statement = (
        "Complexity-associated effects CANNOT be reliably separated from target-layer composition "
        "with the available sample in every cell: " +
        ("no within-layer group pair reaches the >=5-family-per-side threshold on both sides"
         if not comparable_pairs else
         f"{len(comparable_pairs)} within-layer group pair(s) DO reach the threshold on both sides "
         "(listed in comparable_group_pairs_meeting_sparsity_threshold) and are reported below as "
         "exploratory/sensitivity comparisons only -- NOT as confirmed confounder-adjusted causal "
         "estimates. Every other group pair, in every other layer, remains inseparable from layer "
         "composition at this sample size.")
    )

    out = {
        "sparsity_cutoff_min_families": MIN_FAMILIES_FOR_CI,
        "group_x_layer_cross_tab": cross_tab,
        "layer_group_metric_descriptives": layer_group_metrics,
        "defect_class_by_group_x_layer": class_by_stratum,
        "reportable_cells_meeting_cutoff": reportable_comparisons,
        "skipped_cells_below_cutoff": skipped_comparisons,
        "comparable_group_pairs_meeting_sparsity_threshold": comparable_pairs,
        "separability_statement": overall_separability_statement,
    }
    write_json(F25_OUT / "tables" / "target_layer_confounding.json", out)
    return out


if __name__ == "__main__":
    out = run()
    print("Cross tab:", out["group_x_layer_cross_tab"])
    print("Comparable pairs:", out["comparable_group_pairs_meeting_sparsity_threshold"])
