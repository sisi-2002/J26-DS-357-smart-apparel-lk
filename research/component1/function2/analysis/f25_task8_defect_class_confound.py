"""F2.5 Task 8 -- defect-class confounding audit.

Recomputes complexity-group x defect-class and defect-class x target-layer
tables from real data, both among the original 60-image / 92-instance
selection and among the 65 paired instances (these differ). Flags any
defect class effectively confined to 1-2 complexity groups. Does not build
an over-parameterized adjustment model for the available family-level N --
states explicitly where class and complexity are inseparable.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT, load_all, write_json
from research.component1.function2.analysis.f25_bootstrap import MIN_FAMILIES_FOR_CI

GROUPS = ["Simple", "Moderate", "High"]
LAYER_NAMES = {"model.model[15]": "P3", "model.model[18]": "P4", "model.model[21]": "P5"}


def cross_tab(rows, row_key, col_key, row_vals, col_vals):
    tab = {rv: {cv: 0 for cv in col_vals} for rv in row_vals}
    for r in rows:
        rv = r[row_key]
        cv = r[col_key]
        if rv in tab and cv in tab[rv]:
            tab[rv][cv] += 1
        else:
            tab.setdefault(rv, {}).setdefault(cv, 0)
            tab[rv][cv] += 1
    return tab


def run():
    d = load_all()
    acc = d["accounting"]            # 92 instances, original 60-image selection
    paired = d["paired_explanations"]  # 65 instances

    all_classes_92 = sorted({r["gt_defect_class"] for r in acc})
    all_classes_65 = sorted({r["gt_defect_class"] for r in paired})

    tab_92 = cross_tab(acc, "complexity_group", "gt_defect_class", GROUPS, all_classes_92)
    tab_65 = cross_tab(paired, "complexity_group", "gt_defect_class", GROUPS, all_classes_65)

    # classes confined to 1-2 complexity groups (in the 65-pair pool, the
    # pool actually used for every downstream metric)
    confined_classes_65 = {}
    for cls in all_classes_65:
        groups_present = [g for g in GROUPS if tab_65[g].get(cls, 0) > 0]
        if len(groups_present) <= 2:
            confined_classes_65[cls] = {"groups_present": groups_present,
                                         "counts": {g: tab_65[g].get(cls, 0) for g in GROUPS}}
    confined_classes_92 = {}
    for cls in all_classes_92:
        groups_present = [g for g in GROUPS if tab_92[g].get(cls, 0) > 0]
        if len(groups_present) <= 2:
            confined_classes_92[cls] = {"groups_present": groups_present,
                                         "counts": {g: tab_92[g].get(cls, 0) for g in GROUPS}}

    # defect-class x target-layer, among the 65 paired instances (target_layer only exists post-match)
    layers_present = sorted({LAYER_NAMES.get(r["target_layer"], r["target_layer"]) for r in paired})
    class_layer_tab = cross_tab(paired, "gt_defect_class", "target_layer", all_classes_65,
                                 list(LAYER_NAMES.keys()))
    class_layer_tab_named = {cls: {LAYER_NAMES[l]: n for l, n in layer_counts.items()}
                              for cls, layer_counts in class_layer_tab.items()}

    # family-level count per (group, class) in the 65-pair pool, for overlap/sparsity assessment
    fam_by_group_class = {}
    for g in GROUPS:
        fam_by_group_class[g] = {}
        for cls in all_classes_65:
            fams = {r["source_family_id"] for r in paired
                    if r["complexity_group"] == g and r["gt_defect_class"] == cls}
            if fams:
                fam_by_group_class[g][cls] = len(fams)

    # classes with enough family overlap across >=2 groups for a within-class comparison
    within_class_comparable = []
    for cls in all_classes_65:
        groups_with_enough = [g for g in GROUPS if fam_by_group_class.get(g, {}).get(cls, 0) >= MIN_FAMILIES_FOR_CI]
        if len(groups_with_enough) >= 2:
            within_class_comparable.append({"defect_class": cls, "groups_with_ge_cutoff_families": groups_with_enough})

    out = {
        "sparsity_cutoff_min_families": MIN_FAMILIES_FOR_CI,
        "complexity_x_defect_class_ORIGINAL_92_instances": tab_92,
        "complexity_x_defect_class_PAIRED_65_instances": tab_65,
        "defect_class_x_target_layer_PAIRED_65_instances": class_layer_tab_named,
        "classes_confined_to_le2_groups_in_92": confined_classes_92,
        "classes_confined_to_le2_groups_in_65": confined_classes_65,
        "family_count_by_group_x_class_65_pool": fam_by_group_class,
        "within_class_comparisons_with_enough_family_overlap": within_class_comparable,
        "statement": (
            "No over-parameterized adjustment model (e.g. a per-class-per-layer-per-group "
            "regression) is fit here -- the family-level N (48 families total in the 65-pair "
            "pool) cannot support one without severe overfitting. Where "
            "within_class_comparisons_with_enough_family_overlap is empty or near-empty, defect "
            "class and complexity group are INSEPARABLE for that subset at this sample size: a "
            "class effect and a complexity-group effect cannot be told apart from the available "
            "data, and this is stated as a limitation, not resolved by modeling."
        ),
    }
    write_json(F25_OUT / "tables" / "defect_class_confounding.json", out)
    return out


if __name__ == "__main__":
    out = run()
    print("Confined classes (65-pool):", out["classes_confined_to_le2_groups_in_65"])
    print("Within-class comparable:", out["within_class_comparisons_with_enough_family_overlap"])
