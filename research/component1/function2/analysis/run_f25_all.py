"""Run the full F2.5 PP1 analysis pipeline, in dependency order, end to end.

Reproducible: re-running this script from a clean results tree produces
byte-identical JSON/CSV outputs (all computations are deterministic; the
one randomized step -- the Task 11 family-level bootstrap -- is seeded at
357 and verified reproducible by test_f25_analysis.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis import (
    f25_task1_audit, f25_task2_datasets, f25_task3_plan,
    f25_task4_method_comparison, f25_task5_complexity_stratified,
    f25_task6_matching_bias, f25_task7_layer_confound,
    f25_task8_defect_class_confound, f25_task9_stability_gap,
    f25_task10_weight_sensitivity, f25_task11_uncertainty_summary,
    f25_task13_pixel_iou, f25_figures,
)
from research.component1.function2.analysis.f25_common import F25_OUT, write_json


def main():
    audit = f25_task1_audit.run()
    from research.component1.function2.analysis.f25_common import write_json as _wj
    _wj(F25_OUT / "checksums" / "task1_integrity_reaudit.json", audit)
    if audit["overall_status"] != "PASS":
        print("TASK 1 BLOCKED -- stopping pipeline.")
        for b in audit["blockers"]:
            print(" BLOCKER:", b)
        sys.exit(1)
    print("Task 1: PASS (", len(audit["checks"]), "checks )")

    f25_task3_plan.__dict__  # plan is a static module-level constant
    plan_path = F25_OUT / "config" / "f25_preregistered_analysis_plan.json"
    write_json(plan_path, f25_task3_plan.PLAN)
    print("Task 3: plan written")

    _, _, _, unit_summary = f25_task2_datasets.run()
    print("Task 2: units built;", unit_summary["n_distinct_source_families_in_paired_pool"], "families")

    f25_task4_method_comparison.run()
    print("Task 4: method comparison written")

    f25_task5_complexity_stratified.run()
    print("Task 5: complexity-stratified written")

    f25_task6_matching_bias.run()
    print("Task 6: matching/selection-bias written")

    f25_task7_layer_confound.run()
    print("Task 7: target-layer confounding written")

    f25_task8_defect_class_confound.run()
    print("Task 8: defect-class confounding written")

    f25_task9_stability_gap.run()
    print("Task 9: stability gap investigation written")

    f25_task10_weight_sensitivity.run()
    print("Task 10: weight-sensitivity check written")

    f25_task11_uncertainty_summary.run()
    print("Task 11: uncertainty/multiplicity disclosure written")

    f25_task13_pixel_iou.run()
    print("Task 13: pixel-IoU report written")

    figs = f25_figures.run()
    print("Figures:", len(figs))

    print("\nPipeline complete. Outputs under:", F25_OUT)


if __name__ == "__main__":
    main()
