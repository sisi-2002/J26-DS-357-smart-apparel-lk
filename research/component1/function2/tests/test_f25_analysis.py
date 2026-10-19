"""F2.5 analysis test suite (Task 15).

Covers: correct matching/joining of F2.4-B metric records, correct
source-family aggregation, correct method-pairing, correct
metric-directionality handling, correct denominator handling per metric,
correct inclusion/exclusion of invalid rows, complete unmatched-instance
accounting, correct layer/class cross-tab computation, no comparison
reported below the declared sparsity threshold, deterministic bootstrap
reproducibility, zero fabricated pixel-IoU values, reproducible end-to-end
outputs, zero TEST-split access, and protected-upstream-artifact
preservation (byte-unchanged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.analysis.f25_common import (
    F25_OUT, F24_FULL, F24_PILOT, F23_FULL, E02_CHECKPOINT, load_all, sha256_file,
)
from research.component1.function2.analysis import f25_task1_audit
from research.component1.function2.analysis.f25_task2_datasets import (
    build_instance_level, build_image_level, build_family_level,
)
from research.component1.function2.analysis.f25_bootstrap import (
    bootstrap_family_mean, MIN_FAMILIES_FOR_CI,
)
from research.component1.function2.analysis import f25_task7_layer_confound, f25_task8_defect_class_confound
from research.component1.function2.analysis import f25_task13_pixel_iou


@pytest.fixture(scope="module")
def data():
    return load_all()


@pytest.fixture(scope="module")
def instance_rows(data):
    return build_instance_level(data)


@pytest.fixture(scope="module")
def family_rows(instance_rows):
    return build_family_level(instance_rows)


# ---------------------------------------------------------------- Task 1 --

def test_task1_audit_passes_with_no_blockers():
    rep = f25_task1_audit.run()
    assert rep["overall_status"] == "PASS", rep["blockers"]
    assert len(rep["checks"]) >= 50


def test_task1_recomputed_headline_matches_documented_values():
    rep = f25_task1_audit.run()
    h = rep["recomputed_headline"]
    assert h["n_images"] == 60
    assert h["n_gt_instances"] == 92
    assert h["n_matched"] == 66
    assert h["n_paired_success"] == 65
    assert h["n_degenerate_one_sided"] == 1
    assert h["per_group"]["Simple"]["matched"] == 20
    assert h["per_group"]["Moderate"]["matched"] == 27
    assert h["per_group"]["High"]["matched"] == 19


def test_complete_unmatched_instance_accounting(data):
    acc = data["accounting"]
    assert len(acc) == 92
    # every one of the 92 has a non-blank value for every eligibility flag
    flag_cols = ["detection_matched", "paired_explanation_success", "degenerate_one_sided_failure",
                 "localization_eligible", "faithfulness_eligible", "stability_eligible"]
    for r in acc:
        for col in flag_cols:
            assert r[col] in ("True", "False"), (r["gt_instance_id"], col, r[col])


# ---------------------------------------------------------------- Task 2 --

def test_instance_level_row_count_is_130(instance_rows):
    assert len(instance_rows) == 130  # 65 paired instances x 2 methods


def test_instance_level_no_duplicate_instance_method_pairs(instance_rows):
    keys = [(r["gt_instance_id"], r["method"]) for r in instance_rows]
    assert len(keys) == len(set(keys))


def test_family_level_aggregation_matches_manual_recount(data, instance_rows, family_rows):
    paired = data["paired_explanations"]
    distinct_families = {r["source_family_id"] for r in paired}
    family_rows_families = {r["source_family_id"] for r in family_rows}
    assert distinct_families == family_rows_families
    # per-method row count == number of distinct families x 2 methods
    assert len(family_rows) == len(distinct_families) * 2


def test_family_level_mean_is_correct_for_a_multi_instance_family(instance_rows, family_rows):
    # the family that contributes 8 of the 65 paired rows (WA0048 image)
    target_fam = "singleton_IMG-20230818-WA0048_jpg.rf.ed38be1191a7958b2767256cdd9d15f4.jpg"
    gc_instances = [r for r in instance_rows if r["source_family_id"] == target_fam and r["method"] == "gradcam"
                    and r["energy_status"] == "valid"]
    assert len(gc_instances) == 8
    manual_mean = sum(r["energy_fraction"] for r in gc_instances) / len(gc_instances)
    fam_row = next(r for r in family_rows if r["source_family_id"] == target_fam and r["method"] == "gradcam")
    assert abs(fam_row["energy_fraction_mean"] - manual_mean) < 1e-9


def test_image_level_aggregation_one_row_per_image_per_method(data, instance_rows):
    image_rows = build_image_level(instance_rows)
    paired = data["paired_explanations"]
    n_images = len({r["image_id"] for r in paired})
    assert len(image_rows) == n_images * 2


# ---------------------------------------------------------------- Task 4 --

def test_method_pairing_same_target_layer_for_both_methods(data):
    linkage = data["target_linkage"]
    for r in linkage:
        if r["match_status"] == "matched":
            assert r["same_target_for_both_methods"] == "True", r["image_id"]


def test_metric_directionality_documented_for_insertion_and_deletion():
    from research.component1.function2.analysis.f25_task3_plan import PLAN
    conv = PLAN["metric_directionality_conventions"]
    assert "higher = better" in conv["insertion_auc"]
    assert "LOWER = better" in conv["deletion_auc"]


def test_denominators_differ_across_metrics_and_are_reported():
    with open(F25_OUT / "tables" / "method_comparison_overall.json", encoding="utf-8") as fh:
        mc = json.load(fh)
    loc_n = mc["localization"]["gradcam"]["pointing_hit_rate"]["valid_n"]
    stab_n = mc["stability_v1"]["gradcam"]["mean_similarity_overall"]["valid_n"]
    assert loc_n == 65
    assert stab_n == 385  # per F2.4-B: 385/390 valid for Grad-CAM v1
    assert loc_n != stab_n


def test_no_fabricated_rows_for_unmatched_or_failed_instances(data):
    acc = data["accounting"]
    non_pairable = {r["gt_instance_id"] for r in acc if r["paired_explanation_success"] != "True"}
    for key in ["localization", "faithfulness", "stability_v1", "stability_v2"]:
        ids = {r["gt_instance_id"] for r in data[key]}
        assert not (ids & non_pairable), f"{key} leaks unmatched/failed instances"


# ---------------------------------------------------------------- Task 7/8 --

def test_sparsity_threshold_enforced_no_comparison_below_cutoff():
    out = f25_task7_layer_confound.run()
    for cell in out["skipped_cells_below_cutoff"]:
        assert cell["n_families"] < MIN_FAMILIES_FOR_CI
    for pair in out["comparable_group_pairs_meeting_sparsity_threshold"]:
        assert pair["n_families_a"] >= MIN_FAMILIES_FOR_CI
        assert pair["n_families_b"] >= MIN_FAMILIES_FOR_CI


def test_layer_group_cross_tab_totals_match_65_pool(data):
    out = f25_task7_layer_confound.run()
    total_instances = sum(cell["n_instances"] for g in out["group_x_layer_cross_tab"].values()
                          for cell in g.values())
    assert total_instances == 65


def test_defect_class_confound_flags_confined_classes():
    out = f25_task8_defect_class_confound.run()
    confined = out["classes_confined_to_le2_groups_in_65"]
    assert "baekra" in confined  # present only in High in the 65-pair pool
    assert set(confined["baekra"]["groups_present"]) <= {"Simple", "Moderate", "High"}


# -------------------------------------------------------------- Task 11 --

def test_bootstrap_reproducible_same_seed(family_rows):
    vals = {r["source_family_id"]: r["energy_fraction_mean"] for r in family_rows
            if r["method"] == "gradcam" and r["energy_fraction_mean"] is not None}
    r1 = bootstrap_family_mean(vals, seed=357)
    r2 = bootstrap_family_mean(vals, seed=357)
    assert r1["ci_lo"] == r2["ci_lo"]
    assert r1["ci_hi"] == r2["ci_hi"]
    assert r1["ci_lo"] is not None


def test_bootstrap_skips_ci_below_sparsity_cutoff():
    tiny = {"fam1": 0.5, "fam2": 0.6, "fam3": 0.4}
    r = bootstrap_family_mean(tiny, seed=357)
    assert r["ci_skipped"] is True
    assert r["ci_lo"] is None and r["ci_hi"] is None


def test_bootstrap_handles_none_values_without_crashing():
    vals = {"a": 0.5, "b": None, "c": 0.6, "d": 0.7, "e": 0.8, "f": 0.4}
    r = bootstrap_family_mean(vals, seed=357)
    assert r["n_effective_families"] == 5  # 'b' dropped


# -------------------------------------------------------------- Task 13 --

def test_pixel_iou_report_matches_known_counts():
    out = f25_task13_pixel_iou.run()
    assert out["n_verified_reference_masks_total"] == 9
    assert out["n_complexity_eligible"] == 8
    assert out["n_intersection"] == 2
    assert len(out["evaluated_instance_iou_rows"]) == 2
    for row in out["evaluated_instance_iou_rows"]:
        assert row["iou"] == "0.0"


def test_pixel_iou_no_high_complexity_coverage():
    out = f25_task13_pixel_iou.run()
    assert out["complexity_group_breakdown_of_9_masks"].get("High", 0) == 0


# ----------------------------------------------------------- integrity --

def test_no_test_split_access_anywhere_in_analysis_dir():
    analysis_dir = REPO_ROOT / "research" / "component1" / "function2" / "analysis"
    forbidden = ["CHENAB_TEST_IMAGES" + "_DIR", "CHENAB_TEST_LABELS" + "_DIR"]
    for py in analysis_dir.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for tok in forbidden:
            if py.name == "f25_task1_audit.py":
                continue  # legitimately stores the token list for its own check
            assert tok not in text, f"{py.name} references {tok}"


def test_protected_f24_f23_outputs_byte_unchanged(tmp_path):
    # Re-hash a representative, explicitly-listed set of protected files and
    # compare against the values this same stage recorded at Task 1 time.
    rep_path = F25_OUT / "checksums" / "task1_integrity_reaudit.json"
    with open(rep_path, encoding="utf-8") as fh:
        rep = json.load(fh)
    for rel_path, recorded_hash in rep["scoped_checksums"].items():
        full_path = REPO_ROOT / rel_path
        assert full_path.exists(), rel_path
        assert sha256_file(full_path) == recorded_hash, f"{rel_path} changed since Task 1 audit"


def test_f25_output_directory_structure_exists():
    for sub in ["config", "manifests", "tables", "figures", "uncertainty", "sensitivity", "checksums", "reports"]:
        assert (F25_OUT / sub).is_dir(), sub


def test_six_figure_types_present():
    figs = {p.name for p in (F25_OUT / "figures").glob("*.svg")}
    expected_substrings = ["localization_by_group", "stability_by_group", "faithfulness",
                           "runtime_by_group", "paired_method_differences", "matching_missingness_by_group"]
    for sub in expected_substrings:
        assert any(sub in f for f in figs), f"missing figure type: {sub}"


def test_pipeline_reproducible_end_to_end(tmp_path, monkeypatch):
    """Re-running Task 2's dataset build twice gives byte-identical CSV
    content (deterministic join/aggregation logic, no hidden randomness)."""
    from research.component1.function2.analysis.f25_task2_datasets import run as task2_run
    r1 = task2_run()
    r2 = task2_run()
    assert r1[3] == r2[3]  # unit_definitions_summary dict equal
    assert [tuple(sorted(x.items())) for x in r1[0]] == [tuple(sorted(x.items())) for x in r2[0]]
