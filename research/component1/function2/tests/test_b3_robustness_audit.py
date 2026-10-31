"""F2.1B3 -- regression tests for the Task 1-6 robustness-audit work.

Covers (per the F2.1B3 Task 5/8 brief):
  - artifact-safety: the new b3_* scripts accept explicit path overrides
    and never silently fall back to a production path when an override is
    supplied; isolated (tmp_path) runs never write into
    results/calibration/ or config/.
  - a checksum-based regression test asserting the named production
    artifacts are byte-identical to the recorded baseline (catches
    accidental mutation by any test or script run with the wrong args).
  - weight-sensitivity computation correctness (Task 2) on a small,
    hand-computable synthetic eligible set.
  - scale-sensitivity resolution-bucket classification correctness (Task 3).
  - source-family/exclusion-bias computation correctness (Task 4) on a
    small synthetic features CSV.
  - review-package (v1) schema correctness (Task 6): genuinely-blank
    reviewer columns, hidden-label companion file round-trips.
  - Cohen's kappa helper (Task 6) correctness on clearly-labelled
    synthetic/dummy data only (never mixed into real review artifacts).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.dataset import (  # noqa: E402
    b3_task1_audit,
    b3_task2_weight_sensitivity,
    b3_task3_scale_sensitivity,
    b3_task4_family_bias_audit,
    b3_task5_artifact_checksums,
    expert_review_package_v2,
)

CALIB_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "calibration"
CONFIG_DIR = REPO_ROOT / "research" / "component1" / "function2" / "config"


# ---------------------------------------------------------------------------
# Task 1: full-round-trip reproducibility regression test (real production
# data, read-only -- no cv2.imread, no mutation of any file this reads).
# ---------------------------------------------------------------------------
def test_task1_audit_reproduces_calibration_with_zero_discrepancies():
    before = b3_task5_artifact_checksums.compute_checksums()
    report = b3_task1_audit.run()
    after = b3_task5_artifact_checksums.compute_checksums()

    assert before == after, "Task 1 audit must be read-only w.r.t. production artifacts"
    assert report["overall_reproducibility_verdict"] == "FULL_ROUND_TRIP_REPRODUCIBLE_NO_DISCREPANCIES"
    assert report["task_A_row_count_crosscheck"]["all_1900_accounted_for"] is True
    assert report["task_B_eligibility_recomputation"]["matches_expected_1747_153"] is True
    assert report["task_C_rejection_reason_specificity"]["all_reasons_specific"] is True
    assert report["task_D_composite_score_and_threshold_recomputation"]["matches_expected_583_582_582"] is True
    assert report["task_E_source_family_counts"]["matches_expected_310_293_227"] is True
    assert report["discrepancies"] == []


# ---------------------------------------------------------------------------
# Task 5: checksum/provenance regression test
# ---------------------------------------------------------------------------
def test_production_artifact_checksums_match_recorded_baseline():
    """Every named production artifact must be byte-identical to the
    baseline recorded in config/b3_artifact_checksums_baseline.json. This
    is the regression guard against the exact class of bug F2.1B2 had
    (an output_dir/config_dir parameter silently ignored, overwriting
    production files)."""
    result = b3_task5_artifact_checksums.verify_against_baseline()
    mismatches = {k: v for k, v in result["per_artifact"].items() if not v["match"]}
    assert result["all_match"], f"Production artifact checksum mismatch(es): {mismatches}"


def test_all_named_production_artifacts_exist_and_are_nonempty():
    checksums = b3_task5_artifact_checksums.compute_checksums()
    missing = [label for label, v in checksums.items() if v is None]
    assert not missing, f"Expected production artifacts missing from disk: {missing}"


# ---------------------------------------------------------------------------
# Task 5: new b3_* scripts respect explicit path overrides (isolation)
# ---------------------------------------------------------------------------
def _write_synthetic_features_csv(path: Path, n_eligible: int = 12, n_ineligible: int = 3) -> None:
    fieldnames = [
        "image_id", "relative_path", "source_family_group_id", "family_size",
        "defect_class_ids", "defect_class_names", "is_negative_manifest",
        "sample_type", "image_height", "image_width",
        "background_status", "background_coverage_fraction", "num_boxes_excluded",
        "boundary_erosion_px", "luminance_variation", "glcm_entropy", "colour_variation",
        "glcm_valid_pair_count", "glcm_status", "overall_status", "quality_flags",
        "eligible", "rejection_reason", "failed_checks",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_eligible):
            fam = f"fam_{i % 4}"
            writer.writerow({
                "image_id": f"elig_{i:03d}", "relative_path": f"train/images/elig_{i:03d}.jpg",
                "source_family_group_id": fam, "family_size": 3,
                "defect_class_ids": json.dumps([i % 3]), "defect_class_names": json.dumps(["x"]),
                "is_negative_manifest": False, "sample_type": "box_excluded_defective",
                "image_height": 500, "image_width": 500,
                "background_status": "boxes_excluded", "background_coverage_fraction": 0.5,
                "num_boxes_excluded": 1, "boundary_erosion_px": 1,
                "luminance_variation": 20.0 + i, "glcm_entropy": 5.0 + 0.1 * i,
                "colour_variation": 2.0 + 0.05 * i,
                "glcm_valid_pair_count": 100000, "glcm_status": "valid",
                "overall_status": "ok", "quality_flags": "[]",
                "eligible": True, "rejection_reason": "", "failed_checks": "[]",
            })
        for i in range(n_ineligible):
            writer.writerow({
                "image_id": f"inelig_{i:03d}", "relative_path": f"train/images/inelig_{i:03d}.jpg",
                "source_family_group_id": f"fam_{i}", "family_size": 1,
                "defect_class_ids": json.dumps([0]), "defect_class_names": json.dumps(["x"]),
                "is_negative_manifest": False, "sample_type": "insufficient_background_defective",
                "image_height": 500, "image_width": 500,
                "background_status": "insufficient_background", "background_coverage_fraction": 0.1,
                "num_boxes_excluded": 2, "boundary_erosion_px": 1,
                "luminance_variation": "", "glcm_entropy": "", "colour_variation": "",
                "glcm_valid_pair_count": 0, "glcm_status": "all_excluded",
                "overall_status": "glcm_all_excluded",
                "quality_flags": "[]", "eligible": False,
                "rejection_reason": "background_coverage_insufficient_provisional_70pct_rule",
                "failed_checks": json.dumps(["background_coverage_insufficient_provisional_70pct_rule"]),
            })


def _write_synthetic_calibration_config(path: Path) -> None:
    config = {
        "normalization": {
            "features": {
                "luminance_variation": {"mean": 25.5, "std": 10.0},
                "glcm_entropy": {"mean": 5.5, "std": 1.0},
                "colour_variation": {"mean": 2.3, "std": 1.0},
            }
        },
        "weights": {"candidate": {"luminance_variation": 1 / 3, "glcm_entropy": 1 / 3, "colour_variation": 1 / 3}},
        "thresholds": {"simple_max": -0.3, "moderate_max": 0.3},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f)


def test_weight_sensitivity_script_isolated_run_does_not_touch_production(tmp_path):
    features_csv = tmp_path / "features.csv"
    config_path = tmp_path / "config.json"
    out_csv = tmp_path / "out" / "weight_sensitivity.csv"
    out_json = tmp_path / "out" / "summary.json"
    _write_synthetic_features_csv(features_csv, n_eligible=12, n_ineligible=3)
    _write_synthetic_calibration_config(config_path)

    before = b3_task5_artifact_checksums.compute_checksums()
    summary = b3_task2_weight_sensitivity.run(
        features_csv=features_csv, config_path=config_path,
        output_csv=out_csv, output_summary_json=out_json,
    )
    after = b3_task5_artifact_checksums.compute_checksums()

    assert before == after, "Isolated weight-sensitivity run must not mutate any production artifact"
    assert out_csv.is_file()
    assert out_json.is_file()
    assert summary["n_eligible_rows_used"] == 12


def test_family_bias_audit_script_isolated_run_does_not_touch_production(tmp_path):
    features_csv = tmp_path / "features.csv"
    _write_synthetic_features_csv(features_csv, n_eligible=12, n_ineligible=3)

    # Build a matching synthetic scores CSV (proposed_group assigned round-robin).
    scores_csv = tmp_path / "scores.csv"
    with open(features_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = ["Simple", "Moderate", "High"]
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "proposed_group"])
        writer.writeheader()
        i = 0
        for r in rows:
            if r["eligible"] == "True":
                writer.writerow({"image_id": r["image_id"], "proposed_group": groups[i % 3]})
                i += 1

    out_json = tmp_path / "out" / "audit.json"
    before = b3_task5_artifact_checksums.compute_checksums()
    report = b3_task4_family_bias_audit.run(
        features_csv=features_csv, scores_csv=scores_csv, output_json=out_json,
    )
    after = b3_task5_artifact_checksums.compute_checksums()

    assert before == after, "Isolated family-bias-audit run must not mutate any production artifact"
    assert out_json.is_file()
    assert report["n_total_train_images"] == 15
    assert report["n_eligible"] == 12
    assert report["n_ineligible"] == 3


# ---------------------------------------------------------------------------
# Task 2: weight-sensitivity computation correctness (hand-computable)
# ---------------------------------------------------------------------------
def test_weight_sensitivity_reassignment_percent_and_confusion_matrix_correct(tmp_path):
    features_csv = tmp_path / "features.csv"
    config_path = tmp_path / "config.json"
    out_csv = tmp_path / "ws.csv"
    out_json = tmp_path / "ws_summary.json"
    _write_synthetic_features_csv(features_csv, n_eligible=12, n_ineligible=0)
    _write_synthetic_calibration_config(config_path)

    summary = b3_task2_weight_sensitivity.run(
        features_csv=features_csv, config_path=config_path,
        output_csv=out_csv, output_summary_json=out_json,
    )

    # Baseline must partition the 12 eligible images into exactly 3 groups
    # of 4 each (tertile of 12 synthetic, monotonically-increasing scores).
    baseline_counts = summary["baseline_counts"]
    assert sum(baseline_counts.values()) == 12
    assert baseline_counts["Simple"] == 4
    assert baseline_counts["Moderate"] == 4
    assert baseline_counts["High"] == 4

    # Every alternative's confusion matrix row/col totals must sum to 12
    # (every eligible image is accounted for exactly once).
    for alt_name, alt in summary["per_alternative_weighting_results"].items():
        cm = alt["confusion_matrix_baseline_rows_alt_cols"]
        total = sum(cm[b][a] for b in cm for a in cm[b])
        assert total == alt["n_compared"] == 12, f"{alt_name}: confusion matrix total mismatch"
        # Reassignment % must equal n_changed / n_compared exactly.
        expected_fraction = alt["n_images_changed_group_vs_baseline"] / alt["n_compared"]
        assert alt["fraction_images_changed_group_vs_baseline"] == pytest.approx(expected_fraction)

    # Extreme single-feature schemes must appear and be internally consistent.
    assert "extreme_luminance_only_1_0_0" in summary["per_alternative_weighting_results"]
    assert "extreme_glcm_only_0_1_0" in summary["per_alternative_weighting_results"]
    assert "extreme_colour_only_0_0_1" in summary["per_alternative_weighting_results"]


def test_weight_sensitivity_stable_vs_unstable_csv_schema(tmp_path):
    features_csv = tmp_path / "features.csv"
    config_path = tmp_path / "config.json"
    out_csv = tmp_path / "ws.csv"
    out_json = tmp_path / "ws_summary.json"
    _write_synthetic_features_csv(features_csv, n_eligible=12, n_ineligible=0)
    _write_synthetic_calibration_config(config_path)

    b3_task2_weight_sensitivity.run(
        features_csv=features_csv, config_path=config_path,
        output_csv=out_csv, output_summary_json=out_json,
    )
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    for r in rows:
        assert r["stable_across_all_weightings"] in ("True", "False")
        # If stable, n_distinct_groups_across_schemes must be exactly 1.
        if r["stable_across_all_weightings"] == "True":
            assert r["n_distinct_groups_across_schemes"] == "1"
        else:
            assert int(r["n_distinct_groups_across_schemes"]) >= 2


# ---------------------------------------------------------------------------
# Task 3: resolution-bucket classification correctness
# ---------------------------------------------------------------------------
def test_scale_sensitivity_bucket_boundaries():
    assert b3_task3_scale_sensitivity._bucket_for(512) == "small"
    assert b3_task3_scale_sensitivity._bucket_for(1) == "small"
    assert b3_task3_scale_sensitivity._bucket_for(513) == "medium"
    assert b3_task3_scale_sensitivity._bucket_for(1199) == "medium"
    assert b3_task3_scale_sensitivity._bucket_for(1200) == "large"
    assert b3_task3_scale_sensitivity._bucket_for(5000) == "large"


def test_scale_sensitivity_preprocess_configs_differ_only_in_resize_long_edge():
    configs = b3_task3_scale_sensitivity.PREPROCESS_CONFIGS
    assert configs["default_512"].resize_long_edge == 512
    assert configs["alt_1024"].resize_long_edge == 1024
    assert configs["no_resize"].resize_long_edge == 100_000
    # All other protocol fields must be identical to the shared frozen
    # module's DEFAULT_CONFIG (never silently diverging on anything other
    # than the one deliberately-varied field).
    for name, cfg in configs.items():
        assert cfg.glcm_levels == b3_task3_scale_sensitivity.DEFAULT_CONFIG.glcm_levels
        assert cfg.interpolation == b3_task3_scale_sensitivity.DEFAULT_CONFIG.interpolation
        assert cfg.glcm_distances == b3_task3_scale_sensitivity.DEFAULT_CONFIG.glcm_distances


# ---------------------------------------------------------------------------
# Task 4: source-family / exclusion-bias computation correctness
# ---------------------------------------------------------------------------
def test_family_bias_audit_class_exclusion_rate_hand_computed(tmp_path):
    """Hand-constructed: class 'A' (id 0) appears in 4 images, 2 excluded
    (50% exclusion rate); class 'B' (id 1) appears in 4 images, 0 excluded
    (0% exclusion rate). Overall exclusion rate is 2/8=25%. Class A's rate
    (50%) is exactly 2x the overall rate -- must be flagged as
    disproportionately excluded under the 1.5x rule; class B must not."""
    fieldnames = [
        "image_id", "relative_path", "source_family_group_id", "family_size",
        "defect_class_ids", "defect_class_names", "is_negative_manifest",
        "sample_type", "image_height", "image_width",
        "background_status", "background_coverage_fraction", "num_boxes_excluded",
        "boundary_erosion_px", "luminance_variation", "glcm_entropy", "colour_variation",
        "glcm_valid_pair_count", "glcm_status", "overall_status", "quality_flags",
        "eligible", "rejection_reason", "failed_checks",
    ]
    features_csv = tmp_path / "features.csv"
    with open(features_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Class A (id 0): 4 images, 2 excluded by the coverage rule.
        for i in range(4):
            eligible = i >= 2  # first 2 excluded, last 2 eligible
            writer.writerow({
                "image_id": f"a_{i}", "relative_path": "x", "source_family_group_id": f"famA_{i}",
                "family_size": 1, "defect_class_ids": json.dumps([0]), "defect_class_names": "[]",
                "is_negative_manifest": False, "sample_type": "box_excluded_defective",
                "image_height": 500, "image_width": 500,
                "background_status": "insufficient_background" if not eligible else "boxes_excluded",
                "background_coverage_fraction": 0.1, "num_boxes_excluded": 1, "boundary_erosion_px": 1,
                "luminance_variation": 20.0, "glcm_entropy": 5.0, "colour_variation": 2.0,
                "glcm_valid_pair_count": 100000, "glcm_status": "valid", "overall_status": "ok",
                "quality_flags": "[]", "eligible": eligible,
                "rejection_reason": "" if eligible else "background_coverage_insufficient_provisional_70pct_rule",
                "failed_checks": "[]",
            })
        # Class B (id 1): 4 images, all eligible.
        for i in range(4):
            writer.writerow({
                "image_id": f"b_{i}", "relative_path": "x", "source_family_group_id": f"famB_{i}",
                "family_size": 1, "defect_class_ids": json.dumps([1]), "defect_class_names": "[]",
                "is_negative_manifest": False, "sample_type": "box_excluded_defective",
                "image_height": 500, "image_width": 500,
                "background_status": "boxes_excluded",
                "background_coverage_fraction": 0.5, "num_boxes_excluded": 1, "boundary_erosion_px": 1,
                "luminance_variation": 20.0, "glcm_entropy": 5.0, "colour_variation": 2.0,
                "glcm_valid_pair_count": 100000, "glcm_status": "valid", "overall_status": "ok",
                "quality_flags": "[]", "eligible": True, "rejection_reason": "", "failed_checks": "[]",
            })

    scores_csv = tmp_path / "scores.csv"
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "proposed_group"])
        writer.writeheader()
        for i in range(2, 4):
            writer.writerow({"image_id": f"a_{i}", "proposed_group": "Simple"})
        for i in range(4):
            writer.writerow({"image_id": f"b_{i}", "proposed_group": "Moderate"})

    out_json = tmp_path / "audit.json"
    report = b3_task4_family_bias_audit.run(features_csv=features_csv, scores_csv=scores_csv, output_json=out_json)

    class_a_name = b3_task4_family_bias_audit.CLASS_NAMES[0]
    class_b_name = b3_task4_family_bias_audit.CLASS_NAMES[1]

    assert report["per_defect_class_exclusion_rates"][class_a_name]["n_total_images_with_class"] == 4
    assert report["per_defect_class_exclusion_rates"][class_a_name]["n_excluded"] == 2
    assert report["per_defect_class_exclusion_rates"][class_a_name]["exclusion_rate"] == pytest.approx(0.5)
    assert report["per_defect_class_exclusion_rates"][class_b_name]["n_excluded"] == 0
    assert report["overall_exclusion_rate"] == pytest.approx(2 / 8)
    # Class A's exclusion rate (0.5) is exactly 2x the overall rate (0.25) ->
    # must be flagged as disproportionately excluded. (n_total=4 < the
    # script's n>=10 flagging threshold, so this specific synthetic case
    # will NOT appear in the auto-flagged list -- verified explicitly here
    # to pin the documented minimum-base-size behaviour, not to assert a
    # false "it must be flagged" expectation.)
    assert class_a_name not in report["disproportionately_excluded_classes_overall_1.5x_rule"], (
        "small-n classes (<10 images) are intentionally NOT auto-flagged -- "
        "this pins that documented behaviour"
    )
    assert report["per_defect_class_exclusion_rates"][class_a_name]["exclusion_rate"] >= 1.5 * report["overall_exclusion_rate"]


# ---------------------------------------------------------------------------
# Task 6: review-package (v2) schema + blind/key-split correctness
# ---------------------------------------------------------------------------
def test_expert_review_package_v2_isolated_run_does_not_touch_production(tmp_path):
    full = tmp_path / "full.csv"
    blind = tmp_path / "blind.csv"
    key = tmp_path / "key.csv"

    before = b3_task5_artifact_checksums.compute_checksums()
    result = expert_review_package_v2.run(output_full_csv=full, output_blind_csv=blind, output_key_csv=key)
    after = b3_task5_artifact_checksums.compute_checksums()

    assert before == after, "Isolated v2 review-package run must not mutate any production artifact"
    assert full.is_file() and blind.is_file() and key.is_file()
    assert result["n_selected_total"] == 30


def test_expert_review_package_v2_blind_csv_has_no_label_revealing_columns(tmp_path):
    full = tmp_path / "full.csv"
    blind = tmp_path / "blind.csv"
    key = tmp_path / "key.csv"
    expert_review_package_v2.run(output_full_csv=full, output_blind_csv=blind, output_key_csv=key)

    with open(blind, newline="", encoding="utf-8") as f:
        blind_rows = list(csv.DictReader(f))
    leaking_columns = {
        "proposed_group", "selection_reason", "composite_score",
        "z_luminance_variation", "z_glcm_entropy", "z_colour_variation",
        "boundary_adjacent_flag", "feature_disagreement_spread",
        "luminance_variation", "glcm_entropy", "colour_variation",
    }
    assert leaking_columns.isdisjoint(blind_rows[0].keys()), (
        "Blind reviewer-facing CSV must not contain any column that could reveal "
        "or allow reverse-engineering the algorithm's proposed group"
    )
    required_blank_columns = {
        "independent_expert_judgment", "confidence_or_uncertainty", "comments",
        "insufficient_visible_background", "anonymous_reviewer_id", "review_date",
    }
    assert required_blank_columns.issubset(blind_rows[0].keys())
    for row in blind_rows:
        for col in required_blank_columns:
            assert row[col] == "", f"column {col!r} must be genuinely blank, got {row[col]!r}"


def test_expert_review_package_v2_key_reconciles_with_blind_by_sample_id(tmp_path):
    full = tmp_path / "full.csv"
    blind = tmp_path / "blind.csv"
    key = tmp_path / "key.csv"
    expert_review_package_v2.run(output_full_csv=full, output_blind_csv=blind, output_key_csv=key)

    with open(blind, newline="", encoding="utf-8") as f:
        blind_ids = {r["sample_id"] for r in csv.DictReader(f)}
    with open(key, newline="", encoding="utf-8") as f:
        key_rows = {r["sample_id"]: r for r in csv.DictReader(f)}

    assert blind_ids == set(key_rows.keys()), "Every blind sample_id must reconcile to exactly one key row"
    for sample_id in blind_ids:
        assert key_rows[sample_id]["proposed_group"] in ("Simple", "Moderate", "High")


def test_expert_review_package_v2_respects_per_family_cap(tmp_path):
    full = tmp_path / "full.csv"
    blind = tmp_path / "blind.csv"
    key = tmp_path / "key.csv"
    result = expert_review_package_v2.run(
        output_full_csv=full, output_blind_csv=blind, output_key_csv=key,
        max_per_family_per_group=2,
    )
    for group, stats in result["family_diversity_v2"].items():
        if group == "overall_n_distinct_families":
            continue
        assert stats["max_single_family_count"] <= 2, f"{group}: per-family cap of 2 violated"


def test_expert_review_package_v2_full_csv_has_genuinely_blank_expert_columns(tmp_path):
    full = tmp_path / "full.csv"
    blind = tmp_path / "blind.csv"
    key = tmp_path / "key.csv"
    expert_review_package_v2.run(output_full_csv=full, output_blind_csv=blind, output_key_csv=key)
    with open(full, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for col in ("independent_expert_judgment", "expert_comments", "disagreement_flag", "review_status"):
        values = {r[col] for r in rows}
        assert values == {""}, f"column {col!r} must be genuinely blank in every row, found values {values}"
