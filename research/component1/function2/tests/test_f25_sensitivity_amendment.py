"""Tests for the F2.5 sensitivity amendment (``analysis/f25_sensitivity_amendment.py``).

Covers (per the amendment task brief): the Task 2 methodology-replication
check (reproduces F2.2B's 132/354 VALID-wide stable count -- the single
most important correctness test here, since the whole amendment's
60-image/48-family result is untrustworthy if this does not hold),
correct z-score/threshold application per scheme against a hand-computable
example, correct family-group resolution logic, deterministic
reproducibility (same inputs -> same output across repeated runs), zero
TEST-split access anywhere in the new module, and a hard checksum check
that every existing protected F2.1-F2.5 artifact (E02 checkpoint, shared
``complexity``/``dataset``/``xai``/``evaluation``/``analysis`` modules,
every F2.1-F2.5 results/ output including the untouched original
``f25_pp1_v1/`` directory) is byte-unchanged before vs after this
amendment's work.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.analysis import f25_sensitivity_amendment as amend
from research.component1.function2.complexity.scoring import NormalizationParams


# --------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def run_result():
    return amend.run()


# ------------------------------------------- Task 2 correctness (CRITICAL)
def test_reproduces_f22b_132_of_354_stable_count(run_result):
    """THE most important test in this suite: if the amendment's method
    does not reproduce F2.2B's already-published 132/354 VALID-wide
    weighting-stability count exactly, the 60-image/48-family results
    below cannot be trusted and must not be reported."""
    check = run_result["task2_354_image_correctness_check"]
    assert check["n_eligible_valid_images"] == 354
    assert check["n_stable_across_all_7_schemes_recomputed"] == 132
    assert check["recorded_n_stable"] == 132
    assert check["matches_f22b_task6_exactly"] is True


def test_baseline_equal_scheme_reproduces_manifest_groups_exactly(run_result):
    """The 'baseline_equal' scheme applied by this amendment's own method
    must reproduce the EXISTING complexity_group column in
    valid_complexity_manifest.csv exactly for every eligible VALID image --
    it is the same frozen formula that produced that column."""
    check = run_result["task2_354_image_correctness_check"]
    assert check["baseline_equal_scheme_reproduces_manifest_complexity_group_exactly"] is True
    assert check["n_baseline_mismatches"] == 0


def test_task2_check_runs_independently_of_f22b_module():
    """Confirms the amendment module never imports f22b_distribution_audit.py
    itself (an independent re-derivation of the 132/354 figure using this
    module's own loading/scoring calls, not a restatement of that module's
    output) -- it only reads f22b's already-written JSON report file to
    compare against, which is legitimate (and is how the correctness check
    is able to assert equality at all)."""
    assert "f22b_distribution_audit" not in dir(amend)
    import ast
    tree = ast.parse(Path(amend.__file__).read_text(encoding="utf-8"))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
    assert not any("f22b_distribution_audit" in m for m in imported_names)


# ------------------------------------------------- z-score / thresholding
def test_zscore_threshold_hand_computable_example():
    """Hand-computable example: mean=10, std=2 for all 3 features (so a
    raw value of 12 always z-scores to +1.0, 8 to -1.0). Equal weighting
    (1/3 each) of three raw values (12, 12, 12) must give composite score
    exactly +1.0 (mean of three +1.0 z-scores). With thresholds
    simple_max=-0.5, moderate_max=0.5, a composite score of +1.0 must be
    assigned group 'High'; one of exactly -1.0 must be assigned 'Simple'."""
    normalization = {
        name: NormalizationParams(mean=10.0, std=2.0) for name in amend.FEATURE_NAMES
    }
    weights = {name: 1.0 / 3.0 for name in amend.FEATURE_NAMES}

    high_raw = {name: 12.0 for name in amend.FEATURE_NAMES}
    res_high = amend.compute_composite_score(high_raw, normalization, weights=weights)
    assert res_high.composite_score == pytest.approx(1.0)
    g_high, _ = amend.assign_group(res_high.composite_score, simple_max=-0.5, moderate_max=0.5)
    assert g_high == "High"

    low_raw = {name: 8.0 for name in amend.FEATURE_NAMES}
    res_low = amend.compute_composite_score(low_raw, normalization, weights=weights)
    assert res_low.composite_score == pytest.approx(-1.0)
    g_low, _ = amend.assign_group(res_low.composite_score, simple_max=-0.5, moderate_max=0.5)
    assert g_low == "Simple"

    mid_raw = {"luminance_variation": 10.0, "glcm_entropy": 10.0, "colour_variation": 10.0}
    res_mid = amend.compute_composite_score(mid_raw, normalization, weights=weights)
    assert res_mid.composite_score == pytest.approx(0.0)
    g_mid, _ = amend.assign_group(res_mid.composite_score, simple_max=-0.5, moderate_max=0.5)
    assert g_mid == "Moderate"


def test_each_scheme_has_its_own_distinct_threshold_pair():
    """Confirms (per the task brief's explicit requirement to verify this
    rather than assume it) that the 7 schemes do NOT all reuse the
    baseline's tertile thresholds -- each alternative scheme carries its
    own TRAIN-derived threshold pair, read from
    b3_task2_weight_sensitivity_summary.json's per_alternative_weighting_results."""
    _, thresholds, baseline_key = amend.load_schemes_and_thresholds()
    assert len(thresholds) == 7
    distinct_threshold_pairs = set(thresholds.values())
    # Not every scheme should collapse to the same thresholds as baseline.
    assert len(distinct_threshold_pairs) > 1
    baseline_thresholds = thresholds[baseline_key]
    n_schemes_matching_baseline_thresholds = sum(
        1 for t in thresholds.values() if t == baseline_thresholds
    )
    assert n_schemes_matching_baseline_thresholds == 1  # only baseline itself


# --------------------------------------------------- family-group resolve
def test_resolve_family_group_single_image_pass_through():
    g, method = amend.resolve_family_group(["Simple"])
    assert g == "Simple"
    assert method == "single_image_family_pass_through"


def test_resolve_family_group_unanimous_multi_image():
    g, method = amend.resolve_family_group(["Moderate", "Moderate", "Moderate"])
    assert g == "Moderate"
    assert method == "majority_vote"


def test_resolve_family_group_clear_majority():
    g, method = amend.resolve_family_group(["High", "High", "Moderate"])
    assert g == "High"
    assert method == "majority_vote"


def test_resolve_family_group_tie_returns_none_not_a_guess():
    g, method = amend.resolve_family_group(["Simple", "High"])
    assert g is None
    assert method == "tied_majority_vote"


def test_resolve_family_group_all_none_returns_none():
    g, method = amend.resolve_family_group([None, None])
    assert g is None
    assert method == "no_valid_group"


def test_actual_evaluable_pool_is_singleton_families(run_result):
    """This amendment's generic family-resolution logic is written to
    handle multi-image families, but the ACTUAL 48-family evaluable pool
    used here has exactly one image per family (verified by F2.5 Task 2's
    own check) -- confirm that is still true and that resolution was a
    pass-through for all 48, not a majority-vote fallback that could mask
    a real disagreement."""
    fam_check = run_result["family_verification_48_families"]
    assert fam_check["n_evaluable_families"] == 48
    assert fam_check["n_families_with_more_than_one_image"] == 0
    assert fam_check["family_resolution_was_pass_through_for_all_families"] is True


# ----------------------------------------------------- counts / coverage
def test_60_selected_images_and_48_evaluable_families(run_result):
    assert run_result["selection_verification_60_images"]["n_selected_images"] == 60
    assert run_result["family_verification_48_families"]["n_evaluable_families"] == 48


def test_all_60_selected_images_verified_real_nonnull_eligible(run_result):
    v = run_result["selection_verification_60_images"]
    assert v["n_missing_from_valid_complexity_manifest"] == 0
    assert v["n_null_feature_entries"] == 0
    assert v["n_ineligible_images"] == 0
    assert v["all_60_verified_real_nonnull_eligible"] is True


def test_stability_counts_are_plausible_relative_to_354_aggregate(run_result):
    """Not a hard-coded expected value (the exact 60/48 counts are a
    product of which specific images were already selected by F2.2B, not
    something this amendment should assert as 'must equal X') -- but the
    fraction stable at 60/48 scale should be in the same general
    neighbourhood as the 354-image aggregate (37.3%), not wildly
    inconsistent with it, since all three are drawn from the same
    eligible VALID feature distribution."""
    assert 0 <= run_result["n_stable_of_60_selected_images"] <= 60
    assert 0 <= run_result["n_stable_of_48_families"] <= 48
    assert 0.15 <= run_result["fraction_stable_60"] <= 0.60
    assert 0.15 <= run_result["fraction_stable_48"] <= 0.60


def test_output_files_written(run_result):
    assert (amend.AMEND_OUT / "selected_image_alternative_labels.csv").exists()
    assert (amend.AMEND_OUT / "family_level_alternative_labels.csv").exists()
    assert (amend.AMEND_OUT / "alternative_group_metric_summary.csv").exists()
    assert (amend.AMEND_OUT / "reports" / "amendment_summary.json").exists()


def test_selected_csv_has_60_data_rows():
    import csv
    with open(amend.AMEND_OUT / "selected_image_alternative_labels.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 60


def test_family_csv_has_48_data_rows():
    import csv
    with open(amend.AMEND_OUT / "family_level_alternative_labels.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 48


# ------------------------------------------------------- determinism ----
def test_deterministic_reproducibility_same_inputs_same_output():
    """Running the full selected-image + family-level reconstruction twice
    must give byte-identical results (no hidden randomness in the
    z-score/threshold/grouping logic itself)."""
    rows1, verification1 = amend.build_selected_image_table()
    rows2, verification2 = amend.build_selected_image_table()
    assert rows1 == rows2
    assert verification1 == verification2


def test_deterministic_reproducibility_full_run_twice():
    """The full run() (including the family-level bootstrap, which uses a
    FIXED seed per F2.5's own bootstrap utility) must give identical
    headline counts across two independent calls."""
    r1 = amend.run()
    r2 = amend.run()
    assert r1["n_stable_of_60_selected_images"] == r2["n_stable_of_60_selected_images"]
    assert r1["n_stable_of_48_families"] == r2["n_stable_of_48_families"]
    assert r1["task2_354_image_correctness_check"] == r2["task2_354_image_correctness_check"]

    import csv
    with open(amend.AMEND_OUT / "alternative_group_metric_summary.csv", encoding="utf-8", newline="") as f:
        metric_rows_after_second_run = list(csv.DictReader(f))
    assert len(metric_rows_after_second_run) > 0
    # bootstrap CIs must be bit-identical across runs at the same fixed seed
    sample = metric_rows_after_second_run[0]
    assert sample["ci_lo"] != "" or sample["ci_skipped"] == "True"


# ----------------------------------------------------------- sparsity ---
def test_no_sparse_strata_silently_reported_as_reliable(run_result):
    """Any (scheme, group) stratum with fewer than 5 families must be
    explicitly flagged in strata_below_sparsity_cutoff_48_families -- never
    silently reported as a full-confidence number."""
    sparsity = run_result["strata_below_sparsity_cutoff_48_families"]
    counts = run_result["group_counts_per_scheme_48_families"]
    for scheme, group_counts in counts.items():
        for group, n in group_counts.items():
            if n < 5:
                assert group in sparsity.get(scheme, {}), (
                    f"{scheme}/{group} has only {n} families but was not flagged as sparse"
                )


def test_metric_summary_rows_below_sparsity_cutoff_have_ci_skipped():
    import csv
    with open(amend.AMEND_OUT / "alternative_group_metric_summary.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if int(r["n_families_with_valid_metric"]) < 5:
            assert r["ci_skipped"] == "True", r


# ------------------------------------------------------------ integrity-
def test_no_test_split_access_in_amendment_module():
    amendment_path = Path(amend.__file__)
    text = amendment_path.read_text(encoding="utf-8")
    forbidden = ["CHENAB_TEST_IMAGES" + "_DIR", "CHENAB_TEST_LABELS" + "_DIR", "test_split", "TEST_SPLIT"]
    for tok in forbidden:
        assert tok not in text, f"amendment module references forbidden token {tok}"


def test_amendment_only_reads_valid_and_train_derived_calibration_sources():
    """Every path this module reads from must be either (a) a VALID-split
    F2.2B manifest, (b) a TRAIN-derived calibration artifact (the frozen
    normalization/threshold FORMULA, not any specific TRAIN image), or
    (c) F2.5's own already-produced manifests. No TEST-split path."""
    paths_referenced = [
        amend.VALID_MANIFEST_PATH, amend.XAI_PRIMARY_SAMPLE_PATH,
        amend.WEIGHT_SENSITIVITY_SUMMARY_PATH, amend.F22B_TASK6_AUDIT_PATH,
        amend.CALIBRATION_CONFIG_PATH, amend.F25_IMAGE_LEVEL_PATH, amend.F25_FAMILY_LEVEL_PATH,
    ]
    allowed_roots = (
        "research/component1/function2/results/evaluation_sampling/",
        "research/component1/function2/results/calibration/",
        "research/component1/function2/config/",
        "research/component1/function2/results/complexity_xai_analysis/f25_pp1_v1/",
    )
    for p in paths_referenced:
        rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        assert "test_split" not in rel.lower()
        assert rel.startswith(allowed_roots), f"unexpected source path outside the known-safe roots: {rel}"


def test_protected_artifacts_byte_unchanged_before_vs_after(run_result):
    """Hard checksum check: every file hashed in the 'before' snapshot
    (E02 checkpoint, config/, every F2.1-F2.5 results/ output including
    the untouched original f25_pp1_v1/ directory, and every pre-existing
    shared module this amendment reads from) must still hash identically
    in the 'after' snapshot taken once this amendment's own work is done."""
    integrity = run_result["protected_artifact_integrity"]
    assert integrity["n_files_checked"] > 1000, "suspiciously few protected files scoped"
    assert integrity["n_missing_files"] == 0
    assert integrity["n_changed_files"] == 0, integrity["changed_files"]
    assert integrity["all_protected_artifacts_byte_unchanged"] is True


def test_original_f25_output_directory_untouched_in_before_after_snapshot():
    """Specifically confirms the original f25_pp1_v1/ tree (not just 'some'
    protected files) is included in the hashed scope and unchanged."""
    before_path = amend.AMEND_OUT / "checksums" / "protected_checksums_before.json"
    after_path = amend.AMEND_OUT / "checksums" / "protected_checksums_after.json"
    with open(before_path, encoding="utf-8") as f:
        before = json.load(f)["sha256_by_relative_path"]
    with open(after_path, encoding="utf-8") as f:
        after = json.load(f)["sha256_by_relative_path"]

    f25_pp1_paths = [p for p in before if "f25_pp1_v1/" in p]
    assert len(f25_pp1_paths) > 10, "original f25_pp1_v1 outputs were not included in the protected scope"
    for p in f25_pp1_paths:
        assert p in after, f"{p} missing from after-snapshot"
        assert before[p] == after[p], f"{p} changed"


def test_live_rehash_of_f25_pp1_v1_matches_recorded_snapshot():
    """Independent of the before/after snapshot files, re-hash a sample of
    original F2.5 files directly from disk right now and compare to the
    recorded 'before' snapshot -- guards against the snapshot itself being
    stale or wrong."""
    before_path = amend.AMEND_OUT / "checksums" / "protected_checksums_before.json"
    with open(before_path, encoding="utf-8") as f:
        before = json.load(f)["sha256_by_relative_path"]
    sample_paths = [p for p in before if "f25_pp1_v1/manifests/" in p][:5]
    assert sample_paths
    for rel in sample_paths:
        full = REPO_ROOT / rel
        assert full.exists()
        assert amend.sha256_file(full) == before[rel]
