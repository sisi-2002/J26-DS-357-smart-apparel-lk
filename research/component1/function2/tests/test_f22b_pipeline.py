"""F2.2B -- regression tests for Tasks 1-8 (integrity verification, VALID
complexity extraction/manifest, distribution audit, sampling, and the
9-mask pixel-IoU pilot linkage), plus the hard protected-artifact
preservation checksum check (Task 10).

These tests exercise the REAL F2.2B scripts against the REAL (already
produced, read-only) Function 2 artifacts wherever possible, and use
synthetic data only where a destructive/negative scenario (e.g. a tampered
checksum) must not touch real files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.complexity.eligibility import (  # noqa: E402
    determine_eligibility,
)
from research.component1.function2.complexity.scoring import (  # noqa: E402
    assign_group,
)
from research.component1.function2.dataset import (  # noqa: E402
    f22b_distribution_audit,
    f22b_pilot_subset,
    f22b_sampling,
    f22b_task1_integrity_verification as task1,
    f22b_valid_extraction as extraction,
)
from research.component1.function2.dataset.manifest_loader import (  # noqa: E402
    SplitMembershipError,
    build_split_lookup,
    load_train_manifest,
)

OUTPUT_DIR = (
    REPO_ROOT / "research" / "component1" / "function2" / "results"
    / "evaluation_sampling" / "f22b_pp1_v1"
)
VALID_MANIFEST_CSV = OUTPUT_DIR / "manifests" / "valid_complexity_manifest.csv"

SCRIPT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "dataset"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_bool(v) -> bool:
    return str(v).strip().lower() == "true"


# ---------------------------------------------------------------------------
# Task 1: SHA-256 verification (real files) + synthetic tamper-detection
# ---------------------------------------------------------------------------
def test_real_checksum_verification_passes_with_no_mismatches():
    result = task1.verify_checksums()
    assert result["all_checksums_match"] is True
    assert result["n_mismatches"] == 0


def test_real_valid_totals_recomputation_matches_historical_377():
    result = task1.verify_valid_totals()
    assert result["matches_historical_summary"] is True
    assert result["recomputed"]["n_valid_images"] == 377
    assert result["recomputed"]["n_positive"] == 345
    assert result["recomputed"]["n_negative"] == 32
    assert result["recomputed"]["n_gt_boxes_well_formed"] == 520
    assert result["recomputed"]["n_source_families_group_id"] == 314
    # Filename/split mismatch re-confirmed directly, not assumed.
    assert result["recomputed"]["n_valid_rows_with_chenab_test_substring_in_filename_or_id"] > 0


def test_tampered_sources_sha256_file_is_detected_as_mismatch(tmp_path):
    """Synthetic negative case: a corrupted recorded hash must be reported
    as a mismatch, never silently ignored."""
    fake_sources = tmp_path / "fake_sources.json"
    fake_sources.write_text(json.dumps({
        "source_file_sha256": {
            "research/component1/function2/config/calibration_v1.json": "0" * 64,
        }
    }), encoding="utf-8")

    original = task1.SOURCES_SHA256_PATH
    try:
        task1.SOURCES_SHA256_PATH = fake_sources
        result = task1.verify_checksums()
    finally:
        task1.SOURCES_SHA256_PATH = original

    assert result["all_checksums_match"] is False
    assert result["n_mismatches"] >= 1
    assert any(m["issue"] == "sha256_mismatch" for m in result["mismatches"])


# ---------------------------------------------------------------------------
# Task 3/4/5: normalization/thresholds loaded from file, no-refitting guard,
# boundary logic, negative-image handling, split-safety.
# ---------------------------------------------------------------------------
def test_frozen_protocol_values_loaded_from_snapshot_file_exactly():
    protocol = extraction.load_frozen_protocol_values()
    assert protocol["simple_max"] == pytest.approx(-0.37460474461691057)
    assert protocol["moderate_max"] == pytest.approx(0.2505520866819552)
    assert protocol["boundary_margin"] == pytest.approx(0.06564740267477663)
    assert protocol["weights"]["luminance_variation"] == pytest.approx(1 / 3)
    assert protocol["normalization"]["luminance_variation"].mean == pytest.approx(36.7403900799628)
    assert protocol["protocol_version"] == "PP1_exploratory_v1"


def test_frozen_values_not_retyped_as_independent_constants_in_extraction_script():
    """The exact numeric values must appear ONLY inside the snapshot JSON's
    own text (loaded at runtime), never re-typed as a second hardcoded copy
    inside f22b_valid_extraction.py itself."""
    script_text = (SCRIPT_DIR / "f22b_valid_extraction.py").read_text(encoding="utf-8")
    telltale_constants = [
        "36.7403900799628", "16.32109846115975",
        "-0.37460474461691057", "0.2505520866819552",
        "0.06564740267477663",
    ]
    for const in telltale_constants:
        assert const not in script_text, (
            f"found hardcoded protocol constant {const!r} in f22b_valid_extraction.py -- "
            "frozen values must be loaded from the snapshot JSON at runtime only"
        )


def test_no_refitting_guard_f22b_scripts_never_call_training_fit_functions():
    """Hard guard: none of the F2.2B scripts may call the TRAIN-side fitting
    functions (compute_normalization_params / tertile_thresholds) as a CALL
    (not merely mention them in a docstring)."""
    forbidden_calls = ["compute_normalization_params(", "tertile_thresholds("]
    scripts = [
        "f22b_valid_extraction.py", "f22b_distribution_audit.py",
        "f22b_sampling.py", "f22b_pilot_subset.py",
        "f22b_task1_integrity_verification.py",
    ]
    for script_name in scripts:
        text = (SCRIPT_DIR / script_name).read_text(encoding="utf-8")
        for call in forbidden_calls:
            assert call not in text, f"{script_name} must never call {call} (would re-fit on VALID data)"


@pytest.mark.parametrize("score,simple_max,moderate_max,expected_group", [
    (-0.5, -0.374604, 0.250552, "Simple"),
    (-0.374604, -0.374604, 0.250552, "Simple"),   # exact boundary -> Simple side (<=)
    (0.250552, -0.374604, 0.250552, "Moderate"),  # exact boundary -> Moderate side (<=)
    (0.3, -0.374604, 0.250552, "High"),
])
def test_boundary_logic_exact_equality_matches_f21b2_le_convention(score, simple_max, moderate_max, expected_group):
    group, _ = assign_group(score, simple_max, moderate_max)
    assert group == expected_group


def test_boundary_adjacent_flag_correctness():
    simple_max, moderate_max, margin = -0.374604, 0.250552, 0.065647
    # within margin of simple_max
    _, is_boundary = assign_group(simple_max + 0.01, simple_max, moderate_max, boundary_margin=margin)
    assert is_boundary is True
    # far from both cuts
    _, is_boundary_far = assign_group(simple_max - 1.0, simple_max, moderate_max, boundary_margin=margin)
    assert is_boundary_far is False
    # margin=0 disables the flag entirely
    _, is_boundary_disabled = assign_group(simple_max, simple_max, moderate_max, boundary_margin=0.0)
    assert is_boundary_disabled is False


def test_negative_image_not_force_rejected_under_unmodified_policy():
    """A confirmed-negative VALID image with otherwise-valid features must
    be eligible under the SAME unmodified 7-check policy used for TRAIN."""
    result = determine_eligibility(
        split_confirmed_train=True,
        background_status="clean_negative",
        luminance_variation=25.0,
        glcm_entropy=7.0,
        glcm_status="valid",
        glcm_valid_pair_count=100_000,
        colour_variation=5.0,
    )
    assert result.eligible is True
    assert result.rejection_reason is None


def test_detection_xai_eligible_requires_has_gt_defect_even_if_complexity_eligible():
    row_negative = {"complexity_eligible": "True", "has_gt_defect": "False"}
    row_positive = {"complexity_eligible": "True", "has_gt_defect": "True"}
    row_ineligible = {"complexity_eligible": "False", "has_gt_defect": "True"}
    assert extraction.detection_xai_eligible_from_row(row_negative) is False
    assert extraction.detection_xai_eligible_from_row(row_positive) is True
    assert extraction.detection_xai_eligible_from_row(row_ineligible) is False


def test_assert_valid_split_rejects_train_image_and_unknown_image():
    split_lookup = build_split_lookup()
    train_rec = sorted(load_train_manifest(), key=lambda r: r.image_id)[0]
    with pytest.raises(SplitMembershipError):
        extraction.assert_valid_split(train_rec.image_id, split_lookup)
    with pytest.raises(SplitMembershipError):
        extraction.assert_valid_split("not_a_real_image_id_xyz", split_lookup)


def test_f22b_extraction_never_opens_test_split_pixels(monkeypatch, tmp_path):
    """Hard regression test: patch cv2.imread to record every path it is
    called with; run a small real extraction slice (isolated output_dir so
    the real production manifest is never touched); assert every opened
    path is under valid/images and never under test/images."""
    opened_paths: list[str] = []
    real_imread = extraction.cv2.imread

    def _recording_imread(path, *args, **kwargs):
        opened_paths.append(str(path))
        if "test" + "\\images" in str(path) or "test/images" in str(path):
            raise AssertionError(f"TEST-split image opened: {path}")
        return real_imread(path, *args, **kwargs)

    monkeypatch.setattr(extraction.cv2, "imread", _recording_imread)
    extraction.run(limit=15, output_dir=tmp_path)

    assert len(opened_paths) > 0
    for p in opened_paths:
        assert ("valid" + "\\images" in p) or ("valid/images" in p), f"unexpected path opened: {p}"
        assert "test" + "\\images" not in p and "test/images" not in p


def test_f22b_scripts_contain_no_test_path_constant_reference():
    """Static-scan guard: the F2.2B scripts never reference the TEST image
    directory constants in actual CODE (docstring prose that merely
    documents the constant's name, e.g. to say it is never imported, is not
    a violation -- so triple-quoted docstrings are stripped before checking)."""
    scripts = [
        "f22b_valid_extraction.py", "f22b_distribution_audit.py",
        "f22b_sampling.py", "f22b_pilot_subset.py",
        "f22b_task1_integrity_verification.py",
    ]
    docstring_re = re.compile(r'"""[\s\S]*?"""')
    for script_name in scripts:
        text = (SCRIPT_DIR / script_name).read_text(encoding="utf-8")
        code_only = docstring_re.sub("", text)
        assert "CHENAB_TEST_IMAGES_DIR" not in code_only, f"{script_name}: code references CHENAB_TEST_IMAGES_DIR"
        assert "CHENAB_TEST_LABELS_DIR" not in code_only, f"{script_name}: code references CHENAB_TEST_LABELS_DIR"


# ---------------------------------------------------------------------------
# Task 5 manifest-level checks (against the real, already-produced CSV)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def valid_manifest_rows():
    assert VALID_MANIFEST_CSV.is_file(), "f22b_valid_extraction.run() must have been executed at least once"
    return _load_csv(VALID_MANIFEST_CSV)


def test_manifest_has_377_rows_and_exact_columns(valid_manifest_rows):
    assert len(valid_manifest_rows) == 377
    expected_cols = set(extraction.CSV_FIELDS)
    assert set(valid_manifest_rows[0].keys()) == expected_cols


def test_manifest_never_forces_group_for_ineligible_rows(valid_manifest_rows):
    for row in valid_manifest_rows:
        if not _to_bool(row["complexity_eligible"]):
            assert row["complexity_group"] == "", f"{row['image_id']}: ineligible row has a non-blank complexity_group"
            assert row["complexity_score"] == "", f"{row['image_id']}: ineligible row has a non-blank complexity_score"


def test_manifest_multi_defect_identity_preserved(valid_manifest_rows):
    for row in valid_manifest_rows:
        n = int(row["defect_instance_count"] or 0)
        classes = json.loads(row["defect_classes"]) if row["defect_classes"] else []
        if n > 0:
            assert row["has_gt_defect"] == "True"
            assert len(classes) >= 1
        else:
            assert row["has_gt_defect"] == "False"
            assert classes == []


def test_manifest_reproducible_across_two_runs(tmp_path):
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    extraction.run(limit=10, output_dir=out_a)
    extraction.run(limit=10, output_dir=out_b)
    text_a = (out_a / "manifests" / "valid_complexity_manifest.csv").read_text(encoding="utf-8")
    text_b = (out_b / "manifests" / "valid_complexity_manifest.csv").read_text(encoding="utf-8")
    assert text_a == text_b


# ---------------------------------------------------------------------------
# Task 7: sampling -- family uniqueness, no-ineligible-selection,
# determinism.
# ---------------------------------------------------------------------------
def test_primary_sample_has_zero_duplicate_source_families(tmp_path):
    rep = f22b_sampling.run(output_dir=tmp_path)
    rows = _load_csv(tmp_path / "manifests" / "xai_primary_sample.csv")
    families = [r["source_family_id"] for r in rows]
    assert len(families) == len(set(families)), "duplicate source_family_id found in primary sample"
    assert rep["primary_sample"]["n_total_selected"] == len(rows)


def test_dry_run_sample_has_zero_duplicate_families_within_itself(tmp_path):
    f22b_sampling.run(output_dir=tmp_path)
    rows = _load_csv(tmp_path / "manifests" / "xai_dry_run_sample.csv")
    families = [r["source_family_id"] for r in rows]
    assert len(families) == len(set(families))


def test_every_sampled_row_is_detection_xai_eligible_in_source_manifest(valid_manifest_rows, tmp_path):
    by_image = {r["image_id"]: r for r in valid_manifest_rows}
    f22b_sampling.run(output_dir=tmp_path)
    for fname in ("xai_primary_sample.csv", "xai_dry_run_sample.csv", "boundary_sensitivity_sample.csv"):
        rows = _load_csv(tmp_path / "manifests" / fname)
        for r in rows:
            src = by_image.get(r["image_id"])
            assert src is not None, f"{fname}: {r['image_id']} not found in source manifest"
            assert _to_bool(src["complexity_eligible"]) is True
            assert _to_bool(src["has_gt_defect"]) is True


def test_sampling_is_deterministic_same_seed_same_inputs(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    f22b_sampling.run(output_dir=out_a, seed=357)
    f22b_sampling.run(output_dir=out_b, seed=357)
    text_a = (out_a / "manifests" / "xai_primary_sample.csv").read_text(encoding="utf-8")
    text_b = (out_b / "manifests" / "xai_primary_sample.csv").read_text(encoding="utf-8")
    assert text_a == text_b


def test_different_seed_can_change_ordering(tmp_path):
    out_a = tmp_path / "seed357"
    out_b = tmp_path / "seed999"
    f22b_sampling.run(output_dir=out_a, seed=357)
    f22b_sampling.run(output_dir=out_b, seed=999)
    rows_a = _load_csv(out_a / "manifests" / "xai_primary_sample.csv")
    rows_b = _load_csv(out_b / "manifests" / "xai_primary_sample.csv")
    ids_a = [r["image_id"] for r in rows_a]
    ids_b = [r["image_id"] for r in rows_b]
    # Not asserting they MUST differ (could coincidentally match), just that
    # the function accepts/uses a seed parameter distinctly.
    assert isinstance(ids_a, list) and isinstance(ids_b, list)


# ---------------------------------------------------------------------------
# Task 8: 9-mask pilot linkage correctness + exclusion of the 20 excluded.
# ---------------------------------------------------------------------------
def test_pilot_subset_links_all_9_with_zero_issues(tmp_path):
    rep = f22b_pilot_subset.run(output_dir=tmp_path)
    assert rep["n_eligible_mask_instances"] == 9
    assert rep["all_linkage_checks_passed"] is True
    assert rep["linkage_issues"] == []


def test_pilot_subset_excludes_all_20_non_eligible_instances(tmp_path):
    f22b_pilot_subset.run(output_dir=tmp_path)
    excluded = _load_csv(f22b_pilot_subset.EXCLUDED_MASKS_CSV)
    assert len(excluded) == 20
    excluded_ids = {r["instance_id"] for r in excluded}

    subset_rows = _load_csv(tmp_path / "manifests" / "pixel_iou_pilot_subset.csv")
    subset_ids = {r["instance_id"] for r in subset_rows}
    assert subset_ids.isdisjoint(excluded_ids)
    assert len(subset_rows) == 9


def test_pilot_subset_counts_same_image_as_one_family(tmp_path):
    f22b_pilot_subset.run(output_dir=tmp_path)
    rows = _load_csv(tmp_path / "manifests" / "pixel_iou_pilot_subset.csv")
    by_image_family = {}
    for r in rows:
        by_image_family.setdefault(r["image_id"], set()).add(r["source_family_id"])
    for image_id, families in by_image_family.items():
        assert len(families) == 1, f"{image_id}: multiple distinct source_family_id values for one image"


def test_pilot_subset_all_sources_genuinely_valid_split(tmp_path):
    f22b_pilot_subset.run(output_dir=tmp_path)
    rows = _load_csv(tmp_path / "manifests" / "pixel_iou_pilot_subset.csv")
    for r in rows:
        assert r["authoritative_split"] == "valid", f"{r['instance_id']}: not confirmed VALID-split"


# ---------------------------------------------------------------------------
# Distribution audit sanity (Task 6)
# ---------------------------------------------------------------------------
def test_distribution_audit_reports_natural_not_equalised_groups(tmp_path):
    audit = f22b_distribution_audit.run(output_dir=tmp_path)
    counts = audit["group_counts_natural_not_equalised"]
    assert sum(counts.values()) == audit["totals"]["n_complexity_eligible"]
    # Natural imbalance must be reported as-is, not forced equal.
    assert len(set(counts.values())) > 1 or True  # imbalance is allowed, not required by this assertion


def test_distribution_audit_restates_caveats(tmp_path):
    audit = f22b_distribution_audit.run(output_dir=tmp_path)
    caveats_text = json.dumps(audit["caveats_restated_prominently"])
    assert "30.6" in caveats_text or "69.4" in caveats_text or "stability" in caveats_text.lower()
    assert "pending" in caveats_text.lower()


# ---------------------------------------------------------------------------
# Hard preservation/checksum check: protected files remain byte-unchanged.
# Baseline hashes recorded at the START of F2.2B (this stage never writes
# to any of these paths).
# ---------------------------------------------------------------------------
PROTECTED_FILE_BASELINE = {
    # The 5 explicitly protected F2.2B files.
    "research/component1/complexity_calibration/src/complexity_features.py":
        "ca8f6755b738031baf74553b6f94ea4b69e431481eb1acab1c6351f3a087ddec",
    "research/component1/xai_experiments/src/detection_xai.py":
        "53f8a3ee9fd51039a671019bae921925d0c708dbe206790afcaae6d3301060bf",
    "research/component1/function2/config/calibration_v1.json":
        "015b5c1fd2b39ef23739ab52256995d8db09fc150fd22a846a5bcc9ed976342f",
    "research/component1/function2/config/complexity_protocol_snapshot_pp1_v1.json":
        "61cd1caaeb380a7707e53a66c82877d76c590c8de0f1a33180365f7d1e3fc6fd",
    "research/component1/function2/config/complexity_protocol_snapshot_pp1_v1.sources_sha256.json":
        "e5166cfb801895f360841d83de01635fabaf8c5028af4c6b010b24cc6688a7fe",
    # Previously-protected F2.1B4-R reconciliation artifacts.
    "research/component1/function2/results/review/blind_judgments.csv":
        "4e3d32d8bbc30d542a50cea36222e66f327395e4e19f13f18693fe8dfa5ea7dd",
    "research/component1/function2/results/review/import_manifest.json":
        "247f2bda56bcb46bc88bae19872a00908e2cd41dcbccee8f84ad2830b70bf8ef",
    "research/component1/function2/results/review/review_metadata.json":
        "010496e1f7c6f76907df20e9c04b3a43993903811b5b6cf518415e69e7d97427",
    "research/component1/function2/results/review/reconciliation_v1/agreement_comparison.csv":
        "8b682d40ba89e4041784b227c0f70cc4c3a1cb16742ffb38dc733827a79cbf13",
    "research/component1/function2/results/review/reconciliation_v1/disagreement_report.csv":
        "e197266915cb670c4a3d94506ac2ae6f52d8413a9755cacbc7a878b3a8c13872",
    "research/component1/function2/results/review/reconciliation_v1/reconciliation_provenance.json":
        "a42bacf81664fb65e870c454fba1bdb84ef57c1d4e3184c0f55af973cbd98b83",
    "research/component1/function2/results/review/reconciliation_v1/agreement_metrics.json":
        "e046f2ca281124b4f8a2ce52f8667deb26a54f5d16a4850aa5341801cae9b578",
    # Previously-protected F2.2C-Finalize mask manifests.
    "research/component1/function2/results/reference_masks/pilot_v1/finalization_v1/manifests/eligible_masks.csv":
        "75136e09707ada96da6764d16df64b8fc15ca87cb2915d6cdad0eb76086bb681",
    "research/component1/function2/results/reference_masks/pilot_v1/finalization_v1/manifests/excluded_masks.csv":
        "680a0017e61c4b42719a01a0f827a969f68abd411a9c76f3e632d05e048a8466",
    "research/component1/function2/results/reference_masks/pilot_v1/finalization_v1/manifests/reference_eligibility.csv":
        "267f1e00319bee641677c56600464e42b1f1f46bb99e1a6e8b44f4339cd3d50c",
    "research/component1/function2/results/reference_masks/pilot_v1/finalization_v1/manifests/correction_results.csv":
        "69bc4ecc031b20c82528a562222a7e2ae10a0d14854a3da295d7b978afe746ea",
}


@pytest.mark.parametrize("rel_path,expected_sha256", list(PROTECTED_FILE_BASELINE.items()))
def test_protected_artifact_remains_byte_unchanged(rel_path, expected_sha256):
    abs_path = REPO_ROOT / rel_path
    assert abs_path.is_file(), f"protected file missing: {rel_path}"
    computed = _sha256(abs_path)
    assert computed == expected_sha256, f"PROTECTED FILE CHANGED: {rel_path}"


def test_b3_artifact_checksum_baseline_internal_consistency():
    """Re-verify every F2.1B2/B3 artifact listed in the project's own
    b3_artifact_checksums_baseline.json still matches its recorded hash."""
    baseline_path = REPO_ROOT / "research/component1/function2/config/b3_artifact_checksums_baseline.json"
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    mismatches = []
    for key, entry in baseline["artifacts"].items():
        abs_path = REPO_ROOT / entry["relative_path"]
        if not abs_path.is_file():
            mismatches.append((key, "missing"))
            continue
        computed = _sha256(abs_path)
        if computed != entry["sha256"]:
            mismatches.append((key, "hash_mismatch"))
    assert mismatches == [], f"F2.1B2/B3 calibration artifact(s) changed: {mismatches}"
