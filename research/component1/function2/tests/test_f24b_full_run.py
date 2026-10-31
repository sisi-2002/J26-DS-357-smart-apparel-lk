"""F2.4-B Task 14 -- regression tests for the full-65-pair evaluation
orchestration (``run_f24b_full.py``, ``build_accounting.py``,
``run_f24b_sanity_subset.py``, ``run_f24b_timing_sample.py``) and
cross-cutting integrity guarantees (missingness completeness, no TEST-split
access, protected-artifact preservation, F2.4-A pilot byte-preservation).
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.config.paths import (  # noqa: E402
    CHENAB_DATASET_ROOT, CHENAB_TEST_IMAGES_DIR, CHENAB_TEST_LABELS_DIR, E02_CHECKPOINT,
)
from research.component1.function2.evaluation.protected_snapshot import (  # noqa: E402
    verify_against_snapshot,
)

RUN_DIR = REPO_ROOT / "research/component1/function2/results/xai_execution/f23_full_run_v1"
PAIRED_MANIFEST = RUN_DIR / "manifests/paired_explanations_manifest.csv"
EXECUTION_MANIFEST = RUN_DIR / "manifests/execution_manifest.csv"
F24_PILOT_ROOT = REPO_ROOT / "research/component1/function2/results/xai_evaluation/f24_pilot_v1"
F24_FULL_ROOT = REPO_ROOT / "research/component1/function2/results/xai_evaluation/f24_full_v1"


def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _models_present() -> bool:
    return E02_CHECKPOINT.is_file()


# ===========================================================================
# Input-manifest linkage correctness
# ===========================================================================

class TestManifestLinkage:
    def test_paired_manifest_row_count_is_65(self):
        rows = _load_csv(PAIRED_MANIFEST)
        assert len(rows) == 65

    def test_execution_manifest_row_count_is_92(self):
        rows = _load_csv(EXECUTION_MANIFEST)
        assert len(rows) == 92

    def test_every_paired_row_resolves_to_a_matched_execution_row(self):
        exec_by_id = {r["gt_instance_id"]: r for r in _load_csv(EXECUTION_MANIFEST)}
        for r in _load_csv(PAIRED_MANIFEST):
            exec_row = exec_by_id.get(r["gt_instance_id"])
            assert exec_row is not None, f"{r['gt_instance_id']} missing from execution_manifest"
            assert exec_row["prediction_status"] == "matched"

    def test_matched_minus_paired_equals_one_degenerate_one_sided(self):
        exec_rows = _load_csv(EXECUTION_MANIFEST)
        n_matched = sum(1 for r in exec_rows if r["prediction_status"] == "matched")
        n_paired = len(_load_csv(PAIRED_MANIFEST))
        assert n_matched == 66
        assert n_paired == 65
        assert n_matched - n_paired == 1
        degenerate = [r for r in exec_rows if r["prediction_status"] == "matched"
                      and r.get("gradcam_status") == "degenerate" and r.get("gradcampp_status") == "success"]
        assert len(degenerate) == 1

    def test_per_group_matched_counts(self):
        from collections import Counter
        rows = _load_csv(EXECUTION_MANIFEST)
        total = Counter(r["complexity_group"] for r in rows)
        matched = Counter(r["complexity_group"] for r in rows if r["prediction_status"] == "matched")
        assert total == {"Simple": 25, "Moderate": 34, "High": 33}
        assert matched == {"Simple": 20, "Moderate": 27, "High": 19}


# ===========================================================================
# Missingness accounting completeness
# ===========================================================================

class TestAccountingCompleteness:
    def test_accounting_csv_covers_all_92_instances(self):
        acc_path = F24_FULL_ROOT / "manifests" / "all_instances_accounting.csv"
        if not acc_path.exists():
            pytest.skip("accounting CSV not yet built in this environment")
        rows = _load_csv(acc_path)
        assert len(rows) == 92

    def test_accounting_has_distinct_eligibility_columns(self):
        acc_path = F24_FULL_ROOT / "manifests" / "all_instances_accounting.csv"
        if not acc_path.exists():
            pytest.skip("accounting CSV not yet built in this environment")
        rows = _load_csv(acc_path)
        required = {"detection_matched", "paired_explanation_success", "localization_eligible",
                    "faithfulness_eligible", "stability_eligible", "sanity_subset_member",
                    "timing_subset_member"}
        assert required.issubset(rows[0].keys())
        # these must be genuinely distinct concepts, not all identical columns
        cols = {k: tuple(r[k] for r in rows) for k in required}
        distinct_value_tuples = {v for v in cols.values()}
        assert len(distinct_value_tuples) > 1, "eligibility columns must not all collapse to one identical flag"

    def test_unmatched_instances_present_and_count_26(self):
        acc_path = F24_FULL_ROOT / "manifests" / "all_instances_accounting.csv"
        if not acc_path.exists():
            pytest.skip("accounting CSV not yet built in this environment")
        rows = _load_csv(acc_path)
        n_unmatched = sum(1 for r in rows if r["detection_matched"] == "False")
        assert n_unmatched == 26


# ===========================================================================
# Zero TEST-split access (repeats the F2.3/F2.4-A guarantee for F2.4-B code)
# ===========================================================================

class TestNoTestSplitAccess:
    def test_run_f24b_full_source_never_references_test_dir_constants(self):
        src = (REPO_ROOT / "research/component1/function2/evaluation/run_f24b_full.py").read_text(encoding="utf-8")
        assert "CHENAB_TEST_IMAGES_DIR" not in src
        assert "CHENAB_TEST_LABELS_DIR" not in src
        assert '"test"' not in src.replace("valid", "")  # crude guard: no literal "test" path segment

    def test_sanity_and_timing_scripts_never_reference_test_dir(self):
        for fname in ("run_f24b_sanity_subset.py", "run_f24b_timing_sample.py", "run_f24b_stability_audit.py"):
            src = (REPO_ROOT / "research/component1/function2/evaluation" / fname).read_text(encoding="utf-8")
            assert "CHENAB_TEST_IMAGES_DIR" not in src
            assert "CHENAB_TEST_LABELS_DIR" not in src


# ===========================================================================
# Protected-artifact / F2.4-A pilot byte-preservation
# ===========================================================================

class TestProtectedPreservation:
    def test_full_run_script_output_root_is_f24_full_v1_not_pilot(self):
        """The full-run script's OWN output-root constant must point at
        f24_full_v1/, never f24_pilot_v1/ (the docstring may still mention
        f24_pilot_v1 in prose explaining it is not touched -- that's fine;
        what matters is the actual write destination)."""
        from research.component1.function2.evaluation import run_f24b_full as m
        assert "f24_full_v1" in str(m.OUT_ROOT)
        assert "f24_pilot_v1" not in str(m.OUT_ROOT)

    def test_all_new_f24b_scripts_output_roots_point_at_full_v1(self):
        from research.component1.function2.evaluation import (
            run_f24b_sanity_subset, run_f24b_timing_sample, build_accounting,
        )
        for mod_name, mod in (("run_f24b_sanity_subset", run_f24b_sanity_subset),
                               ("run_f24b_timing_sample", run_f24b_timing_sample)):
            # these two expose their default output dir only in __main__; check
            # the RUN_DIR / PAIRED_MANIFEST constants instead point at f23_full_run_v1
            # (the read-only F2.3-B source), never at f24_pilot_v1.
            assert "f24_pilot_v1" not in str(mod.RUN_DIR)
        assert "f24_pilot_v1" not in str(build_accounting.OUT_ROOT)
        assert "f24_full_v1" in str(build_accounting.OUT_ROOT)

    @pytest.mark.skipif(not _models_present(), reason="E02 checkpoint not present in this environment")
    def test_e02_checksum_matches_frozen_value(self):
        assert _sha256(E02_CHECKPOINT) == "c578003ca158a892ac90de3db7b7552aefabd5292e311c6dd0c76faf717628d3"

    def test_f24a_pilot_baseline_snapshot_still_passes(self):
        import json
        baseline_path = F24_PILOT_ROOT / "checksums" / "pre_stage_baseline.json"
        if not baseline_path.is_file():
            pytest.skip("F2.4-A baseline checksum file not present")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        results = verify_against_snapshot(baseline)
        failures = {k: v for k, v in results.items() if v["status"] != "PASS"}
        assert not failures, f"protected files changed or missing: {failures}"


# ===========================================================================
# Stability v2 is a strict superset annotation of v1 (paired-method identity)
# ===========================================================================

class TestStabilityV1V2Consistency:
    def test_v2_rows_subset_relationship_when_full_run_output_present(self):
        v1_path = F24_FULL_ROOT / "metrics" / "stability.csv"
        v2_path = F24_FULL_ROOT / "metrics" / "stability_v2_degenerate_flagged.csv"
        if not (v1_path.exists() and v2_path.exists()):
            pytest.skip("full-run stability outputs not yet present in this environment")
        v1_rows = _load_csv(v1_path)
        v2_rows = _load_csv(v2_path)
        assert len(v1_rows) == len(v2_rows)
        # every v2 "valid" row must also have been "valid" in v1 (v2 only adds
        # a NEW degenerate exclusion, it never resurrects a v1 target_lost row)
        key = lambda r: (r["gt_instance_id"], r["method"], r["perturbation"])
        v1_status = {key(r): r["status"] for r in v1_rows}
        for r in v2_rows:
            if r["status"] == "valid":
                assert v1_status[key(r)] == "valid"
