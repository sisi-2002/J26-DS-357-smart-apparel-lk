"""F2.1B4-R -- regression tests for the ChatGPT blind-review reconciliation
and agreement-analysis stage (``dataset/f21b4r_reconciliation.py``).

Covers:
  - join-logic correctness: a small, hand-constructed synthetic fixture
    (mirroring the real private_mapping/composite_scores/weight_sensitivity/
    features schema) reconciles exactly as expected.
  - correct REJECTION (never guessing/interpolating) of a blind sample
    whose private-mapping entry, composite-score entry, or
    weight-sensitivity entry is missing -- each checked independently.
  - agreement-metric computation correctness (overall agreement, confusion
    matrix, Cohen's kappa, linear weighted kappa, per-class, stratified by
    confidence/stability) against a small, hand-computable synthetic
    example -- entirely separate from the real 30-row review data.
  - defensive "Cannot Assess" handling (0/30 in the real data, but the code
    must not silently mishandle it if a future review produces it).
  - dominant-raw-feature-by-z-score helper correctness.
  - hard preservation checks: blind_judgments.csv byte-unchanged, the
    F2.1B3 checksum baseline still all_match=True, F2.2A validation-audit
    outputs untouched, SAM pilot masks/review-status untouched -- all
    checked by hashing before and after running this stage's own
    reconciliation script against an ISOLATED output directory (so the
    real production reconciliation_v1/ output is never required to exist
    for these specific tests to be meaningful, though the real run is also
    exercised separately below).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.dataset import (  # noqa: E402
    b3_task5_artifact_checksums,
    f21b4r_reconciliation as recon,
)

FUNCTION2_ROOT = REPO_ROOT / "research" / "component1" / "function2"
REVIEW_DIR = FUNCTION2_ROOT / "results" / "review"
VALIDATION_AUDIT_DIR = FUNCTION2_ROOT / "results" / "validation_audit"
PILOT_DIR = FUNCTION2_ROOT / "results" / "reference_masks" / "pilot_v1"

KNOWN_BLIND_JUDGMENTS_SHA256 = "4e3d32d8bbc30d542a50cea36222e66f327395e4e19f13f18693fe8dfa5ea7dd"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Small, hand-constructed synthetic join fixture (NOT the real 30-row data)
# ---------------------------------------------------------------------------
def _synthetic_join_fixture():
    blind_judgments = [
        {"sample_id": "T001", "judgment": "Simple", "confidence": "High",
         "brief background-only rationale": "flat", "limitations": "none"},
        {"sample_id": "T002", "judgment": "High", "confidence": "Medium",
         "brief background-only rationale": "busy", "limitations": "none"},
        {"sample_id": "T003", "judgment": "Moderate", "confidence": "Low",
         "brief background-only rationale": "mixed", "limitations": "folds excluded"},
    ]
    private_mapping = {
        "T001": {"anonymous_sample_id": "T001", "original_sample_id": "img_A", "source_family_group_id": "fam_1"},
        "T002": {"anonymous_sample_id": "T002", "original_sample_id": "img_B", "source_family_group_id": "fam_2"},
        # T003 deliberately has NO private_mapping entry -> must be rejected.
    }
    composite_scores = {
        "img_A": {"image_id": "img_A", "proposed_group": "Simple", "composite_score": "-0.5",
                   "z_luminance_variation": "-0.4", "z_glcm_entropy": "-0.6", "z_colour_variation": "-0.1",
                   "boundary_adjacent_flag": "False"},
        # img_B deliberately missing from composite_scores -> must be rejected.
    }
    weight_sensitivity = {
        "img_A": {"image_id": "img_A", "group_baseline_equal": "Simple",
                   "stable_across_all_weightings": "True", "n_distinct_groups_across_schemes": "1"},
    }
    features = {
        "img_A": {"image_id": "img_A", "luminance_variation": "10.0", "glcm_entropy": "4.0", "colour_variation": "1.0"},
    }
    return blind_judgments, private_mapping, composite_scores, weight_sensitivity, features


class TestReconcileJoinLogic:
    def test_successful_row_joins_correctly(self):
        blind, mapping, scores, ws, feat = _synthetic_join_fixture()
        reconciled, rejected = recon.reconcile(blind, mapping, scores, ws, feat)
        assert len(reconciled) == 1
        row = reconciled[0]
        assert row["sample_id"] == "T001"
        assert row["image_id"] == "img_A"
        assert row["source_family_group_id"] == "fam_1"
        assert row["algorithm_label"] == "Simple"
        assert row["chatgpt_label"] == "Simple"
        assert row["agree"] is True
        assert row["chatgpt_assessable"] is True
        assert row["weight_stable"] is True
        assert row["algorithm_vs_weight_sensitivity_table_consistent"] is True

    def test_missing_private_mapping_entry_is_rejected(self):
        blind, mapping, scores, ws, feat = _synthetic_join_fixture()
        reconciled, rejected = recon.reconcile(blind, mapping, scores, ws, feat)
        reasons = {r["sample_id"]: r["reason"] for r in rejected}
        assert "T003" in reasons
        assert reasons["T003"] == "no_private_mapping_entry"

    def test_missing_composite_score_entry_is_rejected(self):
        blind, mapping, scores, ws, feat = _synthetic_join_fixture()
        reconciled, rejected = recon.reconcile(blind, mapping, scores, ws, feat)
        reasons = {r["sample_id"]: r["reason"] for r in rejected}
        assert "T002" in reasons
        assert reasons["T002"] == "image_id_not_in_eligible_composite_scores_table"

    def test_missing_weight_sensitivity_entry_is_rejected(self):
        blind, mapping, scores, ws, feat = _synthetic_join_fixture()
        # Add img_B to composite_scores so it gets past that check, but leave
        # it absent from weight_sensitivity -- must still be rejected, not
        # silently treated as "stable" or "unstable" by default.
        scores = dict(scores)
        scores["img_B"] = {"image_id": "img_B", "proposed_group": "High", "composite_score": "1.0",
                            "z_luminance_variation": "1.0", "z_glcm_entropy": "1.0", "z_colour_variation": "1.0",
                            "boundary_adjacent_flag": "False"}
        reconciled, rejected = recon.reconcile(blind, mapping, scores, ws, feat)
        reasons = {r["sample_id"]: r["reason"] for r in rejected}
        assert "T002" in reasons
        assert reasons["T002"] == "image_id_not_in_weight_sensitivity_table"

    def test_no_row_is_ever_guessed_or_interpolated(self):
        """Every rejected row must carry an explicit, non-empty reason string
        -- never silently dropped with no record."""
        blind, mapping, scores, ws, feat = _synthetic_join_fixture()
        reconciled, rejected = recon.reconcile(blind, mapping, scores, ws, feat)
        assert len(reconciled) + len(rejected) == len(blind)
        for r in rejected:
            assert r["reason"]
            assert isinstance(r["reason"], str)

    def test_real_production_join_reconciles_all_30_with_zero_rejections(self):
        """Exercises the REAL production files end-to-end (read-only). This
        is the authoritative check that the actual F2.1B4-R join works, not
        just the synthetic fixture above."""
        reconciled, rejected = recon.reconcile()
        assert len(reconciled) == 30
        assert len(rejected) == 0
        sample_ids = {r["sample_id"] for r in reconciled}
        assert sample_ids == {f"S{i:03d}" for i in range(1, 31)}


class TestLoadBlindJudgmentsValidation:
    def test_duplicate_sample_id_raises(self, tmp_path):
        p = tmp_path / "dup.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "judgment", "confidence", "brief background-only rationale", "limitations"])
            w.writerow(["S001", "Simple", "High", "x", "y"])
            w.writerow(["S001", "High", "High", "x", "y"])
        with pytest.raises(ValueError, match="duplicate sample_id"):
            recon.load_blind_judgments(p)

    def test_invalid_judgment_label_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "judgment", "confidence", "brief background-only rationale", "limitations"])
            w.writerow(["S001", "Extremely Complex", "High", "x", "y"])
        with pytest.raises(ValueError, match="not in allowed vocabulary"):
            recon.load_blind_judgments(p)

    def test_cannot_assess_is_accepted_by_loader(self, tmp_path):
        """0/30 in the real data, but the loader must not hard-reject a
        future review that legitimately uses this label."""
        p = tmp_path / "ca.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "judgment", "confidence", "brief background-only rationale", "limitations"])
            w.writerow(["S001", "Cannot Assess", "Low", "x", "insufficient background visible"])
        rows = recon.load_blind_judgments(p)
        assert rows[0]["judgment"] == "Cannot Assess"


# ---------------------------------------------------------------------------
# Agreement-metric correctness: small, hand-computable synthetic example,
# entirely separate from the real 30-row review data. This is the SAME
# 5-item sequence hand-verified in test_inter_rater_agreement.py's
# test_weighted_kappa_hand_computed_linear_example (kappa_w = 6/11), so the
# unweighted-kappa value here can also be hand-verified independently:
#   p_o = 3/5 = 0.6 (Simple/Simple, Moderate/Moderate, High/High agree;
#         Simple/Moderate and Moderate/High disagree)
#   rater_a (algorithm) marginals: Simple=2/5, Moderate=2/5, High=1/5
#   rater_b (chatgpt)   marginals: Simple=1/5, Moderate=2/5, High=2/5
#   p_e = (2/5*1/5) + (2/5*2/5) + (1/5*2/5) = 2/25 + 4/25 + 2/25 = 8/25 = 0.32
#   kappa = (0.6 - 0.32) / (1 - 0.32) = 0.28 / 0.68 = 7/17 (~0.411764705882)
# ---------------------------------------------------------------------------
def _synthetic_reconciled_for_metrics():
    def row(sample_id, algo, chat, agree, assessable, confidence, stable):
        return {
            "sample_id": sample_id, "algorithm_label": algo, "chatgpt_label": chat,
            "agree": agree, "chatgpt_assessable": assessable, "chatgpt_confidence": confidence,
            "weight_stable": stable,
        }
    return [
        row("T1", "Simple", "Simple", True, True, "High", True),
        row("T2", "Simple", "Moderate", False, True, "Medium", False),
        row("T3", "Moderate", "Moderate", True, True, "High", True),
        row("T4", "Moderate", "High", False, True, "Low", False),
        row("T5", "High", "High", True, True, "High", False),
    ]


class TestAgreementMetricsCorrectness:
    def test_overall_agreement_fraction(self):
        metrics = recon.compute_agreement_metrics(_synthetic_reconciled_for_metrics())
        assert metrics["overall_exact_agreement_fraction"] == pytest.approx(0.6, abs=1e-9)
        assert metrics["n_assessable_used_for_core_stats"] == 5
        assert metrics["n_not_assessable_excluded_from_core_stats"] == 0

    def test_confusion_matrix_full_3x3_with_zeros(self):
        metrics = recon.compute_agreement_metrics(_synthetic_reconciled_for_metrics())
        cm = metrics["confusion_matrix_algorithm_rows_chatgpt_cols"]
        assert cm == {
            "Simple": {"Simple": 1, "Moderate": 1, "High": 0},
            "Moderate": {"Simple": 0, "Moderate": 1, "High": 1},
            "High": {"Simple": 0, "Moderate": 0, "High": 1},
        }

    def test_cohens_kappa_matches_hand_computation(self):
        metrics = recon.compute_agreement_metrics(_synthetic_reconciled_for_metrics())
        assert metrics["cohens_kappa_unweighted"] == pytest.approx(7.0 / 17.0, abs=1e-9)

    def test_weighted_kappa_linear_matches_hand_computation(self):
        metrics = recon.compute_agreement_metrics(_synthetic_reconciled_for_metrics())
        assert metrics["weighted_kappa_linear"] == pytest.approx(6.0 / 11.0, abs=1e-9)

    def test_per_class_recall_precision(self):
        metrics = recon.compute_agreement_metrics(_synthetic_reconciled_for_metrics())
        simple = metrics["per_class"]["Simple"]
        assert simple["n_algorithm_assigned"] == 2
        assert simple["n_chatgpt_assigned"] == 1
        assert simple["n_both_agree_on_this_class"] == 1
        assert simple["recall_vs_algorithm"] == pytest.approx(0.5)
        assert simple["precision_vs_algorithm"] == pytest.approx(1.0)

        high = metrics["per_class"]["High"]
        assert high["n_algorithm_assigned"] == 1
        assert high["n_chatgpt_assigned"] == 2
        assert high["recall_vs_algorithm"] == pytest.approx(1.0)
        assert high["precision_vs_algorithm"] == pytest.approx(0.5)

    def test_stratified_by_confidence(self):
        metrics = recon.compute_agreement_metrics(_synthetic_reconciled_for_metrics())
        by_conf = metrics["by_chatgpt_confidence"]
        assert by_conf["High"] == {"n": 3, "n_agree": 3, "agreement_fraction": pytest.approx(1.0)}
        assert by_conf["Medium"] == {"n": 1, "n_agree": 0, "agreement_fraction": pytest.approx(0.0)}
        assert by_conf["Low"] == {"n": 1, "n_agree": 0, "agreement_fraction": pytest.approx(0.0)}

    def test_stratified_by_weight_stability(self):
        metrics = recon.compute_agreement_metrics(_synthetic_reconciled_for_metrics())
        by_stab = metrics["by_weight_stability"]
        assert by_stab["stable_all_7_schemes"] == {"n": 2, "n_agree": 2, "agreement_fraction": pytest.approx(1.0)}
        assert by_stab["unstable_at_least_1_scheme"]["n"] == 3
        assert by_stab["unstable_at_least_1_scheme"]["n_agree"] == 1
        assert by_stab["unstable_at_least_1_scheme"]["agreement_fraction"] == pytest.approx(1.0 / 3.0, abs=1e-9)

    def test_cannot_assess_rows_excluded_from_core_stats_defensively(self):
        rows = _synthetic_reconciled_for_metrics()
        rows.append({
            "sample_id": "T6", "algorithm_label": "Moderate", "chatgpt_label": "Cannot Assess",
            "agree": False, "chatgpt_assessable": False, "chatgpt_confidence": "Low", "weight_stable": False,
        })
        metrics = recon.compute_agreement_metrics(rows)
        assert metrics["n_total_reconciled"] == 6
        assert metrics["n_not_assessable_excluded_from_core_stats"] == 1
        assert metrics["n_assessable_used_for_core_stats"] == 5
        # Core stats must be IDENTICAL to the 5-row-only case -- the
        # "Cannot Assess" row must not leak into any core statistic.
        baseline = recon.compute_agreement_metrics(_synthetic_reconciled_for_metrics())
        assert metrics["overall_exact_agreement_fraction"] == baseline["overall_exact_agreement_fraction"]
        assert metrics["cohens_kappa_unweighted"] == baseline["cohens_kappa_unweighted"]

    def test_empty_assessable_set_degrades_explicitly_not_silently(self):
        rows = [{
            "sample_id": "T1", "algorithm_label": "Simple", "chatgpt_label": "Cannot Assess",
            "agree": False, "chatgpt_assessable": False, "chatgpt_confidence": "Low", "weight_stable": False,
        }]
        metrics = recon.compute_agreement_metrics(rows)
        assert metrics["n_assessable_used_for_core_stats"] == 0
        assert metrics["overall_exact_agreement_fraction"] is None
        assert metrics["cohens_kappa_unweighted"] is None
        assert metrics["weighted_kappa_linear"] is None


class TestDominantFeatureHelper:
    def test_dominant_feature_is_largest_abs_zscore(self):
        row = {"z_luminance_variation": 1.6, "z_glcm_entropy": -1.7, "z_colour_variation": 0.3}
        assert recon._dominant_feature(row) == "glcm_entropy"

    def test_dominant_feature_handles_missing_values(self):
        row = {"z_luminance_variation": None, "z_glcm_entropy": -0.5, "z_colour_variation": None}
        assert recon._dominant_feature(row) == "glcm_entropy"


# ---------------------------------------------------------------------------
# Hard preservation checks (Task 8). Runs the REAL reconciliation pipeline
# end-to-end but into an ISOLATED tmp_path output directory, then re-checks
# every file this stage must never mutate.
# ---------------------------------------------------------------------------
class TestArtifactPreservation:
    def test_blind_judgments_csv_byte_unchanged_after_full_run(self, tmp_path):
        before_hash = _sha256(REVIEW_DIR / "blind_judgments.csv")
        assert before_hash == KNOWN_BLIND_JUDGMENTS_SHA256
        recon.run(output_dir=tmp_path / "isolated_reconciliation_v1")
        after_hash = _sha256(REVIEW_DIR / "blind_judgments.csv")
        assert after_hash == before_hash == KNOWN_BLIND_JUDGMENTS_SHA256

    def test_other_review_files_byte_unchanged_after_full_run(self, tmp_path):
        files = [REVIEW_DIR / "review_metadata.json", REVIEW_DIR / "review_summary.md",
                 REVIEW_DIR / "import_manifest.json"]
        before = {p: _sha256(p) for p in files}
        recon.run(output_dir=tmp_path / "isolated_reconciliation_v1")
        after = {p: _sha256(p) for p in files}
        assert before == after

    def test_checksum_baseline_still_all_match_after_full_run(self, tmp_path):
        before = b3_task5_artifact_checksums.verify_against_baseline()
        assert before["all_match"] is True
        recon.run(output_dir=tmp_path / "isolated_reconciliation_v1")
        after = b3_task5_artifact_checksums.verify_against_baseline()
        assert after["all_match"] is True

    def test_f22a_validation_audit_outputs_untouched_after_full_run(self, tmp_path):
        files = sorted(VALIDATION_AUDIT_DIR.rglob("*"))
        files = [p for p in files if p.is_file()]
        assert len(files) > 0, "expected F2.2A validation_audit output files to exist"
        before = {p: _sha256(p) for p in files}
        recon.run(output_dir=tmp_path / "isolated_reconciliation_v1")
        after = {p: _sha256(p) for p in files}
        assert before == after

    def test_sam_pilot_masks_and_overlays_untouched_after_full_run(self, tmp_path):
        files = sorted((PILOT_DIR / "sam_proposals").glob("*.png")) + sorted((PILOT_DIR / "overlays").glob("*.png"))
        assert len(files) > 0, "expected SAM pilot mask/overlay PNGs to exist"
        before = {p: _sha256(p) for p in files}
        recon.run(output_dir=tmp_path / "isolated_reconciliation_v1")
        after = {p: _sha256(p) for p in files}
        assert before == after

    def test_sam_pilot_review_status_still_all_unreviewed_after_full_run(self, tmp_path):
        template_path = PILOT_DIR / "quality_review" / "mask_review_template.csv"
        with open(template_path, newline="", encoding="utf-8") as f:
            rows_before = list(csv.DictReader(f))
        assert len(rows_before) > 0
        assert all(r["human_review_status"] == "Unreviewed" for r in rows_before)

        recon.run(output_dir=tmp_path / "isolated_reconciliation_v1")

        with open(template_path, newline="", encoding="utf-8") as f:
            rows_after = list(csv.DictReader(f))
        assert rows_after == rows_before
        assert all(r["human_review_status"] == "Unreviewed" for r in rows_after)

    def test_isolated_run_never_writes_into_real_review_dir(self, tmp_path):
        """The isolated output_dir must be the ONLY place this run writes
        new files -- it must never touch results/review/ itself (only
        results/review/reconciliation_v1/ when run with defaults, which
        this test deliberately avoids by using tmp_path)."""
        before_listing = sorted(p.name for p in REVIEW_DIR.iterdir() if p.is_file())
        recon.run(output_dir=tmp_path / "isolated_reconciliation_v1")
        after_listing = sorted(p.name for p in REVIEW_DIR.iterdir() if p.is_file())
        assert before_listing == after_listing
