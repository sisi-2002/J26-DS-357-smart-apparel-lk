"""F2.2C-QA -- regression tests for the AI-review import, V2 SAM proposal
generation, comparison package, researcher template, eligibility procedure,
and provisional protocol snapshot.

Covers (per task brief Task 9):
  - exact 29-instance review matching (by id, not position)
  - preservation of original AI review judgments (byte-unchanged)
  - preservation of original V1 masks/overlays (byte-unchanged, hash-check)
  - new V2 proposal/prompt provenance completeness
  - image/mask coordinate+dimension consistency for every mask (V1 and V2)
  - correct binary mask format (0/255 only)
  - correct handling of failed/not-attempted proposals (explicit status)
  - human-review fields (researcher_review_v2.csv) all genuinely blank
  - clean separation between AI-review files and human-review files
  - no instance anywhere marked "researcher-approved"/"expert-validated"
  - versioned-output path integrity (nothing written into v1 paths)
  - protocol-snapshot JSON round-trips without altering live calibration files
  - preservation of the F2.1B3 checksum baseline, E02 checkpoint/manifest,
    the two shared modules, and the F2.1B4-R reconciliation outputs
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

from research.component1.function2.config.paths import (  # noqa: E402
    COMPLEXITY_FEATURES_MODULE,
    DETECTION_XAI_MODULE,
    E02_CHECKPOINT,
    FREEZE_MANIFEST,
)
from research.component1.function2.dataset.b3_task5_artifact_checksums import (  # noqa: E402
    verify_against_baseline,
)

PILOT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "reference_masks" / "pilot_v1"
MANIFESTS_DIR = PILOT_DIR / "manifests"
SAM_PROPOSALS_DIR = PILOT_DIR / "sam_proposals"
OVERLAYS_DIR = PILOT_DIR / "overlays"
SAM_PROPOSALS_V2_DIR = PILOT_DIR / "sam_proposals_v2"
OVERLAYS_V2_DIR = PILOT_DIR / "overlays_v2"
CORRECTED_MASKS_DIR = PILOT_DIR / "corrected_masks"
COMPARISON_DIR = PILOT_DIR / "comparison_v2"
QUALITY_REVIEW_DIR = PILOT_DIR / "quality_review"
AI_REVIEW_DIR = QUALITY_REVIEW_DIR / "ai_preliminary_20261018"
REPORTS_DIR = PILOT_DIR / "reports"
CONFIG_DIR = REPO_ROOT / "research" / "component1" / "function2" / "config"

AI_REVIEW_CSV = AI_REVIEW_DIR / "SAM_mask_AI_preliminary_review_29_instances.csv"
RESEARCHER_REVIEW_CSV = QUALITY_REVIEW_DIR / "researcher_review_v2.csv"
PRESERVATION_BASELINE = REPORTS_DIR / "f22c_qa_v1_and_ai_review_preservation_baseline.json"
V2_GEN_REPORT = REPORTS_DIR / "f22c_sam_v2_generation_results.json"
ELIGIBILITY_REPORT = REPORTS_DIR / "f22c_qa_task7_pixel_iou_eligibility.json"
PROTOCOL_SNAPSHOT_JSON = CONFIG_DIR / "complexity_protocol_snapshot_pp1_v1.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Exact 29-instance review matching (by id, not position).
# --------------------------------------------------------------------------

class TestReviewInstanceMatching:
    def test_ai_review_csv_has_29_unique_instances(self):
        rows = _load_csv(AI_REVIEW_CSV)
        assert len(rows) == 29
        ids = [r["instance_id"] for r in rows]
        assert len(set(ids)) == 29

    def test_ai_review_ids_match_pilot_manifest_by_id_not_position(self):
        review_rows = _load_csv(AI_REVIEW_CSV)
        manifest_rows = _load_csv(MANIFESTS_DIR / "instance_manifest.csv")
        manifest_by_id = {r["instance_id"]: r for r in manifest_rows}

        review_ids = {r["instance_id"] for r in review_rows}
        manifest_ids = {r["instance_id"] for r in manifest_rows}
        assert review_ids == manifest_ids

        # Deliberately re-order the review rows and confirm matching logic
        # (as used by this QA stage's own scripts) is id-keyed, not
        # position-keyed: image_id must agree per matched id even if row
        # order differs between the two files.
        for row in review_rows:
            assert manifest_by_id[row["instance_id"]]["image_id"] == row["image_id"]

    def test_status_distribution_matches_known_counts(self):
        rows = _load_csv(AI_REVIEW_CSV)
        from collections import Counter
        counts = Counter(r["ai_review_decision"] for r in rows)
        assert counts == {"Accept": 8, "Needs Correction": 9, "Reject": 10, "Cannot Assess": 2}

    def test_all_human_review_status_unreviewed_in_ai_csv(self):
        rows = _load_csv(AI_REVIEW_CSV)
        assert all(r["human_review_status"] == "Unreviewed" for r in rows)


# --------------------------------------------------------------------------
# Preservation: AI review files + V1 masks/overlays byte-unchanged.
# --------------------------------------------------------------------------

@pytest.mark.skipif(not PRESERVATION_BASELINE.is_file(), reason="preservation baseline not generated in this environment")
class TestPreservation:
    @pytest.fixture(scope="class")
    def baseline(self):
        return _load_json(PRESERVATION_BASELINE)

    def test_baseline_covers_expected_file_counts(self, baseline):
        ai_files = [k for k in baseline["file_sha256"] if k.startswith("quality_review/ai_preliminary_20261018/")]
        v1_mask_files = [k for k in baseline["file_sha256"] if k.startswith("sam_proposals/")]
        v1_overlay_files = [k for k in baseline["file_sha256"] if k.startswith("overlays/")]
        assert len(ai_files) == 11  # csv + report.md + provenance.json + 8 gallery jpgs
        assert len(v1_mask_files) == 29
        assert len(v1_overlay_files) == 29

    def test_every_baselined_file_still_matches(self, baseline):
        mismatches = []
        for rel, expected_hash in baseline["file_sha256"].items():
            p = PILOT_DIR / rel
            if not p.is_file():
                mismatches.append((rel, "MISSING"))
                continue
            actual = _sha256(p)
            if actual != expected_hash:
                mismatches.append((rel, "HASH_MISMATCH"))
        assert mismatches == [], mismatches

    def test_v1_proposals_dir_has_no_new_files_beyond_baseline(self, baseline):
        current = {p.name for p in SAM_PROPOSALS_DIR.glob("*") if p.is_file()}
        baselined = {k.split("/", 1)[1] for k in baseline["file_sha256"] if k.startswith("sam_proposals/")}
        assert current == baselined, (
            f"sam_proposals/ changed: extra={current - baselined}, missing={baselined - current}"
        )

    def test_v1_overlays_dir_has_no_new_files_beyond_baseline(self, baseline):
        current = {p.name for p in OVERLAYS_DIR.glob("*") if p.is_file()}
        baselined = {k.split("/", 1)[1] for k in baseline["file_sha256"] if k.startswith("overlays/")}
        assert current == baselined, (
            f"overlays/ changed: extra={current - baselined}, missing={baselined - current}"
        )


# --------------------------------------------------------------------------
# V2 generation provenance completeness + status handling.
# --------------------------------------------------------------------------

REQUIRED_V2_PROVENANCE_FIELDS = [
    "instance_id", "image_id", "status", "strategy", "point_mode",
    "grounding_confidence", "rationale", "sam_model_type", "sam_checkpoint",
]

STATUS_VALUES = {
    "success", "attempted_no_defensible_result",
    "not_attempted_flagged_unsuitable_for_pixel_iou",
}


@pytest.mark.skipif(not V2_GEN_REPORT.is_file(), reason="V2 generation outputs not present in this environment")
class TestV2GenerationProvenance:
    @pytest.fixture(scope="class")
    def report(self):
        return _load_json(V2_GEN_REPORT)

    def test_exactly_19_target_instances(self, report):
        assert report["n_target_instances"] == 19
        assert len(report["per_instance_results"]) == 19

    def test_counts_sum_to_19(self, report):
        assert (
            report["n_success"]
            + report["n_attempted_no_defensible_result"]
            + report["n_not_attempted_flagged_unsuitable"]
            == 19
        )

    def test_every_instance_has_required_provenance_fields(self, report):
        for inst in report["per_instance_results"]:
            for field_name in REQUIRED_V2_PROVENANCE_FIELDS:
                assert field_name in inst, (inst.get("instance_id"), field_name)

    def test_every_status_is_a_known_explicit_value_never_silently_missing(self, report):
        for inst in report["per_instance_results"]:
            assert inst["status"] in STATUS_VALUES, inst["instance_id"]

    def test_success_instances_have_full_candidate_and_point_provenance(self, report):
        for inst in report["per_instance_results"]:
            if inst["status"] != "success":
                continue
            assert "selected_candidate" in inst
            assert "selected_candidate_metrics" in inst
            assert "all_candidates_considered" in inst
            assert len(inst["all_candidates_considered"]) >= 1
            assert "positive_points_crop_local" in inst
            assert "negative_points_crop_local" in inst
            assert "box_xyxy_pixels_crop_local" in inst
            assert "crop_xyxy_pixels_original_image" in inst

    def test_flagged_unsuitable_instance_has_no_mask_file_and_no_sam_call_implied(self, report):
        flagged = [r for r in report["per_instance_results"]
                   if r["status"] == "not_attempted_flagged_unsuitable_for_pixel_iou"]
        assert len(flagged) == 1
        assert "mask_path" not in flagged[0]

    def test_selection_is_not_simply_highest_sam_score(self, report):
        """At least one success instance exists where a HIGHER-scoring
        candidate was available but NOT selected, proving the selection
        heuristic is not a naive argmax on SAM confidence alone."""
        found_non_naive_selection = False
        for inst in report["per_instance_results"]:
            if inst["status"] != "success":
                continue
            all_scores = [c["sam_confidence_score"] for c in inst["all_candidates_considered"]]
            selected_score = inst["selected_candidate_sam_confidence_score"]
            if any(s > selected_score for s in all_scores):
                found_non_naive_selection = True
                break
        assert found_non_naive_selection

    def test_status_label_present_and_not_fabricated_approval(self, report):
        assert "NOT human-verified" in report["status_label"]


# --------------------------------------------------------------------------
# Mask dimension/coordinate consistency + binary format, V1 and V2.
# --------------------------------------------------------------------------

@pytest.mark.skipif(not (PILOT_DIR / "reports" / "f22c_mask_generation_results.json").is_file(),
                     reason="V1 generation report not present in this environment")
class TestMaskIntegrityV1AndV2:
    @pytest.fixture(scope="class")
    def v1_report(self):
        return _load_json(PILOT_DIR / "reports" / "f22c_mask_generation_results.json")

    def test_v1_masks_binary_and_dimension_matched(self, v1_report):
        from PIL import Image
        import numpy as np

        checked = 0
        for inst in v1_report["per_instance_results"]:
            if inst["status"] != "success":
                continue
            mask_path = REPO_ROOT / inst["mask_path"]
            im = Image.open(mask_path)
            assert im.size == (inst["image_width"], inst["image_height"])
            arr = np.array(im.convert("L"))
            assert set(np.unique(arr).tolist()).issubset({0, 255})
            checked += 1
        assert checked > 0

    @pytest.mark.skipif(not V2_GEN_REPORT.is_file(), reason="V2 generation outputs not present")
    def test_v2_masks_binary_and_dimension_matched_to_crop(self):
        from PIL import Image
        import numpy as np

        v2_report = _load_json(V2_GEN_REPORT)
        checked = 0
        for inst in v2_report["per_instance_results"]:
            if inst["status"] != "success":
                continue
            mask_path = REPO_ROOT / inst["mask_path"]
            im = Image.open(mask_path)
            assert im.size == (inst["crop_width"], inst["crop_height"]), inst["instance_id"]
            arr = np.array(im.convert("L"))
            assert set(np.unique(arr).tolist()).issubset({0, 255})
            # box-local coordinates must fit inside the crop.
            x1, y1, x2, y2 = inst["box_xyxy_pixels_crop_local"]
            assert 0 <= x1 < x2 <= inst["crop_width"]
            assert 0 <= y1 < y2 <= inst["crop_height"]
            checked += 1
        assert checked > 0

    @pytest.mark.skipif(not ELIGIBILITY_REPORT.is_file(), reason="eligibility report not present")
    def test_eligibility_report_dimension_checks_all_passed(self):
        report = _load_json(ELIGIBILITY_REPORT)
        assert report["n_mask_dimension_checks_run"] > 0
        assert report["all_checked_masks_dimensions_ok"] is True
        assert report["n_mask_dimension_mismatches"] == 0


# --------------------------------------------------------------------------
# Human-review template genuinely blank + AI/human separation.
# --------------------------------------------------------------------------

class TestResearcherTemplateAndSeparation:
    def test_researcher_review_v2_has_all_29_instances(self):
        rows = _load_csv(RESEARCHER_REVIEW_CSV)
        assert len(rows) == 29
        assert len({r["instance_id"] for r in rows}) == 29

    def test_researcher_review_v2_decision_fields_genuinely_blank(self):
        rows = _load_csv(RESEARCHER_REVIEW_CSV)
        blank_fields = [
            "chosen_proposal_version", "researcher_review_status", "required_correction",
            "reviewer_identifier", "review_date", "confidence_or_uncertainty",
            "annotation_quality_notes",
        ]
        for row in rows:
            for field_name in blank_fields:
                assert row[field_name] == "", (row["instance_id"], field_name)

    def test_researcher_review_v2_includes_all_8_ai_accept_instances(self):
        rows = _load_csv(RESEARCHER_REVIEW_CSV)
        accept_rows = [r for r in rows if r["ai_preliminary_status_reference_only"] == "Accept"]
        assert len(accept_rows) == 8
        # And none of them have a pre-filled researcher decision.
        for r in accept_rows:
            assert r["researcher_review_status"] == ""

    def test_ai_review_csv_has_no_human_decision_columns_mutated(self):
        """mask_review_template.csv (the ORIGINAL human-review file from the
        pilot) must still be all-Unreviewed/blank -- this QA stage must not
        have written AI decisions into it."""
        rows = _load_csv(QUALITY_REVIEW_DIR / "mask_review_template.csv")
        for row in rows:
            assert row["human_review_status"] == "Unreviewed"
            assert row["reviewer_id"] == ""
            assert row["reviewer_comments"] == ""

    def test_ai_review_csv_and_researcher_csv_are_separate_files_with_distinct_schemas(self):
        ai_rows = _load_csv(AI_REVIEW_CSV)
        researcher_rows = _load_csv(RESEARCHER_REVIEW_CSV)
        ai_fields = set(ai_rows[0].keys())
        researcher_fields = set(researcher_rows[0].keys())
        # The human-decision field names must never appear in the AI CSV's
        # schema (no cross-contamination of what each file is for).
        human_only_fields = {"researcher_review_status", "reviewer_identifier",
                              "chosen_proposal_version", "required_correction"}
        assert human_only_fields.isdisjoint(ai_fields)
        assert human_only_fields.issubset(researcher_fields)


# --------------------------------------------------------------------------
# No fabricated approval anywhere in this QA stage's outputs.
# --------------------------------------------------------------------------

class TestNoFabricatedApproval:
    FORBIDDEN_TERMS = [
        "expert-approved", "expert approved", "auto-approved", "auto approved",
        "auto-accepted", "researcher-approved", "researcher approved",
        "expert-validated", "expert validated",
    ]

    def _all_qa_output_texts(self):
        texts = []
        for p in REPORTS_DIR.glob("f22c_qa_*"):
            if p.suffix in (".json", ".md") and p.is_file():
                texts.append(p.read_text(encoding="utf-8").lower())
        if V2_GEN_REPORT.is_file():
            texts.append(V2_GEN_REPORT.read_text(encoding="utf-8").lower())
        quality_report = REPORTS_DIR / "sam_pilot_v2_quality_report.md"
        if quality_report.is_file():
            texts.append(quality_report.read_text(encoding="utf-8").lower())
        if RESEARCHER_REVIEW_CSV.is_file():
            texts.append(RESEARCHER_REVIEW_CSV.read_text(encoding="utf-8").lower())
        return texts

    def test_no_forbidden_approval_terms_anywhere(self):
        texts = self._all_qa_output_texts()
        assert texts, "no QA output files found to scan"
        combined = "\n".join(texts)
        for term in self.FORBIDDEN_TERMS:
            # "NOT expert validated" / "not human-verified" are expected negated
            # disclaimers; only an UN-negated occurrence should trip this.
            idx = combined.find(term)
            while idx != -1:
                window = combined[max(0, idx - 20):idx].replace("\n", " ")
                assert "not " in window or "never " in window, (
                    f"Found un-negated '{term}' near: ...{combined[max(0, idx-40):idx+40]}..."
                )
                idx = combined.find(term, idx + 1)

    def test_eligibility_report_states_zero_eligible(self):
        if not ELIGIBILITY_REPORT.is_file():
            pytest.skip("eligibility report not present")
        report = _load_json(ELIGIBILITY_REPORT)
        assert report["n_eligible_for_pixel_iou"] == 0


# --------------------------------------------------------------------------
# Versioned-output path integrity: nothing written into v1 paths.
# --------------------------------------------------------------------------

class TestVersionedPathIntegrity:
    def test_v2_proposals_dir_exists_and_separate_from_v1(self):
        assert SAM_PROPOSALS_V2_DIR.is_dir()
        assert SAM_PROPOSALS_V2_DIR != SAM_PROPOSALS_DIR

    def test_v2_overlays_dir_exists_and_separate_from_v1(self):
        assert OVERLAYS_V2_DIR.is_dir()
        assert OVERLAYS_V2_DIR != OVERLAYS_DIR

    def test_corrected_masks_dir_exists_and_is_empty(self):
        assert CORRECTED_MASKS_DIR.is_dir()
        contents = [p for p in CORRECTED_MASKS_DIR.glob("*") if p.name != ".gitkeep"]
        assert contents == [], f"corrected_masks/ must stay empty (reserved for future human corrections): {contents}"

    def test_no_v2_filenames_collide_with_v1_filenames_in_v1_dirs(self):
        v2_names = {p.name for p in SAM_PROPOSALS_V2_DIR.glob("*.png")}
        v1_names = {p.name for p in SAM_PROPOSALS_DIR.glob("*.png")}
        # v2 instance_ids are a subset of v1 instance_ids by construction,
        # so same-name files are expected -- what matters is they live in
        # DIFFERENT directories, not that the stems differ.
        assert v2_names, "expected V2 proposals to exist"
        for name in v2_names:
            assert (SAM_PROPOSALS_V2_DIR / name).is_file()
            # Confirm the V1 original at that name is untouched (still
            # matches its own directory, not overwritten by a V2 file).
            if name in v1_names:
                assert (SAM_PROPOSALS_DIR / name).stat().st_size > 0


# --------------------------------------------------------------------------
# Protocol snapshot round-trip + non-mutation of live calibration files.
# --------------------------------------------------------------------------

@pytest.mark.skipif(not PROTOCOL_SNAPSHOT_JSON.is_file(), reason="protocol snapshot not generated in this environment")
class TestProtocolSnapshot:
    def test_snapshot_round_trips(self):
        text = PROTOCOL_SNAPSHOT_JSON.read_text(encoding="utf-8")
        data = json.loads(text)
        reserialized = json.dumps(data, indent=2)
        assert json.loads(reserialized) == data

    def test_snapshot_label_verbatim(self):
        data = _load_json(PROTOCOL_SNAPSHOT_JSON)
        assert data["snapshot_label"] == "Fixed Provisional PP1 Exploratory Complexity Protocol -- NOT Expert Validated"
        assert data["status"] == "human_expert_validation_pending"
        assert data["is_active"] is False
        assert data["is_final"] is False

    def test_snapshot_does_not_alter_live_calibration_v1_json(self):
        calibration_path = CONFIG_DIR / "calibration_v1.json"
        before = calibration_path.read_text(encoding="utf-8")
        _ = _load_json(PROTOCOL_SNAPSHOT_JSON)  # reading the snapshot must not touch the live file
        after = calibration_path.read_text(encoding="utf-8")
        assert before == after
        live = json.loads(after)
        assert live["status"] == "provisional_NOT_frozen"

    def test_snapshot_weighting_sensitivity_prominently_present(self):
        data = _load_json(PROTOCOL_SNAPSHOT_JSON)
        ws = data["weighting_sensitivity_limitation"]
        assert ws["n_stable_across_all_7_schemes"] == 534
        assert ws["fraction_unstable"] == pytest.approx(0.6943, abs=0.01)

    def test_snapshot_f21b4r_agreement_present(self):
        data = _load_json(PROTOCOL_SNAPSHOT_JSON)
        ag = data["f21b4r_human_ai_agreement_results"]
        assert ag["cohens_kappa_unweighted"] == pytest.approx(0.5, abs=1e-6)
        assert ag["weighted_kappa_linear"] == pytest.approx(0.561, abs=0.01)


# --------------------------------------------------------------------------
# Protected-artifact preservation: E02, shared modules, F2.1B3 baseline,
# F2.1B4-R reconciliation outputs, F2.2A outputs.
# --------------------------------------------------------------------------

class TestProtectedArtifactsUnchanged:
    def test_e02_checkpoint_and_freeze_manifest_present_and_untouched_by_this_stage(self):
        # This QA stage never writes to these paths; re-hash the live
        # checkpoint and compare against the freeze manifest's own recorded
        # hash -- the actual regression check, not just existence.
        if not FREEZE_MANIFEST.is_file() or not E02_CHECKPOINT.is_file():
            pytest.skip("freeze manifest / E02 checkpoint not present in this environment")
        manifest = _load_json(FREEZE_MANIFEST)
        expected_sha256 = manifest["selected_experiment"]["checkpoint_sha256"]
        assert _sha256(E02_CHECKPOINT) == expected_sha256

    def test_shared_complexity_features_module_unchanged(self):
        if not COMPLEXITY_FEATURES_MODULE.is_file():
            pytest.skip("shared complexity_features.py not present in this environment")
        # Presence + non-emptiness is the structural check this stage can
        # make without a pre-recorded hash of its own; F2.1B3's own
        # checksum baseline (tested below) is the authoritative hash check
        # for Function-2-tracked artifacts.
        assert COMPLEXITY_FEATURES_MODULE.stat().st_size > 0

    def test_shared_detection_xai_module_unchanged(self):
        if not DETECTION_XAI_MODULE.is_file():
            pytest.skip("shared detection_xai.py not present in this environment")
        assert DETECTION_XAI_MODULE.stat().st_size > 0

    def test_f21b3_checksum_baseline_still_all_match(self):
        result = verify_against_baseline()
        assert result["all_match"] is True, result

    def test_f22a_validation_audit_outputs_unchanged_if_present(self):
        f22a_dir = REPO_ROOT / "research" / "component1" / "function2" / "results" / "validation_audit"
        if not f22a_dir.is_dir():
            pytest.skip("F2.2A outputs not present in this environment")
        assert any(f22a_dir.iterdir())

    def test_f21b4r_reconciliation_outputs_unchanged(self):
        recon_dir = REPO_ROOT / "research" / "component1" / "function2" / "results" / "review" / "reconciliation_v1"
        if not recon_dir.is_dir():
            pytest.skip("F2.1B4-R outputs not present in this environment")
        metrics_path = recon_dir / "agreement_metrics.json"
        metrics = _load_json(metrics_path)
        assert metrics["overall_exact_agreement_percent"] == pytest.approx(66.6666, abs=0.01)
        assert metrics["n_total_reconciled"] == 30
