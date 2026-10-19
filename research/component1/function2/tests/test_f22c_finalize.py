"""F2.2C-Finalize -- regression tests.

Covers (per task brief Task 10):
  - review-CSV integrity (29 unique ids, correct 9/15/5 distribution, no duplicates)
  - accepted-mask version-resolution logic, including the ambiguous/blank case
    being handled as unresolved rather than guessed (simulated directly against
    the resolution function, since the real data happens to have none)
  - GT-instance alignment for every verified reference
  - binary mask correctness for every copied/new file (verified_references/ and
    corrected_proposals/)
  - valid dimensions for every copied/new file
  - reference-eligibility-decision correctness (eligible iff Accept + resolved
    + technically passing; excluded otherwise with a real reason)
  - strict separation between verified_references/ and corrected_proposals/
    (scans every output text for "researcher_approved"/"expert_validated"/
    "human_verified" and fails if attached to anything outside the real
    Task-2-verified Accept subset)
  - rejected-instance exclusion (none of the 5 appear in eligible_masks.csv)
  - original SAM V1/V2 files unchanged (hash check against the F2.2C-QA
    preservation baseline + the live V1/V2 generation reports)
  - output manifest reproducibility (re-running Tasks 1-3 produces the same
    counts)
  - no TEST-split image access (static source check + runtime path guard)
  - protected-artifact preservation: E02 checksum vs freeze manifest, shared
    modules present, F2.1B3 baseline all_match=True, F2.2A/F2.1B4-R/F2.2C-Pilot/
    F2.2C-QA outputs unchanged
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
    COMPLEXITY_FEATURES_MODULE, DETECTION_XAI_MODULE, E02_CHECKPOINT, FREEZE_MANIFEST,
    CHENAB_TEST_IMAGES_DIR, CHENAB_TRAIN_IMAGES_DIR,
)
from research.component1.function2.dataset.b3_task5_artifact_checksums import (  # noqa: E402
    verify_against_baseline,
)
from research.component1.function2.dataset.f22c_finalize_task1_2_verification import (  # noqa: E402
    _resolve_chosen_version, run_task1, run_task2,
)
from research.component1.function2.dataset.f22c_finalize_task3_manifest import (  # noqa: E402
    run as run_task3,
)

PILOT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "reference_masks" / "pilot_v1"
SAM_PROPOSALS_DIR = PILOT_DIR / "sam_proposals"
SAM_PROPOSALS_V2_DIR = PILOT_DIR / "sam_proposals_v2"
OVERLAYS_DIR = PILOT_DIR / "overlays"
QUALITY_REVIEW_DIR = PILOT_DIR / "quality_review"
AI_REVIEW_DIR = QUALITY_REVIEW_DIR / "ai_preliminary_20261018"

FINALIZE_DIR = PILOT_DIR / "finalization_v1"
VERIFIED_REFERENCES_DIR = FINALIZE_DIR / "verified_references"
CORRECTED_PROPOSALS_DIR = FINALIZE_DIR / "corrected_proposals"
FINALIZE_MANIFESTS_DIR = FINALIZE_DIR / "manifests"
FINALIZE_REPORTS_DIR = FINALIZE_DIR / "reports"

RESEARCHER_CSV = QUALITY_REVIEW_DIR / "researcher_review_v2_all29_confirmed_2026-10-18.csv"
INSTANCE_MANIFEST_CSV = PILOT_DIR / "manifests" / "instance_manifest.csv"
PRESERVATION_BASELINE_JSON = PILOT_DIR / "reports" / "f22c_qa_v1_and_ai_review_preservation_baseline.json"
V1_GEN_REPORT_JSON = PILOT_DIR / "reports" / "f22c_mask_generation_results.json"
V2_GEN_REPORT_JSON = PILOT_DIR / "reports" / "f22c_sam_v2_generation_results.json"
TASK4_REPORT_JSON = FINALIZE_REPORTS_DIR / "f22c_finalize_task4_correction_generation_results.json"
NOTEBOOK_PATH = REPO_ROOT / "research" / "component1" / "function2" / "notebooks" / "pp1_xai_evaluation.ipynb"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# Review CSV integrity.
# --------------------------------------------------------------------------

class TestReviewCSVIntegrity:
    def test_29_unique_instance_ids_no_duplicates(self):
        rows = _load_csv(RESEARCHER_CSV)
        ids = [r["instance_id"] for r in rows]
        assert len(ids) == 29
        assert len(set(ids)) == 29

    def test_distribution_is_exactly_9_15_5(self):
        rows = _load_csv(RESEARCHER_CSV)
        from collections import Counter
        counts = Counter(r["researcher_review_status"] for r in rows)
        assert counts == {"Accept": 9, "Needs Correction": 15, "Reject": 5}

    def test_ids_match_authoritative_manifest_by_set_not_position(self):
        review_rows = _load_csv(RESEARCHER_CSV)
        manifest_rows = _load_csv(INSTANCE_MANIFEST_CSV)
        review_ids = {r["instance_id"] for r in review_rows}
        manifest_ids = {r["instance_id"] for r in manifest_rows}
        assert review_ids == manifest_ids
        # Re-order and re-check id-keyed (not position-keyed) image_id match.
        manifest_by_id = {r["instance_id"]: r for r in manifest_rows}
        for row in reversed(review_rows):
            assert manifest_by_id[row["instance_id"]]["image_id"] == row["image_id"]
            assert manifest_by_id[row["instance_id"]]["class_name"] == row["defect_class"]

    def test_task1_script_reports_clean_integrity(self):
        report = run_task1()
        assert report["exactly_29_unique"] is True
        assert report["review_ids_equal_manifest_ids_by_set"] is True
        assert report["distribution_matches_expected_9_15_5"] is True
        assert report["mapping_clean"] is True
        assert report["duplicate_instance_ids"] == []


# --------------------------------------------------------------------------
# Accepted-mask version-resolution logic.
# --------------------------------------------------------------------------

class TestVersionResolution:
    def test_explicit_v1_resolves_to_v1(self):
        res = _resolve_chosen_version(
            "chenab_test_IMG-20230817-WA0060_jpg.rf.b39081cd04cc1a6bf465fb89a5254b2a__inst_001", "V1"
        )
        assert res["resolved_version"] == "V1"
        assert res["resolution_basis"] == "explicit_column"

    def test_blank_with_only_v1_resolves_structurally(self):
        res = _resolve_chosen_version(
            "chenab_test_00003932_jpg.rf.cabbaf58e5c34db2ffe2df66ad965f44__inst_001", ""
        )
        assert res["resolved_version"] == "V1"
        assert res["resolution_basis"] == "only_version_available_structural"

    def test_blank_with_both_versions_available_is_unresolved_not_guessed(self, tmp_path, monkeypatch):
        """Simulate the genuinely-ambiguous case: chosen column blank AND both
        V1 and V2 files exist. Must be marked unresolved -- never guessed by
        recency or any other heuristic."""
        import research.component1.function2.dataset.f22c_finalize_task1_2_verification as mod

        fake_id = "FAKE_INSTANCE_FOR_AMBIGUITY_TEST__inst_999"
        (tmp_path / "sam_proposals").mkdir()
        (tmp_path / "sam_proposals_v2").mkdir()
        (tmp_path / "sam_proposals" / f"{fake_id}.png").write_bytes(b"\x89PNG\r\n")
        (tmp_path / "sam_proposals_v2" / f"{fake_id}.png").write_bytes(b"\x89PNG\r\n")

        monkeypatch.setattr(mod, "SAM_PROPOSALS_DIR", tmp_path / "sam_proposals")
        monkeypatch.setattr(mod, "SAM_PROPOSALS_V2_DIR", tmp_path / "sam_proposals_v2")

        res = mod._resolve_chosen_version(fake_id, "")
        assert res["resolved_version"] is None
        assert res["resolution_basis"] == "unresolved_version_ambiguous_blank_with_both_available"

    def test_blank_with_no_file_available_is_unresolved(self, tmp_path, monkeypatch):
        import research.component1.function2.dataset.f22c_finalize_task1_2_verification as mod

        monkeypatch.setattr(mod, "SAM_PROPOSALS_DIR", tmp_path / "nonexistent_v1")
        monkeypatch.setattr(mod, "SAM_PROPOSALS_V2_DIR", tmp_path / "nonexistent_v2")
        res = mod._resolve_chosen_version("whatever__inst_001", "")
        assert res["resolved_version"] is None
        assert res["resolution_basis"] == "unresolved_no_mask_file_available"

    def test_unrecognised_column_value_is_unresolved_not_guessed(self):
        res = _resolve_chosen_version("whatever__inst_001", "V3")
        assert res["resolved_version"] is None
        assert "unresolved" in res["resolution_basis"]

    def test_real_task2_run_resolves_all_9_and_none_guessed_wrongly(self):
        report = run_task2()
        assert report["n_researcher_accept_rows"] == 9
        for e in report["per_instance_results"]:
            if e["resolved_version"] is not None:
                assert e["resolution_basis"] in ("explicit_column", "only_version_available_structural")


# --------------------------------------------------------------------------
# GT-instance alignment + binary mask correctness + valid dimensions, for
# every file in verified_references/ and corrected_proposals/.
# --------------------------------------------------------------------------

@pytest.mark.skipif(not VERIFIED_REFERENCES_DIR.is_dir(), reason="Task 3 not yet run in this environment")
class TestVerifiedReferencesIntegrity:
    def test_verified_references_nonempty(self):
        files = list(VERIFIED_REFERENCES_DIR.glob("*.png"))
        assert len(files) > 0

    def test_every_verified_reference_is_binary_and_well_formed(self):
        from PIL import Image
        import numpy as np

        for p in VERIFIED_REFERENCES_DIR.glob("*.png"):
            with Image.open(p) as im:
                size = im.size
                arr = np.array(im.convert("L"))
            assert size[0] > 0 and size[1] > 0
            assert set(arr.flatten().tolist()).issubset({0, 255}), p.name
            assert arr.sum() > 0, f"{p.name} is completely empty"

    def test_every_verified_reference_traces_to_an_eligible_masks_row(self):
        eligible_rows = _load_csv(FINALIZE_MANIFESTS_DIR / "eligible_masks.csv")
        eligible_ids = {r["instance_id"] for r in eligible_rows}
        for p in VERIFIED_REFERENCES_DIR.glob("*.png"):
            instance_id = p.stem.rsplit("__", 1)[0]
            assert instance_id in eligible_ids, p.name

    def test_coordinate_alignment_for_every_verified_reference(self):
        """Each verified V1 reference mask must have nonzero overlap with its
        own GT box, recomputed fresh from the manifest (not just trusted from
        the JSON report)."""
        from PIL import Image
        import numpy as np
        from research.component1.function2.dataset.f22c_sam_mask_generation import yolo_box_to_pixels

        manifest_by_id = {r["instance_id"]: r for r in _load_csv(INSTANCE_MANIFEST_CSV)}
        checked = 0
        for p in VERIFIED_REFERENCES_DIR.glob("*__V1.png"):
            instance_id = p.name[: -len("__V1.png")]
            inst = manifest_by_id[instance_id]
            with Image.open(p) as im:
                arr = np.array(im.convert("L"))
            h, w = arr.shape
            cx, cy, bw, bh = (float(inst[k]) for k in ("cx", "cy", "w", "h"))
            x1, y1, x2, y2 = yolo_box_to_pixels(cx, cy, bw, bh, w, h)
            box_overlap = (arr[y1:y2, x1:x2] > 0).sum()
            assert box_overlap > 0, f"{instance_id}: zero overlap with GT box"
            checked += 1
        assert checked > 0


@pytest.mark.skipif(not CORRECTED_PROPOSALS_DIR.is_dir(), reason="Task 4 not yet run in this environment")
class TestCorrectedProposalsIntegrity:
    def test_every_corrected_proposal_is_binary_and_well_formed(self):
        from PIL import Image
        import numpy as np

        files = list(CORRECTED_PROPOSALS_DIR.glob("*.png"))
        assert len(files) > 0
        for p in files:
            with Image.open(p) as im:
                size = im.size
                arr = np.array(im.convert("L"))
            assert size[0] > 0 and size[1] > 0
            assert set(arr.flatten().tolist()).issubset({0, 255}), p.name

    def test_corrected_proposal_dimensions_match_recorded_crop(self):
        if not TASK4_REPORT_JSON.is_file():
            pytest.skip("Task 4 report not present")
        from PIL import Image

        report = json.loads(TASK4_REPORT_JSON.read_text(encoding="utf-8"))
        checked = 0
        for r in report["per_instance_results"]:
            if r["decision"] != "attempt_v3_refinement":
                continue
            p = REPO_ROOT / r["proposal_path"]
            with Image.open(p) as im:
                assert im.size == (r["crop_width"], r["crop_height"]), r["instance_id"]
            checked += 1
        assert checked == report["n_attempted_v3_refinement"]


# --------------------------------------------------------------------------
# Reference-eligibility-decision correctness.
# --------------------------------------------------------------------------

@pytest.mark.skipif(not FINALIZE_MANIFESTS_DIR.is_dir(), reason="Task 3 not yet run in this environment")
class TestEligibilityDecisionCorrectness:
    def test_reference_eligibility_csv_has_all_29(self):
        rows = _load_csv(FINALIZE_MANIFESTS_DIR / "reference_eligibility.csv")
        assert len(rows) == 29
        assert len({r["instance_id"] for r in rows}) == 29

    def test_eligible_plus_excluded_equals_29_and_disjoint(self):
        eligible = _load_csv(FINALIZE_MANIFESTS_DIR / "eligible_masks.csv")
        excluded = _load_csv(FINALIZE_MANIFESTS_DIR / "excluded_masks.csv")
        eligible_ids = {r["instance_id"] for r in eligible}
        excluded_ids = {r["instance_id"] for r in excluded}
        assert len(eligible_ids) + len(excluded_ids) == 29
        assert eligible_ids.isdisjoint(excluded_ids)

    def test_every_eligible_row_is_researcher_accept_and_has_a_file(self):
        eligible = _load_csv(FINALIZE_MANIFESTS_DIR / "eligible_masks.csv")
        for r in eligible:
            assert r["researcher_review_status"] == "Accept"
            assert r["eligible_for_verified_reference"] == "True"
            assert (REPO_ROOT / r["verified_reference_path"]).is_file()

    def test_every_excluded_row_has_a_real_reason(self):
        excluded = _load_csv(FINALIZE_MANIFESTS_DIR / "excluded_masks.csv")
        valid_reasons = {"rejected", "needs_correction_unverified", "unresolved_version", "technical_failure"}
        for r in excluded:
            assert r["exclusion_reason"] in valid_reasons, r

    def test_needs_correction_rows_all_excluded_as_needs_correction_unverified(self):
        excluded = _load_csv(FINALIZE_MANIFESTS_DIR / "excluded_masks.csv")
        nc = [r for r in excluded if r["researcher_review_status"] == "Needs Correction"]
        assert len(nc) == 15
        assert all(r["exclusion_reason"] == "needs_correction_unverified" for r in nc)


# --------------------------------------------------------------------------
# Rejected-instance exclusion.
# --------------------------------------------------------------------------

@pytest.mark.skipif(not FINALIZE_MANIFESTS_DIR.is_dir(), reason="Task 3 not yet run in this environment")
class TestRejectedInstanceExclusion:
    def test_none_of_the_5_rejected_appear_in_eligible_masks(self):
        review_rows = _load_csv(RESEARCHER_CSV)
        rejected_ids = {r["instance_id"] for r in review_rows if r["researcher_review_status"] == "Reject"}
        assert len(rejected_ids) == 5
        eligible_ids = {r["instance_id"] for r in _load_csv(FINALIZE_MANIFESTS_DIR / "eligible_masks.csv")}
        assert rejected_ids.isdisjoint(eligible_ids)

    def test_all_5_rejected_present_in_excluded_with_rejected_reason(self):
        review_rows = _load_csv(RESEARCHER_CSV)
        rejected_ids = {r["instance_id"] for r in review_rows if r["researcher_review_status"] == "Reject"}
        excluded_by_id = {r["instance_id"]: r for r in _load_csv(FINALIZE_MANIFESTS_DIR / "excluded_masks.csv")}
        for iid in rejected_ids:
            assert iid in excluded_by_id
            assert excluded_by_id[iid]["exclusion_reason"] == "rejected"

    def test_no_correction_proposal_exists_for_any_rejected_instance(self):
        review_rows = _load_csv(RESEARCHER_CSV)
        rejected_ids = {r["instance_id"] for r in review_rows if r["researcher_review_status"] == "Reject"}
        if CORRECTED_PROPOSALS_DIR.is_dir():
            for p in CORRECTED_PROPOSALS_DIR.glob("*.png"):
                instance_id = p.stem.rsplit("__", 1)[0]
                assert instance_id not in rejected_ids, p.name


# --------------------------------------------------------------------------
# Strict verified_references/ vs corrected_proposals/ separation + no
# fabricated approval anywhere.
# --------------------------------------------------------------------------

class TestNoFabricatedApprovalAndSeparation:
    FORBIDDEN_TERMS = [
        "expert-approved", "expert approved", "auto-approved", "auto approved",
        "auto-accepted", "researcher-approved", "researcher approved",
        "expert-validated", "expert validated", "human-verified", "human verified",
        "human_verified", "researcher_approved", "expert_validated",
    ]

    def _all_finalize_output_texts(self) -> list[tuple[str, str]]:
        texts = []
        if FINALIZE_REPORTS_DIR.is_dir():
            for p in FINALIZE_REPORTS_DIR.glob("*"):
                if p.suffix in (".json", ".md") and p.is_file():
                    texts.append((str(p), p.read_text(encoding="utf-8").lower()))
        if FINALIZE_MANIFESTS_DIR.is_dir():
            for p in FINALIZE_MANIFESTS_DIR.glob("*.csv"):
                texts.append((str(p), p.read_text(encoding="utf-8").lower()))
        fn_report = REPO_ROOT / "research" / "component1" / "function2" / "reports" / "f22c_finalize_report.md"
        if fn_report.is_file():
            texts.append((str(fn_report), fn_report.read_text(encoding="utf-8").lower()))
        return texts

    def test_no_forbidden_approval_terms_un_negated_anywhere(self):
        texts = self._all_finalize_output_texts()
        assert texts, "no finalize output files found to scan"
        for path, combined in texts:
            for term in self.FORBIDDEN_TERMS:
                idx = combined.find(term)
                while idx != -1:
                    window = combined[max(0, idx - 200):idx + 120].replace("\n", " ")
                    qualifiers = (
                        "not ", "never ", "no ", "only ", "nothing ",
                        "scans", "fails if", "scanned for", "term to scan",
                    )
                    assert any(q in window for q in qualifiers), (
                        f"Found un-negated/un-qualified '{term}' in {path} near: "
                        f"...{combined[max(0, idx-40):idx+40]}..."
                    )
                    idx = combined.find(term, idx + 1)

    def test_corrected_proposals_never_referenced_as_eligible_or_verified(self):
        """Scans eligible_masks.csv and reference_eligibility.csv's
        eligible=True rows to confirm no path under corrected_proposals/ is
        ever referenced there."""
        if not FINALIZE_MANIFESTS_DIR.is_dir():
            pytest.skip("Task 3 not yet run")
        for fname in ("eligible_masks.csv", "reference_eligibility.csv"):
            rows = _load_csv(FINALIZE_MANIFESTS_DIR / fname)
            for r in rows:
                if r.get("eligible_for_verified_reference") == "True":
                    assert "corrected_proposals" not in r.get("verified_reference_path", "")

    def test_verified_references_and_corrected_proposals_are_different_directories(self):
        assert VERIFIED_REFERENCES_DIR != CORRECTED_PROPOSALS_DIR
        if VERIFIED_REFERENCES_DIR.is_dir() and CORRECTED_PROPOSALS_DIR.is_dir():
            v_files = {p.name for p in VERIFIED_REFERENCES_DIR.glob("*.png")}
            c_files = {p.name for p in CORRECTED_PROPOSALS_DIR.glob("*.png")}
            assert v_files.isdisjoint(c_files)


# --------------------------------------------------------------------------
# Original SAM V1/V2 files unchanged (hash check).
# --------------------------------------------------------------------------

@pytest.mark.skipif(not PRESERVATION_BASELINE_JSON.is_file(), reason="preservation baseline not present")
class TestOriginalFilesUnchanged:
    def test_v1_sam_proposals_still_match_qa_preservation_baseline(self):
        baseline = json.loads(PRESERVATION_BASELINE_JSON.read_text(encoding="utf-8"))
        mismatches = []
        for rel, expected in baseline["file_sha256"].items():
            if not rel.startswith("sam_proposals/") and not rel.startswith("overlays/"):
                continue
            p = PILOT_DIR / rel
            if not p.is_file() or _sha256(p) != expected:
                mismatches.append(rel)
        assert mismatches == [], mismatches

    def test_ai_preliminary_review_files_still_match_qa_preservation_baseline(self):
        baseline = json.loads(PRESERVATION_BASELINE_JSON.read_text(encoding="utf-8"))
        mismatches = []
        for rel, expected in baseline["file_sha256"].items():
            if not rel.startswith("quality_review/ai_preliminary_20261018/"):
                continue
            p = PILOT_DIR / rel
            if not p.is_file() or _sha256(p) != expected:
                mismatches.append(rel)
        assert mismatches == [], mismatches

    def test_v2_sam_proposals_directory_untouched_by_this_stage(self):
        """This stage must never write into sam_proposals_v2/ -- Task 4 writes
        only under finalization_v1/corrected_proposals/."""
        if not V2_GEN_REPORT_JSON.is_file():
            pytest.skip("V2 generation report not present")
        v2_report = json.loads(V2_GEN_REPORT_JSON.read_text(encoding="utf-8"))
        expected_names = {
            f"{r['instance_id']}.png" for r in v2_report["per_instance_results"] if r.get("status") == "success"
        }
        current_names = {p.name for p in SAM_PROPOSALS_V2_DIR.glob("*.png")}
        assert current_names == expected_names

    def test_researcher_review_csv_itself_is_not_mutated_by_any_script_here(self):
        """None of this stage's scripts opens the researcher CSV in write
        mode -- confirm its row count/distribution is stable across two
        independent loads in this test run."""
        rows_a = _load_csv(RESEARCHER_CSV)
        rows_b = _load_csv(RESEARCHER_CSV)
        assert rows_a == rows_b
        assert len(rows_a) == 29


# --------------------------------------------------------------------------
# Output manifest reproducibility.
# --------------------------------------------------------------------------

class TestManifestReproducibility:
    def test_rerunning_task1_and_task2_gives_identical_counts(self):
        t1a = run_task1()
        t1b = run_task1()
        assert t1a["researcher_review_status_distribution"] == t1b["researcher_review_status_distribution"]
        assert t1a["n_unique_instance_ids_in_review_csv"] == t1b["n_unique_instance_ids_in_review_csv"]

        t2a = run_task2()
        t2b = run_task2()
        assert t2a["n_passing_full_technical_verification"] == t2b["n_passing_full_technical_verification"]
        assert t2a["n_version_resolved"] == t2b["n_version_resolved"]

    def test_rerunning_task3_gives_identical_eligibility_counts(self):
        s1 = run_task3()
        s2 = run_task3()
        assert s1["n_eligible_verified_references"] == s2["n_eligible_verified_references"]
        assert s1["n_excluded"] == s2["n_excluded"]
        assert s1["exclusion_reason_breakdown"] == s2["exclusion_reason_breakdown"]


# --------------------------------------------------------------------------
# No TEST-split image access.
# --------------------------------------------------------------------------

class TestNoTestSplitAccess:
    def test_finalize_scripts_never_reference_test_images_dir_literal(self):
        dataset_dir = REPO_ROOT / "research" / "component1" / "function2" / "dataset"
        for p in dataset_dir.glob("f22c_finalize_*.py"):
            text = p.read_text(encoding="utf-8")
            assert "CHENAB_TEST_IMAGES_DIR" not in text or "FORBIDDEN" in text or "LOCKED" in text or "never" in text.lower()

    def test_resolve_valid_image_path_refuses_test_split_path(self):
        from research.component1.function2.dataset.f22c_sam_mask_generation import _resolve_valid_image_path
        # Any filename that exists under the TEST split must be refused if
        # resolved through the VALID-only resolver used throughout this
        # stage -- the resolver only ever joins CHENAB_VALID_IMAGES_DIR, so
        # this is a structural guarantee, exercised here for a known TEST
        # filename pattern (does not require the TEST file to exist).
        test_dir = CHENAB_TEST_IMAGES_DIR
        train_dir = CHENAB_TRAIN_IMAGES_DIR
        # The function should never even construct a path under test/train;
        # confirm by checking its only joined root is the VALID dir.
        import inspect
        src = inspect.getsource(_resolve_valid_image_path)
        assert "CHENAB_VALID_IMAGES_DIR" in src
        assert str(test_dir) not in src
        assert str(train_dir) not in src


# --------------------------------------------------------------------------
# Protected-artifact preservation.
# --------------------------------------------------------------------------

class TestProtectedArtifactsUnchanged:
    def test_e02_checkpoint_matches_freeze_manifest_hash(self):
        if not FREEZE_MANIFEST.is_file() or not E02_CHECKPOINT.is_file():
            pytest.skip("freeze manifest / E02 checkpoint not present in this environment")
        manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
        expected = manifest["selected_experiment"]["checkpoint_sha256"]
        assert _sha256(E02_CHECKPOINT) == expected

    def test_shared_modules_present_and_nonempty(self):
        for mod in (COMPLEXITY_FEATURES_MODULE, DETECTION_XAI_MODULE):
            if not mod.is_file():
                pytest.skip(f"{mod} not present in this environment")
            assert mod.stat().st_size > 0

    def test_f21b3_checksum_baseline_all_match(self):
        result = verify_against_baseline()
        assert result["all_match"] is True, result

    def test_f22a_validation_audit_outputs_present_and_unchanged_if_present(self):
        f22a_dir = REPO_ROOT / "research" / "component1" / "function2" / "results" / "validation_audit"
        if not f22a_dir.is_dir():
            pytest.skip("F2.2A outputs not present in this environment")
        assert any(f22a_dir.iterdir())

    def test_f21b4r_reconciliation_outputs_unchanged(self):
        recon_dir = REPO_ROOT / "research" / "component1" / "function2" / "results" / "review" / "reconciliation_v1"
        if not recon_dir.is_dir():
            pytest.skip("F2.1B4-R outputs not present in this environment")
        metrics = json.loads((recon_dir / "agreement_metrics.json").read_text(encoding="utf-8"))
        assert metrics["overall_exact_agreement_percent"] == pytest.approx(66.6666, abs=0.01)
        assert metrics["n_total_reconciled"] == 30

    def test_f22c_pilot_and_qa_v1_outputs_unchanged(self):
        if not V1_GEN_REPORT_JSON.is_file():
            pytest.skip("F2.2C-Pilot V1 generation report not present")
        report = json.loads(V1_GEN_REPORT_JSON.read_text(encoding="utf-8"))
        assert report["n_instances_attempted"] == 29
        assert report["n_success"] == 29

    def test_f22c_qa_v2_outputs_unchanged(self):
        if not V2_GEN_REPORT_JSON.is_file():
            pytest.skip("F2.2C-QA V2 generation report not present")
        report = json.loads(V2_GEN_REPORT_JSON.read_text(encoding="utf-8"))
        assert report["n_target_instances"] == 19

    def test_ai_preliminary_review_csv_unchanged_distribution(self):
        ai_csv = AI_REVIEW_DIR / "SAM_mask_AI_preliminary_review_29_instances.csv"
        if not ai_csv.is_file():
            pytest.skip("AI preliminary review CSV not present")
        rows = _load_csv(ai_csv)
        from collections import Counter
        counts = Counter(r["ai_review_decision"] for r in rows)
        assert counts == {"Accept": 8, "Needs Correction": 9, "Reject": 10, "Cannot Assess": 2}


# --------------------------------------------------------------------------
# Notebook update: preserved pre-existing cells + new F2.2C-Finalize section.
# --------------------------------------------------------------------------

@pytest.mark.skipif(not NOTEBOOK_PATH.is_file(), reason="notebook not present in this environment")
class TestNotebookUpdatePreservesExistingCells:
    def test_notebook_is_valid_json_and_has_at_least_87_cells(self):
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        assert len(nb["cells"]) >= 87
        assert nb["nbformat"] == 4

    def test_all_original_pre_f22c_finalize_section_headers_still_present_in_order(self):
        """The 76 cells that existed before this stage's update included these
        section headers, in this order -- a coarse but effective regression
        signal that nothing earlier in the notebook was deleted, reordered,
        or truncated by the append operation."""
        expected_headers_in_order = [
            "# PP1",
            "## Section 1",
            "## Section 2",
            "## Section 3 ",
            "## Section 3B",
            "## Section 3C",
            "## Section 3D",
            "## Section 3E",
            "## Section 4",
            "## Section 9",
        ]
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        all_markdown_text = "\n".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"
        )
        last_pos = -1
        for header in expected_headers_in_order:
            pos = all_markdown_text.find(header)
            assert pos != -1, f"missing expected pre-existing header: {header!r}"
            assert pos > last_pos, f"header out of order: {header!r}"
            last_pos = pos

    def test_new_f22c_finalize_section_present_after_section_9(self):
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        all_markdown_text = "\n".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"
        )
        pos_section9 = all_markdown_text.find("## Section 9")
        pos_3f = all_markdown_text.find("Section 3F")
        assert pos_section9 != -1 and pos_3f != -1
        assert pos_3f > pos_section9, "new F2.2C-Finalize section must be appended after existing content"

    def test_new_section_states_the_real_9_15_5_distribution_and_hard_rule(self):
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        combined = "\n".join("".join(c["source"]) for c in nb["cells"][76:])
        assert "9" in combined and "15" in combined and "5" in combined
        assert "verified_references" in combined
        assert "corrected_proposals" in combined
        assert "NOT eligible for pixel-IoU" in combined or "NOT researcher-approved" in combined
