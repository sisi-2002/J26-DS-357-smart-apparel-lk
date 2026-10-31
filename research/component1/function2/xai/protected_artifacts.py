"""Task 7 — extended protected-artifact checksum regression check.

Extends the existing pattern already used by
``research/component1/function2/tests/test_f22b_pipeline.py``
(``PROTECTED_FILE_BASELINE``) and
``research/component1/function2/tests/test_f22c_finalize.py``
(``TestProtectedArtifactsUnchanged``) to the additional items this F2.3-A
stage's brief requires to be verified byte-unchanged:
  - E02 checkpoint + freeze manifest,
  - the two protected shared modules (complexity_features.py, detection_xai.py),
  - the complexity protocol snapshot + its own hash file + calibration_v1.json,
  - F2.1B4-R reconciliation artifacts,
  - F2.2C SAM/verified-reference mask manifests + the 9 verified reference PNGs,
  - F2.2B's own sampling manifests (valid_complexity_manifest.csv,
    xai_primary_sample.csv, xai_dry_run_sample.csv,
    boundary_sensitivity_sample.csv, pixel_iou_pilot_subset.csv),
  - the existing notebook (byte-unchanged -- this stage adds no notebook
    cells; that is explicitly out of scope per the task brief).

The known-good hashes for the first few items are the SAME ones already
recorded in test_f22b_pipeline.py's PROTECTED_FILE_BASELINE (re-declared
here, not recomputed, so a real drift is caught against an independent,
pre-existing source of truth rather than against a baseline this same
stage just generated).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

# Hashes already recorded by the PRIOR (F2.2B) stage's own protected-artifact
# test (test_f22b_pipeline.py::PROTECTED_FILE_BASELINE) -- re-declared here
# as an independent cross-check, not recomputed from the current file state.
KNOWN_GOOD_HASHES = {
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
    "research/component1/function2/results/reference_masks/pilot_v1/finalization_v1/manifests/eligible_masks.csv":
        "75136e09707ada96da6764d16df64b8fc15ca87cb2915d6cdad0eb76086bb681",
    "research/component1/function2/results/reference_masks/pilot_v1/finalization_v1/manifests/excluded_masks.csv":
        "680a0017e61c4b42719a01a0f827a969f68abd411a9c76f3e632d05e048a8466",
    "research/component1/function2/results/reference_masks/pilot_v1/finalization_v1/manifests/reference_eligibility.csv":
        "267f1e00319bee641677c56600464e42b1f1f46bb99e1a6e8b44f4339cd3d50c",
    "research/component1/function2/results/reference_masks/pilot_v1/finalization_v1/manifests/correction_results.csv":
        "69bc4ecc031b20c82528a562222a7e2ae10a0d14854a3da295d7b978afe746ea",
}

# Items with no pre-existing cross-stage test asserting their hash. These are
# verified via before/after comparison WITHIN this stage's own run (snapshot
# taken before any F2.3-A code was written; see
# results/xai_execution/f23_dry_run_v1/checksums/pre_stage_baseline.json),
# not against an independent prior-stage source.
ADDITIONAL_PROTECTED_RELATIVE_PATHS = [
    "runs/component1_training/E02_optimized/weights/best.pt",
    "runs/component1_training/freeze/model_freeze_E02_optimized_v1.json",
    "research/component1/function2/config/b3_artifact_checksums_baseline.json",
    "research/component1/function2/config/f22b_pp1_exploratory_run_v1.json",
    "research/component1/function2/results/evaluation_sampling/f22b_pp1_v1/manifests/valid_complexity_manifest.csv",
    "research/component1/function2/results/evaluation_sampling/f22b_pp1_v1/manifests/xai_primary_sample.csv",
    "research/component1/function2/results/evaluation_sampling/f22b_pp1_v1/manifests/xai_dry_run_sample.csv",
    "research/component1/function2/results/evaluation_sampling/f22b_pp1_v1/manifests/boundary_sensitivity_sample.csv",
    "research/component1/function2/results/evaluation_sampling/f22b_pp1_v1/manifests/pixel_iou_pilot_subset.csv",
    "research/component1/function2/notebooks/pp1_xai_evaluation.ipynb",
]

_VERIFIED_REF_DIR = (
    REPO_ROOT / "research/component1/function2/results/reference_masks"
    "/pilot_v1/finalization_v1/verified_references"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def all_protected_relative_paths() -> list[str]:
    paths = list(KNOWN_GOOD_HASHES.keys()) + list(ADDITIONAL_PROTECTED_RELATIVE_PATHS)
    if _VERIFIED_REF_DIR.is_dir():
        for p in sorted(_VERIFIED_REF_DIR.glob("*.png")):
            paths.append(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
    return paths


def compute_current_hashes() -> dict[str, str]:
    out = {}
    for rel in all_protected_relative_paths():
        p = REPO_ROOT / rel
        if p.is_file():
            out[rel] = _sha256(p)
    return out


def verify_known_good_hashes() -> dict:
    """Compare the subset with an independent, pre-existing known-good hash
    (from the F2.2B stage's own test suite) against the current file state."""
    results = {}
    for rel, expected in KNOWN_GOOD_HASHES.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            results[rel] = {"status": "MISSING", "expected": expected, "actual": None}
            continue
        actual = _sha256(p)
        results[rel] = {
            "status": "PASS" if actual == expected else "FAIL",
            "expected": expected, "actual": actual,
        }
    return results


def verify_against_pre_stage_snapshot(pre_stage_snapshot: dict[str, str]) -> dict:
    """Compare every protected item's current hash against a snapshot taken
    before this stage's work began."""
    results = {}
    for rel, expected in pre_stage_snapshot.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            results[rel] = {"status": "MISSING", "expected": expected, "actual": None}
            continue
        actual = _sha256(p)
        results[rel] = {
            "status": "PASS" if actual == expected else "FAIL",
            "expected": expected, "actual": actual,
        }
    return results
