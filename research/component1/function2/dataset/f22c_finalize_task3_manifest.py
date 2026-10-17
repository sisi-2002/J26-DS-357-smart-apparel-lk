"""F2.2C-Finalize Task 3 -- verified-reference manifest + directory build.

Builds ``results/reference_masks/pilot_v1/finalization_v1/`` with:

  * ``verified_references/``   -- COPIES (never moves) of the mask files for
    the instances that passed Task 2's full technical verification. This is
    the ONLY directory whose contents may ever be described as
    "researcher-approved reference mask" anywhere downstream.
  * ``manifests/reference_eligibility.csv`` -- all 29 instances with their
    eligibility determination + reason.
  * ``manifests/eligible_masks.csv``        -- just the passing subset.
  * ``manifests/excluded_masks.csv``        -- everyone else, with an
    explicit exclusion reason (rejected / needs_correction_unverified /
    unresolved_version / technical_failure).
  * ``corrected_proposals/`` and ``comparison/`` -- created empty here;
    populated by Task 4's script. NEVER treated as verified references.

This script never writes to, overwrites, or renames anything under the
original ``sam_proposals/``, ``sam_proposals_v2/``, ``overlays/`` or
``overlays_v2/`` directories -- it only reads from them and copies into the
new ``finalization_v1/`` tree.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.dataset.f22c_finalize_task1_2_verification import (  # noqa: E402
    run_task1, run_task2, STATUS_ACCEPT, STATUS_NEEDS_CORRECTION, STATUS_REJECT,
    RESEARCHER_CSV, INSTANCE_MANIFEST_CSV, _load_csv,
)

PILOT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "reference_masks" / "pilot_v1"
FINALIZE_DIR = PILOT_DIR / "finalization_v1"
VERIFIED_REFERENCES_DIR = FINALIZE_DIR / "verified_references"
FINALIZE_MANIFESTS_DIR = FINALIZE_DIR / "manifests"
CORRECTED_PROPOSALS_DIR = FINALIZE_DIR / "corrected_proposals"
COMPARISON_DIR = FINALIZE_DIR / "comparison"
FINALIZE_REPORTS_DIR = FINALIZE_DIR / "reports"

SAM_PROPOSALS_DIR = PILOT_DIR / "sam_proposals"
SAM_PROPOSALS_V2_DIR = PILOT_DIR / "sam_proposals_v2"

ELIGIBILITY_FIELDNAMES = [
    "instance_id", "image_id", "defect_class", "researcher_review_status",
    "resolved_version", "resolution_basis", "eligible_for_verified_reference",
    "exclusion_reason", "verified_reference_path", "technical_notes",
]


def run() -> dict:
    for d in (VERIFIED_REFERENCES_DIR, FINALIZE_MANIFESTS_DIR, CORRECTED_PROPOSALS_DIR,
              COMPARISON_DIR, FINALIZE_REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    t1 = run_task1()
    t2 = run_task2()

    review_rows = _load_csv(RESEARCHER_CSV)
    t2_by_id = {e["instance_id"]: e for e in t2["per_instance_results"]}

    eligibility_rows = []
    eligible_rows = []
    excluded_rows = []
    n_copied = 0

    for row in review_rows:
        instance_id = row["instance_id"]
        status = row["researcher_review_status"]
        base = {
            "instance_id": instance_id,
            "image_id": row["image_id"],
            "defect_class": row["defect_class"],
            "researcher_review_status": status,
            "resolved_version": "",
            "resolution_basis": "",
            "eligible_for_verified_reference": "False",
            "exclusion_reason": "",
            "verified_reference_path": "",
            "technical_notes": "",
        }

        if status == STATUS_REJECT:
            base["exclusion_reason"] = "rejected"
            base["technical_notes"] = "Researcher-rejected instance; no replacement/correction attempted (per policy)."
            eligibility_rows.append(base)
            excluded_rows.append(base)
            continue

        if status == STATUS_NEEDS_CORRECTION:
            base["exclusion_reason"] = "needs_correction_unverified"
            base["technical_notes"] = (
                "Researcher requested correction; any Task 4 corrected proposal is an "
                "UNREVIEWED algorithmic candidate, not researcher-verified, and is therefore "
                "not eligible for pixel-IoU reference use regardless of its visual quality."
            )
            eligibility_rows.append(base)
            excluded_rows.append(base)
            continue

        if status == STATUS_ACCEPT:
            t2e = t2_by_id.get(instance_id)
            if t2e is None:
                base["exclusion_reason"] = "technical_failure"
                base["technical_notes"] = "No Task 2 verification record found (unexpected)."
                eligibility_rows.append(base)
                excluded_rows.append(base)
                continue

            base["resolved_version"] = t2e["resolved_version"] or ""
            base["resolution_basis"] = t2e["resolution_basis"]

            if t2e["resolved_version"] is None:
                base["exclusion_reason"] = "unresolved_version"
                base["technical_notes"] = (
                    f"chosen_proposal_version column was ambiguous/blank and could not be "
                    f"safely resolved ({t2e['resolution_basis']}); excluded rather than guessed."
                )
                eligibility_rows.append(base)
                excluded_rows.append(base)
                continue

            tv = t2e["technical_verification"]
            if not tv["overall_pass"]:
                base["exclusion_reason"] = "technical_failure"
                base["technical_notes"] = "; ".join(tv["issues"])
                eligibility_rows.append(base)
                excluded_rows.append(base)
                continue

            # Passes everything -- copy the mask into verified_references/.
            version = t2e["resolved_version"]
            src_dir = SAM_PROPOSALS_DIR if version == "V1" else SAM_PROPOSALS_V2_DIR
            src_path = src_dir / f"{instance_id}.png"
            dst_path = VERIFIED_REFERENCES_DIR / f"{instance_id}__{version}.png"
            shutil.copy2(src_path, dst_path)
            n_copied += 1

            base["eligible_for_verified_reference"] = "True"
            base["verified_reference_path"] = str(dst_path.relative_to(REPO_ROOT).as_posix())
            notes = tv.get("notes", [])
            base["technical_notes"] = "; ".join(notes) if notes else "All technical checks passed cleanly."
            eligibility_rows.append(base)
            eligible_rows.append(base)
            continue

        # Unknown status value -- should never happen given Task 1 checks.
        base["exclusion_reason"] = f"unknown_researcher_review_status:{status!r}"
        eligibility_rows.append(base)
        excluded_rows.append(base)

    def _write_csv(path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ELIGIBILITY_FIELDNAMES)
            w.writeheader()
            w.writerows(rows)

    _write_csv(FINALIZE_MANIFESTS_DIR / "reference_eligibility.csv", eligibility_rows)
    _write_csv(FINALIZE_MANIFESTS_DIR / "eligible_masks.csv", eligible_rows)
    _write_csv(FINALIZE_MANIFESTS_DIR / "excluded_masks.csv", excluded_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "F2.2C-Finalize Task 3",
        "n_total_instances": len(eligibility_rows),
        "n_eligible_verified_references": len(eligible_rows),
        "n_excluded": len(excluded_rows),
        "n_mask_files_copied_into_verified_references": n_copied,
        "exclusion_reason_breakdown": {
            reason: sum(1 for r in excluded_rows if r["exclusion_reason"] == reason)
            for reason in sorted({r["exclusion_reason"] for r in excluded_rows})
        },
        "status_label": (
            "verified_references/ is the ONLY directory in this stage whose "
            "contents may be described as researcher-approved reference masks. "
            "corrected_proposals/ (Task 4) is explicitly NOT eligible for this "
            "label under any circumstance."
        ),
    }
    with open(FINALIZE_REPORTS_DIR / "f22c_finalize_task3_manifest_build.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
