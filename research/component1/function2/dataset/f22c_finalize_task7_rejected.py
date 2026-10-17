"""F2.2C-Finalize Task 7 -- record of the 5 researcher-rejected instances.

Pulls the REAL recorded rejection reason (``required_correction`` column --
this is the column populated with the rejection rationale for Reject rows in
the source CSV) verbatim from
``researcher_review_v2_all29_confirmed_2026-10-18.csv`` for each of the 5
Reject instances, confirms each remains excluded in
``finalization_v1/manifests/excluded_masks.csv``, and confirms (by
construction elsewhere in this stage) that no replacement/correction
attempt was made for any of them -- rejected means rejected, not "needs
correction". This script performs no SAM calls and generates no new mask.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.dataset.f22c_finalize_task1_2_verification import (  # noqa: E402
    RESEARCHER_CSV, _load_csv, STATUS_REJECT,
)
from research.component1.function2.dataset.f22c_finalize_task4_corrections import (  # noqa: E402
    PER_INSTANCE_PLAN,
)

PILOT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "reference_masks" / "pilot_v1"
FINALIZE_DIR = PILOT_DIR / "finalization_v1"
FINALIZE_MANIFESTS_DIR = FINALIZE_DIR / "manifests"
FINALIZE_REPORTS_DIR = FINALIZE_DIR / "reports"
EXCLUDED_MASKS_CSV = FINALIZE_MANIFESTS_DIR / "excluded_masks.csv"
ELIGIBLE_MASKS_CSV = FINALIZE_MANIFESTS_DIR / "eligible_masks.csv"


def run() -> dict:
    review_rows = _load_csv(RESEARCHER_CSV)
    reject_rows = [r for r in review_rows if r["researcher_review_status"] == STATUS_REJECT]
    assert len(reject_rows) == 5, f"expected 5 Reject rows, found {len(reject_rows)}"

    excluded_ids = {r["instance_id"] for r in _load_csv(EXCLUDED_MASKS_CSV)} if EXCLUDED_MASKS_CSV.is_file() else set()
    eligible_ids = {r["instance_id"] for r in _load_csv(ELIGIBLE_MASKS_CSV)} if ELIGIBLE_MASKS_CSV.is_file() else set()

    v3_plan_ids = {c.instance_id for c in PER_INSTANCE_PLAN}  # the 15 Needs-Correction ids only

    entries = []
    for row in reject_rows:
        instance_id = row["instance_id"]
        entries.append({
            "instance_id": instance_id,
            "image_id": row["image_id"],
            "defect_class": row["defect_class"],
            "recorded_rejection_reason_required_correction_field": row["required_correction"],
            "recorded_confidence_or_uncertainty_field": row["confidence_or_uncertainty"],
            "recorded_annotation_quality_notes_field": row["annotation_quality_notes"],
            "still_excluded_in_excluded_masks_csv": instance_id in excluded_ids,
            "absent_from_eligible_masks_csv": instance_id not in eligible_ids,
            "no_correction_attempt_made_for_this_instance": instance_id not in v3_plan_ids,
        })

    all_excluded = all(e["still_excluded_in_excluded_masks_csv"] for e in entries)
    all_absent_from_eligible = all(e["absent_from_eligible_masks_csv"] for e in entries)
    all_no_correction_attempted = all(e["no_correction_attempt_made_for_this_instance"] for e in entries)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "F2.2C-Finalize Task 7",
        "n_rejected": len(entries),
        "all_still_excluded": all_excluded,
        "all_absent_from_eligible_masks_csv": all_absent_from_eligible,
        "all_no_correction_attempted": all_no_correction_attempted,
        "policy_confirmation": (
            "Rejected means rejected: no replacement mask, no V3 correction attempt, and no "
            "re-annotation was generated for any of these 5 instances by this stage."
        ),
        "rejected_instances": entries,
    }

    FINALIZE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINALIZE_REPORTS_DIR / "f22c_finalize_task7_rejected_instances.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
