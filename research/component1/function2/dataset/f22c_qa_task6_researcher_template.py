"""F2.2C-QA Task 6 -- blank researcher-verification template.

Creates ``quality_review/researcher_review_v2.csv`` with ALL 29 reviewed
instances (including the 8 AI-preliminary-Accept ones) and genuinely blank
human-decision fields. This module NEVER writes a judgment, status, or
"approved"/"verified" value into any row -- every decision field is written
as an empty string, full stop. It also never copies the AI review's
``ai_review_decision`` into any of the human-facing columns (that would
blur the AI-review / human-review boundary the task brief requires to stay
strict).

AI-Accept is explicitly NOT a free pass: per the task brief, every one of
the 8 AI-Accept instances still needs independent human inspection before
being eligible for pixel-level IoU use (see Task 7's eligibility
procedure, which structurally requires researcher_review_status=="Accept"
regardless of what the AI preliminary screening said).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PILOT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "reference_masks" / "pilot_v1"
MANIFESTS_DIR = PILOT_DIR / "manifests"
QUALITY_REVIEW_DIR = PILOT_DIR / "quality_review"
AI_REVIEW_CSV = QUALITY_REVIEW_DIR / "ai_preliminary_20261018" / "SAM_mask_AI_preliminary_review_29_instances.csv"

FIELDNAMES = [
    "instance_id",
    "image_id",
    "defect_class",
    "ai_preliminary_status_reference_only",  # context column, NOT a human field
    "chosen_proposal_version",       # blank: V1 / V2 / neither
    "researcher_review_status",      # blank: Accept / Needs Correction / Reject / Cannot Assess
    "required_correction",           # blank
    "reviewer_identifier",           # blank
    "review_date",                   # blank
    "confidence_or_uncertainty",     # blank
    "annotation_quality_notes",      # blank
]

# These are the only columns genuinely required to be blank by the task
# brief. ai_preliminary_status_reference_only and the identifying columns
# are context, not a human decision, and are explicitly allowed to be
# pre-filled (they are read-only reference values copied verbatim from the
# AI review CSV / pilot manifest, never a decision).
BLANK_FIELDS = [
    "chosen_proposal_version",
    "researcher_review_status",
    "required_correction",
    "reviewer_identifier",
    "review_date",
    "confidence_or_uncertainty",
    "annotation_quality_notes",
]


def run() -> Path:
    with open(AI_REVIEW_CSV, newline="", encoding="utf-8") as f:
        ai_rows = list(csv.DictReader(f))
    assert len(ai_rows) == 29, f"expected 29 AI-reviewed instances, got {len(ai_rows)}"

    out_rows = []
    for r in ai_rows:
        row = {
            "instance_id": r["instance_id"],
            "image_id": r["image_id"],
            "defect_class": r["defect_class"],
            "ai_preliminary_status_reference_only": r["ai_review_decision"],
        }
        for field_name in BLANK_FIELDS:
            row[field_name] = ""
        out_rows.append(row)

    assert len(out_rows) == 29

    out_path = QUALITY_REVIEW_DIR / "researcher_review_v2.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    return out_path


if __name__ == "__main__":
    path = run()
    print(f"Wrote {path.relative_to(REPO_ROOT).as_posix()}")
