"""F2.3-B Task 2 — re-verify the frozen F2.2B 60-image PRIMARY sample.

This mirrors the pattern already used by ``manifest_verification.py`` for
the 9-image dry-run sample (F2.3-A), adapted to the primary sample's own
column schema (``xai_primary_sample.csv``: sample_type, image_id,
source_family_id, complexity_group, boundary_adjacent, defect_classes,
defect_instance_count, has_gt_defect, complexity_score,
selection_rank_within_group -- no explicit ``detection_xai_eligible``
column; it is re-derived exactly as F2.2B defines it: complexity_eligible
AND has_gt_defect).

Like ``manifest_verification.py``, this module NEVER trusts the sample
file's own claims -- every row is independently re-resolved against:
  - the authoritative ``split_manifest.csv`` (split + group_id),
  - the authoritative ``valid_complexity_manifest.csv`` (complexity_eligible
    AND has_gt_defect).

It resamples NOTHING and modifies nothing -- read-only verification only.
A hard TEST-split lockout is enforced identically to the dry-run verifier:
any row whose AUTHORITATIVE split is "test" (or anything other than
"valid") raises immediately, before any image/label file is ever opened.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerifiedPrimaryRow:
    image_id: str
    source_family_id: str
    complexity_group: str
    defect_classes: str
    defect_instance_count: int
    authoritative_split: str
    image_filename: str
    image_path: Path
    label_path: Path
    detection_xai_eligible: bool
    verification_notes: list[str]


class PrimarySampleVerificationError(Exception):
    """Raised when a primary-sample row fails cross-verification."""


def _load_csv_by_image_id(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["image_id"]: row for row in csv.DictReader(f)}


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def verify_primary_sample(
    primary_sample_csv: Path,
    split_manifest_csv: Path,
    valid_complexity_manifest_csv: Path,
    dataset_root: Path,
) -> list[VerifiedPrimaryRow]:
    """Cross-verify every row of the frozen 60-image primary sample.

    Raises ``PrimarySampleVerificationError`` on ANY mismatch (split != valid,
    source_family_id mismatch, not detection_xai_eligible, missing image or
    label file, duplicate image_id, duplicate source_family_id, group-count
    drift from 20/20/20) -- this stage must never silently proceed on
    unverifiable input, and must never touch TEST-split files.
    """
    with open(primary_sample_csv, newline="", encoding="utf-8") as f:
        sample_rows = list(csv.DictReader(f))

    seen_ids = set()
    for row in sample_rows:
        if row["image_id"] in seen_ids:
            raise PrimarySampleVerificationError(
                f"Duplicate image_id in primary sample: {row['image_id']}"
            )
        seen_ids.add(row["image_id"])

    split_by_id = _load_csv_by_image_id(split_manifest_csv)
    complexity_by_id = _load_csv_by_image_id(valid_complexity_manifest_csv)

    verified: list[VerifiedPrimaryRow] = []
    group_ids_seen = set()
    group_counts: dict[str, int] = {"Simple": 0, "Moderate": 0, "High": 0}

    for row in sample_rows:
        image_id = row["image_id"]
        notes: list[str] = []

        split_row = split_by_id.get(image_id)
        if split_row is None:
            raise PrimarySampleVerificationError(
                f"{image_id}: not found in authoritative split_manifest.csv"
            )

        authoritative_split = split_row["split"]
        if authoritative_split == "test":
            raise PrimarySampleVerificationError(
                f"{image_id}: authoritative split_manifest.csv marks this "
                f"TEST-split -- refusing to process (hard TEST-split "
                f"lockout). The image_id's own naming (e.g. 'chenab_test_...') "
                f"must never be trusted as a split indicator."
            )
        if authoritative_split != "valid":
            raise PrimarySampleVerificationError(
                f"{image_id}: authoritative split is '{authoritative_split}', "
                f"not 'valid' -- F2.2B sampling scope violated."
            )
        notes.append(
            f"authoritative_split='{authoritative_split}' confirmed via "
            f"split_manifest.csv (image_id prefix is not authoritative)"
        )

        csv_family = row["source_family_id"]
        manifest_group = split_row["group_id"]
        if csv_family != manifest_group and csv_family != f"singleton_{split_row['filename']}":
            raise PrimarySampleVerificationError(
                f"{image_id}: source_family_id mismatch -- sample csv says "
                f"'{csv_family}', authoritative split_manifest.csv group_id "
                f"is '{manifest_group}'"
            )
        if manifest_group in group_ids_seen and manifest_group != "" :
            raise PrimarySampleVerificationError(
                f"{image_id}: authoritative group_id '{manifest_group}' already "
                f"used by another row in this primary sample -- source-family "
                f"uniqueness violated."
            )
        group_ids_seen.add(manifest_group)
        notes.append("source_family_id/group_id cross-checked against split_manifest.csv")

        complexity_row = complexity_by_id.get(image_id)
        if complexity_row is None:
            raise PrimarySampleVerificationError(
                f"{image_id}: not found in valid_complexity_manifest.csv"
            )
        complexity_eligible = _to_bool(complexity_row["complexity_eligible"])
        has_gt_defect = _to_bool(complexity_row["has_gt_defect"])
        detection_xai_eligible = complexity_eligible and has_gt_defect
        if not detection_xai_eligible:
            raise PrimarySampleVerificationError(
                f"{image_id}: not detection_xai_eligible per "
                f"valid_complexity_manifest.csv (complexity_eligible="
                f"{complexity_eligible}, has_gt_defect={has_gt_defect})"
            )
        notes.append(
            "detection_xai_eligible=True re-derived as "
            "complexity_eligible AND has_gt_defect (matches f22b_sampling.py)"
        )

        if complexity_row["complexity_group"] != row["complexity_group"]:
            raise PrimarySampleVerificationError(
                f"{image_id}: complexity_group mismatch between sample csv "
                f"('{row['complexity_group']}') and valid_complexity_manifest.csv "
                f"('{complexity_row['complexity_group']}')"
            )
        if row["complexity_group"] not in group_counts:
            raise PrimarySampleVerificationError(
                f"{image_id}: unexpected complexity_group '{row['complexity_group']}'"
            )
        group_counts[row["complexity_group"]] += 1

        image_path = dataset_root / "valid" / "images" / split_row["filename"]
        label_filename = Path(split_row["filename"]).stem + ".txt"
        label_path = dataset_root / "valid" / "labels" / label_filename
        if not image_path.is_file():
            raise PrimarySampleVerificationError(f"{image_id}: image file missing at {image_path}")
        if not label_path.is_file():
            raise PrimarySampleVerificationError(f"{image_id}: label file missing at {label_path}")

        verified.append(VerifiedPrimaryRow(
            image_id=image_id,
            source_family_id=csv_family,
            complexity_group=row["complexity_group"],
            defect_classes=row["defect_classes"],
            defect_instance_count=int(row["defect_instance_count"]),
            authoritative_split=authoritative_split,
            image_filename=split_row["filename"],
            image_path=image_path,
            label_path=label_path,
            detection_xai_eligible=detection_xai_eligible,
            verification_notes=notes,
        ))

    if group_counts != {"Simple": 20, "Moderate": 20, "High": 20}:
        raise PrimarySampleVerificationError(
            f"Group distribution drift: expected 20/20/20, found {group_counts}"
        )

    if len(sample_rows) != 60:
        raise PrimarySampleVerificationError(
            f"Expected exactly 60 rows in primary sample, found {len(sample_rows)}"
        )

    return verified
