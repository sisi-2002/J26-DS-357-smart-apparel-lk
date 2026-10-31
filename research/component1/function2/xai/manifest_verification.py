"""Task 2 — re-verify the frozen F2.2B 9-image dry-run sample.

Never trust ``xai_dry_run_sample.csv``'s own columns (including its
``image_id`` prefix, e.g. ``chenab_test_...`` does NOT mean TEST split --
it is a legacy source-naming artifact from before the re-split) -- every
row is independently re-resolved against:
  - the authoritative ``split_manifest.csv`` (split + source_family_id/group),
  - the F2.2B ``valid_complexity_manifest.csv`` (complexity_eligible AND
    has_gt_defect, i.e. ``detection_xai_eligible``, per
    ``f22b_sampling.detection_xai_eligible``).

This module resamples NOTHING -- it only verifies that the already-frozen
9-row sample is internally consistent with its authoritative sources, and
resolves each row to its real image/label file paths on disk.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VerifiedSampleRow:
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


class ManifestVerificationError(Exception):
    """Raised when a dry-run sample row fails cross-verification."""


def _load_csv_by_image_id(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["image_id"]: row for row in csv.DictReader(f)}


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def verify_dry_run_sample(
    dry_run_sample_csv: Path,
    split_manifest_csv: Path,
    valid_complexity_manifest_csv: Path,
    dataset_root: Path,
) -> list[VerifiedSampleRow]:
    """Cross-verify every row of the frozen dry-run sample.

    Raises ManifestVerificationError on ANY mismatch (split != valid,
    source_family_id mismatch, not detection_xai_eligible, missing image or
    label file) -- this stage must never silently proceed on unverifiable
    input, and must never touch TEST-split files.
    """
    with open(dry_run_sample_csv, newline="", encoding="utf-8") as f:
        sample_rows = list(csv.DictReader(f))

    split_by_id = _load_csv_by_image_id(split_manifest_csv)
    complexity_by_id = _load_csv_by_image_id(valid_complexity_manifest_csv)

    verified: list[VerifiedSampleRow] = []
    for row in sample_rows:
        image_id = row["image_id"]
        notes: list[str] = []

        split_row = split_by_id.get(image_id)
        if split_row is None:
            raise ManifestVerificationError(
                f"{image_id}: not found in authoritative split_manifest.csv"
            )

        authoritative_split = split_row["split"]
        if authoritative_split == "test":
            raise ManifestVerificationError(
                f"{image_id}: authoritative split_manifest.csv marks this "
                f"TEST-split -- refusing to process (hard TEST-split "
                f"lockout). The image_id's own naming ('chenab_test_...') "
                f"must never be trusted as a split indicator."
            )
        if authoritative_split != "valid":
            raise ManifestVerificationError(
                f"{image_id}: authoritative split is '{authoritative_split}', "
                f"not 'valid' -- F2.2B sampling scope violated."
            )
        notes.append(
            f"authoritative_split='{authoritative_split}' confirmed via "
            f"split_manifest.csv (image_id prefix is not authoritative)"
        )

        csv_family = row["source_family_id"]
        manifest_group = split_row["group_id"]
        # source_family_id in the sample csv is either the real group_id
        # (e.g. 'sg_0411') or a synthesized 'singleton_<filename>' label for
        # ungrouped images -- both are derived from (and must match) the
        # authoritative group_id.
        if csv_family != manifest_group and csv_family != f"singleton_{split_row['filename']}":
            raise ManifestVerificationError(
                f"{image_id}: source_family_id mismatch -- sample csv says "
                f"'{csv_family}', authoritative split_manifest.csv group_id "
                f"is '{manifest_group}'"
            )
        notes.append(f"source_family_id/group_id cross-checked against split_manifest.csv")

        complexity_row = complexity_by_id.get(image_id)
        if complexity_row is None:
            raise ManifestVerificationError(
                f"{image_id}: not found in valid_complexity_manifest.csv"
            )
        complexity_eligible = _to_bool(complexity_row["complexity_eligible"])
        has_gt_defect = _to_bool(complexity_row["has_gt_defect"])
        detection_xai_eligible = complexity_eligible and has_gt_defect
        if not detection_xai_eligible:
            raise ManifestVerificationError(
                f"{image_id}: not detection_xai_eligible per "
                f"valid_complexity_manifest.csv (complexity_eligible="
                f"{complexity_eligible}, has_gt_defect={has_gt_defect})"
            )
        notes.append(
            "detection_xai_eligible=True re-derived as "
            "complexity_eligible AND has_gt_defect (matches f22b_sampling.py)"
        )

        if complexity_row["complexity_group"] != row["complexity_group"]:
            raise ManifestVerificationError(
                f"{image_id}: complexity_group mismatch between sample csv "
                f"('{row['complexity_group']}') and valid_complexity_manifest.csv "
                f"('{complexity_row['complexity_group']}')"
            )

        image_path = dataset_root / "valid" / "images" / split_row["filename"]
        label_filename = Path(split_row["filename"]).stem + ".txt"
        label_path = dataset_root / "valid" / "labels" / label_filename
        if not image_path.is_file():
            raise ManifestVerificationError(f"{image_id}: image file missing at {image_path}")
        if not label_path.is_file():
            raise ManifestVerificationError(f"{image_id}: label file missing at {label_path}")

        verified.append(VerifiedSampleRow(
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

    return verified
