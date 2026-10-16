"""Read-only loader for the authoritative Chenab dataset manifests.

Source of truth:
  data-preprocessing/master_fabric_dataset/processed/chenab_detection_training_v1/
    split_manifest.csv   (image_id, filename, split, group_id, is_negative)
    train_manifest.csv   (image_id, filename, group_id, label_count, class_ids, is_negative)
    train/images/*.jpg, train/labels/*.txt   (YOLO txt format)

This module never writes to the dataset directory and never opens files under
the locked test/ split. ``assert_train_split`` exists specifically so calling
code fails loudly instead of silently reading a non-train image.

"Source family" identification: the Chenab manifests do not use a column
literally named ``source_family``. The equivalent field is ``group_id``,
which groups images that originate from the same pre-augmentation source
photo (see data-preprocessing/master_fabric_dataset/reports/phase3/
split_policy_rules.md, rule R12_deterministic_assignment, and the freeze
manifest's ``split_group_counts`` / ``source_family_leakage_check`` keys,
which operate on this same group_id). We surface it as ``group_id`` rather
than inventing a ``source_family`` name that does not exist in the data.
"""

from __future__ import annotations

import ast
import csv
import dataclasses
from pathlib import Path
from typing import Optional

from research.component1.function2.config.paths import (
    CHENAB_SPLIT_MANIFEST,
    CHENAB_TRAIN_IMAGES_DIR,
    CHENAB_TRAIN_LABELS_DIR,
    CHENAB_TRAIN_MANIFEST,
)


@dataclasses.dataclass(frozen=True)
class SplitRecord:
    image_id: str
    filename: str
    split: str
    group_id: str
    is_negative: bool


@dataclasses.dataclass(frozen=True)
class TrainRecord:
    image_id: str
    filename: str
    group_id: str
    label_count: int
    class_ids: tuple[int, ...]
    is_negative: bool


class SplitMembershipError(RuntimeError):
    """Raised when code attempts to touch an image outside the train split."""


def load_split_manifest(path: Path = CHENAB_SPLIT_MANIFEST) -> list[SplitRecord]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(SplitRecord(
                image_id=row["image_id"],
                filename=row["filename"],
                split=row["split"],
                group_id=row["group_id"],
                is_negative=(row["is_negative"].strip().lower() == "true"),
            ))
    return records


def load_train_manifest(path: Path = CHENAB_TRAIN_MANIFEST) -> list[TrainRecord]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            class_ids = tuple(ast.literal_eval(row["class_ids"])) if row["class_ids"] else ()
            records.append(TrainRecord(
                image_id=row["image_id"],
                filename=row["filename"],
                group_id=row["group_id"],
                label_count=int(row["label_count"]),
                class_ids=class_ids,
                is_negative=(row["is_negative"].strip().lower() == "true"),
            ))
    return records


def build_split_lookup(path: Path = CHENAB_SPLIT_MANIFEST) -> dict[str, SplitRecord]:
    """image_id -> SplitRecord, for O(1) split-membership confirmation."""
    return {r.image_id: r for r in load_split_manifest(path)}


def assert_train_split(image_id: str, split_lookup: dict[str, SplitRecord]) -> SplitRecord:
    """Confirm (from the authoritative manifest, not folder naming) that
    image_id belongs to the train split. Raises SplitMembershipError otherwise.

    This is the single choke point that Task 5/7 code must call before any
    image is opened, so that locked validation/test images can never be
    touched by a bug that mixes up which list an image_id came from.
    """
    record = split_lookup.get(image_id)
    if record is None:
        raise SplitMembershipError(f"image_id {image_id!r} not found in split manifest")
    if record.split != "train":
        raise SplitMembershipError(
            f"image_id {image_id!r} is split={record.split!r}, not 'train'. Refusing to touch it."
        )
    return record


def resolve_train_image_path(filename: str) -> Path:
    return CHENAB_TRAIN_IMAGES_DIR / filename


def resolve_train_label_path(filename: str) -> Path:
    stem = Path(filename).stem
    return CHENAB_TRAIN_LABELS_DIR / f"{stem}.txt"


def read_yolo_label_file(label_path: Path) -> Optional[list[tuple[int, float, float, float, float]]]:
    """Parse a YOLO-format label .txt file.

    Returns:
      - [] (empty list) if the label file EXISTS and is empty
        -> confirmed negative image (no GT boxes), background = full image.
      - a non-empty list of (class_id, cx, cy, w, h) tuples (all normalised
        0-1) if the file exists and has content.
      - None if the label file does NOT exist
        -> annotation status is unresolved/ambiguous. Per the F2.1A design
        decision (Task 4), this must NEVER be treated as "clean background".
    """
    if not label_path.is_file():
        return None
    rows = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue  # malformed line; caller may flag via warnings
        cls_id = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        rows.append((cls_id, cx, cy, w, h))
    return rows


def deterministic_train_sample(n: int, seed_note: str = "sorted by image_id, no RNG") -> list[TrainRecord]:
    """Deterministically pick the first n TRAIN records, sorted by image_id.

    Sorting by image_id (rather than using an RNG) keeps the sample
    reproducible across machines/runs without needing to pin/document a
    random seed, and avoids any risk of accidentally drawing from the
    valid/test lists.
    """
    records = sorted(load_train_manifest(), key=lambda r: r.image_id)
    return records[:n]
