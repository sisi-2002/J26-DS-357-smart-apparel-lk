"""F2.1B2 Task 2 -- large (near-full) TRAIN-split background-feature
extraction for complexity calibration.

Extends the F2.1A/F2.1B1 smoke tests (n=20) to (by default) the FULL 1,900-
image TRAIN split, using the already-corrected F2.1B1 extraction adapter
(``complexity/background_features.py``, mask-aware GLCM) plus the new
F2.1B2 Task 1 eligibility policy (``complexity/eligibility.py``).

Split-safety (hard requirement): every sampled image_id is resolved through
``manifest_loader.assert_train_split`` against the authoritative
``split_manifest.csv`` BEFORE its pixels are ever opened with ``cv2.imread``.
This script never reads ``valid/`` or ``test/`` directory paths at all --
``config/paths.py`` records them only as "do not open" constants, and this
script imports nothing from those constants. The accompanying test
``tests/test_train_extraction_split_safety.py`` re-verifies this against
the real CSV this script produces (not a mock).

Output (new, versioned -- does NOT overwrite any F2.1A/F2.1B1 file):
  - results/calibration/train_features_v1_full.csv
  - results/calibration/train_features_v1_full_metadata.json

Per-family handling: no per-family cap is applied to this extraction run
itself -- every TRAIN image is read exactly once (the dataset's own
intrinsic representation), and the maximum ``group_id`` family size in the
TRAIN split is 12 images out of 1,900 (0.63%), confirmed by direct count
over ``train_manifest.csv`` before this script was written, so no single
family can dominate an image-level average here even without an explicit
cap. The downstream Task 3 distribution-analysis script additionally
reports FAMILY-level (one-row-per-``group_id``) statistics alongside
image-level statistics specifically so that any reader who wants a
family-deduplicated view has one, rather than only ever seeing the
(mildly family-weighted) image-level numbers. This choice -- full
image-level extraction plus a complementary family-level analysis view,
rather than a hard per-family sampling cap -- is the per-family handling
approach this task required to be documented and justified.

Run (from repo root, with a Python environment that has cv2/numpy, e.g.
backend/.venv):

    backend/.venv/Scripts/python.exe -m research.component1.function2.dataset.train_extraction_v1 [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.complexity_calibration.src.complexity_features import (  # noqa: E402
    DEFAULT_CONFIG,
)
from research.component1.function2.complexity.background_features import (  # noqa: E402
    DEFAULT_BOUNDARY_EROSION_PX,
    compute_background_features,
)
from research.component1.function2.complexity.background_region import (  # noqa: E402
    DEFAULT_MAX_DEFECT_COVERAGE,
)
from research.component1.function2.complexity.eligibility import (  # noqa: E402
    POLICY_VERSION as ELIGIBILITY_POLICY_VERSION,
    determine_eligibility,
)
from research.component1.function2.complexity.masked_glcm import (  # noqa: E402
    CODE_VERSION as MASKED_GLCM_CODE_VERSION,
    DEFAULT_MIN_VALID_PAIRS,
)
from research.component1.function2.config.paths import CLASS_NAMES  # noqa: E402
from research.component1.function2.dataset.manifest_loader import (  # noqa: E402
    assert_train_split,
    build_split_lookup,
    load_train_manifest,
    read_yolo_label_file,
    resolve_train_image_path,
    resolve_train_label_path,
)

OUTPUT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "calibration"
CSV_PATH = OUTPUT_DIR / "train_features_v1_full.csv"
JSON_PATH = OUTPUT_DIR / "train_features_v1_full_metadata.json"

CSV_FIELDS = [
    "image_id", "relative_path", "source_family_group_id", "family_size",
    "defect_class_ids", "defect_class_names", "is_negative_manifest",
    "sample_type",
    "image_height", "image_width",
    "background_status", "background_coverage_fraction", "num_boxes_excluded",
    "boundary_erosion_px",
    "luminance_variation", "glcm_entropy", "colour_variation",
    "glcm_valid_pair_count", "glcm_status",
    "overall_status", "quality_flags",
    "eligible", "rejection_reason", "failed_checks",
]


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _classify_sample_type(background_status: str, is_negative_manifest: bool) -> str:
    """Keep confirmed-negative images recorded separately from
    box-excluded defective images, per the task brief's explicit schema
    requirement (and surface the two remaining background_region.py
    outcomes under their own labels too, rather than lumping everything
    non-negative into one "defective" bucket)."""
    if background_status == "clean_negative":
        return "confirmed_negative"
    if background_status == "boxes_excluded":
        return "box_excluded_defective"
    if background_status == "insufficient_background":
        return "insufficient_background_defective"
    if background_status == "clean_no_boxes_provided":
        return "ambiguous_clean_no_boxes" if not is_negative_manifest else "confirmed_negative"
    if background_status == "unresolved_missing_annotation":
        return "unresolved_annotation"
    return "other_" + background_status


def run(limit: int | None = None, output_dir: Path | None = None) -> dict:
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    csv_path = out_dir / "train_features_v1_full.csv"
    json_path = out_dir / "train_features_v1_full_metadata.json"

    split_lookup = build_split_lookup()
    all_train = sorted(load_train_manifest(), key=lambda r: r.image_id)
    sample = all_train if limit is None else all_train[:limit]

    family_sizes: dict[str, int] = {}
    for rec in all_train:
        family_sizes[rec.group_id] = family_sizes.get(rec.group_id, 0) + 1

    rows: list[dict] = []
    status_counts: dict[str, int] = {}
    bg_status_counts: dict[str, int] = {}
    sample_type_counts: dict[str, int] = {}
    eligibility_counts: dict[str, int] = {"eligible": 0, "ineligible": 0}
    rejection_reason_counts: dict[str, int] = {}
    confirmed_splits_touched: set[str] = set()

    t_start = time.time()
    for rec in sample:
        # Hard safety gate -- resolved from the authoritative manifest
        # only, never inferred from filename/image_id substrings (see
        # F2.1A task1 report's 'chenab_test_'-in-train-image_id finding).
        split_record = assert_train_split(rec.image_id, split_lookup)
        assert split_record.split == "train"
        confirmed_splits_touched.add(split_record.split)

        image_path = resolve_train_image_path(rec.filename)
        label_path = resolve_train_label_path(rec.filename)
        rel_path = image_path.relative_to(REPO_ROOT).as_posix()

        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            row = {
                "image_id": rec.image_id, "relative_path": rel_path,
                "source_family_group_id": rec.group_id,
                "family_size": family_sizes[rec.group_id],
                "defect_class_ids": json.dumps(list(rec.class_ids)),
                "defect_class_names": json.dumps(
                    [CLASS_NAMES[c] for c in rec.class_ids if c < len(CLASS_NAMES)]
                ),
                "is_negative_manifest": rec.is_negative,
                "sample_type": "unreadable_image",
                "image_height": None, "image_width": None,
                "background_status": "n/a", "background_coverage_fraction": None,
                "num_boxes_excluded": None, "boundary_erosion_px": None,
                "luminance_variation": None, "glcm_entropy": None,
                "colour_variation": None,
                "glcm_valid_pair_count": None, "glcm_status": None,
                "overall_status": "error_unreadable_image",
                "quality_flags": json.dumps(["cv2.imread_returned_None"]),
                "eligible": False, "rejection_reason": "image_unreadable",
                "failed_checks": json.dumps(["image_unreadable"]),
            }
            rows.append(row)
            status_counts["error_unreadable_image"] = status_counts.get("error_unreadable_image", 0) + 1
            sample_type_counts["unreadable_image"] = sample_type_counts.get("unreadable_image", 0) + 1
            eligibility_counts["ineligible"] += 1
            rejection_reason_counts["image_unreadable"] = rejection_reason_counts.get("image_unreadable", 0) + 1
            continue

        gt_labels = read_yolo_label_file(label_path)
        gt_boxes = None if gt_labels is None else [l[1:] for l in gt_labels]
        is_negative = split_record.is_negative

        result = compute_background_features(
            img_bgr, yolo_boxes=gt_boxes, is_negative=is_negative,
            config=DEFAULT_CONFIG,
            max_defect_coverage=DEFAULT_MAX_DEFECT_COVERAGE,
            min_valid_pairs=DEFAULT_MIN_VALID_PAIRS,
            boundary_erosion_px=DEFAULT_BOUNDARY_EROSION_PX,
        )

        bg_status_counts[result.background_status] = bg_status_counts.get(result.background_status, 0) + 1
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

        sample_type = _classify_sample_type(result.background_status, is_negative)
        sample_type_counts[sample_type] = sample_type_counts.get(sample_type, 0) + 1

        elig = determine_eligibility(
            split_confirmed_train=True,  # guaranteed by assert_train_split above
            background_status=result.background_status,
            luminance_variation=result.luminance_variation,
            glcm_entropy=result.glcm_entropy,
            glcm_status=result.glcm_status,
            glcm_valid_pair_count=result.glcm_valid_pair_count,
            colour_variation=result.colour_variation,
            max_defect_coverage=DEFAULT_MAX_DEFECT_COVERAGE,
            min_valid_pairs=DEFAULT_MIN_VALID_PAIRS,
        )
        eligibility_counts["eligible" if elig.eligible else "ineligible"] += 1
        if not elig.eligible:
            rejection_reason_counts[elig.rejection_reason] = rejection_reason_counts.get(elig.rejection_reason, 0) + 1

        rows.append({
            "image_id": rec.image_id,
            "relative_path": rel_path,
            "source_family_group_id": rec.group_id,
            "family_size": family_sizes[rec.group_id],
            "defect_class_ids": json.dumps(list(rec.class_ids)),
            "defect_class_names": json.dumps(
                [CLASS_NAMES[c] for c in rec.class_ids if c < len(CLASS_NAMES)]
            ),
            "is_negative_manifest": rec.is_negative,
            "sample_type": sample_type,
            "image_height": result.image_height,
            "image_width": result.image_width,
            "background_status": result.background_status,
            "background_coverage_fraction": round(result.background_coverage_fraction, 6),
            "num_boxes_excluded": result.num_boxes_excluded,
            "boundary_erosion_px": result.boundary_erosion_px,
            "luminance_variation": result.luminance_variation,
            "glcm_entropy": result.glcm_entropy,
            "colour_variation": result.colour_variation,
            "glcm_valid_pair_count": result.glcm_valid_pair_count,
            "glcm_status": result.glcm_status,
            "overall_status": result.status,
            "quality_flags": json.dumps(result.quality_flags),
            "eligible": elig.eligible,
            "rejection_reason": elig.rejection_reason or "",
            "failed_checks": json.dumps(elig.failed_checks),
        })

    elapsed_sec = time.time() - t_start

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "F2.1B2",
        "task": "Task 2 -- larger training-side extraction",
        "requested_limit": limit,
        "train_split_total_images": len(all_train),
        "sample_size_produced": len(rows),
        "coverage_fraction_of_train_split": len(rows) / len(all_train),
        "sampling_method": (
            "Deterministic: ALL (or, if --limit was passed, the first N "
            "sorted by image_id ascending) TRAIN-split rows of "
            "train_manifest.csv. No RNG. No held-out split touched -- "
            "every image_id is re-resolved through assert_train_split() "
            "against the authoritative split_manifest.csv before any "
            "pixels are opened."
        ),
        "per_family_handling": (
            "No sampling cap applied -- every TRAIN image is included "
            "exactly once (full dataset representation). Max family "
            "(group_id) size in the TRAIN split is "
            f"{max(family_sizes.values())} images out of {len(all_train)} "
            f"total ({max(family_sizes.values()) / len(all_train):.2%}), "
            "confirmed by direct count before this run, so no single "
            "family can dominate an image-level statistic even without "
            "an explicit cap. Task 3's distribution-analysis script "
            "additionally reports family-level (one-row-per-group_id) "
            "statistics as a complementary, de-duplicated view."
        ),
        "split_touched": sorted(confirmed_splits_touched),
        "all_rows_confirmed_train_split": confirmed_splits_touched == {"train"},
        "elapsed_seconds": round(elapsed_sec, 2),
        "elapsed_seconds_per_image": round(elapsed_sec / len(rows), 4) if rows else None,
        "extraction_config": {
            "complexity_features_protocol": DEFAULT_CONFIG.to_dict(),
            "background_adapter": {
                "max_defect_coverage": DEFAULT_MAX_DEFECT_COVERAGE,
                "source": "research/component1/function2/complexity/background_region.py",
            },
            "masked_glcm": {
                "code_version": MASKED_GLCM_CODE_VERSION,
                "min_valid_pairs": DEFAULT_MIN_VALID_PAIRS,
            },
            "boundary_erosion_px": DEFAULT_BOUNDARY_EROSION_PX,
            "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
        },
        "overall_status_counts": status_counts,
        "background_status_counts": bg_status_counts,
        "sample_type_counts": sample_type_counts,
        "eligibility_counts": eligibility_counts,
        "rejection_reason_counts": rejection_reason_counts,
        "source_modules": {
            "complexity_features": "research/component1/complexity_calibration/src/complexity_features.py (unmodified, shared)",
            "background_region_adapter": "research/component1/function2/complexity/background_region.py (unmodified, F2.1A)",
            "masked_glcm": "research/component1/function2/complexity/masked_glcm.py (unmodified, F2.1B1)",
            "background_features_adapter": "research/component1/function2/complexity/background_features.py (unmodified, F2.1B1)",
            "eligibility_policy": "research/component1/function2/complexity/eligibility.py (new, F2.1B2)",
        },
        "csv_output": _relative_or_absolute(csv_path),
        "note": (
            "This is NOT a calibration-freeze artifact. Eligibility and raw "
            "feature values only; no Simple/Moderate/High label is computed "
            "or stored here."
        ),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of TRAIN images processed "
                              "(deterministic first-N by image_id). Omit for full coverage.")
    args = parser.parse_args()
    meta = run(limit=args.limit)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("extraction_config",)}, indent=2))


if __name__ == "__main__":
    main()
