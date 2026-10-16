"""F2.1B2 Task 4 -- scale/mask sensitivity characterisation.

Purely a MEASUREMENT-STABILITY characterisation, run on a small,
representative, deterministic set of TRAIN-split images. Nothing here is
tuned against any downstream (Grad-CAM/XAI) outcome -- it only compares a
handful of reasonable, already-implemented processing-configuration
variants of the SAME feature-extraction code against each other, to
quantify how much the raw numbers (and resulting eligibility decisions)
move.

Two sensitivity axes investigated, both already present in existing code
(no new maths invented here):

  (A) Boundary-erosion mitigation on/off (``boundary_erosion_px`` = 1 vs 0)
      -- quantifies how much of the measured difference is attributable to
      the F2.1B1 resize-boundary-contamination mitigation itself.
  (B) ``min_valid_pairs`` threshold sensitivity (500 provisional default vs
      a stricter 1075 -- the Miller-Madow-motivated value documented in
      masked_glcm.py's own docstring -- vs a looser 200) -- quantifies how
      many samples flip between "valid"/"insufficient_pairs" purely from
      moving this provisional floor, without changing any pixel value at
      all.

Also documents (not "solves") the real-world GLCM pixel-distance confound:
the shared module's resize (aspect-preserving, long edge capped at 512,
INTER_AREA) means a fixed distance=1 step in RESIZED-pixel space covers a
different real-world spatial distance depending on the image's ORIGINAL
resolution. This script computes, for each sampled image, the implied
real-world distance-1-equivalent in ORIGINAL-resolution pixels
(``1 / resize_scale_factor``), so the magnitude of this confound is
reported with real numbers from real images, not asserted in the abstract.

Output:
  - results/calibration/scale_sensitivity_v1.csv
  - reports/task4_scale_sensitivity.md (written by hand alongside this
    script's numeric output; see that file for the narrative writeup)

Run:
    backend/.venv/Scripts/python.exe -m research.component1.function2.dataset.scale_sensitivity_v1
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.complexity_calibration.src.complexity_features import (  # noqa: E402
    DEFAULT_CONFIG,
)
from research.component1.function2.complexity.background_features import (  # noqa: E402
    compute_background_features,
)
from research.component1.function2.complexity.background_region import (  # noqa: E402
    DEFAULT_MAX_DEFECT_COVERAGE,
)
from research.component1.function2.complexity.masked_glcm import (  # noqa: E402
    DEFAULT_MIN_VALID_PAIRS,
)
from research.component1.function2.dataset.manifest_loader import (  # noqa: E402
    assert_train_split,
    build_split_lookup,
    deterministic_train_sample,
    read_yolo_label_file,
    resolve_train_image_path,
    resolve_train_label_path,
)

OUTPUT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "calibration"
CSV_PATH = OUTPUT_DIR / "scale_sensitivity_v1.csv"

# Deterministic set: the first 12 train-split images by image_id -- a
# strict subset of the F2.1A/F2.1B1 20-image smoke-test sample, re-used
# for continuity (not resampled) rather than freshly drawn.
N_IMAGES = 12
MIN_PAIRS_VARIANTS = [200, 500, 1075]  # loose, provisional-default, bias-motivated-strict


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def run(output_dir: Path | None = None) -> dict:
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    csv_path = out_dir / "scale_sensitivity_v1.csv"

    split_lookup = build_split_lookup()
    sample = deterministic_train_sample(N_IMAGES)

    rows = []
    for rec in sample:
        split_record = assert_train_split(rec.image_id, split_lookup)
        assert split_record.split == "train"

        image_path = resolve_train_image_path(rec.filename)
        label_path = resolve_train_label_path(rec.filename)
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        gt_labels = read_yolo_label_file(label_path)
        gt_boxes = None if gt_labels is None else [l[1:] for l in gt_labels]

        long_edge = max(h, w)
        resize_long_edge = DEFAULT_CONFIG.resize_long_edge
        if long_edge > resize_long_edge:
            scale_factor = resize_long_edge / long_edge
        else:
            scale_factor = 1.0
        real_world_distance1_equivalent_px = (1.0 / scale_factor) if scale_factor > 0 else None

        # --- Axis A: erosion on (1px, default) vs off (0px) ---
        res_erosion_on = compute_background_features(
            img_bgr, yolo_boxes=gt_boxes, is_negative=rec.is_negative,
            config=DEFAULT_CONFIG, max_defect_coverage=DEFAULT_MAX_DEFECT_COVERAGE,
            min_valid_pairs=DEFAULT_MIN_VALID_PAIRS, boundary_erosion_px=1,
        )
        res_erosion_off = compute_background_features(
            img_bgr, yolo_boxes=gt_boxes, is_negative=rec.is_negative,
            config=DEFAULT_CONFIG, max_defect_coverage=DEFAULT_MAX_DEFECT_COVERAGE,
            min_valid_pairs=DEFAULT_MIN_VALID_PAIRS, boundary_erosion_px=0,
        )

        def _delta(a, b):
            if a is None or b is None:
                return None
            return a - b

        row = {
            "image_id": rec.image_id,
            "defect_class_ids": list(rec.class_ids),
            "original_height": h, "original_width": w,
            "resize_scale_factor": round(scale_factor, 6),
            "real_world_distance1_equivalent_px": (
                round(real_world_distance1_equivalent_px, 4)
                if real_world_distance1_equivalent_px is not None else None
            ),
            "background_coverage_fraction": round(res_erosion_on.background_coverage_fraction, 4),
            # Axis A: erosion on vs off
            "lum_erosion_on": res_erosion_on.luminance_variation,
            "lum_erosion_off": res_erosion_off.luminance_variation,
            "lum_delta_erosion": _delta(res_erosion_on.luminance_variation, res_erosion_off.luminance_variation),
            "glcm_erosion_on": res_erosion_on.glcm_entropy,
            "glcm_erosion_off": res_erosion_off.glcm_entropy,
            "glcm_delta_erosion": _delta(res_erosion_on.glcm_entropy, res_erosion_off.glcm_entropy),
            "colour_erosion_on": res_erosion_on.colour_variation,
            "colour_erosion_off": res_erosion_off.colour_variation,
            "colour_delta_erosion": _delta(res_erosion_on.colour_variation, res_erosion_off.colour_variation),
            "glcm_valid_pairs_erosion_on": res_erosion_on.glcm_valid_pair_count,
            "glcm_valid_pairs_erosion_off": res_erosion_off.glcm_valid_pair_count,
        }

        # --- Axis B: min_valid_pairs threshold sensitivity (status flips) ---
        for mvp in MIN_PAIRS_VARIANTS:
            res = compute_background_features(
                img_bgr, yolo_boxes=gt_boxes, is_negative=rec.is_negative,
                config=DEFAULT_CONFIG, max_defect_coverage=DEFAULT_MAX_DEFECT_COVERAGE,
                min_valid_pairs=mvp, boundary_erosion_px=1,
            )
            row[f"glcm_status_min_pairs_{mvp}"] = res.glcm_status

        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            row_out["defect_class_ids"] = json.dumps(row_out["defect_class_ids"])
            writer.writerow(row_out)

    # Summaries used directly in the Task 4 report.
    lum_deltas = [r["lum_delta_erosion"] for r in rows if r["lum_delta_erosion"] is not None]
    glcm_deltas = [r["glcm_delta_erosion"] for r in rows if r["glcm_delta_erosion"] is not None]
    colour_deltas = [r["colour_delta_erosion"] for r in rows if r["colour_delta_erosion"] is not None]

    def _abs_mean(vals):
        return (sum(abs(v) for v in vals) / len(vals)) if vals else None

    status_flip_counts = {}
    for a, b in [(200, 500), (500, 1075), (200, 1075)]:
        flips = sum(
            1 for r in rows
            if r[f"glcm_status_min_pairs_{a}"] != r[f"glcm_status_min_pairs_{b}"]
        )
        status_flip_counts[f"{a}_vs_{b}"] = flips

    summary = {
        "n_images": len(rows),
        "erosion_sensitivity": {
            "n_comparable_luminance": len(lum_deltas),
            "mean_abs_delta_luminance": _abs_mean(lum_deltas),
            "max_abs_delta_luminance": max((abs(v) for v in lum_deltas), default=None),
            "n_comparable_glcm": len(glcm_deltas),
            "mean_abs_delta_glcm": _abs_mean(glcm_deltas),
            "max_abs_delta_glcm": max((abs(v) for v in glcm_deltas), default=None),
            "n_comparable_colour": len(colour_deltas),
            "mean_abs_delta_colour": _abs_mean(colour_deltas),
            "max_abs_delta_colour": max((abs(v) for v in colour_deltas), default=None),
        },
        "min_valid_pairs_status_flip_counts": status_flip_counts,
        "resize_scale_factor_range": {
            "min": min((r["resize_scale_factor"] for r in rows), default=None),
            "max": max((r["resize_scale_factor"] for r in rows), default=None),
        },
        "real_world_distance1_equivalent_px_range": {
            "min": min(
                (r["real_world_distance1_equivalent_px"] for r in rows
                 if r["real_world_distance1_equivalent_px"] is not None), default=None),
            "max": max(
                (r["real_world_distance1_equivalent_px"] for r in rows
                 if r["real_world_distance1_equivalent_px"] is not None), default=None),
        },
        "csv_output": _relative_or_absolute(csv_path),
    }
    return summary


if __name__ == "__main__":
    s = run()
    print(json.dumps(s, indent=2))
