"""F2.1B2 Tasks 5 & 6 -- composite complexity score (candidate) and
provisional Simple/Moderate/High thresholds, computed ONLY from the Task 2
eligible training-side data.

This script:
  1. Loads ``results/calibration/train_features_v1_full.csv`` (Task 2).
  2. Restricts to rows the Task 1 eligibility policy already marked
     eligible (``eligible == True`` column, computed once, upstream, by
     ``train_extraction_v1.py`` -- this script does not re-derive
     eligibility).
  3. Computes z-score normalization parameters per feature from the
     eligible distribution ONLY (``complexity/scoring.py``).
  4. Builds a candidate composite score (equal 1/3 weighting default).
  5. Runs a weight-sensitivity check against 2 alternative plausible
     weightings (NOT tuned against anything XAI-related).
  6. Derives candidate tertile thresholds on the composite-score
     distribution, flags threshold-adjacent samples, and runs a bootstrap
     stability check.
  7. Writes:
       - config/calibration_v1.json           (frozen-FORMAT config file,
         content explicitly marked status="provisional_NOT_frozen")
       - results/calibration/train_composite_scores_v1.csv
       - results/calibration/calibration_scoring_summary_v1.json

Nothing here is tuned against, or even looks at, any Grad-CAM/XAI output --
this stage does not run or import detection_xai.py at all.

Run:
    backend/.venv/Scripts/python.exe -m research.component1.function2.dataset.calibration_scoring_v1
"""

from __future__ import annotations

import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.complexity.scoring import (  # noqa: E402
    FEATURE_NAMES,
    assign_group,
    compute_composite_score,
    compute_normalization_params,
    tertile_thresholds,
)

CALIB_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "calibration"
CONFIG_DIR = REPO_ROOT / "research" / "component1" / "function2" / "config"
INPUT_CSV = CALIB_DIR / "train_features_v1_full.csv"
CONFIG_PATH = CONFIG_DIR / "calibration_v1.json"
SCORES_CSV_PATH = CALIB_DIR / "train_composite_scores_v1.csv"
SUMMARY_JSON_PATH = CALIB_DIR / "calibration_scoring_summary_v1.json"

EQUAL_WEIGHTS = {"luminance_variation": 1 / 3, "glcm_entropy": 1 / 3, "colour_variation": 1 / 3}
ALT_WEIGHTS_A = {"luminance_variation": 0.6, "glcm_entropy": 0.2, "colour_variation": 0.2}  # emphasise luminance
ALT_WEIGHTS_B = {"luminance_variation": 0.2, "glcm_entropy": 0.6, "colour_variation": 0.2}  # emphasise GLCM texture

BOOTSTRAP_N_RESAMPLES = 500
BOOTSTRAP_SEED = 20261018  # fixed, documented seed for reproducibility

# Boundary-adjacent margin: derived from the BOOTSTRAP resampling noise of
# the tertile cut points themselves (see step 6 below), not an arbitrary
# fraction of the full score range. An earlier draft of this script used
# "5% of the observed composite-score range" and flagged 670/1747 (38%) of
# eligible samples as "boundary-adjacent" -- a useless flag, because the
# composite score's range is stretched by colour_variation's long right
# tail (see Task 3 report), so 5% of the full range is a much wider band
# than the actual density of points near the two cut points. Using
# BOUNDARY_MARGIN_BOOTSTRAP_STD_MULTIPLIER * (bootstrap std of each cut
# point) instead ties the flag to a statistically meaningful question:
# "would this sample plausibly land in a different group if the
# thresholds were recomputed on a slightly different draw of the eligible
# training data?" -- which is exactly what Task 6's stability check
# measures directly.
BOUNDARY_MARGIN_BOOTSTRAP_STD_MULTIPLIER = 2.0


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v):
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    import math
    return f if math.isfinite(f) else None


def _to_bool(v):
    return str(v).strip().lower() == "true"


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def run(input_csv: Path | None = None, output_dir: Path | None = None,
        config_dir: Path | None = None) -> dict:
    csv_path = Path(input_csv) if input_csv is not None else INPUT_CSV
    calib_dir = Path(output_dir) if output_dir is not None else CALIB_DIR
    cfg_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    calib_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(csv_path)
    eligible_rows = [r for r in rows if _to_bool(r["eligible"])]

    # --- Step 1: normalization parameters from eligible training data ---
    feature_values = {f: [] for f in FEATURE_NAMES}
    for r in eligible_rows:
        for f in FEATURE_NAMES:
            v = _to_float(r[f])
            if v is not None:
                feature_values[f].append(v)

    normalization_params = {f: compute_normalization_params(feature_values[f]) for f in FEATURE_NAMES}

    # --- Step 2: composite score per eligible row (equal weights, default candidate) ---
    scored_rows = []
    for r in eligible_rows:
        raw = {f: _to_float(r[f]) for f in FEATURE_NAMES}
        result = compute_composite_score(raw, normalization_params, weights=EQUAL_WEIGHTS)
        scored_rows.append((r, result))

    composite_scores = [res.composite_score for _, res in scored_rows if res.composite_score is not None]
    n_scored = len(composite_scores)

    # --- Step 3: candidate tertile thresholds ---
    simple_max, moderate_max = tertile_thresholds(composite_scores)

    # --- Step 3b: bootstrap stability check on the tertile cut points
    # (moved ahead of group assignment because the boundary-adjacent
    # margin below is DERIVED from this check's measured std, not an
    # arbitrary fraction of the score range -- see
    # BOUNDARY_MARGIN_BOOTSTRAP_STD_MULTIPLIER's docstring above) ---
    rng = random.Random(BOOTSTRAP_SEED)
    boot_simple_maxes = []
    boot_moderate_maxes = []
    n_boot = len(composite_scores)
    for _ in range(BOOTSTRAP_N_RESAMPLES):
        resample = [composite_scores[rng.randrange(n_boot)] for _ in range(n_boot)]
        s_max, m_max = tertile_thresholds(resample)
        boot_simple_maxes.append(s_max)
        boot_moderate_maxes.append(m_max)

    def _mean(vals):
        return sum(vals) / len(vals)

    def _std(vals):
        m = _mean(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    simple_max_boot_std = _std(boot_simple_maxes)
    moderate_max_boot_std = _std(boot_moderate_maxes)

    bootstrap_stability = {
        "method": (
            f"{BOOTSTRAP_N_RESAMPLES} bootstrap resamples (with replacement, "
            f"same size n={n_boot}, fixed seed={BOOTSTRAP_SEED}) of the eligible "
            "composite-score distribution; tertile thresholds recomputed on "
            "each resample."
        ),
        "simple_max_bootstrap": {
            "mean": _mean(boot_simple_maxes), "std": simple_max_boot_std,
            "min": min(boot_simple_maxes), "max": max(boot_simple_maxes),
            "point_estimate": simple_max,
        },
        "moderate_max_bootstrap": {
            "mean": _mean(boot_moderate_maxes), "std": moderate_max_boot_std,
            "min": min(boot_moderate_maxes), "max": max(boot_moderate_maxes),
            "point_estimate": moderate_max,
        },
    }

    # Boundary-adjacent margin, applied uniformly to both cut points: the
    # LARGER of the two cut points' bootstrap standard deviations, scaled
    # by BOUNDARY_MARGIN_BOOTSTRAP_STD_MULTIPLIER. Using the larger of the
    # two (rather than a per-cutpoint margin) keeps a single, simple,
    # symmetric definition while still being driven by measured resampling
    # noise rather than an arbitrary fraction of the (outlier-stretched)
    # score range.
    boundary_margin = BOUNDARY_MARGIN_BOOTSTRAP_STD_MULTIPLIER * max(simple_max_boot_std, moderate_max_boot_std)

    group_assignments = {}
    boundary_flags = {}
    for r, res in scored_rows:
        image_id = r["image_id"]
        group, is_boundary = assign_group(res.composite_score, simple_max, moderate_max, boundary_margin)
        group_assignments[image_id] = group
        boundary_flags[image_id] = is_boundary

    group_image_counts = {"Simple": 0, "Moderate": 0, "High": 0, "unassigned": 0}
    group_family_sets = {"Simple": set(), "Moderate": set(), "High": set()}
    for r, res in scored_rows:
        g = group_assignments[r["image_id"]]
        if g is None:
            group_image_counts["unassigned"] += 1
        else:
            group_image_counts[g] += 1
            group_family_sets[g].add(r["source_family_group_id"])
    group_family_counts = {g: len(s) for g, s in group_family_sets.items()}

    n_boundary_adjacent = sum(1 for v in boundary_flags.values() if v)

    # --- Step 4: feature-disagreement flag (for Task 7 sample selection) ---
    # A sample "disagrees" across features if its z-scores span a wide
    # range (e.g. one feature strongly above average while another is
    # strongly below) -- captured here as max(z) - min(z) across the 3
    # features, computed once so Task 7's review-package script can reuse
    # it without recomputing scoring from scratch.
    disagreement_spread = {}
    for r, res in scored_rows:
        zs = [v for v in res.z_scores.values() if v is not None]
        disagreement_spread[r["image_id"]] = (max(zs) - min(zs)) if len(zs) == 3 else None

    # --- Step 5: weight-sensitivity check (candidate alternative weightings) ---
    def _group_counts_for_weights(weights: dict) -> dict:
        scores = []
        assigns = {}
        for r, _ in scored_rows:
            raw = {f: _to_float(r[f]) for f in FEATURE_NAMES}
            res = compute_composite_score(raw, normalization_params, weights=weights)
            if res.composite_score is not None:
                scores.append(res.composite_score)
            assigns[r["image_id"]] = res.composite_score
        s_max, m_max = tertile_thresholds(scores)
        counts = {"Simple": 0, "Moderate": 0, "High": 0}
        groups = {}
        for image_id, score in assigns.items():
            g, _ = assign_group(score, s_max, m_max, boundary_margin=0.0)
            groups[image_id] = g
            if g is not None:
                counts[g] += 1
        return {"simple_max": s_max, "moderate_max": m_max, "counts": counts, "groups": groups}

    baseline_groups = {r["image_id"]: group_assignments[r["image_id"]] for r, _ in scored_rows}
    alt_a = _group_counts_for_weights(ALT_WEIGHTS_A)
    alt_b = _group_counts_for_weights(ALT_WEIGHTS_B)

    def _n_changed(alt_groups: dict) -> int:
        return sum(
            1 for image_id, g in baseline_groups.items()
            if g is not None and alt_groups.get(image_id) is not None and alt_groups[image_id] != g
        )

    weight_sensitivity = {
        "equal_weights_baseline": {
            "weights": EQUAL_WEIGHTS, "simple_max": simple_max, "moderate_max": moderate_max,
            "counts": group_image_counts,
        },
        "alt_weights_A_emphasise_luminance": {
            "weights": ALT_WEIGHTS_A, "simple_max": alt_a["simple_max"],
            "moderate_max": alt_a["moderate_max"], "counts": alt_a["counts"],
            "n_images_changed_group_vs_baseline": _n_changed(alt_a["groups"]),
            "fraction_images_changed_group_vs_baseline": (
                _n_changed(alt_a["groups"]) / len(baseline_groups) if baseline_groups else None
            ),
        },
        "alt_weights_B_emphasise_glcm_entropy": {
            "weights": ALT_WEIGHTS_B, "simple_max": alt_b["simple_max"],
            "moderate_max": alt_b["moderate_max"], "counts": alt_b["counts"],
            "n_images_changed_group_vs_baseline": _n_changed(alt_b["groups"]),
            "fraction_images_changed_group_vs_baseline": (
                _n_changed(alt_b["groups"]) / len(baseline_groups) if baseline_groups else None
            ),
        },
    }

    # --- Write calibration config (candidate, provisional) ---
    config = {
        "config_version": "calibration_v1_PROVISIONAL",
        "status": "provisional_NOT_frozen",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "F2.1B2 Tasks 5 & 6 (calibration_scoring_v1.py)",
        "source_data_csv": _relative_or_absolute(csv_path),
        "n_eligible_samples_used": len(eligible_rows),
        "n_samples_with_composite_score": n_scored,
        "normalization": {
            "method": "zscore (population mean/std, ddof=0, over the eligible TRAIN-split distribution only)",
            "features": {f: normalization_params[f].to_dict() for f in FEATURE_NAMES},
        },
        "weights": {
            "candidate": EQUAL_WEIGHTS,
            "candidate_label": "equal_weighting_default_candidate_NOT_validated",
        },
        "thresholds": {
            "method": "tertile_quantiles_of_eligible_training_composite_score_distribution",
            "simple_max": simple_max,
            "moderate_max": moderate_max,
            "label": "candidate_NOT_expert_validated",
            "boundary_adjacent_margin": boundary_margin,
            "boundary_adjacent_margin_method": (
                f"{BOUNDARY_MARGIN_BOOTSTRAP_STD_MULTIPLIER}x the larger of the two cut "
                "points' bootstrap-resampling standard deviations (see bootstrap_stability "
                "in the scoring summary JSON) -- i.e. samples whose composite score is "
                "close enough to a cut point that a modest resampling of the training "
                "data could plausibly move them to the other side of it."
            ),
        },
        "expert_review_status": "pending",
        "must_not_be_adjusted_using": "any Grad-CAM/Grad-CAM++/XAI-metric result",
    }
    config_path = cfg_dir / "calibration_v1.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # --- Write per-sample composite-score CSV ---
    score_fields = [
        "image_id", "source_family_group_id", "sample_type",
        "luminance_variation", "glcm_entropy", "colour_variation",
        "z_luminance_variation", "z_glcm_entropy", "z_colour_variation",
        "composite_score", "proposed_group", "boundary_adjacent_flag",
        "feature_disagreement_spread",
    ]
    scores_csv_path = calib_dir / "train_composite_scores_v1.csv"
    with open(scores_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=score_fields)
        writer.writeheader()
        for r, res in scored_rows:
            image_id = r["image_id"]
            writer.writerow({
                "image_id": image_id,
                "source_family_group_id": r["source_family_group_id"],
                "sample_type": r["sample_type"],
                "luminance_variation": res.raw_features["luminance_variation"],
                "glcm_entropy": res.raw_features["glcm_entropy"],
                "colour_variation": res.raw_features["colour_variation"],
                "z_luminance_variation": res.z_scores["luminance_variation"],
                "z_glcm_entropy": res.z_scores["glcm_entropy"],
                "z_colour_variation": res.z_scores["colour_variation"],
                "composite_score": res.composite_score,
                "proposed_group": group_assignments[image_id] or "",
                "boundary_adjacent_flag": boundary_flags[image_id],
                "feature_disagreement_spread": disagreement_spread[image_id],
            })

    summary = {
        "generated_at_utc": config["generated_at_utc"],
        "n_eligible_rows": len(eligible_rows),
        "n_samples_with_composite_score": n_scored,
        "normalization_params": {f: normalization_params[f].to_dict() for f in FEATURE_NAMES},
        "candidate_thresholds": {"simple_max": simple_max, "moderate_max": moderate_max},
        "boundary_adjacent_margin": boundary_margin,
        "n_boundary_adjacent_samples": n_boundary_adjacent,
        "group_image_counts": group_image_counts,
        "group_independent_family_counts": group_family_counts,
        "weight_sensitivity": weight_sensitivity,
        "bootstrap_stability": bootstrap_stability,
        "config_output": _relative_or_absolute(config_path),
        "scores_csv_output": _relative_or_absolute(scores_csv_path),
    }
    summary_json_path = calib_dir / "calibration_scoring_summary_v1.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    s = run()
    print(json.dumps(s, indent=2))
