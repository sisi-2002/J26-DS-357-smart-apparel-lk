"""F2.1B3 Task 6 -- improved (v2) QC expert-review package with
source-family diversity capping + blind-review split.

Audit of the F2.1B2 v1 package (30 samples, 10 per group, selected by
representative_central / boundary_adjacent / feature_disagreement):
independently recomputed family spread is 26 distinct families among 30
samples overall, but the High group specifically draws its 10 samples from
only 7 distinct families, with one family (``sg_0106``) contributing 3 of
10 (30%) -- all 3 are augmented variants of the same source photo,
deliberately selected because they demonstrate real feature-disagreement
within a family (a genuine, informative case, not a selection bug -- see
``reports/b2_task7_expert_review_package.md`` section 3). Judged here as
WEAK ENOUGH diversity in that one group to warrant a v2 package with an
explicit per-family cap, without discarding the informative
feature-disagreement examples entirely (the cap allows up to
``MAX_PER_FAMILY_PER_GROUP`` from any one family, not just 1, so a genuine
within-family disagreement case can still be shown once).

This script does NOT overwrite ``expert_review_package_v1.csv`` -- v1
remains exactly as F2.1B2 produced it. v2 is a new, additional, improved
package, still drawn only from the real eligible TRAIN-split data (no
synthetic/fabricated rows).

Blind-review split (new in v2, addressing the task brief's explicit
requirement that a reviewer must not see the algorithm's proposed group
while judging):
  - ``expert_review_package_v2_blind_for_reviewer.csv`` -- what the human
    reviewer actually sees: image path + genuinely blank judgment columns
    ONLY. No composite_score, no z-scores, no proposed_group, no
    selection_reason (all of these would let a motivated reviewer infer or
    reverse-engineer the algorithm's label). Row order is deterministically
    shuffled (fixed seed, documented) so group-clustering in the original
    selection order cannot leak information through adjacency.
  - ``expert_review_package_v2_reconciliation_key.csv`` -- sample_id ->
    proposed_group + all measurements + selection_reason, kept SEPARATE,
    to be joined back to the reviewer's blind judgments only AFTER review
    is complete.
  - ``expert_review_package_v2_full.csv`` -- the full, non-blind table
    (same schema as v1) for audit/traceability purposes only; this is NOT
    what should be handed to the reviewer.

Run:
    backend/.venv/Scripts/python.exe -m research.component1.function2.dataset.expert_review_package_v2
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CALIB_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "calibration"
SCORES_CSV = CALIB_DIR / "train_composite_scores_v1.csv"
FEATURES_CSV = CALIB_DIR / "train_features_v1_full.csv"

OUTPUT_FULL_CSV = CALIB_DIR / "expert_review_package_v2_full.csv"
OUTPUT_BLIND_CSV = CALIB_DIR / "expert_review_package_v2_blind_for_reviewer.csv"
OUTPUT_KEY_CSV = CALIB_DIR / "expert_review_package_v2_reconciliation_key.csv"

N_CENTRAL_PER_GROUP = 4
N_BOUNDARY_PER_GROUP = 3
N_DISAGREEMENT_PER_GROUP = 3
MAX_PER_FAMILY_PER_GROUP = 2  # the v2 diversity cap (v1 had no cap; High group hit 3/10 from one family)

SHUFFLE_SEED = 20261018  # fixed, documented -- blind-CSV row order only, never affects selection

FULL_FIELDS = [
    "sample_id", "relative_image_path", "source_family_group_id",
    "proposed_group", "selection_reason",
    "luminance_variation", "glcm_entropy", "colour_variation",
    "z_luminance_variation", "z_glcm_entropy", "z_colour_variation",
    "composite_score", "feature_disagreement_spread", "boundary_adjacent_flag",
    "independent_expert_judgment", "expert_comments", "disagreement_flag", "review_status",
]

BLIND_FIELDS = [
    "sample_id", "relative_image_path",
    "independent_expert_judgment", "confidence_or_uncertainty", "comments",
    "insufficient_visible_background", "anonymous_reviewer_id", "review_date",
]

KEY_FIELDS = [
    "sample_id", "relative_image_path", "source_family_group_id",
    "proposed_group", "selection_reason",
    "luminance_variation", "glcm_entropy", "colour_variation",
    "z_luminance_variation", "z_glcm_entropy", "z_colour_variation",
    "composite_score", "feature_disagreement_spread", "boundary_adjacent_flag",
]


def _to_float(v):
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    import math
    return f if math.isfinite(f) else None


def run(
    scores_csv: Path | None = None,
    features_csv: Path | None = None,
    output_full_csv: Path | None = None,
    output_blind_csv: Path | None = None,
    output_key_csv: Path | None = None,
    max_per_family_per_group: int = MAX_PER_FAMILY_PER_GROUP,
) -> dict:
    scores_path = Path(scores_csv) if scores_csv is not None else SCORES_CSV
    features_path = Path(features_csv) if features_csv is not None else FEATURES_CSV
    full_path = Path(output_full_csv) if output_full_csv is not None else OUTPUT_FULL_CSV
    blind_path = Path(output_blind_csv) if output_blind_csv is not None else OUTPUT_BLIND_CSV
    key_path = Path(output_key_csv) if output_key_csv is not None else OUTPUT_KEY_CSV

    with open(scores_path, newline="", encoding="utf-8") as f:
        score_rows = list(csv.DictReader(f))
    with open(features_path, newline="", encoding="utf-8") as f:
        rel_path_by_id = {r["image_id"]: r["relative_path"] for r in csv.DictReader(f)}

    by_group: dict[str, list[dict]] = {"Simple": [], "Moderate": [], "High": []}
    for r in score_rows:
        g = r["proposed_group"]
        if g in by_group:
            by_group[g].append(r)

    selected: list[tuple[dict, str]] = []
    seen_ids: set[str] = set()
    family_counts_in_group: dict[str, dict[str, int]] = {"Simple": {}, "Moderate": {}, "High": {}}

    def _family_cap_ok(group: str, family_id: str) -> bool:
        return family_counts_in_group[group].get(family_id, 0) < max_per_family_per_group

    def _record_selection(group: str, row: dict, reason: str) -> None:
        selected.append((row, reason))
        seen_ids.add(row["image_id"])
        fam = row["source_family_group_id"]
        family_counts_in_group[group][fam] = family_counts_in_group[group].get(fam, 0) + 1

    for group, rows in by_group.items():
        rows_sorted_by_score = sorted(rows, key=lambda r: _to_float(r["composite_score"]) or 0.0)
        scores = [_to_float(r["composite_score"]) for r in rows_sorted_by_score]
        median_score = scores[len(scores) // 2] if scores else 0.0

        # Central: closest to the group's median composite score, subject
        # to the per-family cap (skip a candidate whose family already hit
        # the cap WITHIN THIS GROUP, try the next-closest instead).
        central_sorted = sorted(rows, key=lambda r: abs((_to_float(r["composite_score"]) or 0.0) - median_score))
        n_added = 0
        for r in central_sorted:
            if r["image_id"] in seen_ids:
                continue
            if not _family_cap_ok(group, r["source_family_group_id"]):
                continue
            _record_selection(group, r, f"representative_central_{group.lower()}")
            n_added += 1
            if n_added >= N_CENTRAL_PER_GROUP:
                break

        # Boundary-adjacent, same cap.
        boundary_candidates = [r for r in rows if r["boundary_adjacent_flag"] == "True" and r["image_id"] not in seen_ids]
        n_added = 0
        for r in boundary_candidates:
            if not _family_cap_ok(group, r["source_family_group_id"]):
                continue
            _record_selection(group, r, f"boundary_adjacent_{group.lower()}")
            n_added += 1
            if n_added >= N_BOUNDARY_PER_GROUP:
                break

        # Feature-disagreement, same cap (this is the category that, in
        # v1, produced 3/10 High-group samples from one family -- the cap
        # now limits it to at most max_per_family_per_group total across
        # ALL three selection reasons combined for that family+group).
        disagreement_sorted = sorted(
            [r for r in rows if r["image_id"] not in seen_ids],
            key=lambda r: _to_float(r["feature_disagreement_spread"]) or 0.0,
            reverse=True,
        )
        n_added = 0
        for r in disagreement_sorted:
            if not _family_cap_ok(group, r["source_family_group_id"]):
                continue
            _record_selection(group, r, f"feature_disagreement_{group.lower()}")
            n_added += 1
            if n_added >= N_DISAGREEMENT_PER_GROUP:
                break

    # --- Write full (non-blind, audit-only) CSV ---
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FULL_FIELDS)
        writer.writeheader()
        for r, reason in selected:
            writer.writerow({
                "sample_id": r["image_id"],
                "relative_image_path": rel_path_by_id.get(r["image_id"], ""),
                "source_family_group_id": r["source_family_group_id"],
                "proposed_group": r["proposed_group"],
                "selection_reason": reason,
                "luminance_variation": r["luminance_variation"],
                "glcm_entropy": r["glcm_entropy"],
                "colour_variation": r["colour_variation"],
                "z_luminance_variation": r["z_luminance_variation"],
                "z_glcm_entropy": r["z_glcm_entropy"],
                "z_colour_variation": r["z_colour_variation"],
                "composite_score": r["composite_score"],
                "feature_disagreement_spread": r["feature_disagreement_spread"],
                "boundary_adjacent_flag": r["boundary_adjacent_flag"],
                "independent_expert_judgment": "",
                "expert_comments": "",
                "disagreement_flag": "",
                "review_status": "",
            })

    # --- Write reconciliation KEY CSV (same rows, same order as full) ---
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDS)
        writer.writeheader()
        for r, reason in selected:
            writer.writerow({
                "sample_id": r["image_id"],
                "relative_image_path": rel_path_by_id.get(r["image_id"], ""),
                "source_family_group_id": r["source_family_group_id"],
                "proposed_group": r["proposed_group"],
                "selection_reason": reason,
                "luminance_variation": r["luminance_variation"],
                "glcm_entropy": r["glcm_entropy"],
                "colour_variation": r["colour_variation"],
                "z_luminance_variation": r["z_luminance_variation"],
                "z_glcm_entropy": r["z_glcm_entropy"],
                "z_colour_variation": r["z_colour_variation"],
                "composite_score": r["composite_score"],
                "feature_disagreement_spread": r["feature_disagreement_spread"],
                "boundary_adjacent_flag": r["boundary_adjacent_flag"],
            })

    # --- Write BLIND reviewer-facing CSV: genuinely blank judgment columns
    # only, no label-revealing fields, deterministically shuffled order ---
    blind_rows = [
        {
            "sample_id": r["image_id"],
            "relative_image_path": rel_path_by_id.get(r["image_id"], ""),
            "independent_expert_judgment": "",
            "confidence_or_uncertainty": "",
            "comments": "",
            "insufficient_visible_background": "",
            "anonymous_reviewer_id": "",
            "review_date": "",
        }
        for r, _ in selected
    ]
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(blind_rows)

    blind_path.parent.mkdir(parents=True, exist_ok=True)
    with open(blind_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BLIND_FIELDS)
        writer.writeheader()
        for row in blind_rows:
            writer.writerow(row)

    # --- Diversity summary (v1 vs v2) ---
    def _family_diversity(rows_with_group: list[tuple[dict, str]]) -> dict:
        per_group_fam: dict[str, dict[str, int]] = {"Simple": {}, "Moderate": {}, "High": {}}
        for r, _ in rows_with_group:
            g = r["proposed_group"]
            fam = r["source_family_group_id"]
            per_group_fam[g][fam] = per_group_fam[g].get(fam, 0) + 1
        summary = {}
        all_fams = set()
        for g, counts in per_group_fam.items():
            all_fams.update(counts.keys())
            max_count = max(counts.values()) if counts else 0
            total = sum(counts.values())
            summary[g] = {
                "n_distinct_families": len(counts),
                "n_images": total,
                "max_single_family_count": max_count,
                "max_single_family_pct": (max_count / total * 100) if total else None,
            }
        summary["overall_n_distinct_families"] = len(all_fams)
        return summary

    diversity_v2 = _family_diversity(selected)

    result = {
        "n_selected_total": len(selected),
        "n_by_group": {g: sum(1 for r, _ in selected if r["proposed_group"] == g) for g in by_group},
        "max_per_family_per_group_cap_applied": max_per_family_per_group,
        "family_diversity_v2": diversity_v2,
        "full_csv_output": str(full_path.relative_to(REPO_ROOT).as_posix()) if full_path.is_relative_to(REPO_ROOT) else str(full_path),
        "blind_csv_output": str(blind_path.relative_to(REPO_ROOT).as_posix()) if blind_path.is_relative_to(REPO_ROOT) else str(blind_path),
        "key_csv_output": str(key_path.relative_to(REPO_ROOT).as_posix()) if key_path.is_relative_to(REPO_ROOT) else str(key_path),
        "blind_csv_shuffle_seed": SHUFFLE_SEED,
    }
    return result


if __name__ == "__main__":
    s = run()
    print(json.dumps(s, indent=2, default=str))
