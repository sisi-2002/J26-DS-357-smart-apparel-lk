"""F2.1B2 Task 7 -- QC expert-review package (no fabricated review).

Builds a representative-sample CSV for a human expert to review the
candidate Simple/Moderate/High groups proposed in
``results/calibration/train_composite_scores_v1.csv`` (Task 5/6 output).
For each proposed group, selects:
  - a few "central" samples (composite score closest to the group's median),
  - boundary-adjacent samples (flagged by Task 6's bootstrap-derived margin),
  - feature-disagreement samples (large spread across the 3 feature
    z-scores -- e.g. high texture but low luminance variation, or vice
    versa).

The output CSV's independent_expert_judgment / expert_comments /
disagreement_flag / review_status columns are left GENUINELY EMPTY --
nothing in this script writes any value into them. ``selection_reason`` is
a SEPARATE, non-empty column this script DOES fill in (why the row was
chosen for the package), so it is never confused with the expert's own
(blank) disagreement_flag column.

Output:
  - results/calibration/expert_review_package_v1.csv

Run:
    backend/.venv/Scripts/python.exe -m research.component1.function2.dataset.expert_review_package_v1
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CALIB_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "calibration"
SCORES_CSV = CALIB_DIR / "train_composite_scores_v1.csv"
FEATURES_CSV = CALIB_DIR / "train_features_v1_full.csv"
OUTPUT_CSV = CALIB_DIR / "expert_review_package_v1.csv"

N_CENTRAL_PER_GROUP = 4
N_BOUNDARY_PER_GROUP = 3
N_DISAGREEMENT_PER_GROUP = 3

OUTPUT_FIELDS = [
    "sample_id", "relative_image_path", "source_family_group_id",
    "proposed_group", "selection_reason",
    "luminance_variation", "glcm_entropy", "colour_variation",
    "z_luminance_variation", "z_glcm_entropy", "z_colour_variation",
    "composite_score", "feature_disagreement_spread", "boundary_adjacent_flag",
    # Genuinely empty columns -- for the human expert reviewer to fill in.
    "independent_expert_judgment", "expert_comments", "disagreement_flag", "review_status",
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


def run(scores_csv: Path | None = None, features_csv: Path | None = None,
        output_csv: Path | None = None) -> dict:
    scores_path = Path(scores_csv) if scores_csv is not None else SCORES_CSV
    features_path = Path(features_csv) if features_csv is not None else FEATURES_CSV
    out_path = Path(output_csv) if output_csv is not None else OUTPUT_CSV

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

    for group, rows in by_group.items():
        rows_sorted_by_score = sorted(rows, key=lambda r: _to_float(r["composite_score"]) or 0.0)
        scores = [_to_float(r["composite_score"]) for r in rows_sorted_by_score]
        median_score = scores[len(scores) // 2] if scores else 0.0

        # Central: closest to the group's median composite score.
        central_sorted = sorted(rows, key=lambda r: abs((_to_float(r["composite_score"]) or 0.0) - median_score))
        n_added = 0
        for r in central_sorted:
            if r["image_id"] in seen_ids:
                continue
            selected.append((r, f"representative_central_{group.lower()}"))
            seen_ids.add(r["image_id"])
            n_added += 1
            if n_added >= N_CENTRAL_PER_GROUP:
                break

        # Boundary-adjacent (per Task 6's bootstrap-derived margin flag).
        boundary_candidates = [r for r in rows if r["boundary_adjacent_flag"] == "True" and r["image_id"] not in seen_ids]
        for r in boundary_candidates[:N_BOUNDARY_PER_GROUP]:
            selected.append((r, f"boundary_adjacent_{group.lower()}"))
            seen_ids.add(r["image_id"])

        # Feature-disagreement: largest spread across the 3 z-scores.
        disagreement_sorted = sorted(
            [r for r in rows if r["image_id"] not in seen_ids],
            key=lambda r: _to_float(r["feature_disagreement_spread"]) or 0.0,
            reverse=True,
        )
        for r in disagreement_sorted[:N_DISAGREEMENT_PER_GROUP]:
            selected.append((r, f"feature_disagreement_{group.lower()}"))
            seen_ids.add(r["image_id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
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
                # Deliberately blank -- never pre-filled.
                "independent_expert_judgment": "",
                "expert_comments": "",
                "disagreement_flag": "",
                "review_status": "",
            })

    summary = {
        "n_selected_total": len(selected),
        "n_by_group": {g: sum(1 for r, _ in selected if r["proposed_group"] == g) for g in by_group},
        "n_by_selection_reason": {},
        "output_csv": str(out_path.relative_to(REPO_ROOT).as_posix()),
    }
    for _, reason in selected:
        summary["n_by_selection_reason"][reason] = summary["n_by_selection_reason"].get(reason, 0) + 1

    return summary


if __name__ == "__main__":
    s = run()
    print(json.dumps(s, indent=2))
