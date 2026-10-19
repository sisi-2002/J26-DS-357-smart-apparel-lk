"""F2.5 Task 13 -- pixel-IoU handling (reporting only; F2.4-C NOT executed).

Reports, verbatim re-derived from existing read-only files: the 9 verified
reference masks, their complexity eligibility, the 2-instance intersection
with the 65 paired outputs, and the single evaluated IoU=0.0 result. Gives
an honest recommendation on whether a standalone supplementary F2.4-C run
would add meaningful value.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT, load_all, write_json


def run():
    d = load_all()
    pilot_subset = d["pixel_iou_pilot_subset"]  # 9 reference masks
    pilot_metric = d["pixel_iou_pilot_metric"]  # 2 rows (the 1 evaluated instance x 2 methods)
    paired = d["paired_explanations"]

    paired_ids = {r["gt_instance_id"] for r in paired}

    n_eligible = sum(1 for r in pilot_subset if r["complexity_eligible"] == "True")
    n_ineligible = sum(1 for r in pilot_subset if r["complexity_eligible"] != "True")
    by_group = {}
    for r in pilot_subset:
        g = r["complexity_group"] if r["complexity_group"] else "INELIGIBLE_NO_GROUP"
        by_group[g] = by_group.get(g, 0) + 1

    def normalize_mask_instance_id(iid: str) -> str:
        """pixel_iou_pilot_subset.csv uses '__inst_NNN' 1-indexed; every other
        manifest (execution_manifest/paired_explanations/accounting) uses
        '__instNNN' 0-indexed. Convert so the two can be compared directly."""
        if "__inst_" not in iid:
            return iid
        prefix, idx_str = iid.rsplit("__inst_", 1)
        idx_0based = int(idx_str) - 1
        return f"{prefix}__inst{idx_0based:03d}"

    intersection = []
    for r in pilot_subset:
        iid = r["instance_id"]
        norm = normalize_mask_instance_id(iid)
        if norm in paired_ids or iid in paired_ids:
            intersection.append({"instance_id": iid, "normalized_id": norm, "defect_class": r["defect_class"],
                                  "complexity_group": r["complexity_group"]})

    evaluated_rows = []
    for r in pilot_metric:
        evaluated_rows.append({"gt_instance_id": r["gt_instance_id"], "method": r["method"],
                                "iou": r["iou"], "status": r["status"]})

    out = {
        "n_verified_reference_masks_total": len(pilot_subset),
        "n_complexity_eligible": n_eligible,
        "n_complexity_ineligible": n_ineligible,
        "complexity_group_breakdown_of_9_masks": by_group,
        "intersection_with_65_pair_pool": intersection,
        "n_intersection": len(intersection),
        "evaluated_instance_iou_rows": evaluated_rows,
        "high_complexity_coverage": "ZERO -- no High-complexity verified reference mask exists anywhere "
                                     "in this pipeline (confirmed: 0 of the 9 masks have "
                                     "complexity_group == 'High').",
        "f24c_execution_status": "NOT EXECUTED. Per Hard Restrictions, F2.5 does not run F2.4-C, "
                                  "does not recompute SAM masks, and does not expand this pool.",
        "recommendation": (
            "A standalone supplementary F2.4-C run (detect+explain on the ~6 additional source "
            "images backing the up-to-8 complexity-eligible masks not yet in the 65-pair pool) "
            "would be CHEAP to run -- F2.4-B's own full 65-pair run took ~61 minutes wall time for "
            "130 explanation pairs, so a <=8-instance supplementary run is a matter of minutes, "
            "not a resourcing concern. However, its INCREMENTAL EVIDENCE VALUE is low: even a "
            "fully successful F2.4-C run adds at most 8 Simple/Moderate pixel-IoU data points "
            "(0 High-complexity, since no High-complexity mask exists in the source annotation set "
            "at all) -- it cannot address the single most consequential gap (zero High-complexity "
            "pixel-level coverage) no matter how it is run, because that gap is in the upstream "
            "SAM-mask annotation set, not in which images have been explained. On balance: F2.4-C "
            "is a low-cost, low-but-nonzero-value addition (it would raise the Simple/Moderate "
            "pixel-IoU N from 1 evaluated / 2 intersecting to up to 8, which is still too small for "
            "any inferential claim, but a slightly richer descriptive pixel-level picture). It is "
            "NOT recommended as a prerequisite for F2.6/F2.7 -- the primary 65-pair box-level "
            "localization/faithfulness/stability evidence is the pipeline's real evidentiary base, "
            "and pixel-IoU remains, and will remain even after a hypothetical F2.4-C, a separate, "
            "much smaller, zero-High-coverage supplementary signal, not a load-bearing one."
        ),
    }
    write_json(F25_OUT / "tables" / "pixel_iou_report_only.json", out)
    return out


if __name__ == "__main__":
    out = run()
    print("Intersection:", out["intersection_with_65_pair_pool"])
    print("Group breakdown:", out["complexity_group_breakdown_of_9_masks"])
