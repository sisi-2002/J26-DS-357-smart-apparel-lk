"""F2.5 Task 2 -- define analysis units (instance / image / source-family).

Builds three joined, row-level datasets from F2.4-B's metric CSVs + F2.3-B's
linkage data, over the 65 paired-success instances x 2 methods pool:

  - instance-level: one row per (gt_instance_id, method)
  - image-level:    one row per (image_id, method), mean-aggregated first
                     over that image's own paired instances
  - family-level:    one row per (source_family_id, method), mean-aggregated
                     over all paired instances belonging to that family
                     (across possibly multiple images in a singleton family,
                     though in this pool every source_family_id maps to
                     exactly one image_id -- checked explicitly below)

Also recomputes the actual number of distinct eligible source families
among the 65 paired instances (NOT assumed to be 60 or 65).
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import (
    F25_OUT, load_all, to_bool, to_float, write_csv, write_json,
)

METHODS = ["gradcam", "gradcampp"]
STABILITY_PERTURBATIONS = [
    "brightness_plus5pct", "brightness_minus5pct", "contrast_plus5pct",
    "contrast_minus5pct", "gaussian_noise", "pixel_shift",
]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def build_instance_level(d) -> list[dict]:
    paired = d["paired_explanations"]
    loc = d["localization"]
    faith = d["faithfulness"]
    stab1 = d["stability_v1"]
    stab2 = d["stability_v2"]

    loc_by = {(r["gt_instance_id"], r["method"]): r for r in loc}
    faith_by = {(r["gt_instance_id"], r["method"]): r for r in faith}

    stab1_by = {}
    for r in stab1:
        stab1_by.setdefault((r["gt_instance_id"], r["method"]), []).append(r)
    stab2_by = {}
    for r in stab2:
        stab2_by.setdefault((r["gt_instance_id"], r["method"]), []).append(r)

    rows = []
    for pr in paired:
        gid = pr["gt_instance_id"]
        for method in METHODS:
            key = (gid, method)
            lr = loc_by.get(key, {})
            fr = faith_by.get(key, {})
            s1_rows = stab1_by.get(key, [])
            s2_rows = stab2_by.get(key, [])

            s1_valid = [to_float(r["similarity"]) for r in s1_rows if r["status"] == "valid"]
            s1_invalid_reason_counts = {}
            for r in s1_rows:
                if r["status"] != "valid":
                    s1_invalid_reason_counts[r["status"]] = s1_invalid_reason_counts.get(r["status"], 0) + 1

            s2_valid = [to_float(r["similarity"]) for r in s2_rows if r["status"] == "valid"]
            s2_status_counts = {}
            for r in s2_rows:
                s2_status_counts[r["status"]] = s2_status_counts.get(r["status"], 0) + 1

            row = {
                "gt_instance_id": gid,
                "image_id": pr["image_id"],
                "source_family_id": pr["source_family_id"],
                "complexity_group": pr["complexity_group"],
                "defect_class": pr["gt_defect_class"],
                "target_layer": pr["target_layer"],
                "method": method,
                # localization
                "pointing_hit": lr.get("pointing_hit") if lr else None,
                "pointing_status": lr.get("pointing_status"),
                "energy_fraction": to_float(lr.get("energy_fraction")),
                "energy_status": lr.get("energy_status"),
                "box_area_fraction_of_image": to_float(lr.get("box_area_fraction_of_image")),
                # faithfulness
                "insertion_auc": to_float(fr.get("insertion_auc")),
                "insertion_status": fr.get("insertion_status"),
                "deletion_auc": to_float(fr.get("deletion_auc")),
                "deletion_status": fr.get("deletion_status"),
                "random_control_auc": to_float(fr.get("random_control_auc")),
                "random_control_status": fr.get("random_control_status"),
                # stability v1 (headline, per F2.4-B's decision rule)
                "stability_v1_mean_similarity": _mean(s1_valid),
                "stability_v1_n_valid": len(s1_valid),
                "stability_v1_n_total": len(s1_rows),
                "stability_v1_invalid_reasons": s1_invalid_reason_counts,
                # stability v2 (degenerate-aware sensitivity annotation)
                "stability_v2_mean_similarity": _mean(s2_valid),
                "stability_v2_n_valid": len(s2_valid),
                "stability_v2_n_total": len(s2_rows),
                "stability_v2_status_counts": s2_status_counts,
            }
            rows.append(row)
    return rows


def build_image_level(instance_rows: list[dict]) -> list[dict]:
    by_image_method: dict[tuple, list[dict]] = {}
    for r in instance_rows:
        by_image_method.setdefault((r["image_id"], r["method"]), []).append(r)

    rows = []
    for (image_id, method), rs in sorted(by_image_method.items()):
        rows.append({
            "image_id": image_id,
            "method": method,
            "complexity_group": rs[0]["complexity_group"],
            "source_family_id": rs[0]["source_family_id"],
            "n_paired_instances_in_image": len(rs),
            "pointing_hit_rate": _mean([1.0 if r["pointing_hit"] == "True" or r["pointing_hit"] is True else 0.0
                                        for r in rs if r["pointing_status"] == "valid"]),
            "energy_fraction_mean": _mean([r["energy_fraction"] for r in rs if r["energy_status"] == "valid"]),
            "insertion_auc_mean": _mean([r["insertion_auc"] for r in rs if r["insertion_status"] == "valid"]),
            "deletion_auc_mean": _mean([r["deletion_auc"] for r in rs if r["deletion_status"] == "valid"]),
            "stability_v1_mean_similarity": _mean([r["stability_v1_mean_similarity"] for r in rs]),
            "stability_v2_mean_similarity": _mean([r["stability_v2_mean_similarity"] for r in rs]),
        })
    return rows


def build_family_level(instance_rows: list[dict]) -> list[dict]:
    by_family_method: dict[tuple, list[dict]] = {}
    for r in instance_rows:
        by_family_method.setdefault((r["source_family_id"], r["method"]), []).append(r)

    rows = []
    for (fam, method), rs in sorted(by_family_method.items()):
        images_in_family = sorted({r["image_id"] for r in rs})
        rows.append({
            "source_family_id": fam,
            "method": method,
            "complexity_group": rs[0]["complexity_group"],
            "n_images_in_family": len(images_in_family),
            "n_paired_instances_in_family": len(rs),
            "pointing_hit_rate": _mean([1.0 if (r["pointing_hit"] == "True" or r["pointing_hit"] is True) else 0.0
                                        for r in rs if r["pointing_status"] == "valid"]),
            "energy_fraction_mean": _mean([r["energy_fraction"] for r in rs if r["energy_status"] == "valid"]),
            "insertion_auc_mean": _mean([r["insertion_auc"] for r in rs if r["insertion_status"] == "valid"]),
            "deletion_auc_mean": _mean([r["deletion_auc"] for r in rs if r["deletion_status"] == "valid"]),
            "stability_v1_mean_similarity": _mean([r["stability_v1_mean_similarity"] for r in rs]),
            "stability_v2_mean_similarity": _mean([r["stability_v2_mean_similarity"] for r in rs]),
        })
    return rows


def run():
    d = load_all()
    instance_rows = build_instance_level(d)
    image_rows = build_image_level(instance_rows)
    family_rows = build_family_level(instance_rows)

    paired = d["paired_explanations"]
    distinct_families = sorted({r["source_family_id"] for r in paired})
    distinct_images = sorted({r["image_id"] for r in paired})

    # verify each family maps to exactly one image in this pool (singleton assumption check)
    fam_to_images = {}
    for r in paired:
        fam_to_images.setdefault(r["source_family_id"], set()).add(r["image_id"])
    multi_image_families = {f: sorted(imgs) for f, imgs in fam_to_images.items() if len(imgs) > 1}

    instances_per_family = {}
    for r in paired:
        instances_per_family[r["source_family_id"]] = instances_per_family.get(r["source_family_id"], 0) + 1
    max_family = max(instances_per_family.items(), key=lambda kv: kv[1])

    by_group = {}
    for g in ["Simple", "Moderate", "High"]:
        fams_g = {r["source_family_id"] for r in paired if r["complexity_group"] == g}
        by_group[g] = len(fams_g)

    summary = {
        "n_paired_instances": len(paired),
        "n_distinct_images_in_paired_pool": len(distinct_images),
        "n_distinct_source_families_in_paired_pool": len(distinct_families),
        "n_distinct_source_families_by_group": by_group,
        "families_with_more_than_one_image": multi_image_families,
        "max_instances_contributed_by_a_single_family": {"family": max_family[0], "n_instances": max_family[1]},
        "instances_per_family_histogram": {
            str(k): sum(1 for v in instances_per_family.values() if v == k)
            for k in sorted(set(instances_per_family.values()))
        },
    }

    write_csv(F25_OUT / "manifests" / "instance_level_dataset.csv", instance_rows)
    write_csv(F25_OUT / "manifests" / "image_level_dataset.csv", image_rows)
    write_csv(F25_OUT / "manifests" / "family_level_dataset.csv", family_rows)
    write_json(F25_OUT / "manifests" / "task2_unit_definitions_summary.json", summary)
    return instance_rows, image_rows, family_rows, summary


if __name__ == "__main__":
    _, _, fam_rows, summary = run()
    print("n instance rows:", len(_))
    print(summary)
