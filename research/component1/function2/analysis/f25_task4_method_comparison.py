"""F2.5 Task 4 -- overall Grad-CAM vs Grad-CAM++ comparison, every metric
family, both methods reported separately, paired differences on common
valid targets, explicit directionality, per-metric valid denominators.

No single combined "XAI quality score" is computed. No universal winner is
declared -- see f25_preregistered_analysis_plan.json's directionality table
for why insertion/deletion AUC point in opposite "better" directions and
stability measures something different again.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT, load_all, to_bool, to_float, write_csv, write_json
from research.component1.function2.analysis.f25_task2_datasets import build_instance_level, build_family_level, METHODS
from research.component1.function2.analysis.f25_bootstrap import bootstrap_family_mean, bootstrap_family_paired_difference

STAB_PERTS = ["brightness_plus5pct", "brightness_minus5pct", "contrast_plus5pct",
              "contrast_minus5pct", "gaussian_noise_sigma5", "pixel_shift_2px"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def family_values_for_metric(family_rows, method, field):
    return {r["source_family_id"]: r[field] for r in family_rows if r["method"] == method}


def family_paired_diff_for_metric(family_rows, field):
    gc = {r["source_family_id"]: r[field] for r in family_rows if r["method"] == "gradcam"}
    gcpp = {r["source_family_id"]: r[field] for r in family_rows if r["method"] == "gradcampp"}
    diffs = {}
    for fam in set(gc) & set(gcpp):
        a, b = gc[fam], gcpp[fam]
        if a is not None and b is not None:
            diffs[fam] = a - b
    return diffs


def localization_section(instance_rows, family_rows):
    out = {"metric_family": "localization"}
    for method in METHODS:
        rows = [r for r in instance_rows if r["method"] == method]
        valid_pt = [r for r in rows if r["pointing_status"] == "valid"]
        hits = [1.0 if (r["pointing_hit"] in ("True", True)) else 0.0 for r in valid_pt]
        valid_en = [r for r in rows if r["energy_status"] == "valid"]
        out[method] = {
            "pointing_hit_rate": {
                "hits": int(sum(hits)), "valid_n": len(valid_pt), "total_n": len(rows),
                "rate": (sum(hits) / len(valid_pt)) if valid_pt else None,
            },
            "energy_fraction_mean": {"mean": _mean([r["energy_fraction"] for r in valid_en]),
                                      "valid_n": len(valid_en), "total_n": len(rows)},
            "box_area_fraction_of_image_mean_baseline": {
                "mean": _mean([r["box_area_fraction_of_image"] for r in rows]),
                "note": "property of GT box/image, not of the method -- context baseline only",
            },
        }
        fam_vals_energy = family_values_for_metric(family_rows, method, "energy_fraction_mean")
        fam_vals_hit = family_values_for_metric(family_rows, method, "pointing_hit_rate")
        out[method]["family_level_bootstrap"] = {
            "energy_fraction": bootstrap_family_mean(fam_vals_energy),
            "pointing_hit_rate": bootstrap_family_mean(fam_vals_hit),
        }

    # paired differences on common valid targets (instance-level complete case)
    gc_by_id = {r["gt_instance_id"]: r for r in instance_rows if r["method"] == "gradcam"}
    gcpp_by_id = {r["gt_instance_id"]: r for r in instance_rows if r["method"] == "gradcampp"}
    common_energy = [gid for gid in gc_by_id if gid in gcpp_by_id
                     and gc_by_id[gid]["energy_status"] == "valid" and gcpp_by_id[gid]["energy_status"] == "valid"]
    energy_diffs = [gc_by_id[g]["energy_fraction"] - gcpp_by_id[g]["energy_fraction"] for g in common_energy]
    out["paired_instance_level"] = {
        "energy_fraction_diff_gradcam_minus_gradcampp": {
            "mean": _mean(energy_diffs), "n_common_valid": len(common_energy),
        },
    }
    fam_diff_energy = family_paired_diff_for_metric(family_rows, "energy_fraction_mean")
    fam_diff_hit = family_paired_diff_for_metric(family_rows, "pointing_hit_rate")
    out["paired_family_level_bootstrap"] = {
        "energy_fraction_diff": bootstrap_family_paired_difference(fam_diff_energy),
        "pointing_hit_rate_diff": bootstrap_family_paired_difference(fam_diff_hit),
    }
    out["directionality"] = {
        "pointing_hit_rate": "higher = better",
        "energy_fraction": "concentration measure, not inherently better/worse -- see plan",
    }
    return out


def faithfulness_section(instance_rows, family_rows):
    out = {"metric_family": "faithfulness"}
    for method in METHODS:
        rows = [r for r in instance_rows if r["method"] == method]
        valid_ins = [r for r in rows if r["insertion_status"] == "valid"]
        valid_del = [r for r in rows if r["deletion_status"] == "valid"]
        valid_rc = [r for r in rows if r["random_control_status"] == "valid"]
        out[method] = {
            "insertion_auc_mean": {"mean": _mean([r["insertion_auc"] for r in valid_ins]),
                                    "valid_n": len(valid_ins), "total_n": len(rows)},
            "deletion_auc_mean": {"mean": _mean([r["deletion_auc"] for r in valid_del]),
                                   "valid_n": len(valid_del), "total_n": len(rows)},
            "random_control_auc_mean": {"mean": _mean([r["random_control_auc"] for r in valid_rc]),
                                         "valid_n": len(valid_rc), "total_n": len(rows),
                                         "note": "shared-per-instance baseline curve, not a method score"},
        }
        out[method]["family_level_bootstrap"] = {
            "insertion_auc": bootstrap_family_mean(family_values_for_metric(family_rows, method, "insertion_auc_mean")),
            "deletion_auc": bootstrap_family_mean(family_values_for_metric(family_rows, method, "deletion_auc_mean")),
        }
    out["paired_family_level_bootstrap"] = {
        "insertion_auc_diff": bootstrap_family_paired_difference(family_paired_diff_for_metric(family_rows, "insertion_auc_mean")),
        "deletion_auc_diff": bootstrap_family_paired_difference(family_paired_diff_for_metric(family_rows, "deletion_auc_mean")),
    }
    out["directionality"] = {
        "insertion_auc": "higher = better (confidence recovers faster as important pixels are inserted)",
        "deletion_auc": "LOWER = better (confidence should drop faster as important pixels are deleted first)",
        "random_control_auc": "baseline reference only, not a competing method score",
    }
    return out


def stability_section(d, instance_rows, family_rows, version):
    key = f"stability_{version}"
    rows_raw = d[key]
    out = {"metric_family": f"stability_{version}"}
    for method in METHODS:
        m_rows = [r for r in rows_raw if r["method"] == method]
        valid = [r for r in m_rows if r["status"] == "valid"]
        status_counts = {}
        for r in m_rows:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        per_pert = {}
        for pert in STAB_PERTS:
            p_rows = [r for r in m_rows if r["perturbation"] == pert]
            p_valid = [r for r in p_rows if r["status"] == "valid"]
            per_pert[pert] = {
                "valid_n": len(p_valid), "total_n": len(p_rows),
                "mean_similarity": _mean([to_float(r["similarity"]) for r in p_valid]),
            }
        out[method] = {
            "mean_similarity_overall": {
                "mean": _mean([to_float(r["similarity"]) for r in valid]),
                "valid_n": len(valid), "total_n": len(m_rows),
            },
            "status_counts": status_counts,
            "per_perturbation": per_pert,
        }
        field = f"stability_{version}_mean_similarity"
        out[method]["family_level_bootstrap"] = bootstrap_family_mean(
            family_values_for_metric(family_rows, method, field))
    out["paired_family_level_bootstrap"] = bootstrap_family_paired_difference(
        family_paired_diff_for_metric(family_rows, f"stability_{version}_mean_similarity"))
    out["directionality"] = {"stability_similarity": "higher = better (more robust to small nuisance "
                                                      "perturbations only -- not a correctness/faithfulness measure)"}
    return out


def sanity_section(d):
    rows = d["sanity_classification_branch"]
    out = {"metric_family": "sanity_classification_branch", "bounded_subset_n_instances": 9,
           "caveat": "9-instance bounded subset (same as F2.4-B) -- not the full 65 pairs; not "
                     "generalized beyond this pool."}
    for method in METHODS:
        m_rows = [r for r in rows if r["method"] == method]
        valid = [r for r in m_rows if r["status"] == "valid"]
        score_deltas = [to_float(r["score_delta"]) for r in m_rows if r["score_delta"] not in (None, "")]
        sims = [to_float(r["similarity_vs_trained"]) for r in m_rows if r["similarity_vs_trained"] not in (None, "")]
        n_nan_similarity = sum(1 for r in m_rows if r["similarity_vs_trained"] in (None, ""))
        redetect_iou = [to_float(r["redetect_same_class_iou"]) for r in m_rows if r["redetect_same_class_iou"] not in (None, "")]
        redetect_survive = sum(1 for r in m_rows if r["redetect_survives_conf_threshold"] == "True")
        out[method] = {
            "n_rows": len(m_rows),
            "valid_rate": {"valid_n": len(valid), "total_n": len(m_rows)},
            "mean_score_delta": _mean(score_deltas),
            "mean_similarity_vs_trained_excluding_nan": _mean(sims),
            "n_similarity_nan_near_zero_rank_variance": n_nan_similarity,
            "mean_redetect_same_class_iou": _mean(redetect_iou),
            "redetect_survives_conf_threshold_rate": {"n_survive": redetect_survive, "total_n": len(m_rows)},
        }
    out["directionality"] = {
        "valid_rate": "higher = stronger evidence the classification-branch mechanism isolates the "
                      "intended confound (not evidence either method is more correct)",
    }
    return out


def timing_section(d):
    rows = d["timing_sample"]
    out = {"metric_family": "timing", "bounded_subset_n_instances": 9,
           "caveat": "9-instance bounded sample -- never conflated with production latency; "
                     "p95 figures are low-N estimates (5 timed reps per instance)."}
    for method in METHODS:
        m_rows = [r for r in rows if r["method"] == method]
        medians = sorted(to_float(r["median_total_s"]) for r in m_rows)
        p95s = sorted(to_float(r["p95_total_s"]) for r in m_rows)
        stage_sums = {}
        stage_counts = {}
        for r in m_rows:
            stages = json.loads(r["per_stage_medians_s"])
            for k, v in stages.items():
                stage_sums[k] = stage_sums.get(k, 0.0) + v
                stage_counts[k] = stage_counts.get(k, 0) + 1
        stage_means = {k: stage_sums[k] / stage_counts[k] for k in stage_sums}
        out[method] = {
            "n": len(m_rows),
            "median_of_medians_total_s": statistics.median(medians) if medians else None,
            "median_of_p95_total_s": statistics.median(p95s) if p95s else None,
            "per_stage_mean_s": stage_means,
        }
    return out


def run():
    d = load_all()
    instance_rows = build_instance_level(d)
    family_rows = build_family_level(instance_rows)

    sections = {
        "localization": localization_section(instance_rows, family_rows),
        "faithfulness": faithfulness_section(instance_rows, family_rows),
        "stability_v1": stability_section(d, instance_rows, family_rows, "v1"),
        "stability_v2": stability_section(d, instance_rows, family_rows, "v2"),
        "sanity": sanity_section(d),
        "timing": timing_section(d),
    }
    write_json(F25_OUT / "tables" / "method_comparison_overall.json", sections)

    # flat CSV summary for quick scanning
    flat_rows = []
    for fam, sec in sections.items():
        for method in METHODS:
            m = sec.get(method, {})
            flat_rows.append({"metric_family": fam, "method": method, "summary_json": json.dumps(m, default=str)})
    write_csv(F25_OUT / "tables" / "method_comparison_overall_flat.csv", flat_rows,
              fieldnames=["metric_family", "method", "summary_json"])
    return sections


if __name__ == "__main__":
    sections = run()
    print("Sections written:", list(sections.keys()))
