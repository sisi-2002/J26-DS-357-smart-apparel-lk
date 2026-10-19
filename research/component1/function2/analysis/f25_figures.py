"""F2.5 figure generation -- the 6 requested figure types, real data only.

1. localization_by_group.svg
2. stability_by_group.svg
3. faithfulness_by_group.svg
4. runtime_by_group.svg (timing is bounded to the 9-instance subset -- only
   one "group" axis is meaningful: complexity_group of that bounded subset)
5. paired_method_differences.svg (family-first paired differences + CI)
6. matching_missingness_by_group.svg
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT, load_all
from research.component1.function2.analysis.f25_svg_helpers import grouped_bar_chart, COLORS

GROUPS = ["Simple", "Moderate", "High"]


def load_json(name):
    with open(F25_OUT / "tables" / name, encoding="utf-8") as fh:
        return json.load(fh)


def fig1_localization_by_group():
    strat = load_json("complexity_stratified_results.json")
    series = {"gradcam": [], "gradcampp": []}
    for g in GROUPS:
        bg = strat["by_group"][g]
        for m in ["gradcam", "gradcampp"]:
            v = bg[m]["pointing_hit_rate"]["rate"]
            series[m].append(v)
    svg = grouped_bar_chart(
        "Pointing-accuracy hit rate by complexity group (65-pair pool, descriptive)",
        GROUPS, series, {"gradcam": COLORS["gradcam"], "gradcampp": COLORS["gradcampp"]},
        y_label="Hit rate", y_max=1.0,
        note="Complete-case, paired-success instances only. Not a causal claim about complexity.")
    (F25_OUT / "figures" / "localization_by_group.svg").write_text(svg, encoding="utf-8")


def fig2_stability_by_group():
    gap = load_json("stability_gap_investigation.json")
    series = {"gradcam": [], "gradcampp": []}
    for g in GROUPS:
        bg = gap["by_group_v1"][g]
        series["gradcam"].append(bg["gradcam"]["mean_similarity"])
        series["gradcampp"].append(bg["gradcampp"]["mean_similarity"])
    svg = grouped_bar_chart(
        "Stability v1 mean similarity by complexity group",
        GROUPS, series, {"gradcam": COLORS["gradcam"], "gradcampp": COLORS["gradcampp"]},
        y_label="Mean Spearman similarity", y_max=1.0,
        note="Robustness to 6 small nuisance perturbations only -- not a faithfulness/localization measure.")
    (F25_OUT / "figures" / "stability_by_group.svg").write_text(svg, encoding="utf-8")


def fig3_faithfulness_by_group():
    strat = load_json("complexity_stratified_results.json")
    series_ins = {"gradcam": [], "gradcampp": []}
    series_del = {"gradcam": [], "gradcampp": []}
    for g in GROUPS:
        bg = strat["by_group"][g]
        for m in ["gradcam", "gradcampp"]:
            series_ins[m].append(bg[m]["insertion_auc_mean"]["mean"])
            series_del[m].append(bg[m]["deletion_auc_mean"]["mean"])
    svg_ins = grouped_bar_chart(
        "Insertion AUC by complexity group (higher = better)",
        GROUPS, series_ins, {"gradcam": COLORS["gradcam"], "gradcampp": COLORS["gradcampp"]},
        y_label="Insertion AUC", y_max=1.0)
    (F25_OUT / "figures" / "faithfulness_insertion_by_group.svg").write_text(svg_ins, encoding="utf-8")
    svg_del = grouped_bar_chart(
        "Deletion AUC by complexity group (LOWER = better)",
        GROUPS, series_del, {"gradcam": COLORS["gradcam"], "gradcampp": COLORS["gradcampp"]},
        y_label="Deletion AUC", y_max=1.0)
    (F25_OUT / "figures" / "faithfulness_deletion_by_group.svg").write_text(svg_del, encoding="utf-8")


def fig4_runtime_by_group():
    d = load_all()
    rows = d["timing_sample"]
    series = {"gradcam": [], "gradcampp": []}
    for g in GROUPS:
        for m in ["gradcam", "gradcampp"]:
            vals = [float(r["median_total_s"]) for r in rows if r["complexity_group"] == g and r["method"] == m]
            series[m].append(statistics.fmean(vals) if vals else None)
    svg = grouped_bar_chart(
        "Median per-explanation latency by complexity group (bounded 9-instance subset)",
        GROUPS, series, {"gradcam": COLORS["gradcam"], "gradcampp": COLORS["gradcampp"]},
        y_label="Seconds", note="N=9 bounded subset, 1-3 instances per group cell -- not generalized.")
    (F25_OUT / "figures" / "runtime_by_group.svg").write_text(svg, encoding="utf-8")


def fig5_paired_method_differences():
    mc = load_json("method_comparison_overall.json")
    metrics = [
        ("Energy fraction", mc["localization"]["paired_family_level_bootstrap"]["energy_fraction_diff"]),
        ("Pointing hit rate", mc["localization"]["paired_family_level_bootstrap"]["pointing_hit_rate_diff"]),
        ("Insertion AUC", mc["faithfulness"]["paired_family_level_bootstrap"]["insertion_auc_diff"]),
        ("Deletion AUC", mc["faithfulness"]["paired_family_level_bootstrap"]["deletion_auc_diff"]),
        ("Stability v1", mc["stability_v1"]["paired_family_level_bootstrap"]),
        ("Stability v2", mc["stability_v2"]["paired_family_level_bootstrap"]),
    ]
    labels = [m[0] for m in metrics]
    means = [m[1].get("point_estimate_family_first_mean") for m in metrics]
    series = {"Grad-CAM minus Grad-CAM++ (family-first mean diff)": means}
    svg = grouped_bar_chart(
        "Paired family-first differences (Grad-CAM minus Grad-CAM++)",
        labels, series, {"Grad-CAM minus Grad-CAM++ (family-first mean diff)": COLORS["neutral"]},
        y_label="Difference", y_max=max(abs(v) for v in means if v is not None) * 1.3,
        note="Positive = Grad-CAM higher. See uncertainty/ for bootstrap CIs; descriptive, not causal.")
    (F25_OUT / "figures" / "paired_method_differences.svg").write_text(svg, encoding="utf-8")


def fig6_matching_missingness_by_group():
    bias = load_json("matching_selection_bias.json")
    series = {"Matched rate": [], "Paired-success rate (of total)": []}
    for g in GROUPS:
        bg = bias["by_group"][g]
        series["Matched rate"].append(bg["match_rate_pct"] / 100.0)
        series["Paired-success rate (of total)"].append(bg["n_paired_success"] / bg["n_total"])
    svg = grouped_bar_chart(
        "Matching / missingness by complexity group (full 92-instance population)",
        GROUPS, series, {"Matched rate": COLORS["gradcam"], "Paired-success rate (of total)": COLORS["gradcampp"]},
        y_label="Rate", y_max=1.0,
        note="High's lower rate reflects differential, likely non-random missingness -- see matching_selection_bias.md.")
    (F25_OUT / "figures" / "matching_missingness_by_group.svg").write_text(svg, encoding="utf-8")


def run():
    fig1_localization_by_group()
    fig2_stability_by_group()
    fig3_faithfulness_by_group()
    fig4_runtime_by_group()
    fig5_paired_method_differences()
    fig6_matching_missingness_by_group()
    figs = sorted((F25_OUT / "figures").glob("*.svg"))
    return figs


if __name__ == "__main__":
    figs = run()
    for f in figs:
        print("Written:", f)
