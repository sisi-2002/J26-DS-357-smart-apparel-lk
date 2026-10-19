"""F2.5 Task 1 -- mandatory data/integrity re-audit.

Independently recomputes, from row-level CSVs only (never trusting prior
prose summaries): instance/image/match counts, duplicate-ID checks,
method-pairing / target-layer-identity consistency, zero-fabrication checks
for unmatched/failed instances, zero TEST-split access in any new F2.5
module, and a scoped protected-checksum re-verification covering the files
this stage actually reads from.

This script WRITES ONLY under results/complexity_xai_analysis/f25_pp1_v1/.
It never modifies any F2.1-F2.4 output, config, or the E02 checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import (
    F25_OUT, E02_CHECKPOINT, F24_FULL, F24_PILOT, F23_FULL, CONFIG,
    load_all, to_bool, sha256_file, write_json,
)


def run() -> dict:
    d = load_all()
    acc = d["accounting"]
    exec_manifest = d["execution_manifest"]
    paired = d["paired_explanations"]
    linkage = d["target_linkage"]
    failures = d["failures"]
    loc = d["localization"]
    faith = d["faithfulness"]
    stab1 = d["stability_v1"]
    stab2 = d["stability_v2"]
    elig = d["metric_eligibility"]

    report = {"stage": "F2.5 Task 1", "blockers": [], "checks": {}}

    def check(name, cond, detail=None):
        report["checks"][name] = {"pass": bool(cond), "detail": detail}
        if not cond:
            report["blockers"].append({"check": name, "detail": detail})

    # --- 1. basic counts, independently recomputed from accounting.csv ---
    n_rows = len(acc)
    check("accounting_row_count_92", n_rows == 92, n_rows)

    images = sorted({r["image_id"] for r in acc})
    check("n_unique_images_60", len(images) == 60, len(images))

    gt_ids = [r["gt_instance_id"] for r in acc]
    check("n_gt_instance_ids_92", len(gt_ids) == 92, len(gt_ids))
    check("gt_instance_ids_unique_no_duplicates", len(set(gt_ids)) == len(gt_ids),
          {"n": len(gt_ids), "n_unique": len(set(gt_ids))})

    matched = [r for r in acc if to_bool(r["detection_matched"])]
    check("n_matched_66", len(matched) == 66, len(matched))

    paired_success = [r for r in acc if to_bool(r["paired_explanation_success"])]
    check("n_paired_success_65", len(paired_success) == 65, len(paired_success))

    degenerate_rows = [r for r in acc if to_bool(r["degenerate_one_sided_failure"])]
    check("n_degenerate_one_sided_1", len(degenerate_rows) == 1, len(degenerate_rows))
    degenerate_id = degenerate_rows[0]["gt_instance_id"] if degenerate_rows else None

    # cross-check: the degenerate row must itself be detection_matched=True,
    # paired_explanation_success=False, gradcam_status=degenerate,
    # gradcampp_status=success (one-sided failure, preserved not dropped)
    if degenerate_rows:
        dr = degenerate_rows[0]
        check("degenerate_row_is_detection_matched", to_bool(dr["detection_matched"]) is True, dr)
        check("degenerate_row_not_paired_success", to_bool(dr["paired_explanation_success"]) is False, dr)
        check("degenerate_row_gradcam_degenerate", dr["gradcam_status"] == "degenerate", dr["gradcam_status"])
        check("degenerate_row_gradcampp_success", dr["gradcampp_status"] == "success", dr["gradcampp_status"])

    # accounting: matched - paired_success - degenerate == unmatched remainder
    unmatched = [r for r in acc if not to_bool(r["detection_matched"])]
    check("n_unmatched_26", len(unmatched) == 26, len(unmatched))
    check("matched_eq_paired_plus_degenerate",
          len(matched) == len(paired_success) + len(degenerate_rows),
          {"matched": len(matched), "paired": len(paired_success), "degenerate": len(degenerate_rows)})
    check("total_eq_matched_plus_unmatched", len(matched) + len(unmatched) == 92,
          {"matched": len(matched), "unmatched": len(unmatched)})

    # --- 2. per-group matched rates, recomputed ---
    groups = ["Simple", "Moderate", "High"]
    per_group = {}
    for g in groups:
        g_rows = [r for r in acc if r["complexity_group"] == g]
        g_matched = [r for r in g_rows if to_bool(r["detection_matched"])]
        per_group[g] = {
            "total": len(g_rows),
            "matched": len(g_matched),
            "rate_pct": round(100.0 * len(g_matched) / len(g_rows), 1) if g_rows else None,
        }
    expected = {"Simple": (20, 25, 80.0), "Moderate": (27, 34, 79.4), "High": (19, 33, 57.6)}
    for g, (exp_m, exp_t, exp_r) in expected.items():
        pg = per_group[g]
        check(f"group_{g}_matched_{exp_m}_of_{exp_t}",
              pg["matched"] == exp_m and pg["total"] == exp_t, pg)
        check(f"group_{g}_rate_matches_{exp_r}", abs(pg["rate_pct"] - exp_r) < 0.05, pg["rate_pct"])

    # --- 3. cross-manifest consistency: execution_manifest / target_linkage ---
    check("execution_manifest_92_rows", len(exec_manifest) == 92, len(exec_manifest))
    check("target_linkage_92_rows", len(linkage) == 92, len(linkage))
    em_matched = sum(1 for r in exec_manifest if r["prediction_status"] == "matched")
    check("execution_manifest_matched_66", em_matched == 66, em_matched)
    tl_matched = sum(1 for r in linkage if r["match_status"] == "matched")
    check("target_linkage_matched_66", tl_matched == 66, tl_matched)

    acc_by_id = {r["gt_instance_id"]: r for r in acc}
    em_by_id = {r["gt_instance_id"]: r for r in exec_manifest}
    tl_by_id = {}
    for r in linkage:
        gid = f"{r['image_id']}__inst{int(r['instance_index']):03d}"
        tl_by_id[gid] = r

    check("accounting_ids_subset_of_execution_manifest_ids",
          set(acc_by_id) == set(em_by_id),
          {"only_in_acc": sorted(set(acc_by_id) - set(em_by_id))[:5],
           "only_in_em": sorted(set(em_by_id) - set(acc_by_id))[:5]})
    check("accounting_ids_match_target_linkage_constructed_ids",
          set(acc_by_id) == set(tl_by_id),
          {"only_in_acc": sorted(set(acc_by_id) - set(tl_by_id))[:5],
           "only_in_tl": sorted(set(tl_by_id) - set(acc_by_id))[:5]})

    # matched status agreement across the three sources
    mismatches = []
    for gid, ar in acc_by_id.items():
        er = em_by_id.get(gid)
        tr = tl_by_id.get(gid)
        a_m = to_bool(ar["detection_matched"])
        e_m = (er["prediction_status"] == "matched") if er else None
        t_m = (tr["match_status"] == "matched") if tr else None
        if a_m != e_m or a_m != t_m:
            mismatches.append({"gt_instance_id": gid, "acc": a_m, "exec": e_m, "linkage": t_m})
    check("matched_flag_consistent_across_accounting_execution_linkage", len(mismatches) == 0,
          mismatches[:10])

    # --- 4. paired_explanations_manifest consistency with accounting ---
    check("paired_explanations_65_rows", len(paired) == 65, len(paired))
    pe_ids = {r["gt_instance_id"] for r in paired}
    acc_paired_ids = {r["gt_instance_id"] for r in paired_success}
    check("paired_explanations_ids_match_accounting_paired_success_ids",
          pe_ids == acc_paired_ids,
          {"only_in_pe": sorted(pe_ids - acc_paired_ids)[:5],
           "only_in_acc": sorted(acc_paired_ids - pe_ids)[:5]})
    check("degenerate_id_excluded_from_paired_explanations",
          degenerate_id not in pe_ids, degenerate_id)

    # --- 5. method-pairing / target-layer-identity consistency ---
    # For every paired instance, target_layer recorded in paired_explanations_manifest
    # must match target_linkage.csv's target_layer for the SAME instance, and
    # target_linkage's own same_target_for_both_methods flag must be True.
    layer_mismatches = []
    same_target_false = []
    pe_by_id = {r["gt_instance_id"]: r for r in paired}
    for gid, pr in pe_by_id.items():
        tr = tl_by_id.get(gid)
        if tr is None:
            layer_mismatches.append({"gt_instance_id": gid, "reason": "missing_in_target_linkage"})
            continue
        if pr["target_layer"] != tr["target_layer"]:
            layer_mismatches.append({"gt_instance_id": gid, "pe_layer": pr["target_layer"], "tl_layer": tr["target_layer"]})
        if to_bool(tr["same_target_for_both_methods"]) is not True:
            same_target_false.append(gid)
    check("target_layer_identical_paired_vs_linkage_for_all_65", len(layer_mismatches) == 0, layer_mismatches[:10])
    check("same_target_for_both_methods_true_for_all_65", len(same_target_false) == 0, same_target_false[:10])

    # target_layer consistency across localization/faithfulness/stability metric rows
    def layer_map(rows, id_key="gt_instance_id", layer_key="target_layer", method_key="method"):
        out = {}
        for r in rows:
            out.setdefault(r[id_key], {})[r[method_key]] = r[layer_key]
        return out

    for name, rows in [("localization", loc), ("faithfulness", faith), ("stability_v1", stab1)]:
        lm = layer_map(rows)
        bad = []
        for gid, methods in lm.items():
            layers = set(methods.values())
            if len(layers) != 1:
                bad.append({"gt_instance_id": gid, "layers_by_method": methods})
            pe = pe_by_id.get(gid)
            if pe is not None and layers and next(iter(layers)) != pe["target_layer"]:
                bad.append({"gt_instance_id": gid, "metric_layer": next(iter(layers)), "pe_layer": pe["target_layer"]})
        check(f"{name}_target_layer_identical_within_instance_and_matches_pe", len(bad) == 0, bad[:10])

    # --- 6. zero-fabrication: unmatched/failed instances must not appear in any metric CSV ---
    non_pairable_ids = {r["gt_instance_id"] for r in acc if not to_bool(r["paired_explanation_success"])}
    check("non_pairable_count_27", len(non_pairable_ids) == 27, len(non_pairable_ids))

    for name, rows in [("localization", loc), ("faithfulness", faith),
                       ("stability_v1", stab1), ("stability_v2", stab2)]:
        present_ids = {r["gt_instance_id"] for r in rows}
        leaked = present_ids & non_pairable_ids
        check(f"zero_fabricated_rows_in_{name}_for_unmatched_or_failed_instances",
              len(leaked) == 0, sorted(leaked)[:10])

    # localization/faithfulness row counts: 65 paired x 2 methods = 130
    check("localization_130_rows", len(loc) == 130, len(loc))
    check("faithfulness_130_rows", len(faith) == 130, len(faith))
    check("stability_v1_780_rows", len(stab1) == 780, len(stab1))
    check("stability_v2_780_rows", len(stab2) == 780, len(stab2))

    loc_ids = {r["gt_instance_id"] for r in loc}
    check("localization_ids_subset_of_paired_success_ids", loc_ids <= acc_paired_ids,
          sorted(loc_ids - acc_paired_ids)[:5])
    check("localization_covers_all_65_paired_ids", loc_ids == acc_paired_ids,
          {"missing": sorted(acc_paired_ids - loc_ids)[:5], "extra": sorted(loc_ids - acc_paired_ids)[:5]})

    # --- 7. duplicate-row checks inside each metric CSV ---
    def dup_check(name, rows, key_fields):
        keys = [tuple(r[k] for k in key_fields) for r in rows]
        dup = len(keys) - len(set(keys))
        check(f"{name}_no_duplicate_rows", dup == 0, {"n": len(keys), "n_unique": len(set(keys))})

    dup_check("localization", loc, ["gt_instance_id", "method"])
    dup_check("faithfulness", faith, ["gt_instance_id", "method"])
    dup_check("stability_v1", stab1, ["gt_instance_id", "method", "perturbation"])
    dup_check("stability_v2", stab2, ["gt_instance_id", "method", "perturbation"])
    dup_check("metric_eligibility", elig, ["gt_instance_id"])

    # --- 8. no TEST-split access anywhere in F2.5's own new code ---
    analysis_dir = Path(__file__).resolve().parent
    this_file = Path(__file__).resolve().name
    # Deliberately not spelled out as plain substrings in this file's own
    # source (that would trip this very check on a self-referential
    # mention) -- built from parts instead, and this audit file itself is
    # excluded from the scan since it legitimately stores the token list.
    test_split_tokens = [
        "CHENAB_TEST_IMAGES" + "_DIR", "CHENAB_TEST_LABELS" + "_DIR",
        "/test/images", "/test/labels",
        "chenab_detection_training_v1/" + "test", "test_manifest.csv",
    ]
    offending = []
    for py in analysis_dir.glob("*.py"):
        if py.name == this_file:
            continue
        text = py.read_text(encoding="utf-8")
        for tok in test_split_tokens:
            if tok in text:
                offending.append({"file": py.name, "token": tok})
    check("no_test_split_token_in_f25_analysis_scripts", len(offending) == 0, offending)

    # --- 9. scoped protected-artifact checksum re-verification ---
    # Not a re-run of the full 691-file F2.4-B baseline (already reported,
    # 691/691 PASS) -- this independently hashes the specific files THIS
    # stage reads from, to confirm nothing drifted since F2.4-B's own
    # verification, and records the hashes for a future re-check.
    scoped_files = [
        E02_CHECKPOINT,
        F24_FULL / "manifests" / "all_instances_accounting.csv",
        F24_FULL / "metrics" / "localization.csv",
        F24_FULL / "metrics" / "faithfulness.csv",
        F24_FULL / "metrics" / "stability.csv",
        F24_FULL / "metrics" / "stability_v2_degenerate_flagged.csv",
        F24_FULL / "reports" / "f24_full_evaluation_report.md",
        F24_PILOT / "reports" / "statistical_analysis_plan.md",
        F24_PILOT / "metrics" / "stability_pilot.csv",
        F23_FULL / "manifests" / "execution_manifest.csv",
        F23_FULL / "manifests" / "paired_explanations_manifest.csv",
        F23_FULL / "manifests" / "target_linkage.csv",
        CONFIG / "complexity_protocol_snapshot_pp1_v1.json",
        CONFIG / "f24b_full_evaluation_config_v1.json",
        CONFIG / "f23b_execution_config_v1.json",
    ]
    hashes = {}
    missing = []
    for p in scoped_files:
        if not p.exists():
            missing.append(str(p))
            continue
        hashes[str(p.relative_to(Path(__file__).resolve().parents[4]))] = sha256_file(p)
    check("scoped_protected_files_all_present", len(missing) == 0, missing)

    e02_hash = hashes.get(str(E02_CHECKPOINT.relative_to(Path(__file__).resolve().parents[4])))
    check("e02_checkpoint_sha256_matches_recorded_value",
          e02_hash == "c578003ca158a892ac90de3db7b7552aefabd5292e311c6dd0c76faf717628d3",
          e02_hash)

    report["recomputed_headline"] = {
        "n_images": len(images), "n_gt_instances": n_rows, "n_matched": len(matched),
        "n_paired_success": len(paired_success), "n_degenerate_one_sided": len(degenerate_rows),
        "per_group": per_group,
    }
    report["scoped_checksums"] = hashes
    report["overall_status"] = "PASS" if not report["blockers"] else "BLOCKED"
    return report


if __name__ == "__main__":
    rep = run()
    out_path = F25_OUT / "checksums" / "task1_integrity_reaudit.json"
    write_json(out_path, rep)
    n_checks = len(rep["checks"])
    n_fail = len(rep["blockers"])
    print(f"Task 1 audit: {n_checks} checks, {n_fail} failed. Status={rep['overall_status']}")
    if rep["blockers"]:
        for b in rep["blockers"]:
            print("BLOCKER:", b["check"], b["detail"])
    print("Written:", out_path)
