"""F2.5 AMENDMENT -- reconstructed per-selected-image / per-family alternative
complexity-group labels for the 60 XAI-sampled VALID images (and the 48
evaluable source families among them).

Background / what this fixes
-----------------------------
F2.5 Task 10 (``f25_task10_weight_sensitivity.py``) tried to check whether
the 60 XAI-selected images' complexity-group labels (Simple/Moderate/High)
are sensitive to the complexity-scoring protocol's known weighting
instability. It attempted this by joining the 60 (VALID-split) image_ids
against ``results/calibration/weight_sensitivity_v2.csv`` -- a TRAIN-split
artifact (1747 TRAIN images). It found zero overlap and concluded that
per-selected-image alternative labels were "not available", falling back to
a weaker VALID-population-aggregate-only diagnostic (37.3% stable, 354
images, from F2.2B Task 6).

That zero-overlap finding is correct, but the conclusion drawn from it is
not: TRAIN and VALID are *intentionally* disjoint splits (that is correct
dataset hygiene, not a gap), so no image_id join between them was ever
going to work, by design. The actual fix -- already demonstrated at
population scale by ``dataset/f22b_distribution_audit.py`` (which produced
exactly the 132/354 VALID-wide stable count F2.5 cited) -- is to apply the
7 fixed weighting schemes' **own TRAIN-derived tertile thresholds** (a
property of the scoring FORMULA, frozen in
``results/calibration/b3_task2_weight_sensitivity_summary.json``) to each
VALID image's **own already-extracted raw feature values** (from
``valid_complexity_manifest.csv``), using the same frozen TRAIN z-score
normalization parameters (``config/calibration_v1.json``). No TRAIN image_id
is ever looked up for any VALID image -- only the frozen formula is reused.

This module:
  1. Reproduces the full-354-image 132/354 stable count as a correctness
     check (must match ``f22b_task6_distribution_audit.json`` exactly;
     the single most important validation in this amendment).
  2. Scopes the identical method down to the 60 XAI-selected images.
  3. Resolves the per-image alternative labels to the 48 evaluable source
     families (every family in this XAI-evaluated pool maps to exactly one
     image -- verified explicitly, see ``resolve_family_group`` docstring).
  4. Recomputes the F2.5 headline group-stratified metric summaries
     (pointing-hit rate, GT-box energy concentration, Grad-CAM-vs-
     Grad-CAM++ stability gap) under each alternative scheme's group
     assignment, reusing F2.5's own family-level bootstrap utility
     unchanged.

Reads only (never writes to any of these):
  - results/evaluation_sampling/f22b_pp1_v1/manifests/valid_complexity_manifest.csv
  - results/evaluation_sampling/f22b_pp1_v1/manifests/xai_primary_sample.csv
  - results/calibration/b3_task2_weight_sensitivity_summary.json
  - results/evaluation_sampling/f22b_pp1_v1/reports/f22b_task6_distribution_audit.json
  - config/calibration_v1.json
  - complexity/scoring.py (functions reused verbatim, not reimplemented)
  - results/complexity_xai_analysis/f25_pp1_v1/manifests/{image,family}_level_dataset.csv
  - analysis/f25_bootstrap.py (bootstrap utility reused verbatim)

Writes only under:
  - results/complexity_xai_analysis/f25_sensitivity_amendment_v1/
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.analysis.f25_common import (
    F2_ROOT, F22B, CALIBRATION, CONFIG, F25_OUT,
    load_all, read_csv, to_float, write_csv, write_json, sha256_file,
)
from research.component1.function2.analysis.f25_task2_datasets import (
    build_instance_level, build_family_level,
)
from research.component1.function2.analysis.f25_bootstrap import bootstrap_family_mean
from research.component1.function2.complexity.scoring import (
    FEATURE_NAMES,
    NormalizationParams,
    assign_group,
    compute_composite_score,
)

E02_CHECKPOINT = REPO_ROOT / "runs" / "component1_training" / "E02_optimized" / "weights" / "best.pt"
F2_RESULTS_DIR = F2_ROOT / "results"
F2_CONFIG_DIR = F2_ROOT / "config"
# Source modules this amendment reuses (read-only); every existing .py file
# under these dirs must remain byte-identical. The amendment's OWN new
# files (this module + its test file) are excluded explicitly below.
F2_SOURCE_DIRS = ["complexity", "dataset", "xai", "evaluation", "analysis"]
NEW_FILES_THIS_AMENDMENT = {"analysis/f25_sensitivity_amendment.py", "tests/test_f25_sensitivity_amendment.py"}
# The new amendment output tree is excluded from the protected scope -- it
# is this task's own, actively-written output, not a pre-existing artifact.
EXCLUDED_DIR_PREFIX = "results/complexity_xai_analysis/f25_sensitivity_amendment_v1"

# ---------------------------------------------------------------- paths ---
VALID_MANIFEST_PATH = F22B / "manifests" / "valid_complexity_manifest.csv"
XAI_PRIMARY_SAMPLE_PATH = F22B / "manifests" / "xai_primary_sample.csv"
WEIGHT_SENSITIVITY_SUMMARY_PATH = CALIBRATION / "b3_task2_weight_sensitivity_summary.json"
F22B_TASK6_AUDIT_PATH = F22B / "reports" / "f22b_task6_distribution_audit.json"
CALIBRATION_CONFIG_PATH = CONFIG / "calibration_v1.json"
F25_IMAGE_LEVEL_PATH = F25_OUT / "manifests" / "image_level_dataset.csv"
F25_FAMILY_LEVEL_PATH = F25_OUT / "manifests" / "family_level_dataset.csv"

AMEND_OUT = F2_ROOT / "results" / "complexity_xai_analysis" / "f25_sensitivity_amendment_v1"

GROUPS = ("Simple", "Moderate", "High")
METHODS = ("gradcam", "gradcampp")


# ------------------------------------------------------------- loading ---
def load_normalization() -> dict[str, NormalizationParams]:
    with open(CALIBRATION_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    feats = cfg["normalization"]["features"]
    return {name: NormalizationParams(mean=feats[name]["mean"], std=feats[name]["std"]) for name in FEATURE_NAMES}


def load_schemes_and_thresholds() -> tuple[dict[str, dict[str, float]], dict[str, tuple[float, float]], str]:
    """Returns (schemes {name: weights}, thresholds {name: (simple_max, moderate_max)}, baseline_key).

    Each of the 7 schemes carries its OWN distinct TRAIN-derived tertile
    threshold pair (confirmed by reading ``b3_task2_weight_sensitivity.py``
    line-by-line: it calls ``tertile_thresholds(scores_list)`` separately
    inside the per-scheme loop, storing the result in
    ``scheme_thresholds[scheme_name]`` for every scheme including the
    baseline -- these are NOT the baseline's thresholds reused for all 7).
    This is also exactly what ``f22b_distribution_audit.py`` already reads
    and applies at the 354-image population scale (see its
    ``thresholds_by_scheme`` dict, built the same way from this same JSON).
    """
    with open(WEIGHT_SENSITIVITY_SUMMARY_PATH, encoding="utf-8") as f:
        wsens = json.load(f)
    schemes = dict(wsens["weighting_schemes_tested"])
    baseline_key = wsens["baseline_scheme"]
    thresholds = {
        baseline_key: (wsens["baseline_thresholds"]["simple_max"], wsens["baseline_thresholds"]["moderate_max"]),
    }
    for scheme_name, alt in wsens["per_alternative_weighting_results"].items():
        thresholds[scheme_name] = (alt["simple_max"], alt["moderate_max"])
    return schemes, thresholds, baseline_key


def _to_bool(v) -> bool:
    return str(v).strip().lower() == "true"


def load_valid_manifest_rows() -> list[dict]:
    return read_csv(VALID_MANIFEST_PATH)


def load_primary_sample_rows() -> list[dict]:
    return read_csv(XAI_PRIMARY_SAMPLE_PATH)


# ------------------------------------------------------- core scoring ---
def compute_scheme_groups(
    raw_by_id: dict[str, dict[str, Optional[float]]],
    normalization: dict[str, NormalizationParams],
    schemes: dict[str, dict[str, float]],
    thresholds: dict[str, tuple[float, float]],
) -> dict[str, dict[str, Optional[str]]]:
    """For every scheme, recompute the composite score (scheme's own
    weights, the SAME frozen normalization for every scheme) and assign a
    group using THAT scheme's own TRAIN-derived threshold pair. Mirrors
    f22b_distribution_audit.py's weighting-sensitivity loop exactly."""
    scheme_groups: dict[str, dict[str, Optional[str]]] = {}
    for scheme_name, weights in schemes.items():
        simple_max, moderate_max = thresholds[scheme_name]
        groups: dict[str, Optional[str]] = {}
        for image_id, raw in raw_by_id.items():
            res = compute_composite_score(raw, normalization, weights=weights)
            g, _ = assign_group(res.composite_score, simple_max, moderate_max)
            groups[image_id] = g
        scheme_groups[scheme_name] = groups
    return scheme_groups


def stability_flags(
    scheme_groups: dict[str, dict[str, Optional[str]]], ids: list[str]
) -> dict[str, dict]:
    out = {}
    scheme_names = list(scheme_groups.keys())
    for image_id in ids:
        groups_per_scheme = {s: scheme_groups[s].get(image_id) for s in scheme_names}
        distinct = set(g for g in groups_per_scheme.values() if g is not None)
        out[image_id] = {
            "groups_per_scheme": groups_per_scheme,
            "n_distinct_groups_across_schemes": len(distinct),
            "stable_across_all_7": len(distinct) == 1,
        }
    return out


# ------------------------------------------------ Task 2 correctness check
def reproduce_354_check() -> dict:
    """Independently reproduces F2.2B Task 6's 132/354 VALID-wide stable
    count, using ONLY this module's own loading/scoring functions (not by
    importing f22b_distribution_audit.py's internals) -- an independent
    re-derivation, not a restatement. This is the most important
    correctness test in the whole amendment: if it does not match, the
    60-image/48-family result below cannot be trusted."""
    normalization = load_normalization()
    schemes, thresholds, baseline_key = load_schemes_and_thresholds()
    rows = load_valid_manifest_rows()
    eligible_rows = [r for r in rows if _to_bool(r["complexity_eligible"])]

    raw_by_id = {
        r["image_id"]: {f: to_float(r[f]) for f in FEATURE_NAMES} for r in eligible_rows
    }
    baseline_group_by_id = {r["image_id"]: (r["complexity_group"] or None) for r in eligible_rows}

    scheme_groups = compute_scheme_groups(raw_by_id, normalization, schemes, thresholds)
    flags = stability_flags(scheme_groups, list(raw_by_id.keys()))

    n_stable = sum(1 for v in flags.values() if v["stable_across_all_7"])
    n_total = len(flags)

    # Internal consistency: the baseline_equal scheme, applied with this
    # same method, must reproduce the manifest's OWN already-assigned
    # complexity_group for every eligible row exactly (it is the same
    # formula that produced that column in the first place).
    mismatches = [
        img_id for img_id in raw_by_id
        if scheme_groups[baseline_key][img_id] != baseline_group_by_id[img_id]
    ]

    with open(F22B_TASK6_AUDIT_PATH, encoding="utf-8") as f:
        recorded = json.load(f)["weighting_sensitivity_diagnostic_valid_descriptive_only"]

    return {
        "n_eligible_valid_images": n_total,
        "n_stable_across_all_7_schemes_recomputed": n_stable,
        "n_unstable_recomputed": n_total - n_stable,
        "fraction_stable_recomputed": round(n_stable / n_total, 4) if n_total else None,
        "recorded_n_eligible_valid_images": recorded["n_eligible_valid_images_compared"],
        "recorded_n_stable": recorded["n_stable_across_all_7_schemes"],
        "recorded_fraction_stable": recorded["fraction_stable"],
        "matches_f22b_task6_exactly": (
            n_total == recorded["n_eligible_valid_images_compared"]
            and n_stable == recorded["n_stable_across_all_7_schemes"]
        ),
        "baseline_equal_scheme_reproduces_manifest_complexity_group_exactly": len(mismatches) == 0,
        "n_baseline_mismatches": len(mismatches),
        "baseline_mismatch_sample": mismatches[:10],
    }


# ------------------------------------------------------- selected (60) ---
def build_selected_image_table() -> tuple[list[dict], dict]:
    normalization = load_normalization()
    schemes, thresholds, baseline_key = load_schemes_and_thresholds()

    manifest_rows = load_valid_manifest_rows()
    manifest_by_id = {r["image_id"]: r for r in manifest_rows}
    sample_rows = load_primary_sample_rows()

    selected_ids = [r["image_id"] for r in sample_rows]

    # Task 1 verification: every one of the 60 has real, non-null features
    # and is complexity_eligible in the manifest.
    missing_from_manifest = [i for i in selected_ids if i not in manifest_by_id]
    null_feature_images = []
    for i in selected_ids:
        m = manifest_by_id.get(i)
        if m is None:
            continue
        for f in FEATURE_NAMES:
            if to_float(m[f]) is None:
                null_feature_images.append((i, f))
    ineligible_images = [i for i in selected_ids if manifest_by_id.get(i, {}).get("complexity_eligible") != "True"]

    raw_by_id = {i: {f: to_float(manifest_by_id[i][f]) for f in FEATURE_NAMES} for i in selected_ids}
    scheme_groups = compute_scheme_groups(raw_by_id, normalization, schemes, thresholds)
    flags = stability_flags(scheme_groups, selected_ids)

    scheme_names = list(schemes.keys())
    rows = []
    for r in sample_rows:
        image_id = r["image_id"]
        m = manifest_by_id[image_id]
        baseline_manifest_group = m["complexity_group"] or None
        baseline_sample_group = r["complexity_group"] or None
        row = {
            "image_id": image_id,
            "source_family_id": r["source_family_id"],
            "baseline_group_valid_complexity_manifest": baseline_manifest_group,
            "baseline_group_xai_primary_sample": baseline_sample_group,
            "baseline_group_consistency_ok": baseline_manifest_group == baseline_sample_group,
        }
        for s in scheme_names:
            row[f"group_{s}"] = scheme_groups[s].get(image_id)
        row["n_distinct_groups_across_7_schemes"] = flags[image_id]["n_distinct_groups_across_schemes"]
        row["stable_across_all_7_schemes"] = flags[image_id]["stable_across_all_7"]
        rows.append(row)

    verification = {
        "n_selected_images": len(selected_ids),
        "n_missing_from_valid_complexity_manifest": len(missing_from_manifest),
        "missing_from_manifest_sample": missing_from_manifest[:10],
        "n_null_feature_entries": len(null_feature_images),
        "null_feature_entries_sample": null_feature_images[:10],
        "n_ineligible_images": len(ineligible_images),
        "ineligible_images_sample": ineligible_images[:10],
        "all_60_verified_real_nonnull_eligible": (
            len(missing_from_manifest) == 0 and len(null_feature_images) == 0 and len(ineligible_images) == 0
        ),
        "baseline_scheme_key": baseline_key,
        "scheme_names": scheme_names,
    }
    return rows, verification


# ----------------------------------------------------- family-level (48) -
def resolve_family_group(image_groups: list[Optional[str]]) -> tuple[Optional[str], str]:
    """Resolve a source family's alternative group under ONE scheme from
    the (possibly several) images belonging to it under that scheme.

    In the F2.5-evaluable 48-family pool, every source_family_id maps to
    EXACTLY ONE image_id (verified explicitly by F2.5 Task 2's own
    ``multi_image_families`` check in ``f25_task2_datasets.py``, which
    found zero families with more than one image in this pool -- the same
    singleton-family structure already assumed throughout F2.5's baseline
    family-level aggregation, see ``build_family_level``'s docstring there).
    This resolver is therefore a pass-through in every case this amendment
    actually encounters, but is written generically (majority vote across
    the family's images, consistent with how the baseline pipeline treats
    a family-level value as the group shared by its images) so it does not
    silently assume singleton families if that ever changes:
      - all images agree -> that group,
      - images disagree -> the single most common group (mode); a tie is
        reported as None with reason "tied_majority_vote" (never guessed).
    """
    valid = [g for g in image_groups if g is not None]
    if not valid:
        return None, "no_valid_group"
    counts = {g: valid.count(g) for g in set(valid)}
    max_count = max(counts.values())
    winners = [g for g, c in counts.items() if c == max_count]
    if len(valid) == 1:
        return valid[0], "single_image_family_pass_through"
    if len(winners) == 1:
        return winners[0], "majority_vote"
    return None, "tied_majority_vote"


def build_family_level_table(selected_rows: list[dict], scheme_names: list[str]) -> tuple[list[dict], dict]:
    f25_image_rows = read_csv(F25_IMAGE_LEVEL_PATH)
    evaluable_image_ids = sorted({r["image_id"] for r in f25_image_rows})
    image_to_family = {r["image_id"]: r["source_family_id"] for r in f25_image_rows}

    by_image = {r["image_id"]: r for r in selected_rows}
    missing = [i for i in evaluable_image_ids if i not in by_image]

    family_to_images: dict[str, list[str]] = {}
    for img_id in evaluable_image_ids:
        fam = image_to_family[img_id]
        family_to_images.setdefault(fam, []).append(img_id)

    multi_image_families = {f: imgs for f, imgs in family_to_images.items() if len(imgs) > 1}

    rows = []
    for fam in sorted(family_to_images):
        imgs = family_to_images[fam]
        row = {"source_family_id": fam, "n_images_in_family": len(imgs), "image_ids": ";".join(imgs)}
        baseline_groups = [by_image[i]["baseline_group_valid_complexity_manifest"] for i in imgs]
        resolved_baseline, baseline_method = resolve_family_group(baseline_groups)
        row["baseline_group"] = resolved_baseline
        row["baseline_resolution_method"] = baseline_method

        per_scheme_distinct = set()
        for s in scheme_names:
            img_groups = [by_image[i][f"group_{s}"] for i in imgs]
            resolved, method = resolve_family_group(img_groups)
            row[f"group_{s}"] = resolved
            row[f"resolution_method_{s}"] = method
            if resolved is not None:
                per_scheme_distinct.add(resolved)
        row["n_distinct_groups_across_7_schemes"] = len(per_scheme_distinct)
        row["stable_across_all_7_schemes"] = len(per_scheme_distinct) == 1
        rows.append(row)

    verification = {
        "n_evaluable_images": len(evaluable_image_ids),
        "n_evaluable_families": len(family_to_images),
        "n_missing_images_from_selected_table": len(missing),
        "n_families_with_more_than_one_image": len(multi_image_families),
        "multi_image_families": multi_image_families,
        "family_resolution_was_pass_through_for_all_families": len(multi_image_families) == 0,
    }
    return rows, verification


# ------------------------------------------------- Task 3c metric redo ---
def per_scheme_reassignment_summary(rows: list[dict], scheme_names: list[str], baseline_key: str, baseline_field: str) -> dict:
    """Per-scheme n_compared/n_changed-vs-baseline/fraction_changed, plus
    a 3x3 confusion matrix (baseline group rows x alternative group cols),
    mirroring exactly the structure ``b3_task2_weight_sensitivity.py`` and
    ``f22b_distribution_audit.py`` already report (same field names), now
    computed over this 60-image / 48-family amendment's own rows instead
    of the 1747-TRAIN-image or 354-VALID-image populations."""
    out = {}
    for scheme in scheme_names:
        if scheme == baseline_key:
            continue
        confusion = {b: {a: 0 for a in GROUPS} for b in GROUPS}
        n_changed = 0
        n_compared = 0
        for r in rows:
            bgroup = r[baseline_field]
            agroup = r[f"group_{scheme}"]
            if bgroup is None or agroup is None:
                continue
            n_compared += 1
            confusion[bgroup][agroup] += 1
            if agroup != bgroup:
                n_changed += 1
        out[scheme] = {
            "n_compared": n_compared,
            "n_changed_vs_baseline": n_changed,
            "fraction_changed": round(n_changed / n_compared, 4) if n_compared else None,
            "confusion_matrix_baseline_rows_alt_cols": confusion,
        }
    return out


def group_counts_per_scheme(rows: list[dict], scheme_names: list[str]) -> dict[str, dict[str, int]]:
    out = {}
    for scheme in scheme_names:
        counts = {g: 0 for g in GROUPS}
        for r in rows:
            g = r[f"group_{scheme}"]
            if g in counts:
                counts[g] += 1
        out[scheme] = counts
    return out


def _family_metric_values(family_level_dataset_rows: list[dict], method: str, field: str) -> dict[str, Optional[float]]:
    return {
        r["source_family_id"]: to_float(r[field]) if r[field] not in (None, "") else None
        for r in family_level_dataset_rows if r["method"] == method
    }


def _family_pointing_hit_rate(family_level_dataset_rows: list[dict], method: str) -> dict[str, Optional[float]]:
    out = {}
    for r in family_level_dataset_rows:
        if r["method"] != method:
            continue
        v = r["pointing_hit_rate"]
        out[r["source_family_id"]] = float(v) if v not in (None, "") else None
    return out


def _mean(xs: list[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def build_alternative_group_metric_summary(
    instance_rows: list[dict], family_rows: list[dict], scheme_names: list[str]
) -> list[dict]:
    """Recomputes the F2.5 headline per-group metric summaries (pointing
    accuracy, GT-box energy concentration, stability) under each alternative
    scheme's family-level group assignment.

    Reports TWO point estimates per row, for direct comparability with the
    two numbers F2.5 itself reports side by side (see
    ``f25_task5_complexity_stratified.py``):
      - ``instance_pooled_point_estimate``: the same instance-pooled
        statistic F2.5's headline prose quotes (e.g. the 21.5%/33.8%
        pointing-hit-rate numbers in ``f25_results_summary.md``) --
        hits/valid_n pooled over every paired instance in the (scheme,
        group, method) cell, exactly as ``f25_task5_complexity_stratified
        .group_selection_counts``/``run`` computes it.
      - ``family_first_bootstrap_point_estimate`` (+ ``ci_lo``/``ci_hi``):
        the family-level block-bootstrap mean, reusing F2.5's own
        ``bootstrap_family_mean`` (``f25_bootstrap.py``) UNCHANGED, over the
        same per-family mean values F2.5's own family-level dataset already
        carries (``family_level_dataset.csv``) -- re-partitioned here by
        each scheme's alternative group assignment instead of the baseline
        complexity_group.
    This is a sensitivity/robustness check on the EXISTING fixed 48-family
    dataset -- no resampling of which families/images were originally
    selected, no new XAI computation.
    """
    f25_family_dataset = read_csv(F25_FAMILY_LEVEL_PATH)
    group_by_family_by_scheme: dict[str, dict[str, Optional[str]]] = {
        s: {r["source_family_id"]: r[f"group_{s}"] or None for r in family_rows} for s in scheme_names
    }
    # instance's family == instance's image in this evaluable pool (verified
    # 1:1 by build_family_level_table / f25_task2_datasets); map instance
    # rows to family-scheme-group via their own source_family_id directly.

    family_metrics_by_method = {
        m: {
            "pointing_hit_rate": _family_pointing_hit_rate(f25_family_dataset, m),
            "energy_fraction_mean": _family_metric_values(f25_family_dataset, m, "energy_fraction_mean"),
            "insertion_auc_mean": _family_metric_values(f25_family_dataset, m, "insertion_auc_mean"),
            "deletion_auc_mean": _family_metric_values(f25_family_dataset, m, "deletion_auc_mean"),
            "stability_v1_mean_similarity": _family_metric_values(f25_family_dataset, m, "stability_v1_mean_similarity"),
        }
        for m in METHODS
    }
    stab_gc = family_metrics_by_method["gradcam"]["stability_v1_mean_similarity"]
    stab_gcpp = family_metrics_by_method["gradcampp"]["stability_v1_mean_similarity"]
    stability_gap_by_family = {
        f: stab_gc[f] - stab_gcpp[f]
        for f in stab_gc if stab_gc.get(f) is not None and stab_gcpp.get(f) is not None
    }

    out_rows = []
    for scheme in scheme_names:
        group_by_family = group_by_family_by_scheme[scheme]
        for group in GROUPS:
            fams_in_group = {f for f, g in group_by_family.items() if g == group}
            n_families_in_group = len(fams_in_group)
            inst_in_group = [r for r in instance_rows if group_by_family.get(r["source_family_id"]) == group]

            for method in METHODS:
                m_inst = [r for r in inst_in_group if r["method"] == method]

                def _pointing(m_inst=m_inst):
                    valid_pt = [r for r in m_inst if r["pointing_status"] == "valid"]
                    hits = sum(1 for r in valid_pt if r["pointing_hit"] in ("True", True))
                    return (hits / len(valid_pt)) if valid_pt else None, len(valid_pt)

                instance_pooled = {
                    "pointing_hit_rate": _pointing()[0],
                    "energy_fraction_mean": _mean([to_float(r["energy_fraction"]) for r in m_inst if r["energy_status"] == "valid"]),
                    "insertion_auc_mean": _mean([to_float(r["insertion_auc"]) for r in m_inst if r["insertion_status"] == "valid"]),
                    "deletion_auc_mean": _mean([to_float(r["deletion_auc"]) for r in m_inst if r["deletion_status"] == "valid"]),
                    "stability_v1_mean_similarity": _mean([to_float(r["stability_v1_mean_similarity"]) for r in m_inst]),
                }
                instance_pooled_n = {
                    "pointing_hit_rate": _pointing()[1],
                    "energy_fraction_mean": sum(1 for r in m_inst if r["energy_status"] == "valid"),
                    "insertion_auc_mean": sum(1 for r in m_inst if r["insertion_status"] == "valid"),
                    "deletion_auc_mean": sum(1 for r in m_inst if r["deletion_status"] == "valid"),
                    "stability_v1_mean_similarity": sum(1 for r in m_inst if r["stability_v1_mean_similarity"] not in (None, "")),
                }

                for metric_name, values_by_family in family_metrics_by_method[method].items():
                    scoped = {f: v for f, v in values_by_family.items() if f in fams_in_group}
                    boot = bootstrap_family_mean(scoped)
                    out_rows.append({
                        "scheme": scheme,
                        "group": group,
                        "method": method,
                        "metric": metric_name,
                        "n_families_in_group": n_families_in_group,
                        "n_families_with_valid_metric": boot["n_effective_families"],
                        "instance_pooled_point_estimate": instance_pooled[metric_name],
                        "instance_pooled_n": instance_pooled_n[metric_name],
                        "family_first_bootstrap_point_estimate": boot["point_estimate_family_first_mean"],
                        "ci_lo": boot.get("ci_lo"),
                        "ci_hi": boot.get("ci_hi"),
                        "ci_skipped": boot.get("ci_skipped"),
                        "skip_reason": boot.get("skip_reason"),
                    })
            # stability gap (gradcam - gradcampp), method field = "gap"
            scoped_gap = {f: v for f, v in stability_gap_by_family.items() if f in fams_in_group}
            boot_gap = bootstrap_family_mean(scoped_gap)
            gap_inst_pooled = _mean([
                to_float(r["stability_v1_mean_similarity"])
                for r in inst_in_group if r["method"] == "gradcam"
            ])
            gap_inst_pooled_pp = _mean([
                to_float(r["stability_v1_mean_similarity"])
                for r in inst_in_group if r["method"] == "gradcampp"
            ])
            instance_pooled_gap = (
                gap_inst_pooled - gap_inst_pooled_pp
                if gap_inst_pooled is not None and gap_inst_pooled_pp is not None else None
            )
            out_rows.append({
                "scheme": scheme,
                "group": group,
                "method": "gap_gradcam_minus_gradcampp",
                "metric": "stability_v1_mean_similarity",
                "n_families_in_group": n_families_in_group,
                "n_families_with_valid_metric": boot_gap["n_effective_families"],
                "instance_pooled_point_estimate": instance_pooled_gap,
                "instance_pooled_n": None,
                "family_first_bootstrap_point_estimate": boot_gap["point_estimate_family_first_mean"],
                "ci_lo": boot_gap.get("ci_lo"),
                "ci_hi": boot_gap.get("ci_hi"),
                "ci_skipped": boot_gap.get("ci_skipped"),
                "skip_reason": boot_gap.get("skip_reason"),
            })
    return out_rows


# --------------------------------------------------- integrity checksums
def _rel(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix()


def compute_protected_checksums() -> dict[str, str]:
    """Hashes EVERY existing file under the F2.1-F2.5 protected scope:
    the E02 checkpoint, every file already present under
    research/component1/function2/results/ (the entire F2.1-F2.5 output
    tree, including the original, untouched f25_pp1_v1/ directory -- but
    EXCLUDING this amendment's own new, actively-written output
    directory), every file under config/, and every pre-existing .py
    module this amendment reads from (complexity/, dataset/, xai/,
    evaluation/, analysis/ -- excluding this amendment's own two new
    files). Used both to record a baseline before this amendment's work
    and to re-verify nothing drifted afterwards."""
    hashes: dict[str, str] = {}

    if E02_CHECKPOINT.exists():
        hashes[_rel(E02_CHECKPOINT)] = sha256_file(E02_CHECKPOINT)

    for p in sorted(F2_RESULTS_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = _rel(p)
        if rel.startswith(f"research/component1/function2/{EXCLUDED_DIR_PREFIX}"):
            continue
        hashes[rel] = sha256_file(p)

    for p in sorted(F2_CONFIG_DIR.rglob("*")):
        if p.is_file():
            hashes[_rel(p)] = sha256_file(p)

    for sub in F2_SOURCE_DIRS:
        d = F2_ROOT / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*.py")):
            rel_short = f"{sub}/{p.name}"
            if rel_short in NEW_FILES_THIS_AMENDMENT:
                continue
            hashes[_rel(p)] = sha256_file(p)

    tests_dir = F2_ROOT / "tests"
    if tests_dir.exists():
        for p in sorted(tests_dir.glob("*.py")):
            rel_short = f"tests/{p.name}"
            if rel_short in NEW_FILES_THIS_AMENDMENT:
                continue
            hashes[_rel(p)] = sha256_file(p)

    return hashes


def write_protected_checksum_snapshot(label: str) -> Path:
    """Writes a full {relative_path: sha256} snapshot under this
    amendment's own checksums/ directory, named by ``label`` (e.g.
    "before"/"after"). Returns the path written."""
    hashes = compute_protected_checksums()
    out_path = AMEND_OUT / "checksums" / f"protected_checksums_{label}.json"
    write_json(out_path, {"label": label, "n_files": len(hashes), "sha256_by_relative_path": hashes})
    return out_path


def verify_protected_checksums_unchanged() -> dict:
    """Compares the 'before' snapshot (recorded at the start of this
    amendment's work, before any new file was written) against hashes
    recomputed right now. Any difference is a hard failure -- this
    amendment must never alter any existing F2.1-F2.5 artifact, the E02
    checkpoint, or any pre-existing shared module."""
    before_path = AMEND_OUT / "checksums" / "protected_checksums_before.json"
    with open(before_path, encoding="utf-8") as f:
        before = json.load(f)["sha256_by_relative_path"]
    after = compute_protected_checksums()

    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(p for p in (set(before) & set(after)) if before[p] != after[p])

    result = {
        "n_files_checked": len(before),
        "n_missing_files": len(missing),
        "missing_files": missing[:20],
        "n_newly_appeared_files_in_protected_scope": len(added),
        "newly_appeared_files_sample": added[:20],
        "n_changed_files": len(changed),
        "changed_files": changed,
        "all_protected_artifacts_byte_unchanged": len(missing) == 0 and len(changed) == 0,
    }
    write_json(AMEND_OUT / "checksums" / "protected_checksum_verification.json", result)
    return result


# -------------------------------------------------------------- driver ---
def run() -> dict:
    AMEND_OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("manifests", "reports", "checksums", "tests_support"):
        (AMEND_OUT / sub).mkdir(parents=True, exist_ok=True)

    # Record a full protected-artifact checksum snapshot BEFORE writing
    # anything else this run -- the "before" half of the hard before/after
    # integrity check (see verify_protected_checksums_unchanged below).
    write_protected_checksum_snapshot("before")

    task2_check = reproduce_354_check()

    selected_rows, selection_verification = build_selected_image_table()
    schemes, _, baseline_key = load_schemes_and_thresholds()
    scheme_names = list(schemes.keys())

    family_rows, family_verification = build_family_level_table(selected_rows, scheme_names)

    d = load_all()
    instance_rows = build_instance_level(d)
    metric_summary_rows = build_alternative_group_metric_summary(instance_rows, family_rows, scheme_names)

    selected_fieldnames = (
        ["image_id", "source_family_id", "baseline_group_valid_complexity_manifest",
         "baseline_group_xai_primary_sample", "baseline_group_consistency_ok"]
        + [f"group_{s}" for s in scheme_names]
        + ["n_distinct_groups_across_7_schemes", "stable_across_all_7_schemes"]
    )
    write_csv(AMEND_OUT / "selected_image_alternative_labels.csv", selected_rows, selected_fieldnames)

    family_fieldnames = (
        ["source_family_id", "n_images_in_family", "image_ids", "baseline_group", "baseline_resolution_method"]
        + [f for s in scheme_names for f in (f"group_{s}", f"resolution_method_{s}")]
        + ["n_distinct_groups_across_7_schemes", "stable_across_all_7_schemes"]
    )
    write_csv(AMEND_OUT / "family_level_alternative_labels.csv", family_rows, family_fieldnames)

    metric_fieldnames = ["scheme", "group", "method", "metric", "n_families_in_group",
                         "n_families_with_valid_metric", "instance_pooled_point_estimate",
                         "instance_pooled_n", "family_first_bootstrap_point_estimate",
                         "ci_lo", "ci_hi", "ci_skipped", "skip_reason"]
    write_csv(AMEND_OUT / "alternative_group_metric_summary.csv", metric_summary_rows, metric_fieldnames)

    n_stable_60 = sum(1 for r in selected_rows if r["stable_across_all_7_schemes"])
    n_stable_48 = sum(1 for r in family_rows if r["stable_across_all_7_schemes"])

    reassignment_60 = per_scheme_reassignment_summary(
        selected_rows, scheme_names, baseline_key, "baseline_group_valid_complexity_manifest"
    )
    reassignment_48 = per_scheme_reassignment_summary(
        family_rows, scheme_names, baseline_key, "baseline_group"
    )
    group_counts_60 = group_counts_per_scheme(selected_rows, scheme_names)
    group_counts_48 = group_counts_per_scheme(family_rows, scheme_names)
    sparsity_48 = {
        scheme: {g: c for g, c in counts.items() if c < 5}
        for scheme, counts in group_counts_48.items()
    }
    any_sparse_strata_48 = any(sparsity_48[s] for s in sparsity_48)

    summary = {
        "task2_354_image_correctness_check": task2_check,
        "selection_verification_60_images": selection_verification,
        "family_verification_48_families": family_verification,
        "n_stable_of_60_selected_images": n_stable_60,
        "n_unstable_of_60_selected_images": 60 - n_stable_60,
        "fraction_stable_60": round(n_stable_60 / len(selected_rows), 4) if selected_rows else None,
        "n_stable_of_48_families": n_stable_48,
        "n_unstable_of_48_families": 48 - n_stable_48,
        "fraction_stable_48": round(n_stable_48 / len(family_rows), 4) if family_rows else None,
        "baseline_scheme_key": baseline_key,
        "scheme_names": scheme_names,
        "group_counts_per_scheme_60_images": group_counts_60,
        "group_counts_per_scheme_48_families": group_counts_48,
        "reassignment_vs_baseline_60_images": reassignment_60,
        "reassignment_vs_baseline_48_families": reassignment_48,
        "sparsity_cutoff_min_families": 5,
        "strata_below_sparsity_cutoff_48_families": sparsity_48,
        "any_sparse_strata_in_48_family_analysis": any_sparse_strata_48,
    }
    write_json(AMEND_OUT / "reports" / "amendment_summary.json", summary)

    # "after" half of the hard before/after protected-artifact integrity
    # check -- must show zero missing/changed files relative to "before".
    write_protected_checksum_snapshot("after")
    integrity = verify_protected_checksums_unchanged()
    summary["protected_artifact_integrity"] = integrity
    write_json(AMEND_OUT / "reports" / "amendment_summary.json", summary)

    return summary


if __name__ == "__main__":
    out = run()
    print(json.dumps(out["task2_354_image_correctness_check"], indent=2))
    print("n_stable_of_60_selected_images:", out["n_stable_of_60_selected_images"])
    print("n_stable_of_48_families:", out["n_stable_of_48_families"])
