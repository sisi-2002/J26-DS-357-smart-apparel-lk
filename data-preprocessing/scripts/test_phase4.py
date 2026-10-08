"""
Phase 4 targeted tests.
Run: python -I test_phase4.py
"""

import csv
import json
import math
import sys
import io
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
META = MASTER / "metadata"
PHASE3 = MASTER / "reports" / "phase3"
PHASE4 = MASTER / "reports" / "phase4"
EXPORT = MASTER / "processed" / "chenab_detection_v1"
RESEARCH = MASTER / "processed" / "research_subsets"
BG_PATCH = MASTER / "research" / "background_patches"
RAW = MASTER / "raw"

PASS = FAIL = 0


def read_csv_file(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f" | {detail}" if detail else ""))
        FAIL += 1


def s_bool(v):
    return str(v).strip().lower() == "true"


# =========================================================================
# 1. Phase 4.0 Reconciliation
# =========================================================================
def test_reconciliation():
    print("\n--- 1. Reconciliation ---")
    recon = read_csv_file(PHASE4 / "phase4_reconciliation.csv")
    check("3038 recon rows", len(recon) == 3038, f"got {len(recon)}")

    dispositions = Counter(r["disposition"] for r in recon)
    check("2655 exported", dispositions.get("exported", 0) == 2655)
    check("122 quarantined", dispositions.get("quarantined", 0) == 122)
    check("247 aitex", dispositions.get("aitex_research_subset", 0) == 247)

    ineligible = dispositions.get("annotation_excluded", 0) + \
                 dispositions.get("annotation_uncertain", 0)
    check("14 ineligible", ineligible == 14, f"got {ineligible}")

    unaccounted = [r for r in recon if r["disposition"] == "unaccounted"]
    check("0 unaccounted", len(unaccounted) == 0, f"got {len(unaccounted)}")

    check("integrity report exists",
          (PHASE4 / "phase4_dataset_integrity.md").exists())


# =========================================================================
# 2. Background patches
# =========================================================================
def test_background_patches():
    print("\n--- 2. Background patches ---")
    manifest = read_csv_file(BG_PATCH / "background_manifest.csv")
    check("patches extracted", len(manifest) > 100, f"got {len(manifest)}")

    for p in manifest[:5]:
        pp = BG_PATCH / "patches" / p["patch_filename"]
        check(f"patch {p['patch_id']} exists", pp.exists())

    check("all 256x256",
          all(int(p["patch_width"]) == 256 and int(p["patch_height"]) == 256
              for p in manifest))

    test_patches = [p for p in manifest if p["source_split"] == "test"]
    check("no test patches", len(test_patches) == 0, f"got {len(test_patches)}")

    fsm = read_csv_file(PHASE3 / "final_split_manifest.csv")
    quarantined_ids = {r["image_id"] for r in fsm
                       if r["assigned_split"] == "quarantined"}
    q_patches = [p for p in manifest if p["source_image_id"] in quarantined_ids]
    check("no quarantined patches", len(q_patches) == 0, f"got {len(q_patches)}")

    group_counts = Counter(p["group_id"] for p in manifest)
    over = [g for g, c in group_counts.items() if c > 3]
    check("max 3 patches per group", len(over) == 0,
          f"groups over limit: {len(over)}")

    overlap_ok = all(float(p["defect_overlap_fraction"]) < 0.10 for p in manifest)
    check("defect overlap < 10%", overlap_ok)

    config = json.loads((BG_PATCH / "background_extraction_config.json").read_text("utf-8"))
    check("config seed 42", config["random_seed"] == 42)
    check("config patch 256", config["patch_size"] == 256)

    chenab = [p for p in manifest if p["source"] == "chenab"]
    aitex = [p for p in manifest if p["source"] == "aitex"]
    check("chenab patches > 0", len(chenab) > 0, f"got {len(chenab)}")
    check("aitex patches > 0", len(aitex) > 0, f"got {len(aitex)}")
    check("141 aitex patches", len(aitex) == 141, f"got {len(aitex)}")


# =========================================================================
# 3. Complexity features
# =========================================================================
def test_complexity_features():
    print("\n--- 3. Complexity features ---")
    features = read_csv_file(PHASE4 / "complexity_features.csv")
    manifest = read_csv_file(BG_PATCH / "background_manifest.csv")
    check("features match patches", len(features) == len(manifest),
          f"features={len(features)} patches={len(manifest)}")

    ok = [r for r in features if r["status"] == "ok"]
    check("all features ok", len(ok) == len(features),
          f"ok={len(ok)} total={len(features)}")

    for r in ok[:20]:
        lv = float(r["luminance_var"])
        ge = float(r["glcm_entropy"])
        cv = float(r["colour_var"])
        check(f"features finite {r['patch_id']}",
              math.isfinite(lv) and math.isfinite(ge) and math.isfinite(cv) and
              lv >= 0 and ge >= 0 and cv >= 0,
              f"lv={lv} ge={ge} cv={cv}")

    config = json.loads((PHASE4 / "complexity_feature_config.json").read_text("utf-8"))
    check("config has protocol version", "version" in config or "protocol_version" in config)
    check("config has feature defs", "feature_definitions" in config)

    summary = read_csv_file(PHASE4 / "complexity_feature_summary.csv")
    check("3 feature summary rows", len(summary) == 3, f"got {len(summary)}")


# =========================================================================
# 4. Provisional calibration
# =========================================================================
def test_calibration():
    print("\n--- 4. Provisional calibration ---")
    assignments = read_csv_file(PHASE4 / "provisional_complexity_assignments.csv")
    check("assignments exist", len(assignments) > 0)

    check("all provisional_ai",
          all(r["calibration_mode"] == "provisional_ai" for r in assignments))

    check("none qc_validated",
          all(r["qc_validated"] in ("False", "false", "") for r in assignments))

    tiers = Counter(r["provisional_tier"] for r in assignments)
    check("3 tiers present", len(tiers) == 3, f"tiers={dict(tiers)}")
    check("Simple present", tiers.get("Simple", 0) > 0)
    check("Moderate present", tiers.get("Moderate", 0) > 0)
    check("High present", tiers.get("High", 0) > 0)

    thresholds = json.loads((PHASE4 / "provisional_thresholds.json").read_text("utf-8"))
    check("status pending_qc", thresholds["status"] == "pending_qc")
    check("mode provisional_ai", thresholds["mode"] == "provisional_ai")
    check("p33 < p67", thresholds["p33"] < thresholds["p67"])
    check("seed 42", thresholds.get("random_seed") == 42)
    check("WARNING present", "PROVISIONAL" in thresholds.get("WARNING", ""))

    cal_config = json.loads(
        (PHASE4 / "provisional_calibration_config.json").read_text("utf-8"))
    check("cal config pending_qc", cal_config["status"] == "pending_qc")

    test_assignments = [r for r in assignments if r["source_split"] == "test"]
    features = read_csv_file(PHASE4 / "complexity_features.csv")
    test_features = [r for r in features if r["source_split"] == "test"]
    check("no test patches in calibration source", len(test_features) == 0)


# =========================================================================
# 5. Expert workflow artifacts
# =========================================================================
def test_expert_workflow():
    print("\n--- 5. Expert workflow ---")
    check("rating template exists",
          (PHASE4 / "expert_rating_template.csv").exists())
    check("rating guidelines exist",
          (PHASE4 / "expert_rating_guidelines.md").exists())
    check("qc protocol exists",
          (PHASE4 / "qc_validation_protocol.md").exists())

    guidelines = (PHASE4 / "expert_rating_guidelines.md").read_text("utf-8")
    check("guidelines mention blinding", "blind" in guidelines.lower())
    check("guidelines mention raters", "rater" in guidelines.lower())


# =========================================================================
# 6. XAI datasets
# =========================================================================
def test_xai_datasets():
    print("\n--- 6. XAI datasets ---")
    candidates = read_csv_file(PHASE4 / "chenab_xai_candidate_manifest.csv")
    check("chenab candidates exist", len(candidates) > 0)

    queue = read_csv_file(PHASE4 / "chenab_xai_annotation_queue.csv")
    check("annotation queue has images", len(queue) > 0)

    queue_classes = set()
    for r in queue:
        for c in r["defect_classes"].split(";"):
            if c.strip():
                queue_classes.add(int(c.strip()))
    check("queue covers 7 classes", len(queue_classes) == 7,
          f"classes={sorted(queue_classes)}")

    aitex_xai = read_csv_file(PHASE4 / "aitex_xai_candidate_manifest.csv")
    check("aitex xai candidates exist", len(aitex_xai) > 0)
    check("aitex marked class_incompatible",
          all(r["class_compatibility"] == "class_incompatible_with_chenab"
              for r in aitex_xai))

    xai_manifest = read_csv_file(PHASE4 / "xai_evaluation_manifest.csv")
    check("xai manifest has entries", len(xai_manifest) > 0)

    all_have_provenance = all(r.get("original_image_id") for r in xai_manifest)
    check("all xai entries have provenance", all_have_provenance)

    chenab_verified = [r for r in xai_manifest
                       if r["source"] == "chenab"
                       and s_bool(r.get("mask_xai_verified", ""))]
    check("no false chenab mask verification", len(chenab_verified) == 0)

    check("mask guidelines exist",
          (PHASE4 / "chenab_mask_annotation_guidelines.md").exists())


# =========================================================================
# 7. AITEX mask QC
# =========================================================================
def test_aitex_mask_qc():
    print("\n--- 7. AITEX mask QC ---")
    check("aitex mask qc summary exists",
          (PHASE4 / "aitex_mask_qc_summary.md").exists())


# =========================================================================
# 8. Safety invariants
# =========================================================================
def test_safety():
    print("\n--- 8. Safety invariants ---")
    raw_phase4 = []
    for d in RAW.rglob("*"):
        if d.is_dir() and "phase4" in str(d).lower():
            raw_phase4.append(str(d))
    check("no phase4 dirs under raw/", len(raw_phase4) == 0)

    check("frozen export unchanged", (EXPORT / "export_manifest.csv").exists())
    check("frozen data.yaml unchanged", (EXPORT / "data.yaml").exists())

    em = read_csv_file(EXPORT / "export_manifest.csv")
    check("export count unchanged", len(em) == 2655, f"got {len(em)}")


# =========================================================================
# 9. Validation results
# =========================================================================
def test_validation_results():
    print("\n--- 9. Validation results ---")
    checks = read_csv_file(PHASE4 / "phase4_dataset_validation.csv")
    failures = [c for c in checks if c["status"] == "FAIL"]
    check("all validation checks pass", len(failures) == 0,
          f"failures={[c['check'] for c in failures]}")

    check("readiness report exists",
          (PHASE4 / "phase4_readiness_report.md").exists())


# =========================================================================
# 10. Results JSON
# =========================================================================
def test_results_json():
    print("\n--- 10. Results JSON ---")
    results = json.loads((PHASE4 / "phase4_results.json").read_text("utf-8"))
    check("reconciliation total 3038", results["reconciliation"]["total"] == 3038)
    check("0 unaccounted", results["reconciliation"]["unaccounted"] == 0)
    check("calibration pending_qc", results["calibration"]["status"] == "pending_qc")
    check("0 validation failures", results["validation"]["failures"] == 0)


# =========================================================================
# 11. Cross-phase consistency
# =========================================================================
def test_cross_phase():
    print("\n--- 11. Cross-phase consistency ---")
    p3_version = json.loads((PHASE3 / "dataset_version.json").read_text("utf-8"))
    check("phase3 version preserved", bool(p3_version.get("version_id")))

    p3_hashes = read_csv_file(PHASE3 / "dataset_hashes.csv")
    check("phase3 hashes preserved", len(p3_hashes) == 2655,
          f"got {len(p3_hashes)}")

    recon = read_csv_file(PHASE4 / "phase4_reconciliation.csv")
    mv12 = read_csv_file(META / "master_manifest_v1_2.csv")
    recon_ids = {r["image_id"] for r in recon}
    mv_ids = {r["image_id"] for r in mv12}
    check("recon covers all v1.2 IDs", recon_ids == mv_ids)


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 64)
    print("Phase 4 Tests")
    print("=" * 64)

    test_reconciliation()
    test_background_patches()
    test_complexity_features()
    test_calibration()
    test_expert_workflow()
    test_xai_datasets()
    test_aitex_mask_qc()
    test_safety()
    test_validation_results()
    test_results_json()
    test_cross_phase()

    print("\n" + "=" * 64)
    total = PASS + FAIL
    print(f"Result: {PASS}/{total} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
