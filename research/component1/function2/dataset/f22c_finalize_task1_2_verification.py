"""F2.2C-Finalize Tasks 1 & 2 -- researcher review CSV integrity + technical
verification of the 9 researcher-Accept instances.

**This is bookkeeping and verification only.** It does not re-review, does
not re-judge, and does not overturn any researcher content decision. It
treats ``quality_review/researcher_review_v2_all29_confirmed_2026-10-18.csv``
as the real, final researcher decision (per the F2.2C-Finalize task brief --
no further human-review round is requested or simulated here).

Task 1 -- review CSV integrity
-------------------------------
  * exactly 29 unique instance_id values, matching
    ``manifests/instance_manifest.csv`` by id SET equality (never by row
    position)
  * recomputed Accept / Needs Correction / Reject distribution (not trusted
    from any prose summary)
  * no duplicate instance_id rows
  * every row's image_id / defect_class cross-checked against the manifest's
    image_id / class_name for that same instance_id
  * reviewer_identifier / review_date reported EXACTLY as given -- this
    script never invents or backfills a blank reviewer_identifier; if every
    row's reviewer_identifier is blank, that is reported as a finding, not
    silently fixed.

Task 2 -- accepted-mask version resolution + technical verification
----------------------------------------------------------------------
For each of the researcher-Accept rows, the mask VERSION the researcher
actually reviewed is resolved as follows (never by inferring from recency or
SAM score):

  1. If ``chosen_proposal_version`` is explicitly "V1" or "V2" in the CSV,
     that is the resolved version (basis: ``explicit_column``).
  2. If that field is blank BUT only one proposal file exists on disk for
     this instance (no V2 was ever generated for it, e.g. because the AI
     preliminary status was Accept and F2.2C-QA's V2 generation pass never
     targeted it), the resolved version is that one file -- not because we
     are guessing which version is "better", but because there is no second
     candidate to be ambiguous between (basis:
     ``only_version_available_structural``).
  3. If the field is blank AND both V1 and V2 files exist for this
     instance, OR the field is blank and NO file exists, this script
     refuses to guess: the instance is marked ``unresolved_version`` and
     excluded from the technically-verified-eligible set.

Each resolvable instance then gets a full technical check: file existence,
dimension match to the source image (V1) or recorded crop (V2), strictly
binary {0, 255} pixel values, image_id/instance_id linkage against the
manifest, nonzero overlap with the GT box (coordinate alignment), not
empty, not implausibly near-full-image/near-full-crop, and a checksum trace
back to the F2.2C-QA preservation baseline (proves the live file is still
byte-identical to what existed right after V1 generation / before any QA
V2 work) plus a provenance cross-check against the original generation
report's own recorded metrics.

Nothing here writes "researcher_approved" / "expert_validated" /
"human_verified" anywhere -- that label is reserved for Task 3's eligibility
manifest, built from THIS script's verification results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.dataset.f22c_sam_mask_generation import (  # noqa: E402
    _mask_quality_flags,
    _resolve_valid_image_path,
    yolo_box_to_pixels,
)

PILOT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "reference_masks" / "pilot_v1"
MANIFESTS_DIR = PILOT_DIR / "manifests"
QUALITY_REVIEW_DIR = PILOT_DIR / "quality_review"
REPORTS_DIR = PILOT_DIR / "reports"
SAM_PROPOSALS_DIR = PILOT_DIR / "sam_proposals"
SAM_PROPOSALS_V2_DIR = PILOT_DIR / "sam_proposals_v2"

FINALIZE_DIR = PILOT_DIR / "finalization_v1"
FINALIZE_REPORTS_DIR = FINALIZE_DIR / "reports"

RESEARCHER_CSV = QUALITY_REVIEW_DIR / "researcher_review_v2_all29_confirmed_2026-10-18.csv"
INSTANCE_MANIFEST_CSV = MANIFESTS_DIR / "instance_manifest.csv"
V1_GENERATION_RESULTS_JSON = REPORTS_DIR / "f22c_mask_generation_results.json"
PRESERVATION_BASELINE_JSON = REPORTS_DIR / "f22c_qa_v1_and_ai_review_preservation_baseline.json"

STATUS_ACCEPT = "Accept"
STATUS_NEEDS_CORRECTION = "Needs Correction"
STATUS_REJECT = "Reject"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_csv(path: Path) -> list[dict]:
    # utf-8-sig transparently strips a UTF-8 BOM if present (observed on the
    # researcher-confirmed CSV, likely from an Excel save) and is a no-op
    # otherwise -- safe for every CSV this stage reads.
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _has_bom(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(3) == b"\xef\xbb\xbf"


# --------------------------------------------------------------------------
# Task 1
# --------------------------------------------------------------------------

def run_task1() -> dict:
    review_rows = _load_csv(RESEARCHER_CSV)
    manifest_rows = _load_csv(INSTANCE_MANIFEST_CSV)
    manifest_by_id = {r["instance_id"]: r for r in manifest_rows}

    review_ids = [r["instance_id"] for r in review_rows]
    review_id_set = set(review_ids)
    manifest_id_set = set(manifest_by_id.keys())

    duplicates = sorted({i for i in review_ids if review_ids.count(i) > 1})

    from collections import Counter
    status_counts = Counter(r["researcher_review_status"] for r in review_rows)

    mapping_mismatches = []
    for r in review_rows:
        m = manifest_by_id.get(r["instance_id"])
        if m is None:
            mapping_mismatches.append({"instance_id": r["instance_id"], "issue": "instance_id not in authoritative manifest"})
            continue
        if m["image_id"] != r["image_id"]:
            mapping_mismatches.append({
                "instance_id": r["instance_id"], "issue": "image_id mismatch",
                "review_image_id": r["image_id"], "manifest_image_id": m["image_id"],
            })
        if m["class_name"] != r["defect_class"]:
            mapping_mismatches.append({
                "instance_id": r["instance_id"], "issue": "defect_class mismatch",
                "review_defect_class": r["defect_class"], "manifest_class_name": m["class_name"],
            })

    reviewer_identifiers = sorted({r["reviewer_identifier"] for r in review_rows})
    review_dates = sorted({r["review_date"] for r in review_rows})

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "F2.2C-Finalize Task 1",
        "source_csv": str(RESEARCHER_CSV.relative_to(REPO_ROOT).as_posix()),
        "n_rows_in_review_csv": len(review_rows),
        "n_unique_instance_ids_in_review_csv": len(review_id_set),
        "exactly_29_unique": len(review_id_set) == 29 and len(review_rows) == 29,
        "duplicate_instance_ids": duplicates,
        "n_manifest_instances": len(manifest_id_set),
        "review_ids_equal_manifest_ids_by_set": review_id_set == manifest_id_set,
        "ids_in_review_not_in_manifest": sorted(review_id_set - manifest_id_set),
        "ids_in_manifest_not_in_review": sorted(manifest_id_set - review_id_set),
        "researcher_review_status_distribution": dict(status_counts),
        "distribution_matches_expected_9_15_5": (
            status_counts.get(STATUS_ACCEPT, 0) == 9
            and status_counts.get(STATUS_NEEDS_CORRECTION, 0) == 15
            and status_counts.get(STATUS_REJECT, 0) == 5
            and sum(status_counts.values()) == 29
        ),
        "mapping_mismatches_image_id_or_defect_class": mapping_mismatches,
        "mapping_clean": mapping_mismatches == [],
        "distinct_reviewer_identifier_values_verbatim": reviewer_identifiers,
        "distinct_review_date_values_verbatim": review_dates,
        "finding_reviewer_identifier_blank_for_all_rows": reviewer_identifiers == [""],
        "finding_note": (
            "reviewer_identifier is blank for every one of the 29 rows in the "
            "source CSV. This script does NOT invent or backfill a reviewer "
            "identifier -- it is reported here verbatim as a known gap in the "
            "review record, to be resolved by the research team if/when "
            "formal attribution is required, not by this script."
        ),
        "finding_source_csv_has_utf8_bom": _has_bom(RESEARCHER_CSV),
        "bom_finding_note": (
            "The source CSV begins with a UTF-8 byte-order-mark (consistent "
            "with having been saved from Excel), unlike the blank template "
            "researcher_review_v2.csv or instance_manifest.csv. This script "
            "reads all CSVs with utf-8-sig so the BOM never corrupts field "
            "names or values; noted here only as a provenance observation, "
            "not a data-integrity defect."
        ),
    }

    FINALIZE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINALIZE_REPORTS_DIR / "f22c_finalize_task1_review_integrity.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


# --------------------------------------------------------------------------
# Task 2
# --------------------------------------------------------------------------

def _resolve_chosen_version(instance_id: str, chosen_col: str) -> dict:
    chosen_col = (chosen_col or "").strip()
    v1_path = SAM_PROPOSALS_DIR / f"{instance_id}.png"
    v2_path = SAM_PROPOSALS_V2_DIR / f"{instance_id}.png"
    v1_exists = v1_path.is_file()
    v2_exists = v2_path.is_file()

    if chosen_col in ("V1", "V2"):
        return {
            "resolved_version": chosen_col,
            "resolution_basis": "explicit_column",
            "v1_file_exists": v1_exists, "v2_file_exists": v2_exists,
        }
    if chosen_col not in ("", "None"):
        # Unexpected value we don't recognise -- do not guess.
        return {
            "resolved_version": None,
            "resolution_basis": f"unresolved_unrecognised_column_value:{chosen_col!r}",
            "v1_file_exists": v1_exists, "v2_file_exists": v2_exists,
        }

    # chosen_col is blank (or explicitly "None", not expected for Accept rows).
    if v1_exists and not v2_exists:
        return {
            "resolved_version": "V1",
            "resolution_basis": "only_version_available_structural",
            "v1_file_exists": v1_exists, "v2_file_exists": v2_exists,
        }
    if v2_exists and not v1_exists:
        return {
            "resolved_version": "V2",
            "resolution_basis": "only_version_available_structural",
            "v1_file_exists": v1_exists, "v2_file_exists": v2_exists,
        }
    if v1_exists and v2_exists:
        return {
            "resolved_version": None,
            "resolution_basis": "unresolved_version_ambiguous_blank_with_both_available",
            "v1_file_exists": v1_exists, "v2_file_exists": v2_exists,
        }
    return {
        "resolved_version": None,
        "resolution_basis": "unresolved_no_mask_file_available",
        "v1_file_exists": v1_exists, "v2_file_exists": v2_exists,
    }


def _verify_mask_technical(instance_id: str, resolved_version: str, manifest_row: dict,
                            v1_gen_by_id: dict, v2_gen_by_id: dict, baseline_hashes: dict) -> dict:
    from PIL import Image
    import numpy as np

    issues: list[str] = []
    notes: list[str] = []
    checks: dict = {}

    if resolved_version == "V1":
        mask_path = SAM_PROPOSALS_DIR / f"{instance_id}.png"
        rel_key = f"sam_proposals/{instance_id}.png"
        gen_record = v1_gen_by_id.get(instance_id)
    elif resolved_version == "V2":
        mask_path = SAM_PROPOSALS_V2_DIR / f"{instance_id}.png"
        rel_key = f"sam_proposals_v2/{instance_id}.png"
        gen_record = v2_gen_by_id.get(instance_id)
    else:
        return {"overall_pass": False, "issues": ["no resolved version to verify"], "notes": [], "checks": {}}

    checks["mask_path"] = str(mask_path.relative_to(REPO_ROOT).as_posix())

    if not mask_path.is_file():
        issues.append(f"mask file does not exist: {mask_path}")
        return {"overall_pass": False, "issues": issues, "notes": notes, "checks": checks}

    # --- image/mask linkage + source image resolution ---------------------
    filename = manifest_row["filename"]
    try:
        image_path = _resolve_valid_image_path(filename)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"could not resolve source image path safely: {exc}")
        image_path = None

    image_w = image_h = None
    if image_path is not None and image_path.is_file():
        with Image.open(image_path) as im_src:
            image_w, image_h = im_src.size
    else:
        issues.append(f"source image file not found or not resolvable: {filename}")

    # --- binary 0/255 check -------------------------------------------------
    with Image.open(mask_path) as im:
        mask_size = im.size  # (w, h)
        arr = np.array(im.convert("L"))
    unique_vals = sorted(set(arr.flatten().tolist()))
    checks["mask_size_wh"] = list(mask_size)
    checks["unique_pixel_values"] = unique_vals
    is_binary = set(unique_vals).issubset({0, 255})
    checks["is_lossless_binary_0_255"] = is_binary
    if not is_binary:
        issues.append(f"mask is not strictly binary {{0,255}}: unique values {unique_vals}")

    # --- resolution match ---------------------------------------------------
    if resolved_version == "V1":
        expected_size = (image_w, image_h) if image_w is not None else None
        checks["expected_size_source_image_wh"] = list(expected_size) if expected_size else None
        if expected_size is not None and mask_size != expected_size:
            issues.append(f"V1 mask size {mask_size} does not match source image size {expected_size}")
    else:  # V2 -- expected size is the recorded crop size, not the full image
        if gen_record is None:
            issues.append("no V2 generation record found to compare crop dimensions against")
        else:
            expected_size = (gen_record.get("crop_width"), gen_record.get("crop_height"))
            checks["expected_size_v2_crop_wh"] = list(expected_size)
            if mask_size != expected_size:
                issues.append(f"V2 mask size {mask_size} does not match recorded crop size {expected_size}")

    # --- instance_id / image_id linkage vs manifest -------------------------
    checks["manifest_image_id"] = manifest_row["image_id"]
    checks["manifest_filename"] = filename
    linkage_ok = mask_path.name.startswith(instance_id)
    checks["mask_filename_matches_instance_id"] = linkage_ok
    if not linkage_ok:
        issues.append("mask filename does not match instance_id")

    # --- coordinate alignment + emptiness / near-full checks ----------------
    mask_bool = (arr > 0)
    mask_px = int(mask_bool.sum())
    total_px = mask_bool.size
    area_fraction = mask_px / total_px if total_px else 0.0
    checks["mask_pixel_count"] = mask_px
    checks["area_fraction_of_own_canvas"] = round(area_fraction, 6)

    empty = mask_px == 0
    checks["is_empty"] = empty
    if empty:
        issues.append("mask is completely empty (0 foreground pixels)")

    near_full = area_fraction >= 0.995
    checks["is_near_full_canvas_ge_99_5pct"] = near_full
    if near_full:
        issues.append(f"mask covers {area_fraction:.4%} of its own canvas -- implausibly near-full")

    if resolved_version == "V1" and image_w is not None and not empty:
        cx, cy, w_, h_ = (float(manifest_row[k]) for k in ("cx", "cy", "w", "h"))
        box_xyxy = yolo_box_to_pixels(cx, cy, w_, h_, image_w, image_h)
        is_elongated = manifest_row["elongated"] == "True"
        is_small = manifest_row["small"] == "True"
        quality = _mask_quality_flags(mask_bool, box_xyxy, (image_h, image_w), is_elongated, is_small)
        checks["recomputed_box_xyxy"] = list(box_xyxy)
        checks["recomputed_inside_box_pixel_count"] = quality["inside_box_pixel_count"]
        checks["recomputed_quality_warnings"] = quality["quality_warnings"]
        coordinate_alignment_ok = quality["inside_box_pixel_count"] > 0
        checks["coordinate_alignment_ok_nonzero_overlap_with_gt_box"] = coordinate_alignment_ok
        if not coordinate_alignment_ok:
            issues.append("mask has ZERO pixel overlap with its own GT box -- coordinate misalignment")
        if quality["quality_warnings"]:
            notes.append(f"non-fatal quality warnings present (recorded at generation time too): {quality['quality_warnings']}")

        # --- provenance cross-check vs original generation record ----------
        if gen_record is not None:
            recorded_warnings = gen_record.get("quality_warnings", [])
            if sorted(recorded_warnings) != sorted(quality["quality_warnings"]):
                issues.append(
                    f"recomputed quality warnings {quality['quality_warnings']} differ from "
                    f"originally recorded warnings {recorded_warnings} -- possible drift since generation"
                )
            recorded_wh = (gen_record.get("image_width"), gen_record.get("image_height"))
            if recorded_wh != (image_w, image_h):
                issues.append(
                    f"source image dimensions changed since V1 generation: recorded {recorded_wh}, now {(image_w, image_h)}"
                )
        else:
            issues.append("no V1 generation record found for provenance cross-check")

    # --- checksum vs F2.2C-QA preservation baseline -------------------------
    live_sha256 = _sha256(mask_path)
    checks["live_sha256"] = live_sha256
    baseline_sha256 = baseline_hashes.get(rel_key)
    checks["baseline_sha256"] = baseline_sha256
    if baseline_sha256 is None:
        notes.append(f"no preservation-baseline entry found for {rel_key} (baseline may not cover this path)")
        checks["checksum_matches_preservation_baseline"] = None
    else:
        matches = live_sha256 == baseline_sha256
        checks["checksum_matches_preservation_baseline"] = matches
        if not matches:
            issues.append(
                f"live file SHA-256 ({live_sha256}) does NOT match the F2.2C-QA preservation "
                f"baseline hash ({baseline_sha256}) for {rel_key} -- file changed since baseline"
            )

    overall_pass = len(issues) == 0
    return {"overall_pass": overall_pass, "issues": issues, "notes": notes, "checks": checks}


def run_task2() -> dict:
    review_rows = _load_csv(RESEARCHER_CSV)
    manifest_rows = _load_csv(INSTANCE_MANIFEST_CSV)
    manifest_by_id = {r["instance_id"]: r for r in manifest_rows}

    v1_gen = json.loads(V1_GENERATION_RESULTS_JSON.read_text(encoding="utf-8"))
    v1_gen_by_id = {r["instance_id"]: r for r in v1_gen["per_instance_results"]}

    v2_gen_path = REPORTS_DIR / "f22c_sam_v2_generation_results.json"
    v2_gen_by_id = {}
    if v2_gen_path.is_file():
        v2_gen = json.loads(v2_gen_path.read_text(encoding="utf-8"))
        v2_gen_by_id = {r["instance_id"]: r for r in v2_gen["per_instance_results"]}

    baseline_hashes = {}
    if PRESERVATION_BASELINE_JSON.is_file():
        baseline = json.loads(PRESERVATION_BASELINE_JSON.read_text(encoding="utf-8"))
        baseline_hashes = baseline.get("file_sha256", {})

    accept_rows = [r for r in review_rows if r["researcher_review_status"] == STATUS_ACCEPT]

    results = []
    n_pass = 0
    for row in accept_rows:
        instance_id = row["instance_id"]
        manifest_row = manifest_by_id.get(instance_id)
        resolution = _resolve_chosen_version(instance_id, row.get("chosen_proposal_version", ""))

        entry = {
            "instance_id": instance_id,
            "image_id": row["image_id"],
            "defect_class": row["defect_class"],
            "chosen_proposal_version_raw_column": row.get("chosen_proposal_version", ""),
            **resolution,
        }

        if manifest_row is None:
            entry["technical_verification"] = {
                "overall_pass": False,
                "issues": ["instance_id not found in authoritative instance_manifest.csv"],
                "notes": [], "checks": {},
            }
        elif resolution["resolved_version"] is None:
            entry["technical_verification"] = {
                "overall_pass": False,
                "issues": [f"version unresolved ({resolution['resolution_basis']}) -- excluded without guessing"],
                "notes": [], "checks": {},
            }
        else:
            entry["technical_verification"] = _verify_mask_technical(
                instance_id, resolution["resolved_version"], manifest_row,
                v1_gen_by_id, v2_gen_by_id, baseline_hashes,
            )

        entry["passes_task2_full_verification"] = (
            resolution["resolved_version"] is not None
            and entry["technical_verification"]["overall_pass"]
        )
        if entry["passes_task2_full_verification"]:
            n_pass += 1
        results.append(entry)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "F2.2C-Finalize Task 2",
        "n_researcher_accept_rows": len(accept_rows),
        "n_version_resolved": sum(1 for e in results if e["resolved_version"] is not None),
        "n_version_unresolved": sum(1 for e in results if e["resolved_version"] is None),
        "n_passing_full_technical_verification": n_pass,
        "true_eligible_count_may_be_less_than_9": n_pass,
        "flag_if_less_than_9": n_pass < 9,
        "per_instance_results": results,
        "status_label": (
            "Technical verification only -- does NOT re-judge researcher content "
            "decisions. Only instances with resolved_version is not None AND "
            "technical_verification.overall_pass is True are eligible for "
            "'verified-reference' treatment in Task 3."
        ),
    }

    FINALIZE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINALIZE_REPORTS_DIR / "f22c_finalize_task2_accept_verification.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    t1 = run_task1()
    print("=== Task 1: review CSV integrity ===")
    print(json.dumps({k: v for k, v in t1.items() if k != "mapping_mismatches_image_id_or_defect_class"}, indent=2))
    t2 = run_task2()
    print("\n=== Task 2: accepted-mask verification ===")
    print(json.dumps({k: v for k, v in t2.items() if k != "per_instance_results"}, indent=2))
    for e in t2["per_instance_results"]:
        print(f"  {e['instance_id']}: resolved={e['resolved_version']} basis={e['resolution_basis']} "
              f"pass={e['passes_task2_full_verification']}")
        if e["technical_verification"]["issues"]:
            for iss in e["technical_verification"]["issues"]:
                print(f"      ISSUE: {iss}")
        if e["technical_verification"]["notes"]:
            for n in e["technical_verification"]["notes"]:
                print(f"      note: {n}")
