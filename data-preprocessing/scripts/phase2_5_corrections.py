"""
Phase 2.5 – Pre-Split Dataset Correction and Verification
Steps 1–8 combined runner.

Safety rules:
- Never modifies raw/ files.
- All outputs go to versioned directories in master_fabric_dataset/.
- Corrections are written as derived metadata only.
"""

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    import imagehash
    HAS_ALL = True
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
RAW_CHENAB = MASTER / "raw" / "chenab" / "fabricDD.v2i.yolov8"
RAW_AITEX = MASTER / "raw" / "aitex"
PHASE1 = MASTER / "reports" / "phase1_audit"
META = MASTER / "metadata"
DERIVED = MASTER / "derived_annotations"
PHASE25 = MASTER / "reports" / "phase2_5"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CHENAB_SPLITS = ("train", "valid", "test")
CHENAB_CLASSES = {0: "baekra", 1: "color issues", 2: "contamination",
                  3: "cut", 4: "gray stitch", 5: "selvet", 6: "stain"}
NUM_CLASSES = 7

# AITEX authoritative class mapping from Silvestre-Blanes et al. 2019 / AFID dataset.
# Numeric codes extracted from AITEX filename pattern NNNN_DDD_II.
# Sources: datasetninja.com/afid (mirror of original AITEX/AFID release),
# AUTEX Research Journal No. 4 2019.
# The 12 defect types listed by authoritative mirrors are:
#   fuzzyball, nep, broken end, broken pick, cut selvage, broken yarn,
#   crease, warp ball, weft curling, knots, contamination, weft crack.
# Numeric-code-to-name mapping is NOT published in the AFID paper or any
# authoritative source found online. Codes must therefore be marked
# pending_expert_review or verified only from the original aitex.es/afid
# download README (not available here).
# Exception: code 000 = no_defect (universally stated in all mirrors).
# The partial mapping below reflects what can be inferred from cross-referencing
# multiple secondary sources with low confidence — none is marked verified_source.
AITEX_CODE_INFO = {
    "000": {
        "name": "no_defect",
        "afid_class": "no_defect",
        "verification_status": "verified_source_definition",
        "source": "Universal across all AITEX/AFID mirrors",
        "chenab_mapping": None,
        "mapping_status": "not_applicable",
    },
    "002": {
        "name": "broken_end",
        "afid_class": "broken_end",
        "verification_status": "pending_expert_review",
        "source": "datasetninja.com/afid secondary mirror; code-number not confirmed in paper",
        "chenab_mapping": None,
        "mapping_status": "incompatible",
    },
    "006": {
        "name": "broken_pick",
        "afid_class": "broken_pick",
        "verification_status": "pending_expert_review",
        "source": "datasetninja.com/afid secondary mirror; code-number not confirmed in paper",
        "chenab_mapping": None,
        "mapping_status": "incompatible",
    },
    "010": {
        "name": "broken_warp_or_weft",
        "afid_class": "unknown_010",
        "verification_status": "pending_expert_review",
        "source": "Code observed in filenames; class name unverified from authoritative source",
        "chenab_mapping": None,
        "mapping_status": "unknown",
    },
    "016": {
        "name": "nep",
        "afid_class": "nep",
        "verification_status": "pending_expert_review",
        "source": "datasetninja.com/afid secondary mirror; code-number not confirmed in paper",
        "chenab_mapping": None,
        "mapping_status": "incompatible",
    },
    "019": {
        "name": "contamination_or_stain",
        "afid_class": "contamination",
        "verification_status": "pending_expert_review",
        "source": "datasetninja.com/afid lists 'contamination'; partial overlap with Chenab stain(6) unverified",
        "chenab_mapping": 6,
        "mapping_status": "rejected_mapping",
        "mapping_rejection_reason": (
            "Previously proposed as stain→Chenab(6). Rejected: AFID 'contamination' "
            "is a broader category that includes oil, chemical, and mechanical contamination, "
            "not exclusively surface stains. Requires expert review before training use."
        ),
    },
    "022": {
        "name": "weft_related",
        "afid_class": "unknown_022",
        "verification_status": "pending_expert_review",
        "source": "Code observed in filenames; class name unverified from authoritative source",
        "chenab_mapping": None,
        "mapping_status": "unknown",
    },
    "023": {
        "name": "unknown_023",
        "afid_class": "unknown_023",
        "verification_status": "pending_expert_review",
        "source": "Code observed in dataset filenames but not listed in any known AFID documentation",
        "chenab_mapping": None,
        "mapping_status": "unknown",
    },
    "025": {
        "name": "weft_crack_or_related",
        "afid_class": "unknown_025",
        "verification_status": "pending_expert_review",
        "source": "Code observed in filenames; possible weft_crack but unverified",
        "chenab_mapping": None,
        "mapping_status": "unknown",
    },
    "027": {
        "name": "warp_crack_or_related",
        "afid_class": "unknown_027",
        "verification_status": "pending_expert_review",
        "source": "Code observed in filenames; possible warp defect but unverified",
        "chenab_mapping": None,
        "mapping_status": "unknown",
    },
    "029": {
        "name": "fuzzyball_or_knot",
        "afid_class": "unknown_029",
        "verification_status": "pending_expert_review",
        "source": "Code observed in filenames; possible fuzzyball but unverified",
        "chenab_mapping": None,
        "mapping_status": "unknown",
    },
    "030": {
        "name": "cut_selvage_or_cut",
        "afid_class": "unknown_030",
        "verification_status": "pending_expert_review",
        "source": (
            "Previously proposed as cut→Chenab(3). Rejected: AFID 'cut selvage' "
            "refers to a fabric edge defect, semantically different from Chenab 'cut' "
            "which is an interior fabric cut. Requires expert review."
        ),
        "chenab_mapping": 3,
        "mapping_status": "rejected_mapping",
        "mapping_rejection_reason": (
            "AFID cut selvage is an edge defect; Chenab cut(3) is an interior defect. "
            "Different defect semantics — mapping rejected pending expert review."
        ),
    },
    "036": {
        "name": "crease",
        "afid_class": "crease",
        "verification_status": "pending_expert_review",
        "source": "datasetninja.com/afid secondary mirror; code-number not confirmed in paper",
        "chenab_mapping": None,
        "mapping_status": "incompatible",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path.name} ({len(rows)} rows)")


def write_md(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {path.name}")


def make_contact_sheet(images_paths_titles: list, out_path: Path,
                       cols: int = 4, thumb_size: int = 256,
                       title: str = ""):
    """Create a PNG contact sheet from (path, title) pairs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images_paths_titles)
    if n == 0:
        return
    rows = math.ceil(n / cols)
    label_h = 24
    W = cols * thumb_size
    H = rows * (thumb_size + label_h) + (40 if title else 0)
    sheet = Image.new("RGB", (W, H), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((4, 4), title[:100], fill=(220, 220, 220))
    for idx, (img_path, lbl) in enumerate(images_paths_titles):
        col = idx % cols
        row = idx // cols
        y_off = (40 if title else 0) + row * (thumb_size + label_h)
        x_off = col * thumb_size
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((thumb_size, thumb_size))
            # centre in cell
            px = x_off + (thumb_size - img.width) // 2
            py = y_off + (thumb_size - img.height) // 2
            sheet.paste(img, (px, py))
        except Exception as e:
            draw.rectangle([x_off, y_off, x_off + thumb_size, y_off + thumb_size],
                           outline=(200, 60, 60))
            draw.text((x_off + 4, y_off + 4), f"ERR:{str(e)[:30]}", fill=(255, 80, 80))
        draw.text((x_off + 2, y_off + thumb_size + 2), str(lbl)[:40], fill=(200, 200, 200))
    sheet.save(str(out_path))


def make_overlay_sheet(pairs: list, out_path: Path, thumb_size: int = 256, cols: int = 4, title: str = ""):
    """pairs: list of (image_path, mask_path, label). Creates side-by-side image+mask pairs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(pairs)
    if n == 0:
        return
    cell_w = thumb_size * 2
    rows_n = math.ceil(n / cols)
    label_h = 20
    W = cols * cell_w
    H = rows_n * (thumb_size + label_h) + (40 if title else 0)
    sheet = Image.new("RGB", (W, H), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((4, 4), title[:100], fill=(220, 220, 220))
    for idx, (img_p, mask_p, lbl) in enumerate(pairs):
        col = idx % cols
        row = idx // cols
        y_off = (40 if title else 0) + row * (thumb_size + label_h)
        x_off = col * cell_w
        for side, fp in enumerate([img_p, mask_p]):
            px = x_off + side * thumb_size
            try:
                img = Image.open(fp).convert("RGB")
                img.thumbnail((thumb_size, thumb_size))
                sheet.paste(img, (px, y_off))
            except Exception as e:
                draw.rectangle([px, y_off, px + thumb_size, y_off + thumb_size],
                               outline=(180, 60, 60))
                draw.text((px + 2, y_off + 2), str(e)[:25], fill=(255, 80, 80))
        draw.text((x_off + 2, y_off + thumb_size + 2), str(lbl)[:60], fill=(200, 200, 200))
    sheet.save(str(out_path))

# ---------------------------------------------------------------------------
# STEP 1 – Verify existing reports
# ---------------------------------------------------------------------------

def step1_verify():
    print("=== STEP 1 – Verify Existing Reports ===")
    issues = []

    # Re-count Chenab images
    chenab_count = 0
    for s in CHENAB_SPLITS:
        d = RAW_CHENAB / s / "images"
        if d.is_dir():
            chenab_count += sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    # Re-count AITEX images
    aitex_defect = sum(1 for p in (RAW_AITEX / "Defect_images").rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    aitex_nodefect = sum(1 for p in (RAW_AITEX / "NODefect_images").rglob("*")
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    aitex_total = aitex_defect + aitex_nodefect

    print(f"  Chenab: {chenab_count} images (expected 2791)")
    print(f"  AITEX defect: {aitex_defect}, no-defect: {aitex_nodefect}, total: {aitex_total} (expected 247)")

    if chenab_count != 2791:
        issues.append(f"Chenab count mismatch: found {chenab_count}, expected 2791")
    if aitex_total != 247:
        issues.append(f"AITEX total mismatch: found {aitex_total}, expected 247")

    # Check NaN/Infinity in annotation files
    nan_inf_count = 0
    for s in CHENAB_SPLITS:
        lbl_dir = RAW_CHENAB / s / "labels"
        if not lbl_dir.is_dir():
            continue
        for lbl in lbl_dir.rglob("*.txt"):
            for line in lbl.read_text(errors="replace").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    for v in parts[1:]:
                        try:
                            f = float(v)
                            if not math.isfinite(f):
                                nan_inf_count += 1
                                issues.append(f"NaN/Inf in {lbl.name}: {line[:80]}")
                        except ValueError:
                            pass
    print(f"  NaN/Inf annotation values: {nan_inf_count}")

    # Check intentionally empty label files vs missing
    empty_label_count = 0
    for s in ("train", "valid"):
        lbl_dir = RAW_CHENAB / s / "labels"
        if lbl_dir.is_dir():
            for lbl in lbl_dir.rglob("*.txt"):
                if lbl.stat().st_size == 0:
                    empty_label_count += 1
    print(f"  Intentionally empty label files (no annotations): {empty_label_count}")

    # Verify source metadata hashes (immutability check)
    meta_path = PHASE1 / "source_metadata.csv"
    hash_mismatches = []
    if meta_path.is_file():
        for r in read_csv(meta_path):
            fp = MASTER / "raw" / r["relative_path"]
            if fp.is_file():
                current = sha256_file(fp)
                if current != r["sha256"]:
                    hash_mismatches.append(r["relative_path"])
    print(f"  Source file hash mismatches: {len(hash_mismatches)}")
    if hash_mismatches:
        issues.append(f"Hash mismatches in raw files: {hash_mismatches}")

    # Report
    PHASE25.mkdir(parents=True, exist_ok=True)
    verify_lines = [
        "# Step 1 – Verification Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Chenab images: {chenab_count}",
        f"- AITEX images: {aitex_total} (defect={aitex_defect}, no-defect={aitex_nodefect})",
        f"- NaN/Inf annotation values: {nan_inf_count}",
        f"- Empty (no-annotation) label files: {empty_label_count}",
        f"- Source file hash mismatches: {len(hash_mismatches)}",
        "",
        "## Issues Found",
    ]
    if issues:
        for iss in issues:
            verify_lines.append(f"- {iss}")
    else:
        verify_lines.append("- None")

    write_md(PHASE25 / "step1_verification.md", verify_lines)
    print(f"  Issues found: {len(issues)}")
    return {"chenab_count": chenab_count, "aitex_total": aitex_total,
            "nan_inf": nan_inf_count, "hash_mismatches": len(hash_mismatches),
            "issues": issues}

# ---------------------------------------------------------------------------
# STEP 2 – AITEX Class Code Verification
# ---------------------------------------------------------------------------

def step2_aitex_class_verification():
    print("=== STEP 2 – AITEX Class Code Verification ===")
    PHASE25.mkdir(parents=True, exist_ok=True)
    contact_dir = PHASE25 / "aitex_class_review_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    defect_dir = RAW_AITEX / "Defect_images"
    # Collect one representative per code
    code_to_images = defaultdict(list)
    for p in sorted(defect_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            parts = p.stem.split("_")
            code = parts[1] if len(parts) >= 2 else "unknown"
            code_to_images[code].append(p)

    observed_codes = sorted(code_to_images.keys())
    print(f"  Observed AITEX codes: {observed_codes}")

    # Build verification CSV
    verification_rows = []
    for code in observed_codes:
        info = AITEX_CODE_INFO.get(code, {
            "name": f"unknown_{code}",
            "afid_class": f"unknown_{code}",
            "verification_status": "pending_expert_review",
            "source": "Not in AITEX_CODE_INFO table",
            "chenab_mapping": None,
            "mapping_status": "unknown",
        })
        imgs = code_to_images[code]
        verification_rows.append({
            "aitex_code": code,
            "assigned_name": info["name"],
            "afid_class": info.get("afid_class", ""),
            "verification_status": info["verification_status"],
            "source_reference": info.get("source", ""),
            "image_count": len(imgs),
            "sample_images": "|".join(p.name for p in imgs[:3]),
        })

    write_csv(PHASE25 / "aitex_code_verification.csv", verification_rows,
              ["aitex_code", "assigned_name", "afid_class", "verification_status",
               "source_reference", "image_count", "sample_images"])

    # Build class mapping decisions CSV
    mapping_rows = []
    for code in observed_codes:
        info = AITEX_CODE_INFO.get(code, {})
        rejected_reason = info.get("mapping_rejection_reason", "")
        mapping_rows.append({
            "aitex_code": code,
            "aitex_name": info.get("name", f"unknown_{code}"),
            "proposed_chenab_class_id": info.get("chenab_mapping", ""),
            "mapping_status": info.get("mapping_status", "unknown"),
            "rejection_reason": rejected_reason,
            "verification_status": info.get("verification_status", "pending_expert_review"),
            "approved_for_training": "no",
            "human_decision": "pending_review",
            "reviewer": "",
            "review_date": "",
            "notes": "",
        })
    write_csv(PHASE25 / "aitex_class_mapping_decisions.csv", mapping_rows,
              ["aitex_code", "aitex_name", "proposed_chenab_class_id", "mapping_status",
               "rejection_reason", "verification_status", "approved_for_training",
               "human_decision", "reviewer", "review_date", "notes"])

    # Contact sheets: one per code (up to 8 images)
    for code in observed_codes:
        imgs = code_to_images[code][:8]
        info = AITEX_CODE_INFO.get(code, {})
        pairs = [(p, f"{code} | {info.get('name', '?')} | {p.name}") for p in imgs]
        make_contact_sheet(pairs, contact_dir / f"code_{code}.png",
                           cols=4, thumb_size=200,
                           title=f"AITEX Code {code}: {info.get('name','unknown')} (status: {info.get('verification_status','?')})")

    print(f"  Codes verified/reviewed: {len(observed_codes)}")
    print(f"  Contact sheets: {len(observed_codes)}")
    return {"observed_codes": observed_codes, "mapping_rows": mapping_rows}

# ---------------------------------------------------------------------------
# STEP 3 – Chenab Annotation Issue Review
# ---------------------------------------------------------------------------

def step3_chenab_annotation_review():
    print("=== STEP 3 – Chenab Annotation Review ===")
    contact_dir = PHASE25 / "chenab_annotation_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    issues_rows = read_csv(PHASE1 / "chenab_annotation_issues.csv")

    review_rows = []
    correction_rows = []

    for r in issues_rows:
        split = r["split"]
        img_file = r["image_file"]
        raw_line = r["raw_line"]
        issues = r["issues"].split("|")
        img_path = RAW_CHENAB / split / "images" / img_file

        # Parse coords
        parts = raw_line.strip().split()
        record_id = f"{split}_{img_file}_{r['line_num']}"

        proposed_correction = ""
        correction_type = ""
        human_review_status = "needs_human_review"
        auto_correctable = False
        correction_note = ""

        if len(parts) == 5:
            try:
                cid = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                w = h = cx = cy = 0.0
                cid = -1

            # Whole-image box (0.5 0.5 1 1): technically valid YOLO annotation for whole-fabric defect
            if abs(cx - 0.5) < 0.001 and abs(cy - 0.5) < 0.001 and abs(w - 1.0) < 0.001 and abs(h - 1.0) < 0.001:
                human_review_status = "needs_human_review"
                correction_note = (
                    "Whole-image bounding box (0.5 0.5 1 1). Technically valid YOLO format. "
                    "May represent a whole-fabric defect annotation. "
                    "Requires human expert to confirm whether defect truly spans full image."
                )
                correction_type = "whole_image_box_review"

            # Zero-width box: mechanically invalid — width must be > 0
            if "zero_or_neg_width" in "|".join(issues) or w <= 0.0:
                # This is a deterministic technical error: zero width is invalid YOLO
                # Proposed correction: exclude this annotation from training
                proposed_correction = "exclude_annotation"
                correction_type = "invalid_zero_width"
                auto_correctable = True
                human_review_status = "auto_excluded_invalid"
                correction_note = "Width=0 is an invalid YOLO bounding box. Exclude from training labels."

            # Suspiciously small box (non-zero but very tiny)
            if "suspiciously_small_box" in "|".join(issues) and w > 0:
                if not auto_correctable:
                    proposed_correction = "flag_for_review"
                    correction_type = "suspiciously_small"
                    human_review_status = "needs_human_review"
                    correction_note = f"Box {w:.4f}x{h:.4f} is very small. May be annotation noise or a real tiny defect."

        review_rows.append({
            "record_id": record_id,
            "split": split,
            "image_file": img_file,
            "line_num": r["line_num"],
            "raw_annotation": raw_line,
            "issue_types": r["issues"],
            "correction_type": correction_type,
            "proposed_correction": proposed_correction,
            "auto_correctable": auto_correctable,
            "human_review_status": human_review_status,
            "correction_note": correction_note,
            "reviewer": "",
            "review_date": "",
            "final_decision": "pending_review" if not auto_correctable else "auto_excluded_invalid",
        })

        if auto_correctable:
            correction_rows.append({
                "record_id": record_id,
                "split": split,
                "image_file": img_file,
                "line_num": int(r["line_num"]),
                "raw_annotation": raw_line,
                "correction_applied": "exclude_annotation",
                "justification": correction_note,
                "derived_label_path": "",  # populated when derived labels written
            })

    # Create contact sheets grouped by issue type
    # Group 1: whole-image boxes
    whole_img = [r for r in review_rows if r["correction_type"] == "whole_image_box_review"]
    if whole_img:
        pairs = []
        for r in whole_img[:12]:
            img_p = RAW_CHENAB / r["split"] / "images" / r["image_file"]
            if img_p.is_file():
                pairs.append((img_p, f"{r['image_file'][:35]}"))
        if pairs:
            make_contact_sheet(pairs, contact_dir / "whole_image_boxes.png",
                               cols=4, thumb_size=200,
                               title="Whole-image boxes (cx=0.5,cy=0.5,w=1,h=1) — needs expert review")

    # Group 2: small/invalid boxes
    small_boxes = [r for r in review_rows if "small" in r["correction_type"] or "zero" in r["correction_type"]]
    if small_boxes:
        pairs = []
        for r in small_boxes[:12]:
            img_p = RAW_CHENAB / r["split"] / "images" / r["image_file"]
            if img_p.is_file():
                pairs.append((img_p, f"{r['image_file'][:35]} | {r['raw_annotation'][:30]}"))
        if pairs:
            make_contact_sheet(pairs, contact_dir / "small_invalid_boxes.png",
                               cols=4, thumb_size=200,
                               title="Small/invalid boxes — auto-excluded or needs review")

    # Write derived corrected labels (exclude zero-width annotations)
    derived_label_dir = DERIVED / "chenab_corrected_labels"
    derived_label_dir.mkdir(parents=True, exist_ok=True)
    # Map: (split, image_file) -> set of line_nums to exclude
    excludes = defaultdict(set)
    for r in correction_rows:
        excludes[(r["split"], r["image_file"])].add(int(r["line_num"]))

    written_corrections = 0
    for (split, img_file), excluded_lines in excludes.items():
        lbl_path = RAW_CHENAB / split / "labels" / Path(img_file).with_suffix(".txt").name
        if not lbl_path.is_file():
            continue
        orig_lines = [l for l in lbl_path.read_text(errors="replace").splitlines() if l.strip()]
        corrected = [l for i, l in enumerate(orig_lines, 1) if i not in excluded_lines]
        out_path = derived_label_dir / split / Path(img_file).with_suffix(".txt").name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(corrected), encoding="utf-8")
        written_corrections += 1
        # Update path in correction_rows
        for r in correction_rows:
            if r["split"] == split and r["image_file"] == img_file:
                r["derived_label_path"] = str(out_path.relative_to(MASTER))

    write_csv(PHASE25 / "chenab_annotation_review.csv", review_rows,
              ["record_id", "split", "image_file", "line_num", "raw_annotation",
               "issue_types", "correction_type", "proposed_correction", "auto_correctable",
               "human_review_status", "correction_note", "reviewer", "review_date", "final_decision"])
    write_csv(PHASE25 / "chenab_annotation_corrections_proposed.csv", correction_rows,
              ["record_id", "split", "image_file", "line_num", "raw_annotation",
               "correction_applied", "justification", "derived_label_path"])

    auto_fix = sum(1 for r in review_rows if r["auto_correctable"])
    needs_review = sum(1 for r in review_rows if r["human_review_status"] == "needs_human_review")
    print(f"  Total annotation issues: {len(review_rows)}")
    print(f"  Auto-correctable (zero-width): {auto_fix}")
    print(f"  Needs human review (whole-image boxes): {needs_review}")
    print(f"  Derived corrected label files written: {written_corrections}")
    return {"total": len(review_rows), "auto_fix": auto_fix, "needs_review": needs_review}

# ---------------------------------------------------------------------------
# STEP 4 – AITEX Mask QC (improved pairing)
# ---------------------------------------------------------------------------

def step4_aitex_mask_qc():
    print("=== STEP 4 – AITEX Mask QC ===")
    contact_dir = PHASE25 / "aitex_mask_review_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    defect_dir = RAW_AITEX / "Defect_images"
    mask_dir = RAW_AITEX / "Mask_images"

    # Build comprehensive mask lookup: stem -> list of mask paths (handle mask1, mask2, etc.)
    mask_lookup = defaultdict(list)
    for mp in sorted(mask_dir.rglob("*")):
        if not mp.is_file() or mp.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = mp.stem
        # Handle patterns: stem_mask, stem_mask1, stem_mask2, ...
        if "_mask" in stem:
            base = stem[:stem.rfind("_mask")]
            # suffix after _mask (empty, 1, 2, ...)
            suffix = stem[stem.rfind("_mask") + 5:]
            mask_lookup[base].append(mp)
        else:
            mask_lookup[stem].append(mp)

    defect_images = sorted(p for p in defect_dir.rglob("*")
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    qc_rows = []
    correction_rows = []
    verified_candidates = []

    for img_path in defect_images:
        stem = img_path.stem
        masks_for_img = mask_lookup.get(stem, [])
        n_masks = len(masks_for_img)

        record = {
            "image_filename": img_path.name,
            "image_stem": stem,
            "n_mask_candidates": n_masks,
            "mask_filenames": "|".join(m.name for m in masks_for_img),
            "pairing_status": "",
            "image_width": None,
            "image_height": None,
            "mask_width": None,
            "mask_height": None,
            "dimension_match": "",
            "foreground_pixels": "",
            "empty_mask": "",
            "n_connected_components": "",
            "mask_pixel_values": "",
            "qc_status": "",
            "issues": "",
            "review_reason": "",
        }

        if n_masks == 0:
            record["pairing_status"] = "no_mask"
            record["qc_status"] = "failed_no_mask"
            record["issues"] = "no_mask_found"
            record["review_reason"] = "No mask file found for this defect image"
            qc_rows.append(record)
            continue

        # Read image dimensions
        try:
            img = Image.open(img_path)
            img_w, img_h = img.width, img.height
            record["image_width"] = img_w
            record["image_height"] = img_h
        except Exception as e:
            record["qc_status"] = "failed_image_unreadable"
            record["issues"] = f"image_read_error:{e}"
            qc_rows.append(record)
            continue

        if n_masks > 1:
            record["pairing_status"] = "multiple_masks"
            record["review_reason"] = f"Multiple masks found: {record['mask_filenames']}"

        # Use first mask as primary; report all
        primary_mask = masks_for_img[0]
        issues = []

        try:
            mask_img = Image.open(primary_mask).convert("L")
            mask_w, mask_h = mask_img.width, mask_img.height
            record["mask_width"] = mask_w
            record["mask_height"] = mask_h

            if mask_w == img_w and mask_h == img_h:
                record["dimension_match"] = "match"
            else:
                record["dimension_match"] = f"mismatch:img={img_w}x{img_h},mask={mask_w}x{mask_h}"
                issues.append(f"dimension_mismatch:{img_w}x{img_h}!={mask_w}x{mask_h}")

            arr = np.array(mask_img)
            uniq = np.unique(arr)
            record["mask_pixel_values"] = str(uniq[:20].tolist())
            fg_pixels = int(np.sum(arr > 127))
            record["foreground_pixels"] = fg_pixels
            record["empty_mask"] = fg_pixels == 0

            if fg_pixels == 0:
                issues.append("empty_mask")

            # Check for intermediate grayscale (not binary)
            non_binary = [v for v in uniq.tolist() if v not in (0, 255) and not (0 <= v <= 10) and not (245 <= v <= 255)]
            if non_binary:
                issues.append(f"intermediate_grayscale_values:{non_binary[:5]}")

            # Connected components
            binary = (arr > 127).astype(np.uint8)
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
            n_components = max(0, n_labels - 1)  # exclude background
            record["n_connected_components"] = n_components

            # Tiny artifacts (< 50 pixels)
            tiny = sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] < 50)
            if tiny > 0:
                issues.append(f"tiny_components:{tiny}")

        except Exception as e:
            issues.append(f"mask_read_error:{e}")

        if n_masks == 1 and not issues:
            record["pairing_status"] = "single_match"
            record["qc_status"] = "passed"
        elif n_masks == 1 and issues:
            record["pairing_status"] = "single_match_with_issues"
            record["qc_status"] = "passed_with_issues"
        elif n_masks > 1:
            record["qc_status"] = "multiple_masks_needs_review"
        record["issues"] = "|".join(issues) if issues else "ok"

        qc_rows.append(record)

        # Verified candidates: single match, dimensions match, non-empty, no errors
        if (n_masks == 1 and record["dimension_match"] == "match"
                and not record.get("empty_mask") and "mask_read_error" not in record["issues"]
                and "dimension_mismatch" not in record["issues"]):
            verified_candidates.append({
                "image_filename": img_path.name,
                "mask_filename": primary_mask.name,
                "image_stem": stem,
                "n_connected_components": record["n_connected_components"],
                "foreground_pixels": record["foreground_pixels"],
                "mask_qc_status": "mechanically_valid",
                "human_verified": "pending",
                "notes": "",
            })

    # Pairing corrections: images with multiple masks need human choice
    for r in qc_rows:
        if r["pairing_status"] in ("multiple_masks", "multiple_masks_needs_review"):
            correction_rows.append({
                "image_filename": r["image_filename"],
                "image_stem": r["image_stem"],
                "mask_candidates": r["mask_filenames"],
                "issue": "multiple_mask_files",
                "proposed_action": "human_selects_primary_mask",
                "chosen_mask": "",
                "reviewer": "",
                "review_date": "",
                "notes": "",
            })
        elif r["pairing_status"] == "no_mask":
            correction_rows.append({
                "image_filename": r["image_filename"],
                "image_stem": r["image_stem"],
                "mask_candidates": "",
                "issue": "no_mask_found",
                "proposed_action": "exclude_from_xai_ground_truth",
                "chosen_mask": "",
                "reviewer": "",
                "review_date": "",
                "notes": "",
            })

    write_csv(PHASE25 / "aitex_mask_qc.csv", qc_rows,
              ["image_filename", "image_stem", "n_mask_candidates", "mask_filenames",
               "pairing_status", "image_width", "image_height", "mask_width", "mask_height",
               "dimension_match", "foreground_pixels", "empty_mask", "n_connected_components",
               "mask_pixel_values", "qc_status", "issues", "review_reason"])
    write_csv(PHASE25 / "aitex_mask_pairing_corrections.csv", correction_rows,
              ["image_filename", "image_stem", "mask_candidates", "issue",
               "proposed_action", "chosen_mask", "reviewer", "review_date", "notes"])
    write_csv(PHASE25 / "verified_mask_candidates.csv", verified_candidates,
              ["image_filename", "mask_filename", "image_stem", "n_connected_components",
               "foreground_pixels", "mask_qc_status", "human_verified", "notes"])

    # Contact sheets
    # 1. Missing mask images
    missing = [r for r in qc_rows if r["pairing_status"] == "no_mask"]
    if missing:
        pairs = [(defect_dir / r["image_filename"], r["image_filename"]) for r in missing
                 if (defect_dir / r["image_filename"]).is_file()]
        if pairs:
            make_contact_sheet(pairs, contact_dir / "missing_masks.png",
                               cols=3, thumb_size=250, title="Images with no mask")

    # 2. Multiple mask images
    multi = [r for r in qc_rows if r["n_mask_candidates"] > 1]
    if multi:
        pairs = []
        for r in multi:
            img_p = defect_dir / r["image_filename"]
            if img_p.is_file():
                pairs.append((img_p, f"{r['image_filename']} → {r['mask_filenames'][:50]}"))
        make_contact_sheet(pairs, contact_dir / "multiple_masks.png",
                           cols=3, thumb_size=250, title="Images with multiple mask candidates")

    # 3. Overlay contact sheet for valid pairs (sample of 16)
    valid = [r for r in qc_rows if r["qc_status"] == "passed"][:16]
    if valid:
        overlay_pairs = []
        for r in valid:
            img_p = defect_dir / r["image_filename"]
            msk_files = r["mask_filenames"].split("|")
            msk_p = mask_dir / msk_files[0] if msk_files else None
            if img_p.is_file() and msk_p and msk_p.is_file():
                overlay_pairs.append((img_p, msk_p, r["image_filename"][:30]))
        if overlay_pairs:
            make_overlay_sheet(overlay_pairs, contact_dir / "valid_mask_overlays.png",
                               cols=4, thumb_size=200, title="Valid image-mask pairs (sample)")

    total = len(qc_rows)
    passed = sum(1 for r in qc_rows if r["qc_status"] == "passed")
    multi_count = len(multi)
    missing_count = len(missing)
    empty = sum(1 for r in qc_rows if r.get("empty_mask") == True)
    print(f"  Total defect images: {total}")
    print(f"  Passed QC (single match, dim OK, non-empty): {passed}")
    print(f"  Multiple masks (needs review): {multi_count}")
    print(f"  Missing masks: {missing_count}")
    print(f"  Empty masks: {empty}")
    print(f"  Verified mask candidates: {len(verified_candidates)}")
    return {"total": total, "passed": passed, "multi": multi_count,
            "missing": missing_count, "empty": empty, "verified": len(verified_candidates)}

# ---------------------------------------------------------------------------
# STEP 5 – Fix Mask-to-BBox Derivation
# ---------------------------------------------------------------------------

def step5_fix_bbox_derivation():
    print("=== STEP 5 – Fix Mask-to-BBox Derivation ===")
    defect_dir = RAW_AITEX / "Defect_images"
    mask_dir = RAW_AITEX / "Mask_images"
    NOISE_AREA_THRESHOLD = 50  # pixels

    # Rebuild mask lookup with improved multi-mask handling
    mask_lookup = defaultdict(list)
    for mp in sorted(mask_dir.rglob("*")):
        if mp.is_file() and mp.suffix.lower() in IMAGE_EXTS:
            stem = mp.stem
            if "_mask" in stem:
                base = stem[:stem.rfind("_mask")]
                mask_lookup[base].append(mp)

    geometry_rows = []
    box_validation_rows = []
    geometry_dir = DERIVED / "aitex_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(defect_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        stem = img_path.stem
        parts = stem.split("_")
        defect_code = parts[1] if len(parts) >= 2 else "unknown"
        masks = mask_lookup.get(stem, [])

        # Determine class approval status
        info = AITEX_CODE_INFO.get(defect_code, {})
        mapping_status = info.get("mapping_status", "unknown")
        approved_class_id = None  # Never auto-approved

        # Read image dimensions
        try:
            img = Image.open(img_path)
            img_w, img_h = img.width, img.height
        except Exception as e:
            geometry_rows.append({
                "image_filename": img_path.name,
                "defect_code": defect_code,
                "mask_filename": "",
                "annotation_status": "failed_image_unreadable",
                "box_count": 0,
                "geometry_file": "",
                "note": str(e)[:100],
            })
            continue

        if not masks:
            geometry_rows.append({
                "image_filename": img_path.name,
                "defect_code": defect_code,
                "mask_filename": "",
                "annotation_status": "no_mask",
                "box_count": 0,
                "geometry_file": "",
                "note": "No mask available — excluded from geometry",
            })
            continue

        # Use first mask (flag multi-mask as pending review)
        primary_mask = masks[0]
        multi_mask_flag = len(masks) > 1

        try:
            mask_img = Image.open(primary_mask).convert("L")
            mask_w, mask_h = mask_img.width, mask_img.height
        except Exception as e:
            geometry_rows.append({
                "image_filename": img_path.name,
                "defect_code": defect_code,
                "mask_filename": primary_mask.name,
                "annotation_status": "failed_mask_unreadable",
                "box_count": 0,
                "geometry_file": "",
                "note": str(e)[:100],
            })
            continue

        # Dimension check — must match exactly
        if mask_w != img_w or mask_h != img_h:
            geometry_rows.append({
                "image_filename": img_path.name,
                "defect_code": defect_code,
                "mask_filename": primary_mask.name,
                "annotation_status": "failed_dimension_mismatch",
                "box_count": 0,
                "geometry_file": "",
                "note": f"img={img_w}x{img_h}, mask={mask_w}x{mask_h}",
            })
            continue

        arr = np.array(mask_img)
        binary = (arr > 127).astype(np.uint8)
        n_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(binary)

        boxes = []
        discarded_tiny = []
        for comp_id in range(1, n_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < NOISE_AREA_THRESHOLD:
                discarded_tiny.append({"comp_id": comp_id, "area": area, "reason": "below_noise_threshold"})
                continue
            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
            h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            # Clamp and verify
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(1e-6, min(1.0, nw))
            nh = max(1e-6, min(1.0, nh))
            # NaN/Inf check
            for val in [cx, cy, nw, nh]:
                if not math.isfinite(val):
                    discarded_tiny.append({"comp_id": comp_id, "area": area, "reason": "non_finite_coordinate"})
                    continue
            valid_box = (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1)
            if valid_box:
                boxes.append((cx, cy, nw, nh, area))

            box_validation_rows.append({
                "image_filename": img_path.name,
                "comp_id": comp_id,
                "area_pixels": area,
                "cx": f"{cx:.6f}",
                "cy": f"{cy:.6f}",
                "nw": f"{nw:.6f}",
                "nh": f"{nh:.6f}",
                "valid": valid_box and area >= NOISE_AREA_THRESHOLD,
                "discard_reason": "below_noise_threshold" if area < NOISE_AREA_THRESHOLD else ("invalid_coord" if not valid_box else ""),
            })

        # Determine annotation status
        if multi_mask_flag:
            ann_status = "geometry_only_pending_mask_review"
        elif len(boxes) == 0 and n_labels <= 1:
            ann_status = "empty_mask_no_geometry"
        else:
            ann_status = "geometry_only_pending_class_approval"

        # Write geometry file — always geometry-only (class=? replaced by UNVERIFIED marker in comment)
        geo_file = geometry_dir / f"{stem}.txt"
        with open(geo_file, "w", encoding="utf-8") as f:
            f.write(f"# GEOMETRY_ONLY source={primary_mask.name} code={defect_code} "
                    f"status={ann_status} tiny_discarded={len(discarded_tiny)}\n")
            for cx, cy, nw, nh, area in boxes:
                f.write(f"UNVERIFIED {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} # area={area}\n")

        geometry_rows.append({
            "image_filename": img_path.name,
            "defect_code": defect_code,
            "mask_filename": primary_mask.name,
            "annotation_status": ann_status,
            "box_count": len(boxes),
            "geometry_file": str(geo_file.relative_to(MASTER)),
            "note": f"tiny_discarded={len(discarded_tiny)}; multi_mask={multi_mask_flag}",
        })

    write_csv(PHASE25 / "aitex_geometry_manifest.csv", geometry_rows,
              ["image_filename", "defect_code", "mask_filename", "annotation_status",
               "box_count", "geometry_file", "note"])
    write_csv(PHASE25 / "aitex_box_validation.csv", box_validation_rows,
              ["image_filename", "comp_id", "area_pixels", "cx", "cy", "nw", "nh",
               "valid", "discard_reason"])

    total_boxes = sum(r["box_count"] for r in geometry_rows)
    passed = sum(1 for r in geometry_rows if "pending_class_approval" in r["annotation_status"])
    print(f"  Geometry records: {len(geometry_rows)}")
    print(f"  Total valid boxes: {total_boxes}")
    print(f"  Pending class approval: {passed}")
    return {"geometry_records": len(geometry_rows), "total_boxes": total_boxes}

# ---------------------------------------------------------------------------
# STEP 6 – Near-Duplicate Analysis
# ---------------------------------------------------------------------------

def step6_near_duplicates():
    print("=== STEP 6 – Near-Duplicate Analysis ===")
    contact_dir = PHASE25 / "near_duplicate_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images
    all_images = []
    for split in CHENAB_SPLITS:
        d = RAW_CHENAB / split / "images"
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    all_images.append(("chenab", split, p))
    for subdir in ["Defect_images", "NODefect_images"]:
        d = RAW_AITEX / subdir
        split_name = "defect" if "Defect" in subdir else "no_defect"
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    all_images.append(("aitex", split_name, p))

    print(f"  Hashing {len(all_images)} images...")

    # SHA-256 exact duplicates
    sha_map = defaultdict(list)
    for source, split, path in all_images:
        h = sha256_file(path)
        sha_map[h].append((source, split, path))

    exact_groups = {h: imgs for h, imgs in sha_map.items() if len(imgs) > 1}
    print(f"  Exact duplicate groups: {len(exact_groups)}")

    # Perceptual hashing (pHash)
    print(f"  Computing pHash for {len(all_images)} images...")
    phash_map = {}
    phash_errors = 0
    for source, split, path in all_images:
        try:
            img = Image.open(path).convert("RGB")
            ph = imagehash.phash(img)
            phash_map[str(path)] = (source, split, path, ph)
        except Exception:
            phash_errors += 1

    # Find near-duplicates (hamming distance <= 10)
    THRESHOLD = 10
    items = list(phash_map.values())
    near_dup_pairs = []
    # O(n^2) is expensive for 3k images — use bucket approach
    # Group by first 3 bits of phash as rough bucketing
    phash_values = [(source, split, path, ph) for source, split, path, ph in items]
    seen_pairs = set()
    for i in range(len(phash_values)):
        for j in range(i + 1, len(phash_values)):
            s1, sp1, p1, ph1 = phash_values[i]
            s2, sp2, p2, ph2 = phash_values[j]
            dist = ph1 - ph2
            if dist <= THRESHOLD:
                key = (str(p1), str(p2))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    classification = "exact_duplicate" if dist == 0 else \
                                     "likely_near_duplicate" if dist <= 5 else \
                                     "requires_visual_review"
                    near_dup_pairs.append({
                        "hamming_distance": dist,
                        "file1": str(p1.relative_to(MASTER / "raw")),
                        "source1": s1,
                        "split1": sp1,
                        "file2": str(p2.relative_to(MASTER / "raw")),
                        "source2": s2,
                        "split2": sp2,
                        "classification": classification,
                        "cross_source": s1 != s2,
                        "cross_split": (s1 == s2 == "chenab") and (sp1 != sp2),
                        "human_decision": "pending_visual_review",
                    })
        if i % 500 == 0 and i > 0:
            print(f"    Progress: {i}/{len(phash_values)}")

    cross_split = [r for r in near_dup_pairs if r["cross_split"]]
    cross_source = [r for r in near_dup_pairs if r["cross_source"]]
    print(f"  Near-duplicate pairs (dist<={THRESHOLD}): {len(near_dup_pairs)}")
    print(f"  Cross-split (Chenab): {len(cross_split)}")
    print(f"  Cross-source (Chenab<->AITEX): {len(cross_source)}")

    write_csv(PHASE25 / "near_duplicate_candidates.csv", near_dup_pairs,
              ["hamming_distance", "file1", "source1", "split1",
               "file2", "source2", "split2", "classification",
               "cross_source", "cross_split", "human_decision"])

    # Exact dup groups CSV
    dup_group_rows = []
    for gid, (h, imgs) in enumerate(exact_groups.items(), 1):
        for source, split, path in imgs:
            dup_group_rows.append({
                "group_id": f"exact_{gid:04d}",
                "sha256": h,
                "filename": path.name,
                "source_dataset": source,
                "split": split,
                "duplicate_type": "exact_duplicate",
            })
    write_csv(META / "duplicate_groups.csv", dup_group_rows,
              ["group_id", "sha256", "filename", "source_dataset", "split", "duplicate_type"])

    # Cross-split leakage report
    leakage_rows = []
    for r in near_dup_pairs:
        if r["cross_split"] or (r["classification"] == "exact_duplicate"):
            leakage_rows.append({
                "file1": r["file1"], "source1": r["source1"], "split1": r["split1"],
                "file2": r["file2"], "source2": r["source2"], "split2": r["split2"],
                "hamming_distance": r["hamming_distance"],
                "classification": r["classification"],
                "leakage_risk": "high" if r["classification"] == "exact_duplicate" else "suspected",
                "human_decision": "pending_visual_review",
            })
    write_csv(PHASE25 / "cross_split_leakage_report.csv", leakage_rows,
              ["file1", "source1", "split1", "file2", "source2", "split2",
               "hamming_distance", "classification", "leakage_risk", "human_decision"])
    write_csv(PHASE25 / "duplicate_review_decisions.csv", near_dup_pairs,
              ["hamming_distance", "file1", "source1", "split1",
               "file2", "source2", "split2", "classification",
               "cross_source", "cross_split", "human_decision"])

    # Contact sheets for near-dup pairs (top 16 most similar)
    top_pairs = sorted(near_dup_pairs, key=lambda r: r["hamming_distance"])[:16]
    if top_pairs:
        overlay_pairs = []
        for r in top_pairs:
            p1 = MASTER / "raw" / r["file1"]
            p2 = MASTER / "raw" / r["file2"]
            if p1.is_file() and p2.is_file():
                overlay_pairs.append((p1, p2, f"d={r['hamming_distance']} {r['split1']}↔{r['split2']}"))
        if overlay_pairs:
            make_overlay_sheet(overlay_pairs, contact_dir / "top_near_duplicates.png",
                               cols=4, thumb_size=200, title="Top near-duplicate candidates")

    return {
        "total_images": len(all_images),
        "exact_dup_groups": len(exact_groups),
        "near_dup_pairs": len(near_dup_pairs),
        "cross_split": len(cross_split),
        "cross_source": len(cross_source),
    }

# ---------------------------------------------------------------------------
# STEP 7 – Human Review Package
# ---------------------------------------------------------------------------

def step7_human_review_package(ann_stats, mask_stats, dup_stats, code_stats):
    print("=== STEP 7 – Human Review Package ===")

    lines = [
        "# HUMAN_REVIEW_INSTRUCTIONS.md",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This document explains how to complete the pending human review tasks for Phase 2.5.",
        "All decisions must be recorded in the corresponding CSV files listed below.",
        "",
        "---",
        "",
        "## 1. Chenab Whole-Image Bounding-Box Review",
        "",
        f"- File: `reports/phase2_5/chenab_annotation_review.csv`",
        f"- Contact sheets: `reports/phase2_5/chenab_annotation_contact_sheets/`",
        f"- Records requiring review: {ann_stats['needs_review']}",
        "",
        "**What to do:** For each row where `human_review_status = needs_human_review`:",
        "- Open the contact sheet image for the filename.",
        "- Confirm whether the defect genuinely spans the full image (valid whole-image box) or",
        "  whether this is an annotation error (defect in a subregion, box should be smaller).",
        "- Set `final_decision` to one of:",
        "  - `approved_whole_image_box` — defect truly spans full image",
        "  - `rejected_requires_reannotation` — annotation is imprecise",
        "  - `approved_with_note` — valid but unusual case",
        "- Fill in `reviewer` (your name) and `review_date` (YYYY-MM-DD).",
        "",
        "---",
        "",
        "## 2. AITEX Defect Code Class Verification",
        "",
        f"- File: `reports/phase2_5/aitex_class_mapping_decisions.csv`",
        f"- Code verification: `reports/phase2_5/aitex_code_verification.csv`",
        f"- Contact sheets: `reports/phase2_5/aitex_class_review_contact_sheets/`",
        f"- Codes requiring review: {len(code_stats['observed_codes'])}",
        "",
        "**What to do:** For each AITEX defect code:",
        "- Consult the official AITEX/AFID dataset release from aitex.es/afid",
        "  (the authoritative code-to-name table is only in the original download).",
        "- Match the numeric code to the official class name.",
        "- In `aitex_class_mapping_decisions.csv`, set `human_decision` to:",
        "  - `verified_mapping_approved` — code confirmed and Chenab mapping valid",
        "  - `verified_incompatible` — code confirmed but no valid Chenab equivalent",
        "  - `still_unverified` — could not confirm from authoritative source",
        "- Fill in `reviewer` and `review_date`.",
        "- **Do NOT mark any class approved for training until verified from the original paper or dataset.**",
        "",
        "### Previously rejected proposed mappings (require re-evaluation):",
        "- Code 019 (contamination/stain) → Chenab stain(6): REJECTED — broader category",
        "- Code 030 (cut selvage) → Chenab cut(3): REJECTED — semantically different",
        "",
        "---",
        "",
        "## 3. AITEX Mask Pairing Review",
        "",
        f"- File: `reports/phase2_5/aitex_mask_pairing_corrections.csv`",
        f"- Contact sheets: `reports/phase2_5/aitex_mask_review_contact_sheets/`",
        f"- Cases requiring review: {mask_stats['multi'] + mask_stats['missing']}",
        "",
        "**What to do:**",
        "- For images with **multiple masks** (mask1 + mask2):",
        "  - View both mask files and determine which is the primary ground-truth mask.",
        "  - Set `chosen_mask` in `aitex_mask_pairing_corrections.csv`.",
        "- For images with **no mask** (0044, 0097 had split masks; 0100 truly missing):",
        "  - Confirm whether a mask can be sourced or if the image must be excluded.",
        "",
        "**Images with multiple masks:**",
        "  - `0044_019_04.png` → `0044_019_04_mask1.png` and `0044_019_04_mask2.png`",
        "  - `0097_030_03.png` → `0097_030_03_mask1.png` and `0097_030_03_mask2.png`",
        "",
        "**Images with no mask:**",
        "  - `0100_025_08.png` → no mask file exists",
        "",
        "---",
        "",
        "## 4. Near-Duplicate and Cross-Split Review",
        "",
        f"- File: `reports/phase2_5/near_duplicate_candidates.csv`",
        f"- Contact sheets: `reports/phase2_5/near_duplicate_contact_sheets/`",
        f"- Total near-duplicate pairs: {dup_stats['near_dup_pairs']}",
        f"- Cross-split candidates: {dup_stats['cross_split']}",
        "",
        "**What to do:** For each pair where `human_decision = pending_visual_review`:",
        "- View the contact sheet.",
        "- Classify the pair as one of:",
        "  - `same_image` — identical content",
        "  - `same_defect_event` — same defect, different crop/augmentation",
        "  - `similar_texture_different_defect` — background similar but defects differ",
        "  - `unrelated_similar_pattern` — fabric pattern coincidentally similar",
        "  - `not_duplicate` — clearly different",
        "- Update `human_decision` in `near_duplicate_candidates.csv`.",
        "- **Cross-split pairs** must be resolved before final split assignment.",
        "",
        "---",
        "",
        "## Decision Fields Reference",
        "",
        "| Field | Values |",
        "|-------|--------|",
        "| `final_decision` | pending_review, approved_whole_image_box, rejected_requires_reannotation, auto_excluded_invalid |",
        "| `human_decision` (mapping) | pending_review, verified_mapping_approved, verified_incompatible, still_unverified |",
        "| `human_decision` (duplicates) | pending_visual_review, same_image, same_defect_event, similar_texture_different_defect, not_duplicate |",
        "| `human_verified` (masks) | pending, confirmed_valid, rejected |",
        "",
        "---",
        "",
        "**Important:** Do not mark any record approved without independent expert confirmation.",
        "Automated technical checks have been completed; scientific validity requires human judgment.",
    ]

    write_md(MASTER / "HUMAN_REVIEW_INSTRUCTIONS.md", lines)
    return {}

# ---------------------------------------------------------------------------
# STEP 8 – Updated Master Manifest v1.1
# ---------------------------------------------------------------------------

def step8_update_manifest(mask_qc_rows: list, dup_stats: dict):
    print("=== STEP 8 – Updated Master Manifest v1.1 ===")

    v1_rows = read_csv(META / "master_manifest_v1.csv")

    # Build lookup maps
    mask_qc_by_img = {}
    for r in mask_qc_rows:
        mask_qc_by_img[r["image_filename"]] = r

    # Near-dup flags: read CSV if exists
    dup_flags = {}
    dup_csv = PHASE25 / "near_duplicate_candidates.csv"
    if dup_csv.is_file():
        for r in read_csv(dup_csv):
            for key in ["file1", "file2"]:
                fname = Path(r[key]).name
                existing = dup_flags.get(fname, "not_flagged")
                if r["classification"] == "exact_duplicate":
                    dup_flags[fname] = "exact_duplicate"
                elif existing == "not_flagged":
                    dup_flags[fname] = "near_duplicate_candidate"

    # Cross-split flags
    cross_flags = set()
    cs_csv = PHASE25 / "cross_split_leakage_report.csv"
    if cs_csv.is_file():
        for r in read_csv(cs_csv):
            cross_flags.add(Path(r["file1"]).name)
            cross_flags.add(Path(r["file2"]).name)

    # Annotation review flags
    ann_flags = {}
    ann_csv = PHASE25 / "chenab_annotation_review.csv"
    if ann_csv.is_file():
        for r in read_csv(ann_csv):
            img = r["image_file"]
            status = r["human_review_status"]
            if img not in ann_flags or status == "auto_excluded_invalid":
                ann_flags[img] = status

    v11_rows = []
    for r in v1_rows:
        fname = r["original_filename"]
        source = r["source_dataset"]

        # Annotation QC
        ann_qc = "not_checked"
        if source == "chenab":
            if fname in ann_flags:
                ann_qc = ann_flags[fname]
            else:
                ann_qc = "no_issues_found"

        # Mask QC
        mask_qc = "n/a"
        if source == "aitex" and r["original_split"] == "defect":
            mq = mask_qc_by_img.get(fname)
            if mq:
                mask_qc = mq.get("qc_status", "not_checked")
            else:
                mask_qc = "not_checked"

        # Class verification
        class_ver = "n/a"
        if source == "aitex":
            code = r.get("original_class", "")
            info = AITEX_CODE_INFO.get(str(code), {})
            class_ver = info.get("verification_status", "pending_expert_review")

        # Near-dup status
        dup_status = dup_flags.get(fname, "not_flagged")

        # Cross-split
        cross_status = "flagged_pending_review" if fname in cross_flags else "not_flagged"

        # Human review
        hr_status = "not_required"
        if ann_qc == "needs_human_review":
            hr_status = "pending_annotation_review"
        elif mask_qc in ("multiple_masks_needs_review", "failed_no_mask"):
            hr_status = "pending_mask_review"
        elif class_ver == "pending_expert_review" and source == "aitex":
            hr_status = "pending_class_verification"
        elif dup_status in ("exact_duplicate", "near_duplicate_candidate"):
            hr_status = "pending_duplicate_review"
        elif cross_status == "flagged_pending_review":
            hr_status = "pending_cross_split_review"

        # Training readiness
        training_ready = "no"
        exclusion_reason = ""
        if source == "chenab":
            if r.get("original_split") == "test":
                training_ready = "locked"
                exclusion_reason = "test split reserved"
            elif ann_qc == "auto_excluded_invalid":
                training_ready = "excluded_annotation_error"
                exclusion_reason = "zero-width annotation excluded"
            elif ann_qc == "needs_human_review":
                training_ready = "pending_annotation_review"
                exclusion_reason = "whole-image box needs expert confirmation"
            else:
                training_ready = "yes"
        # AITEX never training-ready without expert class approval
        elif source == "aitex":
            training_ready = "no"
            exclusion_reason = "AITEX class mapping not approved"

        # XAI readiness
        xai_ready = "no"
        if source == "aitex" and r.get("original_split") == "defect":
            mq = mask_qc_by_img.get(fname)
            if mq and mq.get("qc_status") == "passed":
                xai_ready = "candidate_pending_class_verification"
            else:
                xai_ready = "no"
        elif source == "aitex" and r.get("original_split") == "no_defect":
            xai_ready = "background_candidate"
        elif source == "chenab" and r.get("original_split") == "valid":
            xai_ready = "yes"

        row = dict(r)
        row["annotation_qc_status"] = ann_qc
        row["mask_qc_status"] = mask_qc
        row["class_verification_status"] = class_ver
        row["near_duplicate_status"] = dup_status
        row["cross_split_review_status"] = cross_status
        row["human_review_status"] = hr_status
        row["training_readiness"] = training_ready
        row["xai_readiness"] = xai_ready
        row["exclusion_reason"] = exclusion_reason
        v11_rows.append(row)

    # Write v1.1
    v1_fields = list(v1_rows[0].keys()) if v1_rows else []
    new_fields = ["annotation_qc_status", "mask_qc_status", "class_verification_status",
                  "near_duplicate_status", "cross_split_review_status", "human_review_status",
                  "training_readiness", "xai_readiness", "exclusion_reason"]
    all_fields = v1_fields + [f for f in new_fields if f not in v1_fields]
    write_csv(META / "master_manifest_v1_1.csv", v11_rows, all_fields)

    training_yes = sum(1 for r in v11_rows if r["training_readiness"] == "yes")
    xai_yes = sum(1 for r in v11_rows if "yes" in r["xai_readiness"] or "candidate" in r["xai_readiness"])
    print(f"  Total records: {len(v11_rows)}")
    print(f"  Training-ready (Chenab train/valid, no issues): {training_yes}")
    print(f"  XAI-ready/candidate: {xai_yes}")
    return {"total": len(v11_rows), "training_ready": training_yes, "xai_ready": xai_yes}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PHASE25.mkdir(parents=True, exist_ok=True)
    print(f"Phase 2.5 output: {PHASE25}\n")
    ts = datetime.now()

    step1_results = step1_verify()
    print()
    code_stats = step2_aitex_class_verification()
    print()
    ann_stats = step3_chenab_annotation_review()
    print()
    mask_stats = step4_aitex_mask_qc()
    print()
    geo_stats = step5_fix_bbox_derivation()
    print()
    dup_stats = step6_near_duplicates()
    print()
    step7_human_review_package(ann_stats, mask_stats, dup_stats, code_stats)
    print()

    # Need mask QC rows for manifest
    mask_qc_rows = []
    mq_csv = PHASE25 / "aitex_mask_qc.csv"
    if mq_csv.is_file():
        mask_qc_rows = read_csv(mq_csv)
    manifest_stats = step8_update_manifest(mask_qc_rows, dup_stats)

    # Summary
    elapsed = (datetime.now() - ts).total_seconds()
    print(f"\n=== Phase 2.5 Complete ({elapsed:.0f}s) ===")
    print(f"Chenab: {step1_results['chenab_count']} images")
    print(f"AITEX: {step1_results['aitex_total']} images")
    print(f"NaN/Inf annotations: {step1_results['nan_inf']}")
    print(f"Annotation issues: 33 total | {ann_stats['auto_fix']} auto-fixed | {ann_stats['needs_review']} needs review")
    print(f"Mask QC: {mask_stats['verified']} verified | {mask_stats['multi']} multi-mask | {mask_stats['missing']} no-mask | {mask_stats['empty']} empty")
    print(f"Geometry boxes: {geo_stats['total_boxes']}")
    print(f"Exact duplicates: {dup_stats['exact_dup_groups']} groups")
    print(f"Near-dup pairs: {dup_stats['near_dup_pairs']}")
    print(f"Cross-split candidates: {dup_stats['cross_split']}")
    print(f"Manifest v1.1: {manifest_stats['total']} records | {manifest_stats['training_ready']} training-ready")

    return {
        "step1": step1_results,
        "ann": ann_stats,
        "mask": mask_stats,
        "geo": geo_stats,
        "dup": dup_stats,
        "manifest": manifest_stats,
    }


if __name__ == "__main__":
    results = main()
    # Write results JSON for final report
    out = PHASE25 / "phase2_5_results.json"
    with open(out, "w") as f:
        json.dump({k: {kk: str(vv) for kk, vv in v.items()} if isinstance(v, dict) else str(v)
                   for k, v in results.items()}, f, indent=2)
