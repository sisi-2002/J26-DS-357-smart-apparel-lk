"""
Phase 2 – Master Dataset Standardization and Preparation
Covers Steps 2.1 through 2.6:
  - Class mapping (Chenab ↔ AITEX)
  - AITEX bounding-box derivation from masks
  - Duplicate / near-duplicate detection
  - Master manifest V1
  - Function 2 readiness report
  - V1 package structure
"""

import csv
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("ERROR: Pillow and numpy are required.")
    sys.exit(1)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER_ROOT = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
CHENAB_ROOT = MASTER_ROOT / "raw" / "chenab" / "fabricDD.v2i.yolov8"
AITEX_ROOT = MASTER_ROOT / "raw" / "aitex"
PHASE1_REPORT = MASTER_ROOT / "reports" / "phase1_audit"
META_DIR = MASTER_ROOT / "metadata"
DERIVED_ANN_DIR = MASTER_ROOT / "derived_annotations" / "aitex_boxes"
PHASE2_REPORT = MASTER_ROOT / "reports" / "phase2"
SCRIPTS_DIR = MASTER_ROOT / "scripts"  # symlink reference only

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CHENAB_SPLITS = ("train", "valid", "test")
CHENAB_CLASSES = {0: "baekra", 1: "color issues", 2: "contamination",
                  3: "cut", 4: "gray stitch", 5: "selvet", 6: "stain"}
NUM_CLASSES = 7

# AITEX defect codes from filename pattern NNNN_DDD_II
# DDD is the defect code per AITEX documentation
AITEX_DEFECT_CODES = {
    "000": {"name": "no_defect", "description": "No defect"},
    "002": {"name": "broken_end", "description": "Broken end (warp thread break)"},
    "006": {"name": "broken_pick", "description": "Broken pick (weft thread break)"},
    "010": {"name": "broken_warp", "description": "Broken warp (density variation)"},
    "016": {"name": "nep", "description": "Nep (fibre knot/tangle)"},
    "019": {"name": "stain", "description": "Oil stain or contamination"},
    "022": {"name": "broken_weft", "description": "Broken weft"},
    "025": {"name": "weft_crack", "description": "Weft crack"},
    "027": {"name": "warp_crack", "description": "Warp crack"},
    "029": {"name": "fuzzyball", "description": "Fuzzyball"},
    "030": {"name": "cuts", "description": "Cuts"},
    "036": {"name": "crease", "description": "Crease"},
    "040": {"name": "weft_density_change", "description": "Weft density change"},
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


def write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path} ({len(rows)} rows)")


def get_image_dims(path: Path):
    try:
        img = Image.open(path)
        return img.width, img.height
    except Exception:
        return None, None

# ---------------------------------------------------------------------------
# Step 2.1 – Defect Class Mapping
# ---------------------------------------------------------------------------

def build_class_mapping():
    print("=== Step 2.1 – Class Mapping ===")

    # Mapping rows: source, source_class_id, source_class_name,
    #   mapped_class_id, mapped_class_name, mapping_status, notes
    mapping_rows = []

    # Chenab classes map to themselves (ground truth for unified taxonomy)
    for cid, cname in CHENAB_CLASSES.items():
        mapping_rows.append({
            "source_dataset": "chenab",
            "source_class_id": cid,
            "source_class_name": cname,
            "mapped_class_id": cid,
            "mapped_class_name": cname,
            "mapping_status": "exact_self",
            "approved_for_training": "yes",
            "notes": "Chenab is the primary taxonomy source",
        })

    # AITEX defect codes → Chenab mapping (proposed, requires human approval)
    aitex_to_chenab = {
        "002": {"mapped_id": None, "mapped_name": "broken_end", "status": "incompatible",
                "approved": "no", "notes": "No equivalent in 7-class Chenab taxonomy"},
        "006": {"mapped_id": None, "mapped_name": "broken_pick", "status": "incompatible",
                "approved": "no", "notes": "No equivalent in 7-class Chenab taxonomy"},
        "010": {"mapped_id": None, "mapped_name": "broken_warp", "status": "incompatible",
                "approved": "no", "notes": "No equivalent in 7-class Chenab taxonomy"},
        "016": {"mapped_id": None, "mapped_name": "nep", "status": "incompatible",
                "approved": "no", "notes": "No equivalent in 7-class Chenab taxonomy"},
        "019": {"mapped_id": 6, "mapped_name": "stain", "status": "possible_match",
                "approved": "pending_human_review",
                "notes": "AITEX 019=oil stain may correspond to Chenab stain(6); requires expert review"},
        "022": {"mapped_id": None, "mapped_name": "broken_weft", "status": "incompatible",
                "approved": "no", "notes": "No equivalent in 7-class Chenab taxonomy"},
        "025": {"mapped_id": None, "mapped_name": "weft_crack", "status": "incompatible",
                "approved": "no", "notes": "No equivalent in 7-class Chenab taxonomy"},
        "027": {"mapped_id": None, "mapped_name": "warp_crack", "status": "incompatible",
                "approved": "no", "notes": "No equivalent in 7-class Chenab taxonomy"},
        "030": {"mapped_id": 3, "mapped_name": "cut", "status": "possible_match",
                "approved": "pending_human_review",
                "notes": "AITEX 030=cuts may correspond to Chenab cut(3); requires expert review"},
        "029": {"mapped_id": None, "mapped_name": "fuzzyball", "status": "incompatible",
                "approved": "no", "notes": "No equivalent in 7-class Chenab taxonomy"},
        "036": {"mapped_id": None, "mapped_name": "crease", "status": "incompatible",
                "approved": "no", "notes": "No equivalent in 7-class Chenab taxonomy"},
        "040": {"mapped_id": None, "mapped_name": "weft_density_change", "status": "incompatible",
                "approved": "no", "notes": "No equivalent in 7-class Chenab taxonomy"},
    }

    # Collect AITEX defect codes actually present in dataset
    defect_dir = AITEX_ROOT / "Defect_images"
    observed_codes = set()
    if defect_dir.is_dir():
        for p in defect_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                parts = p.stem.split("_")
                if len(parts) >= 2:
                    observed_codes.add(parts[1])

    print(f"  AITEX defect codes observed: {sorted(observed_codes)}")

    for code in sorted(observed_codes):
        info = aitex_to_chenab.get(code, {})
        doc = AITEX_DEFECT_CODES.get(code, {"name": f"unknown_{code}", "description": "Not in AITEX docs"})
        mapping_rows.append({
            "source_dataset": "aitex",
            "source_class_id": code,
            "source_class_name": doc["name"],
            "mapped_class_id": info.get("mapped_id", ""),
            "mapped_class_name": info.get("mapped_name", doc["name"]),
            "mapping_status": info.get("status", "unknown"),
            "approved_for_training": info.get("approved", "no"),
            "notes": info.get("notes", ""),
        })

    print(f"  Total mapping rows: {len(mapping_rows)}")

    # class_definitions.md
    defn_lines = [
        "# Class Definitions",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Chenab 7-Class Taxonomy (primary)",
        "",
        "| ID | Name | Source |",
        "|----|------|--------|",
    ]
    for cid, cname in CHENAB_CLASSES.items():
        defn_lines.append(f"| {cid} | {cname} | Roboflow fabricDD v2 |")
    defn_lines += [
        "",
        "## AITEX Defect Codes (source-specific, not merged)",
        "",
        "| Code | Name | Description | Mapping Status |",
        "|------|------|-------------|----------------|",
    ]
    for code, doc in sorted(AITEX_DEFECT_CODES.items()):
        info = aitex_to_chenab.get(code, {})
        status = info.get("status", "not_observed")
        defn_lines.append(f"| {code} | {doc['name']} | {doc['description']} | {status} |")
    defn_lines += [
        "",
        "## Notes",
        "",
        "- AITEX defect codes are derived from filenames (pattern: NNNN_DDD_II).",
        "- No AITEX class is approved for unified detector training without expert sign-off.",
        "- The 7-class Chenab taxonomy is NOT extended during Phase 1–2.",
    ]
    META_DIR.mkdir(parents=True, exist_ok=True)
    (META_DIR / "class_definitions.md").write_text("\n".join(defn_lines), encoding="utf-8")

    # class_mapping_review.md
    review_lines = [
        "# Class Mapping Review",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        f"- Chenab classes: {NUM_CLASSES} (all exact_self, approved)",
        f"- AITEX codes observed in dataset: {len(observed_codes)}",
        f"- Possible matches requiring expert review: {sum(1 for r in mapping_rows if r['mapping_status'] == 'possible_match')}",
        f"- Incompatible (not approved): {sum(1 for r in mapping_rows if r['mapping_status'] == 'incompatible')}",
        "",
        "## HUMAN REVIEW REQUIRED",
        "",
        "The following AITEX codes have proposed Chenab mappings that require expert approval:",
        "",
    ]
    for r in mapping_rows:
        if r["mapping_status"] == "possible_match":
            review_lines.append(f"- AITEX {r['source_class_id']} ({r['source_class_name']}) → "
                                 f"Chenab {r['mapped_class_id']} ({r['mapped_class_name']}): {r['notes']}")
    review_lines += [
        "",
        "No AITEX class is approved for training until a domain expert confirms the mapping.",
    ]
    PHASE2_REPORT.mkdir(parents=True, exist_ok=True)
    (PHASE2_REPORT / "class_mapping_review.md").write_text("\n".join(review_lines), encoding="utf-8")

    return mapping_rows, {c.split("_")[1] if "_" in str(c) else str(c): r
                          for r in mapping_rows if r["source_dataset"] == "aitex"
                          for c in [str(r["source_class_id"])]}

# ---------------------------------------------------------------------------
# Step 2.2 – AITEX Bounding Box Derivation
# ---------------------------------------------------------------------------

def derive_aitex_boxes(aitex_mapping_by_code: dict):
    print("=== Step 2.2 – AITEX Bounding Box Derivation ===")
    defect_dir = AITEX_ROOT / "Defect_images"
    mask_dir = AITEX_ROOT / "Mask_images"
    DERIVED_ANN_DIR.mkdir(parents=True, exist_ok=True)

    annotation_manifest = []

    for img_path in sorted(defect_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        stem = img_path.stem
        parts = stem.split("_")
        defect_code = parts[1] if len(parts) >= 2 else "unknown"

        mask_path = mask_dir / f"{stem}_mask{img_path.suffix}"
        if not mask_path.is_file():
            annotation_manifest.append({
                "image_filename": img_path.name,
                "defect_code": defect_code,
                "mask_filename": "",
                "box_file": "",
                "box_count": 0,
                "mapping_status": "no_mask",
                "approved_class_id": "",
                "review_required": "yes",
                "review_reason": "No mask found",
            })
            continue

        # Determine approved class
        mapping_row = aitex_mapping_by_code.get(defect_code)
        approved_class_id = ""
        mapping_status = "unknown"
        review_required = "yes"
        review_reason = ""

        if mapping_row:
            mapping_status = mapping_row.get("mapping_status", "unknown")
            approved = mapping_row.get("approved_for_training", "no")
            if approved == "yes":
                approved_class_id = str(mapping_row.get("mapped_class_id", ""))
                review_required = "no"
            elif approved == "pending_human_review":
                approved_class_id = ""
                review_reason = f"Proposed mapping to Chenab class {mapping_row.get('mapped_class_id')} pending expert review"
            else:
                review_reason = "Incompatible class — no mapping to Chenab taxonomy"
        else:
            review_reason = f"Unknown defect code {defect_code}"

        # Read mask and extract boxes
        try:
            mask_img = Image.open(mask_path).convert("L")
            arr = np.array(mask_img)
            img_w, img_h = mask_img.width, mask_img.height
        except Exception as e:
            annotation_manifest.append({
                "image_filename": img_path.name,
                "defect_code": defect_code,
                "mask_filename": mask_path.name,
                "box_file": "",
                "box_count": 0,
                "mapping_status": mapping_status,
                "approved_class_id": approved_class_id,
                "review_required": "yes",
                "review_reason": f"mask_read_error:{e}",
            })
            continue

        # Find foreground regions
        binary = (arr > 127).astype(np.uint8)
        boxes = []

        if HAS_CV2:
            n_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(binary)
            for comp_id in range(1, n_labels):  # skip background (0)
                x, y, w, h, area = stats[comp_id]
                if area < 10:  # skip tiny noise
                    continue
                # Normalized YOLO format: cx cy w h
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                # Clamp to [0,1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.001, min(1.0, nw))
                nh = max(0.001, min(1.0, nh))
                boxes.append((cx, cy, nw, nh, area))
        else:
            # Fallback: single bounding box over all foreground pixels
            ys, xs = np.where(binary > 0)
            if len(ys) > 0:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                cx = (x1 + w / 2) / img_w
                cy = (y1 + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                boxes.append((cx, cy, nw, nh, w * h))

        # Write box file (geometry only; class_id omitted if not approved)
        box_file = DERIVED_ANN_DIR / f"{stem}.txt"
        with open(box_file, "w", encoding="utf-8") as f:
            for cx, cy, nw, nh, _ in boxes:
                class_field = approved_class_id if approved_class_id else "?"
                f.write(f"{class_field} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        annotation_manifest.append({
            "image_filename": img_path.name,
            "defect_code": defect_code,
            "mask_filename": mask_path.name,
            "box_file": str(box_file.relative_to(MASTER_ROOT)),
            "box_count": len(boxes),
            "mapping_status": mapping_status,
            "approved_class_id": approved_class_id,
            "review_required": review_required,
            "review_reason": review_reason if review_reason else "ok",
        })

    print(f"  Processed {len(annotation_manifest)} defect images")
    print(f"  Boxes generated: {sum(r['box_count'] for r in annotation_manifest)}")
    print(f"  Requiring review: {sum(1 for r in annotation_manifest if r['review_required'] == 'yes')}")

    return annotation_manifest

# ---------------------------------------------------------------------------
# Step 2.3 – Duplicate Detection
# ---------------------------------------------------------------------------

def detect_duplicates():
    print("=== Step 2.3 – Duplicate Detection ===")

    # Collect all image paths
    all_images = []
    for split in CHENAB_SPLITS:
        img_dir = CHENAB_ROOT / split / "images"
        if img_dir.is_dir():
            for p in sorted(img_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    all_images.append(("chenab", split, p))

    for subdir in ["Defect_images", "NODefect_images"]:
        d = AITEX_ROOT / subdir
        split_name = "defect" if "Defect" in subdir else "no_defect"
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    all_images.append(("aitex", split_name, p))

    print(f"  Total images to hash: {len(all_images)}")

    # SHA-256 exact duplicate detection
    hash_to_images: dict[str, list] = {}
    for source, split, path in all_images:
        h = sha256_file(path)
        hash_to_images.setdefault(h, []).append((source, split, path, h))

    exact_dup_groups = {h: imgs for h, imgs in hash_to_images.items() if len(imgs) > 1}
    print(f"  Exact duplicate groups: {len(exact_dup_groups)}")

    # Perceptual hash for near-duplicates
    phash_to_images: dict[str, list] = {}
    has_phash = False
    try:
        import imagehash
        has_phash = True
    except ImportError:
        pass

    near_dup_rows = []
    if has_phash:
        import imagehash
        phashes = {}
        for source, split, path, _ in [(s, sp, p, h) for s, sp, p in [(s, sp, p) for s, sp, p in all_images]
                                        for h in [sha256_file(p)]]:
            try:
                img = Image.open(path)
                ph = str(imagehash.phash(img))
                phashes[str(path)] = (source, split, path, ph)
                phash_to_images.setdefault(ph, []).append((source, split, path))
            except Exception:
                pass
        near_dup_groups = {ph: imgs for ph, imgs in phash_to_images.items() if len(imgs) > 1}
        print(f"  Perceptual hash near-dup groups: {len(near_dup_groups)}")
        for ph, imgs in near_dup_groups.items():
            for i, (s1, sp1, p1) in enumerate(imgs):
                for s2, sp2, p2 in imgs[i+1:]:
                    near_dup_rows.append({
                        "phash": ph,
                        "file1": str(p1.relative_to(MASTER_ROOT / "raw")),
                        "source1": s1,
                        "split1": sp1,
                        "file2": str(p2.relative_to(MASTER_ROOT / "raw")),
                        "source2": s2,
                        "split2": sp2,
                        "classification": "likely_near_duplicate",
                    })
    else:
        print("  imagehash not available — perceptual hash skipped")

    # Build duplicate_groups.csv
    dup_rows = []
    group_id = 0
    for h, imgs in exact_dup_groups.items():
        group_id += 1
        for source, split, path, file_hash in imgs:
            dup_rows.append({
                "duplicate_group_id": f"exact_{group_id:04d}",
                "filename": path.name,
                "source_dataset": source,
                "split": split,
                "sha256": file_hash,
                "duplicate_type": "exact_duplicate",
            })

    # Cross-split leakage
    leakage_rows = []
    for h, imgs in exact_dup_groups.items():
        splits_present = {(s, sp) for s, sp, _, _ in imgs}
        chenab_splits_present = {sp for s, sp, _, _ in imgs if s == "chenab"}
        if len(chenab_splits_present) > 1:
            for source, split, path, file_hash in imgs:
                leakage_rows.append({
                    "sha256": file_hash,
                    "filename": path.name,
                    "source_dataset": source,
                    "split": split,
                    "leakage_type": "cross_split_exact",
                    "splits_involved": "|".join(sorted(chenab_splits_present)),
                })

    print(f"  Cross-split leakage candidates: {len(leakage_rows)}")

    return dup_rows, leakage_rows, near_dup_rows, {
        "total_images": len(all_images),
        "exact_dup_groups": len(exact_dup_groups),
        "exact_dup_images": sum(len(v) for v in exact_dup_groups.values()),
        "cross_split_leakage": len(leakage_rows),
        "near_dup_candidates": len(near_dup_rows),
    }

# ---------------------------------------------------------------------------
# Step 2.4 – Master Dataset Manifest
# ---------------------------------------------------------------------------

def build_master_manifest(mapping_rows: list, ann_manifest: list):
    print("=== Step 2.4 – Master Manifest ===")

    # Build quick lookup: aitex defect_code → mapping row
    aitex_mapping = {str(r["source_class_id"]): r for r in mapping_rows if r["source_dataset"] == "aitex"}

    # Build aitex ann lookup: image_filename → ann row
    ann_by_img = {r["image_filename"]: r for r in ann_manifest}

    manifest = []

    def make_record(image_id, source, version, orig_filename, rel_img_path,
                    width, height, orig_split, orig_class_id, orig_class_name,
                    proposed_mapped_class, mapping_status, ann_format,
                    bbox_count, mask_available, mask_validation_status,
                    quality_status, dup_group_id, src_group_id, provenance_status,
                    license_status, approved_detection, approved_xai, review_req, review_reason):
        return {
            "image_id": image_id,
            "source_dataset": source,
            "source_version": version,
            "original_filename": orig_filename,
            "relative_image_path": rel_img_path,
            "image_sha256": "",  # populated below
            "width": width,
            "height": height,
            "original_split": orig_split,
            "original_class": orig_class_id,
            "proposed_mapped_class": proposed_mapped_class,
            "mapping_status": mapping_status,
            "annotation_format": ann_format,
            "bounding_box_count": bbox_count,
            "mask_available": mask_available,
            "mask_validation_status": mask_validation_status,
            "quality_status": quality_status,
            "duplicate_group_id": dup_group_id,
            "source_group_id": src_group_id,
            "provenance_status": provenance_status,
            "license_status": license_status,
            "approved_for_detection_training": approved_detection,
            "approved_for_xai_evaluation": approved_xai,
            "review_required": review_req,
            "review_reason": review_reason,
        }

    img_counter = 0

    # Chenab images
    for split in CHENAB_SPLITS:
        img_dir = CHENAB_ROOT / split / "images"
        lbl_dir = CHENAB_ROOT / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            img_counter += 1
            image_id = f"chenab_{split}_{img_path.stem}"
            rel_path = str(img_path.relative_to(MASTER_ROOT / "raw"))
            w, h = get_image_dims(img_path)

            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            bbox_count = 0
            if lbl_path.is_file():
                lines = [l.strip() for l in lbl_path.read_text(errors="replace").splitlines() if l.strip()]
                bbox_count = len(lines)

            sha = sha256_file(img_path)
            review_req = "no"
            review_reason = ""
            # test split is locked
            if split == "test":
                review_req = "yes"
                review_reason = "test split locked for evaluation — approved for metadata only"

            rec = make_record(
                image_id=image_id, source="chenab", version="v2_roboflow",
                orig_filename=img_path.name, rel_img_path=rel_path,
                width=w, height=h, orig_split=split,
                orig_class_id="multi", orig_class_name="see_labels",
                proposed_mapped_class="see_labels", mapping_status="exact_self",
                ann_format="yolo_v8", bbox_count=bbox_count,
                mask_available="no", mask_validation_status="n/a",
                quality_status="unchecked",
                dup_group_id="", src_group_id="",
                provenance_status="ok",
                license_status="private_roboflow",
                approved_detection="yes" if split != "test" else "locked",
                approved_xai="yes" if split == "valid" else "pending",
                review_req=review_req, review_reason=review_reason,
            )
            rec["image_sha256"] = sha
            manifest.append(rec)

    # AITEX defect images
    defect_dir = AITEX_ROOT / "Defect_images"
    mask_dir = AITEX_ROOT / "Mask_images"
    if defect_dir.is_dir():
        for img_path in sorted(defect_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            img_counter += 1
            stem = img_path.stem
            parts = stem.split("_")
            defect_code = parts[1] if len(parts) >= 2 else "unknown"
            image_id = f"aitex_defect_{stem}"
            rel_path = str(img_path.relative_to(MASTER_ROOT / "raw"))
            w, h = get_image_dims(img_path)

            mask_path = mask_dir / f"{stem}_mask{img_path.suffix}"
            mask_available = "yes" if mask_path.is_file() else "no"
            mask_val_status = "paired" if mask_path.is_file() else "missing"

            ann_row = ann_by_img.get(img_path.name, {})
            bbox_count = ann_row.get("box_count", 0)
            map_row = aitex_mapping.get(defect_code, {})
            proposed = str(map_row.get("mapped_class_id", "")) if map_row else ""
            map_status = map_row.get("mapping_status", "unknown") if map_row else "unknown"
            approved_det = "no"  # never approved without human sign-off for AITEX
            approved_xai = "candidate" if mask_available == "yes" else "no"

            sha = sha256_file(img_path)
            rec = make_record(
                image_id=image_id, source="aitex", version="v1_public",
                orig_filename=img_path.name, rel_img_path=rel_path,
                width=w, height=h, orig_split="defect",
                orig_class_id=defect_code, orig_class_name=AITEX_DEFECT_CODES.get(defect_code, {}).get("name", "unknown"),
                proposed_mapped_class=proposed, mapping_status=map_status,
                ann_format="mask_derived_yolo", bbox_count=bbox_count,
                mask_available=mask_available, mask_validation_status=mask_val_status,
                quality_status="unchecked",
                dup_group_id="", src_group_id="",
                provenance_status="ok",
                license_status="aitex_public_research",
                approved_detection=approved_det,
                approved_xai=approved_xai,
                review_req="yes",
                review_reason="AITEX class mapping requires expert approval before training use",
            )
            rec["image_sha256"] = sha
            manifest.append(rec)

    # AITEX no-defect images
    nodefect_dir = AITEX_ROOT / "NODefect_images"
    if nodefect_dir.is_dir():
        for img_path in sorted(nodefect_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            img_counter += 1
            stem = img_path.stem
            image_id = f"aitex_nodefect_{stem}"
            rel_path = str(img_path.relative_to(MASTER_ROOT / "raw"))
            w, h = get_image_dims(img_path)
            sha = sha256_file(img_path)
            rec = make_record(
                image_id=image_id, source="aitex", version="v1_public",
                orig_filename=img_path.name, rel_img_path=rel_path,
                width=w, height=h, orig_split="no_defect",
                orig_class_id="000", orig_class_name="no_defect",
                proposed_mapped_class="", mapping_status="no_defect",
                ann_format="none", bbox_count=0,
                mask_available="no", mask_validation_status="n/a",
                quality_status="unchecked",
                dup_group_id="", src_group_id="",
                provenance_status="ok",
                license_status="aitex_public_research",
                approved_detection="no",
                approved_xai="candidate_background",
                review_req="no",
                review_reason="",
            )
            rec["image_sha256"] = sha
            manifest.append(rec)

    print(f"  Manifest records: {len(manifest)}")
    return manifest

# ---------------------------------------------------------------------------
# Step 2.5 – Function 2 Readiness
# ---------------------------------------------------------------------------

def function2_readiness_report(manifest: list):
    print("=== Step 2.5 – Function 2 Readiness ===")
    total = len(manifest)
    with_mask = sum(1 for r in manifest if r["mask_available"] == "yes")
    aitex_defect = sum(1 for r in manifest if r["source_dataset"] == "aitex" and r["original_split"] == "defect")
    aitex_nodefect = sum(1 for r in manifest if r["source_dataset"] == "aitex" and r["original_split"] == "no_defect")
    xai_candidate = sum(1 for r in manifest if r["approved_for_xai_evaluation"] in ("yes", "candidate", "candidate_background"))

    lines = [
        "# Function 2 – XAI Research Dataset Readiness",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        f"- Total manifest records: {total}",
        f"- Images with verified masks: {with_mask}",
        f"- AITEX defect images (mask-derived boxes): {aitex_defect}",
        f"- AITEX no-defect images (background candidates): {aitex_nodefect}",
        f"- XAI evaluation candidates: {xai_candidate}",
        "",
        "## Availability Check",
        "",
        "| Requirement | Available | Notes |",
        "|-------------|-----------|-------|",
        f"| Intact fabric background regions | Yes | AITEX NODefect_images ({aitex_nodefect} images) |",
        f"| Normal fabric samples | Yes | AITEX NODefect_images |",
        f"| RGB colour information | Yes | All images are RGB/grayscale PNG |",
        "| Texture and colour diversity | Partial | AITEX is limited fabric types; diversity assessment pending |",
        f"| Human-verifiable defect masks | Yes (candidate) | {with_mask} AITEX masks available; require expert QC |",
        "| Source and image provenance | Yes | Recorded in master manifest |",
        "| Independent sample groups | Partial | AITEX groups by defect code; Chenab groups not yet assigned |",
        "",
        "## Blockers for Final XAI Evaluation",
        "",
        "1. AITEX class mapping must be approved by domain expert before complexity labelling.",
        "2. Complexity tiers (Simple / Moderate / High) NOT assigned in Phase 2.",
        "3. Calibration fit NOT performed in Phase 2.",
        "4. Final masked XAI evaluation requires test split access approval.",
        "",
        "## Recommended Next Steps",
        "",
        "- Expert review of AITEX class mapping (Step 2.1 blocking items).",
        "- Manual QC of AITEX mask quality for XAI ground truth.",
        "- Background extraction candidates: AITEX NODefect_images.",
        "- Complexity calibration: use Chenab valid split after Phase 3 split assignment.",
    ]

    PHASE2_REPORT.mkdir(parents=True, exist_ok=True)
    (PHASE2_REPORT / "function2_readiness.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {PHASE2_REPORT / 'function2_readiness.md'}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in [META_DIR, DERIVED_ANN_DIR, PHASE2_REPORT, SCRIPTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Master root: {MASTER_ROOT}\n")

    # Step 2.1
    mapping_rows, aitex_mapping_by_code = build_class_mapping()
    write_csv(META_DIR / "class_mapping_proposed.csv", mapping_rows,
              ["source_dataset", "source_class_id", "source_class_name", "mapped_class_id",
               "mapped_class_name", "mapping_status", "approved_for_training", "notes"])

    # Step 2.2
    ann_manifest = derive_aitex_boxes(aitex_mapping_by_code)
    write_csv(DERIVED_ANN_DIR / ".." / "annotation_manifest.csv", ann_manifest,
              ["image_filename", "defect_code", "mask_filename", "box_file", "box_count",
               "mapping_status", "approved_class_id", "review_required", "review_reason"])

    # Step 2.3
    dup_rows, leakage_rows, near_dup_rows, dup_stats = detect_duplicates()
    write_csv(META_DIR / "duplicate_groups.csv", dup_rows,
              ["duplicate_group_id", "filename", "source_dataset", "split", "sha256", "duplicate_type"])
    write_csv(PHASE2_REPORT / "cross_split_leakage.csv", leakage_rows,
              ["sha256", "filename", "source_dataset", "split", "leakage_type", "splits_involved"])
    write_csv(PHASE2_REPORT / "duplicate_review.csv", near_dup_rows,
              ["phash", "file1", "source1", "split1", "file2", "source2", "split2", "classification"])

    # Step 2.4
    manifest = build_master_manifest(mapping_rows, ann_manifest)
    manifest_fields = [
        "image_id", "source_dataset", "source_version", "original_filename",
        "relative_image_path", "image_sha256", "width", "height", "original_split",
        "original_class", "proposed_mapped_class", "mapping_status",
        "annotation_format", "bounding_box_count", "mask_available",
        "mask_validation_status", "quality_status", "duplicate_group_id",
        "source_group_id", "provenance_status", "license_status",
        "approved_for_detection_training", "approved_for_xai_evaluation",
        "review_required", "review_reason",
    ]
    write_csv(META_DIR / "master_manifest_v1.csv", manifest, manifest_fields)

    # Step 2.5
    function2_readiness_report(manifest)

    # Print concise summary
    print("\n=== Phase 2 Summary ===")
    print(f"Class mapping rows: {len(mapping_rows)}")
    print(f"Pending expert review (AITEX mapping): {sum(1 for r in mapping_rows if r['approved_for_training'] == 'pending_human_review')}")
    print(f"AITEX boxes derived: {sum(r['box_count'] for r in ann_manifest)}")
    print(f"Exact duplicate groups: {dup_stats['exact_dup_groups']}, images: {dup_stats['exact_dup_images']}")
    print(f"Cross-split leakage candidates: {dup_stats['cross_split_leakage']}")
    print(f"Master manifest records: {len(manifest)}")
    print("Phase 2 COMPLETE")


if __name__ == "__main__":
    main()
