"""
Phase 1 – Dataset Audit and Quality Checking
Covers Steps 1.1 through 1.5:
  - Chenab inventory, annotation validation, image quality
  - AITEX inventory, mask pairing, mask validation, image quality
  - Output reports to data-preprocessing/master_fabric_dataset/reports/phase1_audit/
"""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("ERROR: Pillow and numpy are required. Install with: pip install pillow numpy opencv-python")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER_ROOT = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
CHENAB_ROOT = MASTER_ROOT / "raw" / "chenab" / "fabricDD.v2i.yolov8"
AITEX_ROOT = MASTER_ROOT / "raw" / "aitex"
REPORT_DIR = MASTER_ROOT / "reports" / "phase1_audit"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CHENAB_SPLITS = ("train", "valid", "test")
CHENAB_CLASSES = {0: "baekra", 1: "color issues", 2: "contamination",
                  3: "cut", 4: "gray stitch", 5: "selvet", 6: "stain"}
NUM_CLASSES = 7

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image_safe(path: Path):
    """Return (PIL Image, error_string). error_string is None on success."""
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)  # reopen after verify
        img.load()
        return img, None
    except Exception as e:
        return None, str(e)


def image_metrics(path: Path):
    """Return dict with width, height, format, mode; or error."""
    img, err = load_image_safe(path)
    if err:
        return {"width": None, "height": None, "format": None, "mode": None, "error": err}
    w, h = img.size
    return {"width": w, "height": h, "format": img.format or path.suffix[1:].upper(),
            "mode": img.mode, "error": None}


def blur_score(path: Path) -> float | None:
    """Laplacian variance as blur indicator (higher = sharper). Needs cv2."""
    if not HAS_CV2:
        return None
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return None


def brightness_stats(path: Path):
    """Mean and std of luminance. Returns (mean, std) or (None, None)."""
    try:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=float)
        return float(arr.mean()), float(arr.std())
    except Exception:
        return None, None

# ---------------------------------------------------------------------------
# Step 1.1 – Chenab Inventory
# ---------------------------------------------------------------------------

def audit_chenab():
    print("=== Step 1.1 & 1.2 & 1.4 – Chenab ===")
    rows = []
    class_counts = {cid: 0 for cid in range(NUM_CLASSES)}
    annotation_issues = []
    multi_defect_count = 0
    no_annotation_count = 0
    total_images = 0
    total_annotations = 0

    for split in CHENAB_SPLITS:
        img_dir = CHENAB_ROOT / split / "images"
        lbl_dir = CHENAB_ROOT / split / "labels"
        if not img_dir.is_dir():
            print(f"  WARNING: {img_dir} not found")
            continue

        imgs = sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        for img_path in imgs:
            total_images += 1
            rel = img_path.relative_to(CHENAB_ROOT)
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name

            meta = image_metrics(img_path)
            blur = blur_score(img_path)
            bright_mean, bright_std = brightness_stats(img_path)

            # Quality flags
            quality_flags = []
            if meta["error"]:
                quality_flags.append(f"corrupt:{meta['error'][:80]}")
            if meta["width"] and meta["width"] < 64:
                quality_flags.append("very_small_width")
            if meta["height"] and meta["height"] < 64:
                quality_flags.append("very_small_height")
            if blur is not None and blur < 10:
                quality_flags.append(f"low_blur_score:{blur:.1f}")
            if bright_mean is not None and bright_mean < 10:
                quality_flags.append("underexposed")
            if bright_mean is not None and bright_mean > 245:
                quality_flags.append("overexposed")

            # Annotation
            ann_count = 0
            ann_valid = 0
            if not lbl_path.is_file():
                quality_flags.append("missing_label_file")
            else:
                lines = lbl_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                ann_lines = [l for l in lines if l.strip()]
                ann_count = len(ann_lines)
                for line_num, line in enumerate(ann_lines, 1):
                    parts = line.strip().split()
                    issues = []
                    if len(parts) != 5:
                        issues.append(f"wrong_field_count:{len(parts)}")
                    else:
                        try:
                            cid = int(parts[0])
                            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            if cid < 0 or cid >= NUM_CLASSES:
                                issues.append(f"invalid_class_id:{cid}")
                            for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                                if not (0.0 <= val <= 1.0):
                                    issues.append(f"coord_out_of_range:{name}={val:.4f}")
                            if w <= 0:
                                issues.append("zero_or_neg_width")
                            if h <= 0:
                                issues.append("zero_or_neg_height")
                            if cx - w/2 < -0.001 or cx + w/2 > 1.001:
                                issues.append("box_exceeds_width")
                            if cy - h/2 < -0.001 or cy + h/2 > 1.001:
                                issues.append("box_exceeds_height")
                            if w < 0.005 or h < 0.005:
                                issues.append(f"suspiciously_small_box:{w:.4f}x{h:.4f}")
                            if w > 0.99 and h > 0.99:
                                issues.append("suspiciously_large_box")
                            if not issues:
                                class_counts[cid] += 1
                                total_annotations += 1
                                ann_valid += 1
                        except ValueError as e:
                            issues.append(f"parse_error:{e}")
                    if issues:
                        annotation_issues.append({
                            "split": split,
                            "image_file": img_path.name,
                            "label_file": lbl_path.name,
                            "line_num": line_num,
                            "raw_line": line[:120],
                            "issues": "|".join(issues),
                        })

            if ann_count == 0:
                no_annotation_count += 1
            if ann_count > 1:
                multi_defect_count += 1

            rows.append({
                "split": split,
                "filename": img_path.name,
                "relative_path": str(rel),
                "width": meta["width"],
                "height": meta["height"],
                "format": meta["format"],
                "mode": meta["mode"],
                "annotation_count": ann_count,
                "label_file_exists": lbl_path.is_file(),
                "blur_score": f"{blur:.2f}" if blur is not None else "",
                "brightness_mean": f"{bright_mean:.2f}" if bright_mean is not None else "",
                "brightness_std": f"{bright_std:.2f}" if bright_std is not None else "",
                "quality_flags": "|".join(quality_flags) if quality_flags else "ok",
                "file_error": meta["error"] or "",
            })

    print(f"  Total images: {total_images}")
    print(f"  Total valid annotations: {total_annotations}")
    print(f"  Images with no annotations: {no_annotation_count}")
    print(f"  Images with multiple defects: {multi_defect_count}")
    print(f"  Annotation issues found: {len(annotation_issues)}")

    return rows, class_counts, annotation_issues, {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "no_annotation_count": no_annotation_count,
        "multi_defect_count": multi_defect_count,
    }

# ---------------------------------------------------------------------------
# Step 1.1 & 1.3 & 1.4 – AITEX Inventory and Mask Validation
# ---------------------------------------------------------------------------

def audit_aitex():
    print("=== Step 1.1, 1.3 & 1.4 – AITEX ===")
    defect_dir = AITEX_ROOT / "Defect_images"
    nodefect_dir = AITEX_ROOT / "NODefect_images"
    mask_dir = AITEX_ROOT / "Mask_images"

    defect_imgs = sorted(p for p in defect_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if defect_dir.is_dir() else []
    nodefect_imgs = sorted(p for p in nodefect_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if nodefect_dir.is_dir() else []
    mask_imgs = sorted(p for p in mask_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if mask_dir.is_dir() else []

    # Build mask lookup: stem without _mask -> mask path
    mask_lookup = {}
    for mp in mask_imgs:
        stem = mp.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        mask_lookup[stem] = mp

    pairing_rows = []
    quality_rows = []

    for img_path in defect_imgs:
        stem = img_path.stem
        mask_path = mask_lookup.get(stem)
        mask_available = mask_path is not None

        meta = image_metrics(img_path)
        blur = blur_score(img_path)
        bright_mean, bright_std = brightness_stats(img_path)

        quality_flags = []
        if meta["error"]:
            quality_flags.append(f"corrupt:{meta['error'][:80]}")
        if blur is not None and blur < 5:
            quality_flags.append(f"low_blur:{blur:.1f}")
        if bright_mean is not None and bright_mean < 5:
            quality_flags.append("underexposed")
        if bright_mean is not None and bright_mean > 250:
            quality_flags.append("overexposed")

        # Mask validation
        mask_issues = []
        mask_width = mask_height = None
        dim_match = ""
        mask_unique_values = ""
        empty_mask = False
        disconnected_regions = 0

        if mask_path:
            mmeta = image_metrics(mask_path)
            mask_width = mmeta["width"]
            mask_height = mmeta["height"]
            if mmeta["error"]:
                mask_issues.append(f"mask_corrupt:{mmeta['error'][:80]}")
            elif meta["width"] and meta["height"]:
                if mmeta["width"] == meta["width"] and mmeta["height"] == meta["height"]:
                    dim_match = "match"
                else:
                    dim_match = f"mismatch:img={meta['width']}x{meta['height']},mask={mmeta['width']}x{mmeta['height']}"
                    mask_issues.append(dim_match)
            # Check mask content
            try:
                mask_img = Image.open(mask_path).convert("L")
                arr = np.array(mask_img)
                uniq = np.unique(arr).tolist()
                mask_unique_values = str(uniq[:20])
                unexpected = [v for v in uniq if v not in (0, 255) and not (0 <= v <= 10) and not (245 <= v <= 255)]
                if unexpected:
                    mask_issues.append(f"unexpected_pixel_values:{unexpected[:5]}")
                nonzero = np.count_nonzero(arr)
                if nonzero == 0:
                    empty_mask = True
                    mask_issues.append("empty_mask")
                # Count connected components
                if HAS_CV2:
                    binary = (arr > 127).astype(np.uint8)
                    n_labels, _ = cv2.connectedComponents(binary)
                    disconnected_regions = max(0, n_labels - 1)  # subtract background
            except Exception as e:
                mask_issues.append(f"mask_read_error:{e}")
        else:
            mask_issues.append("no_mask_found")

        pairing_rows.append({
            "filename": img_path.name,
            "image_stem": stem,
            "mask_filename": mask_path.name if mask_path else "",
            "mask_available": mask_available,
            "image_width": meta["width"],
            "image_height": meta["height"],
            "mask_width": mask_width,
            "mask_height": mask_height,
            "dimension_match": dim_match,
            "mask_unique_values": mask_unique_values,
            "empty_mask": empty_mask,
            "disconnected_regions": disconnected_regions,
            "mask_issues": "|".join(mask_issues) if mask_issues else "ok",
        })

        quality_rows.append({
            "source": "aitex",
            "split": "defect",
            "filename": img_path.name,
            "width": meta["width"],
            "height": meta["height"],
            "format": meta["format"],
            "blur_score": f"{blur:.2f}" if blur is not None else "",
            "brightness_mean": f"{bright_mean:.2f}" if bright_mean is not None else "",
            "brightness_std": f"{bright_std:.2f}" if bright_std is not None else "",
            "quality_flags": "|".join(quality_flags) if quality_flags else "ok",
        })

    # Non-defect images
    for img_path in nodefect_imgs:
        meta = image_metrics(img_path)
        blur = blur_score(img_path)
        bright_mean, bright_std = brightness_stats(img_path)
        quality_flags = []
        if meta["error"]:
            quality_flags.append(f"corrupt:{meta['error'][:80]}")
        quality_rows.append({
            "source": "aitex",
            "split": "no_defect",
            "filename": img_path.name,
            "width": meta["width"],
            "height": meta["height"],
            "format": meta["format"],
            "blur_score": f"{blur:.2f}" if blur is not None else "",
            "brightness_mean": f"{bright_mean:.2f}" if bright_mean is not None else "",
            "brightness_std": f"{bright_std:.2f}" if bright_std is not None else "",
            "quality_flags": "|".join(quality_flags) if quality_flags else "ok",
        })

    # Orphan masks (mask with no defect image)
    orphan_masks = [m for stem, m in mask_lookup.items() if stem not in {p.stem for p in defect_imgs}]

    print(f"  Defect images: {len(defect_imgs)}")
    print(f"  No-defect images: {len(nodefect_imgs)}")
    print(f"  Mask images: {len(mask_imgs)}")
    print(f"  Paired (defect+mask): {sum(1 for r in pairing_rows if r['mask_available'])}")
    print(f"  Missing masks: {sum(1 for r in pairing_rows if not r['mask_available'])}")
    print(f"  Orphan masks: {len(orphan_masks)}")
    print(f"  Empty masks: {sum(1 for r in pairing_rows if r['empty_mask'])}")

    return pairing_rows, quality_rows, {
        "defect_images": len(defect_imgs),
        "nodefect_images": len(nodefect_imgs),
        "mask_images": len(mask_imgs),
        "paired": sum(1 for r in pairing_rows if r["mask_available"]),
        "missing_masks": sum(1 for r in pairing_rows if not r["mask_available"]),
        "orphan_masks": len(orphan_masks),
        "empty_masks": sum(1 for r in pairing_rows if r["empty_mask"]),
    }

# ---------------------------------------------------------------------------
# Step 1.5 – Source Metadata (hashes for immutability verification)
# ---------------------------------------------------------------------------

def compute_source_metadata():
    print("=== Step 1.5 – Source metadata hashing ===")
    rows = []
    # Sample a subset of files for hash recording (not all 2800+ files for efficiency)
    # Hash data.yaml and README as anchors; sample first image per split
    anchors = [
        CHENAB_ROOT / "data.yaml",
        CHENAB_ROOT / "README.roboflow.txt",
    ]
    for split in CHENAB_SPLITS:
        img_dir = CHENAB_ROOT / split / "images"
        if img_dir.is_dir():
            imgs = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if imgs:
                anchors.append(imgs[0])

    for p in anchors:
        if p.is_file():
            rows.append({
                "source": "chenab",
                "relative_path": str(p.relative_to(MASTER_ROOT / "raw")),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            })

    # AITEX anchors
    for subdir in ["Defect_images", "NODefect_images", "Mask_images"]:
        d = AITEX_ROOT / subdir
        if d.is_dir():
            files = sorted(p for p in d.iterdir() if p.is_file())
            if files:
                rows.append({
                    "source": "aitex",
                    "relative_path": str(files[0].relative_to(MASTER_ROOT / "raw")),
                    "sha256": sha256_file(files[0]),
                    "size_bytes": files[0].stat().st_size,
                })

    print(f"  Recorded {len(rows)} source anchor hashes")
    return rows

# ---------------------------------------------------------------------------
# Write CSV helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path} ({len(rows)} rows)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Report directory: {REPORT_DIR}\n")

    # Chenab
    chenab_rows, class_counts, ann_issues, chenab_stats = audit_chenab()
    # AITEX
    aitex_pairing, aitex_quality, aitex_stats = audit_aitex()
    # Source metadata
    source_meta = compute_source_metadata()

    # Write reports
    print("\n=== Writing reports ===")

    # dataset_inventory.csv
    inv_fields = ["split", "filename", "relative_path", "width", "height", "format", "mode",
                  "annotation_count", "label_file_exists", "blur_score", "brightness_mean",
                  "brightness_std", "quality_flags", "file_error"]
    write_csv(REPORT_DIR / "dataset_inventory.csv", chenab_rows, inv_fields)

    # chenab_class_distribution.csv
    class_rows = [{"class_id": cid, "class_name": CHENAB_CLASSES[cid], "annotation_count": count}
                  for cid, count in class_counts.items()]
    write_csv(REPORT_DIR / "chenab_class_distribution.csv", class_rows,
              ["class_id", "class_name", "annotation_count"])

    # chenab_annotation_issues.csv
    write_csv(REPORT_DIR / "chenab_annotation_issues.csv", ann_issues,
              ["split", "image_file", "label_file", "line_num", "raw_line", "issues"])

    # aitex_mask_pairing.csv
    pair_fields = ["filename", "image_stem", "mask_filename", "mask_available",
                   "image_width", "image_height", "mask_width", "mask_height",
                   "dimension_match", "mask_unique_values", "empty_mask",
                   "disconnected_regions", "mask_issues"]
    write_csv(REPORT_DIR / "aitex_mask_pairing.csv", aitex_pairing, pair_fields)

    # image_quality_report.csv (chenab + aitex combined)
    chenab_quality = [{
        "source": "chenab",
        "split": r["split"],
        "filename": r["filename"],
        "width": r["width"],
        "height": r["height"],
        "format": r["format"],
        "blur_score": r["blur_score"],
        "brightness_mean": r["brightness_mean"],
        "brightness_std": r["brightness_std"],
        "quality_flags": r["quality_flags"],
    } for r in chenab_rows]
    quality_rows = chenab_quality + aitex_quality
    write_csv(REPORT_DIR / "image_quality_report.csv", quality_rows,
              ["source", "split", "filename", "width", "height", "format",
               "blur_score", "brightness_mean", "brightness_std", "quality_flags"])

    # source_metadata.csv
    write_csv(REPORT_DIR / "source_metadata.csv", source_meta,
              ["source", "relative_path", "sha256", "size_bytes"])

    # phase1_summary.md
    chenab_per_split = {}
    for r in chenab_rows:
        s = r["split"]
        chenab_per_split.setdefault(s, {"images": 0, "ann": 0})
        chenab_per_split[s]["images"] += 1
        chenab_per_split[s]["ann"] += r["annotation_count"]

    quality_flagged_chenab = sum(1 for r in chenab_rows if r["quality_flags"] != "ok")
    quality_flagged_aitex = sum(1 for r in aitex_quality if r["quality_flags"] != "ok")

    summary_lines = [
        "# Phase 1 Audit Summary",
        "",
        f"Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Chenab Dataset",
        "",
        f"- Total images: {chenab_stats['total_images']}",
    ]
    for s, c in chenab_per_split.items():
        summary_lines.append(f"  - {s}: {c['images']} images, {c['ann']} annotations")
    summary_lines += [
        f"- Total valid annotations: {chenab_stats['total_annotations']}",
        f"- Images with no annotations: {chenab_stats['no_annotation_count']}",
        f"- Images with multiple defects: {chenab_stats['multi_defect_count']}",
        f"- Annotation issues found: {len(ann_issues)}",
        f"- Images with quality flags: {quality_flagged_chenab}",
        "",
        "### Class Distribution",
    ]
    for cid, cname in CHENAB_CLASSES.items():
        summary_lines.append(f"  - {cid} {cname}: {class_counts[cid]} annotations")
    summary_lines += [
        "",
        "## AITEX Dataset",
        "",
        f"- Defect images: {aitex_stats['defect_images']}",
        f"- No-defect images: {aitex_stats['nodefect_images']}",
        f"- Mask images: {aitex_stats['mask_images']}",
        f"- Defect images with paired mask: {aitex_stats['paired']}",
        f"- Defect images missing mask: {aitex_stats['missing_masks']}",
        f"- Orphan masks (no corresponding defect image): {aitex_stats['orphan_masks']}",
        f"- Empty masks (no foreground pixels): {aitex_stats['empty_masks']}",
        f"- Images with quality flags: {quality_flagged_aitex}",
        "",
        "## Source Limitations",
        "",
        "- Chenab dataset license is Private (Roboflow). Not cleared for public redistribution.",
        "- AITEX: filename encoding appears to be NNNN_DDD_II (image_id, defect_code, instance). "
          "Defect class metadata must be validated against official AITEX documentation before use.",
        "- Class mapping between Chenab and AITEX is NOT yet approved (see Phase 2).",
        "",
        "## Unresolved Verification Requirements",
        "",
        "- AITEX defect class semantics: the numeric codes in filenames (e.g., 002, 006, 010, 016, 019) "
          "require mapping to official AITEX defect class names before any training use.",
        "- Chenab test split is locked for evaluation; metadata-only auditing performed.",
        "- Human review required for all annotation_issues entries before training approval.",
    ]

    summary_path = REPORT_DIR / "phase1_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  Written: {summary_path}")

    # Print concise summary to stdout
    print("\n=== Phase 1 Summary ===")
    print(f"Chenab: {chenab_stats['total_images']} images, {chenab_stats['total_annotations']} annotations, "
          f"{len(ann_issues)} annotation issues")
    print(f"AITEX: {aitex_stats['defect_images']} defect, {aitex_stats['nodefect_images']} no-defect, "
          f"{aitex_stats['mask_images']} masks, {aitex_stats['missing_masks']} missing masks")
    print(f"Quality flagged: Chenab={quality_flagged_chenab}, AITEX={quality_flagged_aitex}")
    print("Phase 1 COMPLETE")


if __name__ == "__main__":
    main()
