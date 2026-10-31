"""
Phase 2.6 – Cross-Split Leakage Resolution
Tasks A–E combined runner.
"""

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
    import imagehash
    HAS_PHASH = True
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
RAW_CHENAB = MASTER / "raw" / "chenab" / "fabricDD.v2i.yolov8"
PHASE25 = MASTER / "reports" / "phase2_5"
PHASE26 = MASTER / "reports" / "phase2_6"
META = MASTER / "metadata"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")

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


def write_csv(path: Path, rows: list, fields: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path.name} ({len(rows)} rows)")


def write_md(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {path.name}")


def make_contact_sheet(pairs: list, out_path: Path, cols=4, thumb=220, title=""):
    """pairs: list of (path, label). Saves PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(pairs)
    if n == 0:
        return
    rows_n = math.ceil(n / cols)
    lh = 22
    W = cols * thumb
    H = rows_n * (thumb + lh) + (36 if title else 0)
    sheet = Image.new("RGB", (W, H), (35, 35, 35))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((4, 4), title[:110], fill=(210, 210, 210))
    for idx, (img_path, lbl) in enumerate(pairs):
        col = idx % cols
        row = idx // cols
        yo = (36 if title else 0) + row * (thumb + lh)
        xo = col * thumb
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((thumb, thumb))
            sheet.paste(img, (xo + (thumb - img.width)//2, yo + (thumb - img.height)//2))
        except Exception as e:
            draw.rectangle([xo, yo, xo+thumb, yo+thumb], outline=(180,50,50))
            draw.text((xo+2, yo+2), str(e)[:25], fill=(255,80,80))
        draw.text((xo+2, yo+thumb+2), str(lbl)[:42], fill=(190,190,190))
    sheet.save(str(out_path))


def make_pair_sheet(pair_list: list, out_path: Path, cols=4, thumb=200, title=""):
    """pair_list: list of (path_a, path_b, label). Side-by-side."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(pair_list)
    if n == 0:
        return
    cw = thumb * 2
    rows_n = math.ceil(n / cols)
    lh = 20
    W = cols * cw
    H = rows_n * (thumb + lh) + (36 if title else 0)
    sheet = Image.new("RGB", (W, H), (30, 30, 30))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((4, 4), title[:110], fill=(210, 210, 210))
    for idx, (pa, pb, lbl) in enumerate(pair_list):
        col = idx % cols
        row = idx // cols
        yo = (36 if title else 0) + row * (thumb + lh)
        xo = col * cw
        for side, fp in enumerate([pa, pb]):
            px = xo + side * thumb
            try:
                img = Image.open(fp).convert("RGB")
                img.thumbnail((thumb, thumb))
                sheet.paste(img, (px, yo))
            except Exception as e:
                draw.rectangle([px, yo, px+thumb, yo+thumb], outline=(180,50,50))
                draw.text((px+2, yo+2), str(e)[:22], fill=(255,80,80))
        draw.text((xo+2, yo+thumb+2), str(lbl)[:55], fill=(190,190,190))
    sheet.save(str(out_path))

# ---------------------------------------------------------------------------
# Filename root extractor
# ---------------------------------------------------------------------------
# Roboflow export pattern: [augmented_]<source_base>_jpg.rf.<32hex>.jpg
# Additional patterns observed:
#   <NNNNNN>_jpg.rf.<32hex>.jpg          (6-digit numeric base)
#   <NNNNNNNN>_jpg.rf.<32hex>.jpg        (8-digit numeric base)
#   <timestamp>_jpg.rf.<32hex>.jpg       (datetime base)
#   <desc>_jpeg_jpg.rf.<32hex>.jpg       (double extension base)
# Augmentation prefix: "augmented_" stripped before root extraction.
# Only the _jpg.rf.<32hex> suffix is stripped — nothing else.

RF_PATTERN = re.compile(
    r'^(augmented_)?(.+?)_jpg\.rf\.[0-9a-f]{32}$',
    re.IGNORECASE
)


def extract_root(stem: str) -> tuple[str, str]:
    """
    Returns (root, parse_status).
    parse_status: 'rf_pattern' | 'rf_fallback' | 'unparseable'
    """
    m = RF_PATTERN.match(stem)
    if m:
        return m.group(2), "rf_pattern"
    # Fallback: find _jpg.rf. anywhere in stem
    idx = stem.lower().find("_jpg.rf.")
    if idx > 0:
        root = stem[:idx]
        if root.lower().startswith("augmented_"):
            root = root[10:]
        return root, "rf_fallback"
    return stem, "unparseable"

# ---------------------------------------------------------------------------
# TASK A – Correct Duplicate Terminology
# ---------------------------------------------------------------------------

def task_a_correct_terminology():
    print("=== TASK A – Correct Duplicate Terminology ===")

    dup_rows = read_csv(PHASE25 / "near_duplicate_candidates.csv")

    # Phase 2.5 incorrectly labelled pHash-0 as "exact_duplicate" — that label
    # should be reserved for SHA-256 byte-identical files. pHash-0 means
    # pixel-identical perceptual hash (same visual content, possibly different
    # encoding/metadata) but is NOT the same as SHA-256 identity.
    # Re-classify all rows with corrected terminology.

    corrected = []
    for r in dup_rows:
        try:
            dist = int(r["hamming_distance"])
        except (ValueError, KeyError):
            dist = -1
        old_cls = r.get("classification", "")

        # Determine correct terminology
        if dist == 0:
            new_cls = "phash_identical"           # same visual hash, not SHA-256
            evidence = "phash_distance_0"
        elif dist <= 5:
            new_cls = "likely_near_duplicate"
            evidence = f"phash_distance_{dist}"
        elif dist <= 10:
            new_cls = "requires_visual_review"
            evidence = f"phash_distance_{dist}"
        else:
            new_cls = "not_similar"
            evidence = f"phash_distance_{dist}"

        cross_split = r.get("cross_split", "False") == "True"
        cross_src = r.get("cross_source", "False") == "True"

        corrected.append({
            "file1": r["file1"],
            "source1": r["source1"],
            "split1": r["split1"],
            "file2": r["file2"],
            "source2": r["source2"],
            "split2": r["split2"],
            "hamming_distance": dist,
            "classification_v25": old_cls,
            "classification_v26": new_cls,
            "evidence_type": evidence,
            "sha256_verified": "no",    # not checked in Phase 2.5
            "pixel_identical_verified": "no",
            "cross_split": cross_split,
            "cross_source": cross_src,
            "human_decision": "pending_visual_review",
        })

    total = len(corrected)
    cross = [r for r in corrected if r["cross_split"]]
    cs_phash0 = [r for r in cross if r["hamming_distance"] == 0]
    cs_near = [r for r in cross if 0 < r["hamming_distance"] <= 5]
    cs_review = [r for r in cross if 5 < r["hamming_distance"] <= 10]
    same_split = [r for r in corrected if not r["cross_split"] and not r["cross_source"]]
    ss_phash0 = [r for r in same_split if r["hamming_distance"] == 0]

    print(f"  Total candidate pairs: {total}")
    print(f"  Cross-split pairs: {len(cross)}")
    print(f"    pHash-0 (phash_identical): {len(cs_phash0)}")
    print(f"    pHash 1-5 (likely_near_duplicate): {len(cs_near)}")
    print(f"    pHash 6-10 (requires_visual_review): {len(cs_review)}")
    print(f"  Same-split pairs: {len(same_split)}")
    print(f"    Same-split pHash-0: {len(ss_phash0)}")

    # Note on Phase 2.5 error: leakage_report.csv included same-split phash-0
    # pairs by mistake (1535 rows vs 1440 true cross-split). Correct count is:
    # cross-split leakage = 1440 pairs; same-split phash-0 were incorrectly included.
    print(f"\n  Phase 2.5 correction note:")
    print(f"    Old 'exact_duplicate' label was applied to pHash-0 pairs (not SHA-256)")
    print(f"    Cross-split leakage report had {len(cs_phash0) + len(ss_phash0)} phash-0 pairs total")
    print(f"    Of these, {len(cs_phash0)} are cross-split (leakage risk)")
    print(f"    And {len(ss_phash0)} are same-split (not leakage, but mislabelled in v2.5)")

    write_csv(PHASE26 / "corrected_duplicate_pairs.csv", corrected,
              ["file1", "source1", "split1", "file2", "source2", "split2",
               "hamming_distance", "classification_v25", "classification_v26",
               "evidence_type", "sha256_verified", "pixel_identical_verified",
               "cross_split", "cross_source", "human_decision"])

    return {
        "total": total, "cross_split": len(cross),
        "cs_phash0": len(cs_phash0), "cs_near": len(cs_near),
        "cs_review": len(cs_review), "same_split": len(same_split),
        "ss_phash0": len(ss_phash0),
    }

# ---------------------------------------------------------------------------
# TASK B – Extract Source Image Families
# ---------------------------------------------------------------------------

def task_b_source_families():
    print("=== TASK B – Extract Source Image Families ===")

    # Collect all Chenab images with roots
    all_imgs = []
    for split in SPLITS:
        d = RAW_CHENAB / split / "images"
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    stem = p.stem
                    root, parse_status = extract_root(stem)
                    is_aug = stem.lower().startswith("augmented_")
                    all_imgs.append({
                        "filename": p.name,
                        "split": split,
                        "stem": stem,
                        "source_root": root,
                        "parse_status": parse_status,
                        "is_augmented": is_aug,
                        "relative_path": f"chenab/{split}/images/{p.name}",
                    })

    total = len(all_imgs)
    unparseable = [r for r in all_imgs if r["parse_status"] == "unparseable"]
    print(f"  Total Chenab images: {total}")
    print(f"  Unparseable filenames: {len(unparseable)}")

    # Root collision analysis
    root_to_rows = defaultdict(list)
    for r in all_imgs:
        root_to_rows[r["source_root"]].append(r)

    unique_roots = len(root_to_rows)
    multi_export = {r: rows for r, rows in root_to_rows.items() if len(rows) > 1}
    cross_split_roots = {r: rows for r, rows in root_to_rows.items()
                         if len({row["split"] for row in rows}) > 1}
    test_touching = {r: rows for r, rows in root_to_rows.items()
                     if any(row["split"] == "test" for row in rows)
                     and len({row["split"] for row in rows}) > 1}

    print(f"  Unique source roots: {unique_roots}")
    print(f"  Roots with >1 export (Roboflow augmentations): {len(multi_export)}")
    print(f"  Roots spanning multiple splits: {len(cross_split_roots)}")
    print(f"  Roots touching test split (cross-split): {len(test_touching)}")

    # Build source_family_candidates.csv
    family_rows = []
    for root, rows in sorted(root_to_rows.items()):
        splits_present = sorted({r["split"] for r in rows})
        n_aug = sum(1 for r in rows if r["is_augmented"])
        n_total = len(rows)
        cross = len(splits_present) > 1
        touches_test = "test" in splits_present and cross
        parse_statuses = list({r["parse_status"] for r in rows})

        family_rows.append({
            "source_root": root,
            "image_count": n_total,
            "augmented_count": n_aug,
            "splits_present": "|".join(splits_present),
            "n_splits": len(splits_present),
            "cross_split": cross,
            "touches_test_split": touches_test,
            "parse_statuses": "|".join(parse_statuses),
            "family_status": (
                "cross_split_test_touch" if touches_test else
                "cross_split" if cross else
                "single_split"
            ),
            "training_block_reason": (
                "touches_locked_test_split" if touches_test else
                "cross_split_pending_resolution" if cross else
                ""
            ),
        })

    write_csv(PHASE26 / "source_family_candidates.csv", family_rows,
              ["source_root", "image_count", "augmented_count", "splits_present",
               "n_splits", "cross_split", "touches_test_split", "parse_statuses",
               "family_status", "training_block_reason"])

    # Per-image root assignment (for manifest join)
    write_csv(PHASE26 / "chenab_image_roots.csv", all_imgs,
              ["filename", "split", "stem", "source_root", "parse_status",
               "is_augmented", "relative_path"])

    return {
        "total_images": total, "unique_roots": unique_roots,
        "cross_split_roots": len(cross_split_roots),
        "test_touching_roots": len(test_touching),
        "unparseable": len(unparseable),
        "root_to_rows": root_to_rows,
        "family_rows": family_rows,
    }

# ---------------------------------------------------------------------------
# TASK C – Build Leakage Groups
# ---------------------------------------------------------------------------

def task_c_leakage_groups(root_to_rows: dict, family_rows: list, dup_stats: dict):
    print("=== TASK C – Build Leakage Groups ===")

    corrected = read_csv(PHASE26 / "corrected_duplicate_pairs.csv")
    cross_pairs = [r for r in corrected if r["cross_split"] == "True"]

    # Confirmed duplicate groups: filename-root ancestry (verifiable)
    # A group is "confirmed" if two images share the same Roboflow source root
    # AND appear in different splits. This is filename-based ancestry — strongest
    # non-visual evidence available without re-downloading originals.
    family_by_root = {r["source_root"]: r for r in family_rows}

    confirmed_rows = []
    ambiguous_rows = []
    conflict_rows = []

    group_id = 0
    processed_roots = set()

    for root, root_rows in sorted(root_to_rows.items()):
        splits = {r["split"] for r in root_rows}
        if len(splits) <= 1:
            continue  # single-split — not a leakage group
        group_id += 1
        gid = f"sfam_{group_id:04d}"
        touches_test = "test" in splits
        is_conflict = True  # any cross-split family is a potential conflict

        for r in root_rows:
            confirmed_rows.append({
                "group_id": gid,
                "source_root": root,
                "filename": r["filename"],
                "split": r["split"],
                "is_augmented": r["is_augmented"],
                "evidence_type": "filename_root_ancestry",
                "crosses_splits": "|".join(sorted(splits)),
                "touches_test": touches_test,
                "group_status": "confirmed_source_family",
            })

        if touches_test:
            conflict_rows.append({
                "group_id": gid,
                "source_root": root,
                "splits_involved": "|".join(sorted(splits)),
                "images_in_group": len(root_rows),
                "conflict_type": "source_family_spans_test_split",
                "resolution_required": "yes",
                "proposed_action": "block_non_test_images_pending_split_policy_approval",
                "human_decision": "pending_review",
            })
        else:
            conflict_rows.append({
                "group_id": gid,
                "source_root": root,
                "splits_involved": "|".join(sorted(splits)),
                "images_in_group": len(root_rows),
                "conflict_type": "source_family_spans_train_valid",
                "resolution_required": "yes",
                "proposed_action": "assign_entire_family_to_single_split_in_phase3",
                "human_decision": "pending_review",
            })

    # Ambiguous edges: cross-split pHash pairs where roots differ
    # (could be same physical fabric, different source image)
    phash_confirmed_files = {r["filename"] for r in confirmed_rows}
    for r in cross_pairs:
        f1 = Path(r["file1"]).name
        f2 = Path(r["file2"]).name
        dist = int(r["hamming_distance"])
        # Extract roots for both
        root1, _ = extract_root(Path(f1).stem)
        root2, _ = extract_root(Path(f2).stem)
        if root1 == root2:
            continue  # already in confirmed groups
        ambiguous_rows.append({
            "file1": f1,
            "split1": r["split1"],
            "root1": root1,
            "file2": f2,
            "split2": r["split2"],
            "root2": root2,
            "hamming_distance": dist,
            "classification": r["classification_v26"],
            "relationship": "different_root_same_phash_proximity",
            "auto_merged": "no",
            "human_decision": "pending_visual_review",
            "notes": "",
        })

    print(f"  Confirmed source family groups: {group_id}")
    print(f"  Images in confirmed groups: {len(confirmed_rows)}")
    print(f"  Cross-split conflict groups: {len(conflict_rows)}")
    print(f"    Touching test split: {sum(1 for r in conflict_rows if r['conflict_type']=='source_family_spans_test_split')}")
    print(f"  Ambiguous pHash edges (different roots): {len(ambiguous_rows)}")

    write_csv(PHASE26 / "confirmed_duplicate_groups.csv", confirmed_rows,
              ["group_id", "source_root", "filename", "split", "is_augmented",
               "evidence_type", "crosses_splits", "touches_test",
               "group_status"])
    write_csv(PHASE26 / "cross_split_group_conflicts.csv", conflict_rows,
              ["group_id", "source_root", "splits_involved", "images_in_group",
               "conflict_type", "resolution_required", "proposed_action",
               "human_decision"])
    write_csv(PHASE26 / "ambiguous_duplicate_edges.csv", ambiguous_rows,
              ["file1", "split1", "root1", "file2", "split2", "root2",
               "hamming_distance", "classification", "relationship",
               "auto_merged", "human_decision", "notes"])

    return {
        "n_groups": group_id,
        "confirmed_images": len(confirmed_rows),
        "conflict_groups": len(conflict_rows),
        "test_conflicts": sum(1 for r in conflict_rows if r["conflict_type"] == "source_family_spans_test_split"),
        "ambiguous_edges": len(ambiguous_rows),
    }

# ---------------------------------------------------------------------------
# TASK D – Visual Review Contact Sheets
# ---------------------------------------------------------------------------

def task_d_visual_review(root_to_rows: dict):
    print("=== TASK D – Visual Review Contact Sheets ===")
    contact_dir = PHASE26 / "visual_review_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    corrected = read_csv(PHASE26 / "corrected_duplicate_pairs.csv")
    cross_pairs = [r for r in corrected if r["cross_split"] == "True"]

    review_records = []

    # Priority 1: Cross-split pHash-0 pairs (65 pairs)
    p0 = sorted([r for r in cross_pairs if int(r["hamming_distance"]) == 0],
                key=lambda x: (x["split1"], x["split2"]))
    # Build pair sheet
    pair_data = []
    for idx, r in enumerate(p0[:24]):
        f1 = MASTER / "raw" / r["file1"]
        f2 = MASTER / "raw" / r["file2"]
        if f1.is_file() and f2.is_file():
            pair_data.append((f1, f2,
                f"#{idx+1} d=0 {r['split1']}/{r['split2']}"))
    if pair_data:
        make_pair_sheet(pair_data, contact_dir / "p1_phash0_cross_split.png",
                        cols=3, thumb=180,
                        title="Priority 1: Cross-split pHash-0 pairs (24 of 65)")

    for idx, r in enumerate(p0):
        root1, _ = extract_root(Path(r["file1"]).name.replace(".jpg",""))
        review_records.append({
            "review_id": f"p1_{idx+1:03d}",
            "priority": 1,
            "pair_type": "cross_split_phash_identical",
            "file1": r["file1"],
            "split1": r["split1"],
            "root1": root1,
            "file2": r["file2"],
            "split2": r["split2"],
            "hamming_distance": 0,
            "evidence": "phash_distance_0",
            "contact_sheet": "p1_phash0_cross_split.png",
            "proposed_action": "confirm_same_source",
            "human_decision": "pending_visual_review",
            "reviewer": "",
            "review_date": "",
            "notes": "",
        })

    # Priority 2: Cross-split near-dup pairs (pHash 1-5, 70 pairs)
    p2 = sorted([r for r in cross_pairs if 0 < int(r["hamming_distance"]) <= 5],
                key=lambda x: int(x["hamming_distance"]))
    pair_data2 = []
    for idx, r in enumerate(p2[:20]):
        f1 = MASTER / "raw" / r["file1"]
        f2 = MASTER / "raw" / r["file2"]
        if f1.is_file() and f2.is_file():
            pair_data2.append((f1, f2,
                f"#{idx+1} d={r['hamming_distance']} {r['split1']}/{r['split2']}"))
    if pair_data2:
        make_pair_sheet(pair_data2, contact_dir / "p2_near_dup_cross_split.png",
                        cols=4, thumb=180,
                        title="Priority 2: Cross-split near-duplicate pairs (20 of 70)")

    for idx, r in enumerate(p2):
        root1, _ = extract_root(Path(r["file1"]).name.replace(".jpg",""))
        review_records.append({
            "review_id": f"p2_{idx+1:03d}",
            "priority": 2,
            "pair_type": "cross_split_near_duplicate",
            "file1": r["file1"],
            "split1": r["split1"],
            "root1": root1,
            "file2": r["file2"],
            "split2": r["split2"],
            "hamming_distance": r["hamming_distance"],
            "evidence": f"phash_distance_{r['hamming_distance']}",
            "contact_sheet": "p2_near_dup_cross_split.png",
            "proposed_action": "confirm_same_source" if int(r["hamming_distance"]) <= 3 else "uncertain",
            "human_decision": "pending_visual_review",
            "reviewer": "",
            "review_date": "",
            "notes": "",
        })

    # Priority 3: Cross-split pairs involving test images
    test_pairs = [r for r in cross_pairs
                  if r["split1"] == "test" or r["split2"] == "test"]
    test_p0 = [r for r in test_pairs if int(r["hamming_distance"]) == 0][:20]
    pair_data3 = []
    for idx, r in enumerate(test_p0[:16]):
        f1 = MASTER / "raw" / r["file1"]
        f2 = MASTER / "raw" / r["file2"]
        if f1.is_file() and f2.is_file():
            pair_data3.append((f1, f2,
                f"#{idx+1} d={r['hamming_distance']} {r['split1']}/{r['split2']}"))
    if pair_data3:
        make_pair_sheet(pair_data3, contact_dir / "p3_test_cross_split.png",
                        cols=4, thumb=180,
                        title="Priority 3: Cross-split pairs involving test split")

    # Priority 4: Source families crossing test split (sample families)
    families = read_csv(PHASE26 / "source_family_candidates.csv")
    test_families = [f for f in families if f["touches_test_split"] == "True"][:8]
    family_imgs = []
    for fam in test_families[:4]:
        root = fam["source_root"]
        for img_row in list((r for r in read_csv(PHASE26 / "confirmed_duplicate_groups.csv")
                             if r["source_root"] == root))[:3]:
            fname = img_row["filename"]
            split = img_row["split"]
            p = RAW_CHENAB / split / "images" / fname
            if p.is_file():
                family_imgs.append((p, f"{split}:{fname[:25]}"))
    if family_imgs:
        make_contact_sheet(family_imgs, contact_dir / "p4_test_touching_families.png",
                           cols=4, thumb=200,
                           title="Priority 4: Source families touching test split (sample)")

    write_csv(PHASE26 / "visual_review_records.csv", review_records,
              ["review_id", "priority", "pair_type", "file1", "split1", "root1",
               "file2", "split2", "hamming_distance", "evidence", "contact_sheet",
               "proposed_action", "human_decision", "reviewer", "review_date", "notes"])

    print(f"  Review records created: {len(review_records)}")
    print(f"    Priority 1 (cross-split pHash-0): {len(p0)}")
    print(f"    Priority 2 (cross-split near-dup): {len(p2)}")
    print(f"    Test-touching pairs: {len(test_pairs)}")
    print(f"  Contact sheets: 4")
    return {"review_records": len(review_records)}

# ---------------------------------------------------------------------------
# TASK E – Phase 3 Grouping Proposal
# ---------------------------------------------------------------------------

def task_e_phase3_proposal(family_data: dict, group_data: dict):
    print("=== TASK E – Phase 3 Grouping Proposal ===")

    # Load v1.1 manifest
    manifest = read_csv(META / "master_manifest_v1_1.csv")
    # Load root assignments
    root_rows = read_csv(PHASE26 / "chenab_image_roots.csv")
    root_by_file = {r["filename"]: r for r in root_rows}
    # Load family data
    families = {r["source_root"]: r for r in read_csv(PHASE26 / "source_family_candidates.csv")}
    # Load conflict groups
    conflicts = read_csv(PHASE26 / "cross_split_group_conflicts.csv")
    conflict_roots = {r["source_root"]: r for r in conflicts}

    proposal_rows = []
    excluded_rows = []

    for rec in manifest:
        fname = rec["original_filename"]
        source = rec["source_dataset"]
        split = rec.get("original_split", "")

        if source != "chenab":
            # AITEX: not part of Roboflow family analysis
            proposal_rows.append({
                "image_id": rec["image_id"],
                "filename": fname,
                "source": source,
                "original_split": split,
                "source_root": "",
                "family_status": "not_applicable",
                "proposed_split_assignment": split,
                "block_reason": rec.get("exclusion_reason", ""),
                "training_block": rec.get("training_readiness", "no") != "yes",
                "group_id": "",
                "human_decision_required": rec.get("human_review_status", "") != "not_required",
            })
            continue

        root_info = root_by_file.get(fname)
        root = root_info["source_root"] if root_info else ""
        family = families.get(root, {})
        conflict = conflict_roots.get(root, {})

        # Determine block status
        block_reason = ""
        training_block = False
        proposed_split = split  # preserve original

        if split == "test":
            block_reason = "locked_test_split"
            training_block = True
        elif family.get("touches_test_split") == "True":
            block_reason = "source_family_touches_locked_test_split"
            training_block = True
        elif family.get("cross_split") == "True":
            block_reason = "source_family_spans_multiple_splits_pending_resolution"
            training_block = True
        elif rec.get("training_readiness") not in ("yes",):
            block_reason = rec.get("exclusion_reason", "pending_qc")
            training_block = True

        group_id = ""
        if conflict:
            group_id = conflict.get("group_id", "")

        proposal_rows.append({
            "image_id": rec["image_id"],
            "filename": fname,
            "source": source,
            "original_split": split,
            "source_root": root,
            "family_status": family.get("family_status", "unknown"),
            "proposed_split_assignment": proposed_split,
            "block_reason": block_reason,
            "training_block": training_block,
            "group_id": group_id,
            "human_decision_required": training_block or rec.get("human_review_status", "") != "not_required",
        })

        if training_block and split != "test":
            excluded_rows.append({
                "image_id": rec["image_id"],
                "filename": fname,
                "source": source,
                "original_split": split,
                "source_root": root,
                "block_reason": block_reason,
                "group_id": group_id,
                "resolution_path": (
                    "resolve_cross_split_family_assignment" if "family" in block_reason else
                    "human_annotation_review" if "annotation" in block_reason else
                    "expert_class_verification" if "class" in block_reason else
                    "review_required"
                ),
            })

    write_csv(PHASE26 / "phase3_grouping_proposal.csv", proposal_rows,
              ["image_id", "filename", "source", "original_split", "source_root",
               "family_status", "proposed_split_assignment", "block_reason",
               "training_block", "group_id", "human_decision_required"])
    write_csv(PHASE26 / "phase3_excluded_pending_review.csv", excluded_rows,
              ["image_id", "filename", "source", "original_split", "source_root",
               "block_reason", "group_id", "resolution_path"])

    # Counts
    chenab_total = sum(1 for r in proposal_rows if r["source"] == "chenab")
    chenab_ok = sum(1 for r in proposal_rows if r["source"] == "chenab" and not r["training_block"])
    chenab_blocked = sum(1 for r in proposal_rows if r["source"] == "chenab" and r["training_block"] and r["original_split"] != "test")
    locked = sum(1 for r in proposal_rows if r["original_split"] == "test" and r["source"] == "chenab")

    print(f"  Chenab total: {chenab_total}")
    print(f"  Approved for Phase 3 (no block): {chenab_ok}")
    print(f"  Blocked (non-test): {chenab_blocked}")
    print(f"  Locked test split: {locked}")
    print(f"  Excluded pending review: {len(excluded_rows)}")

    # Readiness markdown
    lines = [
        "# Phase 3 Leakage Readiness",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Status: BLOCKED",
        "",
        "Phase 3 (final split assignment) cannot proceed until cross-split source families are resolved.",
        "",
        "## Summary",
        "",
        f"| Category | Count |",
        f"|----------|-------|",
        f"| Chenab total | {chenab_total} |",
        f"| Unblocked (no leakage risk) | {chenab_ok} |",
        f"| Blocked pending leakage resolution | {chenab_blocked} |",
        f"| Locked test split | {locked} |",
        f"| Source families spanning splits | {family_data['cross_split_roots']} |",
        f"| Families touching test split | {family_data['test_touching_roots']} |",
        f"| Confirmed cross-split groups | {group_data['n_groups']} |",
        f"| Ambiguous pHash edges | {group_data['ambiguous_edges']} |",
        "",
        "## Blockers for Phase 3",
        "",
        "1. **Cross-split source families** — 492 source roots span multiple Chenab splits.",
        "   Every image in such a family must be assigned to the same Phase 3 split.",
        "   This requires a formal split-assignment policy decision.",
        "",
        f"2. **Test-touching families** — {family_data['test_touching_roots']} source roots appear in",
        "   both the locked test split and train/valid. These images must not be used for",
        "   training or calibration until the split policy formally approves their use.",
        "",
        "3. **Ambiguous pHash edges** — Cross-split pairs with different filename roots and",
        f"   pHash distance ≤ 10 ({group_data['ambiguous_edges']} pairs) cannot be auto-resolved.",
        "   Human visual review is required.",
        "",
        "## What Is Safe to Proceed",
        "",
        f"- {chenab_ok} Chenab images with no cross-split family membership can be included",
        "  in Phase 3 subject to annotation QC completion.",
        "- AITEX images are not affected by Roboflow family analysis but remain blocked",
        "  on class verification.",
        "",
        "## Required Human Decisions",
        "",
        "- Visual review of 65 cross-split pHash-0 pairs (`visual_review_records.csv`).",
        "- Visual review of 70 cross-split near-duplicate pairs.",
        "- Split-assignment policy for cross-split source families.",
        "- Decision on images whose source family spans the test split.",
    ]
    write_md(PHASE26 / "phase3_leakage_readiness.md", lines)

    return {"chenab_ok": chenab_ok, "chenab_blocked": chenab_blocked,
            "excluded": len(excluded_rows)}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PHASE26.mkdir(parents=True, exist_ok=True)
    print(f"Phase 2.6 output: {PHASE26}\n")

    dup_stats = task_a_correct_terminology()
    print()
    family_data = task_b_source_families()
    print()
    group_data = task_c_leakage_groups(family_data["root_to_rows"],
                                       family_data["family_rows"], dup_stats)
    print()
    task_d_visual_review(family_data["root_to_rows"])
    print()
    proposal_data = task_e_phase3_proposal(family_data, group_data)

    print("\n=== Phase 2.6 Complete ===")
    print(f"Corrected terminology: pHash-0 cross-split={dup_stats['cs_phash0']}, "
          f"near-dup={dup_stats['cs_near']}, visual-review={dup_stats['cs_review']}")
    print(f"Source families: {family_data['unique_roots']} unique roots, "
          f"{family_data['cross_split_roots']} cross-split")
    print(f"Confirmed groups: {group_data['n_groups']}, "
          f"test conflicts: {group_data['test_conflicts']}")
    print(f"Ambiguous pHash edges: {group_data['ambiguous_edges']}")
    print(f"Phase 3: {proposal_data['chenab_ok']} unblocked, "
          f"{proposal_data['chenab_blocked']} blocked")

    results = {
        "dup": dup_stats, "family": {k: v for k, v in family_data.items()
                                      if not isinstance(v, dict) and k != "root_to_rows"
                                      and k != "family_rows"},
        "group": group_data, "proposal": proposal_data,
    }
    (PHASE26 / "phase2_6_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
