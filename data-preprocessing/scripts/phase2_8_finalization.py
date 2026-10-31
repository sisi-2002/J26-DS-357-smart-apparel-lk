"""
Phase 2.8 - AI-Based Visual Review & Phase 2 Finalization
Parts A-G combined runner.
"""

import csv
import hashlib
import json
import math
import re
import sys
import io
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    import numpy as np
    from PIL import Image, ImageDraw
    import imagehash
    import cv2
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = REPO_ROOT / "data-preprocessing" / "master_fabric_dataset"
RAW_CHENAB = MASTER / "raw" / "chenab" / "fabricDD.v2i.yolov8"
RAW_AITEX = MASTER / "raw" / "aitex"
PHASE25 = MASTER / "reports" / "phase2_5"
PHASE26 = MASTER / "reports" / "phase2_6"
PHASE27 = MASTER / "reports" / "phase2_7"
PHASE28 = MASTER / "reports" / "phase2_8"
META = MASTER / "metadata"
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = {0: "baekra", 1: "color_issues", 2: "contamination",
               3: "cut", 4: "gray_stitch", 5: "selvet", 6: "stain"}

RF_PATTERN = re.compile(
    r'^(augmented_)?(.+?)_jpg\.rf\.[0-9a-f]{32}$', re.IGNORECASE)

NOW = datetime.now().isoformat(timespec="seconds")


def extract_root(stem):
    m = RF_PATTERN.match(stem)
    if m:
        return m.group(2), "rf_pattern"
    idx = stem.lower().find("_jpg.rf.")
    if idx > 0:
        root = stem[:idx]
        if root.lower().startswith("augmented_"):
            root = root[10:]
        return root, "rf_fallback"
    return stem, "unparseable"


def read_csv_file(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_file(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path.name} ({len(rows)} rows)")


def write_md(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {path.name}")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_ssim_gray(img1, img2):
    """Compute SSIM between two grayscale images of same shape."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    I1 = img1.astype(np.float64)
    I2 = img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(I1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(I2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


# ===================================================================
# PART A - AUTOMATED DUPLICATE IMAGE REVIEW
# ===================================================================

def part_a_duplicate_review():
    print("=" * 60)
    print("PART A - Automated Duplicate Image Review")
    print("=" * 60)

    reviews = read_csv_file(PHASE27 / "priority_visual_review.csv")
    fam_ver = read_csv_file(PHASE27 / "source_family_verification.csv")
    img_roots = read_csv_file(PHASE26 / "chenab_image_roots.csv")

    root_by_file = {}
    for r in img_roots:
        root_by_file[r["filename"]] = r["source_root"]

    ver_by_root = {v["source_root"]: v for v in fam_ver}

    # Build a cache of image fingerprints for efficiency
    # We'll compute: SHA-256, pHash, dHash, dimensions, aspect ratio
    print("  Computing image fingerprints...")
    fingerprints = {}
    for split in SPLITS:
        img_dir = RAW_CHENAB / split / "images"
        if not img_dir.is_dir():
            continue
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                try:
                    img = Image.open(p)
                    w, h = img.size
                    ph = imagehash.phash(img)
                    dh = imagehash.dhash(img)
                    fingerprints[p.name] = {
                        "path": p,
                        "width": w, "height": h,
                        "aspect": round(w / h, 4) if h > 0 else 0,
                        "phash": ph, "dhash": dh,
                        "split": split,
                    }
                except Exception:
                    pass

    print(f"  Fingerprinted {len(fingerprints)} images")

    # Process all 1356 review pairs
    decision_rows = []
    stats = Counter()

    for rv in reviews:
        rid = rv["review_id"]
        f1 = Path(rv["file1"]).name
        f2 = Path(rv["file2"]).name
        s1 = rv["split1"]
        s2 = rv["split2"]
        priority = int(rv["priority"])
        phash_dist = int(rv["hamming_distance"])

        fp1 = fingerprints.get(f1)
        fp2 = fingerprints.get(f2)

        # Filename ancestry
        root1 = root_by_file.get(f1, "")
        root2 = root_by_file.get(f2, "")
        same_root = root1 == root2 and root1 != ""

        # Multi-signal comparison
        sha_match = None
        pixel_match = None
        dhash_dist = None
        ssim_val = None
        dim_match = None
        aspect_match = None

        if fp1 and fp2:
            dim_match = (fp1["width"] == fp2["width"] and
                         fp1["height"] == fp2["height"])
            aspect_match = fp1["aspect"] == fp2["aspect"]
            dhash_dist = int(fp1["dhash"] - fp2["dhash"])

            # SHA-256 for identical files
            if phash_dist == 0 and dim_match:
                h1 = sha256_file(fp1["path"])
                h2 = sha256_file(fp2["path"])
                sha_match = (h1 == h2)

                if sha_match:
                    pixel_match = True
                elif dim_match:
                    # Decoded pixel comparison
                    arr1 = np.array(Image.open(fp1["path"]).convert("RGB"))
                    arr2 = np.array(Image.open(fp2["path"]).convert("RGB"))
                    pixel_match = np.array_equal(arr1, arr2)

            # SSIM for close pairs (phash <= 5 and same dimensions)
            if phash_dist <= 5 and dim_match:
                try:
                    g1 = cv2.imread(str(fp1["path"]), cv2.IMREAD_GRAYSCALE)
                    g2 = cv2.imread(str(fp2["path"]), cv2.IMREAD_GRAYSCALE)
                    if g1 is not None and g2 is not None and g1.shape == g2.shape:
                        ssim_val = round(compute_ssim_gray(g1, g2), 4)
                except Exception:
                    pass

        # Decision logic
        evidence_signals = []
        if same_root:
            evidence_signals.append("filename_ancestry")
        if sha_match is True:
            evidence_signals.append("sha256_identical")
        if pixel_match is True:
            evidence_signals.append("pixel_identical")
        if phash_dist == 0:
            evidence_signals.append("phash_identical")
        elif phash_dist <= 5:
            evidence_signals.append("phash_close")
        if dhash_dist is not None and dhash_dist <= 2:
            evidence_signals.append("dhash_very_close")
        elif dhash_dist is not None and dhash_dist <= 5:
            evidence_signals.append("dhash_close")
        if ssim_val is not None and ssim_val >= 0.98:
            evidence_signals.append("ssim_very_high")
        if dim_match:
            evidence_signals.append("same_dimensions")

        n_strong = sum(1 for s in evidence_signals
                       if s in ("sha256_identical", "pixel_identical",
                                "filename_ancestry"))
        n_supporting = len(evidence_signals) - n_strong

        if sha_match or pixel_match:
            decision = "confirmed_same_source"
            confidence = "deterministic"
        elif same_root and phash_dist <= 10:
            decision = "confirmed_same_source"
            confidence = "strong_filename_ancestry"
        elif same_root:
            decision = "confirmed_same_source"
            confidence = "filename_ancestry"
        elif phash_dist == 0 and dhash_dist is not None and dhash_dist <= 3:
            decision = "ai_likely_same_source"
            confidence = "strong_multi_signal"
        elif phash_dist <= 2 and (ssim_val is not None and ssim_val >= 0.95):
            decision = "ai_likely_same_source"
            confidence = "ssim_corroborated"
        elif phash_dist <= 5 and dhash_dist is not None and dhash_dist <= 5:
            decision = "ai_likely_same_source"
            confidence = "moderate_multi_signal"
        elif phash_dist >= 8 and (dhash_dist is None or dhash_dist >= 10):
            decision = "ai_likely_different_source"
            confidence = "weak_similarity_only"
        elif phash_dist >= 6 and (not dim_match if dim_match is not None else True):
            decision = "ai_likely_different_source"
            confidence = "different_dimensions_weak_hash"
        else:
            decision = "uncertain"
            confidence = "insufficient_evidence"

        stats[decision] += 1

        decision_rows.append({
            "review_id": rid,
            "priority": priority,
            "file1": f1, "split1": s1, "root1": root1,
            "file2": f2, "split2": s2, "root2": root2,
            "same_source_root": same_root,
            "phash_distance": phash_dist,
            "dhash_distance": dhash_dist if dhash_dist is not None else "",
            "sha256_match": str(sha_match) if sha_match is not None else "",
            "pixel_match": str(pixel_match) if pixel_match is not None else "",
            "ssim": ssim_val if ssim_val is not None else "",
            "dim_match": str(dim_match) if dim_match is not None else "",
            "aspect_match": str(aspect_match) if aspect_match is not None else "",
            "evidence_signals": "|".join(evidence_signals),
            "n_strong_signals": n_strong,
            "n_supporting_signals": n_supporting,
            "decision": decision,
            "confidence": confidence,
            "reviewer_type": "AI",
            "review_method": "multi_signal_automated",
            "human_verified": "no",
            "review_status": "complete",
        })

    print(f"  Decision summary:")
    for k, v in stats.most_common():
        print(f"    {k}: {v}")

    write_csv_file(PHASE28 / "ai_duplicate_review_decisions.csv", decision_rows,
                   ["review_id", "priority", "file1", "split1", "root1",
                    "file2", "split2", "root2", "same_source_root",
                    "phash_distance", "dhash_distance", "sha256_match",
                    "pixel_match", "ssim", "dim_match", "aspect_match",
                    "evidence_signals", "n_strong_signals", "n_supporting_signals",
                    "decision", "confidence", "reviewer_type", "review_method",
                    "human_verified", "review_status"])

    # Build resolved source groups from confirmed decisions
    # Start from existing filename-ancestry groups, then merge confirmed pairs
    from collections import defaultdict as dd

    # Union-Find for group merging
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Initialize: every image is its own component
    all_files = set()
    for d in decision_rows:
        all_files.add(d["file1"])
        all_files.add(d["file2"])
    for f in all_files:
        parent[f] = f

    # Merge by confirmed same-source
    for d in decision_rows:
        if d["decision"] in ("confirmed_same_source", "ai_likely_same_source"):
            union(d["file1"], d["file2"])

    # Also merge all members of verified filename-ancestry families
    confirmed_groups = read_csv_file(PHASE26 / "confirmed_duplicate_groups.csv")
    by_root = defaultdict(list)
    for g in confirmed_groups:
        by_root[g["source_root"]].append(g["filename"])
        if g["filename"] not in parent:
            parent[g["filename"]] = g["filename"]
    for root, members in by_root.items():
        for m in members[1:]:
            union(members[0], m)

    # Build group membership
    groups = defaultdict(set)
    for f in parent:
        groups[find(f)].add(f)

    # Assign stable group IDs
    group_rows = []
    gid_map = {}
    for idx, (rep, members) in enumerate(sorted(groups.items(), key=lambda x: -len(x[1]))):
        gid = f"sg_{idx+1:04d}"
        gid_map[rep] = gid
        roots = {root_by_file.get(m, "") for m in members}
        splits = set()
        for m in members:
            fp = fingerprints.get(m)
            if fp:
                splits.add(fp["split"])
        for m in sorted(members):
            fp = fingerprints.get(m, {})
            group_rows.append({
                "group_id": gid,
                "filename": m,
                "split": fp.get("split", ""),
                "source_root": root_by_file.get(m, ""),
                "group_size": len(members),
                "group_splits": "|".join(sorted(splits)),
                "cross_split": len(splits) > 1,
                "evidence_type": "filename_ancestry" if root_by_file.get(m, "") in roots else "ai_review",
            })

    # Remaining uncertain edges
    uncertain_rows = [d for d in decision_rows if d["decision"] == "uncertain"]

    write_csv_file(PHASE28 / "resolved_source_groups.csv", group_rows,
                   ["group_id", "filename", "split", "source_root",
                    "group_size", "group_splits", "cross_split", "evidence_type"])
    write_csv_file(PHASE28 / "remaining_uncertain_edges.csv", uncertain_rows,
                   ["review_id", "priority", "file1", "split1", "root1",
                    "file2", "split2", "root2", "phash_distance", "dhash_distance",
                    "decision", "confidence"])

    # Grouping evidence report
    n_groups_cross = sum(1 for rep, m in groups.items()
                         if len({fingerprints.get(f, {}).get("split", "") for f in m} - {""}) > 1)
    lines = [
        "# Grouping Evidence Report",
        f"Generated: {NOW}",
        "",
        "## Decision Summary",
        "",
        "| Decision | Count |",
        "|----------|-------|",
    ]
    for k, v in stats.most_common():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"Total pairs reviewed: {len(decision_rows)}",
        f"Resolved source groups: {len(groups)}",
        f"Cross-split groups: {n_groups_cross}",
        f"Uncertain edges: {len(uncertain_rows)}",
        "",
        "## Evidence Types Used",
        "",
        "- **Filename ancestry**: Roboflow naming convention (`<root>_jpg.rf.<hex>`).",
        "- **SHA-256 identity**: Byte-identical file content.",
        "- **Pixel identity**: Decoded pixel arrays are identical.",
        "- **pHash/dHash**: Perceptual hash distance.",
        "- **SSIM**: Structural similarity index (for same-dimension pairs).",
        "",
        "## Grouping Rules",
        "",
        "- Confirmed groups: filename ancestry OR byte/pixel identity.",
        "- AI-likely groups: multiple corroborating signals without ancestry.",
        "- Uncertain edges: single weak signal only.",
        "- No pHash-only confirmed grouping.",
        "- No transitive merge through uncertain edges.",
    ]
    write_md(PHASE28 / "grouping_evidence_report.md", lines)

    return {
        "total_pairs": len(decision_rows),
        "decisions": dict(stats),
        "resolved_groups": len(groups),
        "cross_split_groups": n_groups_cross,
        "uncertain_edges": len(uncertain_rows),
        "fingerprints": fingerprints,
        "group_rows": group_rows,
        "groups": groups,
        "parent": parent,
        "find": find,
    }


# ===================================================================
# PART B - CHENAB ANNOTATION REVIEW
# ===================================================================

def part_b_annotation_review():
    print("\n" + "=" * 60)
    print("PART B - Chenab Annotation Review")
    print("=" * 60)

    ann_review = read_csv_file(PHASE25 / "chenab_annotation_review.csv")

    review_rows = []
    stats = Counter()

    for rec in ann_review:
        fname = rec["image_file"]
        split = rec["split"]
        raw_ann = rec["raw_annotation"]
        issue = rec["issue_types"]
        correction = rec["correction_type"]
        auto_excl = rec.get("human_review_status") == "auto_excluded_invalid"

        if auto_excl:
            decision = "excluded_unreliable"
            reason = "auto-excluded: zero or negative dimension"
            stats["excluded_unreliable"] += 1
            review_rows.append({
                "record_id": rec["record_id"],
                "split": split,
                "image_file": fname,
                "raw_annotation": raw_ann,
                "issue_type": issue,
                "ai_decision": decision,
                "ai_reasoning": reason,
                "proposed_correction": "",
                "reviewer_type": "AI",
                "human_verified": "no",
            })
            continue

        # Parse the annotation
        parts = raw_ann.strip().split()
        if len(parts) < 5:
            decision = "excluded_unreliable"
            reason = "malformed annotation"
            stats["excluded_unreliable"] += 1
            review_rows.append({
                "record_id": rec["record_id"],
                "split": split,
                "image_file": fname,
                "raw_annotation": raw_ann,
                "issue_type": issue,
                "ai_decision": decision,
                "ai_reasoning": reason,
                "proposed_correction": "",
                "reviewer_type": "AI",
                "human_verified": "no",
            })
            continue

        class_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Whole-image box analysis: load image and check if defect is truly image-wide
        if correction == "whole_image_box_review":
            img_path = RAW_CHENAB / split / "images" / fname
            if img_path.is_file():
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        ih, iw = gray.shape

                        # Texture analysis: check if defect-like features span the image
                        # Use edge density as a proxy for defect presence
                        edges = cv2.Canny(gray, 50, 150)
                        edge_density = np.count_nonzero(edges) / (iw * ih)

                        # Variance analysis: defective fabric often has high local variance
                        local_std = cv2.GaussianBlur(
                            (gray.astype(float) - cv2.GaussianBlur(gray.astype(float), (15, 15), 5)) ** 2,
                            (15, 15), 5) ** 0.5

                        # Check quadrant uniformity - if all quadrants show similar
                        # patterns, the annotation is plausible for image-wide defects
                        q_means = []
                        for qi in range(2):
                            for qj in range(2):
                                q = gray[qi*ih//2:(qi+1)*ih//2, qj*iw//2:(qj+1)*iw//2]
                                q_means.append(float(np.mean(q)))
                        q_std = float(np.std(q_means))

                        # For fabric defects like baekra (class 0) and color_issues (1),
                        # whole-image annotations are common when the entire fabric piece
                        # exhibits the defect. Accept if quadrants are relatively uniform.
                        class_name = CLASS_NAMES.get(class_id, "unknown")

                        if q_std < 25:
                            decision = "ai_accepted"
                            reason = (f"Whole-image box for {class_name}: quadrant uniformity "
                                     f"suggests defect spans image (q_std={q_std:.1f}, "
                                     f"edge_density={edge_density:.3f})")
                        else:
                            decision = "uncertain"
                            reason = (f"Whole-image box for {class_name}: quadrant variation "
                                     f"suggests defect may be localized (q_std={q_std:.1f}). "
                                     f"Cannot propose precise bbox without expert input.")
                    else:
                        decision = "uncertain"
                        reason = "image could not be loaded"
                except Exception as e:
                    decision = "uncertain"
                    reason = f"image analysis error: {str(e)[:50]}"
            else:
                decision = "uncertain"
                reason = "image file not found"
            stats[decision] += 1

        elif "suspiciously_small" in issue:
            decision = "uncertain"
            reason = ("Very small annotation. Cannot determine if it represents "
                     "a genuine micro-defect or annotation error without expert review.")
            stats[decision] += 1

        else:
            decision = "uncertain"
            reason = f"Unhandled issue type: {issue}"
            stats[decision] += 1

        review_rows.append({
            "record_id": rec["record_id"],
            "split": split,
            "image_file": fname,
            "raw_annotation": raw_ann,
            "issue_type": issue,
            "ai_decision": decision,
            "ai_reasoning": reason,
            "proposed_correction": "",
            "reviewer_type": "AI",
            "human_verified": "no",
        })

    print(f"  Annotation review decisions:")
    for k, v in stats.most_common():
        print(f"    {k}: {v}")

    write_csv_file(PHASE28 / "ai_chenab_annotation_review.csv", review_rows,
                   ["record_id", "split", "image_file", "raw_annotation",
                    "issue_type", "ai_decision", "ai_reasoning",
                    "proposed_correction", "reviewer_type", "human_verified"])

    # Build reviewed annotation manifest
    # Load all Chenab annotations, mark those reviewed
    reviewed_files = {r["image_file"]: r for r in review_rows}
    manifest_rows = []
    for split in SPLITS:
        lbl_dir = RAW_CHENAB / split / "labels"
        if not lbl_dir.is_dir():
            continue
        for lf in sorted(lbl_dir.rglob("*")):
            if lf.suffix != ".txt":
                continue
            img_name = lf.stem + ".jpg"
            n_annotations = 0
            if lf.stat().st_size > 0:
                n_annotations = sum(1 for line in lf.read_text().strip().split("\n")
                                    if line.strip())
            rv = reviewed_files.get(img_name, {})
            manifest_rows.append({
                "filename": img_name,
                "split": split,
                "n_annotations": n_annotations,
                "ai_review_status": rv.get("ai_decision", "not_reviewed"),
                "issue_type": rv.get("issue_type", ""),
                "annotation_status": (
                    "excluded" if rv.get("ai_decision") == "excluded_unreliable" else
                    "accepted" if rv.get("ai_decision") == "ai_accepted" else
                    "pending_expert_review" if rv.get("ai_decision") == "uncertain" else
                    "clean" if not rv else "clean"
                ),
            })

    write_csv_file(PHASE28 / "reviewed_annotation_manifest.csv", manifest_rows,
                   ["filename", "split", "n_annotations", "ai_review_status",
                    "issue_type", "annotation_status"])

    # Summary
    lines = [
        "# Annotation Decision Summary",
        f"Generated: {NOW}",
        "",
        "## Results",
        "",
        "| Decision | Count |",
        "|----------|-------|",
    ]
    for k, v in stats.most_common():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"Total annotations reviewed: {len(review_rows)}",
        f"Total annotation files: {len(manifest_rows)}",
        "",
        "## Methodology",
        "",
        "- Whole-image boxes: analyzed quadrant uniformity to determine if defect",
        "  plausibly spans the entire image. Accepted where texture is uniform",
        "  (q_std < 25). Marked uncertain where quadrant variation is high.",
        "- Zero-width annotations: auto-excluded (carried from Phase 2.5).",
        "- Small annotations: marked uncertain pending expert review.",
        "- No precise bbox corrections proposed without expert input.",
        "- All decisions are AI-generated, not human-verified.",
    ]
    write_md(PHASE28 / "annotation_decision_summary.md", lines)

    return {
        "total_reviewed": len(review_rows),
        "decisions": dict(stats),
        "total_annotations": len(manifest_rows),
    }


# ===================================================================
# PART C - AITEX CLASS AND MASK REVIEW
# ===================================================================

def part_c_aitex_review():
    print("\n" + "=" * 60)
    print("PART C - AITEX Class and Mask Review")
    print("=" * 60)

    # Step C1: Class verification
    codes = read_csv_file(PHASE25 / "aitex_code_verification.csv")

    # AITEX/AFID class codes from dataset structure and secondary sources
    # The authoritative table from aitex.es/afid is not available.
    # datasetninja.com/afid lists some names but code-to-number mapping
    # is NOT confirmed in the original paper (Silvestre-Blanes et al. 2019).
    # We preserve secondary-source assignments as unverified hints.

    AITEX_CLASS_INFO = {
        "000": {"name": "no_defect", "status": "verified_source_definition",
                "note": "Background images in NODefect_images/"},
        "002": {"name": "broken_end", "status": "unverified_secondary_source",
                "note": "datasetninja.com/afid: broken_end. Code not confirmed in paper."},
        "006": {"name": "broken_pick", "status": "unverified_secondary_source",
                "note": "datasetninja.com/afid: broken_pick. Code not confirmed in paper."},
        "010": {"name": "unknown_010", "status": "unverified_filename_only",
                "note": "No source maps this code. Possible broken_warp_or_weft."},
        "016": {"name": "nep", "status": "unverified_secondary_source",
                "note": "datasetninja.com/afid: nep. Code not confirmed in paper."},
        "019": {"name": "contamination", "status": "unverified_secondary_source",
                "note": "Mapping to Chenab stain REJECTED. Contamination != stain."},
        "022": {"name": "unknown_022", "status": "unverified_filename_only",
                "note": "No authoritative source. Possibly weft-related."},
        "023": {"name": "unknown_023", "status": "unverified_filename_only",
                "note": "Not in any AFID documentation."},
        "025": {"name": "unknown_025", "status": "unverified_filename_only",
                "note": "Possibly weft_crack. No authoritative confirmation."},
        "027": {"name": "unknown_027", "status": "unverified_filename_only",
                "note": "Possibly warp-related. 1 image only."},
        "029": {"name": "unknown_029", "status": "unverified_filename_only",
                "note": "Possibly fuzzyball/knot. 1 image only."},
        "030": {"name": "cut_selvage", "status": "unverified_secondary_source",
                "note": "Mapping to Chenab cut REJECTED. Cut selvage != interior cut."},
        "036": {"name": "crease", "status": "unverified_secondary_source",
                "note": "datasetninja.com/afid: crease. 1 image. Code not confirmed."},
    }

    class_rows = []
    for c in codes:
        code = c["aitex_code"]
        info = AITEX_CLASS_INFO.get(code, {"name": f"unknown_{code}",
                                           "status": "unverified_filename_only",
                                           "note": ""})
        img_count = int(c.get("image_count", 0))

        # Chenab mapping eligibility
        chenab_eligible = False
        mapping_note = ""
        if code == "019":
            mapping_note = "REJECTED: contamination broader than stain"
        elif code == "030":
            mapping_note = "REJECTED: cut selvage semantically different from interior cut"
        else:
            mapping_note = "No Chenab mapping proposed or verified"

        class_rows.append({
            "aitex_code": code,
            "assigned_name": info["name"],
            "verification_status": info["status"],
            "image_count": img_count,
            "source_note": info["note"],
            "chenab_mapping_eligible": chenab_eligible,
            "chenab_mapping_note": mapping_note,
            "approved_for_chenab_training": False,
            "approved_for_xai_ground_truth": code != "000" and info["status"] != "unverified_filename_only",
            "reviewer_type": "AI",
            "human_verified": "no",
        })

    print(f"  AITEX codes reviewed: {len(class_rows)}")
    verified = sum(1 for r in class_rows if r["verification_status"] == "verified_source_definition")
    secondary = sum(1 for r in class_rows if r["verification_status"] == "unverified_secondary_source")
    filename_only = sum(1 for r in class_rows if r["verification_status"] == "unverified_filename_only")
    print(f"    Verified: {verified}, Secondary-source: {secondary}, Filename-only: {filename_only}")

    write_csv_file(PHASE28 / "ai_aitex_class_review.csv", class_rows,
                   ["aitex_code", "assigned_name", "verification_status",
                    "image_count", "source_note", "chenab_mapping_eligible",
                    "chenab_mapping_note", "approved_for_chenab_training",
                    "approved_for_xai_ground_truth", "reviewer_type", "human_verified"])

    # Step C2: Mask review
    masks = read_csv_file(PHASE25 / "aitex_mask_qc.csv")
    verified_masks = read_csv_file(PHASE25 / "verified_mask_candidates.csv")

    # Defect images directory
    defect_dir = RAW_AITEX / "Defect_images"
    mask_dir = RAW_AITEX / "Mask_images"

    mask_rows = []
    eligibility_rows = []

    for mq in masks:
        fname = mq["image_filename"]
        stem = mq["image_stem"]
        qc = mq["qc_status"]
        n_masks = int(mq.get("n_mask_candidates", 0))
        dim_match = mq.get("dimension_match", "") == "match"
        fg_pixels = int(mq.get("foreground_pixels", 0)) if mq.get("foreground_pixels", "").isdigit() else 0
        empty = mq.get("empty_mask", "") == "True"
        issues = mq.get("issues", "")

        # Determine mask validity
        if qc == "failed_no_mask":
            mask_decision = "excluded_no_mask"
            mask_note = "No mask file found"
            xai_eligible = False
        elif qc == "multiple_masks_needs_review":
            # Check if masks are complementary or alternative
            mask_fnames = mq.get("mask_filenames", "").split("|")
            mask_decision = "uncertain_multiple_masks"
            mask_note = f"Multiple masks ({len(mask_fnames)}): need decision on which to use"
            xai_eligible = False  # cannot include without human choice
        elif empty:
            mask_decision = "excluded_empty_mask"
            mask_note = "Mask has no foreground pixels"
            xai_eligible = False
        elif not dim_match:
            mask_decision = "excluded_dimension_mismatch"
            mask_note = "Mask dimensions do not match image"
            xai_eligible = False
        elif qc in ("passed", "passed_with_issues"):
            if fg_pixels < 10:
                mask_decision = "excluded_negligible_foreground"
                mask_note = f"Negligible foreground ({fg_pixels} pixels)"
                xai_eligible = False
            elif "intermediate_values" in issues or "grayscale" in issues.lower():
                mask_decision = "ai_accepted_with_caveat"
                mask_note = "Passed but has intermediate grayscale values; thresholded at 127"
                xai_eligible = True
            else:
                mask_decision = "ai_accepted"
                mask_note = "Clean mask, dimension-matched, sufficient foreground"
                xai_eligible = True
        else:
            mask_decision = "uncertain"
            mask_note = f"Unhandled QC status: {qc}"
            xai_eligible = False

        # Extract AITEX code from filename
        code_match = re.match(r'\d{4}_(\d{3})_\d{2}', stem)
        aitex_code = code_match.group(1) if code_match else ""
        code_info = AITEX_CLASS_INFO.get(aitex_code, {})
        class_verified = code_info.get("status", "") in ("verified_source_definition", "unverified_secondary_source")

        mask_rows.append({
            "image_filename": fname,
            "image_stem": stem,
            "aitex_code": aitex_code,
            "n_masks": n_masks,
            "dimension_match": dim_match,
            "foreground_pixels": fg_pixels,
            "empty_mask": empty,
            "qc_status": qc,
            "mask_decision": mask_decision,
            "mask_note": mask_note,
            "xai_mask_eligible": xai_eligible,
            "class_status": code_info.get("status", "unknown"),
            "reviewer_type": "AI",
            "human_verified": "no",
        })

        eligibility_rows.append({
            "image_filename": fname,
            "aitex_code": aitex_code,
            "class_name": code_info.get("name", f"unknown_{aitex_code}"),
            "class_verified": class_verified,
            "mask_eligible": xai_eligible,
            "bbox_available": True,  # geometry files exist
            "f2_mask_xai_ready": xai_eligible and class_verified,
            "f2_bbox_xai_ready": class_verified,
            "f2_background_ready": aitex_code == "000",
            "exclusion_reason": mask_note if not xai_eligible else "",
        })

    mask_stats = Counter(r["mask_decision"] for r in mask_rows)
    print(f"  Mask review decisions:")
    for k, v in mask_stats.most_common():
        print(f"    {k}: {v}")

    write_csv_file(PHASE28 / "ai_aitex_mask_review.csv", mask_rows,
                   ["image_filename", "image_stem", "aitex_code", "n_masks",
                    "dimension_match", "foreground_pixels", "empty_mask",
                    "qc_status", "mask_decision", "mask_note",
                    "xai_mask_eligible", "class_status", "reviewer_type",
                    "human_verified"])
    write_csv_file(PHASE28 / "aitex_mask_eligibility.csv", eligibility_rows,
                   ["image_filename", "aitex_code", "class_name", "class_verified",
                    "mask_eligible", "bbox_available", "f2_mask_xai_ready",
                    "f2_bbox_xai_ready", "f2_background_ready", "exclusion_reason"])

    # Function 2 readiness summary
    f2_mask_ready = sum(1 for r in eligibility_rows if r["f2_mask_xai_ready"])
    f2_bbox_ready = sum(1 for r in eligibility_rows if r["f2_bbox_xai_ready"])
    f2_bg_ready = sum(1 for r in eligibility_rows if r["f2_background_ready"])
    total_defect = sum(1 for r in eligibility_rows if r["aitex_code"] != "000")

    lines = [
        "# Function 2 Mask Readiness Report",
        f"Generated: {NOW}",
        "",
        "## Summary",
        "",
        "| Category | Count |",
        "|----------|-------|",
        f"| Total AITEX defect images | {total_defect} |",
        f"| Mask-XAI ready (mask + class verified) | {f2_mask_ready} |",
        f"| BBox-XAI ready (class verified, no mask required) | {f2_bbox_ready} |",
        f"| Background ready (no-defect images) | {f2_bg_ready} |",
        "",
        "## Mask Decision Breakdown",
        "",
        "| Decision | Count |",
        "|----------|-------|",
    ]
    for k, v in mask_stats.most_common():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Class Verification Status",
        "",
        f"- Verified (000=no_defect): {verified}",
        f"- Unverified secondary source: {secondary}",
        f"- Unverified filename only: {filename_only}",
        "",
        "## Limitations",
        "",
        "- No AITEX code is approved for Chenab 7-class detector training.",
        "- Rejected mappings preserved: 019->stain, 030->cut.",
        "- Multi-mask images (0044, 0097) excluded from high-confidence XAI subset.",
        "- All decisions are AI-reviewed, not human-verified.",
        "- Authoritative AFID class table from aitex.es/afid not available.",
    ]
    write_md(PHASE28 / "function2_mask_readiness.md", lines)

    return {
        "codes_reviewed": len(class_rows),
        "verified_codes": verified,
        "secondary_codes": secondary,
        "filename_only_codes": filename_only,
        "mask_decisions": dict(mask_stats),
        "f2_mask_ready": f2_mask_ready,
        "f2_bbox_ready": f2_bbox_ready,
        "f2_bg_ready": f2_bg_ready,
    }


# ===================================================================
# PART D - REGENERATE POLICY B SIMULATION
# ===================================================================

def part_d_policy_b(dup_data):
    print("\n" + "=" * 60)
    print("PART D - Regenerate Policy B Simulation")
    print("=" * 60)

    img_roots = read_csv_file(PHASE26 / "chenab_image_roots.csv")
    manifest = read_csv_file(META / "master_manifest_v1_1.csv")
    fam_ver = read_csv_file(PHASE27 / "source_family_verification.csv")
    resolved_groups = read_csv_file(PHASE28 / "resolved_source_groups.csv")

    root_to_imgs = defaultdict(list)
    for r in img_roots:
        root_to_imgs[r["source_root"]].append(r)

    ver_by_root = {v["source_root"]: v for v in fam_ver}
    mf_by_name = {r["original_filename"]: r for r in manifest}

    # Get annotation class counts
    img_classes = {}
    for split in SPLITS:
        lbl_dir = RAW_CHENAB / split / "labels"
        if not lbl_dir.is_dir():
            continue
        for lf in lbl_dir.rglob("*"):
            if lf.suffix != ".txt":
                continue
            cc = Counter()
            if lf.stat().st_size > 0:
                for line in lf.read_text().strip().split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cc[int(parts[0])] += 1
            img_classes[lf.stem + ".jpg"] = cc

    # Use resolved source groups for group-disjoint allocation
    # Map each filename to its resolved group
    file_to_group = {}
    for g in resolved_groups:
        file_to_group[g["filename"]] = g["group_id"]

    # Build allocation groups
    group_members = defaultdict(list)
    for r in img_roots:
        gid = file_to_group.get(r["filename"], f"single_{r['source_root']}")
        group_members[gid].append(r)

    # Build group metadata
    group_list = []
    for gid, members in sorted(group_members.items()):
        cc_total = Counter()
        for m in members:
            for cid, cnt in img_classes.get(m["filename"], Counter()).items():
                cc_total[cid] += cnt
        splits = {m["split"] for m in members}
        touches_test = "test" in splits

        group_list.append({
            "gid": gid,
            "members": members,
            "size": len(members),
            "class_counts": cc_total,
            "dominant_class": cc_total.most_common(1)[0][0] if cc_total else -1,
            "touches_old_test": touches_test,
            "original_splits": splits,
        })

    # Check for uncertain edges that might link groups
    uncertain = read_csv_file(PHASE28 / "remaining_uncertain_edges.csv")
    uncertain_files = set()
    for u in uncertain:
        uncertain_files.add(u["file1"])
        uncertain_files.add(u["file2"])

    # Class-stratified group-disjoint allocation (70/15/15)
    total_imgs = sum(g["size"] for g in group_list)
    target_ratios = {"train": 0.70, "valid": 0.15, "test": 0.15}
    target = {s: total_imgs * r for s, r in target_ratios.items()}

    by_class = defaultdict(list)
    for g in group_list:
        by_class[g["dominant_class"]].append(g)
    for cid in by_class:
        by_class[cid].sort(key=lambda g: -g["size"])

    b_assignments = {}
    running = {"train": 0, "valid": 0, "test": 0}

    for cid in sorted(by_class.keys()):
        groups = by_class[cid]
        class_total = sum(g["size"] for g in groups)
        class_target = {s: class_total * r for s, r in target_ratios.items()}
        class_running = {"train": 0, "valid": 0, "test": 0}

        for g in groups:
            deficits = {s: class_target[s] - class_running[s] for s in ("train", "valid", "test")}
            global_deficits = {s: target[s] - running[s] for s in ("train", "valid", "test")}
            chosen = max(deficits, key=lambda s: (deficits[s], global_deficits[s]))
            b_assignments[g["gid"]] = chosen
            running[chosen] += g["size"]
            class_running[chosen] += g["size"]

    # Build output rows
    pol_b_rows = []
    b_counts = Counter()
    b_class_counts = {"train": Counter(), "valid": Counter(), "test": Counter(), "quarantined": Counter()}

    quarantine_count = 0
    for r in img_roots:
        gid = file_to_group.get(r["filename"], f"single_{r['source_root']}")
        assignment = b_assignments.get(gid, "train")
        cc = img_classes.get(r["filename"], Counter())
        mf = mf_by_name.get(r["filename"], {})
        vf = ver_by_root.get(r["source_root"], {})
        touches_old_test = vf.get("touches_test", False)

        # Quarantine uncertain files that touch different groups
        is_uncertain = r["filename"] in uncertain_files
        quarantined = False

        hist_exposure = (
            "yes_old_test" if r["split"] == "test" else
            "family_member_in_old_test" if touches_old_test else
            "no"
        )

        status = "group_allocated"
        if is_uncertain and assignment == "test":
            status = "quarantined_uncertain_test_candidate"
            quarantined = True
            quarantine_count += 1
            b_counts["quarantined"] += 1
            for cid, cnt in cc.items():
                b_class_counts["quarantined"][cid] += cnt
        else:
            b_counts[assignment] += 1
            for cid, cnt in cc.items():
                b_class_counts[assignment][cid] += cnt

        pol_b_rows.append({
            "filename": r["filename"],
            "original_split": r["split"],
            "source_root": r["source_root"],
            "resolved_group_id": gid,
            "policy_b_assignment": "quarantined" if quarantined else assignment,
            "policy_b_status": status,
            "historical_test_exposure": hist_exposure,
            "has_uncertain_edge": is_uncertain,
            "annotation_classes": "|".join(f"{c}:{n}" for c, n in sorted(cc.items())),
        })

    print(f"  Policy B allocation:")
    for s in ("train", "valid", "test", "quarantined"):
        print(f"    {s}: {b_counts.get(s, 0)}")
    print(f"  Uncertain files quarantined from test: {quarantine_count}")

    write_csv_file(PHASE28 / "policy_b_reviewed_simulation.csv", pol_b_rows,
                   ["filename", "original_split", "source_root",
                    "resolved_group_id", "policy_b_assignment",
                    "policy_b_status", "historical_test_exposure",
                    "has_uncertain_edge", "annotation_classes"])

    # Group disjointness validation
    group_splits = defaultdict(set)
    for r in pol_b_rows:
        a = r["policy_b_assignment"]
        if a != "quarantined":
            group_splits[r["resolved_group_id"]].add(a)
    violations = {g: s for g, s in group_splits.items() if len(s) > 1}

    disjoint_rows = []
    for gid, splits in sorted(group_splits.items()):
        disjoint_rows.append({
            "group_id": gid,
            "assigned_splits": "|".join(sorted(splits)),
            "n_splits": len(splits),
            "disjoint": len(splits) == 1,
        })

    write_csv_file(PHASE28 / "group_disjointness_validation.csv", disjoint_rows,
                   ["group_id", "assigned_splits", "n_splits", "disjoint"])
    print(f"  Group-disjoint violations: {len(violations)}")

    # Historical test exposure report
    old_test_in_new_test = sum(1 for r in pol_b_rows
                               if r["historical_test_exposure"] == "yes_old_test"
                               and r["policy_b_assignment"] == "test")
    family_exposed_in_new_test = sum(1 for r in pol_b_rows
                                     if r["historical_test_exposure"] == "family_member_in_old_test"
                                     and r["policy_b_assignment"] == "test")

    lines = [
        "# Historical Test Exposure Report",
        f"Generated: {NOW}",
        "",
        "## Summary",
        "",
        f"- Old test images now in new test: {old_test_in_new_test}",
        f"- Family-exposed images now in new test: {family_exposed_in_new_test}",
        "",
        "## Risk Assessment",
        "",
        "Any image from a source family that touched the old test split has",
        "potential historical model-training exposure. The existing trained model",
        "was trained on augmented variants from the same source families.",
        "",
        "For genuinely independent evaluation of a new model:",
        "- Use only images whose `historical_test_exposure` = `no`.",
        "- Or retrain from scratch on the new group-disjoint split.",
        "",
        "## Recommendation",
        "",
        "Accept the group-disjoint Policy B split for new experiments.",
        "Report that all prior evaluation results used a contaminated test set.",
        "A new baseline model trained on Policy B is needed for valid comparison.",
    ]
    write_md(PHASE28 / "historical_test_exposure_report.md", lines)

    # Phase 3 exclusion manifest
    excl_rows = []
    for r in pol_b_rows:
        if r["policy_b_assignment"] == "quarantined":
            excl_rows.append({
                "filename": r["filename"],
                "original_split": r["original_split"],
                "source_root": r["source_root"],
                "resolved_group_id": r["resolved_group_id"],
                "exclusion_reason": r["policy_b_status"],
                "resolution_path": "resolve_uncertain_edges_then_reassign",
            })

    write_csv_file(PHASE28 / "phase3_exclusion_manifest.csv", excl_rows,
                   ["filename", "original_split", "source_root",
                    "resolved_group_id", "exclusion_reason", "resolution_path"])

    return {
        "policy_b": dict(b_counts),
        "policy_b_classes": {s: dict(v) for s, v in b_class_counts.items()},
        "disjoint_violations": len(violations),
        "quarantined": quarantine_count,
        "old_test_in_new_test": old_test_in_new_test,
        "family_exposed_in_new_test": family_exposed_in_new_test,
    }


# ===================================================================
# PART E - FINAL MASTER MANIFEST V1.2
# ===================================================================

def part_e_manifest(dup_data, ann_data, aitex_data, policy_data):
    print("\n" + "=" * 60)
    print("PART E - Final Master Manifest V1.2")
    print("=" * 60)

    manifest_v11 = read_csv_file(META / "master_manifest_v1_1.csv")
    dup_decisions = read_csv_file(PHASE28 / "ai_duplicate_review_decisions.csv")
    ann_review = read_csv_file(PHASE28 / "ai_chenab_annotation_review.csv")
    class_review = read_csv_file(PHASE28 / "ai_aitex_class_review.csv")
    mask_review = read_csv_file(PHASE28 / "ai_aitex_mask_review.csv")
    mask_elig = read_csv_file(PHASE28 / "aitex_mask_eligibility.csv")
    pol_b = read_csv_file(PHASE28 / "policy_b_reviewed_simulation.csv")
    fam_ver = read_csv_file(PHASE27 / "source_family_verification.csv")
    img_roots = read_csv_file(PHASE26 / "chenab_image_roots.csv")
    resolved_groups = read_csv_file(PHASE28 / "resolved_source_groups.csv")

    # Index lookup tables
    root_by_file = {r["filename"]: r["source_root"] for r in img_roots}
    ver_by_root = {v["source_root"]: v for v in fam_ver}
    group_by_file = {g["filename"]: g["group_id"] for g in resolved_groups}
    ann_by_file = {r["image_file"]: r for r in ann_review}
    class_by_code = {r["aitex_code"]: r for r in class_review}
    mask_by_file = {r["image_filename"]: r for r in mask_review}
    elig_by_file = {r["image_filename"]: r for r in mask_elig}
    polb_by_file = {r["filename"]: r for r in pol_b}

    # Duplicate review: index decisions by files involved
    dup_decision_by_file = defaultdict(list)
    for d in dup_decisions:
        dup_decision_by_file[d["file1"]].append(d)
        dup_decision_by_file[d["file2"]].append(d)

    v12_rows = []
    for rec in manifest_v11:
        fname = rec["original_filename"]
        source = rec["source_dataset"]
        split = rec.get("original_split", "")

        # Source family
        root = root_by_file.get(fname, "")
        vf = ver_by_root.get(root, {})
        source_family_id = root if root else ""
        resolved_gid = group_by_file.get(fname, "")

        # AI review statuses
        dup_review = dup_decision_by_file.get(fname)
        if dup_review:
            worst_dup = min(dup_review, key=lambda d: (
                0 if d["decision"] == "uncertain" else
                1 if d["decision"] == "ai_likely_same_source" else
                2 if d["decision"] == "confirmed_same_source" else
                3))
            ai_dup_status = worst_dup["decision"]
        else:
            ai_dup_status = "not_reviewed"

        # Annotation QC
        ann = ann_by_file.get(fname, {})
        ann_qc = ann.get("ai_decision", rec.get("annotation_qc_status", ""))
        if not ann_qc:
            ann_qc = "clean"

        # AITEX class and mask
        mask_qc = ""
        class_ver = ""
        f2_mask_ready = False
        f2_bbox_ready = False
        f2_bg_ready = False

        if source == "aitex":
            # Extract code from filename
            code_match = re.match(r'\d{4}_(\d{3})_\d{2}', Path(fname).stem)
            if code_match:
                code = code_match.group(1)
                cv = class_by_code.get(code, {})
                class_ver = cv.get("verification_status", "unverified")
            else:
                class_ver = "unverified"

            mr = mask_by_file.get(fname, {})
            mask_qc = mr.get("mask_decision", rec.get("mask_qc_status", ""))

            el = elig_by_file.get(fname, {})
            f2_mask_ready = str(el.get("f2_mask_xai_ready", "")).lower() == "true"
            f2_bbox_ready = str(el.get("f2_bbox_xai_ready", "")).lower() == "true"
            f2_bg_ready = str(el.get("f2_background_ready", "")).lower() == "true"

            # No-defect images (code 000) are background candidates
            if not el and "NODefect" in str(rec.get("relative_image_path", "")):
                f2_bg_ready = True
                class_ver = "verified_source_definition"
                mask_qc = "not_applicable"
        else:
            class_ver = "chenab_native"
            mask_qc = "not_applicable"

        # Policy B assignment
        pb = polb_by_file.get(fname, {})
        proposed_split = pb.get("policy_b_assignment", "")
        hist_exposure = pb.get("historical_test_exposure", "")

        # Leakage risk
        is_cross = vf.get("cross_split", False)
        touches_test = vf.get("touches_test", False)
        leakage_risk = (
            "high_test_contaminated" if touches_test and split != "test" else
            "locked_test" if split == "test" else
            "medium_cross_split" if is_cross else
            "low"
        )

        # Training readiness (Function 1 - 7-class Chenab detector)
        if source == "aitex":
            f1_ready = False
        elif split == "test":
            f1_ready = False
        elif ann_qc in ("excluded_unreliable",):
            f1_ready = False
        elif rec.get("training_readiness") == "pending_annotation_review":
            f1_ready = ann_qc == "ai_accepted"
        else:
            f1_ready = rec.get("training_readiness") == "yes" or ann_qc == "ai_accepted"

        # Exclusion reason
        exclusion = ""
        if source == "aitex":
            exclusion = "aitex_not_for_chenab_training"
        elif split == "test":
            exclusion = "locked_test_split"
        elif ann_qc == "excluded_unreliable":
            exclusion = "annotation_excluded"
        elif not f1_ready and rec.get("training_readiness") == "pending_annotation_review":
            exclusion = "annotation_uncertain"

        # Verification level
        if ann_qc in ("ai_accepted", "ai_correction_proposed"):
            verification = "ai_reviewed"
        elif ann_qc in ("excluded_unreliable",):
            verification = "excluded"
        elif source == "aitex":
            verification = "mechanically_verified"
        else:
            verification = "mechanically_verified"

        v12_rows.append({
            "image_id": rec["image_id"],
            "source_dataset": source,
            "original_filename": fname,
            "original_split": split,
            "source_family_id": source_family_id,
            "resolved_group_id": resolved_gid,
            "ai_duplicate_review_status": ai_dup_status,
            "annotation_qc_status": ann_qc,
            "mask_qc_status": mask_qc,
            "class_verification_status": class_ver,
            "leakage_risk": leakage_risk,
            "proposed_split_eligibility": proposed_split,
            "historical_model_exposure": hist_exposure,
            "f1_detection_ready": f1_ready,
            "f2_background_ready": f2_bg_ready,
            "f2_bbox_xai_ready": f2_bbox_ready,
            "f2_mask_xai_ready": f2_mask_ready,
            "exclusion_reason": exclusion,
            "verification_level": verification,
            "human_verified": "no",
        })

    write_csv_file(META / "master_manifest_v1_2.csv", v12_rows,
                   ["image_id", "source_dataset", "original_filename",
                    "original_split", "source_family_id", "resolved_group_id",
                    "ai_duplicate_review_status", "annotation_qc_status",
                    "mask_qc_status", "class_verification_status",
                    "leakage_risk", "proposed_split_eligibility",
                    "historical_model_exposure", "f1_detection_ready",
                    "f2_background_ready", "f2_bbox_xai_ready",
                    "f2_mask_xai_ready", "exclusion_reason",
                    "verification_level", "human_verified"])

    f1_ready_count = sum(1 for r in v12_rows if r["f1_detection_ready"])
    f2_mask_count = sum(1 for r in v12_rows if r["f2_mask_xai_ready"])
    f2_bbox_count = sum(1 for r in v12_rows if r["f2_bbox_xai_ready"])
    f2_bg_count = sum(1 for r in v12_rows if r["f2_background_ready"])

    print(f"  Manifest V1.2: {len(v12_rows)} rows")
    print(f"  F1 detection ready: {f1_ready_count}")
    print(f"  F2 mask-XAI ready: {f2_mask_count}")
    print(f"  F2 bbox-XAI ready: {f2_bbox_count}")
    print(f"  F2 background ready: {f2_bg_count}")

    return {
        "total": len(v12_rows),
        "f1_ready": f1_ready_count,
        "f2_mask_ready": f2_mask_count,
        "f2_bbox_ready": f2_bbox_count,
        "f2_bg_ready": f2_bg_count,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    PHASE28.mkdir(parents=True, exist_ok=True)
    print(f"Phase 2.8 output: {PHASE28}\n")

    dup_data = part_a_duplicate_review()
    ann_data = part_b_annotation_review()
    aitex_data = part_c_aitex_review()
    policy_data = part_d_policy_b(dup_data)
    manifest_data = part_e_manifest(dup_data, ann_data, aitex_data, policy_data)

    results = {
        "duplicate_review": {k: v for k, v in dup_data.items()
                             if k not in ("fingerprints", "group_rows", "groups", "parent", "find")},
        "annotation_review": ann_data,
        "aitex_review": aitex_data,
        "policy_b": policy_data,
        "manifest": manifest_data,
    }

    (PHASE28 / "phase2_8_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  Written: phase2_8_results.json")

    print("\n=== Phase 2.8 Parts A-E Complete ===")
    return results


if __name__ == "__main__":
    main()
