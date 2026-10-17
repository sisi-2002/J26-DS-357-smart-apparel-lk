"""F2.2C-Pilot Task B2 -- deterministic image/instance selection for the
15-20 image SAM/SAM2 segmentation pilot.

Stage: F2.2C-Pilot (SAM segmentation pilot on VALID-split images). This
module is METADATA-ONLY: it reads the authoritative split manifest
(``manifest_loader.load_split_manifest``) and the F2.2A VALID-split audit
CSVs (``results/validation_audit/valid_image_audit.csv`` /
``valid_box_audit.csv``). It never opens an image's pixels -- image pixel
access happens only in the separate mask-generation stage
(``f22c_sam_mask_generation.py``), which is explicitly permitted by the task
brief for the pilot (unlike the metadata-only F2.2A audit).

Split-membership verification
------------------------------
Every one of the 18 selected ``image_id`` values below is verified against
``split_manifest.csv`` (the ONLY authoritative source) to have
``split == "valid"``. This check is load-bearing: several selected
``image_id`` values literally begin with the substring ``chenab_test_`` or
``chenab_train_`` (e.g. ``chenab_test_00003932_jpg.rf....`` is, per the
manifest, actually a VALID-split image) -- F2.1A/F2.2A already documented
that this dataset's filename prefixes do NOT reliably indicate true split
membership, so this module never infers split from the image_id string; it
always looks the record up in the manifest and asserts the ``split`` field.

Selection method (documented, not randomised)
-----------------------------------------------
Candidates were enumerated per defect class from ``valid_box_audit.csv``,
sorted deterministically by ``image_id``, and manually curated against the
task brief's explicit requirements (class coverage, challenging/oversized
boxes beyond just baekra/gray stitch/selvet, elongated/thin defects, both
single- and multi-defect images, distinct ``group_id`` source families,
<=2 images per ``group_id``). The full candidate pools considered and the
rationale for both inclusion and exclusion are recorded in
``manifests/selection_report.json`` and ``manifests/image_selection.csv``.
No Grad-CAM/XAI output of any kind existed at selection time, and none was
consulted -- selection used only box geometry/class/group_id metadata.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.dataset.manifest_loader import (  # noqa: E402
    load_split_manifest,
)

VALIDATION_AUDIT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "validation_audit"
IMAGE_AUDIT_CSV = VALIDATION_AUDIT_DIR / "valid_image_audit.csv"
BOX_AUDIT_CSV = VALIDATION_AUDIT_DIR / "valid_box_audit.csv"

PILOT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "reference_masks" / "pilot_v1"
MANIFESTS_DIR = PILOT_DIR / "manifests"

# --- The 18 selected image_ids, with per-image selection rationale. -------
# Each entry's rationale cites: defect class(es), box-geometry character
# (full-span/large/elongated/normal, per the same w>0.9 / w>=0.98 /
# aspect>=5.0 thresholds F2.2A Task 3 used), single- vs multi-defect, and
# why this particular group_id/image was chosen over siblings in the same
# candidate pool.
SELECTED_IMAGES = [
    {
        "image_id": "chenab_test_IMG-20230817-WA0060_jpg.rf.b39081cd04cc1a6bf465fb89a5254b2a",
        "rationale": (
            "baekra, single-defect, full-span box (w or h >= 0.98). baekra has "
            "the highest full-span rate of any class (25.0% per F2.2A Task 3) "
            "-- this is the class's characteristic challenging case."
        ),
    },
    {
        "image_id": "chenab_test_IMG_20230731_121423_jpg.rf.8d63939b65ca55211b9819ab88a03acb",
        "rationale": (
            "baekra, single-defect, normal (non-oversized) box -- paired "
            "counter-example to the full-span baekra case above, so the class "
            "is not represented only by its hardest case."
        ),
    },
    {
        "image_id": "chenab_train_20230712_163620_jpeg_jpg.rf.2ed46bd9534cd0b2228fff1ee46ca554",
        "rationale": (
            "color issues, multi-defect (2 boxes), one full-span box. "
            "Demonstrates that oversized/full-span boxes are not exclusive to "
            "baekra/gray stitch/selvet -- color issues shows the same pattern "
            "at a lower rate (8.6% full-span per F2.2A Task 3)."
        ),
    },
    {
        "image_id": "chenab_train_20230719_111738_jpeg_jpg.rf.9191f40d3ab89f40cb905f89f3406505",
        "rationale": "color issues, single-defect, normal box -- easy-case counterpart.",
    },
    {
        "image_id": "chenab_train_IMG-20230907-WA0041_jpg.rf.4c88f25dc0cc916f8cddcad68667606c",
        "rationale": (
            "contamination, single-defect, elongated box (aspect ratio >= 5.0). "
            "contamination has zero large/full-span boxes but a non-trivial "
            "elongated share (17.9%) -- a geometrically distinct challenging "
            "signature from baekra/gray stitch/selvet, per F2.2A Task 3 Sec 2."
        ),
    },
    {
        "image_id": "chenab_train_IMG-20230906-WA0070_jpg.rf.d33f74a1611e49180274d6f370cd6a27",
        "rationale": "contamination, multi-defect (2 boxes), normal geometry -- easy-case counterpart with multi-defect coverage.",
    },
    {
        "image_id": "chenab_train_20230712_141546_jpeg_jpg.rf.009d497c13d53d48c37dced25ab5e828",
        "rationale": (
            "cut, single-defect, full-span box. cut is normally the class with "
            "the lowest large/full-span rate (9.0%/2.6%) -- included specifically "
            "to cover a cut outlier rather than assuming the class is always easy."
        ),
    },
    {
        "image_id": "chenab_train_20230712_141552_jpeg_jpg.rf.015bfa40d3d292429c7442a12c6a7406",
        "rationale": "cut, multi-defect (2 boxes), one large (non-full-span) box -- a second, distinct cut challenging case/group_id.",
    },
    {
        "image_id": "chenab_test_006843_jpg.rf.7d6b35a29774a248e7848185c8764286",
        "rationale": "cut, multi-defect (2 boxes), normal geometry -- easy-case counterpart for the class.",
    },
    {
        "image_id": "chenab_test_00003932_jpg.rf.cabbaf58e5c34db2ffe2df66ad965f44",
        "rationale": (
            "gray stitch, single-defect, full-span AND elongated box. gray "
            "stitch has the highest full-span rate among all classes (41.3%) "
            "and a high elongated rate (27.0%) -- this instance exhibits both."
        ),
    },
    {
        "image_id": "chenab_train_00004515_jpg.rf.053441765bd145dffe80b947db2f2d2f",
        "rationale": "gray stitch, single-defect, full-span AND elongated box -- second, distinct group_id/source-family example of the class's dominant failure mode.",
    },
    {
        "image_id": "chenab_test_augmented_00004155_jpg.rf.ada0517f9bc68574c87a5c03decec691",
        "rationale": "gray stitch, multi-defect (2 boxes), normal geometry -- easy-case/multi-defect counterpart for the class.",
    },
    {
        "image_id": "chenab_test_IMG_20230823_105639_jpg.rf.03ba242a6866d033b9c4878dfb907b48",
        "rationale": (
            "selvet, multi-defect (5 boxes -- the densest instance count in the "
            "pilot), one large box. selvet has the highest overall large-box "
            "rate (30.6%) and a high elongated rate (31.9%); this image stresses "
            "multi-instance mask generation within one frame."
        ),
    },
    {
        "image_id": "chenab_train_20230719_121621_jpeg_jpg.rf.146a27505a26b427877e4dd45eda861c",
        "rationale": (
            "selvet, multi-defect (2 boxes), one elongated+small box. A sibling "
            "image from the same group_id (sg_0156) family was available but "
            "excluded (see selection_report.json exclusions) to respect the "
            "per-family cap while still covering selvet's elongated signature."
        ),
    },
    {
        "image_id": "chenab_test_Snapchat-1013626309_jpg.rf.f733c93408113dc65c3386fbd8451955",
        "rationale": "selvet, single-defect, normal geometry -- easy-case counterpart, distinct group_id (sg_0034).",
    },
    {
        "image_id": "chenab_train_IMG-20230817-WA0064_jpg.rf.72c55fd89707b95438444ffec8b140ac",
        "rationale": (
            "stain, single-defect, large (non-full-span) box. stain is the "
            "largest class (126/345 positive VALID images) but has a low "
            "large-box rate (3.4%) -- included to cover its less-common "
            "challenging case."
        ),
    },
    {
        "image_id": "chenab_train_IMG-20231012-WA0073_jpg.rf.5429ad20d40fb0b199cf338896923264",
        "rationale": "stain, multi-defect (2 boxes), elongated box -- covers stain's thin/elongated signature (3.4% elongated rate) with multi-defect coverage.",
    },
    {
        "image_id": "chenab_test_IMG-20231003-WA0007_jpg.rf.0099a2437e3b5694183d79765c353dcf",
        "rationale": "stain, single-defect, small box (area < 0.01) -- easy/small-case counterpart for the dominant class.",
    },
]

EXCLUSION_NOTES = [
    {
        "reason_category": "sibling in already-selected group_id family",
        "excluded_candidates": [
            "chenab_train_20230719_121621_jpeg_jpg.rf.6b33cb0aefde21affdf061406bb684b9",
            "chenab_train_20230719_121621_jpeg_jpg.rf.61957e9ff9f0f7aeea5f5d14dc47ebf2",
        ],
        "detail": (
            "Both belong to group_id sg_0156, the same source family as the "
            "selected selvet instance chenab_train_20230719_121621_jpeg_jpg."
            "rf.146a27505a26b427877e4dd45eda861c. To respect the <=2-per-"
            "group_id cap while still prioritizing distinct source families "
            "across the whole 18-image pilot, only one sg_0156 instance was kept."
        ),
    },
    {
        "reason_category": "redundant geometry already represented, budget constraint",
        "excluded_candidates": [
            "many additional gray-stitch full-span/elongated candidates (28 "
            "total challenging candidates existed for this class alone)",
            "many additional stain normal-box candidates (116 of stain's 126 "
            "candidates are normal, non-challenging geometry)",
        ],
        "detail": (
            "With a 15-20 image budget across 7 classes, each class received "
            "2-3 slots split between a challenging and a normal representative. "
            "Classes with very large candidate pools (stain: 126 images; gray "
            "stitch: 45 images) necessarily had the large majority of their "
            "candidates excluded; this is a budget/coverage tradeoff, not a "
            "quality judgment about the excluded images."
        ),
    },
    {
        "reason_category": "confirmed-negative images excluded entirely",
        "excluded_candidates": ["all 32 confirmed_negative VALID images"],
        "detail": (
            "Task brief requires POSITIVE VALID images only (images with >=1 "
            "GT box) -- confirmed-negative images have no GT box and so no "
            "box-prompted SAM instance applies to them."
        ),
    },
]


def _flags(w: float, h: float, area: float) -> dict:
    full_span = w >= 0.98 or h >= 0.98
    large = (w > 0.9 or h > 0.9) and not full_span
    aspect = max(w, h) / max(min(w, h), 1e-9)
    elongated = aspect >= 5.0
    small = area < 0.01
    return {
        "full_span": full_span, "large": large, "elongated": elongated,
        "small": small, "aspect_ratio": round(aspect, 4),
    }


def run() -> dict:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    split_lookup = {r.image_id: r for r in load_split_manifest()}

    with open(IMAGE_AUDIT_CSV, newline="", encoding="utf-8") as f:
        image_audit = {r["image_id"]: r for r in csv.DictReader(f)}
    with open(BOX_AUDIT_CSV, newline="", encoding="utf-8") as f:
        box_rows = list(csv.DictReader(f))
    boxes_by_image: dict[str, list[dict]] = defaultdict(list)
    for b in box_rows:
        boxes_by_image[b["image_id"]].append(b)

    selection_rows = []
    instance_rows = []
    class_coverage: Counter = Counter()
    group_id_seen: Counter = Counter()
    n_multi = 0
    n_single = 0
    n_challenging_instances = 0
    verification_failures = []

    for entry in SELECTED_IMAGES:
        image_id = entry["image_id"]

        # --- Hard split-membership check: manifest ONLY, never filename. ---
        split_record = split_lookup.get(image_id)
        if split_record is None:
            verification_failures.append(f"{image_id}: not found in split_manifest.csv at all")
            continue
        if split_record.split != "valid":
            verification_failures.append(
                f"{image_id}: split_manifest.csv says split={split_record.split!r}, not 'valid' -- EXCLUDED"
            )
            continue

        audit_row = image_audit.get(image_id)
        if audit_row is None:
            verification_failures.append(f"{image_id}: not found in valid_image_audit.csv")
            continue

        group_id = split_record.group_id
        group_id_seen[group_id] += 1

        is_multi = audit_row["is_multi_defect"] == "True"
        if is_multi:
            n_multi += 1
        else:
            n_single += 1

        boxes = sorted(boxes_by_image.get(image_id, []), key=lambda b: int(b["line_no"]))
        classes_present = sorted({b["class_name"] for b in boxes})
        for c in classes_present:
            class_coverage[c] += 1

        any_challenging = False
        for idx, b in enumerate(boxes, start=1):
            w, h, area = float(b["w"]), float(b["h"]), float(b["area"])
            flags = _flags(w, h, area)
            if flags["full_span"] or flags["large"] or flags["elongated"]:
                any_challenging = True
                n_challenging_instances += 1
            instance_id = f"{image_id}__inst_{idx:03d}"
            instance_rows.append({
                "instance_id": instance_id,
                "image_id": image_id,
                "filename": split_record.filename,
                "group_id": group_id,
                "split": split_record.split,
                "line_no": b["line_no"],
                "class_id": b["class_id"],
                "class_name": b["class_name"],
                "cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"], "area": b["area"],
                **flags,
            })

        selection_rows.append({
            "image_id": image_id,
            "filename": split_record.filename,
            "split_verified": split_record.split,
            "group_id": group_id,
            "classes_present": json.dumps(classes_present),
            "num_gt_boxes": len(boxes),
            "is_multi_defect": is_multi,
            "any_challenging_box": any_challenging,
            "selection_rationale": entry["rationale"],
        })

    if verification_failures:
        raise RuntimeError(
            "F2.2C-Pilot selection aborted -- split-membership verification "
            "failed for one or more selected images:\n" + "\n".join(verification_failures)
        )

    group_cap_violations = {g: n for g, n in group_id_seen.items() if n > 2}

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "F2.2C-Pilot Task B2",
        "purpose": "Deterministic selection of 15-20 positive VALID images for the SAM/SAM2 segmentation pilot",
        "n_images_selected": len(selection_rows),
        "n_instances_selected": len(instance_rows),
        "n_single_defect_images": n_single,
        "n_multi_defect_images": n_multi,
        "n_challenging_instances": n_challenging_instances,
        "n_normal_instances": len(instance_rows) - n_challenging_instances,
        "defect_class_coverage": dict(class_coverage),
        "all_7_classes_covered": len(class_coverage) == 7,
        "n_distinct_group_ids": len(group_id_seen),
        "group_id_cap_respected_le_2_per_group": len(group_cap_violations) == 0,
        "group_id_cap_violations": group_cap_violations,
        "exclusion_notes": EXCLUSION_NOTES,
        "no_grad_cam_or_xai_output_consulted": True,
        "selection_done_before_any_mask_generation": True,
    }

    sel_fieldnames = list(selection_rows[0].keys())
    with open(MANIFESTS_DIR / "image_selection.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sel_fieldnames)
        writer.writeheader()
        writer.writerows(selection_rows)

    inst_fieldnames = list(instance_rows[0].keys())
    with open(MANIFESTS_DIR / "instance_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=inst_fieldnames)
        writer.writeheader()
        writer.writerows(instance_rows)

    with open(MANIFESTS_DIR / "selection_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run()
    print(json.dumps(rep, indent=2))
