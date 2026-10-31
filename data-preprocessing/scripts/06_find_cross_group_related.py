from pathlib import Path
from datetime import datetime
from collections import defaultdict
import csv
import itertools
import math

from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "fabricDD.v2i.yolov8"
REPORTS_DIR = PROJECT_DIR / "reports"
ACTIVE_REVIEW = (
    REPORTS_DIR / "contact_sheets_20261008_100538_600152" / "group_review.updated.csv"
)
MEMBERS_CSV = (
    REPORTS_DIR
    / "related_images_20261008_095455_744519"
    / "candidate_group_members.csv"
)

SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rel_image(path):
    return str(path.relative_to(DATASET_DIR)).replace("\\", "/")


def hamming(left, right):
    return (left ^ right).bit_count()


def ahash(image, size=8):
    gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= avg)
    return value


def dhash(image, size=8):
    gray = ImageOps.grayscale(image).resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for y in range(size):
        offset = y * (size + 1)
        for x in range(size):
            value = (value << 1) | int(pixels[offset + x] > pixels[offset + x + 1])
    return value


def signature(path):
    variants = []
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        mean = image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
        for angle in (0, 90, 180, 270):
            rotated = image.rotate(angle, expand=True)
            for mirrored, variant in ((False, rotated), (True, ImageOps.mirror(rotated))):
                variants.append({
                    "transform": f"rot{angle}{'_mirror' if mirrored else ''}",
                    "ahash": ahash(variant),
                    "dhash": dhash(variant),
                })
    return {
        "width": width,
        "height": height,
        "aspect": width / height if height else 0,
        "mean_rgb": mean,
        "variants": variants,
    }


def color_distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def best_score(left, right):
    best = None
    for lv in left["variants"]:
        for rv in right["variants"]:
            ah = hamming(lv["ahash"], rv["ahash"])
            dh = hamming(lv["dhash"], rv["dhash"])
            score = ah + dh
            if best is None or score < best["hash_distance"]:
                best = {
                    "hash_distance": score,
                    "ahash_distance": ah,
                    "dhash_distance": dh,
                    "left_transform": lv["transform"],
                    "right_transform": rv["transform"],
                }
    best["color_distance"] = color_distance(left["mean_rgb"], right["mean_rgb"])
    best["aspect_ratio_delta"] = abs(left["aspect"] - right["aspect"])
    return best


review_rows = read_csv(ACTIVE_REVIEW)
member_rows = read_csv(MEMBERS_CSV)
review_by_group = {row["group_id"]: row for row in review_rows}
members_by_group = defaultdict(list)
image_to_group = {}

for row in member_rows:
    image = row["image"].replace("\\", "/")
    members_by_group[row["candidate_group_id"]].append({**row, "image": image})
    image_to_group[image] = row["candidate_group_id"]

units = []
for group_id, rows in sorted(members_by_group.items()):
    rows = sorted(rows, key=lambda row: (row["split"], row["image"]))
    review = review_by_group[group_id]
    if group_id == "C00680" or review["decision"] == "different_photos":
        for index, row in enumerate(rows, 1):
            units.append({
                "unit_id": f"{group_id}::member{index}",
                "display_group_id": group_id,
                "representative_image": row["image"],
                "representative_split": row["split"],
                "member_count": 1,
                "source_type": "reviewed_different_photo_member",
            })
    else:
        units.append({
            "unit_id": group_id,
            "display_group_id": group_id,
            "representative_image": rows[0]["image"],
            "representative_split": rows[0]["split"],
            "member_count": len(rows),
            "source_type": f"reviewed_{review['decision']}",
        })

for split in SPLITS:
    image_dir = DATASET_DIR / split / "images"
    for path in sorted(image_dir.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        image = rel_image(path)
        if image in image_to_group:
            continue
        units.append({
            "unit_id": f"UNGROUPED::{image}",
            "display_group_id": "",
            "representative_image": image,
            "representative_split": split,
            "member_count": 1,
            "source_type": "ungrouped",
        })

output_dir = REPORTS_DIR / datetime.now().strftime("cross_group_related_%Y%m%d_%H%M%S_%f")
output_dir.mkdir(parents=True, exist_ok=False)

errors = []
for index, unit in enumerate(units, 1):
    path = DATASET_DIR / Path(unit["representative_image"])
    try:
        unit["signature"] = signature(path)
    except (OSError, ValueError) as error:
        errors.append({"image": unit["representative_image"], "error": str(error)})
    if index % 500 == 0:
        print(f"Read signatures for {index} source units...", flush=True)

units = [unit for unit in units if "signature" in unit]

candidate_rows = []
for index, (left, right) in enumerate(itertools.combinations(units, 2), 1):
    if left["display_group_id"] and left["display_group_id"] == right["display_group_id"]:
        continue
    score = best_score(left["signature"], right["signature"])
    if (
        score["hash_distance"] <= 20
        and score["color_distance"] <= 85
        and score["aspect_ratio_delta"] <= 1.5
    ):
        candidate_rows.append({
            "left_unit_id": left["unit_id"],
            "left_group_id": left["display_group_id"],
            "left_image": left["representative_image"],
            "left_split": left["representative_split"],
            "left_source_type": left["source_type"],
            "right_unit_id": right["unit_id"],
            "right_group_id": right["display_group_id"],
            "right_image": right["representative_image"],
            "right_split": right["representative_split"],
            "right_source_type": right["source_type"],
            "hash_distance": score["hash_distance"],
            "ahash_distance": score["ahash_distance"],
            "dhash_distance": score["dhash_distance"],
            "color_distance": round(score["color_distance"], 2),
            "aspect_ratio_delta": round(score["aspect_ratio_delta"], 3),
            "left_transform": score["left_transform"],
            "right_transform": score["right_transform"],
        })
    if index % 1000000 == 0:
        print(f"Compared {index} source-unit pairs...", flush=True)

candidate_rows.sort(
    key=lambda row: (
        int(row["hash_distance"]),
        float(row["color_distance"]),
        float(row["aspect_ratio_delta"]),
        row["left_image"],
        row["right_image"],
    )
)

write_csv(
    output_dir / "cross_group_similarity_candidates.csv",
    candidate_rows,
    [
        "left_unit_id", "left_group_id", "left_image", "left_split", "left_source_type",
        "right_unit_id", "right_group_id", "right_image", "right_split", "right_source_type",
        "hash_distance", "ahash_distance", "dhash_distance", "color_distance",
        "aspect_ratio_delta", "left_transform", "right_transform",
    ],
)
write_csv(output_dir / "read_errors.csv", errors, ["image", "error"])

print("\nCross-group candidate search completed.")
print(f"Source units considered: {len(units)}")
print(f"Reviewed grouped images represented by groups/members: {len(image_to_group)}")
print(f"Ungrouped source units: {sum(1 for unit in units if unit['source_type'] == 'ungrouped')}")
print(f"Candidate source-unit pairs: {len(candidate_rows)}")
print(f"Read errors: {len(errors)}")
print(f"Output folder: {output_dir}")
print("Dataset files and reviewed reports were not modified.")
