from pathlib import Path
from datetime import datetime
from collections import defaultdict
import csv
import hashlib
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "fabricDD.v2i.yolov8"

SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

name_groups = defaultdict(list)
hash_groups = defaultdict(list)
errors = []


def candidate_key(image_path):
    name = image_path.stem

    # Remove only a trailing Roboflow export suffix.
    name = re.sub(r"\.rf\.[0-9a-f]+$", "", name, flags=re.IGNORECASE)

    # Treat explicit augmented_ versions as candidate relatives.
    name = re.sub(r"^(?:augmented_)+", "", name, flags=re.IGNORECASE)

    return name.casefold()


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


for split in SPLITS:
    image_dir = DATASET_DIR / split / "images"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")

    print(f"Checking related-image candidates in {split}...")

    for image_path in sorted(image_dir.rglob("*")):
        if image_path.suffix.lower() not in EXTENSIONS:
            continue

        record = {
            "split": split,
            "image": str(image_path.relative_to(DATASET_DIR)),
        }

        name_groups[candidate_key(image_path)].append(record)

        try:
            hash_groups[file_hash(image_path)].append(record)
        except OSError as error:
            errors.append({
                "image": record["image"],
                "error": str(error),
            })


group_summary = []
candidate_members = []

for key, members in sorted(name_groups.items()):
    if len(members) < 2:
        continue

    group_id = f"C{len(group_summary) + 1:05d}"
    present_splits = sorted({member["split"] for member in members})
    cross_split = len(present_splits) > 1

    group_summary.append({
        "candidate_group_id": group_id,
        "candidate_key": key,
        "image_count": len(members),
        "splits": "|".join(present_splits),
        "cross_split": cross_split,
        "review_status": "needs_visual_review",
    })

    for member in members:
        candidate_members.append({
            "candidate_group_id": group_id,
            "candidate_key": key,
            "split": member["split"],
            "image": member["image"],
            "cross_split": cross_split,
        })


exact_duplicates = []
exact_group_count = 0

for digest, members in sorted(hash_groups.items()):
    if len(members) < 2:
        continue

    exact_group_count += 1
    present_splits = {member["split"] for member in members}

    for member in members:
        exact_duplicates.append({
            "exact_group_id": f"E{exact_group_count:05d}",
            "sha256": digest,
            "split": member["split"],
            "image": member["image"],
            "cross_split": len(present_splits) > 1,
        })


report_dir = (
    PROJECT_DIR / "reports"
    / datetime.now().strftime("related_images_%Y%m%d_%H%M%S_%f")
)
report_dir.mkdir(parents=True, exist_ok=False)

write_csv(
    report_dir / "candidate_group_summary.csv",
    group_summary,
    [
        "candidate_group_id", "candidate_key", "image_count",
        "splits", "cross_split", "review_status",
    ],
)

write_csv(
    report_dir / "candidate_group_members.csv",
    candidate_members,
    [
        "candidate_group_id", "candidate_key",
        "split", "image", "cross_split",
    ],
)

write_csv(
    report_dir / "exact_file_duplicates.csv",
    exact_duplicates,
    ["exact_group_id", "sha256", "split", "image", "cross_split"],
)

write_csv(
    report_dir / "read_errors.csv",
    errors,
    ["image", "error"],
)

print("\nRelated-image candidate search completed.")
print(f"Filename candidate groups: {len(group_summary)}")
print(
    "Candidate groups spanning multiple splits: "
    f"{sum(group['cross_split'] for group in group_summary)}"
)
print(f"Exact file duplicate groups: {exact_group_count}")
print(f"File read errors: {len(errors)}")
print(f"\nReports saved to:\n{report_dir}")
print("\nDataset files were not modified.")