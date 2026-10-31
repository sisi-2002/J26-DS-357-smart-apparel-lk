from pathlib import Path
from datetime import datetime
import csv

# Project folder: the parent of the scripts folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "fabricDD.v2i.yolov8"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")


def main():
    # Check folder structure before counting.
    required = [
        DATASET_DIR / split / folder
        for split in SPLITS
        for folder in ("images", "labels")
    ]
    missing_folders = [path for path in required if not path.is_dir()]

    if missing_folders:
        print("Required folders were not found:")
        for path in missing_folders:
            print(f"  {path}")
        print("\nCheck whether the dataset has an extra nested folder.")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_dir = PROJECT_DIR / "reports" / f"inventory_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=False)

    summary = []
    review_rows = []

    print(f"Dataset: {DATASET_DIR}\n")

    for split in SPLITS:
        image_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"

        images = sorted(
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(
            path for path in label_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".txt"
        )

        expected_labels = {
            label_dir / image.relative_to(image_dir).with_suffix(".txt")
            for image in images
        }

        missing_labels = sorted(
            path for path in expected_labels if not path.is_file()
        )
        orphan_labels = sorted(set(labels) - expected_labels)

        empty_labels = []
        unreadable_labels = []

        for label in labels:
            try:
                if not label.read_text(encoding="utf-8-sig").strip():
                    empty_labels.append(label)
            except (OSError, UnicodeError):
                unreadable_labels.append(label)

        row = {
            "split": split,
            "images": len(images),
            "label_files": len(labels),
            "empty_labels": len(empty_labels),
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
            "unreadable_labels": len(unreadable_labels),
        }
        summary.append(row)

        for status, paths in (
            ("empty_label_needs_review", empty_labels),
            ("missing_label", missing_labels),
            ("label_without_image", orphan_labels),
            ("unreadable_label", unreadable_labels),
        ):
            for path in paths:
                review_rows.append({
                    "split": split,
                    "status": status,
                    "path": str(path.relative_to(DATASET_DIR)),
                })

        print(
            f"{split}: images={row['images']}, "
            f"labels={row['label_files']}, "
            f"empty={row['empty_labels']}, "
            f"missing={row['missing_labels']}, "
            f"orphan={row['orphan_labels']}, "
            f"unreadable={row['unreadable_labels']}"
        )

    with (report_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    with (report_dir / "labels_to_review.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["split", "status", "path"]
        )
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"\nTotal images: {sum(row['images'] for row in summary)}")
    print(f"Reports saved to: {report_dir}")
    print("Empty labels need review; they are not automatically errors.")


if __name__ == "__main__":
    main()