from pathlib import Path
from datetime import datetime
from collections import Counter
import csv
import math

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "fabricDD.v2i.yolov8"

# Dataset එකේ data.yaml හි class order එක.
CLASS_NAMES = [
    "baekra",
    "color issues",
    "contamination",
    "cut",
    "gray stitch",
    "selvet",
    "stain",
]

SPLITS = ("train", "valid", "test")

# Floating-point rounding වෙනුවෙන් ඉතා කුඩා tolerance එකක්.
EDGE_TOLERANCE = 1e-6

REPORT_DIR = (
    PROJECT_DIR
    / "reports"
    / datetime.now().strftime("label_validation_%Y%m%d_%H%M%S_%f")
)

issues = []
summary = []
class_counts = Counter()


def add_issue(split, file_path, line_number, issue, details):
    issues.append({
        "split": split,
        "file": str(file_path.relative_to(DATASET_DIR)),
        "line": line_number,
        "issue": issue,
        "details": details,
    })


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    # වැරදි path එකක් නම් හිස් report එකක් හදන්නේ නැහැ.
    for split in SPLITS:
        folder = DATASET_DIR / split / "labels"
        if not folder.is_dir():
            raise FileNotFoundError(f"Labels folder not found: {folder}")

    for split in SPLITS:
        label_files = sorted(
            (DATASET_DIR / split / "labels").rglob("*.txt")
        )

        empty_files = 0
        unreadable_files = 0
        annotation_rows = 0
        passed_rows = 0
        flagged_rows = 0

        for label_path in label_files:
            try:
                text = label_path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError) as error:
                unreadable_files += 1
                add_issue(
                    split, label_path, "", "unreadable_file", str(error)
                )
                continue

            if not text.strip():
                empty_files += 1
                add_issue(
                    split,
                    label_path,
                    "",
                    "empty_label_review",
                    "May be a valid negative image; needs visual review.",
                )
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue

                annotation_rows += 1
                fields = line.split()

                if len(fields) != 5:
                    flagged_rows += 1
                    add_issue(
                        split,
                        label_path,
                        line_number,
                        "wrong_field_count",
                        f"Expected 5 values; found {len(fields)}.",
                    )
                    continue

                try:
                    values = [float(value) for value in fields]
                except ValueError:
                    flagged_rows += 1
                    add_issue(
                        split,
                        label_path,
                        line_number,
                        "non_numeric_value",
                        line,
                    )
                    continue

                if not all(math.isfinite(value) for value in values):
                    flagged_rows += 1
                    add_issue(
                        split,
                        label_path,
                        line_number,
                        "non_finite_value",
                        "NaN or infinity found.",
                    )
                    continue

                class_value, x, y, width, height = values
                problems = []

                # YOLO class ID එක integer token එකක් විය යුතුයි.
                try:
                    class_id = int(fields[0])
                except ValueError:
                    class_id = None

                if class_id is None or not 0 <= class_id < len(CLASS_NAMES):
                    problems.append((
                        "invalid_class_id",
                        f"Expected integer 0–6; found {fields[0]}.",
                    ))
                else:
                    # Coordinates flagged වුවත් හඳුනාගත් class එක ගණනය කරනවා.
                    class_counts[(split, class_id)] += 1

                if not (0 <= x <= 1 and 0 <= y <= 1):
                    problems.append((
                        "invalid_center",
                        f"x={x}, y={y}; expected values within [0, 1].",
                    ))

                if not (0 < width <= 1 and 0 < height <= 1):
                    problems.append((
                        "invalid_box_size",
                        f"width={width}, height={height}; expected (0, 1].",
                    ))

                if width > 0 and height > 0:
                    left = x - width / 2
                    right = x + width / 2
                    top = y - height / 2
                    bottom = y + height / 2

                    overflow = max(
                        0.0, -left, -top, right - 1, bottom - 1
                    )

                    if overflow > EDGE_TOLERANCE:
                        problems.append((
                            "box_outside_image",
                            f"left={left:.8f}, top={top:.8f}, "
                            f"right={right:.8f}, bottom={bottom:.8f}; "
                            f"max_normalized_overflow={overflow:.8f}",
                        ))

                if problems:
                    flagged_rows += 1
                    for issue, details in problems:
                        add_issue(
                            split, label_path, line_number, issue, details
                        )
                else:
                    passed_rows += 1

        summary.append({
            "split": split,
            "label_files": len(label_files),
            "empty_labels_review": empty_files,
            "unreadable_files": unreadable_files,
            "annotation_rows": annotation_rows,
            "passed_rows": passed_rows,
            "flagged_rows": flagged_rows,
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=False)

    write_csv(
        REPORT_DIR / "label_summary.csv",
        summary,
        [
            "split", "label_files", "empty_labels_review",
            "unreadable_files", "annotation_rows",
            "passed_rows", "flagged_rows",
        ],
    )

    write_csv(
        REPORT_DIR / "label_issues.csv",
        issues,
        ["split", "file", "line", "issue", "details"],
    )

    class_rows = [
        {
            "split": split,
            "class_id": class_id,
            "class_name": name,
            "annotation_count": class_counts[(split, class_id)],
        }
        for split in SPLITS
        for class_id, name in enumerate(CLASS_NAMES)
    ]

    write_csv(
        REPORT_DIR / "class_counts.csv",
        class_rows,
        ["split", "class_id", "class_name", "annotation_count"],
    )

    print("\nLabel validation completed.")
    for row in summary:
        print(
            f"{row['split']}: "
            f"passed rows={row['passed_rows']}, "
            f"flagged rows={row['flagged_rows']}, "
            f"empty labels for review={row['empty_labels_review']}"
        )

    print(f"\nReports saved to:\n{REPORT_DIR}")
    print("\nDataset files were not modified.")


if __name__ == "__main__":
    main()