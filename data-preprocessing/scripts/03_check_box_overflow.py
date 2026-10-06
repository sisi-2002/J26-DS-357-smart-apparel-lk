from pathlib import Path
from datetime import datetime
import csv
import math
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "fabricDD.v2i.yolov8"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

rows = []
errors = []

for split in ("train", "valid", "test"):
    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset folders: {split}")

    for image_path in sorted(image_dir.rglob("*")):
        if image_path.suffix.lower() not in EXTENSIONS:
            continue

        relative_path = image_path.relative_to(image_dir)
        label_path = label_dir / relative_path.with_suffix(".txt")

        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size

            lines = label_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()

            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue

                values = list(map(float, line.split()))
                if len(values) != 5:
                    raise ValueError(f"Invalid format at line {line_number}")
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite value at line {line_number}")

                class_id, x, y, width, height = values

                # Zero-size boxes are reviewed separately.
                if width <= 0 or height <= 0:
                    continue

                left = x - width / 2
                right = x + width / 2
                top = y - height / 2
                bottom = y + height / 2

                normalized_overflow = max(
                    0, -left, right - 1, -top, bottom - 1
                )

                # Same tolerance used in the previous validator.
                if normalized_overflow <= 1e-6:
                    continue

                left_px = max(0, -left) * image_width
                right_px = max(0, right - 1) * image_width
                top_px = max(0, -top) * image_height
                bottom_px = max(0, bottom - 1) * image_height

                rows.append({
                    "split": split,
                    "image": str(relative_path),
                    "label_line": line_number,
                    "class_id": class_id,
                    "image_width": image_width,
                    "image_height": image_height,
                    "left_overflow_px": left_px,
                    "right_overflow_px": right_px,
                    "top_overflow_px": top_px,
                    "bottom_overflow_px": bottom_px,
                    "max_overflow_px": max(
                        left_px, right_px, top_px, bottom_px
                    ),
                })

        except (OSError, UnicodeError, ValueError) as error:
            errors.append({
                "split": split,
                "image": str(relative_path),
                "error": str(error),
            })

report_dir = (
    PROJECT_DIR / "reports"
    / datetime.now().strftime("overflow_%Y%m%d_%H%M%S_%f")
)
report_dir.mkdir(parents=True, exist_ok=False)

fields = [
    "split", "image", "label_line", "class_id",
    "image_width", "image_height",
    "left_overflow_px", "right_overflow_px",
    "top_overflow_px", "bottom_overflow_px", "max_overflow_px",
]

with (report_dir / "box_overflow_pixels.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

with (report_dir / "read_errors.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=["split", "image", "error"]
    )
    writer.writeheader()
    writer.writerows(errors)

print(f"Boundary-flagged boxes: {len(rows)}")
print(f"Read errors: {len(errors)}")

if rows:
    print(
        "Largest overflow: "
        f"{max(row['max_overflow_px'] for row in rows):.6f} pixels"
    )

print(f"Reports saved to: {report_dir}")
print("Dataset files were not modified.")