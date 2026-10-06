from pathlib import Path
from datetime import datetime
from collections import defaultdict
import csv
import textwrap

from PIL import Image, ImageDraw, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "fabricDD.v2i.yolov8"
REPORTS_DIR = PROJECT_DIR / "reports"

# Find the latest completed grouping report.
candidates = sorted(
    folder
    for folder in REPORTS_DIR.glob("related_images_*")
    if (folder / "candidate_group_members.csv").is_file()
)

if not candidates:
    raise FileNotFoundError(
        "Run 04_find_related_images.py before this script."
    )

source_dir = candidates[-1]

with (source_dir / "candidate_group_members.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    members = list(csv.DictReader(file))

groups = defaultdict(list)

for member in members:
    groups[member["candidate_group_id"]].append(member)

output_dir = (
    REPORTS_DIR
    / datetime.now().strftime("contact_sheets_%Y%m%d_%H%M%S_%f")
)
output_dir.mkdir(parents=True, exist_ok=False)

# Separate folders make cross-split groups easy to find first.
cross_dir = output_dir / "cross_split"
within_dir = output_dir / "within_split"
cross_dir.mkdir()
within_dir.mkdir()

CELL_WIDTH = 420
CELL_HEIGHT = 460
HEADER_HEIGHT = 70
COLUMNS = 3
PER_PAGE = 6

review_rows = []
errors = []

for group_id, group_members in sorted(groups.items()):
    group_members = sorted(
        group_members, key=lambda row: (row["split"], row["image"])
    )

    cross_split = any(
        row["cross_split"].lower() == "true"
        for row in group_members
    )
    destination = cross_dir if cross_split else within_dir
    page_names = []

    for start in range(0, len(group_members), PER_PAGE):
        batch = group_members[start:start + PER_PAGE]
        page_number = start // PER_PAGE + 1
        row_count = (len(batch) + COLUMNS - 1) // COLUMNS

        sheet = Image.new(
            "RGB",
            (
                COLUMNS * CELL_WIDTH,
                HEADER_HEIGHT + row_count * CELL_HEIGHT,
            ),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        draw.text(
            (12, 10),
            f"Group: {group_id} | Page: {page_number} "
            f"| Images in group: {len(group_members)}",
            fill="black",
        )
        draw.text(
            (12, 30),
            group_members[0]["candidate_key"],
            fill="black",
        )

        for index, member in enumerate(batch):
            column = index % COLUMNS
            row = index // COLUMNS
            x = column * CELL_WIDTH
            y = HEADER_HEIGHT + row * CELL_HEIGHT

            # Reports created on Windows contain backslashes.
            relative_path = Path(
                member["image"].replace("\\", "/")
            )
            image_path = DATASET_DIR / relative_path

            draw.rectangle(
                (x + 4, y + 4, x + CELL_WIDTH - 4, y + CELL_HEIGHT - 4),
                outline="#aaaaaa",
            )

            draw.text(
                (x + 12, y + 10),
                f"Split: {member['split']}",
                fill="black",
            )

            filename_lines = textwrap.wrap(
                relative_path.name, width=55
            )
            draw.multiline_text(
                (x + 12, y + 30),
                "\n".join(filename_lines),
                fill="black",
                spacing=3,
            )

            try:
                with Image.open(image_path) as original:
                    preview = ImageOps.contain(
                        original.convert("RGB"),
                        (CELL_WIDTH - 24, 340),
                    )

                preview_x = x + (CELL_WIDTH - preview.width) // 2
                preview_y = y + 100 + (340 - preview.height) // 2
                sheet.paste(preview, (preview_x, preview_y))

            except (OSError, ValueError) as error:
                draw.text(
                    (x + 12, y + 120),
                    "IMAGE COULD NOT BE READ",
                    fill="red",
                )
                errors.append({
                    "group_id": group_id,
                    "image": member["image"],
                    "error": str(error),
                })

        filename = f"{group_id}_page_{page_number:02d}.jpg"
        sheet.save(destination / filename, quality=92)
        page_names.append(
            str((destination / filename).relative_to(output_dir))
        )

    review_rows.append({
        "group_id": group_id,
        "candidate_key": group_members[0]["candidate_key"],
        "image_count": len(group_members),
        "cross_split": cross_split,
        "contact_sheets": "|".join(page_names),
        "decision": "",
        "notes": "",
    })

    if len(review_rows) % 25 == 0:
        print(f"Contact sheets created for {len(review_rows)} groups...")


with (output_dir / "group_review.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "group_id", "candidate_key", "image_count",
            "cross_split", "contact_sheets", "decision", "notes",
        ],
    )
    writer.writeheader()
    writer.writerows(review_rows)

with (output_dir / "read_errors.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=["group_id", "image", "error"]
    )
    writer.writeheader()
    writer.writerows(errors)

print("\nContact sheets completed.")
print(f"Groups: {len(review_rows)}")
print(f"Image read errors: {len(errors)}")
print(f"Grouping report used: {source_dir}")
print(f"Output folder: {output_dir}")
print("Dataset images and labels were not modified.")