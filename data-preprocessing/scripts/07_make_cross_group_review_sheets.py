from pathlib import Path
import csv
import textwrap

from PIL import Image, ImageDraw, ImageOps


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "fabricDD.v2i.yolov8"
REPORT_DIR = PROJECT_DIR / "reports" / "cross_group_related_20261008_114158_369379"
CANDIDATES = REPORT_DIR / "cross_group_similarity_candidates.csv"
OUTPUT_DIR = REPORT_DIR / "review_sheets"

OUTPUT_DIR.mkdir(exist_ok=True)

with CANDIDATES.open(newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

selected = []
seen_units = set()
for row in rows:
    if int(row["hash_distance"]) > 12:
        continue
    key = tuple(sorted((row["left_unit_id"], row["right_unit_id"])))
    if key in seen_units:
        continue
    seen_units.add(key)
    selected.append(row)

CELL_W = 540
CELL_H = 410
HEADER_H = 90
PAIR_H = HEADER_H + CELL_H
PER_PAGE = 4

for start in range(0, len(selected), PER_PAGE):
    batch = selected[start:start + PER_PAGE]
    page = start // PER_PAGE + 1
    sheet = Image.new("RGB", (CELL_W * 2, PAIR_H * len(batch)), "white")
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(batch):
        y0 = index * PAIR_H
        title = (
            f"Candidate {start + index + 1} | hash={row['hash_distance']} "
            f"color={row['color_distance']} | "
            f"{row['left_group_id'] or 'UNG'} <-> {row['right_group_id'] or 'UNG'}"
        )
        draw.text((10, y0 + 8), title, fill="black")

        for side, x0 in (("left", 0), ("right", CELL_W)):
            image_rel = row[f"{side}_image"]
            group_id = row[f"{side}_group_id"] or "UNGROUPED"
            label = f"{group_id} | {row[f'{side}_split']} | {Path(image_rel).name}"
            draw.rectangle(
                (x0 + 4, y0 + HEADER_H, x0 + CELL_W - 4, y0 + PAIR_H - 4),
                outline="#999999",
            )
            draw.multiline_text(
                (x0 + 10, y0 + 32),
                "\n".join(textwrap.wrap(label, width=75)),
                fill="black",
                spacing=3,
            )
            try:
                with Image.open(DATASET_DIR / Path(image_rel)) as image:
                    preview = ImageOps.contain(
                        ImageOps.exif_transpose(image).convert("RGB"),
                        (CELL_W - 24, CELL_H - 24),
                    )
                px = x0 + (CELL_W - preview.width) // 2
                py = y0 + HEADER_H + (CELL_H - preview.height) // 2
                sheet.paste(preview, (px, py))
            except (OSError, ValueError) as error:
                draw.text((x0 + 10, y0 + HEADER_H + 20), str(error), fill="red")

    sheet.save(OUTPUT_DIR / f"cross_group_candidates_page_{page:02d}.jpg", quality=92)

print(f"Selected candidates: {len(selected)}")
print(f"Review sheets: {OUTPUT_DIR}")
